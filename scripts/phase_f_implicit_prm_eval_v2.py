#!/usr/bin/env python3
"""Evaluate LoRA implicit PRM with repository-aligned ProcessBench metrics.

English
-------
This script is a safer replacement for earlier ad-hoc implicit-PRM probes.
It keeps three things aligned with the rest of the repository:
1. ProcessBench rows are built with the same helper used by explicit Phase E eval.
2. The prompt text is included when computing token log-ratios.
3. The output metrics reuse the repository's ProcessBench metric definitions.

中文
----
这是给隐式 PRM 评测准备的一版更稳的实现，核心目的是和仓库现有口径对齐：
1. ProcessBench 的 prefix 展开沿用显式 Phase E benchmark eval 的 helper。
2. 计算 token log-ratio 时显式包含题目 prompt，而不是只看解题步骤。
3. 输出指标直接复用仓库自己的 ProcessBench 指标定义。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_e.benchmark_eval import (  # noqa: E402
    build_processbench_prefix_records,
    compute_processbench_metrics,
    load_processbench_examples,
    render_phase_e_benchmark_summary_markdown,
)
from ours.phase_b.value_head import (  # noqa: E402
    ensure_tokenizer_has_pad_token,
    maybe_resize_embeddings_for_tokenizer,
)
from ours.phase_e.runtime import (  # noqa: E402
    attach_peft_adapter_for_inference,
    build_max_memory_map,
    import_runtime_deps,
    resolve_dtype,
)
from ours.phase_e.training import (  # noqa: E402
    compute_text_truncation_diagnostics,
    validate_text_truncation_diagnostics,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PRIME-style implicit PRM on ProcessBench with repository-aligned metrics."
    )
    parser.add_argument("--lora-run-dir", type=Path, required=True)
    parser.add_argument("--base-model-path", default="")
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument("--run-name", default="phase_f_implicit_prm_v2")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_f_implicit_prm_v2"),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.0,
        help=(
            "Fixed ProcessBench threshold for the implicit PRM score scale. "
            "Default 0.0 follows the natural PRIME log-ratio sign."
        ),
    )
    parser.add_argument("--oracle-threshold-candidates", type=int, default=50)
    parser.add_argument(
        "--max-truncation-over-limit-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument("--dtype", default="")
    parser.add_argument("--device-map", default="")
    parser.add_argument("--max-gpu-memory-gib", type=int, default=None)
    parser.add_argument("--max-cpu-memory-gib", type=int, default=None)
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.lora_run_dir.exists():
        raise FileNotFoundError(f"--lora-run-dir not found: {args.lora_run_dir}")
    if not args.benchmark_path.exists():
        raise FileNotFoundError(f"--benchmark-path not found: {args.benchmark_path}")
    if args.max_samples is not None and int(args.max_samples) <= 0:
        raise ValueError("--max-samples must be > 0")
    if float(args.beta) <= 0.0:
        raise ValueError("--beta must be > 0")
    if int(args.max_length) <= 8:
        raise ValueError("--max-length must be > 8")
    if int(args.oracle_threshold_candidates) <= 0:
        raise ValueError("--oracle-threshold-candidates must be > 0")
    if not (0.0 <= float(args.max_truncation_over_limit_fraction) <= 1.0):
        raise ValueError("--max-truncation-over-limit-fraction must be in [0, 1]")
    if args.max_gpu_memory_gib is not None and int(args.max_gpu_memory_gib) <= 0:
        raise ValueError("--max-gpu-memory-gib must be > 0")
    if args.max_cpu_memory_gib is not None and int(args.max_cpu_memory_gib) <= 0:
        raise ValueError("--max-cpu-memory-gib must be > 0")
    return args


def _get_vocab_logits(model_out: Any, model: Any) -> Any | None:
    """Extract token logits from one HF forward output.

    English
    -------
    Most causal LM runs expose `model_out.logits` directly.  Some PRM-style
    checkpoints instead return hidden states under a custom wrapper, so we
    recover token logits via the underlying `lm_head`.

    中文
    ----
    大多数 causal LM 直接给 `model_out.logits`。少数 PRM 风格 checkpoint
    会包一层自定义输出，这里就回退到 `hidden_states + lm_head` 的路径。
    """
    if hasattr(model_out, "logits") and model_out.logits is not None:
        logits = model_out.logits
        vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
        # Some PRM wrappers expose reward-head logits with shape [B, T, 1].
        # Accept the direct logits path only when the last dimension really
        # looks like vocabulary logits.
        # 某些 PRM wrapper 会把 reward head 输出也命名成 logits，形状常见为 [B, T, 1]。
        # 只有最后一维看起来像真实词表维时，才直接走这条路径。
        if int(getattr(logits, "ndim", 0)) == 3 and (
            vocab_size <= 0 or int(logits.shape[-1]) == vocab_size
        ):
            return logits
    if hasattr(model_out, "hidden_states") and model_out.hidden_states is not None:
        hidden = model_out.hidden_states[-1]
        base_model = model
        if hasattr(base_model, "base_model"):
            base_model = base_model.base_model
        if hasattr(base_model, "model"):
            base_model = base_model.model
        if hasattr(base_model, "lm_head"):
            return base_model.lm_head(hidden)
    return None


def _compute_log_ratio_per_token(
    *,
    model: Any,
    tokenizer: Any,
    text: str,
    max_length: int,
    beta: float,
    device: Any,
    torch_module: Any,
) -> list[float]:
    """Compute beta-scaled token log-ratios using one adapter-toggled model.

    English
    -------
    To avoid loading both reference and LoRA models at once, we run the same
    PEFT model twice:
    1. with the adapter enabled,
    2. with the adapter temporarily disabled.

    中文
    ----
    为了避免同时占两份 7B 模型显存，这里复用同一个 PEFT 模型做两次前向：
    1. 开 adapter，
    2. 暂时关 adapter。
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_length),
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    if input_ids.shape[1] < 2:
        return []

    model.eval()
    with torch_module.no_grad():
        lora_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        disable_ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
        with disable_ctx:
            ref_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
            )

    lora_logits = _get_vocab_logits(lora_out, model)
    ref_logits = _get_vocab_logits(ref_out, model)
    if lora_logits is None or ref_logits is None:
        return []

    log_probs_lora = torch_module.nn.functional.log_softmax(lora_logits, dim=-1)
    log_probs_ref = torch_module.nn.functional.log_softmax(ref_logits, dim=-1)

    target_ids = input_ids[:, 1:]
    target_mask = attention_mask[:, 1:]
    lp_lora = log_probs_lora[:, :-1, :].gather(dim=2, index=target_ids.unsqueeze(-1)).squeeze(-1)
    lp_ref = log_probs_ref[:, :-1, :].gather(dim=2, index=target_ids.unsqueeze(-1)).squeeze(-1)
    log_ratio = (lp_lora - lp_ref) * float(beta)

    result: list[float] = []
    masked = log_ratio.squeeze(0)
    token_mask = target_mask.squeeze(0)
    for idx in range(masked.shape[0]):
        if float(token_mask[idx].item()) > 0.0:
            result.append(float(masked[idx].item()))
    return result


def _score_processbench_rows(
    *,
    examples: list[Any],
    model: Any,
    tokenizer: Any,
    max_length: int,
    beta: float,
    device: Any,
    torch_module: Any,
) -> tuple[list[Any], list[float], list[dict[str, Any]]]:
    """Score ProcessBench prefixes by mean log-ratio on the newly added step.

    中文
    ----
    这里不是给“整段 prefix 的平均 log-ratio”打分，而是只取相对上一条 prefix
    新增那一步的 token log-ratio 均值。这样更贴近 step-level PRM 的语义。
    """
    rows = build_processbench_prefix_records(examples)
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        grouped.setdefault(str(row.example_id), []).append(row)
    for example_rows in grouped.values():
        example_rows.sort(key=lambda item: int(item.prefix_step_index))

    ordered_rows: list[Any] = []
    ordered_scores: list[float] = []
    scored_rows: list[dict[str, Any]] = []

    for example in examples:
        example_id = str(example.example_id)
        example_rows = grouped[example_id]
        prompt_text = str(example.problem) + "\n\n"
        prev_token_count = len(
            _compute_log_ratio_per_token(
                model=model,
                tokenizer=tokenizer,
                text=prompt_text,
                max_length=int(max_length),
                beta=float(beta),
                device=device,
                torch_module=torch_module,
            )
        )
        for row in example_rows:
            full_text = row.input_text()
            cumulative = _compute_log_ratio_per_token(
                model=model,
                tokenizer=tokenizer,
                text=full_text,
                max_length=int(max_length),
                beta=float(beta),
                device=device,
                torch_module=torch_module,
            )
            step_tokens = cumulative[prev_token_count:]
            step_score = float(sum(step_tokens) / len(step_tokens)) if step_tokens else 0.0
            prev_token_count = len(cumulative)
            ordered_rows.append(row)
            ordered_scores.append(step_score)
            scored_rows.append(
                {
                    "row_id": row.row_id,
                    "example_id": row.example_id,
                    "prefix_step_index": row.prefix_step_index,
                    "label": row.label,
                    "is_good_prefix": row.is_good_prefix,
                    "is_first_bad_prefix": row.is_first_bad_prefix,
                    "score": step_score,
                }
            )
    return ordered_rows, ordered_scores, scored_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    manifest_path = Path(args.lora_run_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Phase E manifest: {manifest_path}")
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_backbone = dict(run_manifest.get("resolved_backbone", {}))
    base_model_path = str(args.base_model_path).strip() or str(resolved_backbone.get("model_path", "")).strip()
    if base_model_path == "":
        raise ValueError("Could not resolve base model path from args or LoRA manifest")

    adapter_path_text = str(resolved_backbone.get("adapter_path") or "").strip()
    adapter_path = Path(adapter_path_text) if adapter_path_text else Path(args.lora_run_dir) / "best_adapter"
    if not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

    torch, AutoModelForCausalLM, AutoTokenizer = import_runtime_deps()
    if bool(args.require_cuda) and not bool(torch.cuda.is_available()):
        raise RuntimeError("--require-cuda was specified but no CUDA device is visible")

    dtype_name = str(args.dtype).strip() or str(resolved_backbone.get("dtype", "bfloat16"))
    device_map = str(args.device_map).strip() or str(resolved_backbone.get("device_map", "auto"))
    max_gpu_memory_gib = (
        int(args.max_gpu_memory_gib)
        if args.max_gpu_memory_gib is not None
        else (
            int(resolved_backbone["max_gpu_memory_gib"])
            if resolved_backbone.get("max_gpu_memory_gib") is not None
            else None
        )
    )
    max_cpu_memory_gib = (
        int(args.max_cpu_memory_gib)
        if args.max_cpu_memory_gib is not None
        else (
            int(resolved_backbone["max_cpu_memory_gib"])
            if resolved_backbone.get("max_cpu_memory_gib") is not None
            else None
        )
    )

    print("=" * 88)
    print("Phase F Implicit PRM Eval v2")
    print("=" * 88)
    print(f"lora_run_dir      : {args.lora_run_dir}")
    print(f"base_model_path   : {base_model_path}")
    print(f"adapter_path      : {adapter_path}")
    print(f"benchmark_path    : {args.benchmark_path}")
    print(f"beta              : {float(args.beta):.4f}")
    print(f"fixed_threshold   : {float(args.fixed_threshold):.4f}")
    print(f"device_map        : {device_map}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
    synthesized_pad_token = ensure_tokenizer_has_pad_token(tokenizer)
    resolved_dtype = resolve_dtype(dtype_name, torch)
    load_kwargs: dict[str, Any] = {
        "device_map": str(device_map),
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "ignore_mismatched_sizes": True,
    }
    from_pretrained_sig = __import__("inspect").signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in from_pretrained_sig.parameters:
        load_kwargs["dtype"] = resolved_dtype
    else:
        load_kwargs["torch_dtype"] = resolved_dtype
    max_memory = build_max_memory_map(
        torch_module=torch,
        max_gpu_memory_gib=max_gpu_memory_gib,
        max_cpu_memory_gib=max_cpu_memory_gib,
    )
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    try:
        from transformers import Qwen2ForCausalLM  # type: ignore

        model_cls = Qwen2ForCausalLM
    except Exception:  # noqa: BLE001
        model_cls = AutoModelForCausalLM
    model = model_cls.from_pretrained(str(base_model_path), **load_kwargs)
    if synthesized_pad_token:
        maybe_resize_embeddings_for_tokenizer(backbone=model, tokenizer=tokenizer)
    model = attach_peft_adapter_for_inference(model, adapter_path)
    if hasattr(model, "enable_adapter"):
        model.enable_adapter()
    model.eval()
    device = next(model.parameters()).device

    examples = load_processbench_examples(Path(args.benchmark_path), max_samples=args.max_samples)
    rows = build_processbench_prefix_records(examples)
    truncation_diagnostics = compute_text_truncation_diagnostics(
        texts=[row.input_text() for row in rows],
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        batch_size=32,
        group_labels=[
            "good_prefix"
            if bool(row.is_good_prefix)
            else ("first_bad_prefix" if bool(row.is_first_bad_prefix) else "bad_prefix")
            for row in rows
        ],
    )
    validate_text_truncation_diagnostics(
        diagnostics=truncation_diagnostics,
        context_label="Implicit PRM v2 ProcessBench eval",
        max_allowed_over_limit_fraction=float(args.max_truncation_over_limit_fraction),
    )

    t0 = time.time()
    ordered_rows, ordered_scores, scored_rows = _score_processbench_rows(
        examples=examples,
        model=model,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        beta=float(args.beta),
        device=device,
        torch_module=torch,
    )
    elapsed = time.time() - t0

    fixed_metrics = compute_processbench_metrics(
        ordered_rows,
        ordered_scores,
        processbench_f1_threshold=float(args.fixed_threshold),
        processbench_f1_threshold_candidates=int(args.oracle_threshold_candidates),
    )
    oracle_metrics = compute_processbench_metrics(
        ordered_rows,
        ordered_scores,
        processbench_f1_threshold=None,
        processbench_f1_threshold_candidates=int(args.oracle_threshold_candidates),
    )

    run_dir = Path(args.output_root) / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "run_name": str(args.run_name),
        "lora_run_dir": str(args.lora_run_dir),
        "base_model_path": str(base_model_path),
        "adapter_path": str(adapter_path),
        "benchmark_path": str(args.benchmark_path),
        "n_examples": int(len(examples)),
        "n_prefix_rows": int(len(ordered_rows)),
        "beta": float(args.beta),
        "fixed_threshold": float(args.fixed_threshold),
        "elapsed_s": float(elapsed),
        "metrics_fixed_threshold": fixed_metrics,
        "metrics_oracle_sweep": oracle_metrics,
        "truncation_diagnostics": truncation_diagnostics,
    }
    manifest = {
        "artifact_stage": "phase_f_implicit_prm_eval_v2",
        "generated_at": summary["generated_at"],
        "run_name": str(args.run_name),
        "run_dir": str(run_dir),
        "lora_run_dir": str(args.lora_run_dir),
        "base_model_path": str(base_model_path),
        "adapter_path": str(adapter_path),
        "benchmark_path": str(args.benchmark_path),
        "config": {
            "beta": float(args.beta),
            "fixed_threshold": float(args.fixed_threshold),
            "max_length": int(args.max_length),
            "oracle_threshold_candidates": int(args.oracle_threshold_candidates),
            "dtype": str(dtype_name),
            "device_map": str(device_map),
            "max_gpu_memory_gib": max_gpu_memory_gib,
            "max_cpu_memory_gib": max_cpu_memory_gib,
        },
    }
    summary_json_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    scored_rows_path = run_dir / "scored_rows.jsonl"
    summary_md_path = run_dir / "summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with scored_rows_path.open("w", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_md_path.write_text(
        render_phase_e_benchmark_summary_markdown(
            title="Phase F Implicit PRM Eval v2",
            metadata={
                "benchmark_path": str(args.benchmark_path),
                "lora_run_dir": str(args.lora_run_dir),
                "beta": float(args.beta),
                "fixed_threshold": float(args.fixed_threshold),
                "n_examples": int(len(examples)),
                "n_prefix_rows": int(len(ordered_rows)),
            },
            metrics={
                **fixed_metrics,
                "oracle_processbench_f1": float(oracle_metrics["processbench_f1"]),
                "oracle_processbench_acc_erroneous": float(oracle_metrics["processbench_acc_erroneous"]),
                "oracle_processbench_acc_correct": float(oracle_metrics["processbench_acc_correct"]),
                "oracle_processbench_f1_threshold": float(oracle_metrics["processbench_f1_threshold"]),
            },
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print("Implicit PRM v2 summary")
    print("=" * 88)
    print(f"n_examples        : {len(examples)}")
    print(f"elapsed_s         : {elapsed:.2f}")
    print(f"pair_auc          : {fixed_metrics['pair_auc_good_vs_bad']:.6f}")
    print(f"first_edge        : {fixed_metrics['first_error_edge_accuracy']:.6f}")
    print(f"f1_fixed          : {fixed_metrics['processbench_f1']:.6f}")
    print(f"f1_oracle         : {oracle_metrics['processbench_f1']:.6f}")
    print(f"summary_json      : {summary_json_path}")
    print(f"summary_md        : {summary_md_path}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
