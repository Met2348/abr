#!/usr/bin/env python3
"""Evaluate same-family trust utility on one Phase E value-head run.

English
-------
This script deliberately ignores cross-dataset transfer.  It asks a narrower
question that is more relevant before RL-style use:

1. on the value head's *own* held-out dataset family,
2. can it support prompt-level reranking,
3. can abstention/rejection improve reliability,
4. and does stronger selection pressure expose failure modes?

中文
----
本脚本刻意忽略跨数据集迁移，只问一个更窄、也更接近 RL 前置门槛的问题：

1. 在这个 value head 自己的 held-out 数据集家族里，
2. 它能不能支持按题目的 rerank，
3. 拒答/保守接收能不能提高可靠性，
4. 更强的选择压力会不会暴露明显漏洞？
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_b.value_head import load_value_head_checkpoint  # noqa: E402
from ours.phase_d.external_pairs import load_external_pair_jsonl  # noqa: E402
from ours.phase_e.runtime import (  # noqa: E402
    build_phase_e_backbone_signature,
    import_runtime_deps,
    load_backbone_and_tokenizer,
    load_or_encode_text_features,
    resolve_checkpoint_path,
    resolve_value_device,
    score_feature_tensor,
)
from ours.phase_e.samefamily_trust import (  # noqa: E402
    build_unique_candidate_rows,
    compute_samefamily_trust_metrics,
    render_samefamily_summary_markdown,
    write_prompt_rows_jsonl,
)
from ours.phase_e.training import (  # noqa: E402
    compute_text_truncation_diagnostics,
    validate_text_truncation_diagnostics,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate same-family rerank/rejection/pressure utility for one Phase E value-head run."
    )
    parser.add_argument("--value-run-dir", type=Path, required=True)
    parser.add_argument("--eval-pairs-jsonl", type=Path, default=None)
    parser.add_argument("--run-name", default="phase_e_samefamily_trust")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_samefamily_eval"),
    )
    parser.add_argument("--checkpoint-name", choices=["best", "final"], default="best")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--max-truncation-over-limit-fraction",
        type=float,
        default=0.10,
        help=(
            "Fail fast if more than this fraction of candidate texts exceed max_length. "
            "This keeps same-family trust reports from silently scoring truncated prefixes."
        ),
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
    parser.add_argument(
        "--feature-cache-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_feature_cache"),
    )
    parser.add_argument(
        "--feature-cache-mode",
        choices=["off", "read", "write", "read_write"],
        default="read_write",
    )
    parser.add_argument("--feature-cache-lock-timeout-sec", type=float, default=600.0)
    parser.add_argument("--edge-weight-mode", choices=["unit", "confidence"], default="unit")
    parser.add_argument(
        "--rejection-coverages",
        default="1.0,0.8,0.6,0.4,0.2",
        help="Comma-separated target coverages for the rejection curve.",
    )
    parser.add_argument(
        "--pressure-sizes",
        default="2,3,4,6,8",
        help="Comma-separated subset sizes for the best-of-N pressure curve.",
    )
    parser.add_argument("--pressure-repeats", type=int, default=4)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.value_run_dir.exists():
        raise FileNotFoundError(f"--value-run-dir not found: {args.value_run_dir}")
    if args.eval_pairs_jsonl is not None and not Path(args.eval_pairs_jsonl).exists():
        raise FileNotFoundError(f"--eval-pairs-jsonl not found: {args.eval_pairs_jsonl}")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_gpu_memory_gib is not None and int(args.max_gpu_memory_gib) <= 0:
        raise ValueError("--max-gpu-memory-gib must be > 0")
    if args.max_cpu_memory_gib is not None and int(args.max_cpu_memory_gib) <= 0:
        raise ValueError("--max-cpu-memory-gib must be > 0")
    if args.max_length is not None and int(args.max_length) <= 8:
        raise ValueError("--max-length must be > 8")
    if not (0.0 <= float(args.max_truncation_over_limit_fraction) <= 1.0):
        raise ValueError("--max-truncation-over-limit-fraction must be in [0, 1]")
    if float(args.feature_cache_lock_timeout_sec) <= 0.0:
        raise ValueError("--feature-cache-lock-timeout-sec must be > 0")
    if int(args.pressure_repeats) <= 0:
        raise ValueError("--pressure-repeats must be > 0")
    args.rejection_coverages = _parse_float_csv(args.rejection_coverages, name="--rejection-coverages", lower=0.0, upper=1.0)
    args.pressure_sizes = _parse_int_csv(args.pressure_sizes, name="--pressure-sizes", lower=2)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    manifest_path = Path(args.value_run_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Phase E manifest: {manifest_path}")
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = resolve_checkpoint_path(
        value_run_dir=Path(args.value_run_dir),
        run_manifest=run_manifest,
        checkpoint_name=str(args.checkpoint_name),
    )
    requested_checkpoint_path = str(
        dict(run_manifest.get("output_files", {})).get(
            "best_value_head" if str(args.checkpoint_name) == "best" else "final_value_head",
            "",
        )
    ).strip()
    checkpoint_resolution = {
        "requested_checkpoint_name": str(args.checkpoint_name),
        "requested_checkpoint_path": requested_checkpoint_path,
        "resolved_checkpoint_path": str(checkpoint_path),
        "fallback_to_final": bool(
            str(args.checkpoint_name) == "best"
            and requested_checkpoint_path != ""
            and Path(requested_checkpoint_path) != checkpoint_path
        ),
    }
    eval_pairs_jsonl = _resolve_eval_pairs_jsonl(args=args, run_manifest=run_manifest)
    value_head, _, _ = load_value_head_checkpoint(checkpoint_path)

    resolved_backbone = dict(run_manifest.get("resolved_backbone", {}))
    model_path = str(resolved_backbone.get("model_path", "")).strip()
    adapter_path_text = str(resolved_backbone.get("adapter_path") or "").strip()
    if model_path == "":
        raise ValueError("Phase E manifest missing resolved_backbone.model_path")

    torch, AutoModelForCausalLM, AutoTokenizer = import_runtime_deps()
    if bool(args.require_cuda) and not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA is required by --require-cuda but no GPU is visible")

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
    max_length = int(args.max_length) if args.max_length is not None else int(
        run_manifest.get("train_config", {}).get("max_length", 1024)
    )
    adapter_path = Path(adapter_path_text) if adapter_path_text else None

    print("=" * 88)
    print("Phase E: Eval Same-Family Trust")
    print("=" * 88)
    print(f"value_run_dir      : {args.value_run_dir}")
    print(f"eval_pairs_jsonl   : {eval_pairs_jsonl}")
    print(f"checkpoint_path    : {checkpoint_path}")
    print(f"checkpoint_fallback: {checkpoint_resolution['fallback_to_final']}")
    print(f"model_path         : {model_path}")
    print(f"adapter_path       : {adapter_path if adapter_path is not None else '<none>'}")
    print(f"batch_size         : {args.batch_size}")
    print(f"max_length         : {max_length}")
    print(f"trunc_overlimit_max: {float(args.max_truncation_over_limit_fraction):.4f}")
    print(f"edge_weight_mode   : {args.edge_weight_mode}")
    print(f"rejection_coverages: {args.rejection_coverages}")
    print(f"pressure_sizes     : {args.pressure_sizes}")
    print(f"pressure_repeats   : {args.pressure_repeats}")
    print(f"max_gpu_memory_gib : {max_gpu_memory_gib if max_gpu_memory_gib is not None else '<none>'}")
    print(f"max_cpu_memory_gib : {max_cpu_memory_gib if max_cpu_memory_gib is not None else '<none>'}")
    print("=" * 88)

    pairs, pair_summary = load_external_pair_jsonl(eval_pairs_jsonl)
    candidate_nodes, pools = build_unique_candidate_rows(pairs)

    load_start = time.perf_counter()
    backbone, tokenizer, _, tokenizer_path = load_backbone_and_tokenizer(
        model_path=model_path,
        adapter_path=adapter_path,
        dtype_name=dtype_name,
        device_map=device_map,
        max_gpu_memory_gib=max_gpu_memory_gib,
        max_cpu_memory_gib=max_cpu_memory_gib,
        torch_module=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
    )
    load_elapsed = time.perf_counter() - load_start
    value_device = resolve_value_device(backbone, torch)
    value_head.to(value_device)
    value_head.eval()

    feature_cache_stats: dict[str, Any] = {"hits": 0, "misses": 0, "writes": 0, "entries": {}}
    backbone_signature = build_phase_e_backbone_signature(
        model_path=model_path,
        adapter_path=(str(adapter_path) if adapter_path is not None else None),
        tokenizer_path=tokenizer_path,
        dtype=dtype_name,
        max_length=int(max_length),
    )

    candidate_texts = [node.input_text() for node in candidate_nodes]
    truncation_diagnostics = compute_text_truncation_diagnostics(
        texts=candidate_texts,
        tokenizer=tokenizer,
        max_length=int(max_length),
        batch_size=int(args.batch_size),
        group_labels=[str(node.source_tag) for node in candidate_nodes],
    )
    validate_text_truncation_diagnostics(
        diagnostics=truncation_diagnostics,
        context_label="Phase E same-family trust eval",
        max_allowed_over_limit_fraction=float(args.max_truncation_over_limit_fraction),
    )
    features = load_or_encode_text_features(
        cache_namespace="phase_e_samefamily_eval",
        cache_kind="candidate_nodes",
        texts=candidate_texts,
        backbone=backbone,
        tokenizer=tokenizer,
        max_length=int(max_length),
        batch_size=int(args.batch_size),
        feature_cache_root=Path(args.feature_cache_root),
        feature_cache_mode=str(args.feature_cache_mode),
        lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
        backbone_signature=backbone_signature,
        extra_signature={
            "value_run_dir": str(args.value_run_dir),
            "eval_pairs_jsonl": str(eval_pairs_jsonl),
            "num_pairs": int(len(pairs)),
            "num_candidates": int(len(candidate_nodes)),
        },
        torch_module=torch,
        feature_cache_stats=feature_cache_stats,
    )
    scores = score_feature_tensor(
        value_head=value_head,
        features=features,
        batch_size=int(args.batch_size),
        torch_module=torch,
    )
    candidate_scores = {
        node.candidate_id: float(score)
        for node, score in zip(candidate_nodes, scores, strict=True)
    }

    trust_result = compute_samefamily_trust_metrics(
        pools=pools,
        candidate_nodes=candidate_nodes,
        candidate_scores=candidate_scores,
        edge_weight_mode=str(args.edge_weight_mode),
        rejection_coverages=tuple(float(v) for v in args.rejection_coverages),
        pressure_sizes=tuple(int(v) for v in args.pressure_sizes),
        pressure_repeats=int(args.pressure_repeats),
    )

    run_dir = Path(args.output_root) / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_rows_path = run_dir / "prompt_rows.jsonl"
    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.json"
    manifest_out_path = run_dir / "manifest.json"
    summary_md_path = run_dir / "summary.md"

    write_prompt_rows_jsonl(prompt_rows_path, trust_result.prompt_rows)
    metrics_payload = {
        **trust_result.metrics,
        "pair_summary": pair_summary,
        "num_eval_pairs": int(len(pairs)),
        "num_prompt_pools_raw": int(len(pools)),
        "num_candidate_nodes_raw": int(len(candidate_nodes)),
        "model_load_elapsed_sec": float(load_elapsed),
        "truncation_diagnostics": truncation_diagnostics,
        "feature_cache": feature_cache_stats,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest_payload = {
        "artifact_stage": "phase_e_samefamily_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": str(args.run_name),
        "value_run_dir": str(args.value_run_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_resolution": checkpoint_resolution,
        "eval_pairs_jsonl": str(eval_pairs_jsonl),
        "edge_weight_mode": str(args.edge_weight_mode),
        "rejection_coverages": [float(v) for v in args.rejection_coverages],
        "pressure_sizes": [int(v) for v in args.pressure_sizes],
        "pressure_repeats": int(args.pressure_repeats),
        "max_truncation_over_limit_fraction": float(args.max_truncation_over_limit_fraction),
        "truncation_diagnostics": truncation_diagnostics,
        "output_files": {
            "prompt_rows": str(prompt_rows_path),
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "summary_md": str(summary_md_path),
            "manifest": str(manifest_out_path),
        },
    }
    manifest_out_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md = render_samefamily_summary_markdown(
            run_name=str(args.run_name),
            value_run_dir=Path(args.value_run_dir),
            eval_pairs_jsonl=Path(eval_pairs_jsonl),
            metrics=metrics_payload,
    )
    summary_md += _render_text_truncation_markdown(truncation_diagnostics)
    summary_md_path.write_text(summary_md, encoding="utf-8")

    print("-" * 88)
    print(f"prompt_pool_top1_acc : {metrics_payload['prompt_pool_top1_accuracy']:.6f}")
    print(f"prompt_pool_regret   : {metrics_payload['prompt_pool_mean_regret']:.6f}")
    if metrics_payload.get("local_last_safe_top1_accuracy") is not None:
        print(f"local_last_safe_top1: {float(metrics_payload['local_last_safe_top1_accuracy']):.6f}")
        print(f"local_first_bad_acc : {float(metrics_payload['local_first_bad_edge_accuracy']):.6f}")
    print(f"metrics_path         : {metrics_path}")
    print(f"summary_path         : {summary_md_path}")
    print(f"manifest_path        : {manifest_out_path}")
    print("=" * 88)
    return 0


def _resolve_eval_pairs_jsonl(*, args: argparse.Namespace, run_manifest: dict[str, Any]) -> Path:
    if args.eval_pairs_jsonl is not None:
        return Path(args.eval_pairs_jsonl)
    input_files = dict(run_manifest.get("input_files", {}))
    raw = str(input_files.get("eval_pairs_jsonl", "")).strip()
    if raw == "":
        raise ValueError("Phase E manifest missing input_files.eval_pairs_jsonl; please pass --eval-pairs-jsonl")
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"Resolved eval_pairs_jsonl not found: {path}")
    return path


def _parse_float_csv(raw: str, *, name: str, lower: float, upper: float) -> list[float]:
    values: list[float] = []
    for item in str(raw).split(","):
        text = item.strip()
        if text == "":
            continue
        value = float(text)
        if not (float(lower) <= value <= float(upper)):
            raise ValueError(f"{name} values must be in [{lower}, {upper}]")
        values.append(float(value))
    if not values:
        raise ValueError(f"{name} must contain at least one numeric value")
    return values


def _parse_int_csv(raw: str, *, name: str, lower: int) -> list[int]:
    values: list[int] = []
    for item in str(raw).split(","):
        text = item.strip()
        if text == "":
            continue
        value = int(text)
        if value < int(lower):
            raise ValueError(f"{name} values must be >= {lower}")
        values.append(int(value))
    if not values:
        raise ValueError(f"{name} must contain at least one integer value")
    return values


def _render_text_truncation_markdown(diagnostics: dict[str, Any]) -> str:
    """Render one compact Markdown appendix for text-level truncation diagnostics."""
    overall = dict(diagnostics["overall"])
    lines = [
        "",
        "## Truncation Diagnostics",
        "",
        f"- frac_texts_over_limit: `{float(overall['frac_texts_over_limit']):.6f}`",
        f"- text_length_p95: `{int(overall['text_length']['p95'])}`",
    ]
    by_group = dict(diagnostics.get("by_group", {}))
    if by_group:
        lines.extend(
            [
                "",
                "| group | frac_over_limit | text_length_p95 |",
                "|---|---:|---:|",
            ]
        )
        for group_label, payload in by_group.items():
            lines.append(
                f"| {group_label} | {float(payload['frac_texts_over_limit']):.4f} | "
                f"{int(payload['text_length']['p95'])} |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
