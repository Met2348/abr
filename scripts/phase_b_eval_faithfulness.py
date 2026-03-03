#!/usr/bin/env python3
"""Evaluate a trained Phase C value head on one held-out Phase C artifact directory.

Why this file exists
--------------------
C2 training should not be trusted only by the metrics saved during training.
This script re-loads a saved value head, re-encodes an eval artifact directory,
and recomputes the calibration/corruption metrics in a clean evaluation path.

Interaction with other files
----------------------------
- `scripts/phase_b_train_value.py`: produces the saved value-head checkpoints
- `src/ours/phase_b/value_data.py`: loads joined eval examples/corruptions
- `src/ours/phase_b/value_head.py`: reloads the head and encodes features
- `src/ours/phase_b/faithfulness_eval.py`: computes the metrics
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    """Add the repo-local `src/` directory to `sys.path`."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_b.faithfulness_eval import (  # noqa: E402
    compute_calibration_summary,
    compute_corruption_summary,
    render_faithfulness_summary_markdown,
)
from ours.phase_b.posthoc_calibration import (  # noqa: E402
    IsotonicCalibrationConfig,
    TemperatureCalibrationConfig,
    apply_posthoc_calibration,
    fit_isotonic_calibrator,
    fit_temperature_scaler,
)
from ours.phase_b.value_data import (  # noqa: E402
    assert_phase_c_compatibility,
    load_corruption_variants,
    load_phase_c_manifest,
    load_value_supervision_examples,
)
from ours.phase_b.value_head import (  # noqa: E402
    encode_text_features,
    freeze_backbone,
    load_value_head_checkpoint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone Phase C value-head evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Phase C value head on one Phase C artifact directory."
    )
    parser.add_argument("--value-run-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-name",
        choices=["best", "final"],
        default="best",
        help="Which saved value-head checkpoint to load from the training run dir.",
    )
    parser.add_argument("--run-name", default="phase_c_value_eval")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_c_eval"),
    )
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-corruption-variants", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--posthoc-calibration",
        choices=["none", "temperature", "isotonic", "from_run"],
        default="none",
        help=(
            "Optional post-hoc calibration in standalone eval. "
            "`temperature`/`isotonic` fit on the current eval set; `from_run` loads saved calibrator."
        ),
    )
    parser.add_argument("--posthoc-temperature-lr", type=float, default=0.05)
    parser.add_argument("--posthoc-temperature-max-iters", type=int, default=200)
    parser.add_argument("--posthoc-temperature-min", type=float, default=0.05)
    parser.add_argument("--posthoc-temperature-max", type=float, default=10.0)
    parser.add_argument("--posthoc-isotonic-min-points", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the standalone C2 evaluation path."""
    args = parse_args(argv)
    if args.posthoc_temperature_lr <= 0.0:
        raise ValueError("--posthoc-temperature-lr must be > 0")
    if args.posthoc_temperature_max_iters <= 0:
        raise ValueError("--posthoc-temperature-max-iters must be > 0")
    if args.posthoc_temperature_min <= 0.0:
        raise ValueError("--posthoc-temperature-min must be > 0")
    if args.posthoc_temperature_max <= args.posthoc_temperature_min:
        raise ValueError("--posthoc-temperature-max must be > --posthoc-temperature-min")
    if args.posthoc_isotonic_min_points <= 0:
        raise ValueError("--posthoc-isotonic-min-points must be > 0")
    # 先校验训练 run 和 eval 数据目录的契约一致性，再做任何推理。
    manifest_path = args.value_run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Value-run manifest not found: {manifest_path}")
    value_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_manifest = load_phase_c_manifest(Path(value_manifest["train_dir"]))
    eval_manifest = load_phase_c_manifest(args.eval_dir)
    assert_phase_c_compatibility(train_manifest, eval_manifest)

    train_metrics_path = args.value_run_dir / "train_metrics.json"
    if not train_metrics_path.exists():
        raise FileNotFoundError(f"Expected train_metrics.json in {args.value_run_dir}")
    train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))
    train_target_mean = float(train_metrics["train_target_mean"])

    # `best` 不存在时回退到 final，避免因中断 run 无法复评。
    checkpoint_name = "best_value_head.pt" if args.checkpoint_name == "best" else "final_value_head.pt"
    checkpoint_path = args.value_run_dir / checkpoint_name
    if args.checkpoint_name == "best" and not checkpoint_path.exists():
        checkpoint_path = args.value_run_dir / "final_value_head.pt"
    value_head, _, _ = load_value_head_checkpoint(checkpoint_path)

    eval_examples, _ = load_value_supervision_examples(
        args.eval_dir,
        max_samples=args.max_eval_samples,
        require_corruptions=False,
    )
    eval_corruptions, _ = load_corruption_variants(
        args.eval_dir,
        max_variants=args.max_corruption_variants,
    )

    torch, AutoModelForCausalLM, AutoTokenizer = _import_runtime_deps()
    resolved = value_manifest["resolved_backbone"]
    max_length = int(args.max_length if args.max_length is not None else value_manifest["train_config"]["max_length"])
    tokenizer_path = _resolve_tokenizer_load_path(
        model_path=str(resolved["model_path"]),
        adapter_path=(Path(resolved["adapter_path"]) if resolved.get("adapter_path") else None),
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    dtype = _resolve_dtype(str(resolved["dtype"]), torch)
    model_load_kwargs: dict[str, Any] = {
        "device_map": str(resolved["device_map"]),
        "trust_remote_code": True,
    }
    from_pretrained_sig = __import__("inspect").signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in from_pretrained_sig.parameters:
        model_load_kwargs["dtype"] = dtype
    else:
        model_load_kwargs["torch_dtype"] = dtype

    backbone = AutoModelForCausalLM.from_pretrained(str(resolved["model_path"]), **model_load_kwargs)
    adapter_path = resolved.get("adapter_path")
    if adapter_path:
        backbone = _attach_peft_adapter_for_inference(backbone, Path(adapter_path))
    freeze_backbone(backbone)

    # 重新编码 eval features，避免复用训练时缓存带来的隐性状态污染。
    clean_features = _encode_text_list(
        texts=[f"{example.prompt_text}{example.prefix_target_text}" for example in eval_examples],
        backbone=backbone,
        tokenizer=tokenizer,
        torch_module=torch,
        max_length=max_length,
        batch_size=int(value_manifest["train_config"]["per_device_eval_batch_size"]),
    )
    corrupt_features = _encode_text_list(
        texts=[f"{variant.prompt_text}{variant.corrupted_prefix_text}" for variant in eval_corruptions],
        backbone=backbone,
        tokenizer=tokenizer,
        torch_module=torch,
        max_length=max_length,
        batch_size=int(value_manifest["train_config"]["per_device_eval_batch_size"]),
    ) if eval_corruptions else None

    value_head.to(clean_features.device)
    value_head.eval()
    with torch.no_grad():
        clean_outputs = value_head(clean_features)
        clean_scores_tensor = clean_outputs["scores"]
        clean_scores = clean_scores_tensor.detach().cpu().tolist()
        clean_logits = clean_outputs["logits"]
        corrupt_scores = value_head(corrupt_features)["scores"].detach().cpu().tolist() if corrupt_features is not None else []

    target_scores = [float(example.target_success_rate) for example in eval_examples]
    calibration_raw = compute_calibration_summary(
        [float(x) for x in clean_scores],
        target_scores,
        reference_mean=float(train_target_mean),
    )
    posthoc_payload = _resolve_posthoc_payload(
        args=args,
        value_run_dir=args.value_run_dir,
        checkpoint_name=args.checkpoint_name,
        logits=clean_logits,
        targets=clean_features.new_tensor(target_scores),
        torch_module=torch,
    )
    calibration_posthoc = None
    posthoc_scores: list[float] | None = None
    if posthoc_payload is not None:
        posthoc_scores_tensor = apply_posthoc_calibration(
            logits=clean_logits,
            scores=clean_scores_tensor,
            calibrator=posthoc_payload,
            torch_module=torch,
        )
        posthoc_scores = [float(x) for x in posthoc_scores_tensor.detach().cpu().tolist()]
        calibration_posthoc = compute_calibration_summary(
            posthoc_scores,
            target_scores,
            reference_mean=float(train_target_mean),
        )
    corruption = None
    if eval_corruptions:
        prefix_score_by_id = {
            example.prefix_id: float(score)
            for example, score in zip(eval_examples, clean_scores, strict=True)
        }
        corruption = compute_corruption_summary(
            [prefix_score_by_id[variant.clean_prefix_id] for variant in eval_corruptions],
            [float(x) for x in corrupt_scores],
            corruption_types=[variant.corruption_type for variant in eval_corruptions],
            corruption_step_indices=[variant.corruption_step_index for variant in eval_corruptions],
        )

    run_dir = args.output_root / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    prefix_scores_path = run_dir / "prefix_scores.jsonl"
    corruption_scores_path = run_dir / "corruption_scores.jsonl"
    summary_md_path = run_dir / "summary.md"
    out_manifest_path = run_dir / "manifest.json"

    prefix_rows = []
    prefix_score_by_id = {}
    for idx, (example, score) in enumerate(zip(eval_examples, clean_scores, strict=True)):
        row = {
            "prefix_id": example.prefix_id,
            "sample_id": example.sample_id,
            "dataset": example.dataset,
            "split": example.split,
            "question": example.question,
            "current_step_role": example.current_step_role,
            "prefix_step_index": example.prefix_step_index,
            # Keep `predicted_value` for compatibility with existing scripts.
            "predicted_value": float(score),
            "predicted_value_raw": float(score),
            "target_success_rate": float(example.target_success_rate),
            "target_parseable_rate": float(example.target_parseable_rate),
        }
        if posthoc_scores is not None:
            row["predicted_value_posthoc"] = float(posthoc_scores[idx])
        prefix_rows.append(row)
        prefix_score_by_id[example.prefix_id] = float(score)
    _write_jsonl(prefix_scores_path, prefix_rows)

    corruption_rows = []
    for variant, corrupt_score in zip(eval_corruptions, corrupt_scores, strict=True):
        corruption_rows.append(
            {
                "corruption_id": variant.corruption_id,
                "clean_prefix_id": variant.clean_prefix_id,
                "sample_id": variant.sample_id,
                "dataset": variant.dataset,
                "split": variant.split,
                "question": variant.question,
                "corruption_type": variant.corruption_type,
                "corruption_step_index": variant.corruption_step_index,
                "clean_value": float(prefix_score_by_id[variant.clean_prefix_id]),
                "corrupted_value": float(corrupt_score),
                "value_margin": float(prefix_score_by_id[variant.clean_prefix_id] - float(corrupt_score)),
            }
        )
    _write_jsonl(corruption_scores_path, corruption_rows)

    # 输出结构与训练脚本保持对齐，便于做 run 间对比。
    metrics = {
        "calibration": calibration_raw,
        "calibration_posthoc": calibration_posthoc,
        "posthoc_calibration": posthoc_payload,
        "corruption": corruption,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(
        render_faithfulness_summary_markdown(
            title="Phase C C2 Standalone Value-Head Evaluation",
            calibration=calibration_raw,
            corruption=corruption,
            metadata={
                "value_run_dir": args.value_run_dir,
                "eval_dir": args.eval_dir,
                "checkpoint_path": checkpoint_path,
                "posthoc_calibration": args.posthoc_calibration,
            },
        ),
        encoding="utf-8",
    )
    out_manifest_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "script": "scripts/phase_b_eval_faithfulness.py",
                "value_run_dir": str(args.value_run_dir),
                "eval_dir": str(args.eval_dir),
                "checkpoint_path": str(checkpoint_path),
                "posthoc_calibration": str(args.posthoc_calibration),
                "output_files": {
                    "metrics": str(metrics_path),
                    "prefix_scores": str(prefix_scores_path),
                    "corruption_scores": str(corruption_scores_path),
                    "summary_md": str(summary_md_path),
                },
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print("Phase C: Eval Faithfulness")
    print("=" * 88)
    print(f"value_run_dir     : {args.value_run_dir}")
    print(f"eval_dir          : {args.eval_dir}")
    print(f"checkpoint_path   : {checkpoint_path}")
    print(f"brier_score       : {calibration_raw['brier_score']:.6f}")
    print(f"pearson           : {calibration_raw['pearson']:.6f}")
    if calibration_posthoc is not None:
        print(f"posthoc_brier     : {calibration_posthoc['brier_score']:.6f}")
        print(f"posthoc_pearson   : {calibration_posthoc['pearson']:.6f}")
    if corruption is not None:
        print(f"corr_pair_acc     : {corruption['pair_accuracy']:.6f}")
        print(f"corr_auc          : {corruption['auc_clean_vs_corrupt']:.6f}")
    print(f"metrics_path      : {metrics_path}")
    print("=" * 88)
    return 0


def _resolve_posthoc_payload(
    *,
    args: argparse.Namespace,
    value_run_dir: Path,
    checkpoint_name: str,
    logits: Any,
    targets: Any,
    torch_module: Any,
) -> dict[str, Any] | None:
    """Resolve which post-hoc calibration payload should be applied.

    Modes
    -----
    - `none`: no post-hoc calibration
    - `temperature`: fit a fresh temperature scaler on this eval set
    - `isotonic`: fit an isotonic calibrator on this eval set
    - `from_run`: load saved payload from the training run directory
    """
    if args.posthoc_calibration == "none":
        return None
    if args.posthoc_calibration == "temperature":
        # 在当前 eval 集上现拟合；用于诊断“可校准性”，不是训练信号。
        cfg = TemperatureCalibrationConfig(
            lr=float(args.posthoc_temperature_lr),
            max_iters=int(args.posthoc_temperature_max_iters),
            min_temperature=float(args.posthoc_temperature_min),
            max_temperature=float(args.posthoc_temperature_max),
            init_temperature=1.0,
        )
        return fit_temperature_scaler(
            logits=logits,
            targets=targets,
            torch_module=torch_module,
            config=cfg,
        )
    if args.posthoc_calibration == "isotonic":
        # 等价于对当前分数做单调分段映射，适合非线性校准偏差。
        cfg = IsotonicCalibrationConfig(
            min_points=int(args.posthoc_isotonic_min_points),
        )
        return fit_isotonic_calibrator(
            scores=torch_module.sigmoid(logits),
            targets=targets,
            torch_module=torch_module,
            config=cfg,
        )
    if args.posthoc_calibration == "from_run":
        # 复用训练时持久化的 calibrator，验证部署路径可复现性。
        name = "best_posthoc_calibration.json" if checkpoint_name == "best" else "final_posthoc_calibration.json"
        path = value_run_dir / name
        if not path.exists() and checkpoint_name == "best":
            path = value_run_dir / "final_posthoc_calibration.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Requested --posthoc-calibration=from_run, but calibrator file is missing: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Post-hoc calibration payload {path} must be a JSON object")
        return payload
    raise ValueError(f"Unsupported posthoc calibration mode: {args.posthoc_calibration!r}")


def _import_runtime_deps():
    """Import heavy runtime dependencies lazily."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return torch, AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(name: str, torch_module: Any):
    """Map a user-facing dtype string onto one torch dtype object."""
    if name == "auto":
        if torch_module.cuda.is_available():
            return torch_module.bfloat16
        return torch_module.float32
    if name == "float32":
        return torch_module.float32
    if name == "float16":
        return torch_module.float16
    if name == "bfloat16":
        return torch_module.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _resolve_tokenizer_load_path(model_path: str, adapter_path: Path | None) -> str:
    """Choose which directory should provide tokenizer files."""
    if adapter_path is None:
        return str(model_path)
    if (adapter_path / "tokenizer_config.json").exists():
        return str(adapter_path)
    return str(model_path)


def _attach_peft_adapter_for_inference(model: Any, adapter_path: Path):
    """Attach one PEFT adapter to a loaded base model."""
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    try:
        from peft import PeftModel  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to import `peft` while attaching adapter for faithfulness eval") from exc
    return PeftModel.from_pretrained(model, str(adapter_path))


def _encode_text_list(
    *,
    texts: list[str],
    backbone: Any,
    tokenizer: Any,
    torch_module: Any,
    max_length: int,
    batch_size: int,
):
    """Encode texts into one pooled-feature tensor using batched forwards.

    Cast to float32 so loaded value-head weights (stored in float32) always see
    matching input dtype regardless of backbone runtime dtype.
    """
    if not texts:
        return None
    chunks = []
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size
    progress_every = max(1, math.ceil(total_batches / 8))
    started = time.time()
    print(
        "cache_eval_texts  : "
        f"start {total} texts in {total_batches} batches "
        f"(bs={batch_size}, progress_every~{progress_every})",
        flush=True,
    )
    for batch_idx, start in enumerate(range(0, total, batch_size), start=1):
        chunks.append(
            encode_text_features(
                backbone=backbone,
                tokenizer=tokenizer,
                texts=texts[start : start + batch_size],
                max_length=max_length,
                torch_module=torch_module,
            )
        )
        if batch_idx % progress_every == 0 or batch_idx == total_batches:
            elapsed = max(time.time() - started, 1e-6)
            done = min(batch_idx * batch_size, total)
            rate = done / elapsed
            print(
                "cache_eval_texts  : "
                f"{batch_idx}/{total_batches} batches ({done}/{total}, {done / total:.1%}) | "
                f"elapsed={elapsed:.1f}s | rate={rate:.3f} text/s",
                flush=True,
            )
    print("cache_eval_texts  : done", flush=True)
    return torch_module.cat(chunks, dim=0).to(dtype=torch_module.float32)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write UTF-8 JSONL rows to disk."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
