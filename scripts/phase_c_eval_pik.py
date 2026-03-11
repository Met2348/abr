#!/usr/bin/env python3
"""Standalone evaluation for trained Phase C P(IK) value heads.

Why this file exists
--------------------
Training-time metrics are not enough for research reporting. This script
re-evaluates a saved P(IK) head from disk on an eval artifact directory and
recomputes calibration/discrimination metrics in a clean path.

What this file does
-------------------
1. Load one P(IK) C2 run directory and one P(IK) C1 eval artifact directory.
2. Rebuild frozen-backbone question features.
3. Run the saved value head checkpoint.
4. Compute calibration metrics (Brier/Pearson/ECE) and known-vs-unknown AUROC.
5. Optionally fit/apply post-hoc calibration in standalone mode.
6. Persist JSON/JSONL/Markdown outputs for auditability.

Interaction with other files
----------------------------
- `scripts/phase_c_train_pik.py`: emits checkpoints + manifests consumed here
- `src/ours/phase_b/pik_data.py`: loads question-level eval examples
- `src/ours/phase_b/value_head.py`: loads value head and encodes features
- `src/ours/phase_b/faithfulness_eval.py`: computes calibration summary + AUC
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
    """Add repo-local `src/` to `sys.path` for script-style execution."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_b.faithfulness_eval import (  # noqa: E402
    compute_binary_auc,
    compute_calibration_summary,
    render_faithfulness_summary_markdown,
)
from ours.phase_b.feature_cache import (  # noqa: E402
    build_backbone_signature,
    build_cache_key,
    feature_cache_can_read,
    feature_cache_can_write,
    hash_float_list,
    hash_text_list,
    move_tensors_to_device,
    save_feature_cache,
    try_load_feature_cache,
    validate_feature_cache_mode,
)
from ours.phase_b.pik_data import (  # noqa: E402
    assert_phase_c_pik_compatibility,
    load_phase_c_pik_manifest,
    load_pik_supervision_examples,
)
from ours.phase_b.posthoc_calibration import (  # noqa: E402
    IsotonicCalibrationConfig,
    TemperatureCalibrationConfig,
    apply_posthoc_calibration,
    fit_isotonic_calibrator,
    fit_temperature_scaler,
)
from ours.phase_b.value_head import (  # noqa: E402
    encode_text_features,
    ensure_tokenizer_has_pad_token,
    freeze_backbone,
    load_value_head_checkpoint,
    maybe_resize_embeddings_for_tokenizer,
    resolve_model_input_device,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone P(IK) evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved P(IK) value head on one P(IK) C1 artifact directory."
    )
    parser.add_argument("--value-run-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", choices=["best", "final"], default="best")
    parser.add_argument("--run-name", default="phase_c_pik_eval")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_c_pik_eval"),
    )
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--feature-cache-root",
        type=Path,
        default=Path("assets/artifacts/phase_c_feature_cache"),
        help="Persistent feature-cache root for standalone P(IK) eval feature encoding.",
    )
    parser.add_argument(
        "--feature-cache-mode",
        choices=["off", "read", "write", "read_write"],
        default="read_write",
        help="Feature-cache behavior for standalone P(IK) eval.",
    )
    parser.add_argument(
        "--feature-cache-lock-timeout-sec",
        type=float,
        default=600.0,
        help="Lock wait timeout for safe concurrent cache writes.",
    )
    parser.add_argument("--known-threshold", type=float, default=None)
    parser.add_argument(
        "--posthoc-calibration",
        choices=["none", "temperature", "isotonic", "from_run"],
        default="none",
        help=(
            "Optional post-hoc calibration mode. "
            "`temperature`/`isotonic` fit on this eval set; `from_run` loads saved calibrator."
        ),
    )
    parser.add_argument("--posthoc-temperature-lr", type=float, default=0.05)
    parser.add_argument("--posthoc-temperature-max-iters", type=int, default=200)
    parser.add_argument("--posthoc-temperature-min", type=float, default=0.05)
    parser.add_argument("--posthoc-temperature-max", type=float, default=10.0)
    parser.add_argument("--posthoc-isotonic-min-points", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run standalone P(IK) re-evaluation."""
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
    args.feature_cache_mode = validate_feature_cache_mode(str(args.feature_cache_mode))
    if float(args.feature_cache_lock_timeout_sec) <= 0.0:
        raise ValueError("--feature-cache-lock-timeout-sec must be > 0")

    # 先做 run/eval artifact 兼容性检查，再做任何推理。
    manifest_path = args.value_run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Value-run manifest not found: {manifest_path}")
    value_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_manifest = load_phase_c_pik_manifest(Path(value_manifest["train_dir"]))
    eval_manifest = load_phase_c_pik_manifest(args.eval_dir)
    assert_phase_c_pik_compatibility(train_manifest, eval_manifest)

    train_metrics_path = args.value_run_dir / "train_metrics.json"
    if not train_metrics_path.exists():
        raise FileNotFoundError(f"Expected train_metrics.json in {args.value_run_dir}")
    train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))
    train_target_mean = float(train_metrics["train_target_mean"])

    # `best` 不存在时回退到 final，保证中断训练也可复评。
    checkpoint_name = "best_value_head.pt" if args.checkpoint_name == "best" else "final_value_head.pt"
    checkpoint_path = args.value_run_dir / checkpoint_name
    checkpoint_fallback = False
    if args.checkpoint_name == "best" and not checkpoint_path.exists():
        checkpoint_path = args.value_run_dir / "final_value_head.pt"
        checkpoint_fallback = True
    value_head, _, _ = load_value_head_checkpoint(checkpoint_path)

    eval_examples, _ = load_pik_supervision_examples(
        args.eval_dir,
        max_samples=args.max_eval_samples,
    )

    torch, AutoModelForCausalLM, AutoTokenizer = _import_runtime_deps()
    resolved = value_manifest["resolved_backbone"]
    max_length = int(args.max_length if args.max_length is not None else value_manifest["train_config"]["max_length"])
    known_threshold = float(
        args.known_threshold
        if args.known_threshold is not None
        else value_manifest["train_config"]["known_threshold"]
    )
    if not (0.0 <= known_threshold <= 1.0):
        raise ValueError("known_threshold must be in [0, 1]")

    tokenizer_path = _resolve_tokenizer_load_path(
        model_path=str(resolved["model_path"]),
        adapter_path=(Path(resolved["adapter_path"]) if resolved.get("adapter_path") else None),
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    synthesized_pad_token = ensure_tokenizer_has_pad_token(tokenizer)

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
    if synthesized_pad_token:
        maybe_resize_embeddings_for_tokenizer(backbone=backbone, tokenizer=tokenizer)
    adapter_path = resolved.get("adapter_path")
    if adapter_path:
        backbone = _attach_peft_adapter_for_inference(backbone, Path(adapter_path))
    freeze_backbone(backbone)

    feature_cache_stats: dict[str, Any] = {
        "mode": str(args.feature_cache_mode),
        "root": str(args.feature_cache_root),
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "entries": {},
    }
    feature_cache_root = Path(args.feature_cache_root)
    eval_batch_size = int(value_manifest["train_config"]["per_device_eval_batch_size"])
    eval_texts = [example.model_input_text() for example in eval_examples]
    backbone_signature = build_backbone_signature(
        model_path=str(resolved["model_path"]),
        adapter_path=(str(resolved["adapter_path"]) if resolved.get("adapter_path") else None),
        tokenizer_path=str(tokenizer_path),
        dtype=str(resolved["dtype"]),
        max_length=int(max_length),
    )
    cache_signature = _build_eval_feature_cache_signature_payload(
        texts=eval_texts,
        examples=eval_examples,
        max_length=int(max_length),
        backbone_signature=backbone_signature,
    )
    cache_key, signature_hash = build_cache_key("phase_c_pik_eval_features", cache_signature)
    features = None
    if feature_cache_can_read(str(args.feature_cache_mode)):
        cached_payload, _, _ = try_load_feature_cache(
            cache_root=feature_cache_root,
            cache_key=cache_key,
            expected_signature_hash=signature_hash,
            torch_module=torch,
        )
        if cached_payload is not None:
            try:
                _validate_cached_feature_tensor_payload(
                    payload=cached_payload,
                    expected_rows=len(eval_texts),
                    torch_module=torch,
                )
                features = move_tensors_to_device(cached_payload, resolve_model_input_device(backbone), torch)
                feature_cache_stats["hits"] += 1
                feature_cache_stats["entries"]["eval_features"] = {
                    "status": "hit",
                    "cache_key": cache_key,
                }
                print(f"feature_cache    : eval_features hit ({cache_key})", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"feature_cache    : eval_features invalid payload, fallback to re-encode ({exc})",
                    flush=True,
                )
    if features is None:
        feature_cache_stats["misses"] += 1
        feature_cache_stats["entries"]["eval_features"] = {
            "status": "miss",
            "cache_key": cache_key,
        }
        print(f"feature_cache    : eval_features miss ({cache_key})", flush=True)
        features = _encode_text_list(
            texts=eval_texts,
            backbone=backbone,
            tokenizer=tokenizer,
            torch_module=torch,
            max_length=max_length,
            batch_size=eval_batch_size,
        )
        if feature_cache_can_write(str(args.feature_cache_mode)):
            save_feature_cache(
                cache_root=feature_cache_root,
                cache_key=cache_key,
                signature_hash=signature_hash,
                payload=features,
                torch_module=torch,
                producer="scripts/phase_c_eval_pik.py:eval_features",
                lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
                extra_metadata={"num_rows": int(len(eval_texts))},
            )
            feature_cache_stats["writes"] += 1
            feature_cache_stats["entries"]["eval_features"]["write"] = True

    value_head.to(features.device)
    value_head.eval()
    with torch.no_grad():
        outputs = value_head(features)
        scores_tensor = outputs["scores"]
        logits_tensor = outputs["logits"]
        scores = [float(x) for x in scores_tensor.detach().cpu().tolist()]
        logits = [float(x) for x in logits_tensor.detach().cpu().tolist()]

    targets = [float(example.target_success_rate) for example in eval_examples]
    calibration_raw = compute_calibration_summary(scores, targets, reference_mean=float(train_target_mean))
    known_labels = [1 if float(t) >= float(known_threshold) else 0 for t in targets]
    known_auc = compute_binary_auc(scores=scores, labels=known_labels)

    posthoc_payload = _resolve_posthoc_payload(
        args=args,
        value_run_dir=args.value_run_dir,
        checkpoint_name=args.checkpoint_name,
        logits=logits_tensor,
        scores=scores_tensor,
        targets=features.new_tensor(targets),
        torch_module=torch,
    )
    calibration_posthoc = None
    posthoc_scores: list[float] | None = None
    known_auc_posthoc = None
    if posthoc_payload is not None:
        posthoc_scores_tensor = apply_posthoc_calibration(
            logits=logits_tensor,
            scores=scores_tensor,
            calibrator=posthoc_payload,
            torch_module=torch,
        )
        posthoc_scores = [float(x) for x in posthoc_scores_tensor.detach().cpu().tolist()]
        calibration_posthoc = compute_calibration_summary(
            posthoc_scores,
            targets,
            reference_mean=float(train_target_mean),
        )
        known_auc_posthoc = compute_binary_auc(scores=posthoc_scores, labels=known_labels)

    run_dir = args.output_root / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    question_scores_path = run_dir / "question_scores.jsonl"
    summary_md_path = run_dir / "summary.md"
    out_manifest_path = run_dir / "manifest.json"

    rows = []
    for idx, (example, score, logit) in enumerate(zip(eval_examples, scores, logits, strict=True)):
        row = {
            "sample_id": example.sample_id,
            "dataset": example.dataset,
            "split": example.split,
            "question": example.question,
            "predicted_value": float(score),
            "predicted_value_raw": float(logit),
            "predicted_logit": float(logit),
            "target_success_rate": float(example.target_success_rate),
            "target_parseable_rate": float(example.target_parseable_rate),
            "target_k_rollouts": int(example.target_k_rollouts),
            "known_label": int(known_labels[idx]),
        }
        if posthoc_scores is not None:
            row["predicted_value_posthoc"] = float(posthoc_scores[idx])
        rows.append(row)
    _write_jsonl(question_scores_path, rows)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "value_run_dir": str(args.value_run_dir),
        "eval_dir": str(args.eval_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_fallback": bool(checkpoint_fallback),
        "feature_cache": feature_cache_stats,
        "calibration": calibration_raw,
        "calibration_posthoc": calibration_posthoc,
        "posthoc_calibration": posthoc_payload,
        "known_threshold": float(known_threshold),
        "known_auc": float(known_auc),
        "known_auc_posthoc": (float(known_auc_posthoc) if known_auc_posthoc is not None else None),
        "n_eval_examples": len(eval_examples),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_md_path.write_text(
        render_faithfulness_summary_markdown(
            title="Phase C P(IK) Standalone Evaluation",
            calibration=calibration_raw,
            corruption=None,
            metadata={
                "value_run_dir": str(args.value_run_dir),
                "eval_dir": str(args.eval_dir),
                "checkpoint_path": str(checkpoint_path),
                "known_threshold": float(known_threshold),
                "known_auc": f"{known_auc:.6f}",
                "posthoc_known_auc": (f"{known_auc_posthoc:.6f}" if known_auc_posthoc is not None else "n/a"),
                "posthoc_calibration": args.posthoc_calibration,
                "feature_cache_mode": args.feature_cache_mode,
                "feature_cache_root": str(args.feature_cache_root),
            },
        ),
        encoding="utf-8",
    )

    out_manifest_path.write_text(
        json.dumps(
            {
                "artifact_stage": "phase_c_pik_eval",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "script": "scripts/phase_c_eval_pik.py",
                "run_name": args.run_name,
                "input": {
                    "value_run_dir": str(args.value_run_dir),
                    "eval_dir": str(args.eval_dir),
                    "checkpoint_name": args.checkpoint_name,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_fallback": bool(checkpoint_fallback),
                },
                "feature_cache_mode": str(args.feature_cache_mode),
                "feature_cache_root": str(args.feature_cache_root),
                "output_files": {
                    "metrics": str(metrics_path),
                    "question_scores": str(question_scores_path),
                    "summary": str(summary_md_path),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print("Phase C: Eval P(IK)")
    print("=" * 88)
    print(f"value_run_dir     : {args.value_run_dir}")
    print(f"eval_dir          : {args.eval_dir}")
    print(f"checkpoint_path   : {checkpoint_path}")
    print(f"feature_cache_mode: {args.feature_cache_mode}")
    print(f"feature_cache_root: {args.feature_cache_root}")
    print(
        "feature_cache_use : "
        f"hits={int(feature_cache_stats['hits'])} "
        f"misses={int(feature_cache_stats['misses'])} "
        f"writes={int(feature_cache_stats['writes'])}"
    )
    print(f"brier_score       : {calibration_raw['brier_score']:.6f}")
    print(f"pearson           : {calibration_raw['pearson']:.6f}")
    if calibration_posthoc is not None:
        print(f"posthoc_brier     : {calibration_posthoc['brier_score']:.6f}")
        print(f"posthoc_pearson   : {calibration_posthoc['pearson']:.6f}")
    print(f"known_auc         : {known_auc:.6f}")
    if known_auc_posthoc is not None:
        print(f"known_auc_posthoc : {known_auc_posthoc:.6f}")
    print(f"metrics_path      : {metrics_path}")
    print("=" * 88)
    return 0


def _resolve_posthoc_payload(
    *,
    args: argparse.Namespace,
    value_run_dir: Path,
    checkpoint_name: str,
    logits: Any,
    scores: Any,
    targets: Any,
    torch_module: Any,
) -> dict[str, Any] | None:
    """Resolve optional standalone post-hoc payload according to CLI mode."""
    if args.posthoc_calibration == "none":
        return None
    if args.posthoc_calibration == "temperature":
        # 在当前 eval 集上拟合温度，仅用于诊断与部署校准。
        cfg = TemperatureCalibrationConfig(
            lr=float(args.posthoc_temperature_lr),
            max_iters=int(args.posthoc_temperature_max_iters),
            min_temperature=float(args.posthoc_temperature_min),
            max_temperature=float(args.posthoc_temperature_max),
            init_temperature=1.0,
        )
        cfg.validate()
        return fit_temperature_scaler(
            logits=logits,
            targets=targets,
            torch_module=torch_module,
            config=cfg,
        )
    if args.posthoc_calibration == "isotonic":
        # 分段单调映射，适合非线性校准偏差。
        cfg = IsotonicCalibrationConfig(min_points=int(args.posthoc_isotonic_min_points))
        cfg.validate()
        return fit_isotonic_calibrator(
            scores=scores,
            targets=targets,
            torch_module=torch_module,
            config=cfg,
        )
    if args.posthoc_calibration == "from_run":
        # 复用训练 run 中保存的 calibrator，验证可复现部署路径。
        name = "best_posthoc_calibration.json" if checkpoint_name == "best" else "final_posthoc_calibration.json"
        path = value_run_dir / name
        if not path.exists() and checkpoint_name == "best":
            path = value_run_dir / "final_posthoc_calibration.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Requested --posthoc-calibration=from_run but no saved calibrator was found at {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Saved post-hoc payload must be a JSON object: {path}")
        return payload
    raise ValueError(f"Unsupported --posthoc-calibration: {args.posthoc_calibration!r}")


def _build_eval_feature_cache_signature_payload(
    *,
    texts: list[str],
    examples: list[Any],
    max_length: int,
    backbone_signature: dict[str, Any],
) -> dict[str, Any]:
    """Build conservative signature payload for standalone P(IK) eval features."""
    return {
        "cache_kind": "phase_c_pik_eval_features",
        "backbone_signature": backbone_signature,
        "max_length": int(max_length),
        "num_rows": int(len(texts)),
        "sample_id_hash": hash_text_list([str(example.sample_id) for example in examples]),
        "question_hash": hash_text_list([str(example.question) for example in examples]),
        "text_hash": hash_text_list(texts),
        "target_success_hash": hash_float_list([float(example.target_success_rate) for example in examples]),
    }


def _validate_cached_feature_tensor_payload(
    *,
    payload: Any,
    expected_rows: int,
    torch_module: Any,
) -> None:
    """Validate cached standalone-eval feature tensor before reusing."""
    if not torch_module.is_tensor(payload):
        raise TypeError("Cached eval feature payload must be tensor")
    if payload.ndim != 2:
        raise ValueError(
            f"Cached eval feature tensor must have shape [batch, hidden], got {tuple(payload.shape)!r}"
        )
    if int(payload.shape[0]) != int(expected_rows):
        raise ValueError(
            f"Cached eval feature row mismatch: expected {expected_rows}, got {int(payload.shape[0])}"
        )


def _encode_text_list(
    *,
    texts: list[str],
    backbone: Any,
    tokenizer: Any,
    torch_module: Any,
    max_length: int,
    batch_size: int,
):
    """Encode texts to pooled features using small deterministic batches."""
    chunks = []
    total = len(texts)
    if total == 0:
        return torch_module.zeros((0, 1), dtype=torch_module.float32)
    total_batches = (total + batch_size - 1) // batch_size
    progress_every = max(1, math.ceil(total_batches / 8))
    started = time.time()
    print(
        "cache_eval_questions: "
        f"start {total} texts in {total_batches} batches "
        f"(bs={batch_size}, progress_every~{progress_every})",
        flush=True,
    )
    for batch_idx, start in enumerate(range(0, total, batch_size), start=1):
        chunk = texts[start : start + batch_size]
        chunks.append(
            encode_text_features(
                backbone=backbone,
                tokenizer=tokenizer,
                texts=chunk,
                max_length=max_length,
                torch_module=torch_module,
            )
        )
        if batch_idx % progress_every == 0 or batch_idx == total_batches:
            elapsed = max(time.time() - started, 1e-6)
            done = min(batch_idx * batch_size, total)
            rate = done / elapsed
            print(
                "cache_eval_questions: "
                f"{batch_idx}/{total_batches} batches ({done}/{total}, {done / total:.1%}) | "
                f"elapsed={elapsed:.1f}s | rate={rate:.3f} text/s",
                flush=True,
            )
    print("cache_eval_questions: done", flush=True)
    return torch_module.cat(chunks, dim=0).to(dtype=torch_module.float32)


def _import_runtime_deps():
    """Import heavy runtime dependencies lazily."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(name: str, torch_module: Any):
    """Map user-facing dtype string onto one torch dtype object."""
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
    """Choose tokenizer source directory (adapter tokenizer overrides base)."""
    if adapter_path is None:
        return str(model_path)
    if (adapter_path / "tokenizer_config.json").exists():
        return str(adapter_path)
    return str(model_path)


def _attach_peft_adapter_for_inference(model: Any, adapter_path: Path):
    """Attach one PEFT adapter to loaded backbone for feature encoding."""
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    try:
        from peft import PeftModel  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to import `peft` while attaching adapter for P(IK) eval") from exc
    return PeftModel.from_pretrained(model, str(adapter_path))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries as UTF-8 JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
