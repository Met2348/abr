#!/usr/bin/env python3
"""Train a question-level P(IK) value head on frozen-backbone features.

Why this file exists
--------------------
This script is the "can the head learn anything at all?" diagnostic for Phase C.
Compared with prefix-level faithfulness training, this path is intentionally
simpler:
- question-level examples only,
- Monte Carlo success-rate supervision,
- calibration-focused objective,
- no corruption contrastive branch.

What this file does
-------------------
1. Load train/eval P(IK) artifact directories.
2. Validate rollout-backbone provenance compatibility.
3. Encode question prompts with a frozen backbone.
4. Train a sigmoid value head with BCE/MSE-style calibration loss.
5. Evaluate Brier/Pearson/ECE and known-vs-unknown AUROC.
6. Persist checkpoints, metrics, run manifest, and Markdown summary.

Interaction with other files
----------------------------
- `scripts/phase_c_prepare_pik_data.py`: writes P(IK) C1 artifacts
- `src/ours/phase_b/pik_data.py`: P(IK) data contracts/loaders
- `src/ours/phase_b/value_head.py`: frozen-backbone feature extraction + head
- `src/ours/phase_b/value_losses.py`: calibration loss helpers
- `src/ours/phase_b/faithfulness_eval.py`: shared calibration metrics + AUC

Example
-------
```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/phase_c_train_pik.py \
  --train-dir assets/artifacts/phase_c_pik_data/strategyqa/strategyqa_pik_train__<fp> \
  --eval-dir assets/artifacts/phase_c_pik_data/strategyqa/strategyqa_pik_eval__<fp> \
  --run-name strategyqa_pik_c2 \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --per-device-train-batch-size 192 \
  --per-device-eval-batch-size 192 \
  --learning-rate 1e-4 \
  --num-train-epochs 8 \
  --calibration-loss bce
```
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
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
    PIKSupervisionExample,
    assert_phase_c_pik_compatibility,
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
    SigmoidValueHead,
    ValueHeadConfig,
    encode_text_features,
    ensure_tokenizer_has_pad_token,
    freeze_backbone,
    infer_backbone_hidden_size,
    maybe_resize_embeddings_for_tokenizer,
    save_value_head_checkpoint,
    write_value_head_config_json,
)
from ours.phase_b.value_losses import (  # noqa: E402
    binary_cross_entropy_calibration_loss,
    mean_squared_calibration_loss,
    mixed_calibration_loss,
)


@dataclass(slots=True)
class TrainConfig:
    """Compact P(IK) C2 hyperparameter snapshot persisted in run manifests."""

    max_length: int
    learning_rate: float
    weight_decay: float
    num_train_epochs: float
    max_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    max_grad_norm: float
    calibration_loss: str
    calibration_mse_weight: float
    calibration_bce_weight: float
    calibration_bce_pos_weight: float
    calibration_target_smoothing: float
    known_threshold: float
    checkpoint_selection_metric: str
    posthoc_calibration: str
    posthoc_temperature_lr: float
    posthoc_temperature_max_iters: int
    posthoc_temperature_min: float
    posthoc_temperature_max: float
    posthoc_isotonic_min_points: int
    dropout_prob: float
    seed: int
    dtype: str
    device_map: str
    require_cuda: bool
    logging_steps: int
    feature_cache_root: str
    feature_cache_mode: str
    feature_cache_lock_timeout_sec: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for P(IK) C2 training."""
    parser = argparse.ArgumentParser(
        description="Train a question-level P(IK) value head on frozen-backbone features."
    )
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--train-dir", type=Path, required=False, default=None)
    parser.add_argument("--eval-dir", type=Path, required=False, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_c_pik_runs"),
        help="Output root for P(IK) C2 run artifacts.",
    )
    parser.add_argument("--run-name", default="phase_c_pik")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--feature-cache-root",
        type=Path,
        default=Path("assets/artifacts/phase_c_feature_cache"),
        help="Persistent feature-cache root for frozen P(IK) question encoding.",
    )
    parser.add_argument(
        "--feature-cache-mode",
        choices=["off", "read", "write", "read_write"],
        default="read_write",
        help="Feature-cache behavior for P(IK) C2.",
    )
    parser.add_argument(
        "--feature-cache-lock-timeout-sec",
        type=float,
        default=600.0,
        help="Lock wait timeout for safe concurrent cache writes.",
    )

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-train-epochs", type=float, default=8.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=192)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=192)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--calibration-loss",
        choices=["bce", "mse", "bce_mse"],
        default="bce",
        help="Calibration objective for P(IK) targets.",
    )
    parser.add_argument("--calibration-mse-weight", type=float, default=1.0)
    parser.add_argument("--calibration-bce-weight", type=float, default=1.0)
    parser.add_argument("--calibration-bce-pos-weight", type=float, default=1.0)
    parser.add_argument(
        "--calibration-target-smoothing",
        type=float,
        default=0.0,
        help="Optional target smoothing epsilon: y'=(1-2e)*y+e.",
    )

    parser.add_argument(
        "--known-threshold",
        type=float,
        default=0.5,
        help="Threshold on empirical success-rate used to derive binary known/unknown labels for AUROC.",
    )
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=["raw_brier", "posthoc_brier"],
        default="raw_brier",
    )
    parser.add_argument(
        "--posthoc-calibration",
        choices=["none", "temperature", "isotonic"],
        default="none",
    )
    parser.add_argument("--posthoc-temperature-lr", type=float, default=0.05)
    parser.add_argument("--posthoc-temperature-max-iters", type=int, default=200)
    parser.add_argument("--posthoc-temperature-min", type=float, default=0.05)
    parser.add_argument("--posthoc-temperature-max", type=float, default=10.0)
    parser.add_argument("--posthoc-isotonic-min-points", type=int, default=32)

    parser.add_argument("--dropout-prob", type=float, default=0.0)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail fast if CUDA is not available.",
    )
    parser.add_argument(
        "--save-best-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist best checkpoint separately from final checkpoint.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments with optional config-json defaults."""
    parser = _build_parser()
    # 两段解析：先读 config 默认值，再由 CLI 覆盖。
    partial, _ = parser.parse_known_args(argv)
    if partial.config_json is not None:
        defaults = _load_config_defaults(partial.config_json)
        valid_keys = {action.dest for action in parser._actions}  # noqa: SLF001
        unknown = sorted(set(defaults.keys()) - valid_keys)
        if unknown:
            raise KeyError(f"Unknown keys in config JSON {partial.config_json}: {unknown}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)

    if args.train_dir is None or args.eval_dir is None:
        parser.error("the following arguments are required: --train-dir, --eval-dir")
    if args.max_length <= 8:
        raise ValueError("--max-length must be > 8")
    if args.per_device_train_batch_size <= 0 or args.per_device_eval_batch_size <= 0:
        raise ValueError("Per-device batch sizes must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be > 0")
    if args.num_train_epochs <= 0.0 and args.max_steps <= 0:
        raise ValueError("Set positive --num-train-epochs or positive --max-steps")
    if args.calibration_mse_weight < 0.0:
        raise ValueError("--calibration-mse-weight must be >= 0")
    if args.calibration_bce_weight < 0.0:
        raise ValueError("--calibration-bce-weight must be >= 0")
    if args.calibration_bce_pos_weight <= 0.0:
        raise ValueError("--calibration-bce-pos-weight must be > 0")
    if not (0.0 <= args.calibration_target_smoothing < 0.5):
        raise ValueError("--calibration-target-smoothing must be in [0, 0.5)")
    if args.calibration_loss == "bce_mse" and (
        args.calibration_mse_weight == 0.0 and args.calibration_bce_weight == 0.0
    ):
        raise ValueError("bce_mse requires at least one non-zero calibration component weight")
    if not (0.0 <= args.known_threshold <= 1.0):
        raise ValueError("--known-threshold must be in [0, 1]")
    if args.checkpoint_selection_metric == "posthoc_brier" and args.posthoc_calibration == "none":
        raise ValueError("posthoc_brier selection requires --posthoc-calibration")
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
    if not (0.0 <= args.dropout_prob < 1.0):
        raise ValueError("--dropout-prob must be in [0, 1)")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run full P(IK) C2 training + epoch-level eval workflow."""
    args = parse_args(argv)

    # Stage 1: 先检查 train/eval 产物契约一致，再触发模型加载。
    train_examples, train_manifest = load_pik_supervision_examples(
        args.train_dir,
        max_samples=args.max_train_samples,
    )
    eval_examples, eval_manifest = load_pik_supervision_examples(
        args.eval_dir,
        max_samples=args.max_eval_samples,
    )
    assert_phase_c_pik_compatibility(train_manifest, eval_manifest)

    rollout_config = dict(train_manifest["rollout_config"])
    model_path = str(rollout_config["model_path"])
    adapter_path = rollout_config.get("adapter_path")

    torch, AutoModelForCausalLM, AutoTokenizer = _import_runtime_deps()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for P(IK) C2 encoding, but no GPU is visible")
    _set_seed(args.seed, torch)

    train_cfg = TrainConfig(
        max_length=int(args.max_length),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        num_train_epochs=float(args.num_train_epochs),
        max_steps=int(args.max_steps),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        warmup_ratio=float(args.warmup_ratio),
        max_grad_norm=float(args.max_grad_norm),
        calibration_loss=str(args.calibration_loss),
        calibration_mse_weight=float(args.calibration_mse_weight),
        calibration_bce_weight=float(args.calibration_bce_weight),
        calibration_bce_pos_weight=float(args.calibration_bce_pos_weight),
        calibration_target_smoothing=float(args.calibration_target_smoothing),
        known_threshold=float(args.known_threshold),
        checkpoint_selection_metric=str(args.checkpoint_selection_metric),
        posthoc_calibration=str(args.posthoc_calibration),
        posthoc_temperature_lr=float(args.posthoc_temperature_lr),
        posthoc_temperature_max_iters=int(args.posthoc_temperature_max_iters),
        posthoc_temperature_min=float(args.posthoc_temperature_min),
        posthoc_temperature_max=float(args.posthoc_temperature_max),
        posthoc_isotonic_min_points=int(args.posthoc_isotonic_min_points),
        dropout_prob=float(args.dropout_prob),
        seed=int(args.seed),
        dtype=str(args.dtype),
        device_map=str(args.device_map),
        require_cuda=bool(args.require_cuda),
        logging_steps=int(args.logging_steps),
        feature_cache_root=str(args.feature_cache_root),
        feature_cache_mode=str(args.feature_cache_mode),
        feature_cache_lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
    )

    run_dir = args.output_root / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    train_metrics_path = run_dir / "train_metrics.json"
    eval_metrics_path = run_dir / "eval_metrics.json"
    summary_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    best_ckpt_path = run_dir / "best_value_head.pt"
    final_ckpt_path = run_dir / "final_value_head.pt"
    best_posthoc_path = run_dir / "best_posthoc_calibration.json"
    final_posthoc_path = run_dir / "final_posthoc_calibration.json"
    config_json_path = run_dir / "value_head_config.json"
    question_scores_path = run_dir / "eval_question_scores.jsonl"
    train_curve_path = run_dir / "train_curve.jsonl"

    print("=" * 88)
    print("Phase C: Train P(IK) Value Head")
    print("=" * 88)
    print(f"train_dir         : {args.train_dir}")
    print(f"eval_dir          : {args.eval_dir}")
    print(f"run_dir           : {run_dir}")
    print(f"model_path        : {model_path}")
    print(f"adapter_path      : {adapter_path if adapter_path is not None else '<none>'}")
    print(f"train_examples    : {len(train_examples)}")
    print(f"eval_examples     : {len(eval_examples)}")
    print(f"calibration_loss  : {args.calibration_loss}")
    print(f"known_threshold   : {args.known_threshold}")
    print(f"posthoc_calib     : {args.posthoc_calibration}")
    print(f"ckpt_metric       : {args.checkpoint_selection_metric}")
    print(f"batch_train       : {args.per_device_train_batch_size}")
    print(f"batch_eval        : {args.per_device_eval_batch_size}")
    print(f"max_length        : {args.max_length}")
    print(f"feat_cache_mode   : {args.feature_cache_mode}")
    print(f"feat_cache_root   : {args.feature_cache_root}")
    print(f"seed              : {args.seed}")
    print("=" * 88)

    # Stage 2: 加载并冻结 backbone，编码 question features 到缓存。
    model_load_start = time.perf_counter()
    resolved_dtype = _resolve_dtype(args.dtype, torch)
    tokenizer_path = _resolve_tokenizer_load_path(model_path=model_path, adapter_path=(Path(adapter_path) if adapter_path else None))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    synthesized_pad_token = ensure_tokenizer_has_pad_token(tokenizer)

    model_load_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": True,
    }
    from_pretrained_sig = __import__("inspect").signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in from_pretrained_sig.parameters:
        model_load_kwargs["dtype"] = resolved_dtype
    else:
        model_load_kwargs["torch_dtype"] = resolved_dtype

    backbone = AutoModelForCausalLM.from_pretrained(model_path, **model_load_kwargs)
    if synthesized_pad_token:
        maybe_resize_embeddings_for_tokenizer(backbone=backbone, tokenizer=tokenizer)
    if adapter_path is not None:
        backbone = _attach_peft_adapter_for_inference(backbone, Path(adapter_path))
    freeze_backbone(backbone)
    hidden_size = infer_backbone_hidden_size(backbone)
    value_head_config = ValueHeadConfig(hidden_size=hidden_size, dropout_prob=args.dropout_prob)
    value_head = SigmoidValueHead(value_head_config)
    value_device = _resolve_value_device(backbone, torch)
    value_head.to(value_device)
    model_load_elapsed = time.perf_counter() - model_load_start

    feature_cache_start = time.perf_counter()
    feature_cache_stats: dict[str, Any] = {
        "mode": str(args.feature_cache_mode),
        "root": str(args.feature_cache_root),
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "entries": {},
    }
    feature_cache_root = Path(args.feature_cache_root)
    model_input_device = _resolve_value_device(backbone, torch)
    backbone_signature = build_backbone_signature(
        model_path=str(model_path),
        adapter_path=(str(adapter_path) if adapter_path is not None else None),
        tokenizer_path=str(tokenizer_path),
        dtype=str(args.dtype),
        max_length=int(args.max_length),
    )
    train_texts = [example.model_input_text() for example in train_examples]
    eval_texts = [example.model_input_text() for example in eval_examples]

    train_cache_signature = _build_pik_example_cache_signature_payload(
        cache_kind="phase_c_pik_train_cache",
        examples=train_examples,
        texts=train_texts,
        max_length=int(args.max_length),
        backbone_signature=backbone_signature,
    )
    train_cache_key, train_signature_hash = build_cache_key(
        "phase_c_pik_train_cache",
        train_cache_signature,
    )
    train_cache = None
    if feature_cache_can_read(str(args.feature_cache_mode)):
        cached_payload, _, _ = try_load_feature_cache(
            cache_root=feature_cache_root,
            cache_key=train_cache_key,
            expected_signature_hash=train_signature_hash,
            torch_module=torch,
        )
        if cached_payload is not None:
            try:
                _validate_cached_pik_example_cache_payload(
                    cache=cached_payload,
                    expected_num_examples=len(train_examples),
                    expected_hidden_size=int(hidden_size),
                    torch_module=torch,
                )
                train_cache = cached_payload
                feature_cache_stats["hits"] += 1
                feature_cache_stats["entries"]["train_cache"] = {
                    "status": "hit",
                    "cache_key": train_cache_key,
                }
                print(f"feature_cache    : train_cache hit ({train_cache_key})", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"feature_cache    : train_cache invalid payload, fallback to re-encode ({exc})",
                    flush=True,
                )
    if train_cache is None:
        feature_cache_stats["misses"] += 1
        feature_cache_stats["entries"]["train_cache"] = {
            "status": "miss",
            "cache_key": train_cache_key,
        }
        print(f"feature_cache    : train_cache miss ({train_cache_key})", flush=True)
        train_cache = _encode_example_cache(
            examples=train_examples,
            texts=train_texts,
            backbone=backbone,
            tokenizer=tokenizer,
            torch_module=torch,
            max_length=args.max_length,
            batch_size=args.per_device_eval_batch_size,
            progress_label="cache_train_questions",
        )
        if feature_cache_can_write(str(args.feature_cache_mode)):
            save_feature_cache(
                cache_root=feature_cache_root,
                cache_key=train_cache_key,
                signature_hash=train_signature_hash,
                payload=train_cache,
                torch_module=torch,
                producer="scripts/phase_c_train_pik.py:train_cache",
                lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
                extra_metadata={"num_examples": int(len(train_examples))},
            )
            feature_cache_stats["writes"] += 1
            feature_cache_stats["entries"]["train_cache"]["write"] = True

    eval_cache_signature = _build_pik_example_cache_signature_payload(
        cache_kind="phase_c_pik_eval_cache",
        examples=eval_examples,
        texts=eval_texts,
        max_length=int(args.max_length),
        backbone_signature=backbone_signature,
    )
    eval_cache_key, eval_signature_hash = build_cache_key(
        "phase_c_pik_eval_cache",
        eval_cache_signature,
    )
    eval_cache = None
    if feature_cache_can_read(str(args.feature_cache_mode)):
        cached_payload, _, _ = try_load_feature_cache(
            cache_root=feature_cache_root,
            cache_key=eval_cache_key,
            expected_signature_hash=eval_signature_hash,
            torch_module=torch,
        )
        if cached_payload is not None:
            try:
                _validate_cached_pik_example_cache_payload(
                    cache=cached_payload,
                    expected_num_examples=len(eval_examples),
                    expected_hidden_size=int(hidden_size),
                    torch_module=torch,
                )
                eval_cache = cached_payload
                feature_cache_stats["hits"] += 1
                feature_cache_stats["entries"]["eval_cache"] = {
                    "status": "hit",
                    "cache_key": eval_cache_key,
                }
                print(f"feature_cache    : eval_cache hit ({eval_cache_key})", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"feature_cache    : eval_cache invalid payload, fallback to re-encode ({exc})",
                    flush=True,
                )
    if eval_cache is None:
        feature_cache_stats["misses"] += 1
        feature_cache_stats["entries"]["eval_cache"] = {
            "status": "miss",
            "cache_key": eval_cache_key,
        }
        print(f"feature_cache    : eval_cache miss ({eval_cache_key})", flush=True)
        eval_cache = _encode_example_cache(
            examples=eval_examples,
            texts=eval_texts,
            backbone=backbone,
            tokenizer=tokenizer,
            torch_module=torch,
            max_length=args.max_length,
            batch_size=args.per_device_eval_batch_size,
            progress_label="cache_eval_questions",
        )
        if feature_cache_can_write(str(args.feature_cache_mode)):
            save_feature_cache(
                cache_root=feature_cache_root,
                cache_key=eval_cache_key,
                signature_hash=eval_signature_hash,
                payload=eval_cache,
                torch_module=torch,
                producer="scripts/phase_c_train_pik.py:eval_cache",
                lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
                extra_metadata={"num_examples": int(len(eval_examples))},
            )
            feature_cache_stats["writes"] += 1
            feature_cache_stats["entries"]["eval_cache"]["write"] = True

    feature_cache_elapsed = time.perf_counter() - feature_cache_start

    # Keep cache tensors on CPU and free the frozen backbone before the head-only loop starts.
    # 长期缓存特征留在 CPU，编码结束后立即释放 frozen backbone，训练时只搬当前 mini-batch。
    del backbone
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Stage 3: 仅训练小 value head，并按 eval 指标保存 best checkpoint。
    optimizer = torch.optim.AdamW(
        [{"params": list(value_head.parameters()), "weight_decay": args.weight_decay}],
        lr=args.learning_rate,
    )
    total_steps = _resolve_total_train_steps(
        num_examples=len(train_examples),
        batch_size=args.per_device_train_batch_size,
        grad_accum_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _linear_warmup_decay(step, warmup_steps=warmup_steps, total_steps=total_steps),
    )
    posthoc_temperature_config = _build_temperature_calibration_config(args)
    posthoc_isotonic_config = _build_isotonic_calibration_config(args)

    train_curve: list[dict[str, Any]] = []
    train_target_mean = float(train_cache["targets"].mean().item())
    best_eval_selection_brier = float("inf")
    best_eval_metrics: dict[str, Any] | None = None
    best_eval_rows: list[dict[str, Any]] | None = None

    train_start = time.perf_counter()
    global_step = 0
    epoch_idx = 0
    optimizer.zero_grad(set_to_none=True)

    while global_step < total_steps:
        epoch_stats = _run_one_train_epoch(
            value_head=value_head,
            train_cache=train_cache,
            torch_module=torch,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=args.per_device_train_batch_size,
            grad_accum_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            calibration_loss=args.calibration_loss,
            calibration_mse_weight=args.calibration_mse_weight,
            calibration_bce_weight=args.calibration_bce_weight,
            calibration_bce_pos_weight=args.calibration_bce_pos_weight,
            calibration_target_smoothing=args.calibration_target_smoothing,
            logging_steps=args.logging_steps,
            global_step_offset=global_step,
            max_steps=(args.max_steps if args.max_steps > 0 else None),
        )
        global_step = int(epoch_stats["global_step"])
        epoch_idx += 1
        if int(epoch_stats["num_batches"]) <= 0:
            raise RuntimeError("P(IK) training made zero progress in one epoch pass")

        eval_metrics, eval_rows = _evaluate_pik_head(
            value_head=value_head,
            eval_cache=eval_cache,
            eval_examples=eval_examples,
            train_target_mean=train_target_mean,
            known_threshold=float(args.known_threshold),
            posthoc_calibration=args.posthoc_calibration,
            posthoc_temperature_config=posthoc_temperature_config,
            posthoc_isotonic_config=posthoc_isotonic_config,
            batch_size=int(args.per_device_eval_batch_size),
            torch_module=torch,
        )
        curve_row = {
            "epoch": epoch_idx,
            "global_step": global_step,
            "train": epoch_stats,
            "eval": eval_metrics,
        }
        train_curve.append(curve_row)
        _append_jsonl(train_curve_path, curve_row)

        current_eval_brier = _resolve_checkpoint_brier(
            eval_metrics=eval_metrics,
            checkpoint_selection_metric=args.checkpoint_selection_metric,
        )
        if current_eval_brier < best_eval_selection_brier:
            best_eval_selection_brier = float(current_eval_brier)
            best_eval_metrics = eval_metrics
            best_eval_rows = eval_rows
            if args.save_best_state:
                save_value_head_checkpoint(
                    best_ckpt_path,
                    value_head=value_head,
                    config=value_head_config,
                    extra_state={
                        "epoch": epoch_idx,
                        "global_step": global_step,
                        "eval_selection_brier": best_eval_selection_brier,
                    },
                )
                _write_posthoc_calibration_payload(
                    path=best_posthoc_path,
                    payload=eval_metrics.get("posthoc_calibration"),
                )

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    train_elapsed = time.perf_counter() - train_start
    save_value_head_checkpoint(
        final_ckpt_path,
        value_head=value_head,
        config=value_head_config,
        extra_state={
            "global_step": global_step,
            "train_elapsed_seconds": train_elapsed,
        },
    )
    write_value_head_config_json(config_json_path, value_head_config)

    final_eval_metrics, final_eval_rows = _evaluate_pik_head(
        value_head=value_head,
        eval_cache=eval_cache,
        eval_examples=eval_examples,
        train_target_mean=train_target_mean,
        known_threshold=float(args.known_threshold),
        posthoc_calibration=args.posthoc_calibration,
        posthoc_temperature_config=posthoc_temperature_config,
        posthoc_isotonic_config=posthoc_isotonic_config,
        batch_size=int(args.per_device_eval_batch_size),
        torch_module=torch,
    )
    _write_posthoc_calibration_payload(
        path=final_posthoc_path,
        payload=final_eval_metrics.get("posthoc_calibration"),
    )

    selected_eval_metrics = best_eval_metrics if best_eval_metrics is not None else final_eval_metrics
    selected_rows = best_eval_rows if best_eval_rows is not None else final_eval_rows
    _write_jsonl(question_scores_path, selected_rows)

    train_metrics = {
        "model_load_seconds": float(model_load_elapsed),
        "feature_cache_seconds": float(feature_cache_elapsed),
        "feature_cache": feature_cache_stats,
        "train_elapsed_seconds": float(train_elapsed),
        "global_step": int(global_step),
        "train_curve": train_curve,
        "train_target_mean": float(train_target_mean),
        "best_eval_selection_brier": float(best_eval_selection_brier),
        "checkpoint_selection_metric": str(args.checkpoint_selection_metric),
    }
    eval_metrics = {
        "selected_checkpoint": ("best" if best_eval_metrics is not None else "final"),
        "best_checkpoint_saved": bool(args.save_best_state),
        "feature_cache": feature_cache_stats,
        "best": best_eval_metrics,
        "final": final_eval_metrics,
    }
    train_metrics_path.write_text(json.dumps(train_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    eval_metrics_path.write_text(json.dumps(eval_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "artifact_stage": "phase_c_pik_c2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/phase_c_train_pik.py",
        "run_name": args.run_name,
        "train_dir": str(args.train_dir),
        "eval_dir": str(args.eval_dir),
        "train_dir_manifest": str(args.train_dir / "manifest.json"),
        "eval_dir_manifest": str(args.eval_dir / "manifest.json"),
        "resolved_backbone": {
            "model_path": model_path,
            "adapter_path": adapter_path,
            "dtype": args.dtype,
            "device_map": args.device_map,
        },
        "train_config": train_cfg.to_dict(),
        "value_head_config": value_head_config.to_dict(),
        "feature_cache": feature_cache_stats,
        "num_train_examples": len(train_examples),
        "num_eval_examples": len(eval_examples),
        "output_files": {
            "best_value_head": str(best_ckpt_path) if args.save_best_state else None,
            "final_value_head": str(final_ckpt_path),
            "best_posthoc_calibration": (str(best_posthoc_path) if best_posthoc_path.exists() else None),
            "final_posthoc_calibration": (str(final_posthoc_path) if final_posthoc_path.exists() else None),
            "value_head_config": str(config_json_path),
            "train_metrics": str(train_metrics_path),
            "eval_metrics": str(eval_metrics_path),
            "eval_question_scores": str(question_scores_path),
            "train_curve": str(train_curve_path),
            "summary": str(summary_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "selected_eval_metrics": selected_eval_metrics,
        "train_metrics": {
            "global_step": int(global_step),
            "train_elapsed_seconds": float(train_elapsed),
            "feature_cache_seconds": float(feature_cache_elapsed),
            "feature_cache": feature_cache_stats,
            "model_load_seconds": float(model_load_elapsed),
            "train_target_mean": float(train_target_mean),
            "checkpoint_selection_metric": str(args.checkpoint_selection_metric),
            "posthoc_calibration": str(args.posthoc_calibration),
            "known_threshold": float(args.known_threshold),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(
        render_faithfulness_summary_markdown(
            title="Phase C P(IK) C2 Summary",
            calibration=selected_eval_metrics["calibration"],
            corruption=None,
            metadata={
                "run_dir": str(run_dir),
                "train_examples": len(train_examples),
                "eval_examples": len(eval_examples),
                "known_threshold": float(args.known_threshold),
                "known_auc": f"{selected_eval_metrics['known_auc']:.6f}",
                "posthoc_known_auc": (
                    f"{selected_eval_metrics['known_auc_posthoc']:.6f}"
                    if selected_eval_metrics.get("known_auc_posthoc") is not None
                    else "n/a"
                ),
                "calibration_loss": args.calibration_loss,
                "posthoc_calibration": args.posthoc_calibration,
                "checkpoint_selection_metric": args.checkpoint_selection_metric,
                "feature_cache_mode": args.feature_cache_mode,
                "feature_cache_root": str(args.feature_cache_root),
            },
        ),
        encoding="utf-8",
    )

    print("-" * 88)
    print(f"global_step       : {global_step}")
    print(f"train_elapsed_sec : {train_elapsed:.2f}")
    print(f"feature_cache_sec : {feature_cache_elapsed:.2f}")
    print(
        "feature_cache_use : "
        f"hits={int(feature_cache_stats['hits'])} "
        f"misses={int(feature_cache_stats['misses'])} "
        f"writes={int(feature_cache_stats['writes'])}"
    )
    print(f"selected_brier    : {selected_eval_metrics['calibration']['brier_score']:.6f}")
    print(f"selected_pearson  : {selected_eval_metrics['calibration']['pearson']:.6f}")
    if selected_eval_metrics.get("calibration_posthoc") is not None:
        print(f"selected_post_bri : {selected_eval_metrics['calibration_posthoc']['brier_score']:.6f}")
        print(f"selected_post_prs : {selected_eval_metrics['calibration_posthoc']['pearson']:.6f}")
    print(f"known_auc         : {selected_eval_metrics['known_auc']:.6f}")
    if selected_eval_metrics.get("known_auc_posthoc") is not None:
        print(f"known_auc_posthoc : {selected_eval_metrics['known_auc_posthoc']:.6f}")
    print(f"train_metrics     : {train_metrics_path}")
    print(f"eval_metrics      : {eval_metrics_path}")
    print(f"manifest          : {manifest_path}")
    print(f"summary           : {summary_path}")
    print("=" * 88)
    return 0


def _move_batch_tensor(
    tensor: Any,
    *,
    device: Any,
    dtype: Any | None = None,
) -> Any:
    """Move one batch tensor to the requested device/dtype only when needed."""
    target_dtype = tensor.dtype if dtype is None else dtype
    if tensor.device == device and tensor.dtype == target_dtype:
        return tensor
    return tensor.to(device=device, dtype=target_dtype)


def _score_features_with_logits_in_batches(
    *,
    value_head: Any,
    features: Any,
    batch_size: int,
    torch_module: Any,
) -> tuple[Any, Any]:
    """Score feature rows in mini-batches and return CPU score/logit tensors."""
    head_device = next(value_head.parameters()).device
    head_dtype = next(value_head.parameters()).dtype
    score_chunks: list[Any] = []
    logit_chunks: list[Any] = []
    effective_bs = max(1, int(batch_size))
    for start in range(0, int(features.shape[0]), effective_bs):
        batch = _move_batch_tensor(
            features[start : start + effective_bs],
            device=head_device,
            dtype=head_dtype,
        )
        outputs = value_head(batch)
        score_chunks.append(outputs["scores"].detach().cpu())
        logit_chunks.append(outputs["logits"].detach().cpu())
    if score_chunks:
        return torch_module.cat(score_chunks, dim=0), torch_module.cat(logit_chunks, dim=0)
    return (
        torch_module.zeros((0,), dtype=torch_module.float32),
        torch_module.zeros((0,), dtype=torch_module.float32),
    )


def _run_one_train_epoch(
    *,
    value_head: Any,
    train_cache: dict[str, Any],
    torch_module: Any,
    optimizer: Any,
    scheduler: Any,
    batch_size: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    calibration_loss: str,
    calibration_mse_weight: float,
    calibration_bce_weight: float,
    calibration_bce_pos_weight: float,
    calibration_target_smoothing: float,
    logging_steps: int,
    global_step_offset: int,
    max_steps: int | None,
) -> dict[str, Any]:
    """Run one epoch over cached question features."""
    value_head.train()
    num_examples = int(train_cache["features"].shape[0])
    head_device = next(value_head.parameters()).device
    head_dtype = next(value_head.parameters()).dtype
    permutation = torch_module.randperm(num_examples, device=train_cache["features"].device)

    running_total = 0.0
    batches = 0
    optimizer_steps = 0

    for batch_start in range(0, num_examples, batch_size):
        if max_steps is not None and global_step_offset + optimizer_steps >= max_steps:
            break
        batch_indices = permutation[batch_start : batch_start + batch_size]
        features = _move_batch_tensor(
            train_cache["features"][batch_indices],
            device=head_device,
            dtype=head_dtype,
        )
        targets = _apply_target_smoothing(
            _move_batch_tensor(
                train_cache["targets"][batch_indices],
                device=head_device,
                dtype=torch_module.float32,
            ),
            epsilon=float(calibration_target_smoothing),
        )
        head_outputs = value_head(features)

        # PIK 只做 calibration，不引入 corruption 对比分支。
        loss = _compute_calibration_loss(
            calibration_loss=calibration_loss,
            head_outputs=head_outputs,
            targets=targets,
            calibration_mse_weight=calibration_mse_weight,
            calibration_bce_weight=calibration_bce_weight,
            calibration_bce_pos_weight=calibration_bce_pos_weight,
            torch_module=torch_module,
        )
        (loss / grad_accum_steps).backward()

        if ((batches + 1) % grad_accum_steps == 0) or (batch_start + batch_size >= num_examples):
            if max_grad_norm > 0.0:
                torch_module.nn.utils.clip_grad_norm_(value_head.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

            if logging_steps > 0 and optimizer_steps % logging_steps == 0:
                print(
                    "train            : "
                    f"step={global_step_offset + optimizer_steps} "
                    f"loss={float(loss.item()):.6f}",
                    flush=True,
                )

        running_total += float(loss.item())
        batches += 1

    return {
        "avg_total_loss": float(running_total / max(batches, 1)),
        "num_batches": int(batches),
        "global_step": int(global_step_offset + optimizer_steps),
    }


def _evaluate_pik_head(
    *,
    value_head: Any,
    eval_cache: dict[str, Any],
    eval_examples: list[PIKSupervisionExample],
    train_target_mean: float,
    known_threshold: float,
    posthoc_calibration: str,
    posthoc_temperature_config: TemperatureCalibrationConfig,
    posthoc_isotonic_config: IsotonicCalibrationConfig,
    batch_size: int,
    torch_module: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one checkpoint on held-out P(IK) question-level examples."""
    value_head.eval()
    with torch_module.no_grad():
        scores_tensor, logits_tensor = _score_features_with_logits_in_batches(
            value_head=value_head,
            features=eval_cache["features"],
            batch_size=int(batch_size),
            torch_module=torch_module,
        )
        scores = [float(v) for v in scores_tensor.detach().cpu().tolist()]
        logits = [float(v) for v in logits_tensor.detach().cpu().tolist()]
    target_scores = [float(v) for v in eval_cache["targets"].detach().cpu().tolist()]

    calibration_raw = compute_calibration_summary(
        scores,
        target_scores,
        reference_mean=float(train_target_mean),
    )
    # known/unknown 标签由 success-rate 阈值离散化得到；阈值会影响 AUC 解释。
    known_labels = [1 if float(t) >= float(known_threshold) else 0 for t in target_scores]
    known_auc = compute_binary_auc(scores=scores, labels=known_labels)

    calibration_posthoc = None
    posthoc_payload = None
    posthoc_scores: list[float] | None = None
    known_auc_posthoc = None
    # 同时保留 raw 与 posthoc 指标，便于区分“表示能力”与“校准映射”问题。
    if posthoc_calibration == "temperature":
        posthoc_payload = fit_temperature_scaler(
            logits=logits_tensor,
            targets=eval_cache["targets"],
            torch_module=torch_module,
            config=posthoc_temperature_config,
        )
    elif posthoc_calibration == "isotonic":
        posthoc_payload = fit_isotonic_calibrator(
            scores=scores_tensor,
            targets=eval_cache["targets"],
            torch_module=torch_module,
            config=posthoc_isotonic_config,
        )

    if posthoc_payload is not None:
        posthoc_scores_tensor = apply_posthoc_calibration(
            logits=logits_tensor,
            scores=scores_tensor,
            calibrator=posthoc_payload,
            torch_module=torch_module,
        )
        posthoc_scores = [float(v) for v in posthoc_scores_tensor.detach().cpu().tolist()]
        calibration_posthoc = compute_calibration_summary(
            posthoc_scores,
            target_scores,
            reference_mean=float(train_target_mean),
        )
        known_auc_posthoc = compute_binary_auc(scores=posthoc_scores, labels=known_labels)

    rows: list[dict[str, Any]] = []
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

    return {
        "calibration": calibration_raw,
        "calibration_posthoc": calibration_posthoc,
        "posthoc_calibration": posthoc_payload,
        "known_threshold": float(known_threshold),
        "known_auc": float(known_auc),
        "known_auc_posthoc": (float(known_auc_posthoc) if known_auc_posthoc is not None else None),
    }, rows


def _compute_calibration_loss(
    *,
    calibration_loss: str,
    head_outputs: dict[str, Any],
    targets: Any,
    calibration_mse_weight: float,
    calibration_bce_weight: float,
    calibration_bce_pos_weight: float,
    torch_module: Any,
):
    """Compute one calibration loss tensor according to selected objective."""
    if calibration_loss == "mse":
        return mean_squared_calibration_loss(
            head_outputs["scores"],
            targets,
            torch_module=torch_module,
        )
    if calibration_loss == "bce":
        return binary_cross_entropy_calibration_loss(
            head_outputs["logits"],
            targets,
            torch_module=torch_module,
            pos_weight=float(calibration_bce_pos_weight),
        )
    if calibration_loss == "bce_mse":
        return mixed_calibration_loss(
            head_outputs["logits"],
            head_outputs["scores"],
            targets,
            torch_module=torch_module,
            bce_weight=float(calibration_bce_weight),
            mse_weight=float(calibration_mse_weight),
            bce_pos_weight=float(calibration_bce_pos_weight),
        )
    raise ValueError(f"Unsupported calibration_loss mode: {calibration_loss!r}")


def _apply_target_smoothing(targets: Any, *, epsilon: float) -> Any:
    """Apply optional label smoothing to soft P(IK) targets."""
    if epsilon <= 0.0:
        return targets
    return ((1.0 - 2.0 * float(epsilon)) * targets) + float(epsilon)


def _resolve_checkpoint_brier(
    *,
    eval_metrics: dict[str, Any],
    checkpoint_selection_metric: str,
) -> float:
    """Resolve Brier score used by best-checkpoint selection."""
    if checkpoint_selection_metric == "raw_brier":
        return float(eval_metrics["calibration"]["brier_score"])
    if checkpoint_selection_metric == "posthoc_brier":
        posthoc = eval_metrics.get("calibration_posthoc")
        if posthoc is None:
            raise ValueError("posthoc_brier selected but post-hoc calibration metrics are missing")
        return float(posthoc["brier_score"])
    raise ValueError(f"Unsupported checkpoint_selection_metric: {checkpoint_selection_metric!r}")


def _write_posthoc_calibration_payload(path: Path, payload: dict[str, Any] | None) -> None:
    """Persist one post-hoc calibrator payload if available."""
    if payload is None:
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_temperature_calibration_config(args: argparse.Namespace) -> TemperatureCalibrationConfig:
    """Build validated temperature-scaling config from CLI arguments."""
    cfg = TemperatureCalibrationConfig(
        lr=float(args.posthoc_temperature_lr),
        max_iters=int(args.posthoc_temperature_max_iters),
        min_temperature=float(args.posthoc_temperature_min),
        max_temperature=float(args.posthoc_temperature_max),
        init_temperature=1.0,
    )
    cfg.validate()
    return cfg


def _build_isotonic_calibration_config(args: argparse.Namespace) -> IsotonicCalibrationConfig:
    """Build validated isotonic-calibration config from CLI arguments."""
    cfg = IsotonicCalibrationConfig(min_points=int(args.posthoc_isotonic_min_points))
    cfg.validate()
    return cfg


def _build_pik_example_cache_signature_payload(
    *,
    cache_kind: str,
    examples: list[PIKSupervisionExample],
    texts: list[str],
    max_length: int,
    backbone_signature: dict[str, Any],
) -> dict[str, Any]:
    """Build conservative cache signature for one P(IK) example-cache artifact."""
    if len(examples) != len(texts):
        raise ValueError(
            "PIK cache signature expects text count == examples count: "
            f"{len(texts)} vs {len(examples)}"
        )
    return {
        "cache_kind": str(cache_kind),
        "backbone_signature": backbone_signature,
        "max_length": int(max_length),
        "num_examples": int(len(examples)),
        "sample_id_hash": hash_text_list([example.sample_id for example in examples]),
        "question_hash": hash_text_list([example.question for example in examples]),
        "text_hash": hash_text_list(texts),
        "target_success_hash": hash_float_list([float(example.target_success_rate) for example in examples]),
        "target_parseable_hash": hash_float_list([float(example.target_parseable_rate) for example in examples]),
        "target_rollout_k_hash": hash_float_list([float(example.target_k_rollouts) for example in examples]),
    }


def _validate_cached_pik_example_cache_payload(
    *,
    cache: Any,
    expected_num_examples: int,
    expected_hidden_size: int,
    torch_module: Any,
) -> None:
    """Validate loaded P(IK) cache payload before reusing."""
    if not isinstance(cache, dict):
        raise TypeError("PIK cache payload must be dict")
    for key in ("features", "targets", "parseable"):
        if key not in cache:
            raise KeyError(f"PIK cache payload missing key `{key}`")
    features = cache["features"]
    if not torch_module.is_tensor(features) or features.ndim != 2:
        raise TypeError("PIK cache `features` must be tensor[batch, hidden]")
    if int(features.shape[0]) != int(expected_num_examples):
        raise ValueError(
            f"PIK cache feature rows mismatch: expected {expected_num_examples}, got {int(features.shape[0])}"
        )
    if int(features.shape[1]) != int(expected_hidden_size):
        raise ValueError(
            f"PIK cache feature hidden mismatch: expected {expected_hidden_size}, got {int(features.shape[1])}"
        )
    for key in ("targets", "parseable"):
        tensor = cache[key]
        if not torch_module.is_tensor(tensor) or tensor.ndim != 1:
            raise TypeError(f"PIK cache `{key}` must be tensor[batch]")
        if int(tensor.shape[0]) != int(expected_num_examples):
            raise ValueError(
                f"PIK cache `{key}` rows mismatch: expected {expected_num_examples}, got {int(tensor.shape[0])}"
            )


def _encode_example_cache(
    *,
    examples: list[PIKSupervisionExample],
    texts: list[str] | None,
    backbone: Any,
    tokenizer: Any,
    torch_module: Any,
    max_length: int,
    batch_size: int,
    progress_label: str,
) -> dict[str, Any]:
    """Encode P(IK) examples into cached pooled features + targets tensors."""
    effective_texts = texts if texts is not None else [example.model_input_text() for example in examples]
    features = _encode_text_list_in_batches(
        texts=effective_texts,
        backbone=backbone,
        tokenizer=tokenizer,
        torch_module=torch_module,
        max_length=max_length,
        batch_size=batch_size,
        progress_label=progress_label,
        progress_every_batches=32,
    )
    device = features.device
    targets = torch_module.tensor(
        [example.target_success_rate for example in examples],
        dtype=torch_module.float32,
        device=device,
    )
    parseable = torch_module.tensor(
        [example.target_parseable_rate for example in examples],
        dtype=torch_module.float32,
        device=device,
    )
    payload = {
        "features": features,
        "targets": targets,
        "parseable": parseable,
    }
    return move_tensors_to_device(payload, torch_module.device("cpu"), torch_module)


def _encode_text_list_in_batches(
    *,
    texts: list[str],
    backbone: Any,
    tokenizer: Any,
    torch_module: Any,
    max_length: int,
    batch_size: int,
    progress_label: str = "cache",
    progress_every_batches: int = 0,
    max_progress_updates: int = 8,
):
    """Encode texts into one stacked pooled-feature tensor in float32."""
    chunks = []
    total_texts = len(texts)
    if total_texts == 0:
        return torch_module.zeros(
            (0, infer_backbone_hidden_size(backbone)),
            dtype=torch_module.float32,
            device=_resolve_value_device(backbone, torch_module),
        )

    total_batches = (total_texts + batch_size - 1) // batch_size
    max_progress_updates = max(int(max_progress_updates), 1)
    bounded_every = max(1, math.ceil(total_batches / max_progress_updates))
    _ = int(progress_every_batches)  # kept for API compatibility.
    effective_every = bounded_every
    print(
        f"{progress_label:16s}: start {total_texts} texts in {total_batches} batches "
        f"(bs={batch_size}, progress_every~{effective_every})",
        flush=True,
    )
    for batch_idx, start in enumerate(range(0, total_texts, batch_size), start=1):
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
        if (
            effective_every > 0
            and (batch_idx % effective_every == 0 or batch_idx == total_batches)
        ):
            print(
                f"{progress_label:16s}: {batch_idx}/{total_batches} batches",
                flush=True,
            )
    stacked = torch_module.cat(chunks, dim=0).to(dtype=torch_module.float32)
    print(f"{progress_label:16s}: done", flush=True)
    return stacked


def _resolve_total_train_steps(
    *,
    num_examples: int,
    batch_size: int,
    grad_accum_steps: int,
    num_train_epochs: float,
    max_steps: int,
) -> int:
    """Resolve total optimizer steps for scheduler construction."""
    if max_steps > 0:
        return int(max_steps)
    batches_per_epoch = max((num_examples + batch_size - 1) // batch_size, 1)
    optimizer_steps_per_epoch = max((batches_per_epoch + grad_accum_steps - 1) // grad_accum_steps, 1)
    return max(int(round(num_train_epochs * optimizer_steps_per_epoch)), 1)


def _linear_warmup_decay(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup + linear decay schedule factor."""
    if total_steps <= 1:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return max(float(step + 1) / float(warmup_steps), 1e-8)
    remaining = max(total_steps - step, 0)
    decay_steps = max(total_steps - warmup_steps, 1)
    return max(float(remaining) / float(decay_steps), 0.0)


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


def _set_seed(seed: int, torch_module: Any) -> None:
    """Seed Python and torch RNGs for reproducible C2 behavior."""
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _import_runtime_deps():
    """Import heavy runtime dependencies lazily."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


def _load_config_defaults(path: Path) -> dict[str, Any]:
    """Load one JSON object used as CLI defaults."""
    if not path.exists():
        raise FileNotFoundError(f"Config JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Config JSON {path} must contain an object at top level")
    return payload


def _resolve_tokenizer_load_path(model_path: str, adapter_path: Path | None) -> str:
    """Choose tokenizer source directory (adapter tokenizer overrides base)."""
    if adapter_path is None:
        return str(model_path)
    if (adapter_path / "tokenizer_config.json").exists():
        return str(adapter_path)
    return str(model_path)


def _attach_peft_adapter_for_inference(model: Any, adapter_path: Path):
    """Attach one PEFT adapter to the loaded backbone for feature encoding."""
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    try:
        from peft import PeftModel  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to import `peft` while attaching adapter for P(IK) C2") from exc
    return PeftModel.from_pretrained(model, str(adapter_path))


def _resolve_value_device(backbone: Any, torch_module: Any):
    """Resolve the device that should host cached features and value head."""
    if hasattr(backbone, "device"):
        return backbone.device
    try:
        return next(backbone.parameters()).device
    except StopIteration:
        return torch_module.device("cpu")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries as UTF-8 JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one dictionary row to one JSONL file."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
