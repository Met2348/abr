#!/usr/bin/env python3
"""Train the Phase C C2 value head on frozen-backbone prefix features.

Why this file exists
--------------------
Phase C does not begin with RL. It begins with a value head that estimates the
future success probability of a reasoning prefix. This script is the official C2
training entrypoint for that stage.

What this file does
-------------------
1. Parse config and runtime flags.
2. Load Phase C train/eval artifact directories.
3. Validate that train/eval artifacts share the same contracts and backbone provenance.
4. Load the frozen backbone once and encode all clean/corrupted prefixes into pooled features.
5. Train only the small value head on cached features.
6. Evaluate calibration and corruption sensitivity on the held-out eval set.
7. Persist checkpoints, metrics, manifests, scored outputs, and a Markdown summary.

Design choice
-------------
The backbone is frozen in C2. That means we can cache pooled prefix features and
train the value head on those tensors directly. This avoids repeated large-model
forwards every epoch and removes a large class of silent training/runtime bugs.

Interaction with other files
----------------------------
- `src/ours/phase_b/value_data.py`: loads/join Phase C artifacts
- `src/ours/phase_b/value_head.py`: value head definition and feature extraction helpers
- `src/ours/phase_b/value_losses.py`: calibration and contrastive losses
- `src/ours/phase_b/faithfulness_eval.py`: evaluation metrics and Markdown rendering
- `scripts/phase_b_prepare_value_data.py`: produces the input artifact directories

Example
-------
```bash
python -u scripts/phase_b_train_value.py \
  --train-dir assets/artifacts/phase_c_data/strategyqa/train_run__abcdef123456 \
  --eval-dir assets/artifacts/phase_c_data/strategyqa/val_run__abcdef123456 \
  --run-name strategyqa_value_smoke \
  --require-cuda
```
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
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
    CorruptionVariant,
    ValueSupervisionExample,
    assert_phase_c_compatibility,
    load_corruption_variants,
    load_phase_c_manifest,
    load_value_supervision_examples,
)
from ours.phase_b.value_head import (  # noqa: E402
    SigmoidValueHead,
    ValueHeadConfig,
    encode_text_features,
    freeze_backbone,
    infer_backbone_hidden_size,
    load_value_head_checkpoint,
    save_value_head_checkpoint,
    write_value_head_config_json,
)
from ours.phase_b.value_losses import (  # noqa: E402
    binary_cross_entropy_calibration_loss,
    contrastive_margin_loss,
    mean_squared_calibration_loss,
    mixed_calibration_loss,
)


@dataclass(slots=True)
class TrainConfig:
    """Compact snapshot of C2 hyperparameters persisted in the manifest."""

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
    calibration_sample_weighting: str
    calibration_weight_floor: float
    calibration_weight_gamma: float
    lambda_contrastive: float
    contrastive_margin: float
    contrastive_pair_filter: str
    contrastive_confidence_threshold: float
    contrastive_parseable_threshold: float
    contrastive_score_gap_min: float
    contrastive_score_gap_max: float
    adaptive_loss_balancing: str
    adaptive_loss_init_log_variance: float
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config payload."""
        return asdict(self)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for Phase C value-head training."""
    parser = argparse.ArgumentParser(
        description="Train the Phase C C2 value head on frozen-backbone prefix features."
    )
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--train-dir", type=Path, required=False, default=None)
    parser.add_argument("--eval-dir", type=Path, required=False, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_c_runs"),
        help="Run output root for C2 training artifacts.",
    )
    parser.add_argument("--run-name", default="phase_c_value")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-corruption-variants-eval", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=64)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--use-contrastive-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use primary corrupted prefixes as an auxiliary contrastive signal.",
    )
    parser.add_argument(
        "--calibration-loss",
        choices=["mse", "bce", "bce_mse"],
        default="mse",
        help=(
            "Calibration objective for rollout targets. "
            "`mse` is the legacy baseline, `bce` uses logits, and `bce_mse` mixes both."
        ),
    )
    parser.add_argument(
        "--calibration-mse-weight",
        type=float,
        default=1.0,
        help="Weight of the MSE term when `--calibration-loss=bce_mse`.",
    )
    parser.add_argument(
        "--calibration-bce-weight",
        type=float,
        default=1.0,
        help="Weight of the BCE term when `--calibration-loss=bce_mse`.",
    )
    parser.add_argument(
        "--calibration-bce-pos-weight",
        type=float,
        default=1.0,
        help="Positive-class weight for BCE calibration losses.",
    )
    parser.add_argument(
        "--calibration-target-smoothing",
        type=float,
        default=0.0,
        help=(
            "Optional smoothing epsilon for rollout targets before calibration loss. "
            "Applied as y' = (1 - 2*eps) * y + eps."
        ),
    )
    parser.add_argument(
        "--calibration-sample-weighting",
        choices=["none", "confidence", "entropy_inverse", "parseable", "confidence_parseable"],
        default="none",
        help=(
            "Optional per-sample weighting for calibration losses. "
            "`confidence` upweights targets far from 0.5; "
            "`entropy_inverse` downweights high-uncertainty targets; "
            "`parseable` uses rollout parseable-rate; "
            "`confidence_parseable` multiplies confidence and parseable signals."
        ),
    )
    parser.add_argument(
        "--calibration-weight-floor",
        type=float,
        default=0.05,
        help="Lower bound applied to calibration sample weights to avoid hard-zero gradients.",
    )
    parser.add_argument(
        "--calibration-weight-gamma",
        type=float,
        default=1.0,
        help="Optional exponent applied to calibration sample weights after floor-clamp.",
    )
    parser.add_argument("--lambda-contrastive", type=float, default=1.0)
    parser.add_argument("--contrastive-margin", type=float, default=0.1)
    parser.add_argument(
        "--contrastive-pair-filter",
        choices=["none", "confidence", "parseable", "confidence_parseable"],
        default="none",
        help=(
            "Optional noisy-pair filtering for contrastive training. "
            "Filters are computed from clean-prefix rollout targets."
        ),
    )
    parser.add_argument(
        "--contrastive-confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum clean-prefix confidence (2*|p-0.5|) for pair inclusion when confidence filter is enabled.",
    )
    parser.add_argument(
        "--contrastive-parseable-threshold",
        type=float,
        default=0.0,
        help="Minimum rollout parseable-rate for pair inclusion when parseable filter is enabled.",
    )
    parser.add_argument(
        "--contrastive-score-gap-min",
        type=float,
        default=-1.0,
        help=(
            "Optional lower bound for contrastive pair inclusion based on current "
            "(clean_score - corrupt_score)."
        ),
    )
    parser.add_argument(
        "--contrastive-score-gap-max",
        type=float,
        default=1.0,
        help=(
            "Optional upper bound for contrastive pair inclusion based on current "
            "(clean_score - corrupt_score). Useful for hard-negative mining."
        ),
    )
    parser.add_argument(
        "--adaptive-loss-balancing",
        choices=["none", "uncertainty"],
        default="none",
        help=(
            "How to balance calibration and contrastive terms. "
            "`uncertainty` adds learnable log-variance weights (Kendall-style)."
        ),
    )
    parser.add_argument(
        "--adaptive-loss-init-log-variance",
        type=float,
        default=0.0,
        help="Initial log-variance for uncertainty balancing parameters.",
    )
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=["raw_brier", "posthoc_brier"],
        default="raw_brier",
        help=(
            "Metric used for best-checkpoint selection. "
            "`posthoc_brier` requires post-hoc calibration enabled."
        ),
    )
    parser.add_argument(
        "--posthoc-calibration",
        choices=["none", "temperature", "isotonic"],
        default="none",
        help="Optional post-hoc calibration evaluated on each eval pass.",
    )
    parser.add_argument(
        "--posthoc-temperature-lr",
        type=float,
        default=0.05,
        help="Optimizer LR for temperature scaling.",
    )
    parser.add_argument(
        "--posthoc-temperature-max-iters",
        type=int,
        default=200,
        help="Max optimization iterations for temperature scaling.",
    )
    parser.add_argument(
        "--posthoc-temperature-min",
        type=float,
        default=0.05,
        help="Lower bound for fitted temperature.",
    )
    parser.add_argument(
        "--posthoc-temperature-max",
        type=float,
        default=10.0,
        help="Upper bound for fitted temperature.",
    )
    parser.add_argument(
        "--posthoc-isotonic-min-points",
        type=int,
        default=32,
        help="Minimum eval points required before fitting isotonic post-hoc calibration.",
    )
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
        help="Fail fast if CUDA is unavailable for the frozen-backbone encoding stage.",
    )
    parser.add_argument(
        "--save-best-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist the best eval-Brier checkpoint separately from the final head.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and optional config defaults."""
    parser = _build_parser()
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
        raise ValueError("Set either a positive --num-train-epochs or a positive --max-steps")
    if args.lambda_contrastive < 0.0:
        raise ValueError("--lambda-contrastive must be >= 0")
    if args.contrastive_margin < 0.0:
        raise ValueError("--contrastive-margin must be >= 0")
    if not (0.0 <= args.contrastive_confidence_threshold <= 1.0):
        raise ValueError("--contrastive-confidence-threshold must be in [0, 1]")
    if not (0.0 <= args.contrastive_parseable_threshold <= 1.0):
        raise ValueError("--contrastive-parseable-threshold must be in [0, 1]")
    if args.contrastive_score_gap_min > args.contrastive_score_gap_max:
        raise ValueError("--contrastive-score-gap-min must be <= --contrastive-score-gap-max")
    if not (-1.0 <= args.contrastive_score_gap_min <= 1.0):
        raise ValueError("--contrastive-score-gap-min must be in [-1, 1]")
    if not (-1.0 <= args.contrastive_score_gap_max <= 1.0):
        raise ValueError("--contrastive-score-gap-max must be in [-1, 1]")
    if args.calibration_mse_weight < 0.0:
        raise ValueError("--calibration-mse-weight must be >= 0")
    if args.calibration_bce_weight < 0.0:
        raise ValueError("--calibration-bce-weight must be >= 0")
    if args.calibration_bce_pos_weight <= 0.0:
        raise ValueError("--calibration-bce-pos-weight must be > 0")
    if not (0.0 <= args.calibration_target_smoothing < 0.5):
        raise ValueError("--calibration-target-smoothing must be in [0, 0.5)")
    if args.calibration_weight_floor < 0.0:
        raise ValueError("--calibration-weight-floor must be >= 0")
    if args.calibration_weight_floor > 1.0:
        raise ValueError("--calibration-weight-floor must be <= 1")
    if args.calibration_weight_gamma <= 0.0:
        raise ValueError("--calibration-weight-gamma must be > 0")
    if args.calibration_loss == "bce_mse" and (
        args.calibration_mse_weight == 0.0 and args.calibration_bce_weight == 0.0
    ):
        raise ValueError("bce_mse requires at least one non-zero calibration component weight")
    if args.adaptive_loss_balancing != "none" and not args.use_contrastive_loss:
        raise ValueError("--adaptive-loss-balancing requires --use-contrastive-loss")
    if args.checkpoint_selection_metric == "posthoc_brier" and args.posthoc_calibration == "none":
        raise ValueError(
            "--checkpoint-selection-metric=posthoc_brier requires post-hoc calibration (temperature or isotonic)"
        )
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
    if not (0.0 <= args.dropout_prob < 1.0):
        raise ValueError("--dropout-prob must be in [0, 1)")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the full C2 training and held-out evaluation workflow."""
    args = parse_args(argv)

    # Stage 1: load and validate the input artifact directories before touching the model.
    train_examples, train_manifest = load_value_supervision_examples(
        args.train_dir,
        max_samples=args.max_train_samples,
        require_corruptions=bool(args.use_contrastive_loss),
    )
    eval_examples, eval_manifest = load_value_supervision_examples(
        args.eval_dir,
        max_samples=args.max_eval_samples,
        require_corruptions=False,
    )
    eval_corruptions, _ = load_corruption_variants(
        args.eval_dir,
        max_variants=args.max_corruption_variants_eval,
    )
    assert_phase_c_compatibility(train_manifest, eval_manifest)

    rollout_config = dict(train_manifest["rollout_config"])
    model_path = str(rollout_config["model_path"])
    adapter_path = rollout_config.get("adapter_path")
    if args.use_contrastive_loss and not any(example.has_primary_corruption() for example in train_examples):
        raise ValueError(
            "Contrastive loss was requested, but none of the loaded train prefixes have a primary corruption"
        )

    torch, AutoModelForCausalLM, AutoTokenizer = _import_runtime_deps()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for C2 encoding, but no GPU is visible")
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
        calibration_sample_weighting=str(args.calibration_sample_weighting),
        calibration_weight_floor=float(args.calibration_weight_floor),
        calibration_weight_gamma=float(args.calibration_weight_gamma),
        lambda_contrastive=float(args.lambda_contrastive if args.use_contrastive_loss else 0.0),
        contrastive_margin=float(args.contrastive_margin),
        contrastive_pair_filter=str(args.contrastive_pair_filter),
        contrastive_confidence_threshold=float(args.contrastive_confidence_threshold),
        contrastive_parseable_threshold=float(args.contrastive_parseable_threshold),
        contrastive_score_gap_min=float(args.contrastive_score_gap_min),
        contrastive_score_gap_max=float(args.contrastive_score_gap_max),
        adaptive_loss_balancing=str(args.adaptive_loss_balancing),
        adaptive_loss_init_log_variance=float(args.adaptive_loss_init_log_variance),
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
    prefix_scores_path = run_dir / "eval_prefix_scores.jsonl"
    corruption_scores_path = run_dir / "eval_corruption_scores.jsonl"
    train_curve_path = run_dir / "train_curve.jsonl"

    print("=" * 88)
    print("Phase C: Train Value Head")
    print("=" * 88)
    print(f"train_dir         : {args.train_dir}")
    print(f"eval_dir          : {args.eval_dir}")
    print(f"run_dir           : {run_dir}")
    print(f"model_path        : {model_path}")
    print(f"adapter_path      : {adapter_path if adapter_path is not None else '<none>'}")
    print(f"train_examples    : {len(train_examples)}")
    print(f"eval_examples     : {len(eval_examples)}")
    print(f"eval_corruptions  : {len(eval_corruptions)}")
    print(f"contrastive_loss  : {args.use_contrastive_loss}")
    print(f"calibration_loss  : {args.calibration_loss}")
    print(f"calib_target_eps : {args.calibration_target_smoothing}")
    print(f"calib_weight_mode : {args.calibration_sample_weighting}")
    print(f"calib_weight_floor: {args.calibration_weight_floor}")
    print(f"calib_weight_gamma: {args.calibration_weight_gamma}")
    print(f"adaptive_balance  : {args.adaptive_loss_balancing}")
    print(f"pair_filter       : {args.contrastive_pair_filter}")
    print(f"pair_conf_thr     : {args.contrastive_confidence_threshold}")
    print(f"pair_parse_thr    : {args.contrastive_parseable_threshold}")
    print(f"pair_gap_min      : {args.contrastive_score_gap_min}")
    print(f"pair_gap_max      : {args.contrastive_score_gap_max}")
    print(f"posthoc_calib     : {args.posthoc_calibration}")
    print(f"ckpt_metric       : {args.checkpoint_selection_metric}")
    print(f"batch_train       : {args.per_device_train_batch_size}")
    print(f"batch_eval        : {args.per_device_eval_batch_size}")
    print(f"max_length        : {args.max_length}")
    print(f"seed              : {args.seed}")
    print("=" * 88)

    # Stage 2: load the frozen backbone once and cache pooled features.
    model_load_start = time.perf_counter()
    resolved_dtype = _resolve_dtype(args.dtype, torch)
    tokenizer_path = _resolve_tokenizer_load_path(model_path=model_path, adapter_path=(Path(adapter_path) if adapter_path else None))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

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
    train_cache = _encode_example_cache(
        examples=train_examples,
        backbone=backbone,
        tokenizer=tokenizer,
        torch_module=torch,
        max_length=args.max_length,
        batch_size=args.per_device_eval_batch_size,
        use_primary_corruption=args.use_contrastive_loss,
        calibration_sample_weighting=args.calibration_sample_weighting,
        calibration_weight_floor=args.calibration_weight_floor,
        calibration_weight_gamma=args.calibration_weight_gamma,
    )
    eval_cache = _encode_example_cache(
        examples=eval_examples,
        backbone=backbone,
        tokenizer=tokenizer,
        torch_module=torch,
        max_length=args.max_length,
        batch_size=args.per_device_eval_batch_size,
        use_primary_corruption=False,
        calibration_sample_weighting="none",
        calibration_weight_floor=0.0,
        calibration_weight_gamma=1.0,
    )
    eval_corruption_cache = _encode_corruption_variant_cache(
        variants=eval_corruptions,
        backbone=backbone,
        tokenizer=tokenizer,
        torch_module=torch,
        max_length=args.max_length,
        batch_size=args.per_device_eval_batch_size,
    )
    feature_cache_elapsed = time.perf_counter() - feature_cache_start

    # Stage 3: train only the small value head on the cached features.
    # Trick-3 (adaptive loss balancing):
    # Build optional learnable balancing scalars once, keep them in the same
    # optimizer so updates remain synchronized with the value-head step updates.
    adaptive_loss_state = _build_adaptive_loss_state(
        adaptive_mode=args.adaptive_loss_balancing,
        init_log_variance=args.adaptive_loss_init_log_variance,
        value_device=value_device,
        torch_module=torch,
    )
    optimizer_param_groups: list[dict[str, Any]] = [
        {
            "params": list(value_head.parameters()),
            "weight_decay": args.weight_decay,
        }
    ]
    if adaptive_loss_state is not None:
        # Keep uncertainty-balancing scalars free of weight decay.
        optimizer_param_groups.append(
            {
                "params": [
                    adaptive_loss_state["log_var_calibration"],
                    adaptive_loss_state["log_var_contrastive"],
                ],
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
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
    # Trick-2 (post-hoc calibration):
    # Keep one validated temperature-scaling config and reuse it across eval
    # passes so results are comparable across epochs/checkpoints.
    posthoc_temperature_config = _build_temperature_calibration_config(args)
    posthoc_isotonic_config = _build_isotonic_calibration_config(args)

    train_curve: list[dict[str, Any]] = []
    train_target_mean = float(train_cache["targets"].mean().item())
    best_eval_selection_brier = float("inf")
    best_eval_metrics: dict[str, Any] | None = None
    best_eval_scored: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None

    train_start = time.perf_counter()
    global_step = 0
    epoch_idx = 0
    optimizer.zero_grad(set_to_none=True)

    # Drive the loop by optimizer steps rather than integer epochs so fractional
    # `num_train_epochs` values behave as requested instead of being silently rounded.
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
            lambda_contrastive=(args.lambda_contrastive if args.use_contrastive_loss else 0.0),
            contrastive_margin=args.contrastive_margin,
            contrastive_pair_filter=args.contrastive_pair_filter,
            contrastive_confidence_threshold=args.contrastive_confidence_threshold,
            contrastive_parseable_threshold=args.contrastive_parseable_threshold,
            contrastive_score_gap_min=args.contrastive_score_gap_min,
            contrastive_score_gap_max=args.contrastive_score_gap_max,
            adaptive_loss_balancing=args.adaptive_loss_balancing,
            adaptive_loss_state=adaptive_loss_state,
            logging_steps=args.logging_steps,
            global_step_offset=global_step,
            max_steps=(args.max_steps if args.max_steps > 0 else None),
        )
        global_step = int(epoch_stats["global_step"])
        epoch_idx += 1
        if int(epoch_stats["num_batches"]) <= 0:
            raise RuntimeError("C2 training made zero progress in one epoch pass")

        eval_metrics, eval_prefix_scores, eval_corruption_scores = _evaluate_value_head(
            value_head=value_head,
            eval_cache=eval_cache,
            eval_examples=eval_examples,
            eval_corruption_cache=eval_corruption_cache,
            eval_corruptions=eval_corruptions,
            train_target_mean=train_target_mean,
            posthoc_calibration=args.posthoc_calibration,
            posthoc_temperature_config=posthoc_temperature_config,
            posthoc_isotonic_config=posthoc_isotonic_config,
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
            best_eval_scored = (eval_prefix_scores, eval_corruption_scores)
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

    final_eval_metrics, final_prefix_scores, final_corruption_scores = _evaluate_value_head(
        value_head=value_head,
        eval_cache=eval_cache,
        eval_examples=eval_examples,
        eval_corruption_cache=eval_corruption_cache,
        eval_corruptions=eval_corruptions,
        train_target_mean=train_target_mean,
        posthoc_calibration=args.posthoc_calibration,
        posthoc_temperature_config=posthoc_temperature_config,
        posthoc_isotonic_config=posthoc_isotonic_config,
        torch_module=torch,
    )
    _write_posthoc_calibration_payload(
        path=final_posthoc_path,
        payload=final_eval_metrics.get("posthoc_calibration"),
    )

    selected_eval_metrics = best_eval_metrics if best_eval_metrics is not None else final_eval_metrics
    selected_prefix_scores, selected_corruption_scores = (
        best_eval_scored if best_eval_scored is not None else (final_prefix_scores, final_corruption_scores)
    )
    _write_jsonl(prefix_scores_path, selected_prefix_scores)
    _write_jsonl(corruption_scores_path, selected_corruption_scores)

    train_metrics = {
        "model_load_seconds": float(model_load_elapsed),
        "feature_cache_seconds": float(feature_cache_elapsed),
        "train_elapsed_seconds": float(train_elapsed),
        "global_step": int(global_step),
        "train_curve": train_curve,
        "train_target_mean": float(train_target_mean),
        "best_eval_selection_brier": float(best_eval_selection_brier),
        "checkpoint_selection_metric": str(args.checkpoint_selection_metric),
        "adaptive_loss_state": (
            {
                "mode": str(args.adaptive_loss_balancing),
                "log_var_calibration": float(adaptive_loss_state["log_var_calibration"].detach().cpu().item()),
                "log_var_contrastive": float(adaptive_loss_state["log_var_contrastive"].detach().cpu().item()),
            }
            if adaptive_loss_state is not None
            else None
        ),
    }
    eval_metrics = {
        "selected_checkpoint": ("best" if best_eval_metrics is not None else "final"),
        "best_checkpoint_saved": bool(args.save_best_state),
        "best": best_eval_metrics,
        "final": final_eval_metrics,
    }
    train_metrics_path.write_text(json.dumps(train_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    eval_metrics_path.write_text(json.dumps(eval_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "artifact_stage": "phase_c_c2_value_head",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/phase_b_train_value.py",
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
        "num_train_examples": len(train_examples),
        "num_eval_examples": len(eval_examples),
        "num_eval_corruptions": len(eval_corruptions),
        "output_files": {
            "best_value_head": str(best_ckpt_path) if args.save_best_state else None,
            "final_value_head": str(final_ckpt_path),
            "best_posthoc_calibration": (str(best_posthoc_path) if best_posthoc_path.exists() else None),
            "final_posthoc_calibration": (str(final_posthoc_path) if final_posthoc_path.exists() else None),
            "value_head_config": str(config_json_path),
            "train_metrics": str(train_metrics_path),
            "eval_metrics": str(eval_metrics_path),
            "eval_prefix_scores": str(prefix_scores_path),
            "eval_corruption_scores": str(corruption_scores_path),
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
            "model_load_seconds": float(model_load_elapsed),
            "train_target_mean": float(train_target_mean),
            "checkpoint_selection_metric": str(args.checkpoint_selection_metric),
            "posthoc_calibration": str(args.posthoc_calibration),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(
        render_faithfulness_summary_markdown(
            title="Phase C C2 Value-Head Summary",
            calibration=selected_eval_metrics["calibration"],
            corruption=selected_eval_metrics.get("corruption"),
            metadata={
                "run_dir": run_dir,
                "train_dir": args.train_dir,
                "eval_dir": args.eval_dir,
                "model_path": model_path,
                "adapter_path": (adapter_path if adapter_path is not None else "<none>"),
                "global_step": global_step,
                "train_examples": len(train_examples),
                "eval_examples": len(eval_examples),
                "eval_corruptions": len(eval_corruptions),
            },
        ),
        encoding="utf-8",
    )

    print("-" * 88)
    print(f"global_step       : {global_step}")
    print(f"train_elapsed_sec : {train_elapsed:.2f}")
    print(f"feature_cache_sec : {feature_cache_elapsed:.2f}")
    print(f"selected_raw_brier: {selected_eval_metrics['calibration']['brier_score']:.6f}")
    if selected_eval_metrics.get("calibration_posthoc") is not None:
        print(f"selected_post_bri : {selected_eval_metrics['calibration_posthoc']['brier_score']:.6f}")
    print(f"selected_criterion: {args.checkpoint_selection_metric}")
    print(f"selected_brier    : {_resolve_checkpoint_brier(eval_metrics=selected_eval_metrics, checkpoint_selection_metric=args.checkpoint_selection_metric):.6f}")
    print(f"selected_pearson  : {selected_eval_metrics['calibration']['pearson']:.6f}")
    if selected_eval_metrics.get("calibration_posthoc") is not None:
        print(f"selected_post_prs : {selected_eval_metrics['calibration_posthoc']['pearson']:.6f}")
    if adaptive_loss_state is not None:
        print(f"adaptive_logvar_c : {float(adaptive_loss_state['log_var_calibration'].detach().cpu().item()):.6f}")
        print(f"adaptive_logvar_t : {float(adaptive_loss_state['log_var_contrastive'].detach().cpu().item()):.6f}")
    if selected_eval_metrics.get("corruption") is not None:
        print(f"corr_pair_acc     : {selected_eval_metrics['corruption']['pair_accuracy']:.6f}")
        print(f"corr_auc          : {selected_eval_metrics['corruption']['auc_clean_vs_corrupt']:.6f}")
    print(f"train_metrics     : {train_metrics_path}")
    print(f"eval_metrics      : {eval_metrics_path}")
    print(f"manifest          : {manifest_path}")
    print(f"summary           : {summary_path}")
    print("=" * 88)
    return 0


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
    lambda_contrastive: float,
    contrastive_margin: float,
    contrastive_pair_filter: str,
    contrastive_confidence_threshold: float,
    contrastive_parseable_threshold: float,
    contrastive_score_gap_min: float,
    contrastive_score_gap_max: float,
    adaptive_loss_balancing: str,
    adaptive_loss_state: dict[str, Any] | None,
    logging_steps: int,
    global_step_offset: int,
    max_steps: int | None,
) -> dict[str, Any]:
    """Train the head for one epoch over cached feature tensors."""
    value_head.train()
    num_examples = int(train_cache["clean_features"].shape[0])
    permutation = torch_module.randperm(num_examples, device=train_cache["clean_features"].device)
    running_total = 0.0
    running_cal = 0.0
    running_ctr = 0.0
    running_effective_ctr_weight = 0.0
    batches = 0
    optimizer_steps = 0

    for batch_start in range(0, num_examples, batch_size):
        if max_steps is not None and global_step_offset + optimizer_steps >= max_steps:
            break
        batch_indices = permutation[batch_start : batch_start + batch_size]
        clean_features = train_cache["clean_features"][batch_indices]
        raw_targets = train_cache["targets"][batch_indices]
        targets = _apply_target_smoothing(
            raw_targets,
            epsilon=float(calibration_target_smoothing),
        )
        sample_weights = train_cache["calibration_weights"][batch_indices]
        head_outputs = value_head(clean_features)
        loss_cal = _compute_calibration_loss(
            calibration_loss=calibration_loss,
            head_outputs=head_outputs,
            targets=targets,
            sample_weights=sample_weights,
            calibration_mse_weight=calibration_mse_weight,
            calibration_bce_weight=calibration_bce_weight,
            calibration_bce_pos_weight=calibration_bce_pos_weight,
            torch_module=torch_module,
        )
        loss_ctr = torch_module.zeros((), device=clean_features.device)
        used_ctr_pairs = False
        if lambda_contrastive > 0.0:
            corruption_mask = train_cache["has_primary_corruption"][batch_indices]
            corruption_mask = _apply_contrastive_pair_filter(
                batch_indices=batch_indices,
                base_mask=corruption_mask,
                train_cache=train_cache,
                pair_filter=contrastive_pair_filter,
                confidence_threshold=float(contrastive_confidence_threshold),
                parseable_threshold=float(contrastive_parseable_threshold),
                torch_module=torch_module,
            )
            if bool(corruption_mask.any().item()):
                corrupt_features = train_cache["primary_corruption_features"][batch_indices][corruption_mask]
                clean_scores_for_pairs = head_outputs["scores"][corruption_mask]
                corrupt_scores = value_head(corrupt_features)["scores"]
                clean_scores_for_pairs, corrupt_scores = _apply_contrastive_score_gap_filter(
                    clean_scores_for_pairs=clean_scores_for_pairs,
                    corrupt_scores=corrupt_scores,
                    score_gap_min=float(contrastive_score_gap_min),
                    score_gap_max=float(contrastive_score_gap_max),
                    torch_module=torch_module,
                )
                if clean_scores_for_pairs.numel() > 0:
                    loss_ctr = contrastive_margin_loss(
                        clean_scores_for_pairs,
                        corrupt_scores,
                        margin=contrastive_margin,
                        torch_module=torch_module,
                    )
                    used_ctr_pairs = True

        loss, effective_ctr_weight = _compose_total_loss(
            loss_cal=loss_cal,
            loss_ctr=loss_ctr,
            lambda_contrastive=float(lambda_contrastive),
            adaptive_loss_balancing=adaptive_loss_balancing,
            adaptive_loss_state=adaptive_loss_state,
            used_ctr_pairs=used_ctr_pairs,
            torch_module=torch_module,
        )
        (loss / grad_accum_steps).backward()

        if ((batches + 1) % grad_accum_steps == 0) or (batch_start + batch_size >= num_examples):
            if max_grad_norm > 0.0:
                torch_module.nn.utils.clip_grad_norm_(value_head.parameters(), max_grad_norm)
                if adaptive_loss_state is not None:
                    torch_module.nn.utils.clip_grad_norm_(
                        [
                            adaptive_loss_state["log_var_calibration"],
                            adaptive_loss_state["log_var_contrastive"],
                        ],
                        max_grad_norm,
                    )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

            if logging_steps > 0 and optimizer_steps % logging_steps == 0:
                print(
                    "train            : "
                    f"step={global_step_offset + optimizer_steps} "
                    f"loss={float(loss.item()):.6f} "
                    f"cal={float(loss_cal.item()):.6f} "
                    f"ctr={float(loss_ctr.item()):.6f} "
                    f"ctr_w={float(effective_ctr_weight):.4f}",
                    flush=True,
                )

        running_total += float(loss.item())
        running_cal += float(loss_cal.item())
        running_ctr += float(loss_ctr.item())
        running_effective_ctr_weight += float(effective_ctr_weight)
        batches += 1

    return {
        "avg_total_loss": float(running_total / max(batches, 1)),
        "avg_calibration_loss": float(running_cal / max(batches, 1)),
        "avg_contrastive_loss": float(running_ctr / max(batches, 1)),
        "avg_effective_contrastive_weight": float(running_effective_ctr_weight / max(batches, 1)),
        "num_batches": int(batches),
        "global_step": int(global_step_offset + optimizer_steps),
    }


def _evaluate_value_head(
    *,
    value_head: Any,
    eval_cache: dict[str, Any],
    eval_examples: list[ValueSupervisionExample],
    eval_corruption_cache: dict[str, Any],
    eval_corruptions: list[CorruptionVariant],
    train_target_mean: float,
    posthoc_calibration: str,
    posthoc_temperature_config: TemperatureCalibrationConfig,
    posthoc_isotonic_config: IsotonicCalibrationConfig,
    torch_module: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Score the held-out eval set and compute C2 metrics."""
    value_head.eval()
    with torch_module.no_grad():
        prefix_head_outputs = value_head(eval_cache["clean_features"])
        prefix_scores_tensor = prefix_head_outputs["scores"]
        prefix_logits_tensor = prefix_head_outputs["logits"]
        prefix_scores = [float(v) for v in prefix_scores_tensor.detach().cpu().tolist()]
        target_scores = [float(v) for v in eval_cache["targets"].detach().cpu().tolist()]

    calibration_raw = compute_calibration_summary(
        prefix_scores,
        target_scores,
        reference_mean=float(train_target_mean),
    )
    calibration_posthoc = None
    posthoc_payload = None
    posthoc_scores: list[float] | None = None
    if posthoc_calibration == "temperature":
        posthoc_payload = fit_temperature_scaler(
            logits=prefix_logits_tensor,
            targets=eval_cache["targets"],
            torch_module=torch_module,
            config=posthoc_temperature_config,
        )
    elif posthoc_calibration == "isotonic":
        posthoc_payload = fit_isotonic_calibrator(
            scores=prefix_scores_tensor,
            targets=eval_cache["targets"],
            torch_module=torch_module,
            config=posthoc_isotonic_config,
        )
    if posthoc_payload is not None:
        posthoc_scores_tensor = apply_posthoc_calibration(
            logits=prefix_logits_tensor,
            scores=prefix_scores_tensor,
            calibrator=posthoc_payload,
            torch_module=torch_module,
        )
        posthoc_scores = [float(v) for v in posthoc_scores_tensor.detach().cpu().tolist()]
        calibration_posthoc = compute_calibration_summary(
            posthoc_scores,
            target_scores,
            reference_mean=float(train_target_mean),
        )

    prefix_rows: list[dict[str, Any]] = []
    prefix_score_by_id: dict[str, float] = {}
    for idx, (example, score) in enumerate(zip(eval_examples, prefix_scores, strict=True)):
        row = {
            "prefix_id": example.prefix_id,
            "sample_id": example.sample_id,
            "dataset": example.dataset,
            "split": example.split,
            "question": example.question,
            "current_step_role": example.current_step_role,
            "prefix_step_index": example.prefix_step_index,
            "num_reasoning_steps_seen": example.num_reasoning_steps_seen,
            "num_reasoning_steps_total": example.num_reasoning_steps_total,
            # Keep `predicted_value` for backward compatibility with existing
            # ad-hoc analysis notebooks; `predicted_value_raw` is the explicit key.
            "predicted_value": float(score),
            "predicted_value_raw": float(score),
            "target_success_rate": float(example.target_success_rate),
            "target_parseable_rate": float(example.target_parseable_rate),
        }
        if posthoc_scores is not None:
            row["predicted_value_posthoc"] = float(posthoc_scores[idx])
        prefix_rows.append(row)
        prefix_score_by_id[example.prefix_id] = float(score)

    corruption_metrics = None
    corruption_rows: list[dict[str, Any]] = []
    if eval_corruptions:
        with torch_module.no_grad():
            corrupt_scores_tensor = value_head(eval_corruption_cache["corruption_features"])["scores"]
            corrupt_scores = [float(v) for v in corrupt_scores_tensor.detach().cpu().tolist()]
        clean_scores = [prefix_score_by_id[variant.clean_prefix_id] for variant in eval_corruptions]
        corruption_metrics = compute_corruption_summary(
            clean_scores,
            corrupt_scores,
            corruption_types=[variant.corruption_type for variant in eval_corruptions],
            corruption_step_indices=[variant.corruption_step_index for variant in eval_corruptions],
        )
        for variant, clean_score, corrupt_score in zip(eval_corruptions, clean_scores, corrupt_scores, strict=True):
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
                    "current_step_role": variant.current_step_role,
                    "clean_value": float(clean_score),
                    "corrupted_value": float(corrupt_score),
                    "value_margin": float(clean_score - corrupt_score),
                }
            )

    return {
        "calibration": calibration_raw,
        "calibration_posthoc": calibration_posthoc,
        "posthoc_calibration": posthoc_payload,
        "corruption": corruption_metrics,
    }, prefix_rows, corruption_rows


def _compute_calibration_loss(
    *,
    calibration_loss: str,
    head_outputs: dict[str, Any],
    targets: Any,
    sample_weights: Any,
    calibration_mse_weight: float,
    calibration_bce_weight: float,
    calibration_bce_pos_weight: float,
    torch_module: Any,
):
    """Compute one calibration loss tensor according to the selected mode.

    Why this helper exists
    ----------------------
    Keeping objective routing here avoids silent loss drift when adding new C2
    experiments. The training loop always calls one function and logs one scalar.
    """
    if calibration_loss == "mse":
        return mean_squared_calibration_loss(
            head_outputs["scores"],
            targets,
            torch_module=torch_module,
            sample_weights=sample_weights,
        )
    if calibration_loss == "bce":
        return binary_cross_entropy_calibration_loss(
            head_outputs["logits"],
            targets,
            torch_module=torch_module,
            pos_weight=float(calibration_bce_pos_weight),
            sample_weights=sample_weights,
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
            sample_weights=sample_weights,
        )
    raise ValueError(f"Unsupported calibration_loss mode: {calibration_loss!r}")


def _build_adaptive_loss_state(
    *,
    adaptive_mode: str,
    init_log_variance: float,
    value_device: Any,
    torch_module: Any,
) -> dict[str, Any] | None:
    """Build optional learnable state for adaptive loss balancing.

    Current mode:
    - `none`: no adaptive state
    - `uncertainty`: two learnable log-variance scalars for calibration and
      contrastive losses (Kendall-style weighting).
    """
    if adaptive_mode == "none":
        return None
    if adaptive_mode != "uncertainty":
        raise ValueError(f"Unsupported adaptive loss balancing mode: {adaptive_mode!r}")
    return {
        "mode": "uncertainty",
        "log_var_calibration": torch_module.nn.Parameter(
            torch_module.tensor(float(init_log_variance), dtype=torch_module.float32, device=value_device)
        ),
        "log_var_contrastive": torch_module.nn.Parameter(
            torch_module.tensor(float(init_log_variance), dtype=torch_module.float32, device=value_device)
        ),
    }


def _compose_total_loss(
    *,
    loss_cal: Any,
    loss_ctr: Any,
    lambda_contrastive: float,
    adaptive_loss_balancing: str,
    adaptive_loss_state: dict[str, Any] | None,
    used_ctr_pairs: bool,
    torch_module: Any,
) -> tuple[Any, float]:
    """Compose the final training loss with optional adaptive balancing.

    Returns
    -------
    tuple[tensor, float]
        `(total_loss, effective_contrastive_weight_for_logging)`
    """
    if adaptive_loss_balancing == "none":
        effective = float(lambda_contrastive) if used_ctr_pairs else 0.0
        return loss_cal + float(lambda_contrastive) * loss_ctr, effective
    if adaptive_loss_balancing != "uncertainty":
        raise ValueError(f"Unsupported adaptive loss balancing mode: {adaptive_loss_balancing!r}")
    if adaptive_loss_state is None:
        raise ValueError("Adaptive loss balancing requested, but adaptive_loss_state is missing")

    log_var_cal = adaptive_loss_state["log_var_calibration"]
    log_var_ctr = adaptive_loss_state["log_var_contrastive"]
    total = torch_module.exp(-log_var_cal) * loss_cal + log_var_cal
    effective_ctr_weight = 0.0

    # Only include the contrastive branch when this batch actually has pairs.
    # Otherwise the log-variance term alone would drift and create silent bias.
    if float(lambda_contrastive) > 0.0 and used_ctr_pairs:
        ctr_term = torch_module.exp(-log_var_ctr) * loss_ctr + log_var_ctr
        total = total + float(lambda_contrastive) * ctr_term
        effective_ctr_weight = float(lambda_contrastive) * float(torch_module.exp(-log_var_ctr).detach().item())
    return total, effective_ctr_weight


def _resolve_checkpoint_brier(
    *,
    eval_metrics: dict[str, Any],
    checkpoint_selection_metric: str,
) -> float:
    """Resolve which Brier score drives best-checkpoint selection."""
    if checkpoint_selection_metric == "raw_brier":
        return float(eval_metrics["calibration"]["brier_score"])
    if checkpoint_selection_metric == "posthoc_brier":
        posthoc = eval_metrics.get("calibration_posthoc")
        if posthoc is None:
            raise ValueError(
                "Checkpoint selection requested posthoc_brier, but no post-hoc calibration metrics are available"
            )
        return float(posthoc["brier_score"])
    raise ValueError(f"Unsupported checkpoint_selection_metric: {checkpoint_selection_metric!r}")


def _write_posthoc_calibration_payload(path: Path, payload: dict[str, Any] | None) -> None:
    """Persist one post-hoc calibration payload if available."""
    if payload is None:
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_temperature_calibration_config(args: argparse.Namespace) -> TemperatureCalibrationConfig:
    """Build validated temperature-scaling config from CLI args."""
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
    """Build validated isotonic-calibration config from CLI args."""
    cfg = IsotonicCalibrationConfig(
        min_points=int(args.posthoc_isotonic_min_points),
    )
    cfg.validate()
    return cfg


def _build_calibration_sample_weights(
    *,
    targets: Any,
    parseable: Any,
    mode: str,
    floor: float,
    gamma: float,
    torch_module: Any,
) -> Any:
    """Build per-sample calibration weights from rollout-derived reliability signals.

    These weights are intentionally simple and deterministic:
    - confidence signal from distance to 0.5 success-rate
    - parseable-rate signal from rollout extraction stability
    """
    if mode == "none":
        weights = torch_module.ones_like(targets)
    else:
        confidence = torch_module.abs(targets - 0.5) * 2.0
        confidence = confidence.clamp(0.0, 1.0)
        parseable_clamped = parseable.clamp(0.0, 1.0)
        if mode == "confidence":
            weights = confidence
        elif mode == "entropy_inverse":
            eps = 1e-6
            p = targets.clamp(eps, 1.0 - eps)
            entropy = -(p * torch_module.log(p) + (1.0 - p) * torch_module.log(1.0 - p))
            entropy_norm = entropy / float(torch_module.log(torch_module.tensor(2.0, device=targets.device)))
            weights = (1.0 - entropy_norm).clamp(0.0, 1.0)
        elif mode == "parseable":
            weights = parseable_clamped
        elif mode == "confidence_parseable":
            weights = (confidence * parseable_clamped).clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unsupported calibration sample weighting mode: {mode!r}")
    if floor > 0.0:
        weights = weights.clamp_min(float(floor))
    if gamma != 1.0:
        weights = weights.pow(float(gamma))
    return weights


def _apply_contrastive_pair_filter(
    *,
    batch_indices: Any,
    base_mask: Any,
    train_cache: dict[str, Any],
    pair_filter: str,
    confidence_threshold: float,
    parseable_threshold: float,
    torch_module: Any,
) -> Any:
    """Filter contrastive pairs using rollout-derived confidence/parseable gates."""
    if pair_filter == "none":
        return base_mask

    confidence = train_cache["target_confidence"][batch_indices]
    parseable = train_cache["target_parseable"][batch_indices]

    if pair_filter == "confidence":
        extra_mask = confidence >= float(confidence_threshold)
    elif pair_filter == "parseable":
        extra_mask = parseable >= float(parseable_threshold)
    elif pair_filter == "confidence_parseable":
        extra_mask = (confidence >= float(confidence_threshold)) & (
            parseable >= float(parseable_threshold)
        )
    else:
        raise ValueError(f"Unsupported contrastive pair filter: {pair_filter!r}")

    if extra_mask.dtype != torch_module.bool:
        extra_mask = extra_mask.to(dtype=torch_module.bool)
    return base_mask & extra_mask


def _apply_target_smoothing(targets: Any, *, epsilon: float) -> Any:
    """Apply optional label smoothing to rollout targets.

    For soft rollout targets in [0, 1], smoothing is:
    y' = (1 - 2*eps) * y + eps
    """
    if epsilon <= 0.0:
        return targets
    return ((1.0 - 2.0 * float(epsilon)) * targets) + float(epsilon)


def _apply_contrastive_score_gap_filter(
    *,
    clean_scores_for_pairs: Any,
    corrupt_scores: Any,
    score_gap_min: float,
    score_gap_max: float,
    torch_module: Any,
) -> tuple[Any, Any]:
    """Filter contrastive pairs by current score-gap band.

    Gap is defined as `clean_score - corrupt_score`.
    """
    if clean_scores_for_pairs.numel() == 0:
        return clean_scores_for_pairs, corrupt_scores
    if score_gap_min <= -1.0 and score_gap_max >= 1.0:
        return clean_scores_for_pairs, corrupt_scores

    gaps = clean_scores_for_pairs - corrupt_scores
    keep_mask = (gaps >= float(score_gap_min)) & (gaps <= float(score_gap_max))
    if keep_mask.dtype != torch_module.bool:
        keep_mask = keep_mask.to(dtype=torch_module.bool)
    if not bool(keep_mask.any().item()):
        empty = clean_scores_for_pairs[:0]
        return empty, corrupt_scores[:0]
    return clean_scores_for_pairs[keep_mask], corrupt_scores[keep_mask]


def _encode_example_cache(
    *,
    examples: list[ValueSupervisionExample],
    backbone: Any,
    tokenizer: Any,
    torch_module: Any,
    max_length: int,
    batch_size: int,
    use_primary_corruption: bool,
    calibration_sample_weighting: str,
    calibration_weight_floor: float,
    calibration_weight_gamma: float,
) -> dict[str, Any]:
    """Encode clean examples and optional primary corruptions into cached tensors."""
    clean_features = _encode_text_list_in_batches(
        texts=[example.clean_input_text() for example in examples],
        backbone=backbone,
        tokenizer=tokenizer,
        torch_module=torch_module,
        max_length=max_length,
        batch_size=batch_size,
        progress_label="cache_train_clean" if use_primary_corruption else "cache_eval_clean",
        progress_every_batches=32,
    )
    device = clean_features.device
    targets = torch_module.tensor(
        [example.target_success_rate for example in examples],
        dtype=torch_module.float32,
        device=device,
    )
    target_parseable = torch_module.tensor(
        [example.target_parseable_rate for example in examples],
        dtype=torch_module.float32,
        device=device,
    )
    target_confidence = torch_module.abs(targets - 0.5) * 2.0
    calibration_weights = _build_calibration_sample_weights(
        targets=targets,
        parseable=target_parseable,
        mode=calibration_sample_weighting,
        floor=float(calibration_weight_floor),
        gamma=float(calibration_weight_gamma),
        torch_module=torch_module,
    )
    has_primary_corruption = torch_module.tensor(
        [example.has_primary_corruption() if use_primary_corruption else False for example in examples],
        dtype=torch_module.bool,
        device=device,
    )
    corruption_features = torch_module.zeros_like(clean_features)
    if use_primary_corruption and bool(has_primary_corruption.any().item()):
        corruption_indices = [idx for idx, example in enumerate(examples) if example.has_primary_corruption()]
        corruption_texts = [examples[idx].primary_corruption_input_text() for idx in corruption_indices]
        encoded_corrupt = _encode_text_list_in_batches(
            texts=corruption_texts,
            backbone=backbone,
            tokenizer=tokenizer,
            torch_module=torch_module,
            max_length=max_length,
            batch_size=batch_size,
            progress_label="cache_train_primary_corrupt",
            progress_every_batches=32,
        )
        for local_idx, global_idx in enumerate(corruption_indices):
            corruption_features[global_idx] = encoded_corrupt[local_idx]

    return {
        "clean_features": clean_features,
        "targets": targets,
        "target_parseable": target_parseable,
        "target_confidence": target_confidence,
        "calibration_weights": calibration_weights,
        "has_primary_corruption": has_primary_corruption,
        "primary_corruption_features": corruption_features,
    }


def _encode_corruption_variant_cache(
    *,
    variants: list[CorruptionVariant],
    backbone: Any,
    tokenizer: Any,
    torch_module: Any,
    max_length: int,
    batch_size: int,
) -> dict[str, Any]:
    """Encode all eval corruption variants into one cached tensor."""
    if not variants:
        hidden_size = infer_backbone_hidden_size(backbone)
        empty = torch_module.zeros((0, hidden_size), dtype=torch_module.float32, device=_resolve_value_device(backbone, torch_module))
        return {"corruption_features": empty}
    corruption_features = _encode_text_list_in_batches(
        texts=[variant.corrupted_input_text() for variant in variants],
        backbone=backbone,
        tokenizer=tokenizer,
        torch_module=torch_module,
        max_length=max_length,
        batch_size=batch_size,
        progress_label="cache_eval_corruptions",
        progress_every_batches=32,
    )
    return {"corruption_features": corruption_features}


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
):
    """Encode a list of texts into one stacked pooled-feature tensor.

    C2 trains the small value head in float32 by default. Cast cached features
    to float32 here so backbone runtime dtype (often bf16) cannot trigger silent
    matmul dtype mismatches later in training/evaluation.

    Parameters
    ----------
    progress_label:
        Short tag used in console progress logs.
    progress_every_batches:
        Log every N batches. Use 0 to disable periodic logs.
    """
    chunks = []
    total_texts = len(texts)
    if total_texts == 0:
        return torch_module.zeros((0, infer_backbone_hidden_size(backbone)), dtype=torch_module.float32, device=_resolve_value_device(backbone, torch_module))

    total_batches = (total_texts + batch_size - 1) // batch_size
    print(
        f"{progress_label:16s}: start {total_texts} texts in {total_batches} batches (bs={batch_size})",
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
            progress_every_batches > 0
            and (batch_idx % progress_every_batches == 0 or batch_idx == total_batches)
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
    """Simple linear warmup + linear decay schedule factor."""
    if total_steps <= 1:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return max(float(step + 1) / float(warmup_steps), 1e-8)
    remaining = max(total_steps - step, 0)
    decay_steps = max(total_steps - warmup_steps, 1)
    return max(float(remaining) / float(decay_steps), 0.0)


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
        raise TypeError(f"Config JSON {path} must contain an object at the top level")
    return payload


def _resolve_tokenizer_load_path(model_path: str, adapter_path: Path | None) -> str:
    """Choose which directory should supply tokenizer files."""
    if adapter_path is None:
        return str(model_path)
    if (adapter_path / "tokenizer_config.json").exists():
        return str(adapter_path)
    return str(model_path)


def _attach_peft_adapter_for_inference(model: Any, adapter_path: Path):
    """Attach a PEFT adapter to a loaded base model."""
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    try:
        from peft import PeftModel  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to import `peft` while attaching adapter for C2") from exc
    return PeftModel.from_pretrained(model, str(adapter_path))


def _resolve_value_device(backbone: Any, torch_module: Any):
    """Resolve the device that should host cached features and the value head."""
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
    """Append one dictionary row to a JSONL file."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
