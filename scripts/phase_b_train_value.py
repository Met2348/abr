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
import hashlib
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
from ours.phase_b.feature_cache import (  # noqa: E402
    build_backbone_signature,
    build_cache_key,
    feature_cache_can_read,
    feature_cache_can_write,
    hash_float_list,
    hash_int_list,
    hash_jsonable,
    hash_text_list,
    move_tensors_to_device,
    save_feature_cache,
    try_load_feature_cache,
    validate_feature_cache_mode,
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
    anti_saturation_logit_penalty,
    binary_cross_entropy_calibration_loss,
    contrastive_margin_loss,
    mean_squared_calibration_loss,
    mixed_calibration_loss,
)
from ours.phase_d.external_pairs import (  # noqa: E402
    ExternalPairRecord,
    load_external_pair_jsonl,
)


@dataclass(slots=True)
class TrainConfig:
    """Compact snapshot of C2 hyperparameters persisted in the manifest."""

    max_length: int
    train_mode: str
    two_stage_ranking_ratio: float
    contrastive_max_corruptions_per_prefix: int
    external_pair_jsonl: str | None
    external_pair_weight: float
    external_pair_max_train_samples: int | None
    external_pair_source_balance: str
    external_pair_permutation_mode: str
    external_pair_domain_filter: str | None
    external_pair_min_confidence: float
    external_pair_use_confidence_weights: bool
    external_pair_only: bool
    target_source: str
    target_source_missing_policy: str
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
    anti_saturation_weight: float
    anti_saturation_logit_threshold: float
    lambda_contrastive: float
    contrastive_margin: float
    contrastive_pair_filter: str
    contrastive_confidence_threshold: float
    contrastive_parseable_threshold: float
    contrastive_label_delta_q_min: float
    contrastive_label_z_min: float
    contrastive_label_pair_weight_min: float
    contrastive_require_pair_pass_gate: bool
    contrastive_score_gap_min: float
    contrastive_score_gap_max: float
    contrastive_use_pair_weights: bool
    contrastive_stratified_sampling: bool
    contrastive_stratify_step_bucket_size: int
    contrastive_stratify_include_no_corruption: bool
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
    strict_determinism: bool
    logging_steps: int
    feature_cache_root: str
    feature_cache_mode: str
    feature_cache_lock_timeout_sec: float
    init_value_head_path: str | None

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
    parser.add_argument(
        "--init-value-head-path",
        type=Path,
        default=None,
        help=(
            "Optional warm-start checkpoint for the small value head itself. "
            "This is the key hook for bridge experiments: stage-1 can learn on "
            "external triplets, then stage-2 continues on in-domain C1/CQR pairs "
            "from the saved `best_value_head.pt`."
        ),
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-corruption-variants-eval", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--feature-cache-root",
        type=Path,
        default=Path("assets/artifacts/phase_c_feature_cache"),
        help=(
            "Persistent on-disk cache root for frozen-backbone encoded features. "
            "Shared safely across reruns when signatures match."
        ),
    )
    parser.add_argument(
        "--feature-cache-mode",
        choices=["off", "read", "write", "read_write"],
        default="read_write",
        help=(
            "Feature-cache behavior: "
            "`off` disables cache; "
            "`read` only reuse existing cache; "
            "`write` only write new cache; "
            "`read_write` reuse and populate."
        ),
    )
    parser.add_argument(
        "--feature-cache-lock-timeout-sec",
        type=float,
        default=600.0,
        help="Lock wait timeout for safe concurrent cache writes.",
    )
    parser.add_argument(
        "--train-mode",
        choices=["joint", "ranking_only", "calibration_only", "two_stage"],
        default="joint",
        help=(
            "C2 optimization mode. "
            "`joint` keeps legacy mixed training; "
            "`ranking_only` trains contrastive branch only; "
            "`calibration_only` disables contrastive updates; "
            "`two_stage` runs ranking first, then calibration."
        ),
    )
    parser.add_argument(
        "--two-stage-ranking-ratio",
        type=float,
        default=0.5,
        help=(
            "When --train-mode=two_stage, fraction of optimizer steps spent in "
            "ranking-only stage before switching to calibration-only."
        ),
    )
    parser.add_argument(
        "--contrastive-max-corruptions-per-prefix",
        type=int,
        default=1,
        help=(
            "Maximum number of ranked corruption candidates per clean prefix "
            "loaded into contrastive training. 1 preserves historical behavior."
        ),
    )
    parser.add_argument(
        "--external-pair-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional Phase D external-pair artifact file "
            "(train_pairs.jsonl) used as additional ranking supervision."
        ),
    )
    parser.add_argument(
        "--external-pair-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for external pair ranking loss branch. "
            "0 disables external pair training."
        ),
    )
    parser.add_argument(
        "--external-pair-max-train-samples",
        type=int,
        default=None,
        help="Optional max number of external pairs loaded for training.",
    )
    parser.add_argument(
        "--external-pair-source-balance",
        choices=["none", "uniform"],
        default="none",
        help=(
            "Sampling policy for external pairs. "
            "`uniform` does round-robin across source_tag buckets."
        ),
    )
    parser.add_argument(
        "--external-pair-permutation-mode",
        choices=["random", "stable_hash"],
        default="random",
        help=(
            "How to permute external pairs inside each epoch. "
            "`random` follows seed-dependent shuffling; "
            "`stable_hash` uses pair_id-hash deterministic ordering to reduce seed variance."
        ),
    )
    parser.add_argument(
        "--external-pair-domain-filter",
        default="",
        help=(
            "Comma-separated domain_tag allow-list for external pairs. "
            "Empty means keep all domains."
        ),
    )
    parser.add_argument(
        "--external-pair-min-confidence",
        type=float,
        default=0.0,
        help="Drop external pairs with confidence lower than this threshold.",
    )
    parser.add_argument(
        "--external-pair-use-confidence-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pair_confidence as per-pair sample weights in external ranking loss.",
    )
    parser.add_argument(
        "--external-pair-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable internal C1 clean-vs-corrupt contrastive updates and train "
            "ranking from external pairs only."
        ),
    )
    parser.add_argument(
        "--target-source",
        choices=["q_mean_smoothed", "q_teacher", "q_fused"],
        default="q_mean_smoothed",
        help=(
            "Supervision target source for C2. "
            "`q_mean_smoothed`=MC baseline, `q_teacher`=teacher-only, "
            "`q_fused`=D2 fused labels."
        ),
    )
    parser.add_argument(
        "--target-source-missing-policy",
        choices=["fail", "fallback_mc"],
        default="fail",
        help=(
            "How to handle missing target-source values. "
            "`fail` is strict for promotion runs; `fallback_mc` uses q_mean_smoothed."
        ),
    )

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
        choices=[
            "none",
            "confidence",
            "entropy_inverse",
            "parseable",
            "confidence_parseable",
            "q_weight",
            "q_weight_parseable",
        ],
        default="none",
        help=(
            "Optional per-sample weighting for calibration losses. "
            "`confidence` upweights targets far from 0.5; "
            "`entropy_inverse` downweights high-uncertainty targets; "
            "`parseable` uses rollout parseable-rate; "
            "`confidence_parseable` multiplies confidence and parseable signals; "
            "`q_weight` uses C1 uncertainty-derived reliability; "
            "`q_weight_parseable` multiplies reliability and parseable signals."
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
    parser.add_argument(
        "--anti-saturation-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for anti-saturation regularization on value-head logits. "
            "0 disables this branch."
        ),
    )
    parser.add_argument(
        "--anti-saturation-logit-threshold",
        type=float,
        default=4.0,
        help=(
            "Safe-band threshold for anti-saturation penalty. "
            "Penalty applies on relu(|logit|-threshold)^2."
        ),
    )
    parser.add_argument("--lambda-contrastive", type=float, default=1.0)
    parser.add_argument("--contrastive-margin", type=float, default=0.1)
    parser.add_argument(
        "--contrastive-pair-filter",
        choices=[
            "none",
            "confidence",
            "parseable",
            "confidence_parseable",
            "label_quality",
            "confidence_parseable_label",
        ],
        default="none",
        help=(
            "Optional noisy-pair filtering for contrastive training. "
            "Filters are computed from clean-prefix rollout targets and/or "
            "C1 pair-quality labels."
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
        "--contrastive-label-delta-q-min",
        type=float,
        default=0.0,
        help="Minimum label-side delta_q to include a contrastive pair when label-quality filtering is enabled.",
    )
    parser.add_argument(
        "--contrastive-label-z-min",
        type=float,
        default=0.0,
        help="Minimum label-side z_delta to include a contrastive pair when label-quality filtering is enabled.",
    )
    parser.add_argument(
        "--contrastive-label-pair-weight-min",
        type=float,
        default=0.0,
        help="Minimum label-side pair_weight to include a contrastive pair when label-quality filtering is enabled.",
    )
    parser.add_argument(
        "--contrastive-require-pair-pass-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require pair_pass_gate=true from C1 pair_quality records when label-quality filtering is enabled.",
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
        "--contrastive-use-pair-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use C1 label-side pair_weight as a per-pair weight in the contrastive margin loss.",
    )
    parser.add_argument(
        "--contrastive-stratified-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable type-aware stratified sampling for C2 training batches. "
            "When enabled, examples are interleaved across "
            "`primary_corruption_type x prefix_step_bucket` strata."
        ),
    )
    parser.add_argument(
        "--contrastive-stratify-step-bucket-size",
        type=int,
        default=2,
        help=(
            "Bucket size for prefix_step_index in stratified sampling. "
            "Example: 0-1, 2-3, 4-5 when bucket size is 2."
        ),
    )
    parser.add_argument(
        "--contrastive-stratify-include-no-corruption",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When stratified sampling is enabled, include no-primary-corruption "
            "examples in dedicated strata so calibration coverage remains stable."
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
        choices=["raw_brier", "posthoc_brier", "corr_pair_acc", "corr_auc", "ranking_score"],
        default="raw_brier",
        help=(
            "Metric used for best-checkpoint selection. "
            "`raw_brier`/`posthoc_brier` are calibration-driven (lower is better); "
            "`corr_pair_acc`/`corr_auc`/`ranking_score` are ranking-driven (higher is better). "
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
        "--strict-determinism",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable stricter deterministic backend settings "
            "(deterministic kernels, disable cudnn benchmark, disable TF32). "
            "Useful for seed-stability diagnosis."
        ),
    )
    parser.add_argument(
        "--save-best-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist the best eval-Brier checkpoint separately from the final head.",
    )
    return parser


def _initialize_value_head_from_checkpoint(
    *,
    value_head: Any,
    current_config: ValueHeadConfig,
    checkpoint_path: Path | None,
) -> dict[str, Any] | None:
    """Warm-start the current value head from an earlier saved checkpoint.

    This is the bridge hook for Phase D:
    1. Stage-1 learns ranking on external triplets.
    2. Stage-2 reuses the learned head weights and continues on in-domain
       StrategyQA C1/CQR supervision.

    We do not restore optimizer/scheduler state here. The point is to transfer
    the head parameters, not to resume one interrupted run.
    """
    if checkpoint_path is None:
        return None
    resolved = Path(checkpoint_path)
    init_head, init_config, init_extra = load_value_head_checkpoint(resolved, map_location="cpu")
    if int(init_config.hidden_size) != int(current_config.hidden_size):
        raise ValueError(
            "Warm-start checkpoint hidden_size mismatch: "
            f"checkpoint={init_config.hidden_size}, current={current_config.hidden_size}"
        )
    if str(init_config.pooling) != str(current_config.pooling):
        raise ValueError(
            "Warm-start checkpoint pooling mismatch: "
            f"checkpoint={init_config.pooling!r}, current={current_config.pooling!r}"
        )
    incompatible = value_head.load_state_dict(init_head.state_dict(), strict=True)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        raise RuntimeError(
            "Warm-start checkpoint could not be loaded cleanly: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "path": str(resolved),
        "checkpoint_config": init_config.to_dict(),
        "checkpoint_extra_state": init_extra,
        "config_mismatch_notes": {
            # dropout/init_std do not change tensor shapes, so we allow them but
            # persist the difference for later debugging and report writing.
            "dropout_prob_changed": (
                float(init_config.dropout_prob) != float(current_config.dropout_prob)
            ),
            "init_std_changed": float(init_config.init_std) != float(current_config.init_std),
        },
    }


def _filter_corruptions_to_loaded_eval_examples(
    *,
    eval_examples: list[ValueSupervisionExample],
    eval_corruptions: list[CorruptionVariant],
    max_variants: int | None,
) -> tuple[list[CorruptionVariant], dict[str, int]]:
    """Keep only corruption rows whose clean prefix still exists in eval examples.

    Why this matters:
    - `--max-eval-samples` truncates the clean eval prefix table.
    - Corruption artifacts are stored separately and therefore do not
      automatically follow that truncation.
    - If we do not realign them here, later evaluation will try to look up a
      clean prefix score that was never computed and crash with `KeyError`.
    """
    allowed_prefix_ids = {str(example.prefix_id) for example in eval_examples}
    filtered = [
        variant
        for variant in eval_corruptions
        if str(variant.clean_prefix_id) in allowed_prefix_ids
    ]
    stats = {
        "before_alignment": int(len(eval_corruptions)),
        "after_alignment": int(len(filtered)),
        "dropped_for_missing_clean_prefix": int(len(eval_corruptions) - len(filtered)),
    }
    if max_variants is not None:
        filtered = filtered[: int(max_variants)]
    stats["after_max_variants"] = int(len(filtered))
    return filtered, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and optional config defaults."""
    parser = _build_parser()
    # Parse in two passes so config-json defaults are loaded before explicit CLI overrides.
    # 两段解析：先吸收 config-json 默认值，再解析完整 CLI 覆盖。
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
    if float(args.feature_cache_lock_timeout_sec) <= 0.0:
        raise ValueError("--feature-cache-lock-timeout-sec must be > 0")
    args.feature_cache_mode = validate_feature_cache_mode(str(args.feature_cache_mode))
    if not (0.0 < args.two_stage_ranking_ratio < 1.0):
        raise ValueError("--two-stage-ranking-ratio must be in (0, 1)")
    if args.contrastive_max_corruptions_per_prefix <= 0:
        raise ValueError("--contrastive-max-corruptions-per-prefix must be > 0")
    if args.external_pair_max_train_samples is not None and args.external_pair_max_train_samples <= 0:
        raise ValueError("--external-pair-max-train-samples must be > 0")
    if args.external_pair_weight < 0.0:
        raise ValueError("--external-pair-weight must be >= 0")
    if not (0.0 <= args.external_pair_min_confidence <= 1.0):
        raise ValueError("--external-pair-min-confidence must be in [0, 1]")
    if args.external_pair_weight > 0.0 and args.external_pair_jsonl is None:
        raise ValueError("--external-pair-weight > 0 requires --external-pair-jsonl")
    if args.external_pair_jsonl is not None and not Path(args.external_pair_jsonl).exists():
        raise FileNotFoundError(f"External pair JSONL not found: {args.external_pair_jsonl}")
    if bool(args.external_pair_only) and args.external_pair_weight <= 0.0:
        raise ValueError("--external-pair-only requires --external-pair-weight > 0")
    if args.target_source not in {"q_mean_smoothed", "q_teacher", "q_fused"}:
        raise ValueError("--target-source has an unsupported value")
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
    if not (-1.0 <= args.contrastive_label_delta_q_min <= 1.0):
        raise ValueError("--contrastive-label-delta-q-min must be in [-1, 1]")
    if args.contrastive_label_z_min < 0.0:
        raise ValueError("--contrastive-label-z-min must be >= 0")
    if not (0.0 <= args.contrastive_label_pair_weight_min <= 1.0):
        raise ValueError("--contrastive-label-pair-weight-min must be in [0, 1]")
    if args.contrastive_score_gap_min > args.contrastive_score_gap_max:
        raise ValueError("--contrastive-score-gap-min must be <= --contrastive-score-gap-max")
    if not (-1.0 <= args.contrastive_score_gap_min <= 1.0):
        raise ValueError("--contrastive-score-gap-min must be in [-1, 1]")
    if not (-1.0 <= args.contrastive_score_gap_max <= 1.0):
        raise ValueError("--contrastive-score-gap-max must be in [-1, 1]")
    if args.contrastive_stratify_step_bucket_size <= 0:
        raise ValueError("--contrastive-stratify-step-bucket-size must be > 0")
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
    if args.anti_saturation_weight < 0.0:
        raise ValueError("--anti-saturation-weight must be >= 0")
    if args.anti_saturation_logit_threshold <= 0.0:
        raise ValueError("--anti-saturation-logit-threshold must be > 0")
    if args.calibration_loss == "bce_mse" and (
        args.calibration_mse_weight == 0.0 and args.calibration_bce_weight == 0.0
    ):
        raise ValueError("bce_mse requires at least one non-zero calibration component weight")
    if args.adaptive_loss_balancing != "none" and not args.use_contrastive_loss:
        raise ValueError("--adaptive-loss-balancing requires --use-contrastive-loss")
    if args.train_mode != "joint" and args.adaptive_loss_balancing != "none":
        raise ValueError(
            "--adaptive-loss-balancing currently supports --train-mode=joint only"
        )
    if args.train_mode == "ranking_only" and not args.use_contrastive_loss:
        raise ValueError("--train-mode=ranking_only requires --use-contrastive-loss")
    if bool(args.external_pair_only) and not bool(args.use_contrastive_loss):
        raise ValueError("--external-pair-only requires --use-contrastive-loss")
    if args.train_mode == "two_stage" and not args.use_contrastive_loss:
        raise ValueError("--train-mode=two_stage requires --use-contrastive-loss")
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

    # Stage 1 validates artifact contracts before any model load so dirty data fails fast on CPU.
    # Stage 1: 先做 artifact 契约检查，再加载模型，避免 GPU 时间浪费在脏数据上。
    train_examples, train_manifest = load_value_supervision_examples(
        args.train_dir,
        max_samples=args.max_train_samples,
        require_corruptions=bool(args.use_contrastive_loss),
        max_primary_corruptions=int(args.contrastive_max_corruptions_per_prefix),
    )
    eval_examples, eval_manifest = load_value_supervision_examples(
        args.eval_dir,
        max_samples=args.max_eval_samples,
        require_corruptions=False,
        max_primary_corruptions=1,
    )
    eval_corruptions, _ = load_corruption_variants(
        args.eval_dir,
        max_variants=None,
    )
    eval_corruptions, eval_corruption_alignment_stats = _filter_corruptions_to_loaded_eval_examples(
        eval_examples=eval_examples,
        eval_corruptions=eval_corruptions,
        max_variants=args.max_corruption_variants_eval,
    )
    assert_phase_c_compatibility(train_manifest, eval_manifest)

    # D3 centralizes target-source routing so missing teacher/fused labels never trigger silent fallbacks.
    # D3: 根据 target_source 选择监督目标列。
    # 这一步统一处理 teacher/fused 缺失策略，避免后续训练环节静默回退。
    train_target_values, train_target_source_stats = _resolve_supervision_targets(
        examples=train_examples,
        target_source=str(args.target_source),
        missing_policy=str(args.target_source_missing_policy),
        split_label="train",
    )
    eval_target_values, eval_target_source_stats = _resolve_supervision_targets(
        examples=eval_examples,
        target_source=str(args.target_source),
        missing_policy=str(args.target_source_missing_policy),
        split_label="eval",
    )
    external_pairs: list[ExternalPairRecord] = []
    external_pair_stats: dict[str, Any] | None = None
    external_domain_filter = _parse_csv_allow_list(str(args.external_pair_domain_filter))
    if args.external_pair_jsonl is not None:
        # External pairs are D-stage ranking supervision, so filter them on CPU before any expensive encoding.
        # external pair 是 D 阶段从外部数据构造的排序监督。
        # 这里先在 CPU 侧做 domain/confidence 过滤，避免后面把一堆无效 pair 也编码进 cache。
        external_pairs, external_pair_stats = load_external_pair_jsonl(
            Path(args.external_pair_jsonl),
            max_samples=args.external_pair_max_train_samples,
            min_confidence=float(args.external_pair_min_confidence),
            allowed_domains=external_domain_filter,
        )
        if args.external_pair_weight > 0.0 and not external_pairs:
            # Treat an empty filtered branch as a hard configuration error, not as an implicit ablation.
            # 这是典型配置错误，例如 min_confidence 设太高，必须显式失败而不是悄悄退化成无 external pair。
            raise ValueError(
                "External pair branch was enabled, but no external pairs survived filtering"
            )

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
    _set_seed(
        args.seed,
        torch,
        strict_determinism=bool(args.strict_determinism),
    )
    # Seed setup must happen before any random sampling, otherwise seed-stability diagnosis becomes meaningless.
    # seed 设置必须放在所有随机行为之前，否则尤其在 D6-T 里会把“同 seed 复现”做成假象。

    train_cfg = TrainConfig(
        max_length=int(args.max_length),
        train_mode=str(args.train_mode),
        two_stage_ranking_ratio=float(args.two_stage_ranking_ratio),
        contrastive_max_corruptions_per_prefix=int(args.contrastive_max_corruptions_per_prefix),
        external_pair_jsonl=(
            str(args.external_pair_jsonl) if args.external_pair_jsonl is not None else None
        ),
        external_pair_weight=float(args.external_pair_weight),
        external_pair_max_train_samples=(
            int(args.external_pair_max_train_samples)
            if args.external_pair_max_train_samples is not None
            else None
        ),
        external_pair_source_balance=str(args.external_pair_source_balance),
        external_pair_permutation_mode=str(args.external_pair_permutation_mode),
        external_pair_domain_filter=(
            ",".join(sorted(external_domain_filter)) if external_domain_filter else None
        ),
        external_pair_min_confidence=float(args.external_pair_min_confidence),
        external_pair_use_confidence_weights=bool(args.external_pair_use_confidence_weights),
        external_pair_only=bool(args.external_pair_only),
        target_source=str(args.target_source),
        target_source_missing_policy=str(args.target_source_missing_policy),
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
        anti_saturation_weight=float(args.anti_saturation_weight),
        anti_saturation_logit_threshold=float(args.anti_saturation_logit_threshold),
        lambda_contrastive=float(args.lambda_contrastive if args.use_contrastive_loss else 0.0),
        contrastive_margin=float(args.contrastive_margin),
        contrastive_pair_filter=str(args.contrastive_pair_filter),
        contrastive_confidence_threshold=float(args.contrastive_confidence_threshold),
        contrastive_parseable_threshold=float(args.contrastive_parseable_threshold),
        contrastive_label_delta_q_min=float(args.contrastive_label_delta_q_min),
        contrastive_label_z_min=float(args.contrastive_label_z_min),
        contrastive_label_pair_weight_min=float(args.contrastive_label_pair_weight_min),
        contrastive_require_pair_pass_gate=bool(args.contrastive_require_pair_pass_gate),
        contrastive_score_gap_min=float(args.contrastive_score_gap_min),
        contrastive_score_gap_max=float(args.contrastive_score_gap_max),
        contrastive_use_pair_weights=bool(args.contrastive_use_pair_weights),
        contrastive_stratified_sampling=bool(args.contrastive_stratified_sampling),
        contrastive_stratify_step_bucket_size=int(args.contrastive_stratify_step_bucket_size),
        contrastive_stratify_include_no_corruption=bool(
            args.contrastive_stratify_include_no_corruption
        ),
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
        strict_determinism=bool(args.strict_determinism),
        logging_steps=int(args.logging_steps),
        feature_cache_root=str(args.feature_cache_root),
        feature_cache_mode=str(args.feature_cache_mode),
        feature_cache_lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
        init_value_head_path=(
            str(args.init_value_head_path) if args.init_value_head_path is not None else None
        ),
    )

    run_dir = args.output_root / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Materialize every intermediate/final artifact so suites and gate scripts never depend on console logs.
    # 所有中间/最终产物都显式落盘到 run_dir，后续 suite 汇总、门控脚本和诊断才能稳定工作。
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
    print(
        f"init_value_head   : {args.init_value_head_path if args.init_value_head_path is not None else '<none>'}"
    )
    print(f"train_mode        : {args.train_mode}")
    print(f"stage_ratio       : {args.two_stage_ranking_ratio}")
    print(f"target_source     : {args.target_source}")
    print(f"target_missing    : {args.target_source_missing_policy}")
    print(f"target_cov_train  : {train_target_source_stats['coverage_ratio']:.4f}")
    print(f"target_cov_eval   : {eval_target_source_stats['coverage_ratio']:.4f}")
    print(f"teacher_dis_train : {train_target_source_stats['teacher_disagree_ratio']:.4f}")
    print(f"teacher_dis_eval  : {eval_target_source_stats['teacher_disagree_ratio']:.4f}")
    print(f"train_examples    : {len(train_examples)}")
    print(f"eval_examples     : {len(eval_examples)}")
    print(f"eval_corruptions  : {len(eval_corruptions)}")
    if eval_corruption_alignment_stats["dropped_for_missing_clean_prefix"] > 0:
        print(
            "eval_corr_align  : "
            f"dropped={eval_corruption_alignment_stats['dropped_for_missing_clean_prefix']} "
            f"from={eval_corruption_alignment_stats['before_alignment']} "
            f"after={eval_corruption_alignment_stats['after_alignment']}"
        )
    print(f"contrastive_loss  : {args.use_contrastive_loss}")
    print(f"calibration_loss  : {args.calibration_loss}")
    print(f"calib_target_eps : {args.calibration_target_smoothing}")
    print(f"calib_weight_mode : {args.calibration_sample_weighting}")
    print(f"calib_weight_floor: {args.calibration_weight_floor}")
    print(f"calib_weight_gamma: {args.calibration_weight_gamma}")
    print(f"anti_sat_weight  : {args.anti_saturation_weight}")
    print(f"anti_sat_thr     : {args.anti_saturation_logit_threshold}")
    print(f"adaptive_balance  : {args.adaptive_loss_balancing}")
    print(f"pair_filter       : {args.contrastive_pair_filter}")
    print(f"pair_conf_thr     : {args.contrastive_confidence_threshold}")
    print(f"pair_parse_thr    : {args.contrastive_parseable_threshold}")
    print(f"pair_deltaq_thr   : {args.contrastive_label_delta_q_min}")
    print(f"pair_z_thr        : {args.contrastive_label_z_min}")
    print(f"pair_w_thr        : {args.contrastive_label_pair_weight_min}")
    print(f"pair_req_gate     : {args.contrastive_require_pair_pass_gate}")
    print(f"pair_gap_min      : {args.contrastive_score_gap_min}")
    print(f"pair_gap_max      : {args.contrastive_score_gap_max}")
    print(f"pair_weight_loss  : {args.contrastive_use_pair_weights}")
    print(f"pair_stratified   : {args.contrastive_stratified_sampling}")
    print(f"pair_step_bucket  : {args.contrastive_stratify_step_bucket_size}")
    print(f"pair_include_noc  : {args.contrastive_stratify_include_no_corruption}")
    print(f"pair_k_per_prefix : {args.contrastive_max_corruptions_per_prefix}")
    print(f"external_pair_file: {args.external_pair_jsonl if args.external_pair_jsonl is not None else '<none>'}")
    print(f"external_pair_w   : {args.external_pair_weight}")
    print(f"external_pair_bal : {args.external_pair_source_balance}")
    print(f"external_pair_ord : {args.external_pair_permutation_mode}")
    print(f"external_pair_dom : {args.external_pair_domain_filter if args.external_pair_domain_filter else '<all>'}")
    print(f"external_pair_minc: {args.external_pair_min_confidence}")
    print(f"external_pair_conf: {args.external_pair_use_confidence_weights}")
    print(f"external_pair_only: {bool(args.external_pair_only)}")
    print(f"external_num_pairs: {len(external_pairs)}")
    print(f"posthoc_calib     : {args.posthoc_calibration}")
    print(f"ckpt_metric       : {args.checkpoint_selection_metric}")
    print(f"batch_train       : {args.per_device_train_batch_size}")
    print(f"batch_eval        : {args.per_device_eval_batch_size}")
    print(f"max_length        : {args.max_length}")
    print(f"feat_cache_mode   : {args.feature_cache_mode}")
    print(f"feat_cache_root   : {args.feature_cache_root}")
    print(f"seed              : {args.seed}")
    print(f"strict_determ     : {bool(args.strict_determinism)}")
    if external_pair_stats is not None:
        print(
            "external_sources : "
            + ", ".join(
                f"{k}:{v}" for k, v in sorted(external_pair_stats.get("by_source", {}).items())
            )
        )
    print("=" * 88)

    # Stage 2 freezes the backbone and caches features once so the rest of training is cheap and comparable.
    # Stage 2: 冻结 backbone 并一次性缓存特征；后续仅训练小 value head。
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
    init_value_head_info = _initialize_value_head_from_checkpoint(
        value_head=value_head,
        current_config=value_head_config,
        checkpoint_path=args.init_value_head_path,
    )
    if init_value_head_info is not None:
        print(
            "init_value_head_ok: "
            f"path={init_value_head_info['path']} "
            f"dropout_changed={init_value_head_info['config_mismatch_notes']['dropout_prob_changed']} "
            f"init_std_changed={init_value_head_info['config_mismatch_notes']['init_std_changed']}"
        )
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
    backbone_signature = build_backbone_signature(
        model_path=str(model_path),
        adapter_path=(str(adapter_path) if adapter_path is not None else None),
        tokenizer_path=str(tokenizer_path),
        dtype=str(args.dtype),
        max_length=int(args.max_length),
    )

    train_cache_signature = _build_value_example_cache_signature_payload(
        examples=train_examples,
        target_values=train_target_values,
        use_primary_corruption=bool(args.use_contrastive_loss),
        contrastive_max_corruptions_per_prefix=int(args.contrastive_max_corruptions_per_prefix),
        calibration_sample_weighting=str(args.calibration_sample_weighting),
        calibration_weight_floor=float(args.calibration_weight_floor),
        calibration_weight_gamma=float(args.calibration_weight_gamma),
        max_length=int(args.max_length),
        backbone_signature=backbone_signature,
    )
    train_cache_key, train_signature_hash = build_cache_key(
        "phase_b_value_train_cache",
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
                _validate_cached_example_cache_payload(
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
            target_values=train_target_values,
            backbone=backbone,
            tokenizer=tokenizer,
            torch_module=torch,
            max_length=args.max_length,
            batch_size=args.per_device_eval_batch_size,
            use_primary_corruption=args.use_contrastive_loss,
            contrastive_max_corruptions_per_prefix=int(args.contrastive_max_corruptions_per_prefix),
            calibration_sample_weighting=args.calibration_sample_weighting,
            calibration_weight_floor=args.calibration_weight_floor,
            calibration_weight_gamma=args.calibration_weight_gamma,
        )
        if feature_cache_can_write(str(args.feature_cache_mode)):
            save_feature_cache(
                cache_root=feature_cache_root,
                cache_key=train_cache_key,
                signature_hash=train_signature_hash,
                payload=train_cache,
                torch_module=torch,
                producer="scripts/phase_b_train_value.py:train_cache",
                lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
                extra_metadata={"num_examples": int(len(train_examples))},
            )
            feature_cache_stats["writes"] += 1
            feature_cache_stats["entries"]["train_cache"]["write"] = True

    eval_cache_signature = _build_value_example_cache_signature_payload(
        examples=eval_examples,
        target_values=eval_target_values,
        use_primary_corruption=False,
        contrastive_max_corruptions_per_prefix=1,
        calibration_sample_weighting="none",
        calibration_weight_floor=0.0,
        calibration_weight_gamma=1.0,
        max_length=int(args.max_length),
        backbone_signature=backbone_signature,
    )
    eval_cache_key, eval_signature_hash = build_cache_key(
        "phase_b_value_eval_cache",
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
                _validate_cached_example_cache_payload(
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
            target_values=eval_target_values,
            backbone=backbone,
            tokenizer=tokenizer,
            torch_module=torch,
            max_length=args.max_length,
            batch_size=args.per_device_eval_batch_size,
            use_primary_corruption=False,
            contrastive_max_corruptions_per_prefix=1,
            calibration_sample_weighting="none",
            calibration_weight_floor=0.0,
            calibration_weight_gamma=1.0,
        )
        if feature_cache_can_write(str(args.feature_cache_mode)):
            save_feature_cache(
                cache_root=feature_cache_root,
                cache_key=eval_cache_key,
                signature_hash=eval_signature_hash,
                payload=eval_cache,
                torch_module=torch,
                producer="scripts/phase_b_train_value.py:eval_cache",
                lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
                extra_metadata={"num_examples": int(len(eval_examples))},
            )
            feature_cache_stats["writes"] += 1
            feature_cache_stats["entries"]["eval_cache"]["write"] = True

    eval_corruption_signature = _build_corruption_variant_cache_signature_payload(
        variants=eval_corruptions,
        max_length=int(args.max_length),
        backbone_signature=backbone_signature,
    )
    eval_corruption_key, eval_corruption_hash = build_cache_key(
        "phase_b_value_eval_corruption_cache",
        eval_corruption_signature,
    )
    eval_corruption_cache = None
    if feature_cache_can_read(str(args.feature_cache_mode)):
        cached_payload, _, _ = try_load_feature_cache(
            cache_root=feature_cache_root,
            cache_key=eval_corruption_key,
            expected_signature_hash=eval_corruption_hash,
            torch_module=torch,
        )
        if cached_payload is not None:
            try:
                _validate_cached_corruption_cache_payload(
                    cache=cached_payload,
                    expected_num_variants=len(eval_corruptions),
                    expected_hidden_size=int(hidden_size),
                    torch_module=torch,
                )
                eval_corruption_cache = cached_payload
                feature_cache_stats["hits"] += 1
                feature_cache_stats["entries"]["eval_corruption_cache"] = {
                    "status": "hit",
                    "cache_key": eval_corruption_key,
                }
                print(f"feature_cache    : eval_corruption_cache hit ({eval_corruption_key})", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(
                    "feature_cache    : eval_corruption_cache invalid payload, "
                    f"fallback to re-encode ({exc})",
                    flush=True,
                )
    if eval_corruption_cache is None:
        feature_cache_stats["misses"] += 1
        feature_cache_stats["entries"]["eval_corruption_cache"] = {
            "status": "miss",
            "cache_key": eval_corruption_key,
        }
        print(f"feature_cache    : eval_corruption_cache miss ({eval_corruption_key})", flush=True)
        eval_corruption_cache = _encode_corruption_variant_cache(
            variants=eval_corruptions,
            backbone=backbone,
            tokenizer=tokenizer,
            torch_module=torch,
            max_length=args.max_length,
            batch_size=args.per_device_eval_batch_size,
        )
        if feature_cache_can_write(str(args.feature_cache_mode)):
            save_feature_cache(
                cache_root=feature_cache_root,
                cache_key=eval_corruption_key,
                signature_hash=eval_corruption_hash,
                payload=eval_corruption_cache,
                torch_module=torch,
                producer="scripts/phase_b_train_value.py:eval_corruption_cache",
                lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
                extra_metadata={"num_variants": int(len(eval_corruptions))},
            )
            feature_cache_stats["writes"] += 1
            feature_cache_stats["entries"]["eval_corruption_cache"]["write"] = True

    external_pair_cache = None
    if external_pairs:
        external_signature = _build_external_pair_cache_signature_payload(
            pairs=external_pairs,
            use_confidence_weights=bool(args.external_pair_use_confidence_weights),
            max_length=int(args.max_length),
            backbone_signature=backbone_signature,
        )
        external_key, external_hash = build_cache_key(
            "phase_b_value_external_pair_cache",
            external_signature,
        )
        if feature_cache_can_read(str(args.feature_cache_mode)):
            cached_payload, _, _ = try_load_feature_cache(
                cache_root=feature_cache_root,
                cache_key=external_key,
                expected_signature_hash=external_hash,
                torch_module=torch,
            )
            if cached_payload is not None:
                try:
                    _validate_cached_external_pair_cache_payload(
                        cache=cached_payload,
                        expected_num_pairs=len(external_pairs),
                        expected_hidden_size=int(hidden_size),
                        torch_module=torch,
                    )
                    external_pair_cache = cached_payload
                    feature_cache_stats["hits"] += 1
                    feature_cache_stats["entries"]["external_pair_cache"] = {
                        "status": "hit",
                        "cache_key": external_key,
                    }
                    print(f"feature_cache    : external_pair_cache hit ({external_key})", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(
                        "feature_cache    : external_pair_cache invalid payload, "
                        f"fallback to re-encode ({exc})",
                        flush=True,
                    )
        if external_pair_cache is None:
            feature_cache_stats["misses"] += 1
            feature_cache_stats["entries"]["external_pair_cache"] = {
                "status": "miss",
                "cache_key": external_key,
            }
            print(f"feature_cache    : external_pair_cache miss ({external_key})", flush=True)
            external_pair_cache = _encode_external_pair_cache(
                pairs=external_pairs,
                backbone=backbone,
                tokenizer=tokenizer,
                torch_module=torch,
                max_length=args.max_length,
                batch_size=args.per_device_eval_batch_size,
                use_confidence_weights=bool(args.external_pair_use_confidence_weights),
            )
            if feature_cache_can_write(str(args.feature_cache_mode)):
                save_feature_cache(
                    cache_root=feature_cache_root,
                    cache_key=external_key,
                    signature_hash=external_hash,
                    payload=external_pair_cache,
                    torch_module=torch,
                    producer="scripts/phase_b_train_value.py:external_pair_cache",
                    lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
                    extra_metadata={"num_pairs": int(len(external_pairs))},
                )
                feature_cache_stats["writes"] += 1
                feature_cache_stats["entries"]["external_pair_cache"]["write"] = True

    feature_cache_elapsed = time.perf_counter() - feature_cache_start
    if bool(args.contrastive_stratified_sampling):
        strata_summary = _summarize_strata_for_logging(
            train_cache=train_cache,
            step_bucket_size=int(args.contrastive_stratify_step_bucket_size),
            include_no_corruption=bool(args.contrastive_stratify_include_no_corruption),
        )
        print(
            "pair_strata      : "
            f"num_strata={strata_summary['num_strata']} "
            f"max={strata_summary['max_size']} "
            f"median={strata_summary['median_size']} "
            f"min={strata_summary['min_size']}",
            flush=True,
        )
        print(
            "pair_strata_top  : "
            + ", ".join(str(item) for item in strata_summary["top_strata"]),
            flush=True,
        )

    # Keep long-lived feature caches on CPU and release the frozen backbone once
    # encoding is done; training should only move mini-batches to the head device.
    # 长期缓存特征统一留在 CPU，编码结束后尽快释放 frozen backbone，训练阶段只搬小 batch。
    del backbone
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Stage 3: 仅在缓存特征上训练 value head，避免重复大模型前向。
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
    # D6: best-checkpoint selection can be calibration-driven (lower is better)
    # or ranking-driven (higher is better). Determine direction once and keep it
    # fixed for this run to avoid silent metric-direction drift.
    selection_higher_is_better = _checkpoint_metric_higher_is_better(
        str(args.checkpoint_selection_metric)
    )
    best_eval_selection_value = (
        float("-inf") if selection_higher_is_better else float("inf")
    )
    best_eval_metrics: dict[str, Any] | None = None
    best_eval_scored: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None

    train_start = time.perf_counter()
    global_step = 0
    epoch_idx = 0
    optimizer.zero_grad(set_to_none=True)

    # 用 optimizer steps 驱动而不是整 epoch，保证小数 epoch 语义准确。
    while global_step < total_steps:
        stage_name, stage_calibration_enabled, stage_contrastive_enabled = _resolve_training_stage(
            train_mode=str(args.train_mode),
            global_step=int(global_step),
            total_steps=int(total_steps),
            two_stage_ranking_ratio=float(args.two_stage_ranking_ratio),
            use_contrastive_loss=bool(args.use_contrastive_loss),
        )
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
            anti_saturation_weight=float(args.anti_saturation_weight),
            anti_saturation_logit_threshold=float(args.anti_saturation_logit_threshold),
            lambda_contrastive=(args.lambda_contrastive if args.use_contrastive_loss else 0.0),
            contrastive_margin=args.contrastive_margin,
            contrastive_pair_filter=args.contrastive_pair_filter,
            contrastive_confidence_threshold=args.contrastive_confidence_threshold,
            contrastive_parseable_threshold=args.contrastive_parseable_threshold,
            contrastive_label_delta_q_min=args.contrastive_label_delta_q_min,
            contrastive_label_z_min=args.contrastive_label_z_min,
            contrastive_label_pair_weight_min=args.contrastive_label_pair_weight_min,
            contrastive_require_pair_pass_gate=args.contrastive_require_pair_pass_gate,
            contrastive_score_gap_min=args.contrastive_score_gap_min,
            contrastive_score_gap_max=args.contrastive_score_gap_max,
            contrastive_use_pair_weights=args.contrastive_use_pair_weights,
            contrastive_stratified_sampling=args.contrastive_stratified_sampling,
            contrastive_stratify_step_bucket_size=args.contrastive_stratify_step_bucket_size,
            contrastive_stratify_include_no_corruption=args.contrastive_stratify_include_no_corruption,
            adaptive_loss_balancing=args.adaptive_loss_balancing,
            adaptive_loss_state=adaptive_loss_state,
            external_pair_cache=external_pair_cache,
            external_pair_weight=float(args.external_pair_weight),
            external_pair_source_balance=str(args.external_pair_source_balance),
            external_pair_permutation_mode=str(args.external_pair_permutation_mode),
            logging_steps=args.logging_steps,
            global_step_offset=global_step,
            max_steps=(args.max_steps if args.max_steps > 0 else None),
            calibration_enabled=bool(stage_calibration_enabled),
            contrastive_enabled=bool(stage_contrastive_enabled),
            enable_internal_contrastive=(not bool(args.external_pair_only)),
        )
        global_step = int(epoch_stats["global_step"])
        epoch_idx += 1
        if int(epoch_stats["num_batches"]) <= 0:
            raise RuntimeError("C2 training made zero progress in one epoch pass")
        if int(epoch_stats.get("optimizer_steps", 0)) <= 0:
            raise RuntimeError(
                "C2 training produced zero optimizer steps in one epoch pass. "
                "Likely all contrastive pairs were filtered out; relax pair gates "
                "or switch training mode."
            )

        eval_metrics, eval_prefix_scores, eval_corruption_scores = _evaluate_value_head(
            value_head=value_head,
            eval_cache=eval_cache,
            eval_examples=eval_examples,
            eval_corruption_cache=eval_corruption_cache,
            eval_corruptions=eval_corruptions,
            train_target_mean=train_target_mean,
            target_source=str(args.target_source),
            posthoc_calibration=args.posthoc_calibration,
            posthoc_temperature_config=posthoc_temperature_config,
            posthoc_isotonic_config=posthoc_isotonic_config,
            batch_size=int(args.per_device_eval_batch_size),
            torch_module=torch,
        )
        curve_row = {
            "epoch": epoch_idx,
            "global_step": global_step,
            "train_stage": str(stage_name),
            "train": epoch_stats,
            "eval": eval_metrics,
        }
        train_curve.append(curve_row)
        _append_jsonl(train_curve_path, curve_row)

        current_selection_value, current_higher_is_better = _resolve_checkpoint_selection_value(
            eval_metrics=eval_metrics,
            checkpoint_selection_metric=args.checkpoint_selection_metric,
        )
        if current_higher_is_better != selection_higher_is_better:
            raise RuntimeError(
                "Checkpoint-selection metric direction changed within one run, "
                "which should be impossible."
            )
        is_better = (
            current_selection_value > best_eval_selection_value
            if selection_higher_is_better
            else current_selection_value < best_eval_selection_value
        )
        if is_better:
            best_eval_selection_value = float(current_selection_value)
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
                        "eval_selection_metric": str(args.checkpoint_selection_metric),
                        "eval_selection_value": float(best_eval_selection_value),
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
        target_source=str(args.target_source),
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
    selected_prefix_scores, selected_corruption_scores = (
        best_eval_scored if best_eval_scored is not None else (final_prefix_scores, final_corruption_scores)
    )
    _write_jsonl(prefix_scores_path, selected_prefix_scores)
    _write_jsonl(corruption_scores_path, selected_corruption_scores)

    train_metrics = {
        "model_load_seconds": float(model_load_elapsed),
        "feature_cache_seconds": float(feature_cache_elapsed),
        "feature_cache": feature_cache_stats,
        "train_elapsed_seconds": float(train_elapsed),
        "global_step": int(global_step),
        "train_curve": train_curve,
        "train_target_mean": float(train_target_mean),
        "target_source": str(args.target_source),
        "target_source_missing_policy": str(args.target_source_missing_policy),
        "external_pair_stats": external_pair_stats,
        "external_pair_weight": float(args.external_pair_weight),
        "external_pair_source_balance": str(args.external_pair_source_balance),
        "external_pair_permutation_mode": str(args.external_pair_permutation_mode),
        "external_pair_only": bool(args.external_pair_only),
        "anti_saturation_weight": float(args.anti_saturation_weight),
        "anti_saturation_logit_threshold": float(args.anti_saturation_logit_threshold),
        "strict_determinism": bool(args.strict_determinism),
        "feature_cache": feature_cache_stats,
        "target_source_stats": {
            "train": train_target_source_stats,
            "eval": eval_target_source_stats,
        },
        # Backward-compatible field names are kept for downstream readers.
        "best_eval_selection_value": float(best_eval_selection_value),
        "best_eval_selection_brier": (
            float(best_eval_selection_value)
            if str(args.checkpoint_selection_metric) in {"raw_brier", "posthoc_brier"}
            else None
        ),
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
        "init_value_head": init_value_head_info,
        "num_train_examples": len(train_examples),
        "num_eval_examples": len(eval_examples),
        "num_eval_corruptions": len(eval_corruptions),
        "num_external_pairs": len(external_pairs),
        "target_source": str(args.target_source),
        "target_source_missing_policy": str(args.target_source_missing_policy),
        "external_pair_stats": external_pair_stats,
        "external_pair_weight": float(args.external_pair_weight),
        "external_pair_source_balance": str(args.external_pair_source_balance),
        "external_pair_permutation_mode": str(args.external_pair_permutation_mode),
        "external_pair_only": bool(args.external_pair_only),
        "teacher_coverage_ratio_train": float(train_target_source_stats["teacher_available_ratio"]),
        "teacher_coverage_ratio_eval": float(eval_target_source_stats["teacher_available_ratio"]),
        "teacher_disagreement_ratio_train": float(train_target_source_stats["teacher_disagree_ratio"]),
        "teacher_disagreement_ratio_eval": float(eval_target_source_stats["teacher_disagree_ratio"]),
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
            "feature_cache": feature_cache_stats,
            "model_load_seconds": float(model_load_elapsed),
            "train_target_mean": float(train_target_mean),
            "target_source": str(args.target_source),
            "target_source_missing_policy": str(args.target_source_missing_policy),
            "target_source_stats": {
                "train": train_target_source_stats,
                "eval": eval_target_source_stats,
            },
            "checkpoint_selection_metric": str(args.checkpoint_selection_metric),
            "posthoc_calibration": str(args.posthoc_calibration),
            "anti_saturation_weight": float(args.anti_saturation_weight),
            "anti_saturation_logit_threshold": float(args.anti_saturation_logit_threshold),
            "external_pair_permutation_mode": str(args.external_pair_permutation_mode),
            "strict_determinism": bool(args.strict_determinism),
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
                "target_source": str(args.target_source),
                "target_source_missing_policy": str(args.target_source_missing_policy),
                "global_step": global_step,
                "train_examples": len(train_examples),
                "eval_examples": len(eval_examples),
                "eval_corruptions": len(eval_corruptions),
                "external_pairs": len(external_pairs),
                "external_pair_weight": float(args.external_pair_weight),
                "external_pair_source_balance": str(args.external_pair_source_balance),
                "external_pair_permutation_mode": str(args.external_pair_permutation_mode),
                "external_pair_only": bool(args.external_pair_only),
                "anti_saturation_weight": float(args.anti_saturation_weight),
                "anti_saturation_logit_threshold": float(args.anti_saturation_logit_threshold),
                "strict_determinism": bool(args.strict_determinism),
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
    print(f"selected_raw_brier: {selected_eval_metrics['calibration']['brier_score']:.6f}")
    if selected_eval_metrics.get("calibration_posthoc") is not None:
        print(f"selected_post_bri : {selected_eval_metrics['calibration_posthoc']['brier_score']:.6f}")
    print(f"selected_criterion: {args.checkpoint_selection_metric}")
    selected_metric_value, _ = _resolve_checkpoint_selection_value(
        eval_metrics=selected_eval_metrics,
        checkpoint_selection_metric=args.checkpoint_selection_metric,
    )
    print(f"selected_metric_v : {float(selected_metric_value):.6f}")
    print(f"selected_pearson  : {selected_eval_metrics['calibration']['pearson']:.6f}")
    print(
        "teacher_cov_train : "
        f"{float(train_target_source_stats['teacher_available_ratio']):.4f}"
    )
    print(
        "teacher_dis_train : "
        f"{float(train_target_source_stats['teacher_disagree_ratio']):.4f}"
    )
    print(
        "teacher_cov_eval  : "
        f"{float(eval_target_source_stats['teacher_available_ratio']):.4f}"
    )
    print(
        "teacher_dis_eval  : "
        f"{float(eval_target_source_stats['teacher_disagree_ratio']):.4f}"
    )
    if selected_eval_metrics.get("calibration_posthoc") is not None:
        print(f"selected_post_prs : {selected_eval_metrics['calibration_posthoc']['pearson']:.6f}")
    if adaptive_loss_state is not None:
        print(f"adaptive_logvar_c : {float(adaptive_loss_state['log_var_calibration'].detach().cpu().item()):.6f}")
        print(f"adaptive_logvar_t : {float(adaptive_loss_state['log_var_contrastive'].detach().cpu().item()):.6f}")
    corr_pair_acc, corr_auc = _extract_corruption_console_metrics(selected_eval_metrics)
    if corr_pair_acc is not None:
        print(f"corr_pair_acc     : {float(corr_pair_acc):.6f}")
    if corr_auc is not None:
        print(f"corr_auc          : {float(corr_auc):.6f}")
    else:
        print("corr_auc          : <none>")
    print(f"external_pairs    : {len(external_pairs)}")
    print(f"external_pair_w   : {float(args.external_pair_weight):.4f}")
    print(f"external_pair_only: {bool(args.external_pair_only)}")
    print(f"anti_sat_weight   : {float(args.anti_saturation_weight):.6f}")
    print(f"anti_sat_thr      : {float(args.anti_saturation_logit_threshold):.4f}")
    if train_curve:
        last_train_stats = dict(train_curve[-1].get("train", {}))
        if "avg_anti_saturation_loss" in last_train_stats:
            print(f"last_avg_sat_loss : {float(last_train_stats['avg_anti_saturation_loss']):.6f}")
    print(f"train_metrics     : {train_metrics_path}")
    print(f"eval_metrics      : {eval_metrics_path}")
    print(f"manifest          : {manifest_path}")
    print(f"summary           : {summary_path}")
    print("=" * 88)
    return 0


def _build_stratified_train_permutation(
    *,
    train_cache: dict[str, Any],
    torch_module: Any,
    step_bucket_size: int,
    include_no_corruption: bool,
) -> Any:
    """Build a type-aware training permutation for CQR-4 stratified sampling.

    Strategy
    --------
    1. Build strata keyed by `(corruption_type, step_bucket)`.
    2. Shuffle each stratum locally.
    3. Round-robin interleave non-empty strata.

    This keeps mini-batches from collapsing to one dominant corruption type while
    preserving stochastic order each epoch.
    """
    device = train_cache["clean_features"].device
    num_examples = int(train_cache["clean_features"].shape[0])
    if num_examples <= 1:
        return torch_module.arange(num_examples, device=device)

    has_primary = train_cache["has_primary_corruption"].detach().cpu().tolist()
    step_index = train_cache["prefix_step_index"].detach().cpu().tolist()
    corruption_types = list(train_cache["primary_corruption_type"])

    # Each stratum is keyed by corruption type and coarse step bucket to keep batches mixed.
    # key = (corruption_type, step_bucket)，保证 batch 不会长期被单一类型占满。
    strata: dict[tuple[str, int], list[int]] = {}
    fallback_indices: list[int] = []
    for idx in range(num_examples):
        has_corr = bool(has_primary[idx])
        if has_corr:
            ctype = str(corruption_types[idx] or "__unknown__")
        else:
            if include_no_corruption:
                ctype = "__no_corruption__"
            else:
                fallback_indices.append(idx)
                continue
        bucket = int(step_index[idx]) // max(step_bucket_size, 1)
        strata.setdefault((ctype, bucket), []).append(idx)

    for key in list(strata.keys()):
        bucket = strata[key]
        if len(bucket) <= 1:
            continue
        order = torch_module.randperm(len(bucket)).tolist()
        strata[key] = [bucket[pos] for pos in order]

    pointers: dict[tuple[str, int], int] = {key: 0 for key in strata}
    ordered: list[int] = []
    # Interleave strata round-robin so one corruption family does not dominate long stretches of training.
    # 轮转交织：每轮从各活跃分层取 1 个样本，减少 type/step 偏置。
    while True:
        active = [
            key
            for key, values in strata.items()
            if pointers[key] < len(values)
        ]
        if not active:
            break
        if len(active) > 1:
            shuffle_order = torch_module.randperm(len(active)).tolist()
            active = [active[pos] for pos in shuffle_order]
        for key in active:
            pos = pointers[key]
            ordered.append(strata[key][pos])
            pointers[key] = pos + 1

    if fallback_indices:
        if len(fallback_indices) > 1:
            order = torch_module.randperm(len(fallback_indices)).tolist()
            fallback_indices = [fallback_indices[pos] for pos in order]
        ordered.extend(fallback_indices)

    if len(ordered) != num_examples:
        # Safety fallback: never allow silent sample dropping in training.
        return torch_module.randperm(num_examples, device=device)
    return torch_module.tensor(ordered, dtype=torch_module.long, device=device)


def _summarize_strata_for_logging(
    *,
    train_cache: dict[str, Any],
    step_bucket_size: int,
    include_no_corruption: bool,
) -> dict[str, Any]:
    """Summarize stratification buckets for quick operator visibility."""
    has_primary = train_cache["has_primary_corruption"].detach().cpu().tolist()
    step_index = train_cache["prefix_step_index"].detach().cpu().tolist()
    corruption_types = list(train_cache["primary_corruption_type"])
    counts: dict[tuple[str, int], int] = {}
    for idx, has_corr in enumerate(has_primary):
        if bool(has_corr):
            ctype = str(corruption_types[idx] or "__unknown__")
        else:
            if not include_no_corruption:
                continue
            ctype = "__no_corruption__"
        bucket = int(step_index[idx]) // max(step_bucket_size, 1)
        key = (ctype, bucket)
        counts[key] = counts.get(key, 0) + 1
    sizes = sorted(counts.values())
    if not sizes:
        return {
            "num_strata": 0,
            "min_size": 0,
            "max_size": 0,
            "median_size": 0,
            "top_strata": [],
        }
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    return {
        "num_strata": int(len(sizes)),
        "min_size": int(sizes[0]),
        "max_size": int(sizes[-1]),
        "median_size": int(sizes[len(sizes) // 2]),
        "top_strata": top,
    }


def _build_external_pair_permutation(
    *,
    external_pair_cache: dict[str, Any],
    torch_module: Any,
    source_balance: str,
    permutation_mode: str,
) -> Any:
    """Build external-pair permutation with optional source balancing."""
    num_pairs = int(external_pair_cache["num_pairs"])
    device = external_pair_cache["chosen_features"].device
    if num_pairs <= 1:
        return torch_module.arange(num_pairs, device=device)
    if permutation_mode not in {"random", "stable_hash"}:
        raise ValueError(
            f"Unsupported external pair permutation mode: {permutation_mode!r}"
        )
    pair_ids = list(external_pair_cache.get("pair_ids") or [])
    if len(pair_ids) != num_pairs:
        # Backward compatibility for older caches without pair_ids.
        pair_ids = [f"pair_{idx}" for idx in range(num_pairs)]

    def _stable_order(values: list[int]) -> list[int]:
        # stable_hash removes shuffle noise so cross-seed differences are less entangled with pair order.
        # stable_hash 模式不依赖当前 seed 的随机打乱，而是按 pair_id 哈希排序。
        # 这样“训练结果差异”更多归因于优化过程本身，而不是每个 seed 恰好看到了不同 pair 顺序。
        return sorted(
            values,
            key=lambda idx: hashlib.sha256(str(pair_ids[idx]).encode("utf-8")).hexdigest(),
        )

    if source_balance == "none":
        if permutation_mode == "random":
            return torch_module.randperm(num_pairs, device=device)
        return torch_module.tensor(
            _stable_order(list(range(num_pairs))),
            dtype=torch_module.long,
            device=device,
        )
    if source_balance != "uniform":
        raise ValueError(f"Unsupported external pair source balance mode: {source_balance!r}")

    source_tags = list(external_pair_cache["source_tags"])
    buckets: dict[str, list[int]] = {}
    for idx, tag in enumerate(source_tags):
        buckets.setdefault(str(tag), []).append(int(idx))
    for key in list(buckets.keys()):
        values = buckets[key]
        if len(values) <= 1:
            continue
        if permutation_mode == "random":
            order = torch_module.randperm(len(values)).tolist()
            buckets[key] = [values[pos] for pos in order]
        else:
            buckets[key] = _stable_order(values)

    pointers: dict[str, int] = {key: 0 for key in buckets}
    ordered: list[int] = []
    key_order = sorted(buckets.keys())
    while True:
        active = [key for key in key_order if pointers[key] < len(buckets[key])]
        if not active:
            break
        # Uniform balance means rotating over sources, not forcing exact per-source sample equality.
        # uniform balance 的目标不是“完全平均每个 source 的样本数”，而是避免训练顺序长期被单一 source 垄断。
        # That is why we round-robin buckets instead of concatenating them.
        # 因此这里采用分桶后轮转抽取，而不是简单 concat。
        if permutation_mode == "random" and len(active) > 1:
            shuffle_order = torch_module.randperm(len(active)).tolist()
            active = [active[pos] for pos in shuffle_order]
        for key in active:
            pos = pointers[key]
            ordered.append(buckets[key][pos])
            pointers[key] = pos + 1
    if len(ordered) != num_pairs:
        if permutation_mode == "random":
            return torch_module.randperm(num_pairs, device=device)
        return torch_module.tensor(
            _stable_order(list(range(num_pairs))),
            dtype=torch_module.long,
            device=device,
        )
    return torch_module.tensor(ordered, dtype=torch_module.long, device=device)


def _next_external_pair_batch(
    *,
    external_pair_cache: dict[str, Any],
    torch_module: Any,
    permutation: Any,
    cursor: int,
    batch_size: int,
    source_balance: str,
    permutation_mode: str,
) -> tuple[Any, int, Any]:
    """Return one external-pair mini-batch with wrap-around sampling."""
    num_pairs = int(external_pair_cache["num_pairs"])
    device = external_pair_cache["chosen_features"].device
    if batch_size <= 0 or num_pairs <= 0:
        empty = torch_module.zeros((0,), dtype=torch_module.long, device=device)
        return empty, 0, permutation
    if permutation is None or int(permutation.shape[0]) != num_pairs:
        # Rebuild the traversal order whenever the cache is first seen or its size changes.
        # 当缓存第一次使用，或外部 pair 数发生变化时，重新建 permutation。
        permutation = _build_external_pair_permutation(
            external_pair_cache=external_pair_cache,
            torch_module=torch_module,
            source_balance=source_balance,
            permutation_mode=permutation_mode,
        )
        cursor = 0
    cursor = int(cursor) % num_pairs
    needed = int(batch_size)
    chunks: list[Any] = []
    while needed > 0:
        if cursor >= num_pairs:
            # Wrap around and rebuild order so each optimizer step still sees a full external-pair batch.
            # 走到尾部后重新洗牌/重排，但 batch 继续补齐。
            # 这样一个 optimizer step 永远拿到固定 batch_size 的 pair，不会因为尾 batch 太小而让梯度抖动。
            permutation = _build_external_pair_permutation(
                external_pair_cache=external_pair_cache,
                torch_module=torch_module,
                source_balance=source_balance,
                permutation_mode=permutation_mode,
            )
            cursor = 0
        take = min(needed, num_pairs - cursor)
        chunks.append(permutation[cursor : cursor + take])
        cursor += take
        needed -= take
    if cursor >= num_pairs:
        permutation = _build_external_pair_permutation(
            external_pair_cache=external_pair_cache,
            torch_module=torch_module,
            source_balance=source_balance,
            permutation_mode=permutation_mode,
        )
        cursor = 0
    if len(chunks) == 1:
        return chunks[0], cursor, permutation
    return torch_module.cat(chunks, dim=0), cursor, permutation


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


def _score_feature_tensor_with_logits_in_batches(
    *,
    value_head: Any,
    features: Any,
    batch_size: int,
    torch_module: Any,
) -> tuple[list[float], Any]:
    """Score a feature matrix in mini-batches and return CPU logits for calibration."""
    head_device = next(value_head.parameters()).device
    head_dtype = next(value_head.parameters()).dtype
    scores: list[float] = []
    logits_chunks: list[Any] = []
    effective_bs = max(1, int(batch_size))
    for start in range(0, int(features.shape[0]), effective_bs):
        batch = _move_batch_tensor(
            features[start : start + effective_bs],
            device=head_device,
            dtype=head_dtype,
        )
        outputs = value_head(batch)
        scores.extend(float(v) for v in outputs["scores"].detach().cpu().tolist())
        logits_chunks.append(outputs["logits"].detach().cpu())
    if logits_chunks:
        logits_tensor = torch_module.cat(logits_chunks, dim=0)
    else:
        logits_tensor = torch_module.zeros((0,), dtype=torch_module.float32)
    return scores, logits_tensor


def _score_feature_tensor_in_batches(
    *,
    value_head: Any,
    features: Any,
    batch_size: int,
    torch_module: Any,
) -> list[float]:
    """Score a feature matrix in mini-batches."""
    scores, _ = _score_feature_tensor_with_logits_in_batches(
        value_head=value_head,
        features=features,
        batch_size=batch_size,
        torch_module=torch_module,
    )
    return scores


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
    anti_saturation_weight: float,
    anti_saturation_logit_threshold: float,
    lambda_contrastive: float,
    contrastive_margin: float,
    contrastive_pair_filter: str,
    contrastive_confidence_threshold: float,
    contrastive_parseable_threshold: float,
    contrastive_label_delta_q_min: float,
    contrastive_label_z_min: float,
    contrastive_label_pair_weight_min: float,
    contrastive_require_pair_pass_gate: bool,
    contrastive_score_gap_min: float,
    contrastive_score_gap_max: float,
    contrastive_use_pair_weights: bool,
    contrastive_stratified_sampling: bool,
    contrastive_stratify_step_bucket_size: int,
    contrastive_stratify_include_no_corruption: bool,
    adaptive_loss_balancing: str,
    adaptive_loss_state: dict[str, Any] | None,
    external_pair_cache: dict[str, Any] | None,
    external_pair_weight: float,
    external_pair_source_balance: str,
    external_pair_permutation_mode: str,
    logging_steps: int,
    global_step_offset: int,
    max_steps: int | None,
    calibration_enabled: bool = True,
    contrastive_enabled: bool = True,
    enable_internal_contrastive: bool = True,
) -> dict[str, Any]:
    """Train the head for one epoch over cached feature tensors."""
    value_head.train()
    num_examples = int(train_cache["clean_features"].shape[0])
    head_device = next(value_head.parameters()).device
    head_dtype = next(value_head.parameters()).dtype
    if bool(contrastive_stratified_sampling):
        # Stratified sampling prevents long runs of batches from collapsing to one corruption family.
        # 分层采样避免 batch 被单一 corruption 类型主导。
        permutation = _build_stratified_train_permutation(
            train_cache=train_cache,
            torch_module=torch_module,
            step_bucket_size=int(contrastive_stratify_step_bucket_size),
            include_no_corruption=bool(contrastive_stratify_include_no_corruption),
        )
    else:
        permutation = torch_module.randperm(num_examples, device=train_cache["clean_features"].device)
    running_total = 0.0
    running_cal = 0.0
    running_ctr = 0.0
    running_ctr_external = 0.0
    running_anti_sat = 0.0
    running_effective_ctr_weight = 0.0
    batches = 0
    optimizer_steps = 0
    external_pairs_seen = 0

    external_enabled = (
        external_pair_cache is not None
        and float(external_pair_weight) > 0.0
        and int(external_pair_cache["num_pairs"]) > 0
    )
    external_cursor = 0
    external_perm = None
    if external_enabled:
        external_perm = _build_external_pair_permutation(
            external_pair_cache=external_pair_cache,
            torch_module=torch_module,
            source_balance=str(external_pair_source_balance),
            permutation_mode=str(external_pair_permutation_mode),
        )
    anti_saturation_enabled = (
        float(anti_saturation_weight) > 0.0
        and float(anti_saturation_logit_threshold) > 0.0
    )

    for batch_start in range(0, num_examples, batch_size):
        if max_steps is not None and global_step_offset + optimizer_steps >= max_steps:
            break
        batch_indices = permutation[batch_start : batch_start + batch_size]
        clean_features = _move_batch_tensor(
            train_cache["clean_features"][batch_indices],
            device=head_device,
            dtype=head_dtype,
        )
        raw_targets = _move_batch_tensor(
            train_cache["targets"][batch_indices],
            device=head_device,
            dtype=torch_module.float32,
        )
        targets = _apply_target_smoothing(
            raw_targets,
            epsilon=float(calibration_target_smoothing),
        )
        sample_weights = _move_batch_tensor(
            train_cache["calibration_weights"][batch_indices],
            device=head_device,
            dtype=torch_module.float32,
        )
        head_outputs = value_head(clean_features)
        loss_sat = torch_module.zeros((), device=clean_features.device)
        if anti_saturation_enabled:
            loss_sat = loss_sat + anti_saturation_logit_penalty(
                head_outputs["logits"],
                logit_threshold=float(anti_saturation_logit_threshold),
                torch_module=torch_module,
            )
        if calibration_enabled:
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
        else:
            loss_cal = torch_module.zeros((), device=clean_features.device)
        loss_ctr = torch_module.zeros((), device=clean_features.device)
        loss_ctr_external = torch_module.zeros((), device=clean_features.device)
        used_ctr_pairs = False
        if enable_internal_contrastive and contrastive_enabled and lambda_contrastive > 0.0:
            # Apply pair-quality gates before computing margin loss so noisy pairs do not dominate contrastive updates.
            # 对比分支先过 pair filter，再算 margin loss，尽量抑制噪声 pair。
            sample_mask = train_cache["has_primary_corruption"][batch_indices]
            sample_mask = _apply_contrastive_pair_filter(
                batch_indices=batch_indices,
                base_mask=sample_mask,
                train_cache=train_cache,
                pair_filter=contrastive_pair_filter,
                confidence_threshold=float(contrastive_confidence_threshold),
                parseable_threshold=float(contrastive_parseable_threshold),
                label_delta_q_min=float(contrastive_label_delta_q_min),
                label_z_min=float(contrastive_label_z_min),
                label_pair_weight_min=float(contrastive_label_pair_weight_min),
                require_pair_pass_gate=bool(contrastive_require_pair_pass_gate),
                torch_module=torch_module,
            )
            candidate_mask = train_cache["contrastive_candidate_mask"][batch_indices].clone()
            if candidate_mask.ndim == 1:
                candidate_mask = candidate_mask.unsqueeze(-1)
            candidate_mask = candidate_mask & sample_mask.unsqueeze(1)
            candidate_mask = _apply_contrastive_candidate_quality_filter(
                candidate_mask=candidate_mask,
                candidate_delta_q=train_cache["contrastive_candidate_delta_q"][batch_indices],
                candidate_z_delta=train_cache["contrastive_candidate_z_delta"][batch_indices],
                candidate_pair_weight=train_cache["contrastive_candidate_pair_weight"][batch_indices],
                candidate_pass_gate=train_cache["contrastive_candidate_pass_gate"][batch_indices],
                label_delta_q_min=float(contrastive_label_delta_q_min),
                label_z_min=float(contrastive_label_z_min),
                label_pair_weight_min=float(contrastive_label_pair_weight_min),
                require_pair_pass_gate=bool(contrastive_require_pair_pass_gate),
            )
            if bool(candidate_mask.any().item()):
                candidate_mask_cpu = candidate_mask
                candidate_mask_device = _move_batch_tensor(
                    candidate_mask_cpu,
                    device=head_device,
                    dtype=torch_module.bool,
                )
                corrupt_features = _move_batch_tensor(
                    train_cache["contrastive_corruption_features"][batch_indices][candidate_mask_cpu],
                    device=head_device,
                    dtype=head_dtype,
                )
                expanded_clean_scores = head_outputs["scores"].unsqueeze(1).expand_as(
                    train_cache["contrastive_candidate_pair_weight"][batch_indices]
                )
                clean_scores_for_pairs = expanded_clean_scores[candidate_mask_device]
                corrupt_outputs = value_head(corrupt_features)
                corrupt_scores = corrupt_outputs["scores"]
                if anti_saturation_enabled:
                    loss_sat = loss_sat + anti_saturation_logit_penalty(
                        corrupt_outputs["logits"],
                        logit_threshold=float(anti_saturation_logit_threshold),
                        torch_module=torch_module,
                    )
                pair_weights = None
                if bool(contrastive_use_pair_weights):
                    pair_weights = _move_batch_tensor(
                        train_cache["contrastive_candidate_pair_weight"][batch_indices][candidate_mask_cpu],
                        device=head_device,
                        dtype=torch_module.float32,
                    )
                clean_scores_for_pairs, corrupt_scores, pair_weights = _apply_contrastive_score_gap_filter(
                    clean_scores_for_pairs=clean_scores_for_pairs,
                    corrupt_scores=corrupt_scores,
                    score_gap_min=float(contrastive_score_gap_min),
                    score_gap_max=float(contrastive_score_gap_max),
                    pair_weights=pair_weights,
                    torch_module=torch_module,
                )
                if clean_scores_for_pairs.numel() > 0:
                    loss_ctr = contrastive_margin_loss(
                        clean_scores_for_pairs,
                        corrupt_scores,
                        margin=contrastive_margin,
                        torch_module=torch_module,
                        sample_weights=pair_weights,
                    )
                    used_ctr_pairs = True

        if contrastive_enabled and external_enabled:
            ext_count = int(batch_indices.shape[0])
            ext_indices, external_cursor, external_perm = _next_external_pair_batch(
                external_pair_cache=external_pair_cache,
                torch_module=torch_module,
                permutation=external_perm,
                cursor=external_cursor,
                batch_size=ext_count,
                source_balance=str(external_pair_source_balance),
                permutation_mode=str(external_pair_permutation_mode),
            )
            if ext_indices.numel() > 0:
                chosen_features = _move_batch_tensor(
                    external_pair_cache["chosen_features"][ext_indices],
                    device=head_device,
                    dtype=head_dtype,
                )
                rejected_features = _move_batch_tensor(
                    external_pair_cache["rejected_features"][ext_indices],
                    device=head_device,
                    dtype=head_dtype,
                )
                chosen_outputs = value_head(chosen_features)
                rejected_outputs = value_head(rejected_features)
                chosen_scores = chosen_outputs["scores"]
                rejected_scores = rejected_outputs["scores"]
                if anti_saturation_enabled:
                    loss_sat = loss_sat + 0.5 * (
                        anti_saturation_logit_penalty(
                            chosen_outputs["logits"],
                            logit_threshold=float(anti_saturation_logit_threshold),
                            torch_module=torch_module,
                        )
                        + anti_saturation_logit_penalty(
                            rejected_outputs["logits"],
                            logit_threshold=float(anti_saturation_logit_threshold),
                            torch_module=torch_module,
                        )
                    )
                ext_weights = _move_batch_tensor(
                    external_pair_cache["pair_weights"][ext_indices],
                    device=head_device,
                    dtype=torch_module.float32,
                )
                loss_ctr_external = contrastive_margin_loss(
                    chosen_scores,
                    rejected_scores,
                    margin=contrastive_margin,
                    torch_module=torch_module,
                    sample_weights=ext_weights,
                )
                external_pairs_seen += int(ext_indices.shape[0])
                loss_ctr = loss_ctr + (float(external_pair_weight) * loss_ctr_external)
                used_ctr_pairs = True

        base_loss, effective_ctr_weight = _compose_total_loss(
            loss_cal=loss_cal,
            loss_ctr=loss_ctr,
            lambda_contrastive=float(lambda_contrastive),
            adaptive_loss_balancing=adaptive_loss_balancing,
            adaptive_loss_state=adaptive_loss_state,
            used_ctr_pairs=used_ctr_pairs,
            torch_module=torch_module,
        )
        loss = base_loss + (float(anti_saturation_weight) * loss_sat)
        if not bool(getattr(loss, "requires_grad", False)):
            # Ranking-only stage may hit batches with no surviving pairs after
            # filtering. Skip these batches to avoid backward on constants.
            running_total += float(loss.item())
            running_cal += float(loss_cal.item())
            running_ctr += float(loss_ctr.item())
            running_anti_sat += float(loss_sat.item())
            running_effective_ctr_weight += float(effective_ctr_weight)
            batches += 1
            continue
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
                    f"ctr_ext={float(loss_ctr_external.item()):.6f} "
                    f"sat={float(loss_sat.item()):.6f} "
                    f"ctr_w={float(effective_ctr_weight):.4f}",
                    flush=True,
                )

        running_total += float(loss.item())
        running_cal += float(loss_cal.item())
        running_ctr += float(loss_ctr.item())
        running_ctr_external += float(loss_ctr_external.item())
        running_anti_sat += float(loss_sat.item())
        running_effective_ctr_weight += float(effective_ctr_weight)
        batches += 1

    return {
        "avg_total_loss": float(running_total / max(batches, 1)),
        "avg_calibration_loss": float(running_cal / max(batches, 1)),
        "avg_contrastive_loss": float(running_ctr / max(batches, 1)),
        "avg_external_contrastive_loss": float(running_ctr_external / max(batches, 1)),
        "avg_anti_saturation_loss": float(running_anti_sat / max(batches, 1)),
        "avg_effective_contrastive_weight": float(running_effective_ctr_weight / max(batches, 1)),
        "num_batches": int(batches),
        "optimizer_steps": int(optimizer_steps),
        "external_pairs_seen": int(external_pairs_seen),
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
    target_source: str,
    posthoc_calibration: str,
    posthoc_temperature_config: TemperatureCalibrationConfig,
    posthoc_isotonic_config: IsotonicCalibrationConfig,
    batch_size: int,
    torch_module: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Score the held-out eval set and compute C2 metrics."""
    value_head.eval()
    with torch_module.no_grad():
        prefix_scores, prefix_logits_tensor = _score_feature_tensor_with_logits_in_batches(
            value_head=value_head,
            features=eval_cache["clean_features"],
            batch_size=int(batch_size),
            torch_module=torch_module,
        )
        prefix_scores_tensor = torch_module.tensor(
            prefix_scores,
            dtype=prefix_logits_tensor.dtype if prefix_logits_tensor.numel() > 0 else torch_module.float32,
        )
        target_scores = [float(v) for v in eval_cache["targets"].detach().cpu().tolist()]

    # Keep both raw and post-hoc calibration metrics so later analysis can separate training quality from calibration fixes.
    # 先算 raw 校准指标，再可选做 post-hoc，二者都保留方便复盘。
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
        selected_target = float(eval_cache["targets"][idx].detach().cpu().item())
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
            "target_source": str(target_source),
            "target_selected_value": selected_target,
            "target_success_rate": float(example.target_success_rate),
            "target_q_mean_smoothed": float(example.target_q_mean_smoothed),
            "target_q_teacher": (
                float(example.target_q_teacher) if example.target_q_teacher is not None else None
            ),
            "target_q_fused": (
                float(example.target_q_fused) if example.target_q_fused is not None else None
            ),
            "target_teacher_available": bool(example.target_teacher_available),
            "target_teacher_disagree": bool(example.target_teacher_disagree),
            "target_q_weight": float(example.target_q_weight),
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
            corrupt_scores = _score_feature_tensor_in_batches(
                value_head=value_head,
                features=eval_corruption_cache["corruption_features"],
                batch_size=int(batch_size),
                torch_module=torch_module,
            )
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


def _resolve_supervision_targets(
    *,
    examples: list[ValueSupervisionExample],
    target_source: str,
    missing_policy: str,
    split_label: str,
) -> tuple[list[float], dict[str, Any]]:
    """Resolve per-example scalar targets according to D3 target-source policy.

    Notes
    -----
    - `q_mean_smoothed`: pure MC label (Phase C baseline)
    - `q_teacher`: pure teacher label (D1 sidecar)
    - `q_fused`: D2 fused label
    - For missing values, `fail` aborts immediately and `fallback_mc` falls back to MC while recording the rate.

    说明
    ----
    - `q_mean_smoothed`: 纯 MC 标签（Phase C 基线）
    - `q_teacher`: 纯 teacher 标签（D1 sidecar）
    - `q_fused`: D2 融合标签
    - 对于缺失值，`fail` 会直接中止；`fallback_mc` 会回退到 MC，且统计回退比例。
    """
    values: list[float] = []
    missing_prefix_ids: list[str] = []
    fallback_count = 0
    available_count = 0
    teacher_available_count = 0
    teacher_disagree_count = 0
    for example in examples:
        source_value: float | None
        if target_source == "q_mean_smoothed":
            source_value = float(example.target_q_mean_smoothed)
            available_count += 1
        elif target_source == "q_teacher":
            source_value = (
                float(example.target_q_teacher)
                if example.target_q_teacher is not None
                else None
            )
            if source_value is not None:
                available_count += 1
        elif target_source == "q_fused":
            source_value = (
                float(example.target_q_fused)
                if example.target_q_fused is not None
                else None
            )
            if source_value is not None:
                available_count += 1
        else:
            raise ValueError(f"Unsupported target_source: {target_source!r}")

        if bool(example.target_teacher_available):
            teacher_available_count += 1
        if bool(example.target_teacher_disagree):
            teacher_disagree_count += 1

        if source_value is None:
            missing_prefix_ids.append(str(example.prefix_id))
            if missing_policy == "fail":
                continue
            if missing_policy == "fallback_mc":
                values.append(float(example.target_q_mean_smoothed))
                fallback_count += 1
                continue
            raise ValueError(f"Unsupported missing policy: {missing_policy!r}")
        values.append(float(source_value))

    if missing_prefix_ids and missing_policy == "fail":
        preview = ", ".join(missing_prefix_ids[:10])
        raise ValueError(
            f"Missing target source `{target_source}` in split={split_label} for "
            f"{len(missing_prefix_ids)} examples. Example prefix_ids: {preview}"
        )
    total = len(examples)
    coverage_ratio = float(available_count / total) if total else 0.0
    stats = {
        "split": str(split_label),
        "target_source": str(target_source),
        "missing_policy": str(missing_policy),
        "num_examples": int(total),
        "available_count": int(available_count),
        "missing_count": int(total - available_count),
        "coverage_ratio": float(coverage_ratio),
        "fallback_count": int(fallback_count),
        "fallback_ratio": (float(fallback_count / total) if total else 0.0),
        "teacher_available_count": int(teacher_available_count),
        "teacher_available_ratio": (
            float(teacher_available_count / total) if total else 0.0
        ),
        "teacher_disagree_count": int(teacher_disagree_count),
        "teacher_disagree_ratio": (
            float(teacher_disagree_count / teacher_available_count)
            if teacher_available_count > 0
            else 0.0
        ),
    }
    return values, stats


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


def _checkpoint_metric_higher_is_better(checkpoint_selection_metric: str) -> bool:
    """Return whether the selection metric should be maximized."""
    return checkpoint_selection_metric in {"corr_pair_acc", "corr_auc", "ranking_score"}


def _extract_corruption_console_metrics(
    eval_metrics: dict[str, Any],
) -> tuple[float | None, float | None]:
    """Return optional `(pair_accuracy, auc)` metrics for console reporting."""
    corruption = eval_metrics.get("corruption")
    if not isinstance(corruption, dict):
        return None, None
    pair_acc = corruption.get("pair_accuracy")
    auc = corruption.get("auc_clean_vs_corrupt")
    return (
        (float(pair_acc) if pair_acc is not None else None),
        (float(auc) if auc is not None else None),
    )


def _resolve_checkpoint_selection_value(
    *,
    eval_metrics: dict[str, Any],
    checkpoint_selection_metric: str,
) -> tuple[float, bool]:
    """Resolve checkpoint selection value and optimization direction.

    Returns
    -------
    tuple[float, bool]
        `(selection_value, higher_is_better)`.
    """
    if checkpoint_selection_metric == "raw_brier":
        return float(eval_metrics["calibration"]["brier_score"]), False
    if checkpoint_selection_metric == "posthoc_brier":
        posthoc = eval_metrics.get("calibration_posthoc")
        if posthoc is None:
            raise ValueError(
                "Checkpoint selection requested posthoc_brier, but no post-hoc calibration metrics are available"
            )
        return float(posthoc["brier_score"]), False
    corruption = eval_metrics.get("corruption") or {}
    if checkpoint_selection_metric == "corr_pair_acc":
        pair_acc = corruption.get("pair_accuracy")
        if pair_acc is None:
            raise ValueError(
                "Checkpoint selection requested corr_pair_acc, "
                "but corruption.pair_accuracy is unavailable"
            )
        return float(pair_acc), True
    if checkpoint_selection_metric == "corr_auc":
        auc = corruption.get("auc_clean_vs_corrupt")
        if auc is None:
            raise ValueError(
                "Checkpoint selection requested corr_auc, "
                "but corruption.auc_clean_vs_corrupt is unavailable"
            )
        return float(auc), True
    if checkpoint_selection_metric == "ranking_score":
        pair_acc = corruption.get("pair_accuracy")
        auc = corruption.get("auc_clean_vs_corrupt")
        if pair_acc is None or auc is None:
            raise ValueError(
                "Checkpoint selection requested ranking_score, but one of "
                "corruption.pair_accuracy / corruption.auc_clean_vs_corrupt is unavailable"
            )
        return float((float(pair_acc) + float(auc)) / 2.0), True
    raise ValueError(f"Unsupported checkpoint_selection_metric: {checkpoint_selection_metric!r}")


def _resolve_training_stage(
    *,
    train_mode: str,
    global_step: int,
    total_steps: int,
    two_stage_ranking_ratio: float,
    use_contrastive_loss: bool,
) -> tuple[str, bool, bool]:
    """Resolve stage-specific optimization flags for one training pass."""
    if train_mode == "joint":
        return "joint", True, bool(use_contrastive_loss)
    if train_mode == "ranking_only":
        return "ranking_only", False, bool(use_contrastive_loss)
    if train_mode == "calibration_only":
        return "calibration_only", True, False
    if train_mode == "two_stage":
        switch_step = max(1, int(total_steps * float(two_stage_ranking_ratio)))
        if global_step < switch_step:
            return "two_stage_ranking", False, bool(use_contrastive_loss)
        return "two_stage_calibration", True, False
    raise ValueError(f"Unsupported train_mode: {train_mode!r}")


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
    q_weight: Any,
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
        elif mode == "q_weight":
            # q_weight comes from the C1 rollout CI-width mapping: wider uncertainty means lower weight.
            # q_weight 来自 C1 rollout 的 CI 宽度映射（越不确定，权重越低）。
            weights = q_weight.clamp(0.0, 1.0)
        elif mode == "q_weight_parseable":
            # Combine uncertainty and parseability so unstable labels are down-weighted most aggressively.
            # 进一步叠加 parseable 约束，过滤“可解析性差 + 不确定性高”的样本。
            weights = (q_weight.clamp(0.0, 1.0) * parseable_clamped).clamp(0.0, 1.0)
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
    label_delta_q_min: float,
    label_z_min: float,
    label_pair_weight_min: float,
    require_pair_pass_gate: bool,
    torch_module: Any,
) -> Any:
    """Filter contrastive pairs using rollout and label-side quality gates.

    Supported modes:
    - `none`
    - `confidence`
    - `parseable`
    - `confidence_parseable`
    - `label_quality`
    - `confidence_parseable_label`
    """
    if pair_filter == "none":
        return base_mask

    confidence = train_cache["target_confidence"][batch_indices]
    parseable = train_cache["target_parseable"][batch_indices]
    label_delta_q = train_cache["primary_pair_delta_q"][batch_indices]
    label_z = train_cache["primary_pair_z_delta"][batch_indices]
    label_pair_weight = train_cache["primary_pair_weight"][batch_indices]
    label_pair_pass = train_cache["primary_pair_pass_gate"][batch_indices]
    # label_quality uses C1-side metadata so filtering is based on precomputed pair quality, not noisy in-flight scores.
    # label_quality 分支使用 C1 预先估计的 pair 质量标签，而不是训练时瞬时分数。
    label_mask = (
        (label_delta_q >= float(label_delta_q_min))
        & (label_z >= float(label_z_min))
        & (label_pair_weight >= float(label_pair_weight_min))
    )
    if require_pair_pass_gate:
        # Add the hard gate only when explicitly requested so old artifacts remain reusable by default.
        # 额外硬约束：只允许通过 pair_pass_gate 的样本参与对比学习。
        label_mask = label_mask & label_pair_pass

    if pair_filter == "confidence":
        extra_mask = confidence >= float(confidence_threshold)
    elif pair_filter == "parseable":
        extra_mask = parseable >= float(parseable_threshold)
    elif pair_filter == "confidence_parseable":
        extra_mask = (confidence >= float(confidence_threshold)) & (
            parseable >= float(parseable_threshold)
        )
    elif pair_filter == "label_quality":
        extra_mask = label_mask
    elif pair_filter == "confidence_parseable_label":
        # 三重门控：
        # 1) clean 置信度；
        # 2) clean parseable 率；
        # 3) label-side pair 质量。
        extra_mask = (
            (confidence >= float(confidence_threshold))
            & (parseable >= float(parseable_threshold))
            & label_mask
        )
    else:
        raise ValueError(f"Unsupported contrastive pair filter: {pair_filter!r}")

    if extra_mask.dtype != torch_module.bool:
        extra_mask = extra_mask.to(dtype=torch_module.bool)
    return base_mask & extra_mask


def _apply_contrastive_candidate_quality_filter(
    *,
    candidate_mask: Any,
    candidate_delta_q: Any,
    candidate_z_delta: Any,
    candidate_pair_weight: Any,
    candidate_pass_gate: Any,
    label_delta_q_min: float,
    label_z_min: float,
    label_pair_weight_min: float,
    require_pair_pass_gate: bool,
) -> Any:
    """Filter top-k corruption candidates using candidate-level label quality."""
    keep_mask = candidate_mask
    keep_mask = keep_mask & (candidate_delta_q >= float(label_delta_q_min))
    keep_mask = keep_mask & (candidate_z_delta >= float(label_z_min))
    keep_mask = keep_mask & (candidate_pair_weight >= float(label_pair_weight_min))
    if require_pair_pass_gate:
        keep_mask = keep_mask & candidate_pass_gate
    return keep_mask


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
    pair_weights: Any | None,
    torch_module: Any,
) -> tuple[Any, Any, Any | None]:
    """Filter contrastive pairs by current score-gap band.

    Gap is defined as `clean_score - corrupt_score`.
    """
    if clean_scores_for_pairs.numel() == 0:
        return clean_scores_for_pairs, corrupt_scores, pair_weights
    if score_gap_min <= -1.0 and score_gap_max >= 1.0:
        return clean_scores_for_pairs, corrupt_scores, pair_weights

    gaps = clean_scores_for_pairs - corrupt_scores
    keep_mask = (gaps >= float(score_gap_min)) & (gaps <= float(score_gap_max))
    if keep_mask.dtype != torch_module.bool:
        keep_mask = keep_mask.to(dtype=torch_module.bool)
    if not bool(keep_mask.any().item()):
        empty = clean_scores_for_pairs[:0]
        return empty, corrupt_scores[:0], (pair_weights[:0] if pair_weights is not None else None)
    filtered_weights = pair_weights[keep_mask] if pair_weights is not None else None
    return clean_scores_for_pairs[keep_mask], corrupt_scores[keep_mask], filtered_weights


def _encode_example_cache(
    *,
    examples: list[ValueSupervisionExample],
    target_values: list[float],
    backbone: Any,
    tokenizer: Any,
    torch_module: Any,
    max_length: int,
    batch_size: int,
    use_primary_corruption: bool,
    contrastive_max_corruptions_per_prefix: int,
    calibration_sample_weighting: str,
    calibration_weight_floor: float,
    calibration_weight_gamma: float,
) -> dict[str, Any]:
    """Encode clean examples and optional primary corruptions into cached tensors."""
    if len(target_values) != len(examples):
        raise ValueError(
            "target_values length must match examples length: "
            f"{len(target_values)} vs {len(examples)}"
        )
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
        [float(value) for value in target_values],
        dtype=torch_module.float32,
        device=device,
    )
    target_q_weight = torch_module.tensor(
        [example.target_q_weight for example in examples],
        dtype=torch_module.float32,
        device=device,
    )
    target_parseable = torch_module.tensor(
        [example.target_parseable_rate for example in examples],
        dtype=torch_module.float32,
        device=device,
    )
    target_confidence = torch_module.abs(targets - 0.5) * 2.0
    # target_confidence measures label sharpness, not model confidence.
    # target_confidence 不是模型预测置信度，而是标签离 0.5 有多远。
    # Targets near 0/1 are clearer rollout labels, while targets near 0.5 are inherently ambiguous.
    # 目标越接近 0 或 1，说明 rollout 标签越明确；越接近 0.5，说明这个前缀本身更不确定。
    calibration_weights = _build_calibration_sample_weights(
        targets=targets,
        parseable=target_parseable,
        q_weight=target_q_weight,
        mode=calibration_sample_weighting,
        floor=float(calibration_weight_floor),
        gamma=float(calibration_weight_gamma),
        torch_module=torch_module,
    )
    max_candidates = (
        int(contrastive_max_corruptions_per_prefix)
        if use_primary_corruption
        else 1
    )
    candidate_lists: list[list[dict[str, Any]]] = []
    for example in examples:
        if use_primary_corruption:
            candidate_lists.append(
                _extract_contrastive_candidates(
                    example=example,
                    max_items=max_candidates,
                )
            )
        else:
            candidate_lists.append([])

    has_primary_corruption = torch_module.tensor(
        [bool(candidates) for candidates in candidate_lists],
        dtype=torch_module.bool,
        device=device,
    )
    primary_corruption_type = [
        str(candidates[0].get("corruption_type", "__none__"))
        if candidates
        else "__none__"
        for candidates in candidate_lists
    ]
    primary_pair_delta_q = torch_module.tensor(
        [
            float(candidates[0].get("pair_delta_q", 0.0) or 0.0)
            if candidates
            else 0.0
            for candidates in candidate_lists
        ],
        dtype=torch_module.float32,
        device=device,
    )
    primary_pair_z_delta = torch_module.tensor(
        [
            float(candidates[0].get("pair_z_delta", 0.0) or 0.0)
            if candidates
            else 0.0
            for candidates in candidate_lists
        ],
        dtype=torch_module.float32,
        device=device,
    )
    primary_pair_weight = torch_module.tensor(
        [
            float(candidates[0].get("pair_weight", 0.0) or 0.0)
            if candidates
            else 0.0
            for candidates in candidate_lists
        ],
        dtype=torch_module.float32,
        device=device,
    )
    primary_pair_pass_gate = torch_module.tensor(
        [
            bool(candidates[0].get("pair_pass_gate", False))
            if candidates
            else False
            for candidates in candidate_lists
        ],
        dtype=torch_module.bool,
        device=device,
    )
    prefix_step_index = torch_module.tensor(
        [int(example.prefix_step_index) for example in examples],
        dtype=torch_module.long,
        device=device,
    )

    max_candidates_in_batch = max((len(candidates) for candidates in candidate_lists), default=0)
    if max_candidates_in_batch <= 0:
        max_candidates_in_batch = 1
    # Pack the irregular "0..k candidates per sample" structure into fixed-shape tensor banks up front.
    # 这里把“每个样本可有 0..k 个 corruption 候选”的不规则结构，显式打包成固定形状 tensor bank。
    # That keeps the training loop tensorized and avoids repeated Python-side shape handling in every batch.
    # 这样后面的训练循环可以纯 tensor 化，避免在每个 batch 中反复走 Python list 和隐性 shape bug。
    hidden_size = int(clean_features.shape[1])
    corruption_features_bank = torch_module.zeros(
        (len(examples), max_candidates_in_batch, hidden_size),
        dtype=torch_module.float32,
        device=device,
    )
    corruption_candidate_mask = torch_module.zeros(
        (len(examples), max_candidates_in_batch),
        dtype=torch_module.bool,
        device=device,
    )
    corruption_candidate_pair_weight = torch_module.zeros(
        (len(examples), max_candidates_in_batch),
        dtype=torch_module.float32,
        device=device,
    )
    corruption_candidate_delta_q = torch_module.zeros(
        (len(examples), max_candidates_in_batch),
        dtype=torch_module.float32,
        device=device,
    )
    corruption_candidate_z_delta = torch_module.zeros(
        (len(examples), max_candidates_in_batch),
        dtype=torch_module.float32,
        device=device,
    )
    corruption_candidate_pass_gate = torch_module.zeros(
        (len(examples), max_candidates_in_batch),
        dtype=torch_module.bool,
        device=device,
    )
    flat_corruption_texts: list[str] = []
    flat_corruption_positions: list[tuple[int, int]] = []
    for example_idx, candidates in enumerate(candidate_lists):
        for candidate_idx, candidate in enumerate(candidates):
            if candidate_idx >= max_candidates_in_batch:
                break
            corruption_candidate_mask[example_idx, candidate_idx] = True
            corruption_candidate_pair_weight[example_idx, candidate_idx] = float(
                candidate.get("pair_weight", 0.0) or 0.0
            )
            corruption_candidate_delta_q[example_idx, candidate_idx] = float(
                candidate.get("pair_delta_q", 0.0) or 0.0
            )
            corruption_candidate_z_delta[example_idx, candidate_idx] = float(
                candidate.get("pair_z_delta", 0.0) or 0.0
            )
            corruption_candidate_pass_gate[example_idx, candidate_idx] = bool(
                candidate.get("pair_pass_gate", False)
            )
            flat_corruption_texts.append(
                f"{examples[example_idx].prompt_text}{candidate['corrupted_prefix_text']}"
            )
            flat_corruption_positions.append((example_idx, candidate_idx))
    if flat_corruption_texts:
        # Flatten first and encode once so corruption features are produced by one stable batched pass.
        # 先展平再统一编码，比“每个样本逐个编码 corruption”更稳定，也更高效。
        # `flat_corruption_positions` maps the flat outputs back into `[sample, candidate]` coordinates.
        # `flat_corruption_positions` 负责把编码结果还原回 `[sample, candidate]` 坐标。
        encoded_corrupt = _encode_text_list_in_batches(
            texts=flat_corruption_texts,
            backbone=backbone,
            tokenizer=tokenizer,
            torch_module=torch_module,
            max_length=max_length,
            batch_size=batch_size,
            progress_label="cache_train_primary_corrupt",
            progress_every_batches=32,
        )
        for encoded_idx, (example_idx, candidate_idx) in enumerate(flat_corruption_positions):
            corruption_features_bank[example_idx, candidate_idx] = encoded_corrupt[encoded_idx]
    corruption_features = corruption_features_bank[:, 0, :]

    payload = {
        "clean_features": clean_features,
        "targets": targets,
        "target_q_weight": target_q_weight,
        "target_parseable": target_parseable,
        "target_confidence": target_confidence,
        "calibration_weights": calibration_weights,
        "primary_pair_delta_q": primary_pair_delta_q,
        "primary_pair_z_delta": primary_pair_z_delta,
        "primary_pair_weight": primary_pair_weight,
        "primary_pair_pass_gate": primary_pair_pass_gate,
        "prefix_step_index": prefix_step_index,
        "has_primary_corruption": has_primary_corruption,
        "primary_corruption_type": primary_corruption_type,
        "primary_corruption_features": corruption_features,
        "contrastive_corruption_features": corruption_features_bank,
        "contrastive_candidate_mask": corruption_candidate_mask,
        "contrastive_candidate_pair_weight": corruption_candidate_pair_weight,
        "contrastive_candidate_delta_q": corruption_candidate_delta_q,
        "contrastive_candidate_z_delta": corruption_candidate_z_delta,
        "contrastive_candidate_pass_gate": corruption_candidate_pass_gate,
    }
    return move_tensors_to_device(payload, torch_module.device("cpu"), torch_module)


def _extract_contrastive_candidates(
    *,
    example: ValueSupervisionExample,
    max_items: int,
) -> list[dict[str, Any]]:
    """Resolve ranked corruption candidates for one clean example.

    Notes
    -----
    - Prefer `example.corruption_candidates` (top-k prepared in C1 loader).
    - Fall back to legacy primary-corruption fields for backward compatibility.
    """
    if max_items <= 0:
        raise ValueError("`max_items` must be > 0")
    candidates: list[dict[str, Any]] = []
    if example.corruption_candidates:
        # Prefer the new C1 top-k candidates because they already carry ranking-quality metadata.
        # 新链路优先使用 C1 已经排好序、带质量字段的 top-k 候选。
        # Those candidates usually include delta_q / z_delta / pair_weight / gate results.
        # 这些候选往往已经包含 delta_q / z_delta / pair_weight / gate 结果。
        for item in example.corruption_candidates:
            if len(candidates) >= max_items:
                break
            text = item.get("corrupted_prefix_text")
            ctype = item.get("corruption_type")
            cidx = item.get("corruption_step_index")
            if not isinstance(text, str) or text.strip() == "":
                continue
            if not isinstance(ctype, str) or ctype.strip() == "":
                continue
            if not isinstance(cidx, int) or cidx < 0:
                continue
            candidates.append(
                {
                    "corrupted_prefix_text": text,
                    "corruption_type": ctype,
                    "corruption_step_index": int(cidx),
                    "pair_delta_q": float(item.get("pair_delta_q", 0.0) or 0.0),
                    "pair_z_delta": float(item.get("pair_z_delta", 0.0) or 0.0),
                    "pair_weight": float(item.get("pair_weight", 0.0) or 0.0),
                    "pair_pass_gate": bool(item.get("pair_pass_gate", False)),
                }
            )
        return candidates
    if example.primary_corruption_text is None:
        return candidates
    # Keep the legacy fallback so older artifacts remain replayable without backfilling new fields.
    # 这里保留 legacy 回退，是为了兼容旧 artifact，否则很多历史 Phase C 结果会无法重跑。
    candidates.append(
        {
            "corrupted_prefix_text": str(example.primary_corruption_text),
            "corruption_type": str(example.primary_corruption_type or "__unknown__"),
            "corruption_step_index": int(example.primary_corruption_step_index or 0),
            "pair_delta_q": float(example.primary_pair_delta_q or 0.0),
            "pair_z_delta": float(example.primary_pair_z_delta or 0.0),
            "pair_weight": float(example.primary_pair_weight or 0.0),
            "pair_pass_gate": bool(example.primary_pair_pass_gate),
        }
    )
    return candidates


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
        return move_tensors_to_device(
            {"corruption_features": empty},
            torch_module.device("cpu"),
            torch_module,
        )
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
    return move_tensors_to_device(
        {"corruption_features": corruption_features},
        torch_module.device("cpu"),
        torch_module,
    )


def _encode_external_pair_cache(
    *,
    pairs: list[ExternalPairRecord],
    backbone: Any,
    tokenizer: Any,
    torch_module: Any,
    max_length: int,
    batch_size: int,
    use_confidence_weights: bool,
) -> dict[str, Any]:
    """Encode external chosen/rejected pair texts into cached tensors."""
    if not pairs:
        hidden_size = infer_backbone_hidden_size(backbone)
        device = _resolve_value_device(backbone, torch_module)
        empty_features = torch_module.zeros((0, hidden_size), dtype=torch_module.float32, device=device)
        empty_weights = torch_module.zeros((0,), dtype=torch_module.float32, device=device)
        return move_tensors_to_device(
            {
            "num_pairs": 0,
            "chosen_features": empty_features,
            "rejected_features": empty_features.clone(),
            "pair_weights": empty_weights,
            "pair_ids": [],
            "source_tags": [],
            "domain_tags": [],
            },
            torch_module.device("cpu"),
            torch_module,
        )
    chosen_features = _encode_text_list_in_batches(
        texts=[pair.chosen_input_text() for pair in pairs],
        backbone=backbone,
        tokenizer=tokenizer,
        torch_module=torch_module,
        max_length=max_length,
        batch_size=batch_size,
        progress_label="cache_ext_chosen",
        progress_every_batches=32,
    )
    rejected_features = _encode_text_list_in_batches(
        texts=[pair.rejected_input_text() for pair in pairs],
        backbone=backbone,
        tokenizer=tokenizer,
        torch_module=torch_module,
        max_length=max_length,
        batch_size=batch_size,
        progress_label="cache_ext_rejected",
        progress_every_batches=32,
    )
    device = chosen_features.device
    pair_weights = torch_module.tensor(
        [
            (float(pair.pair_confidence) if use_confidence_weights else 1.0)
            for pair in pairs
        ],
        dtype=torch_module.float32,
        device=device,
    )
    payload = {
        "num_pairs": int(len(pairs)),
        "chosen_features": chosen_features,
        "rejected_features": rejected_features,
        "pair_weights": pair_weights,
        "pair_ids": [str(pair.pair_id) for pair in pairs],
        "source_tags": [str(pair.source_tag) for pair in pairs],
        "domain_tags": [str(pair.domain_tag) for pair in pairs],
    }
    return move_tensors_to_device(payload, torch_module.device("cpu"), torch_module)


def _build_value_example_cache_signature_payload(
    *,
    examples: list[ValueSupervisionExample],
    target_values: list[float],
    use_primary_corruption: bool,
    contrastive_max_corruptions_per_prefix: int,
    calibration_sample_weighting: str,
    calibration_weight_floor: float,
    calibration_weight_gamma: float,
    max_length: int,
    backbone_signature: dict[str, Any],
) -> dict[str, Any]:
    """Build conservative signature payload for one example-cache artifact."""
    if len(examples) != len(target_values):
        raise ValueError(
            "Signature build expects target_values length == examples length: "
            f"{len(target_values)} vs {len(examples)}"
        )
    clean_texts = [example.clean_input_text() for example in examples]
    selected_candidates: list[list[dict[str, Any]]] = []
    max_items = int(contrastive_max_corruptions_per_prefix if use_primary_corruption else 1)
    for example in examples:
        candidates = (
            _extract_contrastive_candidates(example=example, max_items=max_items)
            if use_primary_corruption
            else []
        )
        selected_candidates.append(
            [
                {
                    "corrupted_prefix_text": str(item.get("corrupted_prefix_text", "")),
                    "corruption_type": str(item.get("corruption_type", "")),
                    "corruption_step_index": int(item.get("corruption_step_index", 0)),
                    "pair_delta_q": float(item.get("pair_delta_q", 0.0) or 0.0),
                    "pair_z_delta": float(item.get("pair_z_delta", 0.0) or 0.0),
                    "pair_weight": float(item.get("pair_weight", 0.0) or 0.0),
                    "pair_pass_gate": bool(item.get("pair_pass_gate", False)),
                }
                for item in candidates
            ]
        )
    return {
        "cache_kind": "value_example_cache",
        "backbone_signature": backbone_signature,
        "max_length": int(max_length),
        "num_examples": int(len(examples)),
        "use_primary_corruption": bool(use_primary_corruption),
        "contrastive_max_corruptions_per_prefix": int(contrastive_max_corruptions_per_prefix),
        "calibration_sample_weighting": str(calibration_sample_weighting),
        "calibration_weight_floor": float(calibration_weight_floor),
        "calibration_weight_gamma": float(calibration_weight_gamma),
        "prefix_id_hash": hash_text_list([example.prefix_id for example in examples]),
        "clean_text_hash": hash_text_list(clean_texts),
        "target_value_hash": hash_float_list([float(v) for v in target_values]),
        "target_q_weight_hash": hash_float_list([float(example.target_q_weight) for example in examples]),
        "target_parseable_hash": hash_float_list([float(example.target_parseable_rate) for example in examples]),
        "prefix_step_index_hash": hash_int_list([int(example.prefix_step_index) for example in examples]),
        "candidate_hash": hash_jsonable(selected_candidates),
    }


def _build_corruption_variant_cache_signature_payload(
    *,
    variants: list[CorruptionVariant],
    max_length: int,
    backbone_signature: dict[str, Any],
) -> dict[str, Any]:
    """Build conservative signature payload for eval corruption-variant cache."""
    return {
        "cache_kind": "value_eval_corruption_cache",
        "backbone_signature": backbone_signature,
        "max_length": int(max_length),
        "num_variants": int(len(variants)),
        "corruption_id_hash": hash_text_list([variant.corruption_id for variant in variants]),
        "clean_prefix_id_hash": hash_text_list([variant.clean_prefix_id for variant in variants]),
        "corruption_text_hash": hash_text_list(
            [variant.corrupted_input_text() for variant in variants]
        ),
        "corruption_type_hash": hash_text_list([variant.corruption_type for variant in variants]),
        "corruption_step_hash": hash_int_list([int(variant.corruption_step_index) for variant in variants]),
    }


def _build_external_pair_cache_signature_payload(
    *,
    pairs: list[ExternalPairRecord],
    use_confidence_weights: bool,
    max_length: int,
    backbone_signature: dict[str, Any],
) -> dict[str, Any]:
    """Build conservative signature payload for external-pair cache."""
    return {
        "cache_kind": "value_external_pair_cache",
        "backbone_signature": backbone_signature,
        "max_length": int(max_length),
        "num_pairs": int(len(pairs)),
        "use_confidence_weights": bool(use_confidence_weights),
        "pair_id_hash": hash_text_list([pair.pair_id for pair in pairs]),
        "source_hash": hash_text_list([pair.source_tag for pair in pairs]),
        "domain_hash": hash_text_list([pair.domain_tag for pair in pairs]),
        "chosen_text_hash": hash_text_list([pair.chosen_input_text() for pair in pairs]),
        "rejected_text_hash": hash_text_list([pair.rejected_input_text() for pair in pairs]),
        "pair_confidence_hash": hash_float_list([float(pair.pair_confidence) for pair in pairs]),
    }


def _validate_cached_example_cache_payload(
    *,
    cache: Any,
    expected_num_examples: int,
    expected_hidden_size: int,
    torch_module: Any,
) -> None:
    """Validate loaded example-cache payload shape contract before reuse."""
    if not isinstance(cache, dict):
        raise TypeError("Example cache payload must be dict")
    required_keys = {
        "clean_features",
        "targets",
        "target_q_weight",
        "target_parseable",
        "target_confidence",
        "calibration_weights",
        "has_primary_corruption",
        "primary_corruption_type",
        "primary_corruption_features",
        "contrastive_corruption_features",
        "contrastive_candidate_mask",
        "contrastive_candidate_pair_weight",
        "contrastive_candidate_delta_q",
        "contrastive_candidate_z_delta",
        "contrastive_candidate_pass_gate",
    }
    missing = sorted(required_keys - set(cache.keys()))
    if missing:
        raise KeyError(f"Example cache payload missing keys: {missing}")
    clean = cache["clean_features"]
    if not torch_module.is_tensor(clean) or clean.ndim != 2:
        raise TypeError("Example cache `clean_features` must be tensor[batch, hidden]")
    if int(clean.shape[0]) != int(expected_num_examples):
        raise ValueError(
            "Example cache batch mismatch: "
            f"expected {expected_num_examples}, got {int(clean.shape[0])}"
        )
    if int(clean.shape[1]) != int(expected_hidden_size):
        raise ValueError(
            "Example cache hidden mismatch: "
            f"expected {expected_hidden_size}, got {int(clean.shape[1])}"
        )
    for name in ("targets", "target_q_weight", "target_parseable", "target_confidence", "calibration_weights"):
        tensor = cache[name]
        if not torch_module.is_tensor(tensor) or tensor.ndim != 1:
            raise TypeError(f"Example cache `{name}` must be tensor[batch]")
        if int(tensor.shape[0]) != int(expected_num_examples):
            raise ValueError(
                f"Example cache `{name}` batch mismatch: expected {expected_num_examples}, got {int(tensor.shape[0])}"
            )
    for name in ("has_primary_corruption", "primary_pair_delta_q", "primary_pair_z_delta", "primary_pair_weight", "primary_pair_pass_gate", "prefix_step_index"):
        tensor = cache[name]
        if not torch_module.is_tensor(tensor) or tensor.ndim != 1:
            raise TypeError(f"Example cache `{name}` must be tensor[batch]")
        if int(tensor.shape[0]) != int(expected_num_examples):
            raise ValueError(
                f"Example cache `{name}` batch mismatch: expected {expected_num_examples}, got {int(tensor.shape[0])}"
            )
    primary_types = cache["primary_corruption_type"]
    if not isinstance(primary_types, list) or len(primary_types) != int(expected_num_examples):
        raise TypeError("Example cache `primary_corruption_type` must be list[batch]")
    primary_corrupt = cache["primary_corruption_features"]
    if not torch_module.is_tensor(primary_corrupt) or primary_corrupt.ndim != 2:
        raise TypeError("Example cache `primary_corruption_features` must be tensor[batch, hidden]")
    if int(primary_corrupt.shape[0]) != int(expected_num_examples) or int(primary_corrupt.shape[1]) != int(expected_hidden_size):
        raise ValueError("Example cache `primary_corruption_features` shape mismatch")
    candidate_bank = cache["contrastive_corruption_features"]
    if not torch_module.is_tensor(candidate_bank) or candidate_bank.ndim != 3:
        raise TypeError("Example cache `contrastive_corruption_features` must be tensor[batch, k, hidden]")
    if int(candidate_bank.shape[0]) != int(expected_num_examples) or int(candidate_bank.shape[2]) != int(expected_hidden_size):
        raise ValueError("Example cache `contrastive_corruption_features` shape mismatch")
    candidate_width = int(candidate_bank.shape[1])
    for name in (
        "contrastive_candidate_mask",
        "contrastive_candidate_pair_weight",
        "contrastive_candidate_delta_q",
        "contrastive_candidate_z_delta",
        "contrastive_candidate_pass_gate",
    ):
        tensor = cache[name]
        if not torch_module.is_tensor(tensor) or tensor.ndim != 2:
            raise TypeError(f"Example cache `{name}` must be tensor[batch, k]")
        if int(tensor.shape[0]) != int(expected_num_examples) or int(tensor.shape[1]) != candidate_width:
            raise ValueError(f"Example cache `{name}` shape mismatch")


def _validate_cached_corruption_cache_payload(
    *,
    cache: Any,
    expected_num_variants: int,
    expected_hidden_size: int,
    torch_module: Any,
) -> None:
    """Validate loaded corruption-cache payload shape contract before reuse."""
    if not isinstance(cache, dict):
        raise TypeError("Corruption cache payload must be dict")
    if "corruption_features" not in cache:
        raise KeyError("Corruption cache payload missing key `corruption_features`")
    feats = cache["corruption_features"]
    if not torch_module.is_tensor(feats) or feats.ndim != 2:
        raise TypeError("Corruption cache `corruption_features` must be tensor[batch, hidden]")
    if int(feats.shape[0]) != int(expected_num_variants):
        raise ValueError(
            "Corruption cache batch mismatch: "
            f"expected {expected_num_variants}, got {int(feats.shape[0])}"
        )
    if int(feats.shape[1]) != int(expected_hidden_size):
        raise ValueError(
            "Corruption cache hidden mismatch: "
            f"expected {expected_hidden_size}, got {int(feats.shape[1])}"
        )


def _validate_cached_external_pair_cache_payload(
    *,
    cache: Any,
    expected_num_pairs: int,
    expected_hidden_size: int,
    torch_module: Any,
) -> None:
    """Validate loaded external-pair cache payload contract before reuse."""
    if not isinstance(cache, dict):
        raise TypeError("External pair cache payload must be dict")
    required_keys = {
        "num_pairs",
        "chosen_features",
        "rejected_features",
        "pair_weights",
        "source_tags",
        "domain_tags",
    }
    missing = sorted(required_keys - set(cache.keys()))
    if missing:
        raise KeyError(f"External pair cache payload missing keys: {missing}")
    num_pairs = int(cache["num_pairs"])
    if num_pairs != int(expected_num_pairs):
        raise ValueError(
            f"External pair cache count mismatch: expected {expected_num_pairs}, got {num_pairs}"
        )
    for name in ("chosen_features", "rejected_features"):
        tensor = cache[name]
        if not torch_module.is_tensor(tensor) or tensor.ndim != 2:
            raise TypeError(f"External pair cache `{name}` must be tensor[batch, hidden]")
        if int(tensor.shape[0]) != int(expected_num_pairs):
            raise ValueError(
                f"External pair cache `{name}` batch mismatch: expected {expected_num_pairs}, got {int(tensor.shape[0])}"
            )
        if int(tensor.shape[1]) != int(expected_hidden_size):
            raise ValueError(
                f"External pair cache `{name}` hidden mismatch: expected {expected_hidden_size}, got {int(tensor.shape[1])}"
            )
    weights = cache["pair_weights"]
    if not torch_module.is_tensor(weights) or weights.ndim != 1:
        raise TypeError("External pair cache `pair_weights` must be tensor[batch]")
    if int(weights.shape[0]) != int(expected_num_pairs):
        raise ValueError(
            "External pair cache `pair_weights` batch mismatch: "
            f"expected {expected_num_pairs}, got {int(weights.shape[0])}"
        )
    if "pair_ids" in cache:
        pair_ids = cache["pair_ids"]
        if not isinstance(pair_ids, list):
            raise TypeError("External pair cache `pair_ids` must be a list when present")
        if len(pair_ids) != int(expected_num_pairs):
            raise ValueError(
                "External pair cache `pair_ids` length mismatch: "
                f"expected {expected_num_pairs}, got {len(pair_ids)}"
            )
    for name in ("source_tags", "domain_tags"):
        values = cache[name]
        if not isinstance(values, list) or len(values) != int(expected_num_pairs):
            raise TypeError(f"External pair cache `{name}` must be list[batch]")


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


def _set_seed(
    seed: int,
    torch_module: Any,
    *,
    strict_determinism: bool = False,
) -> None:
    """Seed RNGs; optionally enable stricter deterministic backend behavior."""
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    if bool(strict_determinism):
        # Best-effort deterministic mode for seed-stability diagnostics.
        # We intentionally use warn_only to avoid hard failures on unsupported ops.
        torch_module.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch_module.backends, "cudnn"):
            torch_module.backends.cudnn.deterministic = True
            torch_module.backends.cudnn.benchmark = False
        if hasattr(torch_module.backends, "cuda") and hasattr(torch_module.backends.cuda, "matmul"):
            torch_module.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch_module.backends, "cudnn"):
            torch_module.backends.cudnn.allow_tf32 = False


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


def _parse_csv_allow_list(text: str) -> set[str] | None:
    """Parse comma-separated allow-list; return None when empty."""
    cleaned = [item.strip() for item in str(text).split(",") if item.strip() != ""]
    if not cleaned:
        return None
    return set(cleaned)


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
