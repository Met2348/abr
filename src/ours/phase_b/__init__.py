"""Expose the public surface of the Phase B/Phase C training package.

Why this file exists
--------------------
Callers such as `scripts/phase_b_train_sft.py` should import a compact Phase B API
without needing to know which submodule defines each symbol.

What this file contains
-----------------------
- `PhaseBTrainRow`: validated training-row contract
- `load_phase_b_rows(...)`: strict JSONL loader
- `summarize_rows(...)`: compact dataset/run summary helper
- `SupervisionPlan` and helpers: supervision transforms and token-weight planning
- Phase C prefix/rollout/corruption contracts and builders
"""

from .contracts import PhaseBTrainRow
from .corruptions import (
    CorruptionArtifact,
    CorruptionBuildConfig,
    build_corruptions_for_prefixes,
    summarize_corruptions,
)
from .faithfulness_eval import (
    compute_binary_auc,
    compute_calibration_summary,
    compute_corruption_summary,
    render_faithfulness_summary_markdown,
)
from .data import load_phase_b_rows, summarize_rows
from .supervision import (
    SupervisionPlan,
    build_supervision_plan,
    list_target_transforms,
    split_reasoning_and_answer,
)
from .value_data import (
    CorruptionVariant,
    ValueSupervisionExample,
    assert_phase_c_compatibility,
    load_corruption_variants,
    load_phase_c_manifest,
    load_value_supervision_examples,
)
from .value_head import (
    SigmoidValueHead,
    ValueHeadConfig,
    encode_text_features,
    freeze_backbone,
    infer_backbone_hidden_size,
    load_value_head_checkpoint,
    pool_last_token,
    resolve_model_input_device,
    save_value_head_checkpoint,
    write_value_head_config_json,
)
from .value_losses import (
    bellman_consistency_loss,
    contrastive_margin_loss,
    mean_squared_calibration_loss,
)
from .value_targets import (
    PrefixArtifact,
    PrefixBuildConfig,
    RolloutPredictionRecord,
    RolloutTargetRecord,
    build_prefix_artifacts,
    build_step_sequence_from_phase_b_row,
    summarize_prefix_artifacts,
    summarize_rollout_targets,
)

__all__ = [
    "PhaseBTrainRow",
    "load_phase_b_rows",
    "summarize_rows",
    "SupervisionPlan",
    "build_supervision_plan",
    "list_target_transforms",
    "split_reasoning_and_answer",
    "PrefixArtifact",
    "PrefixBuildConfig",
    "RolloutPredictionRecord",
    "RolloutTargetRecord",
    "build_prefix_artifacts",
    "build_step_sequence_from_phase_b_row",
    "summarize_prefix_artifacts",
    "summarize_rollout_targets",
    "CorruptionArtifact",
    "CorruptionBuildConfig",
    "build_corruptions_for_prefixes",
    "summarize_corruptions",
    "ValueSupervisionExample",
    "CorruptionVariant",
    "load_phase_c_manifest",
    "assert_phase_c_compatibility",
    "load_value_supervision_examples",
    "load_corruption_variants",
    "ValueHeadConfig",
    "SigmoidValueHead",
    "freeze_backbone",
    "infer_backbone_hidden_size",
    "resolve_model_input_device",
    "encode_text_features",
    "pool_last_token",
    "save_value_head_checkpoint",
    "load_value_head_checkpoint",
    "write_value_head_config_json",
    "mean_squared_calibration_loss",
    "contrastive_margin_loss",
    "bellman_consistency_loss",
    "compute_calibration_summary",
    "compute_corruption_summary",
    "compute_binary_auc",
    "render_faithfulness_summary_markdown",
]
