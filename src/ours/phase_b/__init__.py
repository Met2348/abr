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
from .data import load_phase_b_rows, summarize_rows
from .supervision import (
    SupervisionPlan,
    build_supervision_plan,
    list_target_transforms,
    split_reasoning_and_answer,
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
]
