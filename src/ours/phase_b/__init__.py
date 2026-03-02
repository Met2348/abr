"""Expose the small public surface of the Phase B training package.

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
"""

from .contracts import PhaseBTrainRow
from .data import load_phase_b_rows, summarize_rows
from .supervision import (
    SupervisionPlan,
    build_supervision_plan,
    list_target_transforms,
    split_reasoning_and_answer,
)

__all__ = [
    "PhaseBTrainRow",
    "load_phase_b_rows",
    "summarize_rows",
    "SupervisionPlan",
    "build_supervision_plan",
    "list_target_transforms",
    "split_reasoning_and_answer",
]
