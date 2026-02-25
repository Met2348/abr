"""Data utilities for the OURS project.

Exports:
- canonical sample schema
- dataset loading entry points
- step-level preprocessing builders
"""

from .schema import CanonicalSample, ensure_canonical_samples
from .loaders import DATASET_LOADERS, load_dataset_canonical
from .step_builder import (
    ReasoningStep,
    SplitMode,
    StepBuildConfig,
    StepSequence,
    StepRole,
    build_step_sequence,
    build_step_sequences,
    iter_flat_steps,
    split_reasoning_text,
)

__all__ = [
    "CanonicalSample",
    "ensure_canonical_samples",
    "DATASET_LOADERS",
    "load_dataset_canonical",
    "StepRole",
    "SplitMode",
    "StepBuildConfig",
    "ReasoningStep",
    "StepSequence",
    "build_step_sequence",
    "build_step_sequences",
    "iter_flat_steps",
    "split_reasoning_text",
]
