"""Unit tests for stratified-sampling helpers in scripts/phase_b_train_value.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_phase_c_train_value_module():
    """Load the Phase C value-head training script as a Python module."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_b_train_value.py"
    spec = importlib.util.spec_from_file_location("phase_b_train_value", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_stratified_train_permutation_keeps_all_indices() -> None:
    """Stratified permutation must keep full coverage with no dropping."""
    module = _load_phase_c_train_value_module()
    train_cache = {
        "clean_features": torch.zeros((6, 3), dtype=torch.float32),
        "has_primary_corruption": torch.tensor(
            [True, True, True, False, True, False], dtype=torch.bool
        ),
        "prefix_step_index": torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
        "primary_corruption_type": [
            "numeric_perturb",
            "step_drop",
            "negation_flip",
            "__none__",
            "comparator_flip",
            "__none__",
        ],
    }
    perm = module._build_stratified_train_permutation(
        train_cache=train_cache,
        torch_module=torch,
        step_bucket_size=2,
        include_no_corruption=True,
    )
    ordered = perm.detach().cpu().tolist()
    assert len(ordered) == 6
    assert sorted(ordered) == [0, 1, 2, 3, 4, 5]


def test_summarize_strata_for_logging_reports_non_empty_stats() -> None:
    """The strata logger helper should return useful aggregate counts."""
    module = _load_phase_c_train_value_module()
    train_cache = {
        "clean_features": torch.zeros((5, 2), dtype=torch.float32),
        "has_primary_corruption": torch.tensor(
            [True, True, False, True, False], dtype=torch.bool
        ),
        "prefix_step_index": torch.tensor([0, 2, 1, 4, 7], dtype=torch.long),
        "primary_corruption_type": [
            "numeric_perturb",
            "step_drop",
            "__none__",
            "entity_substitution",
            "__none__",
        ],
    }
    summary = module._summarize_strata_for_logging(
        train_cache=train_cache,
        step_bucket_size=2,
        include_no_corruption=True,
    )
    assert summary["num_strata"] >= 2
    assert summary["max_size"] >= summary["min_size"]
    assert isinstance(summary["top_strata"], list)
