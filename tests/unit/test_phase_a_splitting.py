"""Unit tests for deterministic split helpers."""

from __future__ import annotations

import pytest

from ours.phase_a import SplitConfig, assign_split, split_ids


def test_assign_split_is_deterministic() -> None:
    cfg = SplitConfig(seed=123)
    s1 = assign_split("sample-a", cfg)
    s2 = assign_split("sample-a", cfg)
    assert s1 == s2


def test_assign_split_changes_with_seed() -> None:
    ids = [f"sample-{i}" for i in range(200)]
    cfg1 = SplitConfig(seed=1)
    cfg2 = SplitConfig(seed=999)

    assignments_1 = [assign_split(sample_id, cfg1) for sample_id in ids]
    assignments_2 = [assign_split(sample_id, cfg2) for sample_id in ids]

    # With two different seeds across many ids, we expect at least one
    # assignment difference. This avoids brittle single-id assumptions.
    assert assignments_1 != assignments_2


def test_split_config_ratio_validation() -> None:
    cfg = SplitConfig(train_ratio=0.8, validation_ratio=0.3, test_ratio=0.1)
    with pytest.raises(ValueError):
        cfg.validate()


def test_split_ids_covers_all_inputs() -> None:
    ids = [f"id-{i}" for i in range(10)]
    buckets = split_ids(ids, SplitConfig(seed=42))
    recovered = set(buckets["train"] + buckets["validation"] + buckets["test"])
    assert recovered == set(ids)
