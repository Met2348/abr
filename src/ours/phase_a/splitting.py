"""Deterministic split helpers for Phase A.

Why deterministic splitting?
----------------------------
If the same sample can jump between train/validation/test across runs,
metrics become noisy and hard to trust. We use stable hash-based split.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(slots=True)
class SplitConfig:
    """Ratios for deterministic local split."""

    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42

    def validate(self) -> None:
        ratios = [self.train_ratio, self.validation_ratio, self.test_ratio]
        if any(r < 0 or r > 1 for r in ratios):
            raise ValueError("All ratios must be in [0, 1]")
        total = sum(ratios)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Ratios must sum to 1.0, got train+validation+test={total}"
            )


def assign_split(sample_id: str, config: SplitConfig) -> str:
    """Assign one sample id to train/validation/test deterministically."""
    config.validate()
    token = f"{config.seed}:{sample_id}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    # Convert first 16 hex chars into a stable float in [0,1).
    value = int(digest[:16], 16) / float(16**16)

    train_bound = config.train_ratio
    validation_bound = config.train_ratio + config.validation_ratio

    if value < train_bound:
        return "train"
    if value < validation_bound:
        return "validation"
    return "test"


def split_ids(sample_ids: list[str], config: SplitConfig) -> dict[str, list[str]]:
    """Split many sample IDs deterministically."""
    buckets: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for sample_id in sample_ids:
        split = assign_split(sample_id=sample_id, config=config)
        buckets[split].append(sample_id)
    return buckets
