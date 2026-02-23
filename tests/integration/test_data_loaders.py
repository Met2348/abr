"""Integration smoke tests for data loaders.

These tests are intentionally lightweight:
- they only load a few samples
- they skip gracefully when optional parquet runtime is unavailable
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ours.data.loaders import load_dataset_canonical


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "assets" / "datasets"
CACHE_DIR = REPO_ROOT / "assets" / "hf_cache" / "datasets_test_cache"


def test_strategyqa_loader_smoke() -> None:
    path = DATASET_ROOT / "strategyqa" / "strategyQA_train.json"
    if not path.exists():
        pytest.skip("StrategyQA train json not found in local assets.")

    samples = load_dataset_canonical(
        "strategyqa",
        dataset_root=DATASET_ROOT,
        split="train",
        limit=2,
        cache_dir=CACHE_DIR,
    )
    assert len(samples) == 2
    assert all(s.dataset == "strategyqa" for s in samples)
    assert all(s.question for s in samples)
    assert all(s.answer for s in samples)


def test_gsm8k_loader_smoke_optional_parquet_runtime() -> None:
    path = DATASET_ROOT / "gsm8k" / "main"
    if not path.exists():
        pytest.skip("GSM8K local parquet folder not found.")

    try:
        samples = load_dataset_canonical(
            "gsm8k",
            dataset_root=DATASET_ROOT,
            split="train",
            limit=2,
            cache_dir=CACHE_DIR,
            config="main",
        )
    except RuntimeError as exc:
        pytest.skip(f"Parquet runtime unavailable in this environment: {exc}")

    assert len(samples) == 2
    assert all(s.dataset == "gsm8k" for s in samples)
    assert all(s.question for s in samples)

