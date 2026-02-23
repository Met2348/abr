"""Unit tests for canonical data schema."""

from __future__ import annotations

import pytest

from ours.data.schema import CanonicalSample, ensure_canonical_samples


def test_canonical_sample_validate_success() -> None:
    sample = CanonicalSample(
        id="gsm8k:main:train:0",
        dataset="gsm8k",
        question="What is 1 + 1?",
        answer="2",
        cot="1 + 1 = 2",
        metadata={"source_split": "train"},
    )
    sample.validate()
    payload = sample.to_dict()
    assert payload["answer"] == "2"


def test_canonical_sample_rejects_empty_question() -> None:
    sample = CanonicalSample(
        id="x",
        dataset="gsm8k",
        question="   ",
        answer="2",
    )
    with pytest.raises(ValueError, match="question"):
        sample.validate()


def test_ensure_canonical_samples_accepts_dicts() -> None:
    rows = [
        {
            "id": "strategyqa:q1",
            "dataset": "strategyqa",
            "question": "Is water wet?",
            "answer": "yes",
            "cot": None,
            "metadata": {"source": "test"},
        }
    ]
    samples = ensure_canonical_samples(rows, source_name="unit_test")
    assert len(samples) == 1
    assert samples[0].dataset == "strategyqa"

