"""Unit tests for Phase A prompt building utilities."""

from __future__ import annotations

import pytest

from ours.data.schema import CanonicalSample
from ours.phase_a import build_prepared_sample, list_template_versions, resolve_template


def test_list_template_versions_contains_v1() -> None:
    versions = list_template_versions("qa_direct")
    assert "1.0.0" in versions


def test_resolve_template_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        resolve_template("unknown_template", "1.0.0")


def test_list_template_versions_contains_math_direct() -> None:
    versions = list_template_versions("qa_math_direct_final")
    assert "1.0.0" in versions


def test_build_prepared_sample_answer_only() -> None:
    sample = CanonicalSample(
        id="toy:1",
        dataset="gsm8k",
        question="What is 1+1?",
        answer="2",
        cot="1+1=2",
    )
    prepared = build_prepared_sample(
        sample=sample,
        split="train",
        target_style="answer_only",
        template_id="qa_direct",
        template_version="1.0.0",
    )

    assert prepared.target_text == "2"
    assert "[SYSTEM]" in prepared.prompt_text
    assert "[USER]" in prepared.prompt_text
    assert "[ASSISTANT]" in prepared.prompt_text


def test_build_prepared_sample_cot_then_answer_fallback_without_cot() -> None:
    sample = CanonicalSample(
        id="toy:2",
        dataset="strategyqa",
        question="Is water wet?",
        answer="yes",
        cot=None,
    )
    prepared = build_prepared_sample(
        sample=sample,
        split="validation",
        target_style="cot_then_answer",
        template_id="qa_cot_then_final",
        template_version="1.0.0",
    )

    assert prepared.target_text == "Final answer: yes"
