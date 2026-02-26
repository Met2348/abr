"""Unit tests for small helper functions in loaders.py."""

from __future__ import annotations

import pytest

from ours.data.loaders import (
    _build_drop_question,
    _build_proofwriter_question,
    _extract_hendrycks_final_answer,
    _extract_last_boxed_content,
    _normalize_split,
)


def test_build_drop_question_includes_passage_and_question() -> None:
    prompt = _build_drop_question("A short passage.", "What is asked?")
    assert prompt.startswith("Passage: A short passage.")
    assert "\nQuestion: What is asked?" in prompt


def test_build_proofwriter_question_includes_theory_and_question() -> None:
    prompt = _build_proofwriter_question("All cats are mammals.", "Is Tom a mammal?")
    assert prompt.startswith("Theory: All cats are mammals.")
    assert "\nQuestion: Is Tom a mammal?" in prompt


def test_extract_last_boxed_content_simple() -> None:
    text = "Hence the result is \\boxed{42}."
    assert _extract_last_boxed_content(text) == "42"


def test_extract_last_boxed_content_nested() -> None:
    text = "Final answer: \\boxed{\\frac{1}{2}}."
    assert _extract_last_boxed_content(text) == "\\frac{1}{2}"


def test_extract_hendrycks_answer_prefers_boxed() -> None:
    solution = "Work...\nTherefore, answer is \\boxed{0}."
    answer, method = _extract_hendrycks_final_answer(solution)
    assert answer == "0"
    assert method == "boxed"


def test_extract_hendrycks_answer_fallback_last_line() -> None:
    solution = "Step 1\nStep 2\nTherefore, 17"
    answer, method = _extract_hendrycks_final_answer(solution)
    assert answer == "17"
    assert method == "last_line"


def test_normalize_split_accepts_alias_to_validation() -> None:
    assert _normalize_split("val", available={"train", "validation"}, fallback="validation") == "validation"


def test_normalize_split_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        _normalize_split("trian", available={"train", "test"}, fallback="test")
