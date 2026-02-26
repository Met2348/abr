"""Unit tests for answer extraction and evaluator behavior."""

from __future__ import annotations

from ours.phase_a import PredictionRecord, evaluate_predictions, extract_answer


def test_extract_strategyqa_yes() -> None:
    ext = extract_answer("Yes, definitely.", dataset="strategyqa")
    assert ext.text == "yes"
    assert not ext.parse_error


def test_extract_strategyqa_chat_leakage_prefix() -> None:
    ext = extract_answer("noHuman: Is the following statement true?", dataset="strategyqa")
    assert ext.text == "no"
    assert not ext.parse_error


def test_extract_strategyqa_final_answer_tag() -> None:
    ext = extract_answer("Reasoning...\nFinal answer: true", dataset="strategyqa")
    assert ext.text == "yes"
    assert not ext.parse_error


def test_extract_strategyqa_uses_last_binary_token() -> None:
    ext = extract_answer("At first I guessed no, but final answer is yes", dataset="strategyqa")
    assert ext.text == "yes"
    assert not ext.parse_error


def test_extract_gsm8k_hash_marker() -> None:
    ext = extract_answer("Work... #### 42", dataset="gsm8k")
    assert ext.text == "42"
    assert ext.method == "gsm8k_hash_marker"


def test_extract_gsm8k_final_answer_is() -> None:
    ext = extract_answer("After solving, final answer is 37", dataset="gsm8k")
    assert ext.text == "37"
    assert ext.method == "final_answer_tag"


def test_extract_gsm8k_final_answer_with_units() -> None:
    ext = extract_answer("Final answer: 10 meters.", dataset="gsm8k")
    assert ext.text == "10"
    assert ext.method == "final_answer_tag"


def test_extract_hendrycks_boxed() -> None:
    ext = extract_answer("Hence \\boxed{1/2}", dataset="hendrycks_math")
    assert ext.text == "1/2"
    assert ext.method == "boxed"


def test_evaluator_numeric_equivalence_fraction_vs_decimal() -> None:
    records = [
        PredictionRecord(
            sample_id="m1",
            dataset="hendrycks_math",
            split="test",
            raw_prediction="Final answer: 0.5",
            gold_answer="1/2",
        )
    ]
    scored, summary = evaluate_predictions(records)
    assert scored[0].is_correct
    assert summary.accuracy == 1.0


def test_evaluator_numeric_equivalence_with_units_text() -> None:
    records = [
        PredictionRecord(
            sample_id="m2",
            dataset="gsm8k",
            split="validation",
            raw_prediction="Final answer: $140.",
            gold_answer="140",
        ),
        PredictionRecord(
            sample_id="m3",
            dataset="gsm8k",
            split="validation",
            raw_prediction="Final answer: 75% of flowers are not roses.",
            gold_answer="75",
        ),
    ]
    scored, summary = evaluate_predictions(records)
    assert scored[0].is_correct
    assert scored[1].is_correct
    assert summary.accuracy == 1.0


def test_evaluator_allows_empty_prediction_and_marks_parse_error() -> None:
    records = [
        PredictionRecord(
            sample_id="q-empty",
            dataset="strategyqa",
            split="validation",
            raw_prediction="",
            gold_answer="yes",
        )
    ]
    scored, summary = evaluate_predictions(records)
    assert scored[0].parse_error
    assert not scored[0].is_correct
    assert summary.n_parse_error == 1
    assert summary.accuracy == 0.0


def test_evaluator_strategyqa_accuracy() -> None:
    records = [
        PredictionRecord(
            sample_id="q1",
            dataset="strategyqa",
            split="validation",
            raw_prediction="Yes",
            gold_answer="yes",
        ),
        PredictionRecord(
            sample_id="q2",
            dataset="strategyqa",
            split="validation",
            raw_prediction="No",
            gold_answer="yes",
        ),
    ]
    _, summary = evaluate_predictions(records)
    assert summary.n_total == 2
    assert summary.n_correct == 1
    assert summary.accuracy == 0.5
