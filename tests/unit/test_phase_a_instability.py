from __future__ import annotations

from ours.phase_a.instability import (
    compute_pairwise_prediction_flip,
    extract_final_answer_sequence,
    summarize_strategyqa_instability,
)


def test_extract_final_answer_sequence_normalizes_binary_tokens() -> None:
    raw = (
        "Reasoning... Final answer: yes. "
        "Correction: Answer: false. "
        "Final answer: TRUE"
    )
    assert extract_final_answer_sequence(raw) == ["yes", "no", "yes"]


def test_summarize_strategyqa_instability_counts_and_rates() -> None:
    rows = [
        {
            "sample_id": "s1",
            "raw_prediction": "Final answer: yes",
            "extracted_prediction": "yes",
            "is_correct": True,
            "parse_error": False,
        },
        {
            "sample_id": "s2",
            "raw_prediction": "Final answer: yes ... Final answer: no",
            "extracted_prediction": "no",
            "is_correct": False,
            "parse_error": False,
        },
        {
            "sample_id": "s3",
            "raw_prediction": "No final tag here",
            "extracted_prediction": "",
            "is_correct": False,
            "parse_error": True,
        },
        {
            "sample_id": "s4",
            "raw_prediction": "Answer: true. Final answer: yes",
            "extracted_prediction": "yes",
            "is_correct": True,
            "parse_error": False,
        },
    ]

    out = summarize_strategyqa_instability(rows)

    assert out["n_total"] == 4
    assert out["n_correct"] == 2
    assert out["n_parse_error"] == 1

    assert out["n_with_final_tag"] == 3
    assert out["n_multi_final_tag"] == 2
    assert out["n_first_last_disagree"] == 1
    assert out["n_with_tag_switch"] == 1

    assert out["accuracy"] == 0.5
    assert out["parse_error_rate"] == 0.25
    assert out["with_final_tag_rate"] == 0.75
    assert out["multi_final_tag_rate"] == 0.5
    assert out["first_last_disagree_rate"] == 0.25
    assert out["tag_switch_rate"] == 0.25


def test_compute_pairwise_prediction_flip_with_overlap() -> None:
    a = {
        "s1": {"extracted_prediction": "yes", "is_correct": True},
        "s2": {"extracted_prediction": "no", "is_correct": False},
        "s3": {"extracted_prediction": "yes", "is_correct": True},
    }
    b = {
        "s1": {"extracted_prediction": "no", "is_correct": False},
        "s2": {"extracted_prediction": "no", "is_correct": False},
        "s4": {"extracted_prediction": "yes", "is_correct": True},
    }

    out = compute_pairwise_prediction_flip(a, b)

    assert out["n_overlap"] == 2
    assert out["n_pred_flip"] == 1
    assert out["pred_flip_rate"] == 0.5
    assert out["n_correctness_flip"] == 1
    assert out["correctness_flip_rate"] == 0.5
    assert out["n_yes_to_no"] == 1
    assert out["n_no_to_yes"] == 0
