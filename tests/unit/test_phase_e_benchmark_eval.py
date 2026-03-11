"""Unit tests for Phase E benchmark evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from ours.phase_e.benchmark_eval import (
    build_processbench_prefix_records,
    compute_pair_ranking_metrics,
    compute_processbench_f1,
    compute_processbench_metrics,
    load_prmbench_preview_pairs,
    load_processbench_examples,
)


def test_processbench_metrics_separate_good_and_bad_prefixes(tmp_path: Path) -> None:
    path = tmp_path / "processbench.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "ex0",
                    "generator": "stub",
                    "problem": "Problem A",
                    "steps": ["good step", "first bad step", "later bad step"],
                    "label": 1,
                    "final_answer_correct": False,
                },
                {
                    "id": "ex1",
                    "generator": "stub",
                    "problem": "Problem B",
                    "steps": ["good 1", "good 2"],
                    "label": -1,
                    "final_answer_correct": True,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    examples = load_processbench_examples(path)
    rows = build_processbench_prefix_records(examples)
    scores = [0.9, 0.2, 0.1, 0.8, 0.7]
    metrics = compute_processbench_metrics(rows, scores)
    assert metrics["num_examples"] == 2
    assert metrics["num_error_examples"] == 1
    assert metrics["pair_accuracy_good_vs_bad"] > 0.99
    assert metrics["pair_auc_good_vs_bad"] > 0.99
    assert metrics["first_error_edge_accuracy"] > 0.99


def test_prmbench_preview_loader_and_metrics(tmp_path: Path) -> None:
    path = tmp_path / "prmbench_preview.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Compute 3 + 4.",
                "original_process": ["Add the two numbers.", "The answer is 7."],
                "modified_process": ["Add the two numbers.", "The answer is 8."],
                "error_steps": [2],
                "classification": "confidence",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_prmbench_preview_pairs(path)
    metrics = compute_pair_ranking_metrics(
        pair_ids=[row.pair_id for row in rows],
        group_keys=[row.classification for row in rows],
        chosen_scores=[0.9],
        rejected_scores=[0.1],
    )
    assert len(rows) == 1
    assert metrics["pair_accuracy"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["by_group"]["confidence"]["positive_margin_rate"] == 1.0


def test_prmbench_preview_loader_normalizes_one_based_step_indices(tmp_path: Path) -> None:
    path = tmp_path / "prmbench_preview_last_step.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Compute 3 + 4.",
                "original_process": ["Add the two numbers.", "The answer is 7."],
                "modified_process": ["Add the two numbers.", "The answer is 8."],
                "error_steps": [2],
                "classification": "confidence",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_prmbench_preview_pairs(path)

    assert len(rows) == 1
    assert rows[0].error_step_index == 1
    assert rows[0].chosen_prefix_text == "Add the two numbers.\nThe answer is 7."
    assert rows[0].rejected_prefix_text == "Add the two numbers.\nThe answer is 8."


def test_compute_processbench_f1_perfect_model(tmp_path: Path) -> None:
    path = tmp_path / "processbench.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "err0",
                    "generator": "stub",
                    "problem": "Error at step 1",
                    "steps": ["good step", "bad step", "later bad step"],
                    "label": 1,
                    "final_answer_correct": False,
                },
                {
                    "id": "ok0",
                    "generator": "stub",
                    "problem": "All correct",
                    "steps": ["good 1", "good 2"],
                    "label": -1,
                    "final_answer_correct": True,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    examples = load_processbench_examples(path)
    rows = build_processbench_prefix_records(examples)
    # Perfect model: good prefixes score high, bad prefixes score low.
    # err0: step0=good (0.9), step1=bad (0.2), step2=bad (0.1)
    # ok0: step0=good (0.9), step1=good (0.8)
    scores = [0.9, 0.2, 0.1, 0.9, 0.8]
    metrics = compute_processbench_f1(rows, scores, threshold=0.5)
    assert metrics["processbench_f1"] == 1.0
    assert metrics["processbench_acc_erroneous"] == 1.0
    assert metrics["processbench_acc_correct"] == 1.0
    assert metrics["processbench_f1_threshold_selection"] == "fixed"
    assert metrics["processbench_f1_is_oracle"] is False

    # Auto-tuned threshold should also find F1=1.0.
    metrics_auto = compute_processbench_f1(rows, scores)
    assert metrics_auto["processbench_f1"] == 1.0
    assert metrics_auto["processbench_f1_threshold_selection"] == "oracle_sweep"
    assert metrics_auto["processbench_f1_is_oracle"] is True


def test_compute_processbench_f1_random_model(tmp_path: Path) -> None:
    path = tmp_path / "processbench.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "err0",
                    "generator": "stub",
                    "problem": "Error at step 1",
                    "steps": ["good step", "bad step"],
                    "label": 1,
                    "final_answer_correct": False,
                },
                {
                    "id": "ok0",
                    "generator": "stub",
                    "problem": "All correct",
                    "steps": ["good 1", "good 2"],
                    "label": -1,
                    "final_answer_correct": True,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    examples = load_processbench_examples(path)
    rows = build_processbench_prefix_records(examples)
    # All scores above threshold: model predicts everything as "all correct".
    # Acc_erroneous=0 because error example not detected.
    # Acc_correct=1 because all-correct example correctly predicted.
    # F1 = 0 because harmonic mean includes 0.
    scores = [0.9, 0.8, 0.9, 0.9]
    metrics = compute_processbench_f1(rows, scores, threshold=0.5)
    assert metrics["processbench_f1"] == 0.0
    assert metrics["processbench_acc_erroneous"] == 0.0
    assert metrics["processbench_acc_correct"] == 1.0


def test_compute_processbench_f1_included_in_processbench_metrics(tmp_path: Path) -> None:
    path = tmp_path / "processbench.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "err0",
                    "generator": "stub",
                    "problem": "Error at step 1",
                    "steps": ["good step", "bad step"],
                    "label": 1,
                    "final_answer_correct": False,
                },
                {
                    "id": "ok0",
                    "generator": "stub",
                    "problem": "All correct",
                    "steps": ["good 1", "good 2"],
                    "label": -1,
                    "final_answer_correct": True,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    examples = load_processbench_examples(path)
    rows = build_processbench_prefix_records(examples)
    scores = [0.9, 0.2, 0.9, 0.9]
    metrics = compute_processbench_metrics(rows, scores)
    assert "processbench_f1" in metrics
    assert "processbench_acc_erroneous" in metrics
    assert "processbench_acc_correct" in metrics
    assert "processbench_f1_threshold" in metrics
    assert "processbench_f1_threshold_selection" in metrics
    assert "processbench_f1_is_oracle" in metrics


def test_load_processbench_examples_max_samples_preserves_all_correct_slice(tmp_path: Path) -> None:
    path = tmp_path / "processbench_ordered.json"
    payload = []
    for idx in range(8):
        payload.append(
            {
                "id": f"err_{idx}",
                "generator": "stub",
                "problem": f"Problem err {idx}",
                "steps": ["good", "bad"],
                "label": 1,
                "final_answer_correct": False,
            }
        )
    for idx in range(4):
        payload.append(
            {
                "id": f"ok_{idx}",
                "generator": "stub",
                "problem": f"Problem ok {idx}",
                "steps": ["good", "still good"],
                "label": -1,
                "final_answer_correct": True,
            }
        )
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    rows = load_processbench_examples(path, max_samples=6)

    assert len(rows) == 6
    assert sum(1 for row in rows if int(row.label) < 0) >= 1
    assert sum(1 for row in rows if int(row.label) >= 0) >= 1


def test_load_prmbench_preview_pairs_rejects_ambiguous_index_base(tmp_path: Path) -> None:
    path = tmp_path / "prmbench_preview_ambiguous.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Compute 3 + 4.",
                "original_process": ["step 1", "step 2", "step 3"],
                "modified_process": ["step 1", "wrong", "step 3"],
                "error_steps": [1],
                "classification": "confidence",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        load_prmbench_preview_pairs(path)
    except RuntimeError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("expected ambiguous PRMBench index base to raise")
