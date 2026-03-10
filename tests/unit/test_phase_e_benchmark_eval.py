"""Unit tests for Phase E benchmark evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from ours.phase_e.benchmark_eval import (
    build_processbench_prefix_records,
    compute_pair_ranking_metrics,
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

