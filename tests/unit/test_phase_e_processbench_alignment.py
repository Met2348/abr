"""Unit tests for ProcessBench alignment diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from ours.phase_e.processbench_alignment import (
    compute_alignment_distances,
    summarize_pair_jsonl_alignment,
    summarize_processbench_topology,
    summarize_scored_rows_alignment,
)


def test_summarize_pair_jsonl_alignment_tracks_pair_types(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "pair_id": "p0",
                        "source_tag": "math_shepherd",
                        "domain_tag": "general_math",
                        "prompt_text": "Q\n\n",
                        "chosen_text": "good0\n",
                        "rejected_text": "good0\ngood1\nbad2\n",
                        "pair_confidence": 0.7,
                        "quality_flags": {},
                        "metadata": {
                            "positive_step_index": 0,
                            "negative_step_index": 2,
                            "first_negative_index": 2,
                            "step_gap": 2,
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "pair_id": "p1",
                        "source_tag": "math_shepherd",
                        "domain_tag": "general_math",
                        "prompt_text": "Q\n\n",
                        "chosen_text": "good0\ngood1\n",
                        "rejected_text": "good0\ngood1\nbad2\nlater3\n",
                        "pair_confidence": 0.7,
                        "quality_flags": {},
                        "metadata": {
                            "positive_step_index": 1,
                            "negative_step_index": 3,
                            "first_negative_index": 2,
                            "step_gap": 2,
                        },
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_pair_jsonl_alignment(path)

    assert summary["num_pairs"] == 2
    assert summary["num_pairs_with_step_metadata"] == 2
    assert summary["pair_type_counts"] == {
        "earlygood_vs_firstbad": 1,
        "lastsafe_vs_laterbad": 1,
    }
    assert summary["gap_bucket_counts"] == {"gap2": 2}


def test_summarize_scored_rows_alignment_slices_processbench_pairs(tmp_path: Path) -> None:
    processbench_path = tmp_path / "processbench.json"
    processbench_path.write_text(
        json.dumps(
            [
                {
                    "id": "ex0",
                    "generator": "stub",
                    "problem": "Problem A",
                    "steps": ["good0", "good1", "bad2", "bad3"],
                    "label": 2,
                    "final_answer_correct": False,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    scored_rows_path = tmp_path / "scored_rows.jsonl"
    scored_rows_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "row_id": "ex0:prefix:0",
                        "example_id": "ex0",
                        "prefix_step_index": 0,
                        "label": 2,
                        "is_good_prefix": True,
                        "is_first_bad_prefix": False,
                        "score": 0.9,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "row_id": "ex0:prefix:1",
                        "example_id": "ex0",
                        "prefix_step_index": 1,
                        "label": 2,
                        "is_good_prefix": True,
                        "is_first_bad_prefix": False,
                        "score": 0.8,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "row_id": "ex0:prefix:2",
                        "example_id": "ex0",
                        "prefix_step_index": 2,
                        "label": 2,
                        "is_good_prefix": False,
                        "is_first_bad_prefix": True,
                        "score": 0.3,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "row_id": "ex0:prefix:3",
                        "example_id": "ex0",
                        "prefix_step_index": 3,
                        "label": 2,
                        "is_good_prefix": False,
                        "is_first_bad_prefix": False,
                        "score": 0.85,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_scored_rows_alignment(
        scored_rows_path=scored_rows_path,
        processbench_path=processbench_path,
    )

    assert summary["pair_accuracy_good_vs_bad"] == 0.75
    assert summary["first_error_edge_accuracy"] == 1.0
    assert summary["gap_bucket_metrics"]["gap1"]["accuracy"] == 1.0
    assert summary["gap_bucket_metrics"]["gap2"]["accuracy"] == 0.5
    assert summary["aggregate_metrics"]["anygood_vs_firstbad"]["accuracy"] == 1.0
    assert summary["aggregate_metrics"]["good_vs_laterbad"]["accuracy"] == 0.5


def test_compute_alignment_distances_compares_training_and_benchmark_distributions(tmp_path: Path) -> None:
    pair_path = tmp_path / "pairs.jsonl"
    pair_path.write_text(
        json.dumps(
            {
                "pair_id": "p0",
                "source_tag": "math_shepherd",
                "domain_tag": "general_math",
                "prompt_text": "Q\n\n",
                "chosen_text": "good0\ngood1\n",
                "rejected_text": "good0\ngood1\nbad2\n",
                "pair_confidence": 0.7,
                "quality_flags": {},
                "metadata": {
                    "positive_step_index": 1,
                    "negative_step_index": 2,
                    "first_negative_index": 2,
                    "step_gap": 1,
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    processbench_path = tmp_path / "processbench.json"
    processbench_path.write_text(
        json.dumps(
            [
                {
                    "id": "ex0",
                    "generator": "stub",
                    "problem": "Problem A",
                    "steps": ["good0", "good1", "bad2", "bad3"],
                    "label": 2,
                    "final_answer_correct": False,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pair_summary = summarize_pair_jsonl_alignment(pair_path)
    benchmark_summary = summarize_processbench_topology(processbench_path)
    distance = compute_alignment_distances(
        pair_summary=pair_summary,
        benchmark_summary=benchmark_summary,
    )

    assert distance["pair_type_l1_distance"] > 0.0
    assert distance["gap_bucket_l1_distance"] > 0.0
