"""Unit tests for Phase D external pair schema and loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ours.phase_d.external_pairs import (
    ExternalPairRecord,
    load_external_pair_jsonl,
    summarize_external_pairs,
)
from ours.phase_d.external_pairs_adapters import (
    PairBuildConfig,
    load_math_shepherd_pairs,
    load_prm800k_pairs,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_external_pair_record_validation_rejects_identical_text() -> None:
    row = ExternalPairRecord(
        pair_id="p0",
        source_tag="r_prm",
        domain_tag="general_math",
        prompt_text="Question: 1+1=?\nAnswer:",
        chosen_text="2",
        rejected_text="2",
        pair_confidence=0.9,
    )
    with pytest.raises(ValueError):
        row.validate()


def test_load_external_pair_jsonl_filters_by_confidence_source_domain(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            {
                "pair_id": "a",
                "source_tag": "r_prm",
                "domain_tag": "general_math",
                "prompt_text": "Q1\n",
                "chosen_text": "good",
                "rejected_text": "bad",
                "pair_confidence": 0.9,
                "quality_flags": {"ok": True},
                "metadata": {"x": 1},
            },
            {
                "pair_id": "b",
                "source_tag": "math_shepherd",
                "domain_tag": "gsm8k_math",
                "prompt_text": "Q2\n",
                "chosen_text": "better",
                "rejected_text": "worse",
                "pair_confidence": 0.5,
                "quality_flags": {},
                "metadata": {},
            },
            {
                "pair_id": "c",
                "source_tag": "prmbench_preview",
                "domain_tag": "general_math",
                "prompt_text": "Q3\n",
                "chosen_text": "yes",
                "rejected_text": "no",
                "pair_confidence": 0.8,
                "quality_flags": {},
                "metadata": {},
            },
        ],
    )

    rows, stats = load_external_pair_jsonl(
        path,
        min_confidence=0.75,
        allowed_sources={"r_prm", "prmbench_preview"},
        allowed_domains={"general_math"},
    )

    assert [row.pair_id for row in rows] == ["a", "c"]
    assert stats["num_pairs"] == 2
    assert stats["by_source"] == {"prmbench_preview": 1, "r_prm": 1}
    assert stats["by_domain"] == {"general_math": 2}


def test_summarize_external_pairs_counts_sources_and_confidence() -> None:
    rows = [
        ExternalPairRecord(
            pair_id="1",
            source_tag="r_prm",
            domain_tag="general_math",
            prompt_text="Q\n",
            chosen_text="good",
            rejected_text="bad",
            pair_confidence=0.7,
        ),
        ExternalPairRecord(
            pair_id="2",
            source_tag="r_prm",
            domain_tag="general_math",
            prompt_text="Q2\n",
            chosen_text="good2",
            rejected_text="bad2",
            pair_confidence=0.9,
            metadata={"pair_build_mode": "r_prm_direct_pair", "pair_semantics": "direct_preference_pair"},
        ),
    ]
    stats = summarize_external_pairs(rows)
    assert stats["num_pairs"] == 2
    assert stats["by_source"] == {"r_prm": 2}
    assert stats["by_domain"] == {"general_math": 2}
    assert stats["by_pair_build_mode"] == {"r_prm_direct_pair": 1, "unspecified": 1}
    assert stats["by_pair_semantics"] == {"direct_preference_pair": 1, "unspecified": 1}
    assert abs(float(stats["mean_pair_confidence"]) - 0.8) < 1e-8


def test_load_math_shepherd_pairs_uses_first_bad_edge_strict_by_default(tmp_path: Path) -> None:
    path = tmp_path / "math_shepherd.jsonl"
    path.write_text(
        json.dumps(
            {
                "input": "Toy question",
                "label": (
                    "Toy question\n"
                    "Step 1: clean setup. +\n"
                    "Step 2: still clean. +\n"
                    "Step 3: first wrong step. -\n"
                    "Step 4: later wrong step. -\n"
                ),
                "task": "GSM8K",
            },
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps(
            {
                "input": "All wrong question",
                "label": (
                    "All wrong question\n"
                    "Step 1: already wrong. -\n"
                    "Step 2: still wrong. -\n"
                ),
                "task": "GSM8K",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_math_shepherd_pairs(
        path=path,
        config=PairBuildConfig(
            min_chars=4,
            max_length_ratio=10.0,
            max_token_overlap=1.0,
            max_pairs_per_sample=2,
        ),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.metadata["pair_build_mode"] == "step_label_first_bad_edge_strict"
    assert row.metadata["pair_semantics"] == "local_first_bad_edge"
    assert row.metadata["positive_step_index"] == 1
    assert row.metadata["negative_step_index"] == 2
    assert "Step 2: still clean." in row.chosen_text
    assert "Step 3: first wrong step." in row.rejected_text
    assert "Step 4: later wrong step." not in row.rejected_text


def test_load_prm800k_pairs_supports_official_step_completion_schema(tmp_path: Path) -> None:
    path = tmp_path / "prm800k_like.jsonl"
    _write_jsonl(
        path,
        [
            {
                "question": {"problem": "What is 7 + 5?"},
                "label": {
                    "steps": [
                        {
                            "completions": [
                                {
                                    "text": "Compute the sum carefully: seven plus five equals twelve.",
                                    "rating": 1,
                                },
                                {
                                    "text": "Use an incorrect operation and claim seven plus five equals thirty-five.",
                                    "rating": -1,
                                },
                            ],
                            "chosen_completion": 0,
                        },
                        {
                            "completions": [
                                {
                                    "text": "State the final answer: the result is twelve.",
                                    "rating": 1,
                                },
                                {
                                    "text": "State a wrong final answer: the result is ten.",
                                    "rating": 0,
                                },
                            ],
                            "chosen_completion": 0,
                        },
                    ]
                },
            }
        ],
    )

    rows = load_prm800k_pairs(
        path=path,
        config=PairBuildConfig(
            min_chars=12,
            max_length_ratio=8.0,
            max_token_overlap=1.0,
            max_pairs_per_sample=2,
        ),
    )
    assert len(rows) >= 1
    assert all(row.source_tag == "prm800k" for row in rows)
    assert all(row.domain_tag == "general_math" for row in rows)
