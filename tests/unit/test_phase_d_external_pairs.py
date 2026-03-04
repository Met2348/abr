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
        ),
    ]
    stats = summarize_external_pairs(rows)
    assert stats["num_pairs"] == 2
    assert stats["by_source"] == {"r_prm": 2}
    assert stats["by_domain"] == {"general_math": 2}
    assert abs(float(stats["mean_pair_confidence"]) - 0.8) < 1e-8
