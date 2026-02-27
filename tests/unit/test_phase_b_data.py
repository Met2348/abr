"""Unit tests for Phase B data contracts/loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ours.phase_b import load_phase_b_rows, summarize_rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def test_load_phase_b_rows_and_summary(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "sample_id": "strategyqa:a",
                "dataset": "strategyqa",
                "split": "train",
                "prompt_text": "Q1",
                "target_text": " yes",
                "answer": "yes",
            },
            {
                "sample_id": "strategyqa:b",
                "dataset": "strategyqa",
                "split": "train",
                "prompt_text": "Q2",
                "target_text": " no",
                "answer": "no",
            },
        ],
    )

    rows = load_phase_b_rows(path)
    assert len(rows) == 2
    summary = summarize_rows(rows)
    assert summary["num_rows"] == 2
    assert summary["dataset_counts"] == {"strategyqa": 2}
    assert summary["split_counts"] == {"train": 2}


def test_load_phase_b_rows_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "dup.jsonl"
    _write_jsonl(
        path,
        [
            {
                "sample_id": "dup:1",
                "dataset": "strategyqa",
                "split": "train",
                "prompt_text": "Q1",
                "target_text": " yes",
                "answer": "yes",
            },
            {
                "sample_id": "dup:1",
                "dataset": "strategyqa",
                "split": "train",
                "prompt_text": "Q2",
                "target_text": " no",
                "answer": "no",
            },
        ],
    )

    with pytest.raises(ValueError, match="Duplicate sample_id"):
        load_phase_b_rows(path)
