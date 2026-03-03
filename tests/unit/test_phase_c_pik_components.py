"""Unit tests for Phase C P(IK) contracts and loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ours.phase_b.pik_data import (
    PIKTargetRecord,
    assert_phase_c_pik_compatibility,
    load_phase_c_pik_manifest,
    load_pik_supervision_examples,
    summarize_pik_targets,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _pik_manifest(*, model_path: str = "m", adapter_path: str | None = None, dtype: str = "bfloat16") -> dict:
    return {
        "artifact_stage": "phase_c_pik_c1",
        "run_name": "unit",
        "rollout_config": {
            "model_path": model_path,
            "adapter_path": adapter_path,
            "dtype": dtype,
        },
    }


def _make_pik_dir(root: Path, name: str) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "manifest.json", _pik_manifest())
    _write_jsonl(
        run_dir / "pik_targets.jsonl",
        [
            {
                "sample_id": "s0",
                "dataset": "strategyqa",
                "split": "train",
                "question": "Q0",
                "prompt_text": "Question: Q0\nAnswer:",
                "answer": "yes",
                "k_rollouts": 4,
                "n_correct": 1,
                "n_parse_error": 0,
                "success_rate": 0.25,
                "parseable_rate": 1.0,
                "mean_generated_char_count": 90.0,
                "metadata": {},
            },
            {
                "sample_id": "s1",
                "dataset": "strategyqa",
                "split": "train",
                "question": "Q1",
                "prompt_text": "Question: Q1\nAnswer:",
                "answer": "no",
                "k_rollouts": 4,
                "n_correct": 3,
                "n_parse_error": 0,
                "success_rate": 0.75,
                "parseable_rate": 1.0,
                "mean_generated_char_count": 110.0,
                "metadata": {},
            },
        ],
    )
    return run_dir


def test_load_pik_supervision_examples(tmp_path: Path) -> None:
    run_dir = _make_pik_dir(tmp_path, "pik")
    examples, manifest = load_pik_supervision_examples(run_dir)

    assert manifest["artifact_stage"] == "phase_c_pik_c1"
    assert len(examples) == 2
    assert examples[0].sample_id == "s0"
    assert examples[0].target_success_rate == pytest.approx(0.25)
    assert examples[1].model_input_text() == "Question: Q1\nAnswer:"


def test_phase_c_pik_compatibility_rejects_backbone_mismatch() -> None:
    left = _pik_manifest(model_path="base_a")
    right = _pik_manifest(model_path="base_b")
    with pytest.raises(ValueError, match="mismatch on model_path"):
        assert_phase_c_pik_compatibility(left, right)


def test_load_phase_c_pik_manifest_rejects_wrong_stage(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "manifest.json", {"artifact_stage": "not_phase_c_pik"})
    with pytest.raises(ValueError, match=r"not a Phase C P\(IK\) C1 artifact dir"):
        load_phase_c_pik_manifest(run_dir)


def test_summarize_pik_targets() -> None:
    targets = [
        PIKTargetRecord(
            sample_id="a",
            dataset="strategyqa",
            split="train",
            question="Q",
            prompt_text="Question: Q\nAnswer:",
            answer="yes",
            k_rollouts=4,
            n_correct=1,
            n_parse_error=0,
            success_rate=0.25,
            parseable_rate=1.0,
            mean_generated_char_count=100.0,
        ),
        PIKTargetRecord(
            sample_id="b",
            dataset="strategyqa",
            split="train",
            question="Q2",
            prompt_text="Question: Q2\nAnswer:",
            answer="no",
            k_rollouts=4,
            n_correct=3,
            n_parse_error=0,
            success_rate=0.75,
            parseable_rate=1.0,
            mean_generated_char_count=120.0,
        ),
    ]
    summary = summarize_pik_targets(targets)
    assert summary["num_questions"] == 2
    assert summary["mean_success_rate"] == pytest.approx(0.5)
    assert summary["mean_parseable_rate"] == pytest.approx(1.0)
