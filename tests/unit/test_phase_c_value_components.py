"""Unit tests for Phase C C2 value-head data/metrics components."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ours.phase_b.faithfulness_eval import (
    compute_binary_auc,
    compute_calibration_summary,
    compute_corruption_summary,
)
from ours.phase_b.value_data import (
    assert_phase_c_compatibility,
    load_corruption_variants,
    load_phase_c_manifest,
    load_value_supervision_examples,
)
from ours.phase_b.value_head import (
    ValueHeadConfig,
    SigmoidValueHead,
    load_value_head_checkpoint,
    save_value_head_checkpoint,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _phase_c_manifest(*, model_path: str = "m", adapter_path: str | None = None, dtype: str = "bfloat16") -> dict:
    return {
        "artifact_stage": "phase_c_c0_c1",
        "run_name": "unit",
        "step_config_signature": "step_sig",
        "prefix_config_signature": "prefix_sig",
        "rollout_config": {
            "model_path": model_path,
            "adapter_path": adapter_path,
            "dtype": dtype,
        },
    }


def _make_phase_c_dir(root: Path, name: str) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "manifest.json", _phase_c_manifest())
    _write_jsonl(
        run_dir / "prefixes.jsonl",
        [
            {
                "prefix_id": "p0",
                "sample_id": "s0",
                "dataset": "strategyqa",
                "split": "train",
                "question": "Q0",
                "prompt_text": "Question: Q0\nAnswer:",
                "prefix_target_text": "",
                "current_step_role": "question_only",
                "current_step_id": "q0",
                "prefix_step_index": 0,
                "num_reasoning_steps_seen": 0,
                "num_reasoning_steps_total": 2,
                "metadata": {},
            },
            {
                "prefix_id": "p1",
                "sample_id": "s1",
                "dataset": "strategyqa",
                "split": "train",
                "question": "Q1",
                "prompt_text": "Question: Q1\nAnswer:",
                "prefix_target_text": "Reasoning step\n",
                "current_step_role": "reasoning",
                "current_step_id": "r1",
                "prefix_step_index": 1,
                "num_reasoning_steps_seen": 1,
                "num_reasoning_steps_total": 2,
                "metadata": {},
            },
        ],
    )
    _write_jsonl(
        run_dir / "rollout_targets.jsonl",
        [
            {
                "prefix_id": "p0",
                "success_rate": 0.25,
                "parseable_rate": 1.0,
                "k_rollouts": 4,
                "mean_generated_char_count": 100.0,
                "metadata": {},
            },
            {
                "prefix_id": "p1",
                "success_rate": 0.75,
                "parseable_rate": 1.0,
                "k_rollouts": 4,
                "mean_generated_char_count": 120.0,
                "metadata": {},
            },
        ],
    )
    _write_jsonl(
        run_dir / "corruptions.jsonl",
        [
            {
                "corruption_id": "c1",
                "clean_prefix_id": "p1",
                "sample_id": "s1",
                "dataset": "strategyqa",
                "split": "train",
                "corruption_type": "numeric_perturb",
                "corrupted_prefix_text": "Reasoning step (wrong)\n",
                "original_step_text": "Reasoning step",
                "corrupted_step_text": "Reasoning step (wrong)",
                "corruption_step_index": 1,
                "metadata": {},
            }
        ],
    )
    return run_dir


def test_load_value_supervision_examples_and_corruptions(tmp_path: Path) -> None:
    run_dir = _make_phase_c_dir(tmp_path, "phase_c")
    examples, manifest = load_value_supervision_examples(run_dir, require_corruptions=True)
    variants, _ = load_corruption_variants(run_dir)

    assert manifest["artifact_stage"] == "phase_c_c0_c1"
    assert len(examples) == 2
    assert len(variants) == 1
    assert examples[0].prefix_id == "p0"
    assert examples[1].primary_corruption_type == "numeric_perturb"
    assert variants[0].clean_prefix_id == "p1"


def test_phase_c_compatibility_check_rejects_rollout_mismatch(tmp_path: Path) -> None:
    left = _phase_c_manifest(model_path="base-a")
    right = _phase_c_manifest(model_path="base-b")
    with pytest.raises(ValueError, match="mismatch on model_path"):
        assert_phase_c_compatibility(left, right)


def test_load_phase_c_manifest_rejects_wrong_stage(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "manifest.json", {"artifact_stage": "not_phase_c"})
    with pytest.raises(ValueError, match="not a Phase C C0/C1 artifact dir"):
        load_phase_c_manifest(run_dir)


def test_calibration_and_corruption_metrics_basic() -> None:
    calibration = compute_calibration_summary(
        [0.1, 0.9, 0.8, 0.2],
        [0.0, 1.0, 1.0, 0.0],
        reference_mean=0.5,
    )
    assert calibration["n"] == 4
    assert calibration["brier_improvement_vs_baseline"] > 0.0

    corruption = compute_corruption_summary(
        [0.8, 0.7, 0.6],
        [0.2, 0.3, 0.4],
        corruption_types=["numeric_perturb", "step_drop", "numeric_perturb"],
        corruption_step_indices=[1, 2, 1],
    )
    assert corruption["n_pairs"] == 3
    assert corruption["pair_accuracy"] == 1.0
    assert corruption["auc_clean_vs_corrupt"] >= 0.9


def test_compute_binary_auc_degenerate_labels() -> None:
    assert compute_binary_auc([0.1, 0.2], [1, 1]) == 0.5
    assert compute_binary_auc([0.1, 0.2], [0, 0]) == 0.5


def test_value_head_checkpoint_round_trip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    config = ValueHeadConfig(hidden_size=8, dropout_prob=0.0)
    head = SigmoidValueHead(config)
    with torch.no_grad():
        # Make state non-default so round-trip checks something real.
        for param in head.parameters():
            param.add_(0.123)

    ckpt_path = tmp_path / "value_head.pt"
    save_value_head_checkpoint(
        ckpt_path,
        value_head=head,
        config=config,
        extra_state={"tag": "unit"},
    )
    loaded_head, loaded_cfg, extra = load_value_head_checkpoint(ckpt_path)

    assert loaded_cfg.hidden_size == 8
    assert extra["tag"] == "unit"
    for left, right in zip(head.state_dict().values(), loaded_head.state_dict().values(), strict=True):
        assert torch.allclose(left, right)

