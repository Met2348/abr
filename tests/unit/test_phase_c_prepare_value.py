"""Unit tests for Phase C C0/C1 prefix and corruption preparation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ours.phase_b.contracts import PhaseBTrainRow
from ours.phase_b.corruptions import (
    CorruptionBuildConfig,
    build_corruptions_for_prefixes,
)
from ours.phase_b.value_targets import (
    PrefixBuildConfig,
    RolloutTargetRecord,
    build_prefix_artifacts,
    build_step_sequence_from_phase_b_row,
)
from ours.data.step_builder import StepBuildConfig


def _load_prepare_value_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_b_prepare_value_data.py"
    spec = importlib.util.spec_from_file_location("phase_b_prepare_value_data", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_step_sequence_from_phase_b_row_requires_reasoning_by_default() -> None:
    row = PhaseBTrainRow(
        sample_id="strategyqa:no_reasoning",
        dataset="strategyqa",
        split="train",
        prompt_text="Question: Is the sky blue?\nAnswer:",
        target_text="yes",
        answer="yes",
        question="Is the sky blue?",
    )

    with pytest.raises(ValueError, match="requires at least one reasoning step"):
        build_step_sequence_from_phase_b_row(row)


def test_build_prefix_artifacts_are_deterministic_and_include_question_prefix() -> None:
    row = PhaseBTrainRow(
        sample_id="strategyqa:1",
        dataset="strategyqa",
        split="train",
        prompt_text="Question: Is the sky blue?\nAnswer:",
        target_text="The sky appears blue during the day.\nFinal answer: yes",
        answer="yes",
        question="Is the sky blue?",
    )
    step_cfg = StepBuildConfig()
    prefix_cfg = PrefixBuildConfig()

    sequence, meta = build_step_sequence_from_phase_b_row(
        row,
        step_config=step_cfg,
        prefix_config=prefix_cfg,
    )
    prefixes_a = build_prefix_artifacts(
        row=row,
        step_sequence=sequence,
        build_meta=meta,
        prefix_config=prefix_cfg,
    )
    prefixes_b = build_prefix_artifacts(
        row=row,
        step_sequence=sequence,
        build_meta=meta,
        prefix_config=prefix_cfg,
    )

    assert len(prefixes_a) == 2
    assert [p.prefix_id for p in prefixes_a] == [p.prefix_id for p in prefixes_b]
    assert prefixes_a[0].num_reasoning_steps_seen == 0
    assert prefixes_a[0].prefix_target_text == ""
    assert prefixes_a[1].num_reasoning_steps_seen == 1
    assert "The sky appears blue during the day." in prefixes_a[1].prefix_target_text
    assert prefixes_a[1].rollout_input_text() == (
        row.prompt_text + prefixes_a[1].prefix_target_text
    )


def test_build_corruptions_for_prefixes_changes_the_prefix_text() -> None:
    row = PhaseBTrainRow(
        sample_id="gsm8k:1",
        dataset="gsm8k",
        split="train",
        prompt_text="Question: 2+3?\nAnswer:",
        target_text="2 + 3 = 5\nFinal answer: 5",
        answer="5",
        question="2+3?",
    )
    sequence, meta = build_step_sequence_from_phase_b_row(
        row,
        step_config=StepBuildConfig(),
        prefix_config=PrefixBuildConfig(),
    )
    prefixes = build_prefix_artifacts(
        row=row,
        step_sequence=sequence,
        build_meta=meta,
        prefix_config=PrefixBuildConfig(),
    )
    reasoning_prefix = prefixes[-1]

    corruptions = build_corruptions_for_prefixes(
        [reasoning_prefix],
        config=CorruptionBuildConfig(max_corruptions_per_prefix=2),
    )
    assert corruptions
    assert any(
        artifact.corrupted_prefix_text != reasoning_prefix.prefix_target_text
        for artifact in corruptions
    )


def test_build_corruptions_cqr_balanced_reduces_step_drop_dominance() -> None:
    """CQR mode should keep semantic/non-drop variants when they exist."""
    row = PhaseBTrainRow(
        sample_id="strategyqa:cqr1",
        dataset="strategyqa",
        split="train",
        prompt_text="Question: Is Julius Caesar before Augustus?\nAnswer:",
        target_text=(
            "If Julius Caesar is not before Augustus, then Julius Caesar is greater than Augustus.\n"
            "Final answer: no"
        ),
        answer="no",
        question="Is Julius Caesar before Augustus?",
    )
    sequence, meta = build_step_sequence_from_phase_b_row(
        row,
        step_config=StepBuildConfig(),
        prefix_config=PrefixBuildConfig(),
    )
    prefixes = build_prefix_artifacts(
        row=row,
        step_sequence=sequence,
        build_meta=meta,
        prefix_config=PrefixBuildConfig(),
    )
    reasoning_prefix = prefixes[-1]
    corruptions = build_corruptions_for_prefixes(
        [reasoning_prefix],
        config=CorruptionBuildConfig(
            max_corruptions_per_prefix=4,
            selection_policy="cqr_balanced",
            min_non_step_drop_per_prefix=1,
            max_step_drop_per_prefix=1,
        ),
    )
    assert corruptions
    step_drop_count = sum(1 for item in corruptions if item.corruption_type == "step_drop")
    assert step_drop_count <= 1
    assert any(item.corruption_type != "step_drop" for item in corruptions)
    assert any(
        item.corruption_type in {"negation_flip", "comparator_flip", "condition_reversal", "entity_substitution"}
        for item in corruptions
    )


def test_prepare_value_script_builds_artifacts_without_rollouts(tmp_path: Path) -> None:
    module = _load_prepare_value_module()
    input_path = tmp_path / "train.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "sample_id": "strategyqa:10",
                "dataset": "strategyqa",
                "split": "train",
                "prompt_text": "Question: Is water wet?\nAnswer:",
                "target_text": "Water is described as wet in common usage.\nFinal answer: yes",
                "answer": "yes",
                "question": "Is water wet?",
            }
        ],
    )
    output_root = tmp_path / "phase_c_data"

    exit_code = module.main(
        [
            "--input-jsonl",
            str(input_path),
            "--output-root",
            str(output_root),
            "--run-name",
            "smoke",
            "--max-samples",
            "1",
            "--build-corruptions",
            "--no-build-rollouts",
        ]
    )
    assert exit_code == 0

    run_dirs = list((output_root / "strategyqa").glob("smoke__*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "step_sequences.jsonl").exists()
    assert (run_dir / "prefixes.jsonl").exists()
    assert (run_dir / "corruptions.jsonl").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.json").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_step_sequences"] == 1
    assert summary["num_prefixes"] == 2


def test_rollout_config_to_dict_is_json_serializable() -> None:
    module = _load_prepare_value_module()
    config = module.RolloutConfig(
        model_path="assets/models/Qwen2.5-7B-Instruct",
        adapter_path=None,
        batch_size=64,
        rollout_count=4,
        max_new_tokens=96,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        seed=42,
        dtype="bfloat16",
        device_map="auto",
        require_cuda=True,
        oom_backoff=True,
        log_every=25,
    )

    payload = config.to_dict()
    assert payload["model_path"] == "assets/models/Qwen2.5-7B-Instruct"
    assert payload["batch_size"] == 64


def test_select_uncertain_prefix_ids_uses_band_or_ci_width() -> None:
    module = _load_prepare_value_module()
    targets = [
        RolloutTargetRecord(
            prefix_id="p_certain",
            sample_id="s0",
            dataset="strategyqa",
            split="train",
            k_rollouts=8,
            n_correct=8,
            n_parse_error=0,
            success_rate=1.0,
            parseable_rate=1.0,
            q_mean_smoothed=0.95,
            q_std_error=0.01,
            q_ci_low=0.93,
            q_ci_high=0.97,
            q_ci_width=0.04,
            q_weight=0.9,
            mean_generated_char_count=50.0,
            metadata={},
        ),
        RolloutTargetRecord(
            prefix_id="p_band",
            sample_id="s1",
            dataset="strategyqa",
            split="train",
            k_rollouts=8,
            n_correct=4,
            n_parse_error=0,
            success_rate=0.5,
            parseable_rate=1.0,
            q_mean_smoothed=0.52,
            q_std_error=0.05,
            q_ci_low=0.42,
            q_ci_high=0.62,
            q_ci_width=0.20,
            q_weight=0.8,
            mean_generated_char_count=50.0,
            metadata={},
        ),
        RolloutTargetRecord(
            prefix_id="p_ci",
            sample_id="s2",
            dataset="strategyqa",
            split="train",
            k_rollouts=8,
            n_correct=7,
            n_parse_error=0,
            success_rate=0.875,
            parseable_rate=1.0,
            q_mean_smoothed=0.88,
            q_std_error=0.12,
            q_ci_low=0.64,
            q_ci_high=1.0,
            q_ci_width=0.36,
            q_weight=0.4,
            mean_generated_char_count=50.0,
            metadata={},
        ),
    ]
    picked = module._select_uncertain_prefix_ids(  # noqa: SLF001
        targets=targets,
        band=0.2,
        ci_width_threshold=0.3,
    )
    assert "p_band" in picked
    assert "p_ci" in picked
    assert "p_certain" not in picked
