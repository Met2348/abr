"""Unit tests for Phase E pair artifact preparation."""

from __future__ import annotations

import json
from pathlib import Path

from ours.phase_d.external_pairs_adapters import PairBuildConfig
from ours.phase_e.contracts import PhaseEPairSourceSpec
from ours.phase_e.pairs import prepare_phase_e_pair_artifact


def test_prepare_phase_e_pair_artifact_from_prmbench_preview(tmp_path: Path) -> None:
    preview_path = tmp_path / "preview.jsonl"
    preview_path.write_text(
        json.dumps(
            {
                "question": "What is 2 + 2?",
                "original_process": ["We add 2 and 2.", "The result is 4."],
                "modified_process": ["We add 2 and 2.", "The result is 5."],
                "error_steps": [2],
                "classification": "calculation",
                "idx": "row0",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    artifact = prepare_phase_e_pair_artifact(
        run_name="phase_e_test_pairs",
        output_root=tmp_path / "artifacts",
        source_specs=[
            PhaseEPairSourceSpec(
                source_id="prmbench_preview",
                source_type="prmbench_preview",
                description="test",
                default_path=str(preview_path),
            )
        ],
        build_config=PairBuildConfig(
            min_chars=4,
            max_length_ratio=10.0,
            max_token_overlap=1.0,
            max_pairs_per_sample=2,
        ),
        seed=42,
        validation_ratio=0.2,
        max_pairs_total=None,
        max_pairs_per_source=None,
        min_pair_confidence=0.0,
        resume=False,
        overwrite=False,
    )

    assert artifact.run_dir.exists()
    assert artifact.train_pairs_path.exists()
    assert artifact.validation_pairs_path.exists()
    assert artifact.summary["num_rows_after_dedup"] == 1
    assert artifact.manifest["artifact_stage"] == "phase_e_pairs_v2"
    combined_modes = {}
    for payload in (
        artifact.summary["train_summary"]["by_pair_build_mode"],
        artifact.summary["validation_summary"]["by_pair_build_mode"],
    ):
        for key, value in payload.items():
            combined_modes[key] = combined_modes.get(key, 0) + int(value)
    assert combined_modes == {"prmbench_explicit_error_step": 1}


def test_prepare_phase_e_pair_artifact_preserves_source_weight_override(tmp_path: Path) -> None:
    preview_path = tmp_path / "preview.jsonl"
    preview_path.write_text(
        json.dumps(
            {
                "question": "What is 3 + 3?",
                "original_process": ["We add 3 and 3.", "The result is 6."],
                "modified_process": ["We add 3 and 3.", "The result is 7."],
                "error_steps": [2],
                "classification": "calculation",
                "idx": "row1",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    artifact = prepare_phase_e_pair_artifact(
        run_name="phase_e_weighted_pairs",
        output_root=tmp_path / "artifacts",
        source_specs=[
            PhaseEPairSourceSpec(
                source_id="prmbench_preview",
                source_type="prmbench_preview",
                description="test",
                default_path=str(preview_path),
            )
        ],
        build_config=PairBuildConfig(
            min_chars=4,
            max_length_ratio=10.0,
            max_token_overlap=1.0,
            max_pairs_per_sample=2,
        ),
        seed=42,
        validation_ratio=0.2,
        max_pairs_total=None,
        max_pairs_per_source=None,
        min_pair_confidence=0.0,
        source_weight_overrides={"prmbench_preview": 0.25},
        resume=False,
        overwrite=False,
    )

    train_rows = artifact.train_pairs_path.read_text(encoding="utf-8").splitlines()
    validation_rows = artifact.validation_pairs_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads((train_rows + validation_rows)[0])
    assert payload["metadata"]["source_weight"] == 0.25
    assert artifact.summary["build_config"]["source_weight_overrides"] == {
        "prmbench_preview": 0.25
    }
