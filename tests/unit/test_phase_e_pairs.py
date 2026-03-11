"""Unit tests for Phase E pair artifact preparation."""

from __future__ import annotations

import json
from pathlib import Path

from ours.phase_d.external_pairs import ExternalPairRecord
from ours.phase_d.external_pairs_adapters import PairBuildConfig
from ours.phase_e.contracts import PhaseEPairSourceSpec
from ours.phase_e.pairs import (
    _apply_global_cap,
    _split_train_validation,
    prepare_phase_e_pair_artifact,
)


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


def test_split_train_validation_keeps_same_source_sample_together() -> None:
    rows = [
        ExternalPairRecord(
            pair_id="pair_a0",
            source_tag="prmbench_preview",
            domain_tag="general_math",
            prompt_text="Q0\n\n",
            chosen_text="good 0",
            rejected_text="bad 0",
            pair_confidence=0.9,
            metadata={"split_group_id": "sample_a"},
        ),
        ExternalPairRecord(
            pair_id="pair_a1",
            source_tag="prmbench_preview",
            domain_tag="general_math",
            prompt_text="Q0\n\n",
            chosen_text="good 1",
            rejected_text="bad 1",
            pair_confidence=0.9,
            metadata={"split_group_id": "sample_a"},
        ),
        ExternalPairRecord(
            pair_id="pair_b0",
            source_tag="prmbench_preview",
            domain_tag="general_math",
            prompt_text="Q1\n\n",
            chosen_text="good 2",
            rejected_text="bad 2",
            pair_confidence=0.9,
            metadata={"split_group_id": "sample_b"},
        ),
    ]

    train_rows, validation_rows = _split_train_validation(
        rows=rows,
        seed=42,
        validation_ratio=0.5,
        split_granularity="source_sample",
    )
    train_ids = {row.pair_id for row in train_rows}
    validation_ids = {row.pair_id for row in validation_rows}

    assert {"pair_a0", "pair_a1"} <= train_ids or {"pair_a0", "pair_a1"} <= validation_ids
    assert train_ids.isdisjoint(validation_ids)


def test_apply_global_cap_balanced_support_bucket_preserves_rare_semantics() -> None:
    rows = [
        ExternalPairRecord(
            pair_id=f"pair_local_{idx}",
            source_tag="math_shepherd",
            domain_tag="general_math",
            prompt_text="Q\n\n",
            chosen_text=f"good {idx}",
            rejected_text=f"bad {idx}",
            pair_confidence=0.9,
            metadata={"pair_semantics": "local_first_bad_edge"},
        )
        for idx in range(6)
    ] + [
        ExternalPairRecord(
            pair_id=f"pair_terminal_{idx}",
            source_tag="math_shepherd",
            domain_tag="general_math",
            prompt_text="Q\n\n",
            chosen_text=f"full {idx}",
            rejected_text=f"prefix {idx}",
            pair_confidence=0.7,
            metadata={"pair_semantics": "terminal_completion_anchor"},
        )
        for idx in range(2)
    ]

    capped_rows, cap_summary = _apply_global_cap(
        rows=rows,
        max_pairs_total=4,
        global_cap_mode="balanced_support_bucket",
    )

    kept_semantics = {
        str(row.metadata.get("pair_semantics"))
        for row in capped_rows
    }
    assert kept_semantics == {"local_first_bad_edge", "terminal_completion_anchor"}
    assert cap_summary["bucket_summary_after"] == {
        "math_shepherd|local_first_bad_edge": 2,
        "math_shepherd|terminal_completion_anchor": 2,
    }


def test_apply_global_cap_pair_id_head_keeps_legacy_prefix_of_sorted_rows() -> None:
    rows = [
        ExternalPairRecord(
            pair_id="pair_b",
            source_tag="math_shepherd",
            domain_tag="general_math",
            prompt_text="Q\n\n",
            chosen_text="good b",
            rejected_text="bad b",
            pair_confidence=0.9,
            metadata={"pair_semantics": "terminal_completion_anchor"},
        ),
        ExternalPairRecord(
            pair_id="pair_a",
            source_tag="math_shepherd",
            domain_tag="general_math",
            prompt_text="Q\n\n",
            chosen_text="good a",
            rejected_text="bad a",
            pair_confidence=0.9,
            metadata={"pair_semantics": "local_first_bad_edge"},
        ),
        ExternalPairRecord(
            pair_id="pair_c",
            source_tag="math_shepherd",
            domain_tag="general_math",
            prompt_text="Q\n\n",
            chosen_text="good c",
            rejected_text="bad c",
            pair_confidence=0.9,
            metadata={"pair_semantics": "local_first_bad_edge"},
        ),
    ]

    capped_rows, cap_summary = _apply_global_cap(
        rows=rows,
        max_pairs_total=2,
        global_cap_mode="pair_id_head",
    )

    assert [row.pair_id for row in capped_rows] == ["pair_a", "pair_b"]
    assert cap_summary["bucket_summary_after"] == {
        "math_shepherd|local_first_bad_edge": 1,
        "math_shepherd|terminal_completion_anchor": 1,
    }
