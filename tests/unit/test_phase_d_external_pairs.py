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
    load_prmbench_preview_pairs,
    load_prm800k_pairs,
    load_r_prm_dpo_pairs,
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


def test_load_math_shepherd_pairs_supports_first_bad_fanout(tmp_path: Path) -> None:
    path = tmp_path / "math_shepherd_fanout.jsonl"
    path.write_text(
        json.dumps(
            {
                "input": "Toy question",
                "label": (
                    "Toy question\n"
                    "Step 1: clean setup. +\n"
                    "Step 2: still clean. +\n"
                    "Step 3: another good prefix. +\n"
                    "Step 4: first wrong step. -\n"
                    "Step 5: later wrong step. -\n"
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
            max_pairs_per_sample=4,
            step_label_pair_mode="first_bad_fanout",
        ),
    )

    assert len(rows) == 3
    assert {row.metadata["pair_build_mode"] for row in rows} == {"step_label_first_bad_fanout"}
    assert {row.metadata["pair_semantics"] for row in rows} == {"first_bad_fanout_prefix_ranking"}
    assert {row.metadata["negative_step_index"] for row in rows} == {3}
    assert {row.metadata["positive_step_index"] for row in rows} == {0, 1, 2}


def test_load_prmbench_preview_pairs_preserves_local_step_indices(tmp_path: Path) -> None:
    path = tmp_path / "prmbench_preview.jsonl"
    _write_jsonl(
        path,
        [
            {
                "idx": "toy-1",
                "question": "Toy question",
                "classification": "toy",
                "original_process": [
                    "Step 1: correct setup.",
                    "Step 2: still correct.",
                    "Step 3: final correct answer.",
                ],
                "modified_process": [
                    "Step 1: correct setup.",
                    "Step 2: now incorrect.",
                    "Step 3: still incorrect.",
                ],
                "error_steps": [2],
            }
        ],
    )

    rows = load_prmbench_preview_pairs(
        path=path,
        config=PairBuildConfig(
            min_chars=4,
            max_length_ratio=10.0,
            max_token_overlap=1.0,
            max_pairs_per_sample=2,
            prmbench_error_step_index_base="one_based",
        ),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.source_tag == "prmbench_preview"
    assert row.metadata["source_idx"] == "toy-1"
    assert row.metadata["error_step_index"] == 1
    assert row.metadata["error_step_index_base"] == 1
    assert row.metadata["positive_step_index"] == 1
    assert row.metadata["negative_step_index"] == 1
    assert row.metadata["pair_build_mode"] == "prmbench_explicit_error_step"
    assert row.metadata["pair_semantics"] == "local_modified_process_error_step"
    assert "Step 2: still correct." in row.chosen_text
    assert "Step 2: now incorrect." in row.rejected_text


def test_load_prmbench_preview_pairs_auto_detects_zero_based_indices(tmp_path: Path) -> None:
    path = tmp_path / "prmbench_preview_zero_based.jsonl"
    _write_jsonl(
        path,
        [
            {
                "idx": "toy-0",
                "question": "Toy zero-based question",
                "classification": "toy",
                "original_process": [
                    "Step 1: correct setup.",
                    "Step 2: final correct answer.",
                ],
                "modified_process": [
                    "Step 1: incorrect setup.",
                    "Step 2: still incorrect.",
                ],
                "error_steps": [0],
            }
        ],
    )

    rows = load_prmbench_preview_pairs(
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
    assert row.metadata["error_step_index"] == 0
    assert row.metadata["error_step_index_base"] == 0
    assert row.metadata["positive_step_index"] == 0
    assert row.metadata["negative_step_index"] == 0


def test_load_prmbench_preview_pairs_rejects_mixed_index_bases(tmp_path: Path) -> None:
    path = tmp_path / "prmbench_preview_mixed_base.jsonl"
    _write_jsonl(
        path,
        [
            {
                "idx": "zero",
                "question": "Zero-based row",
                "classification": "toy",
                "original_process": ["Step 1", "Step 2"],
                "modified_process": ["Wrong 1", "Wrong 2"],
                "error_steps": [0],
            },
            {
                "idx": "one",
                "question": "One-based row",
                "classification": "toy",
                "original_process": ["Step 1", "Step 2"],
                "modified_process": ["Step 1", "Wrong 2"],
                "error_steps": [2],
            },
        ],
    )

    with pytest.raises(RuntimeError, match="mixed between 0-based and 1-based"):
        load_prmbench_preview_pairs(
            path=path,
            config=PairBuildConfig(
                min_chars=4,
                max_length_ratio=10.0,
                max_token_overlap=1.0,
                max_pairs_per_sample=4,
            ),
        )


def test_load_prmbench_preview_pairs_rejects_ambiguous_auto_base(tmp_path: Path) -> None:
    path = tmp_path / "prmbench_preview_ambiguous.jsonl"
    _write_jsonl(
        path,
        [
            {
                "idx": "amb",
                "question": "Ambiguous row",
                "classification": "toy",
                "original_process": ["Step 1", "Step 2", "Step 3"],
                "modified_process": ["Step 1", "Wrong 2", "Wrong 3"],
                "error_steps": [2],
            }
        ],
    )

    with pytest.raises(RuntimeError, match="index base is ambiguous"):
        load_prmbench_preview_pairs(
            path=path,
            config=PairBuildConfig(
                min_chars=4,
                max_length_ratio=10.0,
                max_token_overlap=1.0,
                max_pairs_per_sample=4,
            ),
        )


def test_load_math_shepherd_pairs_supports_all_good_vs_all_bad(tmp_path: Path) -> None:
    path = tmp_path / "math_shepherd_fullgrid.jsonl"
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
        + "\n",
        encoding="utf-8",
    )

    rows = load_math_shepherd_pairs(
        path=path,
        config=PairBuildConfig(
            min_chars=4,
            max_length_ratio=10.0,
            max_token_overlap=1.0,
            max_pairs_per_sample=6,
            step_label_pair_mode="all_good_vs_all_bad",
        ),
    )

    seen_pairs = {
        (int(row.metadata["positive_step_index"]), int(row.metadata["negative_step_index"]))
        for row in rows
    }
    assert seen_pairs == {(1, 2), (0, 2), (1, 3), (0, 3)}
    assert {row.metadata["pair_build_mode"] for row in rows} == {"step_label_all_good_vs_all_bad"}
    assert {row.metadata["pair_semantics"] for row in rows} == {"good_bad_prefix_grid"}


def test_load_math_shepherd_pairs_supports_terminal_anchors_for_all_positive_rows(tmp_path: Path) -> None:
    path = tmp_path / "math_shepherd_terminal.jsonl"
    path.write_text(
        json.dumps(
            {
                "input": "Toy question",
                "label": (
                    "Toy question\n"
                    "Step 1: clean setup. +\n"
                    "Step 2: still clean. +\n"
                    "Step 3: nearly done. +\n"
                    "Step 4: final correct answer. +\n"
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
            max_pairs_per_sample=3,
            step_label_terminal_anchor_mode="all_positive_fanout",
        ),
    )

    assert len(rows) == 3
    assert {row.metadata["pair_build_mode"] for row in rows} == {
        "step_label_all_positive_terminal_anchor"
    }
    assert {row.metadata["pair_semantics"] for row in rows} == {
        "terminal_completion_anchor"
    }
    assert {int(row.metadata["positive_step_index"]) for row in rows} == {3}
    assert {int(row.metadata["negative_step_index"]) for row in rows} == {0, 1, 2}
    assert all("Step 4: final correct answer." in row.chosen_text for row in rows)
    assert any("Step 3: nearly done." in row.rejected_text for row in rows)


def test_terminal_anchor_mode_does_not_add_extra_rows_when_negatives_exist(tmp_path: Path) -> None:
    path = tmp_path / "math_shepherd_mixed_terminal.jsonl"
    path.write_text(
        json.dumps(
            {
                "input": "Toy question",
                "label": (
                    "Toy question\n"
                    "Step 1: clean setup. +\n"
                    "Step 2: still clean. +\n"
                    "Step 3: first wrong step. -\n"
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
            max_pairs_per_sample=3,
            step_label_terminal_anchor_mode="all_positive_fanout",
        ),
    )

    assert len(rows) == 1
    assert rows[0].metadata["pair_build_mode"] == "step_label_first_bad_edge_strict"


def test_load_math_shepherd_pairs_balances_source_cap_when_terminal_anchors_arrive_late(
    tmp_path: Path,
) -> None:
    path = tmp_path / "math_shepherd_terminal_late.jsonl"
    rows = []
    for idx in range(3):
        rows.append(
            {
                "input": f"Question {idx}",
                "label": (
                    f"Question {idx}\n"
                    "Step 1: clean setup. +\n"
                    "Step 2: first wrong step. -\n"
                ),
                "task": "GSM8K",
            }
        )
    rows.append(
        {
            "input": "Question terminal",
            "label": (
                "Question terminal\n"
                "Step 1: clean setup. +\n"
                "Step 2: still clean. +\n"
                "Step 3: final correct answer. +\n"
            ),
            "task": "GSM8K",
        }
    )
    _write_jsonl(path, rows)

    loaded = load_math_shepherd_pairs(
        path=path,
        config=PairBuildConfig(
            min_chars=4,
            max_length_ratio=10.0,
            max_token_overlap=1.0,
            max_pairs_per_sample=2,
            step_label_terminal_anchor_mode="all_positive_fanout",
        ),
        max_pairs=2,
    )

    assert len(loaded) == 2
    assert {row.metadata["pair_semantics"] for row in loaded} == {
        "local_first_bad_edge",
        "terminal_completion_anchor",
    }


def test_load_math_shepherd_pairs_respects_terminal_anchor_fraction(tmp_path: Path) -> None:
    path = tmp_path / "math_shepherd_terminal_fraction.jsonl"
    rows = []
    for idx in range(12):
        rows.append(
            {
                "input": f"Question neg {idx}",
                "label": (
                    f"Question neg {idx}\n"
                    "Step 1: clean setup. +\n"
                    "Step 2: first wrong step. -\n"
                ),
                "task": "GSM8K",
            }
        )
    for idx in range(6):
        rows.append(
            {
                "input": f"Question pos {idx}",
                "label": (
                    f"Question pos {idx}\n"
                    "Step 1: clean setup. +\n"
                    "Step 2: still clean. +\n"
                    "Step 3: final correct answer. +\n"
                ),
                "task": "GSM8K",
            }
        )
    _write_jsonl(path, rows)

    loaded = load_math_shepherd_pairs(
        path=path,
        config=PairBuildConfig(
            min_chars=4,
            max_length_ratio=10.0,
            max_token_overlap=1.0,
            max_pairs_per_sample=2,
            step_label_terminal_anchor_mode="all_positive_fanout",
            step_label_terminal_anchor_fraction=0.25,
        ),
        max_pairs=8,
    )

    counts = {}
    for row in loaded:
        key = str(row.metadata["pair_semantics"])
        counts[key] = counts.get(key, 0) + 1
    assert counts == {
        "local_first_bad_edge": 6,
        "terminal_completion_anchor": 2,
    }


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
                                    "text": "State the final answer carefully: the result is twelve.",
                                    "rating": 0,
                                },
                                {
                                    "text": "State a wrong final answer: the result is ten.",
                                    "rating": -1,
                                },
                            ],
                            "chosen_completion": 0,
                        },
                        {
                            "completions": [
                                {
                                    "text": "Repeat the correct answer briefly: twelve.",
                                    "rating": 1,
                                },
                                {
                                    "text": "Repeat the answer in a neutral paraphrase: 12.",
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
    assert len(rows) == 2
    assert all(row.source_tag == "prm800k" for row in rows)
    assert all(row.domain_tag == "general_math" for row in rows)
    assert {(row.metadata["rating_positive"], row.metadata["rating_negative"]) for row in rows} == {
        (1.0, -1.0),
        (0.0, -1.0),
    }
    assert all(row.metadata["rating_policy"] == "non_negative_positive" for row in rows)


def test_load_r_prm_dpo_pairs_compact_verdict_rewrites_long_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "r_prm"
    source_dir = root / "dpo" / "train"
    source_dir.mkdir(parents=True)
    (source_dir / "part_00001.parquet").write_bytes(b"stub")

    payload = {
        "instruction": (
            "You are an excellent math teacher.\n\n"
            "Question: What is 2 + 3?\n"
            "Previous Steps: Step 1: We need to add two numbers.\n"
            "Now Step: Step 2: The sum is 5.\n"
            "Please carefully analyze the correctness of the Now Step.\n"
            "Reply: "
        ),
        "chosen": (
            "Analysis: The arithmetic is correct.\n"
            "Verification: Is the step correct (Yes/No)? Yes"
        ),
        "rejected": (
            "Analysis: The arithmetic is wrong.\n"
            "Verification: Is the step correct (Yes/No)? No"
        ),
    }

    def fake_iter_parquet_rows(*, files, columns):
        del files, columns
        yield payload

    monkeypatch.setattr(
        "ours.phase_d.external_pairs_adapters._iter_parquet_rows",
        fake_iter_parquet_rows,
    )

    rows = load_r_prm_dpo_pairs(
        root=root,
        split="train",
        config=PairBuildConfig(
            min_chars=4,
            max_length_ratio=10.0,
            max_token_overlap=1.0,
            max_pairs_per_sample=2,
            r_prm_pair_mode="compact_verdict",
        ),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.metadata["pair_build_mode"] == "r_prm_compact_verdict_pair"
    assert row.metadata["pair_semantics"] == "same_prompt_binary_verdict"
    assert row.metadata["chosen_verdict"] == "yes"
    assert row.metadata["rejected_verdict"] == "no"
    assert "Task: Decide whether the Now Step is correct." in row.prompt_text
    assert row.chosen_text == "The Now Step is correct. Final answer: Yes.\n"
    assert row.rejected_text == "The Now Step is incorrect. Final answer: No.\n"
    assert len(row.prompt_text) < len(payload["instruction"])
    assert len(row.chosen_text) < len(payload["chosen"])
    assert len(row.rejected_text) < len(payload["rejected"])


def test_load_r_prm_dpo_pairs_legacy_mode_preserves_direct_pair_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "r_prm"
    source_dir = root / "dpo" / "validation"
    source_dir.mkdir(parents=True)
    (source_dir / "part_00001.parquet").write_bytes(b"stub")

    payload = {
        "instruction": (
            "Question: Q\n"
            "Previous Steps: Step 1: A\n"
            "Now Step: Step 2: B\n"
            "Please carefully analyze the correctness of the Now Step.\n"
            "Reply: "
        ),
        "chosen": "Long chosen analysis\nVerification: Is the step correct (Yes/No)? Yes",
        "rejected": "Long rejected analysis\nVerification: Is the step correct (Yes/No)? No",
    }

    def fake_iter_parquet_rows(*, files, columns):
        del files, columns
        yield payload

    monkeypatch.setattr(
        "ours.phase_d.external_pairs_adapters._iter_parquet_rows",
        fake_iter_parquet_rows,
    )

    rows = load_r_prm_dpo_pairs(
        root=root,
        split="validation",
        config=PairBuildConfig(
            min_chars=4,
            max_length_ratio=10.0,
            max_token_overlap=1.0,
            max_pairs_per_sample=2,
            r_prm_pair_mode="direct_pair_legacy",
        ),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.metadata["pair_build_mode"] == "r_prm_direct_pair"
    assert row.metadata["pair_semantics"] == "direct_preference_pair"
    assert row.prompt_text == payload["instruction"]
    assert row.chosen_text == payload["chosen"]
    assert row.rejected_text == payload["rejected"]
