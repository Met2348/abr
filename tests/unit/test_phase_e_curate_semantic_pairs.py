"""Unit tests for scripts/phase_e_curate_semantic_pairs.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_e_curate_semantic_pairs.py"
    spec = importlib.util.spec_from_file_location("phase_e_curate_semantic_pairs", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_artifact(artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [
        {
            "pair_id": "m0",
            "source_tag": "math_shepherd",
            "domain_tag": "general_math",
            "prompt_text": "Q\n\n",
            "chosen_text": "good",
            "rejected_text": "bad",
            "pair_confidence": 0.9,
            "quality_flags": {},
            "metadata": {"pair_semantics": "first_bad_fanout_prefix_ranking"},
        },
        {
            "pair_id": "p0",
            "source_tag": "prmbench_preview",
            "domain_tag": "general_math",
            "prompt_text": "Q\n\n",
            "chosen_text": "good2",
            "rejected_text": "bad2",
            "pair_confidence": 0.9,
            "quality_flags": {},
            "metadata": {"pair_semantics": "terminal_completion_anchor"},
        },
    ]
    val_rows = [
        {
            "pair_id": "v0",
            "source_tag": "math_shepherd",
            "domain_tag": "general_math",
            "prompt_text": "Q\n\n",
            "chosen_text": "good3",
            "rejected_text": "bad3",
            "pair_confidence": 0.9,
            "quality_flags": {},
            "metadata": {"pair_semantics": "first_bad_fanout_prefix_ranking"},
        }
    ]
    (artifact_dir / "train_pairs.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in train_rows) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "validation_pairs.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in val_rows) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "summary.json").write_text(
        json.dumps({"train_summary": {"by_pair_semantics": {"first_bad_fanout_prefix_ranking": 1}}}),
        encoding="utf-8",
    )


def test_parse_slice_spec_supports_semantic_and_source_filters(tmp_path: Path) -> None:
    module = _load_module()
    artifact_dir = tmp_path / "artifact"
    _write_artifact(artifact_dir)
    spec = module._parse_slice_spec(
        f"ms_local={artifact_dir}|first_bad_fanout_prefix_ranking|math_shepherd|16|4"
    )
    assert spec.label == "ms_local"
    assert spec.artifact_dir == artifact_dir
    assert spec.semantic_filters == ("first_bad_fanout_prefix_ranking",)
    assert spec.source_filters == ("math_shepherd",)
    assert spec.train_cap == 16
    assert spec.val_cap == 4


def test_main_filters_and_tags_rows(tmp_path: Path) -> None:
    module = _load_module()
    artifact_dir = tmp_path / "artifact"
    output_root = tmp_path / "out"
    _write_artifact(artifact_dir)

    exit_code = module.main(
        [
            "--slice",
            f"ms_local={artifact_dir}|first_bad_fanout_prefix_ranking|math_shepherd|1|1",
            "--run-name",
            "toy_curated",
            "--output-root",
            str(output_root),
            "--seed",
            "7",
        ]
    )
    assert exit_code == 0

    run_dirs = sorted(output_root.glob("toy_curated__*"))
    assert len(run_dirs) == 1
    train_rows = [
        json.loads(line)
        for line in (run_dirs[0] / "train_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(train_rows) == 1
    assert train_rows[0]["source_tag"] == "math_shepherd"
    assert train_rows[0]["metadata"]["artifact_mix_source_label"] == "ms_local"
    assert train_rows[0]["metadata"]["curated_semantic_filters"] == [
        "first_bad_fanout_prefix_ranking"
    ]
