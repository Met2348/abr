from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module(script_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_pair_pool_compare_flags_missing_local_semantics(tmp_path: Path) -> None:
    module = _load_module("phase_e_compare_pair_pools.py")
    pair_dir = tmp_path / "pairs"
    pair_dir.mkdir()
    _write_jsonl(
        pair_dir / "train_pairs.jsonl",
        [
            {
                "pair_id": "a",
                "source_tag": "math_step_dpo",
                "pair_confidence": 0.9,
                "metadata": {"pair_semantics": "sibling_branch"},
            },
            {
                "pair_id": "b",
                "source_tag": "math_shepherd",
                "pair_confidence": 0.95,
                "metadata": {"pair_semantics": "terminal_completion_anchor"},
            },
        ],
    )
    _write_jsonl(pair_dir / "validation_pairs.jsonl", [])

    summary = module._summarize_pair_pool(
        label="bad_pool",
        pair_dir=pair_dir,
        terminal_fraction_warn_threshold=0.1,
    )

    assert "missing_local_first_bad_edge" in summary.findings
    assert any(item.startswith("terminal_fraction_high:") for item in summary.findings)
