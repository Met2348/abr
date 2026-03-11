"""Unit tests for scripts/phase_e_prepare_prmbench_terminal_anchor_pairs.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_e_prepare_prmbench_terminal_anchor_pairs.py"
    spec = importlib.util.spec_from_file_location(
        "phase_e_prepare_prmbench_terminal_anchor_pairs",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_accepts_terminal_anchor_ratio(tmp_path: Path) -> None:
    module = _load_script_module()
    input_path = tmp_path / "prmbench.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "question": "Q",
                "original_process": ["s0", "s1"],
                "modified_process": ["m0", "m1"],
                "error_steps": [2],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    args = module.parse_args(
        [
            "--input-jsonl",
            str(input_path),
            "--terminal-anchor-ratio",
            "0.25",
        ]
    )

    assert args.terminal_anchor_ratio == 0.25
