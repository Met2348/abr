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


def test_apply_profile_preserves_local_semantics_and_caps_terminal() -> None:
    module = _load_module("phase_e_curate_semantic_consensus_pairs.py")
    rows = []
    for idx in range(10):
        rows.append(
            {
                "pair_id": f"local_{idx}",
                "source_tag": "math_shepherd",
                "pair_confidence": 0.9 - idx * 0.01,
                "metadata": {"pair_semantics": "local_first_bad_edge"},
            }
        )
    for idx in range(8):
        rows.append(
            {
                "pair_id": f"sibling_{idx}",
                "source_tag": "math_step_dpo",
                "pair_confidence": 0.95 - idx * 0.01,
                "metadata": {"pair_semantics": "sibling_branch"},
            }
        )
    for idx in range(8):
        rows.append(
            {
                "pair_id": f"terminal_{idx}",
                "source_tag": "math_shepherd",
                "pair_confidence": 0.99 - idx * 0.01,
                "metadata": {"pair_semantics": "terminal_completion_anchor"},
            }
        )

    curated, summary = module._apply_profile(
        rows,
        profile="semantic_consensus_v1",
        max_terminal_fraction=0.12,
    )

    counts = summary["selection_counts_after_terminal_cap"]
    assert counts["local_first_bad_edge"] >= 1
    assert counts["sibling_branch"] >= 1
    assert counts["terminal_completion_anchor"] < summary["selection_counts_before_terminal_cap"]["terminal_completion_anchor"]
    payload = json.dumps(curated)
    assert "local_first_bad_edge" in payload
