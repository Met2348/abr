"""Unit tests for helper logic in `scripts/phase_d_eval_external_pairs.py`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_d_eval_external_pairs.py"
    spec = importlib.util.spec_from_file_location("phase_d_eval_external_pairs", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_checkpoint_path_fails_by_default_when_best_missing(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    final_path = run_dir / "final_value_head.pt"
    final_path.write_text("final\n", encoding="utf-8")
    run_manifest = {
        "output_files": {
            "best_value_head": str(run_dir / "missing_best_value_head.pt"),
            "final_value_head": str(final_path),
        }
    }

    with pytest.raises(FileNotFoundError):
        module._resolve_checkpoint_path(
            value_run_dir=run_dir,
            run_manifest=run_manifest,
            checkpoint_name="best",
            checkpoint_missing_policy="fail",
        )


def test_resolve_checkpoint_path_can_explicitly_fallback(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    final_path = run_dir / "final_value_head.pt"
    final_path.write_text("final\n", encoding="utf-8")
    run_manifest = {
        "output_files": {
            "best_value_head": str(run_dir / "missing_best_value_head.pt"),
            "final_value_head": str(final_path),
        }
    }

    resolved = module._resolve_checkpoint_path(
        value_run_dir=run_dir,
        run_manifest=run_manifest,
        checkpoint_name="best",
        checkpoint_missing_policy="fallback_final",
    )

    assert resolved["requested_checkpoint_name"] == "best"
    assert resolved["resolved_checkpoint_name"] == "final"
    assert resolved["fallback_to_final"] is True
    assert resolved["resolved_checkpoint_path"] == str(final_path)
