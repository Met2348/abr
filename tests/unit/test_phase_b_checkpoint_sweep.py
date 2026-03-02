"""Unit tests for `scripts/phase_b_checkpoint_sweep.py` helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_phase_b_checkpoint_sweep_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_b_checkpoint_sweep.py"
    spec = importlib.util.spec_from_file_location("phase_b_checkpoint_sweep", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_checkpoint_targets_uses_all_and_final(tmp_path: Path) -> None:
    module = _load_phase_b_checkpoint_sweep_module()
    run_dir = tmp_path / "phase_b_run"
    (run_dir / "checkpoints" / "checkpoint-100").mkdir(parents=True)
    (run_dir / "checkpoints" / "checkpoint-300").mkdir(parents=True)

    targets = module._resolve_checkpoint_targets(run_dir, "")
    assert targets == [
        ("100", run_dir / "checkpoints" / "checkpoint-100"),
        ("300", run_dir / "checkpoints" / "checkpoint-300"),
        ("final", None),
    ]


def test_resolve_checkpoint_targets_honors_requested_labels(tmp_path: Path) -> None:
    module = _load_phase_b_checkpoint_sweep_module()
    run_dir = tmp_path / "phase_b_run"
    (run_dir / "checkpoints" / "checkpoint-100").mkdir(parents=True)
    (run_dir / "checkpoints" / "checkpoint-200").mkdir(parents=True)

    targets = module._resolve_checkpoint_targets(run_dir, "200")
    assert targets == [
        ("200", run_dir / "checkpoints" / "checkpoint-200"),
        ("final", None),
    ]
