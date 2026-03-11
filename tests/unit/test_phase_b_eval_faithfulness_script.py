"""Unit tests for helper logic in `scripts/phase_b_eval_faithfulness.py`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_b_eval_faithfulness.py"
    spec = importlib.util.spec_from_file_location("phase_b_eval_faithfulness", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_requested_checkpoint_path_prefers_best(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    best_path = run_dir / "best_value_head.pt"
    final_path = run_dir / "final_value_head.pt"
    best_path.write_text("best\n", encoding="utf-8")
    final_path.write_text("final\n", encoding="utf-8")

    resolved = module._resolve_requested_checkpoint_path(
        value_run_dir=run_dir,
        checkpoint_name="best",
        checkpoint_missing_policy="fail",
    )

    assert resolved["requested"] == "best"
    assert resolved["resolved"] == "best"
    assert resolved["fallback_to_final"] is False
    assert resolved["path"] == str(best_path)


def test_resolve_requested_checkpoint_path_fails_by_default_when_best_missing(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    final_path = run_dir / "final_value_head.pt"
    final_path.write_text("final\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        module._resolve_requested_checkpoint_path(
            value_run_dir=run_dir,
            checkpoint_name="best",
            checkpoint_missing_policy="fail",
        )


def test_resolve_requested_checkpoint_path_records_best_to_final_fallback_when_explicit(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    final_path = run_dir / "final_value_head.pt"
    final_path.write_text("final\n", encoding="utf-8")

    resolved = module._resolve_requested_checkpoint_path(
        value_run_dir=run_dir,
        checkpoint_name="best",
        checkpoint_missing_policy="fallback_final",
    )

    assert resolved["requested"] == "best"
    assert resolved["resolved"] == "final"
    assert resolved["fallback_to_final"] is True
    assert resolved["path"] == str(final_path)


def test_resolve_dtype_accepts_short_aliases() -> None:
    module = _load_module()
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        float32="fp32",
        float16="fp16",
        bfloat16="bf16",
    )

    assert module._resolve_dtype("fp32", torch_stub) == "fp32"
    assert module._resolve_dtype("fp16", torch_stub) == "fp16"
    assert module._resolve_dtype("bf16", torch_stub) == "bf16"
