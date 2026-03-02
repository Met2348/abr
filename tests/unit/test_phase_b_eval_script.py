"""Unit tests for the Phase B evaluation bridge script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_phase_b_eval_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_b_eval.py"
    spec = importlib.util.spec_from_file_location("phase_b_eval", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_eval_model_paths_for_direct_model_arg() -> None:
    module = _load_phase_b_eval_module()
    args = module.parse_args(
        [
            "--input-jsonl",
            "validation.jsonl",
            "--model-path",
            "assets/models/Qwen2.5-7B-Instruct",
        ]
    )

    model_path, adapter_path = module._resolve_eval_model_paths(args)
    assert model_path == "assets/models/Qwen2.5-7B-Instruct"
    assert adapter_path is None


def test_resolve_eval_model_paths_for_peft_run_dir(tmp_path: Path) -> None:
    module = _load_phase_b_eval_module()
    run_dir = tmp_path / "phase_b_run"
    final_model = run_dir / "final_model"
    final_model.mkdir(parents=True)
    (final_model / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "assets/models/Qwen2.5-7B-Instruct"}),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_path": "assets/models/Qwen2.5-7B-Instruct",
                "effective_training_mode": "peft",
            }
        ),
        encoding="utf-8",
    )

    args = module.parse_args(
        [
            "--input-jsonl",
            "validation.jsonl",
            "--phase-b-run-dir",
            str(run_dir),
        ]
    )
    model_path, adapter_path = module._resolve_eval_model_paths(args)
    assert model_path == "assets/models/Qwen2.5-7B-Instruct"
    assert adapter_path == final_model


def test_resolve_eval_model_paths_rejects_dual_inputs(tmp_path: Path) -> None:
    module = _load_phase_b_eval_module()
    run_dir = tmp_path / "phase_b_run"
    run_dir.mkdir()
    args = module.parse_args(
        [
            "--input-jsonl",
            "validation.jsonl",
            "--model-path",
            "assets/models/Qwen2.5-7B-Instruct",
            "--phase-b-run-dir",
            str(run_dir),
        ]
    )

    with pytest.raises(ValueError, match="either --model-path or --phase-b-run-dir"):
        module._resolve_eval_model_paths(args)
