"""Unit tests for scripts/phase_f_grpo_lite.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_f_grpo_lite.py"
    spec = importlib.util.spec_from_file_location("phase_f_grpo_lite", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_accepts_modern_trl_flags(tmp_path: Path) -> None:
    module = _load_module()
    value_run_dir = tmp_path / "value_run"
    value_run_dir.mkdir()

    args = module._build_parser().parse_args(
        [
            "--value-run-dir",
            str(value_run_dir),
            "--trl-loss-type",
            "dr_grpo",
            "--trl-scale-rewards",
            "batch",
            "--trl-beta",
            "0.04",
            "--trl-mask-truncated-completions",
            "--trl-temperature",
            "0.8",
            "--trl-use-replay-buffer",
            "--trl-replay-buffer-size",
            "96",
            "--max-steps",
            "20",
        ]
    )

    assert args.trl_loss_type == "dr_grpo"
    assert args.trl_scale_rewards == "batch"
    assert args.trl_beta == 0.04
    assert args.trl_mask_truncated_completions is True
    assert args.trl_use_replay_buffer is True
    assert args.trl_replay_buffer_size == 96
    assert args.max_steps == 20


def test_build_grpo_config_kwargs_maps_none_scale_rewards(tmp_path: Path) -> None:
    module = _load_module()
    value_run_dir = tmp_path / "value_run"
    value_run_dir.mkdir()
    args = module._build_parser().parse_args(
        [
            "--value-run-dir",
            str(value_run_dir),
            "--trl-scale-rewards",
            "none",
            "--trl-loss-type",
            "dapo",
            "--save-only-model",
            "--trl-use-vllm",
            "--trl-vllm-gpu-memory-utilization",
            "0.42",
            "--trl-epsilon-high",
            "0.28",
            "--trl-delta",
            "1.5",
            "--trl-use-replay-buffer",
            "--trl-replay-buffer-size",
            "80",
        ]
    )

    kwargs = module.build_grpo_config_kwargs(args)

    assert kwargs["scale_rewards"] is False
    assert kwargs["loss_type"] == "dapo"
    assert kwargs["save_only_model"] is True
    assert kwargs["use_vllm"] is True
    assert kwargs["vllm_gpu_memory_utilization"] == 0.42
    assert kwargs["epsilon_high"] == 0.28
    assert kwargs["delta"] == 1.5
    assert kwargs["replay_buffer_size"] == 80
