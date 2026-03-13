"""Unit tests for repository-safe direct CLI defaults.

These tests protect against a recurring failure mode in this repo:
wrappers get hardened, but direct script entrypoints quietly keep legacy
defaults and can reintroduce already-audited research pitfalls.
"""

from __future__ import annotations

import inspect
import importlib.util
import sys
from pathlib import Path

import pytest

from ours.phase_b.corruptions import CorruptionBuildConfig
from ours.phase_d.external_pairs_adapters import PairBuildConfig
from ours.phase_e.pairs import prepare_phase_e_pair_artifact


def _load_script_module(script_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase_b_prepare_value_defaults_to_cqr_balanced(tmp_path: Path) -> None:
    module = _load_script_module("phase_b_prepare_value_data.py")
    input_jsonl = tmp_path / "train.jsonl"
    input_jsonl.write_text("", encoding="utf-8")

    args = module.parse_args(["--input-jsonl", str(input_jsonl)])

    assert args.corruption_selection_policy == "cqr_balanced"
    assert CorruptionBuildConfig().selection_policy == "cqr_balanced"


def test_phase_e_prepare_pairs_defaults_to_safe_split_and_cap() -> None:
    module = _load_script_module("phase_e_prepare_pairs.py")

    args = module.parse_args(["--source-bundle", "prmbench_preview"])

    assert args.split_granularity == "source_sample"
    assert args.global_cap_mode == "balanced_support_bucket"
    assert args.r_prm_pair_mode == "compact_verdict"
    signature = inspect.signature(prepare_phase_e_pair_artifact)
    assert signature.parameters["split_granularity"].default == "source_sample"
    assert signature.parameters["global_cap_mode"].default == "balanced_support_bucket"


def test_phase_e_eval_benchmark_defaults_to_fixed_half_threshold(tmp_path: Path) -> None:
    module = _load_script_module("phase_e_eval_benchmark.py")
    value_run_dir = tmp_path / "run"
    value_run_dir.mkdir()

    args = module.parse_args(
        [
            "--value-run-dir",
            str(value_run_dir),
            "--benchmark-id",
            "processbench_math",
        ]
    )

    assert args.processbench_f1_threshold_policy == "fixed"
    assert args.processbench_f1_threshold == 0.5


def test_phase_e_eval_benchmark_rejects_threshold_with_oracle_policy(tmp_path: Path) -> None:
    module = _load_script_module("phase_e_eval_benchmark.py")
    value_run_dir = tmp_path / "run"
    value_run_dir.mkdir()

    with pytest.raises(
        ValueError,
        match="--processbench-f1-threshold must be omitted when --processbench-f1-threshold-policy=oracle_sweep",
    ):
        module.parse_args(
            [
                "--value-run-dir",
                str(value_run_dir),
                "--benchmark-id",
                "processbench_math",
                "--processbench-f1-threshold-policy",
                "oracle_sweep",
                "--processbench-f1-threshold",
                "0.4",
            ]
        )


def test_phase_d_prepare_external_pairs_defaults_to_compact_rprm(tmp_path: Path) -> None:
    module = _load_script_module("phase_d_prepare_external_pairs.py")
    math_shepherd_path = tmp_path / "math_shepherd.jsonl"
    math_shepherd_path.write_text("", encoding="utf-8")

    args = module.parse_args(["--math-shepherd-path", str(math_shepherd_path)])

    assert args.split_granularity == "source_sample"
    assert args.r_prm_pair_mode == "compact_verdict"
    assert PairBuildConfig().r_prm_pair_mode == "compact_verdict"
