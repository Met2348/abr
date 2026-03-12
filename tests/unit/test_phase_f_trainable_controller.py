"""Unit tests for `scripts/phase_f_train_trainable_controller.py`.

中文
----
重点回归 `robust_lambda` 这条线是否真的改变梯度，而不是只改日志数字。

English
-------
These tests specifically guard that `robust_lambda` changes gradients instead
of only changing logging / bookkeeping.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / "phase_f_train_trainable_controller.py"
    spec = importlib.util.spec_from_file_location("phase_f_train_trainable_controller", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_policy_gradient_loss_uses_real_worst_generator_gradient() -> None:
    module = _load_module()

    def run_case(robust_lambda: float):
        log_a = torch.tensor(0.0, requires_grad=True)
        log_b1 = torch.tensor(0.0, requires_grad=True)
        log_b2 = torch.tensor(0.0, requires_grad=True)
        loss, debug = module._build_policy_gradient_loss(
            episode_rows=[
                {"log_prob_sum": log_a, "reward": 1.0, "generator": "gen_a"},
                {"log_prob_sum": log_b1, "reward": -1.0, "generator": "gen_b"},
                {"log_prob_sum": log_b2, "reward": -0.2, "generator": "gen_b"},
            ],
            baseline=0.0,
            robust_lambda=robust_lambda,
        )
        loss.backward()
        grads = [float(log.grad.item()) for log in (log_a, log_b1, log_b2)]
        return grads, debug

    base_grads, base_debug = run_case(robust_lambda=0.0)
    robust_grads, robust_debug = run_case(robust_lambda=2.0)

    assert base_debug["worst_generator"] is None
    assert robust_debug["worst_generator"] == "gen_b"
    assert robust_debug["worst_generator_count"] == 2
    assert robust_debug["worst_generator_mean_reward"] == pytest.approx(-0.6)

    # 非最差 generator 的 base 项保持不变。 The non-worst slice keeps the same base gradient.
    assert robust_grads[0] == pytest.approx(base_grads[0])
    # 最差 generator 的两个 episode 必须得到不同于 base 的真实梯度变化。
    # The worst-generator episodes must receive real gradient changes, not a no-op constant penalty.
    assert robust_grads[1] != pytest.approx(base_grads[1])
    assert robust_grads[2] != pytest.approx(base_grads[2])


def test_split_traces_emits_explicit_test_slice() -> None:
    module = _load_module()

    class _Trace:
        def __init__(self, example_id: str, is_all_correct: bool) -> None:
            self.example_id = example_id
            self.is_all_correct = is_all_correct

    traces = [
        _Trace("a0", True),
        _Trace("a1", True),
        _Trace("a2", True),
        _Trace("a3", True),
        _Trace("a4", False),
        _Trace("a5", False),
        _Trace("a6", False),
        _Trace("a7", False),
        _Trace("b0", True),
        _Trace("b1", True),
        _Trace("b2", True),
        _Trace("b3", True),
        _Trace("b4", False),
        _Trace("b5", False),
        _Trace("b6", False),
        _Trace("b7", False),
    ]
    generator_map = {
        **{f"a{i}": "gen_a" for i in range(8)},
        **{f"b{i}": "gen_b" for i in range(8)},
    }

    train, dev, test = module.split_traces(
        traces,
        generator_map,
        seed=42,
        dev_fraction=0.25,
        test_fraction=0.25,
    )

    assert train
    assert dev
    assert test
    seen = {id(item) for item in train} | {id(item) for item in dev} | {id(item) for item in test}
    assert len(seen) == len(traces)


def test_render_summary_md_marks_full_eval_as_non_generalization() -> None:
    module = _load_module()
    text = module.render_summary_md(
        Path("assets/artifacts/phase_f_rl_like/fake"),
        {
            "objective": "reinforce_mean_balanced",
            "selection_metric": "balanced_f1",
            "cases": [
                {
                    "case_id": "demo",
                    "num_train": 10,
                    "num_dev": 3,
                    "num_test": 3,
                    "best_dev_eval": {"metrics": {"balanced_f1": 0.6}, "worst_generator": None},
                    "test_eval": {
                        "metrics": {"balanced_f1": 0.5},
                        "efficiency": {"mean_step_fraction": 0.7},
                        "worst_generator": None,
                    },
                }
            ],
        },
    )
    assert "test_eval" in text
    assert "full_eval" in text
    assert "in-benchmark" in text
