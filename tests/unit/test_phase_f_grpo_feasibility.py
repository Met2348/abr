from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from phase_f_grpo_feasibility import ExampleTrace, StepScore, simulate_grpo_advantages


def _build_trace(example_id: str, label: int, scores: list[float]) -> ExampleTrace:
    steps = [
        StepScore(
            step_index=idx,
            score=score,
            is_first_bad=(label != -1 and idx == label),
            label=label,
        )
        for idx, score in enumerate(scores)
    ]
    return ExampleTrace(example_id=example_id, label=label, steps=steps)


def test_mixed_pool_outcome_advantage_is_not_forced_to_zero() -> None:
    traces = [
        _build_trace("c1", -1, [0.9, 0.88]),
        _build_trace("c2", -1, [0.86, 0.84]),
        _build_trace("e1", 1, [0.7, 0.2]),
        _build_trace("e2", 1, [0.72, 0.25]),
    ]

    result = simulate_grpo_advantages(
        traces,
        k_completions=4,
        reward_method="outcome",
        seed=0,
        group_sampling="mixed_pool",
    )

    assert result["group_sampling"] == "mixed_pool"
    assert result["n_examples"] > 0
    assert result["mean_abs_adv"] > 0.0


def test_label_matched_outcome_advantage_remains_degenerate() -> None:
    traces = [
        _build_trace("c1", -1, [0.9, 0.88]),
        _build_trace("c2", -1, [0.86, 0.84]),
        _build_trace("e1", 1, [0.7, 0.2]),
        _build_trace("e2", 1, [0.72, 0.25]),
    ]

    result = simulate_grpo_advantages(
        traces,
        k_completions=4,
        reward_method="outcome",
        seed=0,
        group_sampling="label_matched",
    )

    assert result["group_sampling"] == "label_matched"
    assert result["n_examples"] == 0
    assert result["mean_abs_adv"] == 0.0
