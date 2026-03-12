#!/usr/bin/env python3
"""Train a small stochastic controller policy on pre-scored ProcessBench traces.

这个脚本不是直接做大模型 RL，而是先在现有 `scored_rows.jsonl` 上训练一个
小型随机 controller policy，回答两个更接近 Phase F 的问题：
1. 训练式 controller 会不会自动学到类似 `threshold_only / guarded_drop` 的好规则？
2. 如果把目标改成 robust / worst-generator 导向，policy 会不会更稳？

This is an offline RL-like controller trainer on top of already-scored traces.
It uses a tiny stochastic policy and REINFORCE-style updates to test whether a
learned controller can match or improve over the heuristic families discovered in
`phase_f_controller_policy_sweep.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from phase_f_abr_lite_simulation import compute_efficiency, compute_f1_from_sims
from phase_f_controller_policy_sweep import ExampleTrace, PrefixRow, load_example_traces
from phase_f_controller_generator_robustness import load_generator_map


@dataclass
class EpisodeResult:
    predicted_erroneous: bool
    correct_decision: bool
    steps_processed: int
    total_steps: int
    stop_probability: float
    generator: str
    reward: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class LinearPolicy(nn.Module):
    """A tiny stochastic stop/continue policy.

    一个非常小的 controller policy，只吃 prefix 级数值特征。
    The policy outputs the probability of `stop/backtrack` at the current step.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features)).squeeze(-1)


def build_features(trace: ExampleTrace, idx: int) -> list[float]:
    """Build low-dimensional features for one prefix state.

    特征只用当前 score 轨迹，不引入任何新模型推理：
    1. current score
    2. previous score
    3. current drop
    4. normalized step position
    5. running min score
    6. running max score
    7. gap-to-running-max
    8. score-minus-running-min
    9. score z-ish centered around 0.5
    10. quadratic confidence distance
    """

    row = trace.rows[idx]
    score = float(row.score)
    prev_score = float(trace.rows[idx - 1].score) if idx > 0 else 1.0
    delta = prev_score - score
    prefix = trace.rows[: idx + 1]
    running_min = min(float(r.score) for r in prefix)
    running_max = max(float(r.score) for r in prefix)
    pos = float(idx + 1) / float(max(trace.num_steps, 1))
    centered = score - 0.5
    quad = centered * centered
    return [
        score,
        prev_score,
        delta,
        pos,
        running_min,
        running_max,
        running_max - score,
        score - running_min,
        centered,
        quad,
    ]


def split_traces(
    traces: list[ExampleTrace],
    generator_map: dict[str, str],
    *,
    seed: int,
    dev_fraction: float,
    test_fraction: float,
) -> tuple[list[ExampleTrace], list[ExampleTrace], list[ExampleTrace]]:
    """Stratified train/dev/test split by generator and label family.

    这里必须显式切出 test set，而不能只做 train/dev 再在整份 traces 上报主结果。
    否则 controller 的 summary 会把训练样本重新混回最终指标，导致 in-benchmark
    结果被系统性高估。

    A dedicated test split is required here.  Reporting the main metric on the
    full trace pool after only a train/dev split would silently mix training
    examples back into the "final" score and overstate controller quality.
    """

    buckets: dict[tuple[str, int], list[ExampleTrace]] = {}
    for trace in traces:
        gen = generator_map.get(trace.example_id, "unknown")
        label_group = -1 if trace.is_all_correct else 1
        buckets.setdefault((gen, label_group), []).append(trace)

    rng = random.Random(seed)
    train, dev, test = [], [], []
    for key, group in buckets.items():
        rng.shuffle(group)
        if len(group) >= 6:
            n_dev = max(1, int(round(len(group) * dev_fraction)))
            n_test = max(1, int(round(len(group) * test_fraction)))
            if n_dev + n_test >= len(group):
                n_test = max(1, len(group) // 4)
                n_dev = max(1, len(group) // 4)
        elif len(group) >= 3:
            n_dev = 1
            n_test = 1 if len(group) >= 4 else 0
        else:
            n_dev = 0
            n_test = 0
        dev.extend(group[:n_dev])
        test.extend(group[n_dev : n_dev + n_test])
        train.extend(group[n_dev + n_test :])
    if not train:
        train, dev = dev, []
    if not test and dev:
        test, dev = dev[-1:], dev[:-1]
    return train, dev, test


def compute_class_weights(traces: list[ExampleTrace]) -> dict[str, float]:
    """Compute simple class-balanced weights for all-correct vs erroneous traces.

    用来避免 `all-stop` 这种利用类别不均衡的 reward hack。
    This counteracts the easiest degenerate policy: stop on almost everything.
    """

    n_total = max(len(traces), 1)
    n_correct = max(sum(1 for trace in traces if trace.is_all_correct), 1)
    n_err = max(n_total - n_correct, 1)
    return {
        "all_correct": n_total / (2.0 * n_correct),
        "erroneous": n_total / (2.0 * n_err),
    }


def rollout_episode(
    trace: ExampleTrace,
    policy: LinearPolicy,
    generator_map: dict[str, str],
    *,
    stochastic: bool,
    reward_mode: str,
    class_weights: dict[str, float],
) -> tuple[list[torch.Tensor], EpisodeResult]:
    """Roll one trace through the current policy.

    如果 `stochastic=True`，用于 REINFORCE 训练。
    如果 `False`，用 `p>=0.5` 的 greedy 决策做评估。
    """

    log_probs: list[torch.Tensor] = []
    stop_probability = 0.0
    stopped_at = None
    last_prob = 0.0
    for idx, _row in enumerate(trace.rows):
        feats = torch.tensor(build_features(trace, idx), dtype=torch.float32).unsqueeze(0)
        prob = policy(feats)[0]
        last_prob = float(prob.detach().item())
        if stochastic:
            dist = torch.distributions.Bernoulli(probs=prob)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            do_stop = bool(action.item() >= 0.5)
        else:
            do_stop = last_prob >= 0.5
        if do_stop:
            stopped_at = idx
            stop_probability = last_prob
            break

    predicted_erroneous = stopped_at is not None
    if trace.is_all_correct:
        correct_decision = not predicted_erroneous
    else:
        correct_decision = predicted_erroneous and stopped_at <= trace.label
    steps_processed = (stopped_at + 1) if stopped_at is not None else trace.num_steps
    step_fraction = steps_processed / max(trace.num_steps, 1)

    # Reward shaping:
    # - correct all-correct continuation should be rewarded strongly
    # - correct erroneous early stop gets extra compute bonus
    # - false stop on all-correct gets strong penalty
    # - missed bad trace gets strong penalty
    if reward_mode == "balanced":
        w_correct = float(class_weights["all_correct"])
        w_err = float(class_weights["erroneous"])
        if trace.is_all_correct:
            reward = (
                +1.0 * w_correct
                if correct_decision
                else -1.4 * w_correct - 0.6 * (1.0 - step_fraction)
            )
        else:
            if correct_decision:
                reward = +1.0 * w_err + 0.35 * (1.0 - step_fraction)
            elif predicted_erroneous:
                reward = -0.7 * w_err
            else:
                reward = -1.0 * w_err
    else:
        if trace.is_all_correct:
            reward = 1.0 if correct_decision else -1.2 - 0.5 * (1.0 - step_fraction)
        else:
            if correct_decision:
                reward = 1.0 + 0.35 * (1.0 - step_fraction)
            elif predicted_erroneous:
                reward = -0.6
            else:
                reward = -1.0

    return log_probs, EpisodeResult(
        predicted_erroneous=predicted_erroneous,
        correct_decision=correct_decision,
        steps_processed=steps_processed,
        total_steps=trace.num_steps,
        stop_probability=stop_probability if predicted_erroneous else last_prob,
        generator=generator_map.get(trace.example_id, "unknown"),
        reward=reward,
    )


def evaluate_policy(
    traces: list[ExampleTrace],
    policy: LinearPolicy,
    generator_map: dict[str, str],
    *,
    reward_mode: str,
    class_weights: dict[str, float],
) -> dict[str, Any]:
    """Evaluate greedy policy on a trace set."""

    sims = []
    rewards = []
    by_gen: dict[str, list[EpisodeResult]] = {}
    for trace in traces:
        _logs, result = rollout_episode(
            trace,
            policy,
            generator_map,
            stochastic=False,
            reward_mode=reward_mode,
            class_weights=class_weights,
        )
        sims.append(
            {
                "predicted_erroneous": result.predicted_erroneous,
                "correct_decision": result.correct_decision,
                "steps_processed": result.steps_processed,
                "total_steps": result.total_steps,
            }
        )
        rewards.append(result.reward)
        by_gen.setdefault(result.generator, []).append(result)
    metrics = compute_f1_from_sims(sims, traces)
    efficiency = compute_efficiency(sims)
    gen_rows = []
    for gen, rows in sorted(by_gen.items()):
        gen_sims = [
            {
                "predicted_erroneous": r.predicted_erroneous,
                "correct_decision": r.correct_decision,
                "steps_processed": r.steps_processed,
                "total_steps": r.total_steps,
            }
            for r in rows
        ]
        gen_traces = [trace for trace in traces if generator_map.get(trace.example_id, "unknown") == gen]
        gen_rows.append(
            {
                "generator": gen,
                "metrics": compute_f1_from_sims(gen_sims, gen_traces),
                "mean_reward": sum(r.reward for r in rows) / max(len(rows), 1),
            }
        )
    worst_generator = min(gen_rows, key=lambda row: row["metrics"]["balanced_f1"]) if gen_rows else None
    return {
        "metrics": metrics,
        "efficiency": efficiency,
        "mean_reward": sum(rewards) / max(len(rewards), 1),
        "by_generator": gen_rows,
        "worst_generator": worst_generator,
    }


def _build_policy_gradient_loss(
    *,
    episode_rows: list[dict[str, Any]],
    baseline: float,
    robust_lambda: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build a differentiable REINFORCE loss from collected episodes.

    中文
    ----
    `robust_lambda` 不能只加一个常数惩罚项，否则日志里看起来像是在做
    worst-generator 优化，实际梯度却完全不变。
    这里把 robust 项变成对当前最差 generator episode 的额外 policy-gradient
    权重，这样目标变化会真实影响参数更新。

    English
    -------
    `robust_lambda` must not be implemented as a constant scalar penalty,
    otherwise the logs look "robust" while the gradients remain unchanged.
    This helper turns the robust term into an extra policy-gradient weight on
    episodes from the current worst generator slice so parameter updates change
    in the intended direction.
    """

    if not episode_rows:
        raise RuntimeError("episode_rows must not be empty")

    rewards_by_generator: dict[str, list[float]] = {}
    for row in episode_rows:
        rewards_by_generator.setdefault(str(row["generator"]), []).append(float(row["reward"]))

    worst_generator: str | None = None
    worst_generator_mean_reward = 0.0
    worst_generator_count = 0
    if robust_lambda > 0.0 and rewards_by_generator:
        worst_generator = min(
            rewards_by_generator,
            key=lambda name: sum(rewards_by_generator[name]) / max(len(rewards_by_generator[name]), 1),
        )
        worst_values = rewards_by_generator[worst_generator]
        worst_generator_mean_reward = sum(worst_values) / max(len(worst_values), 1)
        worst_generator_count = len(worst_values)

    num_episodes = len(episode_rows)
    losses: list[torch.Tensor] = []
    for row in episode_rows:
        log_prob_sum = row["log_prob_sum"]
        reward = float(row["reward"])
        generator = str(row["generator"])
        base_advantage = reward - float(baseline)
        # 主目标始终优化总体平均 reward。 Always optimize the overall mean reward.
        loss_term = -(log_prob_sum * base_advantage) / float(num_episodes)
        if robust_lambda > 0.0 and worst_generator is not None and generator == worst_generator:
            robust_advantage = reward - float(worst_generator_mean_reward)
            # 对当前最差 generator 额外加压，避免 robust 目标退化成常数项。
            # Add real gradient pressure on the current worst generator slice
            # instead of a no-op constant penalty.
            loss_term = loss_term - (
                log_prob_sum * robust_advantage * float(robust_lambda)
            ) / float(max(worst_generator_count, 1))
        losses.append(loss_term)

    loss = torch.stack(losses).sum()
    return loss, {
        "worst_generator": worst_generator,
        "worst_generator_mean_reward": float(worst_generator_mean_reward),
        "worst_generator_count": int(worst_generator_count),
    }


def train_policy(
    train_traces: list[ExampleTrace],
    dev_traces: list[ExampleTrace],
    generator_map: dict[str, str],
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    robust_lambda: float,
    selection_metric: str,
    reward_mode: str,
    init_policy: LinearPolicy | None = None,
) -> tuple[LinearPolicy, list[dict[str, Any]], dict[str, Any]]:
    """Train stochastic controller with REINFORCE-style updates.

    `robust_lambda > 0` 会额外惩罚最差 generator 的 reward 落后。
    `selection_metric`:
    - `balanced_f1`
    - `worst_generator_balanced_f1`
    """

    seed_everything(seed)
    policy = init_policy if init_policy is not None else LinearPolicy(input_dim=10, hidden_dim=hidden_dim)
    optim = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    curve = []
    best_state = None
    best_score = -1e9
    best_eval: dict[str, Any] | None = None
    baseline = 0.0
    class_weights = compute_class_weights(train_traces)

    for epoch in range(epochs):
        random.shuffle(train_traces)
        optim.zero_grad()
        rewards = []
        by_gen_rewards: dict[str, list[float]] = {}
        episode_rows: list[dict[str, Any]] = []
        for trace in train_traces:
            log_probs, result = rollout_episode(
                trace,
                policy,
                generator_map,
                stochastic=True,
                reward_mode=reward_mode,
                class_weights=class_weights,
            )
            reward = result.reward
            rewards.append(reward)
            by_gen_rewards.setdefault(result.generator, []).append(reward)
            if not log_probs:
                continue
            episode_rows.append(
                {
                    "log_prob_sum": torch.stack(log_probs).sum(),
                    "reward": float(reward),
                    "generator": str(result.generator),
                }
            )

        if not episode_rows:
            raise RuntimeError("No trainable episodes produced non-empty log_probs.")

        mean_reward = sum(rewards) / max(len(rewards), 1)
        baseline = 0.9 * baseline + 0.1 * mean_reward
        loss, loss_debug = _build_policy_gradient_loss(
            episode_rows=episode_rows,
            baseline=baseline,
            robust_lambda=robust_lambda,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        optim.step()

        train_eval = evaluate_policy(
            train_traces,
            policy,
            generator_map,
            reward_mode=reward_mode,
            class_weights=class_weights,
        )
        dev_eval = evaluate_policy(
            dev_traces,
            policy,
            generator_map,
            reward_mode=reward_mode,
            class_weights=class_weights,
        ) if dev_traces else train_eval
        if selection_metric == "worst_generator_balanced_f1":
            score = (
                dev_eval["worst_generator"]["metrics"]["balanced_f1"]
                if dev_eval["worst_generator"] is not None
                else dev_eval["metrics"]["balanced_f1"]
            )
        else:
            score = dev_eval["metrics"]["balanced_f1"]
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
            best_eval = dev_eval

        curve.append(
            {
                "epoch": epoch,
                "train_mean_reward": mean_reward,
                "train_balanced_f1": train_eval["metrics"]["balanced_f1"],
                "dev_balanced_f1": dev_eval["metrics"]["balanced_f1"],
                "dev_worst_generator_balanced_f1": (
                    dev_eval["worst_generator"]["metrics"]["balanced_f1"]
                    if dev_eval["worst_generator"] is not None
                    else None
                ),
                "train_worst_generator": loss_debug["worst_generator"],
                "train_worst_generator_mean_reward": loss_debug["worst_generator_mean_reward"],
                "loss": float(loss.detach().item()),
            }
        )

    assert best_state is not None and best_eval is not None
    policy.load_state_dict(best_state)
    return policy, curve, best_eval


def render_summary_md(run_dir: Path, summary: dict[str, Any]) -> str:
    lines = [
        "# Phase F RL-like Controller Exploration",
        "",
        f"- run_dir: `{run_dir}`",
        f"- objective: `{summary['objective']}`",
        f"- selection_metric: `{summary['selection_metric']}`",
        "",
        "## Cases",
        "",
        "| case_id | train_examples | dev_examples | test_examples | dev_balanced_f1 | test_balanced_f1 | test_worst_gen_f1 | test_step_frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        dev_worst = case["best_dev_eval"]["worst_generator"]["metrics"]["balanced_f1"] if case["best_dev_eval"]["worst_generator"] else None
        test_worst = case["test_eval"]["worst_generator"]["metrics"]["balanced_f1"] if case["test_eval"]["worst_generator"] else None
        lines.append(
            "| {case_id} | {n_train} | {n_dev} | {n_test} | {dev_bal:.4f} | {test_bal:.4f} | {test_worst} | {step_frac:.4f} |".format(
                case_id=case["case_id"],
                n_train=case["num_train"],
                n_dev=case["num_dev"],
                n_test=case["num_test"],
                dev_bal=case["best_dev_eval"]["metrics"]["balanced_f1"],
                test_bal=case["test_eval"]["metrics"]["balanced_f1"],
                test_worst=(f"{test_worst:.4f}" if test_worst is not None else "N/A"),
                step_frac=case["test_eval"]["efficiency"]["mean_step_fraction"],
            )
        )
    lines.extend([
        "",
        "## Reading Guide",
        "",
        "- 这是离线 RL-like 小策略实验，不是大模型 RL。",
        "- 主结果现在固定看 `test_eval`，避免把 train/dev 样本混回最终指标。",
        "- `full_eval` 只保留为 in-benchmark 上界参考，不能当外部泛化结果解释。",
        "- 如果 learned policy 接近或超过 heuristic baseline，说明 Phase F 可以认真考虑 trainable controller。",
        "- 如果 robust objective 的最差 generator 指标更稳，说明后续 controller RL 应显式优化 shift robustness。",
        "",
    ])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline RL-like trainable controller on scored traces.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--case", action="append", default=[], help="CASE_ID|PATH_TO_SCORED_ROWS_JSONL|PATH_TO_PROCESSBENCH_JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--robust-lambda", type=float, default=0.0)
    parser.add_argument("--selection-metric", choices=("balanced_f1", "worst_generator_balanced_f1"), default="balanced_f1")
    parser.add_argument("--reward-mode", choices=("naive", "balanced"), default="balanced")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.case:
        raise SystemExit("At least one --case CASE_ID|SCORED_ROWS|PROCESSBENCH_JSON is required.")

    objective = (("reinforce_robust" if args.robust_lambda > 0 else "reinforce_mean") + f"_{args.reward_mode}")
    run_dir = Path("assets/artifacts/phase_f_rl_like") / (
        f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    cases_out = []
    for raw_case in args.case:
        case_id, scored_rows_path, processbench_json = raw_case.split("|", 2)
        traces = load_example_traces(Path(scored_rows_path), fallback_benchmark_id=case_id)
        generator_map = load_generator_map(Path(processbench_json))
        train_traces, dev_traces, test_traces = split_traces(
            traces,
            generator_map,
            seed=args.seed,
            dev_fraction=args.dev_fraction,
            test_fraction=args.test_fraction,
        )
        policy, curve, best_dev_eval = train_policy(
            train_traces,
            dev_traces,
            generator_map,
            seed=args.seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            robust_lambda=args.robust_lambda,
            selection_metric=args.selection_metric,
            reward_mode=args.reward_mode,
        )
        test_eval = evaluate_policy(
            test_traces if test_traces else traces,
            policy,
            generator_map,
            reward_mode=args.reward_mode,
            class_weights=compute_class_weights(train_traces),
        )
        full_eval = evaluate_policy(
            traces,
            policy,
            generator_map,
            reward_mode=args.reward_mode,
            class_weights=compute_class_weights(train_traces),
        )
        torch.save(policy.state_dict(), run_dir / f"{case_id}_policy.pt")
        (run_dir / f"{case_id}_curve.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in curve),
            encoding="utf-8",
        )
        cases_out.append(
            {
                "case_id": case_id,
                "scored_rows_jsonl": scored_rows_path,
                "processbench_json": processbench_json,
                "num_train": len(train_traces),
                "num_dev": len(dev_traces),
                "num_test": len(test_traces),
                "best_dev_eval": best_dev_eval,
                "test_eval": test_eval,
                "full_eval": full_eval,
                "evaluation_scope": "benchmark_internal_train_dev_test_split",
                "full_eval_warning": "full_eval reuses training-family traces and must not be treated as external generalization.",
            }
        )
        print(
            "{case:>16} | objective={objective:<16} | dev={dev:.4f} | test={ev:.4f} | worst={worst}".format(
                case=case_id,
                objective=objective,
                dev=best_dev_eval["metrics"]["balanced_f1"],
                ev=test_eval["metrics"]["balanced_f1"],
                worst=(
                    f"{test_eval['worst_generator']['metrics']['balanced_f1']:.4f}"
                    if test_eval["worst_generator"] is not None
                    else "N/A"
                ),
            )
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "objective": objective,
        "selection_metric": args.selection_metric,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "dev_fraction": args.dev_fraction,
        "test_fraction": args.test_fraction,
        "robust_lambda": args.robust_lambda,
        "reward_mode": args.reward_mode,
        "evaluation_scope": "benchmark_internal_train_dev_test_split",
        "scope_warning": "test_eval is the main metric; full_eval is in-benchmark and should not be used as external generalization evidence.",
        "cases": cases_out,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(render_summary_md(run_dir, summary), encoding="utf-8")
    print(f"summary_json: {run_dir / 'summary.json'}")
    print(f"summary_md: {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
