#!/usr/bin/env python3
"""Phase F GRPO feasibility analysis — offline signal study.

这个脚本不做真实的 GRPO 训练。它分析现有的 ProcessBench prefix 分数，
检验 GRPO-style 的 process reward signal 是否足够干净，能支撑 LM 级别的 RL。

This script does NOT perform real GRPO training.
Instead, it analyses existing PBR19 ProcessBench prefix scores to assess whether
GRPO-style process rewards are clean enough to support LM-level RL.

Key questions answered:
1. Group relative advantage distribution: are group advantages well-separated?
2. Reward sparsity: what fraction of steps carry meaningful signal (|adv| > threshold)?
3. Step-score variance per example: how noisy is the reward within one rollout?
4. KL-penalty simulation: at what beta does KL penalty dominate process reward?
5. Comparison with outcome reward: is process reward strictly better than outcome?

GRPO formulation recap (DeepSeekMath / arXiv:2402.03300):
  For K completions {o_1, ..., o_K} per question q:
    advantage_i = (R(q, o_i) - mean_j R(q, o_j)) / std_j R(q, o_j)
  where R is a reward (process or outcome).
  We simulate this offline using ProcessBench examples as "questions" and
  their prefix step scores as "K=1 rollout per question" (since we have one
  trajectory per example). To simulate K>1, we use bootstrap sampling.

Usage:
  python scripts/phase_f_grpo_feasibility.py \\
    --scored-rows-math assets/artifacts/phase_e_eval/pbr19_dpo_mathms_joint_math_fulleval_20260311T123421Z/scored_rows.jsonl \\
    --scored-rows-gsm assets/artifacts/phase_e_eval/pbr19_dpo_mathms_joint_gsm_fulleval_20260311T123421Z/scored_rows.jsonl \\
    --output-dir assets/artifacts/phase_f_grpo_feasibility \\
    --n-bootstrap 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StepScore:
    step_index: int
    score: float
    is_first_bad: bool
    label: int  # -1 = all-correct


@dataclass
class ExampleTrace:
    example_id: str
    label: int
    steps: list[StepScore] = field(default_factory=list)

    @property
    def is_all_correct(self) -> bool:
        return self.label == -1

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def final_outcome_reward(self) -> float:
        """Outcome reward: +1 if all-correct, -1 if erroneous."""
        return 1.0 if self.is_all_correct else -1.0

    def process_reward(self, step_index: int, *, method: str = "last_step") -> float:
        """Process reward at a given step.

        method:
          'last_step': use the score at step_index
          'cumulative': mean of scores up to step_index
          'min_score': min score seen so far (pessimistic)
        """
        scores_so_far = [s.score for s in self.steps if s.step_index <= step_index]
        if not scores_so_far:
            return 0.5  # prior
        if method == "last_step":
            return scores_so_far[-1]
        elif method == "cumulative":
            return statistics.mean(scores_so_far)
        elif method == "min_score":
            return min(scores_so_far)
        raise ValueError(method)

    def step_scores(self) -> list[float]:
        return [s.score for s in sorted(self.steps, key=lambda x: x.step_index)]


def load_traces(scored_rows_jsonl: Path) -> list[ExampleTrace]:
    by_example: dict[str, ExampleTrace] = {}
    for line in scored_rows_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        eid = row["example_id"]
        if eid not in by_example:
            by_example[eid] = ExampleTrace(example_id=eid, label=row["label"])
        by_example[eid].steps.append(
            StepScore(
                step_index=row["prefix_step_index"],
                score=row["score"],
                is_first_bad=row.get("is_first_bad_prefix", False),
                label=row["label"],
            )
        )
    traces = list(by_example.values())
    for tr in traces:
        tr.steps.sort(key=lambda s: s.step_index)
    return traces


# ---------------------------------------------------------------------------
# GRPO signal analysis
# ---------------------------------------------------------------------------


def simulate_grpo_advantages(
    traces: list[ExampleTrace],
    *,
    k_completions: int = 4,
    reward_method: str = "last_step",
    seed: int = 42,
    group_sampling: str = "mixed_pool",
) -> dict[str, Any]:
    """Simulate GRPO group relative advantages via bootstrap sampling.

    `mixed_pool` is the safer default for offline analysis here:
    1. one ProcessBench example only gives us one trace,
    2. so we do not have true multi-completion groups per question,
    3. sampling only from the same label family makes outcome reward degenerate
       to a constant and silently biases the comparison against outcome-only RL.

    `mixed_pool` 是这里更安全的离线默认值：
    1. 每个 ProcessBench 样本通常只有一条现成轨迹，
    2. 没法直接恢复同题多 completion 的真实 GRPO 分组，
    3. 如果只从同标签池采样，outcome reward 会退化成常数，导致对 outcome-only RL 的比较失真。
    """
    rng = random.Random(seed)
    erroneous = [t for t in traces if not t.is_all_correct]
    correct = [t for t in traces if t.is_all_correct]
    all_pool = list(traces)

    all_advantages: list[float] = []
    all_step_scores: list[float] = []
    step_adv_by_index: dict[int, list[float]] = defaultdict(list)
    per_example_adv_std: list[float] = []

    for trace in traces:
        if group_sampling == "label_matched":
            pool = erroneous if not trace.is_all_correct else correct
        elif group_sampling == "mixed_pool":
            pool = all_pool
        else:
            raise ValueError(f"Unsupported group_sampling: {group_sampling}")
        group_pool = [candidate for candidate in pool if candidate.example_id != trace.example_id]
        if not group_pool:
            group_pool = pool
        group_traces = [trace] + rng.choices(group_pool, k=max(k_completions - 1, 0))

        # reward at last step (process reward) or final correctness (outcome reward)
        # 用最后一步 process reward 或最终对错 outcome reward 来构造相对优势。
        group_rewards = [
            t.process_reward(t.n_steps - 1, method=reward_method) if reward_method != "outcome" else t.final_outcome_reward
            for t in group_traces
        ]
        g_mean = statistics.mean(group_rewards)
        g_std = statistics.stdev(group_rewards) if len(group_rewards) > 1 else 1.0
        if g_std < 1e-6:
            continue

        adv = (group_rewards[0] - g_mean) / (g_std + 1e-8)
        all_advantages.append(adv)

        for step in trace.steps:
            all_step_scores.append(step.score)
            step_adv_by_index[step.step_index].append(abs(adv))

        per_example_adv_std.append(abs(adv))

    adv_abs = [abs(a) for a in all_advantages]
    return {
        "group_sampling": group_sampling,
        "n_examples": len(all_advantages),
        "mean_adv": statistics.mean(all_advantages) if all_advantages else 0.0,
        "std_adv": statistics.stdev(all_advantages) if len(all_advantages) > 1 else 0.0,
        "mean_abs_adv": statistics.mean(adv_abs) if adv_abs else 0.0,
        "frac_large_adv": sum(1 for a in adv_abs if a > 0.5) / max(len(adv_abs), 1),
        "frac_small_adv": sum(1 for a in adv_abs if a < 0.1) / max(len(adv_abs), 1),
        "per_step_mean_signal": {
            k: statistics.mean(v)
            for k, v in sorted(step_adv_by_index.items())
            if v
        },
    }


def analyze_score_distribution(traces: list[ExampleTrace]) -> dict[str, Any]:
    """Analyze step score distributions by label type and step position."""
    correct_scores: list[float] = []
    erroneous_good_scores: list[float] = []  # steps before first-bad
    erroneous_bad_scores: list[float] = []   # first-bad step and after
    step_variances: list[float] = []

    for trace in traces:
        scores = trace.step_scores()
        if len(scores) > 1:
            step_variances.append(statistics.variance(scores))

        if trace.is_all_correct:
            correct_scores.extend(scores)
        else:
            for step in trace.steps:
                if step.step_index < trace.label:
                    erroneous_good_scores.append(step.score)
                else:
                    erroneous_bad_scores.append(step.score)

    def summarize(scores: list[float]) -> dict[str, Any]:
        if not scores:
            return {}
        return {
            "n": len(scores),
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "p10": sorted(scores)[len(scores) // 10],
            "p50": sorted(scores)[len(scores) // 2],
            "p90": sorted(scores)[int(len(scores) * 0.9)],
            "frac_above_08": sum(1 for s in scores if s > 0.8) / len(scores),
            "frac_below_05": sum(1 for s in scores if s < 0.5) / len(scores),
        }

    return {
        "all_correct": summarize(correct_scores),
        "erroneous_good_steps": summarize(erroneous_good_scores),
        "erroneous_bad_steps": summarize(erroneous_bad_scores),
        "mean_within_example_variance": statistics.mean(step_variances) if step_variances else 0.0,
        "separation": (
            statistics.mean(correct_scores) - statistics.mean(erroneous_bad_scores)
            if correct_scores and erroneous_bad_scores else 0.0
        ),
    }


def simulate_kl_penalty(
    traces: list[ExampleTrace],
    *,
    betas: list[float] | None = None,
    nominal_kl_per_step: float = 0.05,
) -> dict[str, Any]:
    """Simulate KL-penalty impact at different betas.

    In GRPO: final_reward = R(q, o) - beta * KL(π_θ || π_ref)
    We estimate whether the KL penalty dominates the process reward.

    Assumes nominal KL divergence per step = nominal_kl_per_step (bits/step).
    """
    if betas is None:
        betas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    # Estimate raw process reward magnitude
    all_rewards = []
    for trace in traces:
        scores = trace.step_scores()
        if scores:
            all_rewards.append(scores[-1])  # use final step score as process reward
    mean_r = statistics.mean(all_rewards) if all_rewards else 0.5
    std_r = statistics.stdev(all_rewards) if len(all_rewards) > 1 else 0.1

    mean_n_steps = statistics.mean(t.n_steps for t in traces)
    nominal_kl_total = nominal_kl_per_step * mean_n_steps

    results = []
    for beta in betas:
        kl_term = beta * nominal_kl_total
        snr = std_r / (kl_term + 1e-8)  # signal-to-noise ratio
        results.append({
            "beta": beta,
            "kl_term": kl_term,
            "signal_std": std_r,
            "snr": snr,
            "snr_ok": snr > 2.0,
        })

    return {
        "mean_process_reward": mean_r,
        "std_process_reward": std_r,
        "mean_n_steps": mean_n_steps,
        "betas": results,
        "recommended_max_beta": max((r["beta"] for r in results if r["snr_ok"]), default=0.0),
    }


def analyze_reward_discriminability(traces: list[ExampleTrace]) -> dict[str, Any]:
    """Compute how well process reward discriminates erroneous vs all-correct at each step position.

    Uses AUC-style metric at each step position.
    """
    from collections import defaultdict
    step_scores_by_correctness: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"correct": [], "erroneous": []}
    )

    for trace in traces:
        for step in trace.steps:
            key = "correct" if trace.is_all_correct else "erroneous"
            step_scores_by_correctness[step.step_index][key].append(step.score)

    step_aucs = {}
    for step_idx, groups in sorted(step_scores_by_correctness.items()):
        cors = groups["correct"]
        errs = groups["erroneous"]
        if len(cors) < 5 or len(errs) < 5:
            continue
        # AUC = P(score_correct > score_erroneous)
        n_pairs = len(cors) * len(errs)
        wins = sum(1 for c in cors for e in errs if c > e)
        auc = wins / n_pairs if n_pairs > 0 else 0.5
        step_aucs[step_idx] = {
            "auc": auc,
            "n_correct": len(cors),
            "n_erroneous": len(errs),
            "mean_correct": statistics.mean(cors),
            "mean_erroneous": statistics.mean(errs),
        }
    return step_aucs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-rows-math", type=Path, required=True)
    parser.add_argument("--scored-rows-gsm", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=8, help="K in GRPO group sampling")
    parser.add_argument(
        "--group-sampling",
        choices=("mixed_pool", "label_matched"),
        default="mixed_pool",
        help="Offline proxy for how GRPO comparison groups are formed.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    datasets: dict[str, Path] = {"math": args.scored_rows_math}
    if args.scored_rows_gsm and args.scored_rows_gsm.exists():
        datasets["gsm8k"] = args.scored_rows_gsm

    all_results: dict[str, Any] = {}

    for dataset_name, path in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name} ({path})")
        print(f"{'=' * 60}")

        traces = load_traces(path)
        n_correct = sum(1 for t in traces if t.is_all_correct)
        n_erroneous = len(traces) - n_correct
        mean_steps = statistics.mean(t.n_steps for t in traces)
        print(f"  n_examples={len(traces)}, n_correct={n_correct}, n_erroneous={n_erroneous}")
        print(f"  mean_steps_per_example={mean_steps:.2f}")
        print(f"  group_sampling={args.group_sampling}")

        # 1. Score distribution analysis
        print("\n[1] Score distribution analysis:")
        dist = analyze_score_distribution(traces)
        print(f"  all-correct step mean: {dist['all_correct'].get('mean', '?'):.3f}")
        print(f"  erroneous good-step mean: {dist['erroneous_good_steps'].get('mean', '?'):.3f}")
        print(f"  erroneous bad-step mean: {dist['erroneous_bad_steps'].get('mean', '?'):.3f}")
        print(f"  separation (correct - bad): {dist['separation']:.3f}")
        print(f"  mean within-example variance: {dist['mean_within_example_variance']:.4f}")

        # 2. Step-level discriminability (AUC per step position)
        print("\n[2] Step-level AUC (process reward discriminability):")
        step_aucs = analyze_reward_discriminability(traces)
        for step_idx, metrics in sorted(step_aucs.items())[:8]:
            print(
                f"  step={step_idx}  AUC={metrics['auc']:.3f}  "
                f"mean(cor)={metrics['mean_correct']:.3f}  "
                f"mean(err)={metrics['mean_erroneous']:.3f}  "
                f"n={metrics['n_correct']}+{metrics['n_erroneous']}"
            )

        # 3. GRPO advantage analysis (multiple reward methods)
        print("\n[3] GRPO group advantage analysis:")
        adv_results = {}
        for method in ["last_step", "cumulative", "min_score", "outcome"]:
            adv = simulate_grpo_advantages(
                traces,
                k_completions=args.n_bootstrap,
                reward_method=method,
                seed=args.seed,
                group_sampling=args.group_sampling,
            )
            adv_results[method] = adv
            print(
                f"  reward={method:15s}: "
                f"mean_abs_adv={adv['mean_abs_adv']:.3f}  "
                f"frac_large(>0.5)={adv['frac_large_adv']:.2%}  "
                f"frac_small(<0.1)={adv['frac_small_adv']:.2%}"
            )

        # 4. KL penalty simulation
        print("\n[4] KL penalty simulation (recommend max beta):")
        kl_sim = simulate_kl_penalty(traces, betas=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
        print(f"  mean_process_reward={kl_sim['mean_process_reward']:.3f}")
        print(f"  std_process_reward={kl_sim['std_process_reward']:.3f}")
        for b in kl_sim["betas"]:
            ok = "✓ OK" if b["snr_ok"] else "✗ TOO HIGH"
            print(
                f"  beta={b['beta']:.3f}: KL_term={b['kl_term']:.4f} "
                f"SNR={b['snr']:.2f} {ok}"
            )
        print(f"  → Recommended max beta: {kl_sim['recommended_max_beta']:.3f}")

        # 5. Feasibility verdict
        sep = dist["separation"]
        mean_abs_adv_last = adv_results["last_step"]["mean_abs_adv"]
        mean_abs_adv_outcome = adv_results["outcome"]["mean_abs_adv"]
        recommended_beta = kl_sim["recommended_max_beta"]

        print("\n[5] GRPO Feasibility Verdict:")
        verdict_lines = []
        if sep > 0.2:
            verdict_lines.append(f"  ✓ Score separation ({sep:.3f}) > 0.2 — process reward is discriminative")
        else:
            verdict_lines.append(f"  ✗ Score separation ({sep:.3f}) ≤ 0.2 — process reward may be too noisy")

        if mean_abs_adv_last > 0.3:
            verdict_lines.append(f"  ✓ Mean |advantage| ({mean_abs_adv_last:.3f}) > 0.3 — sufficient RL signal")
        else:
            verdict_lines.append(f"  ✗ Mean |advantage| ({mean_abs_adv_last:.3f}) ≤ 0.3 — weak RL signal")

        if recommended_beta >= 0.02:
            verdict_lines.append(f"  ✓ Recommended beta ({recommended_beta:.3f}) ≥ 0.02 — KL penalty is manageable")
        else:
            verdict_lines.append(f"  ✗ Recommended beta ({recommended_beta:.3f}) < 0.02 — KL penalty may dominate")

        all_ok = sep > 0.2 and mean_abs_adv_last > 0.3 and recommended_beta >= 0.02
        overall = "FEASIBLE ✓" if all_ok else "BORDERLINE !"
        verdict_lines.append(f"\n  Overall GRPO feasibility for {dataset_name}: {overall}")

        for line in verdict_lines:
            print(line)

        all_results[dataset_name] = {
            "n_examples": len(traces),
            "n_correct": n_correct,
            "n_erroneous": n_erroneous,
            "mean_steps": mean_steps,
            "score_distribution": dist,
            "step_aucs": step_aucs,
            "group_sampling": args.group_sampling,
            "grpo_advantages": adv_results,
            "kl_penalty": kl_sim,
            "feasibility": {
                "separation": sep,
                "mean_abs_adv_process": mean_abs_adv_last,
                "mean_abs_adv_outcome": mean_abs_adv_outcome,
                "recommended_max_beta": recommended_beta,
                "verdict": overall,
            },
        }

    # Save results
    out_file = args.output_dir / f"grpo_feasibility_{run_ts}.json"
    out_file.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved to {out_file}")

    # Summary
    print("\n" + "=" * 60)
    print("GRPO Feasibility Summary")
    print("=" * 60)
    for ds, res in all_results.items():
        f = res["feasibility"]
        print(
            f"  {ds:8s}: separation={f['separation']:.3f}  "
            f"|adv|={f['mean_abs_adv_process']:.3f}  "
            f"max_beta={f['recommended_max_beta']:.3f}  "
            f"→ {f['verdict']}"
        )

    # GRPO implementation recommendations
    print("\n=== GRPO Implementation Recommendations ===")
    math_res = all_results.get("math", {}).get("feasibility", {})
    if math_res.get("verdict", "").startswith("FEASIBLE"):
        print("  1. Reward method: 'last_step' process reward (higher |adv| than outcome)")
        print(f"  2. KL beta: start at {min(0.05, math_res.get('recommended_max_beta', 0.05)):.3f}")
        print(f"  3. Group size K: 4-8 with {args.group_sampling} bootstrap as the offline proxy")
        print("  4. PRM scorer: use the strongest verifier slice available for each domain; avoid hardcoded legacy scorer names")
        print("  5. Generator: Qwen2.5-Math-7B or Math-7B-Instruct (requires live generation harness)")
        print("  6. Training recipe: TRL GRPOTrainer or custom PPO on LoRA backbone")
    else:
        print("  Process reward signal too weak for reliable GRPO. Recommendations:")
        print("  - Use outcome reward (is solution correct?) as primary signal")
        print("  - Use PRM as reranker, not per-step gradient signal")


if __name__ == "__main__":
    main()
