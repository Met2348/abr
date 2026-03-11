#!/usr/bin/env python3
"""Phase F F2: ABR-lite deterministic controller offline simulation.

Uses already-scored ProcessBench prefix records (from phase_e_eval_benchmark.py)
to simulate ABR-lite controller decisions without running any new inference.

The controller rule:
  - Walk through prefix scores step by step.
  - If V(prefix_t) < threshold tau, predict "error at step t" and stop.
  - If no step triggers the stop condition, predict "all correct".

Key outputs:
  1. F1 (same as existing benchmark eval, as a sanity check).
  2. Compute efficiency: avg fraction of steps processed before controller stops.
  3. Comparison: value-guided controller vs fixed-schedule (stop after K steps).
  4. Threshold sensitivity analysis across tau values.
  5. Controller action histogram.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PrefixRow:
    example_id: str
    step_index: int
    label: int          # first-error step index; -1 = all-correct
    is_good: bool
    is_first_bad: bool
    score: float


@dataclass
class ExampleSim:
    example_id: str
    label: int          # -1 = all-correct, k >= 0 = first bad step index
    steps: list[PrefixRow] = field(default_factory=list)

    @property
    def is_all_correct(self) -> bool:
        return self.label == -1

    @property
    def num_steps(self) -> int:
        return len(self.steps)


# ---------------------------------------------------------------------------
# Controller simulation
# ---------------------------------------------------------------------------


def simulate_abr_lite(
    example: ExampleSim,
    *,
    tau: float,
    delta_drop: float = 0.15,
    max_backtracks: int = 3,
) -> dict[str, Any]:
    """Simulate the ABR-lite stop/continue controller on one example.

    Returns a dict with keys:
      stopped_at_step (int or None): step where controller triggered stop
      predicted_erroneous (bool): controller predicts error
      correct_decision (bool): controller was right
      steps_processed (int): how many steps were processed
      reason (str): 'threshold' | 'delta_drop' | 'end_of_trace'
    """
    prev_score = 1.0
    stopped_at = None
    reason = "end_of_trace"

    for row in example.steps:
        # Check absolute threshold
        if row.score < tau:
            stopped_at = row.step_index
            reason = "threshold"
            break
        # Check relative drop
        if prev_score - row.score > delta_drop:
            stopped_at = row.step_index
            reason = "delta_drop"
            break
        prev_score = row.score

    predicted_erroneous = stopped_at is not None
    steps_processed = (stopped_at + 1) if stopped_at is not None else len(example.steps)

    # Was the decision correct?
    if example.is_all_correct:
        correct_decision = not predicted_erroneous
    else:
        # We want to correctly flag the error (at or before the actual first-error step)
        correct_decision = (
            predicted_erroneous and stopped_at <= example.label
        )

    return {
        "stopped_at": stopped_at,
        "predicted_erroneous": predicted_erroneous,
        "correct_decision": correct_decision,
        "steps_processed": steps_processed,
        "total_steps": example.num_steps,
        "reason": reason,
    }


def simulate_fixed_schedule(
    example: ExampleSim,
    *,
    stop_after_k: int,
) -> dict[str, Any]:
    """Simulate a fixed-schedule controller: stop after K steps and flag error."""
    if len(example.steps) <= stop_after_k:
        # Full trace processed
        predicted_erroneous = False
        steps_processed = len(example.steps)
    else:
        predicted_erroneous = True
        steps_processed = stop_after_k + 1

    if example.is_all_correct:
        correct_decision = not predicted_erroneous
    else:
        correct_decision = predicted_erroneous and stop_after_k <= example.label

    return {
        "predicted_erroneous": predicted_erroneous,
        "correct_decision": correct_decision,
        "steps_processed": steps_processed,
        "total_steps": example.num_steps,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_f1_from_sims(
    sims: list[dict[str, Any]],
    examples: list[ExampleSim],
) -> dict[str, float]:
    """Compute controller metrics from simulation results.

    这里同时返回两种常见但容易混淆的 F1 定义，避免后续文档误读。
    This returns two different F1-style summaries because they answer
    slightly different questions and are easy to conflate in reports.

    - `balanced_f1`: 用 `acc_erroneous` 和 `acc_correct` 做调和平均，更像控制器视角的
      “错误样本能否拦下 + 正确样本能否放行”的平衡分数。
      Harmonic mean of `acc_erroneous` and `acc_correct`.
    - `positive_f1`: 标准二分类正类 F1，把“erroneous”视为正类。
      Standard positive-class F1 with "erroneous" as the positive label.
    """
    tp = tn = fp = fn = 0
    for sim, ex in zip(sims, examples):
        pred_err = sim["predicted_erroneous"]
        true_err = not ex.is_all_correct
        if pred_err and true_err:
            tp += 1
        elif not pred_err and not true_err:
            tn += 1
        elif pred_err and not true_err:
            fp += 1
        else:
            fn += 1

    acc_err = tp / max(tp + fn, 1)
    acc_cor = tn / max(tn + fp, 1)
    balanced_f1 = (2 * acc_err * acc_cor) / max(acc_err + acc_cor, 1e-12)
    precision_err = tp / max(tp + fp, 1)
    positive_f1 = (2 * precision_err * acc_err) / max(precision_err + acc_err, 1e-12)
    accuracy = (tp + tn) / max(len(sims), 1)

    return {
        "f1": balanced_f1,
        "balanced_f1": balanced_f1,
        "positive_f1": positive_f1,
        "precision_erroneous": precision_err,
        "acc_erroneous": acc_err,
        "acc_correct": acc_cor,
        "accuracy": accuracy,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def compute_efficiency(sims: list[dict[str, Any]]) -> dict[str, float]:
    """Compute average compute efficiency (fraction of steps processed)."""
    fractions = []
    for sim in sims:
        total = sim.get("total_steps", 1)
        processed = sim.get("steps_processed", total)
        fractions.append(processed / max(total, 1))

    return {
        "mean_step_fraction": statistics.mean(fractions),
        "median_step_fraction": statistics.median(fractions),
        "stopped_early_rate": sum(1 for f in fractions if f < 1.0) / len(fractions),
    }


def render_summary_markdown(
    *,
    scored_rows_path: Path,
    tau: float,
    delta_drop: float,
    n_examples: int,
    n_all_correct: int,
    n_erroneous: int,
    abr_metrics: dict[str, Any],
    abr_efficiency: dict[str, Any],
    fixed_rows: list[dict[str, Any]],
) -> str:
    """Render a compact markdown summary for artifact inspection.

    生成供人工审计的 markdown 摘要，避免只剩原始 JSON 导致口径漂移。
    Build a human-readable markdown summary so audits do not depend on
    re-parsing ad hoc console output.
    """
    best_fixed = max(fixed_rows, key=lambda row: row["metrics"]["balanced_f1"])
    lines = [
        "# Phase F ABR-lite Simulation Summary",
        "",
        f"- scored_rows_jsonl: `{scored_rows_path}`",
        f"- tau: `{tau:.3f}`",
        f"- delta_drop: `{delta_drop:.3f}`",
        f"- n_examples: `{n_examples}`",
        f"- n_all_correct: `{n_all_correct}`",
        f"- n_erroneous: `{n_erroneous}`",
        "",
        "## ABR-lite Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| balanced_f1 | {abr_metrics['balanced_f1']:.4f} |",
        f"| positive_f1 | {abr_metrics['positive_f1']:.4f} |",
        f"| precision_erroneous | {abr_metrics['precision_erroneous']:.4f} |",
        f"| acc_erroneous | {abr_metrics['acc_erroneous']:.4f} |",
        f"| acc_correct | {abr_metrics['acc_correct']:.4f} |",
        f"| accuracy | {abr_metrics['accuracy']:.4f} |",
        f"| stopped_early_rate | {abr_efficiency['stopped_early_rate']:.4f} |",
        f"| mean_step_fraction | {abr_efficiency['mean_step_fraction']:.4f} |",
        "",
        "## Fixed-Schedule Comparison",
        "",
        "| schedule | balanced_f1 | positive_f1 | acc_erroneous | acc_correct | step_fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in fixed_rows:
        lines.append(
            "| stop@{k} | {bf1:.4f} | {pf1:.4f} | {ae:.4f} | {ac:.4f} | {sf:.4f} |".format(
                k=row["stop_after_k"],
                bf1=row["metrics"]["balanced_f1"],
                pf1=row["metrics"]["positive_f1"],
                ae=row["metrics"]["acc_erroneous"],
                ac=row["metrics"]["acc_correct"],
                sf=row["efficiency"]["mean_step_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `balanced_f1` 是控制器视角的平衡分数，强调“拦错”和“放对”同时成立。",
            "- `positive_f1` 是标准二分类正类 F1，把 erroneous 当作正类。",
            "- 如果 `acc_erroneous` 很高但 `acc_correct` 很低，控制器会过度早停，不适合作为 RL 前置门。",
            "- 当前最强固定 schedule："
            f" `stop@{best_fixed['stop_after_k']}`"
            f" (`balanced_f1={best_fixed['metrics']['balanced_f1']:.4f}`"
            f", `step_fraction={best_fixed['efficiency']['mean_step_fraction']:.4f}`)",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Simulate ABR-lite controller on pre-scored ProcessBench prefix rows."
    )
    p.add_argument(
        "--scored-rows-jsonl",
        required=True,
        type=Path,
        help="scored_rows.jsonl from phase_e_eval_benchmark.py",
    )
    p.add_argument(
        "--tau",
        type=float,
        default=0.38,
        help="ABR-lite threshold: stop if score < tau (default 0.38 for PBR19+MATH)",
    )
    p.add_argument(
        "--delta-drop",
        type=float,
        default=0.15,
        help="Relative drop trigger: stop if prev_score - score > delta_drop",
    )
    p.add_argument(
        "--tau-sweep",
        action="store_true",
        help="Sweep tau from 0.1 to 0.9 and show F1 at each value",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write results (default: prints to stdout)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # Load scored rows
    rows = [json.loads(line) for line in args.scored_rows_jsonl.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(rows)} scored prefix rows")

    # Group by example
    by_example: dict[str, ExampleSim] = {}
    for r in rows:
        eid = r["example_id"]
        if eid not in by_example:
            by_example[eid] = ExampleSim(example_id=eid, label=r["label"])
        ex = by_example[eid]
        ex.steps.append(PrefixRow(
            example_id=eid,
            step_index=r["prefix_step_index"],
            label=r["label"],
            is_good=r["is_good_prefix"],
            is_first_bad=r["is_first_bad_prefix"],
            score=r["score"],
        ))

    # Sort steps within each example
    for ex in by_example.values():
        ex.steps.sort(key=lambda r: r.step_index)

    examples = list(by_example.values())
    n_all_correct = sum(1 for ex in examples if ex.is_all_correct)
    n_erroneous = len(examples) - n_all_correct
    print(f"Examples: {len(examples)} total | {n_all_correct} all-correct | {n_erroneous} erroneous")
    print()

    # --- Tau sweep ---
    if args.tau_sweep:
        print("=" * 80)
        print("Tau sweep: F1 vs threshold")
        print("=" * 80)
        print(f"{'tau':>6} | {'F1':>7} | {'Acc_err':>8} | {'Acc_cor':>8} | {'stop_rate':>9} | {'step_frac':>9}")
        print("-" * 65)
        best_f1, best_tau = 0.0, 0.38
        for tau_v in [i / 100 for i in range(10, 91, 2)]:
            sims = [simulate_abr_lite(ex, tau=tau_v, delta_drop=args.delta_drop) for ex in examples]
            m = compute_f1_from_sims(sims, examples)
            eff = compute_efficiency(sims)
            print(
                f"{tau_v:>6.2f} | {m['f1']:>7.4f} | {m['acc_erroneous']:>8.4f} | "
                f"{m['acc_correct']:>8.4f} | {eff['stopped_early_rate']:>9.4f} | "
                f"{eff['mean_step_fraction']:>9.4f}"
            )
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_tau = tau_v
        print(f"\nBest tau: {best_tau:.2f} -> F1={best_f1:.4f}")
        print()

    # --- ABR-lite at specified tau ---
    tau = args.tau
    print("=" * 80)
    print(f"ABR-lite controller (tau={tau:.2f}, delta_drop={args.delta_drop:.2f})")
    print("=" * 80)
    sims = [simulate_abr_lite(ex, tau=tau, delta_drop=args.delta_drop) for ex in examples]
    m_abr = compute_f1_from_sims(sims, examples)
    eff_abr = compute_efficiency(sims)

    print(f"Balanced F1   : {m_abr['balanced_f1']:.4f}")
    print(f"Positive F1   : {m_abr['positive_f1']:.4f}")
    print(f"Acc_erroneous : {m_abr['acc_erroneous']:.4f}")
    print(f"Acc_correct   : {m_abr['acc_correct']:.4f}")
    print(f"Accuracy      : {m_abr['accuracy']:.4f}")
    print(f"Stop rate     : {eff_abr['stopped_early_rate']:.4f}  (fraction that triggered early stop)")
    print(f"Mean step frac: {eff_abr['mean_step_fraction']:.4f}  (avg steps processed / total steps)")
    print()

    # Reason distribution
    reason_counts: dict[str, int] = defaultdict(int)
    for s in sims:
        reason_counts[s.get("reason", "end_of_trace")] += 1
    print("Stop reasons:")
    for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {cnt} ({cnt/len(sims):.1%})")
    print()

    # --- Comparison: fixed schedule ---
    print("=" * 80)
    print("Fixed-schedule baselines (stop after K steps, predict error for rest)")
    print("=" * 80)
    avg_steps = statistics.mean(ex.num_steps for ex in examples)
    print(f"Average steps per example: {avg_steps:.1f}")
    print()
    print(f"{'schedule':>12} | {'F1':>7} | {'Acc_err':>8} | {'Acc_cor':>8} | {'step_frac':>9}")
    print("-" * 57)
    fixed_rows: list[dict[str, Any]] = []
    for k in [1, 2, 3, 4, 5, 6]:
        sims_k = [simulate_fixed_schedule(ex, stop_after_k=k) for ex in examples]
        m_k = compute_f1_from_sims(sims_k, examples)
        eff_k = compute_efficiency(sims_k)
        fixed_rows.append(
            {
                "stop_after_k": k,
                "metrics": m_k,
                "efficiency": eff_k,
            }
        )
        print(
            f"  stop@{k}:{' ':4} | {m_k['f1']:>7.4f} | {m_k['acc_erroneous']:>8.4f} | "
            f"{m_k['acc_correct']:>8.4f} | {eff_k['mean_step_fraction']:>9.4f}"
        )
    print()

    # --- No-controller baseline ---
    print("=" * 80)
    print("No-controller baselines")
    print("=" * 80)
    # Always predict all-correct
    always_correct = [{"predicted_erroneous": False, "steps_processed": ex.num_steps, "total_steps": ex.num_steps} for ex in examples]
    m_ac = compute_f1_from_sims(always_correct, examples)
    print(f"Always-correct (predict nothing is wrong): F1={m_ac['f1']:.4f}, Acc_err={m_ac['acc_erroneous']:.4f}, Acc_cor={m_ac['acc_correct']:.4f}")
    # Always predict erroneous
    always_err = [{"predicted_erroneous": True, "stopped_at": 0, "steps_processed": 1, "total_steps": ex.num_steps} for ex in examples]
    m_ae = compute_f1_from_sims(always_err, examples)
    print(f"Always-erroneous (flag everything):        F1={m_ae['f1']:.4f}, Acc_err={m_ae['acc_erroneous']:.4f}, Acc_cor={m_ae['acc_correct']:.4f}")
    # Majority class (50/50 so both should be 0.5)
    print(f"Random (50/50): Acc_err=0.5, Acc_cor=0.5, F1=0.5000")
    print()

    # --- Summary ---
    print("=" * 80)
    print("Summary comparison")
    print("=" * 80)
    print(f"  ABR-lite (tau={tau:.2f}): F1={m_abr['f1']:.4f}, step_frac={eff_abr['mean_step_fraction']:.4f}")
    print(f"  Fixed@tau-equiv:       (see sweep above)")
    print(f"  No-controller:         F1=0.5 (chance for balanced dataset)")
    print()

    margin = m_abr["f1"] - 0.5
    if margin > 0.15:
        verdict = "STRONG PASS — value head provides substantial guidance over random."
    elif margin > 0.05:
        verdict = "PASS — value head provides useful guidance (above chance)."
    elif margin > 0:
        verdict = "WEAK — marginal improvement over random; needs further investigation."
    else:
        verdict = "FAIL — controller performs no better than random."
    print(f"F2 ABR-lite Assessment: {verdict}")
    print()

    # Save output if requested
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scored_rows_path": str(args.scored_rows_jsonl),
            "tau": tau,
            "delta_drop": args.delta_drop,
            "n_examples": len(examples),
            "n_all_correct": n_all_correct,
            "n_erroneous": n_erroneous,
            "abr_lite": {**m_abr, **eff_abr},
            "fixed_schedules": fixed_rows,
            "always_correct": m_ac,
            "always_erroneous": m_ae,
        }
        out = args.output_dir / "f2_simulation_results.json"
        out.write_text(json.dumps(result, indent=2))
        summary_json = args.output_dir / "summary.json"
        summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary_md = args.output_dir / "summary.md"
        summary_md.write_text(
            render_summary_markdown(
                scored_rows_path=args.scored_rows_jsonl,
                tau=tau,
                delta_drop=args.delta_drop,
                n_examples=len(examples),
                n_all_correct=n_all_correct,
                n_erroneous=n_erroneous,
                abr_metrics=m_abr,
                abr_efficiency=eff_abr,
                fixed_rows=fixed_rows,
            ),
            encoding="utf-8",
        )
        print(f"Results written to: {out}")
        print(f"summary_json: {summary_json}")
        print(f"summary_md: {summary_md}")


if __name__ == "__main__":
    main()
