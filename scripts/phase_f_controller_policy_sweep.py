#!/usr/bin/env python3
"""Phase F controller policy sweep on pre-scored ProcessBench prefixes.

这个脚本用于系统比较多种 `ABR-lite` 控制策略，回答两个问题：
1. 当前 controller 失败，是因为 verifier 分数本身不行，还是因为 stop rule 设计错误？
2. 哪一类控制策略最能缓解 `all-correct` 轨迹被过早拦截的问题？

This script compares multiple controller families on already-scored
`ProcessBench` prefix traces. It is intentionally offline and CPU-friendly:
we reuse existing `scored_rows.jsonl` artifacts instead of running any new model
inference, which keeps VRAM pressure close to zero.

Current policy families:
1. `baseline_immediate`: original "score below tau OR drop > delta" rule.
2. `threshold_only`: stop only on absolute low score.
3. `delayed_drop`: allow drop-trigger only after a warmup step.
4. `drop_needs_low`: drop-trigger must also land on a low-ish score.
5. `guarded_drop`: very low score stops immediately; drop-trigger is delayed and guarded.
6. `two_strike`: require two consecutive bad signals before stopping.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase_f_abr_lite_simulation import compute_efficiency, compute_f1_from_sims


@dataclass
class PrefixRow:
    """One scored prefix row from `phase_e_eval_benchmark.py`.

    一条 prefix 级打分记录。
    One prefix-level scored record.
    """

    step_index: int
    score: float


@dataclass
class ExampleTrace:
    """A single ProcessBench example represented as a score trace.

    `label=-1` 表示 all-correct；否则表示 first-bad step index。
    `label=-1` means all-correct; otherwise it is the first-bad step index.
    """

    example_id: str
    benchmark_id: str
    label: int
    rows: list[PrefixRow] = field(default_factory=list)

    @property
    def is_all_correct(self) -> bool:
        return self.label == -1

    @property
    def num_steps(self) -> int:
        return len(self.rows)


def load_example_traces(scored_rows_jsonl: Path, *, fallback_benchmark_id: str) -> list[ExampleTrace]:
    """Load one scored benchmark artifact into grouped traces.

    把 `scored_rows.jsonl` 聚合成按 example 划分的完整分数轨迹。
    Group `scored_rows.jsonl` into per-example score traces.
    """

    by_example: dict[str, ExampleTrace] = {}
    for line in scored_rows_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        example_id = row["example_id"]
        trace = by_example.get(example_id)
        if trace is None:
            trace = ExampleTrace(
                example_id=example_id,
                benchmark_id=row.get("benchmark_id", fallback_benchmark_id),
                label=row["label"],
            )
            by_example[example_id] = trace
        trace.rows.append(
            PrefixRow(
                step_index=row["prefix_step_index"],
                score=row["score"],
            )
        )
    traces = list(by_example.values())
    for trace in traces:
        trace.rows.sort(key=lambda row: row.step_index)
    return traces


def _trigger_baseline_immediate(
    *,
    score: float,
    prev_score: float,
    step_index: int,
    params: dict[str, float | int],
) -> bool:
    """Baseline: stop immediately on low score or large score drop."""

    return score < float(params["tau"]) or (prev_score - score) > float(params["delta"])


def _trigger_threshold_only(
    *,
    score: float,
    prev_score: float,
    step_index: int,
    params: dict[str, float | int],
) -> bool:
    """Threshold-only: ignore drop signals and stop only on low score."""

    del prev_score, step_index
    return score < float(params["tau"])


def _trigger_delayed_drop(
    *,
    score: float,
    prev_score: float,
    step_index: int,
    params: dict[str, float | int],
) -> bool:
    """Delayed-drop: low score always allowed; drop trigger only after warmup."""

    delta = prev_score - score
    return score < float(params["tau"]) or (
        step_index >= int(params["min_step"]) and delta > float(params["delta"])
    )


def _trigger_drop_needs_low(
    *,
    score: float,
    prev_score: float,
    step_index: int,
    params: dict[str, float | int],
) -> bool:
    """Drop-needs-low: drop must also land in a moderately low region."""

    del step_index
    delta = prev_score - score
    return score < float(params["tau"]) or (
        delta > float(params["delta"]) and score < float(params["tau_drop"])
    )


def _trigger_guarded_drop(
    *,
    score: float,
    prev_score: float,
    step_index: int,
    params: dict[str, float | int],
) -> bool:
    """Guarded-drop: immediate stop only for very low score; drop is delayed and guarded.

    这是针对当前主失败模式设计的策略：
    - 极低分数可以立刻停；
    - 普通波动不能在很早阶段触发；
    - drop 必须发生在 warmup 之后，且当前分数已经进入较低区间。
    This directly targets the current failure mode:
    - very low score can stop immediately;
    - ordinary early fluctuations cannot trigger stop;
    - drop-based stop is allowed only after warmup and only when the current
      score has already moved into a lower-confidence region.
    """

    delta = prev_score - score
    return score < float(params["tau_low"]) or (
        step_index >= int(params["min_step"])
        and delta > float(params["delta"])
        and score < float(params["tau_guard"])
    )


def simulate_two_strike(
    trace: ExampleTrace,
    *,
    tau: float,
    delta: float,
    min_step: int,
    strike_count: int,
) -> dict[str, Any]:
    """Require multiple consecutive bad signals before stopping.

    连续 bad signal 达到 `strike_count` 才允许 stop。
    Require `strike_count` consecutive bad signals before stopping.
    """

    prev_score = 1.0
    bad_streak = 0
    stopped_at = None
    for row in trace.rows:
        step_index = row.step_index
        score = row.score
        delta_now = prev_score - score
        bad_signal = score < tau or (step_index >= min_step and delta_now > delta)
        bad_streak = bad_streak + 1 if bad_signal else 0
        if bad_streak >= strike_count:
            stopped_at = step_index
            break
        prev_score = score

    predicted_erroneous = stopped_at is not None
    if trace.is_all_correct:
        correct_decision = not predicted_erroneous
    else:
        correct_decision = predicted_erroneous and stopped_at <= trace.label
    steps_processed = (stopped_at + 1) if stopped_at is not None else trace.num_steps
    return {
        "predicted_erroneous": predicted_erroneous,
        "correct_decision": correct_decision,
        "steps_processed": steps_processed,
        "total_steps": trace.num_steps,
    }


POLICY_FAMILIES: dict[str, list[dict[str, Any]]] = {
    "baseline_immediate": [
        {"tau": tau, "delta": delta}
        for tau in (0.32, 0.35, 0.38, 0.42)
        for delta in (0.12, 0.15, 0.20)
    ],
    "threshold_only": [
        {"tau": tau}
        for tau in (0.28, 0.32, 0.35, 0.38, 0.42, 0.46, 0.50)
    ],
    "delayed_drop": [
        {"tau": tau, "delta": delta, "min_step": min_step}
        for tau in (0.32, 0.35, 0.38, 0.42)
        for delta in (0.15, 0.20, 0.25)
        for min_step in (2, 3, 4)
    ],
    "drop_needs_low": [
        {"tau": tau, "delta": delta, "tau_drop": tau_drop}
        for tau in (0.32, 0.35, 0.38, 0.42)
        for delta in (0.15, 0.20, 0.25)
        for tau_drop in (0.45, 0.50, 0.55, 0.60)
    ],
    "guarded_drop": [
        {"tau_low": tau_low, "delta": delta, "tau_guard": tau_guard, "min_step": min_step}
        for tau_low in (0.20, 0.25, 0.30, 0.35)
        for delta in (0.15, 0.20, 0.25)
        for tau_guard in (0.45, 0.50, 0.55)
        for min_step in (2, 3, 4)
    ],
    "two_strike": [
        {"tau": tau, "delta": delta, "min_step": min_step, "strike_count": 2}
        for tau in (0.35, 0.38, 0.42, 0.46)
        for delta in (0.15, 0.20, 0.25)
        for min_step in (1, 2, 3)
    ],
}


def simulate_policy(trace: ExampleTrace, family: str, params: dict[str, Any]) -> dict[str, Any]:
    """Simulate one policy family on a single trace."""

    if family == "two_strike":
        return simulate_two_strike(
            trace,
            tau=float(params["tau"]),
            delta=float(params["delta"]),
            min_step=int(params["min_step"]),
            strike_count=int(params["strike_count"]),
        )

    trigger_fn = {
        "baseline_immediate": _trigger_baseline_immediate,
        "threshold_only": _trigger_threshold_only,
        "delayed_drop": _trigger_delayed_drop,
        "drop_needs_low": _trigger_drop_needs_low,
        "guarded_drop": _trigger_guarded_drop,
    }[family]

    prev_score = 1.0
    stopped_at = None
    for row in trace.rows:
        if trigger_fn(
            score=row.score,
            prev_score=prev_score,
            step_index=row.step_index,
            params=params,
        ):
            stopped_at = row.step_index
            break
        prev_score = row.score

    predicted_erroneous = stopped_at is not None
    if trace.is_all_correct:
        correct_decision = not predicted_erroneous
    else:
        correct_decision = predicted_erroneous and stopped_at <= trace.label
    steps_processed = (stopped_at + 1) if stopped_at is not None else trace.num_steps
    return {
        "predicted_erroneous": predicted_erroneous,
        "correct_decision": correct_decision,
        "steps_processed": steps_processed,
        "total_steps": trace.num_steps,
    }


def evaluate_family(traces: list[ExampleTrace], family: str) -> dict[str, Any]:
    """Evaluate one policy family and return the best configuration."""

    labels = [trace.label for trace in traces]
    best: dict[str, Any] | None = None
    for params in POLICY_FAMILIES[family]:
        sims = [simulate_policy(trace, family, params) for trace in traces]
        metrics = compute_f1_from_sims(sims, traces)
        efficiency = compute_efficiency(sims)
        row = {
            "family": family,
            "params": params,
            "metrics": metrics,
            "efficiency": efficiency,
        }
        if best is None or row["metrics"]["balanced_f1"] > best["metrics"]["balanced_f1"]:
            best = row
    assert best is not None
    return best


def fixed_schedule_baseline(traces: list[ExampleTrace]) -> dict[str, Any]:
    """Return the best fixed schedule among `stop@1..6` as a reference baseline."""

    best: dict[str, Any] | None = None
    for stop_after_k in range(1, 7):
        sims = []
        for trace in traces:
            predicted_erroneous = trace.num_steps > stop_after_k
            if trace.is_all_correct:
                correct_decision = not predicted_erroneous
            else:
                correct_decision = predicted_erroneous and stop_after_k <= trace.label
            steps_processed = min(trace.num_steps, stop_after_k + 1)
            sims.append(
                {
                    "predicted_erroneous": predicted_erroneous,
                    "correct_decision": correct_decision,
                    "steps_processed": steps_processed,
                    "total_steps": trace.num_steps,
                }
            )
        metrics = compute_f1_from_sims(sims, traces)
        efficiency = compute_efficiency(sims)
        row = {
            "family": "fixed_schedule",
            "params": {"stop_after_k": stop_after_k},
            "metrics": metrics,
            "efficiency": efficiency,
        }
        if best is None or row["metrics"]["balanced_f1"] > best["metrics"]["balanced_f1"]:
            best = row
    assert best is not None
    return best


def render_summary_markdown(run_dir: Path, case_rows: list[dict[str, Any]]) -> str:
    """Render a markdown summary for easy review."""

    lines = [
        "# Phase F Controller Policy Sweep",
        "",
        f"- run_dir: `{run_dir}`",
        "",
        "## Best Policy Per Case",
        "",
        "| case_id | best_family | balanced_f1 | positive_f1 | acc_err | acc_cor | step_frac |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in case_rows:
        best = case["best_overall"]
        lines.append(
            "| {case_id} | {family} | {bf1:.4f} | {pf1:.4f} | {ae:.4f} | {ac:.4f} | {sf:.4f} |".format(
                case_id=case["case_id"],
                family=best["family"],
                bf1=best["metrics"]["balanced_f1"],
                pf1=best["metrics"]["positive_f1"],
                ae=best["metrics"]["acc_erroneous"],
                ac=best["metrics"]["acc_correct"],
                sf=best["efficiency"]["mean_step_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-Family Winners",
            "",
            "| case_id | family | balanced_f1 | positive_f1 | acc_err | acc_cor | step_frac | params |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for case in case_rows:
        for row in case["family_winners"]:
            lines.append(
                "| {case_id} | {family} | {bf1:.4f} | {pf1:.4f} | {ae:.4f} | {ac:.4f} | {sf:.4f} | `{params}` |".format(
                    case_id=case["case_id"],
                    family=row["family"],
                    bf1=row["metrics"]["balanced_f1"],
                    pf1=row["metrics"]["positive_f1"],
                    ae=row["metrics"]["acc_erroneous"],
                    ac=row["metrics"]["acc_correct"],
                    sf=row["efficiency"]["mean_step_fraction"],
                    params=json.dumps(row["params"], sort_keys=True),
                )
            )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `balanced_f1` 是 controller 视角的主指标：既要拦错，也要放对。",
            "- `positive_f1` 是标准正类 F1，辅助看“错误类”抓取能力。",
            "- 如果 `balanced_f1` 很高同时 `step_frac` 很低，说明这是高收益 controller 方向。",
            "- 如果 `acc_err` 很高但 `acc_cor` 低，说明 controller 仍然在过度早停。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description="Sweep multiple ABR-lite controller policies on scored ProcessBench traces."
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Output run name under assets/artifacts/phase_f_controller_sweep/",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="CASE_ID|PATH_TO_SCORED_ROWS_JSONL",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the sweep and write machine-readable + markdown summaries."""

    args = build_parser().parse_args(argv)
    if not args.case:
        raise SystemExit("At least one --case CASE_ID|PATH is required.")

    run_dir = Path("assets/artifacts/phase_f_controller_sweep") / (
        f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict[str, Any]] = []
    for raw_case in args.case:
        case_id, path_str = raw_case.split("|", 1)
        scored_rows_jsonl = Path(path_str)
        traces = load_example_traces(scored_rows_jsonl, fallback_benchmark_id=case_id)
        family_winners = [evaluate_family(traces, family) for family in POLICY_FAMILIES]
        fixed_best = fixed_schedule_baseline(traces)
        family_winners.append(fixed_best)
        best_overall = max(family_winners, key=lambda row: row["metrics"]["balanced_f1"])
        case_row = {
            "case_id": case_id,
            "scored_rows_jsonl": str(scored_rows_jsonl),
            "num_examples": len(traces),
            "num_all_correct": sum(trace.is_all_correct for trace in traces),
            "num_erroneous": sum(not trace.is_all_correct for trace in traces),
            "family_winners": family_winners,
            "best_overall": best_overall,
        }
        case_rows.append(case_row)
        print(
            "{case_id:>16} | best={family:<16} | balanced_f1={bf1:.4f} | positive_f1={pf1:.4f} | "
            "acc_err={ae:.4f} | acc_cor={ac:.4f} | step_frac={sf:.4f}".format(
                case_id=case_id,
                family=best_overall["family"],
                bf1=best_overall["metrics"]["balanced_f1"],
                pf1=best_overall["metrics"]["positive_f1"],
                ae=best_overall["metrics"]["acc_erroneous"],
                ac=best_overall["metrics"]["acc_correct"],
                sf=best_overall["efficiency"]["mean_step_fraction"],
            )
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "policy_families": list(POLICY_FAMILIES.keys()) + ["fixed_schedule"],
        "cases": case_rows,
    }
    summary_json = run_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md = run_dir / "summary.md"
    summary_md.write_text(render_summary_markdown(run_dir, case_rows), encoding="utf-8")
    print(f"summary_json: {summary_json}")
    print(f"summary_md: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
