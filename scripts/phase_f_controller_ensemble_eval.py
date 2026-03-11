#!/usr/bin/env python3
"""Phase F weak-verifier score ensemble evaluation.

这个脚本评估“弱 verifier 组合”是否能提升 controller 表现。
它不重新训练模型，而是离线组合多个 `scored_rows.jsonl` 的 prefix 分数，
再在组合后的分数轨迹上跑 controller policy search。

This script evaluates whether score-level ensembling of multiple verifier
candidates can improve controller quality without any new training. It aligns
multiple `scored_rows.jsonl` files by `row_id`, combines their scores, and then
re-runs the controller policy search on the combined traces.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase_f_controller_policy_sweep import (
    ExampleTrace,
    PrefixRow,
    POLICY_FAMILIES,
    evaluate_family,
)


ENSEMBLE_SPECS: dict[str, dict[str, Any]] = {
    "mean_50": {"mode": "weighted_mean", "alpha": 0.50},
    "mean_75a": {"mode": "weighted_mean", "alpha": 0.75},
    "mean_25a": {"mode": "weighted_mean", "alpha": 0.25},
    "min": {"mode": "min"},
    "max": {"mode": "max"},
}


def load_scored_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Load `row_id -> row` for one scored artifact."""

    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["row_id"]] = row
    return rows


def combine_scores(score_a: float, score_b: float, spec: dict[str, Any]) -> float:
    """Combine two verifier scores according to an ensemble spec."""

    mode = spec["mode"]
    if mode == "weighted_mean":
        alpha = float(spec["alpha"])
        return alpha * score_a + (1.0 - alpha) * score_b
    if mode == "min":
        return min(score_a, score_b)
    if mode == "max":
        return max(score_a, score_b)
    raise ValueError(f"Unsupported ensemble mode: {mode}")


def build_combined_traces(
    rows_a: dict[str, dict[str, Any]],
    rows_b: dict[str, dict[str, Any]],
    *,
    fallback_benchmark_id: str,
    spec: dict[str, Any],
) -> list[ExampleTrace]:
    """Build synthetic traces from two aligned scored-row tables."""

    common_keys = sorted(set(rows_a).intersection(rows_b))
    by_example: dict[str, ExampleTrace] = {}
    for row_id in common_keys:
        row_a = rows_a[row_id]
        row_b = rows_b[row_id]
        example_id = row_a["example_id"]
        trace = by_example.get(example_id)
        if trace is None:
            trace = ExampleTrace(
                example_id=example_id,
                benchmark_id=row_a.get("benchmark_id", fallback_benchmark_id),
                label=row_a["label"],
            )
            by_example[example_id] = trace
        trace.rows.append(
            PrefixRow(
                step_index=row_a["prefix_step_index"],
                score=combine_scores(float(row_a["score"]), float(row_b["score"]), spec),
            )
        )
    traces = list(by_example.values())
    for trace in traces:
        trace.rows.sort(key=lambda row: row.step_index)
    return traces


def render_summary_markdown(run_dir: Path, case_rows: list[dict[str, Any]]) -> str:
    """Render markdown summary."""

    lines = [
        "# Phase F Weak-Verifier Ensemble Evaluation",
        "",
        f"- run_dir: `{run_dir}`",
        "",
        "## Best Ensemble Per Case",
        "",
        "| case_id | ensemble | best_policy_family | balanced_f1 | positive_f1 | step_frac |",
        "|---|---|---|---:|---:|---:|",
    ]
    for case in case_rows:
        best = case["best_ensemble"]
        policy = best["best_policy"]
        lines.append(
            "| {case_id} | {ensemble} | {family} | {bf1:.4f} | {pf1:.4f} | {sf:.4f} |".format(
                case_id=case["case_id"],
                ensemble=best["ensemble_name"],
                family=policy["family"],
                bf1=policy["metrics"]["balanced_f1"],
                pf1=policy["metrics"]["positive_f1"],
                sf=policy["efficiency"]["mean_step_fraction"],
            )
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate score-level verifier ensembles for Phase F controller use."
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Output run name under assets/artifacts/phase_f_controller_ensemble/",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="CASE_ID|ROWS_A|ROWS_B",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.case:
        raise SystemExit("At least one --case CASE_ID|ROWS_A|ROWS_B is required.")

    run_dir = Path("assets/artifacts/phase_f_controller_ensemble") / (
        f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    case_rows = []
    for raw_case in args.case:
        case_id, rows_a_path, rows_b_path = raw_case.split("|", 2)
        rows_a = load_scored_rows(Path(rows_a_path))
        rows_b = load_scored_rows(Path(rows_b_path))

        ensemble_rows = []
        for ensemble_name, spec in ENSEMBLE_SPECS.items():
            traces = build_combined_traces(
                rows_a,
                rows_b,
                fallback_benchmark_id=case_id,
                spec=spec,
            )
            family_winners = [evaluate_family(traces, family) for family in POLICY_FAMILIES]
            best_policy = max(family_winners, key=lambda row: row["metrics"]["balanced_f1"])
            ensemble_rows.append(
                {
                    "ensemble_name": ensemble_name,
                    "spec": spec,
                    "best_policy": best_policy,
                }
            )
        best_ensemble = max(
            ensemble_rows,
            key=lambda row: row["best_policy"]["metrics"]["balanced_f1"],
        )
        case_rows.append(
            {
                "case_id": case_id,
                "rows_a": rows_a_path,
                "rows_b": rows_b_path,
                "ensembles": ensemble_rows,
                "best_ensemble": best_ensemble,
            }
        )
        best_policy = best_ensemble["best_policy"]
        print(
            "{case_id:>20} | ensemble={ensemble:<8} | family={family:<15} | "
            "balanced_f1={bf1:.4f} | positive_f1={pf1:.4f}".format(
                case_id=case_id,
                ensemble=best_ensemble["ensemble_name"],
                family=best_policy["family"],
                bf1=best_policy["metrics"]["balanced_f1"],
                pf1=best_policy["metrics"]["positive_f1"],
            )
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "ensemble_specs": ENSEMBLE_SPECS,
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
