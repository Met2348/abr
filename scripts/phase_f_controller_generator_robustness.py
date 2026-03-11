#!/usr/bin/env python3
"""Phase F generator-robust controller search.

这个脚本专门回答一个更接近部署的问题：
如果不再只看 overall controller 指标，而是看最差 generator 子分布，
哪些 controller family 仍然稳？

This script searches for controller policies that remain strong under generator
subpopulation shift. Instead of selecting by overall score only, it ranks
policies by their worst-generator `balanced_f1` while keeping track of the
overall metric.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase_f_abr_lite_simulation import compute_efficiency, compute_f1_from_sims
from phase_f_controller_policy_sweep import (
    POLICY_FAMILIES,
    ExampleTrace,
    load_example_traces,
    simulate_policy,
)


def load_generator_map(processbench_json: Path) -> dict[str, str]:
    """Load `example_id -> generator` mapping from raw ProcessBench json."""

    rows = json.loads(processbench_json.read_text(encoding="utf-8"))
    return {row["id"]: row["generator"] for row in rows}


def evaluate_config_on_group(
    traces: list[ExampleTrace],
    family: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one controller configuration on a trace group."""

    sims = [simulate_policy(trace, family, params) for trace in traces]
    metrics = compute_f1_from_sims(sims, traces)
    efficiency = compute_efficiency(sims)
    return {
        "metrics": metrics,
        "efficiency": efficiency,
    }


def score_policy_with_generators(
    traces: list[ExampleTrace],
    generator_map: dict[str, str],
    family: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Score one controller configuration overall and by generator slice."""

    overall = evaluate_config_on_group(traces, family, params)

    by_generator: dict[str, list[ExampleTrace]] = {}
    for trace in traces:
        generator = generator_map.get(trace.example_id, "unknown")
        by_generator.setdefault(generator, []).append(trace)

    generator_rows = []
    for generator, group in sorted(by_generator.items()):
        scored = evaluate_config_on_group(group, family, params)
        generator_rows.append(
            {
                "generator": generator,
                "num_examples": len(group),
                "metrics": scored["metrics"],
                "efficiency": scored["efficiency"],
            }
        )

    worst_generator = min(
        generator_rows,
        key=lambda row: row["metrics"]["balanced_f1"],
    )
    return {
        "family": family,
        "params": params,
        "overall": overall,
        "generators": generator_rows,
        "worst_generator": worst_generator,
    }


def render_summary_markdown(run_dir: Path, case_rows: list[dict[str, Any]]) -> str:
    """Render markdown summary for generator-robust controller search."""

    lines = [
        "# Phase F Generator-Robust Controller Search",
        "",
        f"- run_dir: `{run_dir}`",
        "",
        "## Best Robust Policy Per Case",
        "",
        "| case_id | family | overall_balanced_f1 | worst_generator | worst_gen_balanced_f1 | overall_step_frac |",
        "|---|---|---:|---|---:|---:|",
    ]
    for case in case_rows:
        best = case["best_robust"]
        lines.append(
            "| {case_id} | {family} | {overall:.4f} | {generator} | {worst:.4f} | {step_frac:.4f} |".format(
                case_id=case["case_id"],
                family=best["family"],
                overall=best["overall"]["metrics"]["balanced_f1"],
                generator=best["worst_generator"]["generator"],
                worst=best["worst_generator"]["metrics"]["balanced_f1"],
                step_frac=best["overall"]["efficiency"]["mean_step_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- 这里不是选 overall 最优，而是选 `worst_generator balanced_f1` 最优。",
            "- 如果 `worst_generator balanced_f1` 仍然高，说明 controller 在 policy-shift 下更稳。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search controller policies by worst-generator robustness."
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Output run name under assets/artifacts/phase_f_controller_robustness/",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="CASE_ID|PATH_TO_SCORED_ROWS_JSONL|PATH_TO_PROCESSBENCH_JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.case:
        raise SystemExit("At least one --case CASE_ID|SCORED_ROWS|PROCESSBENCH_JSON is required.")

    run_dir = Path("assets/artifacts/phase_f_controller_robustness") / (
        f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    case_rows = []
    for raw_case in args.case:
        case_id, scored_rows_path, processbench_json = raw_case.split("|", 2)
        traces = load_example_traces(
            Path(scored_rows_path),
            fallback_benchmark_id=case_id,
        )
        generator_map = load_generator_map(Path(processbench_json))

        scored_rows = []
        for family in ("threshold_only", "delayed_drop", "drop_needs_low", "guarded_drop"):
            for params in POLICY_FAMILIES[family]:
                scored_rows.append(score_policy_with_generators(traces, generator_map, family, params))

        best_robust = max(
            scored_rows,
            key=lambda row: (
                row["worst_generator"]["metrics"]["balanced_f1"],
                row["overall"]["metrics"]["balanced_f1"],
            ),
        )
        case_rows.append(
            {
                "case_id": case_id,
                "scored_rows_jsonl": scored_rows_path,
                "processbench_json": processbench_json,
                "best_robust": best_robust,
            }
        )
        print(
            "{case_id:>16} | family={family:<15} | overall={overall:.4f} | "
            "worst_gen={generator} | worst={worst:.4f}".format(
                case_id=case_id,
                family=best_robust["family"],
                overall=best_robust["overall"]["metrics"]["balanced_f1"],
                generator=best_robust["worst_generator"]["generator"],
                worst=best_robust["worst_generator"]["metrics"]["balanced_f1"],
            )
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
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
