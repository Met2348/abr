#!/usr/bin/env python3
"""Sweep cheap-to-strong verifier handoff thresholds on ProcessBench scored rows.

English
-------
This script tests one concrete 2025-2026 verifier-system idea without training
new models: do we improve benchmark behavior if a cheap verifier only handles
"easy / confident" prefixes and defers the ambiguous ones to a stronger
verifier?

The current repository already has many `ProcessBench` scored-row artifacts.
What was missing is one reproducible sweep that answers:
1. how to quantify cheap-verifier confidence,
2. when replacing low-confidence cheap scores with strong scores helps,
3. how much strong-verifier usage is needed for that gain.

中文
----
这个脚本把一个很关键的 2025-2026 verifier 系统思路落成可复用实验：
让便宜 verifier 只处理“简单 / 有把握”的 prefix，把模糊样本升级给更强 verifier。

仓库里已经有很多 `ProcessBench scored_rows` 产物，但之前缺一个统一 sweep：
1. 如何定义 cheap verifier 的置信度；
2. 在什么阈值下，用 strong score 替换 weak score 会变好；
3. 这种收益需要消耗多少 strong-verifier 覆盖率。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_e.benchmark_eval import (  # noqa: E402
    build_processbench_prefix_records,
    compute_processbench_metrics,
    load_processbench_examples,
)


@dataclass(slots=True)
class GateCaseSpec:
    """One cheap-vs-strong gate case.

    一个 weak/strong verifier 门控实验的定义。
    """

    case_name: str
    benchmark_path: Path
    weak_scored_rows_path: Path
    strong_scored_rows_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep cheap-to-strong verifier handoff thresholds on ProcessBench."
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help=(
            "case definition in the form "
            "`name=benchmark.json::weak/scored_rows.jsonl::strong/scored_rows.jsonl`"
        ),
    )
    parser.add_argument(
        "--thresholds",
        default="0.00,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.49",
        help="Comma-separated confidence thresholds over |score-0.5|.",
    )
    parser.add_argument("--run-name", default="phase_e_weak_strong_gate")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_gate_sweeps"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    thresholds = _parse_thresholds(args.thresholds)
    cases = _parse_cases(args.case)

    run_dir = args.output_root / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    summary_rows: list[dict[str, Any]] = []
    for case in cases:
        case_summary = _run_case(case=case, thresholds=thresholds, run_dir=run_dir)
        summary_rows.append(case_summary)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "thresholds": thresholds,
        "cases": summary_rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(_render_summary_md(summary), encoding="utf-8")

    print("=" * 88)
    print("Phase E Weak-Strong Gate Sweep")
    print("=" * 88)
    for row in summary_rows:
        best = row["best_threshold_result"]
        print(
            f"{row['case_name']:24} "
            f"weak_auc={row['weak_metrics']['pair_auc_good_vs_bad']:.4f} "
            f"strong_auc={row['strong_metrics']['pair_auc_good_vs_bad']:.4f} "
            f"best_tau={best['threshold']:.2f} "
            f"best_auc={best['metrics']['pair_auc_good_vs_bad']:.4f} "
            f"strong_usage={best['strong_usage_rate']:.4f}"
        )
    print(f"summary_json_path : {run_dir / 'summary.json'}")
    print(f"summary_md_path   : {run_dir / 'summary.md'}")
    print("=" * 88)
    return 0


def _parse_thresholds(raw: str) -> list[float]:
    values = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        value = float(text)
        if value < 0.0 or value > 0.5:
            raise ValueError(f"Threshold must be in [0, 0.5], got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one threshold is required")
    return values


def _parse_cases(raw_items: list[str]) -> list[GateCaseSpec]:
    cases: list[GateCaseSpec] = []
    for item in raw_items:
        if "=" not in item or "::" not in item:
            raise ValueError(
                "--case must look like "
                "`name=benchmark.json::weak/scored_rows.jsonl::strong/scored_rows.jsonl`"
            )
        name, raw_paths = item.split("=", 1)
        bench_text, weak_text, strong_text = raw_paths.split("::", 2)
        spec = GateCaseSpec(
            case_name=name.strip(),
            benchmark_path=Path(bench_text).expanduser(),
            weak_scored_rows_path=Path(weak_text).expanduser(),
            strong_scored_rows_path=Path(strong_text).expanduser(),
        )
        for path in (spec.benchmark_path, spec.weak_scored_rows_path, spec.strong_scored_rows_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing path for case {spec.case_name!r}: {path}")
        cases.append(spec)
    return cases


def _run_case(*, case: GateCaseSpec, thresholds: list[float], run_dir: Path) -> dict[str, Any]:
    examples = load_processbench_examples(case.benchmark_path)
    rows = build_processbench_prefix_records(examples)
    weak_scores = _load_score_by_row_id(case.weak_scored_rows_path)
    strong_scores = _load_score_by_row_id(case.strong_scored_rows_path)

    ordered_weak_scores: list[float] = []
    ordered_strong_scores: list[float] = []
    row_ids: list[str] = []
    for row in rows:
        row_id = str(row.row_id)
        if row_id not in weak_scores:
            raise RuntimeError(f"Weak scored rows missing row_id={row_id!r}")
        if row_id not in strong_scores:
            raise RuntimeError(f"Strong scored rows missing row_id={row_id!r}")
        row_ids.append(row_id)
        ordered_weak_scores.append(float(weak_scores[row_id]))
        ordered_strong_scores.append(float(strong_scores[row_id]))

    weak_metrics = compute_processbench_metrics(rows, ordered_weak_scores)
    strong_metrics = compute_processbench_metrics(rows, ordered_strong_scores)

    threshold_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        mixed_scores = []
        strong_usage = 0
        confidence_values = []
        for weak_score, strong_score in zip(ordered_weak_scores, ordered_strong_scores, strict=True):
            confidence = abs(float(weak_score) - 0.5)
            confidence_values.append(confidence)
            if confidence < float(threshold):
                mixed_scores.append(float(strong_score))
                strong_usage += 1
            else:
                mixed_scores.append(float(weak_score))
        metrics = compute_processbench_metrics(rows, mixed_scores)
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "strong_usage_rate": float(strong_usage / len(mixed_scores)) if mixed_scores else 0.0,
                "mean_weak_confidence": float(statistics.mean(confidence_values)) if confidence_values else 0.0,
                "metrics": metrics,
            }
        )

    best_threshold_result = max(
        threshold_rows,
        key=lambda row: (
            float(row["metrics"]["pair_auc_good_vs_bad"]),
            float(row["metrics"]["first_error_edge_accuracy"]),
            -float(row["strong_usage_rate"]),
        ),
    )

    case_summary = {
        "case_name": case.case_name,
        "benchmark_path": str(case.benchmark_path),
        "weak_scored_rows_path": str(case.weak_scored_rows_path),
        "strong_scored_rows_path": str(case.strong_scored_rows_path),
        "num_prefix_rows": int(len(rows)),
        "weak_metrics": weak_metrics,
        "strong_metrics": strong_metrics,
        "best_threshold_result": best_threshold_result,
        "threshold_sweep": threshold_rows,
    }
    case_dir = run_dir / case.case_name
    case_dir.mkdir(parents=True, exist_ok=False)
    (case_dir / "summary.json").write_text(json.dumps(case_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (case_dir / "summary.md").write_text(_render_case_md(case_summary), encoding="utf-8")
    return case_summary


def _load_score_by_row_id(path: Path) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mapping[str(row["row_id"])] = float(row["score"])
    if not mapping:
        raise RuntimeError(f"No scored rows loaded from {path}")
    return mapping


def _render_case_md(summary: dict[str, Any]) -> str:
    lines = [
        f"# Phase E Weak-Strong Gate Case: {summary['case_name']}",
        "",
        "## Inputs",
        "",
        f"- benchmark_path: `{summary['benchmark_path']}`",
        f"- weak_scored_rows_path: `{summary['weak_scored_rows_path']}`",
        f"- strong_scored_rows_path: `{summary['strong_scored_rows_path']}`",
        f"- num_prefix_rows: `{summary['num_prefix_rows']}`",
        "",
        "## Baselines",
        "",
        f"- weak_auc: `{summary['weak_metrics']['pair_auc_good_vs_bad']:.6f}`",
        f"- weak_first_edge: `{summary['weak_metrics']['first_error_edge_accuracy']:.6f}`",
        f"- strong_auc: `{summary['strong_metrics']['pair_auc_good_vs_bad']:.6f}`",
        f"- strong_first_edge: `{summary['strong_metrics']['first_error_edge_accuracy']:.6f}`",
        "",
        "## Best Threshold",
        "",
        f"- threshold: `{summary['best_threshold_result']['threshold']:.2f}`",
        f"- strong_usage_rate: `{summary['best_threshold_result']['strong_usage_rate']:.6f}`",
        f"- best_auc: `{summary['best_threshold_result']['metrics']['pair_auc_good_vs_bad']:.6f}`",
        f"- best_first_edge: `{summary['best_threshold_result']['metrics']['first_error_edge_accuracy']:.6f}`",
        "",
        "## Sweep",
        "",
        "| threshold | strong_usage_rate | auc | first_edge |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary["threshold_sweep"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['strong_usage_rate']:.4f} | "
            f"{row['metrics']['pair_auc_good_vs_bad']:.4f} | "
            f"{row['metrics']['first_error_edge_accuracy']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase E Weak-Strong Gate Sweep Summary",
        "",
        "| case | weak_auc | strong_auc | best_tau | best_auc | best_first_edge | strong_usage_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["cases"]:
        best = row["best_threshold_result"]
        lines.append(
            f"| {row['case_name']} | "
            f"{row['weak_metrics']['pair_auc_good_vs_bad']:.4f} | "
            f"{row['strong_metrics']['pair_auc_good_vs_bad']:.4f} | "
            f"{best['threshold']:.2f} | "
            f"{best['metrics']['pair_auc_good_vs_bad']:.4f} | "
            f"{best['metrics']['first_error_edge_accuracy']:.4f} | "
            f"{best['strong_usage_rate']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
