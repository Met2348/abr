#!/usr/bin/env python3
"""Diagnose RL-facing transfer risk for Phase E value heads.

English
-------
This script exists because the current repository already has:
1. same-family utility reports,
2. benchmark-native ProcessBench reports,
3. and one loose RL-readiness heuristic.

What is still missing is a tighter *failure-structure* diagnosis:
1. how much the benchmark margin collapses relative to same-family scoring,
2. whether the head can still detect the local first-bad edge,
3. whether it undervalues fully-correct final prefixes,
4. and whether length/support drift is a plausible explanation.

中文
----
当前仓库已经有：
1. same-family utility 报告，
2. benchmark-native ProcessBench 报告，
3. 以及一套偏宽松的 RL-readiness heuristic。

但还缺一层更贴近“失败结构”的诊断：
1. benchmark 上的分数 margin 相比 same-family 到底塌了多少，
2. head 还能不能抓住局部 first-bad edge，
3. 会不会系统性低估“完整正确”的最终 prefix，
4. 长度 / support drift 是否是一个合理解释。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose same-family vs ProcessBench transfer behavior for one or more Phase E checkpoints."
    )
    parser.add_argument(
        "--audit-spec",
        action="append",
        required=True,
        metavar="AUDIT_ID|SOURCE_FAMILY|SAMEFAMILY_DIR|PB_GSM_DIR|PB_MATH_DIR",
        help=(
            "One audit target. Repeat this flag for multiple checkpoints. "
            "Each directory should point to an existing artifact run directory."
        ),
    )
    parser.add_argument("--run-name", default="phase_e_transfer_diag")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_transfer_diag"),
    )
    parser.add_argument("--strict-samefamily-top1-min", type=float, default=0.90)
    parser.add_argument("--strict-samefamily-local-min", type=float, default=0.90)
    parser.add_argument("--strict-pressure-top1-min", type=float, default=0.90)
    parser.add_argument("--strict-benchmark-auc-min", type=float, default=0.60)
    parser.add_argument("--strict-first-edge-min", type=float, default=0.60)
    parser.add_argument("--strict-terminal-top1-min", type=float, default=0.50)
    parser.add_argument("--strict-terminal-gap-min", type=float, default=-0.05)
    parser.add_argument("--length-drift-warning-ratio", type=float, default=2.0)
    parser.add_argument("--margin-collapse-warning-ratio", type=float, default=0.20)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.audit_spec:
        raise ValueError("At least one --audit-spec is required")
    for name in (
        "strict_samefamily_top1_min",
        "strict_samefamily_local_min",
        "strict_pressure_top1_min",
        "strict_benchmark_auc_min",
        "strict_first_edge_min",
        "strict_terminal_top1_min",
        "length_drift_warning_ratio",
        "margin_collapse_warning_ratio",
    ):
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = _allocate_run_dir(output_root=Path(args.output_root), run_name=str(args.run_name))
    run_dir.mkdir(parents=True, exist_ok=False)
    audit_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []

    thresholds = {
        "strict_samefamily_top1_min": float(args.strict_samefamily_top1_min),
        "strict_samefamily_local_min": float(args.strict_samefamily_local_min),
        "strict_pressure_top1_min": float(args.strict_pressure_top1_min),
        "strict_benchmark_auc_min": float(args.strict_benchmark_auc_min),
        "strict_first_edge_min": float(args.strict_first_edge_min),
        "strict_terminal_top1_min": float(args.strict_terminal_top1_min),
        "strict_terminal_gap_min": float(args.strict_terminal_gap_min),
        "length_drift_warning_ratio": float(args.length_drift_warning_ratio),
        "margin_collapse_warning_ratio": float(args.margin_collapse_warning_ratio),
    }

    for spec in args.audit_spec:
        audit = _run_one_audit(spec_text=str(spec), thresholds=thresholds)
        audit_rows.append(audit["audit_row"])
        benchmark_rows.extend(audit["benchmark_rows"])

    summary = {
        "run_dir": str(run_dir),
        "thresholds": thresholds,
        "audits": audit_rows,
        "benchmarks": benchmark_rows,
    }
    summary_json_path = run_dir / "summary.json"
    benchmark_rows_path = run_dir / "benchmark_rows.jsonl"
    audit_rows_path = run_dir / "audit_rows.jsonl"
    summary_md_path = run_dir / "summary.md"

    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(audit_rows_path, audit_rows)
    _write_jsonl(benchmark_rows_path, benchmark_rows)
    summary_md_path.write_text(
        _render_summary_markdown(thresholds=thresholds, audit_rows=audit_rows, benchmark_rows=benchmark_rows),
        encoding="utf-8",
    )

    print("=" * 88)
    print("Phase E Transfer Diagnosis")
    print("=" * 88)
    for row in audit_rows:
        print(
            f"{row['audit_id']:>14} | strict_rl_ready={int(bool(row['strict_rl_ready']))} "
            f"| samefamily_top1={row['samefamily_top1']:.4f} "
            f"| pb_min_auc={row['benchmark_min_auc']:.4f} "
            f"| pb_min_first_edge={row['benchmark_min_first_edge_accuracy']:.4f} "
            f"| pb_min_terminal_top1={row['benchmark_min_terminal_top1_accuracy']:.4f} "
            f"| assessment={row['assessment']}"
        )
    print(f"summary_json      : {summary_json_path}")
    print(f"summary_md        : {summary_md_path}")
    print("=" * 88)
    return 0


def _run_one_audit(*, spec_text: str, thresholds: dict[str, float]) -> dict[str, Any]:
    parts = spec_text.split("|")
    if len(parts) != 5:
        raise ValueError(
            "--audit-spec must have exactly five pipe-separated parts: "
            "AUDIT_ID|SOURCE_FAMILY|SAMEFAMILY_DIR|PB_GSM_DIR|PB_MATH_DIR"
        )
    audit_id, source_family, samefamily_dir_text, pb_gsm_dir_text, pb_math_dir_text = [part.strip() for part in parts]
    if not audit_id or not source_family:
        raise ValueError(f"Invalid audit spec: {spec_text}")

    samefamily_dir = Path(samefamily_dir_text)
    pb_gsm_dir = Path(pb_gsm_dir_text)
    pb_math_dir = Path(pb_math_dir_text)
    for path in (samefamily_dir, pb_gsm_dir, pb_math_dir):
        if not path.exists():
            raise FileNotFoundError(f"Audit directory not found: {path}")

    samefamily_manifest = _load_json(samefamily_dir / "manifest.json")
    samefamily_metrics = _load_json(samefamily_dir / "metrics.json")
    prompt_rows = _load_jsonl(samefamily_dir / "prompt_rows.jsonl")

    samefamily_top1 = float(samefamily_metrics.get("prompt_pool_top1_accuracy", 0.0))
    samefamily_local_first_bad = _maybe_float(samefamily_metrics.get("local_first_bad_edge_accuracy"))
    samefamily_pressure8 = _extract_pressure_top1(samefamily_metrics, subset_size=8)
    samefamily_rej40_gain = _extract_rejection_gain(samefamily_metrics, target_coverage=0.40)
    samefamily_random_top1 = _compute_prompt_random_top1(prompt_rows)
    samefamily_gap = float(samefamily_metrics.get("prompt_pool_mean_score_gap", 0.0))
    samefamily_p95 = float(
        samefamily_manifest["truncation_diagnostics"]["overall"]["text_length"]["p95"]
    )

    benchmark_rows = []
    benchmark_summaries = []
    for benchmark_name, bench_dir in (
        ("processbench_gsm8k", pb_gsm_dir),
        ("processbench_math", pb_math_dir),
    ):
        summary_payload = _load_json(bench_dir / "summary.json")
        scored_rows = _load_jsonl(bench_dir / "scored_rows.jsonl")
        benchmark_row = _analyze_processbench(
            audit_id=audit_id,
            source_family=source_family,
            benchmark_name=benchmark_name,
            summary_payload=summary_payload,
            scored_rows=scored_rows,
            samefamily_p95=samefamily_p95,
            samefamily_gap=samefamily_gap,
            thresholds=thresholds,
        )
        benchmark_rows.append(benchmark_row)
        benchmark_summaries.append(benchmark_row)

    samefamily_green = (
        samefamily_top1 >= thresholds["strict_samefamily_top1_min"]
        and (samefamily_pressure8 is None or samefamily_pressure8 >= thresholds["strict_pressure_top1_min"])
        and (
            samefamily_local_first_bad is None
            or samefamily_local_first_bad >= thresholds["strict_samefamily_local_min"]
        )
    )
    benchmark_green = all(
        row["benchmark_auc"] >= thresholds["strict_benchmark_auc_min"]
        and row["first_edge_accuracy"] >= thresholds["strict_first_edge_min"]
        and row["all_correct_terminal_top1_accuracy"] >= thresholds["strict_terminal_top1_min"]
        and row["all_correct_terminal_gap_mean"] >= thresholds["strict_terminal_gap_min"]
        for row in benchmark_summaries
    )
    strict_rl_ready = bool(samefamily_green and benchmark_green and samefamily_local_first_bad is not None)

    failure_modes: list[str] = []
    if samefamily_local_first_bad is None:
        failure_modes.append("missing_samefamily_local_gate")
    if not samefamily_green:
        failure_modes.append("samefamily_gate_not_clean")
    for row in benchmark_summaries:
        failure_modes.extend(str(tag) for tag in row["failure_tags"])
    failure_modes = sorted(set(failure_modes))

    if strict_rl_ready:
        assessment = "strict_rl_ready"
    elif any("terminal_completion_undervalued" == tag for tag in failure_modes):
        assessment = "not_rl_ready_terminal_completion_risk"
    elif any("benchmark_local_error_weak" == tag for tag in failure_modes):
        assessment = "not_rl_ready_local_transfer_weak"
    elif samefamily_green:
        assessment = "samefamily_strong_but_benchmark_incomplete"
    else:
        assessment = "not_rl_ready"

    audit_row = {
        "audit_id": audit_id,
        "source_family": source_family,
        "value_run_dir": samefamily_manifest.get("value_run_dir"),
        "samefamily_run_dir": str(samefamily_dir),
        "samefamily_top1": samefamily_top1,
        "samefamily_random_top1": samefamily_random_top1,
        "samefamily_top1_lift": float(samefamily_top1 - samefamily_random_top1),
        "samefamily_rejection_040_gain": samefamily_rej40_gain,
        "samefamily_pressure_008_top1": samefamily_pressure8,
        "samefamily_local_first_bad_accuracy": samefamily_local_first_bad,
        "samefamily_mean_score_gap": samefamily_gap,
        "samefamily_text_p95": samefamily_p95,
        "benchmark_min_auc": float(min(row["benchmark_auc"] for row in benchmark_summaries)),
        "benchmark_min_first_edge_accuracy": float(
            min(row["first_edge_accuracy"] for row in benchmark_summaries)
        ),
        "benchmark_min_terminal_top1_accuracy": float(
            min(row["all_correct_terminal_top1_accuracy"] for row in benchmark_summaries)
        ),
        "samefamily_green": bool(samefamily_green),
        "benchmark_green": bool(benchmark_green),
        "strict_rl_ready": bool(strict_rl_ready),
        "failure_modes": failure_modes,
        "assessment": assessment,
    }
    return {
        "audit_row": audit_row,
        "benchmark_rows": benchmark_rows,
    }


def _analyze_processbench(
    *,
    audit_id: str,
    source_family: str,
    benchmark_name: str,
    summary_payload: dict[str, Any],
    scored_rows: list[dict[str, Any]],
    samefamily_p95: float,
    samefamily_gap: float,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    metrics = dict(summary_payload.get("metrics", {}))
    truncation = dict(summary_payload.get("truncation_diagnostics", {}))
    overall_trunc = dict(truncation.get("overall", {}))
    bench_p95 = float(overall_trunc.get("text_length", {}).get("p95", 0.0) or 0.0)
    bench_over_limit = float(overall_trunc.get("frac_texts_over_limit", 0.0) or 0.0)
    benchmark_auc = _resolve_auc(metrics)
    first_edge_accuracy = float(metrics.get("first_error_edge_accuracy", 0.0) or 0.0)
    mean_good = float(metrics.get("mean_good_prefix_score", 0.0) or 0.0)
    mean_bad = float(metrics.get("mean_bad_prefix_score", 0.0) or 0.0)
    mean_all_correct_last = float(metrics.get("mean_all_correct_last_score", 0.0) or 0.0)
    completion_penalty = float(mean_all_correct_last - mean_good)
    margin_collapse_ratio = float((mean_good - mean_bad) / max(float(samefamily_gap), 1e-9))
    length_ratio = float(bench_p95 / max(float(samefamily_p95), 1e-9))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in scored_rows:
        grouped.setdefault(str(row["example_id"]), []).append(row)

    all_correct_top1_hits: list[float] = []
    all_correct_gap_values: list[float] = []
    last_safe_first_bad_hits: list[float] = []
    last_safe_first_bad_gaps: list[float] = []
    last_safe_final_hits: list[float] = []
    last_safe_final_gaps: list[float] = []
    label_zero_error_examples = 0

    for example_rows in grouped.values():
        ordered = sorted(example_rows, key=lambda item: int(item["prefix_step_index"]))
        label = int(ordered[0]["label"])
        final_score = float(ordered[-1]["score"])
        prev_scores = [float(row["score"]) for row in ordered[:-1]]
        prev_max = max(prev_scores) if prev_scores else final_score
        if label < 0:
            all_correct_top1_hits.append(1.0 if final_score >= prev_max else 0.0)
            all_correct_gap_values.append(float(final_score - prev_max))
            continue
        if label == 0:
            label_zero_error_examples += 1
            continue
        if label >= len(ordered):
            continue
        last_safe_score = float(ordered[label - 1]["score"])
        first_bad_score = float(ordered[label]["score"])
        last_safe_first_bad_hits.append(1.0 if last_safe_score > first_bad_score else 0.0)
        last_safe_first_bad_gaps.append(float(last_safe_score - first_bad_score))
        last_safe_final_hits.append(1.0 if last_safe_score > final_score else 0.0)
        last_safe_final_gaps.append(float(last_safe_score - final_score))

    all_correct_terminal_top1_accuracy = _safe_mean(all_correct_top1_hits)
    all_correct_terminal_gap_mean = _safe_mean(all_correct_gap_values)
    last_safe_first_bad_accuracy = _safe_mean(last_safe_first_bad_hits)
    last_safe_first_bad_gap_mean = _safe_mean(last_safe_first_bad_gaps)
    last_safe_final_bad_accuracy = _safe_mean(last_safe_final_hits)
    last_safe_final_bad_gap_mean = _safe_mean(last_safe_final_gaps)

    failure_tags: list[str] = []
    if benchmark_auc < thresholds["strict_benchmark_auc_min"] or first_edge_accuracy < thresholds["strict_first_edge_min"]:
        failure_tags.append("benchmark_local_error_weak")
    if (
        all_correct_terminal_top1_accuracy < thresholds["strict_terminal_top1_min"]
        or all_correct_terminal_gap_mean < thresholds["strict_terminal_gap_min"]
    ):
        failure_tags.append("terminal_completion_undervalued")
    if length_ratio >= thresholds["length_drift_warning_ratio"] or bench_over_limit > 0.02:
        failure_tags.append("support_length_drift")
    if margin_collapse_ratio < thresholds["margin_collapse_warning_ratio"]:
        failure_tags.append("margin_collapse")

    return {
        "audit_id": audit_id,
        "source_family": source_family,
        "benchmark_name": benchmark_name,
        "benchmark_run_dir": str(summary_payload.get("run_dir", "")),
        "benchmark_auc": benchmark_auc,
        "pair_accuracy_good_vs_bad": float(metrics.get("pair_accuracy_good_vs_bad", 0.0) or 0.0),
        "first_edge_accuracy": first_edge_accuracy,
        "mean_good_prefix_score": mean_good,
        "mean_bad_prefix_score": mean_bad,
        "mean_all_correct_last_score": mean_all_correct_last,
        "completion_penalty": completion_penalty,
        "all_correct_terminal_top1_accuracy": all_correct_terminal_top1_accuracy,
        "all_correct_terminal_gap_mean": all_correct_terminal_gap_mean,
        "last_safe_first_bad_accuracy": last_safe_first_bad_accuracy,
        "last_safe_first_bad_gap_mean": last_safe_first_bad_gap_mean,
        "last_safe_final_bad_accuracy": last_safe_final_bad_accuracy,
        "last_safe_final_bad_gap_mean": last_safe_final_bad_gap_mean,
        "samefamily_text_p95": float(samefamily_p95),
        "benchmark_text_p95": bench_p95,
        "length_ratio_vs_samefamily": length_ratio,
        "benchmark_over_limit_fraction": bench_over_limit,
        "samefamily_gap": float(samefamily_gap),
        "margin_collapse_ratio": margin_collapse_ratio,
        "label_zero_error_examples": int(label_zero_error_examples),
        "failure_tags": failure_tags,
    }


def _extract_rejection_gain(metrics: dict[str, Any], *, target_coverage: float) -> float | None:
    base_top1 = _maybe_float(metrics.get("prompt_pool_top1_accuracy"))
    if base_top1 is None:
        return None
    for point in metrics.get("rejection_curve", []):
        if math.isclose(float(point.get("target_coverage", -1.0)), float(target_coverage), abs_tol=1e-6):
            top1 = _maybe_float(point.get("top1_accuracy"))
            if top1 is None:
                return None
            return float(top1 - base_top1)
    return None


def _extract_pressure_top1(metrics: dict[str, Any], *, subset_size: int) -> float | None:
    for point in metrics.get("pressure_curve", []):
        if int(point.get("subset_size", -1)) == int(subset_size):
            return _maybe_float(point.get("top1_accuracy"))
    return None


def _compute_prompt_random_top1(prompt_rows: list[dict[str, Any]]) -> float:
    if not prompt_rows:
        return 0.0
    values = []
    for row in prompt_rows:
        num_candidates = max(int(row.get("num_candidates", 1) or 1), 1)
        gold_count = len(list(row.get("gold_top_candidate_ids", [])))
        values.append(float(gold_count / num_candidates))
    return _safe_mean(values)


def _resolve_auc(metrics: dict[str, Any]) -> float:
    if "pair_auc_good_vs_bad" in metrics:
        return float(metrics["pair_auc_good_vs_bad"])
    if "auc" in metrics:
        return float(metrics["auc"])
    return 0.0


def _render_summary_markdown(
    *,
    thresholds: dict[str, float],
    audit_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase E Transfer Diagnosis",
        "",
        "## Strict Gate",
        "",
        f"- samefamily_top1 >= `{thresholds['strict_samefamily_top1_min']:.2f}`",
        f"- samefamily_local_first_bad >= `{thresholds['strict_samefamily_local_min']:.2f}` when available",
        f"- pressure@8_top1 >= `{thresholds['strict_pressure_top1_min']:.2f}` when available",
        f"- ProcessBench auc >= `{thresholds['strict_benchmark_auc_min']:.2f}` on both GSM8K and Math",
        f"- ProcessBench first_error_edge_accuracy >= `{thresholds['strict_first_edge_min']:.2f}` on both GSM8K and Math",
        f"- all-correct final-prefix top1 >= `{thresholds['strict_terminal_top1_min']:.2f}` on both GSM8K and Math",
        f"- all-correct final-prefix mean gap >= `{thresholds['strict_terminal_gap_min']:.2f}` on both GSM8K and Math",
        "",
        "## Audit Summary",
        "",
        "| audit_id | source | sf_top1 | sf_lift | sf_local | rej40_gain | p8_top1 | pb_min_auc | pb_min_edge | pb_min_terminal_top1 | strict_rl_ready | assessment |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in audit_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["audit_id"]),
                    str(row["source_family"]),
                    _fmt(row.get("samefamily_top1")),
                    _fmt(row.get("samefamily_top1_lift")),
                    _fmt(row.get("samefamily_local_first_bad_accuracy")),
                    _fmt(row.get("samefamily_rejection_040_gain")),
                    _fmt(row.get("samefamily_pressure_008_top1")),
                    _fmt(row.get("benchmark_min_auc")),
                    _fmt(row.get("benchmark_min_first_edge_accuracy")),
                    _fmt(row.get("benchmark_min_terminal_top1_accuracy")),
                    "1" if bool(row.get("strict_rl_ready")) else "0",
                    str(row.get("assessment", "")),
                ]
            )
            + " |"
        )
        failure_modes = row.get("failure_modes", [])
        if failure_modes:
            lines.append(f"  failure_modes: `{', '.join(str(item) for item in failure_modes)}`")

    lines.extend(
        [
            "",
            "## Benchmark Structure",
            "",
            "| audit_id | benchmark | auc | first_edge | terminal_top1 | terminal_gap | completion_penalty | len_ratio | collapse_ratio | over_limit | failure_tags |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in benchmark_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["audit_id"]),
                    str(row["benchmark_name"]),
                    _fmt(row.get("benchmark_auc")),
                    _fmt(row.get("first_edge_accuracy")),
                    _fmt(row.get("all_correct_terminal_top1_accuracy")),
                    _fmt(row.get("all_correct_terminal_gap_mean")),
                    _fmt(row.get("completion_penalty")),
                    _fmt(row.get("length_ratio_vs_samefamily")),
                    _fmt(row.get("margin_collapse_ratio")),
                    _fmt(row.get("benchmark_over_limit_fraction")),
                    ", ".join(str(item) for item in row.get("failure_tags", [])) or "-",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _allocate_run_dir(*, output_root: Path, run_name: str) -> Path:
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)
    candidates = sorted(output_root.glob(f"{run_name}_*"))
    index = len(candidates)
    while True:
        candidate = output_root / f"{run_name}_{index:02d}"
        if not candidate.exists():
            return candidate
        index += 1


if __name__ == "__main__":
    raise SystemExit(main())
