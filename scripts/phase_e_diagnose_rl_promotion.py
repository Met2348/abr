#!/usr/bin/env python3
"""Diagnose whether Phase E candidates are close to an RL-promotion gate.

English
-------
The repository already has:
1. same-family trust reports,
2. ProcessBench benchmark summaries,
3. and one stricter transfer diagnosis.

What was still missing is a *slice-aware promotion gate* that answers the
practical question:
1. if we wanted to promote one value head into conservative RL-style use,
2. which exact slice still blocks promotion,
3. and which repair direction is helping versus over-correcting?

This script treats RL promotion as a conjunction of several offline canaries:
1. same-family prompt-level utility must still be strong,
2. ProcessBench local first-bad behavior must survive,
3. broader later-bad ranking cannot collapse,
4. all-correct terminal completion cannot stay systematically undervalued.

中文
----
仓库里已经有：
1. same-family trust 报告，
2. ProcessBench benchmark 摘要，
3. 以及一版更严格的 transfer diagnosis。

但还缺一层更贴近“能不能晋升为 RL 候选”的 slice 级 gate。这个脚本专门回答：
1. 如果现在要把某个 value head 提升到保守 RL 用途，
2. 到底是哪一类 slice 还在拦路，
3. 哪种 repair 是真修复，哪种只是过度修正？

这里把 RL promotion 看成几条离线 canary 的合取：
1. same-family prompt-level utility 不能掉，
2. ProcessBench 上的局部 first-bad 不能塌，
3. later-bad 这类更非局部的排序不能坏掉，
4. all-correct terminal completion 也不能继续被系统性低估。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
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
    ProcessBenchPrefixRecord,
    build_processbench_prefix_records,
    load_processbench_examples,
)
from ours.phase_e.processbench_alignment import summarize_scored_rows_alignment  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a slice-aware RL promotion diagnosis for one or more Phase E candidates."
    )
    parser.add_argument(
        "--audit-spec",
        action="append",
        required=True,
        metavar="AUDIT_ID|SOURCE_FAMILY|SAMEFAMILY_DIR|PB_GSM_DIR|PB_MATH_DIR",
        help=(
            "One candidate audit. Repeat this flag for multiple candidates. "
            "Each directory should point to an existing artifact run directory."
        ),
    )
    parser.add_argument("--run-name", default="phase_e_rl_promotion_diag")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_rl_promotion_diag"),
    )
    parser.add_argument("--samefamily-top1-min", type=float, default=0.90)
    parser.add_argument("--samefamily-local-min", type=float, default=0.90)
    parser.add_argument("--pressure-top1-min", type=float, default=0.90)
    parser.add_argument("--benchmark-auc-min", type=float, default=0.60)
    parser.add_argument("--first-edge-min", type=float, default=0.60)
    parser.add_argument("--anygood-firstbad-min", type=float, default=0.60)
    parser.add_argument("--good-laterbad-min", type=float, default=0.60)
    parser.add_argument("--terminal-top1-min", type=float, default=0.50)
    parser.add_argument("--terminal-gap-min", type=float, default=-0.05)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.audit_spec:
        raise ValueError("At least one --audit-spec is required")
    for name in (
        "samefamily_top1_min",
        "samefamily_local_min",
        "pressure_top1_min",
        "benchmark_auc_min",
        "first_edge_min",
        "anygood_firstbad_min",
        "good_laterbad_min",
        "terminal_top1_min",
    ):
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = _allocate_run_dir(output_root=Path(args.output_root), run_name=str(args.run_name))
    run_dir.mkdir(parents=True, exist_ok=False)

    thresholds = {
        "samefamily_top1_min": float(args.samefamily_top1_min),
        "samefamily_local_min": float(args.samefamily_local_min),
        "pressure_top1_min": float(args.pressure_top1_min),
        "benchmark_auc_min": float(args.benchmark_auc_min),
        "first_edge_min": float(args.first_edge_min),
        "anygood_firstbad_min": float(args.anygood_firstbad_min),
        "good_laterbad_min": float(args.good_laterbad_min),
        "terminal_top1_min": float(args.terminal_top1_min),
        "terminal_gap_min": float(args.terminal_gap_min),
    }

    audit_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    for raw_spec in args.audit_spec:
        audit_row, bench_rows = _run_one_audit(spec_text=str(raw_spec), thresholds=thresholds)
        audit_rows.append(audit_row)
        benchmark_rows.extend(bench_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "thresholds": thresholds,
        "audits": audit_rows,
        "benchmarks": benchmark_rows,
    }
    summary_json_path = run_dir / "summary.json"
    audit_rows_path = run_dir / "audit_rows.jsonl"
    benchmark_rows_path = run_dir / "benchmark_rows.jsonl"
    summary_md_path = run_dir / "summary.md"

    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_jsonl(audit_rows_path, audit_rows)
    _write_jsonl(benchmark_rows_path, benchmark_rows)
    summary_md_path.write_text(
        _render_summary_markdown(thresholds=thresholds, audit_rows=audit_rows, benchmark_rows=benchmark_rows),
        encoding="utf-8",
    )

    print("=" * 88)
    print("Phase E RL Promotion Diagnosis")
    print("=" * 88)
    for row in audit_rows:
        print(
            f"{row['audit_id']:>18} | gate={int(bool(row['strict_rl_promotion_ready']))} "
            f"| samefamily_top1={row['samefamily_top1']:.4f} "
            f"| pb_min_auc={row['benchmark_min_auc']:.4f} "
            f"| pb_min_laterbad={row['benchmark_min_good_vs_laterbad_accuracy']:.4f} "
            f"| pb_min_terminal_top1={row['benchmark_min_terminal_top1_accuracy']:.4f} "
            f"| assessment={row['assessment']}"
        )
    print(f"summary_json : {summary_json_path}")
    print(f"summary_md   : {summary_md_path}")
    print("=" * 88)
    return 0


def _run_one_audit(
    *,
    spec_text: str,
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parts = [part.strip() for part in str(spec_text).split("|")]
    if len(parts) != 5:
        raise ValueError(
            "--audit-spec must have five pipe-separated parts: "
            "AUDIT_ID|SOURCE_FAMILY|SAMEFAMILY_DIR|PB_GSM_DIR|PB_MATH_DIR"
        )
    audit_id, source_family, samefamily_dir_text, pb_gsm_dir_text, pb_math_dir_text = parts
    if audit_id == "" or source_family == "":
        raise ValueError(f"Invalid audit spec: {spec_text}")

    samefamily_dir = Path(samefamily_dir_text)
    pb_gsm_dir = Path(pb_gsm_dir_text)
    pb_math_dir = Path(pb_math_dir_text)
    for path in (samefamily_dir, pb_gsm_dir, pb_math_dir):
        if not path.exists():
            raise FileNotFoundError(f"Audit directory not found: {path}")

    samefamily_metrics = _load_json(samefamily_dir / "metrics.json")
    samefamily_manifest = _load_json(samefamily_dir / "manifest.json")
    prompt_rows = _load_jsonl(samefamily_dir / "prompt_rows.jsonl")

    samefamily_top1 = float(samefamily_metrics.get("prompt_pool_top1_accuracy", 0.0))
    samefamily_random_top1 = _compute_random_top1(prompt_rows)
    samefamily_pressure8 = _extract_pressure_top1(samefamily_metrics, subset_size=8)
    samefamily_local = _maybe_float(
        samefamily_metrics.get("local_first_bad_edge_accuracy")
        or samefamily_metrics.get("local_safe_vs_bad_pair_accuracy")
    )
    samefamily_green = (
        samefamily_top1 >= thresholds["samefamily_top1_min"]
        and (samefamily_pressure8 is None or samefamily_pressure8 >= thresholds["pressure_top1_min"])
        and (samefamily_local is None or samefamily_local >= thresholds["samefamily_local_min"])
    )

    benchmark_rows = [
        _analyze_benchmark(
            audit_id=audit_id,
            source_family=source_family,
            benchmark_name="processbench_gsm8k",
            benchmark_dir=pb_gsm_dir,
            thresholds=thresholds,
        ),
        _analyze_benchmark(
            audit_id=audit_id,
            source_family=source_family,
            benchmark_name="processbench_math",
            benchmark_dir=pb_math_dir,
            thresholds=thresholds,
        ),
    ]
    benchmark_green = all(bool(row["benchmark_green"]) for row in benchmark_rows)
    strict_rl_promotion_ready = bool(samefamily_green and benchmark_green and samefamily_local is not None)

    failure_tags = set()
    if samefamily_top1 < thresholds["samefamily_top1_min"]:
        failure_tags.add("samefamily_top1_weak")
    if samefamily_pressure8 is not None and samefamily_pressure8 < thresholds["pressure_top1_min"]:
        failure_tags.add("samefamily_pressure_weak")
    if samefamily_local is None:
        failure_tags.add("samefamily_local_missing")
    elif samefamily_local < thresholds["samefamily_local_min"]:
        failure_tags.add("samefamily_local_weak")
    for row in benchmark_rows:
        failure_tags.update(str(tag) for tag in row["failure_tags"])

    if strict_rl_promotion_ready:
        assessment = "strict_rl_promotion_ready"
    elif any(tag == "terminal_completion_weak" for tag in failure_tags):
        if all(tag not in failure_tags for tag in ("benchmark_first_edge_weak", "benchmark_good_laterbad_weak")):
            assessment = "near_rl_ready_but_terminal_gap"
        else:
            assessment = "terminal_and_local_tradeoff_unresolved"
    elif any(tag == "benchmark_good_laterbad_weak" for tag in failure_tags):
        assessment = "not_rl_ready_laterbad_generalization_weak"
    elif any(tag == "benchmark_first_edge_weak" for tag in failure_tags):
        assessment = "not_rl_ready_local_error_detection_weak"
    elif samefamily_green:
        assessment = "samefamily_strong_but_benchmark_incomplete"
    else:
        assessment = "not_rl_ready"

    audit_row = {
        "audit_id": audit_id,
        "source_family": source_family,
        "samefamily_dir": str(samefamily_dir),
        "samefamily_manifest_value_run_dir": str(samefamily_manifest.get("value_run_dir") or ""),
        "samefamily_top1": samefamily_top1,
        "samefamily_random_top1": samefamily_random_top1,
        "samefamily_top1_lift": float(samefamily_top1 - samefamily_random_top1),
        "samefamily_pressure8_top1": samefamily_pressure8,
        "samefamily_local_accuracy": samefamily_local,
        "benchmark_min_auc": float(min(row["benchmark_auc"] for row in benchmark_rows)),
        "benchmark_min_first_edge_accuracy": float(min(row["first_edge_accuracy"] for row in benchmark_rows)),
        "benchmark_min_anygood_vs_firstbad_accuracy": float(
            min(row["anygood_vs_firstbad_accuracy"] for row in benchmark_rows)
        ),
        "benchmark_min_good_vs_laterbad_accuracy": float(
            min(row["good_vs_laterbad_accuracy"] for row in benchmark_rows)
        ),
        "benchmark_min_terminal_top1_accuracy": float(
            min(row["terminal_top1_accuracy"] for row in benchmark_rows)
        ),
        "benchmark_min_terminal_gap_mean": float(min(row["terminal_gap_mean"] for row in benchmark_rows)),
        "samefamily_green": bool(samefamily_green),
        "benchmark_green": bool(benchmark_green),
        "strict_rl_promotion_ready": bool(strict_rl_promotion_ready),
        "failure_tags": sorted(failure_tags),
        "assessment": assessment,
    }
    return audit_row, benchmark_rows


def _analyze_benchmark(
    *,
    audit_id: str,
    source_family: str,
    benchmark_name: str,
    benchmark_dir: Path,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    summary_payload = _load_json(benchmark_dir / "summary.json")
    benchmark_path = Path(summary_payload["benchmark_path"])
    scored_rows_path = benchmark_dir / "scored_rows.jsonl"
    slice_summary = summarize_scored_rows_alignment(
        scored_rows_path=scored_rows_path,
        processbench_path=benchmark_path,
    )
    joined_rows = _load_joined_processbench_rows(
        benchmark_path=benchmark_path,
        scored_rows_path=scored_rows_path,
    )
    terminal_metrics = _summarize_all_correct_examples(joined_rows)
    benchmark_auc = float(summary_payload["metrics"]["pair_auc_good_vs_bad"])
    first_edge_accuracy = float(slice_summary["first_error_edge_accuracy"])
    anygood_vs_firstbad_accuracy = float(
        slice_summary["aggregate_metrics"]["anygood_vs_firstbad"]["accuracy"]
    )
    good_vs_laterbad_accuracy = float(
        slice_summary["aggregate_metrics"]["good_vs_laterbad"]["accuracy"]
    )
    terminal_top1_accuracy = float(terminal_metrics["terminal_top1_accuracy"])
    terminal_gap_mean = float(terminal_metrics["terminal_gap_mean"])

    failure_tags: list[str] = []
    if benchmark_auc < thresholds["benchmark_auc_min"]:
        failure_tags.append("benchmark_auc_weak")
    if first_edge_accuracy < thresholds["first_edge_min"]:
        failure_tags.append("benchmark_first_edge_weak")
    if anygood_vs_firstbad_accuracy < thresholds["anygood_firstbad_min"]:
        failure_tags.append("benchmark_anygood_firstbad_weak")
    if good_vs_laterbad_accuracy < thresholds["good_laterbad_min"]:
        failure_tags.append("benchmark_good_laterbad_weak")
    if (
        terminal_top1_accuracy < thresholds["terminal_top1_min"]
        or terminal_gap_mean < thresholds["terminal_gap_min"]
    ):
        failure_tags.append("terminal_completion_weak")

    benchmark_green = len(failure_tags) == 0
    return {
        "audit_id": audit_id,
        "source_family": source_family,
        "benchmark_name": benchmark_name,
        "benchmark_dir": str(benchmark_dir),
        "benchmark_auc": benchmark_auc,
        "first_edge_accuracy": first_edge_accuracy,
        "anygood_vs_firstbad_accuracy": anygood_vs_firstbad_accuracy,
        "good_vs_laterbad_accuracy": good_vs_laterbad_accuracy,
        "terminal_top1_accuracy": terminal_top1_accuracy,
        "terminal_gap_mean": terminal_gap_mean,
        "benchmark_green": bool(benchmark_green),
        "failure_tags": failure_tags,
    }


def _load_joined_processbench_rows(
    *,
    benchmark_path: Path,
    scored_rows_path: Path,
) -> list[tuple[ProcessBenchPrefixRecord, float]]:
    scored_rows = _load_jsonl(scored_rows_path)
    score_by_row_id = {str(row["row_id"]): float(row["score"]) for row in scored_rows}
    examples = load_processbench_examples(benchmark_path)
    rows = build_processbench_prefix_records(examples)
    joined_rows = []
    for row in rows:
        score = score_by_row_id.get(str(row.row_id))
        if score is None:
            continue
        joined_rows.append((row, score))
    if not joined_rows:
        raise RuntimeError(f"No overlapping ProcessBench rows found in {scored_rows_path}")
    return joined_rows


def _summarize_all_correct_examples(
    joined_rows: list[tuple[ProcessBenchPrefixRecord, float]],
) -> dict[str, float]:
    grouped: dict[str, list[tuple[ProcessBenchPrefixRecord, float]]] = {}
    for row, score in joined_rows:
        grouped.setdefault(str(row.example_id), []).append((row, float(score)))

    terminal_top1_total = 0
    terminal_top1_positive = 0
    terminal_gaps: list[float] = []
    for example_rows in grouped.values():
        example_rows.sort(key=lambda item: item[0].prefix_step_index)
        if int(example_rows[0][0].label) >= 0:
            continue
        final_score = float(example_rows[-1][1])
        prev_scores = [float(score) for _, score in example_rows[:-1]]
        prev_max = max(prev_scores) if prev_scores else final_score
        terminal_top1_total += 1
        if final_score >= prev_max:
            terminal_top1_positive += 1
        terminal_gaps.append(float(final_score - prev_max))

    return {
        "terminal_top1_accuracy": (
            float(terminal_top1_positive / terminal_top1_total) if terminal_top1_total > 0 else 0.0
        ),
        "terminal_gap_mean": float(statistics.mean(terminal_gaps)) if terminal_gaps else 0.0,
    }


def _extract_pressure_top1(metrics: dict[str, Any], *, subset_size: int) -> float | None:
    for point in metrics.get("pressure_curve", []):
        if int(point.get("subset_size", -1)) == int(subset_size):
            return float(point.get("top1_accuracy", 0.0))
    return None


def _compute_random_top1(prompt_rows: list[dict[str, Any]]) -> float:
    if not prompt_rows:
        return 0.0
    values = []
    for row in prompt_rows:
        num_candidates = max(int(row.get("num_candidates", 1)), 1)
        num_gold = len(row.get("gold_top_candidate_ids", []))
        values.append(float(num_gold / num_candidates))
    return float(sum(values) / len(values)) if values else 0.0


def _allocate_run_dir(*, output_root: Path, run_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_root / f"{run_name}_{timestamp}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt_opt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _render_summary_markdown(
    *,
    thresholds: dict[str, float],
    audit_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase E RL Promotion Diagnosis",
        "",
        "## Thresholds",
        "",
        f"- samefamily_top1_min: {thresholds['samefamily_top1_min']:.4f}",
        f"- samefamily_local_min: {thresholds['samefamily_local_min']:.4f}",
        f"- pressure_top1_min: {thresholds['pressure_top1_min']:.4f}",
        f"- benchmark_auc_min: {thresholds['benchmark_auc_min']:.4f}",
        f"- first_edge_min: {thresholds['first_edge_min']:.4f}",
        f"- anygood_firstbad_min: {thresholds['anygood_firstbad_min']:.4f}",
        f"- good_laterbad_min: {thresholds['good_laterbad_min']:.4f}",
        f"- terminal_top1_min: {thresholds['terminal_top1_min']:.4f}",
        f"- terminal_gap_min: {thresholds['terminal_gap_min']:.4f}",
        "",
        "## Audit Table",
        "",
        "| audit_id | gate | samefamily_top1 | pressure8 | local | pb_min_auc | pb_min_first_edge | pb_min_laterbad | pb_min_terminal_top1 | pb_min_terminal_gap | assessment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in audit_rows:
        lines.append(
            "| {audit_id} | {gate} | {samefamily_top1} | {pressure8} | {local} | {pb_auc} | {pb_first} | {pb_later} | {pb_terminal} | {pb_gap} | {assessment} |".format(
                audit_id=row["audit_id"],
                gate=int(bool(row["strict_rl_promotion_ready"])),
                samefamily_top1=f"{row['samefamily_top1']:.4f}",
                pressure8=_fmt_opt(row["samefamily_pressure8_top1"]),
                local=_fmt_opt(row["samefamily_local_accuracy"]),
                pb_auc=f"{row['benchmark_min_auc']:.4f}",
                pb_first=f"{row['benchmark_min_first_edge_accuracy']:.4f}",
                pb_later=f"{row['benchmark_min_good_vs_laterbad_accuracy']:.4f}",
                pb_terminal=f"{row['benchmark_min_terminal_top1_accuracy']:.4f}",
                pb_gap=f"{row['benchmark_min_terminal_gap_mean']:.4f}",
                assessment=row["assessment"],
            )
        )
    lines.extend(
        [
            "",
            "## Benchmark Slices",
            "",
            "| audit_id | benchmark | auc | first_edge | anygood_vs_firstbad | good_vs_laterbad | terminal_top1 | terminal_gap | failure_tags |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in benchmark_rows:
        lines.append(
            "| {audit_id} | {benchmark} | {auc:.4f} | {first_edge:.4f} | {anygood:.4f} | {laterbad:.4f} | {terminal_top1:.4f} | {terminal_gap:.4f} | {tags} |".format(
                audit_id=row["audit_id"],
                benchmark=row["benchmark_name"],
                auc=row["benchmark_auc"],
                first_edge=row["first_edge_accuracy"],
                anygood=row["anygood_vs_firstbad_accuracy"],
                laterbad=row["good_vs_laterbad_accuracy"],
                terminal_top1=row["terminal_top1_accuracy"],
                terminal_gap=row["terminal_gap_mean"],
                tags=",".join(row["failure_tags"]) if row["failure_tags"] else "clean",
            )
        )
    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `samefamily_top1` and `pressure8` guard against promoting a candidate whose own-family rerank utility is unstable.",
            "- `good_vs_laterbad` is the main non-local canary; it catches candidates that only know the easiest first-bad edge.",
            "- `terminal_top1` and `terminal_gap` expose all-correct completion undervaluation or over-correction directly.",
            "- A candidate is only `gate=1` when all of those slices stay above threshold together.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
