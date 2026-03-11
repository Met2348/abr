#!/usr/bin/env python3
"""Analyze ProcessBench failure structure against one training pair artifact.

English
-------
This script exists because a single benchmark AUC is not enough to explain
why same-source ranking can look strong while ProcessBench transfer remains
weak.  It therefore makes the mismatch explicit from two sides:

1. training-pair semantics
   - local first-bad-edge vs terminal-anchor vs direct pair
2. ProcessBench evaluation structure
   - all-correct examples
   - early / mid / late first-error positions
   - step-count buckets

The output is a compact diagnostic report that can be compared across
checkpoints and data recipes.

中文
----
单个 benchmark AUC 并不能解释：
为什么同源 held-out 排序已经很强，但到了 ProcessBench 还是迁移困难。

因此这个脚本会把错位从两侧显式拆出来：

1. 训练监督到底在教什么
   - local first-bad-edge
   - terminal-anchor
   - direct pair
2. ProcessBench 到底在考什么
   - all-correct 完整终点
   - first error 的 early / mid / late 位置
   - 长短轨迹分布

输出是可直接跨 checkpoint / 数据配方比较的诊断报告。
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

from ours.phase_d.external_pairs import load_external_pair_jsonl  # noqa: E402
from ours.phase_e.benchmark_eval import (  # noqa: E402
    ProcessBenchPrefixRecord,
    build_processbench_prefix_records,
    load_processbench_examples,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze ProcessBench failure structure against one Phase E training pair artifact."
    )
    parser.add_argument("--value-run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-eval-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="phase_e_processbench_failure_analysis")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_processbench_analysis"),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.value_run_dir.exists():
        raise FileNotFoundError(f"--value-run-dir not found: {args.value_run_dir}")
    if not args.benchmark_eval_dir.exists():
        raise FileNotFoundError(f"--benchmark-eval-dir not found: {args.benchmark_eval_dir}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    value_manifest = _load_json(Path(args.value_run_dir) / "manifest.json")
    bench_summary = _load_json(Path(args.benchmark_eval_dir) / "summary.json")
    scored_rows = _load_jsonl(Path(args.benchmark_eval_dir) / "scored_rows.jsonl")

    train_pairs_path = Path(value_manifest["input_files"]["train_pairs_jsonl"])
    benchmark_path = Path(bench_summary["benchmark_path"])
    scored_example_ids = {str(row["example_id"]) for row in scored_rows}

    train_pairs, train_pair_summary = load_external_pair_jsonl(train_pairs_path)
    examples = [
        example
        for example in load_processbench_examples(benchmark_path)
        if str(example.example_id) in scored_example_ids
    ]
    prefix_rows = build_processbench_prefix_records(examples)
    if len(prefix_rows) != len(scored_rows):
        raise RuntimeError("ProcessBench prefix rows and scored_rows length mismatch")
    score_by_row_id = {str(row["row_id"]): float(row["score"]) for row in scored_rows}
    joined_rows = []
    for row in prefix_rows:
        score = score_by_row_id.get(str(row.row_id))
        if score is None:
            continue
        joined_rows.append((row, score))

    run_dir = _allocate_run_dir(Path(args.output_root), str(args.run_name))
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = _build_summary(
        value_run_dir=Path(args.value_run_dir),
        benchmark_eval_dir=Path(args.benchmark_eval_dir),
        benchmark_path=benchmark_path,
        train_pairs=train_pairs,
        train_pair_summary=train_pair_summary,
        joined_rows=joined_rows,
        raw_metrics=dict(bench_summary.get("metrics", {})),
    )
    summary_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    example_rows_path = run_dir / "example_bucket_rows.jsonl"
    prefix_rows_path = run_dir / "prefix_bucket_rows.jsonl"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")
    _write_jsonl(example_rows_path, summary["example_bucket_rows"])
    _write_jsonl(prefix_rows_path, summary["prefix_bucket_rows"])

    print("=" * 88)
    print("Phase E ProcessBench Failure Analysis")
    print("=" * 88)
    print(f"value_run_dir        : {args.value_run_dir}")
    print(f"benchmark_eval_dir   : {args.benchmark_eval_dir}")
    print(f"benchmark_id         : {summary['benchmark_id']}")
    print(f"pair_semantics       : {summary['train_summary']['by_pair_semantics']}")
    print(f"terminal_anchor_frac : {summary['mismatch']['train_terminal_anchor_fraction']:.4f}")
    print(f"local_pair_frac      : {summary['mismatch']['train_local_error_fraction']:.4f}")
    print(f"all_correct_ratio    : {summary['benchmark_structure']['all_correct_example_fraction']:.4f}")
    print(f"late_error_ratio     : {summary['benchmark_structure']['late_error_fraction']:.4f}")
    print(f"summary_md           : {summary_md_path}")
    print("=" * 88)
    return 0


def _build_summary(
    *,
    value_run_dir: Path,
    benchmark_eval_dir: Path,
    benchmark_path: Path,
    train_pairs: list[Any],
    train_pair_summary: dict[str, Any],
    joined_rows: list[tuple[ProcessBenchPrefixRecord, float]],
    raw_metrics: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[ProcessBenchPrefixRecord, float]]] = {}
    for row, score in joined_rows:
        grouped.setdefault(str(row.example_id), []).append((row, float(score)))

    example_bucket_rows = []
    prefix_bucket_rows = []
    bucket_payloads: dict[str, list[list[tuple[ProcessBenchPrefixRecord, float]]]] = {}
    prefix_payloads: dict[str, list[tuple[ProcessBenchPrefixRecord, float]]] = {}

    all_correct_examples = 0
    early_errors = 0
    mid_errors = 0
    late_errors = 0
    step_counts = []

    for example_rows in grouped.values():
        example_rows.sort(key=lambda item: item[0].prefix_step_index)
        label = int(example_rows[0][0].label)
        num_steps = int(len(example_rows))
        step_counts.append(num_steps)
        bucket = _bucket_example(label=label, num_steps=num_steps)
        if bucket == "all_correct":
            all_correct_examples += 1
        elif bucket == "early_error":
            early_errors += 1
        elif bucket == "mid_error":
            mid_errors += 1
        elif bucket == "late_error":
            late_errors += 1
        bucket_payloads.setdefault(bucket, []).append(example_rows)

        for row, score in example_rows:
            prefix_payloads.setdefault(_bucket_prefix(row=row, num_steps=num_steps), []).append((row, score))

    for bucket, payload in sorted(bucket_payloads.items()):
        example_bucket_rows.append(_summarize_example_bucket(bucket=bucket, payload=payload))
    for bucket, payload in sorted(prefix_payloads.items()):
        prefix_bucket_rows.append(_summarize_prefix_bucket(bucket=bucket, payload=payload))

    num_examples = int(len(grouped))
    benchmark_structure = {
        "num_examples": num_examples,
        "mean_num_steps": float(statistics.mean(step_counts)) if step_counts else 0.0,
        "p95_num_steps": int(_p_quantile(step_counts, 0.95)) if step_counts else 0,
        "all_correct_example_fraction": _safe_fraction(all_correct_examples, num_examples),
        "early_error_fraction": _safe_fraction(early_errors, num_examples),
        "mid_error_fraction": _safe_fraction(mid_errors, num_examples),
        "late_error_fraction": _safe_fraction(late_errors, num_examples),
    }

    pair_semantics = dict(train_pair_summary.get("by_pair_semantics", {}))
    local_pairs = int(pair_semantics.get("local_first_bad_edge", 0))
    local_modified_pairs = int(pair_semantics.get("local_modified_process_error_step", 0))
    fanout_pairs = int(pair_semantics.get("first_bad_fanout_prefix_ranking", 0))
    grid_pairs = int(pair_semantics.get("good_bad_prefix_grid", 0))
    terminal_pairs = int(train_pair_summary.get("by_pair_semantics", {}).get("terminal_completion_anchor", 0))
    direct_pairs = int(pair_semantics.get("direct_preference_pair", 0))
    same_prompt_binary = int(pair_semantics.get("same_prompt_binary_verdict", 0))
    same_prompt_correctness = int(pair_semantics.get("same_prompt_binary_correctness", 0))
    num_train_pairs = int(train_pair_summary.get("num_pairs", 0))
    local_error_pairs = int(local_pairs + local_modified_pairs + fanout_pairs)

    train_prompt_chars = [len(str(pair.prompt_text)) for pair in train_pairs]
    train_chosen_chars = [len(str(pair.chosen_text)) for pair in train_pairs]
    train_rejected_chars = [len(str(pair.rejected_text)) for pair in train_pairs]

    mismatch = {
        "train_local_first_bad_fraction": _safe_fraction(local_pairs, num_train_pairs),
        "train_local_modified_fraction": _safe_fraction(local_modified_pairs, num_train_pairs),
        "train_first_bad_fanout_fraction": _safe_fraction(fanout_pairs, num_train_pairs),
        "train_good_bad_grid_fraction": _safe_fraction(grid_pairs, num_train_pairs),
        "train_local_error_fraction": _safe_fraction(local_error_pairs, num_train_pairs),
        "train_terminal_anchor_fraction": _safe_fraction(terminal_pairs, num_train_pairs),
        "train_direct_preference_fraction": _safe_fraction(direct_pairs, num_train_pairs),
        "train_same_prompt_binary_fraction": _safe_fraction(
            same_prompt_binary + same_prompt_correctness,
            num_train_pairs,
        ),
        "all_correct_supervision_gap": bool(
            benchmark_structure["all_correct_example_fraction"] >= 0.30
            and _safe_fraction(terminal_pairs, num_train_pairs) < 0.05
        ),
        "late_error_coverage_gap": bool(
            benchmark_structure["late_error_fraction"] >= 0.15
            and _safe_fraction(local_error_pairs + grid_pairs, num_train_pairs) < 0.20
        ),
        "local_only_without_terminal_gap": bool(
            _safe_fraction(local_error_pairs, num_train_pairs) >= 0.50
            and _safe_fraction(terminal_pairs, num_train_pairs) < 0.05
        ),
        "train_prompt_chars_p95": int(_p_quantile(train_prompt_chars, 0.95)) if train_prompt_chars else 0,
        "train_chosen_chars_p95": int(_p_quantile(train_chosen_chars, 0.95)) if train_chosen_chars else 0,
        "train_rejected_chars_p95": int(_p_quantile(train_rejected_chars, 0.95)) if train_rejected_chars else 0,
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "value_run_dir": str(value_run_dir),
        "benchmark_eval_dir": str(benchmark_eval_dir),
        "benchmark_id": str(bench_summary_benchmark_id(raw_metrics=raw_metrics, benchmark_eval_dir=benchmark_eval_dir)),
        "benchmark_path": str(benchmark_path),
        "raw_metrics": raw_metrics,
        "train_summary": {
            **train_pair_summary,
            "train_prompt_chars_p95": mismatch["train_prompt_chars_p95"],
            "train_chosen_chars_p95": mismatch["train_chosen_chars_p95"],
            "train_rejected_chars_p95": mismatch["train_rejected_chars_p95"],
        },
        "benchmark_structure": benchmark_structure,
        "mismatch": mismatch,
        "example_bucket_rows": example_bucket_rows,
        "prefix_bucket_rows": prefix_bucket_rows,
    }


def _bucket_example(*, label: int, num_steps: int) -> str:
    if label < 0:
        return "all_correct"
    if label == 0:
        return "error_at_step0"
    if num_steps <= 1:
        return "early_error"
    frac = float(label / max(num_steps - 1, 1))
    if frac <= (1.0 / 3.0):
        return "early_error"
    if frac >= (2.0 / 3.0):
        return "late_error"
    return "mid_error"


def _bucket_prefix(*, row: ProcessBenchPrefixRecord, num_steps: int) -> str:
    if row.label < 0:
        if row.prefix_step_index == (num_steps - 1):
            return "all_correct_terminal"
        return "all_correct_nonterminal"
    if row.is_first_bad_prefix:
        return "first_bad_prefix"
    if row.is_good_prefix:
        return "safe_prefix"
    return "later_bad_prefix"


def _summarize_example_bucket(
    *,
    bucket: str,
    payload: list[list[tuple[ProcessBenchPrefixRecord, float]]],
) -> dict[str, Any]:
    pair_total = 0
    pair_positive = 0
    first_edge_total = 0
    first_edge_positive = 0
    terminal_top1_total = 0
    terminal_top1_positive = 0
    terminal_gaps = []
    mean_scores = []

    for example_rows in payload:
        good = [float(score) for row, score in example_rows if row.is_good_prefix]
        bad = [float(score) for row, score in example_rows if not row.is_good_prefix]
        for g in good:
            for b in bad:
                pair_total += 1
                if g > b:
                    pair_positive += 1
        label = int(example_rows[0][0].label)
        if label > 0 and label < len(example_rows):
            prev_score = float(example_rows[label - 1][1])
            bad_score = float(example_rows[label][1])
            first_edge_total += 1
            if prev_score > bad_score:
                first_edge_positive += 1
        if label < 0:
            final_score = float(example_rows[-1][1])
            prev_scores = [float(score) for _, score in example_rows[:-1]]
            prev_max = max(prev_scores) if prev_scores else final_score
            terminal_top1_total += 1
            if final_score >= prev_max:
                terminal_top1_positive += 1
            terminal_gaps.append(float(final_score - prev_max))
        mean_scores.extend(float(score) for _, score in example_rows)

    return {
        "bucket": bucket,
        "num_examples": int(len(payload)),
        "pair_accuracy_good_vs_bad": float(pair_positive / pair_total) if pair_total > 0 else None,
        "num_good_bad_pairs": int(pair_total),
        "first_error_edge_accuracy": float(first_edge_positive / first_edge_total) if first_edge_total > 0 else None,
        "num_first_error_edges": int(first_edge_total),
        "all_correct_terminal_top1_accuracy": (
            float(terminal_top1_positive / terminal_top1_total) if terminal_top1_total > 0 else None
        ),
        "all_correct_terminal_gap_mean": (
            float(statistics.mean(terminal_gaps)) if terminal_gaps else None
        ),
        "mean_prefix_score": float(statistics.mean(mean_scores)) if mean_scores else 0.0,
    }


def _summarize_prefix_bucket(
    *,
    bucket: str,
    payload: list[tuple[ProcessBenchPrefixRecord, float]],
) -> dict[str, Any]:
    scores = [float(score) for _, score in payload]
    step_indices = [int(row.prefix_step_index) for row, _ in payload]
    labels = [int(row.label) for row, _ in payload]
    return {
        "bucket": bucket,
        "num_prefixes": int(len(payload)),
        "mean_score": float(statistics.mean(scores)) if scores else 0.0,
        "p10_score": float(_p_quantile(scores, 0.10)) if scores else 0.0,
        "p90_score": float(_p_quantile(scores, 0.90)) if scores else 0.0,
        "mean_step_index": float(statistics.mean(step_indices)) if step_indices else 0.0,
        "label_min": int(min(labels)) if labels else 0,
        "label_max": int(max(labels)) if labels else 0,
    }


def bench_summary_benchmark_id(*, raw_metrics: dict[str, Any], benchmark_eval_dir: Path) -> str:
    summary_path = benchmark_eval_dir / "summary.json"
    payload = _load_json(summary_path)
    return str(payload.get("benchmark_id", benchmark_eval_dir.name))


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase E ProcessBench Failure Analysis",
        "",
        f"- value_run_dir: `{summary['value_run_dir']}`",
        f"- benchmark_eval_dir: `{summary['benchmark_eval_dir']}`",
        f"- benchmark_id: `{summary['benchmark_id']}`",
        f"- benchmark_path: `{summary['benchmark_path']}`",
        "",
        "## Raw Benchmark Metrics",
    ]
    raw_metrics = dict(summary["raw_metrics"])
    for key in (
        "pair_accuracy_good_vs_bad",
        "pair_auc_good_vs_bad",
        "first_error_edge_accuracy",
        "mean_good_prefix_score",
        "mean_bad_prefix_score",
        "mean_all_correct_last_score",
        "num_examples",
        "num_all_correct_examples",
    ):
        if key in raw_metrics:
            value = raw_metrics[key]
            if isinstance(value, float):
                lines.append(f"- {key}: `{value:.6f}`")
            else:
                lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Training Pair Semantics",
            "",
            f"- by_pair_semantics: `{summary['train_summary'].get('by_pair_semantics', {})}`",
            f"- by_pair_build_mode: `{summary['train_summary'].get('by_pair_build_mode', {})}`",
            f"- train_prompt_chars_p95: `{summary['train_summary'].get('train_prompt_chars_p95')}`",
            f"- train_chosen_chars_p95: `{summary['train_summary'].get('train_chosen_chars_p95')}`",
            f"- train_rejected_chars_p95: `{summary['train_summary'].get('train_rejected_chars_p95')}`",
            "",
            "## ProcessBench Structure",
            "",
            f"- all_correct_example_fraction: `{summary['benchmark_structure']['all_correct_example_fraction']:.6f}`",
            f"- early_error_fraction: `{summary['benchmark_structure']['early_error_fraction']:.6f}`",
            f"- mid_error_fraction: `{summary['benchmark_structure']['mid_error_fraction']:.6f}`",
            f"- late_error_fraction: `{summary['benchmark_structure']['late_error_fraction']:.6f}`",
            f"- mean_num_steps: `{summary['benchmark_structure']['mean_num_steps']:.6f}`",
            f"- p95_num_steps: `{summary['benchmark_structure']['p95_num_steps']}`",
            "",
            "## Mismatch Flags",
            "",
            f"- train_local_first_bad_fraction: `{summary['mismatch']['train_local_first_bad_fraction']:.6f}`",
            f"- train_local_modified_fraction: `{summary['mismatch']['train_local_modified_fraction']:.6f}`",
            f"- train_first_bad_fanout_fraction: `{summary['mismatch']['train_first_bad_fanout_fraction']:.6f}`",
            f"- train_good_bad_grid_fraction: `{summary['mismatch']['train_good_bad_grid_fraction']:.6f}`",
            f"- train_local_error_fraction: `{summary['mismatch']['train_local_error_fraction']:.6f}`",
            f"- train_terminal_anchor_fraction: `{summary['mismatch']['train_terminal_anchor_fraction']:.6f}`",
            f"- train_direct_preference_fraction: `{summary['mismatch']['train_direct_preference_fraction']:.6f}`",
            f"- train_same_prompt_binary_fraction: `{summary['mismatch']['train_same_prompt_binary_fraction']:.6f}`",
            f"- all_correct_supervision_gap: `{summary['mismatch']['all_correct_supervision_gap']}`",
            f"- late_error_coverage_gap: `{summary['mismatch']['late_error_coverage_gap']}`",
            f"- local_only_without_terminal_gap: `{summary['mismatch']['local_only_without_terminal_gap']}`",
            "",
            "## Example Buckets",
            "",
            "| bucket | n_examples | pair_acc | first_edge_acc | terminal_top1 | terminal_gap | mean_score |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["example_bucket_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["bucket"]),
                    str(int(row["num_examples"])),
                    _fmt_opt(row["pair_accuracy_good_vs_bad"]),
                    _fmt_opt(row["first_error_edge_accuracy"]),
                    _fmt_opt(row["all_correct_terminal_top1_accuracy"]),
                    _fmt_opt(row["all_correct_terminal_gap_mean"]),
                    _fmt_opt(row["mean_prefix_score"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Prefix Buckets",
            "",
            "| bucket | n_prefixes | mean_score | p10_score | p90_score | mean_step_index |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["prefix_bucket_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["bucket"]),
                    str(int(row["num_prefixes"])),
                    _fmt_opt(row["mean_score"]),
                    _fmt_opt(row["p10_score"]),
                    _fmt_opt(row["p90_score"]),
                    _fmt_opt(row["mean_step_index"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _allocate_run_dir(output_root: Path, run_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_root / f"{run_name}_{timestamp}"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_no} must be JSON object")
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_fraction(num: int, den: int) -> float:
    return 0.0 if int(den) <= 0 else float(num / den)


def _p_quantile(values: list[float] | list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = int(max(0, min(len(ordered) - 1, p * (len(ordered) - 1))))
    return float(ordered[idx])


def _fmt_opt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
