#!/usr/bin/env python3
"""Compare multiple Phase E ProcessBench transfer cases in one diagnostic table.

English
-------
The repository already has per-run tools that answer:
1. how one training-pair artifact aligns with ProcessBench geometry,
2. how one benchmark eval fails by bucket.

What was still missing is a compact cross-run view that puts those signals
side-by-side.  This script fills that gap so benchmark-transfer experiments can
be judged on more than one collapsed AUC number.

中文
----
仓库里已经有单运行诊断工具，可以分别回答：
1. 某个训练 pair artifact 与 ProcessBench 几何是否对齐；
2. 某个 benchmark eval 具体失败在哪些 bucket。

但还缺一层“多运行横向对照”。本脚本就是把这些信号并排放在一起，避免后续实验只
盯着一个 AUC。
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
from ours.phase_e.processbench_alignment import (  # noqa: E402
    compute_alignment_distances,
    summarize_pair_jsonl_alignment,
    summarize_processbench_topology,
    summarize_scored_rows_alignment,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare multiple Phase E ProcessBench transfer cases with one shared diagnostic table."
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help=(
            "Case definition in the form "
            "`case_name=path/to/value_run_dir::path/to/benchmark_eval_dir`."
        ),
    )
    parser.add_argument("--run-name", default="phase_e_processbench_transfer_compare")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_transfer_compare"),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.case:
        raise ValueError("At least one --case must be provided")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = _parse_case_args(args.case)
    run_dir = Path(args.output_root) / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    benchmark_cache: dict[str, dict[str, Any]] = {}
    case_rows = []
    for case_name, value_run_dir, benchmark_eval_dir in cases:
        value_manifest = _load_json(value_run_dir / "manifest.json")
        benchmark_summary = _load_json(benchmark_eval_dir / "summary.json")
        train_pairs_jsonl = Path(value_manifest["input_files"]["train_pairs_jsonl"])
        benchmark_path = Path(benchmark_summary["benchmark_path"])
        cache_key = str(benchmark_path)
        if cache_key not in benchmark_cache:
            benchmark_cache[cache_key] = summarize_processbench_topology(benchmark_path)

        pair_summary = summarize_pair_jsonl_alignment(train_pairs_jsonl)
        scored_summary = summarize_scored_rows_alignment(
            scored_rows_path=benchmark_eval_dir / "scored_rows.jsonl",
            processbench_path=benchmark_path,
        )
        alignment = compute_alignment_distances(
            pair_summary=pair_summary,
            benchmark_summary=benchmark_cache[cache_key],
        )
        case_rows.append(
            _build_case_row(
                case_name=case_name,
                value_run_dir=value_run_dir,
                benchmark_eval_dir=benchmark_eval_dir,
                benchmark_summary=benchmark_summary,
                benchmark_topology=benchmark_cache[cache_key],
                pair_summary=pair_summary,
                scored_summary=scored_summary,
            )
            | {"alignment_distance": alignment}
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "cases": case_rows,
    }
    summary_json_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")

    print("=" * 88)
    print("Phase E ProcessBench Transfer Compare")
    print("=" * 88)
    for row in case_rows:
        print(
            f"{row['case_name']:28} auc={row['metrics']['pair_auc_good_vs_bad']:.4f} "
            f"first_edge={row['slice_metrics']['first_error_edge_accuracy']:.4f} "
            f"terminal_top1={_fmt_opt(row['all_correct_metrics']['terminal_top1_accuracy'])} "
            f"pair_type_l1={row['alignment_distance']['pair_type_l1_distance']:.4f}"
        )
    print(f"summary_json_path : {summary_json_path}")
    print(f"summary_md_path   : {summary_md_path}")
    print("=" * 88)
    return 0


def _parse_case_args(raw_items: list[str]) -> list[tuple[str, Path, Path]]:
    parsed: list[tuple[str, Path, Path]] = []
    for item in raw_items:
        text = str(item).strip()
        if "=" not in text or "::" not in text:
            raise ValueError(
                "--case must look like case_name=value_run_dir::benchmark_eval_dir, "
                f"got {item!r}"
            )
        case_name, raw_paths = text.split("=", 1)
        value_run_dir_text, benchmark_eval_dir_text = raw_paths.split("::", 1)
        normalized_name = str(case_name).strip()
        value_run_dir = Path(value_run_dir_text).expanduser()
        benchmark_eval_dir = Path(benchmark_eval_dir_text).expanduser()
        if normalized_name == "":
            raise ValueError(f"--case is missing the case name: {item!r}")
        if not value_run_dir.exists():
            raise FileNotFoundError(f"--case value_run_dir not found: {value_run_dir}")
        if not benchmark_eval_dir.exists():
            raise FileNotFoundError(f"--case benchmark_eval_dir not found: {benchmark_eval_dir}")
        parsed.append((normalized_name, value_run_dir, benchmark_eval_dir))
    return parsed


def _build_case_row(
    *,
    case_name: str,
    value_run_dir: Path,
    benchmark_eval_dir: Path,
    benchmark_summary: dict[str, Any],
    benchmark_topology: dict[str, Any],
    pair_summary: dict[str, Any],
    scored_summary: dict[str, Any],
) -> dict[str, Any]:
    joined_rows = _load_joined_processbench_rows(
        benchmark_path=Path(benchmark_summary["benchmark_path"]),
        scored_rows_path=benchmark_eval_dir / "scored_rows.jsonl",
    )
    prefix_bucket_means = _summarize_prefix_bucket_means(joined_rows)
    all_correct_metrics = _summarize_all_correct_examples(joined_rows)
    base_pair_summary = dict(pair_summary.get("base_summary", {}))
    by_pair_semantics = dict(base_pair_summary.get("by_pair_semantics", {}))
    num_pairs = int(pair_summary.get("num_pairs", 0))
    local_error_pairs = int(by_pair_semantics.get("local_first_bad_edge", 0)) + int(
        by_pair_semantics.get("local_modified_process_error_step", 0)
    )
    first_bad_fanout_pairs = int(by_pair_semantics.get("first_bad_fanout_prefix_ranking", 0))
    good_bad_grid_pairs = int(by_pair_semantics.get("good_bad_prefix_grid", 0))
    terminal_pairs = int(by_pair_semantics.get("terminal_completion_anchor", 0))

    return {
        "case_name": case_name,
        "value_run_dir": str(value_run_dir),
        "benchmark_eval_dir": str(benchmark_eval_dir),
        "benchmark_id": str(benchmark_summary["benchmark_id"]),
        "benchmark_path": str(benchmark_summary["benchmark_path"]),
        "train_pair_summary": {
            "num_pairs": num_pairs,
            "by_pair_semantics": by_pair_semantics,
            "local_error_fraction": _safe_fraction(local_error_pairs, num_pairs),
            "first_bad_fanout_fraction": _safe_fraction(first_bad_fanout_pairs, num_pairs),
            "good_bad_grid_fraction": _safe_fraction(good_bad_grid_pairs, num_pairs),
            "terminal_anchor_fraction": _safe_fraction(terminal_pairs, num_pairs),
            "prefix_relation_rate": float(pair_summary.get("prefix_relation_rate", 0.0)),
            "pair_type_distribution": dict(pair_summary.get("pair_type_distribution", {})),
            "gap_bucket_distribution": dict(pair_summary.get("gap_bucket_distribution", {})),
        },
        "benchmark_topology": {
            "num_examples": int(benchmark_topology.get("num_examples", 0)),
            "num_all_correct_examples": int(benchmark_topology.get("num_all_correct_examples", 0)),
            "pair_type_distribution": dict(benchmark_topology.get("pair_type_distribution", {})),
            "gap_bucket_distribution": dict(benchmark_topology.get("gap_bucket_distribution", {})),
        },
        "metrics": {
            "pair_accuracy_good_vs_bad": float(benchmark_summary["metrics"]["pair_accuracy_good_vs_bad"]),
            "pair_auc_good_vs_bad": float(benchmark_summary["metrics"]["pair_auc_good_vs_bad"]),
            "first_error_edge_accuracy": float(benchmark_summary["metrics"]["first_error_edge_accuracy"]),
        },
        "slice_metrics": {
            "first_error_edge_accuracy": float(scored_summary["first_error_edge_accuracy"]),
            "anygood_vs_firstbad_accuracy": float(
                scored_summary["aggregate_metrics"]["anygood_vs_firstbad"]["accuracy"]
            ),
            "good_vs_laterbad_accuracy": float(
                scored_summary["aggregate_metrics"]["good_vs_laterbad"]["accuracy"]
            ),
            "gap1_accuracy": _safe_gap_bucket_accuracy(scored_summary, "gap1"),
            "gap2_accuracy": _safe_gap_bucket_accuracy(scored_summary, "gap2"),
            "gap3_4_accuracy": _safe_gap_bucket_accuracy(scored_summary, "gap3_4"),
            "gap5p_accuracy": _safe_gap_bucket_accuracy(scored_summary, "gap5p"),
        },
        "all_correct_metrics": all_correct_metrics,
        "prefix_bucket_means": prefix_bucket_means,
    }


def _safe_gap_bucket_accuracy(scored_summary: dict[str, Any], bucket_name: str) -> float:
    bucket_metrics = dict(scored_summary.get("gap_bucket_metrics", {}))
    bucket_row = dict(bucket_metrics.get(bucket_name, {}))
    return float(bucket_row.get("accuracy", 0.0))


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


def _summarize_prefix_bucket_means(
    joined_rows: list[tuple[ProcessBenchPrefixRecord, float]],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {
        "safe_prefix": [],
        "first_bad_prefix": [],
        "later_bad_prefix": [],
        "all_correct_nonterminal": [],
        "all_correct_terminal": [],
    }
    by_example: dict[str, list[tuple[ProcessBenchPrefixRecord, float]]] = {}
    for row, score in joined_rows:
        by_example.setdefault(str(row.example_id), []).append((row, float(score)))
    for example_rows in by_example.values():
        num_steps = int(max(row.prefix_step_index for row, _ in example_rows) + 1)
        for row, score in example_rows:
            grouped[_bucket_prefix(row=row, num_steps=num_steps)].append(float(score))
    return {
        key: float(statistics.mean(values)) if values else 0.0 for key, values in sorted(grouped.items())
    }


def _summarize_all_correct_examples(
    joined_rows: list[tuple[ProcessBenchPrefixRecord, float]],
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[ProcessBenchPrefixRecord, float]]] = {}
    for row, score in joined_rows:
        grouped.setdefault(str(row.example_id), []).append((row, float(score)))

    terminal_top1_total = 0
    terminal_top1_positive = 0
    terminal_gaps = []
    terminal_scores = []
    nonterminal_scores = []
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
        terminal_scores.append(final_score)
        nonterminal_scores.extend(prev_scores)
    return {
        "terminal_top1_accuracy": (
            float(terminal_top1_positive / terminal_top1_total) if terminal_top1_total > 0 else None
        ),
        "terminal_gap_mean": float(statistics.mean(terminal_gaps)) if terminal_gaps else None,
        "mean_terminal_score": float(statistics.mean(terminal_scores)) if terminal_scores else None,
        "mean_nonterminal_score": float(statistics.mean(nonterminal_scores)) if nonterminal_scores else None,
    }


def _bucket_prefix(*, row: ProcessBenchPrefixRecord, num_steps: int) -> str:
    if row.label < 0:
        if row.prefix_step_index == (num_steps - 1):
            return "all_correct_terminal"
        return "all_correct_nonterminal"
    if row.prefix_step_index < row.label:
        return "safe_prefix"
    if row.prefix_step_index == row.label:
        return "first_bad_prefix"
    return "later_bad_prefix"


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase E ProcessBench Transfer Compare",
        "",
        "| case | benchmark | auc | pair_acc | first_edge | anygood_vs_firstbad | good_vs_laterbad | terminal_top1 | terminal_gap | pair_type_l1 | terminal_anchor_frac | fanout_frac | grid_frac |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["cases"]:
        lines.append(
            "| {case} | {benchmark} | {auc} | {pair_acc} | {first_edge} | {anygood} | {laterbad} | {terminal_top1} | {terminal_gap} | {pair_l1} | {terminal_frac} | {fanout_frac} | {grid_frac} |".format(
                case=row["case_name"],
                benchmark=row["benchmark_id"],
                auc=_fmt_float(row["metrics"]["pair_auc_good_vs_bad"]),
                pair_acc=_fmt_float(row["metrics"]["pair_accuracy_good_vs_bad"]),
                first_edge=_fmt_float(row["slice_metrics"]["first_error_edge_accuracy"]),
                anygood=_fmt_float(row["slice_metrics"]["anygood_vs_firstbad_accuracy"]),
                laterbad=_fmt_float(row["slice_metrics"]["good_vs_laterbad_accuracy"]),
                terminal_top1=_fmt_opt(row["all_correct_metrics"]["terminal_top1_accuracy"]),
                terminal_gap=_fmt_opt(row["all_correct_metrics"]["terminal_gap_mean"]),
                pair_l1=_fmt_float(row["alignment_distance"]["pair_type_l1_distance"]),
                terminal_frac=_fmt_float(row["train_pair_summary"]["terminal_anchor_fraction"]),
                fanout_frac=_fmt_float(row["train_pair_summary"]["first_bad_fanout_fraction"]),
                grid_frac=_fmt_float(row["train_pair_summary"]["good_bad_grid_fraction"]),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `anygood_vs_firstbad` isolates whether the model can keep multiple good prefixes above the first bad prefix.",
            "- `good_vs_laterbad` isolates whether later bad prefixes are still ranked below good prefixes.",
            "- `terminal_top1` and `terminal_gap` only use all-correct ProcessBench examples; they expose terminal-completion collapse directly.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_fraction(numerator: int | float, denominator: int | float) -> float:
    if float(denominator) <= 0.0:
        return 0.0
    return float(float(numerator) / float(denominator))


def _fmt_float(value: float) -> str:
    return f"{float(value):.4f}"


def _fmt_opt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
