#!/usr/bin/env python3
"""Audit fixed-threshold stability and generator-shift robustness for Phase F.

English
-------
Phase F is not blocked by "can the verifier rank at all?" anymore.
The remaining pre-RL question is narrower:

1. if we freeze one current strong verifier candidate,
2. does its decision quality stay stable around one deployment threshold,
3. and does that threshold survive moderate policy/generator shift?

This script answers that question offline from existing benchmark eval artifacts.

中文
----
Phase F 现在卡住的已经不是“verifier 会不会排序”，而是更接近部署的两个问题：

1. 如果我们冻结当前强候选之一，
2. 它在一个固定阈值附近的行为是否稳定，
3. 这个阈值在中等程度的 policy / generator shift 下还能不能站住。

这个脚本完全离线消费已有 benchmark eval artifact，不重新训练模型。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
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
    ProcessBenchExample,
    ProcessBenchPrefixRecord,
    build_processbench_prefix_records,
    compute_processbench_f1,
    load_processbench_examples,
)


@dataclass(slots=True)
class JoinedPrefixRow:
    """One ProcessBench prefix row joined with its score and generator label.

    中文
    ----
    这里把 benchmark 原始 generator 信息补回 prefix 级行，是为了显式量化
    "policy shift"：不同 generator 产生的推理轨迹，本质上就是不同策略分布。
    """

    row: ProcessBenchPrefixRecord
    score: float
    generator: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze threshold stability and generator-shift robustness from ProcessBench eval artifacts."
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="CASE_ID|EVAL_DIR",
        help="One benchmark eval artifact. Repeat this flag for multiple candidates / benchmarks.",
    )
    parser.add_argument("--run-name", default="phase_f_threshold_shift")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_f_threshold_shift"),
    )
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--grid-min", type=float, default=0.20)
    parser.add_argument("--grid-max", type=float, default=0.80)
    parser.add_argument("--grid-step", type=float, default=0.02)
    parser.add_argument(
        "--near-best-ratio",
        type=float,
        default=0.95,
        help="Thresholds with F1 >= ratio * best_F1 are counted as near-best window.",
    )
    parser.add_argument(
        "--min-generator-examples",
        type=int,
        default=12,
        help="Ignore generators with fewer than this many benchmark examples.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.case:
        raise ValueError("At least one --case is required")
    if not (0.0 < float(args.grid_step) <= 1.0):
        raise ValueError("--grid-step must be in (0, 1]")
    if float(args.grid_min) >= float(args.grid_max):
        raise ValueError("--grid-min must be smaller than --grid-max")
    if not (0.0 < float(args.near_best_ratio) <= 1.0):
        raise ValueError("--near-best-ratio must be in (0, 1]")
    if int(args.min_generator_examples) <= 0:
        raise ValueError("--min-generator-examples must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = _allocate_run_dir(output_root=Path(args.output_root), run_name=str(args.run_name))
    run_dir.mkdir(parents=True, exist_ok=False)

    case_rows: list[dict[str, Any]] = []
    generator_rows: list[dict[str, Any]] = []
    for raw_case in args.case:
        case_summary, per_generator = _run_one_case(
            raw_case=str(raw_case),
            fixed_threshold=float(args.fixed_threshold),
            grid_min=float(args.grid_min),
            grid_max=float(args.grid_max),
            grid_step=float(args.grid_step),
            near_best_ratio=float(args.near_best_ratio),
            min_generator_examples=int(args.min_generator_examples),
        )
        case_rows.append(case_summary)
        generator_rows.extend(per_generator)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "settings": {
            "fixed_threshold": float(args.fixed_threshold),
            "grid_min": float(args.grid_min),
            "grid_max": float(args.grid_max),
            "grid_step": float(args.grid_step),
            "near_best_ratio": float(args.near_best_ratio),
            "min_generator_examples": int(args.min_generator_examples),
        },
        "cases": case_rows,
        "generator_rows": generator_rows,
    }
    summary_json_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    case_rows_path = run_dir / "case_rows.jsonl"
    generator_rows_path = run_dir / "generator_rows.jsonl"

    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_jsonl(case_rows_path, case_rows)
    _write_jsonl(generator_rows_path, generator_rows)
    summary_md_path.write_text(
        _render_summary_markdown(case_rows=case_rows, generator_rows=generator_rows),
        encoding="utf-8",
    )

    print("=" * 88)
    print("Phase F Threshold / Shift Audit")
    print("=" * 88)
    for row in case_rows:
        print(
            f"{row['case_id']:>12} | {row['benchmark_id']:<18} "
            f"| f1@0.5={row['fixed_f1']:.4f} | best={row['best_f1']:.4f}@{row['best_threshold']:.3f} "
            f"| near_best_width={row['near_best_window_width']:.3f} "
            f"| gen_tau_std={row['generator_best_threshold_std']:.4f} "
            f"| worst_logo_f1={row['worst_generator_logo_f1']:.4f}"
        )
    print(f"summary_json : {summary_json_path}")
    print(f"summary_md   : {summary_md_path}")
    print("=" * 88)
    return 0


def _run_one_case(
    *,
    raw_case: str,
    fixed_threshold: float,
    grid_min: float,
    grid_max: float,
    grid_step: float,
    near_best_ratio: float,
    min_generator_examples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parts = [part.strip() for part in str(raw_case).split("|")]
    if len(parts) != 2:
        raise ValueError("--case must look like CASE_ID|EVAL_DIR")
    case_id, eval_dir_text = parts
    eval_dir = Path(eval_dir_text)
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

    summary_payload = _load_json(eval_dir / "summary.json")
    benchmark_path = Path(summary_payload["benchmark_path"])
    benchmark_id = str(summary_payload.get("benchmark_id", "processbench"))
    scored_rows = _load_jsonl(eval_dir / "scored_rows.jsonl")

    joined_rows = _load_joined_rows(benchmark_path=benchmark_path, scored_rows=scored_rows)
    rows = [item.row for item in joined_rows]
    scores = [float(item.score) for item in joined_rows]
    thresholds = _build_threshold_grid(grid_min=grid_min, grid_max=grid_max, grid_step=grid_step)
    threshold_metrics = [_compute_threshold_metrics(rows=rows, scores=scores, threshold=tau) for tau in thresholds]
    best_row = max(threshold_metrics, key=lambda item: (item["f1"], -abs(item["threshold"] - fixed_threshold)))
    fixed_row = _compute_threshold_metrics(rows=rows, scores=scores, threshold=fixed_threshold)

    near_best = [
        item for item in threshold_metrics if float(item["f1"]) >= float(best_row["f1"]) * float(near_best_ratio)
    ]
    near_best_min = min(float(item["threshold"]) for item in near_best)
    near_best_max = max(float(item["threshold"]) for item in near_best)

    generator_rows = _analyze_generators(
        case_id=case_id,
        benchmark_id=benchmark_id,
        joined_rows=joined_rows,
        fixed_threshold=fixed_threshold,
        min_generator_examples=min_generator_examples,
    )
    generator_best_thresholds = [float(row["generator_best_threshold"]) for row in generator_rows]
    generator_fixed_f1s = [float(row["fixed_f1"]) for row in generator_rows]
    generator_logo_f1s = [float(row["logo_f1"]) for row in generator_rows]

    case_summary = {
        "case_id": case_id,
        "benchmark_id": benchmark_id,
        "eval_dir": str(eval_dir),
        "value_run_dir": str(summary_payload.get("value_run_dir", "")),
        "best_threshold": float(best_row["threshold"]),
        "best_f1": float(best_row["f1"]),
        "best_acc_correct": float(best_row["acc_correct"]),
        "best_acc_erroneous": float(best_row["acc_erroneous"]),
        "fixed_threshold": float(fixed_threshold),
        "fixed_f1": float(fixed_row["f1"]),
        "fixed_acc_correct": float(fixed_row["acc_correct"]),
        "fixed_acc_erroneous": float(fixed_row["acc_erroneous"]),
        "near_best_window_min": float(near_best_min),
        "near_best_window_max": float(near_best_max),
        "near_best_window_width": float(near_best_max - near_best_min),
        "generator_count": int(len(generator_rows)),
        "generator_best_threshold_std": (
            float(statistics.pstdev(generator_best_thresholds)) if len(generator_best_thresholds) >= 2 else 0.0
        ),
        "worst_generator_fixed_f1": float(min(generator_fixed_f1s)) if generator_fixed_f1s else 0.0,
        "worst_generator_logo_f1": float(min(generator_logo_f1s)) if generator_logo_f1s else 0.0,
        "mean_generator_logo_gap": (
            float(statistics.mean(float(row["generator_best_f1"]) - float(row["logo_f1"]) for row in generator_rows))
            if generator_rows
            else 0.0
        ),
        "mean_generator_fixed_gap": (
            float(statistics.mean(float(row["generator_best_f1"]) - float(row["fixed_f1"]) for row in generator_rows))
            if generator_rows
            else 0.0
        ),
    }
    return case_summary, generator_rows


def _analyze_generators(
    *,
    case_id: str,
    benchmark_id: str,
    joined_rows: list[JoinedPrefixRow],
    fixed_threshold: float,
    min_generator_examples: int,
) -> list[dict[str, Any]]:
    grouped_examples: dict[str, list[JoinedPrefixRow]] = {}
    for item in joined_rows:
        grouped_examples.setdefault(str(item.row.example_id), []).append(item)

    example_generators: dict[str, str] = {}
    for example_id, rows in grouped_examples.items():
        example_generators[example_id] = str(rows[0].generator)

    generator_to_example_ids: dict[str, list[str]] = {}
    for example_id, generator in example_generators.items():
        generator_to_example_ids.setdefault(str(generator), []).append(str(example_id))

    rows_by_generator: dict[str, list[JoinedPrefixRow]] = {}
    for item in joined_rows:
        rows_by_generator.setdefault(str(item.generator), []).append(item)

    results: list[dict[str, Any]] = []
    for generator, items in sorted(rows_by_generator.items()):
        example_ids = generator_to_example_ids.get(str(generator), [])
        if len(example_ids) < int(min_generator_examples):
            continue
        rows = [item.row for item in items]
        scores = [float(item.score) for item in items]
        generator_best = compute_processbench_f1(rows=rows, scores=scores, threshold=None)
        fixed = compute_processbench_f1(rows=rows, scores=scores, threshold=float(fixed_threshold))

        train_rows = [item.row for item in joined_rows if str(item.generator) != str(generator)]
        train_scores = [float(item.score) for item in joined_rows if str(item.generator) != str(generator)]
        if not train_rows:
            continue
        logo_train = compute_processbench_f1(rows=train_rows, scores=train_scores, threshold=None)
        logo = compute_processbench_f1(
            rows=rows,
            scores=scores,
            threshold=float(logo_train["processbench_f1_threshold"]),
        )

        results.append(
            {
                "case_id": case_id,
                "benchmark_id": benchmark_id,
                "generator": str(generator),
                "num_examples": int(len(example_ids)),
                "generator_best_threshold": float(generator_best["processbench_f1_threshold"]),
                "generator_best_f1": float(generator_best["processbench_f1"]),
                "fixed_threshold": float(fixed_threshold),
                "fixed_f1": float(fixed["processbench_f1"]),
                "logo_threshold": float(logo_train["processbench_f1_threshold"]),
                "logo_f1": float(logo["processbench_f1"]),
                "logo_acc_correct": float(logo["processbench_acc_correct"]),
                "logo_acc_erroneous": float(logo["processbench_acc_erroneous"]),
            }
        )
    return results


def _load_joined_rows(*, benchmark_path: Path, scored_rows: list[dict[str, Any]]) -> list[JoinedPrefixRow]:
    score_by_row_id = {str(row["row_id"]): float(row["score"]) for row in scored_rows}
    examples = load_processbench_examples(benchmark_path)
    example_by_id = {str(example.example_id): example for example in examples}
    joined: list[JoinedPrefixRow] = []
    for row in build_processbench_prefix_records(examples):
        score = score_by_row_id.get(str(row.row_id))
        if score is None:
            continue
        example = example_by_id[str(row.example_id)]
        joined.append(
            JoinedPrefixRow(
                row=row,
                score=float(score),
                generator=str(example.generator),
            )
        )
    if not joined:
        raise RuntimeError(f"No joined ProcessBench rows found for {benchmark_path}")
    return joined


def _compute_threshold_metrics(
    *,
    rows: list[ProcessBenchPrefixRecord],
    scores: list[float],
    threshold: float,
) -> dict[str, float]:
    payload = compute_processbench_f1(rows=rows, scores=scores, threshold=float(threshold))
    return {
        "threshold": float(threshold),
        "f1": float(payload["processbench_f1"]),
        "acc_correct": float(payload["processbench_acc_correct"]),
        "acc_erroneous": float(payload["processbench_acc_erroneous"]),
    }


def _build_threshold_grid(*, grid_min: float, grid_max: float, grid_step: float) -> list[float]:
    values: list[float] = []
    current = float(grid_min)
    while current <= float(grid_max) + 1e-9:
        values.append(round(float(current), 6))
        current += float(grid_step)
    if not values:
        raise RuntimeError("Threshold grid is empty")
    return values


def _allocate_run_dir(*, output_root: Path, run_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_root / f"{run_name}_{timestamp}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _render_summary_markdown(
    *,
    case_rows: list[dict[str, Any]],
    generator_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase F Threshold / Shift Audit",
        "",
        "## Case Summary",
        "",
        "| case_id | benchmark | best_tau | best_f1 | f1@0.5 | near_best_width | gen_tau_std | worst_gen_f1@0.5 | worst_gen_logo_f1 | mean_logo_gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in case_rows:
        lines.append(
            "| {case_id} | {benchmark} | {best_tau:.3f} | {best_f1:.4f} | {fixed_f1:.4f} | {width:.3f} | {tau_std:.4f} | {worst_fixed:.4f} | {worst_logo:.4f} | {logo_gap:.4f} |".format(
                case_id=row["case_id"],
                benchmark=row["benchmark_id"],
                best_tau=row["best_threshold"],
                best_f1=row["best_f1"],
                fixed_f1=row["fixed_f1"],
                width=row["near_best_window_width"],
                tau_std=row["generator_best_threshold_std"],
                worst_fixed=row["worst_generator_fixed_f1"],
                worst_logo=row["worst_generator_logo_f1"],
                logo_gap=row["mean_generator_logo_gap"],
            )
        )

    lines.extend(
        [
            "",
            "## Generator Breakdown",
            "",
            "| case_id | benchmark | generator | n_examples | best_tau | best_f1 | f1@0.5 | logo_tau | logo_f1 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in generator_rows:
        lines.append(
            "| {case_id} | {benchmark} | {generator} | {n_examples} | {best_tau:.3f} | {best_f1:.4f} | {fixed_f1:.4f} | {logo_tau:.3f} | {logo_f1:.4f} |".format(
                case_id=row["case_id"],
                benchmark=row["benchmark_id"],
                generator=row["generator"],
                n_examples=row["num_examples"],
                best_tau=row["generator_best_threshold"],
                best_f1=row["generator_best_f1"],
                fixed_f1=row["fixed_f1"],
                logo_tau=row["logo_threshold"],
                logo_f1=row["logo_f1"],
            )
        )

    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `near_best_width` 越窄，说明 deployment threshold 越脆弱。",
            "- `gen_tau_std` 越大，说明不同 generator / policy 子分布需要的阈值差异越大。",
            "- `logo_f1` 是 leave-one-generator-out 阈值迁移结果，越低说明 policy-shift 越危险。",
            "- `f1@0.5` 与 `best_f1` 的差距越大，说明 fixed-threshold 部署会更吃亏。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
