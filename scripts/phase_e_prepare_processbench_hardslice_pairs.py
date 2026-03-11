#!/usr/bin/env python3
"""Prepare hard-slice ProcessBench pairs for benchmark-side adjudication.

English
-------
This script converts one existing ProcessBench benchmark-eval artifact into a
judge-friendly pair dataset.

It focuses on slices where the current value model is already known to fail:
1. first-error edge failures,
2. later-bad outranking failures.

The output is an external-pair JSONL that can be fed into the existing
`phase_e_pairwise_judge_benchmark.py` pipeline.

中文
----
这个脚本把已有的 ProcessBench benchmark-eval artifact 转成 judge 更容易处理
的 pair 数据。

它只关注当前 value model 已经明确失败的 slice：
1. first-error edge fail，
2. later-bad outranking fail。

输出仍然是 external-pair JSONL，可以直接复用现有
`phase_e_pairwise_judge_benchmark.py`。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_d.external_pairs import ExternalPairRecord  # noqa: E402
from ours.phase_e.benchmark_eval import build_processbench_prefix_records, load_processbench_examples  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ProcessBench hard-slice adjudication pairs.")
    parser.add_argument("--benchmark-eval-dir", type=Path, required=True)
    parser.add_argument("--max-edge-pairs", type=int, default=16)
    parser.add_argument("--max-laterbad-pairs", type=int, default=16)
    parser.add_argument("--run-name", default="phase_e_processbench_hardslice_pairs")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_pairs"),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.benchmark_eval_dir.exists():
        raise FileNotFoundError(f"--benchmark-eval-dir not found: {args.benchmark_eval_dir}")
    if int(args.max_edge_pairs) < 0 or int(args.max_laterbad_pairs) < 0:
        raise ValueError("max pair counts must be >= 0")
    return args


def _bucket_error_position(*, label: int, num_steps: int) -> str:
    if num_steps <= 0 or label < 0:
        return "unknown"
    frac = float((label + 1) / max(num_steps, 1))
    if frac <= 1.0 / 3.0:
        return "early"
    if frac <= 2.0 / 3.0:
        return "mid"
    return "late"


def _round_robin_limit(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not rows:
        return []
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["error_bucket"])].append(row)
    ordered = sorted(buckets)
    selected: list[dict[str, Any]] = []
    cursor = {key: 0 for key in ordered}
    while len(selected) < int(limit):
        advanced = False
        for key in ordered:
            idx = cursor[key]
            if idx >= len(buckets[key]):
                continue
            selected.append(buckets[key][idx])
            cursor[key] += 1
            advanced = True
            if len(selected) >= int(limit):
                break
        if not advanced:
            break
    return selected


def _prefix_text(problem: str, steps: list[str], step_index: int) -> str:
    rendered = "\n".join(f"Step {idx + 1}: {steps[idx]}" for idx in range(step_index + 1))
    return f"{rendered}\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads((Path(args.benchmark_eval_dir) / "summary.json").read_text(encoding="utf-8"))
    benchmark_path = Path(summary["benchmark_path"])
    scored_rows = [json.loads(line) for line in (Path(args.benchmark_eval_dir) / "scored_rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = {str(example.example_id): example for example in load_processbench_examples(benchmark_path)}
    prefix_rows = build_processbench_prefix_records(list(examples.values()))
    prefix_by_row_id = {str(row.row_id): row for row in prefix_rows}

    joined_by_example: dict[str, list[tuple[Any, float]]] = defaultdict(list)
    for row in scored_rows:
        prefix = prefix_by_row_id.get(str(row["row_id"]))
        if prefix is None:
            continue
        joined_by_example[str(prefix.example_id)].append((prefix, float(row["score"])))

    edge_candidates: list[dict[str, Any]] = []
    laterbad_candidates: list[dict[str, Any]] = []
    for example_id, records in joined_by_example.items():
        example = examples.get(example_id)
        if example is None or int(example.label) < 0:
            continue
        ordered = sorted(records, key=lambda item: int(item[0].prefix_step_index))
        label = int(example.label)
        num_steps = int(len(example.steps))
        bucket = _bucket_error_position(label=label, num_steps=num_steps)
        last_safe = next((item for item in reversed(ordered) if bool(item[0].is_good_prefix)), None)
        first_bad = next((item for item in ordered if bool(item[0].is_first_bad_prefix)), None)
        if last_safe is not None and first_bad is not None:
            safe_row, safe_score = last_safe
            first_bad_row, first_bad_score = first_bad
            if float(safe_score) <= float(first_bad_score):
                edge_candidates.append(
                    {
                        "pair": ExternalPairRecord(
                            pair_id=f"{example_id}::edge_fail",
                            source_tag="processbench_adjudication",
                            domain_tag=str(summary.get("benchmark_id", "processbench")),
                            prompt_text=f"{example.problem.strip()}\n\n",
                            chosen_text=_prefix_text(example.problem, example.steps, int(safe_row.prefix_step_index)),
                            rejected_text=_prefix_text(example.problem, example.steps, int(first_bad_row.prefix_step_index)),
                            pair_confidence=1.0,
                            quality_flags={"benchmark_side_adjudication": True},
                            metadata={
                                "pair_semantics": "processbench_edge_fail_adjudication",
                                "benchmark_id": str(summary.get("benchmark_id", "processbench")),
                                "example_id": str(example_id),
                                "error_bucket": bucket,
                                "gold_first_bad_step_index": int(label),
                                "chosen_prefix_step_index": int(safe_row.prefix_step_index),
                                "rejected_prefix_step_index": int(first_bad_row.prefix_step_index),
                                "chosen_model_score": float(safe_score),
                                "rejected_model_score": float(first_bad_score),
                                "model_score_gap": float(safe_score - first_bad_score),
                            },
                        ).to_dict(),
                        "error_bucket": bucket,
                    }
                )

        later_bad_rows = [item for item in ordered if int(item[0].prefix_step_index) > int(label)]
        if last_safe is not None and later_bad_rows:
            safe_row, safe_score = last_safe
            hardest_later_bad = max(later_bad_rows, key=lambda item: float(item[1]))
            later_bad_row, later_bad_score = hardest_later_bad
            if float(safe_score) <= float(later_bad_score):
                laterbad_candidates.append(
                    {
                        "pair": ExternalPairRecord(
                            pair_id=f"{example_id}::laterbad_fail",
                            source_tag="processbench_adjudication",
                            domain_tag=str(summary.get("benchmark_id", "processbench")),
                            prompt_text=f"{example.problem.strip()}\n\n",
                            chosen_text=_prefix_text(example.problem, example.steps, int(safe_row.prefix_step_index)),
                            rejected_text=_prefix_text(example.problem, example.steps, int(later_bad_row.prefix_step_index)),
                            pair_confidence=1.0,
                            quality_flags={"benchmark_side_adjudication": True},
                            metadata={
                                "pair_semantics": "processbench_laterbad_fail_adjudication",
                                "benchmark_id": str(summary.get("benchmark_id", "processbench")),
                                "example_id": str(example_id),
                                "error_bucket": bucket,
                                "gold_first_bad_step_index": int(label),
                                "chosen_prefix_step_index": int(safe_row.prefix_step_index),
                                "rejected_prefix_step_index": int(later_bad_row.prefix_step_index),
                                "chosen_model_score": float(safe_score),
                                "rejected_model_score": float(later_bad_score),
                                "model_score_gap": float(safe_score - later_bad_score),
                            },
                        ).to_dict(),
                        "error_bucket": bucket,
                    }
                )

    selected_edge = _round_robin_limit(edge_candidates, limit=int(args.max_edge_pairs))
    selected_laterbad = _round_robin_limit(laterbad_candidates, limit=int(args.max_laterbad_pairs))
    combined = [item["pair"] for item in selected_edge] + [item["pair"] for item in selected_laterbad]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root) / f"{args.run_name}__{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    pairs_path = run_dir / "pairs.jsonl"
    edge_path = run_dir / "edge_pairs.jsonl"
    laterbad_path = run_dir / "laterbad_pairs.jsonl"

    for path, rows in [
        (pairs_path, combined),
        (edge_path, [item["pair"] for item in selected_edge]),
        (laterbad_path, [item["pair"] for item in selected_laterbad]),
    ]:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _bucket_counts(items: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            key = str(item["error_bucket"])
            counts[key] = counts.get(key, 0) + 1
        return counts

    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "benchmark_eval_dir": str(args.benchmark_eval_dir),
        "benchmark_id": str(summary.get("benchmark_id", "processbench")),
        "benchmark_path": str(benchmark_path),
        "num_scored_rows": int(len(scored_rows)),
        "num_examples": int(len(joined_by_example)),
        "num_edge_candidates": int(len(edge_candidates)),
        "num_laterbad_candidates": int(len(laterbad_candidates)),
        "num_edge_selected": int(len(selected_edge)),
        "num_laterbad_selected": int(len(selected_laterbad)),
        "edge_bucket_counts": _bucket_counts(selected_edge),
        "laterbad_bucket_counts": _bucket_counts(selected_laterbad),
        "output_files": {
            "pairs": str(pairs_path),
            "edge_pairs": str(edge_path),
            "laterbad_pairs": str(laterbad_path),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_stage": "phase_e_processbench_hardslice_pairs_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "benchmark_eval_dir": str(args.benchmark_eval_dir),
                "benchmark_path": str(benchmark_path),
                "benchmark_id": str(summary.get("benchmark_id", "processbench")),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_md = [
        "# Phase E ProcessBench Hard-Slice Pairs",
        "",
        f"- benchmark_eval_dir: `{args.benchmark_eval_dir}`",
        f"- benchmark_id: `{summary_payload['benchmark_id']}`",
        f"- num_edge_candidates: `{len(edge_candidates)}`",
        f"- num_laterbad_candidates: `{len(laterbad_candidates)}`",
        f"- num_edge_selected: `{len(selected_edge)}`",
        f"- num_laterbad_selected: `{len(selected_laterbad)}`",
        "",
        "## Outputs",
        "",
        f"- pairs: `{pairs_path}`",
        f"- edge_pairs: `{edge_path}`",
        f"- laterbad_pairs: `{laterbad_path}`",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    print("=" * 88)
    print("Phase E: Prepare ProcessBench Hard-Slice Pairs")
    print("=" * 88)
    print(f"benchmark_eval_dir     : {args.benchmark_eval_dir}")
    print(f"benchmark_id           : {summary_payload['benchmark_id']}")
    print(f"num_edge_candidates    : {len(edge_candidates)}")
    print(f"num_laterbad_candidates: {len(laterbad_candidates)}")
    print(f"num_edge_selected      : {len(selected_edge)}")
    print(f"num_laterbad_selected  : {len(selected_laterbad)}")
    print(f"pairs_path             : {pairs_path}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
