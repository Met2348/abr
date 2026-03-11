#!/usr/bin/env python3
"""审计训练 pair 与 ProcessBench 评测几何是否对齐。 Audit whether training pairs are aligned with ProcessBench evaluation geometry.

这个脚本不是再跑一次训练，也不是替代主 benchmark eval。
它的职责更像“研究诊断仪表板”：
1. 读取实际训练用的 pair artifact，
2. 读取 ProcessBench 原始数据与已有 scored_rows，
3. 用统一 slice 解释训练支持面与 benchmark 要求是否错位。

This script is not a replacement for training or the main benchmark eval.
Its job is closer to a research debugging dashboard:
1. inspect the actual training pair artifact,
2. inspect raw ProcessBench data plus existing scored rows,
3. explain, in one shared slicing language, where training support and
   benchmark demand do or do not match.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_e.processbench_alignment import (  # noqa: E402
    compute_alignment_distances,
    render_processbench_alignment_markdown,
    summarize_pair_jsonl_alignment,
    summarize_processbench_topology,
    summarize_scored_rows_alignment,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit how one training pair artifact aligns with ProcessBench evaluation geometry."
    )
    parser.add_argument("--pair-artifact-dir", type=Path, required=True)
    parser.add_argument("--processbench-path", type=Path, required=True)
    parser.add_argument("--run-name", default="phase_e_processbench_alignment")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_alignment_audit"),
    )
    parser.add_argument(
        "--scored-run",
        action="append",
        default=[],
        help=(
            "Optional scored_rows input in the form "
            "`run_name=path/to/scored_rows.jsonl`. Repeat this flag for multiple runs."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 CLI 参数并尽早发现明显操作错误。 Parse CLI arguments and fail fast on obvious operator mistakes."""
    args = _build_parser().parse_args(argv)
    if not args.pair_artifact_dir.exists():
        raise FileNotFoundError(f"--pair-artifact-dir not found: {args.pair_artifact_dir}")
    if not args.processbench_path.exists():
        raise FileNotFoundError(f"--processbench-path not found: {args.processbench_path}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pair_jsonl_path = Path(args.pair_artifact_dir) / "train_pairs.jsonl"
    if not pair_jsonl_path.exists():
        raise FileNotFoundError(f"Missing train_pairs.jsonl in pair artifact dir: {pair_jsonl_path}")

    scored_runs = parse_scored_run_args(args.scored_run)
    run_dir = Path(args.output_root) / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pair_summary = summarize_pair_jsonl_alignment(pair_jsonl_path)
    benchmark_summary = summarize_processbench_topology(Path(args.processbench_path))
    alignment_distance = compute_alignment_distances(
        pair_summary=pair_summary,
        benchmark_summary=benchmark_summary,
    )
    scored_run_summaries = {
        run_name: summarize_scored_rows_alignment(
            scored_rows_path=scored_rows_path,
            processbench_path=Path(args.processbench_path),
        )
        for run_name, scored_rows_path in sorted(scored_runs.items())
    }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "pair_artifact_dir": str(args.pair_artifact_dir),
        "pair_summary": pair_summary,
        "benchmark_summary": benchmark_summary,
        "alignment_distance": alignment_distance,
        "scored_runs": scored_run_summaries,
    }
    manifest = {
        "artifact_stage": "phase_e_processbench_alignment_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pair_artifact_dir": str(args.pair_artifact_dir),
        "pair_jsonl_path": str(pair_jsonl_path),
        "processbench_path": str(args.processbench_path),
        "scored_runs": {key: str(value) for key, value in sorted(scored_runs.items())},
    }

    summary_json_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    summary_md_path = run_dir / "summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(
        render_processbench_alignment_markdown(
            title="Phase E ProcessBench Alignment Audit",
            pair_summary=pair_summary,
            benchmark_summary=benchmark_summary,
            alignment_distance=alignment_distance,
            scored_run_summaries=scored_run_summaries,
        ),
        encoding="utf-8",
    )

    print("=" * 88)
    print("Phase E: ProcessBench Alignment Audit")
    print("=" * 88)
    print(f"pair_artifact_dir    : {args.pair_artifact_dir}")
    print(f"processbench_path    : {args.processbench_path}")
    print(f"pair_type_l1_distance: {float(alignment_distance['pair_type_l1_distance']):.6f}")
    print(f"gap_bucket_l1_dist   : {float(alignment_distance['gap_bucket_l1_distance']):.6f}")
    print(f"summary_json_path    : {summary_json_path}")
    print(f"summary_md_path      : {summary_md_path}")
    if scored_run_summaries:
        print("-" * 88)
        for run_name, payload in sorted(scored_run_summaries.items()):
            print(
                f"{run_name:22} pair_acc={float(payload['pair_accuracy_good_vs_bad']):.4f} "
                f"first_edge={float(payload['first_error_edge_accuracy']):.4f}"
            )
    print("=" * 88)
    return 0


def parse_scored_run_args(raw_items: list[str]) -> dict[str, Path]:
    """解析重复的 `--scored-run name=path` 参数。 Parse repeated `--scored-run name=path` arguments."""
    parsed: dict[str, Path] = {}
    for item in raw_items:
        text = str(item).strip()
        if "=" not in text:
            raise ValueError(f"--scored-run must look like name=path, got: {item!r}")
        run_name, raw_path = text.split("=", 1)
        normalized_name = str(run_name).strip()
        normalized_path = Path(raw_path).expanduser()
        if normalized_name == "":
            raise ValueError(f"--scored-run is missing a run name: {item!r}")
        if not normalized_path.exists():
            raise FileNotFoundError(f"--scored-run path not found: {normalized_path}")
        parsed[normalized_name] = normalized_path
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
