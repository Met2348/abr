#!/usr/bin/env python3
"""Select the most trustworthy Phase E value-head candidate from suite outputs.

Why this file exists
--------------------
Phase E no longer asks only "can we train a value head once?".
For later RL-style stages, we need a stricter question:

1. which suite configuration is strongest on held-out source-family pairs,
2. which one remains least fragile across seeds,
3. which checkpoint file should be promoted as the current best candidate?

This script turns those judgement rules into a repeatable report instead of
leaving them as ad-hoc manual reading.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SeedCandidateRow:
    """One seed-level row extracted from a Phase E suite seed result file."""

    seed: int
    value_run_dir: str
    heldout_pair_acc: float
    heldout_auc: float
    heldout_ranking_score: float
    benchmarks: dict[str, dict[str, float]]


@dataclass(slots=True)
class GroupCandidateSummary:
    """Aggregate trust summary for one suite group."""

    group_id: str
    group_title: str
    status: str
    suite_log_dir: str
    summary_path: str
    num_seeds: int
    mean_heldout_pair_acc: float
    mean_heldout_auc: float
    mean_heldout_ranking_score: float
    std_heldout_pair_acc: float | None
    std_heldout_auc: float | None
    worst_seed_heldout_pair_acc: float
    worst_seed_heldout_auc: float
    benchmark_means: dict[str, dict[str, float]]
    benchmark_stds: dict[str, dict[str, float]]
    eligible_for_candidate: bool
    gate_pass: bool
    trust_score: float | None
    best_seed: int | None
    best_checkpoint_path: str | None
    best_seed_score: float | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select the most trustworthy Phase E value-head checkpoint from suite logs."
    )
    parser.add_argument(
        "--suite-log-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more Phase E suite log directories that contain final_summary.md and seed_results.jsonl",
    )
    parser.add_argument(
        "--required-benchmark-ids",
        nargs="*",
        default=["processbench_gsm8k", "processbench_math"],
        help="Benchmarks that must be present before a group can be promoted as a trustworthy candidate.",
    )
    parser.add_argument("--min-heldout-pair-acc", type=float, default=0.70)
    parser.add_argument("--min-heldout-auc", type=float, default=0.70)
    parser.add_argument("--min-worst-seed-heldout-pair-acc", type=float, default=0.55)
    parser.add_argument("--min-worst-seed-heldout-auc", type=float, default=0.55)
    parser.add_argument("--max-heldout-pair-std", type=float, default=0.12)
    parser.add_argument("--max-heldout-auc-std", type=float, default=0.12)
    parser.add_argument("--min-benchmark-pair-acc", type=float, default=0.50)
    parser.add_argument("--min-benchmark-auc", type=float, default=0.50)
    parser.add_argument("--run-name", default="phase_e_candidate")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_candidates"),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    for suite_dir in args.suite_log_dirs:
        if not suite_dir.exists():
            raise FileNotFoundError(f"--suite-log-dirs entry not found: {suite_dir}")
    if not args.required_benchmark_ids:
        raise ValueError("--required-benchmark-ids must contain at least one benchmark")
    return args


def _parse_summary_header(summary_path: Path) -> dict[str, str]:
    text = summary_path.read_text(encoding="utf-8")

    def grab(key: str) -> str:
        match = re.search(rf"^- {re.escape(key)}: (.+)$", text, flags=re.MULTILINE)
        return match.group(1).strip() if match else ""

    return {
        "group_id": grab("group_id"),
        "group_title": grab("group_title"),
        "status": grab("status"),
    }


def _load_seed_rows(rows_path: Path) -> list[SeedCandidateRow]:
    rows: list[SeedCandidateRow] = []
    for raw in rows_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        record = json.loads(raw)
        rows.append(
            SeedCandidateRow(
                seed=int(record["seed"]),
                value_run_dir=str(record["value_run_dir"]),
                heldout_pair_acc=float(record["heldout_pair_acc"]),
                heldout_auc=float(record["heldout_auc"]),
                heldout_ranking_score=float(record.get("heldout_ranking_score", 0.0)),
                benchmarks={
                    str(bench_id): {
                        "pair_acc": float(item.get("pair_acc", 0.0)),
                        "auc": float(item.get("auc", 0.0)),
                    }
                    for bench_id, item in dict(record.get("benchmarks", {})).items()
                },
            )
        )
    return rows


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def _std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return None
    return float(statistics.pstdev(values))


def _seed_score(row: SeedCandidateRow, required_benchmark_ids: list[str]) -> float:
    # The per-seed score prefers ranking quality first, but still rewards
    # benchmark-native behavior. This is intentionally not a calibrated
    # probability; it is a checkpoint-promotion heuristic.
    # 这里的 seed score 首先看排序质量，其次看 benchmark-native 行为。
    # 它不是“真实概率”，只是一个固定的 checkpoint 晋级启发式。
    benchmark_pair_mean = _mean(
        [float(row.benchmarks.get(bench_id, {}).get("pair_acc", 0.0)) for bench_id in required_benchmark_ids]
    )
    benchmark_auc_mean = _mean(
        [float(row.benchmarks.get(bench_id, {}).get("auc", 0.0)) for bench_id in required_benchmark_ids]
    )
    return (
        0.40 * float(row.heldout_ranking_score)
        + 0.25 * float(row.heldout_auc)
        + 0.15 * benchmark_pair_mean
        + 0.20 * benchmark_auc_mean
    )


def _build_group_summary(
    *,
    suite_log_dir: Path,
    required_benchmark_ids: list[str],
    args: argparse.Namespace,
) -> GroupCandidateSummary:
    summary_path = suite_log_dir / "final_summary.md"
    rows_path = suite_log_dir / "seed_results.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing final_summary.md: {summary_path}")
    if not rows_path.exists():
        raise FileNotFoundError(f"Missing seed_results.jsonl: {rows_path}")

    header = _parse_summary_header(summary_path)
    seed_rows = _load_seed_rows(rows_path)
    if not seed_rows:
        raise ValueError(f"No seed rows found in {rows_path}")

    notes: list[str] = []
    heldout_pair_accs = [row.heldout_pair_acc for row in seed_rows]
    heldout_aucs = [row.heldout_auc for row in seed_rows]
    heldout_rankings = [row.heldout_ranking_score for row in seed_rows]

    benchmark_means: dict[str, dict[str, float]] = {}
    benchmark_stds: dict[str, dict[str, float]] = {}
    eligible_for_candidate = True
    for bench_id in required_benchmark_ids:
        pair_values: list[float] = []
        auc_values: list[float] = []
        for row in seed_rows:
            bench_metrics = row.benchmarks.get(bench_id)
            if bench_metrics is None:
                eligible_for_candidate = False
                notes.append(f"Missing required benchmark metrics: {bench_id}")
                break
            pair_values.append(float(bench_metrics["pair_acc"]))
            auc_values.append(float(bench_metrics["auc"]))
        if len(pair_values) != len(seed_rows):
            continue
        benchmark_means[bench_id] = {
            "pair_acc": _mean(pair_values),
            "auc": _mean(auc_values),
        }
        benchmark_stds[bench_id] = {
            "pair_acc": _std(pair_values) or 0.0,
            "auc": _std(auc_values) or 0.0,
        }

    std_pair = _std(heldout_pair_accs)
    std_auc = _std(heldout_aucs)
    worst_pair = min(heldout_pair_accs)
    worst_auc = min(heldout_aucs)

    gate_pass = bool(eligible_for_candidate)
    if _mean(heldout_pair_accs) < float(args.min_heldout_pair_acc):
        gate_pass = False
        notes.append("mean held-out pair_acc below threshold")
    if _mean(heldout_aucs) < float(args.min_heldout_auc):
        gate_pass = False
        notes.append("mean held-out auc below threshold")
    if worst_pair < float(args.min_worst_seed_heldout_pair_acc):
        gate_pass = False
        notes.append("worst-seed held-out pair_acc below threshold")
    if worst_auc < float(args.min_worst_seed_heldout_auc):
        gate_pass = False
        notes.append("worst-seed held-out auc below threshold")
    if std_pair is not None and std_pair > float(args.max_heldout_pair_std):
        gate_pass = False
        notes.append("held-out pair_acc seed std above threshold")
    if std_auc is not None and std_auc > float(args.max_heldout_auc_std):
        gate_pass = False
        notes.append("held-out auc seed std above threshold")
    for bench_id in required_benchmark_ids:
        bench_mean = benchmark_means.get(bench_id)
        if bench_mean is None:
            continue
        if float(bench_mean["pair_acc"]) < float(args.min_benchmark_pair_acc):
            gate_pass = False
            notes.append(f"{bench_id} mean pair_acc below threshold")
        if float(bench_mean["auc"]) < float(args.min_benchmark_auc):
            gate_pass = False
            notes.append(f"{bench_id} mean auc below threshold")

    trust_score: float | None = None
    if eligible_for_candidate:
        benchmark_pair_mean = _mean(
            [benchmark_means[bench_id]["pair_acc"] for bench_id in required_benchmark_ids if bench_id in benchmark_means]
        )
        benchmark_auc_mean = _mean(
            [benchmark_means[bench_id]["auc"] for bench_id in required_benchmark_ids if bench_id in benchmark_means]
        )
        std_pair_term = 1.0 - min(float(std_pair or 0.0), 1.0)
        std_auc_term = 1.0 - min(float(std_auc or 0.0), 1.0)
        trust_score = float(
            0.30 * _mean(heldout_rankings)
            + 0.20 * _mean(heldout_aucs)
            + 0.20 * benchmark_pair_mean
            + 0.20 * benchmark_auc_mean
            + 0.05 * std_pair_term
            + 0.05 * std_auc_term
        )

    best_seed: int | None = None
    best_checkpoint_path: str | None = None
    best_seed_score: float | None = None
    if eligible_for_candidate:
        per_seed_scored = sorted(
            ((_seed_score(row, required_benchmark_ids), row) for row in seed_rows),
            key=lambda item: item[0],
            reverse=True,
        )
        best_seed_score, best_row = per_seed_scored[0]
        best_seed = int(best_row.seed)
        best_checkpoint_path = str(Path(best_row.value_run_dir) / "best_value_head.pt")

    return GroupCandidateSummary(
        group_id=str(header.get("group_id") or suite_log_dir.name),
        group_title=str(header.get("group_title") or ""),
        status=str(header.get("status") or "unknown"),
        suite_log_dir=str(suite_log_dir),
        summary_path=str(summary_path),
        num_seeds=len(seed_rows),
        mean_heldout_pair_acc=_mean(heldout_pair_accs),
        mean_heldout_auc=_mean(heldout_aucs),
        mean_heldout_ranking_score=_mean(heldout_rankings),
        std_heldout_pair_acc=std_pair,
        std_heldout_auc=std_auc,
        worst_seed_heldout_pair_acc=worst_pair,
        worst_seed_heldout_auc=worst_auc,
        benchmark_means=benchmark_means,
        benchmark_stds=benchmark_stds,
        eligible_for_candidate=eligible_for_candidate,
        gate_pass=gate_pass,
        trust_score=trust_score,
        best_seed=best_seed,
        best_checkpoint_path=best_checkpoint_path,
        best_seed_score=best_seed_score,
        notes=notes,
    )


def _render_markdown(
    *,
    output_path: Path,
    groups: list[GroupCandidateSummary],
    required_benchmark_ids: list[str],
    selected_group: GroupCandidateSummary | None,
    selected_mode: str,
    args: argparse.Namespace,
    json_report_path: Path,
) -> None:
    lines = [
        "# Phase E Candidate Report",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- required_benchmarks: {', '.join(required_benchmark_ids)}",
        f"- min_heldout_pair_acc: {args.min_heldout_pair_acc}",
        f"- min_heldout_auc: {args.min_heldout_auc}",
        f"- min_worst_seed_pair_acc: {args.min_worst_seed_heldout_pair_acc}",
        f"- min_worst_seed_auc: {args.min_worst_seed_heldout_auc}",
        f"- max_heldout_pair_std: {args.max_heldout_pair_std}",
        f"- max_heldout_auc_std: {args.max_heldout_auc_std}",
        f"- min_benchmark_pair_acc: {args.min_benchmark_pair_acc}",
        f"- min_benchmark_auc: {args.min_benchmark_auc}",
        f"- json_report: `{json_report_path}`",
        "",
    ]
    if selected_group is not None:
        lines.extend(
            [
                "## Recommended Candidate",
                "",
                f"- mode: `{selected_mode}`",
                f"- group_id: `{selected_group.group_id}`",
                f"- group_title: `{selected_group.group_title}`",
                f"- trust_score: `{float(selected_group.trust_score or 0.0):.6f}`",
                f"- best_seed: `{selected_group.best_seed}`",
                f"- best_checkpoint_path: `{selected_group.best_checkpoint_path}`",
                f"- summary_path: `{selected_group.summary_path}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Recommended Candidate",
                "",
                "- No eligible group was found.",
                "",
            ]
        )

    lines.extend(
        [
            "## Group Comparison",
            "",
            "| group_id | status | eligible | gate_pass | mean_hold_pair | mean_hold_auc | worst_hold_pair | worst_hold_auc | std_hold_pair | std_hold_auc | trust_score |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in groups:
        def fmt(value: float | None) -> str:
            if value is None:
                return "N/A"
            return f"{float(value):.4f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    group.group_id,
                    group.status,
                    "1" if group.eligible_for_candidate else "0",
                    "1" if group.gate_pass else "0",
                    fmt(group.mean_heldout_pair_acc),
                    fmt(group.mean_heldout_auc),
                    fmt(group.worst_seed_heldout_pair_acc),
                    fmt(group.worst_seed_heldout_auc),
                    fmt(group.std_heldout_pair_acc),
                    fmt(group.std_heldout_auc),
                    fmt(group.trust_score),
                ]
            )
            + " |"
        )
        for bench_id in required_benchmark_ids:
            bench = group.benchmark_means.get(bench_id)
            if bench is None:
                continue
            lines.append(
                f"  {bench_id}: pair_acc=`{bench['pair_acc']:.4f}`, auc=`{bench['auc']:.4f}`"
            )
        if group.best_checkpoint_path:
            lines.append(f"  best_checkpoint: `{group.best_checkpoint_path}`")
        if group.notes:
            lines.append(f"  notes: {'; '.join(group.notes)}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    groups = [
        _build_group_summary(
            suite_log_dir=Path(suite_dir),
            required_benchmark_ids=[str(item) for item in args.required_benchmark_ids],
            args=args,
        )
        for suite_dir in args.suite_log_dirs
    ]
    groups.sort(key=lambda item: (item.trust_score is not None, item.trust_score or -1.0), reverse=True)

    gated_groups = [group for group in groups if group.gate_pass and group.trust_score is not None]
    selected_mode = "gated"
    selected_group: GroupCandidateSummary | None = gated_groups[0] if gated_groups else None
    if selected_group is None:
        eligible_groups = [group for group in groups if group.eligible_for_candidate and group.trust_score is not None]
        selected_group = eligible_groups[0] if eligible_groups else None
        selected_mode = "provisional" if selected_group is not None else "none"

    run_dir = Path(args.output_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = run_dir / "candidate_report.json"
    report_md_path = run_dir / "candidate_report.md"
    manifest_path = run_dir / "manifest.json"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite_log_dirs": [str(Path(item)) for item in args.suite_log_dirs],
        "required_benchmark_ids": [str(item) for item in args.required_benchmark_ids],
        "selected_mode": selected_mode,
        "selected_group": selected_group.to_dict() if selected_group is not None else None,
        "groups": [group.to_dict() for group in groups],
        "thresholds": {
            "min_heldout_pair_acc": float(args.min_heldout_pair_acc),
            "min_heldout_auc": float(args.min_heldout_auc),
            "min_worst_seed_heldout_pair_acc": float(args.min_worst_seed_heldout_pair_acc),
            "min_worst_seed_heldout_auc": float(args.min_worst_seed_heldout_auc),
            "max_heldout_pair_std": float(args.max_heldout_pair_std),
            "max_heldout_auc_std": float(args.max_heldout_auc_std),
            "min_benchmark_pair_acc": float(args.min_benchmark_pair_acc),
            "min_benchmark_auc": float(args.min_benchmark_auc),
        },
    }
    report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _render_markdown(
        output_path=report_md_path,
        groups=groups,
        required_benchmark_ids=[str(item) for item in args.required_benchmark_ids],
        selected_group=selected_group,
        selected_mode=selected_mode,
        args=args,
        json_report_path=report_json_path,
    )
    manifest = {
        "artifact_stage": "phase_e_candidate_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_json_path": str(report_json_path),
        "report_md_path": str(report_md_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 88)
    print("Phase E: Select Candidate")
    print("=" * 88)
    print(f"run_dir            : {run_dir}")
    print(f"selected_mode      : {selected_mode}")
    if selected_group is not None:
        print(f"selected_group_id  : {selected_group.group_id}")
        print(f"selected_trust     : {float(selected_group.trust_score or 0.0):.6f}")
        print(f"selected_checkpoint: {selected_group.best_checkpoint_path}")
    else:
        print("selected_group_id  : <none>")
    print(f"report_json_path   : {report_json_path}")
    print(f"report_md_path     : {report_md_path}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
