#!/usr/bin/env python3
"""Analyze Phase E same-source pair errors with source-specific stratification.

English
-------
Phase E already writes `eval_pair_scores.jsonl`, but that file alone is too thin
for root-cause diagnosis.  To understand why one source learns and another does
not, we need to join the scores back to the original pair metadata and then
slice the failures using source-specific fields.

This script does exactly that:
1. load canonical pair JSONL,
2. join with one `eval_pair_scores.jsonl`,
3. compute overall and stratified metrics,
4. persist the hardest false pairs for manual inspection.

中文
----
Phase E 已经会落盘 `eval_pair_scores.jsonl`，但光看这个文件不足以做根因诊断。
要理解“为什么某个 source 学得起来、另一个学不起来”，必须把预测分数重新和
原始 pair metadata 对齐，然后按 source 自己的字段去切分失败模式。

这个脚本的职责就是：
1. 读取 canonical pair JSONL，
2. 与一次 `eval_pair_scores.jsonl` 对齐，
3. 计算整体与分层指标，
4. 把最值得人工阅读的错误 pair 持久化出来。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Phase E same-source pair errors and emit source-aware diagnostics."
    )
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--eval-pair-scores-jsonl", type=Path, required=True)
    parser.add_argument("--run-name", default="phase_e_pair_error_analysis")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_error_analysis"),
    )
    parser.add_argument(
        "--dataset-profile",
        choices=["auto", "math_shepherd", "prmbench_preview", "r_prm"],
        default="auto",
        help="How to choose source-specific breakdown fields.",
    )
    parser.add_argument("--top-k-errors", type=int, default=25)
    return parser


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No JSONL rows loaded from {path}")
    return rows


def _bucket_numeric(value: Any, *, edges: list[float], labels: list[str]) -> str:
    if value is None:
        return "<missing>"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "<non_numeric>"
    for edge, label in zip(edges, labels, strict=True):
        if number <= edge:
            return label
    return labels[-1]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


def _resolve_profile(rows: list[dict[str, Any]], explicit: str) -> str:
    if explicit != "auto":
        return explicit
    source_tags = {str(row.get("source_tag", "")) for row in rows[: min(20, len(rows))]}
    if "math_shepherd" in source_tags:
        return "math_shepherd"
    if "prmbench_preview" in source_tags:
        return "prmbench_preview"
    if "r_prm" in source_tags:
        return "r_prm"
    return "auto"


def _iter_group_specs(profile: str) -> list[tuple[str, str]]:
    common = [
        ("source_tag", "source_tag"),
        ("domain_tag", "domain_tag"),
    ]
    if profile == "math_shepherd":
        return common + [
            ("task", "task"),
            ("first_negative_bucket", "first_negative_index"),
            ("num_steps_bucket", "num_step_labels"),
        ]
    if profile == "prmbench_preview":
        return common + [
            ("classification", "classification"),
            ("error_step_bucket", "error_step_index"),
            ("num_error_steps_bucket", "num_error_steps"),
        ]
    if profile == "r_prm":
        return common + [
            ("chosen_verdict", "chosen_verdict"),
            ("prompt_chars_bucket", "compact_prompt_chars"),
            ("raw_instruction_chars_bucket", "raw_instruction_chars"),
        ]
    return common


def _extract_group_value(enriched: dict[str, Any], *, group_name: str, field_name: str) -> str:
    meta = enriched.get("metadata") or {}
    if group_name == "first_negative_bucket":
        return _bucket_numeric(
            meta.get(field_name),
            edges=[1, 2, 3, 5, 999999],
            labels=["<=1", "<=2", "<=3", "<=5", ">5"],
        )
    if group_name == "num_steps_bucket":
        return _bucket_numeric(
            meta.get(field_name),
            edges=[2, 3, 5, 8, 999999],
            labels=["<=2", "<=3", "<=5", "<=8", ">8"],
        )
    if group_name == "error_step_bucket":
        return _bucket_numeric(
            meta.get(field_name),
            edges=[3, 6, 10, 15, 999999],
            labels=["<=3", "<=6", "<=10", "<=15", ">15"],
        )
    if group_name == "num_error_steps_bucket":
        return _bucket_numeric(
            meta.get(field_name),
            edges=[1, 2, 4, 8, 999999],
            labels=["1", "<=2", "<=4", "<=8", ">8"],
        )
    if group_name == "prompt_chars_bucket":
        return _bucket_numeric(
            meta.get(field_name),
            edges=[400, 700, 1000, 1400, 999999],
            labels=["<=400", "<=700", "<=1000", "<=1400", ">1400"],
        )
    if group_name == "raw_instruction_chars_bucket":
        return _bucket_numeric(
            meta.get(field_name),
            edges=[1200, 1800, 2400, 3200, 999999],
            labels=["<=1200", "<=1800", "<=2400", "<=3200", ">3200"],
        )
    if field_name in {"source_tag", "domain_tag"}:
        return str(enriched.get(field_name, "<missing>"))
    return str(meta.get(field_name, "<missing>"))


def _compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "pair_acc": 0.0,
            "pair_acc_with_tie": 0.0,
            "mean_margin": 0.0,
            "median_margin": 0.0,
            "p10_margin": 0.0,
            "wrong_rate": 0.0,
        }
    margins = [float(row["margin"]) for row in rows]
    wrong = [row for row in rows if float(row["margin"]) <= 0.0]
    ordered = sorted(margins)
    mid = len(ordered) // 2
    median = float(ordered[mid]) if len(ordered) % 2 == 1 else float((ordered[mid - 1] + ordered[mid]) / 2.0)
    return {
        "n": int(len(rows)),
        "pair_acc": float(sum(1 for value in margins if value > 0.0) / len(margins)),
        "pair_acc_with_tie": float(sum(1 for value in margins if value >= 0.0) / len(margins)),
        "mean_margin": float(sum(margins) / len(margins)),
        "median_margin": median,
        "p10_margin": _percentile(margins, 0.10),
        "wrong_rate": float(len(wrong) / len(rows)),
    }


def _top_error_rows(rows: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    wrong = [row for row in rows if float(row["margin"]) <= 0.0]
    wrong.sort(key=lambda item: (float(item["margin"]), float(item["chosen_score"])))
    return wrong[: max(1, int(top_k))]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    pair_rows = _load_jsonl(Path(args.pairs_jsonl))
    score_rows = _load_jsonl(Path(args.eval_pair_scores_jsonl))
    pairs_by_id = {str(row["pair_id"]): row for row in pair_rows}

    enriched_rows: list[dict[str, Any]] = []
    missing_pair_ids: list[str] = []
    for score_row in score_rows:
        pair_id = str(score_row["pair_id"])
        pair_row = pairs_by_id.get(pair_id)
        if pair_row is None:
            missing_pair_ids.append(pair_id)
            continue
        enriched_rows.append(
            {
                "pair_id": pair_id,
                "source_tag": pair_row.get("source_tag"),
                "domain_tag": pair_row.get("domain_tag"),
                "pair_confidence": pair_row.get("pair_confidence"),
                "margin": float(score_row["margin"]),
                "chosen_score": float(score_row["chosen_score"]),
                "rejected_score": float(score_row["rejected_score"]),
                "metadata": pair_row.get("metadata") or {},
                "quality_flags": pair_row.get("quality_flags") or {},
                "prompt_text": pair_row.get("prompt_text", ""),
                "chosen_text": pair_row.get("chosen_text", ""),
                "rejected_text": pair_row.get("rejected_text", ""),
            }
        )
    if not enriched_rows:
        raise RuntimeError("No joined rows produced; pair ids did not match eval outputs")

    profile = _resolve_profile(enriched_rows, str(args.dataset_profile))
    group_specs = _iter_group_specs(profile)

    grouped_metrics: dict[str, dict[str, Any]] = {}
    for group_name, field_name in group_specs:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in enriched_rows:
            buckets[_extract_group_value(row, group_name=group_name, field_name=field_name)].append(row)
        grouped_metrics[group_name] = {
            key: _compute_metrics(value_rows)
            for key, value_rows in sorted(buckets.items(), key=lambda item: item[0])
        }

    top_errors = _top_error_rows(enriched_rows, top_k=int(args.top_k_errors))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root) / f"{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pairs_jsonl": str(Path(args.pairs_jsonl)),
        "eval_pair_scores_jsonl": str(Path(args.eval_pair_scores_jsonl)),
        "dataset_profile": profile,
        "num_joined_rows": int(len(enriched_rows)),
        "num_missing_pair_ids": int(len(missing_pair_ids)),
        "overall": _compute_metrics(enriched_rows),
        "grouped_metrics": grouped_metrics,
        "top_error_count": int(len(top_errors)),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (run_dir / "top_errors.jsonl").open("w", encoding="utf-8") as handle:
        for row in top_errors:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Phase E Pair Error Analysis",
        "",
        f"- dataset_profile: `{profile}`",
        f"- pairs_jsonl: `{args.pairs_jsonl}`",
        f"- eval_pair_scores_jsonl: `{args.eval_pair_scores_jsonl}`",
        f"- num_joined_rows: `{len(enriched_rows)}`",
        f"- num_missing_pair_ids: `{len(missing_pair_ids)}`",
        "",
        "## Overall",
        "",
        f"- pair_acc: `{summary['overall']['pair_acc']:.6f}`",
        f"- pair_acc_with_tie: `{summary['overall']['pair_acc_with_tie']:.6f}`",
        f"- mean_margin: `{summary['overall']['mean_margin']:.6f}`",
        f"- median_margin: `{summary['overall']['median_margin']:.6f}`",
        f"- p10_margin: `{summary['overall']['p10_margin']:.6f}`",
        f"- wrong_rate: `{summary['overall']['wrong_rate']:.6f}`",
        "",
    ]
    for group_name, payload in grouped_metrics.items():
        lines.extend(
            [
                f"## {group_name}",
                "",
                "| bucket | n | pair_acc | mean_margin | p10_margin | wrong_rate |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        items = sorted(payload.items(), key=lambda item: (-int(item[1]["n"]), item[0]))
        for bucket, metrics in items:
            lines.append(
                "| "
                f"{bucket} | {metrics['n']} | {metrics['pair_acc']:.4f} | "
                f"{metrics['mean_margin']:.4f} | {metrics['p10_margin']:.4f} | {metrics['wrong_rate']:.4f} |"
            )
        lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 88)
    print("Phase E Pair Error Analysis")
    print("=" * 88)
    print(f"dataset_profile      : {profile}")
    print(f"num_joined_rows      : {len(enriched_rows)}")
    print(f"num_missing_pair_ids : {len(missing_pair_ids)}")
    print(f"pair_acc             : {summary['overall']['pair_acc']:.6f}")
    print(f"mean_margin          : {summary['overall']['mean_margin']:.6f}")
    print(f"wrong_rate           : {summary['overall']['wrong_rate']:.6f}")
    print(f"summary_path         : {run_dir / 'summary.json'}")
    print(f"summary_md           : {run_dir / 'summary.md'}")
    print(f"top_errors_path      : {run_dir / 'top_errors.jsonl'}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
