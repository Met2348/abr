#!/usr/bin/env python3
"""Audit PRM800K adapter semantics, especially how rating `0` is interpreted.

English
-------
The repo previously treated PRM800K completion ratings with the rule:
`positive = rating > 0`, `negative = rating <= 0`.

That is dangerous for public PRM800K mirrors because many rows use the
`1 / 0 / -1` convention:
- `1`: clearly positive
- `0`: neutral/acceptable
- `-1`: negative

If `0` is folded into the negative bucket, the loader can silently fabricate
pairs where the "rejected" completion is not actually wrong.

This script quantifies that risk on the local PRM800K snapshot and compares:
1. the legacy policy (`>0` vs `<=0`)
2. the current policy (`>=0` vs `<0`)

中文
----
这个脚本专门审计 `PRM800K` completion rating 的语义，重点回答：

1. 本地数据里 `1 / 0 / -1` 的分布到底是什么？
2. 旧策略（`>0` / `<=0`）会在多大程度上把 `0` 当成负例？
3. 新策略（`>=0` / `<0`）会让 pair 构造发生多大变化？

它的目标不是训练模型，而是先把 source 契约问题钉死，避免后续把“适配器读错”
误解释成“数据本身没用”。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

from ours.phase_d.external_pairs import summarize_external_pairs  # noqa: E402
from ours.phase_d.external_pairs_adapters import (  # noqa: E402
    PairBuildConfig,
    _build_record,
    _collect_prm800k_files,
    _extract_prm800k_prompt,
    _iter_json_records,
    _join_steps_as_prefix,
    _safe_rating_value,
    load_prm800k_pairs,
)


@dataclass(slots=True)
class PolicyPairStats:
    num_pairs: int = 0
    num_rows_with_pairs: int = 0
    num_steps_pairable: int = 0
    num_pairs_zero_as_positive: int = 0
    num_pairs_zero_as_negative: int = 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit PRM800K adapter semantics on a local snapshot.")
    parser.add_argument(
        "--prm800k-path",
        type=Path,
        default=Path("assets/external_datasets/openai_prm800k"),
    )
    parser.add_argument("--run-name", default="phase_e_prm800k_adapter_audit")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_prm800k_audit"),
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-length-ratio", type=float, default=4.0)
    parser.add_argument("--max-token-overlap", type=float, default=0.995)
    parser.add_argument("--max-pairs-per-sample", type=int, default=2)
    return parser


def _extract_pairs_with_policy(
    *,
    payload: dict[str, Any],
    prompt_text: str,
    source_file: Path,
    source_row_index: int,
    config: PairBuildConfig,
    non_negative_is_positive: bool,
) -> tuple[list[Any], PolicyPairStats]:
    label_obj = payload.get("label")
    if not isinstance(label_obj, dict):
        return [], PolicyPairStats()
    steps = label_obj.get("steps")
    if not isinstance(steps, list) or not steps:
        return [], PolicyPairStats()

    rows: list[Any] = []
    stats = PolicyPairStats()
    history_steps: list[str] = []
    saw_pair_for_row = False
    for step_idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        completions = step.get("completions")
        if not isinstance(completions, list):
            continue

        positive: tuple[str, float] | None = None
        negative: tuple[str, float] | None = None
        saw_zero = False
        for item in completions:
            if not isinstance(item, dict):
                continue
            text_raw = item.get("text")
            if not isinstance(text_raw, str):
                continue
            text = text_raw.strip()
            if text == "":
                continue
            rating = _safe_rating_value(item.get("rating"))
            if rating is None:
                continue
            if abs(float(rating)) < 1e-12:
                saw_zero = True
            is_positive = bool(rating >= 0.0) if non_negative_is_positive else bool(rating > 0.0)
            if is_positive:
                if positive is None or float(rating) > float(positive[1]):
                    positive = (text, float(rating))
            else:
                if negative is None or float(rating) < float(negative[1]):
                    negative = (text, float(rating))

        if positive is not None and negative is not None:
            stats.num_steps_pairable += 1
            chosen_prefix = _join_steps_as_prefix([*history_steps, positive[0]], len(history_steps))
            rejected_prefix = _join_steps_as_prefix([*history_steps, negative[0]], len(history_steps))
            record = _build_record(
                source_tag="prm800k",
                domain_tag="general_math",
                prompt_text=f"{prompt_text}\n\n",
                chosen_text=chosen_prefix,
                rejected_text=rejected_prefix,
                pair_confidence=0.7,
                metadata={
                    "source_file": str(source_file),
                    "source_row_index": int(source_row_index),
                    "positive_step_index": int(step_idx),
                    "negative_step_index": int(step_idx),
                    "rating_positive": float(positive[1]),
                    "rating_negative": float(negative[1]),
                    "rating_policy": (
                        "non_negative_positive" if non_negative_is_positive else "strict_positive_only"
                    ),
                },
                config=config,
            )
            if record is not None:
                rows.append(record)
                saw_pair_for_row = True
                stats.num_pairs += 1
                if abs(float(positive[1])) < 1e-12:
                    stats.num_pairs_zero_as_positive += 1
                if abs(float(negative[1])) < 1e-12:
                    stats.num_pairs_zero_as_negative += 1
                if len(rows) >= int(config.max_pairs_per_sample):
                    break

        chosen_idx = step.get("chosen_completion")
        next_history = None
        if isinstance(chosen_idx, int) and 0 <= chosen_idx < len(completions):
            chosen_item = completions[chosen_idx]
            if isinstance(chosen_item, dict):
                text_raw = chosen_item.get("text")
                if isinstance(text_raw, str) and text_raw.strip():
                    next_history = text_raw.strip()
        if next_history is None and positive is not None:
            next_history = positive[0]
        if next_history is not None:
            history_steps.append(next_history)
        del saw_zero

    if saw_pair_for_row:
        stats.num_rows_with_pairs = 1
    return rows, stats


def _render_markdown(summary: dict[str, Any]) -> str:
    legacy = dict(summary["legacy_policy"])
    current = dict(summary["current_policy"])
    lines = [
        "# PRM800K Adapter Audit",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- prm800k_path: `{summary['prm800k_path']}`",
        f"- scanned_rows: `{summary['scanned_rows']}`",
        "",
        "## Rating Histogram",
        "",
        f"- +1: `{summary['rating_histogram'].get('1', 0)}`",
        f"- 0: `{summary['rating_histogram'].get('0', 0)}`",
        f"- -1: `{summary['rating_histogram'].get('-1', 0)}`",
        f"- other: `{summary['rating_histogram'].get('other', 0)}`",
        "",
        "## Policy Comparison",
        "",
        f"- legacy_num_pairs: `{legacy['num_pairs']}`",
        f"- legacy_zero_as_negative: `{legacy['num_pairs_zero_as_negative']}`",
        f"- current_num_pairs: `{current['num_pairs']}`",
        f"- current_zero_as_positive: `{current['num_pairs_zero_as_positive']}`",
        "",
        "## Current Loader Summary",
        "",
        f"- num_pairs: `{summary['current_loader_summary']['num_pairs']}`",
        f"- mean_pair_confidence: `{float(summary['current_loader_summary']['mean_pair_confidence']):.6f}`",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = PairBuildConfig(
        min_chars=int(args.min_chars),
        max_length_ratio=float(args.max_length_ratio),
        max_token_overlap=float(args.max_token_overlap),
        max_pairs_per_sample=int(args.max_pairs_per_sample),
    )
    config.validate()

    run_dir = Path(args.output_root) / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_json = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"

    rating_hist = Counter()
    scanned_rows = 0
    legacy_rows_total = 0
    legacy_policy = PolicyPairStats()
    current_policy = PolicyPairStats()

    files = _collect_prm800k_files(Path(args.prm800k_path))
    for file_path in files:
        for row_idx, payload in enumerate(_iter_json_records(file_path), start=1):
            scanned_rows += 1
            label_obj = payload.get("label")
            if isinstance(label_obj, dict):
                steps = label_obj.get("steps")
                if isinstance(steps, list):
                    for step in steps:
                        if not isinstance(step, dict):
                            continue
                        completions = step.get("completions")
                        if not isinstance(completions, list):
                            continue
                        for item in completions:
                            if not isinstance(item, dict):
                                continue
                            rating = _safe_rating_value(item.get("rating"))
                            if rating is None:
                                continue
                            if abs(float(rating) - 1.0) < 1e-12:
                                rating_hist["1"] += 1
                            elif abs(float(rating)) < 1e-12:
                                rating_hist["0"] += 1
                            elif abs(float(rating) + 1.0) < 1e-12:
                                rating_hist["-1"] += 1
                            else:
                                rating_hist["other"] += 1

            prompt = _extract_prm800k_prompt(payload)
            _, legacy_stats = _extract_pairs_with_policy(
                payload=payload,
                prompt_text=prompt,
                source_file=file_path,
                source_row_index=int(row_idx),
                config=config,
                non_negative_is_positive=False,
            )
            _, current_stats = _extract_pairs_with_policy(
                payload=payload,
                prompt_text=prompt,
                source_file=file_path,
                source_row_index=int(row_idx),
                config=config,
                non_negative_is_positive=True,
            )
            for key in ("num_pairs", "num_rows_with_pairs", "num_steps_pairable", "num_pairs_zero_as_positive", "num_pairs_zero_as_negative"):
                setattr(legacy_policy, key, int(getattr(legacy_policy, key) + getattr(legacy_stats, key)))
                setattr(current_policy, key, int(getattr(current_policy, key) + getattr(current_stats, key)))

            if args.max_rows is not None and scanned_rows >= int(args.max_rows):
                break
        if args.max_rows is not None and scanned_rows >= int(args.max_rows):
            break

    current_loader_rows = load_prm800k_pairs(
        path=Path(args.prm800k_path),
        config=config,
        max_pairs=(None if args.max_rows is None else int(args.max_rows) * int(args.max_pairs_per_sample)),
    )
    current_loader_summary = summarize_external_pairs(current_loader_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "prm800k_path": str(args.prm800k_path),
        "scanned_rows": int(scanned_rows),
        "rating_histogram": dict(rating_hist),
        "legacy_policy": {
            "num_pairs": int(legacy_policy.num_pairs),
            "num_rows_with_pairs": int(legacy_policy.num_rows_with_pairs),
            "num_steps_pairable": int(legacy_policy.num_steps_pairable),
            "num_pairs_zero_as_positive": int(legacy_policy.num_pairs_zero_as_positive),
            "num_pairs_zero_as_negative": int(legacy_policy.num_pairs_zero_as_negative),
        },
        "current_policy": {
            "num_pairs": int(current_policy.num_pairs),
            "num_rows_with_pairs": int(current_policy.num_rows_with_pairs),
            "num_steps_pairable": int(current_policy.num_steps_pairable),
            "num_pairs_zero_as_positive": int(current_policy.num_pairs_zero_as_positive),
            "num_pairs_zero_as_negative": int(current_policy.num_pairs_zero_as_negative),
        },
        "current_loader_summary": current_loader_summary,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md.write_text(_render_markdown(summary), encoding="utf-8")

    print("=" * 88)
    print("PRM800K Adapter Audit")
    print("=" * 88)
    print(f"run_dir                  : {run_dir}")
    print(f"scanned_rows             : {scanned_rows}")
    print(f"rating_histogram         : {dict(rating_hist)}")
    print(f"legacy_num_pairs         : {legacy_policy.num_pairs}")
    print(f"legacy_zero_as_negative  : {legacy_policy.num_pairs_zero_as_negative}")
    print(f"current_num_pairs        : {current_policy.num_pairs}")
    print(f"current_zero_as_positive : {current_policy.num_pairs_zero_as_positive}")
    print(f"summary_json             : {summary_json}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
