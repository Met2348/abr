#!/usr/bin/env python3
"""Audit the current R-PRM contract used by Phase E.

English
-------
This script answers a narrower question than training:
given the raw R-PRM DPO rows, what exactly survives the current
`compact_verdict` contract, and what structural signal remains after the
rewrite?

It therefore reports:
1. raw-row counts and compact-contract drop reasons,
2. chosen-verdict balance,
3. prompt / step statistics,
4. token-length and cutoff-risk summaries under several `max_length` values.

中文
----
这个脚本不关心“最后训得好不好”，而是先回答一个更基础的问题：
给定原始 R-PRM DPO 行，当前 `compact_verdict` 合同到底保留了什么，
又丢掉了什么。

因此它会汇报：
1. 原始行数与 compact 合同的丢弃原因，
2. chosen verdict 的分布，
3. prompt / step 统计，
4. 在多个 `max_length` 下的 token 长度与 cutoff 风险。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_d.external_pairs_adapters import (  # noqa: E402
    PairBuildConfig,
    _R_PRM_CASE_RE,
    _build_record,
    _extract_r_prm_compact_prompt,
    _extract_r_prm_verdict,
    _iter_parquet_rows,
    _render_r_prm_verdict_text,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the current compact-verdict R-PRM contract used by Phase E."
    )
    parser.add_argument("--r-prm-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--run-name", default="phase_e_rprm_contract_audit")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_rprm_audit"),
    )
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--max-lengths", type=int, nargs="+", default=[1024, 1280, 1536, 2048])
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    source_root = Path(args.r_prm_root) / "dpo" / str(args.split)
    if not source_root.exists():
        raise FileNotFoundError(f"R-PRM split dir not found: {source_root}")
    if args.adapter_path is not None and not Path(args.adapter_path).exists():
        raise FileNotFoundError(f"--adapter-path not found: {args.adapter_path}")
    if int(args.max_rows) <= 0:
        raise ValueError("--max-rows must be > 0")
    if not args.max_lengths:
        raise ValueError("--max-lengths must not be empty")
    if any(int(value) <= 8 for value in args.max_lengths):
        raise ValueError("--max-lengths values must all be > 8")
    return args


def _stats(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    ordered = sorted(float(v) for v in values)

    def pick(q: float) -> int:
        idx = int(round(q * (len(ordered) - 1)))
        return int(ordered[idx])

    return {
        "count": int(len(ordered)),
        "mean": float(mean(ordered)),
        "min": int(ordered[0]),
        "p50": pick(0.50),
        "p75": pick(0.75),
        "p90": pick(0.90),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": int(ordered[-1]),
    }


def _count_steps(previous_steps: str) -> int:
    return int(len(re.findall(r"(?m)^Step\\s+\\d+:", str(previous_steps))))


def _tokenize_pair(tokenizer: Any, prompt_text: str, chosen_text: str, rejected_text: str) -> dict[str, int]:
    chosen_ids = tokenizer.encode(f"{prompt_text}{chosen_text}", add_special_tokens=False)
    rejected_ids = tokenizer.encode(f"{prompt_text}{rejected_text}", add_special_tokens=False)
    first_diff = min(len(chosen_ids), len(rejected_ids))
    for idx, (left, right) in enumerate(zip(chosen_ids, rejected_ids)):
        if int(left) != int(right):
            first_diff = idx
            break
    return {
        "chosen_length": int(len(chosen_ids)),
        "rejected_length": int(len(rejected_ids)),
        "first_diff_token_index": int(first_diff),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.adapter_path if args.adapter_path is not None else args.model_path),
        trust_remote_code=True,
    )
    config = PairBuildConfig(r_prm_pair_mode="compact_verdict")
    source_root = Path(args.r_prm_root) / "dpo" / str(args.split)
    files = sorted(source_root.glob("*.parquet"))

    reason_counts: Counter[str] = Counter()
    chosen_verdict_counts: Counter[str] = Counter()
    prompt_chars: list[int] = []
    prompt_token_lengths: list[int] = []
    first_diff_token_indices: list[int] = []
    previous_step_counts: list[int] = []
    now_step_chars: list[int] = []
    accepted_rows = 0
    total_rows = 0
    over_limit_by_length: dict[int, int] = {int(v): 0 for v in args.max_lengths}
    hidden_diff_by_length: dict[int, int] = {int(v): 0 for v in args.max_lengths}

    for row_idx, payload in enumerate(
        _iter_parquet_rows(files=files, columns=("instruction", "chosen", "rejected")),
        start=0,
    ):
        if row_idx >= int(args.max_rows):
            break
        total_rows += 1
        raw_instruction = str(payload.get("instruction", ""))
        raw_chosen = str(payload.get("chosen", ""))
        raw_rejected = str(payload.get("rejected", ""))

        compact_prompt = _extract_r_prm_compact_prompt(raw_instruction)
        if compact_prompt is None:
            reason_counts["compact_prompt_parse_fail"] += 1
            continue
        chosen_verdict = _extract_r_prm_verdict(raw_chosen)
        if chosen_verdict is None:
            reason_counts["chosen_verdict_parse_fail"] += 1
            continue
        rejected_verdict = _extract_r_prm_verdict(raw_rejected)
        if rejected_verdict is None:
            reason_counts["rejected_verdict_parse_fail"] += 1
            continue
        if chosen_verdict == rejected_verdict:
            reason_counts["same_verdict"] += 1
            continue

        chosen_text = _render_r_prm_verdict_text(chosen_verdict)
        rejected_text = _render_r_prm_verdict_text(rejected_verdict)
        record = _build_record(
            source_tag="r_prm",
            domain_tag="general_math",
            prompt_text=compact_prompt,
            chosen_text=chosen_text,
            rejected_text=rejected_text,
            pair_confidence=0.86,
            metadata={
                "source_split": str(args.split),
                "source_row_index": int(row_idx),
                "pair_build_mode": "r_prm_compact_verdict_pair",
                "pair_semantics": "same_prompt_binary_verdict",
                "chosen_verdict": chosen_verdict,
                "rejected_verdict": rejected_verdict,
            },
            config=config,
        )
        if record is None:
            reason_counts["quality_filter_drop"] += 1
            continue

        accepted_rows += 1
        chosen_verdict_counts[chosen_verdict] += 1
        prompt_chars.append(len(compact_prompt))
        token_payload = _tokenize_pair(tokenizer, compact_prompt, chosen_text, rejected_text)
        prompt_token_lengths.append(int(token_payload["chosen_length"]))
        first_diff_token_indices.append(int(token_payload["first_diff_token_index"]))
        for max_length in over_limit_by_length:
            if int(token_payload["chosen_length"]) > int(max_length) or int(token_payload["rejected_length"]) > int(max_length):
                over_limit_by_length[int(max_length)] += 1
            if int(token_payload["first_diff_token_index"]) >= int(max_length):
                hidden_diff_by_length[int(max_length)] += 1

        match = _R_PRM_CASE_RE.search(str(raw_instruction).replace("\r", "\n").strip())
        if match is not None:
            previous_step_counts.append(_count_steps(match.group(2).strip()))
            now_step_chars.append(len(match.group(3).strip()))

    rejected_rows = int(total_rows - accepted_rows)
    acceptance_rate = float(accepted_rows / total_rows) if total_rows else 0.0
    compact_prompt_over_limit = {
        str(max_length): {
            "frac_pairs_over_limit": float(over_limit_by_length[int(max_length)] / accepted_rows) if accepted_rows else 0.0,
            "frac_pairs_hidden_diff_after_cutoff": float(hidden_diff_by_length[int(max_length)] / accepted_rows) if accepted_rows else 0.0,
        }
        for max_length in sorted({int(v) for v in args.max_lengths})
    }

    run_dir = Path(args.output_root) / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "r_prm_root": str(args.r_prm_root),
        "split": str(args.split),
        "max_rows": int(args.max_rows),
        "raw_rows_audited": int(total_rows),
        "accepted_rows": int(accepted_rows),
        "rejected_rows": int(rejected_rows),
        "acceptance_rate": float(acceptance_rate),
        "reject_reason_counts": dict(reason_counts),
        "chosen_verdict_counts": dict(chosen_verdict_counts),
        "prompt_chars": _stats(prompt_chars),
        "prompt_token_lengths": _stats(prompt_token_lengths),
        "first_diff_token_index": _stats(first_diff_token_indices),
        "previous_step_count": _stats(previous_step_counts),
        "now_step_chars": _stats(now_step_chars),
        "cutoff_risk_by_max_length": compact_prompt_over_limit,
    }

    summary_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(_render_markdown(summary), encoding="utf-8")

    print("=" * 88)
    print("Phase E: Audit R-PRM Compact Contract")
    print("=" * 88)
    print(f"r_prm_root         : {args.r_prm_root}")
    print(f"split              : {args.split}")
    print(f"raw_rows_audited   : {total_rows}")
    print(f"accepted_rows      : {accepted_rows}")
    print(f"acceptance_rate    : {acceptance_rate:.4f}")
    print(f"chosen_yes         : {int(chosen_verdict_counts.get('yes', 0))}")
    print(f"chosen_no          : {int(chosen_verdict_counts.get('no', 0))}")
    for max_length in sorted({int(v) for v in args.max_lengths}):
        payload = compact_prompt_over_limit[str(max_length)]
        print(
            "cutoff_diag        : "
            f"max_length={max_length} "
            f"over_limit={payload['frac_pairs_over_limit']:.4f} "
            f"hidden_diff_after_cut={payload['frac_pairs_hidden_diff_after_cutoff']:.4f}"
        )
    print(f"summary_json       : {summary_path}")
    print(f"summary_md         : {summary_md_path}")
    print("=" * 88)
    return 0


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase E R-PRM Compact Contract Audit",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- r_prm_root: `{summary['r_prm_root']}`",
        f"- split: `{summary['split']}`",
        f"- raw_rows_audited: `{summary['raw_rows_audited']}`",
        f"- accepted_rows: `{summary['accepted_rows']}`",
        f"- acceptance_rate: `{summary['acceptance_rate']:.4f}`",
        "",
        "## Reject Reasons",
        "",
        "| reason | count |",
        "|---|---:|",
    ]
    for reason, count in sorted(dict(summary["reject_reason_counts"]).items()):
        lines.append(f"| {reason} | {int(count)} |")
    lines.extend(
        [
            "",
            "## Verdict Balance",
            "",
            "| chosen_verdict | count |",
            "|---|---:|",
        ]
    )
    for verdict, count in sorted(dict(summary["chosen_verdict_counts"]).items()):
        lines.append(f"| {verdict} | {int(count)} |")
    lines.extend(
        [
            "",
            "## Structural Stats",
            "",
            f"- prompt_chars: `{summary['prompt_chars']}`",
            f"- prompt_token_lengths: `{summary['prompt_token_lengths']}`",
            f"- first_diff_token_index: `{summary['first_diff_token_index']}`",
            f"- previous_step_count: `{summary['previous_step_count']}`",
            f"- now_step_chars: `{summary['now_step_chars']}`",
            "",
            "## Cutoff Risk",
            "",
            "| max_length | frac_pairs_over_limit | frac_pairs_hidden_diff_after_cutoff |",
            "|---|---:|---:|",
        ]
    )
    for max_length, payload in dict(summary["cutoff_risk_by_max_length"]).items():
        lines.append(
            f"| {max_length} | {float(payload['frac_pairs_over_limit']):.4f} | "
            f"{float(payload['frac_pairs_hidden_diff_after_cutoff']):.4f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
