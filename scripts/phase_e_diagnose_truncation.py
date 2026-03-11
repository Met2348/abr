#!/usr/bin/env python3
"""诊断一份 Phase E pair artifact 在不同 `max_length` 下的截断风险。 Diagnose how one Phase E pair artifact behaves under different `max_length` cutoffs.

这个脚本存在的目的，是把“训练学不会”与“监督信号在进模型前就被截断破坏”这两类问题拆开。
This script exists to separate "the model cannot learn" from "the supervision signal was already damaged by truncation before entering the model".

控制流很简单：
1. 读取一份 canonical pair JSONL；
2. 用真实 tokenizer 对 chosen / rejected 全文本做无截断分词；
3. 对多个 `max_length` 计算截断风险；
4. 把结构化 JSON 和可读 Markdown 一起落盘。
The control flow is simple:
1. load one canonical pair JSONL;
2. tokenize full chosen / rejected texts with the real tokenizer and no truncation;
3. compute truncation-risk summaries for multiple `max_length` values;
4. write both structured JSON and readable Markdown outputs.
"""

from __future__ import annotations

import argparse
import json
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
from ours.phase_e.runtime import resolve_tokenizer_load_path  # noqa: E402
from ours.phase_e.training import compute_pair_truncation_diagnostics  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose truncation risk for one Phase E pair artifact under multiple max-length settings."
    )
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--run-name", default="phase_e_truncation_diag")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_truncation_diagnostics"),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-lengths", type=int, nargs="+", required=True)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """校验 CLI 参数是否合法。 Validate CLI arguments before any tokenizer or dataset work starts."""
    args = _build_parser().parse_args(argv)
    if not Path(args.pairs_jsonl).exists():
        raise FileNotFoundError(f"--pairs-jsonl not found: {args.pairs_jsonl}")
    if args.adapter_path is not None and not Path(args.adapter_path).exists():
        raise FileNotFoundError(f"--adapter-path not found: {args.adapter_path}")
    if args.max_samples is not None and int(args.max_samples) <= 0:
        raise ValueError("--max-samples must be > 0")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")
    if not args.max_lengths:
        raise ValueError("--max-lengths must contain at least one value")
    if any(int(value) <= 8 for value in args.max_lengths):
        raise ValueError("--max-lengths values must all be > 8")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pairs, pair_stats = load_external_pair_jsonl(
        Path(args.pairs_jsonl),
        max_samples=args.max_samples,
    )
    if not pairs:
        raise RuntimeError("No pairs were loaded for truncation diagnosis")

    from transformers import AutoTokenizer

    tokenizer_path = resolve_tokenizer_load_path(
        str(args.model_path),
        Path(args.adapter_path) if args.adapter_path is not None else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    normalized_lengths = sorted({int(value) for value in args.max_lengths})

    diagnostics_by_length: dict[str, Any] = {}
    for max_length in normalized_lengths:
        diagnostics_by_length[str(max_length)] = compute_pair_truncation_diagnostics(
            pairs=pairs,
            tokenizer=tokenizer,
            max_length=int(max_length),
            batch_size=int(args.batch_size),
        )

    run_dir = Path(args.output_root) / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "pairs_jsonl": str(args.pairs_jsonl),
        "model_path": str(args.model_path),
        "adapter_path": (str(args.adapter_path) if args.adapter_path is not None else None),
        "tokenizer_path": str(tokenizer_path),
        "num_pairs": int(len(pairs)),
        "pair_stats": pair_stats,
        "batch_size": int(args.batch_size),
        "max_lengths": [int(value) for value in normalized_lengths],
        "diagnostics_by_max_length": diagnostics_by_length,
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")

    print("=" * 88)
    print("Phase E: Truncation Diagnostic")
    print("=" * 88)
    print(f"pairs_jsonl        : {args.pairs_jsonl}")
    print(f"tokenizer_path     : {tokenizer_path}")
    print(f"num_pairs          : {len(pairs)}")
    print(f"max_lengths        : {normalized_lengths}")
    for max_length in normalized_lengths:
        overall = diagnostics_by_length[str(max_length)]["overall"]
        print(
            "diag               : "
            f"max_length={max_length} "
            f"over_limit={float(overall['frac_pairs_over_limit']):.4f} "
            f"collapse_after_cut={float(overall['frac_pairs_identical_after_truncation']):.4f} "
            f"hidden_diff_after_cut={float(overall['frac_pairs_first_diff_after_cutoff']):.4f}"
        )
    print(f"summary_json       : {summary_path}")
    print(f"summary_md         : {summary_md_path}")
    print("=" * 88)
    return 0


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    """把多组 `max_length` 的诊断结果压成可读 Markdown。 Render readable Markdown for multiple max-length truncation diagnostics."""
    lines = [
        "# Phase E Truncation Diagnostic Summary",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- run_dir: {summary['run_dir']}",
        f"- pairs_jsonl: `{summary['pairs_jsonl']}`",
        f"- tokenizer_path: `{summary['tokenizer_path']}`",
        f"- num_pairs: `{summary['num_pairs']}`",
        "",
        "## Max-Length Comparison",
        "",
        "| max_length | frac_over_limit | frac_collapse_after_cut | frac_hidden_diff_after_cut | chosen_p95 | first_diff_p95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    diagnostics_by_length = dict(summary["diagnostics_by_max_length"])
    for max_length in summary["max_lengths"]:
        overall = diagnostics_by_length[str(max_length)]["overall"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(max_length),
                    f"{float(overall['frac_pairs_over_limit']):.4f}",
                    f"{float(overall['frac_pairs_identical_after_truncation']):.4f}",
                    f"{float(overall['frac_pairs_first_diff_after_cutoff']):.4f}",
                    f"{int(overall['chosen_length']['p95'])}",
                    f"{int(overall['first_diff_token_index']['p95'])}",
                ]
            )
            + " |"
        )
    lines.append("")

    for max_length in summary["max_lengths"]:
        diagnostics = diagnostics_by_length[str(max_length)]
        lines.extend(
            [
                f"## By Source @ {max_length}",
                "",
                "| source_tag | frac_over_limit | frac_collapse_after_cut | frac_hidden_diff_after_cut | chosen_p95 | first_diff_p95 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for source_tag, payload in diagnostics.get("by_source", {}).items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(source_tag),
                        f"{float(payload['frac_pairs_over_limit']):.4f}",
                        f"{float(payload['frac_pairs_identical_after_truncation']):.4f}",
                        f"{float(payload['frac_pairs_first_diff_after_cutoff']):.4f}",
                        f"{int(payload['chosen_length']['p95'])}",
                        f"{int(payload['first_diff_token_index']['p95'])}",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
