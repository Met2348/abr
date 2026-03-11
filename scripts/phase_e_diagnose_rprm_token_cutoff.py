#!/usr/bin/env python3
"""Diagnose whether R-PRM pair artifacts have their decision point hidden by token truncation.

English
-------
Unlike `phase_e_audit_rprm_contract.py` (which audits the raw parquet source),
this script operates on an already-filtered Phase E pair artifact JSONL.  It
answers the question:

    "For the pairs that actually enter training, what fraction have
     first_diff_token_index >= max_length?"

If that fraction is high (e.g. > 20 %), the backbone never sees any
difference between chosen and rejected — the pair is invisible to the
value-head gradient and contributes only noise.  This is a plausible
explanation for the observed ~28 % train-fit → held-out generalization gap.

中文
----
这个脚本不同于 phase_e_audit_rprm_contract.py（后者审计原始 parquet）,
而是作用于已过滤的 Phase E pair artifact JSONL。它回答的核心问题是:

    "进入训练的 pairs 里,有多少比例的 first_diff_token_index >= max_length?"

如果这个比例很高 (e.g. > 20 %), 说明 backbone 对 chosen/rejected 看到的
是完全一样的 token 序列, value head 梯度为零, 这些 pair 只产生噪声。
这是 R-PRM 28% 泛化差距的候选根因。

Usage
-----
    python scripts/phase_e_diagnose_rprm_token_cutoff.py \\
        --pairs-jsonl assets/artifacts/phase_e_pairs/XXX/train_pairs.jsonl \\
        --tokenizer-path assets/models/Qwen2.5-7B-Instruct \\
        --max-length 1024 \\
        --source-filter r_prm

    # Also run on validation split to see if cutoff risk differs:
    python scripts/phase_e_diagnose_rprm_token_cutoff.py \\
        --pairs-jsonl assets/artifacts/phase_e_pairs/XXX/validation_pairs.jsonl \\
        --tokenizer-path assets/models/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Diagnose token-cutoff risk in a Phase E pair artifact JSONL."
    )
    p.add_argument(
        "--pairs-jsonl",
        type=Path,
        required=True,
        help="Path to train_pairs.jsonl or validation_pairs.jsonl from a Phase E artifact.",
    )
    p.add_argument(
        "--tokenizer-path",
        required=True,
        help="Path to backbone tokenizer (e.g. assets/models/Qwen2.5-7B-Instruct).",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Truncation limit used during training (default: 1024).",
    )
    p.add_argument(
        "--extra-max-lengths",
        type=int,
        nargs="*",
        default=[512, 768, 1280, 1536],
        help="Additional max_length thresholds to report alongside --max-length.",
    )
    p.add_argument(
        "--source-filter",
        default="r_prm",
        help=(
            "Only analyse pairs whose source_tag contains this substring. "
            "Pass '' to analyse all pairs. Default: 'r_prm'."
        ),
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Cap the number of pairs analysed (useful for large artifacts).",
    )
    p.add_argument(
        "--run-name",
        default="rprm_cutoff_diag",
        help="Sub-directory name under --output-root.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_diagnostics"),
    )
    p.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of hidden-diff example pairs to print / save.",
    )
    return p


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.pairs_jsonl.exists():
        raise FileNotFoundError(f"--pairs-jsonl not found: {args.pairs_jsonl}")
    return args


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(round(q * (len(sorted_vals) - 1)))
    return float(sorted_vals[idx])


def _stats(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": 0.0, "p5": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0, "max": 0}
    sv = sorted(float(v) for v in values)
    return {
        "count": len(sv),
        "mean": float(mean(sv)),
        "p5": int(_percentile(sv, 0.05)),
        "p25": int(_percentile(sv, 0.25)),
        "p50": int(_percentile(sv, 0.50)),
        "p75": int(_percentile(sv, 0.75)),
        "p95": int(_percentile(sv, 0.95)),
        "max": int(sv[-1]),
    }


# ---------------------------------------------------------------------------
# Core tokenization helper
# ---------------------------------------------------------------------------

def _find_first_diff_token_index(
    tokenizer: Any,
    prompt_text: str,
    chosen_text: str,
    rejected_text: str,
) -> dict[str, int]:
    """Tokenize both inputs and return token lengths + first divergence position."""
    chosen_ids: list[int] = tokenizer.encode(
        prompt_text + chosen_text, add_special_tokens=False
    )
    rejected_ids: list[int] = tokenizer.encode(
        prompt_text + rejected_text, add_special_tokens=False
    )
    first_diff = min(len(chosen_ids), len(rejected_ids))
    for idx, (a, b) in enumerate(zip(chosen_ids, rejected_ids)):
        if int(a) != int(b):
            first_diff = idx
            break
    return {
        "chosen_length": len(chosen_ids),
        "rejected_length": len(rejected_ids),
        "first_diff_token_index": first_diff,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase E R-PRM Token Cutoff Diagnostic",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- pairs_jsonl: `{summary['pairs_jsonl']}`",
        f"- source_filter: `{summary['source_filter']}`",
        f"- total_pairs_in_file: `{summary['total_pairs_in_file']}`",
        f"- pairs_matching_filter: `{summary['pairs_matching_filter']}`",
        f"- pairs_analysed: `{summary['pairs_analysed']}`",
        "",
        "## Cutoff Risk",
        "",
        "| max_length | frac_over_limit | frac_hidden_diff | hidden_diff_count |",
        "|---:|---:|---:|---:|",
    ]
    for ml, row in sorted(summary["cutoff_risk_by_max_length"].items(), key=lambda x: int(x[0])):
        lines.append(
            f"| {ml} "
            f"| {float(row['frac_over_limit']):.4f} "
            f"| {float(row['frac_hidden_diff']):.4f} "
            f"| {int(row['hidden_diff_count'])} |"
        )
    lines.extend([
        "",
        "## Length Distributions",
        "",
        f"- chosen_length: `{summary['chosen_length_stats']}`",
        f"- rejected_length: `{summary['rejected_length_stats']}`",
        f"- first_diff_token_index: `{summary['first_diff_stats']}`",
        "",
        "## Interpretation",
        "",
    ])
    primary = int(summary.get("primary_max_length", 1024))
    risk_row = summary["cutoff_risk_by_max_length"].get(str(primary), {})
    frac_hidden = float(risk_row.get("frac_hidden_diff", 0.0))
    if frac_hidden >= 0.20:
        lines.append(
            f"⚠️  **HIGH RISK**: {frac_hidden:.1%} of pairs have their first_diff_token_index "
            f">= {primary}. The backbone sees identical token sequences for these pairs — "
            f"they contribute zero discriminative gradient. This is a strong candidate for "
            f"the observed train-fit → held-out generalization gap."
        )
    elif frac_hidden >= 0.05:
        lines.append(
            f"⚡ **MODERATE RISK**: {frac_hidden:.1%} of pairs have hidden diff at max_length={primary}. "
            f"Unlikely to be the primary cause, but worth filtering out."
        )
    else:
        lines.append(
            f"✅ **LOW RISK**: Only {frac_hidden:.1%} of pairs have hidden diff at max_length={primary}. "
            f"Token cutoff is NOT the primary explanation for the generalization gap."
        )
    if summary.get("hidden_diff_examples"):
        lines.extend(["", "## Hidden-Diff Examples (first_diff >= max_length)", ""])
        for i, ex in enumerate(summary["hidden_diff_examples"], 1):
            lines.extend([
                f"### Example {i}",
                f"- pair_id: `{ex['pair_id']}`",
                f"- source_tag: `{ex['source_tag']}`",
                f"- chosen_length: {ex['chosen_length']}, rejected_length: {ex['rejected_length']}, "
                  f"first_diff: {ex['first_diff_token_index']}",
                f"- chosen_input tail (chars -200): `{ex['chosen_tail']}`",
                f"- rejected_input tail (chars -200): `{ex['rejected_tail']}`",
                "",
            ])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print(f"Loading tokenizer from: {args.tokenizer_path}")
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path), trust_remote_code=True)

    all_max_lengths = sorted(set([args.max_length] + list(args.extra_max_lengths)))

    # Counters
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    first_diff_indices: list[int] = []
    over_limit_counts: dict[int, int] = {ml: 0 for ml in all_max_lengths}
    hidden_diff_counts: dict[int, int] = {ml: 0 for ml in all_max_lengths}
    hidden_diff_examples: list[dict[str, Any]] = []

    total_pairs_in_file = 0
    pairs_matching = 0
    pairs_analysed = 0

    print(f"Reading pairs from: {args.pairs_jsonl}")
    with args.pairs_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total_pairs_in_file += 1
            record = json.loads(line)

            # Source filter
            source_tag = str(record.get("source_tag", ""))
            if args.source_filter and args.source_filter not in source_tag:
                continue
            pairs_matching += 1

            if args.max_pairs is not None and pairs_analysed >= args.max_pairs:
                continue

            prompt_text = str(record.get("prompt_text", ""))
            chosen_text = str(record.get("chosen_text", ""))
            rejected_text = str(record.get("rejected_text", ""))

            tok = _find_first_diff_token_index(tokenizer, prompt_text, chosen_text, rejected_text)
            cl = tok["chosen_length"]
            rl = tok["rejected_length"]
            fd = tok["first_diff_token_index"]

            chosen_lengths.append(cl)
            rejected_lengths.append(rl)
            first_diff_indices.append(fd)
            pairs_analysed += 1

            for ml in all_max_lengths:
                if cl > ml or rl > ml:
                    over_limit_counts[ml] += 1
                if fd >= ml:
                    hidden_diff_counts[ml] += 1
                    if ml == args.max_length and len(hidden_diff_examples) < args.num_examples:
                        chosen_input = prompt_text + chosen_text
                        rejected_input = prompt_text + rejected_text
                        hidden_diff_examples.append({
                            "pair_id": str(record.get("pair_id", "")),
                            "source_tag": source_tag,
                            "chosen_length": cl,
                            "rejected_length": rl,
                            "first_diff_token_index": fd,
                            "chosen_tail": chosen_input[-200:],
                            "rejected_tail": rejected_input[-200:],
                        })

            if pairs_analysed % 500 == 0:
                print(f"  ... analysed {pairs_analysed} pairs", flush=True)

    if pairs_analysed == 0:
        print(f"ERROR: No pairs matched source_filter={args.source_filter!r}. Exiting.")
        return 1

    cutoff_risk: dict[str, dict[str, Any]] = {}
    for ml in all_max_lengths:
        cutoff_risk[str(ml)] = {
            "frac_over_limit": float(over_limit_counts[ml] / pairs_analysed),
            "frac_hidden_diff": float(hidden_diff_counts[ml] / pairs_analysed),
            "over_limit_count": int(over_limit_counts[ml]),
            "hidden_diff_count": int(hidden_diff_counts[ml]),
        }

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pairs_jsonl": str(args.pairs_jsonl),
        "tokenizer_path": str(args.tokenizer_path),
        "source_filter": args.source_filter,
        "primary_max_length": args.max_length,
        "total_pairs_in_file": total_pairs_in_file,
        "pairs_matching_filter": pairs_matching,
        "pairs_analysed": pairs_analysed,
        "chosen_length_stats": _stats(chosen_lengths),
        "rejected_length_stats": _stats(rejected_lengths),
        "first_diff_stats": _stats(first_diff_indices),
        "cutoff_risk_by_max_length": cutoff_risk,
        "hidden_diff_examples": hidden_diff_examples,
    }

    # Output
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root) / f"{args.run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(_render_markdown(summary), encoding="utf-8")

    # Console report
    primary_risk = cutoff_risk[str(args.max_length)]
    print()
    print("=" * 80)
    print("Phase E: R-PRM Token Cutoff Diagnostic")
    print("=" * 80)
    print(f"pairs_jsonl          : {args.pairs_jsonl}")
    print(f"source_filter        : {args.source_filter!r}")
    print(f"total_in_file        : {total_pairs_in_file}")
    print(f"matching_filter      : {pairs_matching}")
    print(f"analysed             : {pairs_analysed}")
    print()
    print(f"chosen_len  p50/p95/max : {_stats(chosen_lengths)['p50']} / {_stats(chosen_lengths)['p95']} / {_stats(chosen_lengths)['max']}")
    print(f"rejected_len p50/p95/max: {_stats(rejected_lengths)['p50']} / {_stats(rejected_lengths)['p95']} / {_stats(rejected_lengths)['max']}")
    print(f"first_diff   p50/p95/max: {_stats(first_diff_indices)['p50']} / {_stats(first_diff_indices)['p95']} / {_stats(first_diff_indices)['max']}")
    print()
    print("Cutoff risk:")
    for ml in all_max_lengths:
        row = cutoff_risk[str(ml)]
        marker = " <-- PRIMARY" if ml == args.max_length else ""
        print(
            f"  max_length={ml:5d}  over_limit={row['frac_over_limit']:.4f}  "
            f"hidden_diff={row['frac_hidden_diff']:.4f}{marker}"
        )
    print()
    frac_hd = float(primary_risk["frac_hidden_diff"])
    if frac_hd >= 0.20:
        verdict = f"HIGH RISK ({frac_hd:.1%}) — token cutoff is a strong candidate for R-PRM generalization gap"
    elif frac_hd >= 0.05:
        verdict = f"MODERATE RISK ({frac_hd:.1%}) — may contribute marginally"
    else:
        verdict = f"LOW RISK ({frac_hd:.1%}) — token cutoff is NOT the primary cause"
    print(f"Verdict: {verdict}")
    print()
    print(f"summary_json  : {summary_path}")
    print(f"summary_md    : {summary_md_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
