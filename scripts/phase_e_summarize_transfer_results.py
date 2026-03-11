#!/usr/bin/env python3
"""Summarise Phase E benchmark evaluation results across multiple runs.

English
-------
Reads summary.json from one or more phase_e_eval directories and produces a
Markdown table comparing key transfer metrics.  Intended as a quick post-hoc
tool after running a sweep (e.g. the terminal-anchor ratio sweep).

Key metrics extracted per run:
- pair_accuracy_good_vs_bad
- pair_auc_good_vs_bad
- first_error_edge_accuracy
- mean_all_correct_last_score  (proxy for terminal completion quality)
- mean_good_prefix_score / mean_bad_prefix_score
- num_all_correct_examples / num_error_examples

中文
----
读取多个 phase_e_eval 目录中的 summary.json，生成对比 Markdown 表格。
专为 terminal-anchor ratio sweep 等批量实验设计的快速汇总工具。

Usage
-----
    # Pass explicit directories:
    python scripts/phase_e_summarize_transfer_results.py \\
        --eval-dirs \\
            assets/artifacts/phase_e_eval/run_a \\
            assets/artifacts/phase_e_eval/run_b \\
        --output-path results.md

    # Glob by tag substring:
    python scripts/phase_e_summarize_transfer_results.py \\
        --eval-root assets/artifacts/phase_e_eval \\
        --tag-filter ta_ratio \\
        --output-path results.md

    # Both together (dirs take priority, root used as fallback):
    python scripts/phase_e_summarize_transfer_results.py \\
        --eval-root assets/artifacts/phase_e_eval \\
        --tag-filter processbench_math \\
        --sort-by pair_auc_good_vs_bad
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


_METRICS_KEYS = [
    "pair_accuracy_good_vs_bad",
    "pair_auc_good_vs_bad",
    "first_error_edge_accuracy",
    "mean_all_correct_last_score",
    "mean_good_prefix_score",
    "mean_bad_prefix_score",
    "num_all_correct_examples",
    "num_error_examples",
    "num_good_bad_pairs",
]

_SORT_CHOICES = [
    "pair_auc_good_vs_bad",
    "pair_accuracy_good_vs_bad",
    "first_error_edge_accuracy",
    "mean_all_correct_last_score",
    "run_name",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Summarise Phase E benchmark eval results across multiple runs."
    )
    p.add_argument(
        "--eval-dirs",
        type=Path,
        nargs="*",
        default=[],
        help="Explicit list of phase_e_eval directories to include.",
    )
    p.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help=(
            "Root directory to scan for eval subdirectories. "
            "Combined with --tag-filter to select matching runs."
        ),
    )
    p.add_argument(
        "--tag-filter",
        default="",
        help="Only include eval dirs whose name contains this substring (used with --eval-root).",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Write Markdown table to this file. Default: print to stdout.",
    )
    p.add_argument(
        "--sort-by",
        choices=_SORT_CHOICES,
        default="pair_auc_good_vs_bad",
        help="Column to sort rows by (descending). Default: pair_auc_good_vs_bad.",
    )
    p.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Also write a JSON summary to this path.",
    )
    return p


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.eval_dirs and args.eval_root is None:
        raise ValueError("Provide --eval-dirs and/or --eval-root.")
    if args.eval_root is not None and not args.eval_root.exists():
        raise FileNotFoundError(f"--eval-root not found: {args.eval_root}")
    return args


def _collect_eval_dirs(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = list(args.eval_dirs)
    if args.eval_root is not None:
        for d in sorted(args.eval_root.iterdir()):
            if not d.is_dir():
                continue
            if args.tag_filter and args.tag_filter not in d.name:
                continue
            if d not in dirs:
                dirs.append(d)
    return dirs


def _load_metrics(eval_dir: Path) -> dict[str, Any] | None:
    """Try summary.json first (newer format), then metrics.json."""
    for name in ("summary.json", "metrics.json"):
        candidate = eval_dir / name
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                # summary.json may have metrics nested or flat
                if "pair_accuracy_good_vs_bad" in data:
                    return data
                if "metrics" in data and isinstance(data["metrics"], dict):
                    return data["metrics"]
                # Try to find the metrics in any top-level dict value
                for v in data.values():
                    if isinstance(v, dict) and "pair_accuracy_good_vs_bad" in v:
                        return v
                # Fallback: return the whole dict (flat summary)
                return data
            except (json.JSONDecodeError, OSError):
                pass
    return None


def _fmt(value: Any, key: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if "num_" in key:
            return str(int(value))
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _render_table(rows: list[dict[str, Any]], sort_by: str) -> str:
    if not rows:
        return "_No matching eval directories found._\n"

    # Sort
    def _sort_key(r: dict[str, Any]) -> tuple[float, str]:
        v = r.get(sort_by)
        if isinstance(v, (int, float)):
            return (-float(v), r.get("run_name", ""))
        return (0.0, str(r.get("run_name", "")))

    rows = sorted(rows, key=_sort_key)

    # Find best values for highlighting
    best: dict[str, float] = {}
    for key in _METRICS_KEYS:
        vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            best[key] = max(vals) if "score" in key or "accuracy" in key or "auc" in key else max(vals)

    # Column headers
    short_keys = {
        "pair_accuracy_good_vs_bad": "pair_acc",
        "pair_auc_good_vs_bad": "AUC",
        "first_error_edge_accuracy": "1st_edge_acc",
        "mean_all_correct_last_score": "allcorr_last",
        "mean_good_prefix_score": "good_score",
        "mean_bad_prefix_score": "bad_score",
        "num_all_correct_examples": "n_allcorr",
        "num_error_examples": "n_error",
        "num_good_bad_pairs": "n_pairs",
    }
    col_order = list(short_keys.keys())
    headers = ["run_name"] + [short_keys[k] for k in col_order]
    sep = [":---"] + ["---:" for _ in col_order]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in rows:
        cells = [r.get("run_name", "?")]
        for key in col_order:
            val = r.get(key)
            cell = _fmt(val, key)
            # Bold the best value (only for float metrics)
            if (
                isinstance(val, (int, float))
                and key in best
                and abs(float(val) - best[key]) < 1e-9
            ):
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    eval_dirs = _collect_eval_dirs(args)

    if not eval_dirs:
        print("ERROR: No eval directories found.", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for d in eval_dirs:
        metrics = _load_metrics(d)
        if metrics is None:
            missing.append(str(d))
            continue
        row: dict[str, Any] = {"run_name": d.name}
        for key in _METRICS_KEYS:
            row[key] = metrics.get(key)
        rows.append(row)

    if missing:
        print(f"WARNING: {len(missing)} dir(s) had no readable metrics:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)

    table_md = _render_table(rows, args.sort_by)

    header = (
        f"# Phase E Transfer Results Summary\n\n"
        f"Sorted by: `{args.sort_by}`  |  Runs: {len(rows)}  |  "
        f"Missing: {len(missing)}\n\n"
        f"Bold = best value in column.\n\n"
    )
    full_md = header + table_md

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(full_md, encoding="utf-8")
        print(f"Wrote Markdown table to: {args.output_path}")
    else:
        print(full_md)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps({"rows": rows, "missing": missing}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote JSON summary to: {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
