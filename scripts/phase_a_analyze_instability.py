#!/usr/bin/env python3
"""Analyze inference inconsistency indicators from Phase A scored artifacts.

This script can be used in two ways:
1) Standalone artifact analysis for existing runs (manual diagnosis/reporting).
2) Programmatic utility used by benchmark suite summary logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_a.instability import (  # noqa: E402
    compute_pairwise_prediction_flip,
    index_rows_by_sample_id,
    summarize_strategyqa_instability,
)


@dataclass(slots=True)
class AnalysisInput:
    label: str
    scored_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze multi-final-answer and flip indicators from scored predictions."
    )
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        default=[],
        help=(
            "Run directories containing scored_predictions.jsonl "
            "(for example: assets/artifacts/phase_a_runs/<run_dir>)."
        ),
    )
    parser.add_argument(
        "--scored-jsonl",
        nargs="*",
        default=[],
        help="Direct scored_predictions.jsonl file paths.",
    )
    parser.add_argument(
        "--no-pairwise",
        action="store_true",
        help="Disable pairwise overlap/flip analysis across provided runs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path for machine-readable summary.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional output markdown path for report-ready table.",
    )
    return parser.parse_args()


def _load_scored_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue
        rows.append(json.loads(line))
    return rows


def _collect_inputs(args: argparse.Namespace) -> list[AnalysisInput]:
    items: list[AnalysisInput] = []
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        scored_path = run_dir / "scored_predictions.jsonl"
        if not scored_path.exists():
            raise FileNotFoundError(f"Missing scored file in run dir: {scored_path}")
        items.append(AnalysisInput(label=run_dir.name, scored_path=scored_path))

    for scored_str in args.scored_jsonl:
        scored_path = Path(scored_str)
        if not scored_path.exists():
            raise FileNotFoundError(f"Missing scored file: {scored_path}")
        label = scored_path.parent.name
        items.append(AnalysisInput(label=label, scored_path=scored_path))

    if not items:
        raise ValueError("No inputs provided. Use --run-dirs and/or --scored-jsonl.")
    return items


def _render_markdown(
    run_rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("## Inconsistency Analysis")
    lines.append("")
    lines.append(
        "| Run | n | acc | parse_err | multi_tag_rate | first_last_disagree | "
        "tag_switch_rate | mean_tag_count_tagged |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in run_rows:
        lines.append(
            f"| {row['label']} | {row['n_total']} | {row['accuracy']:.4f} | "
            f"{row['parse_error_rate']:.4f} | {row['multi_final_tag_rate']:.4f} | "
            f"{row['first_last_disagree_rate']:.4f} | {row['tag_switch_rate']:.4f} | "
            f"{row['mean_final_tag_count_tagged']:.2f} |"
        )

    if pairwise_rows:
        lines.append("")
        lines.append("### Pairwise Flip Analysis")
        lines.append("")
        lines.append(
            "| Run A | Run B | overlap | pred_flip_rate | correctness_flip_rate | "
            "yes->no | no->yes |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in pairwise_rows:
            lines.append(
                f"| {row['label_a']} | {row['label_b']} | {row['n_overlap']} | "
                f"{row['pred_flip_rate']:.4f} | {row['correctness_flip_rate']:.4f} | "
                f"{row['n_yes_to_no']} | {row['n_no_to_yes']} |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    items = _collect_inputs(args)

    run_rows: list[dict[str, Any]] = []
    index_by_label: dict[str, dict[str, dict[str, Any]]] = {}
    for item in items:
        scored_rows = _load_scored_rows(item.scored_path)
        inst = summarize_strategyqa_instability(scored_rows)
        run_rows.append(
            {
                "label": item.label,
                "scored_path": str(item.scored_path),
                **inst,
            }
        )
        index_by_label[item.label] = index_rows_by_sample_id(scored_rows)

    pairwise_rows: list[dict[str, Any]] = []
    if not args.no_pairwise and len(items) >= 2:
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = items[i]
                b = items[j]
                flip = compute_pairwise_prediction_flip(
                    rows_a_by_id=index_by_label[a.label],
                    rows_b_by_id=index_by_label[b.label],
                )
                if int(flip["n_overlap"]) == 0:
                    continue
                pairwise_rows.append(
                    {
                        "label_a": a.label,
                        "label_b": b.label,
                        **flip,
                    }
                )

    payload = {
        "runs": run_rows,
        "pairwise": pairwise_rows,
    }

    md = _render_markdown(run_rows=run_rows, pairwise_rows=pairwise_rows)
    print(md, end="")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

