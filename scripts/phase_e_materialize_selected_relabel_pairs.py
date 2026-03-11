#!/usr/bin/env python3
"""Merge untouched pairs with judge-kept selected-relabel pairs.

English
-------
This is the second half of a selected-relabel experiment:
1. untouched pairs stay as-is,
2. the judged slice contributes only label-preserving kept rows,
3. the merged artifact becomes the new training set.

中文
----
这个脚本负责 selected-relabel 实验的第二步：
1. 未被选中的训练 pair 原样保留，
2. 被 judge 审过的子集只引入 label-preserving keep 的样本，
3. 合并后的 artifact 作为新的训练集。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize one selected-relabel Phase E pair artifact.")
    parser.add_argument("--retained-train-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--judge-kept-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--validation-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--selected-train-pairs-jsonl", type=Path, default=None)
    parser.add_argument("--run-name", default="phase_e_selected_relabel_materialized")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_pairs"),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    for name in [
        "retained_train_pairs_jsonl",
        "judge_kept_pairs_jsonl",
        "validation_pairs_jsonl",
    ]:
        path = Path(getattr(args, name))
        if not path.exists():
            raise FileNotFoundError(f"--{name.replace('_', '-')} not found: {path}")
    if args.selected_train_pairs_jsonl is not None and not Path(args.selected_train_pairs_jsonl).exists():
        raise FileNotFoundError(f"--selected-train-pairs-jsonl not found: {args.selected_train_pairs_jsonl}")
    return args


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _annotate_rows(rows: list[dict[str, Any]], *, judge_selected: bool, judge_kept: bool) -> list[dict[str, Any]]:
    payloads = []
    for row in rows:
        payload = json.loads(json.dumps(row, ensure_ascii=False))
        metadata = dict(payload.get("metadata") or {})
        metadata["selected_relabel_judge_selected"] = bool(judge_selected)
        metadata["selected_relabel_judge_kept"] = bool(judge_kept)
        payload["metadata"] = metadata
        payloads.append(payload)
    return payloads


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    retained_rows = _annotate_rows(_load_rows(Path(args.retained_train_pairs_jsonl)), judge_selected=False, judge_kept=False)
    judge_rows = _annotate_rows(_load_rows(Path(args.judge_kept_pairs_jsonl)), judge_selected=True, judge_kept=True)
    selected_rows = _load_rows(Path(args.selected_train_pairs_jsonl)) if args.selected_train_pairs_jsonl is not None else None

    merged_rows = [*retained_rows, *judge_rows]
    merged_pair_ids = [str(row.get("pair_id")) for row in merged_rows]
    if len(set(merged_pair_ids)) != len(merged_pair_ids):
        raise ValueError("Merged selected-relabel artifact contains duplicate pair_id values")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root) / f"{args.run_name}__{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    train_path = run_dir / "train_pairs.jsonl"
    validation_path = run_dir / "validation_pairs.jsonl"

    with train_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    validation_path.write_text(Path(args.validation_pairs_jsonl).read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "num_retained_rows": int(len(retained_rows)),
        "num_judge_kept_rows": int(len(judge_rows)),
        "num_merged_rows": int(len(merged_rows)),
        "num_selected_rows": (None if selected_rows is None else int(len(selected_rows))),
        "selected_keep_rate": (
            None
            if selected_rows is None or len(selected_rows) == 0
            else float(len(judge_rows) / len(selected_rows))
        ),
        "output_files": {
            "train_pairs": str(train_path),
            "validation_pairs": str(validation_path),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_stage": "phase_e_selected_relabel_materialized_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "retained_train_pairs_jsonl": str(args.retained_train_pairs_jsonl),
                "judge_kept_pairs_jsonl": str(args.judge_kept_pairs_jsonl),
                "validation_pairs_jsonl": str(args.validation_pairs_jsonl),
                "selected_train_pairs_jsonl": (
                    None if args.selected_train_pairs_jsonl is None else str(args.selected_train_pairs_jsonl)
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_md = [
        "# Phase E Selected Relabel Materialization",
        "",
        f"- num_retained_rows: `{len(retained_rows)}`",
        f"- num_judge_kept_rows: `{len(judge_rows)}`",
        f"- num_merged_rows: `{len(merged_rows)}`",
    ]
    if summary["selected_keep_rate"] is not None:
        summary_md.append(f"- selected_keep_rate: `{float(summary['selected_keep_rate']):.4f}`")
    summary_md.extend(
        [
            "",
            "## Outputs",
            "",
            f"- train_pairs: `{train_path}`",
            f"- validation_pairs: `{validation_path}`",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    print("=" * 88)
    print("Phase E: Materialize Selected Relabel Pairs")
    print("=" * 88)
    print(f"num_retained_rows     : {len(retained_rows)}")
    print(f"num_judge_kept_rows   : {len(judge_rows)}")
    print(f"num_merged_rows       : {len(merged_rows)}")
    if summary["selected_keep_rate"] is not None:
        print(f"selected_keep_rate    : {float(summary['selected_keep_rate']):.4f}")
    print(f"train_pairs_path      : {train_path}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
