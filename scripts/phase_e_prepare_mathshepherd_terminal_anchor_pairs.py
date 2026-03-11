#!/usr/bin/env python3
"""Build a Math-Shepherd artifact augmented with terminal-completion anchors.

English
-------
Math-Shepherd's default conservative conversion only teaches:
1. the last clean prefix before the first bad step
2. should outrank the first bad prefix.

That local supervision is useful, but it does not explicitly teach:
1. on all-correct trajectories, the *full* correct completion
2. should outrank an earlier safe prefix from the same reasoning chain.

This script keeps the local pairs and adds a capped set of terminal anchors
from all-positive trajectories:
1. chosen = full correct trajectory
2. rejected = a shorter safe prefix, usually the penultimate prefix

中文
----
Math-Shepherd 默认的保守转换只教一件事：
1. first bad 之前最后一个 clean prefix
2. 应该高于 first bad prefix。

这种局部监督有用，但它不会显式教会模型：
1. 在全程都正确的轨迹上，
2. “完整正确解答”应该高于更早的安全前缀。

这个脚本会：
1. 保留原有 local pair，
2. 再从 all-positive 轨迹中补一批 terminal anchors：
   - chosen = 完整正确轨迹
   - rejected = 更短的安全前缀，默认取倒数第二个 prefix
"""

from __future__ import annotations

import argparse
import hashlib
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

from ours.phase_d.external_pairs import ExternalPairRecord, summarize_external_pairs  # noqa: E402
from ours.phase_d.external_pairs_adapters import (  # noqa: E402
    PairBuildConfig,
    _build_record,
    _extract_problem_prefix,
    _extract_step_labels_from_math_shepherd,
    _join_steps_as_prefix,
    load_math_shepherd_pairs,
)
from ours.phase_e.pairs import (  # noqa: E402
    _count_split_units,
    _deduplicate_pairs,
    _split_train_validation,
    _stable_fingerprint,
    _write_jsonl,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a Math-Shepherd pair artifact augmented with terminal-completion anchors."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl"),
    )
    parser.add_argument("--run-name", default="phase_e_mathshepherd_terminal_anchor_pairs")
    parser.add_argument("--output-root", type=Path, default=Path("assets/artifacts/phase_e_pairs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-granularity", choices=["pair_id", "source_sample"], default="source_sample")
    parser.add_argument(
        "--max-local-pairs",
        type=int,
        default=None,
        help="Optional deterministic cap on local non-anchor pairs before mixing anchors.",
    )
    parser.add_argument("--min-pair-confidence", type=float, default=0.72)
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-length-ratio", type=float, default=4.0)
    parser.add_argument("--max-token-overlap", type=float, default=0.995)
    parser.add_argument("--max-pairs-per-sample", type=int, default=2)
    parser.add_argument(
        "--step-label-pair-mode",
        choices=["first_bad_edge_strict", "first_bad_fanout", "all_good_vs_all_bad", "legacy_nearest"],
        default="first_bad_edge_strict",
    )
    parser.add_argument("--terminal-anchor-confidence", type=float, default=0.82)
    parser.add_argument(
        "--terminal-anchor-ratio",
        type=float,
        default=0.50,
        help="Cap terminal anchors to this ratio times the number of local pairs.",
    )
    parser.add_argument(
        "--terminal-anchor-prefix-mode",
        choices=["penultimate", "late75"],
        default="penultimate",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"--input-jsonl not found: {args.input_jsonl}")
    if not (0.0 < float(args.validation_ratio) < 0.5):
        raise ValueError("--validation-ratio must be in (0, 0.5)")
    if args.max_local_pairs is not None and int(args.max_local_pairs) <= 0:
        raise ValueError("--max-local-pairs must be > 0")
    if float(args.terminal_anchor_ratio) < 0.0:
        raise ValueError("--terminal-anchor-ratio must be >= 0")
    if not (0.0 <= float(args.min_pair_confidence) <= 1.0):
        raise ValueError("--min-pair-confidence must be in [0, 1]")
    if not (0.0 <= float(args.terminal_anchor_confidence) <= 1.0):
        raise ValueError("--terminal-anchor-confidence must be in [0, 1]")
    if int(args.min_chars) <= 0:
        raise ValueError("--min-chars must be > 0")
    if float(args.max_length_ratio) <= 1.0:
        raise ValueError("--max-length-ratio must be > 1")
    if not (0.0 <= float(args.max_token_overlap) <= 1.0):
        raise ValueError("--max-token-overlap must be in [0, 1]")
    if int(args.max_pairs_per_sample) <= 0:
        raise ValueError("--max-pairs-per-sample must be > 0")
    if args.max_rows is not None and int(args.max_rows) <= 0:
        raise ValueError("--max-rows must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_config = PairBuildConfig(
        min_chars=int(args.min_chars),
        max_length_ratio=float(args.max_length_ratio),
        max_token_overlap=float(args.max_token_overlap),
        max_pairs_per_sample=int(args.max_pairs_per_sample),
        step_label_pair_mode=str(args.step_label_pair_mode),
        r_prm_pair_mode="direct_pair_legacy",
    )

    local_rows = load_math_shepherd_pairs(
        path=Path(args.input_jsonl),
        config=build_config,
        max_pairs=(int(args.max_rows) if args.max_rows is not None else None),
    )
    if args.max_local_pairs is not None:
        local_rows = _select_capped_rows(
            rows=local_rows,
            max_rows=int(args.max_local_pairs),
            key_fn=lambda row: _stable_row_hash(row, prefix="local"),
        )
    raw_anchor_rows = _build_terminal_anchor_rows(
        path=Path(args.input_jsonl),
        config=build_config,
        terminal_anchor_confidence=float(args.terminal_anchor_confidence),
        prefix_mode=str(args.terminal_anchor_prefix_mode),
        max_rows=(int(args.max_rows) if args.max_rows is not None else None),
    )
    max_anchor_rows = (
        int(len(local_rows) * float(args.terminal_anchor_ratio))
        if float(args.terminal_anchor_ratio) > 0.0
        else 0
    )
    anchor_rows = _select_capped_anchor_rows(
        rows=raw_anchor_rows,
        max_rows=max_anchor_rows,
    )

    all_rows = list(local_rows) + list(anchor_rows)
    filtered_rows = [
        row for row in all_rows if float(row.pair_confidence) >= float(args.min_pair_confidence)
    ]
    dedup_rows = _deduplicate_pairs(filtered_rows)
    dedup_rows.sort(key=lambda item: item.pair_id)
    train_rows, validation_rows = _split_train_validation(
        rows=dedup_rows,
        seed=int(args.seed),
        validation_ratio=float(args.validation_ratio),
        split_granularity=str(args.split_granularity),
    )

    fingerprint = _stable_fingerprint(
        {
            "run_name": str(args.run_name),
            "input_jsonl": str(args.input_jsonl),
            "seed": int(args.seed),
            "validation_ratio": float(args.validation_ratio),
            "split_granularity": str(args.split_granularity),
            "min_pair_confidence": float(args.min_pair_confidence),
            "max_local_pairs": args.max_local_pairs,
            "terminal_anchor_confidence": float(args.terminal_anchor_confidence),
            "terminal_anchor_ratio": float(args.terminal_anchor_ratio),
            "terminal_anchor_prefix_mode": str(args.terminal_anchor_prefix_mode),
            "step_label_pair_mode": str(args.step_label_pair_mode),
            "max_rows": args.max_rows,
            "build_config": {
                "min_chars": int(build_config.min_chars),
                "max_length_ratio": float(build_config.max_length_ratio),
                "max_token_overlap": float(build_config.max_token_overlap),
                "max_pairs_per_sample": int(build_config.max_pairs_per_sample),
            },
        }
    )
    run_dir = Path(args.output_root) / f"{args.run_name}__{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_path = run_dir / "train_pairs.jsonl"
    validation_path = run_dir / "validation_pairs.jsonl"
    summary_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    manifest_path = run_dir / "manifest.json"

    _write_jsonl(train_path, [row.to_dict() for row in train_rows])
    _write_jsonl(validation_path, [row.to_dict() for row in validation_rows])

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "input_jsonl": str(args.input_jsonl),
        "num_rows_before_filter": int(len(all_rows)),
        "num_rows_after_confidence_filter": int(len(filtered_rows)),
        "num_rows_after_dedup": int(len(dedup_rows)),
        "num_train_rows": int(len(train_rows)),
        "num_validation_rows": int(len(validation_rows)),
        "num_split_units_after_dedup": int(
            _count_split_units(rows=dedup_rows, split_granularity=str(args.split_granularity))
        ),
        "num_train_split_units": int(
            _count_split_units(rows=train_rows, split_granularity=str(args.split_granularity))
        ),
        "num_validation_split_units": int(
            _count_split_units(rows=validation_rows, split_granularity=str(args.split_granularity))
        ),
        "row_counts_by_origin": {
            "local_pairs": int(len(local_rows)),
            "raw_terminal_anchor_pairs": int(len(raw_anchor_rows)),
            "selected_terminal_anchor_pairs": int(len(anchor_rows)),
        },
        "overall_summary": summarize_external_pairs(dedup_rows),
        "train_summary": summarize_external_pairs(train_rows),
        "validation_summary": summarize_external_pairs(validation_rows),
        "build_config": {
            "seed": int(args.seed),
            "validation_ratio": float(args.validation_ratio),
            "split_granularity": str(args.split_granularity),
            "min_pair_confidence": float(args.min_pair_confidence),
            "max_local_pairs": args.max_local_pairs,
            "terminal_anchor_confidence": float(args.terminal_anchor_confidence),
            "terminal_anchor_ratio": float(args.terminal_anchor_ratio),
            "terminal_anchor_prefix_mode": str(args.terminal_anchor_prefix_mode),
            "step_label_pair_mode": str(args.step_label_pair_mode),
            "max_rows": args.max_rows,
            "min_chars": int(build_config.min_chars),
            "max_length_ratio": float(build_config.max_length_ratio),
            "max_token_overlap": float(build_config.max_token_overlap),
            "max_pairs_per_sample": int(build_config.max_pairs_per_sample),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")

    manifest = {
        "artifact_stage": "phase_e_pairs_mathshepherd_terminal_anchor_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": str(args.run_name),
        "fingerprint": str(fingerprint),
        "run_dir": str(run_dir),
        "input_jsonl": str(args.input_jsonl),
        "output_files": {
            "train_pairs": str(train_path),
            "validation_pairs": str(validation_path),
            "summary": str(summary_path),
            "summary_md": str(summary_md_path),
        },
        "summary_snapshot": summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 88)
    print("Phase E: Prepare Math-Shepherd Terminal-Anchor Pair Artifact")
    print("=" * 88)
    print(f"run_dir             : {run_dir}")
    print(f"local_pairs         : {len(local_rows)}")
    print(f"raw_terminal_anchor : {len(raw_anchor_rows)}")
    print(f"selected_terminal   : {len(anchor_rows)}")
    print(f"rows_after_dedup    : {len(dedup_rows)}")
    print(f"train_pairs_path    : {train_path}")
    print(f"validation_pairs    : {validation_path}")
    print("=" * 88)
    return 0


def _build_terminal_anchor_rows(
    *,
    path: Path,
    config: PairBuildConfig,
    terminal_anchor_confidence: float,
    prefix_mode: str,
    max_rows: int | None,
) -> list[ExternalPairRecord]:
    rows: list[ExternalPairRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if text == "":
                continue
            payload = json.loads(text)
            label_text = str(payload.get("label", ""))
            if label_text == "":
                continue
            step_labels = _extract_step_labels_from_math_shepherd(label_text)
            if len(step_labels) < 2:
                continue
            if any(label != "+" for _, label in step_labels):
                if max_rows is not None and line_no >= int(max_rows):
                    break
                continue
            prefix_index = _select_terminal_anchor_prefix_index(
                num_steps=len(step_labels),
                prefix_mode=prefix_mode,
            )
            if prefix_index < 0 or prefix_index >= len(step_labels) - 1:
                if max_rows is not None and line_no >= int(max_rows):
                    break
                continue
            prompt = _extract_problem_prefix(label_text)
            if prompt.strip() == "":
                if max_rows is not None and line_no >= int(max_rows):
                    break
                continue
            chosen_text = _join_steps_as_prefix([step for step, _ in step_labels], len(step_labels) - 1)
            rejected_text = _join_steps_as_prefix([step for step, _ in step_labels], prefix_index)
            task = str(payload.get("task", "")).strip().lower()
            domain_tag = "gsm8k_math" if task == "gsm8k" else "general_math"
            record = _build_record(
                source_tag="math_shepherd",
                domain_tag=domain_tag,
                prompt_text=f"{prompt}\n\n",
                chosen_text=chosen_text,
                rejected_text=rejected_text,
                pair_confidence=float(terminal_anchor_confidence),
                metadata={
                    "source_line": int(line_no),
                    "task": payload.get("task"),
                    "anchor_prefix_index": int(prefix_index),
                    "terminal_prefix_index": int(len(step_labels) - 1),
                    "num_step_labels": int(len(step_labels)),
                    "pair_build_mode": f"math_shepherd_terminal_anchor_{prefix_mode}",
                    "pair_semantics": "terminal_completion_anchor",
                    "split_group_id": f"math_shepherd|source_line={line_no}",
                },
                config=config,
            )
            if record is not None:
                rows.append(record)
            if max_rows is not None and line_no >= int(max_rows):
                break
    return rows


def _select_terminal_anchor_prefix_index(*, num_steps: int, prefix_mode: str) -> int:
    if int(num_steps) < 2:
        return -1
    if prefix_mode == "penultimate":
        return int(num_steps) - 2
    if prefix_mode == "late75":
        late_idx = max(0, int(round(float(num_steps) * 0.75)) - 1)
        return min(int(num_steps) - 2, late_idx)
    raise ValueError(f"Unsupported prefix_mode: {prefix_mode}")


def _select_capped_anchor_rows(*, rows: list[ExternalPairRecord], max_rows: int) -> list[ExternalPairRecord]:
    return _select_capped_rows(
        rows=rows,
        max_rows=max_rows,
        key_fn=lambda row: _stable_row_hash(row, prefix="anchor"),
    )


def _select_capped_rows(
    *,
    rows: list[ExternalPairRecord],
    max_rows: int,
    key_fn,
) -> list[ExternalPairRecord]:
    if max_rows <= 0 or not rows:
        return []
    if len(rows) <= int(max_rows):
        return list(rows)
    ranked = sorted(
        rows,
        key=key_fn,
    )
    return ranked[: int(max_rows)]


def _stable_row_hash(row: ExternalPairRecord, *, prefix: str) -> str:
    payload = (
        f"{prefix}|{row.source_tag}|{row.metadata.get('source_line')}|"
        f"{row.metadata.get('anchor_prefix_index')}|{row.metadata.get('positive_step_index')}|"
        f"{row.metadata.get('negative_step_index')}|{row.metadata.get('pair_build_mode')}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase E Math-Shepherd Terminal-Anchor Pair Summary",
        "",
        f"- generated_at: {summary.get('generated_at')}",
        f"- run_dir: {summary.get('run_dir')}",
        f"- input_jsonl: {summary.get('input_jsonl')}",
        f"- rows_before_filter: {summary.get('num_rows_before_filter')}",
        f"- rows_after_conf_filter: {summary.get('num_rows_after_confidence_filter')}",
        f"- rows_after_dedup: {summary.get('num_rows_after_dedup')}",
        f"- train_rows: {summary.get('num_train_rows')}",
        f"- validation_rows: {summary.get('num_validation_rows')}",
        f"- split_granularity: {summary.get('build_config', {}).get('split_granularity')}",
        "",
        "## Origins",
        "",
        f"- local_pairs: {summary.get('row_counts_by_origin', {}).get('local_pairs')}",
        f"- raw_terminal_anchor_pairs: {summary.get('row_counts_by_origin', {}).get('raw_terminal_anchor_pairs')}",
        f"- selected_terminal_anchor_pairs: {summary.get('row_counts_by_origin', {}).get('selected_terminal_anchor_pairs')}",
        "",
        "## Train Summary",
        "",
        f"- num_pairs: {summary.get('train_summary', {}).get('num_pairs')}",
        f"- by_pair_build_mode: {summary.get('train_summary', {}).get('by_pair_build_mode')}",
        f"- by_pair_semantics: {summary.get('train_summary', {}).get('by_pair_semantics')}",
        "",
        "## Validation Summary",
        "",
        f"- num_pairs: {summary.get('validation_summary', {}).get('num_pairs')}",
        f"- by_pair_build_mode: {summary.get('validation_summary', {}).get('by_pair_build_mode')}",
        f"- by_pair_semantics: {summary.get('validation_summary', {}).get('by_pair_semantics')}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
