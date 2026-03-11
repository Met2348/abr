#!/usr/bin/env python3
"""Build an experimental PRMBench pair artifact with terminal anchors.

English
-------
PRMBench Preview currently contributes only local error-step pairs:
1. a correct prefix up to one step,
2. versus a modified prefix that first turns wrong at that step.

That supervision is useful for local error discrimination, but it does not
explicitly teach the value head that a *complete correct solution* should beat
an earlier safe prefix. This script builds a research-only artifact that keeps
the original local pairs and adds one terminal-completion anchor per source row:
1. chosen = full original correct process,
2. rejected = shorter original safe prefix near the first modified error step.

中文
----
PRMBench Preview 当前只提供局部 error-step pair：
1. 一个到某步为止的正确 prefix，
2. 对比一个在该步第一次出错的 modified prefix。

这种监督适合学局部错误判别，但并不会显式教会 value head：
“完整正确解答”应该高于“中间安全前缀”。本脚本就是为此构造一个研究用 artifact：
1. 保留原始 local error pairs，
2. 再为每条源样本额外加入一个 terminal-completion anchor：
   - chosen = 完整 original 正确过程，
   - rejected = 靠近第一次错误位置的较短 original 安全前缀。
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

from ours.phase_d.external_pairs import ExternalPairRecord, summarize_external_pairs  # noqa: E402
from ours.phase_d.external_pairs_adapters import (  # noqa: E402
    PairBuildConfig,
    _build_record,
    _join_steps_as_prefix,
    load_prmbench_preview_pairs,
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
        description="Prepare an experimental PRMBench Preview pair artifact with terminal-completion anchors."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("assets/external_datasets/hitsmy_prmbench_preview/prmbench_preview.jsonl"),
    )
    parser.add_argument("--run-name", default="phase_e_prmbench_terminal_anchor_pairs")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_pairs"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument(
        "--split-granularity",
        choices=["pair_id", "source_sample"],
        default="source_sample",
    )
    parser.add_argument("--min-pair-confidence", type=float, default=0.80)
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-length-ratio", type=float, default=4.0)
    parser.add_argument("--max-token-overlap", type=float, default=0.995)
    parser.add_argument("--max-pairs-per-sample", type=int, default=4)
    parser.add_argument(
        "--terminal-anchor-confidence",
        type=float,
        default=0.84,
        help="Confidence assigned to synthetic terminal anchors.",
    )
    parser.add_argument(
        "--terminal-anchor-ratio",
        type=float,
        default=1.0,
        help=(
            "Cap selected terminal anchors to this ratio times the number of local pairs. "
            "This is useful for light-touch RL repair experiments where completion support "
            "should help without overwhelming local error supervision."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on raw PRMBench rows for quick smoke experiments.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"--input-jsonl not found: {args.input_jsonl}")
    if not (0.0 < float(args.validation_ratio) < 0.5):
        raise ValueError("--validation-ratio must be in (0, 0.5)")
    if str(args.split_granularity) not in {"pair_id", "source_sample"}:
        raise ValueError("--split-granularity must be one of {'pair_id', 'source_sample'}")
    if not (0.0 <= float(args.min_pair_confidence) <= 1.0):
        raise ValueError("--min-pair-confidence must be in [0, 1]")
    if not (0.0 <= float(args.terminal_anchor_confidence) <= 1.0):
        raise ValueError("--terminal-anchor-confidence must be in [0, 1]")
    if float(args.terminal_anchor_ratio) < 0.0:
        raise ValueError("--terminal-anchor-ratio must be >= 0")
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
        step_label_pair_mode="first_bad_edge_strict",
        r_prm_pair_mode="compact_verdict",
    )

    local_rows = load_prmbench_preview_pairs(
        path=Path(args.input_jsonl),
        config=build_config,
        max_pairs=None,
    )
    raw_anchor_rows = _build_terminal_anchor_rows(
        path=Path(args.input_jsonl),
        config=build_config,
        terminal_anchor_confidence=float(args.terminal_anchor_confidence),
        max_rows=(int(args.max_rows) if args.max_rows is not None else None),
    )
    max_anchor_rows = (
        int(len(local_rows) * float(args.terminal_anchor_ratio))
        if float(args.terminal_anchor_ratio) > 0.0
        else 0
    )
    anchor_rows = _select_capped_rows(
        rows=raw_anchor_rows,
        max_rows=max_anchor_rows,
        key_fn=lambda row: _stable_row_hash(row, prefix="anchor"),
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
            "terminal_anchor_confidence": float(args.terminal_anchor_confidence),
            "terminal_anchor_ratio": float(args.terminal_anchor_ratio),
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
            "terminal_anchor_confidence": float(args.terminal_anchor_confidence),
            "terminal_anchor_ratio": float(args.terminal_anchor_ratio),
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
        "artifact_stage": "phase_e_pairs_prmbench_terminal_anchor_v1",
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
    print("Phase E: Prepare PRMBench Terminal-Anchor Pair Artifact")
    print("=" * 88)
    print(f"run_dir             : {run_dir}")
    print(f"local_pairs         : {len(local_rows)}")
    print(f"raw_terminal_anchor : {len(raw_anchor_rows)}")
    print(f"terminal_anchor     : {len(anchor_rows)}")
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
    max_rows: int | None,
) -> list[ExternalPairRecord]:
    rows: list[ExternalPairRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if text == "":
                continue
            payload = json.loads(text)
            question = str(
                payload.get("question")
                or payload.get("modified_question")
                or payload.get("original_question")
                or ""
            ).strip()
            original_process = payload.get("original_process")
            error_steps = payload.get("error_steps")
            if not isinstance(original_process, list) or len(original_process) < 2:
                continue
            if not isinstance(error_steps, list) or not error_steps:
                continue
            error_indices = _normalize_error_steps(error_steps=error_steps, num_steps=len(original_process))
            if not error_indices:
                continue
            prefix_index = error_indices[0]
            if prefix_index >= len(original_process) - 1:
                continue
            chosen_text = _join_steps_as_prefix(original_process, len(original_process) - 1)
            rejected_text = _join_steps_as_prefix(original_process, prefix_index)
            record = _build_record(
                source_tag="prmbench_preview",
                domain_tag="general_math",
                prompt_text=f"{question}\n\n",
                chosen_text=chosen_text,
                rejected_text=rejected_text,
                pair_confidence=float(terminal_anchor_confidence),
                metadata={
                    "source_row_line": int(line_no),
                    "source_idx": payload.get("idx"),
                    "classification": payload.get("classification"),
                    "anchor_prefix_index": int(prefix_index),
                    "terminal_prefix_index": int(len(original_process) - 1),
                    "num_original_steps": int(len(original_process)),
                    "num_error_steps": int(len(error_indices)),
                    "pair_build_mode": "prmbench_terminal_anchor_full_vs_prefix",
                    "pair_semantics": "terminal_completion_anchor",
                    "split_group_id": f"prmbench_preview|source_row_line={line_no}",
                },
                config=config,
            )
            if record is not None:
                rows.append(record)
            if max_rows is not None and line_no >= int(max_rows):
                break
    return rows


def _normalize_error_steps(*, error_steps: list[Any], num_steps: int) -> list[int]:
    indices: list[int] = []
    for raw in error_steps:
        try:
            idx = int(raw)
        except Exception:  # noqa: BLE001
            continue
        idx = idx - 1 if idx > 0 else idx
        if 0 <= idx < int(num_steps):
            indices.append(int(idx))
    return sorted(set(indices))


def _select_capped_rows(
    *,
    rows: list[ExternalPairRecord],
    max_rows: int,
    key_fn,
) -> list[ExternalPairRecord]:
    """Select a deterministic capped subset.

    中文
    ----
    这里不用随机采样，而是按稳定键排序后截头。
    这样即使我们只想保留很轻的 terminal-anchor 比例，也能保证：
    1. 不同机器上结果一致，
    2. warm-start 对照实验可重复，
    3. ratio 变化只改变“保留多少”，不改变采样噪声。
    """
    if max_rows <= 0 or not rows:
        return []
    if len(rows) <= int(max_rows):
        return list(rows)
    ranked = sorted(rows, key=key_fn)
    return ranked[: int(max_rows)]


def _stable_row_hash(row: ExternalPairRecord, *, prefix: str) -> str:
    return f"{prefix}:{row.pair_id}"


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase E PRMBench Terminal-Anchor Pair Summary",
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
        f"- terminal_anchor_ratio: {summary.get('build_config', {}).get('terminal_anchor_ratio')}",
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
