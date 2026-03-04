#!/usr/bin/env python3
"""Prepare normalized external pair artifacts for Phase D4.

Why this file exists
--------------------
Phase D4 introduces external pair supervision (direct preference pairs and
step-label conversions). Different sources have different schemas, so we need a
single canonical artifact before C2 can consume them safely.

What this file does
-------------------
1. Load selected external datasets from local downloaded files.
2. Normalize rows into one canonical pair schema.
3. Apply deterministic quality filters and de-duplication.
4. Split into train/validation deterministically.
5. Write artifact files + manifest + summaries for reproducibility.
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
    load_math_shepherd_pairs,
    load_prmbench_preview_pairs,
    load_r_prm_dpo_pairs,
    load_rlhflow_pairs,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare canonical external-pair artifacts for Phase D4."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_d_external_pairs"),
    )
    parser.add_argument("--run-name", default="phase_d_external_pairs")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing artifact dir when outputs already exist.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate outputs even if matching artifact dir already exists.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--max-pairs-total", type=int, default=None)
    parser.add_argument("--max-pairs-per-source", type=int, default=None)
    parser.add_argument("--min-pair-confidence", type=float, default=0.0)

    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-length-ratio", type=float, default=4.0)
    parser.add_argument("--max-token-overlap", type=float, default=0.995)
    parser.add_argument("--max-pairs-per-sample", type=int, default=2)

    parser.add_argument(
        "--r-prm-root",
        type=Path,
        default=None,
        help="Path to local kevinpro_r_prm dataset root.",
    )
    parser.add_argument(
        "--r-prm-split",
        choices=["train", "validation", "both"],
        default="train",
    )
    parser.add_argument(
        "--prmbench-preview-path",
        type=Path,
        default=None,
        help="Path to prmbench_preview.jsonl.",
    )
    parser.add_argument(
        "--math-shepherd-path",
        type=Path,
        default=None,
        help="Path to math-shepherd.jsonl.",
    )
    parser.add_argument(
        "--rlhflow-mistral-root",
        type=Path,
        default=None,
        help="Path to RLHFlow mistral dataset root.",
    )
    parser.add_argument(
        "--rlhflow-deepseek-path",
        type=Path,
        default=None,
        help="Path to deepseek_instruct_data.jsonl.",
    )
    parser.add_argument(
        "--build-step-converted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable step-label conversion sources (Math-Shepherd / RLHFlow).",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not (0.0 < float(args.validation_ratio) < 0.5):
        raise ValueError("--validation-ratio must be in (0, 0.5)")
    if args.max_pairs_total is not None and int(args.max_pairs_total) <= 0:
        raise ValueError("--max-pairs-total must be > 0")
    if args.max_pairs_per_source is not None and int(args.max_pairs_per_source) <= 0:
        raise ValueError("--max-pairs-per-source must be > 0")
    if not (0.0 <= float(args.min_pair_confidence) <= 1.0):
        raise ValueError("--min-pair-confidence must be in [0, 1]")
    if int(args.min_chars) <= 0:
        raise ValueError("--min-chars must be > 0")
    if float(args.max_length_ratio) <= 1.0:
        raise ValueError("--max-length-ratio must be > 1")
    if not (0.0 <= float(args.max_token_overlap) <= 1.0):
        raise ValueError("--max-token-overlap must be in [0, 1]")
    if int(args.max_pairs_per_sample) <= 0:
        raise ValueError("--max-pairs-per-sample must be > 0")
    if (
        args.r_prm_root is None
        and args.prmbench_preview_path is None
        and args.math_shepherd_path is None
        and args.rlhflow_mistral_root is None
        and args.rlhflow_deepseek_path is None
    ):
        raise ValueError("No external source configured. Provide at least one source path.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PairBuildConfig(
        min_chars=int(args.min_chars),
        max_length_ratio=float(args.max_length_ratio),
        max_token_overlap=float(args.max_token_overlap),
        max_pairs_per_sample=int(args.max_pairs_per_sample),
    )
    config.validate()

    fingerprint = _stable_fingerprint(
        {
            "run_name": str(args.run_name),
            "seed": int(args.seed),
            "validation_ratio": float(args.validation_ratio),
            "max_pairs_total": args.max_pairs_total,
            "max_pairs_per_source": args.max_pairs_per_source,
            "min_pair_confidence": float(args.min_pair_confidence),
            "min_chars": int(args.min_chars),
            "max_length_ratio": float(args.max_length_ratio),
            "max_token_overlap": float(args.max_token_overlap),
            "max_pairs_per_sample": int(args.max_pairs_per_sample),
            "r_prm_root": str(args.r_prm_root) if args.r_prm_root is not None else None,
            "r_prm_split": str(args.r_prm_split),
            "prmbench_preview_path": (
                str(args.prmbench_preview_path) if args.prmbench_preview_path is not None else None
            ),
            "math_shepherd_path": (
                str(args.math_shepherd_path) if args.math_shepherd_path is not None else None
            ),
            "rlhflow_mistral_root": (
                str(args.rlhflow_mistral_root) if args.rlhflow_mistral_root is not None else None
            ),
            "rlhflow_deepseek_path": (
                str(args.rlhflow_deepseek_path) if args.rlhflow_deepseek_path is not None else None
            ),
            "build_step_converted": bool(args.build_step_converted),
        }
    )
    run_dir = args.output_root / f"{args.run_name}__{fingerprint}"
    train_path = run_dir / "train_pairs.jsonl"
    validation_path = run_dir / "validation_pairs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    summary_md_path = run_dir / "summary.md"
    required_outputs = [train_path, validation_path, summary_path, manifest_path, summary_md_path]

    if run_dir.exists() and not bool(args.overwrite) and bool(args.resume):
        if all(path.exists() for path in required_outputs):
            print("=" * 88)
            print("Phase D: Prepare External Pairs")
            print("=" * 88)
            print(f"run_dir          : {run_dir}")
            print("status           : resume-hit (all outputs already exist)")
            print("=" * 88)
            return 0

    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Phase D: Prepare External Pairs")
    print("=" * 88)
    print(f"run_dir          : {run_dir}")
    print(f"seed             : {args.seed}")
    print(f"validation_ratio : {args.validation_ratio}")
    print(f"max_pairs_total  : {args.max_pairs_total}")
    print(f"max_pairs_source : {args.max_pairs_per_source}")
    print(f"min_confidence   : {args.min_pair_confidence}")
    print("=" * 88)

    all_rows: list[ExternalPairRecord] = []
    source_rows_before_filter: dict[str, int] = {}

    if args.r_prm_root is not None:
        splits = ["train", "validation"] if args.r_prm_split == "both" else [str(args.r_prm_split)]
        for split in splits:
            rows = load_r_prm_dpo_pairs(
                root=args.r_prm_root,
                split=split,
                config=config,
                max_pairs=args.max_pairs_per_source,
            )
            tag = f"r_prm_{split}"
            source_rows_before_filter[tag] = len(rows)
            all_rows.extend(rows)

    if args.prmbench_preview_path is not None:
        rows = load_prmbench_preview_pairs(
            path=args.prmbench_preview_path,
            config=config,
            max_pairs=args.max_pairs_per_source,
        )
        source_rows_before_filter["prmbench_preview"] = len(rows)
        all_rows.extend(rows)

    if bool(args.build_step_converted):
        if args.math_shepherd_path is not None:
            rows = load_math_shepherd_pairs(
                path=args.math_shepherd_path,
                config=config,
                max_pairs=args.max_pairs_per_source,
            )
            source_rows_before_filter["math_shepherd"] = len(rows)
            all_rows.extend(rows)
        if args.rlhflow_mistral_root is not None or args.rlhflow_deepseek_path is not None:
            rows = load_rlhflow_pairs(
                mistral_root=args.rlhflow_mistral_root,
                deepseek_path=args.rlhflow_deepseek_path,
                config=config,
                max_pairs_per_source=args.max_pairs_per_source,
            )
            by_source = _count_by_source(rows)
            for key, value in by_source.items():
                source_rows_before_filter[key] = int(value)
            all_rows.extend(rows)

    filtered_rows = [
        row
        for row in all_rows
        if float(row.pair_confidence) >= float(args.min_pair_confidence)
    ]
    dedup_rows = _deduplicate_pairs(filtered_rows)
    dedup_rows.sort(key=lambda item: item.pair_id)
    if args.max_pairs_total is not None:
        dedup_rows = dedup_rows[: int(args.max_pairs_total)]

    train_rows, validation_rows = _split_train_validation(
        rows=dedup_rows,
        seed=int(args.seed),
        validation_ratio=float(args.validation_ratio),
    )

    _write_jsonl(train_path, [row.to_dict() for row in train_rows])
    _write_jsonl(validation_path, [row.to_dict() for row in validation_rows])

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "num_rows_before_filter": int(len(all_rows)),
        "num_rows_after_confidence_filter": int(len(filtered_rows)),
        "num_rows_after_dedup": int(len(dedup_rows)),
        "num_train_rows": int(len(train_rows)),
        "num_validation_rows": int(len(validation_rows)),
        "source_rows_before_filter": dict(sorted(source_rows_before_filter.items())),
        "overall_summary": summarize_external_pairs(dedup_rows),
        "train_summary": summarize_external_pairs(train_rows),
        "validation_summary": summarize_external_pairs(validation_rows),
        "build_config": {
            "min_chars": int(config.min_chars),
            "max_length_ratio": float(config.max_length_ratio),
            "max_token_overlap": float(config.max_token_overlap),
            "max_pairs_per_sample": int(config.max_pairs_per_sample),
            "min_pair_confidence": float(args.min_pair_confidence),
            "max_pairs_per_source": args.max_pairs_per_source,
            "max_pairs_total": args.max_pairs_total,
            "validation_ratio": float(args.validation_ratio),
            "seed": int(args.seed),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "artifact_stage": "phase_d_external_pairs_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/phase_d_prepare_external_pairs.py",
        "run_name": str(args.run_name),
        "fingerprint": str(fingerprint),
        "run_dir": str(run_dir),
        "source_inputs": {
            "r_prm_root": str(args.r_prm_root) if args.r_prm_root is not None else None,
            "r_prm_split": str(args.r_prm_split),
            "prmbench_preview_path": (
                str(args.prmbench_preview_path) if args.prmbench_preview_path is not None else None
            ),
            "math_shepherd_path": (
                str(args.math_shepherd_path) if args.math_shepherd_path is not None else None
            ),
            "rlhflow_mistral_root": (
                str(args.rlhflow_mistral_root) if args.rlhflow_mistral_root is not None else None
            ),
            "rlhflow_deepseek_path": (
                str(args.rlhflow_deepseek_path) if args.rlhflow_deepseek_path is not None else None
            ),
            "build_step_converted": bool(args.build_step_converted),
        },
        "output_files": {
            "train_pairs": str(train_path),
            "validation_pairs": str(validation_path),
            "summary": str(summary_path),
            "summary_md": str(summary_md_path),
        },
        "summary_snapshot": summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")

    print("-" * 88)
    print(f"num_total_pairs   : {summary['num_rows_after_dedup']}")
    print(f"num_train_pairs   : {summary['num_train_rows']}")
    print(f"num_val_pairs     : {summary['num_validation_rows']}")
    print(f"train_pairs_path  : {train_path}")
    print(f"val_pairs_path    : {validation_path}")
    print(f"summary_path      : {summary_path}")
    print(f"manifest_path     : {manifest_path}")
    print("=" * 88)
    return 0


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _deduplicate_pairs(rows: list[ExternalPairRecord]) -> list[ExternalPairRecord]:
    dedup: dict[str, ExternalPairRecord] = {}
    for row in rows:
        if row.pair_id not in dedup:
            dedup[row.pair_id] = row
    return list(dedup.values())


def _split_train_validation(
    *,
    rows: list[ExternalPairRecord],
    seed: int,
    validation_ratio: float,
) -> tuple[list[ExternalPairRecord], list[ExternalPairRecord]]:
    train_rows: list[ExternalPairRecord] = []
    val_rows: list[ExternalPairRecord] = []
    for row in rows:
        if _is_validation_pair(pair_id=row.pair_id, seed=seed, ratio=validation_ratio):
            val_rows.append(row)
        else:
            train_rows.append(row)
    if not train_rows and val_rows:
        train_rows.append(val_rows.pop())
    if not val_rows and train_rows:
        val_rows.append(train_rows.pop())
    return train_rows, val_rows


def _is_validation_pair(*, pair_id: str, seed: int, ratio: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).hexdigest()
    value = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return value < float(ratio)


def _count_by_source(rows: list[ExternalPairRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.source_tag] = counts.get(row.source_tag, 0) + 1
    return counts


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase D External Pair Summary",
        "",
        f"- generated_at: {summary.get('generated_at')}",
        f"- run_dir: {summary.get('run_dir')}",
        f"- rows_before_filter: {summary.get('num_rows_before_filter')}",
        f"- rows_after_conf_filter: {summary.get('num_rows_after_confidence_filter')}",
        f"- rows_after_dedup: {summary.get('num_rows_after_dedup')}",
        f"- train_rows: {summary.get('num_train_rows')}",
        f"- validation_rows: {summary.get('num_validation_rows')}",
        "",
        "## By Source (Before Filter)",
        "",
    ]
    for key, value in sorted((summary.get("source_rows_before_filter") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Train Summary",
            "",
            f"- num_pairs: {summary.get('train_summary', {}).get('num_pairs')}",
            f"- mean_pair_confidence: {summary.get('train_summary', {}).get('mean_pair_confidence')}",
            "",
            "## Validation Summary",
            "",
            f"- num_pairs: {summary.get('validation_summary', {}).get('num_pairs')}",
            f"- mean_pair_confidence: {summary.get('validation_summary', {}).get('mean_pair_confidence')}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

