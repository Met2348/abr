#!/usr/bin/env python3
"""Validate and preview normalized datasets before training.

Beginner-friendly workflow
--------------------------
1. Download datasets into ``assets/datasets``.
2. Run:
   ``python scripts/check_data.py --datasets gsm8k strategyqa --split train --limit 3``
3. Confirm:
   - schema is valid
   - sample preview looks correct
   - counts and non-empty ratios look reasonable

If this script fails, fix data issues before writing model code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    """Allow running this script from repo root without installing package."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.data.loaders import DATASET_LOADERS, load_dataset_canonical  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check and preview normalized dataset samples."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "strategyqa"],
        help=(
            "Dataset names to load. "
            f"Supported: {', '.join(sorted(DATASET_LOADERS.keys()))}"
        ),
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Requested split (train/validation/test). Loaders may map unsupported splits.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Max samples to load per dataset for quick validation.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("assets/datasets"),
        help="Local dataset root directory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("assets/hf_cache/datasets"),
        help="HF datasets cache dir used for parquet/script loading.",
    )
    parser.add_argument(
        "--gsm8k-config",
        default="main",
        choices=["main", "socratic"],
        help="GSM8K config to use.",
    )
    parser.add_argument(
        "--bbh-task",
        default="boolean_expressions",
        help="BBH task folder name (used when dataset is bbh/bigbench_hard).",
    )
    parser.add_argument(
        "--hendrycks-subset",
        default="algebra",
        help="Hendrycks Math subset folder name.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[tuple[str, str]] = []

    print("=" * 80)
    print("Data Pipeline Check")
    print("=" * 80)
    print(f"dataset_root : {args.dataset_root}")
    print(f"cache_dir    : {args.cache_dir}")
    print(f"requested_split : {args.split}")
    print(f"sample_limit : {args.limit}")
    print()

    for dataset_name in args.datasets:
        print("-" * 80)
        print(f"[Dataset] {dataset_name}")
        print("-" * 80)
        kwargs = _dataset_specific_kwargs(dataset_name, args)
        try:
            samples = load_dataset_canonical(
                dataset_name=dataset_name,
                dataset_root=args.dataset_root,
                split=args.split,
                limit=args.limit,
                cache_dir=args.cache_dir,
                **kwargs,
            )
            _print_summary(samples, dataset_name)
            _print_example(samples)
            print(f"[OK] {dataset_name} loaded and validated.\n")
        except Exception as exc:  # noqa: BLE001 - explicit reporting is desired
            failures.append((dataset_name, str(exc)))
            print(f"[ERROR] {dataset_name} failed: {exc}\n")

    print("=" * 80)
    if failures:
        print("Result: FAILED")
        print("Failed datasets:")
        for name, err in failures:
            print(f"- {name}: {err}")
        print("=" * 80)
        return 1

    print("Result: SUCCESS")
    print("=" * 80)
    return 0


def _dataset_specific_kwargs(dataset_name: str, args: argparse.Namespace) -> dict[str, Any]:
    name = dataset_name.lower()
    kwargs: dict[str, Any] = {}
    if name == "gsm8k":
        kwargs["config"] = args.gsm8k_config
    elif name in {"bbh", "bigbench_hard"}:
        kwargs["task"] = args.bbh_task
    elif name == "hendrycks_math":
        kwargs["subset"] = args.hendrycks_subset
    return kwargs


def _print_summary(samples, dataset_name: str) -> None:
    n = len(samples)
    if n == 0:
        print(f"[WARN] {dataset_name}: no samples returned.")
        return

    question_non_empty = sum(1 for s in samples if s.question.strip() != "")
    answer_non_empty = sum(1 for s in samples if s.answer.strip() != "")
    cot_present = sum(1 for s in samples if s.cot is not None and s.cot.strip() != "")

    print(f"count: {n}")
    print(f"question non-empty: {question_non_empty}/{n}")
    print(f"answer non-empty  : {answer_non_empty}/{n}")
    print(f"cot present       : {cot_present}/{n}")


def _print_example(samples) -> None:
    if not samples:
        return
    example = samples[0].to_dict()
    print("example[0]:")
    print(json.dumps(example, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())

