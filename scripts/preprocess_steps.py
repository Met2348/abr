#!/usr/bin/env python3
"""Preprocess canonical datasets into step-level training artifacts.

What this script does
---------------------
1. Loads canonical samples from dataset loaders.
2. Converts each sample into a deterministic step sequence.
3. Writes reusable artifacts to disk (JSONL + summary + manifest).

Why this script matters
-----------------------
Training/eval loops should consume ready-to-use data artifacts.
Doing preprocessing once and reusing artifacts makes experiments faster,
more reproducible, and easier to debug.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


def _bootstrap_src_path() -> None:
    """Allow running script from repo root without package installation."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.data.loaders import DATASET_LOADERS, load_dataset_canonical  # noqa: E402
from ours.data.step_builder import (  # noqa: E402
    StepBuildConfig,
    build_step_sequence,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for step-artifact preprocessing.

    Example
    -------
    ```python
    args = parse_args()
    ```
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build step-level artifacts from canonical dataset samples. "
            "Outputs are reusable across training/evaluation runs."
        )
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "strategyqa"],
        help=(
            "Dataset names to preprocess. "
            f"Supported: {', '.join(sorted(DATASET_LOADERS.keys()))}"
        ),
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Requested split name (loaders may map unsupported splits).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample cap for smoke runs (default: no cap).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("assets/datasets"),
        help="Root directory containing local datasets.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("assets/hf_cache/datasets"),
        help="HF datasets cache directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/artifacts/steps"),
        help="Root directory where preprocessing artifacts will be written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Number of samples processed per batch.",
    )

    # Step-building config.
    parser.add_argument(
        "--split-mode",
        choices=["auto", "newline", "sentence"],
        default="auto",
        help="How to split CoT into reasoning steps.",
    )
    parser.add_argument(
        "--include-question-step0",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include question as step index 0 (default: true).",
    )
    parser.add_argument(
        "--include-answer-terminal-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include final answer as terminal step (default: true).",
    )
    parser.add_argument(
        "--normalize-whitespace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize whitespace in step fragments (default: true).",
    )
    parser.add_argument(
        "--strip-list-markers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove list markers like '-', '1.' after splitting (default: true).",
    )
    parser.add_argument(
        "--min-fragment-chars",
        type=int,
        default=1,
        help="Drop reasoning fragments shorter than this length.",
    )

    # Dataset-specific options.
    parser.add_argument(
        "--gsm8k-config",
        default="main",
        choices=["main", "socratic"],
        help="GSM8K config.",
    )
    parser.add_argument(
        "--bbh-task",
        default="boolean_expressions",
        help="BBH task name.",
    )
    parser.add_argument(
        "--hendrycks-subset",
        default="algebra",
        help="Hendrycks MATH subset.",
    )

    # Pipeline controls.
    parser.add_argument(
        "--write-flat-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write an extra flat step-level JSONL file (default: true).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip dataset run when matching artifacts already exist (default: true).",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate artifacts even if same run directory already exists.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail immediately on the first sample-level preprocessing error.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the step-preprocessing workflow for all requested datasets.

    Returns
    -------
    int
        `0` on success, otherwise `1` when one or more datasets fail.

    Example
    -------
    ```bash
    python scripts/preprocess_steps.py --datasets gsm8k strategyqa --split train --limit 200
    ```
    """
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("`--batch-size` must be >= 1")

    step_config = StepBuildConfig(
        split_mode=args.split_mode,
        include_question_as_step0=args.include_question_step0,
        include_final_answer_as_terminal_step=args.include_answer_terminal_step,
        normalize_whitespace=args.normalize_whitespace,
        strip_list_markers=args.strip_list_markers,
        min_fragment_chars=args.min_fragment_chars,
    )
    step_config.validate()

    print("=" * 88)
    print("Step Preprocessing Run")
    print("=" * 88)
    print(f"datasets      : {args.datasets}")
    print(f"dataset_root  : {args.dataset_root}")
    print(f"cache_dir     : {args.cache_dir}")
    print(f"output_dir    : {args.output_dir}")
    print(f"split         : {args.split}")
    print(f"limit         : {args.limit}")
    print(f"batch_size    : {args.batch_size}")
    print(f"step_config   : {step_config.to_dict()} (signature={step_config.stable_signature()})")
    print()

    total_failures: list[tuple[str, str]] = []

    for dataset_name in args.datasets:
        print("-" * 88)
        print(f"[Dataset] {dataset_name}")
        print("-" * 88)

        dataset_kwargs = _dataset_specific_kwargs(dataset_name, args)
        run_spec = {
            "dataset": dataset_name,
            "split": args.split,
            "limit": args.limit,
            "dataset_kwargs": dataset_kwargs,
            "step_config": step_config.to_dict(),
        }
        run_fingerprint = _stable_fingerprint(run_spec)
        variant = _dataset_variant_name(dataset_name, dataset_kwargs)

        run_dir = (
            args.output_dir
            / dataset_name.lower()
            / f"{_safe_name(args.split)}__{_safe_name(variant)}__{run_fingerprint}"
        )

        try:
            summary = _preprocess_one_dataset(
                dataset_name=dataset_name,
                split=args.split,
                limit=args.limit,
                dataset_root=args.dataset_root,
                cache_dir=args.cache_dir,
                dataset_kwargs=dataset_kwargs,
                step_config=step_config,
                batch_size=args.batch_size,
                run_dir=run_dir,
                run_fingerprint=run_fingerprint,
                write_flat_steps=args.write_flat_steps,
                resume=args.resume,
                overwrite=args.overwrite,
                strict=args.strict,
            )
            if summary is None:
                print("[SKIP] Matching artifacts already exist; reuse previous run.")
            else:
                print(f"[OK] Wrote artifacts to: {run_dir}")
                print(f"      samples_ok={summary['samples_ok']}, total_steps={summary['total_steps']}")
        except Exception as exc:  # noqa: BLE001 - keep failure context explicit
            total_failures.append((dataset_name, str(exc)))
            print(f"[ERROR] {dataset_name}: {exc}")

        print()

    print("=" * 88)
    if total_failures:
        print("Result: FAILED")
        for name, err in total_failures:
            print(f"- {name}: {err}")
        print("=" * 88)
        return 1

    print("Result: SUCCESS")
    print("=" * 88)
    return 0


def _preprocess_one_dataset(
    dataset_name: str,
    split: str,
    limit: int | None,
    dataset_root: Path,
    cache_dir: Path,
    dataset_kwargs: dict[str, Any],
    step_config: StepBuildConfig,
    batch_size: int,
    run_dir: Path,
    run_fingerprint: str,
    write_flat_steps: bool,
    resume: bool,
    overwrite: bool,
    strict: bool,
) -> dict[str, Any] | None:
    """Preprocess one dataset and write artifacts.

    Returns
    -------
    dict | None
        Summary dict if work was done, or `None` if skipped due to resume.
    """
    samples_jsonl = run_dir / "sample_sequences.jsonl"
    steps_jsonl = run_dir / "flat_steps.jsonl"
    errors_jsonl = run_dir / "errors.jsonl"
    summary_json = run_dir / "summary.json"
    manifest_json = run_dir / "manifest.json"

    if run_dir.exists() and resume and not overwrite:
        if samples_jsonl.exists() and summary_json.exists() and manifest_json.exists():
            old_manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
            if old_manifest.get("run_fingerprint") == run_fingerprint:
                return None

    if run_dir.exists() and overwrite:
        for path in [
            samples_jsonl,
            steps_jsonl,
            errors_jsonl,
            summary_json,
            manifest_json,
        ]:
            if path.exists():
                path.unlink()

    run_dir.mkdir(parents=True, exist_ok=True)

    print("Loading canonical samples...")
    samples = load_dataset_canonical(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        split=split,
        limit=limit,
        cache_dir=cache_dir,
        **dataset_kwargs,
    )
    print(f"Loaded {len(samples)} canonical samples.")

    started_at = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()

    role_counts = {"question": 0, "reasoning": 0, "answer": 0}
    total_steps = 0
    total_reasoning_steps = 0
    samples_ok = 0
    samples_error = 0
    samples_with_cot = 0

    with (
        samples_jsonl.open("w", encoding="utf-8") as sample_writer,
        (steps_jsonl.open("w", encoding="utf-8") if write_flat_steps else _null_writer()) as step_writer,
        errors_jsonl.open("w", encoding="utf-8") as error_writer,
    ):
        for start in range(0, len(samples), batch_size):
            end = min(start + batch_size, len(samples))
            batch = samples[start:end]

            for sample in batch:
                try:
                    sequence = build_step_sequence(sample=sample, config=step_config)
                    _write_jsonl(sample_writer, sequence.to_dict())

                    if write_flat_steps:
                        for step in sequence.steps:
                            _write_jsonl(step_writer, step.to_dict())

                    for step in sequence.steps:
                        role_counts[step.role] += 1

                    total_steps += sequence.num_steps
                    total_reasoning_steps += sum(
                        1 for step in sequence.steps if step.role == "reasoning"
                    )
                    samples_with_cot += 1 if sequence.has_cot else 0
                    samples_ok += 1
                except Exception as exc:  # noqa: BLE001 - persist error details for debug
                    samples_error += 1
                    error_record = {
                        "dataset": dataset_name,
                        "sample_id": getattr(sample, "id", "<unknown>"),
                        "error": str(exc),
                    }
                    _write_jsonl(error_writer, error_record)
                    if strict:
                        raise RuntimeError(
                            f"Strict mode failure on sample_id={error_record['sample_id']}: {exc}"
                        ) from exc

            print(
                f"Processed batch {start}:{end} "
                f"(ok={samples_ok}, errors={samples_error}, total_steps={total_steps})"
            )

    duration_sec = time.time() - started_at
    ended_iso = datetime.now(timezone.utc).isoformat()

    summary = {
        "dataset": dataset_name,
        "split": split,
        "limit": limit,
        "dataset_kwargs": dataset_kwargs,
        "step_config": step_config.to_dict(),
        "run_fingerprint": run_fingerprint,
        "samples_total": len(samples),
        "samples_ok": samples_ok,
        "samples_error": samples_error,
        "samples_with_cot": samples_with_cot,
        "total_steps": total_steps,
        "total_reasoning_steps": total_reasoning_steps,
        "role_counts": role_counts,
        "avg_steps_per_ok_sample": (total_steps / samples_ok) if samples_ok else 0.0,
        "avg_reasoning_steps_per_ok_sample": (
            total_reasoning_steps / samples_ok
        )
        if samples_ok
        else 0.0,
        "started_at_utc": started_iso,
        "ended_at_utc": ended_iso,
        "duration_seconds": round(duration_sec, 3),
        "artifacts": {
            "sample_sequences_jsonl": str(samples_jsonl),
            "flat_steps_jsonl": str(steps_jsonl) if write_flat_steps else None,
            "errors_jsonl": str(errors_jsonl),
            "summary_json": str(summary_json),
            "manifest_json": str(manifest_json),
        },
    }

    manifest = {
        "schema_version": 1,
        "script": "scripts/preprocess_steps.py",
        "run_fingerprint": run_fingerprint,
        "dataset": dataset_name,
        "split": split,
        "limit": limit,
        "dataset_kwargs": dataset_kwargs,
        "step_config": step_config.to_dict(),
        "step_config_signature": step_config.stable_signature(),
        "created_at_utc": ended_iso,
    }

    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _dataset_specific_kwargs(dataset_name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Build dataset-specific kwargs in one central place."""
    name = dataset_name.lower()
    kwargs: dict[str, Any] = {}
    if name == "gsm8k":
        kwargs["config"] = args.gsm8k_config
    elif name in {"bbh", "bigbench_hard"}:
        kwargs["task"] = args.bbh_task
    elif name == "hendrycks_math":
        kwargs["subset"] = args.hendrycks_subset
    return kwargs


def _dataset_variant_name(dataset_name: str, kwargs: dict[str, Any]) -> str:
    """Return a concise variant name used in output directory naming."""
    name = dataset_name.lower()
    if name == "gsm8k":
        return f"config-{kwargs.get('config', 'main')}"
    if name in {"bbh", "bigbench_hard"}:
        return f"task-{kwargs.get('task', 'unknown')}"
    if name == "hendrycks_math":
        return f"subset-{kwargs.get('subset', 'unknown')}"
    return "default"


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    """Return a short deterministic fingerprint for one run specification.

    Example
    -------
    ```python
    fingerprint = _stable_fingerprint({"dataset": "gsm8k", "split": "train"})
    ```
    """
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _safe_name(value: str) -> str:
    """Create filesystem-safe compact token for path components."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _write_jsonl(writer: TextIO, record: dict[str, Any]) -> None:
    """Write one JSON object as one line (JSONL format)."""
    writer.write(json.dumps(record, ensure_ascii=False) + "\n")


class _NullWriter:
    """Small no-op writer used when optional outputs are disabled."""

    def write(self, _: str) -> int:
        """Pretend to write one string and report zero bytes written."""
        return 0

    def __enter__(self) -> "_NullWriter":
        """Support `with`-statement usage for optional file handles."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Exit the context manager without performing cleanup."""
        return None


def _null_writer() -> _NullWriter:
    """Create a `_NullWriter` for branches that skip optional artifact output.

    Example
    -------
    ```python
    writer = _null_writer()
    ```
    """
    return _NullWriter()


if __name__ == "__main__":
    raise SystemExit(main())
