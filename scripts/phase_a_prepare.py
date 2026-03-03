#!/usr/bin/env python3
"""Prepare Phase A prompt/target artifacts from canonical dataset samples.

Why this file exists
--------------------
Raw datasets are normalized into canonical samples first, but models still need
prompt text and supervised target text. This script builds those model-ready Phase A
artifacts in a reproducible format.

What this file does
-------------------
1. Load canonical samples from supported datasets.
2. Assign each sample to train/validation/test using either official or hash-based
   splitting.
3. Convert each sample into a `PreparedSample` with prompt/target text.
4. Write `train.jsonl`, `validation.jsonl`, `test.jsonl`, plus summary/manifest
   files under a deterministic run directory.

Interaction with other files
----------------------------
- `src/ours/data/loaders.py`: provides canonical samples.
- `src/ours/phase_a/prompt_builder.py`: converts canonical samples into prepared rows.
- `scripts/phase_a_generate_and_eval.py`: consumes the JSONL outputs of this script.

Example
-------
```bash
python scripts/phase_a_prepare.py \
  --datasets strategyqa \
  --source-split train \
  --split-policy hash \
  --target-style answer_only \
  --template-id qa_direct
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


def _bootstrap_src_path() -> None:
    """Add the repo-local `src/` directory to `sys.path`.

    Example
    -------
    ```bash
    python scripts/phase_a_prepare.py --help
    ```
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.data.loaders import DATASET_LOADERS, load_dataset_canonical  # noqa: E402
from ours.phase_a import (  # noqa: E402
    PROMPT_TEMPLATE_REGISTRY,
    SplitConfig,
    assign_split,
    build_prepared_sample,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for artifact preparation.

    Example
    -------
    ```python
    args = parse_args(["--datasets", "strategyqa"])
    ```
    """
    parser = argparse.ArgumentParser(
        description="Prepare Phase A prompt/target artifacts from canonical datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["strategyqa"],
        help=f"Supported datasets: {', '.join(sorted(DATASET_LOADERS.keys()))}",
    )
    parser.add_argument(
        "--source-split",
        default="train",
        help="Split used to load raw canonical samples.",
    )
    parser.add_argument(
        "--split-policy",
        choices=["official", "hash"],
        default="hash",
        help=(
            "official: keep all records in source split; "
            "hash: deterministic local train/validation/test split by sample id."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for quick experiments.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("assets/datasets"),
        help="Dataset root directory.",
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
        default=Path("assets/artifacts/phase_a_prepared"),
        help="Output root for prepared artifacts.",
    )
    parser.add_argument(
        "--template-id",
        default="qa_direct",
        choices=sorted(PROMPT_TEMPLATE_REGISTRY.keys()),
        help="Prompt template id.",
    )
    parser.add_argument(
        "--template-version",
        default="1.0.0",
        help="Prompt template version.",
    )
    parser.add_argument(
        "--target-style",
        default="answer_only",
        choices=["answer_only", "cot_then_answer"],
        help="Supervised target style.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used only for deterministic hash splitting.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing matching artifact run if present.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force regenerate files in run directory.",
    )

    # Dataset-specific options.
    parser.add_argument("--gsm8k-config", default="main", choices=["main", "socratic"])
    parser.add_argument("--bbh-task", default="boolean_expressions")
    parser.add_argument("--hendrycks-subset", default="algebra")
    return parser.parse_args()


def main() -> int:
    """Run the Phase A preparation workflow for all requested datasets.

    Returns
    -------
    int
        `0` on success, `1` if one or more datasets fail.
    """
    args = parse_args()

    # 先固定切分配置，确保本次运行里所有数据集都使用同一套比例和 seed。
    # 这样跨数据集结果更可比，也便于复现实验。
    split_cfg = SplitConfig(
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    split_cfg.validate()

    print("=" * 88)
    print("Phase A: Prepare Artifacts")
    print("=" * 88)
    print(f"datasets       : {args.datasets}")
    print(f"source_split   : {args.source_split}")
    print(f"split_policy   : {args.split_policy}")
    print(f"target_style   : {args.target_style}")
    print(f"template       : {args.template_id}@{args.template_version}")
    print(f"limit          : {args.limit}")
    print()

    failures: list[tuple[str, str]] = []

    for dataset in args.datasets:
        # Each dataset gets its own deterministic run directory derived from the full spec.
        print("-" * 88)
        print(f"[Dataset] {dataset}")
        print("-" * 88)

        dataset_kwargs = _dataset_specific_kwargs(dataset, args)
        run_spec = {
            "dataset": dataset,
            "source_split": args.source_split,
            "split_policy": args.split_policy,
            "limit": args.limit,
            "target_style": args.target_style,
            "template_id": args.template_id,
            "template_version": args.template_version,
            "split_config": {
                "train": args.train_ratio,
                "validation": args.validation_ratio,
                "test": args.test_ratio,
                "seed": args.seed,
            },
            "dataset_kwargs": dataset_kwargs,
        }
        # run_spec -> run_fingerprint 是 Phase A prepared 目录可复现的核心。
        # 只要 run_spec 不变，指纹和输出目录就稳定不变。
        run_fingerprint = _stable_fingerprint(run_spec)
        run_dir = args.output_dir / dataset.lower() / run_fingerprint

        try:
            # Preparation is isolated per dataset so one failure does not hide all
            # successful outputs from other requested datasets.
            summary = _prepare_one_dataset(
                dataset=dataset,
                source_split=args.source_split,
                split_policy=args.split_policy,
                limit=args.limit,
                dataset_root=args.dataset_root,
                cache_dir=args.cache_dir,
                dataset_kwargs=dataset_kwargs,
                target_style=args.target_style,
                template_id=args.template_id,
                template_version=args.template_version,
                split_cfg=split_cfg,
                run_dir=run_dir,
                run_spec=run_spec,
                run_fingerprint=run_fingerprint,
                resume=args.resume,
                overwrite=args.overwrite,
            )
            if summary is None:
                print("[SKIP] Matching artifacts already exist.")
            else:
                print(f"[OK] {dataset}: wrote {summary['n_total']} records to {run_dir}")
                print(f"     split_counts={summary['split_counts']}")
        except Exception as exc:  # noqa: BLE001
            failures.append((dataset, str(exc)))
            print(f"[ERROR] {dataset}: {exc}")

        print()

    print("=" * 88)
    if failures:
        # Keep failure reporting explicit because preparation problems usually
        # indicate bad loader settings or invalid cached artifacts.
        print("Result: FAILED")
        for name, err in failures:
            print(f"- {name}: {err}")
        print("=" * 88)
        return 1

    print("Result: SUCCESS")
    print("=" * 88)
    return 0


def _prepare_one_dataset(
    dataset: str,
    source_split: str,
    split_policy: str,
    limit: int | None,
    dataset_root: Path,
    cache_dir: Path,
    dataset_kwargs: dict[str, Any],
    target_style: str,
    template_id: str,
    template_version: str,
    split_cfg: SplitConfig,
    run_dir: Path,
    run_spec: dict[str, Any],
    run_fingerprint: str,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any] | None:
    """Prepare one dataset and write its split JSONL artifacts.

    Returns
    -------
    dict[str, Any] | None
        Summary dict if work was performed, or `None` when resume mode reused a
        matching existing artifact directory.

    Example
    -------
    ```python
    summary = _prepare_one_dataset(
        dataset="strategyqa",
        source_split="train",
        split_policy="hash",
        limit=200,
        dataset_root=Path("assets/datasets"),
        cache_dir=Path("assets/hf_cache/datasets"),
        dataset_kwargs={},
        target_style="answer_only",
        template_id="qa_direct",
        template_version="1.0.0",
        split_cfg=split_cfg,
        run_dir=run_dir,
        run_spec=run_spec,
        run_fingerprint="abc123",
        resume=True,
        overwrite=False,
    )
    ```
    """
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"

    # resume 语义：已有同指纹且 JSONL 可解析就直接复用，避免重复准备。
    if run_dir.exists() and resume and not overwrite:
        if manifest_path.exists() and summary_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("run_fingerprint") == run_fingerprint:
                valid, reason = _validate_jsonl_outputs(run_dir)
                if valid:
                    return None
                print(
                    "WARN: Existing prepared artifacts failed JSONL validation; regenerating.",
                    f"dataset={dataset} run_dir={run_dir} reason={reason}",
                )
                for p in run_dir.glob("*.json"):
                    p.unlink()
                for p in run_dir.glob("*.jsonl"):
                    p.unlink()

    # overwrite 语义：仅清理本目录下的 json/jsonl，再按当前参数重建。
    if run_dir.exists() and overwrite:
        for p in run_dir.glob("*.json"):
            p.unlink()
        for p in run_dir.glob("*.jsonl"):
            p.unlink()

    run_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset_canonical(
        dataset_name=dataset,
        dataset_root=dataset_root,
        split=source_split,
        limit=limit,
        cache_dir=cache_dir,
        **dataset_kwargs,
    )

    writers = {
        "train": (run_dir / "train.jsonl").open("w", encoding="utf-8"),
        "validation": (run_dir / "validation.jsonl").open("w", encoding="utf-8"),
        "test": (run_dir / "test.jsonl").open("w", encoding="utf-8"),
    }

    split_counts = {"train": 0, "validation": 0, "test": 0}
    try:
        for sample in samples:
            # official: 全部样本保留在 source_split 对应桶；
            # hash: 由 sample_id + seed 稳定映射到 train/validation/test。
            if split_policy == "official":
                target_split = source_split
            else:
                target_split = assign_split(sample.id, split_cfg)

            # For official mode we still keep output files constrained to
            # known split names; unsupported names map to `validation`.
            if target_split not in writers:
                target_split = "validation"

            prepared = build_prepared_sample(
                sample=sample,
                split=target_split,
                target_style=target_style,
                template_id=template_id,
                template_version=template_version,
                extra_metadata={"source_split": source_split},
            )
            _write_jsonl(writers[target_split], prepared.to_dict())
            split_counts[target_split] += 1
    finally:
        for writer in writers.values():
            writer.close()

    created_at = datetime.now(timezone.utc).isoformat()
    n_total = sum(split_counts.values())

    summary = {
        "dataset": dataset,
        "source_split": source_split,
        "split_policy": split_policy,
        "n_total": n_total,
        "split_counts": split_counts,
        "target_style": target_style,
        "template_id": template_id,
        "template_version": template_version,
        "run_fingerprint": run_fingerprint,
        "created_at_utc": created_at,
        "files": {
            "train": str(run_dir / "train.jsonl"),
            "validation": str(run_dir / "validation.jsonl"),
            "test": str(run_dir / "test.jsonl"),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
    }
    manifest = {
        "schema_version": 1,
        "script": "scripts/phase_a_prepare.py",
        "run_fingerprint": run_fingerprint,
        "run_spec": run_spec,
        "created_at_utc": created_at,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _dataset_specific_kwargs(dataset_name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Build dataset-specific loader kwargs in one central place.

    Example
    -------
    ```python
    kwargs = _dataset_specific_kwargs("gsm8k", args)
    ```
    """
    name = dataset_name.lower()
    kwargs: dict[str, Any] = {}
    # 数据集私有参数统一在这里集中管理，避免散落在主流程里难追踪。
    if name == "gsm8k":
        kwargs["config"] = args.gsm8k_config
    elif name in {"bbh", "bigbench_hard"}:
        kwargs["task"] = args.bbh_task
    elif name == "hendrycks_math":
        kwargs["subset"] = args.hendrycks_subset
    return kwargs


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    """Return a short deterministic fingerprint for one preparation spec."""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _write_jsonl(writer: TextIO, record: dict[str, Any]) -> None:
    """Write one JSON object as one JSONL line with safe line-separator escaping.

    Example
    -------
    ```python
    _write_jsonl(writer, {"sample_id": "x"})
    ```
    """
    # Keep JSONL robust for line-based tooling by escaping Unicode line separators.
    text = json.dumps(record, ensure_ascii=False)
    text = text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    writer.write(text + "\n")


def _validate_jsonl_outputs(run_dir: Path) -> tuple[bool, str]:
    """Check whether prepared split JSONLs are parseable row-by-row.

    Example
    -------
    ```python
    valid, reason = _validate_jsonl_outputs(run_dir)
    ```
    """
    for split_name in ("train", "validation", "test"):
        path = run_dir / f"{split_name}.jsonl"
        if not path.exists():
            return False, f"missing_file:{path}"
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                if line.strip() == "":
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    return False, f"invalid_json:{path}:line={idx}:{exc}"
    return True, "ok"


if __name__ == "__main__":
    raise SystemExit(main())
