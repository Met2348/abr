#!/usr/bin/env python3
"""Delete incomplete experiment artifact directories under ``assets/artifacts``.

Why this file exists
--------------------
The repo has accumulated many experiment outputs across Phase A/B/C/D. Some
directories were created by runs that crashed before writing their manifest or
summary files. Those half-written directories are dangerous because later suite
scripts may auto-resolve "latest artifact" by name and accidentally pick an
incomplete directory.

This utility gives one consistent definition of "incomplete" per artifact
family and can either:

1. dry-run and report what would be deleted, or
2. actually delete the incomplete directories.

Design principles
-----------------
- Conservative on scope: only scan known experiment roots.
- Strict on completeness: if a directory is missing core files that should
  always exist for that artifact family, treat it as incomplete.
- Shallow on traversal: scan only run directories, not nested checkpoint/model
  subdirectories.

Examples
--------
Dry-run:

    python scripts/cleanup_incomplete_artifacts.py

Delete the detected incomplete artifact directories:

    python scripts/cleanup_incomplete_artifacts.py --delete
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScanSpec:
    """Describe one artifact family and the files that define completeness."""

    name: str
    root: Path
    required_files: tuple[str, ...]
    levels_below_root: int = 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete incomplete directories. Default is dry-run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also emit a machine-readable JSON summary to stdout at the end.",
    )
    return parser.parse_args()


def main() -> int:
    """Scan known artifact roots, print findings, and optionally delete them."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    specs = build_scan_specs(repo_root)

    incomplete_rows: list[dict[str, object]] = []
    print("=" * 88)
    print("Cleanup Incomplete Artifacts")
    print("=" * 88)
    for spec in specs:
        rows = scan_spec(spec)
        incomplete_rows.extend(rows)
        print(
            f"[{spec.name}] root={spec.root} | scanned={count_candidate_dirs(spec)} "
            f"| incomplete={len(rows)}"
        )
        for row in rows[:12]:
            missing = ",".join(row["missing_files"])  # type: ignore[index]
            print(f"  - {row['path']} | missing=[{missing}]")
        if len(rows) > 12:
            print(f"  ... {len(rows) - 12} more")

    print("-" * 88)
    print(f"total_incomplete_dirs : {len(incomplete_rows)}")

    delete_mode = bool(args.delete)
    emit_json = bool(args.json)

    if delete_mode:
        deleted = 0
        for row in incomplete_rows:
            path = Path(str(row.get("abs_path", row["path"])))
            if path.exists():
                shutil.rmtree(path)
                deleted += 1
        print(f"deleted_dirs          : {deleted}")
    else:
        print("delete_mode           : False (dry-run)")

    if emit_json:
        payload = {
            "total_incomplete_dirs": int(len(incomplete_rows)),
            "delete_mode": bool(delete_mode),
            "rows": incomplete_rows,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def build_scan_specs(repo_root: Path) -> list[ScanSpec]:
    """Return the artifact families that should be scanned."""
    # 不同阶段产物的“完整性”定义并不一样。
    # 例如 Phase C data 需要 rollout_targets，而 Phase C eval 只需要 metrics。
    # 因此不要用“一套通用必需文件”去硬套所有目录。
    return [
        ScanSpec(
            name="phase_a_runs",
            root=repo_root / "assets/artifacts/phase_a_runs",
            required_files=(
                "manifest.json",
                "metrics.json",
                "predictions.jsonl",
                "scored_predictions.jsonl",
            ),
            levels_below_root=1,
        ),
        ScanSpec(
            name="phase_b_runs",
            root=repo_root / "assets/artifacts/phase_b_runs",
            required_files=(
                "manifest.json",
                "summary.json",
                "train_metrics.json",
                "eval_metrics.json",
            ),
            levels_below_root=1,
        ),
        ScanSpec(
            name="phase_b_logs",
            root=repo_root / "assets/artifacts/phase_b_logs",
            required_files=("final_summary.md", "suite.log"),
            levels_below_root=1,
        ),
        ScanSpec(
            name="phase_c_data",
            root=repo_root / "assets/artifacts/phase_c_data",
            required_files=(
                "manifest.json",
                "summary.json",
                "prefixes.jsonl",
                "rollout_targets.jsonl",
            ),
            levels_below_root=2,
        ),
        ScanSpec(
            name="phase_c_runs",
            root=repo_root / "assets/artifacts/phase_c_runs",
            required_files=(
                "manifest.json",
                "summary.json",
                "train_metrics.json",
                "eval_metrics.json",
                "best_value_head.pt",
            ),
            levels_below_root=1,
        ),
        ScanSpec(
            name="phase_c_eval",
            root=repo_root / "assets/artifacts/phase_c_eval",
            required_files=("manifest.json", "metrics.json", "summary.md"),
            levels_below_root=1,
        ),
        ScanSpec(
            name="phase_c_logs",
            root=repo_root / "assets/artifacts/phase_c_logs",
            required_files=("final_summary.md", "suite.log"),
            levels_below_root=1,
        ),
        ScanSpec(
            name="phase_c_pik_data",
            root=repo_root / "assets/artifacts/phase_c_pik_data",
            required_files=(
                "manifest.json",
                "summary.json",
                "questions.jsonl",
                "pik_targets.jsonl",
            ),
            levels_below_root=2,
        ),
        ScanSpec(
            name="phase_c_pik_runs",
            root=repo_root / "assets/artifacts/phase_c_pik_runs",
            required_files=(
                "manifest.json",
                "summary.json",
                "train_metrics.json",
                "eval_metrics.json",
                "best_value_head.pt",
            ),
            levels_below_root=1,
        ),
        ScanSpec(
            name="phase_c_pik_eval",
            root=repo_root / "assets/artifacts/phase_c_pik_eval",
            required_files=("manifest.json", "metrics.json", "summary.md"),
            levels_below_root=1,
        ),
        ScanSpec(
            name="phase_c_pik_logs",
            root=repo_root / "assets/artifacts/phase_c_pik_logs",
            required_files=("final_summary.md", "suite.log"),
            levels_below_root=1,
        ),
        ScanSpec(
            name="phase_d_external_pairs",
            root=repo_root / "assets/artifacts/phase_d_external_pairs",
            required_files=(
                "manifest.json",
                "summary.json",
                "train_pairs.jsonl",
                "validation_pairs.jsonl",
            ),
            levels_below_root=1,
        ),
        ScanSpec(
            name="phase_d_gate_reports",
            root=repo_root / "assets/artifacts/phase_d_gate_reports",
            required_files=("gate_report.json", "summary.md"),
            levels_below_root=1,
        ),
        ScanSpec(
            name="phase_d6t_logs",
            root=repo_root / "assets/artifacts/phase_d6t_logs",
            required_files=("final_summary.md", "seed_results.jsonl", "suite.log"),
            levels_below_root=1,
        ),
    ]


def count_candidate_dirs(spec: ScanSpec) -> int:
    """Count the number of run directories that would be scanned for one spec."""
    return sum(1 for _ in iter_candidate_dirs(spec))


def iter_candidate_dirs(spec: ScanSpec):
    """Yield candidate run directories for one scan spec."""
    if not spec.root.exists():
        return
    if spec.levels_below_root == 1:
        for child in sorted(spec.root.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                yield child
        return
    if spec.levels_below_root == 2:
        for dataset_dir in sorted(spec.root.iterdir()):
            if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
                continue
            for run_dir in sorted(dataset_dir.iterdir()):
                if run_dir.is_dir() and not run_dir.name.startswith("."):
                    yield run_dir
        return
    raise ValueError(f"Unsupported levels_below_root: {spec.levels_below_root}")


def scan_spec(spec: ScanSpec) -> list[dict[str, object]]:
    """Return incomplete directories for one artifact family."""
    rows: list[dict[str, object]] = []
    for run_dir in iter_candidate_dirs(spec):
        missing = [name for name in spec.required_files if not (run_dir / name).exists()]
        if not missing:
            continue
        rows.append(
            {
                "spec_name": spec.name,
                # `path` 用 repo-relative，便于日志阅读；
                # `abs_path` 保留绝对路径，删除时不依赖当前工作目录，避免误删失败。
                "path": str(run_dir.relative_to(spec.root.parents[2])),
                "abs_path": str(run_dir),
                "missing_files": missing,
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
