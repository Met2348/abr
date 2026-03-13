#!/usr/bin/env python3
"""Build a semantics-aware consensus-filtered Phase E pair artifact.

English
-------
Naive confidence filtering can accidentally delete the very supervision geometry
that makes a verifier useful on ProcessBench. This script keeps consensus-style
filtering, but applies it per semantic bucket so local first-bad supervision is
not silently removed.

中文
----
朴素的 confidence 过滤很容易把 ProcessBench 最需要的监督几何一起删掉。
这个脚本保留“共识过滤”的思路，但按语义 bucket 分开筛选，避免
`local_first_bad_edge` 这类关键监督被静默清空。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROFILE_SPECS: dict[str, dict[str, float]] = {
    # Keep local-first-bad coverage dominant, keep sibling pairs moderately, and
    # downweight terminal anchors so they remain auxiliary.
    # 让 local-first-bad 继续占主导，保留中等比例 sibling pair，同时压低
    # terminal anchor 占比，让它继续扮演辅助监督而不是主导监督。
    "semantic_consensus_v1": {
        "local_first_bad_edge": 0.70,
        "sibling_branch": 0.65,
        "terminal_completion_anchor": 0.35,
    }
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curate a semantics-aware consensus-filtered Phase E pair artifact.")
    parser.add_argument("--train-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--validation-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--run-name", default="phase_e_semantic_consensus_pairs")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_pairs"),
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_SPECS),
        default="semantic_consensus_v1",
    )
    parser.add_argument(
        "--max-terminal-fraction",
        type=float,
        default=0.12,
        help="Final cap on terminal-anchor fraction after bucket-level filtering.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.train_pairs_jsonl.exists():
        raise FileNotFoundError(f"--train-pairs-jsonl not found: {args.train_pairs_jsonl}")
    if not args.validation_pairs_jsonl.exists():
        raise FileNotFoundError(f"--validation-pairs-jsonl not found: {args.validation_pairs_jsonl}")
    if not (0.0 < float(args.max_terminal_fraction) < 1.0):
        raise ValueError("--max-terminal-fraction must be in (0, 1)")
    return args


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            rows.append(json.loads(raw))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _semantic_of(row: dict[str, Any]) -> str:
    return str(dict(row.get("metadata") or {}).get("pair_semantics", "unknown"))


def _stable_rank_key(row: dict[str, Any]) -> tuple[float, str]:
    return (-float(row.get("pair_confidence", 0.0)), str(row.get("pair_id", "")))


def _apply_profile(train_rows: list[dict[str, Any]], profile: str, max_terminal_fraction: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keep_ratios = PROFILE_SPECS[profile]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        buckets[_semantic_of(row)].append(row)

    selected: list[dict[str, Any]] = []
    selection_counts: dict[str, int] = {}
    raw_counts: dict[str, int] = {key: len(value) for key, value in sorted(buckets.items())}
    for semantic, rows in sorted(buckets.items()):
        sorted_rows = sorted(rows, key=_stable_rank_key)
        ratio = float(keep_ratios.get(semantic, 1.0))
        keep_n = max(1, int(math.ceil(len(sorted_rows) * ratio)))
        chosen_rows = sorted_rows[:keep_n]
        selection_counts[semantic] = int(len(chosen_rows))
        selected.extend(chosen_rows)

    selected.sort(key=_stable_rank_key)
    capped_rows = _cap_terminal_fraction(selected, max_terminal_fraction=max_terminal_fraction)
    capped_counts = Counter(_semantic_of(row) for row in capped_rows)
    summary = {
        "profile": profile,
        "raw_counts": raw_counts,
        "selection_counts_before_terminal_cap": dict(sorted(selection_counts.items())),
        "selection_counts_after_terminal_cap": dict(sorted(capped_counts.items())),
    }
    return capped_rows, summary


def _cap_terminal_fraction(rows: list[dict[str, Any]], *, max_terminal_fraction: float) -> list[dict[str, Any]]:
    terminal_rows = [row for row in rows if _semantic_of(row) == "terminal_completion_anchor"]
    non_terminal_rows = [row for row in rows if _semantic_of(row) != "terminal_completion_anchor"]
    if not terminal_rows:
        return rows
    max_terminal = int(math.floor((len(non_terminal_rows) * float(max_terminal_fraction)) / max(1e-9, 1.0 - float(max_terminal_fraction))))
    max_terminal = max(1, max_terminal)
    if len(terminal_rows) <= max_terminal:
        return rows
    kept_terminal = sorted(terminal_rows, key=_stable_rank_key)[:max_terminal]
    merged = list(non_terminal_rows) + list(kept_terminal)
    merged.sort(key=_stable_rank_key)
    return merged


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_semantics = Counter(_semantic_of(row) for row in rows)
    by_source = Counter(str(row.get("source_tag", "unknown")) for row in rows)
    total = len(rows)
    return {
        "num_pairs": int(total),
        "by_source": dict(sorted(by_source.items())),
        "by_semantics": dict(sorted(by_semantics.items())),
        "semantics_fraction": {
            key: (float(value) / float(total) if total else 0.0) for key, value in sorted(by_semantics.items())
        },
        "mean_pair_confidence": (
            float(sum(float(row.get("pair_confidence", 0.0)) for row in rows) / total) if total else 0.0
        ),
    }


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    return (
        "# Phase E Semantic Consensus Pair Curation\n\n"
        f"- profile: `{summary['profile']}`\n"
        f"- train_pairs_before: `{summary['train_pairs_before']}`\n"
        f"- train_pairs_after: `{summary['train_pairs_after']}`\n"
        f"- validation_pairs: `{summary['validation_pairs']}`\n"
        f"- raw_counts: `{json.dumps(summary['raw_counts'], ensure_ascii=False, sort_keys=True)}`\n"
        f"- selection_counts_before_terminal_cap: `{json.dumps(summary['selection_counts_before_terminal_cap'], ensure_ascii=False, sort_keys=True)}`\n"
        f"- selection_counts_after_terminal_cap: `{json.dumps(summary['selection_counts_after_terminal_cap'], ensure_ascii=False, sort_keys=True)}`\n"
        f"- train_summary: `{json.dumps(summary['train_summary'], ensure_ascii=False, sort_keys=True)}`\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.output_root) / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _read_jsonl(Path(args.train_pairs_jsonl))
    validation_rows = _read_jsonl(Path(args.validation_pairs_jsonl))
    curated_rows, curation_summary = _apply_profile(
        train_rows,
        profile=str(args.profile),
        max_terminal_fraction=float(args.max_terminal_fraction),
    )

    _write_jsonl(run_dir / "train_pairs.jsonl", curated_rows)
    _write_jsonl(run_dir / "validation_pairs.jsonl", validation_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "profile": str(args.profile),
        "input_train_pairs_jsonl": str(args.train_pairs_jsonl),
        "input_validation_pairs_jsonl": str(args.validation_pairs_jsonl),
        "train_pairs_before": int(len(train_rows)),
        "train_pairs_after": int(len(curated_rows)),
        "validation_pairs": int(len(validation_rows)),
        **curation_summary,
        "train_summary": _summarize(curated_rows),
        "validation_summary": _summarize(validation_rows),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_stage": "phase_e_semantic_consensus_pairs_v1",
                "generated_at": summary["generated_at"],
                "run_dir": str(run_dir),
                "profile": str(args.profile),
                "input_files": {
                    "train_pairs_jsonl": str(args.train_pairs_jsonl),
                    "validation_pairs_jsonl": str(args.validation_pairs_jsonl),
                },
                "build_config": {
                    "max_terminal_fraction": float(args.max_terminal_fraction),
                    "profile_keep_ratios": PROFILE_SPECS[str(args.profile)],
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text(_render_summary_markdown(summary), encoding="utf-8")

    print("=" * 88)
    print("Phase E Semantic Consensus Pair Curation")
    print("=" * 88)
    print(f"run_dir          : {run_dir}")
    print(f"profile          : {args.profile}")
    print(f"train_before     : {len(train_rows)}")
    print(f"train_after      : {len(curated_rows)}")
    print(f"validation_pairs : {len(validation_rows)}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
