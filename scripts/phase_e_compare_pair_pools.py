#!/usr/bin/env python3
"""Compare Phase E pair-pool artifacts and surface semantics-coverage risks.

English
-------
This script audits pair artifacts before training so data-path mistakes can be
caught without spending GPU time. The main questions are:
1. how large is each train/validation pool,
2. which sources and pair semantics dominate,
3. whether critical semantics such as `local_first_bad_edge` disappeared.

中文
----
这个脚本在训练前先审计 pair artifact，避免把 GPU 时间浪费在明显有问题的
数据池上。它主要回答三件事：
1. train/validation 各有多大，
2. 哪些 source 和 pair semantics 占主导，
3. 像 `local_first_bad_edge` 这样的关键语义是否被错误地筛掉了。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PairPoolSummary:
    label: str
    pair_dir: str
    num_train_pairs: int
    num_validation_pairs: int
    mean_train_confidence: float
    by_source: dict[str, int]
    by_semantics: dict[str, int]
    semantics_fraction: dict[str, float]
    findings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Phase E pair-pool artifacts.")
    parser.add_argument(
        "--pair-dir",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        required=True,
        help="One pair pool entry as LABEL PATH.",
    )
    parser.add_argument("--run-name", default="phase_e_pair_pool_compare")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_analysis"),
    )
    parser.add_argument(
        "--terminal-fraction-warn-threshold",
        type=float,
        default=0.18,
        help="Warn when terminal anchors exceed this train-pool fraction.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    normalized: list[tuple[str, Path]] = []
    seen_labels: set[str] = set()
    for label, path_text in args.pair_dir:
        if label in seen_labels:
            raise ValueError(f"Duplicate --pair-dir label: {label}")
        seen_labels.add(label)
        pair_dir = Path(path_text)
        train_path = pair_dir / "train_pairs.jsonl"
        validation_path = pair_dir / "validation_pairs.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing train_pairs.jsonl for {label}: {train_path}")
        if not validation_path.exists():
            raise FileNotFoundError(f"Missing validation_pairs.jsonl for {label}: {validation_path}")
        normalized.append((label, pair_dir))
    if float(args.terminal_fraction_warn_threshold) <= 0.0:
        raise ValueError("--terminal-fraction-warn-threshold must be > 0")
    args.pair_dir = normalized
    return args


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            rows.append(json.loads(raw))
    return rows


def _summarize_pair_pool(
    *,
    label: str,
    pair_dir: Path,
    terminal_fraction_warn_threshold: float,
) -> PairPoolSummary:
    train_rows = _read_jsonl(pair_dir / "train_pairs.jsonl")
    validation_rows = _read_jsonl(pair_dir / "validation_pairs.jsonl")

    by_source = Counter()
    by_semantics = Counter()
    confidences: list[float] = []
    for row in train_rows:
        by_source[str(row.get("source_tag", "unknown"))] += 1
        metadata = dict(row.get("metadata") or {})
        by_semantics[str(metadata.get("pair_semantics", "unknown"))] += 1
        confidences.append(float(row.get("pair_confidence", 0.0)))

    total = len(train_rows)
    semantics_fraction = {
        key: (float(value) / float(total) if total else 0.0)
        for key, value in sorted(by_semantics.items())
    }

    findings: list[str] = []
    if "local_first_bad_edge" not in by_semantics:
        findings.append("missing_local_first_bad_edge")
    if "sibling_branch" not in by_semantics:
        findings.append("missing_sibling_branch")
    terminal_fraction = semantics_fraction.get("terminal_completion_anchor", 0.0)
    if terminal_fraction > float(terminal_fraction_warn_threshold):
        findings.append(f"terminal_fraction_high:{terminal_fraction:.3f}")
    if total and len(by_semantics) <= 1:
        findings.append("single_semantics_pool")

    return PairPoolSummary(
        label=label,
        pair_dir=str(pair_dir),
        num_train_pairs=int(len(train_rows)),
        num_validation_pairs=int(len(validation_rows)),
        mean_train_confidence=(float(sum(confidences) / len(confidences)) if confidences else 0.0),
        by_source=dict(sorted(by_source.items())),
        by_semantics=dict(sorted(by_semantics.items())),
        semantics_fraction=semantics_fraction,
        findings=findings,
    )


def _render_summary_markdown(summaries: list[PairPoolSummary]) -> str:
    lines = [
        "# Phase E Pair Pool Comparison",
        "",
        "| label | train_pairs | val_pairs | mean_conf | local_frac | sibling_frac | terminal_frac | findings |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        local_frac = item.semantics_fraction.get("local_first_bad_edge", 0.0)
        sibling_frac = item.semantics_fraction.get("sibling_branch", 0.0)
        terminal_frac = item.semantics_fraction.get("terminal_completion_anchor", 0.0)
        lines.append(
            "| {label} | {train_pairs} | {val_pairs} | {mean_conf:.3f} | {local_frac:.3f} | "
            "{sibling_frac:.3f} | {terminal_frac:.3f} | {findings} |".format(
                label=item.label,
                train_pairs=item.num_train_pairs,
                val_pairs=item.num_validation_pairs,
                mean_conf=item.mean_train_confidence,
                local_frac=local_frac,
                sibling_frac=sibling_frac,
                terminal_frac=terminal_frac,
                findings=", ".join(item.findings) if item.findings else "ok",
            )
        )
    lines.extend(["", "## Detailed Breakdown", ""])
    for item in summaries:
        lines.extend(
            [
                f"### {item.label}",
                "",
                f"- pair_dir: `{item.pair_dir}`",
                f"- num_train_pairs: `{item.num_train_pairs}`",
                f"- num_validation_pairs: `{item.num_validation_pairs}`",
                f"- mean_train_confidence: `{item.mean_train_confidence:.4f}`",
                f"- by_source: `{json.dumps(item.by_source, ensure_ascii=False, sort_keys=True)}`",
                f"- by_semantics: `{json.dumps(item.by_semantics, ensure_ascii=False, sort_keys=True)}`",
                f"- findings: `{item.findings or ['ok']}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.output_root) / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        _summarize_pair_pool(
            label=label,
            pair_dir=pair_dir,
            terminal_fraction_warn_threshold=float(args.terminal_fraction_warn_threshold),
        )
        for label, pair_dir in args.pair_dir
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "summaries": [item.to_dict() for item in summaries],
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(_render_summary_markdown(summaries), encoding="utf-8")

    print("=" * 88)
    print("Phase E Pair Pool Comparison")
    print("=" * 88)
    print(f"run_dir: {run_dir}")
    for item in summaries:
        print(
            f"- {item.label}: train={item.num_train_pairs} val={item.num_validation_pairs} "
            f"findings={item.findings or ['ok']}"
        )
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
