#!/usr/bin/env python3
"""Compare pre/post Phase A benchmark metrics for one Phase B training run.

Why this file exists
--------------------
Phase B training produces optimizer-centric metrics such as loss, but the project
needs benchmark-centric answers such as "did PEFT improve accuracy on held-out
StrategyQA or GSM8K examples?" This script converts paired pre/post Phase A metrics
into one report that is easier to trust and present.

What this file does
-------------------
1. Load one or more `(before_metrics, after_metrics)` pairs.
2. Validate that each pair scored the same number of samples.
3. Compute per-split deltas for:
   - accuracy,
   - parse error rate,
   - parseable accuracy,
   - correct / parse-error counts.
4. Compute a held-out aggregate across all listed splits.
5. Print and optionally persist JSON/Markdown summaries.

Interaction with other files
----------------------------
- `scripts/phase_b_eval.py`: produces the post-train metrics this script compares.
- `scripts/phase_a_generate_and_eval.py`: produces the `metrics.json` files being
  loaded here.
- `scripts/run_phase_b_training_suite.sh`: can call this script automatically at the
  end of a named experiment group.

Example
-------
```bash
python -u scripts/phase_b_compare_eval.py \
  --dataset strategyqa \
  --phase-b-run-dir assets/artifacts/phase_b_runs/<phase_b_run_dir> \
  --compare validation before_validation_metrics.json after_validation_metrics.json \
  --compare test before_test_metrics.json after_test_metrics.json
```
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MetricSnapshot:
    """Normalized Phase A metric view for one evaluated split.

    Example
    -------
    ```python
    snap = MetricSnapshot.from_metrics_path(Path("metrics.json"))
    ```
    """

    path: Path
    n_total: int
    n_parseable: int
    accuracy: float
    parse_error_rate: float
    accuracy_parseable: float
    sample_per_second: float | None

    @classmethod
    def from_metrics_path(cls, path: Path) -> "MetricSnapshot":
        """Load and validate a Phase A `metrics.json` file.

        Parameters
        ----------
        path:
            File produced by `scripts/phase_a_generate_and_eval.py`.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        generation_stats = payload.get("generation_stats") or {}
        return cls(
            path=path,
            n_total=int(payload["n_total"]),
            n_parseable=int(payload["n_parseable"]),
            accuracy=float(payload["accuracy"]),
            parse_error_rate=float(payload["parse_error_rate"]),
            accuracy_parseable=float(payload["accuracy_parseable"]),
            sample_per_second=(
                float(generation_stats["sample_per_second"])
                if generation_stats.get("sample_per_second") is not None
                else None
            ),
        )

    @property
    def n_correct(self) -> int:
        """Return the integer count of correct predictions."""
        return int(round(self.accuracy * self.n_total))

    @property
    def n_parse_error(self) -> int:
        """Return the integer count of parse-error predictions."""
        return self.n_total - self.n_parseable

    @property
    def n_correct_parseable(self) -> int:
        """Return the integer count of correct parseable predictions."""
        return int(round(self.accuracy_parseable * self.n_parseable))


@dataclass(slots=True)
class PairSummary:
    """Delta summary for one benchmark split."""

    label: str
    before: MetricSnapshot
    after: MetricSnapshot

    def to_dict(self) -> dict[str, Any]:
        """Serialize one comparison into a JSON-friendly dict."""
        return {
            "label": self.label,
            "n_total": self.before.n_total,
            "before_metrics_path": str(self.before.path),
            "after_metrics_path": str(self.after.path),
            "before_accuracy": self.before.accuracy,
            "after_accuracy": self.after.accuracy,
            "delta_accuracy": self.after.accuracy - self.before.accuracy,
            "before_parse_error_rate": self.before.parse_error_rate,
            "after_parse_error_rate": self.after.parse_error_rate,
            "delta_parse_error_rate": self.after.parse_error_rate
            - self.before.parse_error_rate,
            "before_accuracy_parseable": self.before.accuracy_parseable,
            "after_accuracy_parseable": self.after.accuracy_parseable,
            "delta_accuracy_parseable": self.after.accuracy_parseable
            - self.before.accuracy_parseable,
            "before_n_correct": self.before.n_correct,
            "after_n_correct": self.after.n_correct,
            "delta_n_correct": self.after.n_correct - self.before.n_correct,
            "before_n_parse_error": self.before.n_parse_error,
            "after_n_parse_error": self.after.n_parse_error,
            "delta_n_parse_error": self.after.n_parse_error - self.before.n_parse_error,
            "before_sample_per_second": self.before.sample_per_second,
            "after_sample_per_second": self.after.sample_per_second,
            "delta_sample_per_second": (
                (self.after.sample_per_second - self.before.sample_per_second)
                if self.before.sample_per_second is not None
                and self.after.sample_per_second is not None
                else None
            ),
        }


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for PEFT gain comparison."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare before/after Phase A benchmark metrics for one Phase B run."
        )
    )
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--phase-b-run-dir", type=Path, default=None)
    parser.add_argument(
        "--title",
        default="Phase B PEFT Gain Summary",
        help="Human-readable title used in console and markdown output.",
    )
    parser.add_argument(
        "--compare",
        nargs=3,
        action="append",
        metavar=("LABEL", "BEFORE_METRICS", "AFTER_METRICS"),
        required=True,
        help=(
            "One split comparison triple. Repeat this flag for validation/test/etc."
        ),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for the comparison report.

    Example
    -------
    ```python
    args = parse_args([
        "--dataset", "strategyqa",
        "--compare", "validation", "before.json", "after.json",
    ])
    ```
    """
    return _build_parser().parse_args(argv)


def _load_pairs(args: argparse.Namespace) -> list[PairSummary]:
    """Load and validate all requested comparison pairs."""
    pairs: list[PairSummary] = []
    for label, before_str, after_str in args.compare:
        before = MetricSnapshot.from_metrics_path(Path(before_str))
        after = MetricSnapshot.from_metrics_path(Path(after_str))
        if before.n_total != after.n_total:
            raise ValueError(
                f"Split `{label}` compares mismatched totals: "
                f"before={before.n_total} after={after.n_total}"
            )
        pairs.append(PairSummary(label=label, before=before, after=after))
    return pairs


def _aggregate_pairs(pairs: list[PairSummary]) -> dict[str, Any]:
    """Compute held-out aggregate counts and deltas across all comparisons."""
    n_total = sum(pair.before.n_total for pair in pairs)
    before_correct = sum(pair.before.n_correct for pair in pairs)
    after_correct = sum(pair.after.n_correct for pair in pairs)
    before_parse_error = sum(pair.before.n_parse_error for pair in pairs)
    after_parse_error = sum(pair.after.n_parse_error for pair in pairs)
    before_parseable = sum(pair.before.n_parseable for pair in pairs)
    after_parseable = sum(pair.after.n_parseable for pair in pairs)
    before_correct_parseable = sum(pair.before.n_correct_parseable for pair in pairs)
    after_correct_parseable = sum(pair.after.n_correct_parseable for pair in pairs)

    return {
        "n_total": int(n_total),
        "before_n_correct": int(before_correct),
        "after_n_correct": int(after_correct),
        "delta_n_correct": int(after_correct - before_correct),
        "before_accuracy": float(before_correct / n_total if n_total else 0.0),
        "after_accuracy": float(after_correct / n_total if n_total else 0.0),
        "delta_accuracy": float(
            (after_correct - before_correct) / n_total if n_total else 0.0
        ),
        "before_n_parse_error": int(before_parse_error),
        "after_n_parse_error": int(after_parse_error),
        "delta_n_parse_error": int(after_parse_error - before_parse_error),
        "before_parse_error_rate": float(
            before_parse_error / n_total if n_total else 0.0
        ),
        "after_parse_error_rate": float(after_parse_error / n_total if n_total else 0.0),
        "delta_parse_error_rate": float(
            (after_parse_error - before_parse_error) / n_total if n_total else 0.0
        ),
        "before_n_parseable": int(before_parseable),
        "after_n_parseable": int(after_parseable),
        "before_accuracy_parseable": float(
            before_correct_parseable / before_parseable if before_parseable else 0.0
        ),
        "after_accuracy_parseable": float(
            after_correct_parseable / after_parseable if after_parseable else 0.0
        ),
        "delta_accuracy_parseable": float(
            (after_correct_parseable / after_parseable if after_parseable else 0.0)
            - (before_correct_parseable / before_parseable if before_parseable else 0.0)
        ),
    }


def _build_report_payload(args: argparse.Namespace, pairs: list[PairSummary]) -> dict[str, Any]:
    """Build one combined JSON-friendly report payload."""
    aggregate = _aggregate_pairs(pairs)
    direction = "improved"
    if aggregate["delta_accuracy"] < 0:
        direction = "declined"
    elif aggregate["delta_accuracy"] == 0:
        direction = "stayed_flat"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "title": args.title,
        "dataset": args.dataset,
        "phase_b_run_dir": str(args.phase_b_run_dir) if args.phase_b_run_dir else None,
        "pairs": [pair.to_dict() for pair in pairs],
        "aggregate": aggregate,
        "headline": {
            "direction": direction,
            "delta_accuracy": aggregate["delta_accuracy"],
            "delta_n_correct": aggregate["delta_n_correct"],
            "delta_parse_error_rate": aggregate["delta_parse_error_rate"],
        },
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    """Render a human-readable markdown report for the comparison payload."""
    lines = [
        f"# {payload['title']}",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- dataset: {payload['dataset']}",
        f"- phase_b_run_dir: {payload['phase_b_run_dir'] or 'N/A'}",
        "",
        "## Split Comparisons",
        "",
        "| split | n | acc_before | acc_after | delta_acc | parse_before | parse_after | delta_parse | correct_before | correct_after | delta_correct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for pair in payload["pairs"]:
        lines.append(
            "| {label} | {n_total} | {before_accuracy:.4f} | {after_accuracy:.4f} | "
            "{delta_accuracy:+.4f} | {before_parse_error_rate:.4f} | "
            "{after_parse_error_rate:.4f} | {delta_parse_error_rate:+.4f} | "
            "{before_n_correct} | {after_n_correct} | {delta_n_correct:+d} |".format(
                **pair
            )
        )

    agg = payload["aggregate"]
    lines.extend(
        [
            "",
            "## Held-Out Aggregate",
            "",
            f"- n_total: {agg['n_total']}",
            f"- accuracy_before: {agg['before_accuracy']:.4f}",
            f"- accuracy_after: {agg['after_accuracy']:.4f}",
            f"- delta_accuracy: {agg['delta_accuracy']:+.4f}",
            f"- correct_before: {agg['before_n_correct']}",
            f"- correct_after: {agg['after_n_correct']}",
            f"- delta_correct: {agg['delta_n_correct']:+d}",
            f"- parse_error_rate_before: {agg['before_parse_error_rate']:.4f}",
            f"- parse_error_rate_after: {agg['after_parse_error_rate']:.4f}",
            f"- delta_parse_error_rate: {agg['delta_parse_error_rate']:+.4f}",
            f"- acc_parseable_before: {agg['before_accuracy_parseable']:.4f}",
            f"- acc_parseable_after: {agg['after_accuracy_parseable']:.4f}",
            f"- delta_acc_parseable: {agg['delta_accuracy_parseable']:+.4f}",
            "",
            "## Headline",
            "",
            "- "
            f"PEFT {payload['headline']['direction']} held-out accuracy by "
            f"{payload['headline']['delta_accuracy']:+.4f} "
            f"({payload['headline']['delta_n_correct']:+d} correct predictions).",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_outputs(
    payload: dict[str, Any],
    *,
    output_json: Path | None,
    output_markdown: Path | None,
) -> None:
    """Persist optional JSON/Markdown artifacts."""
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(_render_markdown(payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Generate and optionally persist the PEFT gain report.

    Example
    -------
    ```bash
    python scripts/phase_b_compare_eval.py \
      --dataset strategyqa \
      --compare validation before.json after.json
    ```
    """
    args = parse_args(argv)
    pairs = _load_pairs(args)
    payload = _build_report_payload(args, pairs)
    markdown = _render_markdown(payload)
    print("=" * 88)
    print(args.title)
    print("=" * 88)
    print(markdown.rstrip())
    _write_outputs(
        payload,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
