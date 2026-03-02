#!/usr/bin/env python3
"""Evaluate multiple checkpoints from one Phase B run against held-out metrics.

Why this file exists
--------------------
Trainer loss alone is not enough to explain GSM8K regressions. A run may reach a
better benchmark state in the middle of training and then drift by the final
checkpoint. This script evaluates saved checkpoints and the final adapter/model
through the frozen Phase A evaluator and reports where the best held-out metric
actually occurs.

What this file does
-------------------
1. Load a finished Phase B run directory.
2. Run a baseline held-out evaluation on the frozen base model.
3. Evaluate each requested checkpoint plus the final saved artifact.
4. Aggregate validation/test metrics into one sweep report.
5. Persist JSON and Markdown summaries that can be linked from suite logs.

Interaction with other files
----------------------------
- `scripts/phase_b_eval.py`: performs each individual checkpoint evaluation.
- `scripts/phase_a_generate_and_eval.py`: remains the frozen benchmark engine.
- `scripts/run_phase_b_training_suite.sh`: can call this script automatically as
  part of a named diagnostic group.

Example
-------
```bash
python -u scripts/phase_b_checkpoint_sweep.py \
  --phase-b-run-dir assets/artifacts/phase_b_runs/my_run_20260302T000000Z \
  --dataset gsm8k \
  --run-name-prefix gsm8k_ckpt_sweep \
  --batch-size 64 \
  --eval-spec validation assets/artifacts/phase_a_prepared/gsm8k/e3/train.jsonl freeform 192 \
  --eval-spec test assets/artifacts/phase_a_prepared/gsm8k/e3/test.jsonl freeform 192
```
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MetricSnapshot:
    """Normalized Phase A metric view used by the checkpoint sweep report.

    Example
    -------
    ```python
    snap = MetricSnapshot.from_path(Path("metrics.json"))
    ```
    """

    path: Path
    n_total: int
    n_parseable: int
    accuracy: float
    parse_error_rate: float

    @classmethod
    def from_path(cls, path: Path) -> "MetricSnapshot":
        """Load one Phase A `metrics.json` file.

        Example
        -------
        ```python
        snap = MetricSnapshot.from_path(Path("metrics.json"))
        ```
        """

        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            n_total=int(payload["n_total"]),
            n_parseable=int(payload["n_parseable"]),
            accuracy=float(payload["accuracy"]),
            parse_error_rate=float(payload["parse_error_rate"]),
        )

    @property
    def n_correct(self) -> int:
        """Return the integer count of correct predictions."""

        return int(round(self.accuracy * self.n_total))


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for checkpoint-sweep diagnostics."""

    parser = argparse.ArgumentParser(
        description="Evaluate a Phase B checkpoint sweep against held-out metrics."
    )
    parser.add_argument("--phase-b-run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument(
        "--title",
        default="Phase B Checkpoint Sweep",
        help="Human-readable title printed in console and markdown output.",
    )
    parser.add_argument(
        "--run-name-prefix",
        default="phase_b_checkpoint_sweep",
        help="Prefix used for the generated Phase A eval run names.",
    )
    parser.add_argument(
        "--checkpoint-labels",
        default="",
        help=(
            "Comma-separated checkpoint step numbers to evaluate, for example "
            "`100,200,300`. Leave empty to evaluate every retained checkpoint. "
            "The final saved artifact is always included."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--eval-spec",
        nargs=4,
        action="append",
        metavar=("LABEL", "INPUT_JSONL", "DECODE_MODE", "MAX_NEW_TOKENS"),
        required=True,
        help="One held-out split spec. Repeat for validation/test.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded into scripts/phase_b_eval.py.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse checkpoint-sweep CLI arguments.

    Example
    -------
    ```python
    args = parse_args([
        "--phase-b-run-dir", "assets/artifacts/phase_b_runs/demo",
        "--eval-spec", "validation", "validation.jsonl", "freeform", "192",
    ])
    ```
    """

    return _build_parser().parse_args(argv)


def _latest_phase_a_metrics_for_name(run_name: str) -> Path:
    """Resolve the newest Phase A `metrics.json` for one eval run-name prefix."""

    candidates = sorted(
        Path("assets/artifacts/phase_a_runs").glob(f"{run_name}_*/metrics.json")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No Phase A metrics found for run-name prefix: {run_name}"
        )
    return candidates[-1]


def _resolve_base_model_path(run_dir: Path) -> str:
    """Read the base model path from one finished Phase B run manifest."""

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Phase B manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = str(manifest.get("model_path", "")).strip()
    if model_path == "":
        raise ValueError(f"Missing `model_path` in manifest: {manifest_path}")
    return model_path


def _resolve_checkpoint_targets(run_dir: Path, checkpoint_labels: str) -> list[tuple[str, Path | None]]:
    """Resolve requested checkpoint labels into filesystem targets.

    Returns
    -------
    list[tuple[str, Path | None]]
        Pairs of `(label, checkpoint_dir_or_none)`. `None` means the run's
        `final_model/` directory should be used.
    """

    checkpoint_root = run_dir / "checkpoints"
    available = {}
    for path in sorted(checkpoint_root.glob("checkpoint-*")):
        label = path.name.removeprefix("checkpoint-")
        available[label] = path

    targets: list[tuple[str, Path | None]] = []
    if checkpoint_labels.strip():
        labels = [item.strip() for item in checkpoint_labels.split(",") if item.strip()]
        for label in labels:
            if label not in available:
                raise FileNotFoundError(
                    f"Requested checkpoint-{label} not found under {checkpoint_root}"
                )
            targets.append((label, available[label]))
    else:
        for label, path in sorted(available.items(), key=lambda item: int(item[0])):
            targets.append((label, path))

    targets.append(("final", None))
    return targets


def _run_eval(
    *,
    run_name: str,
    input_jsonl: str,
    decode_mode: str,
    max_new_tokens: str,
    batch_size: int,
    require_cuda: bool,
    model_path: str | None = None,
    phase_b_run_dir: Path | None = None,
    phase_b_checkpoint_dir: Path | None = None,
    extra_args: list[str],
) -> Path:
    """Run one held-out eval via the Phase B evaluation bridge.

    Example
    -------
    ```python
    metrics_path = _run_eval(
        run_name="demo_validation",
        input_jsonl="validation.jsonl",
        decode_mode="freeform",
        max_new_tokens="192",
        batch_size=64,
        require_cuda=True,
        model_path="assets/models/Qwen2.5-7B-Instruct",
        extra_args=[],
    )
    ```
    """

    repo_root = Path(__file__).resolve().parents[1]
    target_script = repo_root / "scripts" / "phase_b_eval.py"
    cmd = [
        sys.executable,
        "-u",
        str(target_script),
        "--input-jsonl",
        input_jsonl,
        "--run-name",
        run_name,
        "--batch-size",
        str(batch_size),
        "--strategyqa-decode-mode",
        decode_mode,
        "--max-new-tokens",
        str(max_new_tokens),
    ]

    if model_path is not None:
        cmd.extend(["--model-path", model_path])
    elif phase_b_run_dir is not None:
        cmd.extend(["--phase-b-run-dir", str(phase_b_run_dir)])
    elif phase_b_checkpoint_dir is not None:
        cmd.extend(["--phase-b-checkpoint-dir", str(phase_b_checkpoint_dir)])
    else:
        raise ValueError("One evaluation source must be provided.")

    cmd.append("--require-cuda" if require_cuda else "--no-require-cuda")
    if extra_args:
        cmd.append("--extra-args")
        cmd.extend(extra_args)

    print(f"[eval] {' '.join(cmd)}")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Checkpoint sweep eval failed with exit code {completed.returncode}: "
            f"{' '.join(cmd)}"
        )
    return _latest_phase_a_metrics_for_name(run_name)


def _aggregate_snapshots(
    baseline: dict[str, MetricSnapshot],
    after: dict[str, MetricSnapshot],
) -> dict[str, Any]:
    """Aggregate split metrics into one held-out summary row."""

    total = sum(item.n_total for item in baseline.values())
    before_correct = sum(item.n_correct for item in baseline.values())
    after_correct = sum(item.n_correct for item in after.values())
    before_parse = sum(item.n_total - item.n_parseable for item in baseline.values())
    after_parse = sum(item.n_total - item.n_parseable for item in after.values())
    return {
        "n_total": total,
        "before_accuracy": before_correct / total if total else 0.0,
        "after_accuracy": after_correct / total if total else 0.0,
        "delta_accuracy": (after_correct - before_correct) / total if total else 0.0,
        "before_n_correct": before_correct,
        "after_n_correct": after_correct,
        "delta_n_correct": after_correct - before_correct,
        "before_parse_error_rate": before_parse / total if total else 0.0,
        "after_parse_error_rate": after_parse / total if total else 0.0,
    }


def _render_markdown(
    *,
    title: str,
    dataset: str,
    run_dir: Path,
    checkpoint_rows: list[dict[str, Any]],
) -> str:
    """Render the checkpoint sweep as a compact Markdown report."""

    best_row = max(checkpoint_rows, key=lambda row: row["aggregate"]["after_accuracy"])
    lines = [
        f"# {title}",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- dataset: {dataset}",
        f"- phase_b_run_dir: `{run_dir}`",
        f"- best_checkpoint: `{best_row['checkpoint_label']}`",
        f"- best_after_accuracy: `{best_row['aggregate']['after_accuracy']:.4f}`",
        "",
        "| Checkpoint | Held-out acc | Delta acc | Delta correct | Parse err after |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in checkpoint_rows:
        agg = row["aggregate"]
        lines.append(
            "| "
            f"{row['checkpoint_label']} | "
            f"{agg['after_accuracy']:.4f} | "
            f"{agg['delta_accuracy']:+.4f} | "
            f"{agg['delta_n_correct']:+d} | "
            f"{agg['after_parse_error_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Split Details",
        ]
    )
    for row in checkpoint_rows:
        lines.append(f"- `{row['checkpoint_label']}`")
        for label, metrics in row["splits"].items():
            lines.append(
                "  "
                f"- {label}: before={metrics['before_accuracy']:.4f}, "
                f"after={metrics['after_accuracy']:.4f}, "
                f"delta={metrics['delta_accuracy']:+.4f}"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the full checkpoint-sweep diagnostic.

    Example
    -------
    ```bash
    python scripts/phase_b_checkpoint_sweep.py \
      --phase-b-run-dir assets/artifacts/phase_b_runs/demo \
      --eval-spec validation validation.jsonl freeform 192
    ```
    """

    args = parse_args(argv)
    if not args.phase_b_run_dir.exists():
        raise FileNotFoundError(
            f"Phase B run directory not found: {args.phase_b_run_dir}"
        )

    print("=" * 88)
    print("Phase B: Checkpoint Sweep")
    print("=" * 88)
    print(f"phase_b_run_dir   : {args.phase_b_run_dir}")
    print(f"dataset           : {args.dataset}")
    print(f"batch_size        : {args.batch_size}")
    print(f"checkpoint_labels : {args.checkpoint_labels or '<all retained>'}")

    base_model_path = _resolve_base_model_path(args.phase_b_run_dir)
    baseline_by_split: dict[str, MetricSnapshot] = {}
    checkpoint_rows: list[dict[str, Any]] = []

    for label, input_jsonl, decode_mode, max_new_tokens in args.eval_spec:
        run_name = f"{args.run_name_prefix}_pre_{label}"
        metrics_path = _run_eval(
            run_name=run_name,
            input_jsonl=input_jsonl,
            decode_mode=decode_mode,
            max_new_tokens=max_new_tokens,
            batch_size=args.batch_size,
            require_cuda=args.require_cuda,
            model_path=base_model_path,
            extra_args=args.extra_args,
        )
        baseline_by_split[label] = MetricSnapshot.from_path(metrics_path)

    for checkpoint_label, checkpoint_dir in _resolve_checkpoint_targets(
        args.phase_b_run_dir,
        args.checkpoint_labels,
    ):
        split_rows: dict[str, Any] = {}
        after_by_split: dict[str, MetricSnapshot] = {}
        for label, input_jsonl, decode_mode, max_new_tokens in args.eval_spec:
            run_name = f"{args.run_name_prefix}_ckpt{checkpoint_label}_{label}"
            metrics_path = _run_eval(
                run_name=run_name,
                input_jsonl=input_jsonl,
                decode_mode=decode_mode,
                max_new_tokens=max_new_tokens,
                batch_size=args.batch_size,
                require_cuda=args.require_cuda,
                phase_b_run_dir=(
                    args.phase_b_run_dir if checkpoint_dir is None else None
                ),
                phase_b_checkpoint_dir=checkpoint_dir,
                extra_args=args.extra_args,
            )
            after_snap = MetricSnapshot.from_path(metrics_path)
            after_by_split[label] = after_snap
            before_snap = baseline_by_split[label]
            split_rows[label] = {
                "before_metrics_path": str(before_snap.path),
                "after_metrics_path": str(after_snap.path),
                "before_accuracy": before_snap.accuracy,
                "after_accuracy": after_snap.accuracy,
                "delta_accuracy": after_snap.accuracy - before_snap.accuracy,
                "before_parse_error_rate": before_snap.parse_error_rate,
                "after_parse_error_rate": after_snap.parse_error_rate,
            }

        checkpoint_rows.append(
            {
                "checkpoint_label": checkpoint_label,
                "checkpoint_dir": (
                    str(checkpoint_dir) if checkpoint_dir is not None else None
                ),
                "splits": split_rows,
                "aggregate": _aggregate_snapshots(baseline_by_split, after_by_split),
            }
        )

    report = {
        "title": args.title,
        "dataset": args.dataset,
        "phase_b_run_dir": str(args.phase_b_run_dir),
        "base_model_path": base_model_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_rows": checkpoint_rows,
    }

    markdown = _render_markdown(
        title=args.title,
        dataset=args.dataset,
        run_dir=args.phase_b_run_dir,
        checkpoint_rows=checkpoint_rows,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown + "\n", encoding="utf-8")

    print("-" * 88)
    print(markdown)
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
