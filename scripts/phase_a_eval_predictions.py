#!/usr/bin/env python3
"""Evaluate prediction JSONL files with dataset-aware extraction.

Input JSONL contract (one object per line)
------------------------------------------
Required fields:
- sample_id
- dataset
- split
- raw_prediction
- gold_answer

Optional fields:
- question
- metadata
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_a import PredictionRecord, evaluate_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score prediction JSONL files using Phase A evaluators."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to prediction JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/artifacts/phase_a_eval"),
        help="Directory where scored outputs/metrics are written.",
    )
    parser.add_argument(
        "--run-name",
        default="phase_a_eval",
        help="Friendly label used in output folder naming.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(f"Prediction file not found: {args.predictions}")

    records = list(_load_prediction_records(args.predictions))
    scored, summary = evaluate_predictions(records)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / f"{args.run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scored_path = run_dir / "scored_predictions.jsonl"
    metrics_path = run_dir / "metrics.json"

    with scored_path.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    metrics = summary.to_dict()
    metrics["source_predictions_path"] = str(args.predictions)
    metrics["scored_predictions_path"] = str(scored_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print("Phase A: Evaluation Result")
    print("=" * 88)
    print(f"input_file       : {args.predictions}")
    print(f"n_total          : {summary.n_total}")
    print(f"accuracy         : {summary.accuracy:.4f}")
    print(f"parse_error_rate : {summary.parse_error_rate:.4f}")
    print(f"output_dir       : {run_dir}")
    print("=" * 88)
    return 0


def _load_prediction_records(path: Path) -> Iterable[PredictionRecord]:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip() == "":
            continue
        payload = json.loads(line)
        try:
            record = PredictionRecord(
                sample_id=str(payload["sample_id"]),
                dataset=str(payload["dataset"]),
                split=str(payload["split"]),
                raw_prediction=str(payload["raw_prediction"]),
                gold_answer=str(payload["gold_answer"]),
                question=str(payload["question"]) if "question" in payload and payload["question"] is not None else None,
                metadata=dict(payload.get("metadata", {})),
            )
            record.validate()
            yield record
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Invalid prediction row at line={idx+1} in file={path}: {exc}"
            ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
