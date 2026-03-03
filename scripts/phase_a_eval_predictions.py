#!/usr/bin/env python3
"""Evaluate prediction JSONL files with dataset-aware answer extraction.

Why this file exists
--------------------
Sometimes inference has already finished and only evaluation needs to be rerun.
This script scores saved prediction artifacts without repeating generation.

What this file does
-------------------
1. Parse the prediction-file path and output location.
2. Load prediction records from JSONL with strict validation.
3. Delegate scoring to the shared Phase A evaluator.
4. Persist scored predictions and summary metrics to a fresh run directory.

Input JSONL contract
--------------------
Required fields:
- sample_id
- dataset
- split
- raw_prediction
- gold_answer

Optional fields:
- question
- metadata

Interaction with other files
----------------------------
- `src/ours/phase_a/contracts.py`: prediction/scored row contracts
- `src/ours/phase_a/evaluator.py`: actual extraction/evaluation logic
- `scripts/phase_a_generate_and_eval.py`: upstream producer of prediction JSONL files

Example
-------
```bash
python scripts/phase_a_eval_predictions.py \
  --predictions assets/artifacts/phase_a_runs/example_run/predictions.jsonl \
  --run-name reeval_example
```
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _bootstrap_src_path() -> None:
    """Add the repo-local `src/` directory to `sys.path`.

    Example
    -------
    ```bash
    python scripts/phase_a_eval_predictions.py --help
    ```
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_a import PredictionRecord, evaluate_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline evaluation of saved predictions.

    Example
    -------
    ```python
    args = parse_args()
    ```
    """
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
    """Run offline evaluation on a saved prediction JSONL file.

    Returns
    -------
    int
        `0` when evaluation completed successfully.

    Example
    -------
    ```bash
    python scripts/phase_a_eval_predictions.py --predictions predictions.jsonl
    ```
    """
    args = parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(f"Prediction file not found: {args.predictions}")

    # 这里严格按 PredictionRecord 合约回读 JSONL，目的是保证离线复评估
    # 与在线 generate+eval 路径使用完全一致的打分输入结构。
    records = list(_load_prediction_records(args.predictions))
    scored, summary = evaluate_predictions(records)

    # 输出目录按时间戳隔离，避免覆盖历史复评估结果。
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
    print(f"n_parseable      : {summary.n_parseable}")
    print(f"acc_parseable    : {summary.accuracy_parseable:.4f}")
    print(f"output_dir       : {run_dir}")
    print("=" * 88)
    return 0


def _load_prediction_records(path: Path) -> Iterable[PredictionRecord]:
    """Yield validated `PredictionRecord` objects from a JSONL file.

    Example
    -------
    ```python
    records = list(_load_prediction_records(Path("predictions.jsonl")))
    ```
    """
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if line.strip() == "":
                continue
            # 每一行都先做 JSON 解析，再做 PredictionRecord 级别字段校验。
            # 任意坏行都直接报错，防止“静默跳过坏数据”造成指标失真。
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL row at line={idx} in file={path}: {exc}"
                ) from exc
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
                    f"Invalid prediction row at line={idx} in file={path}: {exc}"
                ) from exc


if __name__ == "__main__":
    raise SystemExit(main())
