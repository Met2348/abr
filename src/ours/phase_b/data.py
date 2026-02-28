"""Load and summarize Phase B training rows from prepared JSONL artifacts.

Why this file exists
--------------------
The training script should not mix JSON parsing, schema validation, duplicate-id
checks, and dataset summaries into one long function. This module isolates those
responsibilities.

What this file contains
-----------------------
- `load_phase_b_rows(...)`: strict JSONL loader with duplicate-id detection
- `summarize_rows(...)`: compact row statistics for manifests and debug prints

Interaction with other files
----------------------------
- `src/ours/phase_b/contracts.py`: row schema and field validation.
- `scripts/phase_b_train_sft.py`: main caller during training startup.

Example
-------
```python
rows = load_phase_b_rows(Path("train.jsonl"), max_samples=256)
summary = summarize_rows(rows)
```
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .contracts import PhaseBTrainRow


def load_phase_b_rows(
    path: Path,
    max_samples: int | None = None,
) -> list[PhaseBTrainRow]:
    """Load and validate Phase B rows from a prepared JSONL file.

    This function is strict on duplicate IDs because duplicates can silently
    bias training and evaluation.

    Example
    -------
    ```python
    rows = load_phase_b_rows(Path("train.jsonl"), max_samples=128)
    ```
    """

    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    rows: list[PhaseBTrainRow] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            row = PhaseBTrainRow.from_dict(payload)
            if row.sample_id in seen_ids:
                raise ValueError(
                    f"Duplicate sample_id detected in {path} at line {line_no}: "
                    f"{row.sample_id!r}"
                )
            seen_ids.add(row.sample_id)
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break

    return rows


def summarize_rows(rows: list[PhaseBTrainRow]) -> dict[str, Any]:
    """Return compact stats for run manifests and debug prints.

    Example
    -------
    ```python
    summary = summarize_rows(rows)
    print(summary["dataset_counts"])
    ```
    """

    by_dataset = Counter(row.dataset for row in rows)
    by_split = Counter(row.split for row in rows)
    return {
        "num_rows": len(rows),
        "dataset_counts": dict(by_dataset),
        "split_counts": dict(by_split),
    }
