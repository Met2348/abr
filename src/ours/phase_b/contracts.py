"""Define the validated row contract used by Phase B training.

Why this file exists
--------------------
Phase B training reads JSONL records produced earlier in the pipeline. Those records
need one strict contract so the training code can fail early instead of discovering
schema issues inside tokenization or model code.

What this file contains
-----------------------
- `PhaseBTrainRow`: one normalized training record
- small validation helpers used by dataclass methods

Execution logic
---------------
Call `PhaseBTrainRow.from_dict(...)` on raw JSON payloads, then use
`PhaseBTrainRow.validate()` and `PhaseBTrainRow.to_dict()` when persisting or
re-checking values.

Interaction with other files
----------------------------
- `src/ours/phase_b/data.py` loads JSONL lines and instantiates this dataclass.
- `scripts/phase_b_train_sft.py` consumes validated row objects during training.

Example
-------
```python
row = PhaseBTrainRow.from_dict(payload)
row.validate()
```
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PhaseBTrainRow:
    """One normalized training row loaded from prepared JSONL.

    Required fields are aligned with `scripts/phase_a_prepare.py` outputs.

    中文要点
    --------
    这是 B 阶段最基础的数据契约；字段一旦变更，会影响 loader、训练与评测全链路。

    Example
    -------
    ```python
    row = PhaseBTrainRow(
        sample_id="strategyqa-train-1",
        dataset="strategyqa",
        split="train",
        prompt_text="Question: ...",
        target_text="yes",
        answer="yes",
    )
    ```
    """

    sample_id: str
    dataset: str
    split: str
    prompt_text: str
    target_text: str
    answer: str
    question: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate field types and required non-empty strings.

        Example
        -------
        ```python
        row.validate()
        ```
        """
        _validate_non_empty_str(self.sample_id, "sample_id")
        _validate_non_empty_str(self.dataset, "dataset")
        _validate_non_empty_str(self.split, "split")
        _validate_non_empty_str(self.prompt_text, "prompt_text")
        _validate_non_empty_str(self.target_text, "target_text")
        _validate_non_empty_str(self.answer, "answer")
        if self.question is not None and not isinstance(self.question, str):
            raise TypeError("`question` must be str or None")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        """Convert the row into a validated plain dictionary.

        Example
        -------
        ```python
        payload = row.to_dict()
        ```
        """
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhaseBTrainRow":
        """Build and validate one row from a JSON-like dictionary.

        Example
        -------
        ```python
        row = PhaseBTrainRow.from_dict(payload)
        ```
        """
        required = [
            "sample_id",
            "dataset",
            "split",
            "prompt_text",
            "target_text",
            "answer",
        ]
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"Missing required keys: {missing}")

        row = cls(
            sample_id=str(payload["sample_id"]),
            dataset=str(payload["dataset"]),
            split=str(payload["split"]),
            prompt_text=str(payload["prompt_text"]),
            target_text=str(payload["target_text"]),
            answer=str(payload["answer"]),
            question=payload.get("question"),
            metadata=dict(payload.get("metadata", {})),
        )
        row.validate()
        return row


def _validate_non_empty_str(value: Any, field_name: str) -> None:
    """Validate that a field is a non-empty string after trimming.

    Example
    -------
    ```python
    _validate_non_empty_str("strategyqa", "dataset")
    ```
    """
    # 保持校验严格，尽早在数据层失败，而不是在 tokenizer 或训练时才暴露。
    if not isinstance(value, str):
        raise TypeError(f"`{field_name}` must be str, got {type(value)!r}")
    if value.strip() == "":
        raise ValueError(f"`{field_name}` must be a non-empty string")
