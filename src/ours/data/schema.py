"""Canonical data schema used across all datasets in this project.

Why this file exists
--------------------
Different datasets use different field names and structures.
If we do not normalize them early, training and evaluation code will become
dataset-specific and fragile.

This module defines one unified sample shape:
    id, dataset, question, answer, cot, metadata

All dataset loaders must output this format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CanonicalSample:
    """A normalized data sample used by all training/evaluation pipelines.

    Attributes
    ----------
    id:
        Stable unique identifier inside this project.
    dataset:
        Dataset name (for example: ``gsm8k``).
    question:
        Input question/prompt string.
    answer:
        Expected target answer string.
    cot:
        Optional chain-of-thought style reasoning text.
    metadata:
        Extra dataset-specific fields kept for debugging and analysis.
    """

    id: str
    dataset: str
    question: str
    answer: str
    cot: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate the object and raise a clear exception if invalid."""
        _validate_non_empty_str(self.id, "id")
        _validate_non_empty_str(self.dataset, "dataset")
        _validate_non_empty_str(self.question, "question")
        _validate_non_empty_str(self.answer, "answer")
        if self.cot is not None and not isinstance(self.cot, str):
            raise TypeError(f"`cot` must be str or None, got {type(self.cot)!r}")
        if not isinstance(self.metadata, dict):
            raise TypeError(
                f"`metadata` must be a dict[str, Any], got {type(self.metadata)!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalSample":
        """Create and validate a ``CanonicalSample`` from a dict."""
        required = ["id", "dataset", "question", "answer"]
        missing = [k for k in required if k not in payload]
        if missing:
            raise KeyError(f"Missing required keys: {missing}")

        sample = cls(
            id=str(payload["id"]),
            dataset=str(payload["dataset"]),
            question=str(payload["question"]),
            answer=str(payload["answer"]),
            cot=payload.get("cot"),
            metadata=dict(payload.get("metadata", {})),
        )
        sample.validate()
        return sample


def ensure_canonical_samples(
    samples: list[CanonicalSample | dict[str, Any]], source_name: str = "unknown"
) -> list[CanonicalSample]:
    """Validate a list of samples and coerce dict entries to ``CanonicalSample``.

    Parameters
    ----------
    samples:
        Input list containing either ``CanonicalSample`` objects or dictionaries.
    source_name:
        Friendly label used in error messages to speed up debugging.
    """
    canonical: list[CanonicalSample] = []
    for idx, item in enumerate(samples):
        try:
            if isinstance(item, CanonicalSample):
                sample = item
                sample.validate()
            elif isinstance(item, dict):
                sample = CanonicalSample.from_dict(item)
            else:
                raise TypeError(
                    f"Expected CanonicalSample or dict, got {type(item)!r}"
                )
            canonical.append(sample)
        except Exception as exc:  # noqa: BLE001 - explicit debug context is useful
            raise ValueError(
                f"Invalid sample at index={idx} from source={source_name}: {exc}"
            ) from exc
    return canonical


def _validate_non_empty_str(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"`{field_name}` must be str, got {type(value)!r}")
    if value.strip() == "":
        raise ValueError(f"`{field_name}` must be a non-empty string")

