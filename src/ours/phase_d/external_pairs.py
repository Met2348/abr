"""Canonical schema and loaders for Phase D external pair artifacts.

This module provides a strict in-memory contract for external pair rows used by
Phase D4. The goal is to keep source diversity (R-PRM, PRMBench preview,
step-labeled conversions, etc.) while making C2 integration deterministic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExternalPairRecord:
    """One normalized external preference pair for ranking supervision."""

    pair_id: str
    source_tag: str
    domain_tag: str
    prompt_text: str
    chosen_text: str
    rejected_text: str
    pair_confidence: float
    quality_flags: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate field types and numeric ranges."""
        for name in (
            "pair_id",
            "source_tag",
            "domain_tag",
            "prompt_text",
            "chosen_text",
            "rejected_text",
        ):
            _validate_non_empty_str(getattr(self, name), name)
        if self.chosen_text.strip() == self.rejected_text.strip():
            raise ValueError("`chosen_text` and `rejected_text` must be different")
        if not isinstance(self.pair_confidence, (int, float)):
            raise TypeError("`pair_confidence` must be float in [0, 1]")
        confidence = float(self.pair_confidence)
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("`pair_confidence` must be in [0, 1]")
        if not isinstance(self.quality_flags, dict):
            raise TypeError("`quality_flags` must be dict[str, Any]")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        """Return a validated JSON-serializable dictionary."""
        self.validate()
        return asdict(self)

    def chosen_input_text(self) -> str:
        """Return full chosen text input for frozen-backbone encoding."""
        return f"{self.prompt_text}{self.chosen_text}"

    def rejected_input_text(self) -> str:
        """Return full rejected text input for frozen-backbone encoding."""
        return f"{self.prompt_text}{self.rejected_text}"


def load_external_pair_jsonl(
    path: Path,
    *,
    max_samples: int | None = None,
    min_confidence: float = 0.0,
    allowed_sources: set[str] | None = None,
    allowed_domains: set[str] | None = None,
) -> tuple[list[ExternalPairRecord], dict[str, Any]]:
    """Load canonical external pairs from one JSONL artifact.

    Parameters
    ----------
    path:
        Path to a JSONL file where each row follows `ExternalPairRecord`.
    max_samples:
        Optional cap after deterministic sort by `pair_id`.
    min_confidence:
        Drop rows with confidence lower than this threshold.
    allowed_sources / allowed_domains:
        Optional inclusion filters.
    """
    if not path.exists():
        raise FileNotFoundError(f"External pair JSONL not found: {path}")
    if not (0.0 <= float(min_confidence) <= 1.0):
        raise ValueError("`min_confidence` must be in [0, 1]")

    rows: list[ExternalPairRecord] = []
    line_no = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line_no += 1
            text = raw.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_no} must be a JSON object")
            record = ExternalPairRecord(
                pair_id=str(payload.get("pair_id", "")).strip(),
                source_tag=str(payload.get("source_tag", "")).strip(),
                domain_tag=str(payload.get("domain_tag", "")).strip(),
                prompt_text=str(payload.get("prompt_text", "")),
                chosen_text=str(payload.get("chosen_text", "")),
                rejected_text=str(payload.get("rejected_text", "")),
                pair_confidence=float(payload.get("pair_confidence", 0.0)),
                quality_flags=dict(payload.get("quality_flags", {}) or {}),
                metadata=dict(payload.get("metadata", {}) or {}),
            )
            record.validate()
            if float(record.pair_confidence) < float(min_confidence):
                continue
            if allowed_sources is not None and record.source_tag not in allowed_sources:
                continue
            if allowed_domains is not None and record.domain_tag not in allowed_domains:
                continue
            rows.append(record)

    rows.sort(key=lambda item: item.pair_id)
    if max_samples is not None:
        rows = rows[: max(0, int(max_samples))]
    return rows, summarize_external_pairs(rows)


def summarize_external_pairs(rows: list[ExternalPairRecord]) -> dict[str, Any]:
    """Return lightweight summary stats for external pair rows."""
    by_source: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    by_pair_build_mode: dict[str, int] = {}
    by_pair_semantics: dict[str, int] = {}
    confidence_sum = 0.0
    for row in rows:
        by_source[row.source_tag] = by_source.get(row.source_tag, 0) + 1
        by_domain[row.domain_tag] = by_domain.get(row.domain_tag, 0) + 1
        pair_build_mode = str((row.metadata or {}).get("pair_build_mode", "unspecified")).strip() or "unspecified"
        pair_semantics = str((row.metadata or {}).get("pair_semantics", "unspecified")).strip() or "unspecified"
        by_pair_build_mode[pair_build_mode] = by_pair_build_mode.get(pair_build_mode, 0) + 1
        by_pair_semantics[pair_semantics] = by_pair_semantics.get(pair_semantics, 0) + 1
        confidence_sum += float(row.pair_confidence)
    count = len(rows)
    return {
        "num_pairs": int(count),
        "mean_pair_confidence": (float(confidence_sum / count) if count > 0 else 0.0),
        "by_source": dict(sorted(by_source.items())),
        "by_domain": dict(sorted(by_domain.items())),
        "by_pair_build_mode": dict(sorted(by_pair_build_mode.items())),
        "by_pair_semantics": dict(sorted(by_pair_semantics.items())),
    }


def _validate_non_empty_str(value: Any, name: str) -> None:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"`{name}` must be a non-empty string")
