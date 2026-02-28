"""Evaluator for Phase A prediction files.

This module converts raw model generations into robust correctness metrics using:
- dataset-specific extraction,
- dataset-aware normalization/equivalence logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .answer_extraction import answers_equivalent, extract_answer, normalize_gold_answer
from .contracts import PredictionRecord, ScoredPrediction

EVALUATOR_VERSION = "1.1.0"


@dataclass(slots=True)
class EvalSummary:
    """Aggregate metrics for one evaluation run."""

    n_total: int
    n_correct: int
    n_parse_error: int
    n_parseable: int
    accuracy: float
    parse_error_rate: float
    accuracy_parseable: float
    by_dataset: dict[str, dict[str, Any]]
    evaluator_version: str = EVALUATOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert aggregate metrics into a JSON-serializable dictionary.

        Example
        -------
        ```python
        payload = summary.to_dict()
        ```
        """
        return {
            "n_total": self.n_total,
            "n_correct": self.n_correct,
            "n_parse_error": self.n_parse_error,
            "n_parseable": self.n_parseable,
            "accuracy": self.accuracy,
            "parse_error_rate": self.parse_error_rate,
            "accuracy_parseable": self.accuracy_parseable,
            "by_dataset": self.by_dataset,
            "evaluator_version": self.evaluator_version,
        }


def score_prediction(record: PredictionRecord) -> ScoredPrediction:
    """Score one prediction record."""
    record.validate()

    extracted = extract_answer(raw_text=record.raw_prediction, dataset=record.dataset)
    gold_norm = normalize_gold_answer(record.gold_answer, dataset=record.dataset)
    is_correct = answers_equivalent(
        pred=extracted.text,
        gold=record.gold_answer,
        dataset=record.dataset,
    )

    scored = ScoredPrediction(
        sample_id=record.sample_id,
        dataset=record.dataset,
        split=record.split,
        raw_prediction=record.raw_prediction,
        extracted_prediction=extracted.text,
        normalized_gold=gold_norm,
        is_correct=is_correct,
        parse_error=extracted.parse_error,
        extraction_method=extracted.method,
        question=record.question,
        metadata=record.metadata,
    )
    scored.validate()
    return scored


def evaluate_predictions(records: list[PredictionRecord]) -> tuple[list[ScoredPrediction], EvalSummary]:
    """Score a list of predictions and compute aggregate metrics."""
    scored_rows: list[ScoredPrediction] = [score_prediction(rec) for rec in records]

    n_total = len(scored_rows)
    n_correct = sum(1 for row in scored_rows if row.is_correct)
    n_parse_error = sum(1 for row in scored_rows if row.parse_error)
    n_parseable = n_total - n_parse_error
    n_correct_parseable = sum(
        1 for row in scored_rows if (not row.parse_error and row.is_correct)
    )

    by_dataset: dict[str, dict[str, Any]] = {}
    for row in scored_rows:
        d = row.dataset
        if d not in by_dataset:
            by_dataset[d] = {
                "n": 0,
                "n_correct": 0,
                "n_parse_error": 0,
                "n_parseable": 0,
                "n_correct_parseable": 0,
            }
        by_dataset[d]["n"] += 1
        by_dataset[d]["n_correct"] += 1 if row.is_correct else 0
        by_dataset[d]["n_parse_error"] += 1 if row.parse_error else 0
        if not row.parse_error:
            by_dataset[d]["n_parseable"] += 1
            by_dataset[d]["n_correct_parseable"] += 1 if row.is_correct else 0

    for d, stat in by_dataset.items():
        n = stat["n"]
        stat["accuracy"] = (stat["n_correct"] / n) if n else 0.0
        stat["parse_error_rate"] = (stat["n_parse_error"] / n) if n else 0.0
        stat["accuracy_parseable"] = (
            stat["n_correct_parseable"] / stat["n_parseable"]
            if stat["n_parseable"]
            else 0.0
        )
        by_dataset[d] = stat

    summary = EvalSummary(
        n_total=n_total,
        n_correct=n_correct,
        n_parse_error=n_parse_error,
        n_parseable=n_parseable,
        accuracy=(n_correct / n_total) if n_total else 0.0,
        parse_error_rate=(n_parse_error / n_total) if n_total else 0.0,
        accuracy_parseable=(
            n_correct_parseable / n_parseable if n_parseable else 0.0
        ),
        by_dataset=by_dataset,
    )
    return scored_rows, summary
