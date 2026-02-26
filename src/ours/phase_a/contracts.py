"""Core dataclasses for Phase A baseline pipeline.

Design goal
-----------
Keep all stage-A records explicit and typed so that:
- the pipeline is easy to read for beginners,
- artifacts are stable and self-describing,
- later stages (B/C/D) can reuse these structures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TargetStyle = Literal["answer_only", "cot_then_answer"]


@dataclass(slots=True)
class PromptTemplateSpec:
    """A versioned prompt-template definition.

    Why version this?
    -----------------
    Small wording changes can alter model behavior a lot.
    We store template id+version in artifacts so experiments are reproducible.
    """

    template_id: str
    template_version: str
    description: str
    system_prompt: str
    user_prefix: str
    answer_prefix: str

    def validate(self) -> None:
        _validate_non_empty_str(self.template_id, "template_id")
        _validate_non_empty_str(self.template_version, "template_version")
        _validate_non_empty_str(self.description, "description")
        _validate_non_empty_str(self.system_prompt, "system_prompt")
        _validate_non_empty_str(self.user_prefix, "user_prefix")
        # `answer_prefix` can be empty (for direct-answer templates).
        _validate_str(self.answer_prefix, "answer_prefix")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class PreparedSample:
    """One model-ready training/evaluation record for Phase A."""

    sample_id: str
    dataset: str
    split: str
    question: str
    answer: str
    cot: str | None
    target_style: TargetStyle
    template_id: str
    template_version: str
    prompt_text: str
    target_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _validate_non_empty_str(self.sample_id, "sample_id")
        _validate_non_empty_str(self.dataset, "dataset")
        _validate_non_empty_str(self.split, "split")
        _validate_non_empty_str(self.question, "question")
        _validate_non_empty_str(self.answer, "answer")
        if self.cot is not None and not isinstance(self.cot, str):
            raise TypeError("`cot` must be str or None")
        if self.target_style not in {"answer_only", "cot_then_answer"}:
            raise ValueError(
                "`target_style` must be one of {'answer_only', 'cot_then_answer'}"
            )
        _validate_non_empty_str(self.template_id, "template_id")
        _validate_non_empty_str(self.template_version, "template_version")
        _validate_non_empty_str(self.prompt_text, "prompt_text")
        _validate_non_empty_str(self.target_text, "target_text")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class PredictionRecord:
    """One prediction row for evaluator input/output."""

    sample_id: str
    dataset: str
    split: str
    raw_prediction: str
    gold_answer: str
    question: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _validate_non_empty_str(self.sample_id, "sample_id")
        _validate_non_empty_str(self.dataset, "dataset")
        _validate_non_empty_str(self.split, "split")
        _validate_non_empty_str(self.raw_prediction, "raw_prediction")
        _validate_non_empty_str(self.gold_answer, "gold_answer")
        if self.question is not None and not isinstance(self.question, str):
            raise TypeError("`question` must be str or None")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class ScoredPrediction:
    """Prediction row enriched by extraction + correctness judgment."""

    sample_id: str
    dataset: str
    split: str
    raw_prediction: str
    extracted_prediction: str
    normalized_gold: str
    is_correct: bool
    parse_error: bool
    extraction_method: str
    question: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _validate_non_empty_str(self.sample_id, "sample_id")
        _validate_non_empty_str(self.dataset, "dataset")
        _validate_non_empty_str(self.split, "split")
        _validate_non_empty_str(self.raw_prediction, "raw_prediction")
        if not isinstance(self.extracted_prediction, str):
            raise TypeError("`extracted_prediction` must be str")
        if not isinstance(self.normalized_gold, str):
            raise TypeError("`normalized_gold` must be str")
        if not isinstance(self.is_correct, bool):
            raise TypeError("`is_correct` must be bool")
        if not isinstance(self.parse_error, bool):
            raise TypeError("`parse_error` must be bool")
        _validate_non_empty_str(self.extraction_method, "extraction_method")
        if self.question is not None and not isinstance(self.question, str):
            raise TypeError("`question` must be str or None")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _validate_non_empty_str(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"`{field_name}` must be str, got {type(value)!r}")
    if value.strip() == "":
        raise ValueError(f"`{field_name}` must be a non-empty string")


def _validate_str(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"`{field_name}` must be str, got {type(value)!r}")
