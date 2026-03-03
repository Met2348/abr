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
        """Validate the prompt-template fields before use or persistence.

        Example
        -------
        ```python
        spec.validate()
        ```
        """
        _validate_non_empty_str(self.template_id, "template_id")
        _validate_non_empty_str(self.template_version, "template_version")
        _validate_non_empty_str(self.description, "description")
        _validate_non_empty_str(self.system_prompt, "system_prompt")
        _validate_non_empty_str(self.user_prefix, "user_prefix")
        # `answer_prefix` can be empty (for direct-answer templates).
        _validate_str(self.answer_prefix, "answer_prefix")

    def to_dict(self) -> dict[str, Any]:
        """Convert the template spec into a validated plain dictionary."""
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class PreparedSample:
    """One model-ready training/evaluation record for Phase A.

    Example
    -------
    ```python
    prepared = PreparedSample(
        sample_id="id-1",
        dataset="strategyqa",
        split="validation",
        question="Is the sky blue?",
        answer="yes",
        cot=None,
        target_style="answer_only",
        template_id="qa_direct",
        template_version="1.0.0",
        prompt_text="...",
        target_text="yes",
    )
    ```
    """

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
    # target_text 是 Phase A 到 Phase B/C 的监督桥梁字段：
    # 后续 SFT、value data 构建都会直接读取它。
    target_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate a prepared sample before writing or consuming it."""
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
        """Convert the prepared sample into a validated plain dictionary."""
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class PredictionRecord:
    """One prediction row for evaluator input/output.

    Example
    -------
    ```python
    record = PredictionRecord(
        sample_id="id-1",
        dataset="strategyqa",
        split="validation",
        raw_prediction="yes",
        gold_answer="yes",
    )
    ```
    """

    sample_id: str
    dataset: str
    split: str
    raw_prediction: str
    gold_answer: str
    question: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate prediction fields before scoring or persistence."""
        _validate_non_empty_str(self.sample_id, "sample_id")
        _validate_non_empty_str(self.dataset, "dataset")
        _validate_non_empty_str(self.split, "split")
        # Empty prediction is a valid inference outcome (parse error case),
        # so we only require type=str here.
        _validate_str(self.raw_prediction, "raw_prediction")
        _validate_non_empty_str(self.gold_answer, "gold_answer")
        if self.question is not None and not isinstance(self.question, str):
            raise TypeError("`question` must be str or None")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        """Convert the prediction row into a validated plain dictionary."""
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class ScoredPrediction:
    """Prediction row enriched by extraction and correctness judgment.

    Example
    -------
    ```python
    scored = ScoredPrediction(
        sample_id="id-1",
        dataset="strategyqa",
        split="validation",
        raw_prediction="yes",
        extracted_prediction="yes",
        normalized_gold="yes",
        is_correct=True,
        parse_error=False,
        extraction_method="strategyqa_yes_no",
    )
    ```
    """

    sample_id: str
    dataset: str
    split: str
    raw_prediction: str
    extracted_prediction: str
    normalized_gold: str
    is_correct: bool
    # parse_error=True 表示“答案提取失败或不可信”，并不等同于模型一定答错；
    # 该标志会影响 parseable 子集指标。
    parse_error: bool
    extraction_method: str
    question: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate scored-prediction fields before writing artifacts."""
        _validate_non_empty_str(self.sample_id, "sample_id")
        _validate_non_empty_str(self.dataset, "dataset")
        _validate_non_empty_str(self.split, "split")
        # Keep empty raw prediction representable for audit/debug visibility.
        _validate_str(self.raw_prediction, "raw_prediction")
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
        """Convert the scored row into a validated plain dictionary."""
        self.validate()
        return asdict(self)


def _validate_non_empty_str(value: Any, field_name: str) -> None:
    """Validate that a field is a non-empty string after trimming."""
    if not isinstance(value, str):
        raise TypeError(f"`{field_name}` must be str, got {type(value)!r}")
    if value.strip() == "":
        raise ValueError(f"`{field_name}` must be a non-empty string")


def _validate_str(value: Any, field_name: str) -> None:
    """Validate that a field is a string, allowing empty content."""
    if not isinstance(value, str):
        raise TypeError(f"`{field_name}` must be str, got {type(value)!r}")
