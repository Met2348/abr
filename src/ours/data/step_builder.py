"""Step-level reasoning builder for the OURS research pipeline.

Beginner mental model
---------------------
Raw dataset samples are usually organized as:
- question
- optional chain-of-thought text (cot)
- final answer

For BCR/ABR style research, we often need *step-level* structure.
This module converts one canonical sample into an ordered list of steps.

Why this module exists
----------------------
- Keep all step-splitting logic in one place.
- Keep behavior deterministic for reproducible experiments.
- Keep design configurable so we can run ablations (with/without question step,
  with/without answer terminal step, different split modes).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Sequence

from .schema import CanonicalSample

# Literal types make allowed values explicit for static checkers and readers.
StepRole = Literal["question", "reasoning", "answer"]
SplitMode = Literal["auto", "newline", "sentence"]


@dataclass(slots=True)
class StepBuildConfig:
    """Configuration controlling how step sequences are built.

    Attributes
    ----------
    split_mode:
        - "newline": split CoT by lines
        - "sentence": split CoT by sentence punctuation
        - "auto": choose a reasonable split rule from data shape
    include_question_as_step0:
        If True, prepend question as step index 0.
    include_final_answer_as_terminal_step:
        If True, append final answer as the terminal step.
    normalize_whitespace:
        Collapse repeated spaces/tabs/newlines inside each fragment.
    strip_list_markers:
        Remove common list prefixes ("-", "1.", "(a)") after splitting.
    min_fragment_chars:
        Minimum non-space character length for a reasoning fragment.
        Shorter fragments are dropped as likely noise.
    """

    split_mode: SplitMode = "auto"
    include_question_as_step0: bool = True
    include_final_answer_as_terminal_step: bool = True
    normalize_whitespace: bool = True
    strip_list_markers: bool = True
    min_fragment_chars: int = 1

    def validate(self) -> None:
        """Validate config values and fail early with clear errors."""
        if self.split_mode not in {"auto", "newline", "sentence"}:
            raise ValueError(
                "`split_mode` must be one of {'auto', 'newline', 'sentence'}"
            )
        if self.min_fragment_chars < 1:
            raise ValueError("`min_fragment_chars` must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable config payload for manifests/caching."""
        self.validate()
        return asdict(self)

    def stable_signature(self) -> str:
        """Return stable hash for reproducibility checks.

        We hash sorted JSON to guarantee the same config => same signature.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(slots=True)
class ReasoningStep:
    """One atomic step in a step sequence."""

    step_id: str
    sample_id: str
    dataset: str
    index: int
    role: StepRole
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _validate_non_empty_str(self.step_id, "step_id")
        _validate_non_empty_str(self.sample_id, "sample_id")
        _validate_non_empty_str(self.dataset, "dataset")
        if not isinstance(self.index, int) or self.index < 0:
            raise ValueError("`index` must be a non-negative int")
        if self.role not in {"question", "reasoning", "answer"}:
            raise ValueError(
                "`role` must be one of {'question', 'reasoning', 'answer'}"
            )
        _validate_non_empty_str(self.text, "text")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class StepSequence:
    """All steps derived from one canonical sample.

    This object is the natural output of step preprocessing.
    """

    sample_id: str
    dataset: str
    steps: list[ReasoningStep]
    has_cot: bool
    config_signature: str

    def validate(self) -> None:
        _validate_non_empty_str(self.sample_id, "sample_id")
        _validate_non_empty_str(self.dataset, "dataset")
        if not isinstance(self.steps, list):
            raise TypeError("`steps` must be list[ReasoningStep]")
        for idx, step in enumerate(self.steps):
            if not isinstance(step, ReasoningStep):
                raise TypeError(f"steps[{idx}] must be ReasoningStep")
            step.validate()
            if step.index != idx:
                raise ValueError(
                    f"Step index mismatch in sample={self.sample_id!r}: "
                    f"expected {idx}, got {step.index}"
                )
        if not isinstance(self.has_cot, bool):
            raise TypeError("`has_cot` must be bool")
        _validate_non_empty_str(self.config_signature, "config_signature")

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "num_steps": self.num_steps,
            "has_cot": self.has_cot,
            "config_signature": self.config_signature,
            "steps": [step.to_dict() for step in self.steps],
        }


def build_step_sequence(
    sample: CanonicalSample,
    config: StepBuildConfig | None = None,
) -> StepSequence:
    """Convert one canonical sample into one deterministic step sequence.

    Determinism rules in this function:
    - no randomness
    - stable step ordering
    - stable step IDs derived from sample_id + index + role
    """
    sample.validate()
    cfg = config or StepBuildConfig()
    cfg.validate()

    steps: list[ReasoningStep] = []
    next_index = 0

    # Optional step 0: include the problem statement as an explicit state.
    if cfg.include_question_as_step0:
        question_text = _normalize_fragment(sample.question, cfg)
        if question_text:
            steps.append(
                _make_step(
                    sample=sample,
                    index=next_index,
                    role="question",
                    text=question_text,
                    metadata={"source_field": "question"},
                )
            )
            next_index += 1

    reasoning_fragments = split_reasoning_text(
        cot_text=sample.cot,
        dataset=sample.dataset,
        config=cfg,
    )

    for fragment in reasoning_fragments:
        steps.append(
            _make_step(
                sample=sample,
                index=next_index,
                role="reasoning",
                text=fragment,
                metadata={"source_field": "cot"},
            )
        )
        next_index += 1

    # Optional terminal step: include final answer as explicit endpoint.
    if cfg.include_final_answer_as_terminal_step:
        answer_text = _normalize_fragment(sample.answer, cfg)
        if answer_text:
            steps.append(
                _make_step(
                    sample=sample,
                    index=next_index,
                    role="answer",
                    text=answer_text,
                    metadata={"source_field": "answer"},
                )
            )
            next_index += 1

    if not steps:
        raise ValueError(
            "No steps were produced. Check config: if both question/answer are "
            "excluded and cot is empty, sequence becomes empty."
        )

    sequence = StepSequence(
        sample_id=sample.id,
        dataset=sample.dataset,
        steps=steps,
        has_cot=bool(sample.cot and sample.cot.strip()),
        config_signature=cfg.stable_signature(),
    )
    sequence.validate()
    return sequence


def build_step_sequences(
    samples: Sequence[CanonicalSample],
    config: StepBuildConfig | None = None,
    source_name: str = "unknown",
) -> list[StepSequence]:
    """Build step sequences for a batch of canonical samples.

    Parameters
    ----------
    samples:
        Input canonical samples.
    config:
        Optional `StepBuildConfig`. Uses defaults when omitted.
    source_name:
        Human-readable source label used in error messages.
    """
    cfg = config or StepBuildConfig()
    cfg.validate()

    outputs: list[StepSequence] = []
    for idx, sample in enumerate(samples):
        try:
            outputs.append(build_step_sequence(sample=sample, config=cfg))
        except Exception as exc:  # noqa: BLE001 - caller needs context-rich errors
            raise ValueError(
                f"Failed building step sequence from source={source_name}, "
                f"sample_index={idx}, sample_id={getattr(sample, 'id', '<unknown>')}: {exc}"
            ) from exc
    return outputs


def iter_flat_steps(sequences: Iterable[StepSequence]) -> Iterable[ReasoningStep]:
    """Yield every `ReasoningStep` across all sequences in order."""
    for sequence in sequences:
        sequence.validate()
        for step in sequence.steps:
            yield step


def split_reasoning_text(
    cot_text: str | None,
    dataset: str,
    config: StepBuildConfig,
) -> list[str]:
    """Split chain-of-thought text into clean step fragments.

    Important: this function only works on `cot` text. Question/answer handling
    is controlled by `build_step_sequence`.
    """
    if cot_text is None or cot_text.strip() == "":
        return []

    mode = config.split_mode
    if mode == "auto":
        mode = _choose_split_mode(cot_text=cot_text, dataset=dataset)

    if mode == "newline":
        fragments = _split_by_newline(cot_text)
    elif mode == "sentence":
        fragments = _split_by_sentence(cot_text)
    else:
        # Defensive branch for future maintenance.
        raise ValueError(f"Unsupported split mode: {mode}")

    cleaned: list[str] = []
    for fragment in fragments:
        text = _normalize_fragment(fragment, config)
        if config.strip_list_markers:
            text = _strip_list_prefix(text)
        if len(text.strip()) < config.min_fragment_chars:
            continue
        if text.strip() == "":
            continue
        cleaned.append(text)

    return cleaned


def _choose_split_mode(cot_text: str, dataset: str) -> SplitMode:
    """Choose a default split strategy based on text shape and dataset.

    Heuristic goals:
    - If explicit line structure exists, respect it.
    - Otherwise use sentence splitting.
    - Keep behavior deterministic and easy to reason about.
    """
    text = cot_text.strip()
    dataset = dataset.strip().lower()

    # Dataset hint: strategyqa decomposition and proofwriter proofs are often
    # line/bullet structured, so newline splitting keeps intended granularity.
    if dataset in {"strategyqa", "proofwriter"}:
        return "newline"

    # If there are multiple lines, newline split usually preserves author intent.
    if "\n" in text:
        return "newline"

    # Very long single-line explanations are usually sentence-like.
    return "sentence"


def _split_by_newline(text: str) -> list[str]:
    """Split text by physical lines.

    Empty lines are expected in many datasets and are filtered later.
    """
    return text.splitlines()


def _split_by_sentence(text: str) -> list[str]:
    """Split by sentence boundaries using a conservative regex.

    Notes for beginners
    -------------------
    Regex sentence splitting is imperfect for abbreviations/math notation.
    We keep it simple and deterministic; improvements can be added later.
    """
    normalized = text.replace("\n", " ").strip()
    if normalized == "":
        return []

    # Split after terminal punctuation followed by whitespace.
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return parts


def _normalize_fragment(text: str, config: StepBuildConfig) -> str:
    """Normalize one fragment according to config."""
    value = text.strip()
    if config.normalize_whitespace:
        value = re.sub(r"\s+", " ", value)
    return value.strip()


def _strip_list_prefix(text: str) -> str:
    """Remove common list-style prefixes from a fragment.

    Examples removed:
    - "- reason"
    - "1. reason"
    - "(a) reason"
    """
    # Keep this intentionally small and readable; avoid overfitting regex.
    patterns = [
        r"^[-*]\s+",              # bullet prefix
        r"^\d+[\.)]\s+",         # numeric list: 1. / 2)
        r"^\([a-zA-Z]\)\s+",     # alpha list: (a) / (B)
    ]
    output = text
    for pattern in patterns:
        output = re.sub(pattern, "", output)
    return output.strip()


def _make_step(
    sample: CanonicalSample,
    index: int,
    role: StepRole,
    text: str,
    metadata: dict[str, Any],
) -> ReasoningStep:
    step = ReasoningStep(
        step_id=_build_step_id(sample.id, index, role),
        sample_id=sample.id,
        dataset=sample.dataset,
        index=index,
        role=role,
        text=text,
        metadata=metadata,
    )
    step.validate()
    return step


def _build_step_id(sample_id: str, index: int, role: StepRole) -> str:
    """Create deterministic step ID.

    We avoid random UUIDs here because deterministic IDs are easier for:
    - diffing artifacts
    - cache reuse
    - error tracing
    """
    return f"{sample_id}::step::{index:04d}::{role}"


def _validate_non_empty_str(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"`{field_name}` must be str, got {type(value)!r}")
    if value.strip() == "":
        raise ValueError(f"`{field_name}` must be a non-empty string")
