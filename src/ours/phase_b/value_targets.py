"""Phase C prefix and rollout-target contracts built on top of Phase B rows.

Why this file exists
--------------------
Phase B training consumes plain `(prompt_text, target_text)` rows. Phase C needs
much richer artifacts:
- deterministic step prefixes,
- rollout-based empirical success targets,
- traceable IDs and metadata for later value-head training.

This module centralizes those contracts and the deterministic conversion from a
validated `PhaseBTrainRow` into:
- one `StepSequence`,
- many `PrefixArtifact` rows,
- optional `RolloutTargetRecord` summaries later on.

Keeping this logic out of the training script matters because silent data-shape
bugs are expensive at this stage. The rule here is simple: if a prefix cannot be
described clearly, it should not reach training.

Interaction with other files
----------------------------
- `src/ours/phase_b/contracts.py`: source row contract (`PhaseBTrainRow`)
- `src/ours/data/step_builder.py`: deterministic stepization
- `src/ours/phase_b/corruptions.py`: corruption artifacts built from prefixes
- `scripts/phase_b_prepare_value_data.py`: orchestration script for Phase C C0/C1

Example
-------
```python
from ours.phase_b.contracts import PhaseBTrainRow

row = PhaseBTrainRow(
    sample_id="strategyqa:1",
    dataset="strategyqa",
    split="train",
    prompt_text="Question: Is the sky blue?\nAnswer:",
    target_text="Blue things in the sky are usually visible.\nFinal answer: yes",
    answer="yes",
    question="Is the sky blue?",
)

step_sequence, build_meta = build_step_sequence_from_phase_b_row(row)
prefixes = build_prefix_artifacts(row=row, step_sequence=step_sequence)
assert prefixes[0].sample_id == "strategyqa:1"
```
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from ours.data.schema import CanonicalSample
from ours.data.step_builder import (
    ReasoningStep,
    StepBuildConfig,
    StepSequence,
    build_step_sequence,
)

from .contracts import PhaseBTrainRow
from .supervision import split_reasoning_and_answer


@dataclass(slots=True)
class PrefixBuildConfig:
    """Configuration for converting step sequences into Phase C prefixes.

    Attributes
    ----------
    include_question_only_prefix:
        If true, emit a prefix whose generated target text is still empty. This
        corresponds to the state right after reading the question and before any
        reasoning step is emitted.
    require_reasoning_steps:
        If true, rows without at least one reasoning step are rejected. This is
        the safe default for early value-head work because answer-only rows do
        not provide a meaningful process trajectory.
    fallback_question_to_prompt_text:
        If a Phase B row does not carry a clean `question` field, use
        `prompt_text` as a fallback for step-building. This keeps the pipeline
        usable on older artifacts while still recording the fallback.
    """

    include_question_only_prefix: bool = True
    require_reasoning_steps: bool = True
    fallback_question_to_prompt_text: bool = True

    def validate(self) -> None:
        """Validate configuration values before artifact construction."""
        if not isinstance(self.include_question_only_prefix, bool):
            raise TypeError("`include_question_only_prefix` must be bool")
        if not isinstance(self.require_reasoning_steps, bool):
            raise TypeError("`require_reasoning_steps` must be bool")
        if not isinstance(self.fallback_question_to_prompt_text, bool):
            raise TypeError("`fallback_question_to_prompt_text` must be bool")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration payload."""
        self.validate()
        return asdict(self)

    def stable_signature(self) -> str:
        """Return a short stable signature used in artifact IDs/manifests."""
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(slots=True)
class PrefixArtifact:
    """One deterministic Phase C prefix record.

    A prefix artifact is the core unit for value-head training. It describes one
    partial reasoning state:
    - which sample it belongs to,
    - which step boundary it stops at,
    - what prompt text the model sees,
    - what target text has been generated so far,
    - what the gold answer is.

    Important
    ---------
    `prefix_target_text` intentionally excludes the question text because the
    question already lives in `prompt_text`. This mirrors how Phase B SFT rows
    concatenate `prompt_text + target_text`.
    """

    prefix_id: str
    sample_id: str
    dataset: str
    split: str
    question: str
    prompt_text: str
    gold_answer: str
    terminal_answer_text: str
    full_target_text: str
    prefix_target_text: str
    prefix_step_index: int
    current_step_id: str
    current_step_role: str
    current_step_text: str
    source_step_ids: list[str]
    source_step_roles: list[str]
    num_reasoning_steps_seen: int
    num_reasoning_steps_total: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate field types and invariants.

        Example
        -------
        ```python
        prefix.validate()
        ```
        """
        _validate_non_empty_str(self.prefix_id, "prefix_id")
        _validate_non_empty_str(self.sample_id, "sample_id")
        _validate_non_empty_str(self.dataset, "dataset")
        _validate_non_empty_str(self.split, "split")
        _validate_non_empty_str(self.question, "question")
        _validate_non_empty_str(self.prompt_text, "prompt_text")
        _validate_non_empty_str(self.gold_answer, "gold_answer")
        _validate_non_empty_str(self.terminal_answer_text, "terminal_answer_text")
        _validate_non_empty_str(self.full_target_text, "full_target_text")
        _validate_str(self.prefix_target_text, "prefix_target_text")
        if not isinstance(self.prefix_step_index, int) or self.prefix_step_index < 0:
            raise ValueError("`prefix_step_index` must be a non-negative int")
        _validate_non_empty_str(self.current_step_id, "current_step_id")
        _validate_non_empty_str(self.current_step_role, "current_step_role")
        _validate_non_empty_str(self.current_step_text, "current_step_text")
        _validate_string_list(self.source_step_ids, "source_step_ids")
        _validate_string_list(self.source_step_roles, "source_step_roles")
        if len(self.source_step_ids) != len(self.source_step_roles):
            raise ValueError(
                "`source_step_ids` and `source_step_roles` must have equal length"
            )
        if not isinstance(self.num_reasoning_steps_seen, int) or self.num_reasoning_steps_seen < 0:
            raise ValueError("`num_reasoning_steps_seen` must be a non-negative int")
        if not isinstance(self.num_reasoning_steps_total, int) or self.num_reasoning_steps_total < 0:
            raise ValueError("`num_reasoning_steps_total` must be a non-negative int")
        if self.num_reasoning_steps_seen > self.num_reasoning_steps_total:
            raise ValueError(
                "`num_reasoning_steps_seen` cannot exceed `num_reasoning_steps_total`"
            )
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        """Convert the record into a validated plain dictionary."""
        self.validate()
        return asdict(self)

    def rollout_input_text(self) -> str:
        """Return the exact model input string used for rollout generation.

        Example
        -------
        ```python
        prompt = prefix.rollout_input_text()
        ```
        """
        return f"{self.prompt_text}{self.prefix_target_text}"


@dataclass(slots=True)
class RolloutPredictionRecord:
    """One sampled continuation from one prefix.

    This is the lowest-level debug artifact for rollout target generation. It is
    intentionally verbose so later target mismatches can be audited from disk.
    """

    prefix_id: str
    sample_id: str
    dataset: str
    split: str
    rollout_index: int
    raw_continuation: str
    full_prediction: str
    extracted_prediction: str
    extraction_method: str
    is_correct: bool
    parse_error: bool
    generated_char_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate one rollout prediction record."""
        _validate_non_empty_str(self.prefix_id, "prefix_id")
        _validate_non_empty_str(self.sample_id, "sample_id")
        _validate_non_empty_str(self.dataset, "dataset")
        _validate_non_empty_str(self.split, "split")
        if not isinstance(self.rollout_index, int) or self.rollout_index < 0:
            raise ValueError("`rollout_index` must be a non-negative int")
        _validate_str(self.raw_continuation, "raw_continuation")
        _validate_str(self.full_prediction, "full_prediction")
        _validate_str(self.extracted_prediction, "extracted_prediction")
        _validate_non_empty_str(self.extraction_method, "extraction_method")
        if not isinstance(self.is_correct, bool):
            raise TypeError("`is_correct` must be bool")
        if not isinstance(self.parse_error, bool):
            raise TypeError("`parse_error` must be bool")
        if not isinstance(self.generated_char_count, int) or self.generated_char_count < 0:
            raise ValueError("`generated_char_count` must be a non-negative int")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        """Convert the record into a validated plain dictionary."""
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class RolloutTargetRecord:
    """Aggregate empirical success target for one prefix."""

    prefix_id: str
    sample_id: str
    dataset: str
    split: str
    k_rollouts: int
    n_correct: int
    n_parse_error: int
    success_rate: float
    parseable_rate: float
    mean_generated_char_count: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate one rollout target summary."""
        _validate_non_empty_str(self.prefix_id, "prefix_id")
        _validate_non_empty_str(self.sample_id, "sample_id")
        _validate_non_empty_str(self.dataset, "dataset")
        _validate_non_empty_str(self.split, "split")
        if not isinstance(self.k_rollouts, int) or self.k_rollouts <= 0:
            raise ValueError("`k_rollouts` must be a positive int")
        if not isinstance(self.n_correct, int) or self.n_correct < 0:
            raise ValueError("`n_correct` must be a non-negative int")
        if not isinstance(self.n_parse_error, int) or self.n_parse_error < 0:
            raise ValueError("`n_parse_error` must be a non-negative int")
        if self.n_correct > self.k_rollouts:
            raise ValueError("`n_correct` cannot exceed `k_rollouts`")
        if self.n_parse_error > self.k_rollouts:
            raise ValueError("`n_parse_error` cannot exceed `k_rollouts`")
        _validate_unit_interval(self.success_rate, "success_rate")
        _validate_unit_interval(self.parseable_rate, "parseable_rate")
        if not isinstance(self.mean_generated_char_count, float):
            raise TypeError("`mean_generated_char_count` must be float")
        if self.mean_generated_char_count < 0.0:
            raise ValueError("`mean_generated_char_count` must be >= 0")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        """Convert the record into a validated plain dictionary."""
        self.validate()
        return asdict(self)


def build_step_sequence_from_phase_b_row(
    row: PhaseBTrainRow,
    *,
    step_config: StepBuildConfig | None = None,
    prefix_config: PrefixBuildConfig | None = None,
) -> tuple[StepSequence, dict[str, Any]]:
    """Convert one Phase B training row into a deterministic step sequence.

    Parameters
    ----------
    row:
        Validated prepared Phase B row.
    step_config:
        Configuration for step segmentation.
    prefix_config:
        Configuration controlling fallback and reasoning-step requirements.

    Returns
    -------
    tuple[StepSequence, dict[str, Any]]
        The built sequence and a small metadata payload describing how the row
        was interpreted.
    """
    row.validate()
    cfg = step_config or StepBuildConfig()
    prefix_cfg = prefix_config or PrefixBuildConfig()
    cfg.validate()
    prefix_cfg.validate()

    reasoning_text, answer_target_text, answer_signal_found = split_reasoning_and_answer(
        row.target_text
    )

    question_text = (row.question or "").strip()
    question_fallback_used = False
    if question_text == "":
        if not prefix_cfg.fallback_question_to_prompt_text:
            raise ValueError(
                "Phase C requires `question` text or explicit prompt fallback. "
                f"sample_id={row.sample_id!r}"
            )
        question_text = row.prompt_text.strip()
        question_fallback_used = True

    terminal_answer_text = answer_target_text.strip() or row.answer.strip()
    canonical = CanonicalSample(
        id=row.sample_id,
        dataset=row.dataset,
        question=question_text,
        answer=terminal_answer_text,
        cot=(reasoning_text.strip() or None),
        metadata={
            "split": row.split,
            "prompt_text": row.prompt_text,
            "gold_answer": row.answer,
            "answer_signal_found": bool(answer_signal_found),
            "question_fallback_used": bool(question_fallback_used),
        },
    )
    canonical.validate()

    step_sequence = build_step_sequence(canonical, config=cfg)
    reasoning_steps = [step for step in step_sequence.steps if step.role == "reasoning"]
    if prefix_cfg.require_reasoning_steps and not reasoning_steps:
        raise ValueError(
            "Phase C prefix building requires at least one reasoning step, "
            f"but none were found for sample_id={row.sample_id!r}. "
            "Use a CoT-style prepared artifact or disable the guard explicitly."
        )

    build_meta = {
        "answer_signal_found": bool(answer_signal_found),
        "question_fallback_used": bool(question_fallback_used),
        "terminal_answer_text": terminal_answer_text,
        "reasoning_text": reasoning_text.strip(),
        "num_reasoning_steps": len(reasoning_steps),
        "step_config_signature": cfg.stable_signature(),
        "prefix_config_signature": prefix_cfg.stable_signature(),
    }
    return step_sequence, build_meta


def build_prefix_artifacts(
    *,
    row: PhaseBTrainRow,
    step_sequence: StepSequence,
    build_meta: dict[str, Any] | None = None,
    prefix_config: PrefixBuildConfig | None = None,
) -> list[PrefixArtifact]:
    """Build all deterministic prefixes for one step sequence.

    Prefixes are built over:
    - the optional question-only state,
    - each reasoning-step boundary,
    - but not the terminal answer step.
    """
    row.validate()
    step_sequence.validate()
    prefix_cfg = prefix_config or PrefixBuildConfig()
    prefix_cfg.validate()
    meta = dict(build_meta or {})

    question_steps = [step for step in step_sequence.steps if step.role == "question"]
    reasoning_steps = [step for step in step_sequence.steps if step.role == "reasoning"]
    if prefix_cfg.require_reasoning_steps and not reasoning_steps:
        raise ValueError(
            "Cannot build prefixes without reasoning steps when "
            "`require_reasoning_steps=True`."
        )

    terminal_answer_text = str(meta.get("terminal_answer_text") or row.answer).strip()
    full_target_text = str(row.target_text)
    question_text = str(row.question or row.prompt_text).strip()

    prefixes: list[PrefixArtifact] = []
    step_config_signature = str(meta.get("step_config_signature") or step_sequence.config_signature)
    prefix_config_signature = str(meta.get("prefix_config_signature") or prefix_cfg.stable_signature())

    if prefix_cfg.include_question_only_prefix:
        if question_steps:
            question_step = question_steps[0]
        else:
            question_step = ReasoningStep(
                step_id=_stable_hash("phase_c_question_fallback", row.sample_id, "0")[:16],
                sample_id=row.sample_id,
                dataset=row.dataset,
                index=0,
                role="question",
                text=question_text,
                metadata={"synthetic_question_step": True},
            )
        prefixes.append(
            _make_prefix_artifact(
                row=row,
                question_text=question_text,
                terminal_answer_text=terminal_answer_text,
                full_target_text=full_target_text,
                prefix_target_text="",
                current_step=question_step,
                source_steps=[question_step],
                num_reasoning_steps_seen=0,
                num_reasoning_steps_total=len(reasoning_steps),
                step_config_signature=step_config_signature,
                prefix_config_signature=prefix_config_signature,
                metadata={
                    "prefix_kind": "question_only",
                    "answer_signal_found": bool(meta.get("answer_signal_found", False)),
                    "question_fallback_used": bool(meta.get("question_fallback_used", False)),
                },
            )
        )

    accumulated_reasoning: list[ReasoningStep] = []
    for reasoning_step in reasoning_steps:
        accumulated_reasoning.append(reasoning_step)
        prefix_target_text = "\n".join(step.text for step in accumulated_reasoning).strip()
        source_steps: list[ReasoningStep] = []
        if question_steps:
            source_steps.append(question_steps[0])
        source_steps.extend(accumulated_reasoning)
        prefixes.append(
            _make_prefix_artifact(
                row=row,
                question_text=question_text,
                terminal_answer_text=terminal_answer_text,
                full_target_text=full_target_text,
                prefix_target_text=prefix_target_text,
                current_step=reasoning_step,
                source_steps=source_steps,
                num_reasoning_steps_seen=len(accumulated_reasoning),
                num_reasoning_steps_total=len(reasoning_steps),
                step_config_signature=step_config_signature,
                prefix_config_signature=prefix_config_signature,
                metadata={
                    "prefix_kind": "reasoning_prefix",
                    "answer_signal_found": bool(meta.get("answer_signal_found", False)),
                    "question_fallback_used": bool(meta.get("question_fallback_used", False)),
                },
            )
        )

    validate_unique_prefix_ids(prefixes)
    return prefixes


def summarize_prefix_artifacts(prefixes: list[PrefixArtifact]) -> dict[str, Any]:
    """Return compact summary statistics for prefix artifacts."""
    validate_unique_prefix_ids(prefixes)
    dataset_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    max_reasoning_steps_seen = 0
    question_only_count = 0
    for prefix in prefixes:
        prefix.validate()
        dataset_counts[prefix.dataset] = dataset_counts.get(prefix.dataset, 0) + 1
        split_counts[prefix.split] = split_counts.get(prefix.split, 0) + 1
        max_reasoning_steps_seen = max(
            max_reasoning_steps_seen, int(prefix.num_reasoning_steps_seen)
        )
        if prefix.num_reasoning_steps_seen == 0:
            question_only_count += 1
    return {
        "num_prefixes": len(prefixes),
        "dataset_counts": dataset_counts,
        "split_counts": split_counts,
        "question_only_prefixes": question_only_count,
        "max_reasoning_steps_seen": max_reasoning_steps_seen,
    }


def summarize_rollout_targets(targets: list[RolloutTargetRecord]) -> dict[str, Any]:
    """Return aggregate stats for one rollout-target artifact file."""
    if not targets:
        return {
            "num_prefixes": 0,
            "mean_success_rate": 0.0,
            "mean_parseable_rate": 0.0,
            "mean_generated_char_count": 0.0,
        }
    total_success = 0.0
    total_parseable = 0.0
    total_chars = 0.0
    for target in targets:
        target.validate()
        total_success += float(target.success_rate)
        total_parseable += float(target.parseable_rate)
        total_chars += float(target.mean_generated_char_count)
    n = float(len(targets))
    return {
        "num_prefixes": len(targets),
        "mean_success_rate": total_success / n,
        "mean_parseable_rate": total_parseable / n,
        "mean_generated_char_count": total_chars / n,
    }


def validate_unique_prefix_ids(prefixes: list[PrefixArtifact]) -> None:
    """Fail fast if a prefix artifact list contains duplicate prefix IDs."""
    seen: set[str] = set()
    for idx, prefix in enumerate(prefixes):
        prefix.validate()
        if prefix.prefix_id in seen:
            raise ValueError(
                "Duplicate prefix_id detected at index="
                f"{idx}: {prefix.prefix_id!r}"
            )
        seen.add(prefix.prefix_id)


def _make_prefix_artifact(
    *,
    row: PhaseBTrainRow,
    question_text: str,
    terminal_answer_text: str,
    full_target_text: str,
    prefix_target_text: str,
    current_step: ReasoningStep,
    source_steps: list[ReasoningStep],
    num_reasoning_steps_seen: int,
    num_reasoning_steps_total: int,
    step_config_signature: str,
    prefix_config_signature: str,
    metadata: dict[str, Any],
) -> PrefixArtifact:
    """Build one `PrefixArtifact` in a single canonical place."""
    prefix_id = _stable_hash(
        "phase_c_prefix",
        row.sample_id,
        str(current_step.index),
        current_step.role,
        step_config_signature,
        prefix_config_signature,
    )[:24]
    prefix = PrefixArtifact(
        prefix_id=prefix_id,
        sample_id=row.sample_id,
        dataset=row.dataset,
        split=row.split,
        question=question_text,
        prompt_text=row.prompt_text,
        gold_answer=row.answer,
        terminal_answer_text=terminal_answer_text,
        full_target_text=full_target_text,
        prefix_target_text=prefix_target_text,
        prefix_step_index=int(current_step.index),
        current_step_id=current_step.step_id,
        current_step_role=current_step.role,
        current_step_text=current_step.text,
        source_step_ids=[step.step_id for step in source_steps],
        source_step_roles=[step.role for step in source_steps],
        num_reasoning_steps_seen=int(num_reasoning_steps_seen),
        num_reasoning_steps_total=int(num_reasoning_steps_total),
        metadata=metadata,
    )
    prefix.validate()
    return prefix


def _stable_hash(*parts: str) -> str:
    """Return a deterministic SHA256 hex digest from string parts."""
    payload = "||".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _validate_string_list(values: Any, field_name: str) -> None:
    """Validate that a field is a list of strings."""
    if not isinstance(values, list):
        raise TypeError(f"`{field_name}` must be list[str], got {type(values)!r}")
    for idx, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(
                f"`{field_name}[{idx}]` must be str, got {type(value)!r}"
            )


def _validate_unit_interval(value: Any, field_name: str) -> None:
    """Validate that a numeric value lies in `[0, 1]`."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"`{field_name}` must be numeric, got {type(value)!r}")
    numeric = float(value)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"`{field_name}` must be in [0, 1], got {numeric}")
