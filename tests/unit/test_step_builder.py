"""Unit tests for step-level preprocessing builders.

These tests intentionally use tiny in-memory samples so beginners can
understand expected behavior without downloading extra data.
"""

from __future__ import annotations

from ours.data.schema import CanonicalSample
from ours.data.step_builder import (
    StepBuildConfig,
    build_step_sequence,
    build_step_sequences,
    split_reasoning_text,
)


def test_build_step_sequence_default_includes_question_reasoning_answer() -> None:
    sample = CanonicalSample(
        id="toy:1",
        dataset="gsm8k",
        question="What is 2+2?",
        answer="4",
        cot="Step 1: add numbers.\nStep 2: result is 4.",
    )

    sequence = build_step_sequence(sample)

    assert sequence.num_steps == 4
    assert [step.role for step in sequence.steps] == [
        "question",
        "reasoning",
        "reasoning",
        "answer",
    ]
    assert sequence.steps[0].text == "What is 2+2?"
    assert sequence.steps[-1].text == "4"


def test_build_step_sequence_can_exclude_question_and_answer() -> None:
    sample = CanonicalSample(
        id="toy:2",
        dataset="strategyqa",
        question="Is sky blue?",
        answer="yes",
        cot="- depends on weather\n- usually yes",
    )
    config = StepBuildConfig(
        include_question_as_step0=False,
        include_final_answer_as_terminal_step=False,
        split_mode="newline",
    )

    sequence = build_step_sequence(sample, config=config)

    assert sequence.num_steps == 2
    assert all(step.role == "reasoning" for step in sequence.steps)
    assert sequence.steps[0].text == "depends on weather"


def test_split_reasoning_text_drops_empty_fragments_after_newline_split() -> None:
    config = StepBuildConfig(split_mode="newline")
    cot = "First line\n\n   \nSecond line"
    fragments = split_reasoning_text(cot_text=cot, dataset="gsm8k", config=config)
    assert fragments == ["First line", "Second line"]


def test_split_reasoning_text_sentence_mode() -> None:
    config = StepBuildConfig(split_mode="sentence")
    cot = "Compute A. Then compute B! Finally answer C?"
    fragments = split_reasoning_text(cot_text=cot, dataset="gsm8k", config=config)
    assert fragments == ["Compute A.", "Then compute B!", "Finally answer C?"]


def test_build_step_sequence_is_deterministic_for_same_input() -> None:
    sample = CanonicalSample(
        id="toy:3",
        dataset="proofwriter",
        question="Given theory, can we infer X?",
        answer="yes",
        cot="Rule 1 applies.\nRule 2 applies.",
    )
    config = StepBuildConfig(split_mode="auto")

    seq1 = build_step_sequence(sample, config)
    seq2 = build_step_sequence(sample, config)

    assert seq1.to_dict() == seq2.to_dict()


def test_build_step_sequences_batch() -> None:
    samples = [
        CanonicalSample(
            id="toy:a",
            dataset="gsm8k",
            question="Q1",
            answer="A1",
            cot="R1",
        ),
        CanonicalSample(
            id="toy:b",
            dataset="gsm8k",
            question="Q2",
            answer="A2",
            cot="R2",
        ),
    ]
    outputs = build_step_sequences(samples, source_name="unit-test")
    assert len(outputs) == 2
    assert outputs[0].sample_id == "toy:a"
    assert outputs[1].sample_id == "toy:b"
