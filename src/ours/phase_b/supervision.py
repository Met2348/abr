"""Phase B supervision transforms and token-weight planning helpers.

Why this file exists
--------------------
Phase B training needs two kinds of supervision controls that should stay
separate from the generic training loop:

1. target transforms:
   - for example, compressing a long GSM8K rationale into a shorter chain of
     thought while preserving the final answer line,
2. token-weight planning:
   - for example, assigning a larger loss weight to the final-answer segment
     than to the rationale segment.

Keeping this logic in one small module makes the training script easier to read
and easier to test.

What this file does
-------------------
1. split a target into reasoning and answer segments,
2. apply named target transforms,
3. return one normalized supervision plan that the trainer can tokenize,
4. expose helper metadata so manifests and diagnostics can record exactly what
   supervision policy was used.

Interaction with other files
----------------------------
- `scripts/phase_b_train_sft.py` uses these helpers before tokenization.
- `phase_B_report.md` and other diagnostics rely on the transform names being
  stable and human-readable.

Example
-------
```python
plan = build_supervision_plan(
    target_text=(
        "Step 1: 2+2=4\n"
        "Step 2: 4+4=8\n"
        "Final answer: 8"
    ),
    target_transform="gsm8k_short_cot_last2",
    target_max_reasoning_lines=2,
)
assert plan.transformed_target_text.endswith("Final answer: 8")
```
"""

from __future__ import annotations

from dataclasses import dataclass


ANSWER_PREFIX_CANDIDATES: tuple[str, ...] = (
    "Final answer:",
    "Verdict:",
)


@dataclass(slots=True)
class SupervisionPlan:
    """Normalized supervision layout after optional target transformation.

    Attributes
    ----------
    transformed_target_text:
        Final target text that should be supervised.
    reasoning_text:
        Reasoning-only prefix of the transformed target, if any.
    answer_text:
        Final-answer segment of the transformed target.
    answer_signal_found:
        Whether a recognized final-answer line was found.
    transform_name:
        Stable identifier of the applied target transform.

    Example
    -------
    ```python
    plan = SupervisionPlan(
        transformed_target_text="Final answer: 72",
        reasoning_text="",
        answer_text="Final answer: 72",
        answer_signal_found=True,
        transform_name="none",
    )
    ```
    """

    transformed_target_text: str
    reasoning_text: str
    answer_text: str
    answer_signal_found: bool
    transform_name: str


def list_target_transforms() -> list[str]:
    """Return the supported target-transform identifiers.

    Example
    -------
    ```python
    assert "gsm8k_short_cot_last2" in list_target_transforms()
    ```
    """

    return [
        "none",
        "gsm8k_short_cot_last2",
    ]


def split_reasoning_and_answer(target_text: str) -> tuple[str, str, bool]:
    """Split target text into reasoning and final-answer segments.

    The function prefers line-based final-answer detection because prepared
    Phase A artifacts typically place the answer signal on the last line.

    Parameters
    ----------
    target_text:
        Raw supervised target text.

    Returns
    -------
    tuple[str, str, bool]
        `(reasoning_text, answer_text, answer_signal_found)`.

    Example
    -------
    ```python
    reasoning, answer, found = split_reasoning_and_answer(
        "1+1=2\\nFinal answer: 2"
    )
    assert found is True
    assert answer == "Final answer: 2"
    ```
    """

    normalized = target_text.strip()
    if normalized == "":
        return "", "", False

    lines = [line.rstrip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return "", "", False

    last_line = lines[-1].strip()
    if any(last_line.startswith(prefix) for prefix in ANSWER_PREFIX_CANDIDATES):
        reasoning = "\n".join(line.strip() for line in lines[:-1]).strip()
        return reasoning, last_line, True

    for prefix in ANSWER_PREFIX_CANDIDATES:
        idx = normalized.rfind(prefix)
        if idx != -1:
            reasoning = normalized[:idx].strip()
            answer = normalized[idx:].strip()
            return reasoning, answer, True

    return "", normalized, False


def build_supervision_plan(
    *,
    target_text: str,
    target_transform: str = "none",
    target_max_reasoning_lines: int | None = None,
) -> SupervisionPlan:
    """Apply one target transform and return the normalized supervision plan.

    Supported transforms
    --------------------
    `none`
        Keep the original target unchanged.
    `gsm8k_short_cot_last2`
        Keep at most the last N reasoning lines and always preserve the final
        answer line. This is intended for GSM8K diagnostics where long CoT may
        be teaching style more than truthful arithmetic.

    Example
    -------
    ```python
    plan = build_supervision_plan(
        target_text="a\\nb\\nc\\nFinal answer: 5",
        target_transform="gsm8k_short_cot_last2",
        target_max_reasoning_lines=2,
    )
    assert plan.transformed_target_text == "b\\nc\\nFinal answer: 5"
    ```
    """

    if target_transform not in list_target_transforms():
        raise ValueError(
            f"Unknown target_transform={target_transform!r}. "
            f"Supported: {list_target_transforms()}"
        )

    reasoning_text, answer_text, answer_signal_found = split_reasoning_and_answer(
        target_text
    )
    reasoning_lines = [
        line.strip() for line in reasoning_text.splitlines() if line.strip()
    ]

    if target_transform == "gsm8k_short_cot_last2":
        limit = target_max_reasoning_lines or 2
        if limit <= 0:
            reasoning_lines = []
        else:
            reasoning_lines = reasoning_lines[-limit:]

    reasoning_text = "\n".join(reasoning_lines).strip()

    pieces = []
    if reasoning_text:
        pieces.append(reasoning_text)
    if answer_text:
        pieces.append(answer_text.strip())

    transformed_target_text = "\n".join(pieces).strip()
    if transformed_target_text == "":
        transformed_target_text = target_text.strip()
        reasoning_text, answer_text, answer_signal_found = split_reasoning_and_answer(
            transformed_target_text
        )

    return SupervisionPlan(
        transformed_target_text=transformed_target_text,
        reasoning_text=reasoning_text,
        answer_text=answer_text,
        answer_signal_found=answer_signal_found,
        transform_name=target_transform,
    )
