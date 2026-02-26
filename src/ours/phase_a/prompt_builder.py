"""Dataset-agnostic prompt building for Phase A.

Key idea
--------
All datasets are already normalized into `CanonicalSample`.
So prompt-building should depend on that canonical shape, not dataset internals.
"""

from __future__ import annotations

from typing import Any

from ours.data.schema import CanonicalSample

from .contracts import PreparedSample, PromptTemplateSpec, TargetStyle


# Registry: template_id -> version -> spec
PROMPT_TEMPLATE_REGISTRY: dict[str, dict[str, PromptTemplateSpec]] = {
    "qa_direct": {
        "1.0.0": PromptTemplateSpec(
            template_id="qa_direct",
            template_version="1.0.0",
            description=(
                "Direct QA template. Model should answer concisely. "
                "Used for answer-only SFT baseline."
            ),
            system_prompt=(
                "You are a careful reasoning assistant. "
                "Return only the final answer when possible."
            ),
            user_prefix="Question:\n{question}\n\nAnswer:",
            answer_prefix="",
        )
    },
    "qa_math_direct_final": {
        "1.0.0": PromptTemplateSpec(
            template_id="qa_math_direct_final",
            template_version="1.0.0",
            description=(
                "Math-focused direct template. "
                "Forces a final-answer line format to reduce ambiguous extraction."
            ),
            system_prompt=(
                "You are a careful math assistant. "
                "Solve internally, then output exactly one line: "
                "'Final answer: <number>'. "
                "Do not include extra explanation."
            ),
            user_prefix=(
                "Question:\n{question}\n\n"
                "Return only one line in this exact format:\n"
                "Final answer: <number>"
            ),
            answer_prefix="Final answer: ",
        )
    },
    "qa_cot_then_final": {
        "1.0.0": PromptTemplateSpec(
            template_id="qa_cot_then_final",
            template_version="1.0.0",
            description=(
                "Reasoning-first template. Model can produce steps and must end "
                "with a final answer line."
            ),
            system_prompt=(
                "You are a careful reasoning assistant. "
                "Show your reasoning briefly, then write 'Final answer: <answer>'."
            ),
            user_prefix="Question:\n{question}\n\nLet's think step by step.",
            answer_prefix="Final answer: ",
        )
    },
    "qa_binary_strict": {
        "1.0.0": PromptTemplateSpec(
            template_id="qa_binary_strict",
            template_version="1.0.0",
            description=(
                "Strict yes/no output template for binary QA. "
                "Designed to maximize answer-format compliance and reduce "
                "token waste from long explanations."
            ),
            system_prompt=(
                "You are a careful assistant for yes/no questions. "
                "Output exactly one word: yes or no. "
                "Do not output explanations, punctuation, or extra words. "
                "Do not default to one label; decide from question evidence."
            ),
            user_prefix=(
                "Question:\n{question}\n\n"
                "Respond with one word only (yes or no).\n"
                "Answer:"
            ),
            answer_prefix="",
        )
    },
}


def list_template_versions(template_id: str) -> list[str]:
    """Return available versions for one template id."""
    if template_id not in PROMPT_TEMPLATE_REGISTRY:
        raise KeyError(
            f"Unknown template_id={template_id!r}. "
            f"Supported: {sorted(PROMPT_TEMPLATE_REGISTRY.keys())}"
        )
    return sorted(PROMPT_TEMPLATE_REGISTRY[template_id].keys())


def resolve_template(template_id: str, template_version: str) -> PromptTemplateSpec:
    """Resolve and validate one template from registry."""
    if template_id not in PROMPT_TEMPLATE_REGISTRY:
        raise KeyError(
            f"Unknown template_id={template_id!r}. "
            f"Supported: {sorted(PROMPT_TEMPLATE_REGISTRY.keys())}"
        )
    versions = PROMPT_TEMPLATE_REGISTRY[template_id]
    if template_version not in versions:
        raise KeyError(
            f"Unknown template_version={template_version!r} for template_id={template_id!r}. "
            f"Supported versions: {sorted(versions.keys())}"
        )
    spec = versions[template_version]
    spec.validate()
    return spec


def build_prepared_sample(
    sample: CanonicalSample,
    split: str,
    target_style: TargetStyle,
    template_id: str,
    template_version: str,
    extra_metadata: dict[str, Any] | None = None,
) -> PreparedSample:
    """Convert one canonical sample into a model-ready record.

    Output fields include:
    - prompt_text: what user/system asks model
    - target_text: supervised target used in SFT
    """
    sample.validate()

    if target_style not in {"answer_only", "cot_then_answer"}:
        raise ValueError(
            "`target_style` must be one of {'answer_only', 'cot_then_answer'}"
        )

    spec = resolve_template(template_id=template_id, template_version=template_version)

    user_text = spec.user_prefix.format(question=sample.question)
    prompt_text = _compose_chat_like_prompt(
        system_prompt=spec.system_prompt,
        user_text=user_text,
    )

    target_text = _build_target_text(
        answer=sample.answer,
        cot=sample.cot,
        target_style=target_style,
        answer_prefix=spec.answer_prefix,
    )

    metadata = {
        "template_description": spec.description,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    prepared = PreparedSample(
        sample_id=sample.id,
        dataset=sample.dataset,
        split=split,
        question=sample.question,
        answer=sample.answer,
        cot=sample.cot,
        target_style=target_style,
        template_id=spec.template_id,
        template_version=spec.template_version,
        prompt_text=prompt_text,
        target_text=target_text,
        metadata=metadata,
    )
    prepared.validate()
    return prepared


def _compose_chat_like_prompt(system_prompt: str, user_text: str) -> str:
    """Compose a simple, explicit plain-text prompt structure.

    We intentionally keep this format readable and framework-agnostic.
    Later you can map it to model-specific chat templates if needed.
    """
    return (
        "[SYSTEM]\n"
        f"{system_prompt.strip()}\n\n"
        "[USER]\n"
        f"{user_text.strip()}\n\n"
        "[ASSISTANT]\n"
    )


def _build_target_text(
    answer: str,
    cot: str | None,
    target_style: TargetStyle,
    answer_prefix: str,
) -> str:
    """Build supervised target text based on selected style."""
    answer_clean = answer.strip()
    cot_clean = cot.strip() if cot is not None else ""

    if target_style == "answer_only":
        return f"{answer_prefix}{answer_clean}".strip()

    # target_style == "cot_then_answer"
    if cot_clean:
        return f"{cot_clean}\n{answer_prefix}{answer_clean}".strip()
    # Fallback: if CoT missing, still produce usable target.
    return f"{answer_prefix}{answer_clean}".strip()
