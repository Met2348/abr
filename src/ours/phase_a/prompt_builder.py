"""Build prompt/target pairs for Phase A from canonical dataset samples.

Why this file exists
--------------------
Different datasets need different prompt wording, but the rest of the pipeline
should not care about those wording details. This module centralizes prompt-template
definitions and converts canonical samples into model-ready prompt/target records.

What this file contains
-----------------------
- a prompt-template registry keyed by template id and version
- helpers for resolving templates
- the main conversion function `build_prepared_sample(...)`
- low-level helpers for composing the final prompt text and target text

Execution logic
---------------
`build_prepared_sample(...)` validates a canonical sample -> resolves the template ->
formats the prompt -> formats the target -> emits a validated `PreparedSample`.

Interaction with other files
----------------------------
- `src/ours/data/schema.py`: defines `CanonicalSample`.
- `src/ours/phase_a/contracts.py`: defines `PreparedSample` and template specs.
- `scripts/phase_a_prepare.py`: main caller that writes prepared JSONL artifacts.

Example
-------
```python
prepared = build_prepared_sample(
    sample=canonical_sample,
    split="train",
    target_style="answer_only",
    template_id="qa_direct",
    template_version="1.0.0",
)
```
"""

from __future__ import annotations

from typing import Any

from ours.data.schema import CanonicalSample

from .contracts import PreparedSample, PromptTemplateSpec, TargetStyle


# Registry: template_id -> version -> spec
# 模板库是 Phase A 可复现实验的核心之一：改模板文案时建议升 template_version，
# 避免“同名模板行为变化”导致历史结果难对比。
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
    "qa_strategyqa_minimal_binary": {
        "1.0.0": PromptTemplateSpec(
            template_id="qa_strategyqa_minimal_binary",
            template_version="1.0.0",
            description=(
                "StrategyQA style A: minimal binary decision contract. "
                "Optimized for short answers and low format drift."
            ),
            system_prompt=(
                "You are a careful assistant for yes/no questions. "
                "Think silently, then output one token: yes or no."
            ),
            user_prefix=(
                "Question:\n{question}\n\n"
                "Return exactly one token: yes or no.\n"
                "Answer:"
            ),
            answer_prefix="",
        )
    },
    "qa_strategyqa_cot_compact": {
        "1.0.0": PromptTemplateSpec(
            template_id="qa_strategyqa_cot_compact",
            template_version="1.0.0",
            description=(
                "StrategyQA style B: compact reasoning then explicit final line. "
                "Balances reasoning visibility and extraction stability."
            ),
            system_prompt=(
                "You are a careful reasoning assistant. "
                "Use at most two short reasoning lines, then write "
                "'Final answer: yes' or 'Final answer: no'."
            ),
            user_prefix=(
                "Question:\n{question}\n\n"
                "Reason briefly, then give the final answer line."
            ),
            answer_prefix="Final answer: ",
        )
    },
    "qa_strategyqa_evidence_verdict": {
        "1.0.0": PromptTemplateSpec(
            template_id="qa_strategyqa_evidence_verdict",
            template_version="1.0.0",
            description=(
                "StrategyQA style C: evidence-first structure with explicit verdict tag."
            ),
            system_prompt=(
                "You are a fact-checking assistant. "
                "Give one short evidence line, then one final verdict line "
                "in the form 'Verdict: yes' or 'Verdict: no'."
            ),
            user_prefix=(
                "Question:\n{question}\n\n"
                "Format:\n"
                "Evidence: <short reason>\n"
                "Verdict: <yes or no>"
            ),
            answer_prefix="Verdict: ",
        )
    },
    "qa_gsm8k_direct_final_only": {
        "1.0.0": PromptTemplateSpec(
            template_id="qa_gsm8k_direct_final_only",
            template_version="1.0.0",
            description=(
                "GSM8K style A: direct numeric answer with strict final-answer line."
            ),
            system_prompt=(
                "You are a careful math assistant. "
                "Solve internally, then output exactly one line: "
                "'Final answer: <number>'."
            ),
            user_prefix=(
                "Question:\n{question}\n\n"
                "Output exactly one line:\n"
                "Final answer: <number>"
            ),
            answer_prefix="Final answer: ",
        )
    },
    "qa_gsm8k_cot_compact_final": {
        "1.0.0": PromptTemplateSpec(
            template_id="qa_gsm8k_cot_compact_final",
            template_version="1.0.0",
            description=(
                "GSM8K style B: short chain-of-thought and strict final line."
            ),
            system_prompt=(
                "You are a careful math tutor. "
                "Show concise reasoning, then write one final line: "
                "'Final answer: <number>'."
            ),
            user_prefix=(
                "Question:\n{question}\n\n"
                "Use up to 3 short reasoning lines, then output:\n"
                "Final answer: <number>"
            ),
            answer_prefix="Final answer: ",
        )
    },
    "qa_gsm8k_equation_then_final": {
        "1.0.0": PromptTemplateSpec(
            template_id="qa_gsm8k_equation_then_final",
            template_version="1.0.0",
            description=(
                "GSM8K style C: enforce one equation line before the final number."
            ),
            system_prompt=(
                "You are a precise math assistant. "
                "Provide one equation line and then a final answer line."
            ),
            user_prefix=(
                "Question:\n{question}\n\n"
                "Format:\n"
                "Equation: <main equation>\n"
                "Final answer: <number>"
            ),
            answer_prefix="Final answer: ",
        )
    },
}


def list_template_versions(template_id: str) -> list[str]:
    """Return available versions for one template id.

    Example
    -------
    ```python
    versions = list_template_versions("qa_direct")
    ```
    """
    if template_id not in PROMPT_TEMPLATE_REGISTRY:
        raise KeyError(
            f"Unknown template_id={template_id!r}. "
            f"Supported: {sorted(PROMPT_TEMPLATE_REGISTRY.keys())}"
        )
    return sorted(PROMPT_TEMPLATE_REGISTRY[template_id].keys())


def resolve_template(template_id: str, template_version: str) -> PromptTemplateSpec:
    """Resolve and validate one template from the registry.

    Example
    -------
    ```python
    spec = resolve_template("qa_direct", "1.0.0")
    ```
    """
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
    # 统一在这里做模板存在性和版本校验，调用方只拿已验证的 spec。
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

    Example
    -------
    ```python
    prepared = build_prepared_sample(
        sample=sample,
        split="validation",
        target_style="cot_then_answer",
        template_id="qa_strategyqa_cot_compact",
        template_version="1.0.0",
    )
    ```

    说明
    --------
    - 本函数是 `Phase A -> Phase B/Phase C` 的监督桥梁。
    - 产出的 `target_text` 会直接进入后续 SFT 与前缀价值学习流程。
    - `target_style` 决定监督中是否包含显式推理轨迹。
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

    # 要点：target_text 是后续 Phase B/Phase C 的核心监督来源。
    # answer_only 仅答案；cot_then_answer 包含推理与最终答案行。
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

    Example
    -------
    ```python
    prompt_text = _compose_chat_like_prompt("You are helpful.", "Question: ...")
    ```
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
    """Build supervised target text based on selected style.

    Example
    -------
    ```python
    target = _build_target_text(
        answer="yes",
        cot="Because ...",
        target_style="cot_then_answer",
        answer_prefix="Final answer: ",
    )
    ```

    说明
    --------
    - `answer_only`：只监督最终答案，过程信息较少。
    - `cot_then_answer`：监督“推理轨迹 + 最终答案”，更适合后续前缀级分析。
    """
    answer_clean = answer.strip()
    cot_clean = cot.strip() if cot is not None else ""

    if target_style == "answer_only":
        # 仅监督最终答案，不注入显式推理轨迹。
        return f"{answer_prefix}{answer_clean}".strip()

    # target_style == "cot_then_answer"
    if cot_clean:
        # 显式写入“推理轨迹 + 最终答案”，供后续前缀化与价值学习使用。
        return f"{cot_clean}\n{answer_prefix}{answer_clean}".strip()
    # Fallback: if CoT missing, still produce usable target.
    return f"{answer_prefix}{answer_clean}".strip()
