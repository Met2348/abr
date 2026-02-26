"""Inference-time answer extraction utilities for Phase A.

Problem this solves
-------------------
Model outputs are free-form text. We need a canonical predicted answer
before scoring. Direct raw string equality is too brittle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal


ExtractMethod = Literal[
    "strategyqa_yes_no",
    "strategyqa_final_answer_tag",
    "strategyqa_last_binary_token",
    "final_answer_tag",
    "gsm8k_hash_marker",
    "boxed",
    "last_number",
    "plain_text_fallback",
]


@dataclass(slots=True)
class ExtractedAnswer:
    """Canonical answer extracted from raw generation text."""

    text: str
    method: ExtractMethod
    parse_error: bool


def extract_answer(raw_text: str, dataset: str) -> ExtractedAnswer:
    """Dataset-aware extraction entry point."""
    dataset = dataset.strip().lower()
    if dataset == "strategyqa":
        return _extract_strategyqa(raw_text)
    if dataset in {"gsm8k", "hendrycks_math"}:
        return _extract_math_style(raw_text)
    return _extract_plain(raw_text)


def normalize_gold_answer(gold: str, dataset: str) -> str:
    """Normalize gold answer into comparable canonical text."""
    dataset = dataset.strip().lower()
    if dataset == "strategyqa":
        return _normalize_yes_no(gold)
    if dataset in {"gsm8k", "hendrycks_math"}:
        numeric = _normalize_numeric_text(gold)
        return numeric if numeric is not None else _normalize_text(gold)
    return _normalize_text(gold)


def answers_equivalent(pred: str, gold: str, dataset: str) -> bool:
    """Task-aware equivalence check.

    We still use equality eventually, but only after normalization/parsing.
    """
    dataset = dataset.strip().lower()
    if dataset == "strategyqa":
        return _normalize_yes_no(pred) == _normalize_yes_no(gold)

    if dataset in {"gsm8k", "hendrycks_math"}:
        pred_num = _normalize_numeric_text(pred)
        gold_num = _normalize_numeric_text(gold)
        if pred_num is not None and gold_num is not None:
            return pred_num == gold_num
        # Relaxed fallback: handle outputs like "10 meters" or "$140".
        # We only apply this when strict numeric parsing fails.
        if gold_num is not None:
            pred_relaxed_num = _normalize_numeric_text_relaxed(pred)
            if pred_relaxed_num is not None:
                return pred_relaxed_num == gold_num
        return _normalize_text(pred) == _normalize_text(gold)

    return _normalize_text(pred) == _normalize_text(gold)


def _extract_strategyqa(raw_text: str) -> ExtractedAnswer:
    text = _truncate_chat_leakage(raw_text.strip())
    if text == "":
        return ExtractedAnswer(text="", method="plain_text_fallback", parse_error=True)

    # 1) Exact or near-exact one-token answer first.
    yes_no = _normalize_yes_no(text)
    if yes_no in {"yes", "no"}:
        return ExtractedAnswer(
            text=yes_no,
            method="strategyqa_yes_no",
            parse_error=False,
        )

    lowered = text.lower()
    # 2) Prefix form, including leakage-adjacent strings like "noHuman:".
    prefix = re.match(
        r"^\s*(yes|no|true|false)(?:\b|(?=human:|user:|assistant:|\[|<|$))",
        lowered,
    )
    if prefix:
        parsed = _normalize_yes_no(prefix.group(1))
        if parsed in {"yes", "no"}:
            return ExtractedAnswer(
                text=parsed,
                method="strategyqa_yes_no",
                parse_error=False,
            )

    # 3) Explicit answer tags are strongest in free-form reasoning outputs.
    final_matches = re.findall(
        r"(?:final\s*answer|answer)\s*[:\-]\s*(yes|no|true|false)\b",
        lowered,
        flags=re.IGNORECASE,
    )
    if final_matches:
        parsed = _normalize_yes_no(final_matches[-1])
        if parsed in {"yes", "no"}:
            return ExtractedAnswer(
                text=parsed,
                method="strategyqa_final_answer_tag",
                parse_error=False,
            )

    # 4) Fallback: use the last binary token rather than first.
    # This is safer for CoT outputs because final decision usually appears late.
    binary_tokens = re.findall(r"\b(yes|no|true|false)\b", lowered)
    if binary_tokens:
        parsed = _normalize_yes_no(binary_tokens[-1])
        if parsed in {"yes", "no"}:
            return ExtractedAnswer(
                text=parsed,
                method="strategyqa_last_binary_token",
                parse_error=False,
            )

    return ExtractedAnswer(
        text=_normalize_text(text),
        method="plain_text_fallback",
        parse_error=True,
    )


def _extract_math_style(raw_text: str) -> ExtractedAnswer:
    text = raw_text.strip()

    # 1) explicit final-answer tags
    m = re.search(
        r"final\s*answer(?:\s*is)?\s*:\s*(.+)|final\s*answer\s*is\s+(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        candidate = (m.group(1) or m.group(2) or "").strip().splitlines()[0].strip()
        relaxed_num = _normalize_numeric_text_relaxed(candidate)
        if relaxed_num is not None:
            return ExtractedAnswer(
                text=relaxed_num,
                method="final_answer_tag",
                parse_error=False,
            )
        return ExtractedAnswer(
            text=_strip_wrapping(candidate),
            method="final_answer_tag",
            parse_error=False,
        )

    # 2) GSM8K marker
    if "####" in text:
        candidate = text.rsplit("####", 1)[-1].strip()
        return ExtractedAnswer(
            text=_strip_wrapping(candidate),
            method="gsm8k_hash_marker",
            parse_error=False,
        )

    # 3) last boxed expression
    boxed = _extract_last_boxed(text)
    if boxed is not None:
        return ExtractedAnswer(
            text=_strip_wrapping(boxed),
            method="boxed",
            parse_error=False,
        )

    # 4) last number fallback
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", text)
    if numbers:
        return ExtractedAnswer(
            text=numbers[-1].strip(),
            method="last_number",
            parse_error=False,
        )

    return ExtractedAnswer(
        text=_normalize_text(text),
        method="plain_text_fallback",
        parse_error=True,
    )


def _extract_plain(raw_text: str) -> ExtractedAnswer:
    text = _normalize_text(raw_text)
    if text == "":
        return ExtractedAnswer(text="", method="plain_text_fallback", parse_error=True)
    return ExtractedAnswer(text=text, method="plain_text_fallback", parse_error=False)


def _extract_last_boxed(text: str) -> str | None:
    marker = "\\boxed"
    pos = text.rfind(marker)
    while pos != -1:
        brace_start = text.find("{", pos)
        if brace_start != -1:
            depth = 0
            for i in range(brace_start, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[brace_start + 1 : i].strip()
        pos = text.rfind(marker, 0, pos)
    return None


def _normalize_text(text: str) -> str:
    # Lowercase + whitespace normalization + remove trailing punctuation noise.
    value = re.sub(r"\s+", " ", text.strip().lower())
    return value.strip(" .,!;:")


def _normalize_yes_no(text: str) -> str:
    value = _normalize_text(text)
    mapping = {
        "yes": "yes",
        "true": "yes",
        "1": "yes",
        "no": "no",
        "false": "no",
        "0": "no",
    }
    return mapping.get(value, value)


def _normalize_numeric_text(text: str) -> str | None:
    value = _strip_wrapping(text.strip())
    value = value.replace(",", "")
    # Support simple fractions such as 1/2
    if re.fullmatch(r"[-+]?\d+/\d+", value):
        num, den = value.split("/", 1)
        try:
            frac = Decimal(num) / Decimal(den)
        except (InvalidOperation, ZeroDivisionError):
            return None
        return _decimal_to_str(frac)

    try:
        dec = Decimal(value)
    except InvalidOperation:
        # Not purely numeric
        return None
    return _decimal_to_str(dec)


def _normalize_numeric_text_relaxed(text: str) -> str | None:
    """Try strict numeric parse, then recover first numeric token from free text.

    Examples recovered by this helper:
    - "10 meters" -> "10"
    - "$140." -> "140"
    - "75% of flowers" -> "75"
    """
    strict = _normalize_numeric_text(text)
    if strict is not None:
        return strict

    token = _extract_first_numeric_token(text)
    if token is None:
        return None
    return _normalize_numeric_text(token)


def _extract_first_numeric_token(text: str) -> str | None:
    value = _strip_wrapping(str(text).strip()).replace(",", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:/\d+)?", value)
    if not m:
        return None
    return m.group(0)


def _decimal_to_str(value: Decimal) -> str:
    # Normalize 1.2300 -> 1.23, 2.0 -> 2
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _strip_wrapping(text: str) -> str:
    value = text.strip()
    # Remove common wrappers around predicted answer.
    wrappers = ["$", "`"]
    for ch in wrappers:
        if value.startswith(ch) and value.endswith(ch) and len(value) >= 2:
            value = value[1:-1].strip()
    return value.strip()


def _truncate_chat_leakage(text: str) -> str:
    """Cut text at common prompt-leak markers.

    Why this exists
    ---------------
    Some generations continue into synthetic next-turn text, e.g.:
    ``noHuman: Is the following statement ...``.
    For StrategyQA we only need the first binary decision token.
    """
    lowered = text.lower()
    markers = [
        "[system]",
        "[user]",
        "[assistant]",
        "human:",
        "user:",
        "assistant:",
    ]
    cut_positions = [lowered.find(marker) for marker in markers]
    cut_positions = [pos for pos in cut_positions if pos >= 0]
    if not cut_positions:
        return text.strip()
    cut = min(cut_positions)
    return text[:cut].strip()
