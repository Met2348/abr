"""Inference-stability diagnostics for Phase A artifacts.

Why this module exists
----------------------
Phase A experiments revealed a repeated pattern in long free-form outputs:
- multiple ``Final answer: yes/no`` tags in one response,
- occasional answer flips across decode budgets.

This module provides reusable metrics for:
1) per-run instability signatures,
2) pairwise flip analysis across runs.
"""

from __future__ import annotations

import re
from typing import Any

FINAL_ANSWER_PATTERN = re.compile(
    r"(?:final\s*answer|answer)\s*:\s*(yes|no|true|false)\b",
    re.IGNORECASE,
)


def _normalize_binary_token(text: str | None) -> str | None:
    """Normalize common binary-answer variants to `yes` or `no`.

    Example
    -------
    ```python
    _normalize_binary_token("True")  # -> "yes"
    ```
    """
    if text is None:
        return None
    token = str(text).strip().lower()
    mapping = {
        "yes": "yes",
        "true": "yes",
        "1": "yes",
        "no": "no",
        "false": "no",
        "0": "no",
    }
    return mapping.get(token)


def extract_final_answer_sequence(raw_prediction: str) -> list[str]:
    """Extract normalized yes/no sequence from ``Final answer`` tags."""
    matches = FINAL_ANSWER_PATTERN.findall(raw_prediction or "")
    seq: list[str] = []
    for token in matches:
        normalized = _normalize_binary_token(token)
        if normalized is not None:
            seq.append(normalized)
    return seq


def summarize_strategyqa_instability(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute run-level instability metrics from scored prediction rows.

    Notes
    -----
    This function is dataset-agnostic at runtime, but metrics are most meaningful
    for StrategyQA-style yes/no outputs.
    """
    n_total = len(scored_rows)
    if n_total == 0:
        return {
            "n_total": 0,
            "n_correct": 0,
            "accuracy": 0.0,
            "n_parse_error": 0,
            "parse_error_rate": 0.0,
            "n_with_final_tag": 0,
            "with_final_tag_rate": 0.0,
            "n_multi_final_tag": 0,
            "multi_final_tag_rate": 0.0,
            "n_first_last_disagree": 0,
            "first_last_disagree_rate": 0.0,
            "n_with_tag_switch": 0,
            "tag_switch_rate": 0.0,
            "mean_final_tag_count_all": 0.0,
            "mean_final_tag_count_tagged": 0.0,
            "mean_switch_count_tagged": 0.0,
        }

    n_correct = 0
    n_parse_error = 0
    n_with_final_tag = 0
    n_multi_final_tag = 0
    n_first_last_disagree = 0
    n_with_tag_switch = 0

    sum_tag_count_all = 0
    sum_tag_count_tagged = 0
    sum_switch_count_tagged = 0

    # run 内统计：观测单条输出里是否出现多次 final-answer 及标签切换。
    for row in scored_rows:
        if bool(row.get("is_correct", False)):
            n_correct += 1
        if bool(row.get("parse_error", False)):
            n_parse_error += 1

        seq = extract_final_answer_sequence(str(row.get("raw_prediction", "") or ""))
        tag_count = len(seq)
        sum_tag_count_all += tag_count
        if tag_count == 0:
            continue

        n_with_final_tag += 1
        sum_tag_count_tagged += tag_count

        if tag_count >= 2:
            n_multi_final_tag += 1
            switch_count = 0
            for i in range(1, tag_count):
                if seq[i] != seq[i - 1]:
                    switch_count += 1
            if switch_count > 0:
                n_with_tag_switch += 1
            sum_switch_count_tagged += switch_count
            if seq[0] != seq[-1]:
                n_first_last_disagree += 1

    n_tagged = max(n_with_final_tag, 1)
    return {
        "n_total": int(n_total),
        "n_correct": int(n_correct),
        "accuracy": float(n_correct / n_total),
        "n_parse_error": int(n_parse_error),
        "parse_error_rate": float(n_parse_error / n_total),
        "n_with_final_tag": int(n_with_final_tag),
        "with_final_tag_rate": float(n_with_final_tag / n_total),
        "n_multi_final_tag": int(n_multi_final_tag),
        "multi_final_tag_rate": float(n_multi_final_tag / n_total),
        "n_first_last_disagree": int(n_first_last_disagree),
        "first_last_disagree_rate": float(n_first_last_disagree / n_total),
        "n_with_tag_switch": int(n_with_tag_switch),
        "tag_switch_rate": float(n_with_tag_switch / n_total),
        "mean_final_tag_count_all": float(sum_tag_count_all / n_total),
        "mean_final_tag_count_tagged": float(sum_tag_count_tagged / n_tagged),
        "mean_switch_count_tagged": float(sum_switch_count_tagged / n_tagged),
    }


def index_rows_by_sample_id(scored_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index scored rows by sample id for overlap/flip analysis."""
    index: dict[str, dict[str, Any]] = {}
    for row in scored_rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id == "":
            continue
        index[sample_id] = row
    return index


def compute_pairwise_prediction_flip(
    rows_a_by_id: dict[str, dict[str, Any]],
    rows_b_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute pairwise flip rates for overlapping sample IDs."""
    overlap_ids = sorted(set(rows_a_by_id.keys()) & set(rows_b_by_id.keys()))
    n_overlap = len(overlap_ids)
    if n_overlap == 0:
        return {
            "n_overlap": 0,
            "n_pred_flip": 0,
            "pred_flip_rate": 0.0,
            "n_correctness_flip": 0,
            "correctness_flip_rate": 0.0,
            "n_yes_to_no": 0,
            "n_no_to_yes": 0,
        }

    n_pred_flip = 0
    n_correctness_flip = 0
    n_yes_to_no = 0
    n_no_to_yes = 0

    # run 间统计：同一 sample 在两次实验中的预测是否翻转。
    for sid in overlap_ids:
        row_a = rows_a_by_id[sid]
        row_b = rows_b_by_id[sid]

        pred_a = _normalize_binary_token(str(row_a.get("extracted_prediction", "") or ""))
        pred_b = _normalize_binary_token(str(row_b.get("extracted_prediction", "") or ""))
        if pred_a is not None and pred_b is not None and pred_a != pred_b:
            n_pred_flip += 1
            if pred_a == "yes" and pred_b == "no":
                n_yes_to_no += 1
            elif pred_a == "no" and pred_b == "yes":
                n_no_to_yes += 1

        if bool(row_a.get("is_correct", False)) != bool(row_b.get("is_correct", False)):
            n_correctness_flip += 1

    return {
        "n_overlap": int(n_overlap),
        "n_pred_flip": int(n_pred_flip),
        "pred_flip_rate": float(n_pred_flip / n_overlap),
        "n_correctness_flip": int(n_correctness_flip),
        "correctness_flip_rate": float(n_correctness_flip / n_overlap),
        "n_yes_to_no": int(n_yes_to_no),
        "n_no_to_yes": int(n_no_to_yes),
    }
