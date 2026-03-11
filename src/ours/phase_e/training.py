"""Training helpers for Phase E external-pair-only value head runs.

English
-------
This module contains the low-level mechanics behind the Phase E trainer.

High-level goal:
1. encode chosen/rejected texts into frozen backbone features,
2. train a small head so `chosen_score > rejected_score`,
3. optionally add BCE-style auxiliary pressure on logits.

Even though the idea is simple, the implementation still has several concepts
that are easy for beginners to mix up:
1. feature caching,
2. pair weighting,
3. source balancing,
4. checkpoint-selection metrics,
5. score-space vs logit-space objectives.

中文
----
这个模块负责 Phase E 训练的底层实现细节。

高层目标其实很简单：
1. 把 chosen/rejected 文本编码成冻结 backbone 特征，
2. 训练一个小 head，让 `chosen_score > rejected_score`，
3. 必要时再加 BCE 风格的 logits 辅助目标。

但真正实现时，有几块特别容易让新手混淆：
1. 特征缓存怎么用，
2. pair 权重怎么进损失，
3. 多 source 时怎么平衡遍历顺序，
4. checkpoint 用哪个指标选，
5. 某个目标到底作用在 sigmoid 前还是 sigmoid 后。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from ours.phase_b.faithfulness_eval import compute_binary_auc
from ours.phase_b.value_losses import (
    anti_saturation_logit_penalty,
    binary_cross_entropy_calibration_loss,
    contrastive_margin_loss,
    reward_centering_penalty,
)
from ours.phase_d.external_pairs import ExternalPairRecord

from .runtime import load_or_encode_text_features, stable_hash_order


def compute_pair_truncation_diagnostics(
    *,
    pairs: list[ExternalPairRecord],
    tokenizer: Any,
    max_length: int,
    batch_size: int,
) -> dict[str, Any]:
    """计算 pair 文本在给定 `max_length` 下的截断风险。 Compute truncation risk for one pair set under a given `max_length`.

    这个诊断回答的不是“模型训得好不好”，而是“监督信号在进入 backbone 之前是否已经被截坏了”。
    This diagnostic does not ask whether the model learns well; it asks whether the supervision signal is already damaged before it reaches the backbone.
    """
    if int(max_length) <= 0:
        raise ValueError("`max_length` must be > 0")
    if int(batch_size) <= 0:
        raise ValueError("`batch_size` must be > 0")
    if not pairs:
        raise ValueError("truncation diagnostics expect at least one pair")

    overall_bucket = _new_truncation_bucket()
    by_source: dict[str, dict[str, Any]] = {}

    for start in range(0, len(pairs), int(batch_size)):
        batch_pairs = pairs[start : start + int(batch_size)]
        chosen_inputs = [pair.chosen_input_text() for pair in batch_pairs]
        rejected_inputs = [pair.rejected_input_text() for pair in batch_pairs]
        chosen_tokens = tokenizer(
            chosen_inputs,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        rejected_tokens = tokenizer(
            rejected_inputs,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        for pair, chosen_ids, rejected_ids in zip(batch_pairs, chosen_tokens, rejected_tokens, strict=True):
            _update_truncation_bucket(
                bucket=overall_bucket,
                chosen_ids=chosen_ids,
                rejected_ids=rejected_ids,
                max_length=int(max_length),
            )
            source_bucket = by_source.setdefault(str(pair.source_tag), _new_truncation_bucket())
            _update_truncation_bucket(
                bucket=source_bucket,
                chosen_ids=chosen_ids,
                rejected_ids=rejected_ids,
                max_length=int(max_length),
            )

    return {
        "overall": _finalize_truncation_bucket(overall_bucket, max_length=int(max_length)),
        "by_source": {
            key: _finalize_truncation_bucket(bucket, max_length=int(max_length))
            for key, bucket in sorted(by_source.items())
        },
    }


def compute_text_truncation_diagnostics(
    *,
    texts: list[str],
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    group_labels: list[str] | None = None,
) -> dict[str, Any]:
    """计算一组原始文本在给定 `max_length` 下的截断风险。 Compute truncation risk for one text list under a given `max_length`.

    中文
    ----
    这个版本不要求显式 chosen/rejected pair，只回答：
    1. 文本本身有多少会超长，
    2. 长度分布长什么样，
    3. 哪些 group 的超长比例异常高。

    English
    -------
    This text-only variant is used when we do not have explicit chosen/rejected
    pairs but still need to detect whether evaluation inputs are being silently
    truncated before scoring.
    """
    if int(max_length) <= 0:
        raise ValueError("`max_length` must be > 0")
    if int(batch_size) <= 0:
        raise ValueError("`batch_size` must be > 0")
    if not texts:
        raise ValueError("text truncation diagnostics expect at least one text")
    if group_labels is not None and len(group_labels) != len(texts):
        raise ValueError("`group_labels` must align 1:1 with `texts`")

    overall_bucket = _new_text_truncation_bucket()
    by_group: dict[str, dict[str, Any]] = {}

    for start in range(0, len(texts), int(batch_size)):
        batch_texts = texts[start : start + int(batch_size)]
        batch_group_labels = (
            group_labels[start : start + int(batch_size)]
            if group_labels is not None
            else [None] * len(batch_texts)
        )
        batch_tokens = tokenizer(
            batch_texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        for token_ids, group_label in zip(batch_tokens, batch_group_labels, strict=True):
            _update_text_truncation_bucket(
                bucket=overall_bucket,
                token_ids=token_ids,
                max_length=int(max_length),
            )
            if group_label is None:
                continue
            group_bucket = by_group.setdefault(str(group_label), _new_text_truncation_bucket())
            _update_text_truncation_bucket(
                bucket=group_bucket,
                token_ids=token_ids,
                max_length=int(max_length),
            )

    return {
        "overall": _finalize_text_truncation_bucket(overall_bucket, max_length=int(max_length)),
        "by_group": {
            key: _finalize_text_truncation_bucket(bucket, max_length=int(max_length))
            for key, bucket in sorted(by_group.items())
        },
    }


def _new_truncation_bucket() -> dict[str, Any]:
    """创建一个可累加的截断统计桶。 Create one mutable accumulator bucket for truncation stats."""
    return {
        "num_pairs": 0,
        "num_pairs_over_limit": 0,
        "num_pairs_identical_after_truncation": 0,
        "num_pairs_first_diff_after_cutoff": 0,
        "num_pairs_originally_identical": 0,
        "chosen_lengths": [],
        "rejected_lengths": [],
        "first_diff_positions": [],
    }


def _new_text_truncation_bucket() -> dict[str, Any]:
    """创建文本级截断统计桶。 Create one mutable text-level truncation bucket."""
    return {
        "num_texts": 0,
        "num_texts_over_limit": 0,
        "text_lengths": [],
    }


def _update_truncation_bucket(
    *,
    bucket: dict[str, Any],
    chosen_ids: list[int],
    rejected_ids: list[int],
    max_length: int,
) -> None:
    """把一对样本的长度与截断信息写入桶中。 Fold one pair's length and cutoff statistics into one bucket."""
    chosen_length = int(len(chosen_ids))
    rejected_length = int(len(rejected_ids))
    first_diff = _first_difference_token_index(chosen_ids, rejected_ids)
    over_limit = chosen_length > int(max_length) or rejected_length > int(max_length)
    originally_identical = chosen_ids == rejected_ids
    identical_after_truncation = (not originally_identical) and (
        chosen_ids[: int(max_length)] == rejected_ids[: int(max_length)]
    )

    bucket["num_pairs"] = int(bucket["num_pairs"]) + 1
    bucket["num_pairs_over_limit"] = int(bucket["num_pairs_over_limit"]) + int(over_limit)
    bucket["num_pairs_identical_after_truncation"] = (
        int(bucket["num_pairs_identical_after_truncation"]) + int(identical_after_truncation)
    )
    bucket["num_pairs_first_diff_after_cutoff"] = (
        int(bucket["num_pairs_first_diff_after_cutoff"]) + int(first_diff >= int(max_length))
    )
    bucket["num_pairs_originally_identical"] = (
        int(bucket["num_pairs_originally_identical"]) + int(originally_identical)
    )
    bucket["chosen_lengths"].append(int(chosen_length))
    bucket["rejected_lengths"].append(int(rejected_length))
    bucket["first_diff_positions"].append(int(first_diff))


def _update_text_truncation_bucket(
    *,
    bucket: dict[str, Any],
    token_ids: list[int],
    max_length: int,
) -> None:
    """把一条文本的长度与超限信息写入桶中。 Fold one text length and over-limit flag into one bucket."""
    text_length = int(len(token_ids))
    over_limit = text_length > int(max_length)
    bucket["num_texts"] = int(bucket["num_texts"]) + 1
    bucket["num_texts_over_limit"] = int(bucket["num_texts_over_limit"]) + int(over_limit)
    bucket["text_lengths"].append(int(text_length))


def _first_difference_token_index(chosen_ids: list[int], rejected_ids: list[int]) -> int:
    """返回两条 token 序列第一次分歧的位置。 Return the first token position where two token sequences diverge."""
    for index, (chosen_token, rejected_token) in enumerate(zip(chosen_ids, rejected_ids, strict=False)):
        if int(chosen_token) != int(rejected_token):
            return int(index)
    if len(chosen_ids) != len(rejected_ids):
        return int(min(len(chosen_ids), len(rejected_ids)))
    return int(len(chosen_ids))


def _finalize_truncation_bucket(bucket: dict[str, Any], *, max_length: int) -> dict[str, Any]:
    """把可变统计桶冻结成 JSON 友好的摘要。 Freeze one mutable truncation bucket into a JSON-friendly summary."""
    num_pairs = int(bucket["num_pairs"])
    num_pairs_over_limit = int(bucket["num_pairs_over_limit"])
    num_pairs_identical_after_truncation = int(bucket["num_pairs_identical_after_truncation"])
    num_pairs_first_diff_after_cutoff = int(bucket["num_pairs_first_diff_after_cutoff"])
    num_pairs_originally_identical = int(bucket["num_pairs_originally_identical"])
    return {
        "num_pairs": int(num_pairs),
        "max_length": int(max_length),
        "num_pairs_over_limit": int(num_pairs_over_limit),
        "frac_pairs_over_limit": _safe_fraction(num_pairs_over_limit, num_pairs),
        "num_pairs_identical_after_truncation": int(num_pairs_identical_after_truncation),
        "frac_pairs_identical_after_truncation": _safe_fraction(
            num_pairs_identical_after_truncation,
            num_pairs,
        ),
        "num_pairs_first_diff_after_cutoff": int(num_pairs_first_diff_after_cutoff),
        "frac_pairs_first_diff_after_cutoff": _safe_fraction(
            num_pairs_first_diff_after_cutoff,
            num_pairs,
        ),
        "num_pairs_originally_identical": int(num_pairs_originally_identical),
        "chosen_length": _summarize_numeric_series(bucket["chosen_lengths"]),
        "rejected_length": _summarize_numeric_series(bucket["rejected_lengths"]),
        "first_diff_token_index": _summarize_numeric_series(bucket["first_diff_positions"]),
        "warning_flags": _build_truncation_warning_flags(
            frac_pairs_over_limit=_safe_fraction(num_pairs_over_limit, num_pairs),
            frac_pairs_identical_after_truncation=_safe_fraction(
                num_pairs_identical_after_truncation,
                num_pairs,
            ),
            frac_pairs_first_diff_after_cutoff=_safe_fraction(
                num_pairs_first_diff_after_cutoff,
                num_pairs,
            ),
        ),
    }


def _summarize_numeric_series(values: list[int]) -> dict[str, Any]:
    """把一组整数序列压成分位点摘要。 Summarize one integer series into quantile-style statistics."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0,
            "p50": 0,
            "p75": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    ordered = sorted(int(value) for value in values)
    count = int(len(ordered))
    return {
        "count": int(count),
        "mean": float(sum(ordered) / count),
        "min": int(ordered[0]),
        "p50": int(_quantile_from_sorted(ordered, 0.50)),
        "p75": int(_quantile_from_sorted(ordered, 0.75)),
        "p90": int(_quantile_from_sorted(ordered, 0.90)),
        "p95": int(_quantile_from_sorted(ordered, 0.95)),
        "p99": int(_quantile_from_sorted(ordered, 0.99)),
        "max": int(ordered[-1]),
    }


def _finalize_text_truncation_bucket(bucket: dict[str, Any], *, max_length: int) -> dict[str, Any]:
    """把文本级统计桶冻结成 JSON 友好摘要。 Freeze one text-level bucket into a JSON-friendly summary."""
    num_texts = int(bucket["num_texts"])
    num_texts_over_limit = int(bucket["num_texts_over_limit"])
    frac_texts_over_limit = _safe_fraction(num_texts_over_limit, num_texts)
    return {
        "num_texts": int(num_texts),
        "max_length": int(max_length),
        "num_texts_over_limit": int(num_texts_over_limit),
        "frac_texts_over_limit": float(frac_texts_over_limit),
        "text_length": _summarize_numeric_series(bucket["text_lengths"]),
        "warning_flags": _build_text_truncation_warning_flags(
            frac_texts_over_limit=float(frac_texts_over_limit),
        ),
    }


def _quantile_from_sorted(sorted_values: list[int], quantile: float) -> int:
    """从已排序序列中取一个稳定分位点。 Read one stable quantile from an already sorted integer series."""
    if not sorted_values:
        return 0
    if float(quantile) <= 0.0:
        return int(sorted_values[0])
    if float(quantile) >= 1.0:
        return int(sorted_values[-1])
    position = int(float(quantile) * float(len(sorted_values) - 1))
    return int(sorted_values[position])


def _safe_fraction(numerator: int, denominator: int) -> float:
    """返回安全比例，避免空分母。 Return a safe fraction and avoid division by zero for empty buckets."""
    if int(denominator) <= 0:
        return 0.0
    return float(int(numerator) / int(denominator))


def _build_truncation_warning_flags(
    *,
    frac_pairs_over_limit: float,
    frac_pairs_identical_after_truncation: float,
    frac_pairs_first_diff_after_cutoff: float,
) -> dict[str, bool]:
    """把关键比例转成易读告警。 Convert key truncation fractions into readable warning flags."""
    over_limit = float(frac_pairs_over_limit)
    identical = float(frac_pairs_identical_after_truncation)
    hidden_diff = float(frac_pairs_first_diff_after_cutoff)
    return {
        "has_any_over_limit_pairs": bool(over_limit > 0.0),
        "high_over_limit_fraction": bool(over_limit >= 0.10),
        "has_any_pairs_collapsing_after_truncation": bool(identical > 0.0),
        "high_pairs_collapsing_after_truncation_fraction": bool(identical >= 0.05),
        "has_any_first_diff_after_cutoff": bool(hidden_diff > 0.0),
        "high_first_diff_after_cutoff_fraction": bool(hidden_diff >= 0.05),
    }


def _build_text_truncation_warning_flags(*, frac_texts_over_limit: float) -> dict[str, bool]:
    """把文本级超限比例转成易读告警。 Convert text-only over-limit fraction into readable warning flags."""
    over_limit = float(frac_texts_over_limit)
    return {
        "has_any_over_limit_texts": bool(over_limit > 0.0),
        "high_over_limit_fraction": bool(over_limit >= 0.10),
    }


def validate_pair_truncation_diagnostics(
    *,
    diagnostics: dict[str, Any],
    context_label: str,
    max_allowed_over_limit_fraction: float,
) -> None:
    """在训练/评测前强制检查 pair 级截断风险。 Enforce pair-level truncation limits before training/eval."""
    if not (0.0 <= float(max_allowed_over_limit_fraction) <= 1.0):
        raise ValueError("`max_allowed_over_limit_fraction` must be in [0, 1]")
    overall = dict(diagnostics["overall"])
    failures = _collect_pair_truncation_failures(
        payload=overall,
        max_allowed_over_limit_fraction=float(max_allowed_over_limit_fraction),
    )
    if not failures:
        return
    by_source_failures: list[str] = []
    for source_tag, payload in sorted(dict(diagnostics.get("by_source", {})).items()):
        source_failures = _collect_pair_truncation_failures(
            payload=dict(payload),
            max_allowed_over_limit_fraction=float(max_allowed_over_limit_fraction),
        )
        if source_failures:
            by_source_failures.append(f"{source_tag}: {', '.join(source_failures)}")
    detail_suffix = f" | by_source: {'; '.join(by_source_failures)}" if by_source_failures else ""
    raise ValueError(
        f"{context_label} truncation risk too high: {', '.join(failures)}{detail_suffix}"
    )


def validate_text_truncation_diagnostics(
    *,
    diagnostics: dict[str, Any],
    context_label: str,
    max_allowed_over_limit_fraction: float,
) -> None:
    """在编码前强制检查文本级截断风险。 Enforce text-level truncation limits before encoding."""
    if not (0.0 <= float(max_allowed_over_limit_fraction) <= 1.0):
        raise ValueError("`max_allowed_over_limit_fraction` must be in [0, 1]")
    overall = dict(diagnostics["overall"])
    failures = _collect_text_truncation_failures(
        payload=overall,
        max_allowed_over_limit_fraction=float(max_allowed_over_limit_fraction),
    )
    if not failures:
        return
    by_group_failures: list[str] = []
    for group_label, payload in sorted(dict(diagnostics.get("by_group", {})).items()):
        group_failures = _collect_text_truncation_failures(
            payload=dict(payload),
            max_allowed_over_limit_fraction=float(max_allowed_over_limit_fraction),
        )
        if group_failures:
            by_group_failures.append(f"{group_label}: {', '.join(group_failures)}")
    detail_suffix = f" | by_group: {'; '.join(by_group_failures)}" if by_group_failures else ""
    raise ValueError(
        f"{context_label} truncation risk too high: {', '.join(failures)}{detail_suffix}"
    )


def _collect_pair_truncation_failures(
    *,
    payload: dict[str, Any],
    max_allowed_over_limit_fraction: float,
) -> list[str]:
    """把 pair 级截断诊断压成失败原因列表。 Summarize pair-level truncation diagnostics into failure reasons.

    English
    -------
    Earlier Phase E code treated *any* non-zero `identical_after_truncation`
    or `first_diff_after_cutoff` fraction as an unconditional failure.
    That was too strict for large real-world datasets such as R-PRM, where a
    tiny tail of pathological rows can survive upstream filtering.

    We now apply the same operator-configurable tolerance to all three pair
    risks:
    1. over-limit pairs,
    2. pairs that collapse to identical texts after truncation,
    3. pairs whose first chosen/rejected difference appears after the cutoff.

    中文
    ----
    早期 Phase E 的逻辑把 `identical_after_truncation` 和
    `first_diff_after_cutoff` 这两类风险写成了“只要大于 0 就直接失败”。
    这对真实数据集过于苛刻，尤其像 R-PRM 这种规模更大、长尾更明显的数据，
    少量边角样本不应直接阻断整次训练。

    现在改成：三类 pair 级风险统一使用同一个可配置容忍阈值：
    1. 超长 pair 比例，
    2. 截断后 chosen/rejected 变得完全相同的比例，
    3. chosen/rejected 的首个差异被截断掉的比例。
    """
    failures: list[str] = []
    over_limit = float(payload.get("frac_pairs_over_limit", 0.0))
    identical = float(payload.get("frac_pairs_identical_after_truncation", 0.0))
    hidden_diff = float(payload.get("frac_pairs_first_diff_after_cutoff", 0.0))
    if over_limit > float(max_allowed_over_limit_fraction):
        failures.append(
            "over_limit_fraction="
            f"{over_limit:.4f} exceeds {float(max_allowed_over_limit_fraction):.4f}"
        )
    if identical > float(max_allowed_over_limit_fraction):
        failures.append(
            "collapse_after_cut_fraction="
            f"{identical:.4f} exceeds {float(max_allowed_over_limit_fraction):.4f}"
        )
    if hidden_diff > float(max_allowed_over_limit_fraction):
        failures.append(
            "hidden_diff_after_cut_fraction="
            f"{hidden_diff:.4f} exceeds {float(max_allowed_over_limit_fraction):.4f}"
        )
    return failures


def _collect_text_truncation_failures(
    *,
    payload: dict[str, Any],
    max_allowed_over_limit_fraction: float,
) -> list[str]:
    """把文本级截断诊断压成失败原因列表。 Summarize text-level truncation diagnostics into failure reasons."""
    failures: list[str] = []
    over_limit = float(payload.get("frac_texts_over_limit", 0.0))
    if over_limit > float(max_allowed_over_limit_fraction):
        failures.append(
            "over_limit_fraction="
            f"{over_limit:.4f} exceeds {float(max_allowed_over_limit_fraction):.4f}"
        )
    return failures


def build_pair_feature_cache(
    *,
    pairs: list[ExternalPairRecord],
    split_label: str,
    backbone: Any,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    feature_cache_root: Path,
    feature_cache_mode: str,
    lock_timeout_sec: float,
    backbone_signature: dict[str, Any],
    pair_weight_mode: str,
    nonfinite_feature_policy: str,
    torch_module: Any,
    feature_cache_stats: dict[str, Any],
) -> dict[str, Any]:
    """Encode chosen/rejected texts once and package them as one pair cache.

    English
    -------
    The returned object is the tensor-level dataset actually consumed by the
    training loop:
    1. chosen pooled features,
    2. rejected pooled features,
    3. pair weights,
    4. light metadata such as pair ids and source tags.

    中文
    ----
    返回值可以理解成“训练循环真正消费的张量版数据集”：
    1. chosen 侧 pooled features，
    2. rejected 侧 pooled features，
    3. 每对样本的权重，
    4. 轻量元信息（pair id、source tag 等）。
    """
    chosen_texts = [pair.chosen_input_text() for pair in pairs]
    rejected_texts = [pair.rejected_input_text() for pair in pairs]
    # Chosen/rejected text lists are cached separately, but both are tied to the
    # same pair digest so cache hits still correspond to the exact same pair set.
    # chosen/rejected 文本列表分别缓存，但都绑定到同一个 pair digest，
    # 这样缓存命中时仍然对应“完全同一批 pair”。
    extra_signature = {
        "split_label": str(split_label),
        "pair_digest": _hash_pair_ids(pairs),
        "pair_weight_mode": str(pair_weight_mode),
    }
    chosen_features = load_or_encode_text_features(
        cache_namespace="phase_e_pair_features",
        cache_kind=f"{split_label}_chosen",
        texts=chosen_texts,
        backbone=backbone,
        tokenizer=tokenizer,
        max_length=int(max_length),
        batch_size=int(batch_size),
        feature_cache_root=feature_cache_root,
        feature_cache_mode=feature_cache_mode,
        lock_timeout_sec=float(lock_timeout_sec),
        backbone_signature=backbone_signature,
        extra_signature=extra_signature,
        torch_module=torch_module,
        feature_cache_stats=feature_cache_stats,
    )
    rejected_features = load_or_encode_text_features(
        cache_namespace="phase_e_pair_features",
        cache_kind=f"{split_label}_rejected",
        texts=rejected_texts,
        backbone=backbone,
        tokenizer=tokenizer,
        max_length=int(max_length),
        batch_size=int(batch_size),
        feature_cache_root=feature_cache_root,
        feature_cache_mode=feature_cache_mode,
        lock_timeout_sec=float(lock_timeout_sec),
        backbone_signature=backbone_signature,
        extra_signature=extra_signature,
        torch_module=torch_module,
        feature_cache_stats=feature_cache_stats,
    )
    pairs, chosen_features, rejected_features, nonfinite_summary = _apply_nonfinite_feature_policy(
        pairs=pairs,
        chosen_features=chosen_features,
        rejected_features=rejected_features,
        split_label=str(split_label),
        nonfinite_feature_policy=str(nonfinite_feature_policy),
        torch_module=torch_module,
    )

    if pair_weight_mode not in {
        "none",
        "confidence",
        "semantic",
        "confidence_semantic",
        "verdict_balance",
        "confidence_verdict_balance",
        "group_balance",
        "confidence_group_balance",
    }:
        raise ValueError(f"Unsupported pair_weight_mode: {pair_weight_mode!r}")

    verdict_counts: dict[str, int] = {}
    if pair_weight_mode in {"verdict_balance", "confidence_verdict_balance"}:
        for pair in pairs:
            verdict = str((pair.metadata or {}).get("chosen_verdict", "")).strip().lower()
            if verdict:
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    group_counts: dict[str, int] = {}
    if pair_weight_mode in {"group_balance", "confidence_group_balance"}:
        for pair in pairs:
            group_label = _resolve_pair_balance_group_label(pair)
            group_counts[group_label] = group_counts.get(group_label, 0) + 1

    weights = compute_pair_weights(
        pairs=pairs,
        pair_weight_mode=pair_weight_mode,
        verdict_counts=verdict_counts,
        group_counts=group_counts,
    )
    local_route_weights, terminal_route_weights = compute_pair_route_weights(pairs=pairs)
    device = chosen_features.device
    return {
        "chosen_features": chosen_features,
        "rejected_features": rejected_features,
        "pair_weights": torch_module.tensor(weights, dtype=chosen_features.dtype, device=device),
        "local_route_weights": torch_module.tensor(
            local_route_weights,
            dtype=chosen_features.dtype,
            device=device,
        ),
        "terminal_route_weights": torch_module.tensor(
            terminal_route_weights,
            dtype=chosen_features.dtype,
            device=device,
        ),
        "pairs": list(pairs),
        "pair_ids": [pair.pair_id for pair in pairs],
        "source_tags": [pair.source_tag for pair in pairs],
        "num_pairs": int(len(pairs)),
        "nonfinite_feature_summary": nonfinite_summary,
    }


def build_pair_tokenized_cache(
    *,
    pairs: list[ExternalPairRecord],
    tokenizer: Any,
    max_length: int,
    pair_weight_mode: str,
    torch_module: Any,
) -> dict[str, Any]:
    """Tokenize chosen/rejected texts and store raw input_ids for LoRA training.

    English
    -------
    This is the LoRA-mode alternative to `build_pair_feature_cache`.  Instead of
    running backbone forward passes up-front and caching feature vectors, we only
    tokenize the texts and store the raw integer token tensors.

    During LoRA training the backbone forward is executed per mini-batch with
    gradient tracking enabled, so pre-cached features would be stale after any
    LoRA weight update.

    Returned dict keys:
    - ``chosen_input_ids``       [N, max_length] int64 CPU tensor
    - ``chosen_attention_mask``  [N, max_length] int64 CPU tensor
    - ``rejected_input_ids``     [N, max_length] int64 CPU tensor
    - ``rejected_attention_mask``[N, max_length] int64 CPU tensor
    - ``pair_weights``           [N] float32 CPU tensor
    - ``local_route_weights``    [N] float32 CPU tensor
    - ``terminal_route_weights`` [N] float32 CPU tensor
    - ``pair_ids``               list[str]
    - ``source_tags``            list[str]
    - ``num_pairs``              int

    中文
    ----
    LoRA 模式下，backbone 权重在训练过程中会不断更新，预先缓存的特征会立刻过时。
    这个函数只做 tokenization，存储原始的 input_ids，
    供后续每个 mini-batch 用当前的 backbone 实时编码。
    """
    chosen_texts = [pair.chosen_input_text() for pair in pairs]
    rejected_texts = [pair.rejected_input_text() for pair in pairs]

    def _tokenize_texts(texts: list[str]) -> tuple[Any, Any]:
        """Tokenize a list of texts → (input_ids, attention_mask) on CPU."""
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=int(max_length),
        )
        return enc["input_ids"].cpu(), enc["attention_mask"].cpu()

    chosen_ids, chosen_mask = _tokenize_texts(chosen_texts)
    rejected_ids, rejected_mask = _tokenize_texts(rejected_texts)

    # Build pair weights using existing helper (same logic as feature cache).
    if pair_weight_mode not in {
        "none",
        "confidence",
        "semantic",
        "confidence_semantic",
        "verdict_balance",
        "confidence_verdict_balance",
        "group_balance",
        "confidence_group_balance",
    }:
        raise ValueError(f"Unsupported pair_weight_mode: {pair_weight_mode!r}")

    verdict_counts: dict[str, int] = {}
    if pair_weight_mode in {"verdict_balance", "confidence_verdict_balance"}:
        for pair in pairs:
            verdict = str((pair.metadata or {}).get("chosen_verdict", "")).strip().lower()
            if verdict:
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    group_counts: dict[str, int] = {}
    if pair_weight_mode in {"group_balance", "confidence_group_balance"}:
        for pair in pairs:
            group_label = _resolve_pair_balance_group_label(pair)
            group_counts[group_label] = group_counts.get(group_label, 0) + 1

    weights = compute_pair_weights(
        pairs=pairs,
        pair_weight_mode=pair_weight_mode,
        verdict_counts=verdict_counts,
        group_counts=group_counts,
    )
    local_route_weights, terminal_route_weights = compute_pair_route_weights(pairs=pairs)

    return {
        "chosen_input_ids": chosen_ids,
        "chosen_attention_mask": chosen_mask,
        "rejected_input_ids": rejected_ids,
        "rejected_attention_mask": rejected_mask,
        "pair_weights": torch_module.tensor(weights, dtype=torch_module.float32),
        "local_route_weights": torch_module.tensor(local_route_weights, dtype=torch_module.float32),
        "terminal_route_weights": torch_module.tensor(
            terminal_route_weights, dtype=torch_module.float32
        ),
        "pairs": list(pairs),
        "pair_ids": [pair.pair_id for pair in pairs],
        "source_tags": [pair.source_tag for pair in pairs],
        "num_pairs": int(len(pairs)),
    }


def encode_tokenized_cache_with_backbone(
    *,
    tokenized_cache: dict[str, Any],
    backbone: Any,
    torch_module: Any,
    batch_size: int,
    head_dtype: Any,
    grad_enabled: bool = False,
) -> dict[str, Any]:
    """Encode a tokenized pair cache using the current backbone state.

    English
    -------
    Converts a ``tokenized_cache`` (from ``build_pair_tokenized_cache``) into a
    standard feature cache compatible with ``evaluate_pair_cache``.

    The backbone runs on all pairs in mini-batches.  The returned feature
    tensors are on CPU to avoid OOM during LoRA training where both the
    backbone and the value head need GPU memory simultaneously.

    When ``grad_enabled=True`` the call is wrapped in a context that preserves
    the computation graph (for training).  When ``grad_enabled=False`` (default,
    used at eval time) ``torch.no_grad`` is applied.

    中文
    ----
    将 ``build_pair_tokenized_cache`` 产出的 tokenized cache 转换成标准 feature
    cache 格式，与 ``evaluate_pair_cache`` 兼容。
    backbone 在 mini-batch 下运行，结果放在 CPU，避免 LoRA 训练中骨干模型和
    value head 同时竞争显存。
    ``grad_enabled=True`` 时保留计算图（训练用）；
    ``grad_enabled=False`` 时禁用梯度（eval 用）。
    """
    from ours.phase_b.value_head import pool_last_token, resolve_model_input_device  # type: ignore

    backbone_device = resolve_model_input_device(backbone)
    chosen_ids = tokenized_cache["chosen_input_ids"]
    chosen_mask = tokenized_cache["chosen_attention_mask"]
    rejected_ids = tokenized_cache["rejected_input_ids"]
    rejected_mask = tokenized_cache["rejected_attention_mask"]
    num_pairs = int(tokenized_cache["num_pairs"])

    chosen_features_list: list[Any] = []
    rejected_features_list: list[Any] = []

    ctx = torch_module.enable_grad() if grad_enabled else torch_module.no_grad()
    with ctx:
        for sl in _batched_index_ranges(num_pairs, int(batch_size)):
            # Chosen side
            cids = chosen_ids[sl].to(backbone_device)
            cmask = chosen_mask[sl].to(backbone_device)
            c_out = backbone(
                input_ids=cids,
                attention_mask=cmask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            c_last = c_out.hidden_states[-1]
            c_mask_aligned = cmask.to(c_last.device)
            c_pooled = pool_last_token(c_last, c_mask_aligned, torch_module=torch_module)
            if grad_enabled:
                chosen_features_list.append(c_pooled.to(dtype=head_dtype))
            else:
                chosen_features_list.append(c_pooled.detach().cpu().to(dtype=head_dtype))

            # Rejected side
            rids = rejected_ids[sl].to(backbone_device)
            rmask = rejected_mask[sl].to(backbone_device)
            r_out = backbone(
                input_ids=rids,
                attention_mask=rmask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            r_last = r_out.hidden_states[-1]
            r_mask_aligned = rmask.to(r_last.device)
            r_pooled = pool_last_token(r_last, r_mask_aligned, torch_module=torch_module)
            if grad_enabled:
                rejected_features_list.append(r_pooled.to(dtype=head_dtype))
            else:
                rejected_features_list.append(r_pooled.detach().cpu().to(dtype=head_dtype))

    chosen_features = torch_module.cat(chosen_features_list, dim=0)
    rejected_features = torch_module.cat(rejected_features_list, dim=0)

    device = chosen_features.device if grad_enabled else torch_module.device("cpu")
    pair_weights = tokenized_cache["pair_weights"].to(device=device, dtype=head_dtype)
    local_rw = tokenized_cache["local_route_weights"].to(device=device, dtype=head_dtype)
    terminal_rw = tokenized_cache["terminal_route_weights"].to(device=device, dtype=head_dtype)

    return {
        "chosen_features": chosen_features,
        "rejected_features": rejected_features,
        "pair_weights": pair_weights,
        "local_route_weights": local_rw,
        "terminal_route_weights": terminal_rw,
        "pair_ids": tokenized_cache["pair_ids"],
        "source_tags": tokenized_cache["source_tags"],
        "num_pairs": num_pairs,
        "nonfinite_feature_summary": {},
    }


def _apply_nonfinite_feature_policy(
    *,
    pairs: list[ExternalPairRecord],
    chosen_features: Any,
    rejected_features: Any,
    split_label: str,
    nonfinite_feature_policy: str,
    torch_module: Any,
) -> tuple[list[ExternalPairRecord], Any, Any, dict[str, Any]]:
    """Check whether cached features are finite and either fail or filter.

    English
    -------
    Non-finite pooled features are catastrophic for reward/value training:
    1. the head can emit NaNs on the very first forward pass,
    2. the optimizer then produces meaningless checkpoints,
    3. and the run summary may misleadingly look like a normal failed experiment.

    This helper makes that state explicit.

    中文
    ----
    pooled feature 一旦含有非有限值，对 reward/value 训练就是灾难性的：
    1. head 在第一步前向就可能出 NaN，
    2. 优化器随后只会生成无意义 checkpoint，
    3. run summary 还可能伪装成一次“普通实验失败”。

    这里的职责就是把这种状态显式拦住。
    """
    if nonfinite_feature_policy not in {"error", "drop"}:
        raise ValueError(
            "`nonfinite_feature_policy` must be one of {'error', 'drop'}"
        )
    chosen_finite = torch_module.isfinite(chosen_features).all(dim=1)
    rejected_finite = torch_module.isfinite(rejected_features).all(dim=1)
    row_finite = chosen_finite & rejected_finite
    num_bad = int((~row_finite).sum().item())
    if num_bad <= 0:
        return pairs, chosen_features, rejected_features, {
            "split_label": str(split_label),
            "policy": str(nonfinite_feature_policy),
            "num_rows_before": int(len(pairs)),
            "num_bad_rows": 0,
            "num_rows_after": int(len(pairs)),
            "sample_bad_pair_ids": [],
        }

    bad_indices = torch_module.nonzero(~row_finite, as_tuple=False).view(-1).tolist()
    sample_bad_pair_ids = [str(pairs[int(idx)].pair_id) for idx in bad_indices[:5]]
    sample_bad_semantics = [
        str((pairs[int(idx)].metadata or {}).get("pair_semantics", ""))
        for idx in bad_indices[:5]
    ]
    message = (
        "Non-finite pooled features detected. "
        f"split_label={split_label} "
        f"num_bad_rows={num_bad} "
        f"sample_bad_pair_ids={sample_bad_pair_ids} "
        f"sample_bad_semantics={sample_bad_semantics}"
    )
    if nonfinite_feature_policy == "error":
        raise RuntimeError(message)

    keep_indices = torch_module.nonzero(row_finite, as_tuple=False).view(-1)
    kept_pairs = [pairs[int(idx)] for idx in keep_indices.tolist()]
    if not kept_pairs:
        raise RuntimeError(
            f"{message} | policy=drop would remove every row, so the split is unusable."
        )
    filtered_chosen = chosen_features[keep_indices]
    filtered_rejected = rejected_features[keep_indices]
    print(
        "nonfinite_filter  : "
        f"split={split_label} "
        f"dropped_rows={num_bad} "
        f"kept_rows={len(kept_pairs)} "
        f"sample_bad_pair_ids={sample_bad_pair_ids}",
        flush=True,
    )
    return kept_pairs, filtered_chosen, filtered_rejected, {
        "split_label": str(split_label),
        "policy": str(nonfinite_feature_policy),
        "num_rows_before": int(len(pairs)),
        "num_bad_rows": int(num_bad),
        "num_rows_after": int(len(kept_pairs)),
        "sample_bad_pair_ids": sample_bad_pair_ids,
    }


def compute_pair_weights(
    *,
    pairs: list[ExternalPairRecord],
    pair_weight_mode: str,
    verdict_counts: dict[str, int] | None = None,
    group_counts: dict[str, int] | None = None,
) -> list[float]:
    """Compute scalar training weights for one pair list.

    English
    -------
    The current ProcessBench-transfer work needs more than plain confidence:
    some supervision families are intentionally auxiliary repairs, not primary
    trust anchors.  `semantic_weight` lets curated artifacts express that
    directly in metadata and keep the weighting logic explicit.

    中文
    ----
    当前这轮 ProcessBench 迁移修复，不能只靠 `pair_confidence`。
    某些监督族被故意设计成“辅助修复信号”，不是“主监督锚点”。
    `semantic_weight` 就是把这个意图显式写进 metadata，再由这里统一进损失。
    """
    if pair_weight_mode not in {
        "none",
        "confidence",
        "semantic",
        "confidence_semantic",
        "verdict_balance",
        "confidence_verdict_balance",
        "group_balance",
        "confidence_group_balance",
    }:
        raise ValueError(f"Unsupported pair_weight_mode: {pair_weight_mode!r}")

    normalized_verdict_counts = dict(verdict_counts or {})
    normalized_group_counts = dict(group_counts or {})
    weights: list[float] = []
    for pair in pairs:
        metadata = dict(pair.metadata or {})
        source_weight = float(metadata.get("source_weight", 1.0))
        if source_weight <= 0.0:
            raise ValueError(
                f"Pair {pair.pair_id!r} carries non-positive source_weight={source_weight}"
            )
        if pair_weight_mode in {"confidence", "confidence_semantic", "confidence_verdict_balance", "confidence_group_balance"}:
            base_weight = float(pair.pair_confidence)
        else:
            base_weight = 1.0
        if pair_weight_mode in {"semantic", "confidence_semantic"}:
            semantic_weight = float(metadata.get("semantic_weight", 1.0))
            if semantic_weight <= 0.0:
                raise ValueError(
                    f"Pair {pair.pair_id!r} carries non-positive semantic_weight={semantic_weight}"
                )
            base_weight *= semantic_weight
        if pair_weight_mode in {"verdict_balance", "confidence_verdict_balance"}:
            verdict = str(metadata.get("chosen_verdict", "")).strip().lower()
            count = int(normalized_verdict_counts.get(verdict, 0))
            if verdict and count > 0:
                base_weight *= float(len(pairs)) / float(len(normalized_verdict_counts) * count)
        if pair_weight_mode in {"group_balance", "confidence_group_balance"}:
            group_label = _resolve_pair_balance_group_label(pair)
            count = int(normalized_group_counts.get(group_label, 0))
            if count > 0:
                base_weight *= float(len(pairs)) / float(len(normalized_group_counts) * count)
        # Source weight is the cleanest way to express:
        # "keep this source in the mixture, but do not let it dominate".
        # 用显式 source weight 表达“保留该 source，但不让它主导训练”，
        # 比偷偷裁样本更可解释，也更容易在日志中复盘。
        weights.append(float(base_weight * source_weight))
    return weights


def compute_pair_route_weights(
    *,
    pairs: list[ExternalPairRecord],
) -> tuple[list[float], list[float]]:
    """Resolve `(local_weight, terminal_weight)` for each pair.

    English
    -------
    This keeps the current Phase E trainer architecture-light but lets
    supervision semantics matter during optimization whenever the head exposes
    separate local and terminal branches.

    中文
    ----
    这里的目标不是把 trainer 改成全新框架，而是给现有 Phase E 增加一层“语义可见”的
    路由能力：如果 head 暴露了 `local/terminal` 两个分支，就让不同 pair 主要更新
    各自对应的分支。
    """
    local_weights: list[float] = []
    terminal_weights: list[float] = []
    for pair in pairs:
        local_weight, terminal_weight = _resolve_pair_training_route_weights(pair)
        local_weights.append(float(local_weight))
        terminal_weights.append(float(terminal_weight))
    return local_weights, terminal_weights


def _resolve_pair_training_route_weights(pair: ExternalPairRecord) -> tuple[float, float]:
    """Map one pair to training-time branch weights.

    Priority:
    1. explicit `metadata.train_route`
    2. `pair_semantics`
    3. conservative default = local-only
    """
    metadata = dict(pair.metadata or {})
    explicit_route = str(metadata.get("train_route", "")).strip().lower()
    if explicit_route == "local":
        return 1.0, 0.0
    if explicit_route == "terminal":
        return 0.0, 1.0
    if explicit_route == "both":
        return 0.5, 0.5

    semantics = str(metadata.get("pair_semantics", "")).strip()
    if semantics == "terminal_completion_anchor":
        return 0.0, 1.0
    if semantics == "good_bad_prefix_grid":
        return 0.5, 0.5
    if semantics in {
        "local_first_bad_edge",
        "local_modified_process_error_step",
        "first_bad_fanout_prefix_ranking",
    }:
        return 1.0, 0.0
    return 1.0, 0.0


def _resolve_pair_balance_group_label(pair: ExternalPairRecord) -> str:
    """Resolve the group label used by group-balance pair weighting.

    Priority:
    1. explicit mixed-artifact source label
    2. pair semantics
    3. source tag
    """
    metadata = dict(pair.metadata or {})
    mixed_label = str(metadata.get("artifact_mix_source_label", "")).strip()
    if mixed_label:
        return mixed_label
    semantics = str(metadata.get("pair_semantics", "")).strip()
    if semantics:
        return semantics
    return str(pair.source_tag)


def build_pair_permutation(
    *,
    pair_cache: dict[str, Any],
    torch_module: Any,
    permutation_mode: str,
    source_balance: str,
) -> Any:
    """Build pair traversal order with optional source balancing.

    English
    -------
    This function determines the order in which training examples are visited.
    That matters when different sources have very different dataset sizes.

    Modes:
    1. `source_balance="none"`
       - treat all pairs as one flat list
    2. `source_balance="uniform"`
       - round-robin across sources so large sources do not monopolize updates

    中文
    ----
    这个函数决定训练时样本被看到的顺序。不同 source 的样本规模差距很大时，
    这个顺序会直接影响优化行为。

    两种模式的直觉是：
    1. `source_balance="none"`
       - 把所有 pair 看成一个大列表
    2. `source_balance="uniform"`
       - 按 source 轮转，防止大 source 在一个 epoch 里过早垄断更新
    """
    num_pairs = int(pair_cache["num_pairs"])
    device = pair_cache["chosen_features"].device
    if num_pairs <= 1:
        return torch_module.arange(num_pairs, device=device)
    if permutation_mode not in {"random", "stable_hash"}:
        raise ValueError(f"Unsupported permutation_mode: {permutation_mode!r}")
    if source_balance not in {"none", "uniform"}:
        raise ValueError(f"Unsupported source_balance: {source_balance!r}")

    pair_ids = list(pair_cache["pair_ids"])
    base_indices = list(range(num_pairs))

    def _order(values: list[int]) -> list[int]:
        if permutation_mode == "random":
            order = torch_module.randperm(len(values)).tolist()
            return [values[pos] for pos in order]
        ordered_ids = [pair_ids[idx] for idx in values]
        return stable_hash_order(values, ids=ordered_ids)

    if source_balance == "none":
        return torch_module.tensor(
            _order(base_indices),
            dtype=torch_module.long,
            device=device,
        )

    buckets: dict[str, list[int]] = {}
    for idx, source_tag in enumerate(pair_cache["source_tags"]):
        buckets.setdefault(str(source_tag), []).append(int(idx))
    for key in list(buckets.keys()):
        buckets[key] = _order(buckets[key])
    ordered: list[int] = []
    pointers = {key: 0 for key in buckets}
    key_order = sorted(buckets)
    # "Uniform" means interleaving sources, not discarding extra rows from large
    # sources.  Large sources still contribute all rows; they are just spread
    # out more evenly through the epoch.
    # `uniform` 指的是不同 source 交错出现，而不是把大 source 直接裁掉。
    # 大 source 的样本仍会全部参与，只是被更均匀地分散在整个 epoch 中。
    while True:
        active = [key for key in key_order if pointers[key] < len(buckets[key])]
        if not active:
            break
        if permutation_mode == "random" and len(active) > 1:
            shuffle_order = torch_module.randperm(len(active)).tolist()
            active = [active[pos] for pos in shuffle_order]
        for key in active:
            ordered.append(buckets[key][pointers[key]])
            pointers[key] += 1
    return torch_module.tensor(ordered, dtype=torch_module.long, device=device)


def compute_external_pair_metrics(
    *,
    chosen_scores: list[float],
    rejected_scores: list[float],
    source_tags: list[str],
) -> dict[str, Any]:
    """Compute external-pair ranking metrics.

    English
    -------
    The central derived quantity is:

    `margin = chosen_score - rejected_score`

    From that we derive:
    1. pair accuracy,
    2. global AUC,
    3. mean/median margin,
    4. per-source margin breakdowns.

    中文
    ----
    这里最核心的派生量是：

    `margin = chosen_score - rejected_score`

    再基于这个 margin 计算：
    1. pair accuracy，
    2. 全局 AUC，
    3. mean/median margin，
    4. 按 source 拆开的 margin 统计。
    """
    if not (
        len(chosen_scores) == len(rejected_scores) == len(source_tags)
    ):
        raise ValueError("chosen_scores/rejected_scores/source_tags must have equal length")
    margins = [float(c - r) for c, r in zip(chosen_scores, rejected_scores, strict=True)]
    by_source: dict[str, list[float]] = {}
    for source_tag, margin in zip(source_tags, margins, strict=True):
        by_source.setdefault(str(source_tag), []).append(float(margin))
    return {
        "num_pairs": int(len(margins)),
        "pair_accuracy": float(sum(1 for m in margins if m > 0.0) / len(margins)),
        "pair_accuracy_with_ties": float(sum(1 for m in margins if m >= 0.0) / len(margins)),
        "auc": float(
            compute_binary_auc(
                scores=[*chosen_scores, *rejected_scores],
                labels=[1] * len(chosen_scores) + [0] * len(rejected_scores),
            )
        ),
        "mean_margin": float(sum(margins) / len(margins)),
        "median_margin": float(_median(margins)),
        "by_source": {
            key: {
                "n": int(len(values)),
                "mean_margin": float(sum(values) / len(values)),
                "positive_margin_rate": float(sum(1 for v in values if v > 0.0) / len(values)),
            }
            for key, values in sorted(by_source.items())
        },
    }


def evaluate_pair_cache(
    *,
    value_head: Any,
    pair_cache: dict[str, Any],
    batch_size: int,
    objective_mode: str,
    ranking_target_space: str,
    ranking_margin: float,
    lambda_ranking: float,
    lambda_bce: float,
    anti_saturation_weight: float,
    anti_saturation_logit_threshold: float,
    reward_centering_weight: float,
    torch_module: Any,
    lambda_terminal_bce: float = 0.0,
) -> tuple[dict[str, Any], list[float], list[float]]:
    """Score one held-out pair cache and compute eval metrics.

    中文
    ----
    评测阶段会复用与训练一致的目标定义，但在 `torch.no_grad()` 下执行并禁用参数更新。
    这样可以保证：
    1. eval loss 与 train loss 语义一致；
    2. 排序指标就是基于模型真实输出分数计算的。
    """
    chosen_features = pair_cache["chosen_features"]
    rejected_features = pair_cache["rejected_features"]
    pair_weights = pair_cache["pair_weights"]
    local_route_weights = pair_cache.get("local_route_weights")
    terminal_route_weights = pair_cache.get("terminal_route_weights")
    source_tags = list(pair_cache["source_tags"])

    head_device = next(value_head.parameters()).device
    head_dtype = next(value_head.parameters()).dtype
    chosen_scores: list[float] = []
    rejected_scores: list[float] = []
    losses: list[float] = []

    with torch_module.no_grad():
        for batch_indices in _batched_index_ranges(int(chosen_features.shape[0]), int(batch_size)):
            chosen_batch = chosen_features[batch_indices]
            rejected_batch = rejected_features[batch_indices]
            weight_batch = pair_weights[batch_indices]
            local_route_batch = local_route_weights[batch_indices] if local_route_weights is not None else None
            terminal_route_batch = (
                terminal_route_weights[batch_indices] if terminal_route_weights is not None else None
            )
            if chosen_batch.device != head_device or chosen_batch.dtype != head_dtype:
                chosen_batch = chosen_batch.to(device=head_device, dtype=head_dtype)
            if rejected_batch.device != head_device or rejected_batch.dtype != head_dtype:
                rejected_batch = rejected_batch.to(device=head_device, dtype=head_dtype)
            if weight_batch.device != head_device or weight_batch.dtype != head_dtype:
                weight_batch = weight_batch.to(device=head_device, dtype=head_dtype)
            if local_route_batch is not None and (
                local_route_batch.device != head_device or local_route_batch.dtype != head_dtype
            ):
                local_route_batch = local_route_batch.to(device=head_device, dtype=head_dtype)
            if terminal_route_batch is not None and (
                terminal_route_batch.device != head_device or terminal_route_batch.dtype != head_dtype
            ):
                terminal_route_batch = terminal_route_batch.to(device=head_device, dtype=head_dtype)
            chosen_out = value_head(chosen_batch)
            rejected_out = value_head(rejected_batch)
            chosen_scores.extend(float(v) for v in chosen_out["scores"].detach().cpu().tolist())
            rejected_scores.extend(float(v) for v in rejected_out["scores"].detach().cpu().tolist())
            batch_loss = compute_pair_objective(
                chosen_logits=chosen_out["logits"],
                rejected_logits=rejected_out["logits"],
                chosen_scores=chosen_out["scores"],
                rejected_scores=rejected_out["scores"],
                pair_weights=weight_batch,
                objective_mode=objective_mode,
                ranking_target_space=ranking_target_space,
                ranking_margin=ranking_margin,
                lambda_ranking=lambda_ranking,
                lambda_bce=lambda_bce,
                anti_saturation_weight=anti_saturation_weight,
                anti_saturation_logit_threshold=anti_saturation_logit_threshold,
                reward_centering_weight=reward_centering_weight,
                chosen_local_logits=chosen_out.get("local_logits"),
                rejected_local_logits=rejected_out.get("local_logits"),
                chosen_local_scores=chosen_out.get("local_scores"),
                rejected_local_scores=rejected_out.get("local_scores"),
                local_pair_weights=(
                    weight_batch * local_route_batch if local_route_batch is not None else None
                ),
                chosen_terminal_logits=chosen_out.get("terminal_logits"),
                rejected_terminal_logits=rejected_out.get("terminal_logits"),
                chosen_terminal_scores=chosen_out.get("terminal_scores"),
                rejected_terminal_scores=rejected_out.get("terminal_scores"),
                terminal_pair_weights=(
                    weight_batch * terminal_route_batch if terminal_route_batch is not None else None
                ),
                lambda_terminal_bce=lambda_terminal_bce,
                torch_module=torch_module,
            )
            losses.append(float(batch_loss.detach().cpu().item()))

    metrics = compute_external_pair_metrics(
        chosen_scores=chosen_scores,
        rejected_scores=rejected_scores,
        source_tags=source_tags,
    )
    metrics["pair_loss"] = float(sum(losses) / len(losses)) if losses else 0.0
    # ranking_score is the suite-friendly single scalar used for checkpoint selection by default.
    # ranking_score 是 suite 默认使用的单值指标，等权综合 pair accuracy 与 AUC。
    metrics["ranking_score"] = float((metrics["pair_accuracy"] + metrics["auc"]) / 2.0)
    return metrics, chosen_scores, rejected_scores


def compute_pair_objective(
    *,
    chosen_logits: Any,
    rejected_logits: Any,
    chosen_scores: Any,
    rejected_scores: Any,
    pair_weights: Any,
    objective_mode: str,
    ranking_target_space: str,
    ranking_margin: float,
    lambda_ranking: float,
    lambda_bce: float,
    anti_saturation_weight: float,
    anti_saturation_logit_threshold: float,
    reward_centering_weight: float,
    chosen_local_logits: Any | None = None,
    rejected_local_logits: Any | None = None,
    chosen_local_scores: Any | None = None,
    rejected_local_scores: Any | None = None,
    local_pair_weights: Any | None = None,
    chosen_terminal_logits: Any | None = None,
    rejected_terminal_logits: Any | None = None,
    chosen_terminal_scores: Any | None = None,
    rejected_terminal_scores: Any | None = None,
    terminal_pair_weights: Any | None = None,
    lambda_terminal_bce: float = 0.0,
    torch_module: Any,
):
    """Compute one external-pair training objective.

    English
    -------
    Supported modes:
    1. `ranking_only`
       - optimize only pair ordering
    2. `pair_bce_only`
       - optimize only chosen=1 / rejected=0 classification
    3. `joint`
       - combine both

    Important implementation detail:
    1. ranking can use either post-sigmoid `scores` or raw `logits`
    2. BCE always uses pre-sigmoid `logits`

    That is why this function receives both representations plus an explicit
    `ranking_target_space` switch.

    中文
    ----
    支持三种目标模式：
    1. `ranking_only`
       - 只优化 pair 排序
    2. `pair_bce_only`
       - 只做 chosen=1 / rejected=0 的二分类监督
    3. `joint`
       - 两者联合

    一个非常容易忽视的实现细节是：
    1. ranking 分支现在可以选择：
       - sigmoid 之后的 `scores`
       - 或 sigmoid 之前的 `logits`
    2. BCE 分支始终使用 sigmoid 之前的 `logits`

    所以这里必须同时接收 logits、scores，以及显式的 `ranking_target_space`。
    """
    dual_head_requested = all(
        item is not None
        for item in (
            chosen_local_logits,
            rejected_local_logits,
            chosen_local_scores,
            rejected_local_scores,
            local_pair_weights,
            chosen_terminal_logits,
            rejected_terminal_logits,
            chosen_terminal_scores,
            rejected_terminal_scores,
            terminal_pair_weights,
        )
    )
    if dual_head_requested:
        # 中文：显式双头时，不应该再把所有监督压到一个混合分数上训练。
        # local 类 pair 走 local head，terminal 类 pair 走 terminal head，
        # 这样才能真正检验“任务分解”而不是“容量变大”。
        # English: once the head exposes explicit local/terminal branches, we
        # should train them with routed supervision rather than collapsing
        # everything onto the blended inference score.
        return _compute_single_pair_objective(
            chosen_logits=chosen_local_logits,
            rejected_logits=rejected_local_logits,
            chosen_scores=chosen_local_scores,
            rejected_scores=rejected_local_scores,
            pair_weights=local_pair_weights,
            objective_mode=objective_mode,
            ranking_target_space=ranking_target_space,
            ranking_margin=ranking_margin,
            lambda_ranking=lambda_ranking,
            lambda_bce=lambda_bce,
            anti_saturation_weight=anti_saturation_weight,
            anti_saturation_logit_threshold=anti_saturation_logit_threshold,
            reward_centering_weight=reward_centering_weight,
            torch_module=torch_module,
        ) + _compute_single_pair_objective(
            chosen_logits=chosen_terminal_logits,
            rejected_logits=rejected_terminal_logits,
            chosen_scores=chosen_terminal_scores,
            rejected_scores=rejected_terminal_scores,
            pair_weights=terminal_pair_weights,
            objective_mode=objective_mode,
            ranking_target_space=ranking_target_space,
            ranking_margin=ranking_margin,
            lambda_ranking=lambda_ranking,
            lambda_bce=lambda_bce,
            anti_saturation_weight=anti_saturation_weight,
            anti_saturation_logit_threshold=anti_saturation_logit_threshold,
            reward_centering_weight=reward_centering_weight,
            torch_module=torch_module,
            terminal_pair_weights_for_bce=terminal_pair_weights,
            lambda_terminal_bce=lambda_terminal_bce,
        )
    return _compute_single_pair_objective(
        chosen_logits=chosen_logits,
        rejected_logits=rejected_logits,
        chosen_scores=chosen_scores,
        rejected_scores=rejected_scores,
        pair_weights=pair_weights,
        objective_mode=objective_mode,
        ranking_target_space=ranking_target_space,
        ranking_margin=ranking_margin,
        lambda_ranking=lambda_ranking,
        lambda_bce=lambda_bce,
        anti_saturation_weight=anti_saturation_weight,
        anti_saturation_logit_threshold=anti_saturation_logit_threshold,
        reward_centering_weight=reward_centering_weight,
        torch_module=torch_module,
        terminal_pair_weights_for_bce=terminal_pair_weights,
        lambda_terminal_bce=lambda_terminal_bce,
    )


def _compute_single_pair_objective(
    *,
    chosen_logits: Any,
    rejected_logits: Any,
    chosen_scores: Any,
    rejected_scores: Any,
    pair_weights: Any,
    objective_mode: str,
    ranking_target_space: str,
    ranking_margin: float,
    lambda_ranking: float,
    lambda_bce: float,
    anti_saturation_weight: float,
    anti_saturation_logit_threshold: float,
    reward_centering_weight: float,
    torch_module: Any,
    terminal_pair_weights_for_bce: Any | None = None,
    lambda_terminal_bce: float = 0.0,
):
    """Compute the original single-head objective.

    This helper is kept separate so dual-head routing can reuse exactly the same
    loss semantics branch-by-branch.
    """
    total = chosen_scores.new_zeros(())
    if objective_mode not in {"ranking_only", "pair_bce_only", "joint"}:
        raise ValueError(f"Unsupported objective_mode: {objective_mode!r}")
    if ranking_target_space == "score":
        ranking_chosen = chosen_scores
        ranking_rejected = rejected_scores
    elif ranking_target_space == "logit":
        ranking_chosen = chosen_logits
        ranking_rejected = rejected_logits
    else:
        raise ValueError(f"Unsupported ranking_target_space: {ranking_target_space!r}")
    if objective_mode in {"ranking_only", "joint"}:
        total = total + float(lambda_ranking) * contrastive_margin_loss(
            ranking_chosen,
            ranking_rejected,
            margin=float(ranking_margin),
            torch_module=torch_module,
            sample_weights=pair_weights,
        )
    if objective_mode in {"pair_bce_only", "joint"}:
        ones = torch_module.ones_like(chosen_logits)
        zeros = torch_module.zeros_like(rejected_logits)
        chosen_loss = binary_cross_entropy_calibration_loss(
            chosen_logits,
            ones,
            torch_module=torch_module,
            sample_weights=pair_weights,
        )
        rejected_loss = binary_cross_entropy_calibration_loss(
            rejected_logits,
            zeros,
            torch_module=torch_module,
            sample_weights=pair_weights,
        )
        total = total + float(lambda_bce) * 0.5 * (chosen_loss + rejected_loss)
    if float(reward_centering_weight) > 0.0:
        # 中文：reward/value 的绝对零点没有稳定语义，因此显式把 batch 均值拉回 0，
        # 能减少 source 间整体抬升/下压带来的迁移漂移。
        # English: reward/value logits are only identifiable up to a shift, so
        # we explicitly keep the batch mean near zero to reduce source-level drift.
        centered_logits = torch_module.cat([chosen_logits, rejected_logits], dim=0)
        centered_weights = torch_module.cat([pair_weights, pair_weights], dim=0)
        total = total + float(reward_centering_weight) * reward_centering_penalty(
            centered_logits,
            torch_module=torch_module,
            sample_weights=centered_weights,
        )
    if float(anti_saturation_weight) > 0.0:
        total = total + float(anti_saturation_weight) * 0.5 * (
            anti_saturation_logit_penalty(
                chosen_logits,
                logit_threshold=float(anti_saturation_logit_threshold),
                torch_module=torch_module,
            )
            + anti_saturation_logit_penalty(
                rejected_logits,
                logit_threshold=float(anti_saturation_logit_threshold),
                torch_module=torch_module,
            )
        )
    if float(lambda_terminal_bce) > 0.0 and terminal_pair_weights_for_bce is not None:
        # English: BiRM-style terminal BCE term.  For terminal anchor pairs the
        # contrastive ranking loss alone is insufficient because the ground truth
        # signal is absolute (a complete correct trajectory should score near 1),
        # not relative.  We add a separate BCE term gated by terminal route weights
        # so it only fires for pairs tagged as terminal_completion_anchor.
        # 中文：对 terminal_completion_anchor 类 pair，单纯的对比 ranking loss 不够，
        # 因为这类 pair 要求模型对完整正确轨迹给出接近 1 的绝对得分，而不只是"比 rejected 高"。
        # 用 terminal route weight 作为 sample weight，让这个 BCE 只作用于 terminal 对。
        ones = torch_module.ones_like(chosen_logits)
        zeros = torch_module.zeros_like(rejected_logits)
        terminal_chosen_loss = binary_cross_entropy_calibration_loss(
            chosen_logits,
            ones,
            torch_module=torch_module,
            sample_weights=terminal_pair_weights_for_bce,
        )
        terminal_rejected_loss = binary_cross_entropy_calibration_loss(
            rejected_logits,
            zeros,
            torch_module=torch_module,
            sample_weights=terminal_pair_weights_for_bce,
        )
        total = total + float(lambda_terminal_bce) * 0.5 * (terminal_chosen_loss + terminal_rejected_loss)
    return total


def select_metric_value(metrics: dict[str, Any], *, selection_metric: str) -> tuple[float, bool]:
    """Resolve one selection metric and whether higher is better.

    中文
    ----
    返回的不只是数值本身，还包含“越大越好还是越小越好”这个布尔标记。
    这样外层 checkpoint 选择逻辑就能统一处理，而不用到处特判 `pair_loss`。
    """
    if selection_metric == "pair_acc":
        return float(metrics["pair_accuracy"]), True
    if selection_metric == "auc":
        return float(metrics["auc"]), True
    if selection_metric == "ranking_score":
        return float(metrics["ranking_score"]), True
    if selection_metric == "pair_loss":
        return float(metrics["pair_loss"]), False
    raise ValueError(f"Unsupported selection_metric: {selection_metric!r}")


def _hash_pair_ids(pairs: list[ExternalPairRecord]) -> dict[str, Any]:
    """Build a deterministic digest for the exact pair list.

    中文
    ----
    这个 digest 的主要用途不是安全，而是缓存签名：
    只要 pair_id 列表变了，缓存命中条件就应该随之变化。
    """
    digest = hashlib.sha256()
    for pair in pairs:
        token = str(pair.pair_id).encode("utf-8")
        digest.update(len(token).to_bytes(4, byteorder="little", signed=False))
        digest.update(token)
    return {"count": int(len(pairs)), "sha256": digest.hexdigest()}


def _batched_index_ranges(total: int, batch_size: int) -> list[slice]:
    """Split one range into contiguous batch slices.

    中文
    ----
    这里返回 `slice` 而不是索引数组，是因为后面对张量切片更直接，也能避免
    一些不必要的中间对象。
    """
    ranges: list[slice] = []
    start = 0
    while start < total:
        stop = min(total, start + max(1, int(batch_size)))
        ranges.append(slice(start, stop))
        start = stop
    return ranges


def _median(values: list[float]) -> float:
    """Return median with a safe fallback for empty input.

    中文
    ----
    空列表时返回 `0.0` 是一个工程兜底，目的是让 summary 生成流程在极端情况下
    也保持稳定，不因为空输入直接崩掉。
    """
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)
