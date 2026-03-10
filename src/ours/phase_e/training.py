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
)
from ours.phase_d.external_pairs import ExternalPairRecord

from .runtime import load_or_encode_text_features, stable_hash_order


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
    use_confidence_weights: bool,
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
        "use_confidence_weights": bool(use_confidence_weights),
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
    weights = []
    for pair in pairs:
        source_weight = float((pair.metadata or {}).get("source_weight", 1.0))
        if source_weight <= 0.0:
            raise ValueError(
                f"Pair {pair.pair_id!r} carries non-positive source_weight={source_weight}"
            )
        base_weight = float(pair.pair_confidence) if use_confidence_weights else 1.0
        # Source weight is the cleanest way to express:
        # "keep this source in the mixture, but do not let it dominate".
        # 用显式 source weight 表达“保留该 source，但不让它主导训练”，
        # 比偷偷裁样本更可解释，也更容易在日志中复盘。
        weights.append(base_weight * source_weight)
    device = chosen_features.device
    return {
        "chosen_features": chosen_features,
        "rejected_features": rejected_features,
        "pair_weights": torch_module.tensor(weights, dtype=chosen_features.dtype, device=device),
        "pair_ids": [pair.pair_id for pair in pairs],
        "source_tags": [pair.source_tag for pair in pairs],
        "num_pairs": int(len(pairs)),
    }


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
    ranking_margin: float,
    lambda_ranking: float,
    lambda_bce: float,
    anti_saturation_weight: float,
    anti_saturation_logit_threshold: float,
    torch_module: Any,
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
            if chosen_batch.device != head_device or chosen_batch.dtype != head_dtype:
                chosen_batch = chosen_batch.to(device=head_device, dtype=head_dtype)
            if rejected_batch.device != head_device or rejected_batch.dtype != head_dtype:
                rejected_batch = rejected_batch.to(device=head_device, dtype=head_dtype)
            if weight_batch.device != head_device or weight_batch.dtype != head_dtype:
                weight_batch = weight_batch.to(device=head_device, dtype=head_dtype)
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
                ranking_margin=ranking_margin,
                lambda_ranking=lambda_ranking,
                lambda_bce=lambda_bce,
                anti_saturation_weight=anti_saturation_weight,
                anti_saturation_logit_threshold=anti_saturation_logit_threshold,
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
    ranking_margin: float,
    lambda_ranking: float,
    lambda_bce: float,
    anti_saturation_weight: float,
    anti_saturation_logit_threshold: float,
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
    1. ranking uses post-sigmoid `scores`
    2. BCE uses pre-sigmoid `logits`

    That is why this function receives both representations.

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
    1. ranking 分支使用 sigmoid 之后的 `scores`
    2. BCE 分支使用 sigmoid 之前的 `logits`

    所以这里必须同时接收 logits 和 scores。
    """
    total = chosen_scores.new_zeros(())
    if objective_mode not in {"ranking_only", "pair_bce_only", "joint"}:
        raise ValueError(f"Unsupported objective_mode: {objective_mode!r}")
    if objective_mode in {"ranking_only", "joint"}:
        total = total + float(lambda_ranking) * contrastive_margin_loss(
            chosen_scores,
            rejected_scores,
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
