"""Compute calibration and corruption metrics for Phase C value heads.

Why this file exists
--------------------
C2 is not judged by answer accuracy directly. It is judged by whether the value
head produces a useful prefix-level signal. That requires dedicated metrics for:
- calibration against rollout targets,
- corruption sensitivity,
- value margins on clean vs corrupted prefixes.

This module keeps those metrics explicit, deterministic, and testable.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, median
from typing import Any


def compute_calibration_summary(
    predicted_scores: list[float],
    target_scores: list[float],
    *,
    reference_mean: float,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Compute the main C2 calibration metrics.

    Metrics include:
    - Brier score / MSE
    - RMSE
    - MAE
    - Pearson correlation
    - ECE-like bin mismatch between predicted score and empirical target
    - improvement over a constant baseline predictor
    """
    _validate_equal_non_empty(predicted_scores, target_scores, "calibration")
    if not (0.0 <= reference_mean <= 1.0):
        raise ValueError("`reference_mean` must lie in [0, 1]")
    if num_bins <= 0:
        raise ValueError("`num_bins` must be positive")

    # 这里的 Brier/MAE/RMSE 评估“概率数值是否靠谱”，
    # Pearson 评估“排序相关性”，ECE 评估“分箱后的校准偏差”。
    n = len(predicted_scores)
    sq_errors = [(p - t) ** 2 for p, t in zip(predicted_scores, target_scores, strict=True)]
    abs_errors = [abs(p - t) for p, t in zip(predicted_scores, target_scores, strict=True)]
    baseline_sq_errors = [(reference_mean - t) ** 2 for t in target_scores]

    brier = mean(sq_errors)
    baseline_brier = mean(baseline_sq_errors)
    rmse = math.sqrt(brier)
    mae = mean(abs_errors)
    corr = _pearson(predicted_scores, target_scores)
    ece = _expected_calibration_error(predicted_scores, target_scores, num_bins=num_bins)

    return {
        "n": n,
        "target_mean": float(mean(target_scores)),
        "prediction_mean": float(mean(predicted_scores)),
        "brier_score": float(brier),
        "rmse": float(rmse),
        "mae": float(mae),
        "pearson": float(corr),
        "ece": float(ece),
        "baseline_mean_predictor": float(reference_mean),
        "baseline_brier_score": float(baseline_brier),
        "brier_improvement_vs_baseline": float(baseline_brier - brier),
    }


def compute_corruption_summary(
    clean_scores: list[float],
    corrupted_scores: list[float],
    *,
    corruption_types: list[str],
    corruption_step_indices: list[int],
) -> dict[str, Any]:
    """Compute faithfulness metrics over clean/corrupted prefix pairs."""
    _validate_equal_non_empty(clean_scores, corrupted_scores, "corruption")
    if len(clean_scores) != len(corruption_types) or len(clean_scores) != len(corruption_step_indices):
        raise ValueError("Corruption metadata must align one-to-one with score pairs")

    margins = [clean - corrupt for clean, corrupt in zip(clean_scores, corrupted_scores, strict=True)]
    pair_accuracy = sum(1 for margin in margins if margin > 0.0) / len(margins)
    pair_accuracy_with_ties = sum(1 for margin in margins if margin >= 0.0) / len(margins)
    auc = compute_binary_auc(
        scores=list(clean_scores) + list(corrupted_scores),
        labels=[1] * len(clean_scores) + [0] * len(corrupted_scores),
    )

    by_type: dict[str, dict[str, Any]] = {}
    grouped_type: dict[str, list[float]] = defaultdict(list)
    for corruption_type, margin in zip(corruption_types, margins, strict=True):
        grouped_type[corruption_type].append(float(margin))
    for corruption_type, rows in grouped_type.items():
        by_type[corruption_type] = {
            "n": len(rows),
            "mean_margin": float(mean(rows)),
            "median_margin": float(median(rows)),
            "positive_margin_rate": float(sum(1 for margin in rows if margin > 0.0) / len(rows)),
        }

    by_step_index: dict[str, dict[str, Any]] = {}
    grouped_step: dict[int, list[float]] = defaultdict(list)
    for step_index, margin in zip(corruption_step_indices, margins, strict=True):
        grouped_step[int(step_index)].append(float(margin))
    for step_index, rows in sorted(grouped_step.items()):
        by_step_index[str(step_index)] = {
            "n": len(rows),
            "mean_margin": float(mean(rows)),
            "positive_margin_rate": float(sum(1 for margin in rows if margin > 0.0) / len(rows)),
        }

    return {
        "n_pairs": len(margins),
        "pair_accuracy": float(pair_accuracy),
        "pair_accuracy_with_ties": float(pair_accuracy_with_ties),
        "mean_margin": float(mean(margins)),
        "median_margin": float(median(margins)),
        "auc_clean_vs_corrupt": float(auc),
        "by_corruption_type": by_type,
        "by_corruption_step_index": by_step_index,
    }


def compute_binary_auc(scores: list[float], labels: list[int]) -> float:
    """Compute binary AUC without external dependencies.

    Returns `0.5` when the label set does not contain both classes.
    """
    if len(scores) != len(labels):
        raise ValueError("`scores` and `labels` must have equal length")
    if not scores:
        raise ValueError("AUC requires at least one example")
    positives = sum(1 for label in labels if int(label) == 1)
    negatives = sum(1 for label in labels if int(label) == 0)
    if positives == 0 or negatives == 0:
        return 0.5

    ranked = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(ranked):
        j = idx
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[idx][0]:
            j += 1
        avg_rank = (idx + j + 2) / 2.0  # 1-based average rank
        positives_in_block = sum(1 for _, label in ranked[idx : j + 1] if int(label) == 1)
        rank_sum += positives_in_block * avg_rank
        idx = j + 1

    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def render_faithfulness_summary_markdown(
    *,
    title: str,
    calibration: dict[str, Any],
    corruption: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Render a compact Markdown report for saved Phase C evaluations."""
    lines = [f"# {title}", ""]
    if metadata:
        for key, value in metadata.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    lines.extend(
        [
            "## Calibration",
            f"- n: `{calibration['n']}`",
            f"- prediction_mean: `{calibration['prediction_mean']:.4f}`",
            f"- target_mean: `{calibration['target_mean']:.4f}`",
            f"- brier_score: `{calibration['brier_score']:.6f}`",
            f"- baseline_brier_score: `{calibration['baseline_brier_score']:.6f}`",
            f"- brier_improvement_vs_baseline: `{calibration['brier_improvement_vs_baseline']:.6f}`",
            f"- rmse: `{calibration['rmse']:.6f}`",
            f"- mae: `{calibration['mae']:.6f}`",
            f"- pearson: `{calibration['pearson']:.6f}`",
            f"- ece: `{calibration['ece']:.6f}`",
        ]
    )
    if corruption is not None:
        lines.extend(
            [
                "",
                "## Corruption Sensitivity",
                f"- n_pairs: `{corruption['n_pairs']}`",
                f"- pair_accuracy: `{corruption['pair_accuracy']:.6f}`",
                f"- pair_accuracy_with_ties: `{corruption['pair_accuracy_with_ties']:.6f}`",
                f"- mean_margin: `{corruption['mean_margin']:.6f}`",
                f"- median_margin: `{corruption['median_margin']:.6f}`",
                f"- auc_clean_vs_corrupt: `{corruption['auc_clean_vs_corrupt']:.6f}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        raise ValueError("Pearson correlation requires equal non-empty lists")
    mean_x = mean(xs)
    mean_y = mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return float(num / (den_x * den_y))


def _expected_calibration_error(
    predicted_scores: list[float],
    target_scores: list[float],
    *,
    num_bins: int,
) -> float:
    # ECE 的 bin = 预测概率区间桶；num_bins 越大，分辨率越高但方差也更大。
    bins: list[list[tuple[float, float]]] = [[] for _ in range(num_bins)]
    for prediction, target in zip(predicted_scores, target_scores, strict=True):
        clipped = min(max(float(prediction), 0.0), 1.0)
        bin_idx = min(int(clipped * num_bins), num_bins - 1)
        bins[bin_idx].append((clipped, float(target)))

    total = len(predicted_scores)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        bucket_weight = len(bucket) / total
        bucket_pred_mean = mean(pair[0] for pair in bucket)
        bucket_target_mean = mean(pair[1] for pair in bucket)
        ece += bucket_weight * abs(bucket_pred_mean - bucket_target_mean)
    return float(ece)


def _validate_equal_non_empty(xs: list[float], ys: list[float], label: str) -> None:
    if len(xs) != len(ys):
        raise ValueError(f"{label}: expected equal-length lists, got {len(xs)} and {len(ys)}")
    if not xs:
        raise ValueError(f"{label}: expected a non-empty list")
