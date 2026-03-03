"""Loss helpers for Phase C value-head training.

Why this file exists
--------------------
C2 should keep its loss definitions explicit and testable. Hiding calibration or
contrastive objectives inside one long training loop makes it harder to verify
that the value head is learning the intended signal.

This module starts with the conservative C2 losses:
- rollout-target calibration loss,
- corruption contrastive margin loss,
- Bellman consistency helper for the later BCR-lite stage.

As C2 diagnostics expanded, this module also added a BCE-with-logits calibration
path and a mixed calibration helper. These are useful when MSE-only calibration
is too weak for prefix targets concentrated near 0/1.
"""

from __future__ import annotations

from typing import Any


def binary_cross_entropy_calibration_loss(
    predicted_logits: Any,
    target_scores: Any,
    *,
    torch_module: Any,
    pos_weight: float | None = None,
    sample_weights: Any | None = None,
):
    """Return BCE-with-logits loss between predicted logits and rollout targets.

    Why this loss exists
    --------------------
    MSE treats all score-space errors uniformly. In our C2 setting, some prefixes
    are near-extreme targets (close to 0 or 1), and BCE can produce a sharper
    calibration signal for those regions while still supporting soft targets.

    Parameters
    ----------
    predicted_logits:
        Raw logits from the value head (before sigmoid).
    target_scores:
        Target success rates in `[0, 1]`.
    pos_weight:
        Optional positive-class weight, passed to
        `binary_cross_entropy_with_logits`. Keep this close to `1.0` unless
        there is a clear positive/negative imbalance reason.

    Example
    -------
    ```python
    loss = binary_cross_entropy_calibration_loss(
        logits, target, torch_module=torch, pos_weight=1.0
    )
    ```
    """
    if predicted_logits.shape != target_scores.shape:
        raise ValueError(
            "BCE calibration loss expects equal shapes, got "
            f"{tuple(predicted_logits.shape)!r} and {tuple(target_scores.shape)!r}"
        )
    if pos_weight is not None and float(pos_weight) <= 0.0:
        raise ValueError("`pos_weight` must be > 0 when provided")
    if target_scores.numel() == 0:
        raise ValueError("BCE calibration loss expects a non-empty tensor")

    weight_tensor = None
    if pos_weight is not None:
        # BCEWithLogitsLoss expects a tensor on the same device/dtype.
        weight_tensor = torch_module.tensor(
            float(pos_weight),
            dtype=predicted_logits.dtype,
            device=predicted_logits.device,
        )

    raw = torch_module.nn.functional.binary_cross_entropy_with_logits(
        predicted_logits,
        target_scores,
        reduction="none",
        pos_weight=weight_tensor,
    )
    return _weighted_mean(raw, sample_weights=sample_weights, torch_module=torch_module)


def mean_squared_calibration_loss(
    predicted_scores: Any,
    target_scores: Any,
    *,
    torch_module: Any,
    sample_weights: Any | None = None,
):
    """Return MSE between predicted value scores and rollout targets.

    Example
    -------
    ```python
    loss = mean_squared_calibration_loss(pred, target, torch_module=torch)
    ```
    """
    if predicted_scores.shape != target_scores.shape:
        raise ValueError(
            f"Calibration loss expects equal shapes, got {tuple(predicted_scores.shape)!r} and {tuple(target_scores.shape)!r}"
        )
    raw = (predicted_scores - target_scores) ** 2
    return _weighted_mean(raw, sample_weights=sample_weights, torch_module=torch_module)


def mixed_calibration_loss(
    predicted_logits: Any,
    predicted_scores: Any,
    target_scores: Any,
    *,
    torch_module: Any,
    bce_weight: float,
    mse_weight: float,
    bce_pos_weight: float | None = None,
    sample_weights: Any | None = None,
):
    """Return a weighted sum of BCE and MSE calibration losses.

    This helper keeps the combination explicit in one place so C2 logs can
    report exactly what objective is used.
    """
    if float(bce_weight) < 0.0 or float(mse_weight) < 0.0:
        raise ValueError("`bce_weight` and `mse_weight` must be >= 0")
    if float(bce_weight) == 0.0 and float(mse_weight) == 0.0:
        raise ValueError("At least one of `bce_weight` or `mse_weight` must be > 0")
    bce = binary_cross_entropy_calibration_loss(
        predicted_logits,
        target_scores,
        torch_module=torch_module,
        pos_weight=bce_pos_weight,
        sample_weights=sample_weights,
    )
    mse = mean_squared_calibration_loss(
        predicted_scores,
        target_scores,
        torch_module=torch_module,
        sample_weights=sample_weights,
    )
    return float(bce_weight) * bce + float(mse_weight) * mse


def contrastive_margin_loss(
    clean_scores: Any,
    corrupted_scores: Any,
    *,
    margin: float,
    torch_module: Any,
):
    """Encourage clean prefixes to score above corrupted ones.

    The loss is:
    `max(0, margin - clean + corrupt)`
    """
    if clean_scores.shape != corrupted_scores.shape:
        raise ValueError(
            f"Contrastive loss expects equal shapes, got {tuple(clean_scores.shape)!r} and {tuple(corrupted_scores.shape)!r}"
        )
    if margin < 0:
        raise ValueError("`margin` must be non-negative")
    return torch_module.relu(float(margin) - clean_scores + corrupted_scores).mean()


def bellman_consistency_loss(
    current_scores: Any,
    next_scores_stopgrad: Any,
    *,
    rewards: Any | None,
    gamma: float,
    torch_module: Any,
):
    """Return a Bellman-style consistency loss for later BCR-lite stages.

    `next_scores_stopgrad` should already be detached by the caller.
    C2 does not use this loss yet, but defining it now keeps the later stage
    consistent with the current contracts.
    """
    if current_scores.shape != next_scores_stopgrad.shape:
        raise ValueError(
            f"Bellman loss expects equal shapes, got {tuple(current_scores.shape)!r} and {tuple(next_scores_stopgrad.shape)!r}"
        )
    if not (0.0 <= float(gamma) <= 1.0):
        raise ValueError("`gamma` must be in [0, 1]")
    if rewards is None:
        rewards = torch_module.zeros_like(current_scores)
    if rewards.shape != current_scores.shape:
        raise ValueError(
            f"Bellman loss expects rewards with shape {tuple(current_scores.shape)!r}, got {tuple(rewards.shape)!r}"
        )
    target = rewards + float(gamma) * next_scores_stopgrad
    return torch_module.mean((current_scores - target) ** 2)


def _weighted_mean(
    values: Any,
    *,
    sample_weights: Any | None,
    torch_module: Any,
):
    """Return weighted or unweighted mean for per-sample loss values.

    `sample_weights` are expected in `[0, +inf)` with the same shape as `values`.
    A tiny epsilon is added to the denominator to avoid divide-by-zero when all
    weights are exactly zero after filtering.
    """
    if sample_weights is None:
        return torch_module.mean(values)
    if values.shape != sample_weights.shape:
        raise ValueError(
            "Sample-weighted loss expects equal shapes, got "
            f"{tuple(values.shape)!r} and {tuple(sample_weights.shape)!r}"
        )
    if bool((sample_weights < 0).any().item()):
        raise ValueError("`sample_weights` must be non-negative")
    denom = torch_module.sum(sample_weights).clamp_min(1e-8)
    return torch_module.sum(values * sample_weights) / denom
