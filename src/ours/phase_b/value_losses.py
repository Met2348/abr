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
"""

from __future__ import annotations

from typing import Any


def mean_squared_calibration_loss(
    predicted_scores: Any,
    target_scores: Any,
    *,
    torch_module: Any,
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
    return torch_module.mean((predicted_scores - target_scores) ** 2)


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
