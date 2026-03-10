"""Unit tests for the lightweight value head module."""

from __future__ import annotations

import torch

from ours.phase_b.value_head import SigmoidValueHead, ValueHeadConfig


def test_value_head_linear_forward_shape() -> None:
    config = ValueHeadConfig(hidden_size=16, architecture="linear")
    head = SigmoidValueHead(config)
    features = torch.randn(4, 16)
    out = head(features)
    assert set(out) == {"logits", "scores"}
    assert tuple(out["logits"].shape) == (4,)
    assert tuple(out["scores"].shape) == (4,)



def test_value_head_mlp_forward_shape() -> None:
    config = ValueHeadConfig(
        hidden_size=16,
        architecture="mlp",
        mlp_hidden_size=32,
        activation="relu",
        dropout_prob=0.1,
    )
    head = SigmoidValueHead(config)
    features = torch.randn(5, 16)
    out = head(features)
    assert tuple(out["logits"].shape) == (5,)
    assert tuple(out["scores"].shape) == (5,)
    assert torch.all(out["scores"] >= 0.0)
    assert torch.all(out["scores"] <= 1.0)
