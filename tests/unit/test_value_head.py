"""Unit tests for the lightweight value head module."""

from __future__ import annotations

import torch

from ours.phase_b.value_head import (
    SigmoidValueHead,
    ValueHeadConfig,
    ensure_tokenizer_has_pad_token,
    maybe_resize_embeddings_for_tokenizer,
)


def test_value_head_linear_forward_shape() -> None:
    config = ValueHeadConfig(hidden_size=16, architecture="linear")
    head = SigmoidValueHead(config)
    features = torch.randn(4, 16)
    out = head(features)
    assert set(out) == {"logits", "scores"}
    assert tuple(out["logits"].shape) == (4,)
    assert tuple(out["scores"].shape) == (4,)


def test_value_head_proj_alias_preserved_for_backward_compatibility() -> None:
    config = ValueHeadConfig(hidden_size=8, architecture="linear")
    head = SigmoidValueHead(config)
    assert head.proj is head.net[-1]
    with torch.no_grad():
        head.proj.weight.fill_(0.25)
        head.proj.bias.fill_(0.5)
    out = head(torch.ones(2, 8))
    assert torch.all(out["scores"] > 0.5)



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
    assert head.proj is head.net[-1]


def test_value_head_gated_mlp_forward_shape() -> None:
    config = ValueHeadConfig(
        hidden_size=16,
        architecture="gated_mlp",
        mlp_hidden_size=32,
        activation="gelu",
        dropout_prob=0.05,
    )
    head = SigmoidValueHead(config)
    features = torch.randn(6, 16)
    out = head(features)
    assert tuple(out["logits"].shape) == (6,)
    assert tuple(out["scores"].shape) == (6,)
    assert torch.all(out["scores"] >= 0.0)
    assert torch.all(out["scores"] <= 1.0)
    assert head.proj is head.final_proj


def test_value_head_dual_head_forward_shape() -> None:
    config = ValueHeadConfig(
        hidden_size=16,
        architecture="dual_head",
        mlp_hidden_size=32,
        activation="gelu",
        dropout_prob=0.05,
        inference_alpha=0.25,
    )
    head = SigmoidValueHead(config)
    features = torch.randn(3, 16)
    out = head(features)
    assert set(out) == {
        "logits",
        "scores",
        "local_logits",
        "local_scores",
        "terminal_logits",
        "terminal_scores",
    }
    assert tuple(out["logits"].shape) == (3,)
    assert tuple(out["local_logits"].shape) == (3,)
    assert tuple(out["terminal_logits"].shape) == (3,)
    expected = 0.25 * out["local_logits"] + 0.75 * out["terminal_logits"]
    assert torch.allclose(out["logits"], expected)


def test_ensure_tokenizer_has_pad_token_reuses_eos_when_available() -> None:
    class _Tokenizer:
        pad_token_id = None
        eos_token = "</s>"
        pad_token = None

        def add_special_tokens(self, payload):
            raise AssertionError(f"unexpected vocab growth: {payload}")

    tokenizer = _Tokenizer()
    synthesized = ensure_tokenizer_has_pad_token(tokenizer)
    assert synthesized is False
    assert tokenizer.pad_token == "</s>"


def test_ensure_tokenizer_has_pad_token_synthesizes_when_needed() -> None:
    class _Tokenizer:
        def __init__(self) -> None:
            self.pad_token_id = None
            self.eos_token = None
            self.add_calls = 0

        def add_special_tokens(self, payload):
            self.add_calls += 1
            assert payload == {"pad_token": "<|pad|>"}

    tokenizer = _Tokenizer()
    synthesized = ensure_tokenizer_has_pad_token(tokenizer)
    assert synthesized is True
    assert tokenizer.add_calls == 1


def test_maybe_resize_embeddings_for_tokenizer_resizes_on_vocab_growth() -> None:
    class _Embeddings:
        num_embeddings = 10

    class _Backbone:
        def __init__(self) -> None:
            self.resize_calls: list[int] = []

        def get_input_embeddings(self):
            return _Embeddings()

        def resize_token_embeddings(self, size: int):
            self.resize_calls.append(int(size))

    class _Tokenizer:
        def __len__(self) -> int:
            return 11

    backbone = _Backbone()
    resized = maybe_resize_embeddings_for_tokenizer(
        backbone=backbone,
        tokenizer=_Tokenizer(),
    )
    assert resized is True
    assert backbone.resize_calls == [11]
