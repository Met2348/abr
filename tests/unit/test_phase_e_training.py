"""Unit tests for Phase E training helpers."""

from __future__ import annotations

import pytest
import torch

from ours.phase_b.value_losses import contrastive_margin_loss
from ours.phase_b.value_losses import binary_cross_entropy_calibration_loss
from ours.phase_d.external_pairs import ExternalPairRecord
from ours.phase_e.training import (
    _resolve_pair_balance_group_label,
    compute_pair_route_weights,
    compute_pair_objective,
    compute_pair_weights,
    compute_pair_truncation_diagnostics,
    compute_text_truncation_diagnostics,
    validate_pair_truncation_diagnostics,
    validate_text_truncation_diagnostics,
)


class _ToyTokenizer:
    """Tiny tokenizer stub that maps each character to one token id."""

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        truncation: bool,
        padding: bool,
        return_attention_mask: bool,
    ) -> dict[str, list[list[int]]]:
        assert add_special_tokens is True
        assert truncation is False
        assert padding is False
        assert return_attention_mask is False
        return {"input_ids": [[ord(ch) for ch in text] for text in texts]}


def test_compute_pair_objective_supports_logit_ranking_space() -> None:
    chosen_logits = torch.tensor([2.0, 0.4], dtype=torch.float32)
    rejected_logits = torch.tensor([1.0, 0.3], dtype=torch.float32)
    chosen_scores = torch.sigmoid(chosen_logits)
    rejected_scores = torch.sigmoid(rejected_logits)
    pair_weights = torch.ones_like(chosen_logits)

    loss = compute_pair_objective(
        chosen_logits=chosen_logits,
        rejected_logits=rejected_logits,
        chosen_scores=chosen_scores,
        rejected_scores=rejected_scores,
        pair_weights=pair_weights,
        objective_mode="ranking_only",
        ranking_target_space="logit",
        ranking_margin=0.5,
        lambda_ranking=1.0,
        lambda_bce=0.0,
        anti_saturation_weight=0.0,
        anti_saturation_logit_threshold=4.0,
        reward_centering_weight=0.0,
        torch_module=torch,
    )
    expected = contrastive_margin_loss(
        chosen_logits,
        rejected_logits,
        margin=0.5,
        torch_module=torch,
        sample_weights=pair_weights,
    )
    assert torch.allclose(loss, expected)


def test_compute_pair_objective_rejects_unknown_ranking_space() -> None:
    logits = torch.tensor([0.0], dtype=torch.float32)
    scores = torch.sigmoid(logits)
    weights = torch.ones_like(logits)

    with pytest.raises(ValueError, match="ranking_target_space"):
        compute_pair_objective(
            chosen_logits=logits,
            rejected_logits=logits,
            chosen_scores=scores,
            rejected_scores=scores,
            pair_weights=weights,
            objective_mode="ranking_only",
            ranking_target_space="mystery",
            ranking_margin=0.0,
            lambda_ranking=1.0,
            lambda_bce=0.0,
            anti_saturation_weight=0.0,
            anti_saturation_logit_threshold=4.0,
            reward_centering_weight=0.0,
            torch_module=torch,
        )


def test_compute_pair_objective_adds_reward_centering_penalty() -> None:
    chosen_logits = torch.tensor([2.0, 2.0], dtype=torch.float32)
    rejected_logits = torch.tensor([2.0, 2.0], dtype=torch.float32)
    chosen_scores = torch.sigmoid(chosen_logits)
    rejected_scores = torch.sigmoid(rejected_logits)
    pair_weights = torch.ones_like(chosen_logits)

    loss = compute_pair_objective(
        chosen_logits=chosen_logits,
        rejected_logits=rejected_logits,
        chosen_scores=chosen_scores,
        rejected_scores=rejected_scores,
        pair_weights=pair_weights,
        objective_mode="ranking_only",
        ranking_target_space="logit",
        ranking_margin=0.0,
        lambda_ranking=0.0,
        lambda_bce=0.0,
        anti_saturation_weight=0.0,
        anti_saturation_logit_threshold=4.0,
        reward_centering_weight=0.5,
        torch_module=torch,
    )
    assert torch.allclose(loss, torch.tensor(2.0, dtype=torch.float32))


def test_compute_pair_weights_supports_confidence_semantic_mode() -> None:
    pairs = [
        ExternalPairRecord(
            pair_id="p0",
            source_tag="math_shepherd",
            domain_tag="general_math",
            prompt_text="Q\n\n",
            chosen_text="good",
            rejected_text="bad",
            pair_confidence=0.8,
            metadata={"semantic_weight": 0.5},
        ),
        ExternalPairRecord(
            pair_id="p1",
            source_tag="math_shepherd",
            domain_tag="general_math",
            prompt_text="Q\n\n",
            chosen_text="good2",
            rejected_text="bad2",
            pair_confidence=0.6,
            metadata={"semantic_weight": 1.25},
        ),
    ]

    weights = compute_pair_weights(
        pairs=pairs,
        pair_weight_mode="confidence_semantic",
    )

    assert weights == pytest.approx([0.4, 0.75])


def test_compute_pair_weights_rejects_non_positive_semantic_weight() -> None:
    pairs = [
        ExternalPairRecord(
            pair_id="p0",
            source_tag="math_shepherd",
            domain_tag="general_math",
            prompt_text="Q\n\n",
            chosen_text="good",
            rejected_text="bad",
            pair_confidence=0.8,
            metadata={"semantic_weight": 0.0},
        )
    ]

    with pytest.raises(ValueError, match="semantic_weight"):
        compute_pair_weights(
            pairs=pairs,
            pair_weight_mode="semantic",
        )


def test_validate_pair_truncation_diagnostics_rejects_hidden_diff_after_cutoff() -> None:
    tokenizer = _ToyTokenizer()
    pairs = [
        ExternalPairRecord(
            pair_id="p0",
            source_tag="r_prm",
            domain_tag="general_math",
            prompt_text="P",
            chosen_text="abcdeX",
            rejected_text="abcdeY",
            pair_confidence=0.9,
        )
    ]
    diagnostics = compute_pair_truncation_diagnostics(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=6,
        batch_size=1,
    )

    with pytest.raises(ValueError, match="hidden_diff_after_cut_fraction=1.0000"):
        validate_pair_truncation_diagnostics(
            diagnostics=diagnostics,
            context_label="toy pair audit",
            max_allowed_over_limit_fraction=0.0,
        )


def test_compute_pair_truncation_diagnostics_reports_per_source_root_cause() -> None:
    tokenizer = _ToyTokenizer()
    pairs = [
        ExternalPairRecord(
            pair_id="legacy_bad",
            source_tag="r_prm_legacy",
            domain_tag="general_math",
            prompt_text="P",
            chosen_text="abcdefX",
            rejected_text="abcdefY",
            pair_confidence=0.9,
        ),
        ExternalPairRecord(
            pair_id="compact_ok",
            source_tag="r_prm_compact",
            domain_tag="general_math",
            prompt_text="P",
            chosen_text="good",
            rejected_text="bad",
            pair_confidence=0.9,
        ),
    ]

    diagnostics = compute_pair_truncation_diagnostics(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=7,
        batch_size=2,
    )

    overall = diagnostics["overall"]
    legacy = diagnostics["by_source"]["r_prm_legacy"]
    compact = diagnostics["by_source"]["r_prm_compact"]

    assert overall["num_pairs"] == 2
    assert overall["frac_pairs_identical_after_truncation"] == pytest.approx(0.5)
    assert overall["frac_pairs_first_diff_after_cutoff"] == pytest.approx(0.5)
    assert legacy["frac_pairs_identical_after_truncation"] == pytest.approx(1.0)
    assert legacy["frac_pairs_first_diff_after_cutoff"] == pytest.approx(1.0)
    assert compact["frac_pairs_identical_after_truncation"] == pytest.approx(0.0)
    assert compact["frac_pairs_first_diff_after_cutoff"] == pytest.approx(0.0)


def test_validate_pair_truncation_diagnostics_rejects_high_over_limit_fraction() -> None:
    tokenizer = _ToyTokenizer()
    pairs = [
        ExternalPairRecord(
            pair_id="p0",
            source_tag="r_prm",
            domain_tag="general_math",
            prompt_text="P",
            chosen_text="goodlongtext",
            rejected_text="badlongtext",
            pair_confidence=0.9,
        )
    ]
    diagnostics = compute_pair_truncation_diagnostics(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=5,
        batch_size=1,
    )

    with pytest.raises(ValueError, match="over_limit_fraction=1.0000 exceeds 0.1000"):
        validate_pair_truncation_diagnostics(
            diagnostics=diagnostics,
            context_label="toy pair audit",
            max_allowed_over_limit_fraction=0.10,
        )


def test_validate_text_truncation_diagnostics_accepts_safe_lengths() -> None:
    tokenizer = _ToyTokenizer()
    diagnostics = compute_text_truncation_diagnostics(
        texts=["short", "tiny"],
        tokenizer=tokenizer,
        max_length=10,
        batch_size=2,
        group_labels=["a", "b"],
    )

    validate_text_truncation_diagnostics(
        diagnostics=diagnostics,
        context_label="toy text audit",
        max_allowed_over_limit_fraction=0.10,
    )


def test_validate_text_truncation_diagnostics_rejects_high_over_limit_fraction() -> None:
    tokenizer = _ToyTokenizer()
    diagnostics = compute_text_truncation_diagnostics(
        texts=["12345678901", "tiny"],
        tokenizer=tokenizer,
        max_length=10,
        batch_size=2,
        group_labels=["long", "short"],
    )

    with pytest.raises(ValueError, match="over_limit_fraction=0.5000 exceeds 0.1000"):
        validate_text_truncation_diagnostics(
            diagnostics=diagnostics,
            context_label="toy text audit",
            max_allowed_over_limit_fraction=0.10,
        )


def test_resolve_pair_balance_group_label_prefers_mixed_artifact_label() -> None:
    row = ExternalPairRecord(
        pair_id="p0",
        source_tag="prmbench_preview",
        domain_tag="general_math",
        prompt_text="P",
        chosen_text="good",
        rejected_text="bad",
        pair_confidence=0.9,
        metadata={
            "artifact_mix_source_label": "terminal_branch",
            "pair_semantics": "terminal_completion_anchor",
        },
    )
    assert _resolve_pair_balance_group_label(row) == "terminal_branch"


def test_resolve_pair_balance_group_label_falls_back_to_pair_semantics() -> None:
    row = ExternalPairRecord(
        pair_id="p1",
        source_tag="math_shepherd",
        domain_tag="general_math",
        prompt_text="P",
        chosen_text="good",
        rejected_text="bad",
        pair_confidence=0.9,
        metadata={"pair_semantics": "first_bad_fanout_prefix_ranking"},
    )
    assert _resolve_pair_balance_group_label(row) == "first_bad_fanout_prefix_ranking"


def test_compute_pair_route_weights_follow_pair_semantics() -> None:
    pairs = [
        ExternalPairRecord(
            pair_id="p_local",
            source_tag="math_shepherd",
            domain_tag="general_math",
            prompt_text="P",
            chosen_text="good",
            rejected_text="bad",
            pair_confidence=0.9,
            metadata={"pair_semantics": "local_first_bad_edge"},
        ),
        ExternalPairRecord(
            pair_id="p_terminal",
            source_tag="prmbench_preview",
            domain_tag="general_math",
            prompt_text="P",
            chosen_text="good",
            rejected_text="bad",
            pair_confidence=0.9,
            metadata={"pair_semantics": "terminal_completion_anchor"},
        ),
        ExternalPairRecord(
            pair_id="p_grid",
            source_tag="prmbench_preview",
            domain_tag="general_math",
            prompt_text="P",
            chosen_text="good",
            rejected_text="bad",
            pair_confidence=0.9,
            metadata={"pair_semantics": "good_bad_prefix_grid"},
        ),
    ]
    local_weights, terminal_weights = compute_pair_route_weights(pairs=pairs)
    assert local_weights == pytest.approx([1.0, 0.0, 0.5])
    assert terminal_weights == pytest.approx([0.0, 1.0, 0.5])


def test_compute_pair_objective_supports_dual_head_routing() -> None:
    chosen_logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    rejected_logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    chosen_scores = torch.sigmoid(chosen_logits)
    rejected_scores = torch.sigmoid(rejected_logits)
    pair_weights = torch.ones_like(chosen_logits)

    local_chosen_logits = torch.tensor([1.0, 0.0], dtype=torch.float32)
    local_rejected_logits = torch.tensor([0.0, 1.0], dtype=torch.float32)
    local_scores = torch.sigmoid(local_chosen_logits)
    local_rejected_scores = torch.sigmoid(local_rejected_logits)
    local_pair_weights = torch.tensor([1.0, 0.0], dtype=torch.float32)

    terminal_chosen_logits = torch.tensor([0.0, 1.0], dtype=torch.float32)
    terminal_rejected_logits = torch.tensor([1.0, 0.0], dtype=torch.float32)
    terminal_scores = torch.sigmoid(terminal_chosen_logits)
    terminal_rejected_scores = torch.sigmoid(terminal_rejected_logits)
    terminal_pair_weights = torch.tensor([0.0, 1.0], dtype=torch.float32)

    loss = compute_pair_objective(
        chosen_logits=chosen_logits,
        rejected_logits=rejected_logits,
        chosen_scores=chosen_scores,
        rejected_scores=rejected_scores,
        pair_weights=pair_weights,
        objective_mode="ranking_only",
        ranking_target_space="logit",
        ranking_margin=0.5,
        lambda_ranking=1.0,
        lambda_bce=0.0,
        anti_saturation_weight=0.0,
        anti_saturation_logit_threshold=4.0,
        reward_centering_weight=0.0,
        chosen_local_logits=local_chosen_logits,
        rejected_local_logits=local_rejected_logits,
        chosen_local_scores=local_scores,
        rejected_local_scores=local_rejected_scores,
        local_pair_weights=local_pair_weights,
        chosen_terminal_logits=terminal_chosen_logits,
        rejected_terminal_logits=terminal_rejected_logits,
        chosen_terminal_scores=terminal_scores,
        rejected_terminal_scores=terminal_rejected_scores,
        terminal_pair_weights=terminal_pair_weights,
        torch_module=torch,
    )
    # Both routed samples already satisfy the margin, so the routed loss should be zero.
    assert torch.allclose(loss, torch.tensor(0.0, dtype=torch.float32))


def test_compute_pair_objective_dual_head_applies_terminal_bce() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    scores = torch.sigmoid(logits)
    pair_weights = torch.ones_like(logits)

    local_weights = torch.tensor([1.0, 0.0], dtype=torch.float32)
    terminal_weights = torch.tensor([0.0, 1.0], dtype=torch.float32)

    loss = compute_pair_objective(
        chosen_logits=logits,
        rejected_logits=logits,
        chosen_scores=scores,
        rejected_scores=scores,
        pair_weights=pair_weights,
        objective_mode="ranking_only",
        ranking_target_space="logit",
        ranking_margin=0.0,
        lambda_ranking=0.0,
        lambda_bce=0.0,
        anti_saturation_weight=0.0,
        anti_saturation_logit_threshold=4.0,
        reward_centering_weight=0.0,
        chosen_local_logits=logits,
        rejected_local_logits=logits,
        chosen_local_scores=scores,
        rejected_local_scores=scores,
        local_pair_weights=local_weights,
        chosen_terminal_logits=logits,
        rejected_terminal_logits=logits,
        chosen_terminal_scores=scores,
        rejected_terminal_scores=scores,
        terminal_pair_weights=terminal_weights,
        lambda_terminal_bce=2.0,
        torch_module=torch,
    )

    expected = 2.0 * 0.5 * (
        binary_cross_entropy_calibration_loss(
            logits,
            torch.ones_like(logits),
            torch_module=torch,
            sample_weights=terminal_weights,
        )
        + binary_cross_entropy_calibration_loss(
            logits,
            torch.zeros_like(logits),
            torch_module=torch,
            sample_weights=terminal_weights,
        )
    )
    assert torch.allclose(loss, expected)
