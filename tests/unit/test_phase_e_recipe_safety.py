"""Unit tests for Phase E recipe-risk and collapse diagnostics."""

from __future__ import annotations

from ours.phase_e.recipe_safety import (
    assess_phase_e_recipe_risk,
    diagnose_phase_e_training_health,
    enforce_phase_e_recipe_risk,
)


def test_assess_recipe_risk_flags_known_mixed_terminal_logit_recipe() -> None:
    report = assess_phase_e_recipe_risk(
        train_pair_summary={
            "num_pairs": 1000,
            "by_pair_semantics": {
                "local_first_bad_edge": 600,
                "terminal_completion_anchor": 200,
                "good_bad_prefix_grid": 200,
            },
        },
        train_config={
            "objective_mode": "joint",
            "ranking_target_space": "logit",
            "pair_weight_mode": "confidence_semantic",
            "checkpoint_selection_metric": "ranking_score",
        },
    )
    assert report["max_severity"] in {"high", "critical"}
    codes = {item["code"] for item in report["findings"]}
    assert "ANTI_PATTERN_G_FULL" in codes


def test_assess_recipe_risk_keeps_clean_local_recipe_low_risk() -> None:
    report = assess_phase_e_recipe_risk(
        train_pair_summary={
            "num_pairs": 1000,
            "by_pair_semantics": {
                "local_first_bad_edge": 920,
                "sibling_branch": 80,
            },
        },
        train_config={
            "objective_mode": "ranking_only",
            "ranking_target_space": "score",
            "pair_weight_mode": "none",
            "checkpoint_selection_metric": "pair_acc",
        },
    )
    assert report["max_severity"] == "info"


def test_enforce_recipe_risk_errors_on_dangerous_recipe() -> None:
    report = assess_phase_e_recipe_risk(
        train_pair_summary={
            "num_pairs": 1000,
            "by_pair_semantics": {
                "local_first_bad_edge": 600,
                "terminal_completion_anchor": 400,
            },
        },
        train_config={
            "objective_mode": "joint",
            "ranking_target_space": "logit",
            "pair_weight_mode": "confidence_semantic",
            "checkpoint_selection_metric": "ranking_score",
        },
    )
    try:
        enforce_phase_e_recipe_risk(recipe_risk_report=report, policy="error")
    except ValueError as exc:
        assert "Phase E recipe risk rejected by policy" in str(exc)
    else:
        raise AssertionError("Expected dangerous recipe to be rejected")


def test_diagnose_training_health_detects_flat_collapse() -> None:
    diagnostics = diagnose_phase_e_training_health(
        train_curve=[
            {"train": {"avg_loss": 0.7132}, "eval": {"pair_accuracy": 0.501, "auc": 0.499}},
            {"train": {"avg_loss": 0.7130}, "eval": {"pair_accuracy": 0.500, "auc": 0.500}},
            {"train": {"avg_loss": 0.7131}, "eval": {"pair_accuracy": 0.499, "auc": 0.498}},
        ],
        best_eval_metrics={"pair_accuracy": 0.499, "auc": 0.498, "mean_margin": 0.0002},
        chosen_scores=[0.01, 0.02, 0.00, 0.01],
        rejected_scores=[0.01, 0.02, 0.00, 0.01],
        recipe_risk_report={"max_severity": "critical"},
    )
    assert diagnostics["known_collapse_signature"] is True
    assert diagnostics["diagnosis"] == "collapse_detected"
    assert "known_risky_recipe_combination" in diagnostics["likely_causes"]


def test_diagnose_training_health_marks_progressive_run_as_healthy() -> None:
    diagnostics = diagnose_phase_e_training_health(
        train_curve=[
            {"train": {"avg_loss": 1.10}, "eval": {"pair_accuracy": 0.62, "auc": 0.63}},
            {"train": {"avg_loss": 0.82}, "eval": {"pair_accuracy": 0.71, "auc": 0.74}},
            {"train": {"avg_loss": 0.55}, "eval": {"pair_accuracy": 0.84, "auc": 0.86}},
        ],
        best_eval_metrics={"pair_accuracy": 0.84, "auc": 0.86, "mean_margin": 0.21},
        chosen_scores=[0.81, 0.76, 0.79, 0.83],
        rejected_scores=[0.52, 0.55, 0.49, 0.50],
        recipe_risk_report={"max_severity": "info"},
    )
    assert diagnostics["known_collapse_signature"] is False
    assert diagnostics["diagnosis"] == "healthy_or_undetermined"
