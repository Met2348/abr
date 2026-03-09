"""Unit tests for stratified-sampling helpers in scripts/phase_b_train_value.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_phase_c_train_value_module():
    """Load the Phase C value-head training script as a Python module."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_b_train_value.py"
    spec = importlib.util.spec_from_file_location("phase_b_train_value", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_stratified_train_permutation_keeps_all_indices() -> None:
    """Stratified permutation must keep full coverage with no dropping."""
    module = _load_phase_c_train_value_module()
    train_cache = {
        "clean_features": torch.zeros((6, 3), dtype=torch.float32),
        "has_primary_corruption": torch.tensor(
            [True, True, True, False, True, False], dtype=torch.bool
        ),
        "prefix_step_index": torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
        "primary_corruption_type": [
            "numeric_perturb",
            "step_drop",
            "negation_flip",
            "__none__",
            "comparator_flip",
            "__none__",
        ],
    }
    perm = module._build_stratified_train_permutation(
        train_cache=train_cache,
        torch_module=torch,
        step_bucket_size=2,
        include_no_corruption=True,
    )
    ordered = perm.detach().cpu().tolist()
    assert len(ordered) == 6
    assert sorted(ordered) == [0, 1, 2, 3, 4, 5]


def test_summarize_strata_for_logging_reports_non_empty_stats() -> None:
    """The strata logger helper should return useful aggregate counts."""
    module = _load_phase_c_train_value_module()
    train_cache = {
        "clean_features": torch.zeros((5, 2), dtype=torch.float32),
        "has_primary_corruption": torch.tensor(
            [True, True, False, True, False], dtype=torch.bool
        ),
        "prefix_step_index": torch.tensor([0, 2, 1, 4, 7], dtype=torch.long),
        "primary_corruption_type": [
            "numeric_perturb",
            "step_drop",
            "__none__",
            "entity_substitution",
            "__none__",
        ],
    }
    summary = module._summarize_strata_for_logging(
        train_cache=train_cache,
        step_bucket_size=2,
        include_no_corruption=True,
    )
    assert summary["num_strata"] >= 2
    assert summary["max_size"] >= summary["min_size"]
    assert isinstance(summary["top_strata"], list)


def test_next_external_pair_batch_wraparound_keeps_requested_batch_size() -> None:
    """Wrap-around external sampling should always return exactly batch_size ids."""
    module = _load_phase_c_train_value_module()
    external_cache = {
        "num_pairs": 3,
        "chosen_features": torch.zeros((3, 2), dtype=torch.float32),
        "rejected_features": torch.zeros((3, 2), dtype=torch.float32),
        "pair_weights": torch.ones((3,), dtype=torch.float32),
        "source_tags": ["s0", "s1", "s2"],
        "domain_tags": ["d0", "d0", "d0"],
    }
    perm = torch.tensor([0, 1, 2], dtype=torch.long)
    indices, cursor, out_perm = module._next_external_pair_batch(
        external_pair_cache=external_cache,
        torch_module=torch,
        permutation=perm,
        cursor=0,
        batch_size=8,
        source_balance="none",
        permutation_mode="random",
    )
    assert int(indices.shape[0]) == 8
    assert 0 <= int(cursor) < int(external_cache["num_pairs"])
    assert int(out_perm.shape[0]) == int(external_cache["num_pairs"])
    assert all(0 <= int(i) < 3 for i in indices.detach().cpu().tolist())


def test_next_external_pair_batch_multiple_calls_keep_valid_cursor_and_size() -> None:
    """Repeated calls should keep cursor bounded and return full-size batches."""
    module = _load_phase_c_train_value_module()
    external_cache = {
        "num_pairs": 2,
        "chosen_features": torch.zeros((2, 2), dtype=torch.float32),
        "rejected_features": torch.zeros((2, 2), dtype=torch.float32),
        "pair_weights": torch.ones((2,), dtype=torch.float32),
        "source_tags": ["left", "right"],
        "domain_tags": ["d0", "d1"],
    }
    perm = torch.tensor([0, 1], dtype=torch.long)
    cursor = 0
    for _ in range(5):
        indices, cursor, perm = module._next_external_pair_batch(
            external_pair_cache=external_cache,
            torch_module=torch,
            permutation=perm,
            cursor=cursor,
            batch_size=5,
            source_balance="none",
            permutation_mode="random",
        )
        assert int(indices.shape[0]) == 5
        assert 0 <= int(cursor) < int(external_cache["num_pairs"])
        assert int(perm.shape[0]) == int(external_cache["num_pairs"])


def test_extract_corruption_console_metrics_handles_missing_block() -> None:
    """Console metric extraction should be safe when corruption metrics are absent."""
    module = _load_phase_c_train_value_module()
    pair_acc, auc = module._extract_corruption_console_metrics(
        {"calibration": {"brier_score": 0.2}, "corruption": None}
    )
    assert pair_acc is None
    assert auc is None


def test_extract_corruption_console_metrics_reads_pair_acc_and_auc() -> None:
    """Console metric extraction should return numeric pair_acc and auc when present."""
    module = _load_phase_c_train_value_module()
    pair_acc, auc = module._extract_corruption_console_metrics(
        {
            "corruption": {
                "pair_accuracy": 0.75,
                "auc_clean_vs_corrupt": 0.83,
            }
        }
    )
    assert pair_acc == 0.75
    assert auc == 0.83


def test_initialize_value_head_from_checkpoint_loads_matching_weights(tmp_path) -> None:
    """Warm-start helper should load saved head weights into the new head."""
    module = _load_phase_c_train_value_module()

    init_config = module.ValueHeadConfig(hidden_size=4, dropout_prob=0.0)
    init_head = module.SigmoidValueHead(init_config)
    with torch.no_grad():
        init_head.proj.weight.fill_(0.25)
        init_head.proj.bias.fill_(0.5)

    checkpoint_path = tmp_path / "warm_start.pt"
    module.save_value_head_checkpoint(
        checkpoint_path,
        value_head=init_head,
        config=init_config,
        extra_state={"stage": "stage1"},
    )

    current_config = module.ValueHeadConfig(hidden_size=4, dropout_prob=0.1)
    current_head = module.SigmoidValueHead(current_config)
    with torch.no_grad():
        current_head.proj.weight.zero_()
        current_head.proj.bias.zero_()

    info = module._initialize_value_head_from_checkpoint(
        value_head=current_head,
        current_config=current_config,
        checkpoint_path=checkpoint_path,
    )

    assert info is not None
    assert info["path"] == str(checkpoint_path)
    assert info["checkpoint_extra_state"]["stage"] == "stage1"
    assert info["config_mismatch_notes"]["dropout_prob_changed"] is True
    assert torch.allclose(current_head.proj.weight, init_head.proj.weight)
    assert torch.allclose(current_head.proj.bias, init_head.proj.bias)


def test_filter_corruptions_to_loaded_eval_examples_drops_unloaded_prefixes() -> None:
    """Eval corruption rows should be aligned to the truncated clean eval set."""
    module = _load_phase_c_train_value_module()

    eval_examples = [
        module.ValueSupervisionExample(
            prefix_id="keep_a",
            sample_id="s0",
            dataset="strategyqa",
            split="validation",
            question="q0",
            prompt_text="prompt0",
            prefix_target_text="prefix0",
            current_step_role="reasoning",
            current_step_id="step0",
            prefix_step_index=0,
            num_reasoning_steps_seen=1,
            num_reasoning_steps_total=2,
            target_success_rate=0.5,
            target_q_mean_smoothed=0.5,
            target_q_std_error=0.1,
            target_q_ci_width=0.2,
            target_q_weight=1.0,
            target_q_teacher=None,
            target_q_fused=None,
            target_teacher_available=False,
            target_teacher_disagree=False,
            target_teacher_model_id=None,
            target_parseable_rate=1.0,
            target_k_rollouts=8,
            mean_generated_char_count=42.0,
            metadata={},
            corruption_candidates=[],
            primary_corruption_text=None,
            primary_corruption_type=None,
            primary_corruption_step_index=None,
            primary_pair_delta_q=None,
            primary_pair_z_delta=None,
            primary_pair_weight=None,
            primary_pair_pass_gate=None,
        )
    ]
    eval_corruptions = [
        module.CorruptionVariant(
            corruption_id="c_keep",
            clean_prefix_id="keep_a",
            sample_id="s0",
            dataset="strategyqa",
            split="validation",
            prompt_text="prompt0",
            question="q0",
            clean_prefix_target_text="prefix0",
            corrupted_prefix_text="bad0",
            corruption_type="step_drop",
            corruption_step_index=0,
            current_step_role="reasoning",
            metadata={},
        ),
        module.CorruptionVariant(
            corruption_id="c_drop",
            clean_prefix_id="drop_b",
            sample_id="s1",
            dataset="strategyqa",
            split="validation",
            prompt_text="prompt1",
            question="q1",
            clean_prefix_target_text="prefix1",
            corrupted_prefix_text="bad1",
            corruption_type="step_drop",
            corruption_step_index=1,
            current_step_role="reasoning",
            metadata={},
        ),
    ]

    filtered, stats = module._filter_corruptions_to_loaded_eval_examples(
        eval_examples=eval_examples,
        eval_corruptions=eval_corruptions,
        max_variants=None,
    )

    assert [row.corruption_id for row in filtered] == ["c_keep"]
    assert stats["before_alignment"] == 2
    assert stats["after_alignment"] == 1
    assert stats["dropped_for_missing_clean_prefix"] == 1
