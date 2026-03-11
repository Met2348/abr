"""Unit tests for scripts/phase_e_train_value.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ours.phase_b.value_head import SigmoidValueHead, ValueHeadConfig, save_value_head_checkpoint


def _load_phase_e_train_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_e_train_value.py"
    spec = importlib.util.spec_from_file_location("phase_e_train_value", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pair_jsonl(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "pair_id": "p0",
                "source_tag": "math_shepherd",
                "domain_tag": "general_math",
                "prompt_text": "Q\n\n",
                "chosen_text": "good",
                "rejected_text": "bad",
                "pair_confidence": 0.8,
                "quality_flags": {},
                "metadata": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_phase_e_train_parse_args_requires_only_pair_jsonl_and_model(tmp_path: Path) -> None:
    module = _load_phase_e_train_module()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_pair_jsonl(train_path)
    _write_pair_jsonl(eval_path)

    args = module.parse_args(
        [
            "--train-pairs-jsonl",
            str(train_path),
            "--eval-pairs-jsonl",
            str(eval_path),
            "--model-path",
            "assets/models/Qwen2.5-7B-Instruct",
        ]
    )
    assert args.train_pairs_jsonl == train_path
    assert args.eval_pairs_jsonl == eval_path
    assert args.model_path == "assets/models/Qwen2.5-7B-Instruct"


def test_phase_e_train_parse_args_accepts_head_options(tmp_path: Path) -> None:
    module = _load_phase_e_train_module()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_pair_jsonl(train_path)
    _write_pair_jsonl(eval_path)

    args = module.parse_args(
        [
            "--train-pairs-jsonl",
            str(train_path),
            "--eval-pairs-jsonl",
            str(eval_path),
            "--model-path",
            "assets/models/Qwen2.5-7B-Instruct",
            "--ranking-target-space",
            "logit",
            "--head-architecture",
            "mlp",
            "--head-mlp-hidden-size",
            "256",
            "--head-dropout-prob",
            "0.1",
            "--head-activation",
            "relu",
            "--head-inference-alpha",
            "0.35",
            "--reward-centering-weight",
            "0.01",
        ]
    )
    assert args.ranking_target_space == "logit"
    assert args.head_architecture == "mlp"
    assert args.head_mlp_hidden_size == 256
    assert args.head_dropout_prob == 0.1
    assert args.head_activation == "relu"
    assert args.head_inference_alpha == 0.35
    assert args.reward_centering_weight == 0.01


def test_phase_e_train_initialize_value_head_from_checkpoint(tmp_path: Path) -> None:
    module = _load_phase_e_train_module()
    config = ValueHeadConfig(hidden_size=8)
    source_head = SigmoidValueHead(config)
    ckpt_path = tmp_path / "head.pt"
    save_value_head_checkpoint(ckpt_path, value_head=source_head, config=config, extra_state={"epoch": 3})

    target_head = SigmoidValueHead(config)
    info = module._initialize_value_head_from_checkpoint(
        value_head=target_head,
        current_config=config,
        checkpoint_path=ckpt_path,
    )
    assert info is not None
    assert info["path"] == str(ckpt_path)
    assert info["extra_state"]["epoch"] == 3


def test_phase_e_train_initialize_value_head_rejects_config_mismatch(tmp_path: Path) -> None:
    module = _load_phase_e_train_module()
    small_config = ValueHeadConfig(hidden_size=8)
    large_config = ValueHeadConfig(hidden_size=16)
    source_head = SigmoidValueHead(small_config)
    ckpt_path = tmp_path / "head_small.pt"
    save_value_head_checkpoint(ckpt_path, value_head=source_head, config=small_config)

    target_head = SigmoidValueHead(large_config)
    with pytest.raises(ValueError, match="config mismatch"):
        module._initialize_value_head_from_checkpoint(
            value_head=target_head,
            current_config=large_config,
            checkpoint_path=ckpt_path,
        )
