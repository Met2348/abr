"""Unit tests for helper functions in scripts/phase_b_train_sft.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ours.phase_b.contracts import PhaseBTrainRow


def _load_phase_b_train_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_b_train_sft.py"
    spec = importlib.util.spec_from_file_location("phase_b_train_sft", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_config_defaults_requires_dict(tmp_path: Path) -> None:
    module = _load_phase_b_train_module()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

    with pytest.raises(TypeError, match="must contain an object"):
        module._load_config_defaults(path)


def test_parse_args_accepts_train_jsonl_from_config(tmp_path: Path) -> None:
    module = _load_phase_b_train_module()
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "train_jsonl": "assets/artifacts/phase_a_prepared/strategyqa/b0f/train.jsonl",
                "validation_jsonl": "assets/artifacts/phase_a_prepared/strategyqa/b0f/validation.jsonl",
            }
        ),
        encoding="utf-8",
    )

    args = module.parse_args(["--config-json", str(cfg_path)])
    assert args.train_jsonl == Path(
        "assets/artifacts/phase_a_prepared/strategyqa/b0f/train.jsonl"
    )
    assert args.validation_jsonl == Path(
        "assets/artifacts/phase_a_prepared/strategyqa/b0f/validation.jsonl"
    )


def test_build_features_masks_prompt_and_respects_max_seq_length() -> None:
    module = _load_phase_b_train_module()

    class _TokenizerStub:
        eos_token_id = 2

        def __call__(self, text, add_special_tokens=False):
            del add_special_tokens
            # Encode each non-space char as an integer.
            ids = [ord(ch) % 50 + 3 for ch in text if ch != " "]
            return {"input_ids": ids}

    rows = [
        PhaseBTrainRow(
            sample_id="s1",
            dataset="strategyqa",
            split="train",
            prompt_text="ABCD",
            target_text="EF",
            answer="yes",
        )
    ]

    features = module._build_features(
        rows=rows,
        tokenizer=_TokenizerStub(),
        max_seq_length=5,
        target_transform="none",
        target_max_reasoning_lines=2,
        answer_weighting_mode="none",
        reasoning_loss_weight=1.0,
        answer_loss_weight=1.0,
    )
    assert len(features) == 1
    f = features[0]
    assert len(f["input_ids"]) == 5
    assert len(f["labels"]) == 5
    assert len(f["loss_weights"]) == 5
    # At least one supervised label should exist.
    assert any(x != -100 for x in f["labels"])


def test_build_features_short_cot_keeps_last_reasoning_lines() -> None:
    module = _load_phase_b_train_module()

    class _TokenizerStub:
        eos_token_id = None

        def __call__(self, text, add_special_tokens=False):
            del add_special_tokens
            ids = [idx + 10 for idx, _ in enumerate(text.split())]
            return {"input_ids": ids}

    rows = [
        PhaseBTrainRow(
            sample_id="gsm8k:1",
            dataset="gsm8k",
            split="train",
            prompt_text="Question",
            target_text="a\nb\nc\nFinal answer: 5",
            answer="5",
        )
    ]

    features = module._build_features(
        rows=rows,
        tokenizer=_TokenizerStub(),
        max_seq_length=64,
        target_transform="gsm8k_short_cot_last2",
        target_max_reasoning_lines=2,
        answer_weighting_mode="none",
        reasoning_loss_weight=1.0,
        answer_loss_weight=1.0,
    )
    assert len(features) == 1
    # prompt token + 2 kept reasoning lines + 3-token final answer segment
    assert len(features[0]["input_ids"]) == 6


def test_build_features_answer_weighting_boosts_final_answer_tokens() -> None:
    module = _load_phase_b_train_module()

    class _TokenizerStub:
        eos_token_id = None

        def __call__(self, text, add_special_tokens=False):
            del add_special_tokens
            ids = [idx + 20 for idx, _ in enumerate(text.split())]
            return {"input_ids": ids}

    rows = [
        PhaseBTrainRow(
            sample_id="gsm8k:2",
            dataset="gsm8k",
            split="train",
            prompt_text="Question",
            target_text="step one\nstep two\nFinal answer: 7",
            answer="7",
        )
    ]

    features = module._build_features(
        rows=rows,
        tokenizer=_TokenizerStub(),
        max_seq_length=64,
        target_transform="none",
        target_max_reasoning_lines=2,
        answer_weighting_mode="final_answer_line",
        reasoning_loss_weight=0.5,
        answer_loss_weight=3.0,
    )
    weights = features[0]["loss_weights"]
    assert 0.5 in weights
    assert 3.0 in weights


def test_resolve_training_args_tolerates_missing_optional_kwargs() -> None:
    module = _load_phase_b_train_module()

    class _TrainingArgsStub:
        # Deliberately tiny signature: excludes overwrite_output_dir and many others.
        def __init__(
            self,
            output_dir,
            per_device_train_batch_size,
            per_device_eval_batch_size,
            gradient_accumulation_steps,
            learning_rate,
            weight_decay,
            warmup_ratio,
            num_train_epochs,
            max_steps,
            logging_steps,
            save_steps,
            eval_steps,
            save_total_limit,
            max_grad_norm,
            seed,
            report_to,
            remove_unused_columns,
            auto_find_batch_size,
            bf16,
            fp16,
            gradient_checkpointing,
            eval_strategy,
        ):
            self.payload = {
                "output_dir": output_dir,
                "per_device_train_batch_size": per_device_train_batch_size,
                "eval_strategy": eval_strategy,
                "bf16": bf16,
                "fp16": fp16,
                "gradient_checkpointing": gradient_checkpointing,
            }

    args = module.parse_args(
        [
            "--train-jsonl",
            "assets/artifacts/phase_a_prepared/strategyqa/b0f/train.jsonl",
            "--validation-jsonl",
            "assets/artifacts/phase_a_prepared/strategyqa/b0f/validation.jsonl",
        ]
    )
    out = module._resolve_training_args(
        TrainingArguments=_TrainingArgsStub,
        output_dir=Path("tmp/phase_b_test"),
        args=args,
        has_eval=True,
        use_bf16=True,
        use_fp16=False,
    )
    assert out.payload["output_dir"] == "tmp/phase_b_test"
    assert out.payload["eval_strategy"] == "steps"
    assert out.payload["bf16"] is True
