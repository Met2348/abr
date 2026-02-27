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
    )
    assert len(features) == 1
    f = features[0]
    assert len(f["input_ids"]) == 5
    assert len(f["labels"]) == 5
    # At least one supervised label should exist.
    assert any(x != -100 for x in f["labels"])
