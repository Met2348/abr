"""Unit tests for scripts/phase_e_train_value_lora.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_e_train_value_lora.py"
    spec = importlib.util.spec_from_file_location("phase_e_train_value_lora", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pairs(path: Path) -> None:
    rows = [
        {
            "pair_id": "p0",
            "source_tag": "math_shepherd",
            "domain_tag": "gsm8k_math",
            "prompt_text": "Q\n\n",
            "chosen_text": "Step 1: good",
            "rejected_text": "Step 1: bad",
            "pair_confidence": 0.9,
            "quality_flags": {},
            "metadata": {"pair_semantics": "local_first_bad_edge", "semantic_weight": 1.0},
        }
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_parse_args_accepts_minimal_valid_config(tmp_path: Path) -> None:
    module = _load_module()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_pairs(train_path)
    _write_pairs(eval_path)
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
    assert args.objective_mode == "joint"
    assert args.lora_rank == 4
    assert args.gradient_checkpointing is True


def test_attach_lora_resolves_last_k_layers() -> None:
    module = _load_module()

    class DummyModel:
        pass

    captured = {}

    class DummyCfg:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_get_peft_model(model, cfg):
        captured["cfg_obj"] = cfg
        return model

    import peft

    original_lora_config = peft.LoraConfig
    original_get_peft_model = peft.get_peft_model
    original_task_type = peft.TaskType
    try:
        peft.LoraConfig = DummyCfg  # type: ignore[assignment]
        peft.get_peft_model = fake_get_peft_model  # type: ignore[assignment]
        model, spec = module._attach_lora(
            model=DummyModel(),
            num_hidden_layers=28,
            target_modules=["q_proj", "v_proj"],
            rank=4,
            alpha=16,
            dropout=0.05,
            top_k_layers=4,
        )
    finally:
        peft.LoraConfig = original_lora_config  # type: ignore[assignment]
        peft.get_peft_model = original_get_peft_model  # type: ignore[assignment]
        peft.TaskType = original_task_type  # type: ignore[assignment]

    assert spec["layers_to_transform"] == [24, 25, 26, 27]
    assert captured["layers_to_transform"] == [24, 25, 26, 27]
    assert captured["target_modules"] == ["q_proj", "v_proj"]
