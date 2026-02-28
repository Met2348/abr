"""Unit tests for lightweight helpers in phase_a_generate_and_eval script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


def _load_phase_a_generate_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_a_generate_and_eval.py"
    spec = importlib.util.spec_from_file_location("phase_a_generate_and_eval", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclass internals expect module to be present in sys.modules during exec.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_generation_binary_choice_writes_metadata_without_token_tensor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_phase_a_generate_module()

    # Force binary-choice predictor to avoid model/tokenizer dependency in unit test.
    monkeypatch.setattr(
        module,
        "_predict_strategyqa_binary_choice",
        lambda **_: "yes",
    )

    rows = [
        {
            "sample_id": "strategyqa:q1",
            "dataset": "strategyqa",
            "split": "validation",
            "prompt_text": "[SYSTEM]\\n...\\n[USER]\\nQ\\n[ASSISTANT]\\n",
            "answer": "yes",
            "question": "Is water wet?",
            "template_id": "qa_direct",
            "template_version": "1.0.0",
        }
    ]

    output_path = tmp_path / "predictions.jsonl"
    source_path = tmp_path / "prepared.jsonl"
    source_path.write_text("{}", encoding="utf-8")

    class _TokenizerStub:
        pad_token_id = 0
        eos_token_id = 1

    module._run_generation(
        rows=rows,
        model=object(),
        tokenizer=_TokenizerStub(),
        gen_cfg=module.GenerationConfig(max_new_tokens=16, do_sample=False),
        output_path=output_path,
        source_path=source_path,
        torch_module=object(),
        log_every=1,
        max_progress_lines=2,
        strategyqa_decode_mode="binary_choice",
        truncate_chat_markers=True,
        batch_size=1,
        oom_backoff=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["raw_prediction"] == "yes"
    assert payload["metadata"]["generated_tokens"] == 0
    assert payload["metadata"]["hit_token_limit"] is False


def test_run_generation_freeform_batch_preserves_row_order_and_metadata(
    tmp_path: Path,
) -> None:
    module = _load_phase_a_generate_module()

    class _TokenBatch(dict):
        def to(self, _device):
            return self

    class _TokenizerStub:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, prompts, return_tensors="pt", padding=True):
            if isinstance(prompts, str):
                prompts = [prompts]
            seqs = []
            for i, _ in enumerate(prompts):
                # Make variable prompt lengths to exercise padding path.
                seqs.append([10] * (2 + i))
            max_len = max(len(s) for s in seqs)
            ids: list[list[int]] = []
            mask: list[list[int]] = []
            for s in seqs:
                pad = max_len - len(s)
                ids.append(s + [0] * pad)
                mask.append([1] * len(s) + [0] * pad)
            return _TokenBatch(
                {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(mask, dtype=torch.long),
                }
            )

        def decode(self, token_ids, skip_special_tokens=True):
            vals = [int(x) for x in token_ids.tolist() if int(x) not in {0, 2}]
            return " ".join(str(v) for v in vals)

    class _ModelStub:
        device = torch.device("cpu")

        def generate(self, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            bs = int(input_ids.shape[0])
            in_len = int(input_ids.shape[1])
            out = torch.full((bs, in_len + 2), fill_value=0, dtype=torch.long)
            out[:, :in_len] = input_ids
            for i in range(bs):
                if i % 2 == 0:
                    out[i, in_len : in_len + 2] = torch.tensor([11, 2], dtype=torch.long)
                else:
                    out[i, in_len] = 12
            return out

    rows = [
        {
            "sample_id": "gsm8k:r1",
            "dataset": "gsm8k",
            "split": "validation",
            "prompt_text": "P1",
            "answer": "1",
            "question": "Q1",
            "template_id": "qa_direct",
            "template_version": "1.0.0",
        },
        {
            "sample_id": "gsm8k:r2",
            "dataset": "gsm8k",
            "split": "validation",
            "prompt_text": "P2",
            "answer": "2",
            "question": "Q2",
            "template_id": "qa_direct",
            "template_version": "1.0.0",
        },
        {
            "sample_id": "gsm8k:r3",
            "dataset": "gsm8k",
            "split": "validation",
            "prompt_text": "P3",
            "answer": "3",
            "question": "Q3",
            "template_id": "qa_direct",
            "template_version": "1.0.0",
        },
    ]

    output_path = tmp_path / "predictions_batch.jsonl"
    source_path = tmp_path / "prepared.jsonl"
    source_path.write_text("{}", encoding="utf-8")

    stats = module._run_generation(
        rows=rows,
        model=_ModelStub(),
        tokenizer=_TokenizerStub(),
        gen_cfg=module.GenerationConfig(max_new_tokens=4, do_sample=False),
        output_path=output_path,
        source_path=source_path,
        torch_module=torch,
        log_every=1,
        max_progress_lines=3,
        strategyqa_decode_mode="freeform",
        truncate_chat_markers=True,
        batch_size=2,
        oom_backoff=True,
    )
    assert stats.num_samples == 3
    assert stats.batch_size == 2

    rows_out = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r["sample_id"] for r in rows_out] == ["gsm8k:r1", "gsm8k:r2", "gsm8k:r3"]
    assert [r["metadata"]["row_index"] for r in rows_out] == [0, 1, 2]
    # From model stub pattern: batch local even -> "11", odd -> "12".
    assert [r["raw_prediction"] for r in rows_out] == ["11", "12", "11"]
    assert [r["metadata"]["generated_tokens"] for r in rows_out] == [2, 1, 2]


def test_generate_freeform_rows_once_uses_left_padding_and_restores_tokenizer_side() -> None:
    module = _load_phase_a_generate_module()

    class _TokenBatch(dict):
        def to(self, _device):
            return self

    class _TokenizerStub:
        pad_token_id = 0
        eos_token_id = 2

        def __init__(self) -> None:
            self.padding_side = "right"
            self.seen_padding_sides: list[str] = []

        def __call__(self, prompts, return_tensors="pt", padding=True):
            del return_tensors, padding
            self.seen_padding_sides.append(self.padding_side)
            # Left-padding should be active inside this call.
            if self.padding_side == "left":
                ids = [[0, 10], [11, 12]]
                mask = [[0, 1], [1, 1]]
            else:
                ids = [[10, 0], [11, 12]]
                mask = [[1, 0], [1, 1]]
            return _TokenBatch(
                {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(mask, dtype=torch.long),
                }
            )

        def decode(self, token_ids, skip_special_tokens=True):
            del skip_special_tokens
            vals = [int(x) for x in token_ids.tolist() if int(x) not in {0, 2}]
            return " ".join(str(v) for v in vals)

    class _ModelStub:
        device = torch.device("cpu")

        def generate(self, input_ids, attention_mask, **kwargs):
            del attention_mask, kwargs
            bs = int(input_ids.shape[0])
            in_len = int(input_ids.shape[1])
            out = torch.full((bs, in_len + 1), fill_value=0, dtype=torch.long)
            out[:, :in_len] = input_ids
            out[:, in_len] = 13
            return out

    tok = _TokenizerStub()
    outputs = module._generate_freeform_rows_once(
        prompts=["a", "b"],
        model=_ModelStub(),
        tokenizer=tok,
        gen_cfg=module.GenerationConfig(max_new_tokens=4, do_sample=False),
        pad_id=tok.pad_token_id,
        truncate_chat_markers=True,
        torch_module=torch,
    )
    assert tok.seen_padding_sides == ["left"]
    assert tok.padding_side == "right"
    assert len(outputs) == 2
    assert all(item.raw_prediction == "13" for item in outputs)


def test_needs_truncation_recovery_requires_hit_cap_and_missing_signal() -> None:
    module = _load_phase_a_generate_module()

    cfg = module.TruncationRecoveryConfig(
        enabled=True,
        max_rounds=2,
        extra_tokens_per_round=32,
        datasets=("gsm8k",),
        require_final_answer_signal=True,
    )
    needs = module._needs_truncation_recovery(
        dataset="gsm8k",
        result=module.FreeformGenerationResult(
            raw_prediction="Reasoning without final line",
            generated_token_count=64,
            hit_token_limit=True,
        ),
        truncation_recovery_cfg=cfg,
    )
    assert needs is True

    no_need = module._needs_truncation_recovery(
        dataset="gsm8k",
        result=module.FreeformGenerationResult(
            raw_prediction="Final answer: 42",
            generated_token_count=64,
            hit_token_limit=True,
        ),
        truncation_recovery_cfg=cfg,
    )
    assert no_need is False


def test_apply_truncation_recovery_appends_continuation(monkeypatch) -> None:
    module = _load_phase_a_generate_module()

    rows = [
        {
            "sample_id": "gsm8k:r1",
            "dataset": "gsm8k",
            "split": "validation",
            "prompt_text": "Q\\n[ASSISTANT]\\n",
            "answer": "7",
            "question": "q",
        }
    ]
    outputs = [
        module.FreeformGenerationResult(
            raw_prediction="working",
            generated_token_count=32,
            hit_token_limit=True,
        )
    ]

    def _fake_generate_once(**kwargs):
        del kwargs
        return [
            module.FreeformGenerationResult(
                raw_prediction="Final answer: 7",
                generated_token_count=5,
                hit_token_limit=False,
            )
        ]

    monkeypatch.setattr(module, "_generate_freeform_rows_once", _fake_generate_once)

    recovered = module._apply_truncation_recovery_if_needed(
        rows=rows,
        outputs=outputs,
        model=object(),
        tokenizer=object(),
        gen_cfg=module.GenerationConfig(max_new_tokens=32, do_sample=False),
        pad_id=0,
        truncate_chat_markers=True,
        torch_module=object(),
        truncation_recovery_cfg=module.TruncationRecoveryConfig(
            enabled=True,
            max_rounds=2,
            extra_tokens_per_round=32,
            datasets=("gsm8k",),
            require_final_answer_signal=True,
        ),
    )

    assert len(recovered) == 1
    rec = recovered[0]
    assert rec.truncation_recovery_applied is True
    assert rec.truncation_recovery_rounds == 1
    assert "Final answer: 7" in rec.raw_prediction


def test_load_prepared_rows_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    module = _load_phase_a_generate_module()
    path = tmp_path / "prepared_dupe.jsonl"
    row = {
        "sample_id": "dup-1",
        "dataset": "strategyqa",
        "split": "validation",
        "prompt_text": "p",
        "answer": "yes",
        "question": "q",
    }
    path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    try:
        module._load_prepared_rows(path)
    except ValueError as exc:
        assert "Duplicate sample_id" in str(exc)
        return
    raise AssertionError("Expected duplicate sample_id failure")


def test_load_prepared_rows_handles_u2028_without_splitting_jsonl(tmp_path: Path) -> None:
    module = _load_phase_a_generate_module()

    row = {
        "sample_id": "gsm8k:unicode:1",
        "dataset": "gsm8k",
        "split": "validation",
        "prompt_text": "[USER]\\nQ\\n[ASSISTANT]\\n",
        "answer": "42",
        "question": "Line A\u2028Line B",
    }
    path = tmp_path / "prepared_u2028.jsonl"
    # Keep ensure_ascii=False so the JSON line contains literal U+2028.
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = module._load_prepared_rows(path)
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "gsm8k:unicode:1"
    assert rows[0]["question"] == "Line A\u2028Line B"


def test_compare_metrics_marks_evaluator_version_mismatch(tmp_path: Path) -> None:
    module = _load_phase_a_generate_module()

    previous_metrics = tmp_path / "old_metrics.json"
    previous_metrics.write_text(
        json.dumps(
            {
                "accuracy": 0.5,
                "parse_error_rate": 0.1,
                "evaluator_version": "0.9.0",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    curr_pred = tmp_path / "curr_predictions.jsonl"
    prev_pred = tmp_path / "prev_predictions.jsonl"
    curr_pred.write_text("", encoding="utf-8")
    prev_pred.write_text("", encoding="utf-8")

    comparison = module._compare_metrics(
        current_metrics={
            "accuracy": 0.6,
            "parse_error_rate": 0.08,
            "evaluator_version": "1.1.0",
        },
        previous_metrics_path=previous_metrics,
        current_predictions_path=curr_pred,
        previous_predictions_path=prev_pred,
    )
    assert comparison["evaluator_version_match"] is False
    assert "mismatch" in str(comparison["comparison_caution"]).lower()
