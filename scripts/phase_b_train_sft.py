#!/usr/bin/env python3
"""Phase B (B1) training skeleton: first SFT/PEFT training entry point.

This script is intentionally explicit and beginner-friendly.
It is the first training bridge from Phase A prepared artifacts to
trainable checkpoints.

Key design decisions
--------------------
1) PEFT-first by default (LoRA), with safe fallback to full SFT.
2) Batching-first defaults for speed (`per_device_train_batch_size=4`).
3) Safety nets enabled by default:
   - `auto_find_batch_size`,
   - gradient checkpointing,
   - strict data validation and duplicate-id checks.
4) Reproducibility:
   - run manifests,
   - fixed-seed behavior,
   - deterministic split/artifact references logged.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_b import load_phase_b_rows, summarize_rows  # noqa: E402


@dataclass(slots=True)
class LoRAConfigSpec:
    """Minimal LoRA settings saved in manifests."""

    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase B training script (B1): SFT/PEFT skeleton.",
    )

    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON config file. "
            "Values in this file become parser defaults, and CLI flags still override."
        ),
    )

    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=None,
        help=(
            "Training JSONL path. Can be provided directly or via --config-json."
        ),
    )
    parser.add_argument("--validation-jsonl", type=Path, default=None)
    parser.add_argument(
        "--model-path",
        default="assets/models/Qwen2.5-7B-Instruct",
        help="Local HF model path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_b_runs"),
        help="Run output root.",
    )
    parser.add_argument("--run-name", default="phase_b_sft")

    # Data limits for smoke/debug runs.
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=1024)

    # Strategy.
    parser.add_argument(
        "--training-mode",
        choices=["peft", "sft"],
        default="peft",
        help="peft: LoRA adapters; sft: full-model fine-tuning.",
    )
    parser.add_argument(
        "--peft-fallback-to-sft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If PEFT deps fail, fallback to SFT instead of hard-failing.",
    )

    # LoRA settings.
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names for LoRA injection.",
    )

    # Training hyperparameters (batching-first defaults).
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--auto-find-batch-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Safety net for OOM during training.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reduce activation memory at some speed cost.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Logging/checkpoint cadence.
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)

    # Runtime controls.
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail fast if CUDA is unavailable.",
    )
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument(
        "--save-final-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()

    # Two-phase parse so config file can set defaults.
    partial, _ = parser.parse_known_args(argv)
    if partial.config_json is not None:
        defaults = _load_config_defaults(partial.config_json)
        valid_keys = {a.dest for a in parser._actions}  # noqa: SLF001
        unknown = sorted(set(defaults.keys()) - valid_keys)
        if unknown:
            raise KeyError(
                f"Unknown keys in config file {partial.config_json}: {unknown}"
            )
        parser.set_defaults(**defaults)

    args = parser.parse_args(argv)
    if args.train_jsonl is None:
        parser.error(
            "the following arguments are required: --train-jsonl "
            "(or define `train_jsonl` in --config-json)"
        )
    if args.max_seq_length <= 8:
        raise ValueError("--max-seq-length must be > 8")
    if args.per_device_train_batch_size <= 0:
        raise ValueError("--per-device-train-batch-size must be > 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0")
    return args


def _load_config_defaults(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(
            f"Config JSON must contain an object/dict, got {type(payload)!r}"
        )
    return payload


def _import_runtime_deps():
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    return torch, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


def _resolve_dtype(name: str, torch_module):
    if name == "float32":
        return torch_module.float32
    if name == "float16":
        return torch_module.float16
    if name == "bfloat16":
        return torch_module.bfloat16
    if torch_module.cuda.is_available():
        return torch_module.bfloat16
    return torch_module.float32


def _set_seed(seed: int, torch_module) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _fmt_seconds(seconds: float) -> str:
    whole = max(int(seconds), 0)
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_features(
    rows,
    tokenizer,
    max_seq_length: int,
) -> list[dict[str, Any]]:
    """Convert prompt/target rows into causal-LM training features.

    Labels are masked (`-100`) on prompt tokens so loss is computed only on target.
    """

    features: list[dict[str, Any]] = []
    eos_id = tokenizer.eos_token_id

    for row in rows:
        prompt_ids = tokenizer(row.prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(row.target_text, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + target_ids
        if eos_id is not None and (not full_ids or full_ids[-1] != eos_id):
            full_ids = full_ids + [int(eos_id)]

        prompt_len = len(prompt_ids)
        if len(full_ids) > max_seq_length:
            overflow = len(full_ids) - max_seq_length
            full_ids = full_ids[overflow:]
            prompt_len = max(0, prompt_len - overflow)

        labels = list(full_ids)
        for i in range(prompt_len):
            labels[i] = -100

        if all(x == -100 for x in labels):
            # This can happen on extremely short max_seq_length values.
            # Keep at least one supervised token to avoid invalid batches.
            labels[-1] = full_ids[-1]

        features.append(
            {
                "sample_id": row.sample_id,
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )

    return features


class _ListFeatureDataset:
    """Simple dataset wrapper over pre-tokenized feature dicts."""

    def __init__(self, features: list[dict[str, Any]]) -> None:
        self._features = features

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._features[idx]


class _CausalLMCollator:
    """Dynamic-padding collator for causal-LM SFT with label masking."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for f in features:
            cur_len = len(f["input_ids"])
            pad_len = max_len - cur_len
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _resolve_training_args(
    *,
    TrainingArguments,
    output_dir: Path,
    args: argparse.Namespace,
    has_eval: bool,
    use_bf16: bool,
    use_fp16: bool,
):
    """Create version-tolerant TrainingArguments."""

    signature = inspect.signature(TrainingArguments.__init__)
    params = signature.parameters

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "data_seed": args.seed,
        "report_to": [],
        "remove_unused_columns": False,
        "auto_find_batch_size": args.auto_find_batch_size,
    }

    if "bf16" in params:
        kwargs["bf16"] = bool(use_bf16)
    if "fp16" in params:
        kwargs["fp16"] = bool(use_fp16)
    if "gradient_checkpointing" in params:
        kwargs["gradient_checkpointing"] = bool(args.gradient_checkpointing)
    if "save_safetensors" in params:
        kwargs["save_safetensors"] = True

    eval_mode = "steps" if has_eval else "no"
    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = eval_mode
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_mode

    return TrainingArguments(**kwargs)


def _attach_lora_if_requested(model, args: argparse.Namespace):
    """Attach LoRA adapters or fallback safely."""

    effective_mode = args.training_mode
    lora_spec = LoRAConfigSpec(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=[
            m.strip() for m in args.lora_target_modules.split(",") if m.strip()
        ],
    )

    if args.training_mode != "peft":
        return model, effective_mode, lora_spec, None

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:  # noqa: BLE001
        if args.peft_fallback_to_sft:
            msg = (
                "warning     : PEFT import failed; falling back to full SFT. "
                f"reason={exc}"
            )
            return model, "sft", lora_spec, msg
        raise RuntimeError(
            "PEFT mode requested but peft package import failed. "
            "Install `peft` or use --training-mode sft."
        ) from exc

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_spec.rank,
        lora_alpha=lora_spec.alpha,
        lora_dropout=lora_spec.dropout,
        target_modules=lora_spec.target_modules,
        bias="none",
    )
    peft_model = get_peft_model(model, peft_cfg)
    return peft_model, effective_mode, lora_spec, None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / f"{args.run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    train_metrics_path = run_dir / "train_metrics.json"
    eval_metrics_path = run_dir / "eval_metrics.json"
    summary_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    final_model_dir = run_dir / "final_model"

    print("=" * 88)
    print("Phase B: Train SFT/PEFT (B1 Skeleton)")
    print("=" * 88)
    print(f"run_dir          : {run_dir}")
    print(f"train_jsonl      : {args.train_jsonl}")
    print(f"validation_jsonl : {args.validation_jsonl}")
    print(f"model_path       : {args.model_path}")
    print(f"training_mode    : {args.training_mode}")
    print(f"batch_train      : {args.per_device_train_batch_size}")
    print(f"batch_eval       : {args.per_device_eval_batch_size}")
    print(f"max_seq_length   : {args.max_seq_length}")
    print(f"seed             : {args.seed}")
    print()

    train_rows = load_phase_b_rows(args.train_jsonl, max_samples=args.max_train_samples)
    eval_rows = (
        load_phase_b_rows(args.validation_jsonl, max_samples=args.max_eval_samples)
        if args.validation_jsonl is not None
        else []
    )
    train_summary = summarize_rows(train_rows)
    eval_summary = summarize_rows(eval_rows)

    print(f"train_rows       : {train_summary['num_rows']}")
    print(f"eval_rows        : {eval_summary['num_rows']}")
    print(f"train_datasets   : {train_summary['dataset_counts']}")
    print(f"eval_datasets    : {eval_summary['dataset_counts']}")

    (
        torch,
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
    ) = _import_runtime_deps()

    cuda_available = torch.cuda.is_available()
    print(f"torch            : {torch.__version__} (build CUDA={torch.version.cuda})")
    print(f"cuda_avail       : {cuda_available}")
    if cuda_available:
        print(f"cuda_count       : {torch.cuda.device_count()}")
    elif args.require_cuda:
        raise RuntimeError(
            "CUDA is unavailable but --require-cuda was set. "
            "Aborting to avoid accidental CPU training."
        )

    _set_seed(args.seed, torch_module=torch)
    dtype = _resolve_dtype(args.dtype, torch_module=torch)
    use_bf16 = dtype == torch.bfloat16
    use_fp16 = dtype == torch.float16

    load_start = time.perf_counter()
    print("model_load       : start")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    load_elapsed = time.perf_counter() - load_start
    print(f"model_load       : done in {_fmt_seconds(load_elapsed)}")

    model, effective_mode, lora_spec, peft_warning = _attach_lora_if_requested(model, args)
    if peft_warning is not None:
        print(peft_warning)
    print(f"effective_mode   : {effective_mode}")
    if effective_mode == "peft" and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_features = _build_features(
        rows=train_rows,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    eval_features = _build_features(
        rows=eval_rows,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    train_dataset = _ListFeatureDataset(train_features)
    eval_dataset = _ListFeatureDataset(eval_features) if eval_features else None
    collator = _CausalLMCollator(pad_token_id=int(tokenizer.pad_token_id))

    training_args = _resolve_training_args(
        TrainingArguments=TrainingArguments,
        output_dir=run_dir / "checkpoints",
        args=args,
        has_eval=eval_dataset is not None,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
    )

    manifest = {
        "phase": "phase_b",
        "lifecycle": "B1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "config_json": str(args.config_json) if args.config_json else None,
        "model_path": args.model_path,
        "requested_training_mode": args.training_mode,
        "effective_training_mode": effective_mode,
        "lora": asdict(lora_spec),
        "seed": args.seed,
        "dtype": args.dtype,
        "max_seq_length": args.max_seq_length,
        "train_input": str(args.train_jsonl),
        "validation_input": str(args.validation_jsonl) if args.validation_jsonl else None,
        "train_data_summary": train_summary,
        "eval_data_summary": eval_summary,
        "training_args": training_args.to_dict(),
    }
    _write_json(manifest_path, manifest)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    print("train            : start")
    train_start = time.perf_counter()
    train_output = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_elapsed = time.perf_counter() - train_start
    train_metrics = dict(train_output.metrics)
    train_metrics["train_elapsed_seconds"] = float(train_elapsed)
    train_metrics["effective_training_mode"] = effective_mode
    _write_json(train_metrics_path, train_metrics)
    print(f"train            : done in {_fmt_seconds(train_elapsed)}")

    eval_metrics: dict[str, Any] = {}
    if eval_dataset is not None:
        print("eval             : start")
        eval_metrics = dict(trainer.evaluate())
        _write_json(eval_metrics_path, eval_metrics)
        print("eval             : done")

    if args.save_final_model:
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        print(f"final_model      : {final_model_dir}")

    summary = {
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "train_metrics_path": str(train_metrics_path),
        "eval_metrics_path": str(eval_metrics_path) if eval_metrics else None,
        "effective_training_mode": effective_mode,
        "num_train_rows": train_summary["num_rows"],
        "num_eval_rows": eval_summary["num_rows"],
        "train_elapsed_seconds": float(train_elapsed),
        "max_seq_length": args.max_seq_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    _write_json(summary_path, summary)
    summary_md_path.write_text(
        "\n".join(
            [
                "# Phase B Run Summary",
                "",
                f"- run_dir: `{run_dir}`",
                f"- effective_training_mode: `{effective_mode}`",
                f"- num_train_rows: `{train_summary['num_rows']}`",
                f"- num_eval_rows: `{eval_summary['num_rows']}`",
                f"- train_elapsed_seconds: `{train_elapsed:.2f}`",
                f"- train_metrics: `{train_metrics_path}`",
                f"- eval_metrics: `{eval_metrics_path if eval_metrics else 'N/A'}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("-" * 88)
    print(f"train_metrics    : {train_metrics_path}")
    print(f"eval_metrics     : {eval_metrics_path if eval_metrics else 'N/A'}")
    print(f"manifest         : {manifest_path}")
    print(f"summary          : {summary_path}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
