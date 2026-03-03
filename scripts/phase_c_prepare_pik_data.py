#!/usr/bin/env python3
"""Prepare question-level P(IK) rollout artifacts for Phase C.

Why this file exists
--------------------
Current Phase C prefix-value experiments show weak signal. Before adding more
complex objectives, we need a simpler diagnostic: can a value head learn any
useful confidence signal at the question level?

This script builds that diagnostic dataset by sampling multiple answers per
question and aggregating empirical success rates:
- one question prompt,
- K sampled answers,
- one question-level P(IK) target.

What this file does
-------------------
1. Load prepared Phase-B rows.
2. Persist strict question contracts.
3. Generate K rollout answers per question (optional for contract-only tests).
4. Score each rollout with Phase-A extraction/eval semantics.
5. Aggregate per-question empirical success-rate targets.
6. Persist manifest, JSON summaries, and Markdown report.

Interaction with other files
----------------------------
- `src/ours/phase_b/pik_data.py`: question-level P(IK) contracts
- `src/ours/phase_b/data.py`: strict prepared-row loader
- `scripts/phase_c_train_pik.py`: consumes `pik_targets.jsonl`
- `scripts/phase_c_eval_pik.py`: standalone P(IK) evaluation

Example
-------
```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_c_prepare_pik_data.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl \
  --run-name strategyqa_pik_train \
  --max-samples 256 \
  --build-rollouts \
  --rollout-count 32 \
  --batch-size 256 \
  --require-cuda
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _bootstrap_src_path() -> None:
    """Allow running this script from repo root without package installation."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_a import PredictionRecord, score_prediction  # noqa: E402
from ours.phase_b.contracts import PhaseBTrainRow  # noqa: E402
from ours.phase_b.data import load_phase_b_rows, summarize_rows  # noqa: E402
from ours.phase_b.pik_data import (  # noqa: E402
    PIKQuestionRecord,
    PIKRolloutPredictionRecord,
    PIKTargetRecord,
    summarize_pik_targets,
)


@dataclass(slots=True)
class RolloutConfig:
    """Rollout generation config for question-level P(IK) artifact building."""

    model_path: str
    adapter_path: str | None
    batch_size: int
    rollout_count: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    seed: int
    dtype: str
    device_map: str
    require_cuda: bool
    oom_backoff: bool
    log_every: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary for manifests."""
        return asdict(self)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for P(IK) C1 artifact preparation."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare question-level P(IK) rollout artifacts from prepared "
            "Phase-B JSONL rows."
        )
    )
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--input-jsonl", type=Path, required=False, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_c_pik_data"),
        help="Root directory where P(IK) artifacts are written.",
    )
    parser.add_argument("--run-name", default="phase_c_pik_prepare")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse matching run directory when outputs already exist.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate artifacts even when a matching run directory already exists.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail on first row-level conversion error instead of recording it.",
    )

    parser.add_argument(
        "--build-rollouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate sampled answers and aggregate question-level targets.",
    )
    parser.add_argument(
        "--model-path",
        default="assets/models/Qwen2.5-7B-Instruct",
        help="Local HF model path used for rollout generation.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional PEFT adapter directory used on top of --model-path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Rollout generation batch size.",
    )
    parser.add_argument(
        "--rollout-count",
        type=int,
        default=32,
        help="Number of sampled answers per question.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use stochastic decoding for Monte Carlo supervision.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail fast if rollout generation was requested but CUDA is unavailable.",
    )
    parser.add_argument(
        "--oom-backoff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split generation batches recursively on OOM.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=256,
        help="Progress print interval in generated rollout samples.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args with optional config-JSON default injection."""
    parser = _build_parser()
    initial = parser.parse_known_args(argv)[0]
    if initial.config_json is not None:
        defaults = _load_config_defaults(initial.config_json)
        valid_keys = {action.dest for action in parser._actions}  # noqa: SLF001
        unknown = sorted(set(defaults.keys()) - valid_keys)
        if unknown:
            raise KeyError(f"Unknown keys in config JSON {initial.config_json}: {unknown}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.input_jsonl is None:
        parser.error("the following arguments are required: --input-jsonl")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.rollout_count <= 0:
        raise ValueError("--rollout-count must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.log_every < 0:
        raise ValueError("--log-every must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the full P(IK) C1 artifact preparation workflow."""
    args = parse_args(argv)

    rows = load_phase_b_rows(args.input_jsonl, max_samples=args.max_samples)
    row_summary = summarize_rows(rows)
    input_sha256 = _sha256_file(args.input_jsonl)
    datasets = sorted({row.dataset for row in rows}) or ["unknown"]
    dataset_tag = datasets[0] if len(datasets) == 1 else "mixed"

    rollout_config = None
    if args.build_rollouts:
        rollout_config = RolloutConfig(
            model_path=str(args.model_path),
            adapter_path=(str(args.adapter_path) if args.adapter_path is not None else None),
            batch_size=int(args.batch_size),
            rollout_count=int(args.rollout_count),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            do_sample=bool(args.do_sample),
            seed=int(args.seed),
            dtype=str(args.dtype),
            device_map=str(args.device_map),
            require_cuda=bool(args.require_cuda),
            oom_backoff=bool(args.oom_backoff),
            log_every=int(args.log_every),
        )

    run_spec = {
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": input_sha256,
        "max_samples": args.max_samples,
        "build_rollouts": bool(args.build_rollouts),
        "rollout_config": (rollout_config.to_dict() if rollout_config is not None else None),
    }
    run_fingerprint = _stable_hash_json(run_spec)
    run_dir = args.output_root / dataset_tag / f"{args.run_name}__{run_fingerprint}"

    expected_files = _expected_output_files(build_rollouts=bool(args.build_rollouts))
    if run_dir.exists() and not args.overwrite and args.resume and _has_expected_outputs(run_dir, expected_files):
        print("=" * 88)
        print("Phase C: Prepare P(IK) Data")
        print("=" * 88)
        print("status          : resume")
        print(f"run_dir         : {run_dir}")
        print(f"input_jsonl     : {args.input_jsonl}")
        print("=" * 88)
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    question_path = run_dir / "questions.jsonl"
    error_path = run_dir / "errors.jsonl"
    rollout_pred_path = run_dir / "rollout_predictions.jsonl"
    target_path = run_dir / "pik_targets.jsonl"
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"

    print("=" * 88)
    print("Phase C: Prepare P(IK) Data")
    print("=" * 88)
    print(f"input_jsonl      : {args.input_jsonl}")
    print(f"run_dir          : {run_dir}")
    print(f"num_rows         : {row_summary['num_rows']}")
    print(f"datasets         : {row_summary['dataset_counts']}")
    print(f"build_rollouts   : {args.build_rollouts}")
    if rollout_config is not None:
        print(
            "rollout_config   : "
            f"k={rollout_config.rollout_count} batch={rollout_config.batch_size} "
            f"tok={rollout_config.max_new_tokens} sample={rollout_config.do_sample}"
        )
    print("=" * 88)

    questions: list[PIKQuestionRecord] = []
    error_records: list[dict[str, Any]] = []
    for row in rows:
        try:
            question = _build_question_record(row)
            question.validate()
            questions.append(question)
        except Exception as exc:  # noqa: BLE001 - explicit artifact logging is desired.
            payload = {
                "sample_id": row.sample_id,
                "dataset": row.dataset,
                "split": row.split,
                "error": str(exc),
            }
            error_records.append(payload)
            if args.strict:
                raise

    _write_jsonl(question_path, (record.to_dict() for record in questions))
    _write_jsonl(error_path, error_records)

    rollout_predictions: list[PIKRolloutPredictionRecord] = []
    pik_targets: list[PIKTargetRecord] = []
    rollout_runtime: dict[str, Any] | None = None
    if args.build_rollouts:
        rollout_predictions, pik_targets, rollout_runtime = _build_rollout_targets(
            questions=questions,
            config=rollout_config,
        )
        _write_jsonl(rollout_pred_path, (record.to_dict() for record in rollout_predictions))
        _write_jsonl(target_path, (record.to_dict() for record in pik_targets))

    question_summary = {
        "num_questions": len(questions),
        "dataset_counts": row_summary["dataset_counts"],
        "split_counts": row_summary["split_counts"],
    }
    target_summary = summarize_pik_targets(pik_targets) if args.build_rollouts else None

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": input_sha256,
        "num_rows_input": row_summary["num_rows"],
        "num_rows_failed": len(error_records),
        "num_questions": len(questions),
        "question_summary": question_summary,
        "target_summary": target_summary,
        "rollout_runtime": rollout_runtime,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "artifact_stage": "phase_c_pik_c1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/phase_c_prepare_pik_data.py",
        "run_name": args.run_name,
        "run_fingerprint": run_fingerprint,
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": input_sha256,
        "row_summary": row_summary,
        "rollout_config": (rollout_config.to_dict() if rollout_config is not None else None),
        "output_files": {
            "questions": str(question_path),
            "errors": str(error_path),
            "rollout_predictions": (str(rollout_pred_path) if args.build_rollouts else None),
            "pik_targets": (str(target_path) if args.build_rollouts else None),
            "summary": str(summary_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(_render_summary_markdown(summary=summary, manifest=manifest), encoding="utf-8")

    print("-" * 88)
    print(f"num_questions     : {len(questions)}")
    print(f"num_errors        : {len(error_records)}")
    if args.build_rollouts:
        print(f"num_rollout_preds : {len(rollout_predictions)}")
        print(f"num_pik_targets   : {len(pik_targets)}")
        if rollout_runtime is not None:
            print(
                "rollout_runtime   : "
                f"{rollout_runtime['elapsed_seconds']:.2f}s | "
                f"{rollout_runtime['samples_per_second']:.3f} sample/s | "
                f"oom_backoff={rollout_runtime['oom_backoff_events']}"
            )
    print(f"summary_path      : {summary_path}")
    print(f"manifest_path     : {manifest_path}")
    print("=" * 88)
    return 0


def _build_question_record(row: PhaseBTrainRow) -> PIKQuestionRecord:
    """Convert one Phase-B row into one question-level P(IK) record."""
    row.validate()
    question_text = (row.question or "").strip()
    if question_text == "":
        # We keep this fallback explicit in metadata so later diagnostics can
        # isolate whether fallback rows behave differently.
        question_text = row.prompt_text.strip()
        question_fallback_used = True
    else:
        question_fallback_used = False
    return PIKQuestionRecord(
        sample_id=row.sample_id,
        dataset=row.dataset,
        split=row.split,
        question=question_text,
        prompt_text=row.prompt_text,
        answer=row.answer,
        metadata={
            "question_fallback_used": bool(question_fallback_used),
        },
    )


def _build_rollout_targets(
    *,
    questions: list[PIKQuestionRecord],
    config: RolloutConfig,
) -> tuple[list[PIKRolloutPredictionRecord], list[PIKTargetRecord], dict[str, Any]]:
    """Generate rollout predictions and aggregate question-level P(IK) targets."""
    if not questions:
        return [], [], {
            "elapsed_seconds": 0.0,
            "samples_per_second": 0.0,
            "oom_backoff_events": 0,
        }

    random.seed(config.seed)
    runtime = _load_rollout_runtime_dependencies()
    torch = runtime["torch"]
    AutoModelForCausalLM = runtime["AutoModelForCausalLM"]
    AutoTokenizer = runtime["AutoTokenizer"]

    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for rollout generation, but no GPU is visible.")

    resolved_dtype = _resolve_dtype(config.dtype, torch)
    tokenizer_path = _resolve_tokenizer_load_path(
        model_path=config.model_path,
        adapter_path=(Path(config.adapter_path) if config.adapter_path is not None else None),
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=resolved_dtype,
        device_map=config.device_map,
        trust_remote_code=True,
    )
    if config.adapter_path is not None:
        model = _attach_peft_adapter_for_inference(model, Path(config.adapter_path))
    model.eval()

    jobs: list[tuple[PIKQuestionRecord, int]] = []
    for question in questions:
        question.validate()
        for rollout_idx in range(config.rollout_count):
            jobs.append((question, rollout_idx))

    predictions: list[PIKRolloutPredictionRecord] = []
    start_ts = time.perf_counter()
    oom_backoff_events = 0
    done = 0

    for batch_start in range(0, len(jobs), config.batch_size):
        batch = jobs[batch_start : batch_start + config.batch_size]
        prompts = [question.prompt_text for question, _ in batch]
        continuations, local_oom_events = _generate_prompt_batch_with_backoff(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            oom_backoff=config.oom_backoff,
            torch_module=torch,
        )
        oom_backoff_events += local_oom_events

        for (question, rollout_idx), continuation in zip(batch, continuations, strict=True):
            full_prediction = str(continuation)
            scored = score_prediction(
                PredictionRecord(
                    sample_id=question.sample_id,
                    dataset=question.dataset,
                    split=question.split,
                    raw_prediction=full_prediction,
                    gold_answer=question.answer,
                    question=question.question,
                    metadata={
                        "rollout_index": int(rollout_idx),
                    },
                )
            )
            prediction = PIKRolloutPredictionRecord(
                sample_id=question.sample_id,
                dataset=question.dataset,
                split=question.split,
                question=question.question,
                prompt_text=question.prompt_text,
                answer=question.answer,
                rollout_index=int(rollout_idx),
                raw_continuation=str(continuation),
                full_prediction=full_prediction,
                extracted_prediction=scored.extracted_prediction,
                extraction_method=scored.extraction_method,
                is_correct=bool(scored.is_correct),
                parse_error=bool(scored.parse_error),
                generated_char_count=len(str(continuation)),
                metadata={
                    "question_fallback_used": bool(question.metadata.get("question_fallback_used", False)),
                },
            )
            prediction.validate()
            predictions.append(prediction)
            done += 1

        if config.log_every > 0 and (done % config.log_every == 0 or done == len(jobs)):
            elapsed = max(time.perf_counter() - start_ts, 1e-6)
            rate = done / elapsed
            print(
                "rollouts        : "
                f"{done}/{len(jobs)} ({done / max(len(jobs), 1):.1%}) | "
                f"elapsed={elapsed:.1f}s | rate={rate:.3f} sample/s",
                flush=True,
            )

    targets = _aggregate_pik_targets(predictions)
    elapsed = max(time.perf_counter() - start_ts, 1e-6)
    runtime_summary = {
        "elapsed_seconds": float(elapsed),
        "samples_per_second": float(len(jobs) / elapsed if jobs else 0.0),
        "oom_backoff_events": int(oom_backoff_events),
        "k_rollouts": int(config.rollout_count),
    }
    return predictions, targets, runtime_summary


def _aggregate_pik_targets(
    predictions: list[PIKRolloutPredictionRecord],
) -> list[PIKTargetRecord]:
    """Aggregate rollout predictions into one P(IK) target per question."""
    by_sample: dict[str, list[PIKRolloutPredictionRecord]] = {}
    for prediction in predictions:
        prediction.validate()
        by_sample.setdefault(prediction.sample_id, []).append(prediction)

    targets: list[PIKTargetRecord] = []
    for sample_id in sorted(by_sample):
        rows = sorted(by_sample[sample_id], key=lambda item: item.rollout_index)
        first = rows[0]
        k_rollouts = len(rows)
        n_correct = sum(1 for row in rows if row.is_correct)
        n_parse_error = sum(1 for row in rows if row.parse_error)
        parseable = k_rollouts - n_parse_error
        target = PIKTargetRecord(
            sample_id=first.sample_id,
            dataset=first.dataset,
            split=first.split,
            question=first.question,
            prompt_text=first.prompt_text,
            answer=first.answer,
            k_rollouts=k_rollouts,
            n_correct=n_correct,
            n_parse_error=n_parse_error,
            success_rate=(n_correct / k_rollouts if k_rollouts else 0.0),
            parseable_rate=(parseable / k_rollouts if k_rollouts else 0.0),
            mean_generated_char_count=float(sum(row.generated_char_count for row in rows) / max(k_rollouts, 1)),
            metadata={
                "rollout_indices": [row.rollout_index for row in rows],
            },
        )
        target.validate()
        targets.append(target)
    return targets


def _generate_prompt_batch_with_backoff(
    *,
    prompts: list[str],
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    oom_backoff: bool,
    torch_module: Any,
) -> tuple[list[str], int]:
    """Generate one continuation per prompt with optional OOM backoff."""
    try:
        return (
            _generate_prompt_batch_once(
                prompts=prompts,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                torch_module=torch_module,
            ),
            0,
        )
    except RuntimeError as exc:
        is_oom = "out of memory" in str(exc).lower()
        if not (oom_backoff and is_oom and len(prompts) > 1):
            raise
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        mid = len(prompts) // 2
        left, left_events = _generate_prompt_batch_with_backoff(
            prompts=prompts[:mid],
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            oom_backoff=oom_backoff,
            torch_module=torch_module,
        )
        right, right_events = _generate_prompt_batch_with_backoff(
            prompts=prompts[mid:],
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            oom_backoff=oom_backoff,
            torch_module=torch_module,
        )
        return left + right, left_events + right_events + 1


def _generate_prompt_batch_once(
    *,
    prompts: list[str],
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    torch_module: Any,
) -> list[str]:
    """Generate one continuation per prompt for one batch."""
    if not prompts:
        return []
    model_device = _resolve_model_input_device(model)
    original_padding_side = getattr(tokenizer, "padding_side", None)
    if original_padding_side is not None and original_padding_side != "left":
        tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model_device)
    finally:
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update(
            {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
            }
        )

    with torch_module.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    input_seq_len = int(inputs["input_ids"].shape[1])
    outputs: list[str] = []
    for row_idx in range(len(prompts)):
        generated = output_ids[row_idx, input_seq_len:]
        generated = _trim_trailing_pad_tokens(
            token_ids=generated,
            pad_id=tokenizer.pad_token_id,
            eos_id=tokenizer.eos_token_id,
        )
        raw_prediction = tokenizer.decode(generated, skip_special_tokens=True)
        outputs.append(str(raw_prediction))
    return outputs


def _load_rollout_runtime_dependencies() -> dict[str, Any]:
    """Import heavy runtime dependencies only when rollouts are enabled."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Rollout generation requires `torch` and `transformers` in the active environment"
        ) from exc
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _resolve_dtype(dtype_name: str, torch_module: Any) -> Any:
    """Map CLI dtype string into the torch dtype object used at runtime."""
    dtype_name = str(dtype_name).lower().strip()
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name!r}")
    return mapping[dtype_name]


def _resolve_tokenizer_load_path(*, model_path: str, adapter_path: Path | None) -> str:
    """Pick tokenizer path with adapter override fallback to base model path."""
    if adapter_path is not None:
        candidate = adapter_path / "tokenizer_config.json"
        if candidate.exists():
            return str(adapter_path)
    return str(model_path)


def _attach_peft_adapter_for_inference(model: Any, adapter_path: Path) -> Any:
    """Attach one PEFT adapter for rollout generation when requested."""
    try:
        from peft import PeftModel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Adapter path was provided ({adapter_path}), but `peft` is not importable"
        ) from exc
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    return PeftModel.from_pretrained(model, str(adapter_path))


def _resolve_model_input_device(model: Any) -> Any:
    """Resolve one stable device to place tokenizer tensors before generation."""
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for location in model.hf_device_map.values():
            if isinstance(location, str) and location.startswith("cuda"):
                import torch
                return torch.device(location)
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters; cannot resolve input device") from exc


def _trim_trailing_pad_tokens(*, token_ids: Any, pad_id: int | None, eos_id: int | None) -> Any:
    """Trim trailing PAD tokens and one trailing EOS token from generated IDs."""
    if token_ids.numel() == 0:
        return token_ids
    if eos_id is not None and int(token_ids[-1].item()) == int(eos_id):
        token_ids = token_ids[:-1]
    if pad_id is None:
        return token_ids
    end = int(token_ids.shape[0])
    while end > 0 and int(token_ids[end - 1].item()) == int(pad_id):
        end -= 1
    return token_ids[:end]


def _load_config_defaults(path: Path) -> dict[str, Any]:
    """Load one JSON object used as CLI defaults."""
    if not path.exists():
        raise FileNotFoundError(f"Config JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Config JSON {path} must contain an object at top level")
    return payload


def _sha256_file(path: Path) -> str:
    """Return SHA256 digest of one file for deterministic run fingerprints."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _stable_hash_json(payload: dict[str, Any]) -> str:
    """Hash one JSON-serializable object deterministically and return short tag."""
    stable = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:12]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write one iterable of dictionaries to UTF-8 JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _expected_output_files(*, build_rollouts: bool) -> list[str]:
    """Return output filenames that define one complete P(IK) run."""
    names = [
        "questions.jsonl",
        "errors.jsonl",
        "manifest.json",
        "summary.json",
        "summary.md",
    ]
    if build_rollouts:
        names.extend(["rollout_predictions.jsonl", "pik_targets.jsonl"])
    return names


def _has_expected_outputs(run_dir: Path, expected_files: list[str]) -> bool:
    """Check whether one run directory already contains all expected files."""
    return all((run_dir / name).exists() for name in expected_files)


def _render_summary_markdown(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Render one compact Markdown report for human scan and debugging."""
    lines = [
        "# Phase C P(IK) C1 Artifact Summary",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- input_jsonl: `{summary['input_jsonl']}`",
        f"- num_rows_input: `{summary['num_rows_input']}`",
        f"- num_rows_failed: `{summary['num_rows_failed']}`",
        f"- num_questions: `{summary['num_questions']}`",
        "",
        "## Config",
        "",
        f"- rollout_enabled: `{manifest['rollout_config'] is not None}`",
    ]
    if manifest.get("rollout_config") is not None:
        cfg = manifest["rollout_config"]
        lines.extend(
            [
                f"- rollout_count: `{cfg['rollout_count']}`",
                f"- rollout_batch_size: `{cfg['batch_size']}`",
                f"- max_new_tokens: `{cfg['max_new_tokens']}`",
                f"- model_path: `{cfg['model_path']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Question Summary",
            "",
            f"- dataset_counts: `{summary['question_summary']['dataset_counts']}`",
            f"- split_counts: `{summary['question_summary']['split_counts']}`",
        ]
    )
    if summary.get("target_summary") is not None:
        target_summary = summary["target_summary"]
        lines.extend(
            [
                "",
                "## Target Summary",
                "",
                f"- num_questions: `{target_summary['num_questions']}`",
                f"- mean_success_rate: `{target_summary['mean_success_rate']:.4f}`",
                f"- mean_parseable_rate: `{target_summary['mean_parseable_rate']:.4f}`",
                f"- mean_generated_char_count: `{target_summary['mean_generated_char_count']:.2f}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
