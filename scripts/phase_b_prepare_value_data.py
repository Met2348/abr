#!/usr/bin/env python3
"""Prepare Phase C prefix, corruption, and rollout-target artifacts.

Why this file exists
--------------------
Phase C starts with value-head training, not RL. That requires a new data layer:
- step-level prefixes,
- corrupted prefixes for faithfulness tests,
- rollout-based empirical prefix targets.

This script is the official entrypoint for C0/C1:
1. freeze and validate the artifact contracts,
2. build deterministic step/prefix artifacts from prepared Phase B JSONL rows,
3. optionally generate rollout targets using the current backbone/adapter.

Design goals
------------
1. Fail loudly on contract mismatches.
2. Persist enough metadata to debug every prefix later.
3. Reuse Phase A evaluator semantics for rollout scoring.
4. Keep rollout generation optional so data-contract bugs can be tested without
   touching GPU code.

Interaction with other files
----------------------------
- `src/ours/phase_b/value_targets.py`: prefix and rollout-target contracts
- `src/ours/phase_b/corruptions.py`: corruption artifacts
- `src/ours/phase_b/data.py`: prepared-row loader
- `src/ours/phase_a/evaluator.py`: benchmark-compatible scoring

Example
-------
```bash
python -u scripts/phase_b_prepare_value_data.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/ef2ae6864f9c/train.jsonl \
  --run-name strategyqa_value_smoke \
  --max-samples 128 \
  --build-corruptions \
  --no-build-rollouts
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
    """Allow running this script from the repo root without package install."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_a import PredictionRecord, score_prediction  # noqa: E402
from ours.phase_b.contracts import PhaseBTrainRow  # noqa: E402
from ours.phase_b.data import load_phase_b_rows, summarize_rows  # noqa: E402
from ours.phase_b.corruptions import (  # noqa: E402
    CorruptionArtifact,
    CorruptionBuildConfig,
    build_corruptions_for_prefixes,
    summarize_corruptions,
)
from ours.phase_b.value_targets import (  # noqa: E402
    PrefixArtifact,
    PrefixBuildConfig,
    RolloutPredictionRecord,
    RolloutTargetRecord,
    build_prefix_artifacts,
    build_step_sequence_from_phase_b_row,
    summarize_prefix_artifacts,
    summarize_rollout_targets,
)
from ours.data.step_builder import StepBuildConfig, StepSequence  # noqa: E402


@dataclass(slots=True)
class RolloutConfig:
    """Configuration used only when rollout targets are enabled."""

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
        """Return a JSON-serializable payload for manifests."""
        return asdict(self)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for Phase C C0/C1 artifact preparation."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Phase C prefix/corruption/rollout-target artifacts from "
            "prepared Phase B JSONL rows."
        )
    )
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--input-jsonl", type=Path, required=False, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_c_data"),
        help="Root directory where Phase C artifacts are written.",
    )
    parser.add_argument("--run-name", default="phase_c_prepare")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse a matching run directory when all expected outputs already exist.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate artifacts even if a matching run directory already exists.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail on the first row-level artifact error instead of recording it.",
    )

    # Step configuration.
    parser.add_argument(
        "--split-mode",
        choices=["auto", "newline", "sentence"],
        default="auto",
        help="How to split CoT text into reasoning steps.",
    )
    parser.add_argument(
        "--include-question-step0",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend the question as step 0 before reasoning steps.",
    )
    parser.add_argument(
        "--include-answer-terminal-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the target final-answer text as a terminal step.",
    )
    parser.add_argument(
        "--normalize-whitespace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize whitespace inside generated step fragments.",
    )
    parser.add_argument(
        "--strip-list-markers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strip simple list markers after splitting CoT text.",
    )
    parser.add_argument("--min-fragment-chars", type=int, default=1)

    # Prefix configuration.
    parser.add_argument(
        "--include-question-only-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit the state before any reasoning step is generated.",
    )
    parser.add_argument(
        "--require-reasoning-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject answer-only rows by default for early value-head work.",
    )
    parser.add_argument(
        "--fallback-question-to-prompt-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use prompt_text as a fallback if the prepared row lacks question text.",
    )

    # Corruption configuration.
    parser.add_argument(
        "--build-corruptions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write corrupted-prefix artifacts for faithfulness tests.",
    )
    parser.add_argument("--max-corruptions-per-prefix", type=int, default=1)

    # Rollout configuration.
    parser.add_argument(
        "--build-rollouts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate rollout targets with the current model/adapter.",
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
        default=16,
        help="Effective rollout generation batch size (sampled continuations per call).",
    )
    parser.add_argument("--rollout-count", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rollouts should usually sample; disabling this collapses targets to 0/1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Runtime dtype for rollout generation.",
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
        help="If one generation batch OOMs, recursively split it.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=256,
        help="Progress print interval counted in generated rollout samples.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments with optional JSON default injection."""
    parser = _build_parser()
    initial = parser.parse_known_args(argv)[0]
    if initial.config_json is not None:
        defaults = _load_config_defaults(initial.config_json)
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.input_jsonl is None:
        parser.error("the following arguments are required: --input-jsonl")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the Phase C C0/C1 artifact-preparation workflow."""
    args = parse_args(argv)

    step_config = StepBuildConfig(
        split_mode=args.split_mode,
        include_question_as_step0=args.include_question_step0,
        include_final_answer_as_terminal_step=args.include_answer_terminal_step,
        normalize_whitespace=args.normalize_whitespace,
        strip_list_markers=args.strip_list_markers,
        min_fragment_chars=args.min_fragment_chars,
    )
    prefix_config = PrefixBuildConfig(
        include_question_only_prefix=args.include_question_only_prefix,
        require_reasoning_steps=args.require_reasoning_steps,
        fallback_question_to_prompt_text=args.fallback_question_to_prompt_text,
    )
    corruption_config = CorruptionBuildConfig(
        max_corruptions_per_prefix=args.max_corruptions_per_prefix,
    )
    step_config.validate()
    prefix_config.validate()
    corruption_config.validate()

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
        "step_config": step_config.to_dict(),
        "prefix_config": prefix_config.to_dict(),
        "build_corruptions": bool(args.build_corruptions),
        "corruption_config": (
            corruption_config.to_dict() if args.build_corruptions else None
        ),
        "build_rollouts": bool(args.build_rollouts),
        "rollout_config": (rollout_config.to_dict() if rollout_config is not None else None),
    }
    run_fingerprint = _stable_hash_json(run_spec)
    run_dir = args.output_root / dataset_tag / f"{args.run_name}__{run_fingerprint}"

    expected_files = _expected_output_files(build_corruptions=args.build_corruptions, build_rollouts=args.build_rollouts)
    if run_dir.exists() and not args.overwrite and args.resume and _has_expected_outputs(run_dir, expected_files):
        print("=" * 88)
        print("Phase C: Prepare Value Data")
        print("=" * 88)
        print(f"status          : resume")
        print(f"run_dir         : {run_dir}")
        print(f"input_jsonl     : {args.input_jsonl}")
        print("=" * 88)
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    step_path = run_dir / "step_sequences.jsonl"
    prefix_path = run_dir / "prefixes.jsonl"
    error_path = run_dir / "errors.jsonl"
    corruption_path = run_dir / "corruptions.jsonl"
    rollout_pred_path = run_dir / "rollout_predictions.jsonl"
    rollout_target_path = run_dir / "rollout_targets.jsonl"
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"

    print("=" * 88)
    print("Phase C: Prepare Value Data")
    print("=" * 88)
    print(f"input_jsonl      : {args.input_jsonl}")
    print(f"run_dir          : {run_dir}")
    print(f"num_rows         : {row_summary['num_rows']}")
    print(f"datasets         : {row_summary['dataset_counts']}")
    print(f"step_config      : {step_config.to_dict()} (sig={step_config.stable_signature()})")
    print(f"prefix_config    : {prefix_config.to_dict()} (sig={prefix_config.stable_signature()})")
    print(f"build_corruptions: {args.build_corruptions}")
    print(f"build_rollouts   : {args.build_rollouts}")
    if rollout_config is not None:
        print(
            "rollout_config   : "
            f"k={rollout_config.rollout_count} batch={rollout_config.batch_size} "
            f"tok={rollout_config.max_new_tokens} sample={rollout_config.do_sample}"
        )
    print("=" * 88)

    step_sequences: list[StepSequence] = []
    prefixes: list[PrefixArtifact] = []
    error_records: list[dict[str, Any]] = []

    for row in rows:
        try:
            sequence, build_meta = build_step_sequence_from_phase_b_row(
                row,
                step_config=step_config,
                prefix_config=prefix_config,
            )
            step_sequences.append(sequence)
            prefixes.extend(
                build_prefix_artifacts(
                    row=row,
                    step_sequence=sequence,
                    build_meta=build_meta,
                    prefix_config=prefix_config,
                )
            )
        except Exception as exc:  # noqa: BLE001 - explicit error artifact is intentional
            payload = {
                "sample_id": row.sample_id,
                "dataset": row.dataset,
                "split": row.split,
                "error": str(exc),
            }
            error_records.append(payload)
            if args.strict:
                raise

    _write_jsonl(step_path, (sequence.to_dict() for sequence in step_sequences))
    _write_jsonl(prefix_path, (prefix.to_dict() for prefix in prefixes))
    _write_jsonl(error_path, error_records)

    corruption_artifacts: list[CorruptionArtifact] = []
    if args.build_corruptions:
        corruption_artifacts = build_corruptions_for_prefixes(
            prefixes,
            config=corruption_config,
        )
        _write_jsonl(corruption_path, (artifact.to_dict() for artifact in corruption_artifacts))

    rollout_predictions: list[RolloutPredictionRecord] = []
    rollout_targets: list[RolloutTargetRecord] = []
    rollout_runtime: dict[str, Any] | None = None
    if args.build_rollouts:
        rollout_predictions, rollout_targets, rollout_runtime = _build_rollout_targets(
            prefixes=prefixes,
            config=rollout_config,
        )
        _write_jsonl(
            rollout_pred_path,
            (record.to_dict() for record in rollout_predictions),
        )
        _write_jsonl(
            rollout_target_path,
            (record.to_dict() for record in rollout_targets),
        )

    prefix_summary = summarize_prefix_artifacts(prefixes)
    corruption_summary = (
        summarize_corruptions(corruption_artifacts) if args.build_corruptions else None
    )
    rollout_summary = (
        summarize_rollout_targets(rollout_targets) if args.build_rollouts else None
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": input_sha256,
        "num_rows_input": row_summary["num_rows"],
        "num_rows_failed": len(error_records),
        "num_step_sequences": len(step_sequences),
        "num_prefixes": len(prefixes),
        "row_summary": row_summary,
        "prefix_summary": prefix_summary,
        "corruption_summary": corruption_summary,
        "rollout_summary": rollout_summary,
        "rollout_runtime": rollout_runtime,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "artifact_stage": "phase_c_c0_c1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/phase_b_prepare_value_data.py",
        "run_name": args.run_name,
        "run_fingerprint": run_fingerprint,
        "input_jsonl": str(args.input_jsonl),
        "input_sha256": input_sha256,
        "row_summary": row_summary,
        "step_config": step_config.to_dict(),
        "step_config_signature": step_config.stable_signature(),
        "prefix_config": prefix_config.to_dict(),
        "prefix_config_signature": prefix_config.stable_signature(),
        "corruption_config": (
            corruption_config.to_dict() if args.build_corruptions else None
        ),
        "rollout_config": (rollout_config.to_dict() if rollout_config is not None else None),
        "output_files": {
            "step_sequences": str(step_path),
            "prefixes": str(prefix_path),
            "errors": str(error_path),
            "corruptions": str(corruption_path) if args.build_corruptions else None,
            "rollout_predictions": str(rollout_pred_path) if args.build_rollouts else None,
            "rollout_targets": str(rollout_target_path) if args.build_rollouts else None,
            "summary": str(summary_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_md_path.write_text(
        _render_summary_markdown(summary=summary, manifest=manifest),
        encoding="utf-8",
    )

    print("-" * 88)
    print(f"num_step_sequences : {len(step_sequences)}")
    print(f"num_prefixes       : {len(prefixes)}")
    print(f"num_errors         : {len(error_records)}")
    if args.build_corruptions:
        print(f"num_corruptions    : {len(corruption_artifacts)}")
    if args.build_rollouts:
        print(f"num_rollout_preds  : {len(rollout_predictions)}")
        print(f"num_rollout_targets: {len(rollout_targets)}")
        if rollout_runtime is not None:
            print(
                "rollout_runtime   : "
                f"{rollout_runtime['elapsed_seconds']:.2f}s | "
                f"{rollout_runtime['samples_per_second']:.3f} sample/s | "
                f"oom_backoff={rollout_runtime['oom_backoff_events']}"
            )
    print(f"summary_path       : {summary_path}")
    print(f"manifest_path      : {manifest_path}")
    print("=" * 88)
    return 0


def _build_rollout_targets(
    *,
    prefixes: list[PrefixArtifact],
    config: RolloutConfig,
) -> tuple[list[RolloutPredictionRecord], list[RolloutTargetRecord], dict[str, Any]]:
    """Generate rollout predictions and aggregate them into prefix targets."""
    if not prefixes:
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

    jobs: list[tuple[PrefixArtifact, int]] = []
    for prefix in prefixes:
        prefix.validate()
        for rollout_index in range(config.rollout_count):
            jobs.append((prefix, rollout_index))

    predictions: list[RolloutPredictionRecord] = []
    start_ts = time.perf_counter()
    oom_backoff_events = 0
    done = 0

    for batch_start in range(0, len(jobs), config.batch_size):
        batch = jobs[batch_start : batch_start + config.batch_size]
        prompts = [prefix.rollout_input_text() for prefix, _ in batch]
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

        for (prefix, rollout_index), continuation in zip(batch, continuations, strict=True):
            full_prediction = f"{prefix.prefix_target_text}{continuation}"
            record = PredictionRecord(
                sample_id=prefix.sample_id,
                dataset=prefix.dataset,
                split=prefix.split,
                raw_prediction=full_prediction,
                gold_answer=prefix.gold_answer,
                question=prefix.question,
                metadata={
                    "prefix_id": prefix.prefix_id,
                    "rollout_index": int(rollout_index),
                    "prefix_step_index": int(prefix.prefix_step_index),
                },
            )
            scored = score_prediction(record)
            prediction = RolloutPredictionRecord(
                prefix_id=prefix.prefix_id,
                sample_id=prefix.sample_id,
                dataset=prefix.dataset,
                split=prefix.split,
                rollout_index=int(rollout_index),
                raw_continuation=continuation,
                full_prediction=full_prediction,
                extracted_prediction=scored.extracted_prediction,
                extraction_method=scored.extraction_method,
                is_correct=bool(scored.is_correct),
                parse_error=bool(scored.parse_error),
                generated_char_count=len(continuation),
                metadata={
                    "prefix_step_index": int(prefix.prefix_step_index),
                    "num_reasoning_steps_seen": int(prefix.num_reasoning_steps_seen),
                    "num_reasoning_steps_total": int(prefix.num_reasoning_steps_total),
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

    targets = _aggregate_rollout_targets(predictions)
    elapsed = max(time.perf_counter() - start_ts, 1e-6)
    runtime_summary = {
        "elapsed_seconds": float(elapsed),
        "samples_per_second": float(len(jobs) / elapsed if jobs else 0.0),
        "oom_backoff_events": int(oom_backoff_events),
        "k_rollouts": int(config.rollout_count),
    }
    return predictions, targets, runtime_summary


def _aggregate_rollout_targets(
    predictions: list[RolloutPredictionRecord],
) -> list[RolloutTargetRecord]:
    """Aggregate low-level rollout predictions into one target per prefix."""
    by_prefix: dict[str, list[RolloutPredictionRecord]] = {}
    for prediction in predictions:
        prediction.validate()
        by_prefix.setdefault(prediction.prefix_id, []).append(prediction)

    targets: list[RolloutTargetRecord] = []
    for prefix_id in sorted(by_prefix):
        rows = sorted(by_prefix[prefix_id], key=lambda item: item.rollout_index)
        first = rows[0]
        k_rollouts = len(rows)
        n_correct = sum(1 for row in rows if row.is_correct)
        n_parse_error = sum(1 for row in rows if row.parse_error)
        parseable = k_rollouts - n_parse_error
        target = RolloutTargetRecord(
            prefix_id=prefix_id,
            sample_id=first.sample_id,
            dataset=first.dataset,
            split=first.split,
            k_rollouts=k_rollouts,
            n_correct=n_correct,
            n_parse_error=n_parse_error,
            success_rate=(n_correct / k_rollouts if k_rollouts else 0.0),
            parseable_rate=(parseable / k_rollouts if k_rollouts else 0.0),
            mean_generated_char_count=float(
                sum(row.generated_char_count for row in rows) / max(k_rollouts, 1)
            ),
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
    """Generate sampled continuations for a prompt batch with optional OOM backoff."""
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
        left_outputs, left_events = _generate_prompt_batch_with_backoff(
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
        right_outputs, right_events = _generate_prompt_batch_with_backoff(
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
        return left_outputs + right_outputs, left_events + right_events + 1


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
    """Generate one sampled continuation per prompt."""
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
    """Import heavy runtime dependencies only when rollout generation is enabled."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Rollout generation requires `torch` and `transformers` in the active environment."
        ) from exc
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _resolve_dtype(dtype_name: str, torch_module: Any):
    """Map a CLI dtype name onto one torch dtype object."""
    if dtype_name == "auto":
        if torch_module.cuda.is_available():
            return torch_module.bfloat16
        return torch_module.float32
    if dtype_name == "float32":
        return torch_module.float32
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _resolve_tokenizer_load_path(model_path: str, adapter_path: Path | None) -> str:
    """Choose which directory should supply tokenizer files for rollouts."""
    if adapter_path is None:
        return str(model_path)
    if (adapter_path / "tokenizer_config.json").exists():
        return str(adapter_path)
    return str(model_path)


def _attach_peft_adapter_for_inference(model: Any, adapter_path: Path):
    """Attach a PEFT adapter to a loaded base model."""
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    try:
        from peft import PeftModel  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import `peft` while loading --adapter-path for rollout generation."
        ) from exc
    return PeftModel.from_pretrained(model, str(adapter_path))


def _resolve_model_input_device(model: Any):
    """Resolve which device should receive tokenized input tensors."""
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _trim_trailing_pad_tokens(token_ids: Any, pad_id: int | None, eos_id: int | None):
    """Trim right-side pad tokens from one generated token sequence."""
    if pad_id is None:
        return token_ids
    if eos_id is not None and pad_id == eos_id:
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
        raise TypeError(f"Config JSON {path} must contain an object at the top level")
    return payload


def _sha256_file(path: Path) -> str:
    """Return SHA256 of one file for deterministic artifact fingerprints."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _stable_hash_json(payload: dict[str, Any]) -> str:
    """Hash a JSON-serializable object deterministically and return a short tag."""
    stable = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:12]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write one iterable of dictionaries as UTF-8 JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _expected_output_files(*, build_corruptions: bool, build_rollouts: bool) -> list[str]:
    """Return the output filenames that define a complete run."""
    names = [
        "step_sequences.jsonl",
        "prefixes.jsonl",
        "errors.jsonl",
        "manifest.json",
        "summary.json",
        "summary.md",
    ]
    if build_corruptions:
        names.append("corruptions.jsonl")
    if build_rollouts:
        names.extend(["rollout_predictions.jsonl", "rollout_targets.jsonl"])
    return names


def _has_expected_outputs(run_dir: Path, expected_files: list[str]) -> bool:
    """Check whether one run directory already contains all expected files."""
    return all((run_dir / name).exists() for name in expected_files)


def _render_summary_markdown(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Render a compact human-readable Markdown summary."""
    lines = [
        "# Phase C C0/C1 Artifact Summary",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- input_jsonl: `{summary['input_jsonl']}`",
        f"- num_rows_input: `{summary['num_rows_input']}`",
        f"- num_rows_failed: `{summary['num_rows_failed']}`",
        f"- num_step_sequences: `{summary['num_step_sequences']}`",
        f"- num_prefixes: `{summary['num_prefixes']}`",
        "",
        "## Config",
        "",
        f"- step_config_signature: `{manifest['step_config_signature']}`",
        f"- prefix_config_signature: `{manifest['prefix_config_signature']}`",
        f"- rollout_enabled: `{manifest['rollout_config'] is not None}`",
        f"- corruptions_enabled: `{manifest['corruption_config'] is not None}`",
        "",
        "## Prefix Summary",
        "",
        f"- dataset_counts: `{summary['prefix_summary']['dataset_counts']}`",
        f"- split_counts: `{summary['prefix_summary']['split_counts']}`",
        f"- question_only_prefixes: `{summary['prefix_summary']['question_only_prefixes']}`",
        f"- max_reasoning_steps_seen: `{summary['prefix_summary']['max_reasoning_steps_seen']}`",
    ]
    if summary["corruption_summary"] is not None:
        lines.extend(
            [
                "",
                "## Corruption Summary",
                "",
                f"- num_corruptions: `{summary['corruption_summary']['num_corruptions']}`",
                f"- type_counts: `{summary['corruption_summary']['type_counts']}`",
            ]
        )
    if summary["rollout_summary"] is not None:
        lines.extend(
            [
                "",
                "## Rollout Summary",
                "",
                f"- mean_success_rate: `{summary['rollout_summary']['mean_success_rate']:.4f}`",
                f"- mean_parseable_rate: `{summary['rollout_summary']['mean_parseable_rate']:.4f}`",
                f"- mean_generated_char_count: `{summary['rollout_summary']['mean_generated_char_count']:.2f}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
