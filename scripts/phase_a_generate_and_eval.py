#!/usr/bin/env python3
"""Run Phase A inference and evaluation in one Python script.

What this script does
---------------------
1) Reads prepared samples JSONL (for example validation.jsonl from phase_a_prepare).
2) Runs model generation to create predictions JSONL.
3) Runs Phase A evaluator to create scored outputs + metrics.
4) Optionally compares against a previous run to show metric/prediction differences.

Why this script exists
----------------------
You asked for a Python-script workflow instead of many shell commands.
This file is the beginner-friendly, reproducible entry point for
"same experiment again, then compare differences".
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    import torch as torch_type  # pragma: no cover


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_a import PredictionRecord, evaluate_predictions  # noqa: E402


@dataclass(slots=True)
class GenerationConfig:
    """Generation settings saved to run manifest for reproducibility."""

    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50


@dataclass(slots=True)
class GenerationStats:
    """Runtime stats for one generation pass.

    Why this exists
    ---------------
    We need stable, machine-readable throughput numbers to compare
    single-sample vs batched runs fairly.
    """

    num_samples: int
    elapsed_seconds: float
    sample_per_second: float
    batch_size: int
    oom_backoff_events: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": int(self.num_samples),
            "elapsed_seconds": float(self.elapsed_seconds),
            "sample_per_second": float(self.sample_per_second),
            "batch_size": int(self.batch_size),
            "oom_backoff_events": int(self.oom_backoff_events),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase A inference + evaluation + optional comparison."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Prepared input file (for example: .../validation.jsonl).",
    )
    parser.add_argument(
        "--model-path",
        default="assets/models/Qwen2.5-7B-Instruct",
        help="Local HF model directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_a_runs"),
        help="Output root for this script's run folders.",
    )
    parser.add_argument(
        "--run-name",
        default="qwen_phase_a",
        help="Friendly run label used in output folder naming.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Model load dtype.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="HuggingFace device_map argument (default: auto).",
    )
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fail fast if CUDA is unavailable. "
            "Use this in benchmark runs to prevent accidental CPU fallback."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for smoke runs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Generation batch size for free-form decoding. "
            "Use 1 for strict compatibility; use 2/4/8 to speed up when VRAM allows."
        ),
    )
    parser.add_argument(
        "--oom-backoff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If a batch OOM happens, retry automatically by splitting into smaller batches. "
            "Recommended on shared servers for robust sweeps."
        ),
    )

    # Generation params
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sampling mode. Keep false for deterministic baseline.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--strategyqa-decode-mode",
        choices=["freeform", "binary_choice"],
        default="freeform",
        help=(
            "How to decode StrategyQA outputs. "
            "`freeform`: normal text generation. "
            "`binary_choice`: score candidate answers (`yes`/`no`) directly."
        ),
    )
    parser.add_argument(
        "--truncate-chat-markers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Trim generated text at chat-leak markers like `[USER]` or `Human:`. "
            "This often reduces parse errors caused by accidental next-turn spillover."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help=(
            "Candidate stride for generation progress checkpoints."
        ),
    )
    parser.add_argument(
        "--max-progress-lines",
        type=int,
        default=5,
        help=(
            "Maximum number of generation progress lines to print "
            "(including start line)."
        ),
    )
    parser.add_argument(
        "--persist-console-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Persist terminal output to <run_dir>/console.log while still printing "
            "to terminal."
        ),
    )

    # Comparison controls
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        help="Optional previous run metrics.json path for diff.",
    )
    parser.add_argument(
        "--compare-latest-same-name",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If compare-with is absent, compare to latest run with same run-name.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_jsonl}")

    torch, AutoTokenizer, AutoModelForCausalLM = _import_runtime_deps()
    _set_reproducibility(seed=args.seed, torch_module=torch)

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / f"{args.run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = run_dir / "predictions.jsonl"
    scored_path = run_dir / "scored_predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "manifest.json"
    console_log_path = run_dir / "console.log"

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    console_log_writer: TextIO | None = None
    if args.persist_console_log:
        console_log_writer = console_log_path.open("w", encoding="utf-8")
        sys.stdout = _TeeWriter(orig_stdout, console_log_writer)
        sys.stderr = _TeeWriter(orig_stderr, console_log_writer)

    try:
        print("=" * 88)
        print("Phase A: Generate + Evaluate")
        print("=" * 88)
        print(f"input_jsonl : {args.input_jsonl}")
        print(f"model_path  : {args.model_path}")
        print(f"run_dir     : {run_dir}")
        print(f"seed        : {args.seed}")
        print(f"gen_config  : {asdict(gen_cfg)}")
        print(f"decode_mode : strategyqa={args.strategyqa_decode_mode}")
        print(f"trim_markers: {args.truncate_chat_markers}")
        print(f"log_every   : {args.log_every}")
        print(f"max_prog_ln : {args.max_progress_lines}")
        print(f"batch_size  : {args.batch_size}")
        print(f"oom_backoff : {args.oom_backoff}")
        print(f"console_log : {console_log_path if args.persist_console_log else '<disabled>'}")
        print(f"torch       : {torch.__version__} (build CUDA={torch.version.cuda})")
        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count() if cuda_available else 0
        print(f"cuda_avail  : {cuda_available}")
        print(f"cuda_count  : {cuda_count}")
        if cuda_available:
            devices = [torch.cuda.get_device_name(i) for i in range(cuda_count)]
            print(f"cuda_names  : {devices}")
        elif args.require_cuda:
            raise RuntimeError(
                "CUDA is unavailable but --require-cuda was set. "
                "Aborting to avoid accidental CPU benchmarking."
            )
        else:
            print(
                "warning     : CUDA unavailable; this run will use CPU and may be very slow.",
            )

        rows = _load_prepared_rows(args.input_jsonl)
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
        print(f"num_inputs  : {len(rows)}")

        _configure_transformers_progress_bar(disable=True)
        load_start = time.perf_counter()
        print("model_load  : start")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=_resolve_dtype(args.dtype, torch_module=torch),
            device_map=args.device_map,
            trust_remote_code=False,
        )
        load_elapsed = int(time.perf_counter() - load_start)
        print(f"model_load  : done in {_fmt_seconds(load_elapsed)}")
        model.eval()
        _configure_model_generation(model=model, gen_cfg=gen_cfg)
        print(f"first_param : {next(model.parameters()).device}")
        hf_map = getattr(model, "hf_device_map", None)
        if hf_map is not None:
            map_counts = Counter(str(v) for v in hf_map.values())
            print(f"hf_device_map: {dict(sorted(map_counts.items()))}")

        generation_stats = _run_generation(
            rows=rows,
            model=model,
            tokenizer=tokenizer,
            gen_cfg=gen_cfg,
            output_path=predictions_path,
            source_path=args.input_jsonl,
            torch_module=torch,
            log_every=args.log_every,
            max_progress_lines=args.max_progress_lines,
            strategyqa_decode_mode=args.strategyqa_decode_mode,
            truncate_chat_markers=bool(args.truncate_chat_markers),
            batch_size=int(args.batch_size),
            oom_backoff=bool(args.oom_backoff),
        )

        scored_rows, summary = _run_evaluation(predictions_path)

        math_diag = _build_math_generation_diagnostics(
            scored_rows=scored_rows,
            max_new_tokens=gen_cfg.max_new_tokens,
        )
        if math_diag is not None:
            _print_math_generation_diagnostics(math_diag=math_diag)

        with scored_path.open("w", encoding="utf-8") as fout:
            for row in scored_rows:
                fout.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

        metrics = summary.to_dict()
        metrics["run_name"] = args.run_name
        metrics["input_jsonl"] = str(args.input_jsonl)
        metrics["predictions_jsonl"] = str(predictions_path)
        metrics["scored_predictions_jsonl"] = str(scored_path)
        metrics["generation_stats"] = generation_stats.to_dict()
        if math_diag is not None:
            metrics["math_diagnostics"] = math_diag

        previous_metrics_path = _resolve_previous_metrics_path(
            output_root=args.output_root,
            run_name=args.run_name,
            current_run_dir=run_dir,
            compare_with=args.compare_with,
            compare_latest_same_name=args.compare_latest_same_name,
        )

        comparison = None
        if previous_metrics_path is not None:
            comparison = _compare_metrics(
                current_metrics=metrics,
                previous_metrics_path=previous_metrics_path,
                current_predictions_path=predictions_path,
                previous_predictions_path=previous_metrics_path.parent / "predictions.jsonl",
            )
            metrics["comparison"] = comparison

        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        manifest = {
            "schema_version": 1,
            "script": "scripts/phase_a_generate_and_eval.py",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": args.run_name,
            "input_jsonl": str(args.input_jsonl),
            "model_path": args.model_path,
            "seed": args.seed,
            "dtype": args.dtype,
            "device_map": args.device_map,
            "generation_config": asdict(gen_cfg),
            "strategyqa_decode_mode": args.strategyqa_decode_mode,
            "truncate_chat_markers": bool(args.truncate_chat_markers),
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "oom_backoff": bool(args.oom_backoff),
            "persist_console_log": bool(args.persist_console_log),
            "files": {
                "predictions": str(predictions_path),
                "scored_predictions": str(scored_path),
                "metrics": str(metrics_path),
                "console_log": (
                    str(console_log_path)
                    if args.persist_console_log
                    else None
                ),
            },
            "compared_with": str(previous_metrics_path) if previous_metrics_path else None,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print("-" * 88)
        print(f"accuracy         : {summary.accuracy:.4f}")
        print(f"parse_error_rate : {summary.parse_error_rate:.4f}")
        print(f"n_parseable      : {summary.n_parseable}")
        print(f"acc_parseable    : {summary.accuracy_parseable:.4f}")
        print(f"gen_elapsed_sec  : {generation_stats.elapsed_seconds:.2f}")
        print(f"gen_sample_rate  : {generation_stats.sample_per_second:.3f} sample/s")
        print(f"oom_backoff_evts : {generation_stats.oom_backoff_events}")
        if comparison is not None:
            print(f"compared_with    : {previous_metrics_path}")
            print(f"delta_accuracy   : {comparison['delta_accuracy']:+.4f}")
            print(f"delta_parse_err  : {comparison['delta_parse_error_rate']:+.4f}")
            print(f"changed_samples  : {comparison['changed_prediction_count']}")
            if not comparison.get("evaluator_version_match", True):
                print(
                    "compare_warning  : evaluator version mismatch "
                    f"({comparison.get('previous_evaluator_version')} -> "
                    f"{comparison.get('current_evaluator_version')})."
                )
                print(
                    "compare_warning  : metric deltas may include scoring-logic changes.",
                )
        print(f"metrics_path     : {metrics_path}")
        print("=" * 88)
        return 0
    finally:
        if console_log_writer is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            console_log_writer.close()


def _set_reproducibility(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    # Use deterministic algorithms where practical.
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False


def _resolve_dtype(dtype_name: str, torch_module: Any):
    if dtype_name == "auto":
        if torch_module.cuda.is_available():
            # bfloat16 is stable for most modern GPUs and efficient.
            return torch_module.bfloat16
        return torch_module.float32
    if dtype_name == "float32":
        return torch_module.float32
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _load_prepared_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip() == "":
            continue
        row = json.loads(line)
        required = ["sample_id", "dataset", "split", "prompt_text", "answer", "question"]
        missing = [k for k in required if k not in row]
        if missing:
            raise KeyError(f"Missing keys {missing} at line={idx+1} in {path}")
        sample_id = str(row["sample_id"])
        if sample_id in seen_sample_ids:
            raise ValueError(
                f"Duplicate sample_id={sample_id!r} at line={idx+1} in {path}. "
                "Prepared input must have unique sample IDs for trustworthy metrics."
            )
        seen_sample_ids.add(sample_id)
        rows.append(row)
    return rows


def _run_generation(
    rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    gen_cfg: GenerationConfig,
    output_path: Path,
    source_path: Path,
    torch_module: Any,
    log_every: int,
    max_progress_lines: int,
    strategyqa_decode_mode: str,
    truncate_chat_markers: bool,
    batch_size: int,
    oom_backoff: bool,
) -> GenerationStats:
    if log_every <= 0:
        raise ValueError(f"--log-every must be >= 1, got {log_every}")
    if max_progress_lines <= 0:
        raise ValueError(f"--max-progress-lines must be >= 1, got {max_progress_lines}")
    if batch_size <= 0:
        raise ValueError(f"--batch-size must be >= 1, got {batch_size}")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    total = len(rows)
    start_ts = time.perf_counter()
    print(f"generation  : starting {total} samples", flush=True)
    checkpoints = sorted(
        _build_progress_checkpoints(
        total=total,
        log_every=log_every,
        max_progress_lines=max_progress_lines,
        )
    )
    checkpoint_idx = 0
    oom_backoff_events = 0

    with output_path.open("w", encoding="utf-8") as fout:
        done = 0
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_rows = rows[batch_start:batch_end]

            # Keep row order fully deterministic even if we process via mixed routes.
            batch_results: list[dict[str, Any] | None] = [None] * len(batch_rows)

            binary_indices: list[int] = []
            free_indices: list[int] = []
            for local_idx, row in enumerate(batch_rows):
                dataset = str(row["dataset"]).strip().lower()
                if dataset == "strategyqa" and strategyqa_decode_mode == "binary_choice":
                    binary_indices.append(local_idx)
                else:
                    free_indices.append(local_idx)

            # Binary-choice route remains per-sample scoring.
            for local_idx in binary_indices:
                row = batch_rows[local_idx]
                prompt = str(row["prompt_text"])
                raw_prediction = _predict_strategyqa_binary_choice(
                    prompt=prompt,
                    model=model,
                    tokenizer=tokenizer,
                    torch_module=torch_module,
                )
                batch_results[local_idx] = _build_prediction_row(
                    row=row,
                    source_path=source_path,
                    row_index=batch_start + local_idx,
                    raw_prediction=raw_prediction,
                    generated_token_count=0,
                    hit_token_limit=False,
                )

            if free_indices:
                free_rows = [batch_rows[idx] for idx in free_indices]
                free_outputs, local_oom_events = _generate_freeform_rows_with_optional_backoff(
                    rows=free_rows,
                    model=model,
                    tokenizer=tokenizer,
                    gen_cfg=gen_cfg,
                    pad_id=pad_id,
                    truncate_chat_markers=truncate_chat_markers,
                    torch_module=torch_module,
                    oom_backoff=oom_backoff,
                )
                oom_backoff_events += local_oom_events
                for sub_idx, local_idx in enumerate(free_indices):
                    raw_prediction, generated_token_count, hit_token_limit = free_outputs[sub_idx]
                    row = batch_rows[local_idx]
                    batch_results[local_idx] = _build_prediction_row(
                        row=row,
                        source_path=source_path,
                        row_index=batch_start + local_idx,
                        raw_prediction=raw_prediction,
                        generated_token_count=generated_token_count,
                        hit_token_limit=hit_token_limit,
                    )

            for local_idx, pred_row in enumerate(batch_results):
                if pred_row is None:
                    raise RuntimeError(
                        "Internal generation error: missing prediction row. "
                        f"batch_start={batch_start}, local_idx={local_idx}"
                    )
                fout.write(json.dumps(pred_row, ensure_ascii=False) + "\n")
                # Flush every record so external monitoring can track real progress.
                fout.flush()
                done += 1

                while checkpoint_idx < len(checkpoints) and done >= checkpoints[checkpoint_idx]:
                    elapsed = max(time.perf_counter() - start_ts, 1e-6)
                    rate = done / elapsed
                    remaining = total - done
                    eta_seconds = int(remaining / rate) if rate > 0 else -1
                    print(
                        "generation  : "
                        f"{done}/{total} ({done / max(total, 1):.1%}) | "
                        f"elapsed={_fmt_seconds(int(elapsed))} | "
                        f"rate={rate:.3f} sample/s | "
                        f"eta={_fmt_seconds(eta_seconds)}",
                        flush=True,
                    )
                    checkpoint_idx += 1

    total_elapsed = max(time.perf_counter() - start_ts, 1e-6)
    return GenerationStats(
        num_samples=total,
        elapsed_seconds=float(total_elapsed),
        sample_per_second=float(total / total_elapsed if total else 0.0),
        batch_size=int(batch_size),
        oom_backoff_events=int(oom_backoff_events),
    )


def _build_prediction_row(
    row: dict[str, Any],
    source_path: Path,
    row_index: int,
    raw_prediction: str,
    generated_token_count: int,
    hit_token_limit: bool,
) -> dict[str, Any]:
    """Build one prediction record in a single canonical place."""
    return {
        "sample_id": row["sample_id"],
        "dataset": row["dataset"],
        "split": row["split"],
        "raw_prediction": raw_prediction,
        "gold_answer": row["answer"],
        "question": row.get("question"),
        "metadata": {
            "source_prepared_jsonl": str(source_path),
            "template_id": row.get("template_id"),
            "template_version": row.get("template_version"),
            "row_index": row_index,
            "generated_tokens": int(generated_token_count),
            "hit_token_limit": bool(hit_token_limit),
        },
    }


def _generate_freeform_rows_with_optional_backoff(
    rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    gen_cfg: GenerationConfig,
    pad_id: int | None,
    truncate_chat_markers: bool,
    torch_module: Any,
    oom_backoff: bool,
) -> tuple[list[tuple[str, int, bool]], int]:
    """Generate free-form outputs for a row batch.

    Returns
    -------
    tuple[list[tuple[str, int, bool]], int]
        - list per row: (`raw_prediction`, `generated_token_count`, `hit_token_limit`)
        - count of OOM backoff events in this call tree
    """
    prompts = [str(row["prompt_text"]) for row in rows]
    try:
        outputs = _generate_freeform_rows_once(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            gen_cfg=gen_cfg,
            pad_id=pad_id,
            truncate_chat_markers=truncate_chat_markers,
            torch_module=torch_module,
        )
        return outputs, 0
    except RuntimeError as exc:
        is_oom = "out of memory" in str(exc).lower()
        if not (oom_backoff and is_oom and len(rows) > 1):
            raise
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        mid = len(rows) // 2
        left_outputs, left_events = _generate_freeform_rows_with_optional_backoff(
            rows=rows[:mid],
            model=model,
            tokenizer=tokenizer,
            gen_cfg=gen_cfg,
            pad_id=pad_id,
            truncate_chat_markers=truncate_chat_markers,
            torch_module=torch_module,
            oom_backoff=oom_backoff,
        )
        right_outputs, right_events = _generate_freeform_rows_with_optional_backoff(
            rows=rows[mid:],
            model=model,
            tokenizer=tokenizer,
            gen_cfg=gen_cfg,
            pad_id=pad_id,
            truncate_chat_markers=truncate_chat_markers,
            torch_module=torch_module,
            oom_backoff=oom_backoff,
        )
        return left_outputs + right_outputs, left_events + right_events + 1


def _generate_freeform_rows_once(
    prompts: list[str],
    model: Any,
    tokenizer: Any,
    gen_cfg: GenerationConfig,
    pad_id: int | None,
    truncate_chat_markers: bool,
    torch_module: Any,
) -> list[tuple[str, int, bool]]:
    """Run exactly one batched free-form generation call."""
    if not prompts:
        return []

    model_device = _resolve_model_input_device(model)
    # Decoder-only models should use LEFT padding during generation.
    # Right padding can shift attention behavior and degrade output quality.
    original_padding_side = getattr(tokenizer, "padding_side", None)
    if original_padding_side is not None and original_padding_side != "left":
        tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model_device)
    finally:
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": gen_cfg.max_new_tokens,
        "do_sample": gen_cfg.do_sample,
        "pad_token_id": pad_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if gen_cfg.do_sample:
        gen_kwargs.update(
            {
                "temperature": gen_cfg.temperature,
                "top_p": gen_cfg.top_p,
                "top_k": gen_cfg.top_k,
            }
        )

    with torch_module.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)  # core generation call

    input_seq_len = int(inputs["input_ids"].shape[1])
    outputs: list[tuple[str, int, bool]] = []
    for row_idx in range(len(prompts)):
        generated = output_ids[row_idx, input_seq_len:]
        generated = _trim_trailing_pad_tokens(
            token_ids=generated,
            pad_id=pad_id,
            eos_id=tokenizer.eos_token_id,
        )
        generated_token_count = int(generated.shape[0])
        hit_token_limit = bool(generated_token_count >= gen_cfg.max_new_tokens)
        raw_prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if truncate_chat_markers:
            raw_prediction = _truncate_output_at_chat_markers(raw_prediction)
        outputs.append((raw_prediction, generated_token_count, hit_token_limit))
    return outputs


def _resolve_model_input_device(model: Any):
    """Resolve which device should receive tokenized input tensors."""
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _trim_trailing_pad_tokens(token_ids: Any, pad_id: int | None, eos_id: int | None):
    """Trim right-side pad tokens from one generated token sequence.

    We keep the sequence unchanged when `pad_id == eos_id`, because then
    trailing token identity is ambiguous and aggressive trimming may delete
    valid EOS markers.
    """
    if pad_id is None:
        return token_ids
    if eos_id is not None and pad_id == eos_id:
        return token_ids
    end = int(token_ids.shape[0])
    while end > 0 and int(token_ids[end - 1].item()) == int(pad_id):
        end -= 1
    return token_ids[:end]


def _predict_strategyqa_binary_choice(
    prompt: str,
    model: Any,
    tokenizer: Any,
    torch_module: Any,
) -> str:
    """Choose between `yes` and `no` by conditional log-probability.

    Why this mode exists
    --------------------
    StrategyQA is a binary task. Free-form decoding can fail formatting and
    inflate parse-error rates. This mode removes format variance and evaluates
    the model's binary decision directly.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    choices = {
        "yes": " yes",
        "no": " no",
    }
    scores: dict[str, float] = {}

    with torch_module.no_grad():
        for label, suffix in choices.items():
            cand_ids = tokenizer(
                suffix,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(model.device)
            input_ids = torch_module.cat([prompt_ids, cand_ids], dim=1)
            attention_mask = torch_module.ones_like(input_ids)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            # Score each candidate token conditioned on prompt + previous candidate tokens.
            start_pos = prompt_ids.shape[1] - 1
            cand_score = 0.0
            for j in range(cand_ids.shape[1]):
                token_logits = logits[0, start_pos + j, :]
                token_id = int(cand_ids[0, j].item())
                token_logprob = torch_module.log_softmax(token_logits, dim=-1)[token_id]
                cand_score += float(token_logprob.item())
            scores[label] = cand_score

    return max(scores, key=scores.get)


def _truncate_output_at_chat_markers(text: str) -> str:
    """Trim spillover text such as `[USER]` / `Human:` from generation."""
    lowered = text.lower()
    markers = [
        "[system]",
        "[user]",
        "[assistant]",
        "human:",
        "user:",
        "assistant:",
    ]
    positions = [lowered.find(marker) for marker in markers]
    positions = [pos for pos in positions if pos >= 0]
    if not positions:
        return text.strip()
    return text[: min(positions)].strip()


def _configure_model_generation(model: Any, gen_cfg: GenerationConfig) -> None:
    """Align model generation config with run settings.

    Some transformers versions warn when `do_sample=False` but sampling params
    remain set in model generation config. We explicitly clear them here.
    """
    if not hasattr(model, "generation_config"):
        return
    model.generation_config.do_sample = bool(gen_cfg.do_sample)
    if gen_cfg.do_sample:
        model.generation_config.temperature = float(gen_cfg.temperature)
        model.generation_config.top_p = float(gen_cfg.top_p)
        model.generation_config.top_k = int(gen_cfg.top_k)
    else:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None


def _run_evaluation(predictions_path: Path):
    records: list[PredictionRecord] = []
    for idx, line in enumerate(predictions_path.read_text(encoding="utf-8").splitlines()):
        if line.strip() == "":
            continue
        payload = json.loads(line)
        rec = PredictionRecord(
            sample_id=str(payload["sample_id"]),
            dataset=str(payload["dataset"]),
            split=str(payload["split"]),
            raw_prediction=str(payload["raw_prediction"]),
            gold_answer=str(payload["gold_answer"]),
            question=(
                str(payload["question"])
                if "question" in payload and payload["question"] is not None
                else None
            ),
            metadata=dict(payload.get("metadata", {})),
        )
        try:
            rec.validate()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Invalid prediction record at line={idx+1} in {predictions_path}: {exc}"
            ) from exc
        records.append(rec)

    return evaluate_predictions(records)


def _build_math_generation_diagnostics(
    scored_rows: list[Any],
    max_new_tokens: int,
) -> dict[str, Any] | None:
    """Summarize extraction quality for math datasets.

    This helps catch the common failure mode:
    - model emits long partial reasoning,
    - generation hits token cap,
    - evaluator falls back to weak `last_number` extraction.
    """
    math_sets = {"gsm8k", "hendrycks_math"}
    rows = [r for r in scored_rows if str(r.dataset).strip().lower() in math_sets]
    if not rows:
        return None

    n = len(rows)
    method_counts = Counter(str(r.extraction_method) for r in rows)
    n_last_number = method_counts.get("last_number", 0)
    n_hit_cap = sum(
        1
        for r in rows
        if bool((r.metadata or {}).get("hit_token_limit", False))
    )
    n_with_final_tag = sum(
        1 for r in rows if "final answer" in str(r.raw_prediction).lower()
    )
    accuracy = sum(1 for r in rows if bool(r.is_correct)) / n if n else 0.0

    return {
        "n": n,
        "max_new_tokens": int(max_new_tokens),
        "accuracy": float(accuracy),
        "last_number_rate": float(n_last_number / n if n else 0.0),
        "hit_token_limit_rate": float(n_hit_cap / n if n else 0.0),
        "final_answer_tag_rate": float(n_with_final_tag / n if n else 0.0),
        "extraction_method_counts": dict(method_counts),
    }


def _print_math_generation_diagnostics(math_diag: dict[str, Any]) -> None:
    print("-" * 88)
    print("math_diag       : extraction reliability check")
    print(
        "math_diag       : "
        f"n={math_diag['n']} "
        f"acc={math_diag['accuracy']:.4f} "
        f"last_number_rate={math_diag['last_number_rate']:.4f} "
        f"hit_cap_rate={math_diag['hit_token_limit_rate']:.4f} "
        f"final_tag_rate={math_diag['final_answer_tag_rate']:.4f}"
    )
    if (
        float(math_diag["last_number_rate"]) >= 0.8
        and float(math_diag["hit_token_limit_rate"]) >= 0.5
    ):
        print(
            "math_warning    : Many answers use weak last-number extraction after hitting token cap.",
        )
        print(
            "math_warning    : This often indicates truncated reasoning, not clean final answers.",
        )
        print(
            "math_warning    : Try `qa_math_direct_final` template and/or larger `--max-new-tokens`.",
        )


def _resolve_previous_metrics_path(
    output_root: Path,
    run_name: str,
    current_run_dir: Path,
    compare_with: Path | None,
    compare_latest_same_name: bool,
) -> Path | None:
    if compare_with is not None:
        if not compare_with.exists():
            raise FileNotFoundError(f"compare-with file not found: {compare_with}")
        return compare_with

    if not compare_latest_same_name:
        return None

    candidates = sorted(output_root.glob(f"{run_name}_*/metrics.json"))
    candidates = [c for c in candidates if c.parent != current_run_dir]
    if not candidates:
        return None
    return candidates[-1]


def _compare_metrics(
    current_metrics: dict[str, Any],
    previous_metrics_path: Path,
    current_predictions_path: Path,
    previous_predictions_path: Path,
) -> dict[str, Any]:
    previous_metrics = json.loads(previous_metrics_path.read_text(encoding="utf-8"))
    current_eval_version = str(current_metrics.get("evaluator_version", "unknown"))
    previous_eval_version = str(previous_metrics.get("evaluator_version", "unknown"))
    eval_version_match = current_eval_version == previous_eval_version

    comparison = {
        "previous_metrics_path": str(previous_metrics_path),
        "current_accuracy": float(current_metrics.get("accuracy", 0.0)),
        "previous_accuracy": float(previous_metrics.get("accuracy", 0.0)),
        "delta_accuracy": float(current_metrics.get("accuracy", 0.0))
        - float(previous_metrics.get("accuracy", 0.0)),
        "current_parse_error_rate": float(current_metrics.get("parse_error_rate", 0.0)),
        "previous_parse_error_rate": float(previous_metrics.get("parse_error_rate", 0.0)),
        "delta_parse_error_rate": float(current_metrics.get("parse_error_rate", 0.0))
        - float(previous_metrics.get("parse_error_rate", 0.0)),
        "changed_prediction_count": 0,
        "evaluator_version_match": bool(eval_version_match),
        "current_evaluator_version": current_eval_version,
        "previous_evaluator_version": previous_eval_version,
        "comparison_caution": (
            None
            if eval_version_match
            else (
                "Evaluator version mismatch. Deltas may reflect scoring logic changes, "
                "not only model behavior."
            )
        ),
    }

    if previous_predictions_path.exists():
        curr = _load_pred_map(current_predictions_path)
        prev = _load_pred_map(previous_predictions_path)
        shared_ids = set(curr.keys()) & set(prev.keys())
        changed = sum(1 for sid in shared_ids if curr[sid] != prev[sid])
        comparison["changed_prediction_count"] = changed
        comparison["shared_sample_count"] = len(shared_ids)
    else:
        comparison["shared_sample_count"] = 0

    return comparison


def _load_pred_map(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue
        payload = json.loads(line)
        data[str(payload["sample_id"])] = str(payload.get("raw_prediction", ""))
    return data


def _import_runtime_deps():
    """Import heavy runtime deps lazily.

    Why lazy import?
    ----------------
    This lets `--help` work even if the current shell environment has
    incompatible ML package versions. Actual inference still requires a
    correct environment (for example `conda activate bcr`).
    """
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import runtime dependencies (`torch`, `transformers`). "
            "Please activate your ML environment first (for example: `conda activate bcr`)."
        ) from exc
    return torch, AutoTokenizer, AutoModelForCausalLM


def _configure_transformers_progress_bar(disable: bool) -> None:
    """Enable or disable transformers tqdm-style progress bars."""
    from transformers.utils import logging as hf_logging  # lazy import

    if disable:
        hf_logging.disable_progress_bar()
    else:
        hf_logging.enable_progress_bar()


def _build_progress_checkpoints(
    total: int, log_every: int, max_progress_lines: int
) -> set[int]:
    """Select sparse progress checkpoints to avoid noisy logs.

    Rules:
    - `max_progress_lines` includes the start line.
    - We always include final completion checkpoint (`total`) when `total > 0`.
    - Candidate points follow `log_every`.
    - If too many candidate points exist, we downsample evenly.
    """
    if total <= 0:
        return set()

    if max_progress_lines == 1:
        return {total}

    candidates = [i for i in range(log_every, total + 1, log_every)]
    if not candidates or candidates[-1] != total:
        candidates.append(total)

    max_updates = max_progress_lines - 1
    if len(candidates) <= max_updates:
        return set(candidates)

    if max_updates == 1:
        return {total}

    selected: list[int] = []
    denom = max_updates - 1
    span = len(candidates) - 1
    for i in range(max_updates):
        idx = int(round(i * span / denom))
        selected.append(candidates[idx])
    selected[-1] = total
    return set(selected)


class _TeeWriter:
    """Write text to terminal and a file simultaneously.

    This keeps live progress visible while persisting logs for later debugging.
    """

    def __init__(self, primary: TextIO, mirror: TextIO) -> None:
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        written = self._primary.write(data)
        self._mirror.write(data)
        return written

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())

    @property
    def encoding(self) -> str | None:
        return getattr(self._primary, "encoding", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)


def _fmt_seconds(seconds: int) -> str:
    """Format duration in a compact and readable `HH:MM:SS` style."""
    if seconds < 0:
        return "unknown"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
