#!/usr/bin/env python3
"""Run one full Phase A inference-and-evaluation experiment.

Why this file exists
--------------------
Phase A experiments are repeated many times with different prompts, token budgets,
batch sizes, and decoding modes. This script keeps the full loop in one reproducible
entrypoint:
- load prepared prompt/target rows,
- run model inference,
- score outputs with the frozen evaluator,
- persist artifacts and optional comparisons.

What this file does
-------------------
1. Read prepared samples JSONL (for example `validation.jsonl` from
   `scripts/phase_a_prepare.py`).
2. Load a local Hugging Face causal LM and tokenizer.
3. Generate predictions using either:
   - normal free-form generation, or
   - StrategyQA binary-choice scoring.
4. Apply optional batching, OOM backoff, truncation recovery, and chat-marker
   trimming.
5. Evaluate saved predictions into scored outputs and summary metrics.
6. Optionally compare the current run against a previous run with the same name.
7. Persist machine-readable artifacts such as:
   - `predictions.jsonl`
   - `scored_predictions.jsonl`
   - `metrics.json`
   - `manifest.json`
   - `console.log`

What this file contains
-----------------------
- CLI/config parsing
- deterministic runtime setup
- free-form and binary-choice generation helpers
- truncation-recovery helpers
- VRAM telemetry helpers
- evaluation/comparison helpers
- top-level orchestration in `main()`

Execution logic
---------------
`parse_args()` -> runtime dependency import -> reproducibility setup -> run directory
creation -> model/tokenizer load -> generation -> evaluation -> artifact writing ->
optional run comparison.

Interaction with other files
----------------------------
- `scripts/phase_a_prepare.py`: produces the prepared JSONL consumed here.
- `src/ours/phase_a/evaluator.py`: scores predictions after inference.
- `scripts/phase_a_eval_predictions.py`: re-evaluates saved predictions without
  rerunning inference.
- `scripts/run_phase_a_benchmark_suite.sh`: launches this script for named
  experiment groups.

Example
-------
```bash
python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/.../validation.jsonl \
  --run-name strategyqa_baseline \
  --require-cuda \
  --max-new-tokens 64
```
"""

from __future__ import annotations

import argparse
import json
import random
import re
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
    """Add the repo-local `src/` directory to `sys.path`.

    This allows direct execution with `python scripts/phase_a_generate_and_eval.py`
    from the repository root, without requiring an editable package install.

    Example
    -------
    ```bash
    python scripts/phase_a_generate_and_eval.py --help
    ```
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_a import PredictionRecord, evaluate_predictions  # noqa: E402


@dataclass(slots=True)
class GenerationConfig:
    """Generation settings saved to the run manifest for reproducibility.

    Example
    -------
    ```python
    cfg = GenerationConfig(max_new_tokens=128, do_sample=False)
    ```
    """

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
    truncation_recovery_rows: int
    truncation_recovery_rounds: int
    vram_sample_count: int
    vram_mean_total_reserved_gib_sampled: float | None
    vram_max_total_reserved_gib_sampled: float | None
    vram_mean_total_allocated_gib_sampled: float | None
    vram_max_total_allocated_gib_sampled: float | None
    vram_per_device: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert runtime telemetry into a JSON-serializable dictionary.

        Example
        -------
        ```python
        payload = generation_stats.to_dict()
        ```
        """
        return {
            "num_samples": int(self.num_samples),
            "elapsed_seconds": float(self.elapsed_seconds),
            "sample_per_second": float(self.sample_per_second),
            "batch_size": int(self.batch_size),
            "oom_backoff_events": int(self.oom_backoff_events),
            "truncation_recovery_rows": int(self.truncation_recovery_rows),
            "truncation_recovery_rounds": int(self.truncation_recovery_rounds),
            "vram_sample_count": int(self.vram_sample_count),
            "vram_mean_total_reserved_gib_sampled": (
                float(self.vram_mean_total_reserved_gib_sampled)
                if self.vram_mean_total_reserved_gib_sampled is not None
                else None
            ),
            "vram_max_total_reserved_gib_sampled": (
                float(self.vram_max_total_reserved_gib_sampled)
                if self.vram_max_total_reserved_gib_sampled is not None
                else None
            ),
            "vram_mean_total_allocated_gib_sampled": (
                float(self.vram_mean_total_allocated_gib_sampled)
                if self.vram_mean_total_allocated_gib_sampled is not None
                else None
            ),
            "vram_max_total_allocated_gib_sampled": (
                float(self.vram_max_total_allocated_gib_sampled)
                if self.vram_max_total_allocated_gib_sampled is not None
                else None
            ),
            "vram_per_device": list(self.vram_per_device),
        }


@dataclass(slots=True)
class TruncationRecoveryConfig:
    """Control policy for continuation-based truncation recovery.

    Example
    -------
    ```python
    cfg = TruncationRecoveryConfig(
        enabled=True,
        max_rounds=2,
        extra_tokens_per_round=96,
        datasets=("gsm8k", "hendrycks_math"),
    )
    ```
    """

    enabled: bool = True
    max_rounds: int = 2
    extra_tokens_per_round: int = 96
    datasets: tuple[str, ...] = ("gsm8k", "hendrycks_math")
    require_final_answer_signal: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert recovery policy into a JSON-serializable dictionary."""
        return {
            "enabled": bool(self.enabled),
            "max_rounds": int(self.max_rounds),
            "extra_tokens_per_round": int(self.extra_tokens_per_round),
            "datasets": list(self.datasets),
            "require_final_answer_signal": bool(self.require_final_answer_signal),
        }


@dataclass(slots=True)
class FreeformGenerationResult:
    """Hold one free-form generation result plus truncation metadata.

    Example
    -------
    ```python
    result = FreeformGenerationResult(
        raw_prediction="Final answer: 12",
        generated_token_count=8,
        hit_token_limit=False,
    )
    ```
    """
    raw_prediction: str
    generated_token_count: int
    hit_token_limit: bool
    truncation_recovery_applied: bool = False
    truncation_recovery_rounds: int = 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one Phase A inference/evaluation run.

    Returns
    -------
    argparse.Namespace
        Validated run configuration used by `main()`.

    Example
    -------
    ```python
    args = parse_args([
        "--input-jsonl", "validation.jsonl",
        "--run-name", "debug_run",
        "--max-new-tokens", "64",
    ])
    ```
    """
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
        "--adapter-path",
        type=Path,
        default=None,
        help=(
            "Optional PEFT adapter directory to load on top of --model-path during "
            "inference. Use this for Phase B LoRA evaluation."
        ),
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
    parser.add_argument(
        "--truncation-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If a free-form output hits token cap and appears incomplete, "
            "run continuation decoding rounds to recover a clean final answer."
        ),
    )
    parser.add_argument(
        "--truncation-recovery-rounds",
        type=int,
        default=2,
        help="Maximum continuation rounds per sample when truncation recovery is enabled.",
    )
    parser.add_argument(
        "--truncation-recovery-extra-tokens",
        type=int,
        default=96,
        help="Extra decode budget (`max_new_tokens`) used by each recovery round.",
    )
    parser.add_argument(
        "--truncation-recovery-datasets",
        default="gsm8k,hendrycks_math",
        help=(
            "Comma-separated dataset names where truncation recovery is active. "
            "Example: gsm8k,hendrycks_math,strategyqa"
        ),
    )
    parser.add_argument(
        "--truncation-recovery-require-final-answer-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Only trigger recovery when output still lacks a dataset-appropriate "
            "final-answer signal after hitting token cap."
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
    """Run the full Phase A generation, evaluation, and artifact-writing lifecycle.

    This is the core orchestration function for Phase A. It intentionally keeps the
    major stages visible in order so a newcomer can map console output back to code.

    Returns
    -------
    int
        Process exit code. `0` indicates successful completion.

    Example
    -------
    ```bash
    python -u scripts/phase_a_generate_and_eval.py \
      --input-jsonl validation.jsonl \
      --run-name baseline_eval \
      --max-new-tokens 64
    ```
    """
    args = parse_args()
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_jsonl}")

    # Import heavy ML runtime deps lazily so `--help` and lightweight tooling remain fast.
    torch, AutoTokenizer, AutoModelForCausalLM = _import_runtime_deps()

    # Fix RNG behavior before any model work so repeated runs are comparable.
    _set_reproducibility(seed=args.seed, torch_module=torch)

    # Freeze generation-policy and truncation-policy objects early so they can be
    # logged into artifacts and reused consistently across helper calls.
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    trunc_recovery_cfg = TruncationRecoveryConfig(
        enabled=bool(args.truncation_recovery),
        max_rounds=int(args.truncation_recovery_rounds),
        extra_tokens_per_round=int(args.truncation_recovery_extra_tokens),
        datasets=_parse_csv_datasets(args.truncation_recovery_datasets),
        require_final_answer_signal=bool(
            args.truncation_recovery_require_final_answer_signal
        ),
    )

    # Create one timestamped run directory that will contain all persisted outputs.
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
        # Mirror terminal output into a file so long experiments remain debuggable
        # after the terminal session ends.
        console_log_writer = console_log_path.open("w", encoding="utf-8")
        sys.stdout = _TeeWriter(orig_stdout, console_log_writer)
        sys.stderr = _TeeWriter(orig_stderr, console_log_writer)

    try:
        # Stage 1: print a fully explicit run header. This is intentionally verbose
        # because users often paste it directly into experiment notes.
        print("=" * 88)
        print("Phase A: Generate + Evaluate")
        print("=" * 88)
        print(f"input_jsonl : {args.input_jsonl}")
        print(f"model_path  : {args.model_path}")
        print(f"adapter_path: {args.adapter_path if args.adapter_path is not None else '<none>'}")
        print(f"run_dir     : {run_dir}")
        print(f"seed        : {args.seed}")
        print(f"gen_config  : {asdict(gen_cfg)}")
        print(f"decode_mode : strategyqa={args.strategyqa_decode_mode}")
        print(f"trim_markers: {args.truncate_chat_markers}")
        print(f"log_every   : {args.log_every}")
        print(f"max_prog_ln : {args.max_progress_lines}")
        print(f"batch_size  : {args.batch_size}")
        print(f"oom_backoff : {args.oom_backoff}")
        print(
            "trunc_recov : "
            f"enabled={trunc_recovery_cfg.enabled} "
            f"rounds={trunc_recovery_cfg.max_rounds} "
            f"extra_tokens={trunc_recovery_cfg.extra_tokens_per_round} "
            f"datasets={list(trunc_recovery_cfg.datasets)} "
            f"require_final_signal={trunc_recovery_cfg.require_final_answer_signal}"
        )
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

        # Stage 2: load prepared rows and optionally shrink them for a smoke run.
        rows = _load_prepared_rows(args.input_jsonl) # 初次定义rows，调用了上面定义的_load_prepared_rows函数，输入是之前准备好的jsonl文件路径，输出是一个字典列表，每个字典对应jsonl文件中的一行，包含sample_id、dataset、split、prompt_text、answer、question等键值对
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
        print(f"num_inputs  : {len(rows)}")

        # Stage 3: load tokenizer/model and align generation config with runtime settings.
        _configure_transformers_progress_bar(disable=True)
        load_start = time.perf_counter()
        print("model_load  : start")
        tokenizer_load_path = _resolve_tokenizer_load_path(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_load_path,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=_resolve_dtype(args.dtype, torch_module=torch),
            device_map=args.device_map,
            trust_remote_code=False,
        )
        if args.adapter_path is not None:
            model = _attach_peft_adapter_for_inference(
                model=model,
                adapter_path=args.adapter_path,
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

        # Stage 4: run inference and persist raw predictions immediately to JSONL.
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
            truncation_recovery_cfg=trunc_recovery_cfg,
        )

        # Stage 5: score saved predictions through the stable evaluator path.
        scored_rows, summary = _run_evaluation(predictions_path)

        # Stage 6: compute extra diagnostics for math-style datasets where token-cap
        # truncation and weak extraction are common hidden failure modes.
        math_diag = _build_math_generation_diagnostics(
            scored_rows=scored_rows,
            max_new_tokens=gen_cfg.max_new_tokens,
        )
        if math_diag is not None:
            _print_math_generation_diagnostics(math_diag=math_diag)

        with scored_path.open("w", encoding="utf-8") as fout:
            for row in scored_rows:
                fout.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

        # Stage 7: assemble the summary metrics payload that downstream tools and
        # benchmark-suite scripts consume.
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

        # Persist machine-readable metrics before writing the human-readable footer.
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Write a manifest that captures how this run was configured, not just how it scored.
        manifest = {
            "schema_version": 1,
            "script": "scripts/phase_a_generate_and_eval.py",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": args.run_name,
            "input_jsonl": str(args.input_jsonl),
            "model_path": args.model_path,
            "adapter_path": str(args.adapter_path) if args.adapter_path is not None else None,
            "seed": args.seed,
            "dtype": args.dtype,
            "device_map": args.device_map,
            "generation_config": asdict(gen_cfg),
            "strategyqa_decode_mode": args.strategyqa_decode_mode,
            "truncate_chat_markers": bool(args.truncate_chat_markers),
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "oom_backoff": bool(args.oom_backoff),
            "truncation_recovery": trunc_recovery_cfg.to_dict(),
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

        # Final console footer: concise enough for humans, stable enough for logs.
        print("-" * 88)
        print(f"accuracy         : {summary.accuracy:.4f}")
        print(f"parse_error_rate : {summary.parse_error_rate:.4f}")
        print(f"n_parseable      : {summary.n_parseable}")
        print(f"acc_parseable    : {summary.accuracy_parseable:.4f}")
        print(f"gen_elapsed_sec  : {generation_stats.elapsed_seconds:.2f}")
        print(f"gen_sample_rate  : {generation_stats.sample_per_second:.3f} sample/s")
        print(f"oom_backoff_evts : {generation_stats.oom_backoff_events}")
        print(f"trunc_recov_rows : {generation_stats.truncation_recovery_rows}")
        print(f"trunc_recov_rnds : {generation_stats.truncation_recovery_rounds}")
        if generation_stats.vram_sample_count > 0:
            print(
                "vram_mean_gib   : "
                f"{generation_stats.vram_mean_total_reserved_gib_sampled:.2f} "
                "(total reserved, sampled)"
            )
            print(
                "vram_max_gib    : "
                f"{generation_stats.vram_max_total_reserved_gib_sampled:.2f} "
                "(total reserved, sampled)"
            )
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
        # Restore original streams even if the run fails, so later shell output is normal.
        if console_log_writer is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            console_log_writer.close()


def _set_reproducibility(seed: int, torch_module: Any) -> None:
    """Apply deterministic-friendly RNG settings for Python and torch.

    Example
    -------
    ```python
    _set_reproducibility(seed=42, torch_module=torch)
    ```
    """
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    # Use deterministic algorithms where practical.
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False


def _resolve_dtype(dtype_name: str, torch_module: Any):
    """Map a CLI dtype name onto a concrete torch dtype object.

    Example
    -------
    ```python
    dtype = _resolve_dtype("auto", torch)
    ```
    """
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


def _resolve_tokenizer_load_path(model_path: str, adapter_path: Path | None) -> str:
    """Choose which directory should supply tokenizer files for inference.

    For PEFT runs we prefer the adapter directory when it contains tokenizer files,
    because Phase B saves tokenizer assets alongside the adapter.

    Example
    -------
    ```python
    tokenizer_path = _resolve_tokenizer_load_path(
        model_path="assets/models/Qwen2.5-7B-Instruct",
        adapter_path=Path("assets/artifacts/phase_b_runs/run/final_model"),
    )
    ```
    """
    if adapter_path is None:
        return str(model_path)
    if (adapter_path / "tokenizer_config.json").exists():
        return str(adapter_path)
    return str(model_path)


def _attach_peft_adapter_for_inference(model: Any, adapter_path: Path):
    """Load a PEFT adapter onto an already-loaded base model for inference.

    Parameters
    ----------
    model:
        Base causal LM loaded from `--model-path`.
    adapter_path:
        Directory containing PEFT adapter files such as `adapter_config.json`.

    Returns
    -------
    Any
        Adapter-wrapped model ready for evaluation.

    Example
    -------
    ```python
    model = _attach_peft_adapter_for_inference(model, Path("run_dir/final_model"))
    ```
    """
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    try:
        from peft import PeftModel  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import `peft` while loading --adapter-path for inference. "
            "Install `peft` in the active environment."
        ) from exc
    return PeftModel.from_pretrained(model, str(adapter_path))


def _load_prepared_rows(path: Path) -> list[dict[str, Any]]: #这个函数产生后面要切分的rows这个关键变量
    """Load and validate prepared Phase A rows from JSONL.

    This function is strict about missing keys and duplicate `sample_id` values
    because those errors can silently corrupt downstream metrics.

    Example
    -------
    ```python
    rows = _load_prepared_rows(Path("validation.jsonl"))
    The absolute 'path' example: '/home/zling/y/bcr/ref/assets/artifacts/phase_a_prepared/strategyqa/ef2ae6864f9c/validation.jsonl'
    ```
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}. "
            "Pass a valid --input-jsonl path (for example: .../validation.jsonl)."
        )
    if path.is_dir():
        raise IsADirectoryError(
            f"--input-jsonl points to a directory, not a file: {path}. "
            "If you used an env var, verify it with: echo \"$INPUT_JSONL\"."
        )

    rows: list[dict[str, Any]] = [] # 事实上， rows的格式在这里就已经被预定义， 即一个列表，列表中的每个元素都是一个字典，字典中必须包含sample_id、dataset、split、prompt_text、answer、question等键值对，这些键值对在后续的代码中会被直接访问和使用，所以这里的格式定义非常重要，后续的代码才能正确运行
    seen_sample_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f: # 注意， 这个helper函数本质上是一个打开文件然后读取里面内容并返回为对应行的功能，而它完全受其参数path的支配。所以path必须配置对;f的类型是TextIO，全称为_io.TextIOWrapper 也就是一个文本文件的输入输出流对象， 通过这个对象可以读取文件内容， 这里使用了with语句来确保文件在使用完毕后正确关闭， 避免资源泄露等问题。 之后的代码块就是在这个打开的文件对象f上进行enumerate操作， 当成是一个list[str]来遍历并逐行读取每一行的内容。
        for idx, line in enumerate(f, start=1):
            if line.strip() == "":
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL row at line={idx} in {path}: {exc}"
                ) from exc
            required = ["sample_id", "dataset", "split", "prompt_text", "answer", "question"]
            missing = [k for k in required if k not in row] #这里本质上是在对输入的文件当中的每一行进行检查，确保它们都包含了sample_id、dataset、split、prompt_text、answer、question这几个关键字段，如果任何一行只要缺少了其中任何一个字段，就会被记录在missing这个列表当中，最后如果missing列表只要不为空，就会抛出一个KeyError异常，这个异常没有被捕获会直接终止， 这会保证训练的数据是完整的， 不会出现问题， 是一种“零容忍”
            if missing:
                raise KeyError(f"Missing keys {missing} at line={idx} in {path}")
            sample_id = str(row["sample_id"])
            if sample_id in seen_sample_ids:
                raise ValueError(
                    f"Duplicate sample_id={sample_id!r} at line={idx} in {path}. "
                    "Prepared input must have unique sample IDs for trustworthy metrics."
                )
            seen_sample_ids.add(sample_id) #这里是另外一层安全网，防止添加重复的训练片段， 这里把已经看到的sample_id记录在seen_sample_ids这个集合当中，如果再次看到相同的sample_id，就会抛出一个ValueError异常，这也是一种“零容忍”，保证了训练数据的唯一性和完整性
            rows.append(row)
    return rows


def _gib_from_bytes(num_bytes: int) -> float:
    """Convert bytes to GiB using binary units.

    Example
    -------
    ```python
    gib = _gib_from_bytes(1024 ** 3)  # -> 1.0
    ```
    """
    return float(num_bytes) / float(1024**3)


def _create_vram_tracker(torch_module: Any) -> dict[str, Any] | None:
    """Create per-run VRAM tracker state.

    Notes
    -----
    - We sample memory once per processed batch to estimate mean/max usage.
    - We also reset CUDA peak stats so per-device peaks are generation-scoped.
    """
    cuda = getattr(torch_module, "cuda", None) #尝试获取torch的cuda状态
    if cuda is None or not cuda.is_available(): #不支持cuda就自然不跟踪其VRAM状态
        return None

    device_count = int(cuda.device_count()) #cuda设备的总数， 在后面拿来用于对get_device_name里面拿来遍历， 以获取每个设备的情况
    devices: list[dict[str, Any]] = [] #读取的cuda设备信息的结果列表，用来装读取到的信息
    for device_index in range(device_count): #遍历每个cuda设备
        try:
            device_name = str(cuda.get_device_name(device_index)) #有名称的就直接用名称称呼， 没名称的就用序号称呼
        except Exception:
            device_name = f"cuda:{device_index}"
        try:
            cuda.reset_peak_memory_stats(device_index) #这个函数是拿来重置峰值显存用量记录，是个“去皮操作”，把目前已有的显存占用情况记为新的零点，这样即可获知后续本进程真实的额外显存占用情况
        except Exception:
            # Some backends may not support peak reset; continue with sampled stats.
            pass
        devices.append(
            {
                "device_index": int(device_index),
                "device_name": device_name,
                "sum_reserved_bytes": 0,
                "sum_allocated_bytes": 0,
                "max_reserved_bytes_sampled": 0,
                "max_allocated_bytes_sampled": 0,
            }
        )

    return {
        "sample_count": 0,
        "sum_total_reserved_bytes": 0,
        "sum_total_allocated_bytes": 0,
        "max_total_reserved_bytes_sampled": 0,
        "max_total_allocated_bytes_sampled": 0,
        "devices": devices,
    }


def _sample_vram_tracker(vram_tracker: dict[str, Any] | None, torch_module: Any) -> None:
    """Take one VRAM sample from all visible CUDA devices."""
    if vram_tracker is None:
        return

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return

    total_reserved = 0
    total_allocated = 0
    for device_slot in vram_tracker["devices"]:
        device_index = int(device_slot["device_index"])
        reserved = int(cuda.memory_reserved(device_index))
        allocated = int(cuda.memory_allocated(device_index))
        device_slot["sum_reserved_bytes"] += reserved
        device_slot["sum_allocated_bytes"] += allocated
        device_slot["max_reserved_bytes_sampled"] = max(
            int(device_slot["max_reserved_bytes_sampled"]), reserved
        )
        device_slot["max_allocated_bytes_sampled"] = max(
            int(device_slot["max_allocated_bytes_sampled"]), allocated
        )
        total_reserved += reserved
        total_allocated += allocated

    vram_tracker["sample_count"] = int(vram_tracker["sample_count"]) + 1
    vram_tracker["sum_total_reserved_bytes"] += total_reserved
    vram_tracker["sum_total_allocated_bytes"] += total_allocated
    vram_tracker["max_total_reserved_bytes_sampled"] = max(
        int(vram_tracker["max_total_reserved_bytes_sampled"]), total_reserved
    )
    vram_tracker["max_total_allocated_bytes_sampled"] = max(
        int(vram_tracker["max_total_allocated_bytes_sampled"]), total_allocated
    )


def _finalize_vram_tracker(
    vram_tracker: dict[str, Any] | None,
    torch_module: Any,
) -> tuple[int, float | None, float | None, float | None, float | None, list[dict[str, Any]]]:
    """Convert tracker state into report-ready summary fields."""
    if vram_tracker is None:
        return 0, None, None, None, None, []

    sample_count = int(vram_tracker["sample_count"])
    if sample_count <= 0:
        return 0, None, None, None, None, []

    mean_total_reserved = _gib_from_bytes(
        int(vram_tracker["sum_total_reserved_bytes"]) // sample_count
    )
    max_total_reserved = _gib_from_bytes(
        int(vram_tracker["max_total_reserved_bytes_sampled"])
    )
    mean_total_allocated = _gib_from_bytes(
        int(vram_tracker["sum_total_allocated_bytes"]) // sample_count
    )
    max_total_allocated = _gib_from_bytes(
        int(vram_tracker["max_total_allocated_bytes_sampled"])
    )

    cuda = getattr(torch_module, "cuda", None)
    per_device: list[dict[str, Any]] = []
    for device_slot in vram_tracker["devices"]:
        device_index = int(device_slot["device_index"])
        peak_reserved = int(device_slot["max_reserved_bytes_sampled"])
        peak_allocated = int(device_slot["max_allocated_bytes_sampled"])
        if cuda is not None and cuda.is_available():
            try:
                peak_reserved = max(peak_reserved, int(cuda.max_memory_reserved(device_index)))
                peak_allocated = max(
                    peak_allocated, int(cuda.max_memory_allocated(device_index))
                )
            except Exception:
                pass
        per_device.append(
            {
                "device_index": device_index,
                "device_name": str(device_slot["device_name"]),
                "mean_reserved_gib_sampled": _gib_from_bytes(
                    int(device_slot["sum_reserved_bytes"]) // sample_count
                ),
                "max_reserved_gib": _gib_from_bytes(peak_reserved),
                "mean_allocated_gib_sampled": _gib_from_bytes(
                    int(device_slot["sum_allocated_bytes"]) // sample_count
                ),
                "max_allocated_gib": _gib_from_bytes(peak_allocated),
            }
        )

    return (
        sample_count,
        mean_total_reserved,
        max_total_reserved,
        mean_total_allocated,
        max_total_allocated,
        per_device,
    )


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
    truncation_recovery_cfg: TruncationRecoveryConfig | None = None,
) -> GenerationStats:
    """Generate predictions for prepared rows and write them to JSONL.

    This helper owns the main inference loop, including:
    - batch routing,
    - StrategyQA binary-choice handling,
    - OOM backoff,
    - truncation-recovery accounting,
    - progress logging,
    - VRAM telemetry.

    Example
    -------
    ```python
    stats = _run_generation(
        rows=rows,
        model=model,
        tokenizer=tokenizer,
        gen_cfg=gen_cfg,
        output_path=predictions_path,
        source_path=input_path,
        torch_module=torch,
        log_every=10,
        max_progress_lines=5,
        strategyqa_decode_mode="freeform",
        truncate_chat_markers=True,
        batch_size=4,
        oom_backoff=True,
    )
    ```
    """
    if log_every <= 0:
        raise ValueError(f"--log-every must be >= 1, got {log_every}")
    if max_progress_lines <= 0:
        raise ValueError(f"--max-progress-lines must be >= 1, got {max_progress_lines}")
    if batch_size <= 0:
        raise ValueError(f"--batch-size must be >= 1, got {batch_size}")
    if truncation_recovery_cfg is None: # set default truncation recovery config if not provided
        truncation_recovery_cfg = TruncationRecoveryConfig(
            enabled=False,
            max_rounds=0,
            extra_tokens_per_round=96,
            datasets=(),
            require_final_answer_signal=True,
        )
    if truncation_recovery_cfg.max_rounds < 0:
        raise ValueError(
            "--truncation-recovery-rounds must be >= 0, "
            f"got {truncation_recovery_cfg.max_rounds}"
        )
    if truncation_recovery_cfg.extra_tokens_per_round <= 0:
        raise ValueError(
            "--truncation-recovery-extra-tokens must be >= 1, "
            f"got {truncation_recovery_cfg.extra_tokens_per_round}"
        )

    # set default pad_id to eos_token_id if pad_token_id is not set, because some tokenizers don't have a pad token，调试看到的默认值：151643
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
    ) # 用于控制生成进度日志的输出频率和格式
    checkpoint_idx = 0
    oom_backoff_events = 0
    truncation_recovery_rows = 0
    truncation_recovery_rounds = 0
    vram_tracker = _create_vram_tracker(torch_module=torch_module) # 用于监视显存情况
    _sample_vram_tracker(vram_tracker=vram_tracker, torch_module=torch_module)

    with output_path.open("w", encoding="utf-8") as fout: #打开一个文件，准备记录实验的生成情况，调试看到它是每次run里面的predictions.jsonl
        done = 0
        for batch_start in range(0, total, batch_size): #根据batchsize切分定位每个batch的起点和终点index， 然后作为切片依据，用于切分rows，其中total是从输入的jsonl里面读取的行数目
            batch_end = min(batch_start + batch_size, total)
            batch_rows = rows[batch_start:batch_end] #如果batch设置为1，则这个所谓的batchrows里面就只会含有一行

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
                    truncation_recovery_cfg=truncation_recovery_cfg,
                )
                oom_backoff_events += local_oom_events
                for sub_idx, local_idx in enumerate(free_indices):
                    free_result = free_outputs[sub_idx]
                    row = batch_rows[local_idx]
                    if free_result.truncation_recovery_applied:
                        truncation_recovery_rows += 1
                        truncation_recovery_rounds += free_result.truncation_recovery_rounds
                    batch_results[local_idx] = _build_prediction_row(
                        row=row,
                        source_path=source_path,
                        row_index=batch_start + local_idx,
                        raw_prediction=free_result.raw_prediction,
                        generated_token_count=free_result.generated_token_count,
                        hit_token_limit=free_result.hit_token_limit,
                        truncation_recovery_applied=free_result.truncation_recovery_applied,
                        truncation_recovery_rounds=free_result.truncation_recovery_rounds,
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

            # One sample per finished batch for stable, low-overhead VRAM telemetry.
            _sample_vram_tracker(vram_tracker=vram_tracker, torch_module=torch_module)

    total_elapsed = max(time.perf_counter() - start_ts, 1e-6)
    (
        vram_sample_count,
        vram_mean_total_reserved_gib_sampled,
        vram_max_total_reserved_gib_sampled,
        vram_mean_total_allocated_gib_sampled,
        vram_max_total_allocated_gib_sampled,
        vram_per_device,
    ) = _finalize_vram_tracker(vram_tracker=vram_tracker, torch_module=torch_module)
    return GenerationStats(
        num_samples=total,
        elapsed_seconds=float(total_elapsed),
        sample_per_second=float(total / total_elapsed if total else 0.0),
        batch_size=int(batch_size),
        oom_backoff_events=int(oom_backoff_events),
        truncation_recovery_rows=int(truncation_recovery_rows),
        truncation_recovery_rounds=int(truncation_recovery_rounds),
        vram_sample_count=int(vram_sample_count),
        vram_mean_total_reserved_gib_sampled=vram_mean_total_reserved_gib_sampled,
        vram_max_total_reserved_gib_sampled=vram_max_total_reserved_gib_sampled,
        vram_mean_total_allocated_gib_sampled=vram_mean_total_allocated_gib_sampled,
        vram_max_total_allocated_gib_sampled=vram_max_total_allocated_gib_sampled,
        vram_per_device=vram_per_device,
    )


def _build_prediction_row(
    row: dict[str, Any],
    source_path: Path,
    row_index: int,
    raw_prediction: str,
    generated_token_count: int,
    hit_token_limit: bool,
    truncation_recovery_applied: bool = False,
    truncation_recovery_rounds: int = 0,
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
            "truncation_recovery_applied": bool(truncation_recovery_applied),
            "truncation_recovery_rounds": int(truncation_recovery_rounds),
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
    truncation_recovery_cfg: TruncationRecoveryConfig,
) -> tuple[list[FreeformGenerationResult], int]:
    """Generate free-form outputs for a row batch.

    Returns
    -------
    tuple[list[FreeformGenerationResult], int]
        - list per row: generation result with recovery metadata
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
        outputs = _apply_truncation_recovery_if_needed(
            rows=rows,
            outputs=outputs,
            model=model,
            tokenizer=tokenizer,
            gen_cfg=gen_cfg,
            pad_id=pad_id,
            truncate_chat_markers=truncate_chat_markers,
            torch_module=torch_module,
            truncation_recovery_cfg=truncation_recovery_cfg,
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
            truncation_recovery_cfg=truncation_recovery_cfg,
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
            truncation_recovery_cfg=truncation_recovery_cfg,
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
) -> list[FreeformGenerationResult]:
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
    outputs: list[FreeformGenerationResult] = []
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
        outputs.append(
            FreeformGenerationResult(
                raw_prediction=raw_prediction,
                generated_token_count=generated_token_count,
                hit_token_limit=hit_token_limit,
            )
        )
    return outputs


def _apply_truncation_recovery_if_needed(
    rows: list[dict[str, Any]],
    outputs: list[FreeformGenerationResult],
    model: Any,
    tokenizer: Any,
    gen_cfg: GenerationConfig,
    pad_id: int | None,
    truncate_chat_markers: bool,
    torch_module: Any,
    truncation_recovery_cfg: TruncationRecoveryConfig,
) -> list[FreeformGenerationResult]:
    """Apply continuation decoding for rows that likely ended by truncation."""
    if not truncation_recovery_cfg.enabled:
        return outputs
    if truncation_recovery_cfg.max_rounds <= 0:
        return outputs

    recovered_outputs: list[FreeformGenerationResult] = []
    for row, result in zip(rows, outputs, strict=True):
        dataset = str(row.get("dataset", "")).strip().lower()
        if not _needs_truncation_recovery(
            dataset=dataset,
            result=result,
            truncation_recovery_cfg=truncation_recovery_cfg,
        ):
            recovered_outputs.append(result)
            continue
        recovered_outputs.append(
            _recover_truncated_freeform_output(
                row=row,
                result=result,
                model=model,
                tokenizer=tokenizer,
                gen_cfg=gen_cfg,
                pad_id=pad_id,
                truncate_chat_markers=truncate_chat_markers,
                torch_module=torch_module,
                truncation_recovery_cfg=truncation_recovery_cfg,
            )
        )
    return recovered_outputs


def _recover_truncated_freeform_output(
    row: dict[str, Any],
    result: FreeformGenerationResult,
    model: Any,
    tokenizer: Any,
    gen_cfg: GenerationConfig,
    pad_id: int | None,
    truncate_chat_markers: bool,
    torch_module: Any,
    truncation_recovery_cfg: TruncationRecoveryConfig,
) -> FreeformGenerationResult:
    """Continue generation from partial output to reduce cap-hit truncation errors."""
    dataset = str(row.get("dataset", "")).strip().lower()
    current_text = str(result.raw_prediction)
    total_generated_tokens = int(result.generated_token_count)
    current_hit_limit = bool(result.hit_token_limit)
    applied = False
    rounds_used = 0

    recovery_gen_cfg = GenerationConfig(
        max_new_tokens=int(truncation_recovery_cfg.extra_tokens_per_round),
        do_sample=bool(gen_cfg.do_sample),
        temperature=float(gen_cfg.temperature),
        top_p=float(gen_cfg.top_p),
        top_k=int(gen_cfg.top_k),
    )

    for _ in range(int(truncation_recovery_cfg.max_rounds)):
        if not _needs_truncation_recovery(
            dataset=dataset,
            result=FreeformGenerationResult(
                raw_prediction=current_text,
                generated_token_count=total_generated_tokens,
                hit_token_limit=current_hit_limit,
            ),
            truncation_recovery_cfg=truncation_recovery_cfg,
        ):
            break

        rounds_used += 1
        continuation_prompt = f"{row['prompt_text']}{current_text}"
        continuation_result = _generate_freeform_rows_once(
            prompts=[continuation_prompt],
            model=model,
            tokenizer=tokenizer,
            gen_cfg=recovery_gen_cfg,
            pad_id=pad_id,
            truncate_chat_markers=truncate_chat_markers,
            torch_module=torch_module,
        )[0]

        if continuation_result.generated_token_count <= 0:
            break

        applied = True
        current_text = _stitch_prediction_chunks(
            prefix=current_text,
            continuation=continuation_result.raw_prediction,
        )
        total_generated_tokens += int(continuation_result.generated_token_count)
        current_hit_limit = bool(continuation_result.hit_token_limit)

    return FreeformGenerationResult(
        raw_prediction=current_text,
        generated_token_count=total_generated_tokens,
        hit_token_limit=current_hit_limit,
        truncation_recovery_applied=applied,
        truncation_recovery_rounds=(rounds_used if applied else 0),
    )


def _needs_truncation_recovery(
    dataset: str,
    result: FreeformGenerationResult,
    truncation_recovery_cfg: TruncationRecoveryConfig,
) -> bool:
    """Decide whether one output likely needs continuation recovery."""
    if not truncation_recovery_cfg.enabled:
        return False
    if dataset not in set(truncation_recovery_cfg.datasets):
        return False
    if not result.hit_token_limit:
        return False
    if not truncation_recovery_cfg.require_final_answer_signal:
        return True
    return not _has_dataset_final_answer_signal(
        text=result.raw_prediction,
        dataset=dataset,
    )


def _has_dataset_final_answer_signal(text: str, dataset: str) -> bool:
    """Heuristic check for whether output already contains an answer-finalization cue."""
    value = str(text).strip()
    if value == "":
        return False
    lowered = value.lower()

    if dataset in {"gsm8k", "hendrycks_math"}:
        if re.search(r"final\s*answer(?:\s*is)?\s*:", lowered):
            return True
        if "####" in value:
            return True
        if "\\boxed" in value:
            return True
        # Accept a plain numeric-only tail as final signal.
        if re.search(r"[-+]?\d*\.?\d+(?:/\d+)?\s*$", value):
            return True
        return False

    if dataset == "strategyqa":
        if re.search(
            r"(?:final\s*answer|answer|verdict)\s*[:\-]\s*(yes|no|true|false)\b",
            lowered,
        ):
            return True
        if re.search(r"^\s*(yes|no|true|false)\s*$", lowered):
            return True
        return False

    return True


def _stitch_prediction_chunks(prefix: str, continuation: str) -> str:
    """Join continuation text while preserving readable spacing."""
    left = str(prefix).rstrip()
    right = str(continuation).lstrip()
    if left == "":
        return right
    if right == "":
        return left
    return f"{left}\n{right}".strip()


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
    """Load saved predictions and score them with the shared Phase A evaluator.

    Example
    -------
    ```python
    scored_rows, summary = _run_evaluation(predictions_path)
    ```
    """
    records: list[PredictionRecord] = []
    with predictions_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if line.strip() == "":
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid prediction JSONL row at line={idx} in {predictions_path}: {exc}"
                ) from exc
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
                    f"Invalid prediction record at line={idx} in {predictions_path}: {exc}"
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
    """Print human-readable math-generation diagnostics to the console.

    Example
    -------
    ```python
    _print_math_generation_diagnostics(math_diag)
    ```
    """
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
    """Resolve which previous `metrics.json` file should be used for comparison.

    Example
    -------
    ```python
    previous = _resolve_previous_metrics_path(
        output_root=Path("assets/artifacts/phase_a_runs"),
        run_name="baseline",
        current_run_dir=run_dir,
        compare_with=None,
        compare_latest_same_name=True,
    )
    ```
    """
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
    """Compare current metrics/predictions against a previous run.

    Example
    -------
    ```python
    comparison = _compare_metrics(
        current_metrics=metrics,
        previous_metrics_path=prev_metrics,
        current_predictions_path=cur_preds,
        previous_predictions_path=prev_preds,
    )
    ```
    """
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
    """Load a compact `sample_id -> raw_prediction` map from a prediction JSONL file.

    Example
    -------
    ```python
    pred_map = _load_pred_map(predictions_path)
    ```
    """
    data: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if line.strip() == "":
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid prediction JSONL row at line={idx} in {path}: {exc}"
                ) from exc
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


def _parse_csv_datasets(raw: str) -> tuple[str, ...]:
    """Parse comma-separated dataset list into a normalized tuple."""
    parts = [p.strip().lower() for p in str(raw).split(",")]
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part == "":
            continue
        if part in seen:
            continue
        seen.add(part)
        deduped.append(part)
    return tuple(deduped)


class _TeeWriter:
    """Write text to terminal and a file simultaneously.

    This keeps live progress visible while persisting logs for later debugging.
    """

    def __init__(self, primary: TextIO, mirror: TextIO) -> None:
        """Store the live terminal stream and the mirrored log-file stream."""
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        """Write one string to both target streams.

        Example
        -------
        ```python
        tee.write("hello\\n")
        ```
        """
        written = self._primary.write(data)
        self._mirror.write(data)
        return written

    def flush(self) -> None:
        """Flush both target streams so logs stay up to date on disk."""
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        """Proxy TTY detection to the primary stream."""
        return bool(getattr(self._primary, "isatty", lambda: False)())

    @property
    def encoding(self) -> str | None:
        """Expose the primary stream encoding for code that queries it."""
        return getattr(self._primary, "encoding", None)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute access to the primary stream object."""
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
