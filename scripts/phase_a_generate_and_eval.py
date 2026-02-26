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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

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
        "--log-every",
        type=int,
        default=10,
        help=(
            "Print generation progress every N samples. "
            "Use 1 for per-sample logging."
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

    print("=" * 88)
    print("Phase A: Generate + Evaluate")
    print("=" * 88)
    print(f"input_jsonl : {args.input_jsonl}")
    print(f"model_path  : {args.model_path}")
    print(f"run_dir     : {run_dir}")
    print(f"seed        : {args.seed}")
    print(f"gen_config  : {asdict(gen_cfg)}")
    print(f"log_every   : {args.log_every}")
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=_resolve_dtype(args.dtype, torch_module=torch),
        device_map=args.device_map,
        trust_remote_code=False,
    )
    model.eval()
    _configure_model_generation(model=model, gen_cfg=gen_cfg)
    print(f"first_param : {next(model.parameters()).device}")
    hf_map = getattr(model, "hf_device_map", None)
    if hf_map is not None:
        print(f"hf_device_map: {hf_map}")

    _run_generation(
        rows=rows,
        model=model,
        tokenizer=tokenizer,
        gen_cfg=gen_cfg,
        output_path=predictions_path,
        source_path=args.input_jsonl,
        torch_module=torch,
        log_every=args.log_every,
    )

    scored_rows, summary = _run_evaluation(predictions_path)

    with scored_path.open("w", encoding="utf-8") as fout:
        for row in scored_rows:
            fout.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    metrics = summary.to_dict()
    metrics["run_name"] = args.run_name
    metrics["input_jsonl"] = str(args.input_jsonl)
    metrics["predictions_jsonl"] = str(predictions_path)
    metrics["scored_predictions_jsonl"] = str(scored_path)

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

    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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
        "max_samples": args.max_samples,
        "files": {
            "predictions": str(predictions_path),
            "scored_predictions": str(scored_path),
            "metrics": str(metrics_path),
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
    if comparison is not None:
        print(f"compared_with    : {previous_metrics_path}")
        print(f"delta_accuracy   : {comparison['delta_accuracy']:+.4f}")
        print(f"delta_parse_err  : {comparison['delta_parse_error_rate']:+.4f}")
        print(f"changed_samples  : {comparison['changed_prediction_count']}")
    print(f"metrics_path     : {metrics_path}")
    print("=" * 88)
    return 0


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
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip() == "":
            continue
        row = json.loads(line)
        required = ["sample_id", "dataset", "split", "prompt_text", "answer", "question"]
        missing = [k for k in required if k not in row]
        if missing:
            raise KeyError(f"Missing keys {missing} at line={idx+1} in {path}")
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
) -> None:
    if log_every <= 0:
        raise ValueError(f"--log-every must be >= 1, got {log_every}")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    total = len(rows)
    start_ts = time.perf_counter()
    print(f"generation  : starting {total} samples", flush=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            prompt = str(row["prompt_text"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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
                output_ids = model.generate(**inputs, **gen_kwargs)

            generated = output_ids[0, inputs["input_ids"].shape[1] :]
            raw_prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

            pred_row = {
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
                    "row_index": i,
                },
            }
            fout.write(json.dumps(pred_row, ensure_ascii=False) + "\n")
            # Flush every record so that progress can be monitored externally
            # (for example with `wc -l predictions.jsonl` while job is running).
            fout.flush()

            done = i + 1
            should_log = (done % log_every == 0) or (done == total)
            if should_log:
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
