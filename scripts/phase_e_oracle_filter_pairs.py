#!/usr/bin/env python3
"""Filter Phase E pair artifacts with a local PRM oracle.

Why this file exists
--------------------
Current Phase E data still inherits substantial label noise from automatic
Math-Shepherd style supervision.  The repository already has a strong local
process reward model checkpoint (`Qwen2.5-Math-PRM-7B`), so the cheapest next
experiment is not a brand-new judge stack; it is to ask whether a second,
independent process verifier can clean the current pair pool.

This script does exactly that:
1. load canonical train/eval pair JSONL files,
2. score chosen/rejected prefixes with the PRM oracle,
3. keep only pairs whose chosen side looks sufficiently good and rejected side
   looks sufficiently bad,
4. write a new deterministic artifact directory for downstream training.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_d.external_pairs import ExternalPairRecord, load_external_pair_jsonl  # noqa: E402
from ours.phase_e.runtime import resolve_dtype, set_seed  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter Phase E pairs with Qwen2.5-Math-PRM style prefix rewards."
    )
    parser.add_argument("--train-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--eval-pairs-jsonl", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("assets/models/Qwen2.5-Math-PRM-7B"),
        help="Local PRM oracle checkpoint.",
    )
    parser.add_argument("--run-name", default="phase_e_prm_oracle_filter")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_pairs"),
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--chosen-threshold", type=float, default=0.60)
    parser.add_argument("--rejected-threshold", type=float, default=0.40)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _split_reasoning_steps(text: str) -> list[str]:
    steps = [line.strip() for line in str(text).replace("\r\n", "\n").split("\n") if line.strip()]
    if not steps:
        return [str(text).strip()]
    return steps


def _build_conversation(
    *,
    tokenizer: Any,
    prompt_text: str,
    response_text: str,
) -> str:
    steps = _split_reasoning_steps(response_text)
    assistant_text = "<extra_0>".join(steps) + "<extra_0>"
    messages = [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {"role": "user", "content": str(prompt_text).strip()},
        {"role": "assistant", "content": assistant_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _score_texts_with_prm(
    *,
    texts: list[str],
    model: Any,
    tokenizer: Any,
    step_sep_id: int,
    batch_size: int,
    max_length: int,
    torch_module: Any,
) -> list[list[float]]:
    if not texts:
        return []
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    outputs_all: list[list[float]] = []
    for start in range(0, len(texts), int(batch_size)):
        batch_texts = texts[start : start + int(batch_size)]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_length),
        ).to(device)
        with torch_module.no_grad():
            outputs = model(**inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else getattr(outputs, "logits", outputs[0])
            probabilities = torch_module.softmax(logits, dim=-1)
            token_masks = inputs["input_ids"] == int(step_sep_id)
            for row_idx in range(probabilities.shape[0]):
                positions = torch_module.nonzero(token_masks[row_idx], as_tuple=False).view(-1)
                if int(positions.numel()) <= 0:
                    outputs_all.append([])
                    continue
                positive_probs = probabilities[row_idx, positions, 1]
                outputs_all.append([float(v) for v in positive_probs.detach().cpu().tolist()])
    return outputs_all


def _augment_and_filter_pairs(
    *,
    pairs: list[ExternalPairRecord],
    chosen_step_scores: list[list[float]],
    rejected_step_scores: list[list[float]],
    chosen_threshold: float,
    rejected_threshold: float,
    min_margin: float,
) -> tuple[list[ExternalPairRecord], dict[str, Any]]:
    kept: list[ExternalPairRecord] = []
    kept_margins: list[float] = []
    dropped_by_reason = {
        "missing_step_scores": 0,
        "chosen_below_threshold": 0,
        "rejected_above_threshold": 0,
        "margin_below_threshold": 0,
    }
    by_semantics: dict[str, dict[str, int]] = {}
    final_chosen_scores: list[float] = []
    final_rejected_scores: list[float] = []

    for pair, chosen_scores, rejected_scores in zip(
        pairs,
        chosen_step_scores,
        rejected_step_scores,
        strict=True,
    ):
        semantics = str((pair.metadata or {}).get("pair_semantics", "unspecified")).strip() or "unspecified"
        bucket = by_semantics.setdefault(
            semantics,
            {"total": 0, "kept": 0},
        )
        bucket["total"] += 1
        if not chosen_scores or not rejected_scores:
            dropped_by_reason["missing_step_scores"] += 1
            continue
        chosen_final = float(chosen_scores[-1])
        rejected_final = float(rejected_scores[-1])
        margin = float(chosen_final - rejected_final)
        final_chosen_scores.append(chosen_final)
        final_rejected_scores.append(rejected_final)
        if chosen_final < float(chosen_threshold):
            dropped_by_reason["chosen_below_threshold"] += 1
            continue
        if rejected_final > float(rejected_threshold):
            dropped_by_reason["rejected_above_threshold"] += 1
            continue
        if margin < float(min_margin):
            dropped_by_reason["margin_below_threshold"] += 1
            continue
        metadata = dict(pair.metadata or {})
        metadata.update(
            {
                "oracle_filter_model": "Qwen2.5-Math-PRM-7B",
                "oracle_filter_mode": "prm_step_reward_consensus",
                "oracle_chosen_step_scores": chosen_scores,
                "oracle_rejected_step_scores": rejected_scores,
                "oracle_chosen_final_score": chosen_final,
                "oracle_rejected_final_score": rejected_final,
                "oracle_margin": margin,
            }
        )
        kept.append(
            ExternalPairRecord(
                pair_id=str(pair.pair_id),
                source_tag=str(pair.source_tag),
                domain_tag=str(pair.domain_tag),
                prompt_text=str(pair.prompt_text),
                chosen_text=str(pair.chosen_text),
                rejected_text=str(pair.rejected_text),
                pair_confidence=float(pair.pair_confidence),
                quality_flags=dict(pair.quality_flags or {}),
                metadata=metadata,
            )
        )
        kept_margins.append(margin)
        bucket["kept"] += 1

    summary = {
        "num_pairs_before": int(len(pairs)),
        "num_pairs_after": int(len(kept)),
        "keep_rate": float(len(kept) / len(pairs)) if pairs else 0.0,
        "dropped_by_reason": dropped_by_reason,
        "mean_oracle_chosen_final": float(statistics.mean(final_chosen_scores)) if final_chosen_scores else 0.0,
        "mean_oracle_rejected_final": float(statistics.mean(final_rejected_scores)) if final_rejected_scores else 0.0,
        "mean_oracle_margin_kept": float(statistics.mean(kept_margins)) if kept_margins else 0.0,
        "by_pair_semantics": by_semantics,
    }
    return kept, summary


def _write_pairs(path: Path, pairs: list[ExternalPairRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")


def _render_summary_md(*, train_summary: dict[str, Any], eval_summary: dict[str, Any]) -> str:
    def _rows(summary: dict[str, Any]) -> list[str]:
        rows = [
            f"- keep_rate: `{float(summary.get('keep_rate', 0.0)):.4f}`",
            f"- mean_oracle_chosen_final: `{float(summary.get('mean_oracle_chosen_final', 0.0)):.4f}`",
            f"- mean_oracle_rejected_final: `{float(summary.get('mean_oracle_rejected_final', 0.0)):.4f}`",
            f"- mean_oracle_margin_kept: `{float(summary.get('mean_oracle_margin_kept', 0.0)):.4f}`",
            "- dropped_by_reason:",
        ]
        for key, value in sorted(dict(summary.get("dropped_by_reason", {})).items()):
            rows.append(f"  - `{key}`: `{int(value)}`")
        rows.append("- by_pair_semantics:")
        for key, payload in sorted(dict(summary.get("by_pair_semantics", {})).items()):
            rows.append(
                "  - "
                f"`{key}`: kept `{int(payload.get('kept', 0))}` / total `{int(payload.get('total', 0))}`"
            )
        return rows

    lines = [
        "# Phase E PRM Oracle Filter Summary",
        "",
        "## Train",
        *_rows(train_summary),
        "",
        "## Eval",
        *_rows(eval_summary),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.train_pairs_jsonl.exists():
        raise FileNotFoundError(f"--train-pairs-jsonl not found: {args.train_pairs_jsonl}")
    if not args.eval_pairs_jsonl.exists():
        raise FileNotFoundError(f"--eval-pairs-jsonl not found: {args.eval_pairs_jsonl}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"--model-path not found: {args.model_path}")
    if bool(args.require_cuda):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by --require-cuda but no GPU is visible")

    train_pairs, train_stats = load_external_pair_jsonl(
        args.train_pairs_jsonl,
        max_samples=args.max_train_samples,
    )
    eval_pairs, eval_stats = load_external_pair_jsonl(
        args.eval_pairs_jsonl,
        max_samples=args.max_eval_samples,
    )

    import torch
    from transformers import AutoModel, AutoTokenizer

    set_seed(int(args.seed), torch, strict_determinism=False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(args.model_path),
        device_map=str(args.device_map),
        torch_dtype=resolve_dtype(str(args.dtype), torch),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()

    step_sep_tokens = tokenizer.encode("<extra_0>", add_special_tokens=False)
    if len(step_sep_tokens) != 1:
        raise RuntimeError(f"Unexpected <extra_0> tokenization: {step_sep_tokens!r}")
    step_sep_id = int(step_sep_tokens[0])

    run_dir = Path(args.output_root) / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_out = run_dir / "train_pairs.jsonl"
    eval_out = run_dir / "validation_pairs.jsonl"
    summary_json = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"
    manifest_path = run_dir / "manifest.json"

    print("=" * 88)
    print("Phase E: PRM Oracle Filter")
    print("=" * 88)
    print(f"train_pairs_jsonl  : {args.train_pairs_jsonl}")
    print(f"eval_pairs_jsonl   : {args.eval_pairs_jsonl}")
    print(f"model_path         : {args.model_path}")
    print(f"run_dir            : {run_dir}")
    print(f"num_train_pairs    : {len(train_pairs)}")
    print(f"num_eval_pairs     : {len(eval_pairs)}")
    print("=" * 88)

    start = time.perf_counter()
    train_chosen = [
        _build_conversation(tokenizer=tokenizer, prompt_text=pair.prompt_text, response_text=pair.chosen_text)
        for pair in train_pairs
    ]
    train_rejected = [
        _build_conversation(tokenizer=tokenizer, prompt_text=pair.prompt_text, response_text=pair.rejected_text)
        for pair in train_pairs
    ]
    eval_chosen = [
        _build_conversation(tokenizer=tokenizer, prompt_text=pair.prompt_text, response_text=pair.chosen_text)
        for pair in eval_pairs
    ]
    eval_rejected = [
        _build_conversation(tokenizer=tokenizer, prompt_text=pair.prompt_text, response_text=pair.rejected_text)
        for pair in eval_pairs
    ]
    train_chosen_scores = _score_texts_with_prm(
        texts=train_chosen,
        model=model,
        tokenizer=tokenizer,
        step_sep_id=step_sep_id,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        torch_module=torch,
    )
    train_rejected_scores = _score_texts_with_prm(
        texts=train_rejected,
        model=model,
        tokenizer=tokenizer,
        step_sep_id=step_sep_id,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        torch_module=torch,
    )
    eval_chosen_scores = _score_texts_with_prm(
        texts=eval_chosen,
        model=model,
        tokenizer=tokenizer,
        step_sep_id=step_sep_id,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        torch_module=torch,
    )
    eval_rejected_scores = _score_texts_with_prm(
        texts=eval_rejected,
        model=model,
        tokenizer=tokenizer,
        step_sep_id=step_sep_id,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        torch_module=torch,
    )

    filtered_train, train_summary = _augment_and_filter_pairs(
        pairs=train_pairs,
        chosen_step_scores=train_chosen_scores,
        rejected_step_scores=train_rejected_scores,
        chosen_threshold=float(args.chosen_threshold),
        rejected_threshold=float(args.rejected_threshold),
        min_margin=float(args.min_margin),
    )
    filtered_eval, eval_summary = _augment_and_filter_pairs(
        pairs=eval_pairs,
        chosen_step_scores=eval_chosen_scores,
        rejected_step_scores=eval_rejected_scores,
        chosen_threshold=float(args.chosen_threshold),
        rejected_threshold=float(args.rejected_threshold),
        min_margin=float(args.min_margin),
    )
    elapsed = time.perf_counter() - start

    _write_pairs(train_out, filtered_train)
    _write_pairs(eval_out, filtered_eval)

    summary_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "model_path": str(args.model_path),
        "filter_config": {
            "chosen_threshold": float(args.chosen_threshold),
            "rejected_threshold": float(args.rejected_threshold),
            "min_margin": float(args.min_margin),
            "batch_size": int(args.batch_size),
            "max_length": int(args.max_length),
        },
        "input_stats": {
            "train": train_stats,
            "eval": eval_stats,
        },
        "train_summary": train_summary,
        "eval_summary": eval_summary,
        "elapsed_sec": float(elapsed),
        "output_files": {
            "train_pairs": str(train_out),
            "validation_pairs": str(eval_out),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "manifest": str(manifest_path),
        },
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md.write_text(
        _render_summary_md(train_summary=train_summary, eval_summary=eval_summary),
        encoding="utf-8",
    )
    manifest = {
        "artifact_stage": "phase_e_prm_oracle_filter_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": str(args.run_name),
        "model_path": str(args.model_path),
        "train_pairs_jsonl": str(args.train_pairs_jsonl),
        "eval_pairs_jsonl": str(args.eval_pairs_jsonl),
        "summary": summary_payload,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"kept_train_rate    : {float(train_summary['keep_rate']):.4f}")
    print(f"kept_eval_rate     : {float(eval_summary['keep_rate']):.4f}")
    print(f"elapsed_sec        : {elapsed:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
