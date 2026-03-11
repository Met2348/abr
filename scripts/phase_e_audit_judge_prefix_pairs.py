#!/usr/bin/env python3
"""Audit whether a local judge LLM is suitable for Phase E prefix pairs.

Why this file exists
--------------------
The repository now has local judge models, but most Phase E supervision is not
full-solution grading; it is prefix-level comparison. Before wiring a judge into
data cleaning, we need to know whether it can handle this prefix contract:

1. judge the correctness of *shown* steps only,
2. do not punish a prefix for being incomplete,
3. still identify the first incorrect displayed step when one exists.

This script runs that bounded audit and reports agreement with the canonical
chosen/rejected direction of existing pair artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_b.value_head import (  # noqa: E402
    ensure_tokenizer_has_pad_token,
    maybe_resize_embeddings_for_tokenizer,
)
from ours.phase_d.external_pairs import ExternalPairRecord, load_external_pair_jsonl  # noqa: E402
from ours.phase_e.runtime import resolve_dtype, set_seed  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a local judge model on Phase E prefix pairs.")
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("assets/models/Qwen2.5-Math-7B-Instruct"),
    )
    parser.add_argument("--run-name", default="phase_e_judge_prefix_audit")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_judge_audit"),
    )
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _judge_contract() -> str:
    return (
        "You are a rigorous math reasoning judge. "
        "You will receive a math problem and a partial reasoning prefix. "
        "Judge whether every displayed step so far is valid. "
        "Do NOT penalize the prefix for being incomplete or not yet reaching the final answer. "
        "If all shown steps are valid so far, overall_verdict must be 'correct' and first_incorrect_step must be null. "
        "If a shown step is invalid, report the first incorrect displayed step. "
        "Your entire answer must be exactly one JSON object with this schema:\n"
        "{\n"
        '  "step_labels": [\n'
        '    {"step_index": 1, "verdict": "correct" | "incorrect", "reason": "short reason"}\n'
        "  ],\n"
        '  "first_incorrect_step": <integer or null>,\n'
        '  "overall_verdict": "correct" | "incorrect",\n'
        '  "confidence": <number between 0 and 1>\n'
        "}"
    )


def _build_messages(*, pair: ExternalPairRecord, candidate_text: str) -> list[dict[str, str]]:
    user_text = (
        f"{_judge_contract()}\n\n"
        "Problem:\n"
        f"{pair.prompt_text.strip()}\n\n"
        "Candidate reasoning prefix:\n"
        f"{candidate_text.strip()}\n\n"
        "Judge only the displayed steps. Return JSON only."
    )
    return [{"role": "user", "content": user_text}]


def _extract_json(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = str(raw_text or "").strip()
    candidates: list[str] = []
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL):
        candidates.append(match.group(1))

    def _balanced_object_from(start_idx: int) -> str | None:
        depth = 0
        in_string = False
        escaped = False
        begin = None
        for idx in range(start_idx, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if begin is None:
                    begin = idx
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and begin is not None:
                        return text[begin : idx + 1]
        return None

    for marker in ['"overall_verdict"', '"first_incorrect_step"', '"step_labels"']:
        marker_idx = text.find(marker)
        if marker_idx != -1:
            brace_idx = text.rfind("{", 0, marker_idx)
            if brace_idx != -1:
                candidate = _balanced_object_from(brace_idx)
                if candidate is not None:
                    candidates.append(candidate)

    first_brace = text.find("{")
    if first_brace != -1:
        candidate = _balanced_object_from(first_brace)
        if candidate is not None:
            candidates.append(candidate)

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload, None
    return None, "no_valid_json_object_found"


def _stratified_pairs(pairs: list[ExternalPairRecord], *, max_pairs: int) -> list[ExternalPairRecord]:
    buckets: dict[str, list[ExternalPairRecord]] = {}
    for pair in pairs:
        semantics = str((pair.metadata or {}).get("pair_semantics", "unspecified")).strip() or "unspecified"
        buckets.setdefault(semantics, []).append(pair)
    for key in list(buckets.keys()):
        buckets[key] = sorted(buckets[key], key=lambda item: str(item.pair_id))
    ordered_keys = sorted(buckets)
    selected: list[ExternalPairRecord] = []
    pointer = {key: 0 for key in ordered_keys}
    while len(selected) < int(max_pairs):
        advanced = False
        for key in ordered_keys:
            idx = pointer[key]
            if idx >= len(buckets[key]):
                continue
            selected.append(buckets[key][idx])
            pointer[key] += 1
            advanced = True
            if len(selected) >= int(max_pairs):
                break
        if not advanced:
            break
    return selected


def _generate_one(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    torch_module: Any,
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch_module.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output_ids = generated[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.pairs_jsonl.exists():
        raise FileNotFoundError(f"--pairs-jsonl not found: {args.pairs_jsonl}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"--model-path not found: {args.model_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if bool(args.require_cuda) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by --require-cuda but no GPU is visible")

    set_seed(int(args.seed), torch, strict_determinism=False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    synthesized_pad_token = ensure_tokenizer_has_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        device_map=str(args.device_map),
        torch_dtype=resolve_dtype(str(args.dtype), torch),
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    if synthesized_pad_token:
        maybe_resize_embeddings_for_tokenizer(backbone=model, tokenizer=tokenizer)

    pairs, pair_stats = load_external_pair_jsonl(args.pairs_jsonl)
    selected_pairs = _stratified_pairs(pairs, max_pairs=int(args.max_pairs))

    run_dir = Path(args.output_root) / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "judge_records.jsonl"
    summary_json = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"

    print("=" * 88)
    print("Phase E: Judge Prefix Audit")
    print("=" * 88)
    print(f"pairs_jsonl        : {args.pairs_jsonl}")
    print(f"model_path         : {args.model_path}")
    print(f"run_dir            : {run_dir}")
    print(f"selected_pairs     : {len(selected_pairs)}")
    print("=" * 88)

    start = time.perf_counter()
    json_ok_total = 0
    pair_agree_total = 0
    chosen_correct_total = 0
    rejected_incorrect_total = 0
    by_semantics: dict[str, dict[str, int]] = {}

    with records_path.open("w", encoding="utf-8") as handle:
        for pair in selected_pairs:
            semantics = str((pair.metadata or {}).get("pair_semantics", "unspecified")).strip() or "unspecified"
            stats = by_semantics.setdefault(
                semantics,
                {
                    "pairs": 0,
                    "json_ok": 0,
                    "pair_agreement": 0,
                    "chosen_correct": 0,
                    "rejected_incorrect": 0,
                },
            )
            stats["pairs"] += 1
            chosen_raw = _generate_one(
                model=model,
                tokenizer=tokenizer,
                messages=_build_messages(pair=pair, candidate_text=pair.chosen_text),
                max_new_tokens=int(args.max_new_tokens),
                torch_module=torch,
            )
            rejected_raw = _generate_one(
                model=model,
                tokenizer=tokenizer,
                messages=_build_messages(pair=pair, candidate_text=pair.rejected_text),
                max_new_tokens=int(args.max_new_tokens),
                torch_module=torch,
            )
            chosen_payload, chosen_error = _extract_json(chosen_raw)
            rejected_payload, rejected_error = _extract_json(rejected_raw)
            json_ok = chosen_payload is not None and rejected_payload is not None
            if json_ok:
                json_ok_total += 1
                stats["json_ok"] += 1
            chosen_correct = (
                isinstance(chosen_payload, dict)
                and str(chosen_payload.get("overall_verdict", "")).strip().lower() == "correct"
            )
            rejected_incorrect = (
                isinstance(rejected_payload, dict)
                and str(rejected_payload.get("overall_verdict", "")).strip().lower() == "incorrect"
            )
            if chosen_correct:
                chosen_correct_total += 1
                stats["chosen_correct"] += 1
            if rejected_incorrect:
                rejected_incorrect_total += 1
                stats["rejected_incorrect"] += 1
            pair_agreement = bool(chosen_correct and rejected_incorrect)
            if pair_agreement:
                pair_agree_total += 1
                stats["pair_agreement"] += 1

            record = {
                "pair_id": str(pair.pair_id),
                "pair_semantics": semantics,
                "prompt_text": pair.prompt_text,
                "chosen_text": pair.chosen_text,
                "rejected_text": pair.rejected_text,
                "chosen_raw": chosen_raw,
                "rejected_raw": rejected_raw,
                "chosen_payload": chosen_payload,
                "rejected_payload": rejected_payload,
                "chosen_error": chosen_error,
                "rejected_error": rejected_error,
                "json_ok": json_ok,
                "chosen_correct": chosen_correct,
                "rejected_incorrect": rejected_incorrect,
                "pair_agreement": pair_agreement,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - start
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "model_path": str(args.model_path),
        "pairs_jsonl": str(args.pairs_jsonl),
        "input_pair_stats": pair_stats,
        "num_pairs_audited": int(len(selected_pairs)),
        "pair_json_ok_rate": float(json_ok_total / len(selected_pairs)) if selected_pairs else 0.0,
        "chosen_correct_rate": float(chosen_correct_total / len(selected_pairs)) if selected_pairs else 0.0,
        "rejected_incorrect_rate": float(rejected_incorrect_total / len(selected_pairs)) if selected_pairs else 0.0,
        "pair_agreement_rate": float(pair_agree_total / len(selected_pairs)) if selected_pairs else 0.0,
        "by_pair_semantics": by_semantics,
        "elapsed_sec": float(elapsed),
        "output_files": {
            "judge_records": str(records_path),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Phase E Judge Prefix Audit Summary",
        "",
        f"- pair_json_ok_rate: `{float(summary['pair_json_ok_rate']):.4f}`",
        f"- chosen_correct_rate: `{float(summary['chosen_correct_rate']):.4f}`",
        f"- rejected_incorrect_rate: `{float(summary['rejected_incorrect_rate']):.4f}`",
        f"- pair_agreement_rate: `{float(summary['pair_agreement_rate']):.4f}`",
        "",
        "## By Pair Semantics",
    ]
    for key, payload in sorted(by_semantics.items()):
        n = max(1, int(payload.get("pairs", 0)))
        lines.extend(
            [
                f"- `{key}`: n=`{n}`",
                f"  json_ok=`{int(payload.get('json_ok', 0)) / n:.4f}`",
                f"  chosen_correct=`{int(payload.get('chosen_correct', 0)) / n:.4f}`",
                f"  rejected_incorrect=`{int(payload.get('rejected_incorrect', 0)) / n:.4f}`",
                f"  pair_agreement=`{int(payload.get('pair_agreement', 0)) / n:.4f}`",
            ]
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"pair_json_ok_rate  : {float(summary['pair_json_ok_rate']):.4f}")
    print(f"pair_agreement_rate: {float(summary['pair_agreement_rate']):.4f}")
    print(f"elapsed_sec        : {elapsed:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
