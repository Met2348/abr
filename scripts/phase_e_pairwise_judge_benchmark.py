#!/usr/bin/env python3
"""Benchmark one local judge LLM as a pairwise preference judge on Phase E pairs.

English
-------
This script tests the judge setup that current literature and community usage
more strongly support:

1. pairwise comparison instead of pointwise scalar grading,
2. swap-debiasing by judging both `A/B` and `B/A`,
3. light final-block contracts instead of brittle verbose schemas.

It can also emit a label-preserving filtered JSONL when the judge:
1. parses both orders,
2. remains swap-consistent,
3. prefers the original `chosen` side with sufficient confidence.

中文
----
这个脚本专门评测更符合当前论文和社区经验的 judge 用法：

1. 用 pairwise comparison，而不是 pointwise 打分；
2. 用 `A/B` 和 `B/A` 双向判定控制位置偏置；
3. 用轻量 final block，而不是脆弱的冗长结构化合同。

此外，它还能在 judge 满足下列条件时产出一个“保留原标签”的过滤后 JSONL：
1. 两个方向都 parse 成功；
2. swap 后结论一致；
3. judge 以足够置信度支持原始 `chosen` 一侧。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_d.external_pairs import ExternalPairRecord, load_external_pair_jsonl, summarize_external_pairs  # noqa: E402


@dataclass(slots=True)
class JudgePreference:
    """Parsed pairwise decision from one judge output.

    中文
    ----
    `preferred` 只允许是 `A/B/TIE`，后续会再映射回原始 pair 的
    `chosen/rejected/tie` 语义空间。
    """

    preferred: str
    confidence: float | None
    reason: str
    parse_error: str | None
    raw_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferred": self.preferred,
            "confidence": self.confidence,
            "reason": self.reason,
            "parse_error": self.parse_error,
            "raw_text": self.raw_text,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one local judge as a pairwise preference judge on external pair JSONL."
    )
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-name", default="phase_e_pairwise_judge_bench")
    parser.add_argument("--dataset-label", default="", help="Human-readable label for reporting.")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--min-filter-confidence",
        type=float,
        default=0.6,
        help="Confidence threshold for label-preserving keep decisions.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--use-system-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable for DeepSeek-R1 style models if needed.",
    )
    parser.add_argument(
        "--assistant-prefix",
        default="",
        help="Optional assistant prefix appended after chat template.",
    )
    parser.add_argument(
        "--write-filtered-jsonl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write label-preserving kept rows to `filtered_pairs.jsonl`.",
    )
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _resolve_dtype(name: str, torch_module: Any) -> Any | None:
    if name == "auto":
        return None
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    return mapping[name]


def _artifact_run_dir(run_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("assets/artifacts/phase_e_pairwise_judge") / f"{run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _pairwise_contract_prompt() -> str:
    return (
        "You are a rigorous reasoning judge.\n"
        "Two candidate reasoning prefixes solve the same problem.\n"
        "A prefix may be incomplete. Do NOT require a final answer.\n"
        "Prefer the prefix whose shown reasoning is more mathematically correct and reliable so far.\n"
        "Ignore length unless extra steps introduce an error.\n"
        "If both prefixes are equally valid so far, or equally flawed, choose TIE.\n"
        "Do not write analysis before the answer. Return only the final block.\n"
        "End with exactly one final block in this format:\n"
        "[FINAL]\n"
        "PREFERRED=A|B|TIE\n"
        "CONFIDENCE=0.0-1.0\n"
        "REASON=one short sentence\n"
        "[/FINAL]"
    )


def _build_messages(
    *,
    pair: ExternalPairRecord,
    a_text: str,
    b_text: str,
    use_system_prompt: bool,
) -> list[dict[str, str]]:
    user_text = (
        "Problem:\n"
        f"{pair.prompt_text.strip()}\n\n"
        "Candidate A:\n"
        f"{a_text.strip()}\n\n"
        "Candidate B:\n"
        f"{b_text.strip()}\n\n"
        "Which prefix is better so far? End with the final block."
    )
    if use_system_prompt:
        return [
            {"role": "system", "content": _pairwise_contract_prompt()},
            {"role": "user", "content": user_text},
        ]
    return [{"role": "user", "content": f"{_pairwise_contract_prompt()}\n\n{user_text}"}]


def _extract_final_block(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = raw_text.strip()
    body = None
    match = re.search(r"\[FINAL\](.*?)\[/FINAL\]", text, flags=re.DOTALL | re.IGNORECASE)
    if match is not None:
        body = match.group(1).strip()
    else:
        start_match = re.search(r"\[\(?\s*final\s*\]", text, flags=re.IGNORECASE)
        end_match = re.search(r"\[/\s*final\s*\]", text, flags=re.IGNORECASE)
        if start_match is not None:
            start_idx = start_match.end()
            end_idx = end_match.start() if end_match is not None else len(text)
            body = text[start_idx:end_idx].strip()
    if not body:
        # Fallback 1: very short verdicts such as `boxed{A}` / `\\boxed{B}`.
        boxed_match = re.search(r"\\?boxed\s*\{\s*(A|B|TIE)\s*\}", text, flags=re.IGNORECASE)
        if boxed_match is not None:
            return {
                "preferred": boxed_match.group(1).upper(),
                "confidence": None,
                "reason": "",
            }, None
        # Fallback 2: plain preference lines without the enclosing block.
        preferred_match = re.search(
            r"(?:preferred|prefer|choice|answer)\s*[:=]?\s*(A|B|TIE)\b",
            text,
            flags=re.IGNORECASE,
        )
        if preferred_match is not None:
            payload = {
                "preferred": preferred_match.group(1).upper(),
                "confidence": None,
                "reason": "",
            }
            confidence_match = re.search(
                r"confidence\s*[:=]\s*([01](?:\.\d+)?)",
                text,
                flags=re.IGNORECASE,
            )
            if confidence_match is not None:
                payload["confidence"] = max(0.0, min(1.0, float(confidence_match.group(1))))
            return payload, None
        # Fallback 2.5: assistant-prefix completion such as:
        # `A\nCONFIDENCE=0.8\nREASON=...`
        leading_pref = re.match(r"^\s*(A|B|TIE)\b", text, flags=re.IGNORECASE)
        if leading_pref is not None:
            payload = {
                "preferred": leading_pref.group(1).upper(),
                "confidence": None,
                "reason": "",
            }
            confidence_match = re.search(
                r"confidence\s*[:=]\s*([01](?:\.\d+)?)",
                text,
                flags=re.IGNORECASE,
            )
            if confidence_match is not None:
                payload["confidence"] = max(0.0, min(1.0, float(confidence_match.group(1))))
            reason_match = re.search(r"reason\s*[:=]\s*(.+)", text, flags=re.IGNORECASE)
            if reason_match is not None:
                payload["reason"] = reason_match.group(1).strip()
            return payload, None
        # Fallback 3: natural-language conclusions near the end.
        tail = text[-300:].lower()
        if re.search(r"(candidate\s+a|option\s+a|prefix\s+a).{0,40}(better|more correct|prefer)", tail):
            return {"preferred": "A", "confidence": None, "reason": ""}, None
        if re.search(r"(candidate\s+b|option\s+b|prefix\s+b).{0,40}(better|more correct|prefer)", tail):
            return {"preferred": "B", "confidence": None, "reason": ""}, None
        if "tie" in tail or "equally valid" in tail or "equally flawed" in tail:
            return {"preferred": "TIE", "confidence": None, "reason": ""}, None
        return None, "final_block_missing"
    payload: dict[str, Any] = {}
    for line in [line.strip() for line in body.splitlines() if line.strip()]:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = re.sub(r"[^A-Z0-9_]", "", key.strip().upper().replace(" ", "_"))
        value = value.strip()
        if key == "PREFERRED":
            normalized = value.upper().strip()
            if normalized not in {"A", "B", "TIE"}:
                return None, f"invalid_preferred: {value}"
            payload["preferred"] = normalized
        elif key == "CONFIDENCE":
            try:
                confidence = float(value)
            except ValueError:
                return None, f"invalid_confidence: {value}"
            payload["confidence"] = max(0.0, min(1.0, confidence))
        elif key == "REASON":
            payload["reason"] = value
    if "preferred" not in payload:
        return None, "preferred_missing"
    payload.setdefault("confidence", None)
    payload.setdefault("reason", "")
    return payload, None


def _parse_preference(raw_text: str) -> JudgePreference:
    payload, parse_error = _extract_final_block(raw_text)
    if payload is None:
        return JudgePreference(
            preferred="PARSE_ERROR",
            confidence=None,
            reason="",
            parse_error=parse_error,
            raw_text=raw_text,
        )
    return JudgePreference(
        preferred=str(payload["preferred"]),
        confidence=(float(payload["confidence"]) if payload.get("confidence") is not None else None),
        reason=str(payload.get("reason", "")),
        parse_error=None,
        raw_text=raw_text,
    )


def _normalize_preference(raw_preferred: str, *, swapped: bool) -> str:
    if raw_preferred == "TIE":
        return "tie"
    if raw_preferred not in {"A", "B"}:
        return "parse_error"
    if not swapped:
        return "chosen" if raw_preferred == "A" else "rejected"
    return "rejected" if raw_preferred == "A" else "chosen"


def _aggregate_normalized_decisions(decisions: list[str]) -> str:
    votes = [decision for decision in decisions if decision in {"chosen", "rejected", "tie"}]
    if not votes:
        return "parse_error"
    counts = Counter(votes)
    top_count = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == top_count)
    if len(winners) != 1:
        return "tie"
    return winners[0]


def _build_prompt_text(
    *,
    tokenizer: Any,
    pair: ExternalPairRecord,
    a_text: str,
    b_text: str,
    use_system_prompt: bool,
    assistant_prefix: str,
) -> str:
    messages = _build_messages(
        pair=pair,
        a_text=a_text,
        b_text=b_text,
        use_system_prompt=use_system_prompt,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if assistant_prefix:
        prompt_text += assistant_prefix
    return prompt_text


def _batched_generate(
    *,
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    max_input_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
) -> tuple[list[str], list[int]]:
    """Generate one response per prompt and record pre-truncation token lengths.

    中文
    ----
    这里额外返回 prompt token 长度，后续用来统计截断率；这对 judge 很重要，
    因为一旦关键信息在 cutoff 之后，judge 结论会被系统性污染。
    """

    token_lengths = [
        len(tokenizer.encode(prompt_text, add_special_tokens=False))
        for prompt_text in prompt_texts
    ]
    batch = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    batch = {key: value.to(model.device) for key, value in batch.items()}
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature is not None:
        generate_kwargs["temperature"] = temperature
    if top_p is not None:
        generate_kwargs["top_p"] = top_p
    if top_k is not None:
        generate_kwargs["top_k"] = top_k
    with __import__("torch").inference_mode():
        outputs = model.generate(
            **batch,
            **generate_kwargs,
        )
    prompt_token_count = batch["input_ids"].shape[1]
    decoded: list[str] = []
    for row_idx in range(outputs.shape[0]):
        new_tokens = outputs[row_idx, prompt_token_count:]
        decoded.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return decoded, token_lengths


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set but CUDA is unavailable.")

    run_dir = _artifact_run_dir(args.run_name)
    pairs, pair_summary = load_external_pair_jsonl(
        Path(args.pairs_jsonl),
        max_samples=int(args.max_samples),
    )
    model_path = Path(args.model_path).resolve()
    dtype = _resolve_dtype(args.dtype, torch)

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    ).eval()
    load_elapsed = time.perf_counter() - load_start

    rows: list[dict[str, Any]] = []
    kept_pairs: list[dict[str, Any]] = []
    total_batches = (len(pairs) + int(args.batch_size) - 1) // int(args.batch_size)
    truncated_prompt_count = 0
    gen_elapsed_total = 0.0

    for batch_idx in range(total_batches):
        batch_pairs = pairs[batch_idx * int(args.batch_size) : (batch_idx + 1) * int(args.batch_size)]
        prompt_texts: list[str] = []
        request_meta: list[tuple[ExternalPairRecord, bool]] = []
        for pair in batch_pairs:
            prompt_texts.append(
                _build_prompt_text(
                    tokenizer=tokenizer,
                    pair=pair,
                    a_text=pair.chosen_text,
                    b_text=pair.rejected_text,
                    use_system_prompt=bool(args.use_system_prompt),
                    assistant_prefix=str(args.assistant_prefix),
                )
            )
            request_meta.append((pair, False))
            prompt_texts.append(
                _build_prompt_text(
                    tokenizer=tokenizer,
                    pair=pair,
                    a_text=pair.rejected_text,
                    b_text=pair.chosen_text,
                    use_system_prompt=bool(args.use_system_prompt),
                    assistant_prefix=str(args.assistant_prefix),
                )
            )
            request_meta.append((pair, True))

        gen_start = time.perf_counter()
        raw_outputs, token_lengths = _batched_generate(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            max_input_length=int(args.max_input_length),
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.do_sample),
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        gen_elapsed = time.perf_counter() - gen_start
        gen_elapsed_total += gen_elapsed

        parsed_by_pair: dict[str, dict[str, Any]] = {}
        for raw_text, token_length, (pair, swapped) in zip(raw_outputs, token_lengths, request_meta):
            if int(token_length) > int(args.max_input_length):
                truncated_prompt_count += 1
            judge = _parse_preference(raw_text)
            entry = parsed_by_pair.setdefault(
                pair.pair_id,
                {
                    "pair": pair,
                    "ab": None,
                    "ba": None,
                    "prompt_token_len_ab": None,
                    "prompt_token_len_ba": None,
                },
            )
            key = "ba" if swapped else "ab"
            entry[key] = judge
            entry[f"prompt_token_len_{key}"] = int(token_length)

        for pair_id, payload in parsed_by_pair.items():
            pair = payload["pair"]
            ab: JudgePreference = payload["ab"]
            ba: JudgePreference = payload["ba"]
            ab_norm = _normalize_preference(ab.preferred, swapped=False)
            ba_norm = _normalize_preference(ba.preferred, swapped=True)
            aggregate = _aggregate_normalized_decisions([ab_norm, ba_norm])
            both_parse_ok = ab.parse_error is None and ba.parse_error is None
            swap_consistent = both_parse_ok and ab_norm == ba_norm
            mean_conf = None
            conf_values = [value for value in [ab.confidence, ba.confidence] if value is not None]
            if conf_values:
                mean_conf = float(sum(conf_values) / len(conf_values))
            non_tie_usable = both_parse_ok and swap_consistent and aggregate != "tie"
            label_preserving_keep = (
                non_tie_usable
                and aggregate == "chosen"
                and (mean_conf is None or mean_conf >= float(args.min_filter_confidence))
            )
            contradiction = (
                non_tie_usable
                and aggregate == "rejected"
                and (mean_conf is None or mean_conf >= float(args.min_filter_confidence))
            )
            row = {
                "pair_id": pair.pair_id,
                "dataset_label": args.dataset_label,
                "source_tag": pair.source_tag,
                "domain_tag": pair.domain_tag,
                "pair_confidence": float(pair.pair_confidence),
                "pair_semantics": str((pair.metadata or {}).get("pair_semantics", "unspecified")),
                "pair_build_mode": str((pair.metadata or {}).get("pair_build_mode", "unspecified")),
                "prompt_token_len_ab": payload["prompt_token_len_ab"],
                "prompt_token_len_ba": payload["prompt_token_len_ba"],
                "ab": ab.to_dict(),
                "ba": ba.to_dict(),
                "ab_norm": ab_norm,
                "ba_norm": ba_norm,
                "aggregate_decision": aggregate,
                "both_parse_ok": both_parse_ok,
                "swap_consistent": swap_consistent,
                "non_tie_usable": non_tie_usable,
                "mean_confidence": mean_conf,
                "label_preserving_keep": label_preserving_keep,
                "judge_contradiction": contradiction,
            }
            rows.append(row)
            if label_preserving_keep and args.write_filtered_jsonl:
                kept_pairs.append(pair.to_dict())

    def _rate(predicate: list[bool]) -> float:
        return float(sum(predicate) / max(len(predicate), 1))

    pair_acc_ab = _rate([row["ab_norm"] == "chosen" for row in rows])
    pair_acc_majority = _rate([row["aggregate_decision"] == "chosen" for row in rows])
    both_parse_ok_rate = _rate([bool(row["both_parse_ok"]) for row in rows])
    swap_consistency_rate = _rate([bool(row["swap_consistent"]) for row in rows if row["both_parse_ok"]]) if any(
        row["both_parse_ok"] for row in rows
    ) else 0.0
    non_tie_usable_rate = _rate([bool(row["non_tie_usable"]) for row in rows])
    label_preserving_keep_rate = _rate([bool(row["label_preserving_keep"]) for row in rows])
    contradiction_rate = _rate([bool(row["judge_contradiction"]) for row in rows])
    tie_rate = _rate([row["aggregate_decision"] == "tie" for row in rows])
    parse_error_rate = _rate([not row["both_parse_ok"] for row in rows])
    mean_confidence_all = float(
        sum(row["mean_confidence"] for row in rows if row["mean_confidence"] is not None)
        / max(sum(row["mean_confidence"] is not None for row in rows), 1)
    )
    truncation_rate = float(truncated_prompt_count / max(len(rows) * 2, 1))

    by_semantics: dict[str, dict[str, float]] = {}
    semantic_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        semantic_groups.setdefault(str(row["pair_semantics"]), []).append(row)
    for semantic, semantic_rows in sorted(semantic_groups.items()):
        by_semantics[semantic] = {
            "n_rows": int(len(semantic_rows)),
            "pair_acc_majority": float(
                sum(r["aggregate_decision"] == "chosen" for r in semantic_rows) / len(semantic_rows)
            ),
            "swap_consistency_rate": float(
                sum(r["swap_consistent"] for r in semantic_rows if r["both_parse_ok"])
                / max(sum(r["both_parse_ok"] for r in semantic_rows), 1)
            ),
            "label_preserving_keep_rate": float(
                sum(r["label_preserving_keep"] for r in semantic_rows) / len(semantic_rows)
            ),
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "dataset_label": str(args.dataset_label),
        "pairs_jsonl": str(Path(args.pairs_jsonl)),
        "num_pairs": int(len(rows)),
        "pair_summary": pair_summary,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "load_elapsed_sec": round(load_elapsed, 4),
        "generation_elapsed_sec": round(gen_elapsed_total, 4),
        "max_input_length": int(args.max_input_length),
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "use_system_prompt": bool(args.use_system_prompt),
        "assistant_prefix": str(args.assistant_prefix),
        "min_filter_confidence": float(args.min_filter_confidence),
        "both_parse_ok_rate": both_parse_ok_rate,
        "parse_error_rate": parse_error_rate,
        "pair_acc_ab": pair_acc_ab,
        "pair_acc_majority": pair_acc_majority,
        "swap_consistency_rate": swap_consistency_rate,
        "non_tie_usable_rate": non_tie_usable_rate,
        "label_preserving_keep_rate": label_preserving_keep_rate,
        "judge_contradiction_rate": contradiction_rate,
        "tie_rate": tie_rate,
        "mean_confidence_all": mean_confidence_all,
        "truncation_rate": truncation_rate,
        "by_semantics": by_semantics,
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.write_filtered_jsonl:
        with (run_dir / "filtered_pairs.jsonl").open("w", encoding="utf-8") as handle:
            for row in kept_pairs:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_lines = [
        "# Pairwise Judge Benchmark Summary",
        "",
        f"- model_path: `{model_path}`",
        f"- dataset_label: `{args.dataset_label}`",
        f"- pairs_jsonl: `{Path(args.pairs_jsonl)}`",
        f"- num_pairs: `{len(rows)}`",
        f"- both_parse_ok_rate: `{both_parse_ok_rate:.4f}`",
        f"- pair_acc_ab: `{pair_acc_ab:.4f}`",
        f"- pair_acc_majority: `{pair_acc_majority:.4f}`",
        f"- swap_consistency_rate: `{swap_consistency_rate:.4f}`",
        f"- non_tie_usable_rate: `{non_tie_usable_rate:.4f}`",
        f"- label_preserving_keep_rate@{float(args.min_filter_confidence):.2f}: `{label_preserving_keep_rate:.4f}`",
        f"- judge_contradiction_rate@{float(args.min_filter_confidence):.2f}: `{contradiction_rate:.4f}`",
        f"- tie_rate: `{tie_rate:.4f}`",
        f"- mean_confidence_all: `{mean_confidence_all:.4f}`",
        f"- truncation_rate: `{truncation_rate:.4f}`",
        "",
        "## By Semantics",
        "",
        "| pair_semantics | n_rows | pair_acc_majority | swap_consistency_rate | label_preserving_keep_rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for semantic, metrics in by_semantics.items():
        summary_lines.append(
            "| {semantic} | {n_rows} | {pair_acc_majority:.4f} | {swap_consistency_rate:.4f} | {label_preserving_keep_rate:.4f} |".format(
                semantic=semantic,
                **metrics,
            )
        )
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("=" * 88)
    print("Phase E Pairwise Judge Benchmark")
    print("=" * 88)
    print(f"model_path              : {model_path}")
    print(f"dataset_label           : {args.dataset_label}")
    print(f"pairs_jsonl             : {Path(args.pairs_jsonl)}")
    print(f"num_pairs               : {len(rows)}")
    print(f"both_parse_ok_rate      : {both_parse_ok_rate:.4f}")
    print(f"pair_acc_ab             : {pair_acc_ab:.4f}")
    print(f"pair_acc_majority       : {pair_acc_majority:.4f}")
    print(f"swap_consistency_rate   : {swap_consistency_rate:.4f}")
    print(f"non_tie_usable_rate     : {non_tie_usable_rate:.4f}")
    print(f"label_preserving_keep   : {label_preserving_keep_rate:.4f}")
    print(f"judge_contradiction     : {contradiction_rate:.4f}")
    print(f"tie_rate                : {tie_rate:.4f}")
    print(f"mean_confidence_all     : {mean_confidence_all:.4f}")
    print(f"truncation_rate         : {truncation_rate:.4f}")
    print(f"run_dir                 : {run_dir}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
