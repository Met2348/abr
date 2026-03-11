#!/usr/bin/env python3
"""Filter Phase E pair artifacts with one local prefix-correctness judge.

English
-------
This script adds a conservative LLM-as-a-judge layer on top of existing Phase E
pair artifacts.

Design goal:
1. keep the canonical pair JSONL contract unchanged,
2. only audit semantics where a prefix-correctness judge is actually aligned,
3. write one new artifact directory that can be fed directly into the existing
   Phase E trainers.

The current judge contract is intentionally narrow:
- it asks whether *all shown steps so far* are mathematically valid,
- it does not require the prefix to be complete,
- so it is well aligned to local first-bad style pairs,
- but not to terminal-completion-anchor preference pairs.

中文
----
这个脚本是在现有 Phase E pair artifact 之上，加一层保守的 LLM-as-a-judge
过滤。

设计目标是：
1. 不改 canonical pair JSONL 合同，
2. 只审计那些与“prefix correctness” judge 语义真正对齐的 pair，
3. 输出一个新的 artifact 目录，后续可以直接喂给现有 Phase E trainer。

当前 judge 合同故意收得很窄：
- 它只判断“当前前缀里已经出现的步骤是否都正确”，
- 不要求前缀必须已经完整解出答案，
- 因此它适合 local first-bad 这类 pair，
- 但并不天然适合 terminal-completion-anchor 这类偏好 pair。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
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

from ours.phase_b.value_head import (  # noqa: E402
    ensure_tokenizer_has_pad_token,
    maybe_resize_embeddings_for_tokenizer,
)
from ours.phase_d.external_pairs import ExternalPairRecord, load_external_pair_jsonl, summarize_external_pairs  # noqa: E402


DEFAULT_AUDITABLE_SEMANTICS = (
    "local_first_bad_edge",
    "local_modified_process_error_step",
    "first_bad_fanout_prefix_ranking",
    "good_bad_prefix_grid",
)
DEFAULT_BYPASS_SEMANTICS = (
    "terminal_completion_anchor",
    "same_step_completion_preference",
    "same_prompt_binary_verdict",
    "same_prompt_binary_correctness",
    "direct_preference_pair",
)


@dataclass(slots=True)
class PrefixJudgeResult:
    """Structured judge result for one prefix.

    English
    -------
    `overall_verdict` is the primary control signal used by this script.
    `first_incorrect_step` and `confidence` are auxiliary diagnostics that help
    us distinguish contract mismatch from clear label disagreement.

    中文
    ----
    这个脚本真正拿来做过滤决策的主信号是 `overall_verdict`。
    `first_incorrect_step` 与 `confidence` 则主要用于诊断，帮助区分：
    1. 真的标签冲突，
    2. 还是 judge 合同本身与该语义不对齐。
    """

    overall_verdict: str
    first_incorrect_step: int | None
    confidence: float | None
    reason: str
    parse_error: str | None
    raw_output: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_verdict": str(self.overall_verdict),
            "first_incorrect_step": self.first_incorrect_step,
            "confidence": self.confidence,
            "reason": str(self.reason),
            "parse_error": self.parse_error,
            "raw_output": self.raw_output,
        }


@dataclass(slots=True)
class JudgeRequest:
    pair_id: str
    side: str
    prompt_text: str
    reasoning_text: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter one Phase E pair artifact with a local prefix-correctness judge."
    )
    parser.add_argument("--train-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--eval-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-name", default="phase_e_judge_filter")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_pairs"),
    )
    parser.add_argument(
        "--auditable-semantic",
        action="append",
        default=list(DEFAULT_AUDITABLE_SEMANTICS),
        help="Repeatable auditable pair semantics. Defaults to local/fanout/grid semantics.",
    )
    parser.add_argument(
        "--bypass-semantic",
        action="append",
        default=list(DEFAULT_BYPASS_SEMANTICS),
        help="Repeatable bypass semantics. Defaults to terminal/completion-style semantics.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--logging-batches",
        type=int,
        default=4,
        help="Print one progress line every N generation batches.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Drop auditable rows whose chosen/rejected judge confidence is below this threshold.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.train_pairs_jsonl.exists():
        raise FileNotFoundError(f"--train-pairs-jsonl not found: {args.train_pairs_jsonl}")
    if not args.eval_pairs_jsonl.exists():
        raise FileNotFoundError(f"--eval-pairs-jsonl not found: {args.eval_pairs_jsonl}")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")
    if int(args.max_new_tokens) <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if int(args.logging_batches) <= 0:
        raise ValueError("--logging-batches must be > 0")
    if not (0.0 <= float(args.min_confidence) <= 1.0):
        raise ValueError("--min-confidence must be in [0, 1]")
    return args


def _resolve_dtype(name: str, torch_module: Any) -> Any | None:
    if str(name) == "auto":
        return None
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    return mapping[str(name)]


def _system_prompt() -> str:
    return (
        "You are a rigorous math process judge. "
        "I will give you one math problem and one candidate reasoning prefix. "
        "The prefix may be incomplete. Your task is NOT to require a final answer. "
        "Instead, judge whether every numbered step shown so far is mathematically and logically valid. "
        "Return exactly one JSON object and nothing else. Use this schema: "
        '{"overall_verdict":"correct"|"incorrect",'
        '"first_incorrect_step":<integer or null>,'
        '"confidence":<number between 0 and 1>,'
        '"reason":"short explanation"}'
    )


def _build_messages(*, prompt_text: str, reasoning_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": (
                "Problem:\n"
                f"{prompt_text.strip()}\n\n"
                "Candidate reasoning prefix:\n"
                f"{reasoning_text.strip()}\n\n"
                "Judge whether all shown steps so far are valid. Return JSON only."
            ),
        },
    ]


def _extract_json(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = raw_text.strip()
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

    for marker in ['"overall_verdict"', '"first_incorrect_step"', '"confidence"']:
        marker_idx = text.find(marker)
        if marker_idx != -1:
            brace_idx = text.rfind("{", 0, marker_idx)
            if brace_idx != -1:
                candidate = _balanced_object_from(brace_idx)
                if candidate is not None:
                    candidates.append(candidate)

    first_brace = text.find("{")
    if first_brace != -1:
        fallback = _balanced_object_from(first_brace)
        if fallback is not None:
            candidates.append(fallback)

    if not candidates:
        return None, "no_json_object_found"

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload, None
    return None, "json_parse_failed"


def _normalize_judge_payload(payload: dict[str, Any] | None, raw_text: str, parse_error: str | None) -> PrefixJudgeResult:
    if payload is None:
        return PrefixJudgeResult(
            overall_verdict="parse_failed",
            first_incorrect_step=None,
            confidence=None,
            reason="",
            parse_error=str(parse_error or "parse_failed"),
            raw_output=raw_text,
        )
    verdict = str(payload.get("overall_verdict", "")).strip().lower()
    if verdict not in {"correct", "incorrect"}:
        verdict = "parse_failed"
        parse_error = "invalid_overall_verdict"
    first_bad = payload.get("first_incorrect_step")
    if first_bad is None:
        normalized_first_bad = None
    else:
        try:
            normalized_first_bad = int(first_bad)
        except Exception:
            normalized_first_bad = None
            parse_error = parse_error or "invalid_first_incorrect_step"
    confidence = payload.get("confidence")
    normalized_confidence: float | None
    try:
        normalized_confidence = float(confidence) if confidence is not None else None
    except Exception:
        normalized_confidence = None
        parse_error = parse_error or "invalid_confidence"
    if normalized_confidence is not None:
        normalized_confidence = min(1.0, max(0.0, normalized_confidence))
    reason = str(payload.get("reason", "")).strip()
    return PrefixJudgeResult(
        overall_verdict=verdict,
        first_incorrect_step=normalized_first_bad,
        confidence=normalized_confidence,
        reason=reason,
        parse_error=parse_error,
        raw_output=raw_text,
    )


def _resolve_model_input_device(model: Any):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _build_prompt_texts(tokenizer: Any, requests: list[JudgeRequest]) -> list[str]:
    payloads: list[str] = []
    for request in requests:
        payloads.append(
            tokenizer.apply_chat_template(
                _build_messages(prompt_text=request.prompt_text, reasoning_text=request.reasoning_text),
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return payloads


def _run_batch_judge(
    *,
    requests: list[JudgeRequest],
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    torch_module: Any,
) -> dict[tuple[str, str], PrefixJudgeResult]:
    prompt_texts = _build_prompt_texts(tokenizer, requests)
    model_device = _resolve_model_input_device(model)
    batch = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    batch = {key: value.to(model_device) for key, value in batch.items()}
    prompt_lengths = batch["attention_mask"].sum(dim=1).tolist()

    with torch_module.inference_mode():
        outputs = model.generate(
            **batch,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    results: dict[tuple[str, str], PrefixJudgeResult] = {}
    for request, output_row, prompt_len in zip(requests, outputs, prompt_lengths, strict=True):
        new_tokens = output_row[int(prompt_len) :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed, parse_error = _extract_json(raw_text)
        results[(request.pair_id, request.side)] = _normalize_judge_payload(parsed, raw_text, parse_error)
    return results


def _stable_records(rows: list[ExternalPairRecord]) -> list[ExternalPairRecord]:
    return sorted(rows, key=lambda row: hashlib.sha256(str(row.pair_id).encode("utf-8")).hexdigest())


def _should_keep_audited_pair(
    *,
    chosen_result: PrefixJudgeResult,
    rejected_result: PrefixJudgeResult,
    min_confidence: float,
) -> tuple[bool, str]:
    if chosen_result.parse_error is not None or rejected_result.parse_error is not None:
        return False, "parse_failed"
    if chosen_result.confidence is not None and chosen_result.confidence < float(min_confidence):
        return False, "chosen_low_confidence"
    if rejected_result.confidence is not None and rejected_result.confidence < float(min_confidence):
        return False, "rejected_low_confidence"
    if chosen_result.overall_verdict != "correct":
        return False, "chosen_not_correct"
    if rejected_result.overall_verdict != "incorrect":
        return False, "rejected_not_incorrect"
    return True, "judge_agree"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase E Judge Filter Summary",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- model_path: `{summary['model_path']}`",
        f"- num_train_pairs_input: `{summary['num_train_pairs_input']}`",
        f"- num_train_pairs_output: `{summary['num_train_pairs_output']}`",
        f"- keep_rate: `{summary['keep_rate']:.4f}`",
        f"- num_audited_pairs: `{summary['num_audited_pairs']}`",
        f"- num_bypassed_pairs: `{summary['num_bypassed_pairs']}`",
        "",
        "## Decisions",
    ]
    for key, value in sorted(dict(summary["decision_counts"]).items()):
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## By Semantics")
    for key, payload in sorted(dict(summary["by_semantics"]).items()):
        lines.append(
            f"- {key}: kept=`{payload['kept']}` dropped=`{payload['dropped']}` bypassed=`{payload['bypassed']}`"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_pairs, _ = load_external_pair_jsonl(Path(args.train_pairs_jsonl))
    eval_pairs, _ = load_external_pair_jsonl(Path(args.eval_pairs_jsonl))
    train_pairs = _stable_records(train_pairs)
    eval_pairs = _stable_records(eval_pairs)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if bool(args.require_cuda) and not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA is required by --require-cuda but no GPU is visible")

    dtype = _resolve_dtype(str(args.dtype), torch)
    model_path = Path(args.model_path)
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=bool(args.trust_remote_code))
    synthesized_pad_token = ensure_tokenizer_has_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
        device_map=str(args.device_map),
        trust_remote_code=bool(args.trust_remote_code),
    ).eval()
    if synthesized_pad_token:
        maybe_resize_embeddings_for_tokenizer(backbone=model, tokenizer=tokenizer)
    load_elapsed = time.perf_counter() - load_start

    auditable_semantics = {str(item).strip() for item in args.auditable_semantic if str(item).strip()}
    bypass_semantics = {str(item).strip() for item in args.bypass_semantic if str(item).strip()}

    requests: list[JudgeRequest] = []
    pair_lookup = {row.pair_id: row for row in train_pairs}
    semantics_by_pair: dict[str, str] = {}
    for row in train_pairs:
        semantics = str((row.metadata or {}).get("pair_semantics", "unspecified")).strip() or "unspecified"
        semantics_by_pair[row.pair_id] = semantics
        if semantics not in auditable_semantics:
            continue
        requests.append(
            JudgeRequest(
                pair_id=row.pair_id,
                side="chosen",
                prompt_text=row.prompt_text,
                reasoning_text=row.chosen_text,
            )
        )
        requests.append(
            JudgeRequest(
                pair_id=row.pair_id,
                side="rejected",
                prompt_text=row.prompt_text,
                reasoning_text=row.rejected_text,
            )
        )

    judge_results: dict[tuple[str, str], PrefixJudgeResult] = {}
    judge_start = time.perf_counter()
    total_batches = max(1, math.ceil(len(requests) / int(args.batch_size))) if requests else 0
    for batch_idx, start in enumerate(range(0, len(requests), int(args.batch_size)), start=1):
        batch_requests = requests[start : start + int(args.batch_size)]
        judge_results.update(
            _run_batch_judge(
                requests=batch_requests,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=int(args.max_new_tokens),
                torch_module=torch,
            )
        )
        if batch_idx % int(args.logging_batches) == 0 or batch_idx == total_batches:
            print(
                "judge_progress      : "
                f"batch={batch_idx}/{total_batches} "
                f"requests_done={start + len(batch_requests)}/{len(requests)}",
                flush=True,
            )
    judge_elapsed = time.perf_counter() - judge_start

    kept_train_pairs: list[ExternalPairRecord] = []
    audit_rows: list[dict[str, Any]] = []
    decision_counts: dict[str, int] = {}
    by_semantics: dict[str, dict[str, int]] = {}

    for row in train_pairs:
        semantics = semantics_by_pair[row.pair_id]
        semantic_bucket = by_semantics.setdefault(semantics, {"kept": 0, "dropped": 0, "bypassed": 0})
        if semantics in bypass_semantics or semantics not in auditable_semantics:
            kept_train_pairs.append(row)
            semantic_bucket["bypassed"] += 1
            decision_counts["bypass_semantics"] = decision_counts.get("bypass_semantics", 0) + 1
            audit_rows.append(
                {
                    "pair_id": row.pair_id,
                    "pair_semantics": semantics,
                    "decision": "keep",
                    "reason_code": "bypass_semantics",
                }
            )
            continue

        chosen_result = judge_results[(row.pair_id, "chosen")]
        rejected_result = judge_results[(row.pair_id, "rejected")]
        keep_pair, reason_code = _should_keep_audited_pair(
            chosen_result=chosen_result,
            rejected_result=rejected_result,
            min_confidence=float(args.min_confidence),
        )
        if keep_pair:
            kept_train_pairs.append(row)
            semantic_bucket["kept"] += 1
        else:
            semantic_bucket["dropped"] += 1
        decision_counts[reason_code] = decision_counts.get(reason_code, 0) + 1
        audit_rows.append(
            {
                "pair_id": row.pair_id,
                "pair_semantics": semantics,
                "decision": "keep" if keep_pair else "drop",
                "reason_code": reason_code,
                "chosen_judge": chosen_result.to_dict(),
                "rejected_judge": rejected_result.to_dict(),
                "metadata": row.metadata,
            }
        )

    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "train_pairs_jsonl": str(args.train_pairs_jsonl),
                "eval_pairs_jsonl": str(args.eval_pairs_jsonl),
                "model_path": str(args.model_path),
                "auditable_semantics": sorted(auditable_semantics),
                "bypass_semantics": sorted(bypass_semantics),
                "min_confidence": float(args.min_confidence),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:12]
    run_dir = Path(args.output_root) / f"{args.run_name}__{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_path = run_dir / "train_pairs.jsonl"
    eval_path = run_dir / "validation_pairs.jsonl"
    summary_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    manifest_path = run_dir / "manifest.json"
    audit_rows_path = run_dir / "audit_rows.jsonl"

    _write_jsonl(train_path, [row.to_dict() for row in kept_train_pairs])
    _write_jsonl(eval_path, [row.to_dict() for row in eval_pairs])
    _write_jsonl(audit_rows_path, audit_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "model_path": str(args.model_path),
        "load_elapsed_sec": float(load_elapsed),
        "judge_elapsed_sec": float(judge_elapsed),
        "auditable_semantics": sorted(auditable_semantics),
        "bypass_semantics": sorted(bypass_semantics),
        "min_confidence": float(args.min_confidence),
        "num_train_pairs_input": int(len(train_pairs)),
        "num_train_pairs_output": int(len(kept_train_pairs)),
        "num_eval_pairs_output": int(len(eval_pairs)),
        "keep_rate": float(len(kept_train_pairs) / len(train_pairs)) if train_pairs else 0.0,
        "num_audited_pairs": int(sum(1 for row in train_pairs if semantics_by_pair[row.pair_id] in auditable_semantics)),
        "num_bypassed_pairs": int(sum(1 for row in train_pairs if semantics_by_pair[row.pair_id] not in auditable_semantics or semantics_by_pair[row.pair_id] in bypass_semantics)),
        "decision_counts": dict(sorted(decision_counts.items())),
        "by_semantics": {key: dict(value) for key, value in sorted(by_semantics.items())},
        "train_pair_summary": summarize_external_pairs(kept_train_pairs),
        "eval_pair_summary": summarize_external_pairs(eval_pairs),
    }
    manifest = {
        "artifact_stage": "phase_e_judge_filter_pairs_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": str(args.run_name),
        "run_dir": str(run_dir),
        "input_files": {
            "train_pairs_jsonl": str(args.train_pairs_jsonl),
            "eval_pairs_jsonl": str(args.eval_pairs_jsonl),
        },
        "output_files": {
            "train_pairs_jsonl": str(train_path),
            "validation_pairs_jsonl": str(eval_path),
            "audit_rows_jsonl": str(audit_rows_path),
            "summary": str(summary_path),
            "summary_md": str(summary_md_path),
        },
        "filter_config": {
            "model_path": str(args.model_path),
            "auditable_semantics": sorted(auditable_semantics),
            "bypass_semantics": sorted(bypass_semantics),
            "batch_size": int(args.batch_size),
            "max_new_tokens": int(args.max_new_tokens),
            "dtype": str(args.dtype),
            "device_map": str(args.device_map),
            "min_confidence": float(args.min_confidence),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 88)
    print("Phase E: Judge Filter Pairs")
    print("=" * 88)
    print(f"train_pairs_in      : {len(train_pairs)}")
    print(f"train_pairs_out     : {len(kept_train_pairs)}")
    print(f"keep_rate           : {summary['keep_rate']:.4f}")
    print(f"num_audited_pairs   : {summary['num_audited_pairs']}")
    print(f"num_bypassed_pairs  : {summary['num_bypassed_pairs']}")
    print(f"load_elapsed_sec    : {load_elapsed:.2f}")
    print(f"judge_elapsed_sec   : {judge_elapsed:.2f}")
    print(f"run_dir             : {run_dir}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
