#!/usr/bin/env python3
"""Smoke-test one local instruct model as a structured step judge.

Why this file exists
--------------------
Phase E is moving toward LLM-as-a-judge style supervision. Before wiring any
judge into relabeling or active-learning loops, we need one deterministic,
cheap script that can answer three questions:

1. Can the local model load cleanly on the current server?
2. Can it follow a strict JSON judging contract?
3. Does it produce sensible step-level verdicts on tiny sanity examples?

This script answers those questions and writes stable smoke artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    """Allow running from repo root without package installation."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()


@dataclass(slots=True)
class JudgeExample:
    """One tiny reasoning sample used for judge smoke tests."""

    example_id: str
    title: str
    question: str
    reasoning: str
    expected_overall_verdict: str
    expected_first_incorrect_step: int | None


EXAMPLES: dict[str, JudgeExample] = {
    "math_bad": JudgeExample(
        example_id="math_bad",
        title="Simple arithmetic with a clear first wrong step",
        question="If Tom has 3 apples and buys 2 more, how many apples does he have?",
        reasoning=(
            "Step 1: Tom starts with 3 apples.\n"
            "Step 2: Buying 2 more means we subtract 2 from 3.\n"
            "Step 3: 3 - 2 = 1, so Tom has 1 apple.\n"
            "Step 4: Therefore the final answer is 1."
        ),
        expected_overall_verdict="incorrect",
        expected_first_incorrect_step=2,
    ),
    "math_good": JudgeExample(
        example_id="math_good",
        title="Simple arithmetic with all steps correct",
        question="If Tom has 3 apples and buys 2 more, how many apples does he have?",
        reasoning=(
            "Step 1: Tom starts with 3 apples.\n"
            "Step 2: Buying 2 more means we add 2 to 3.\n"
            "Step 3: 3 + 2 = 5.\n"
            "Step 4: Therefore the final answer is 5."
        ),
        expected_overall_verdict="correct",
        expected_first_incorrect_step=None,
    ),
    "algebra_bad": JudgeExample(
        example_id="algebra_bad",
        title="Two-step algebra with a sign error",
        question="Solve for x: 2x + 4 = 10.",
        reasoning=(
            "Step 1: Start from 2x + 4 = 10.\n"
            "Step 2: Subtract 4 from both sides to get 2x = 6.\n"
            "Step 3: Divide both sides by 2 to get x = -3.\n"
            "Step 4: Therefore the final answer is x = -3."
        ),
        expected_overall_verdict="incorrect",
        expected_first_incorrect_step=3,
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load one local instruct model and smoke-test LLM-as-a-judge JSON output."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local Hugging Face model path.",
    )
    parser.add_argument(
        "--run-name",
        default="phase_e_judge_smoke",
        help="Artifact name prefix.",
    )
    parser.add_argument(
        "--example",
        action="append",
        choices=sorted(EXAMPLES.keys()),
        default=None,
        help="Repeatable example selector. Defaults to math_bad + math_good.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast when CUDA is unavailable.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-system-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Put the judging contract in a system prompt. Disable for DeepSeek-R1 style models.",
    )
    parser.add_argument(
        "--assistant-prefix",
        default="",
        help="Optional raw prefix appended after the chat template, e.g. '<think>\\n'.",
    )
    parser.add_argument(
        "--write-artifacts",
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


def _system_prompt() -> str:
    return (
        "You are a rigorous reasoning judge. Read the problem and the candidate "
        "reasoning steps. Judge each step for correctness. Your entire answer "
        "must be exactly one JSON object. No prose. No markdown. No code "
        "fences. No LaTeX. If you add any text outside the JSON object, the "
        "answer is wrong. Use the exact schema:\n"
        "{\n"
        '  "step_labels": [\n'
        '    {"step_index": 1, "verdict": "correct" | "incorrect", "reason": "short reason"}\n'
        "  ],\n"
        '  "first_incorrect_step": <integer or null>,\n'
        '  "overall_verdict": "correct" | "incorrect",\n'
        '  "confidence": <number between 0 and 1>\n'
        "}"
    )


def _build_messages(example: JudgeExample) -> list[dict[str, str]]:
    user_text = (
        f"{_system_prompt()}\n\n"
        "Problem:\n"
        f"{example.question}\n\n"
        "Candidate reasoning:\n"
        f"{example.reasoning}\n\n"
        "Judge every numbered step and return JSON only."
    )
    return [{"role": "user", "content": user_text}]


def _extract_json(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = raw_text.strip()
    candidates: list[str] = []

    # 先尝试 fenced code block，因为很多 instruct 模型会把 JSON 包在 ```json 里。
    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL):
        candidates.append(match.group(1))

    # 再尝试从包含关键 schema 字段的位置向外截出平衡的大括号对象。
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

    for marker in ['"step_labels"', '"overall_verdict"', '"first_incorrect_step"']:
        marker_idx = text.find(marker)
        if marker_idx != -1:
            brace_idx = text.rfind("{", 0, marker_idx)
            if brace_idx != -1:
                candidate = _balanced_object_from(brace_idx)
                if candidate is not None:
                    candidates.append(candidate)

    # 最后退回到从首个大括号开始的最大平衡对象。
    first_brace = text.find("{")
    if first_brace != -1:
        fallback = _balanced_object_from(first_brace)
        if fallback is not None:
            candidates.append(fallback)

    if not candidates:
        return None, "no_json_object_found"

    seen: set[str] = set()
    unique_candidates: list[str] = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            unique_candidates.append(item)

    last_error = "json_decode_error: unknown"
    for candidate in unique_candidates:
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as exc:
            last_error = f"json_decode_error: {exc}"
    return None, last_error


def _score_parse(parsed: dict[str, Any] | None, example: JudgeExample) -> dict[str, Any]:
    if parsed is None:
        return {
            "json_ok": False,
            "overall_match": False,
            "first_incorrect_match": False,
        }
    overall = str(parsed.get("overall_verdict", "")).strip().lower()
    first_bad = parsed.get("first_incorrect_step", None)
    return {
        "json_ok": True,
        "overall_match": overall == example.expected_overall_verdict,
        "first_incorrect_match": first_bad == example.expected_first_incorrect_step,
    }


def _build_run_dir(run_name: str, write_artifacts: bool) -> Path | None:
    if not write_artifacts:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("assets/artifacts/phase_e_judge_smoke") / f"{run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    model_path = (repo_root / args.model_path).resolve() if not Path(args.model_path).is_absolute() else Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set but CUDA is unavailable.")

    dtype = _resolve_dtype(args.dtype, torch)
    example_ids = args.example or ["math_bad", "math_good"]
    examples = [EXAMPLES[item] for item in example_ids]
    run_dir = _build_run_dir(args.run_name, args.write_artifacts)

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

    results: list[dict[str, Any]] = []
    for example in examples:
        if args.use_system_prompt:
            messages = [
                {"role": "system", "content": _system_prompt()},
                {
                    "role": "user",
                    "content": (
                        "Problem:\n"
                        f"{example.question}\n\n"
                        "Candidate reasoning:\n"
                        f"{example.reasoning}\n\n"
                        "Judge every numbered step and return JSON only."
                    ),
                },
            ]
        else:
            messages = _build_messages(example)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if args.assistant_prefix:
            prompt_text += args.assistant_prefix
        batch = tokenizer(prompt_text, return_tensors="pt")
        batch = {key: value.to(model.device) for key, value in batch.items()}

        gen_start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_elapsed = time.perf_counter() - gen_start
        new_tokens = output[0, batch["input_ids"].shape[1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed, parse_error = _extract_json(raw_text)
        checks = _score_parse(parsed, example)
        results.append(
            {
                "example": asdict(example),
                "prompt_text": prompt_text,
                "raw_output": raw_text,
                "parsed_output": parsed,
                "parse_error": parse_error,
                "checks": checks,
                "generation_elapsed_sec": round(gen_elapsed, 4),
            }
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "model_name": model_path.name,
        "load_elapsed_sec": round(load_elapsed, 4),
        "dtype": args.dtype,
        "device_map": args.device_map,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "examples": example_ids,
        "n_examples": len(results),
        "n_json_ok": sum(item["checks"]["json_ok"] for item in results),
        "n_overall_match": sum(item["checks"]["overall_match"] for item in results),
        "n_first_incorrect_match": sum(item["checks"]["first_incorrect_match"] for item in results),
        "results": results,
    }

    if run_dir is not None:
        (run_dir / "results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_lines = [
            f"# Judge Smoke Summary: {model_path.name}",
            "",
            f"- model_path: `{model_path}`",
            f"- dtype: `{args.dtype}`",
            f"- device_map: `{args.device_map}`",
            f"- do_sample: `{args.do_sample}`",
            f"- temperature/top_p/top_k: `{args.temperature}` / `{args.top_p}` / `{args.top_k}`",
            f"- load_elapsed_sec: `{summary['load_elapsed_sec']}`",
            f"- examples: `{', '.join(example_ids)}`",
            f"- json_ok: `{summary['n_json_ok']}/{summary['n_examples']}`",
            f"- overall_match: `{summary['n_overall_match']}/{summary['n_examples']}`",
            f"- first_incorrect_match: `{summary['n_first_incorrect_match']}/{summary['n_examples']}`",
            "",
        ]
        for item in results:
            summary_lines.extend(
                [
                    f"## {item['example']['example_id']}",
                    f"- expected_overall: `{item['example']['expected_overall_verdict']}`",
                    f"- expected_first_incorrect_step: `{item['example']['expected_first_incorrect_step']}`",
                    f"- parse_error: `{item['parse_error']}`",
                    f"- checks: `{item['checks']}`",
                    f"- generation_elapsed_sec: `{item['generation_elapsed_sec']}`",
                    "",
                ]
            )
        (run_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("=" * 88)
    print("Phase E Judge LLM Smoke")
    print("=" * 88)
    print(f"model_path        : {model_path}")
    print(f"load_elapsed_sec  : {summary['load_elapsed_sec']}")
    print(f"dtype             : {args.dtype}")
    print(f"device_map        : {args.device_map}")
    print(f"examples          : {', '.join(example_ids)}")
    print("-" * 88)
    for item in results:
        print(f"example           : {item['example']['example_id']}")
        print(f"json_ok           : {item['checks']['json_ok']}")
        print(f"overall_match     : {item['checks']['overall_match']}")
        print(f"first_bad_match   : {item['checks']['first_incorrect_match']}")
        print(f"parse_error       : {item['parse_error']}")
        print(f"gen_elapsed_sec   : {item['generation_elapsed_sec']}")
        print("raw_output        :")
        print(item["raw_output"].strip() or "<empty>")
        print("-" * 88)
    if run_dir is not None:
        print(f"run_dir           : {run_dir}")
        print(f"results_json      : {run_dir / 'results.json'}")
        print(f"summary_md        : {run_dir / 'summary.md'}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
