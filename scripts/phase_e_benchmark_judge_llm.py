#!/usr/bin/env python3
"""Benchmark one local judge LLM on small ProcessBench slices.

English
-------
This script answers a narrower question than full Phase E training:

1. if we use one local instruct model directly as a judge,
2. how well can it identify bad reasoning steps on real ProcessBench rows,
3. and how stable is its output contract in practice?

It is intentionally cheap:

1. it evaluates only a deterministic small subset,
2. it writes inspectable artifacts,
3. and it focuses on judge-readiness rather than final research claims.

中文
----
这个脚本回答的是一个比完整 Phase E 训练更窄的问题：

1. 如果直接把一个本地 instruct 模型当 judge 用，
2. 它在真实 ProcessBench 样本上能不能识别错误步骤，
3. 输出契约在工程上到底稳不稳？

它故意保持便宜：

1. 只评测一个确定性的子集，
2. 会把中间结果完整留档，
3. 重点看 judge readiness，而不是最终论文结论。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    """Allow running the script from repo root without package installation.

    中文
    ----
    允许在 repo 根目录直接运行，不要求先把项目安装成包。
    """

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_e.benchmark_eval import ProcessBenchExample, load_processbench_examples  # noqa: E402


@dataclass(slots=True)
class JudgeEvalRow:
    """One evaluated example with parsed judge outputs.

    English
    -------
    Each row stores:
    1. the benchmark example,
    2. the raw model output,
    3. the parsed structured verdict,
    4. and downstream correctness checks.

    中文
    ----
    每条记录保留：
    1. benchmark 原样本，
    2. judge 原始输出，
    3. 解析后的结构化判定，
    4. 以及后续 correctness 检查结果。
    """

    example_id: str
    benchmark_id: str
    expected_overall: str
    expected_first_bad: int | None
    raw_output: str
    parsed: dict[str, Any] | None
    parse_error: str | None
    overall_ok: bool
    first_bad_ok: bool
    first_bad_within_one: bool
    step_acc: float
    generation_elapsed_sec: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one local judge LLM on deterministic ProcessBench subsets."
    )
    parser.add_argument("--model-path", required=True, help="Local model path.")
    parser.add_argument(
        "--run-name",
        default="phase_e_judge_bench",
        help="Artifact name prefix.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=["processbench_math", "processbench_gsm8k"],
        default=None,
        help="Repeatable benchmark selector. Defaults to both.",
    )
    parser.add_argument(
        "--max-samples-per-benchmark",
        type=int,
        default=8,
        help="Deterministic cap per benchmark.",
    )
    parser.add_argument(
        "--contract-mode",
        choices=["full_steps", "first_bad_only"],
        default="full_steps",
        help="Judge output contract. `first_bad_only` is lighter and tests whether verbose step lines are the main blocker.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=320)
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
        help="Use a system prompt for the output contract. Disable for DeepSeek-R1 style models.",
    )
    parser.add_argument(
        "--assistant-prefix",
        default="",
        help="Optional raw assistant prefix appended after the chat template.",
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
    """Map CLI dtype strings to torch dtypes.

    中文
    ----
    把 CLI 里的 dtype 字符串映射成 torch dtype；`auto` 返回 `None`。
    """

    if name == "auto":
        return None
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    return mapping[name]


def _contract_prompt(*, contract_mode: str) -> str:
    """Return the line-based judge contract.

    English
    -------
    We intentionally do not require raw JSON here. The goal is to separate:
    1. reasoning quality,
    2. from strict JSON formatting fragility.

    中文
    ----
    这里故意不用原始 JSON 契约。目的就是把：
    1. 模型的推理判断能力，
    2. 和 strict JSON 格式脆弱性
    先分开看。
    """

    if contract_mode == "first_bad_only":
        return (
            "You are a rigorous reasoning judge.\n"
            "Read the problem and candidate reasoning steps.\n"
            "You may think first, but you must end with one final block in exactly this format:\n"
            "[FINAL]\n"
            "OVERALL=correct|incorrect\n"
            "FIRST_BAD=none|1|2|3|...\n"
            "[/FINAL]\n"
            "If all steps are correct, use FIRST_BAD=none."
        )
    return (
        "You are a rigorous reasoning judge.\n"
        "Read the problem and candidate reasoning steps.\n"
        "You may think first, but you must end with one final block in exactly this format:\n"
        "[FINAL]\n"
        "OVERALL=correct|incorrect\n"
        "FIRST_BAD=none|1|2|3|...\n"
        "STEP_1=correct|incorrect\n"
        "STEP_2=correct|incorrect\n"
        "... one STEP_i line for every step ...\n"
        "[/FINAL]\n"
        "Do not omit any step line."
    )


def _format_reasoning_steps(steps: list[str]) -> str:
    """Format numbered reasoning steps for the judge prompt.

    中文
    ----
    把 benchmark 中的 step 列表转成稳定的编号文本，方便后续和 judge 输出对齐。
    """

    return "\n".join(f"Step {idx + 1}: {step}" for idx, step in enumerate(steps))


def _build_messages(
    *,
    example: ProcessBenchExample,
    use_system_prompt: bool,
    contract_mode: str,
) -> list[dict[str, str]]:
    """Build chat messages for one ProcessBench row.

    中文
    ----
    DeepSeek-R1 系列官方建议避免 system prompt，因此这里支持两种消息构造：
    1. `use_system_prompt=True`
    2. `use_system_prompt=False`
    """

    user_text = (
        "Problem:\n"
        f"{example.problem}\n\n"
        "Candidate reasoning:\n"
        f"{_format_reasoning_steps(example.steps)}\n\n"
        "Judge the reasoning and end with the final block."
    )
    if use_system_prompt:
        return [
            {"role": "system", "content": _contract_prompt(contract_mode=contract_mode)},
            {"role": "user", "content": user_text},
        ]
    return [{"role": "user", "content": f"{_contract_prompt(contract_mode=contract_mode)}\n\n{user_text}"}]


def _expected_step_verdicts(example: ProcessBenchExample) -> list[str]:
    """Return expected per-step verdicts from ProcessBench label semantics.

    English
    -------
    Public ProcessBench semantics:
    - `label < 0`: all steps correct
    - otherwise: first incorrect step is 0-based `label`

    中文
    ----
    ProcessBench 公开语义：
    - `label < 0`: 所有步骤都正确
    - 否则 `label` 表示第一处错误的 0-based step index
    """

    if int(example.label) < 0:
        return ["correct"] * len(example.steps)
    return [
        "correct" if idx < int(example.label) else "incorrect"
        for idx in range(len(example.steps))
    ]


def _extract_final_block(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the `[FINAL] ... [/FINAL]` judge contract.

    中文
    ----
    这里只认最后的结构块，不试图把前面的自由文本也当成结构化结果。
    这样能兼容会先思考、最后再给答案的 reasoning model。
    """

    text = raw_text.strip()
    body = None

    # Strict block first.
    match = re.search(r"\[FINAL\](.*?)\[/FINAL\]", text, flags=re.DOTALL | re.IGNORECASE)
    if match is not None:
        body = match.group(1).strip()
    else:
        # Fallback for near-miss variants such as:
        # `[(final] ... [/final]`
        # `[(FINAL] ...`
        # 先兼容一些真实出现过的近似结构块。
        start_match = re.search(r"\[\(?\s*final\s*\]", text, flags=re.IGNORECASE)
        end_match = re.search(r"\[/\s*final\s*\]", text, flags=re.IGNORECASE)
        if start_match is not None:
            start_idx = start_match.end()
            end_idx = end_match.start() if end_match is not None else len(text)
            body = text[start_idx:end_idx].strip()
    if not body:
        return None, "final_block_missing"
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    payload: dict[str, Any] = {"step_verdicts": {}}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = re.sub(r"[^A-Z0-9_]", "", key.strip().upper().replace(" ", "_"))
        value = value.strip().lower()
        if key in {"OVERALL", "OVERALLVERDICT", "OVER_ALL"}:
            payload["overall"] = value
        elif key in {"FIRST_BAD", "FIRSTBAD"}:
            if value == "none":
                payload["first_bad"] = None
            else:
                try:
                    payload["first_bad"] = int(value)
                except ValueError:
                    return None, f"invalid_first_bad: {value}"
        elif key.startswith("STEP_"):
            try:
                step_idx = int(key.split("_", 1)[1])
            except ValueError:
                return None, f"invalid_step_key: {key}"
            payload["step_verdicts"][step_idx] = value
    if "overall" not in payload:
        return None, "overall_missing"
    if "first_bad" not in payload:
        step_verdicts = dict(payload.get("step_verdicts", {}))
        incorrect_steps = sorted(
            int(step_idx)
            for step_idx, verdict in step_verdicts.items()
            if str(verdict).strip().lower() == "incorrect"
        )
        if incorrect_steps:
            payload["first_bad"] = int(incorrect_steps[0])
        elif step_verdicts:
            payload["first_bad"] = None
        else:
            return None, "first_bad_missing"
    return payload, None


def _score_one_example(
    *,
    example: ProcessBenchExample,
    raw_text: str,
    parsed: dict[str, Any] | None,
    parse_error: str | None,
    generation_elapsed_sec: float,
    benchmark_id: str,
) -> JudgeEvalRow:
    """Score one parsed judge output against ProcessBench truth.

    中文
    ----
    这里分开记录：
    1. overall 是否对，
    2. first bad 是否 exact 对，
    3. first bad 是否在 ±1 内，
    4. step verdict 的平均准确率。
    """

    expected_overall = "correct" if int(example.label) < 0 else "incorrect"
    expected_first_bad = None if int(example.label) < 0 else int(example.label) + 1
    expected_steps = _expected_step_verdicts(example)
    if parsed is None:
        return JudgeEvalRow(
            example_id=str(example.example_id),
            benchmark_id=benchmark_id,
            expected_overall=expected_overall,
            expected_first_bad=expected_first_bad,
            raw_output=raw_text,
            parsed=None,
            parse_error=parse_error,
            overall_ok=False,
            first_bad_ok=False,
            first_bad_within_one=False,
            step_acc=0.0,
            generation_elapsed_sec=round(generation_elapsed_sec, 4),
        )

    overall_pred = str(parsed.get("overall", "")).strip().lower()
    first_bad_pred = parsed.get("first_bad", None)
    step_preds: dict[int, str] = {
        int(idx): str(verdict).strip().lower()
        for idx, verdict in dict(parsed.get("step_verdicts", {})).items()
    }
    if not step_preds:
        if first_bad_pred is None:
            step_preds = {idx: "correct" for idx in range(1, len(expected_steps) + 1)}
        elif isinstance(first_bad_pred, int):
            step_preds = {
                idx: ("correct" if idx < int(first_bad_pred) else "incorrect")
                for idx in range(1, len(expected_steps) + 1)
            }
    total_steps = len(expected_steps)
    correct_steps = 0
    for idx, expected in enumerate(expected_steps, start=1):
        predicted = step_preds.get(idx, "")
        if predicted == expected:
            correct_steps += 1
    overall_ok = overall_pred == expected_overall
    first_bad_ok = first_bad_pred == expected_first_bad
    within_one = False
    if expected_first_bad is None:
        within_one = first_bad_pred is None
    elif isinstance(first_bad_pred, int):
        within_one = abs(first_bad_pred - expected_first_bad) <= 1
    return JudgeEvalRow(
        example_id=str(example.example_id),
        benchmark_id=benchmark_id,
        expected_overall=expected_overall,
        expected_first_bad=expected_first_bad,
        raw_output=raw_text,
        parsed=parsed,
        parse_error=parse_error,
        overall_ok=overall_ok,
        first_bad_ok=first_bad_ok,
        first_bad_within_one=within_one,
        step_acc=float(correct_steps / max(total_steps, 1)),
        generation_elapsed_sec=round(generation_elapsed_sec, 4),
    )


def _summarize_rows(rows: list[JudgeEvalRow]) -> dict[str, Any]:
    """Aggregate benchmark-level metrics.

    中文
    ----
    输出的指标尽量直接支持工程决策：
    1. parse 成功率
    2. overall acc
    3. first-bad exact / within-1
    4. step-level 平均准确率
    """

    if not rows:
        return {
            "n_rows": 0,
            "parse_ok_rate": 0.0,
            "overall_acc": 0.0,
            "first_bad_exact_acc": 0.0,
            "first_bad_within_one_acc": 0.0,
            "mean_step_acc": 0.0,
            "mean_generation_elapsed_sec": 0.0,
        }
    error_rows = [row for row in rows if row.expected_first_bad is not None]
    return {
        "n_rows": int(len(rows)),
        "parse_ok_rate": float(sum(row.parsed is not None for row in rows) / len(rows)),
        "overall_acc": float(sum(row.overall_ok for row in rows) / len(rows)),
        "first_bad_exact_acc": float(sum(row.first_bad_ok for row in error_rows) / max(len(error_rows), 1)),
        "first_bad_within_one_acc": float(
            sum(row.first_bad_within_one for row in error_rows) / max(len(error_rows), 1)
        ),
        "mean_step_acc": float(sum(row.step_acc for row in rows) / len(rows)),
        "mean_generation_elapsed_sec": float(
            sum(row.generation_elapsed_sec for row in rows) / len(rows)
        ),
    }


def _artifact_run_dir(run_name: str) -> Path:
    """Create a stable timestamped run directory.

    中文
    ----
    每次 judge benchmark 都单独落盘，避免不同模型/配置互相覆盖。
    """

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path("assets/artifacts/phase_e_judge_bench") / f"{run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set but CUDA is unavailable.")

    benchmark_ids = args.benchmark or ["processbench_math", "processbench_gsm8k"]
    benchmark_paths = {
        "processbench_math": Path("assets/external_datasets/qwen_processbench/math.json"),
        "processbench_gsm8k": Path("assets/external_datasets/qwen_processbench/gsm8k.json"),
    }
    dtype = _resolve_dtype(args.dtype, torch)
    model_path = Path(args.model_path).resolve()
    run_dir = _artifact_run_dir(args.run_name)

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

    all_rows: list[dict[str, Any]] = []
    benchmark_summaries: dict[str, Any] = {}
    for benchmark_id in benchmark_ids:
        examples = load_processbench_examples(
            benchmark_paths[benchmark_id],
            max_samples=int(args.max_samples_per_benchmark),
        )
        scored_rows: list[JudgeEvalRow] = []
        for example in examples:
            messages = _build_messages(
                example=example,
                use_system_prompt=bool(args.use_system_prompt),
                contract_mode=str(args.contract_mode),
            )
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
            parsed, parse_error = _extract_final_block(raw_text)
            scored = _score_one_example(
                example=example,
                raw_text=raw_text,
                parsed=parsed,
                parse_error=parse_error,
                generation_elapsed_sec=gen_elapsed,
                benchmark_id=benchmark_id,
            )
            scored_rows.append(scored)
            all_rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "example_id": scored.example_id,
                    "problem": example.problem,
                    "steps": example.steps,
                    "label": int(example.label),
                    "expected_overall": scored.expected_overall,
                    "expected_first_bad": scored.expected_first_bad,
                    "raw_output": raw_text,
                    "parsed": parsed,
                    "parse_error": parse_error,
                    "overall_ok": scored.overall_ok,
                    "first_bad_ok": scored.first_bad_ok,
                    "first_bad_within_one": scored.first_bad_within_one,
                    "step_acc": scored.step_acc,
                    "generation_elapsed_sec": scored.generation_elapsed_sec,
                }
            )
        benchmark_summaries[benchmark_id] = _summarize_rows(scored_rows)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "model_name": model_path.name,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "load_elapsed_sec": round(load_elapsed, 4),
        "benchmarks": benchmark_ids,
        "max_samples_per_benchmark": int(args.max_samples_per_benchmark),
        "contract_mode": str(args.contract_mode),
        "use_system_prompt": bool(args.use_system_prompt),
        "assistant_prefix": args.assistant_prefix,
        "do_sample": bool(args.do_sample),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "benchmark_summaries": benchmark_summaries,
        "rows": all_rows,
    }
    (run_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_lines = [
        f"# Judge Benchmark Summary: {model_path.name}",
        "",
        f"- model_path: `{model_path}`",
        f"- dtype: `{args.dtype}`",
        f"- device_map: `{args.device_map}`",
        f"- load_elapsed_sec: `{round(load_elapsed, 4)}`",
        f"- benchmarks: `{', '.join(benchmark_ids)}`",
        f"- max_samples_per_benchmark: `{args.max_samples_per_benchmark}`",
        f"- contract_mode: `{args.contract_mode}`",
        f"- use_system_prompt: `{args.use_system_prompt}`",
        f"- assistant_prefix: `{args.assistant_prefix}`",
        f"- do_sample: `{args.do_sample}`",
        f"- temperature/top_p/top_k: `{args.temperature}` / `{args.top_p}` / `{args.top_k}`",
        "",
        "## Benchmark Metrics",
        "",
        "| benchmark | n_rows | parse_ok | overall_acc | first_bad_exact | first_bad_within1 | mean_step_acc | mean_gen_sec |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark_id, summary in benchmark_summaries.items():
        summary_lines.append(
            "| {bench} | {n_rows} | {parse_ok_rate:.4f} | {overall_acc:.4f} | "
            "{first_bad_exact_acc:.4f} | {first_bad_within_one_acc:.4f} | {mean_step_acc:.4f} | {mean_generation_elapsed_sec:.2f} |".format(
                bench=benchmark_id,
                **summary,
            )
        )
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("=" * 88)
    print("Phase E Judge Benchmark")
    print("=" * 88)
    print(f"model_path        : {model_path}")
    print(f"load_elapsed_sec  : {round(load_elapsed, 4)}")
    print(f"dtype             : {args.dtype}")
    print(f"device_map        : {args.device_map}")
    print(f"benchmarks        : {', '.join(benchmark_ids)}")
    print(f"max_samples       : {args.max_samples_per_benchmark}")
    print("-" * 88)
    for benchmark_id, summary in benchmark_summaries.items():
        print(f"benchmark         : {benchmark_id}")
        for key, value in summary.items():
            print(f"{key:18s}: {value}")
        print("-" * 88)
    print(f"run_dir           : {run_dir}")
    print(f"results_json      : {run_dir / 'results.json'}")
    print(f"rows_jsonl        : {run_dir / 'rows.jsonl'}")
    print(f"summary_md        : {run_dir / 'summary.md'}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
