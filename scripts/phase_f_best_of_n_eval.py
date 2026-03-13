#!/usr/bin/env python3
# Changelog (prepend-only, newest first):
# 2026-03-12: Initial creation. Phase F live validation: PRM-guided Best-of-N on GSM8K.
"""Phase F live generation eval: PRM-guided Best-of-N reranking.

English
-------
Validates that our value head (PBR32 or any LoRA-adapted PRM) can guide inference-time
compute allocation by reranking K sampled solutions per math problem.

Experiment design:
1. Load GSM8K test set (up to --num-problems problems).
2. Generate K solutions per problem with Qwen2.5-Math-7B-Instruct (temperature > 0).
3. For each solution, score all step-level prefixes with the value head.
4. Aggregate step scores → solution score (mean of step scores).
5. Select the highest-scored solution (PRM-reranked).
6. Compare:
   - greedy (K=1, temperature=0): baseline accuracy
   - oracle (K=N, select by ground truth): upper bound
   - random-N (K=N, random selection): sampling baseline
   - PRM-reranked (K=N, select by PRM score): our method

This is the "Phase F3" live validation gate: if PRM-reranked > random-N + 2%,
the process reward signal is strong enough to guide search and RL.

中文
----
验证 value head 能否在推理阶段指导计算分配（Best-of-N 重排序）。
如果 PRM 重排 > 随机抽取 + 2%，说明 process reward 信号足以引导搜索，RL 可行。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()


# ── helpers ──────────────────────────────────────────────────────────────────


def _extract_gsm8k_answer(solution_text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K solution."""
    # Standard GSM8K format: "#### <answer>" at end
    m = re.search(r"####\s*([^\n]+)", solution_text)
    if m:
        return m.group(1).strip().replace(",", "")
    # Fallback: look for last number in solution
    nums = re.findall(r"-?\d+(?:\.\d+)?", solution_text)
    return nums[-1] if nums else None


def _extract_gsm8k_gt_answer(gt_answer_text: str) -> str | None:
    """Extract ground truth answer from GSM8K label (#### ... format)."""
    m = re.search(r"####\s*([^\n]+)", gt_answer_text)
    if m:
        return m.group(1).strip().replace(",", "")
    return None


def _extract_math_gt_answer(answer: str) -> str | None:
    """Extract GT answer from PRM800K MATH splits (answer field is already clean)."""
    s = str(answer).strip()
    return s if s else None


def _extract_boxed_answer(solution_text: str) -> str | None:
    """Extract answer from \\boxed{} in generated solution."""
    m = re.search(r"\\boxed\{([^}]+)\}", solution_text)
    if m:
        return m.group(1).strip()
    return None


def _normalize_math_str(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\left|\\right|\\,|\\;|\\!", "", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    return s.lower()


def _answers_match(pred: str | None, gt: str | None) -> bool:
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred.replace(",", "")) - float(gt.replace(",", ""))) < 1e-3
    except ValueError:
        pass
    pn = _normalize_math_str(pred)
    gn = _normalize_math_str(gt)
    if pn == gn:
        return True
    try:
        import ast
        return abs(float(ast.literal_eval(pn)) - float(ast.literal_eval(gn))) < 1e-3
    except Exception:
        pass
    return False


def _split_steps(solution_text: str) -> list[str]:
    """Split solution into step prefixes (cumulative, \n\n delimited)."""
    parts = [p.strip() for p in solution_text.split("\n\n") if p.strip()]
    if not parts:
        return [solution_text.strip()]
    prefixes: list[str] = []
    cumulative = ""
    for part in parts:
        cumulative = (cumulative + "\n\n" + part).lstrip("\n")
        prefixes.append(cumulative)
    return prefixes


# ── value head inference ─────────────────────────────────────────────────────


def _load_value_head(
    *,
    value_run_dir: Path,
    backbone_path: str,
    device: str,
    dtype_str: str,
    torch_module: Any,
) -> tuple[Any, Any, Any]:
    """Load backbone + LoRA adapter + value head for scoring."""
    from transformers import AutoTokenizer

    from ours.phase_b.value_head import (
        SigmoidValueHead,
        ValueHeadConfig,
        ensure_tokenizer_has_pad_token,
        load_value_head_checkpoint,
        pool_last_token,
    )
    from ours.phase_e.runtime import resolve_backbone_loader_family, resolve_dtype

    resolved_dtype = resolve_dtype(dtype_str, torch_module)

    tokenizer = AutoTokenizer.from_pretrained(backbone_path, trust_remote_code=True)
    ensure_tokenizer_has_pad_token(tokenizer)

    _load_kwargs: dict = {
        "torch_dtype": resolved_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    _family = resolve_backbone_loader_family(model_path=backbone_path, trust_remote_code=True)
    if _family == "process_reward_model":
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(backbone_path, **_load_kwargs)
    else:
        from transformers import AutoModelForCausalLM

        backbone = AutoModelForCausalLM.from_pretrained(backbone_path, **_load_kwargs)

    # Try to load LoRA adapter
    adapter_dir = value_run_dir / "best_adapter"
    if adapter_dir.exists():
        from ours.phase_e.runtime import attach_peft_adapter_for_inference

        backbone = attach_peft_adapter_for_inference(backbone, adapter_dir)
        print(f"value_head      : loaded LoRA adapter from {adapter_dir}")
    backbone.eval()
    backbone.to(device)

    # Load value head
    ckpt_path = value_run_dir / "best_value_head.pt"
    value_head, _vh_config, _extra = load_value_head_checkpoint(ckpt_path)
    value_head.eval()
    value_head.to(device)

    return backbone, tokenizer, value_head


def _score_solution(
    *,
    solution_text: str,
    backbone: Any,
    tokenizer: Any,
    value_head: Any,
    max_length: int,
    device: str,
    torch_module: Any,
) -> float:
    """Score a solution by averaging value head scores over all step prefixes."""
    from ours.phase_b.value_head import encode_text_features, pool_last_token

    prefixes = _split_steps(solution_text)
    if not prefixes:
        return 0.5

    scores: list[float] = []
    batch_size = 8
    for i in range(0, len(prefixes), batch_size):
        batch = prefixes[i : i + batch_size]
        with torch_module.no_grad():
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            outputs = backbone(
                **tokens,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]
            pooled = pool_last_token(last_hidden, tokens["attention_mask"], torch_module=torch_module)
            pooled = pooled.to(device=next(value_head.parameters()).device, dtype=next(value_head.parameters()).dtype)
            out = value_head(pooled)
            step_scores = out["scores"].detach().cpu().tolist()
            scores.extend(float(s) for s in step_scores)

    return float(sum(scores) / len(scores)) if scores else 0.5


# ── generation ───────────────────────────────────────────────────────────────


def _build_math_prompt(question: str) -> str:
    """Build a math problem prompt for Qwen2.5-Math-7B-Instruct."""
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}."
        " Each reasoning step should be separated by a blank line.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _generate_solutions(
    *,
    questions: list[str],
    generator_model: Any,
    tokenizer: Any,
    device: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    torch_module: Any,
) -> list[list[str]]:
    """Generate K solutions per question. Returns list[list[str]] (outer=question, inner=solutions)."""
    all_solutions: list[list[str]] = [[] for _ in range(len(questions))]

    prompts = [_build_math_prompt(q) for q in questions]
    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    prompt_len = tokens["input_ids"].shape[1]
    tokens = {k_: v.to(device) for k_, v in tokens.items()}

    for sample_idx in range(k):
        with torch_module.no_grad():
            generated = generator_model.generate(
                **tokens,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=max(float(temperature), 1e-4),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for q_idx in range(len(questions)):
            new_tokens = generated[q_idx, prompt_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_solutions[q_idx].append(text)

    return all_solutions


# ── main evaluation ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase F: PRM-guided Best-of-N evaluation on GSM8K.")
    p.add_argument("--value-run-dir", type=Path, required=True, help="Path to Phase E LoRA run dir.")
    p.add_argument("--backbone-path", default="assets/models/Qwen2.5-Math-PRM-7B")
    p.add_argument("--generator-path", default="assets/models/Qwen2.5-Math-7B-Instruct")
    p.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math"], help="Dataset: gsm8k (default) or math (PRM800K MATH test split).")
    p.add_argument("--gsm8k-path", default="assets/external_datasets/openai_gsm8k/main/test-00000-of-00001.parquet")
    p.add_argument("--math-path", default="assets/external_datasets/openai_prm800k/prm800k/math_splits/test.jsonl")
    p.add_argument("--num-problems", type=int, default=200, help="Number of problems to evaluate.")
    p.add_argument("--k-samples", type=int, default=4, help="Number of sampled solutions per problem (K).")
    p.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens per generated solution.")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    p.add_argument("--max-scoring-length", type=int, default=1024, help="Max token length for value head scoring.")
    p.add_argument("--generator-batch-size", type=int, default=8, help="Problems per generation batch.")
    p.add_argument("--output-root", type=Path, default=Path("assets/artifacts/phase_f_bon"))
    p.add_argument("--run-name", default="pbr_bon_gsm8k")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    import random
    random.seed(args.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if bool(args.require_cuda) and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but not available")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_root) / f"{args.run_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase F: PRM-Guided Best-of-N Evaluation")
    print("=" * 80)
    print(f"value_run_dir    : {args.value_run_dir}")
    print(f"generator_path   : {args.generator_path}")
    print(f"backbone_path    : {args.backbone_path}")
    print(f"num_problems     : {args.num_problems}")
    print(f"k_samples        : {args.k_samples}")
    print(f"temperature      : {args.temperature}")
    print(f"output_dir       : {out_dir}")
    print("=" * 80)

    # Load dataset
    import random as _random
    _random.seed(int(args.seed))

    if args.dataset == "math":
        import json as _json
        items = []
        with open(args.math_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(_json.loads(line))
        _random.shuffle(items)
        items = items[: int(args.num_problems)]
        questions = [d["problem"] for d in items]
        gt_answers = [_extract_math_gt_answer(d["answer"]) for d in items]
        print(f"Loaded {len(questions)} MATH problems from PRM800K test split")
    else:
        import pandas as pd
        df = pd.read_parquet(args.gsm8k_path)
        df = df.sample(frac=1, random_state=args.seed).head(int(args.num_problems)).reset_index(drop=True)
        questions = df["question"].tolist()
        gt_answers_raw = df["answer"].tolist()
        gt_answers = [_extract_gsm8k_gt_answer(str(a)) for a in gt_answers_raw]
        print(f"Loaded {len(questions)} GSM8K problems")

    # Load value head (PRM)
    print("Loading value head (PRM)...")
    t0 = time.perf_counter()
    backbone, prm_tokenizer, value_head = _load_value_head(
        value_run_dir=Path(args.value_run_dir),
        backbone_path=str(args.backbone_path),
        device=device,
        dtype_str=str(args.dtype),
        torch_module=torch,
    )
    print(f"Value head loaded in {time.perf_counter()-t0:.1f}s")

    # Load generator
    print("Loading generator model...")
    t0 = time.perf_counter()
    from ours.phase_e.runtime import resolve_dtype
    gen_dtype = resolve_dtype(str(args.dtype), torch)
    gen_tokenizer = AutoTokenizer.from_pretrained(str(args.generator_path), trust_remote_code=True)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_tokenizer.padding_side = "left"  # required for generation with padding
    gen_model = AutoModelForCausalLM.from_pretrained(
        str(args.generator_path),
        torch_dtype=gen_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    gen_model.eval()
    gen_model.to(device)
    print(f"Generator loaded in {time.perf_counter()-t0:.1f}s")

    # Evaluation loop
    results_per_problem: list[dict[str, Any]] = []
    total_q = len(questions)
    batch_size = int(args.generator_batch_size)

    greedy_correct = 0
    random_n_correct = 0
    oracle_correct = 0
    prm_correct = 0

    print(f"\nGenerating {args.k_samples} solutions × {total_q} problems (batch={batch_size})...")
    for b_start in range(0, total_q, batch_size):
        b_end = min(b_start + batch_size, total_q)
        batch_questions = questions[b_start:b_end]
        batch_gt = gt_answers[b_start:b_end]

        t_gen = time.perf_counter()
        # Generate greedy first (sample 0)
        greedy_solutions = _generate_solutions(
            questions=batch_questions,
            generator_model=gen_model,
            tokenizer=gen_tokenizer,
            device=device,
            k=1,
            max_new_tokens=int(args.max_new_tokens),
            temperature=0.0,  # greedy
            torch_module=torch,
        )
        # Generate K temperature-sampled solutions
        sampled_solutions = _generate_solutions(
            questions=batch_questions,
            generator_model=gen_model,
            tokenizer=gen_tokenizer,
            device=device,
            k=int(args.k_samples),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            torch_module=torch,
        )
        gen_elapsed = time.perf_counter() - t_gen

        # Score sampled solutions with PRM
        t_score = time.perf_counter()
        for q_idx in range(len(batch_questions)):
            q_gt = batch_gt[q_idx]
            greedy_sol = greedy_solutions[q_idx][0]
            sampled_sols = sampled_solutions[q_idx]  # K solutions

            # Score each sampled solution
            prm_scores = [
                _score_solution(
                    solution_text=sol,
                    backbone=backbone,
                    tokenizer=prm_tokenizer,
                    value_head=value_head,
                    max_length=int(args.max_scoring_length),
                    device=device,
                    torch_module=torch,
                )
                for sol in sampled_sols
            ]

            # Correctness of each solution
            _extract_fn = _extract_boxed_answer if args.dataset == "math" else _extract_gsm8k_answer
            greedy_pred = _extract_fn(greedy_sol)
            sampled_preds = [_extract_fn(s) for s in sampled_sols]
            sampled_correct = [_answers_match(p, q_gt) for p in sampled_preds]

            # Select best by PRM
            best_prm_idx = max(range(len(prm_scores)), key=lambda i: prm_scores[i])
            prm_pred = sampled_preds[best_prm_idx]

            # Oracle: pick best correct solution if any
            oracle_pred = sampled_preds[0]
            for idx, is_c in enumerate(sampled_correct):
                if is_c:
                    oracle_pred = sampled_preds[idx]
                    break

            # Random: pick random solution
            random_pred = sampled_preds[random.randint(0, len(sampled_preds) - 1)]

            g_ok = _answers_match(greedy_pred, q_gt)
            p_ok = _answers_match(prm_pred, q_gt)
            o_ok = any(sampled_correct)
            r_ok = _answers_match(random_pred, q_gt)

            greedy_correct += int(g_ok)
            prm_correct += int(p_ok)
            oracle_correct += int(o_ok)
            random_n_correct += int(r_ok)

            results_per_problem.append({
                "question_idx": b_start + q_idx,
                "gt_answer": q_gt,
                "greedy_pred": greedy_pred,
                "greedy_correct": g_ok,
                "prm_best_idx": best_prm_idx,
                "prm_pred": prm_pred,
                "prm_correct": p_ok,
                "oracle_correct": o_ok,
                "random_correct": r_ok,
                "prm_scores": prm_scores,
                "k_correct": sum(sampled_correct),
            })

        score_elapsed = time.perf_counter() - t_score
        n_done = b_end
        pct = 100.0 * n_done / total_q
        print(
            f"  [{n_done}/{total_q} ({pct:.0f}%)] "
            f"gen={gen_elapsed:.1f}s score={score_elapsed:.1f}s | "
            f"greedy={greedy_correct/n_done:.3f} "
            f"prm@{args.k_samples}={prm_correct/n_done:.3f} "
            f"oracle@{args.k_samples}={oracle_correct/n_done:.3f}",
            flush=True,
        )

    n = len(results_per_problem)
    summary = {
        "num_problems": n,
        "k_samples": int(args.k_samples),
        "temperature": float(args.temperature),
        "greedy_accuracy": float(greedy_correct / max(1, n)),
        "random_n_accuracy": float(random_n_correct / max(1, n)),
        "prm_reranked_accuracy": float(prm_correct / max(1, n)),
        "oracle_accuracy": float(oracle_correct / max(1, n)),
        "prm_vs_greedy_delta": float((prm_correct - greedy_correct) / max(1, n)),
        "prm_vs_random_delta": float((prm_correct - random_n_correct) / max(1, n)),
        "value_run_dir": str(args.value_run_dir),
        "generator_path": str(args.generator_path),
    }

    print("\n" + "=" * 80)
    print("Phase F Best-of-N Results")
    print("=" * 80)
    print(f"greedy_accuracy       : {summary['greedy_accuracy']:.4f}")
    print(f"random@{args.k_samples}_accuracy   : {summary['random_n_accuracy']:.4f}")
    print(f"prm_reranked@{args.k_samples}       : {summary['prm_reranked_accuracy']:.4f}")
    print(f"oracle@{args.k_samples}_accuracy    : {summary['oracle_accuracy']:.4f}")
    print(f"prm_vs_greedy_delta   : {summary['prm_vs_greedy_delta']:+.4f}")
    print(f"prm_vs_random_delta   : {summary['prm_vs_random_delta']:+.4f}")
    print("=" * 80)

    # Gate check
    threshold = 0.02
    gate_pass = summary["prm_vs_random_delta"] >= threshold
    gate_str = "PASS" if gate_pass else "FAIL"
    print(f"\nPhase F3 gate (prm_vs_random >= +{threshold:.0%}): {gate_str}")
    summary["phase_f3_gate"] = gate_str
    summary["phase_f3_threshold"] = threshold

    # Save results
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    results_path = out_dir / "per_problem_results.jsonl"
    with results_path.open("w") as f:
        for r in results_per_problem:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
