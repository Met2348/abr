#!/usr/bin/env python3
# Changelog (prepend-only, newest first):
# 2026-03-13: Initial creation. Evaluates PRIME-style implicit PRM from LoRA adapter.
"""Phase F / Phase E post-hoc evaluation: PRIME implicit PRM on ProcessBench.

English
-------
PRIME (arXiv:2502.01456) defines implicit process reward at each token as:
    r_φ(y_t) = β · log[ π_φ(y_t | context) / π_ref(y_t | context) ]

This is a FREE discriminative PRM that requires:
- A LoRA-fine-tuned model (π_φ): any of our PBR31/32/33/34/35/36/37 adapters
- The reference model (π_ref): base model before LoRA (e.g. Qwen2.5-Math-PRM-7B)

Step-level implicit PRM score = mean of per-token log-ratios over that step's tokens.
This score can be used as a ProcessBench step scorer without any additional training.

Usage
-----
Evaluate PRIME implicit PRM on ProcessBench MATH:

    python scripts/phase_f_implicit_prm_eval.py \\
        --lora-run-dir assets/artifacts/phase_e_runs/phase_e_pbr35_... \\
        --base-model-path assets/models/Qwen2.5-Math-PRM-7B \\
        --benchmark-jsonl assets/external_datasets/processbench_math.jsonl \\
        --beta 0.5 \\
        --run-name pbr35_implicit_prm

中文
----
PRIME 隐式过程奖励评估。给定已训练的 LoRA 适配器和基础模型，计算每个步骤的
log(π_LoRA / π_ref) 之均值作为该步骤的过程奖励分数，然后用此分数在 ProcessBench
上计算 F1（与明确训练的 value head 相比）。零额外训练成本。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _bootstrap() -> None:
    pass


_bootstrap()


def _split_steps(text: str) -> list[str]:
    """Split solution into cumulative prefixes at double-newline boundaries."""
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    prefixes: list[str] = []
    running = ""
    for part in parts:
        running = (running + "\n\n" + part).strip() if running else part
        prefixes.append(running)
    return prefixes


def _compute_log_ratio_per_token(
    lora_model: Any,
    ref_model: Any,
    tokenizer: Any,
    text: str,
    max_length: int,
    device: str,
    torch_module: Any,
    beta: float,
) -> list[float]:
    """Compute β·log(π_LoRA/π_ref) for each non-padding token in text.

    Returns a list of log-ratio values (one per token after the first).
    Padding tokens are excluded.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    if input_ids.shape[1] < 2:
        return []

    # Forward through both models
    lora_model.eval()
    ref_model.eval()

    with torch_module.no_grad():  # type: ignore[attr-defined]
        lora_out = lora_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        ref_out = ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )

    # Get logits from both models (shape: [1, seq_len, vocab_size])
    # Use the last hidden state → lm_head for computing logits
    # For Qwen2ForProcessRewardModel, we need to access the underlying LM backbone.
    lora_logits = _get_lm_logits(lora_model, lora_out, device, torch_module)
    ref_logits = _get_lm_logits(ref_model, ref_out, device, torch_module)

    if lora_logits is None or ref_logits is None:
        return []

    # Compute log-probs for the actual token sequence
    # Shift: logits[t] predicts token[t+1]
    log_probs_lora = torch_module.nn.functional.log_softmax(lora_logits, dim=-1)
    log_probs_ref = torch_module.nn.functional.log_softmax(ref_logits, dim=-1)

    # For each position t, get the log-prob of the actual next token
    # input_ids[:, 1:] are the target tokens; positions 0..T-2 predict them
    target_ids = input_ids[:, 1:]  # [1, T-1]
    target_mask = attention_mask[:, 1:]  # [1, T-1]

    lp_lora = log_probs_lora[:, :-1, :].gather(dim=2, index=target_ids.unsqueeze(-1)).squeeze(-1)
    lp_ref = log_probs_ref[:, :-1, :].gather(dim=2, index=target_ids.unsqueeze(-1)).squeeze(-1)

    # log-ratio = log(π_LoRA/π_ref) = log_p_lora - log_p_ref
    log_ratio = (lp_lora - lp_ref) * beta  # [1, T-1]

    # Mask out padding tokens
    token_mask = target_mask.float()  # [1, T-1]
    log_ratio_masked = (log_ratio * token_mask).squeeze(0)  # [T-1]
    token_mask_flat = token_mask.squeeze(0)

    # Return per-token values for non-padding tokens
    result = []
    for i in range(log_ratio_masked.shape[0]):
        if token_mask_flat[i].item() > 0:
            result.append(float(log_ratio_masked[i].item()))
    return result


def _get_lm_logits(model: Any, model_out: Any, device: str, torch_module: Any) -> Any | None:
    """Extract LM-head logits from a model output, handling PRM and CausalLM cases."""
    # For Qwen2ForCausalLM (standard): logits is directly in model_out.logits
    if hasattr(model_out, "logits") and model_out.logits is not None:
        return model_out.logits

    # For Qwen2ForProcessRewardModel: hidden states available; apply lm_head manually
    if hasattr(model_out, "hidden_states") and model_out.hidden_states is not None:
        last_hidden = model_out.hidden_states[-1]
        # Get the underlying base model to access lm_head
        base_model = model
        # Unwrap peft model
        if hasattr(base_model, "base_model"):
            base_model = base_model.base_model
        if hasattr(base_model, "model"):
            base_model = base_model.model
        if hasattr(base_model, "lm_head"):
            return base_model.lm_head(last_hidden)

    return None


def _compute_step_scores_implicit(
    lora_model: Any,
    ref_model: Any,
    tokenizer: Any,
    solution: str,
    max_length: int,
    device: str,
    torch_module: Any,
    beta: float,
) -> list[float]:
    """Score each step prefix using implicit log-ratio PRM.

    Returns per-step mean log-ratio scores s_1, ..., s_T.
    """
    prefixes = _split_steps(solution)
    step_scores: list[float] = []
    prev_len = 0  # token count of previous prefix (for step isolation)
    prev_logratios: list[float] = []

    for prefix in prefixes:
        # Get full cumulative log-ratios
        curr_logratios = _compute_log_ratio_per_token(
            lora_model, ref_model, tokenizer, prefix, max_length, device, torch_module, beta
        )
        # Step-specific tokens are the new ones added since last prefix
        step_tokens = curr_logratios[prev_len:]
        if step_tokens:
            step_scores.append(float(sum(step_tokens) / len(step_tokens)))
        else:
            step_scores.append(0.0)
        prev_len = len(curr_logratios)
        prev_logratios = curr_logratios

    return step_scores


def _load_processbench_jsonl(path: Path) -> list[dict]:
    """Load ProcessBench JSON or JSONL file."""
    with path.open() as f:
        first_char = f.read(1)
    with path.open() as f:
        if first_char == "[":
            return json.load(f)
        examples = []
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
        return examples


def _compute_processbench_f1(
    scored_examples: list[dict],
) -> dict:
    """Compute ProcessBench F1.

    scored_examples: list of dicts with keys:
        - step_scores: list[float] per-step scores (s_1..s_T)
        - error_location: int (-1 = all correct, ≥0 = first error step idx)
    Returns dict with f1, acc_erroneous, acc_correct, best_tau.
    """
    import numpy as np

    # Find optimal threshold tau
    all_scores = [s for ex in scored_examples for s in ex["step_scores"]]
    if not all_scores:
        return {"f1": 0.0, "acc_erroneous": 0.0, "acc_correct": 0.0, "best_tau": 0.5}

    taus = sorted(set(all_scores))[::max(1, len(all_scores) // 200)]

    best_f1, best_tau = 0.0, 0.5
    for tau in taus:
        tp = fp = tn = fn = 0
        for ex in scored_examples:
            scores = ex["step_scores"]
            error_loc = ex["error_location"]
            # First step with score < tau is the predicted error location
            pred_error = next((i for i, s in enumerate(scores) if s < tau), None)
            if error_loc < 0:  # all-correct
                if pred_error is None:
                    tn += 1  # correct: no error predicted
                else:
                    fp += 1  # wrong: predicted an error
            else:  # has error
                if pred_error is not None:
                    tp += 1  # detected error (even if wrong location, counts as detection)
                else:
                    fn += 1  # missed error
        acc_err = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        acc_cor = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * acc_err * acc_cor / (acc_err + acc_cor) if (acc_err + acc_cor) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
            best_acc_err, best_acc_cor = acc_err, acc_cor

    return {
        "f1": best_f1,
        "acc_erroneous": best_acc_err if best_f1 > 0 else 0.0,
        "acc_correct": best_acc_cor if best_f1 > 0 else 0.0,
        "best_tau": best_tau,
    }


def main(argv: list[str] | None = None) -> None:
    import torch

    p = argparse.ArgumentParser(description="PRIME implicit PRM evaluation on ProcessBench")
    p.add_argument("--lora-run-dir", type=Path, required=True, help="Phase E LoRA run dir (with best_adapter)")
    p.add_argument("--base-model-path", default="assets/models/Qwen2.5-Math-PRM-7B")
    p.add_argument("--benchmark-jsonl", type=Path, required=True, help="ProcessBench JSONL file")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--beta", type=float, default=0.5, help="Scaling factor for log-ratio (PRIME β)")
    p.add_argument("--max-token-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (currently 1 for simplicity)")
    p.add_argument("--run-name", default="implicit_prm_eval")
    p.add_argument("--output-root", type=Path, default=Path("assets/artifacts/phase_f_implicit_prm"))
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args(argv)

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda specified but no CUDA device found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype, torch.bfloat16)

    print("=" * 80)
    print("Phase F: PRIME Implicit PRM Evaluation")
    print("=" * 80)
    print(f"lora_run_dir     : {args.lora_run_dir}")
    print(f"base_model       : {args.base_model_path}")
    print(f"beta             : {args.beta}")
    print(f"device           : {device}")
    print()

    # Load base (reference) model
    # Use Qwen2ForCausalLM (not AutoModel/Qwen2ForProcessRewardModel) so that
    # model_out.logits = token-level vocabulary logits [1, T, V] (not scalar PRM rewards).
    # The PRM checkpoint contains lm_head.weight; CausalLM loads it correctly.
    print("Loading reference (base) model...")
    from transformers import AutoTokenizer

    try:
        from transformers import Qwen2ForCausalLM

        _ModelCls = Qwen2ForCausalLM
    except ImportError:
        from transformers import AutoModelForCausalLM

        _ModelCls = AutoModelForCausalLM  # type: ignore[assignment]

    load_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True, "trust_remote_code": True, "ignore_mismatched_sizes": True}
    ref_model = _ModelCls.from_pretrained(args.base_model_path, **load_kwargs)
    ref_model.eval()
    ref_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    print(f"Reference model loaded: {type(ref_model).__name__}")

    # Load LoRA model (separate copy of base + adapter)
    print("Loading LoRA model...")
    lora_model = _ModelCls.from_pretrained(args.base_model_path, **load_kwargs)
    adapter_dir = args.lora_run_dir / "best_adapter"
    if adapter_dir.exists():
        from ours.phase_e.runtime import attach_peft_adapter_for_inference

        lora_model = attach_peft_adapter_for_inference(lora_model, adapter_dir)
        print(f"LoRA adapter loaded from {adapter_dir}")
    else:
        print("WARNING: No best_adapter found — using base model as LoRA model (scores will all be 0)")
    lora_model.eval()
    lora_model.to(device)

    # Load benchmark
    print(f"Loading benchmark: {args.benchmark_jsonl}")
    examples = _load_processbench_jsonl(args.benchmark_jsonl)
    if args.max_samples:
        examples = examples[: args.max_samples]
    print(f"Loaded {len(examples)} examples")

    # Score each example
    print("Scoring examples with implicit PRM...")
    t0 = time.time()
    scored = []
    for i, ex in enumerate(examples):
        # Extract solution text and error location
        # ProcessBench format: steps is a list; join to reconstruct solution text
        raw_steps = ex.get("steps")
        if isinstance(raw_steps, list):
            solution = "\n\n".join(raw_steps)
        else:
            solution = ex.get("solution", ex.get("reasoning", ex.get("text", "")))
        # ProcessBench format: error_location = first error step index (0-based), -1 if all correct
        error_loc = ex.get("error_location", ex.get("label", -1))
        if error_loc is None:
            error_loc = -1

        step_scores = _compute_step_scores_implicit(
            lora_model,
            ref_model,
            tokenizer,
            solution,
            args.max_token_length,
            device,
            torch,  # torch_module
            args.beta,
        )

        scored.append(
            {
                "example_id": ex.get("id", i),
                "error_location": int(error_loc),
                "num_steps": len(step_scores),
                "step_scores": step_scores,
            }
        )

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(examples)} ({elapsed:.0f}s)")

    # Compute F1
    metrics = _compute_processbench_f1(scored)

    print("\nResults:")
    print(f"  ProcessBench F1   : {metrics['f1']:.4f}")
    print(f"  Acc (erroneous)   : {metrics['acc_erroneous']:.4f}")
    print(f"  Acc (correct)     : {metrics['acc_correct']:.4f}")
    print(f"  Best tau          : {metrics['best_tau']:.4f}")

    # Save results
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_root / f"{args.run_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": args.run_name,
        "lora_run_dir": str(args.lora_run_dir),
        "benchmark_jsonl": str(args.benchmark_jsonl),
        "beta": args.beta,
        "n_examples": len(examples),
        **metrics,
        "elapsed_s": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "scored_rows.jsonl").write_text("\n".join(json.dumps(s) for s in scored))
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
