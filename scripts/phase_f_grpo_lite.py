#!/usr/bin/env python3
# Changelog (prepend-only, newest first):
# 2026-03-12: Initial creation. Phase F GRPO-lite: RL training with process reward.
"""Phase F GRPO-lite: RL fine-tuning with outcome + process rewards.

English
-------
Trains a policy LM (Qwen2.5-Math-7B-Instruct) with GRPO using:
  r_total = r_outcome + lambda_process * r_process

where:
  r_outcome = +1 if final answer correct, -1 otherwise (verifiable reward)
  r_process = mean step-level score from PBR32 value head (process reward)

Uses TRL's GRPOTrainer for the RL loop. This is the Phase F4 experiment:
validating that our process reward improves RL training over outcome-only GRPO.

Comparison:
  - GRPO outcome-only (lambda_process=0): standard RLVR
  - GRPO + PRM (lambda_process=0.3): our method
  - Evaluate on GSM8K test (separate from training split)

中文
----
使用 TRL GRPOTrainer + outcome reward + PRM process reward 训练策略模型。
验证 process reward 是否能在相同训练步数内带来更高的 GSM8K 准确率。
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


# ── answer extraction ─────────────────────────────────────────────────────────


def _extract_final_answer(text: str) -> str | None:
    """Extract final answer from generated solution (#### format or \\boxed{} format)."""
    # GSM8K style: #### <number>
    m = re.search(r"####\s*([^\n]+)", text)
    if m:
        return m.group(1).strip().replace(",", "")
    # LaTeX boxed: \boxed{...}
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    # Last number in text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _extract_gsm8k_gt(answer_text: str) -> str | None:
    m = re.search(r"####\s*([^\n]+)", answer_text)
    if m:
        return m.group(1).strip().replace(",", "")
    return None


def _answers_match(pred: str | None, gt: str | None) -> bool:
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < 1e-3
    except ValueError:
        return pred.strip().lower() == gt.strip().lower()


def _split_steps(text: str) -> list[str]:
    """Return cumulative step prefixes for PRM scoring."""
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not parts:
        return [text.strip()]
    prefixes: list[str] = []
    cumulative = ""
    for part in parts:
        cumulative = (cumulative + "\n\n" + part).lstrip("\n")
        prefixes.append(cumulative)
    return prefixes


# ── PRM scoring helper ────────────────────────────────────────────────────────


class PRMScorer:
    """Step-level process reward model scorer.

    Scores a batch of solutions by averaging value head scores over step prefixes.
    """

    def __init__(
        self,
        *,
        value_run_dir: Path,
        backbone_path: str,
        device: str,
        dtype: Any,
        torch_module: Any,
    ) -> None:
        self.device = device
        self.torch = torch_module

        from transformers import AutoTokenizer

        from ours.phase_b.value_head import (
            ensure_tokenizer_has_pad_token,
            load_value_head_checkpoint,
            pool_last_token,
        )
        from ours.phase_e.runtime import resolve_backbone_loader_family

        self._pool_last_token = pool_last_token

        tokenizer = AutoTokenizer.from_pretrained(backbone_path, trust_remote_code=True)
        ensure_tokenizer_has_pad_token(tokenizer)
        self.tokenizer = tokenizer

        _family = resolve_backbone_loader_family(model_path=backbone_path, trust_remote_code=True)
        _load_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True, "trust_remote_code": True}
        if _family == "process_reward_model":
            from transformers import AutoModel

            backbone = AutoModel.from_pretrained(backbone_path, **_load_kwargs)
        else:
            from transformers import AutoModelForCausalLM

            backbone = AutoModelForCausalLM.from_pretrained(backbone_path, **_load_kwargs)

        adapter_dir = value_run_dir / "best_adapter"
        if adapter_dir.exists():
            from ours.phase_e.runtime import attach_peft_adapter_for_inference

            backbone = attach_peft_adapter_for_inference(backbone, adapter_dir)
            print(f"prm_scorer       : loaded LoRA from {adapter_dir}")
        backbone.eval()
        backbone.to(device)
        self.backbone = backbone

        vh, _, _ = load_value_head_checkpoint(value_run_dir / "best_value_head.pt")
        vh.eval()
        vh.to(device)
        self.value_head = vh

    def score_steps(self, solution: str, max_length: int = 1024) -> list[float]:
        """Score a single solution, returning per-step scores s_1, ..., s_T."""
        prefixes = _split_steps(solution)
        if not prefixes:
            return [0.5]
        step_scores: list[float] = []
        batch_sz = 8
        for i in range(0, len(prefixes), batch_sz):
            batch = prefixes[i : i + batch_sz]
            with self.torch.no_grad():
                tok = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                tok = {k: v.to(self.device) for k, v in tok.items()}
                out = self.backbone(**tok, output_hidden_states=True, use_cache=False, return_dict=True)
                last_h = out.hidden_states[-1]
                pooled = self._pool_last_token(last_h, tok["attention_mask"], torch_module=self.torch)
                pooled = pooled.to(
                    device=next(self.value_head.parameters()).device,
                    dtype=next(self.value_head.parameters()).dtype,
                )
                vh_out = self.value_head(pooled)
                step_scores.extend(float(s) for s in vh_out["scores"].detach().cpu().tolist())
        return step_scores

    def score(self, solutions: list[str], max_length: int = 1024) -> list[float]:
        """Score a list of solutions. Returns mean step-level PRM score per solution."""
        return [
            float(sum(ss) / len(ss)) if (ss := self.score_steps(sol, max_length)) else 0.5
            for sol in solutions
        ]

    def score_clip_delta(self, solution: str, max_length: int = 1024, clip_range: float = 0.3) -> float:
        """Clip+Delta reward shaping (Zeng et al. 2024) to prevent reward hacking.

        Instead of using mean PRM score, compute the sum of clipped step-to-step deltas.
        This penalizes models that generate many tiny steps to accumulate small positive scores,
        because the delta of consecutive near-identical scores ≈ 0.

        r = (1/T) * Σ_t clip(s_t - s_{t-1}, -clip_range, +clip_range)
        where s_0 = 0.5 (neutral prior) and T = number of steps.
        """
        import math

        step_scores = self.score_steps(solution, max_length)
        if not step_scores:
            return 0.0
        prev = 0.5  # neutral prior for step 0
        total = 0.0
        for s in step_scores:
            delta = s - prev
            clipped = max(-clip_range, min(clip_range, delta))
            total += clipped
            prev = s
        return total / len(step_scores)


# ── reward function for GRPO ──────────────────────────────────────────────────


def _make_reward_fn(
    *,
    prm_scorer: PRMScorer | None,
    lambda_process: float,
    gt_answers_dict: dict[str, str],
    max_scoring_length: int,
    reward_shaping: str = "clip_delta",
) -> Any:
    """Create a reward function compatible with TRL GRPOTrainer.

    TRL GRPOTrainer calls reward_fn(prompts, completions) -> list[float].

    reward_shaping options:
    - "clip_delta": Zeng et al. (2024) recommended approach to prevent reward hacking.
        r_process = (1/T) Σ_t clip(s_t - s_{t-1}, -0.3, +0.3)
    - "mean_centered": original approach: mean(step_scores) - 0.5
    """

    def reward_fn(prompts: list[str], completions: list[str], completion_ids: Any = None, **kwargs: Any) -> list[float]:
        rewards: list[float] = []
        for prompt, completion in zip(prompts, completions):
            # Extract ground truth from prompt (we embed the GT in a special marker)
            gt_marker = re.search(r"\[GT:(.*?)\]", prompt)
            gt = gt_marker.group(1).strip() if gt_marker else None

            pred = _extract_final_answer(completion)
            outcome_r = 1.0 if _answers_match(pred, gt) else -1.0

            if prm_scorer is not None and float(lambda_process) > 0.0:
                if reward_shaping == "clip_delta":
                    # Clip+Delta: prevents reward hacking by penalizing step proliferation
                    process_r = prm_scorer.score_clip_delta(
                        completion, max_length=max_scoring_length, clip_range=0.3
                    )
                else:
                    # Legacy: mean score centered at 0.5
                    process_r = prm_scorer.score([completion], max_length=max_scoring_length)[0] - 0.5
                total_r = outcome_r + float(lambda_process) * float(process_r)
            else:
                total_r = outcome_r

            rewards.append(float(total_r))
        return rewards

    return reward_fn


# ── evaluate accuracy ─────────────────────────────────────────────────────────


def _evaluate_accuracy(
    *,
    model: Any,
    tokenizer: Any,
    questions: list[str],
    gt_answers: list[str | None],
    device: str,
    max_new_tokens: int,
    torch_module: Any,
) -> float:
    """Greedy decoding accuracy on a set of questions."""
    correct = 0
    bs = 8
    prompts = [
        (
            "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}."
            " Each reasoning step should be separated by a blank line.<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for q in questions
    ]
    for i in range(0, len(prompts), bs):
        batch_p = prompts[i : i + bs]
        batch_gt = gt_answers[i : i + bs]
        tok = tokenizer(batch_p, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tok = {k: v.to(device) for k, v in tok.items()}
        prompt_len = tok["input_ids"].shape[1]
        with torch_module.no_grad():
            gen = model.generate(**tok, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        for j in range(len(batch_p)):
            text = tokenizer.decode(gen[j, prompt_len:], skip_special_tokens=True)
            pred = _extract_final_answer(text)
            if _answers_match(pred, batch_gt[j]):
                correct += 1
    return float(correct / max(1, len(questions)))


# ── main ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase F GRPO-lite: RL training with process reward.")
    p.add_argument("--value-run-dir", type=Path, required=True)
    p.add_argument("--backbone-path", default="assets/models/Qwen2.5-Math-PRM-7B")
    p.add_argument("--policy-path", default="assets/models/Qwen2.5-Math-7B-Instruct")
    p.add_argument("--gsm8k-train-path", default="assets/external_datasets/openai_gsm8k/main/train-00000-of-00001.parquet")
    p.add_argument("--gsm8k-test-path", default="assets/external_datasets/openai_gsm8k/main/test-00000-of-00001.parquet")
    p.add_argument("--num-train-problems", type=int, default=500, help="Problems from GSM8K train split.")
    p.add_argument("--num-eval-problems", type=int, default=200, help="Problems from GSM8K test split for eval.")
    p.add_argument("--lambda-process", type=float, default=0.3, help="Process reward weight. 0.0=outcome-only GRPO.")
    p.add_argument("--k-samples", type=int, default=4, help="Number of GRPO rollouts per problem.")
    p.add_argument("--num-train-epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-scoring-length", type=int, default=1024)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--output-root", type=Path, default=Path("assets/artifacts/phase_f_grpo"))
    p.add_argument("--run-name", default="grpo_lite_gsm8k")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-interval-steps", type=int, default=50, help="Evaluate policy every N optimizer steps.")
    p.add_argument(
        "--reward-shaping",
        default="clip_delta",
        choices=["clip_delta", "mean_centered"],
        help=(
            "Process reward shaping. clip_delta (default): Clip+Delta per Zeng et al. 2024 prevents "
            "reward hacking by computing clipped step-to-step deltas. mean_centered: legacy mean(scores)-0.5."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if bool(args.require_cuda) and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but not available")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from ours.phase_e.runtime import resolve_dtype

    dtype = resolve_dtype(str(args.dtype), torch)

    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_root) / f"{args.run_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase F GRPO-lite: RL Training with Process Reward")
    print("=" * 80)
    print(f"value_run_dir    : {args.value_run_dir}")
    print(f"policy_path      : {args.policy_path}")
    print(f"lambda_process   : {args.lambda_process}")
    print(f"k_samples        : {args.k_samples}")
    print(f"num_train        : {args.num_train_problems}")
    print(f"num_eval         : {args.num_eval_problems}")
    print(f"output_dir       : {out_dir}")
    print("=" * 80)

    # Load datasets
    import pandas as pd

    df_train = pd.read_parquet(args.gsm8k_train_path).sample(
        frac=1, random_state=int(args.seed)
    ).head(int(args.num_train_problems)).reset_index(drop=True)
    df_eval = pd.read_parquet(args.gsm8k_test_path).sample(
        frac=1, random_state=int(args.seed) + 1
    ).head(int(args.num_eval_problems)).reset_index(drop=True)

    train_q = df_train["question"].tolist()
    train_gt = [_extract_gsm8k_gt(str(a)) for a in df_train["answer"].tolist()]
    eval_q = df_eval["question"].tolist()
    eval_gt = [_extract_gsm8k_gt(str(a)) for a in df_eval["answer"].tolist()]

    print(f"Loaded {len(train_q)} train + {len(eval_q)} eval problems")

    # Load PRM scorer
    if float(args.lambda_process) > 0.0:
        print("Loading PRM scorer...")
        t0 = time.perf_counter()
        prm_scorer = PRMScorer(
            value_run_dir=Path(args.value_run_dir),
            backbone_path=str(args.backbone_path),
            device=device,
            dtype=dtype,
            torch_module=torch,
        )
        print(f"PRM loaded in {time.perf_counter()-t0:.1f}s")
    else:
        prm_scorer = None
        print("lambda_process=0 -> outcome-only GRPO (no PRM)")

    # Load policy model
    print("Loading policy model...")
    t0 = time.perf_counter()
    policy_tokenizer = AutoTokenizer.from_pretrained(str(args.policy_path), trust_remote_code=True)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_tokenizer.padding_side = "left"
    policy_model = AutoModelForCausalLM.from_pretrained(
        str(args.policy_path),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    policy_model.to(device)
    print(f"Policy loaded in {time.perf_counter()-t0:.1f}s")

    # Evaluate pre-training accuracy
    print("\nPre-training evaluation...")
    pre_acc = _evaluate_accuracy(
        model=policy_model,
        tokenizer=policy_tokenizer,
        questions=eval_q,
        gt_answers=eval_gt,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
        torch_module=torch,
    )
    print(f"pre_training_accuracy : {pre_acc:.4f}")

    # Build reward function
    reward_fn = _make_reward_fn(
        prm_scorer=prm_scorer,
        lambda_process=float(args.lambda_process),
        gt_answers_dict={q: g for q, g in zip(train_q, train_gt)},
        max_scoring_length=int(args.max_scoring_length),
        reward_shaping=getattr(args, "reward_shaping", "clip_delta"),
    )

    # Build TRL GRPO dataset
    # TRL GRPOTrainer expects a dataset with "prompt" column
    # We embed GT in a [GT:...] marker in the prompt so reward_fn can extract it
    from datasets import Dataset

    def _make_prompt_with_gt(q: str, gt: str | None) -> str:
        gt_str = str(gt) if gt is not None else ""
        return (
            f"[GT:{gt_str}]"
            "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}."
            " Each reasoning step should be separated by a blank line.<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    train_dataset = Dataset.from_dict({
        "prompt": [_make_prompt_with_gt(q, g) for q, g in zip(train_q, train_gt)],
    })

    # Configure GRPO trainer
    from trl import GRPOConfig, GRPOTrainer

    grpo_config = GRPOConfig(
        output_dir=str(out_dir / "grpo_checkpoints"),
        num_train_epochs=int(args.num_train_epochs),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        learning_rate=float(args.learning_rate),
        num_generations=int(args.k_samples),
        max_completion_length=int(args.max_new_tokens),
        seed=int(args.seed),
        logging_steps=10,
        save_steps=int(args.eval_interval_steps),
        save_total_limit=1,
        report_to="none",
        bf16=(str(args.dtype) == "bfloat16"),
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=policy_model,
        processing_class=policy_tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
    )

    print("\nStarting GRPO training...")
    t_train = time.perf_counter()
    trainer.train()
    train_elapsed = time.perf_counter() - t_train
    print(f"GRPO training done in {train_elapsed:.0f}s ({train_elapsed/60:.1f}min)")

    # Post-training evaluation
    print("\nPost-training evaluation...")
    policy_model.eval()
    post_acc = _evaluate_accuracy(
        model=policy_model,
        tokenizer=policy_tokenizer,
        questions=eval_q,
        gt_answers=eval_gt,
        device=device,
        max_new_tokens=int(args.max_new_tokens),
        torch_module=torch,
    )
    print(f"post_training_accuracy : {post_acc:.4f}")
    print(f"accuracy_delta         : {post_acc - pre_acc:+.4f}")

    summary = {
        "num_train_problems": len(train_q),
        "num_eval_problems": len(eval_q),
        "k_samples": int(args.k_samples),
        "lambda_process": float(args.lambda_process),
        "reward_shaping": getattr(args, "reward_shaping", "clip_delta"),
        "pre_training_accuracy": float(pre_acc),
        "post_training_accuracy": float(post_acc),
        "accuracy_delta": float(post_acc - pre_acc),
        "train_elapsed_s": float(train_elapsed),
        "value_run_dir": str(args.value_run_dir),
        "policy_path": str(args.policy_path),
    }

    gate_pass = (post_acc - pre_acc) >= 0.01
    summary["phase_f4_gate"] = "PASS" if gate_pass else "FAIL"
    print(f"\nPhase F4 gate (delta >= +1%): {summary['phase_f4_gate']}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
