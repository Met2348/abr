# Phase A PPT Reference (Supervisor-Oriented)

Last updated: 2026-02-27

## 1. Purpose

This file is a reusable reference for project reporting slides.
It consolidates:
1. what Phase A built,
2. key numeric outcomes,
3. term explanations for non-LLM-specialist audiences,
4. interpretation caveats (including whether current results reflect model potential).

## 2. Slide-Ready Summary (3+1 Pages)

## 2.1 Page 1: Phase A Goal and Deliverables

Goal:
1. Build a reliable baseline pipeline before training-phase research claims.
2. Separate true model quality from evaluation/format artifacts.

Deliverables:
1. Canonical data pipeline and deterministic artifact preparation.
2. Prompt-template versioning and reproducible run manifests.
3. One-script inference+evaluation and one-click benchmark suites (`A1`~`A8`).
4. Reliability hardening on evaluator, extraction, split handling, and batching path.

## 2.2 Page 2: Main Findings

1. Parse-error inflation was largely protocol/extraction/format related.
2. StrategyQA binary-choice mode (`yes/no` scoring) removed parse noise (`parse_error=0.0000`).
3. After evaluator fixes, GSM8K CoT quality was previously underestimated and rose substantially on re-evaluation.
4. Batching is now trustworthy and significantly faster after left-padding fix for decoder-only models.

## 2.3 Page 3: Status and Next Step

1. Phase A baseline infrastructure is complete and reproducible.
2. Known confounds are documented and partly controlled.
3. Phase B is ready:
   - B0 scope freeze complete,
   - B1 training skeleton implemented,
   - next step is first stable smoke training run.

## 2.4 Page 4 (Optional): Numeric Outcomes Table

| Track | Setting | n | Accuracy | Parse Error | Notes |
|---|---|---:|---:|---:|---|
| StrategyQA direct (freeform) | `t16` | 193 | 0.5492 | 0.1762 | baseline direct |
| StrategyQA direct (freeform) | `t64` | 193 | 0.5544 | 0.1658 | best direct sweep point |
| StrategyQA CoT (freeform) | `t128` | 193 | 0.1036 | 0.8497 | severe format failure |
| StrategyQA CoT (freeform) | `t320` | 193 | 0.6684 | 0.1140 | high-quality freeform point |
| StrategyQA CoT (freeform) | `t384` | 193 | 0.6839 | 0.0518 | best freeform CoT accuracy |
| StrategyQA strict template | `strict_t16` | 193 | 0.5285 | 0.0363 | compliance up, quality tradeoff |
| StrategyQA binary-choice | `direct_binchoice` | 193 | 0.6788 | 0.0000 | decision-quality baseline |
| StrategyQA binary-choice | `cot_binchoice` | 193 | 0.5803 | 0.0000 | CoT prompt under direct here |
| StrategyQA batching | `batch1` | 193 | 0.5492 | 0.1762 | 1.180 sample/s, 163.53s |
| StrategyQA batching | `batch4_fix` | 193 | 0.5596 | 0.1658 | 4.189 sample/s, 46.08s |
| GSM8K re-eval | direct `t16` | 172 | 0.3721 | 0.0000 | current evaluator |
| GSM8K re-eval | CoT `t256` | 172 | 0.7035 | 0.0000 | current evaluator |

Determinism confirmations:
1. `strict_t16` repeat: `delta_accuracy=+0.0000`, `changed_samples=0`.
2. `direct_binchoice` repeat: `delta_accuracy=+0.0000`, `changed_samples=0`.
3. `gsm8k_direct_t16_gate` repeat: identical metrics and no changed samples.

## 3. Glossary for Supervisor Audience

1. `n`: number of evaluation samples.
2. `Accuracy`: fraction of correct final answers.
3. `Parse Error`: fraction of outputs that cannot be interpreted into valid answers.
4. `freeform`: model outputs natural language; realistic but format-sensitive.
5. `binary_choice`: model chooses between fixed options (`yes`/`no`); isolates decision quality.
6. `CoT`: chain-of-thought prompting (reasoning before final answer).
7. `t128/t256/...`: `max_new_tokens` decode budget.
8. `batch_size`: number of prompts decoded together (throughput lever).
9. `changed_samples`: count of per-sample prediction changes in rerun diff; near zero indicates reproducibility.

## 4. Interpretation Caveat: Are We Seeing True Model Potential?

Short answer: partially, but not fully.

Reasons current results may still be below model ceiling:
1. Prompt protocol is custom and may not fully match model-native chat template behavior.
2. Split/protocol differs from some public leaderboard setups.
3. Current core baseline is deterministic greedy decode (`--no-do-sample`), while some published highs use stronger inference recipes (for example self-consistency).
4. Token budget and truncation effects can still suppress CoT quality.

Practical interpretation:
1. Current numbers are valid for **our frozen protocol**.
2. They should be treated as **project baseline**, not guaranteed model ceiling.

## 5. Recommended “Ceiling Calibration” Mini-Track (Before Big Claims)

1. Add model-native chat-template variant and compare.
2. Keep one official-split evaluation track where feasible.
3. Add a stronger GSM8K inference variant (for example multi-sample majority vote) as a reference.
4. Maintain two baselines in reports:
   - project-frozen baseline,
   - calibrated-ceiling baseline.

## 6. Source Pointers

1. Experiment history and diagnostics: `result_records.md`
2. Phase A closeout report: `phase_A_report.md`
3. Reliability hardening details: `foundation_reliability_audit.md`
4. Phase B lifecycle and goals: `phase_B_plan.md`

## 7. New Prompt-Template Packs (for Next Evaluation Round)

StrategyQA (3 styles):
1. `qa_strategyqa_minimal_binary`: one-token yes/no contract.
2. `qa_strategyqa_cot_compact`: short reasoning + `Final answer: yes/no`.
3. `qa_strategyqa_evidence_verdict`: `Evidence:` line + `Verdict: yes/no`.

GSM8K (3 styles):
1. `qa_gsm8k_direct_final_only`: one-line `Final answer: <number>`.
2. `qa_gsm8k_cot_compact_final`: short CoT + final numeric line.
3. `qa_gsm8k_equation_then_final`: equation line + final numeric line.

Benchmark param groups:
1. `A7`: StrategyQA prompt-style sweep.
2. `A8`: GSM8K prompt-style sweep.

Shell commands:

```bash
ACTIVE_PARAM_GROUP=A7 \
CUDA_VISIBLE_DEVICES=0 \
RUN_PREFIX=strategyqa_style_sweep \
bash scripts/run_phase_a_benchmark_suite.sh

ACTIVE_PARAM_GROUP=A8 \
CUDA_VISIBLE_DEVICES=0 \
RUN_PREFIX=gsm8k_style_sweep \
bash scripts/run_phase_a_benchmark_suite.sh
```
