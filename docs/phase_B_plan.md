# Phase B Plan: From Phase A Baselines to First SFT/PEFT Run

Last updated: 2026-03-02 10:15:00 +0800

## 0. Official Transition Notice

As of `2026-02-28`, project execution focus is officially switched to **Phase B**.

Operational policy:
1. `scripts/phase_a_*` remain benchmark references and should not be used for new method development.
2. New development for SFT/PEFT/value-head/BCR-lite work must go to `src/ours/phase_b/*` and `scripts/phase_b_*`.
3. Phase A outputs are now used as fixed baselines and training/evaluation inputs for Phase B.

Immediate execution entrypoints:
1. `scripts/run_phase_b_training_suite.sh` for one-click group runs (`B1_SMOKE`, `B1_FIRST`).
2. `scripts/phase_b_train_sft.py --config-json ...` for direct training control.
3. `scripts/phase_b_compare_eval.py` for explicit before/after PEFT gain reporting.

## 1. Phase B Mission

Phase B is complete when we finish a reproducible first training run (SFT or PEFT) with:
1. a saved checkpoint,
2. deterministic run manifest and logs,
3. post-train evaluation report using frozen Phase A evaluator protocol.

Primary objective:
- move from inference-only benchmarking (Phase A) to a stable training pipeline baseline.

## 2. Lifecycle Overview

| Lifecycle | Name | Goal | Exit Gate |
|---|---|---|---|
| B0 | Scope Freeze | Lock first-run contract and protocol | Contract doc approved and committed |
| B1 | Train Pipeline Skeleton | Create train script/config/artifact layout | Tiny smoke run completes end-to-end |
| B2 | Data Contract Wiring | Ensure train/val inputs are deterministic and validated | Data manifest and validation checks pass |
| B3 | Smoke Stability | Validate loss/logging/checkpoint/resume behavior | No NaN/OOM crash on short smoke runs |
| B4 | Dev Tuning | Get first stable quality trend | At least one config outperforms untrained baseline on target metric |
| B5 | First Official Run | Execute frozen first milestone run | Full run artifacts + report generated |
| B6 | Handoff to Phase C/D | Prepare value-head and BCR-lite extension | Phase B report + frozen baseline config |

## 3. Lifecycle B0 (Implemented in this update)

## 3.1 Training Strategy Freeze

Chosen path:
1. PEFT-first (LoRA) as default for first milestone.
2. Keep full SFT as optional fallback if PEFT integration blocks progress.

Reason:
1. lower VRAM and faster iteration on current hardware usage pattern,
2. easier early debugging while preserving meaningful learning signals.

## 3.2 First Milestone Task Freeze

Primary task:
1. StrategyQA direct-answer training baseline (yes/no outputs).

Primary artifact set:
1. `assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/`
   - `train.jsonl`
   - `validation.jsonl`
   - `test.jsonl`

Secondary validation task (after first success):
1. GSM8K direct math baseline using current math prompt/evaluator protocol.

## 3.3 Model and Runtime Freeze

1. Base model: `assets/models/Qwen2.5-7B-Instruct`
2. Precision default: `bfloat16`
3. Reproducibility defaults:
   - fixed seed (`42` unless explicitly overridden),
   - deterministic run manifest logging,
   - evaluator version recorded in every eval output.

## 3.4 Evaluation Protocol Freeze

For StrategyQA first-run reporting:
1. Decision-quality track: binary-choice evaluation.
2. End-to-end track: freeform parse-aware evaluation.
3. Always report together:
   - `accuracy`
   - `parse_error_rate`
   - `n_parseable`
   - `acc_parseable`

For GSM8K follow-up reporting:
1. keep current relaxed numeric equivalence logic,
2. include math diagnostics:
   - cap hit rate,
   - final-answer-tag rate,
   - fallback extraction rate.

## 3.5 Required Outputs for First Official Phase B Run

Each official run directory must include:
1. run config snapshot,
2. training log,
3. checkpoint(s),
4. post-train predictions and scored outputs,
5. metrics JSON with evaluator version,
6. short markdown summary.

## 4. Phase B Deliverables (Holistic)

## 4.1 Code Deliverables

1. `scripts/phase_b_train_sft.py` (or unified `phase_b_train.py` with SFT/PEFT mode)
2. `scripts/phase_b_eval.py`
3. minimal trainer modules for:
   - optimizer/scheduler setup,
   - gradient accumulation,
   - checkpoint save/resume,
   - periodic validation hooks.

## 4.2 Config Deliverables

1. frozen baseline config set for first run:
   - data config,
   - model/PEFT config,
   - training config,
   - eval config.
2. small smoke config variant for quick debugging.

## 4.3 Documentation Deliverables

1. update `readme.md` with Phase B run instructions,
2. update `TODO_ours.md` lifecycle checkboxes,
3. append key outcomes to `result_records.md`,
4. create `phase_B_report.md` at B5 closeout.

## 5. Risks and Guardrails

## 5.1 High-Risk Items

1. evaluator/protocol drift causing invalid comparisons,
2. data contract mismatches between train and eval,
3. unstable training logs hiding NaN/divergence,
4. batching-related train/eval parity bugs.

## 5.2 Guardrails

1. freeze evaluator/template/decode versions in manifest,
2. keep Phase A scripts frozen as benchmark references,
3. run reproducibility pair for any new "official" config,
4. enforce batch correctness checks (padding/order/parity),
5. fail fast on duplicate sample IDs and schema mismatches.

## 6. Immediate Next Actions (Start B1)

1. Create training script skeleton with CLI and run-manifest writer.
2. Add config files for:
   - PEFT smoke run,
   - PEFT first official run candidate.
3. Implement tiny smoke run (`max_steps` small) on StrategyQA train subset.
4. Validate:
   - checkpoint save/load,
   - periodic eval execution,
   - final report generation.
5. Only after stable smoke runs, launch B4 tuning and B5 official run.

## 7. Current Status Note

1. B0 scope freeze: completed.
2. B1 implementation status:
   - training script skeleton implemented (`scripts/phase_b_train_sft.py`),
   - evaluation bridge implemented (`scripts/phase_b_eval.py`),
   - starter configs added under `configs/phase_b/`.
3. B1 smoke exit gate is satisfied: tiny PEFT smoke run completes end-to-end and writes checkpoints, metrics, and adapter artifacts.
4. New B2-style full-dataset gain suites are now available:
   - `B2_STRATEGYQA_FULL`
   - `B2_GSM8K_FULL`
5. GSM8K diagnostic branches are now available to isolate the failure mechanism:
   - `B2_GSM8K_DIAG_LR_5E5`
   - `B2_GSM8K_DIAG_LR_1E4`
   - `B2_GSM8K_DIAG_EPOCH_025`
   - `B2_GSM8K_DIAG_EPOCH_050`
   - `B2_GSM8K_DIAG_DIRECT_STYLE`
   - `B2_GSM8K_DIAG_EQUATION_STYLE`
   - `B2_GSM8K_DIAG_CHECKPOINT_SWEEP`
   - `B2_GSM8K_DIAG_SHORT_COT`
   - `B2_GSM8K_DIAG_ANSWER_WEIGHTED`
6. StrategyQA scaling branches are now available to test whether the current gain can be pushed higher:
   - `B2_STRATEGYQA_DIAG_EPOCH_200`
   - `B2_STRATEGYQA_DIAG_EPOCH_300`
   - `B2_STRATEGYQA_DIAG_LORA_R8`
   - `B2_STRATEGYQA_DIAG_LORA_R32`
7. These groups are designed to answer the milestone question in a reportable way:
   - baseline held-out accuracy before PEFT,
   - held-out accuracy after PEFT,
   - absolute delta and correct-count delta in one summary.
8. GSM8K follow-up diagnosis is now explicitly testing three additional hypotheses:
   - best checkpoint may occur before the final adapter,
   - shorter CoT supervision may preserve arithmetic quality better than long-CoT supervision,
   - rationale tokens may currently dominate the loss too strongly relative to the final answer.
9. A combined GSM8K repair run is now available:
   - `B2_GSM8K_REPAIR_ANSWER_WEIGHTED_CKPT`
   - purpose:
     - combine answer-weighted supervision with dense checkpoint selection,
     - test whether the best retained checkpoint can fully recover the frozen GSM8K baseline,
     - avoid treating the final adapter as the reportable model when late-run drift is known to be severe.
10. A dedicated cross-task interference suite is now part of the remaining Phase B work:
   - evaluate the best StrategyQA adapter on GSM8K,
   - evaluate GSM8K adapters on StrategyQA,
   - compare whether long-CoT GSM8K tuning causes broader cross-task damage than shorter GSM8K styles.
