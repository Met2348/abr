# Phase B Plan: From Phase A Baselines to First SFT/PEFT Run

Last updated: 2026-02-27 01:47:19 +0800

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
3. B1 exit gate is still pending until a tiny smoke run completes end-to-end.
