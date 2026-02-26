# Phase A Report: Current Experiment Status and Handoff

Last updated: 2026-02-27 01:16:23 +0800

## 1. Purpose

This report is a newcomer-oriented summary of what Phase A completed, what was fixed, what remains open, and what the team should do next for BCR/ABR research.

Phase A scope was:
1. Build reproducible data/prompt/eval infrastructure.
2. Establish reliable baseline measurements on StrategyQA.
3. Diagnose whether low scores came from model reasoning limits or output-format/extraction issues.

## 2. Phase A Final Conclusion (Concise)

1. Phase A is concluded and successful as an infrastructure + diagnosis stage.
2. Parse-error inflation was largely due to protocol/extraction/free-form formatting.
3. After binary-choice decoding, parse error is `0.0000` and decision quality can be measured fairly.
4. Remaining quality gap is real model/prompt behavior:
   - direct binary-choice outperforms CoT binary-choice.

## 3. What Was Implemented

## 3.1 Data and preparation

1. Canonical schema and dataset loaders.
2. Deterministic step builder and preprocess scripts.
3. Deterministic split policy and prepared artifacts.

## 3.2 Phase A evaluation stack

1. Prompt template registry with versioning.
2. Dataset-aware answer extraction and evaluation.
3. One-script inference + evaluation + comparison.
4. One-click benchmark suite (`A1..A6`) with persisted logs and final summaries.

## 3.3 Reliability improvements

1. Reduced noisy run logs; persisted run/suite logs.
2. Added parseable coverage metrics:
   - `n_parseable`
   - `acc_parseable`
3. Added StrategyQA decode modes:
   - `freeform`
   - `binary_choice`
4. Batched inference hardening:
   - fixed decoder-only right-padding bug in batch path,
   - kept deterministic order and added OOM backoff safeguards,
   - validated speedup without quality regression on StrategyQA direct baseline.

## 4. Key Experimental Results

All key runs used `n=193` StrategyQA validation samples.

## 4.1 Freeform CoT baseline symptom

Representative result:
- `accuracy=0.1036`
- `parse_error_rate=0.8446`
- `n_parseable=30`
- `acc_parseable=0.6667`

Interpretation:
- parseable-only accuracy looked moderate, but coverage was very low.
- total accuracy was dominated by parse failures.

## 4.2 A6 binary-choice confirmation

`ACTIVE_PARAM_GROUP=A6` result:

1. `direct_binchoice`:
   - `accuracy=0.6788`
   - `parse_error_rate=0.0000`
2. `cot_binchoice`:
   - `accuracy=0.5803`
   - `parse_error_rate=0.0000`
3. Reproducibility:
   - repeat run unchanged (`delta_accuracy=+0.0000`, `changed_samples=0`)

Interpretation:
- parse failures can be removed by protocol choice.
- remaining direct-vs-CoT gap is decision-quality, not parser noise.

## 4.3 Freeform vs binary-choice diagnosis

Binary-choice A2 sweep produced identical values across token budgets, which is expected:
- this mode scores yes/no directly and does not depend on `max_new_tokens`.

This confirms:
1. freeform metrics reflect both reasoning and output-format compliance,
2. binary-choice isolates decision quality.

## 4.4 Batching Validation (post-fix)

StrategyQA direct baseline (`n=193`, `max_new_tokens=32`, no-sample, seed=42):
1. `batch_size=1`:
   - `accuracy=0.5492`
   - `parse_error_rate=0.1762`
   - `gen_sample_rate=1.180`
   - `gen_elapsed_sec=163.53`
2. `batch_size=4` (after padding fix):
   - `accuracy=0.5596`
   - `parse_error_rate=0.1658`
   - `gen_sample_rate=4.189`
   - `gen_elapsed_sec=46.08`

Interpretation:
1. Batching is now a safe default for Phase A throughput.
2. Previous batch-quality collapse was implementation bug, not a model limitation.
3. A reproducibility pair for the fixed batch setting should still be run as routine gate.

## 5. Problems Probed and Fixed

1. Prompt leakage parse failures (example pattern: `noHuman: ...`):
   - fixed by extraction hardening + chat-marker truncation.
2. Misleading reporting via parseable subset only:
   - fixed by reporting both coverage and parseable accuracy.
3. Freeform-format confound:
   - fixed diagnostically via binary-choice decode mode.

## 6. Open (Unsolved) Problems

1. CoT prompt underperforms direct prompt on StrategyQA decision quality.
2. Freeform CoT remains fragile at low token budgets.
3. Phase A has no training-time value head / BCR loss / ABR router yet.
4. Faithfulness metrics beyond parsing/accuracy are not yet integrated end-to-end.
5. Batched-mode reproducibility on all target datasets is not yet fully documented (StrategyQA fixed and validated; GSM8K parity check still recommended).

## 7. What To Do Next

1. Freeze Phase A scripts and metrics schema as benchmark references.
2. Start Phase B in new scripts/modules:
   - model wrapper + value head
   - BCR-lite losses
   - smoke training and eval
3. Keep two StrategyQA tracks in all future reports:
   - decision-quality track (`binary_choice`)
   - end-to-end format track (`freeform`)

## 8. Re-Planned Route to Final BCR/ABR Goal

1. Stage B: BCR-lite training stack (SFT + value temporal consistency).
2. Stage C: faithfulness eval completion (calibration + corruption AUC + localization).
3. Stage D: ABR heuristic router (`gen/ver/fin`) under fixed budgets.
4. Stage E: ABR learned router (RL) after deterministic Stage D stability.
5. Final stage: multi-dataset ablations and paper-grade reproducibility package.

## 9. Handoff Notes for New Team Members

1. Start with:
   - `readme.md`
   - `result_records.md`
   - `TODO_ours.md`
2. Reproduce Phase A quickly with:
   - `scripts/run_phase_a_benchmark_suite.sh`
   - groups `A5` and `A6`.
3. Treat Phase A as frozen baseline; do not mix Phase B code into `scripts/phase_a_*`.

## 10. Retrospective Lessons (What Phase A Actually Taught Us)

1. Evaluation infrastructure is part of the experiment, not a neutral observer.
   - Parser/extractor/evaluator changes can materially alter reported accuracy.
2. Coverage-aware reporting is mandatory.
   - For parse-sensitive tasks, `accuracy` without parseability fields is incomplete.
3. Protocol can be a bigger bottleneck than model reasoning.
   - StrategyQA freeform vs binary-choice showed that format channel can dominate errors.
4. Speed changes can silently alter correctness.
   - Batching required explicit padding-side fixes and regression tests before being trusted.
5. Determinism is a baseline property, not optional polish.
   - Repeat-run equality checks prevented false conclusions from drift/noise.
6. Token budget sweeps without truncation diagnostics can be misleading.
   - Cap pressure must be measured and interpreted together with extraction method usage.

## 11. Pre-Phase-B Commit Checklist (Recommended)

Before tagging/committing Phase A as baseline:
1. Ensure docs and code agree on final baseline settings:
   - StrategyQA binary-choice baseline and freeform baseline,
   - GSM8K direct/CoT baseline with current evaluator semantics.
2. Store one reproducibility pair for:
   - StrategyQA direct baseline,
   - StrategyQA batched baseline,
   - GSM8K direct baseline.
3. Confirm all baseline runs have:
   - manifest,
   - metrics with evaluator version,
   - persisted console log.
4. Keep a frozen "do-not-edit" note for `scripts/phase_a_*` during early Phase B development.
