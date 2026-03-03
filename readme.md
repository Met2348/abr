# OURS Reasoning Research Pipeline

A research codebase for building and evaluating reasoning-faithfulness workflows (Phase A baseline completed, Phase B SFT/PEFT track in progress).

## What This Repo Includes

- Canonical dataset schema and loaders (`src/ours/data/`)
- Step preprocessing pipeline (`src/ours/data/step_builder.py`, `scripts/preprocess_steps.py`)
- Phase A benchmark stack:
  - prompt templates,
  - deterministic splitting,
  - answer extraction + evaluation,
  - one-click benchmark suites (`scripts/run_phase_a_benchmark_suite.sh`)
- Phase B training skeleton:
  - SFT/PEFT training entrypoint (`scripts/phase_b_train_sft.py`)
  - Phase B suite runner (`scripts/run_phase_b_training_suite.sh`)
  - Live Phase B report (`phase_B_report.md`)

## Current Status

- Phase A infrastructure is stable and reproducible.
- Batched inference path is enabled and validated.
- Phase B B0 planning is complete; B1 training skeleton is implemented.
- Phase B is now the official active development track.

## Quick Start

### 1) Prepare environment

Use your preferred Python environment and install required runtime deps (`torch`, `transformers`, `datasets`, `pytest`, etc.).

Recommended env vars:

```bash
export HF_HOME=$PWD/assets/hf_cache
export HF_DATASETS_CACHE=$PWD/assets/hf_cache/datasets
export PYTHONPATH=$PWD/src
```

### 2) Download datasets

```bash
bash download_datasets.sh
```

### 3) Smoke-check canonical loaders

```bash
python scripts/check_data.py --datasets gsm8k strategyqa --split train --limit 5
```

### 4) Run core tests

```bash
python -m pytest -q
```

## Phase A Public Entry Points

Prepare artifacts:

```bash
python scripts/phase_a_prepare.py \
  --datasets strategyqa \
  --source-split train \
  --split-policy hash \
  --target-style answer_only \
  --template-id qa_direct \
  --template-version 1.0.0 \
  --limit 200
```

Generate + evaluate:

```bash
python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl <prepared_validation_jsonl> \
  --run-name baseline_eval \
  --require-cuda \
  --no-do-sample \
  --max-new-tokens 64
```

Run output and `metrics.json` now include sampled VRAM usage stats (`vram_mean_gib`, `vram_max_gib`) for reporting.
Phase A JSONL readers are robust to Unicode line separators (for example `U+2028`) in prompt/prediction text.

Truncation recovery is enabled by default for math datasets (`gsm8k`, `hendrycks_math`).
You can tune it with:
- `--truncation-recovery-rounds`
- `--truncation-recovery-extra-tokens`
- `--truncation-recovery-datasets`
- `--no-truncation-recovery`

Run one-click benchmark suite:

```bash
bash scripts/run_phase_a_benchmark_suite.sh
```

Suite final summaries now include artifact-only instability diagnostics under
`RESULT TABLE` (multi-tag rate, first/last tag disagreement, tag-switch rate, pairwise flip rate).

Standalone artifact analysis (no re-inference):

```bash
python scripts/phase_a_analyze_instability.py \
  --run-dirs assets/artifacts/phase_a_runs/<run_dir_a> assets/artifacts/phase_a_runs/<run_dir_b>
```

Prompt-style sweeps:

```bash
ACTIVE_PARAM_GROUP=A7 RUN_PREFIX=strategyqa_style_sweep bash scripts/run_phase_a_benchmark_suite.sh
ACTIVE_PARAM_GROUP=A8 RUN_PREFIX=gsm8k_style_sweep bash scripts/run_phase_a_benchmark_suite.sh
# Whole-corpus StrategyQA review (train+validation+test aggregate)
ACTIVE_PARAM_GROUP=A11 RUN_PREFIX=strategyqa_whole_2290 bash scripts/run_phase_a_benchmark_suite.sh
# StrategyQA token-stress variant (example)
ACTIVE_PARAM_GROUP=A11_256 RUN_PREFIX=strategyqa_whole_t256 bash scripts/run_phase_a_benchmark_suite.sh
# Whole-corpus GSM8K review (train+validation+test aggregate)
ACTIVE_PARAM_GROUP=A12 RUN_PREFIX=gsm8k_whole_corpus bash scripts/run_phase_a_benchmark_suite.sh
```

For whole-corpus groups (`A11`/`A12`), env runtime overrides such as `BATCH_SIZE=...` are honored.

## Phase B Public Entry Points

Current consolidated Phase B findings:
- `phase_B_report.md`

Official Phase B kickoff (recommended):

```bash
ACTIVE_PHASE_B_GROUP=B1_SMOKE RUN_PREFIX=phase_b_kickoff bash scripts/run_phase_b_training_suite.sh
```

Heavy-run note:
- do not launch multiple full-dataset Phase B suites on one GPU at the same time,
- if a suite exits early, `final_summary.md` now records `status: failed` and `failed_stage`.

Full-dataset gain runs:

```bash
ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_FULL RUN_PREFIX=phase_b_strategyqa_full bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_FULL RUN_PREFIX=phase_b_gsm8k_full bash scripts/run_phase_b_training_suite.sh
```

StrategyQA scaling diagnostics:

```bash
ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_DIAG_EPOCH_200 RUN_PREFIX=strategyqa_diag_e200 bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_DIAG_EPOCH_300 RUN_PREFIX=strategyqa_diag_e300 bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_DIAG_LORA_R8 RUN_PREFIX=strategyqa_diag_r8 bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_DIAG_LORA_R32 RUN_PREFIX=strategyqa_diag_r32 bash scripts/run_phase_b_training_suite.sh
```

GSM8K diagnostic runs:

```bash
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_LR_5E5 RUN_PREFIX=gsm8k_diag_lr5e5 bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_LR_1E4 RUN_PREFIX=gsm8k_diag_lr1e4 bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_EPOCH_025 RUN_PREFIX=gsm8k_diag_e025 bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_EPOCH_050 RUN_PREFIX=gsm8k_diag_e050 bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_DIRECT_STYLE RUN_PREFIX=gsm8k_diag_direct bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_EQUATION_STYLE RUN_PREFIX=gsm8k_diag_equation bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_CHECKPOINT_SWEEP RUN_PREFIX=gsm8k_diag_ckpt_sweep bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_SHORT_COT RUN_PREFIX=gsm8k_diag_short_cot bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_ANSWER_WEIGHTED RUN_PREFIX=gsm8k_diag_answer_weighted bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_REPAIR_ANSWER_WEIGHTED_CKPT RUN_PREFIX=gsm8k_repair_aw_ckpt bash scripts/run_phase_b_training_suite.sh
```

New GSM8K diagnostics now cover three additional hypotheses:
- best checkpoint may occur before the final saved adapter,
- shorter CoT supervision may preserve arithmetic quality better than long CoT,
- final-answer tokens may need more loss weight than rationale tokens.

The current combined GSM8K repair attempt is:
- answer-weighted long-CoT supervision,
- plus dense checkpoint saving and held-out checkpoint sweep,
- so the best checkpoint is selected instead of the final adapter.

Cross-task interference runs:

```bash
ACTIVE_CROSS_TASK_GROUP=B3_XTASK_STRAT_R32_TO_GSM8K RUN_PREFIX=xtask_strat_r32_to_gsm8k bash scripts/run_phase_b_cross_task_suite.sh
ACTIVE_CROSS_TASK_GROUP=B3_XTASK_GSM8K_FULL_TO_STRAT RUN_PREFIX=xtask_gsm8k_full_to_strat bash scripts/run_phase_b_cross_task_suite.sh
ACTIVE_CROSS_TASK_GROUP=B3_XTASK_GSM8K_DIRECT_TO_STRAT RUN_PREFIX=xtask_gsm8k_direct_to_strat bash scripts/run_phase_b_cross_task_suite.sh
ACTIVE_CROSS_TASK_GROUP=B3_XTASK_GSM8K_EQUATION_TO_STRAT RUN_PREFIX=xtask_gsm8k_equation_to_strat bash scripts/run_phase_b_cross_task_suite.sh
```

These groups measure whether a task-specific adapter transfers or interferes when
evaluated on the other task, which is directly relevant to later BCR/ABR work.

## Phase C Public Entry Points

Phase C guidance:
- `phase_C_plan.md`

Current implemented scope:
- `C0`: freeze the Phase C contracts and execution order
- `C1`: build deterministic prefix, corruption, and optional rollout-target artifacts
- `C2`: train and evaluate a frozen-backbone value head on C1 artifacts

Full lifecycle suite entrypoint:
- `scripts/run_phase_c_value_suite.sh`
- question-level P(IK) lifecycle: `scripts/run_phase_c_pik_suite.sh`

Main entrypoint:

```bash
python -u scripts/phase_b_prepare_value_data.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl \
  --run-name strategyqa_value_smoke \
  --max-samples 128 \
  --build-corruptions \
  --no-build-rollouts
```

Phase C C0/C1 output directory:
- `assets/artifacts/phase_c_data/<dataset>/<run_name>__<fingerprint>/`

Key output files:
- `step_sequences.jsonl`
- `prefixes.jsonl`
- `corruptions.jsonl` if enabled
- `rollout_predictions.jsonl` and `rollout_targets.jsonl` if enabled
- `corruption_rollout_targets.jsonl` and `pair_quality.jsonl` when `--build-pair-quality` is enabled
- `manifest.json`
- `summary.json`
- `summary.md`

C2 training entrypoint:

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/phase_b_train_value.py \
  --train-dir assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts__<train_fingerprint> \
  --eval-dir assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts_val__<eval_fingerprint> \
  --run-name strategyqa_value_c2_smoke \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --per-device-train-batch-size 256 \
  --per-device-eval-batch-size 256 \
  --learning-rate 1e-3 \
  --num-train-epochs 5 \
  --use-contrastive-loss \
  --lambda-contrastive 1.0 \
  --contrastive-margin 0.1
```

New C2 options (can be toggled independently):
- calibration objective: `--calibration-loss {mse,bce,bce_mse}`
- post-hoc calibration: `--posthoc-calibration {none,temperature,isotonic}` and `--checkpoint-selection-metric {raw_brier,posthoc_brier}`
- adaptive cal/contrastive balancing: `--adaptive-loss-balancing {none,uncertainty}`
- confidence-aware calibration weighting: `--calibration-sample-weighting` (includes `q_weight`, `q_weight_parseable`), `--calibration-weight-floor`, `--calibration-weight-gamma`
- contrastive pair filtering: `--contrastive-pair-filter` (includes `label_quality`, `confidence_parseable_label`), `--contrastive-confidence-threshold`, `--contrastive-parseable-threshold`, `--contrastive-label-delta-q-min`, `--contrastive-label-z-min`, `--contrastive-label-pair-weight-min`, `--contrastive-require-pair-pass-gate`, `--contrastive-use-pair-weights`
- calibration target smoothing: `--calibration-target-smoothing`
- contrastive score-gap mining: `--contrastive-score-gap-min`, `--contrastive-score-gap-max`

C2 standalone evaluation entrypoint:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_b_eval_faithfulness.py \
  --value-run-dir assets/artifacts/phase_c_runs/strategyqa_value_c2_smoke_<timestamp> \
  --eval-dir assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts_val__<eval_fingerprint> \
  --checkpoint-name best \
  --posthoc-calibration from_run \
  --run-name strategyqa_value_c2_eval
```

One-command lifecycle (C1 train + C1 eval + C2 train + C2 eval):

```bash
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_c_strategyqa_smoke \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_value_suite.sh
```

Quality-first lifecycle examples:

```bash
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_QUALITY_FIRST \
RUN_PREFIX=phase_c_quality_first \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_c_value_suite.sh

ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_QUALITY_FIRST_FULL \
RUN_PREFIX=phase_c_quality_first_full \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_c_value_suite.sh
```

Question-level P(IK) lifecycle (new):

```bash
ACTIVE_PHASE_C_PIK_GROUP=PIK_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_c_pik_smoke \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_pik_suite.sh
```

Why add P(IK):
- Prefix-level value runs are harder and noisier.
- P(IK) is a simpler diagnostic: one question -> `K` sampled answers -> empirical success-rate target.
- It checks whether the value head can learn any confidence signal before returning to prefix-level objectives.

Main P(IK) scripts:
- `scripts/phase_c_prepare_pik_data.py`
- `scripts/phase_c_train_pik.py`
- `scripts/phase_c_eval_pik.py`

Supported lifecycle groups:
1. `C2_STRATEGYQA_SMOKE`
2. `C2_STRATEGYQA_FULL`
3. `C2_STRATEGYQA_TRICK1_BCE`
4. `C2_STRATEGYQA_TRICK2_POSTHOC_TEMP`
5. `C2_STRATEGYQA_TRICK3_ADAPTIVE_BALANCE`
6. `C2_STRATEGYQA_TRICK4_ISOTONIC`
7. `C2_STRATEGYQA_TRICK5_WEIGHTED_CAL`
8. `C2_STRATEGYQA_TRICK6_PAIR_FILTER`
9. `C2_STRATEGYQA_TRICK7_COMBINED`
10. `C2_STRATEGYQA_TRICK8_LABEL_SMOOTH`
11. `C2_STRATEGYQA_TRICK9_HARD_NEG_MINING`
12. `C2_STRATEGYQA_TRICK10_K16_COMBINED`
13. `C2_STRATEGYQA_QUALITY_FIRST`
14. `C2_STRATEGYQA_QUALITY_FIRST_FULL`

Smoke training run:

```bash
python -u scripts/phase_b_train_sft.py \
  --config-json configs/phase_b/peft_smoke_strategyqa.json
```

If PEFT/transformers env errors appear, repair in your env:

```bash
python -m pip install -U "transformers>=4.44,<5" "huggingface-hub>=0.23.2,<1.0" "accelerate>=1.1,<2" "peft>=0.11,<0.14"
```

Training suite:

```bash
bash scripts/run_phase_b_training_suite.sh
```

Full-dataset Phase B groups now run:
1. held-out baseline eval with the frozen base model,
2. whole-train-split PEFT training,
3. held-out post-train eval with the saved adapter/model,
4. one gain report that states the before/after delta.

Post-train evaluation against frozen Phase A metrics:

```bash
python -u scripts/phase_b_eval.py \
  --phase-b-run-dir assets/artifacts/phase_b_runs/<phase_b_run_dir> \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/<prepared_dir>/validation.jsonl \
  --run-name phase_b_eval_after_train \
  --strategyqa-decode-mode binary_choice \
  --max-new-tokens 16
```

This evaluates the trained Phase B artifact with the same Phase A benchmark logic,
so you can compare post-train task accuracy against pre-train baselines.

For the new `B2_*` groups, this comparison is automated and written to:
- `assets/artifacts/phase_b_logs/<RUN_PREFIX>/peft_gain_summary.json`
- `assets/artifacts/phase_b_logs/<RUN_PREFIX>/peft_gain_summary.md`

## Key Documentation

- `phase_A_report.md`: Phase A conclusions and outcomes
- `phase_B_plan.md`: Phase B lifecycle and goals
- `result_records.md`: experiment records and diagnosis history
- `foundation_reliability_audit.md`: reliability risks and hardening notes

## Note for Maintainers

Detailed/private operational notes and hardcoded local workflow details are maintained in `readme_full.md`.

Documentation convention for maintained runtime files:
- every `py`/`sh` file should begin with a short abstract,
- every function/class should carry a beginner-friendly docstring or nearby comment,
- examples should be included when they materially improve readability.
