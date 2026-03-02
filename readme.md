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

Official Phase B kickoff (recommended):

```bash
ACTIVE_PHASE_B_GROUP=B1_SMOKE RUN_PREFIX=phase_b_kickoff bash scripts/run_phase_b_training_suite.sh
```

Full-dataset gain runs:

```bash
ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_FULL RUN_PREFIX=phase_b_strategyqa_full bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_FULL RUN_PREFIX=phase_b_gsm8k_full bash scripts/run_phase_b_training_suite.sh
```

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
