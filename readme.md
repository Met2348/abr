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

Run one-click benchmark suite:

```bash
bash scripts/run_phase_a_benchmark_suite.sh
```

Prompt-style sweeps:

```bash
ACTIVE_PARAM_GROUP=A7 RUN_PREFIX=strategyqa_style_sweep bash scripts/run_phase_a_benchmark_suite.sh
ACTIVE_PARAM_GROUP=A8 RUN_PREFIX=gsm8k_style_sweep bash scripts/run_phase_a_benchmark_suite.sh
```

## Phase B Public Entry Points

Smoke training run:

```bash
python -u scripts/phase_b_train_sft.py \
  --config-json configs/phase_b/peft_smoke_strategyqa.json
```

Training suite:

```bash
bash scripts/run_phase_b_training_suite.sh
```

## Key Documentation

- `phase_A_report.md`: Phase A conclusions and outcomes
- `phase_B_plan.md`: Phase B lifecycle and goals
- `result_records.md`: experiment records and diagnosis history
- `foundation_reliability_audit.md`: reliability risks and hardening notes

## Note for Maintainers

Detailed/private operational notes and hardcoded local workflow details are maintained in `readme_full.md`.
