# OURS Research Project (Independent from external BCR code)

This repository is building an in-house reasoning-faithfulness pipeline from scratch.

Current milestone status:
- First and second data-pipeline milestones are complete:
  - canonical schema
  - multi-dataset loaders
  - data validation CLI
  - starter unit/integration tests
  - step-level builder (`step_builder.py`)
  - preprocessing artifact CLI (`preprocess_steps.py`)

Primary roadmap:
- `TODO_ours.md`

Context files:
- `idea_polish.md`
- `idea_formulation.md`
- `readme_dev.md`

## 1. Model files

Please manually place the 4 model shard files in:

`assets/models/Qwen2.5-7B-Instruct`

Expected filenames:
- `model-00001-of-00004.safetensors`
- `model-00002-of-00004.safetensors`
- `model-00003-of-00004.safetensors`
- `model-00004-of-00004.safetensors`

## 2. Dataset downloads

Use:

```bash
bash download_datasets.sh
```

or:

```bash
bash download_datasets.sh assets/datasets
```

Why this script exists:
- it uses valid, namespaced HF dataset IDs
- it includes fallbacks for datasets with moved/disabled repos

## 3. Quick Start (Novice-Friendly)

```bash
cd /home/zling/y/bcr/ref
conda activate bcr
export HF_HOME=$PWD/assets/hf_cache
export HF_DATASETS_CACHE=$PWD/assets/hf_cache/datasets
export PYTHONPATH=$PWD/src
```

### 3.1 Run data checks (first priority)

Core datasets first:

```bash
python scripts/check_data.py --datasets gsm8k strategyqa --split train --limit 5
```

Additional datasets:

```bash
python scripts/check_data.py --datasets drop proofwriter --split train --limit 5
python scripts/check_data.py --datasets bbh --split test --limit 5 --bbh-task boolean_expressions
python scripts/check_data.py --datasets hendrycks_math --split train --limit 5 --hendrycks-subset algebra
python scripts/check_data.py --datasets logiqa --split train --limit 5
```

Expected success signal:
- `Result: SUCCESS`
- non-empty `question` and `answer` counts
- sensible sample preview

### 3.2 Run tests

Preferred command:

```bash
python -m pytest -q tests/unit/test_data_schema.py tests/integration/test_data_loaders.py
```

If you see `No module named pytest` in `bcr` environment:

```bash
python -m pip install -U pytest
```

## 4. Implemented Data Pipeline Files

- `src/ours/data/schema.py`: canonical sample contract + validation
- `src/ours/data/loaders.py`: per-dataset normalization to canonical schema
- `src/ours/data/step_builder.py`: deterministic canonical-sample -> step-sequence builder
- `scripts/check_data.py`: CLI for readiness checks and sample previews
- `scripts/preprocess_steps.py`: batch preprocessing CLI that writes reusable artifacts
- `tests/unit/test_data_schema.py`: schema unit tests
- `tests/integration/test_data_loaders.py`: loader smoke tests
- `tests/unit/test_step_builder.py`: step-builder behavior tests

## 5. Common Failure Cases

1. `Repository not found` on HF:
- Usually wrong dataset repo ID (use `download_datasets.sh`).

2. `403 DisabledRepoError` for old competition math repo:
- Use `EleutherAI/hendrycks_math` fallback (already in script).

3. `pyarrow` / `datasets` runtime mismatch:
- Activate a clean env (recommended: `bcr`) and reinstall compatible versions.

4. Permission error under default HF cache:
- Ensure `HF_HOME` and `HF_DATASETS_CACHE` point to writable project paths.

## 6. What to do next

Next milestone:
- run step-level preprocessing to generate reusable artifacts,
- then move to baseline SFT/value training.

### 6.1 Build step-level artifacts

Use this command for a first smoke run:

```bash
python scripts/preprocess_steps.py \
  --datasets gsm8k strategyqa \
  --split train \
  --limit 200 \
  --batch-size 64
```

Outputs are written under:

`assets/artifacts/steps/<dataset>/<split__variant__fingerprint>/`

Key files:
- `sample_sequences.jsonl`: one JSON object per sample, containing full step sequence.
- `flat_steps.jsonl`: one JSON object per step (easy for direct model ingestion).
- `summary.json`: aggregate counters and averages for the run.
- `manifest.json`: exact run configuration and deterministic fingerprint.
- `errors.jsonl`: sample-level preprocessing failures (empty if all good).

Resume behavior:
- rerunning the same command will reuse existing artifacts (`--resume` default true).
- use `--overwrite` to force regeneration.

### 6.2 How to read preprocessing outputs

After one run, inspect:

```bash
ls -lah assets/artifacts/steps/strategyqa
ls -lah assets/artifacts/steps/strategyqa/<run_folder>
```

Then verify content:

```bash
sed -n '1,2p' assets/artifacts/steps/strategyqa/<run_folder>/sample_sequences.jsonl
sed -n '1,20p' assets/artifacts/steps/strategyqa/<run_folder>/summary.json
sed -n '1,20p' assets/artifacts/steps/strategyqa/<run_folder>/manifest.json
```

What each file means:
- `sample_sequences.jsonl`: one line per original sample, containing ordered steps.
- `flat_steps.jsonl`: one line per step across all samples (easy for step-level model input).
- `summary.json`: counts (samples, steps, role distribution, averages).
- `manifest.json`: exact run config and fingerprint for reproducibility.
- `errors.jsonl`: per-sample failures (should be empty for clean runs).

## 7. Change History (Novice Digest)

This section summarizes the most recent engineering changes in plain language.

### 7.1 Loader improvements
- DROP and ProofWriter now include context in the canonical `question` field:
  - DROP: `Passage: ...` + `Question: ...`
  - ProofWriter: `Theory: ...` + `Question: ...`
- Hendrycks MATH now separates:
  - `cot`: full worked solution
  - `answer`: extracted final answer (prefer last `\\boxed{...}`)

Why this matters:
- Better training signal quality for reasoning tasks.
- Less information loss from dataset-specific formats.

### 7.2 New step preprocessing pipeline
- Added step builder with deterministic step IDs and configurable splitting:
  - split modes: `auto`, `newline`, `sentence`
  - optional question step 0
  - optional answer terminal step
- Added preprocessing CLI that writes reusable artifacts and supports:
  - `--resume` (default reuse)
  - `--overwrite` (force regeneration)
  - `--strict` (fail fast on bad samples)

Why this matters:
- You preprocess once and reuse outputs across experiments.
- You can compare experiments fairly because run fingerprints are stable.

## 8. Recommended Next Commands

1. Run a tiny smoke test:
```bash
python scripts/preprocess_steps.py --datasets strategyqa --split train --limit 20 --batch-size 8
```

2. Run broader preprocessing:
```bash
python scripts/preprocess_steps.py --datasets gsm8k strategyqa drop proofwriter --split train --limit 2000 --batch-size 128
```

3. Run tests to confirm pipeline stability:
```bash
python -m pytest -q
```
