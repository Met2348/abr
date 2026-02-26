# OURS Research Project (Independent from external BCR code)

This repository is building an in-house reasoning-faithfulness pipeline from scratch.

Current milestone status (2026-02-27):
- Phase A is concluded.
- Phase A core conclusion:
  - parse-error inflation was mostly protocol/extraction/free-form formatting related,
  - after binary-choice decode, StrategyQA parse error reaches `0.0000`,
  - remaining errors are model/prompt decision-quality gaps (not parser-only noise).
- Current StrategyQA decision-quality baseline (`binary_choice`, n=193):
  - `direct_binchoice`: `accuracy=0.6788`
  - `cot_binchoice`: `accuracy=0.5803`
- Batched freeform inference is now validated after padding fix:
  - StrategyQA direct (`max_new_tokens=32`, n=193):
  - `batch_size=1`: `accuracy=0.5492`, `parse_error_rate=0.1762`, `1.180 sample/s`
  - `batch_size=4`: `accuracy=0.5596`, `parse_error_rate=0.1658`, `4.189 sample/s`
- Phase B should start now as a separate training track (value head + BCR-lite), while keeping Phase A scripts frozen as benchmark references.

Primary roadmap:
- `TODO_ours.md`

Context files:
- `idea_polish.md`
- `idea_formulation.md`
- `readme_dev.md`
- `result_records.md` (experiment history + diagnosis)
- `phase_A_report.md` (newcomer-facing Phase A closeout report)
- `foundation_reliability_audit.md` (low-level risk scan + hardening plan before Phase B scale)

## Phase A Retrospective (Before You Start Phase B)

Treat these as operational lessons from the full Phase A cycle:
1. Evaluation logic is part of the experiment. Keep evaluator/version fixed when comparing runs.
2. Always read metrics as a tuple:
   - `accuracy`,
   - `parse_error_rate`,
   - `n_parseable`,
   - `acc_parseable`.
3. For StrategyQA, maintain both tracks:
   - `binary_choice` for decision quality,
   - `freeform` for end-to-end compliance.
4. Batching is now the default speed path, but trust it only with correctness guards:
   - left-padding in batch decode for decoder-only models,
   - deterministic parity checks vs batch size 1.
5. For math tasks, watch truncation/extraction diagnostics before interpreting accuracy:
   - cap hit rate,
   - final-answer tag rate,
   - fallback extraction rate.

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
- `src/ours/phase_a/`: Stage A baseline package (prompting, split policy, extraction, evaluator)
- `scripts/check_data.py`: CLI for readiness checks and sample previews
- `scripts/preprocess_steps.py`: batch preprocessing CLI that writes reusable artifacts
- `scripts/phase_a_prepare.py`: build model-ready Stage A prompt/target records
- `scripts/phase_a_eval_predictions.py`: evaluate model predictions with dataset-aware answer extraction
- `tests/unit/test_data_schema.py`: schema unit tests
- `tests/integration/test_data_loaders.py`: loader smoke tests
- `tests/unit/test_step_builder.py`: step-builder behavior tests
- `tests/unit/test_phase_a_*.py`: Stage A prompt/split/extraction/evaluator tests

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

## 9. Stage A Baseline (New)

Stage A objective:
- build a reproducible baseline harness before BCR/ABR-specific training.

This repo now implements Stage A foundations:
- versioned prompt templates,
- deterministic split policy,
- dataset-aware answer extraction,
- robust evaluator that does more than raw `==`.

### 9.1 Prepare Stage A artifacts

Example (StrategyQA, hash split, answer-only target):

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

Output folder pattern:
- `assets/artifacts/phase_a_prepared/<dataset>/<run_fingerprint>/`

Files created:
- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `summary.json`
- `manifest.json`

Notes:
- `--resume` is on by default (reuse matching run).
- use `--overwrite` to force regeneration.

### 9.2 Evaluate prediction JSONL files

Input JSONL must include:
- `sample_id`
- `dataset`
- `split`
- `raw_prediction`
- `gold_answer`

Run:

```bash
python scripts/phase_a_eval_predictions.py \
  --predictions <path_to_predictions.jsonl> \
  --run-name my_eval
```

Output:
- `assets/artifacts/phase_a_eval/<run_name_timestamp>/scored_predictions.jsonl`
- `assets/artifacts/phase_a_eval/<run_name_timestamp>/metrics.json`

### 9.3 Which template/target pair to start with

Recommended first baseline:
1. `--template-id qa_direct --target-style answer_only`
2. Then compare against:
`--template-id qa_cot_then_final --target-style cot_then_answer`

This A/B comparison is your Stage A baseline matrix.

Dataset-specific note:
- For GSM8K direct-answer baselines, prefer:
  - `--template-id qa_math_direct_final --target-style answer_only`
  - this avoids weak `last_number` extraction from truncated reasoning outputs.

### 9.4 Why this milestone is critical

- You now have reproducible prompt+target generation.
- You now have deterministic data splitting.
- You now have extraction-aware evaluation for natural-language outputs.
- These are prerequisites before adding value heads or ABR router logic.

### 9.5 One-Script Inference + Eval + Diff

Use:

```bash
python scripts/phase_a_generate_and_eval.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/21095d3c688a/validation.jsonl \
  --run-name qwen_strategyqa_val \
  --require-cuda \
  --no-do-sample \
  --max-new-tokens 64 \
  --log-every 10
```

What this script writes:
- `assets/artifacts/phase_a_runs/<run_name_timestamp>/predictions.jsonl`
- `assets/artifacts/phase_a_runs/<run_name_timestamp>/scored_predictions.jsonl`
- `assets/artifacts/phase_a_runs/<run_name_timestamp>/metrics.json`
- `assets/artifacts/phase_a_runs/<run_name_timestamp>/manifest.json`
- `assets/artifacts/phase_a_runs/<run_name_timestamp>/console.log`

Rerun the same experiment to compare differences:
1. Run the same command again (same `--input-jsonl`, `--run-name`, and generation settings).
2. The script auto-compares with the latest previous run of the same `run_name`.
3. It reports:
   - `delta_accuracy`
   - `delta_parse_err`
   - `changed_samples`

Notes for reproducibility:
- keep `--no-do-sample` for deterministic baseline behavior.
- keep `--seed` fixed (default `42`).
- keep input file and generation parameters unchanged.

Notes for long runs:
- use `python -u ...` to force unbuffered console output.
- `--log-every` controls candidate checkpoint stride.
- `--max-progress-lines` caps generation progress verbosity (default `5` lines including start line).
- use `--require-cuda` for strict benchmarking so the script aborts instead of silently using CPU.
- console output is persisted by default in each run folder as `console.log`.
- disable console-log persistence only if needed: add `--no-persist-console-log`.
- model loading progress bars are suppressed; logs show concise load start + completion time.
- `--strategyqa-decode-mode` controls StrategyQA decoding:
  - `freeform` (default): normal text generation,
  - `binary_choice`: score `yes` vs `no` directly (removes most format-related parse errors).
- `--batch-size` controls free-form decode batching:
  - `1` keeps old behavior (safest baseline),
  - `2/4/8` can improve throughput when VRAM allows.
- `--oom-backoff` (default on) auto-splits a failing batch on OOM:
  - useful for robust sweeps on shared servers,
  - turn off with `--no-oom-backoff` for strict failure behavior.
- `--truncate-chat-markers` (default on) trims leaked next-turn markers such as `[USER]` / `Human:`.
- For math datasets, script output now includes `math_diag` (extraction reliability signals):
  - `last_number_rate`,
  - `hit_token_limit_rate`,
  - `final_answer_tag_rate`.
- script output now also prints and stores generation throughput:
  - `gen_elapsed_sec`,
  - `gen_sample_rate`,
  - `oom_backoff_evts`.
- while running, you can monitor file progress:

```bash
wc -l assets/artifacts/phase_a_runs/<run_name_timestamp>/predictions.jsonl
```

### 9.6 One-Command Benchmark Suite (Prepare + Inference + Diagnostics)

Use:

```bash
bash scripts/run_phase_a_benchmark_suite.sh
```

The script now supports **param groups** (`A1`, `A2`, `A3`, `A4`, `A5`) for one-click experiment presets.

One-click switching:
1. Open `scripts/run_phase_a_benchmark_suite.sh`.
2. Change `ACTIVE_PARAM_GROUP` (for example from `A1` to `A2`).
3. Save and rerun the same script.

Group intent summary:
- `A1`: core reproduction (`direct_t32` repeat + `cot_t128` + `cot_t256`)
- `A2`: CoT token sweep (truncation/compliance diagnosis)
- `A3`: direct token sweep (speed/accuracy frontier)
- `A4`: determinism check (repeat same config and compare deltas)
- `A5`: strict yes/no compliance fix (`qa_binary_strict`) to reduce parse errors and token waste
- `A6`: binary-choice decode validation (removes free-form format noise for StrategyQA)

Each group prints:
- intention,
- what to observe,
- expected trend,
- and a summary table with `accuracy`, `parse_error_rate`, `acc_parseable`, plus delta fields when comparison is enabled.

Useful overrides (optional):

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_PREFIX=my_suite \
LIMIT=2000 \
COT_SWEEP_TOKENS="128 192 256 320 384" \
DIRECT_SWEEP_TOKENS="16 24 32 48 64" \
STRATEGYQA_DECODE_MODE=freeform \
TRUNCATE_CHAT_MARKERS=1 \
LOG_EVERY=5 \
MAX_PROGRESS_LINES=5 \
BATCH_SIZE=1 \
OOM_BACKOFF=1 \
bash scripts/run_phase_a_benchmark_suite.sh
```

Notes:
- The script no longer blocks concurrent runs.
- If concurrent runs are detected, it prints a warning and continues.

### 9.7 Batching Hands-On: How To See Speed Difference

Run the same config twice, only changing `--batch-size`.

Example (`batch_size=1` baseline):

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl \
  --run-name strat_batch1 \
  --require-cuda --dtype bfloat16 --device-map auto \
  --no-do-sample --seed 42 \
  --max-new-tokens 32 \
  --batch-size 1 \
  --oom-backoff \
  --log-every 10 --max-progress-lines 5 \
  --no-compare-latest-same-name
```

Then run batched:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl \
  --run-name strat_batch4 \
  --require-cuda --dtype bfloat16 --device-map auto \
  --no-do-sample --seed 42 \
  --max-new-tokens 32 \
  --batch-size 4 \
  --oom-backoff \
  --log-every 10 --max-progress-lines 5 \
  --no-compare-latest-same-name
```

Compare these fields in console or `metrics.json`:
1. `gen_sample_rate` (higher is better),
2. `gen_elapsed_sec` (lower is better),
3. `accuracy` / `parse_error_rate` (should stay comparable).
- If you see warning `decoder-only ... right-padding`, stop and fix tokenizer padding before trusting results.
- Current expected stable setup on A100 80G for StrategyQA direct:
  - `--batch-size 4 --oom-backoff`
  - quality parity with significant speedup.
- Default `CUDA_VISIBLE_DEVICES=0` is chosen for stable, simpler 7B benchmarking.
- Suite-level live logs are persisted to `assets/artifacts/phase_a_logs/<RUN_PREFIX>/suite.log`.
- At the end of a group run, it prints a **final summary block** (group + settings + table) and saves it to:
  - `assets/artifacts/phase_a_logs/<RUN_PREFIX>/final_summary.md`
- Disable suite log persistence if needed:

```bash
ENABLE_PERSISTED_LOGS=0 bash scripts/run_phase_a_benchmark_suite.sh
```
