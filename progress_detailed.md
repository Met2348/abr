# Progress Detailed Changelog

This file is prepend-only: newest entries must be added at the top (right below this header).

## 2026-02-26 18:36:48 +08 (+0800)
- Type: Feature / Docs / Workflow
- Summary: Refactored benchmark shell runner into a param-group system (`A1`..`A4`) for one-click reproducible experiment presets.
- Details:
  - Reworked `scripts/run_phase_a_benchmark_suite.sh` into a configurable group runner.
  - Added one-click group selector:
    - `ACTIVE_PARAM_GROUP=A1|A2|A3|A4`
    - user can switch by editing one variable (or env override) and rerunning the same script.
  - Added four explicit param groups:
    - `A1`: core reproduction (direct baseline + CoT comparison),
    - `A2`: CoT token sweep,
    - `A3`: direct token sweep,
    - `A4`: determinism validation repeats.
  - Added group-level explanatory metadata printed at runtime:
    - intention,
    - what to observe,
    - expected outcome trend.
  - Added generalized summary table output for any group, including:
    - `accuracy`,
    - `parse_error_rate`,
    - `acc_parseable`,
    - delta columns when comparison mode is active.
  - Updated `readme.md` section `9.6` to document:
    - one-click group switching flow,
    - group purpose summary,
    - updated environment overrides.
  - Validation:
    - `bash -n scripts/run_phase_a_benchmark_suite.sh` passed.
    - script marked executable.
- Files changed:
  - `scripts/run_phase_a_benchmark_suite.sh`
  - `readme.md`
  - `progress_detailed.md`
- Breaking changes:
  - [CAUTION] old suite-specific env knobs (`MAX_NEW_TOKENS_DIRECT`, `MAX_NEW_TOKENS_COT_SMALL`, `MAX_NEW_TOKENS_COT_LARGE`) are no longer the primary control path for grouped runs; use `ACTIVE_PARAM_GROUP` and sweep token lists (`COT_SWEEP_TOKENS`, `DIRECT_SWEEP_TOKENS`).

---

## 2026-02-26 18:32:31 +08 (+0800)
- Type: Docs / Knowledge Transfer
- Summary: Added teammate-facing experiment history ledger `result_records.md` to capture meaningful prior Phase A runs and conclusions with timestamps.
- Details:
  - Added new root file: `result_records.md`.
  - Included:
    - prepared artifact lineage (fingerprints, template/target style, creation timestamps),
    - early smoke-run records,
    - main StrategyQA run table (`n=193`) with:
      - run timestamp,
      - input prepared set,
      - generation budget,
      - accuracy,
      - parse error rate,
      - parseable sample count,
      - parseable-only accuracy,
    - operational incident notes (GPU visibility instability and CPU offload slow run),
    - trusted conclusions and next-experiment checklist.
  - Goal:
    - new teammates can understand prior outcomes quickly without digging through many run folders.
- Files changed:
  - `result_records.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-26 17:23:43 +08 (+0800)
- Type: Feature / Docs / Automation
- Summary: Added a one-command benchmark suite shell script that runs Phase A preparation + multi-variant inference/eval + diagnostic summary table.
- Details:
  - Added new executable script: `scripts/run_phase_a_benchmark_suite.sh`.
  - Script workflow:
    - prepares StrategyQA artifacts for both template/target variants,
    - runs deterministic direct baseline twice (for reproducibility check),
    - runs CoT variant at two token budgets (small vs large),
    - prints a compact summary table with:
      - `accuracy`,
      - `parse_error_rate`,
      - `acc_parseable` (accuracy on parseable outputs).
  - Added operational safeguards:
    - checks for concurrent `phase_a_generate_and_eval.py` processes and fails fast by default,
    - defaults to single-GPU benchmarking (`CUDA_VISIBLE_DEVICES=0`) for simpler/stabler 7B runs,
    - supports explicit environment-variable overrides for common benchmark knobs.
  - Updated `readme.md` with a new section:
    - “9.6 One-Command Benchmark Suite (Prepare + Inference + Diagnostics)”
    - includes default usage and override examples.
  - Validation:
    - `bash -n scripts/run_phase_a_benchmark_suite.sh` passed.
    - script marked executable.
- Files changed:
  - `scripts/run_phase_a_benchmark_suite.sh`
  - `readme.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-26 17:06:23 +08 (+0800)
- Type: Reliability / UX / Benchmarking Guard
- Summary: Added explicit CUDA visibility reporting and `--require-cuda` fail-fast mode to prevent accidental CPU benchmarking.
- Details:
  - Updated `scripts/phase_a_generate_and_eval.py`:
    - prints runtime CUDA diagnostics at start (`torch` version, CUDA build, availability, device count, GPU names),
    - prints model placement after load (`first_param` device and `hf_device_map` when present),
    - added `--require-cuda` flag to abort immediately when CUDA is not available.
  - Updated `readme.md` Phase A run command and notes:
    - included `--require-cuda` in benchmark example,
    - documented why this flag is important for strict benchmark reproducibility.
  - Validation:
    - `python -m py_compile scripts/phase_a_generate_and_eval.py` passed.
- Files changed:
  - `scripts/phase_a_generate_and_eval.py`
  - `readme.md`
  - `progress_detailed.md`
- Breaking changes:
  - None (default behavior unchanged unless `--require-cuda` is enabled).

---

## 2026-02-26 17:00:29 +08 (+0800)
- Type: Feature / UX / Observability
- Summary: Added live generation progress logging and ETA to long Phase A runs so job status is visible during benchmarking.
- Details:
  - Updated `scripts/phase_a_generate_and_eval.py`:
    - added `--log-every` (default `10`) to control progress print frequency,
    - prints run-time generation progress lines with:
      - completed/total samples,
      - completion percentage,
      - elapsed wall time,
      - throughput (samples/sec),
      - ETA.
    - explicitly flushes prediction output after each written row, so external monitoring commands (`wc -l`, `tail`) reflect near-real-time progress.
    - added compact duration formatter helper for readable progress output.
  - Updated `readme.md` Phase A section with new usage notes:
    - `--log-every` examples,
    - `python -u` for visible live logs,
    - quick `wc -l` command for progress checks.
  - Validation:
    - `python -m py_compile scripts/phase_a_generate_and_eval.py` passed.
- Files changed:
  - `scripts/phase_a_generate_and_eval.py`
  - `readme.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-26 14:57:01 +08 (+0800)
- Type: Feature / Docs / Fix / Test
- Summary: Added one-script Phase A inference+evaluation+comparison workflow and fixed runtime import/generation issues for novice-friendly reruns.
- Details:
  - Added new script: `scripts/phase_a_generate_and_eval.py`.
  - Script capabilities:
    - loads prepared Phase A JSONL input,
    - runs local model generation,
    - evaluates predictions via Phase A evaluator,
    - writes `predictions.jsonl`, `scored_predictions.jsonl`, `metrics.json`, `manifest.json`,
    - auto-compares with latest previous run of same `run_name` (or explicit `--compare-with`).
  - Implemented lazy runtime dependency imports (`torch`, `transformers`) so `--help` works even in mismatched environments.
  - Fixed runtime bug after lazy-import refactor:
    - [CAUTION] internal bug fixed where `_run_generation` referenced undefined global `torch`; now uses passed runtime module.
  - Added generation-config alignment to reduce confusing warnings when `do_sample=False`.
  - Updated `readme.md` with a dedicated section:
    - “9.5 One-Script Inference + Eval + Diff”
    - exact rerun instructions and reproducibility notes.
  - Validation results:
    - `python3 -m py_compile scripts/phase_a_generate_and_eval.py` passed.
    - `python3 scripts/phase_a_generate_and_eval.py --help` works.
    - `conda run -n bcr python scripts/phase_a_generate_and_eval.py ... --max-samples 2` completed.
    - `conda run -n bcr python scripts/phase_a_generate_and_eval.py ... --max-samples 1` completed and produced run-to-run diff.
- Files changed:
  - `scripts/phase_a_generate_and_eval.py`
  - `readme.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-26 02:49:59 +08 (+0800)
- Type: Feature / Docs / Test
- Summary: Started Phase A critical milestone with reproducible baseline foundations (prompting, split, extraction, evaluator) and runnable CLIs.
- Details:
  - Added new Stage A package `src/ours/phase_a/`:
    - `contracts.py`: typed dataclasses (`PromptTemplateSpec`, `PreparedSample`, `PredictionRecord`, `ScoredPrediction`).
    - `prompt_builder.py`: versioned dataset-agnostic prompt templates and target builders.
    - `splitting.py`: deterministic hash-based split policy (`train/validation/test`).
    - `answer_extraction.py`: dataset-aware prediction extraction (`strategyqa`, `gsm8k`, `hendrycks_math`, fallback).
    - `evaluator.py`: robust scoring pipeline with extraction + normalization + aggregate metrics.
    - `__init__.py`: Phase A public exports.
  - Added new Stage A scripts:
    - `scripts/phase_a_prepare.py`: builds model-ready prompt/target artifacts from canonical samples.
    - `scripts/phase_a_eval_predictions.py`: scores prediction JSONL files and writes metrics.
  - Added new Stage A tests:
    - `tests/unit/test_phase_a_prompt_builder.py`
    - `tests/unit/test_phase_a_splitting.py`
    - `tests/unit/test_phase_a_extraction_eval.py`
  - Updated `readme.md` with a dedicated “Stage A Baseline (New)” section:
    - how to prepare artifacts,
    - how to evaluate predictions,
    - recommended baseline A/B template-target matrix,
    - why this milestone is a prerequisite for ABR.
  - Validation results:
    - `python3 -m py_compile ...` on all new Phase A modules/scripts passed.
    - `python3 -m pytest -q` -> `29 passed, 1 skipped`.
    - Smoke run: `python3 scripts/phase_a_prepare.py --datasets strategyqa --source-split train --limit 10 --split-policy hash ...` succeeded.
    - Smoke run: `python3 scripts/phase_a_eval_predictions.py --predictions .../mock_predictions.jsonl --run-name smoke_eval` succeeded with expected metrics.
- Files changed:
  - `src/ours/phase_a/__init__.py`
  - `src/ours/phase_a/contracts.py`
  - `src/ours/phase_a/prompt_builder.py`
  - `src/ours/phase_a/splitting.py`
  - `src/ours/phase_a/answer_extraction.py`
  - `src/ours/phase_a/evaluator.py`
  - `scripts/phase_a_prepare.py`
  - `scripts/phase_a_eval_predictions.py`
  - `tests/unit/test_phase_a_prompt_builder.py`
  - `tests/unit/test_phase_a_splitting.py`
  - `tests/unit/test_phase_a_extraction_eval.py`
  - `readme.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-26 01:25:56 +08 (+0800)
- Type: Docs
- Summary: Expanded novice-facing documentation with explicit change history and step-preprocessing usage instructions.
- Details:
  - Updated `readme.md` milestone status to include second data-pipeline milestone (`step_builder.py` and `preprocess_steps.py`).
  - Expanded implemented-file list to include:
    - `src/ours/data/step_builder.py`
    - `scripts/preprocess_steps.py`
    - `tests/unit/test_step_builder.py`
  - Added a new section explaining how to inspect and interpret preprocessing artifacts:
    - `sample_sequences.jsonl`
    - `flat_steps.jsonl`
    - `summary.json`
    - `manifest.json`
    - `errors.jsonl`
  - Added a novice digest section summarizing what changed in loader logic and why it matters.
  - Added a concise “recommended next commands” section for immediate execution.
- Files changed:
  - `readme.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-25 15:53:33 +08 (+0800)
- Type: Feature / Docs / Test
- Summary: Implemented step-level preprocessing pipeline (`step_builder.py` + `preprocess_steps.py`) with deterministic artifacts, resume controls, and beginner-focused documentation/comments.
- Details:
  - Added `src/ours/data/step_builder.py`:
    - Introduced typed dataclasses: `StepBuildConfig`, `ReasoningStep`, `StepSequence`.
    - Added deterministic sequence construction via `build_step_sequence` / `build_step_sequences`.
    - Added CoT splitting logic (`auto`, `newline`, `sentence`) with whitespace normalization and list-marker cleanup.
    - Added stable config signature and deterministic step IDs for reproducibility.
  - Added `scripts/preprocess_steps.py`:
    - Implemented batch preprocessing CLI from canonical samples to step artifacts.
    - Wrote reusable artifacts per run: `sample_sequences.jsonl`, `flat_steps.jsonl`, `errors.jsonl`, `summary.json`, `manifest.json`.
    - Added run fingerprinting + `--resume` / `--overwrite` behavior.
    - Added explicit dataset-specific argument handling (`gsm8k`, `bbh`, `hendrycks_math`).
  - Exported step-builder APIs in `src/ours/data/__init__.py`.
  - Added unit tests in `tests/unit/test_step_builder.py`.
  - Updated `readme.md` with preprocessing command examples and artifact explanations.
  - Validation results:
    - `python3 -m py_compile src/ours/data/step_builder.py scripts/preprocess_steps.py src/ours/data/__init__.py tests/unit/test_step_builder.py` passed.
    - `python3 -m pytest -q tests/unit/test_step_builder.py tests/unit/test_data_schema.py tests/unit/test_data_loader_helpers.py` -> `15 passed`.
    - `python3 scripts/preprocess_steps.py --datasets strategyqa --split train --limit 5 --batch-size 2` succeeded and produced artifacts.
    - Re-running same command correctly skipped due to matching artifacts (`--resume` default behavior).
- Files changed:
  - `src/ours/data/step_builder.py`
  - `scripts/preprocess_steps.py`
  - `src/ours/data/__init__.py`
  - `tests/unit/test_step_builder.py`
  - `readme.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-24 02:25:04 +08 (+0800)
- Type: Feature / Test
- Summary: Improved dataset loader logic for DROP/ProofWriter context handling and Hendrycks final-answer extraction; added targeted helper tests.
- Details:
  - Updated `load_drop` to build `question` from both passage and question (`Passage: ...\\nQuestion: ...`) so reading-comprehension context is not lost.
  - Updated `load_proofwriter` to build `question` from both theory and question (`Theory: ...\\nQuestion: ...`) and preserved additional metadata (`QDep`, `QLen`).
  - Updated `load_hendrycks_math` to:
    - store full solution in `cot`,
    - extract concise final answer into `answer` (prefer last `\\boxed{...}`),
    - record extraction strategy in metadata (`answer_extraction_method`).
  - Added helper functions with inline comments:
    - `_build_drop_question`
    - `_build_proofwriter_question`
    - `_extract_hendrycks_final_answer`
    - `_extract_last_boxed_content` (supports nested braces).
  - Added unit tests in `tests/unit/test_data_loader_helpers.py` for prompt builders and Hendrycks extraction logic.
  - Validation run results:
    - `python3 -m py_compile ...` passed
    - `python3 -m pytest -q tests/unit/test_data_loader_helpers.py tests/unit/test_data_schema.py` -> `9 passed`
    - `conda run -n bcr python scripts/check_data.py --datasets drop proofwriter hendrycks_math --split train --limit 2 --hendrycks-subset algebra` -> success
- Files changed:
  - `src/ours/data/loaders.py`
  - `tests/unit/test_data_loader_helpers.py`
  - `progress_detailed.md`
- Breaking changes:
  - [CAUTION] Canonical `question` for `drop` and `proofwriter` now includes context prefixes (`Passage:`/`Theory:`). Any downstream code that expected raw question-only text must adapt (use `metadata.raw_question` if needed).
  - [CAUTION] Canonical `hendrycks_math.answer` now holds extracted final answer (when possible) instead of full solution; full solution moved to `cot`.

---

## 2026-02-24 02:04:36 +08 (+0800)
- Type: Docs
- Summary: Finalized first-milestone documentation pass and refreshed root README with an actionable runbook.
- Details:
  - Updated `readme.md` from setup-only notes to a full novice-friendly project guide.
  - Added explicit quick-start commands (env setup, data checks, test command, pytest fix).
  - Added sections describing implemented pipeline files and common failure cases.
  - Added a clear "what to do next" section pointing to step-level preprocessing as next milestone.
- Files changed:
  - `readme.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-23 15:36:11 +08 (+0800)
- Type: Feature / Docs / Test
- Summary: Implemented first milestone data pipeline scaffold (`schema`, `loaders`, `check_data` CLI, and starter tests) for the independent OURS track.
- Details:
  - Added canonical schema with strong validation and dict coercion helpers.
  - Added multi-dataset loader module with a unified entry point and beginner-friendly error messages.
  - Added `scripts/check_data.py` to validate schema, preview samples, and verify dataset readiness before training.
  - Added starter test suite (`unit` + `integration`) including graceful skip behavior when parquet runtime is unavailable.
  - Ran verification:
    - `python3 -m py_compile ...` on all newly added files.
    - `python3 scripts/check_data.py --datasets strategyqa --split train --limit 2` (success).
    - `conda run -n bcr python scripts/check_data.py --datasets gsm8k --split train --limit 2 --gsm8k-config main` (success).
    - `pytest -q tests/unit/test_data_schema.py tests/integration/test_data_loaders.py` -> `4 passed, 1 skipped`.
- Files changed:
  - `src/ours/__init__.py`
  - `src/ours/data/__init__.py`
  - `src/ours/data/schema.py`
  - `src/ours/data/loaders.py`
  - `scripts/check_data.py`
  - `tests/conftest.py`
  - `tests/unit/test_data_schema.py`
  - `tests/integration/test_data_loaders.py`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-23 14:55:42 +08 (+0800)
- Type: Docs / Process
- Summary: Added persistent system rule for Markdown math rendering compatibility with Notion and Obsidian.
- Details:
  - Updated `chat_system_prompts.md` to require math delimiters: inline `$...$`, display `$$...$$`.
  - Explicitly disallowed legacy delimiters `\(...\)` and `\[...\]` for cross-editor rendering consistency.
  - Marked this requirement as an effective persistent system-level rule.
- Files changed:
  - `chat_system_prompts.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-23 14:38:47 +08 (+0800)
- Type: Docs / Process
- Summary: Added root `chat_system_prompts.md` and recorded system-level rules from the latest two user prompts.
- Details:
  - Created `chat_system_prompts.md` at repository root.
  - Summarized and persisted changelog-maintenance policy for `progress_detailed.md`.
  - Summarized and persisted check/refresh behavior rules for `chat_system_prompts.md`.
- Files changed:
  - `chat_system_prompts.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-23 14:24:49 +08 (+0800)
- Type: Setup / Process
- Summary: Created `progress_detailed.md` and initialized prepend-style changelog tracking.
- Details:
  - Added timestamp-first entry format.
  - Defined that newest changes must appear at the top.
  - Added breaking-change tagging rule: prefix impacted items with `[CAUTION]`.
- Files changed:
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## Log Entry Template
```md
## YYYY-MM-DD HH:MM:SS TZ (±HHMM)
- Type: <Feature | Fix | Refactor | Docs | Setup | Test | Chore>
- Summary: <one-line summary>
- Details:
  - <bullet 1>
  - <bullet 2>
- Files changed:
  - `<path>`
  - `<path>`
- Breaking changes:
  - None.
  - [CAUTION] <describe breaking change + migration note if applicable>
```
