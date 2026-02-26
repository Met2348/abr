# Progress Detailed Changelog

This file is prepend-only: newest entries must be added at the top (right below this header).

## 2026-02-27 01:16:23 +08 (+0800)
- Type: Documentation Consolidation / Phase A Retrospective
- Summary: Performed a full Phase A retrospective pass and added consolidated conclusions, notices, and lessons-learned guidance across core docs to support clean handoff into Phase B.
- Details:
  - Updated `result_records.md`:
    - added section `0.13` with full-period lessons from Phase A:
      - evaluator-semantics impact,
      - coverage-aware reporting requirements,
      - prompt/protocol vs token-budget effects,
      - batching correctness guardrails,
      - truncation-pressure interpretation,
      - Phase-B carry-over operational rules.
  - Updated `phase_A_report.md`:
    - added retrospective lessons section,
    - added explicit pre-Phase-B commit checklist for reproducibility/artifact/freeze discipline.
  - Updated `readme.md`:
    - added top-level "Phase A Retrospective" guidance block for newcomers before Phase B.
  - Updated `TODO_ours.md`:
    - added `0.2 Phase A -> Phase B Handoff Gate (Commit-Time)` with concrete checklist items.
- Files changed:
  - `result_records.md`
  - `phase_A_report.md`
  - `readme.md`
  - `TODO_ours.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

## 2026-02-27 01:07:48 +08 (+0800)
- Type: Documentation Refresh / System-Prompt Update
- Summary: Refreshed project docs to reflect latest batching-fix results and added new batching-first implementation policy to system prompts.
- Details:
  - Updated `chat_system_prompts.md`:
    - added explicit batching-first requirement for new performance-sensitive implementations,
    - required safety nets for batching (fallback/backoff/determinism checks),
    - required explicit attention to batching-introduced bugs.
  - Updated `result_records.md`:
    - added new section `0.12` with validated post-fix batch results:
      - `batch_size=1`: `acc=0.5492`, `parse_err=0.1762`, `1.180 sample/s`,
      - `batch_size=4`: `acc=0.5596`, `parse_err=0.1658`, `4.189 sample/s`,
    - recorded interpretation and next verification checks.
  - Updated `readme.md`:
    - milestone summary now includes validated batching gain and parity,
    - batching hands-on notes now include right-padding warning and stable recommended settings.
  - Updated `phase_A_report.md`:
    - added batched-inference hardening notes and post-fix validation section.
  - Updated `foundation_reliability_audit.md`:
    - added high-severity finding `F9` (right-padding bug) with fixed status and mitigation details.
  - Updated `TODO_ours.md`:
    - engineering rules now include batching-first + batching-correctness guard requirements,
    - scaling section marks padding correctness fix completed in `S2`.
- Files changed:
  - `chat_system_prompts.md`
  - `result_records.md`
  - `readme.md`
  - `phase_A_report.md`
  - `foundation_reliability_audit.md`
  - `TODO_ours.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

## 2026-02-27 00:45:24 +08 (+0800)
- Type: Bug Fix / Batch Inference Correctness
- Summary: Fixed batched decoder-only generation bug caused by right-padding, which was degrading prediction quality and spamming warnings.
- Details:
  - Root cause:
    - batched free-form generation used tokenizer default padding side (right),
    - decoder-only models require left-padding for correct next-token generation context.
  - Code fix in `scripts/phase_a_generate_and_eval.py`:
    - `_generate_freeform_rows_once` now temporarily forces `tokenizer.padding_side = "left"` for batched tokenization,
    - restores original `padding_side` after tokenization to avoid side effects.
  - Regression test added in `tests/unit/test_phase_a_generate_script.py`:
    - verifies batched generation call uses left-padding and restores tokenizer state.
  - Validation:
    - targeted tests: `5 passed`,
    - full suite: `44 passed, 1 skipped`,
    - compile check passed for `scripts/phase_a_generate_and_eval.py`.
- Files changed:
  - `scripts/phase_a_generate_and_eval.py`
  - `tests/unit/test_phase_a_generate_script.py`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-27 00:34:36 +08 (+0800)
- Type: Throughput Upgrade / Batch Inference Implementation
- Summary: Implemented deterministic batched generation for Phase A runner, added OOM backoff, integrated batch controls into benchmark shell, expanded docs for novice usage, and added regression tests.
- Details:
  - Updated `scripts/phase_a_generate_and_eval.py`:
    - added CLI args:
      - `--batch-size` (default `1`),
      - `--oom-backoff` / `--no-oom-backoff`,
    - implemented batched free-form generation path in `_run_generation`:
      - keeps deterministic row order,
      - supports mixed routing (StrategyQA binary-choice + free-form in same batch),
      - preserves per-sample metadata (`generated_tokens`, `hit_token_limit`),
    - added automatic OOM backoff:
      - on OOM, recursively split batch and retry,
      - tracks `oom_backoff_events`,
    - added generation runtime stats object (`GenerationStats`) and persisted metrics fields:
      - `generation_stats.elapsed_seconds`,
      - `generation_stats.sample_per_second`,
      - `generation_stats.batch_size`,
      - `generation_stats.oom_backoff_events`,
    - added duplicate `sample_id` guard in prepared input loader and surfaced clear failure message,
    - added evaluator-version mismatch warning in metric comparison output.
  - Updated `scripts/run_phase_a_benchmark_suite.sh`:
    - added env-driven knobs:
      - `BATCH_SIZE` (default `1`),
      - `OOM_BACKOFF` (default `1`),
    - forwards these flags to `phase_a_generate_and_eval.py`,
    - prints and records batch settings in final suite summary.
  - Updated docs:
    - `readme.md`:
      - added batching + OOM-backoff flag documentation,
      - added `9.7 Batching Hands-On` with copy-paste commands and comparison method,
      - documented generation throughput outputs (`gen_elapsed_sec`, `gen_sample_rate`, `oom_backoff_evts`),
    - `TODO_ours.md`:
      - marked S2 batched-generation milestone as completed,
      - left bucketing as pending.
  - Added/updated tests:
    - `tests/unit/test_phase_a_generate_script.py`:
      - binary-choice metadata regression (branch-safe),
      - new free-form batch behavior test (order + metadata correctness),
      - duplicate sample-id guard test,
      - evaluator-version mismatch comparison payload test.
  - Validation:
    - targeted tests: `24 passed`,
    - full suite: `43 passed, 1 skipped`,
    - syntax checks: `bash -n scripts/run_phase_a_benchmark_suite.sh` and `python -m py_compile scripts/phase_a_generate_and_eval.py` passed.
- Files changed:
  - `scripts/phase_a_generate_and_eval.py`
  - `scripts/run_phase_a_benchmark_suite.sh`
  - `tests/unit/test_phase_a_generate_script.py`
  - `readme.md`
  - `TODO_ours.md`
  - `progress_detailed.md`
- Breaking changes:
  - None. Default behavior remains equivalent with `--batch-size 1`.

---

## 2026-02-26 23:59:32 +08 (+0800)
- Type: Reliability Hardening / Foundation Audit / Phase-B Readiness
- Summary: Performed a strict low-level reliability audit, fixed high-impact vulnerabilities in core evaluation/loading paths, and documented a formal hardening gate before large Phase B runs.
- Details:
  - Reliability fixes in core code:
    - `scripts/phase_a_generate_and_eval.py`:
      - fixed binary-choice branch metadata safety (removed branch-unsafe dependency on freeform token tensor),
      - added duplicate `sample_id` guard in `_load_prepared_rows` (fail-fast on corrupted prepared inputs),
      - added evaluator-version mismatch diagnostics in run comparisons (`evaluator_version_match`, caution message).
    - `src/ours/phase_a/contracts.py`:
      - relaxed `PredictionRecord.raw_prediction` and `ScoredPrediction.raw_prediction` validation to allow empty-string outputs (treated as parse-error cases instead of crashing the pipeline).
    - `src/ours/data/loaders.py`:
      - hardened `_normalize_split` to fail on unknown split typos while preserving known aliases (`val`, `valid`, `dev`).
    - `src/ours/phase_a/evaluator.py`:
      - added `EVALUATOR_VERSION` and persisted `evaluator_version` into summary/metrics payloads.
  - New and updated regression tests:
    - added `tests/unit/test_phase_a_generate_script.py`:
      - binary-choice generation metadata regression,
      - duplicate sample-id detection,
      - evaluator-version mismatch comparison payload.
    - updated `tests/unit/test_phase_a_extraction_eval.py`:
      - empty prediction now parsed/scored safely (no hard crash).
    - updated `tests/unit/test_data_loader_helpers.py`:
      - split alias normalization + typo rejection behavior.
  - Reliability documentation and planning:
    - added `foundation_reliability_audit.md`:
      - full risk scan, fixed findings, open findings, prioritized patch plan (P0/P1/P2),
      - explicit Phase-B entry gate criteria.
    - updated `TODO_ours.md` with section `18. Foundation Reliability Hardening Gate`.
    - updated `result_records.md` with section `0.11` summarizing hardening outcomes.
    - updated `readme.md` context-file index to include reliability audit doc.
  - Validation:
    - targeted tests passed: `27 passed`,
    - full suite passed: `42 passed, 1 skipped`.
- Files changed:
  - `scripts/phase_a_generate_and_eval.py`
  - `src/ours/phase_a/contracts.py`
  - `src/ours/data/loaders.py`
  - `src/ours/phase_a/evaluator.py`
  - `tests/unit/test_phase_a_generate_script.py`
  - `tests/unit/test_phase_a_extraction_eval.py`
  - `tests/unit/test_data_loader_helpers.py`
  - `foundation_reliability_audit.md`
  - `TODO_ours.md`
  - `result_records.md`
  - `readme.md`
  - `progress_detailed.md`
- Breaking changes:
  - [CAUTION] Metrics comparison now surfaces evaluator-version mismatch; historical delta interpretations should be re-checked when versions differ.
  - [CAUTION] Loader split typos now fail fast instead of silently falling back.

---

## 2026-02-26 23:44:42 +08 (+0800)
- Type: Evaluation Fix / GSM8K Math Extraction
- Summary: Fixed math equivalence undercount for CoT outputs by adding relaxed numeric handling for tagged answers that include units/currency text.
- Details:
  - Updated `src/ours/phase_a/answer_extraction.py`:
    - in math datasets (`gsm8k`, `hendrycks_math`), `answers_equivalent` now keeps strict numeric parse first, then applies relaxed numeric fallback when gold is numeric,
    - added helper functions:
      - `_normalize_numeric_text_relaxed`,
      - `_extract_first_numeric_token`,
    - math `final_answer_tag` extraction now normalizes numeric content from tagged text (for example `10 meters.` -> `10`, `$140.` -> `140`).
  - Updated `tests/unit/test_phase_a_extraction_eval.py`:
    - added extraction test for `Final answer: 10 meters.`,
    - added evaluator equivalence tests for currency/percent style outputs.
  - Re-evaluated existing prediction artifacts with the fixed evaluator:
    - `gsm8k_cot_t256_20260226T151603Z`: accuracy `0.2733 -> 0.7035`,
    - `gsm8k_math_direct_t16_20260226T152107Z`: accuracy `0.3663 -> 0.3721`.
  - Updated `result_records.md` with a new diagnosis section (`0.10`) documenting revised interpretation.
- Files changed:
  - `src/ours/phase_a/answer_extraction.py`
  - `tests/unit/test_phase_a_extraction_eval.py`
  - `result_records.md`
  - `progress_detailed.md`
- Breaking changes:
  - [CAUTION] GSM8K/Hendrycks historical metrics are not directly comparable across evaluator versions; re-score old prediction files with the new evaluator before comparing runs.

---

## 2026-02-26 23:32:00 +08 (+0800)
- Type: Planning / Performance / Scaling Strategy
- Summary: Added a detailed scaling and throughput boost plan to `TODO_ours.md`, including bottleneck diagnosis, phased acceleration roadmap, and recommended best-fit strategy for current project stage.
- Details:
  - Updated `TODO_ours.md` with new section `17. Scaling and Throughput Boost Plan (A100 Cluster)`:
    - recorded current runtime context and observed VRAM behavior (`4-15 GiB` per process on A100),
    - documented code-level and hardware-level bottlenecks,
    - specified recommended strategy for this project:
      - deterministic batching + bucketing,
      - one-process-per-GPU shard parallelism,
      - tighter stopping/token policies,
      - defer engine migration until later validation,
    - added phased roadmap (`S1`..`S4`) with expected gains,
    - added planned performance experiment groups and go/no-go criteria.
  - This plan explicitly targets CoT-heavy runtime bottlenecks while preserving reproducibility.
- Files changed:
  - `TODO_ours.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-26 23:11:37 +08 (+0800)
- Type: GSM8K Reliability Fix / Prompting / Evaluation Diagnostics
- Summary: Diagnosed the GSM8K smoke failure mode and implemented fixes to avoid weak truncated-number evaluation.
- Details:
  - Root-cause scan on `gsm8k_direct_smoke_20260226T150012Z` found:
    - `accuracy=0.0500`,
    - `parse_error_rate=0.0000`,
    - extraction methods were `last_number` for all samples (weak fallback over partial reasoning).
  - Added new prompt template in `src/ours/phase_a/prompt_builder.py`:
    - `qa_math_direct_final@1.0.0`
    - enforces one-line `Final answer: <number>` output for math direct baselines.
  - Updated `scripts/phase_a_prepare.py`:
    - template choices now derive dynamically from registry (auto-includes new templates).
  - Improved math extraction in `src/ours/phase_a/answer_extraction.py`:
    - supports both `Final answer: ...` and `Final answer is ...`.
  - Added math reliability diagnostics in `scripts/phase_a_generate_and_eval.py`:
    - prints and stores `math_diagnostics` with:
      - `last_number_rate`,
      - `hit_token_limit_rate`,
      - `final_answer_tag_rate`,
    - emits warnings when runs are dominated by weak capped `last_number` extraction.
    - prediction metadata now stores `generated_tokens` and `hit_token_limit`.
  - Updated tests:
    - `tests/unit/test_phase_a_prompt_builder.py` for new math template presence.
    - `tests/unit/test_phase_a_extraction_eval.py` for `final answer is` parsing.
  - Updated docs/records:
    - `readme.md` with GSM8K template recommendation and math diagnostics note.
    - `result_records.md` section `0.9` with before/after GSM8K smoke results.
  - Validation:
    - `python -m pytest -q tests/unit/test_phase_a_extraction_eval.py tests/unit/test_phase_a_prompt_builder.py` passed.
    - `python -m py_compile scripts/phase_a_prepare.py scripts/phase_a_generate_and_eval.py src/ours/phase_a/answer_extraction.py src/ours/phase_a/prompt_builder.py` passed.
    - `bash -n scripts/run_phase_a_benchmark_suite.sh` passed.
    - smoke recheck:
      - `gsm8k_math_direct_smoke_20260226T150913Z`: `accuracy=0.6000` (from `0.0500`).
      - `gsm8k_math_direct_smoke_t16_20260226T151029Z`: same smoke accuracy with faster runtime.
- Files changed:
  - `src/ours/phase_a/prompt_builder.py`
  - `scripts/phase_a_prepare.py`
  - `src/ours/phase_a/answer_extraction.py`
  - `scripts/phase_a_generate_and_eval.py`
  - `tests/unit/test_phase_a_prompt_builder.py`
  - `tests/unit/test_phase_a_extraction_eval.py`
  - `readme.md`
  - `result_records.md`
  - `progress_detailed.md`
- Breaking changes:
  - [CAUTION] Phase A math-baseline behavior changed; GSM8K comparisons should note template/extraction version.

---

## 2026-02-26 22:40:38 +08 (+0800)
- Type: Documentation / Milestone Closeout / Planning
- Summary: Closed Phase A in core docs, added a newcomer-facing `phase_A_report.md`, and re-planned the route toward final BCR/ABR goals.
- Details:
  - Updated `readme.md`:
    - refreshed top-level milestone status to reflect Phase A conclusion,
    - added explicit links to `result_records.md` and `phase_A_report.md`,
    - clarified that Phase B should start in separate scripts while keeping Phase A benchmarks frozen.
  - Updated `result_records.md`:
    - added Phase A closeout section (`0.8`) with:
      - concise conclusion,
      - achievements,
      - problems fixed,
      - open unsolved problems,
      - immediate next steps,
      - re-planned Stage B->E research route.
    - updated report timestamp.
  - Updated `TODO_ours.md`:
    - added `0.1 Phase A Closeout Status`,
    - replaced outdated immediate-action block with a re-planned execution route:
      - Stage B (BCR-lite build),
      - Stage C (faithfulness metrics completion),
      - Stage D (ABR heuristic),
      - Stage E (ABR-RL),
      - explicit go/no-go gates.
  - Added new document `phase_A_report.md`:
    - comprehensive handoff report for newcomers:
      - current experiment status,
      - key quantitative findings,
      - fixed vs open issues,
      - next-stage execution guidance.
- Files changed:
  - `readme.md`
  - `result_records.md`
  - `TODO_ours.md`
  - `phase_A_report.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-26 22:12:07 +08 (+0800)
- Type: Experiment Diagnosis / Records Update
- Summary: Recorded new A6 binary-choice results and finalized root-cause diagnosis for parse-error vs model-quality separation.
- Details:
  - Updated `result_records.md`:
    - added section `0.7` covering:
      - A6 result table (`direct_binchoice`, `cot_binchoice`, reproducibility repeat),
      - freeform-vs-binary interpretation,
      - explicit root-cause split (protocol/code vs model quality),
      - clarification that binary-choice mode is token-budget independent by design,
      - updated plan for dual-track reporting (decision-quality vs end-to-end format).
    - updated `Last updated` timestamp.
- Files changed:
  - `result_records.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-26 21:32:38 +08 (+0800)
- Type: Evaluation Reliability / Parsing Fix / Benchmark Controls
- Summary: Implemented targeted remedies for high parse-error rates and added a mode to separate formatting failures from model decision quality.
- Details:
  - Updated `src/ours/phase_a/answer_extraction.py`:
    - strengthened StrategyQA extraction with ordered rules:
      - exact/prefix yes-no parse,
      - explicit `Final answer:` / `Answer:` extraction,
      - last binary-token fallback,
    - added chat-leak truncation handling (e.g. `noHuman: ...`) to prevent false parse errors,
    - removed unsafe numeric (`1/0`) fallback from StrategyQA binary-token matching.
  - Updated `src/ours/phase_a/evaluator.py`:
    - added fairness/coverage diagnostics:
      - `n_parseable`,
      - `accuracy_parseable`,
      - dataset-level parseable accuracy.
  - Updated `scripts/phase_a_generate_and_eval.py`:
    - added `--strategyqa-decode-mode`:
      - `freeform` (default),
      - `binary_choice` (score `yes` vs `no` directly),
    - added `--truncate-chat-markers/--no-truncate-chat-markers`,
    - run output now prints `n_parseable` and `acc_parseable`.
  - Updated `scripts/phase_a_eval_predictions.py`:
    - now prints `n_parseable` and `acc_parseable`.
  - Updated `scripts/run_phase_a_benchmark_suite.sh`:
    - added decode controls:
      - `STRATEGYQA_DECODE_MODE`,
      - `TRUNCATE_CHAT_MARKERS`,
    - added new param group `A6` (Binary-Choice Decode Validation),
    - included decode controls in final suite summary metadata.
  - Updated `tests/unit/test_phase_a_extraction_eval.py`:
    - added tests for leakage-prefix parsing, final-answer tag parsing, and last-binary-token behavior.
  - Updated `readme.md`:
    - documented new decode options and `A6` group.
  - Updated `result_records.md`:
    - added parser/decode remedy diagnosis update with old-vs-new metric comparison.
  - Validation:
    - `python -m pytest -q tests/unit/test_phase_a_extraction_eval.py tests/unit/test_phase_a_prompt_builder.py tests/unit/test_phase_a_splitting.py` passed.
    - `bash -n scripts/run_phase_a_benchmark_suite.sh` passed.
    - `python -m py_compile scripts/phase_a_generate_and_eval.py scripts/phase_a_eval_predictions.py src/ours/phase_a/answer_extraction.py src/ours/phase_a/evaluator.py` passed.
- Files changed:
  - `src/ours/phase_a/answer_extraction.py`
  - `src/ours/phase_a/evaluator.py`
  - `scripts/phase_a_generate_and_eval.py`
  - `scripts/phase_a_eval_predictions.py`
  - `scripts/run_phase_a_benchmark_suite.sh`
  - `tests/unit/test_phase_a_extraction_eval.py`
  - `readme.md`
  - `result_records.md`
  - `progress_detailed.md`
- Breaking changes:
  - [CAUTION] StrategyQA extraction behavior changed; historical metrics may shift when re-evaluated with the updated parser.

---

## 2026-02-26 21:17:39 +08 (+0800)
- Type: Experiment Diagnosis / Docs / Process
- Summary: Added a new consolidated diagnosis update for CoT sweeper2 and A5 strict-fix runs; recorded updated conclusions and next plans for teammates.
- Details:
  - Updated `result_records.md`:
    - added latest diagnosis block with problem-focus, result tables, and fix verdicts,
    - recorded CoT sweep (`t128..t384`) and A5 strict runs in one place,
    - added explicit, actionable next-step plan for follow-up experiments.
  - Updated `chat_system_prompts.md`:
    - appended persistent rule to update `result_records.md` with diagnosis/summary/plan when requested for experiment-fix analysis.
- Files changed:
  - `result_records.md`
  - `chat_system_prompts.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-26 21:06:58 +08 (+0800)
- Type: UX / Logging / Runtime Output
- Summary: Reduced run-log spam by suppressing model-load progress bars and capping generation progress lines per run.
- Details:
  - Updated `scripts/phase_a_generate_and_eval.py`:
    - suppressed transformers weight-loading progress bars (removes duplicated/garbled `Loading weights` lines),
    - added concise model-load messages:
      - `model_load : start`
      - `model_load : done in HH:MM:SS`,
    - reduced `hf_device_map` print verbosity to device-count summary instead of full large mapping,
    - added `--max-progress-lines` (default `5`) to cap generation progress output lines,
    - adjusted progress logging logic to downsample checkpoints while preserving completion visibility.
  - Updated `scripts/run_phase_a_benchmark_suite.sh`:
    - added `MAX_PROGRESS_LINES` env control (default `5`),
    - passes `--max-progress-lines` to each inference run,
    - includes this value in startup logs and final summary metadata.
  - Updated `readme.md`:
    - documented new progress-cap behavior and model-load logging style,
    - documented `MAX_PROGRESS_LINES` override.
  - Validation:
    - `python -m py_compile scripts/phase_a_generate_and_eval.py` passed.
    - `bash -n scripts/run_phase_a_benchmark_suite.sh` passed.
- Files changed:
  - `scripts/phase_a_generate_and_eval.py`
  - `scripts/run_phase_a_benchmark_suite.sh`
  - `readme.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-26 20:51:29 +08 (+0800)
- Type: Feature / UX / Reporting
- Summary: Added end-of-suite consolidated experiment summary output (group + full settings + run table) to avoid scrolling through long logs.
- Details:
  - Updated `scripts/run_phase_a_benchmark_suite.sh`:
    - summary now includes, at run end:
      - experiment group id/title,
      - intention/observe/expectation text,
      - detailed settings (dataset/split/limit/seed/dtype/log cadence/CUDA/model),
      - prepared input paths,
      - planned run specs,
      - result table with metrics and deltas,
      - quick best-metric highlights.
    - summary is printed to terminal at the end and persisted to:
      - `assets/artifacts/phase_a_logs/<RUN_PREFIX>/final_summary.md`
    - script startup now prints the summary-file path.
  - Updated `readme.md` section `9.6`:
    - documented final summary print and persisted path.
  - Validation:
    - `bash -n scripts/run_phase_a_benchmark_suite.sh` passed.
- Files changed:
  - `scripts/run_phase_a_benchmark_suite.sh`
  - `readme.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-26 20:37:06 +08 (+0800)
- Type: Workflow / UX
- Summary: Removed hard concurrency blocking from the param-group suite runner to reduce teammate confusion, replaced with non-blocking warnings.
- Details:
  - Updated `scripts/run_phase_a_benchmark_suite.sh`:
    - removed fail-fast concurrency guard,
    - added `warn_if_concurrent_generation()` that logs warnings but does not stop execution.
  - Updated `readme.md` notes for section `9.6`:
    - removed outdated `ALLOW_CONCURRENT` guidance,
    - documented current behavior (warning + continue).
  - Validation:
    - `bash -n scripts/run_phase_a_benchmark_suite.sh` passed.
- Files changed:
  - `scripts/run_phase_a_benchmark_suite.sh`
  - `readme.md`
  - `progress_detailed.md`
- Breaking changes:
  - [CAUTION] Default behavior changed: the suite now continues under concurrency instead of stopping, which can affect throughput/comparison fairness if multiple runs share GPUs.

---

## 2026-02-26 19:14:34 +08 (+0800)
- Type: Feature / Experiment Design / Docs
- Summary: Added a new strict-binary experiment param block (`A5`) and recorded completed direct-token sweep results with analysis in the experiment logger.
- Details:
  - Added new prompt template in `src/ours/phase_a/prompt_builder.py`:
    - `qa_binary_strict@1.0.0`
    - enforces one-word yes/no style and discourages default-label behavior.
  - Updated `scripts/phase_a_prepare.py` template choices:
    - now supports `qa_binary_strict`.
  - Extended param-group runner `scripts/run_phase_a_benchmark_suite.sh`:
    - added `A5` group targeting current issues:
      - format compliance,
      - token waste from long outputs,
      - reproducibility check under strict binary template.
    - added strict-input preparation and routing path.
    - updated supported group list to `A1..A5`.
  - Updated `readme.md` group summary:
    - documented new `A5` purpose.
  - Updated experiment logger `result_records.md`:
    - added completed direct-token sweep table (`t16, t24, t32, t48, t64`),
    - added analysis of efficiency/compliance/bias observations,
    - added executable command and evaluation focus for new `A5` block.
  - Validation:
    - `python -m py_compile scripts/phase_a_prepare.py scripts/phase_a_generate_and_eval.py src/ours/phase_a/prompt_builder.py` passed.
    - `bash -n scripts/run_phase_a_benchmark_suite.sh` passed.
    - `python -m pytest -q tests/unit/test_phase_a_prompt_builder.py` passed (`4 passed`).
- Files changed:
  - `src/ours/phase_a/prompt_builder.py`
  - `scripts/phase_a_prepare.py`
  - `scripts/run_phase_a_benchmark_suite.sh`
  - `readme.md`
  - `result_records.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-26 18:48:56 +08 (+0800)
- Type: Feature / Reliability / Docs
- Summary: Implemented persisted live logging for both single-run inference and param-group sweep runner.
- Details:
  - Updated `scripts/phase_a_generate_and_eval.py`:
    - added `--persist-console-log/--no-persist-console-log` (default enabled),
    - console output is now mirrored to `<run_dir>/console.log` while still printing live to terminal,
    - manifest now records console log persistence status and `console_log` path.
  - Updated `scripts/run_phase_a_benchmark_suite.sh`:
    - added suite-level persistent log capture via `tee`,
    - default suite log path: `assets/artifacts/phase_a_logs/<RUN_PREFIX>/suite.log`,
    - can disable with `ENABLE_PERSISTED_LOGS=0`.
  - Updated `readme.md`:
    - documented new per-run `console.log`,
    - documented suite log file location and disable toggle.
  - Validation:
    - `python -m py_compile scripts/phase_a_generate_and_eval.py` passed.
    - `bash -n scripts/run_phase_a_benchmark_suite.sh` passed.
- Files changed:
  - `scripts/phase_a_generate_and_eval.py`
  - `scripts/run_phase_a_benchmark_suite.sh`
  - `readme.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

## 2026-02-26 18:43:20 +08 (+0800)
- Type: Docs / Process
- Summary: Refreshed `chat_system_prompts.md` with new persistent experiment-workflow requirements (param groups + one-click shell execution).
- Details:
  - Updated `chat_system_prompts.md` by adding new effective rules:
    - organize new experiment settings into named parameter groups (`A1`, `A2`, ...),
    - include per-group comments for intention, observation focus, and expected outcomes,
    - provide new experiment execution via `.sh` one-click workflows (switch group and rerun script).
  - Updated acknowledgement metadata:
    - last explicit refresh instruction date set to `2026-02-26`,
    - source scope expanded to include experiment-workflow requirements.
- Files changed:
  - `chat_system_prompts.md`
  - `progress_detailed.md`
- Breaking changes:
  - None.

---

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
