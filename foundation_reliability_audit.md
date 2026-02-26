# Foundation Reliability Audit (Pre-Phase-B Gate)

Last updated: 2026-02-27 01:07 +0800

## 1. Why This Audit Exists

Before Phase B (training/value learning), we must trust low-level components.
If core utilities are brittle, higher-level experiments can be misleading even when code "runs".

This audit focuses on components that are:
1. heavily reused,
2. easy to overlook,
3. high impact on experiment correctness.

---

## 2. Audit Scope

Reviewed modules:
1. Data schema + loaders:
   - `src/ours/data/schema.py`
   - `src/ours/data/loaders.py`
2. Step preprocessing:
   - `src/ours/data/step_builder.py`
   - `scripts/preprocess_steps.py`
3. Phase A core contracts + evaluation:
   - `src/ours/phase_a/contracts.py`
   - `src/ours/phase_a/answer_extraction.py`
   - `src/ours/phase_a/evaluator.py`
4. Run orchestration and comparisons:
   - `scripts/phase_a_prepare.py`
   - `scripts/phase_a_generate_and_eval.py`
   - `scripts/run_phase_a_benchmark_suite.sh`

Validation method:
1. static code scan (invariants, failure paths, silent fallback behavior),
2. targeted regression tests for critical branches,
3. full test suite run after fixes.

Current test status after hardening:
1. `44 passed, 1 skipped`.

---

## 3. Critical Findings and Fix Status

## F1 (High): Binary-choice generation branch had unsafe token-metadata dependency

Risk:
1. StrategyQA `binary_choice` path could rely on variables only created in freeform generation branch.
2. This can crash or produce undefined metadata behavior.

Fix:
1. Added explicit branch-safe metadata fields:
   - `generated_tokens`,
   - `hit_token_limit`,
   with deterministic defaults for binary-choice mode.
2. Added regression test for binary-choice path:
   - `tests/unit/test_phase_a_generate_script.py`.

Status: Fixed.

---

## F2 (High): Empty raw predictions were rejected by contracts

Risk:
1. Evaluator pipeline aborted when model output became empty string.
2. This turns a normal parse-error case into a hard run failure.

Fix:
1. Relaxed contract validation:
   - `PredictionRecord.raw_prediction` and `ScoredPrediction.raw_prediction` now require `str`, not non-empty.
2. Added regression test:
   - empty prediction is scored as parse error, not pipeline crash.

Status: Fixed.

---

## F3 (High): Split normalization could silently map typos to fallback split

Risk:
1. Typo like `trian` could silently fall back (for example to `test`), contaminating experiments.

Fix:
1. `_normalize_split` now:
   - accepts known aliases (`val`, `valid`, `dev`),
   - fails fast on unknown split tokens.
2. Added tests for alias acceptance and typo rejection.

Status: Fixed.

---

## F4 (High): Cross-version metric comparison had no evaluator-version guard

Risk:
1. We changed extraction/equivalence logic over time.
2. Comparing metrics across scoring versions can produce false "model improvements/regressions".

Fix:
1. Added `evaluator_version` into evaluation summary/metrics.
2. Added comparison diagnostics in generate script:
   - `evaluator_version_match`,
   - explicit caution message when versions differ.
3. Added regression test for mismatch warning payload.

Status: Fixed.

---

## F5 (Medium): Prepared input duplicate `sample_id` was not guarded

Risk:
1. Duplicate sample IDs can bias metrics and break run-to-run comparison logic.

Fix:
1. Added duplicate `sample_id` detection in `_load_prepared_rows`.
2. Script now fails fast with explicit message.
3. Added regression test.

Status: Fixed.

---

## F6 (Medium, Open): Relaxed numeric parsing can over-credit ambiguous answers

Context:
1. We intentionally added relaxed numeric parsing for GSM8K/Hendrycks to avoid false negatives (`$140`, `10 meters`, `75% ...`).
2. This can still over-credit some ambiguous strings.

Plan:
1. Build adversarial fixture set (at least 50 examples) with:
   - should-pass cases,
   - should-fail lookalikes.
2. Add a strict-vs-relaxed scorer report on the same predictions.
3. Gate adoption with fixture pass threshold.

Status: Planned (P1).

---

## F7 (Medium, Open): Script-level contract/schema validation is still shallow

Risk:
1. JSONL files are validated row-by-row but schema-version compatibility is not strongly enforced.
2. Future field changes can silently degrade downstream behavior.

Plan:
1. Add a shared artifact schema validator module.
2. Require `schema_version` compatibility checks before consuming:
   - prepared inputs,
   - predictions,
   - scored outputs.
3. Fail fast with actionable migration errors.

Status: Planned (P1).

---

## F8 (Medium, Open): Low-level reliability checks are not yet first-class CI gates

Risk:
1. We now have key tests, but no explicit "foundation gate" grouping before Phase B runs.

Plan:
1. Add `scripts/check_foundation_reliability.sh`:
   - run critical test subset,
   - verify evaluator version presence in fresh metrics,
   - verify no duplicate IDs in prepared artifacts.
2. Require this gate before Phase B training scripts.

Status: Planned (P1).

---

## F9 (High): Batched decoder-only generation used right-padding (quality regression risk)

Risk:
1. Decoder-only models must use left-padding in batched generation.
2. Right-padding in batch path can corrupt generation behavior and silently degrade quality.
3. Symptom was severe parse-error increase and repeated runtime warnings under `batch_size>1`.

Fix:
1. Forced temporary `padding_side='left'` inside batched freeform tokenization.
2. Restored tokenizer original padding side after call to avoid side effects.
3. Added regression test for left-padding usage + state restoration.
4. Re-validated StrategyQA quality parity with strong throughput gain (`batch_size=4`).

Status: Fixed.

---

## 4. Patch Plan (Prioritized)

### P0 (Completed now)
1. Binary-choice metadata safety.
2. Empty prediction contract fix.
3. Split typo fail-fast.
4. Evaluator-version comparison caution.
5. Duplicate sample-id guard.
6. Regression tests for all above.

### P1 (Must complete before large Phase B sweeps)
1. Add adversarial numeric-equivalence fixture pack.
2. Add artifact schema-version validator.
3. Add foundation reliability gate script + checklist.

### P2 (During Phase B/C)
1. Expand dataset-specific extractor fixtures (DROP/ProofWriter/BBH).
2. Add run-level drift dashboard:
   - evaluator version,
   - template version,
   - decode mode,
   - prompt fingerprint.

---

## 5. Phase-B Entry Criteria (Reliability Gate)

Phase B should proceed only if all are true:
1. Foundation tests pass (`42+` current suite plus new P1 gates).
2. Evaluator-version mismatch is surfaced in comparisons.
3. Prepared artifacts pass duplicate-ID and schema checks.
4. Numeric-equivalence fixture suite reaches agreed threshold.

If any item fails, fix reliability first, then resume model training work.
