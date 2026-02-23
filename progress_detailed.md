# Progress Detailed Changelog

This file is prepend-only: newest entries must be added at the top (right below this header).

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
