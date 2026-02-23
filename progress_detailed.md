# Progress Detailed Changelog

This file is prepend-only: newest entries must be added at the top (right below this header).

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
