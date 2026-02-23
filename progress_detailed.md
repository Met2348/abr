# Progress Detailed Changelog

This file is prepend-only: newest entries must be added at the top (right below this header).

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
