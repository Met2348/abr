# Chat System Prompts (User-Controlled)

This file records persistent, system-level operating requirements provided by the user.
Changes to this file are user-authorized only.

## Effective Rules (Summarized from the last two user prompts)

1. Maintain `docs/progress_detailed.md` as a prepend-only changelog.
2. Whenever any code/file modification is made by the assistant, add a new changelog entry at the top of `docs/progress_detailed.md`. Except: if only docs are changed and codes unaffected, do not document.
3. Every changelog entry must begin with a timestamp.
4. Breaking changes must be explicitly marked with `[CAUTION]`.
5. If a turn is pure chat and no code/files are modified, no changelog entry is required.
6. Maintain `docs/chat_system_prompts.md` for system-level requirements that must be remembered across chats and code modifications.
7. The assistant may add new requirements to `docs/chat_system_prompts.md` only when explicitly instructed by the user.
8. The assistant is **not required** to check `docs/chat_system_prompts.md` at every chat start.
9. The assistant checks `docs/chat_system_prompts.md` only under explicit user instruction.
10. If `docs/chat_system_prompts.md` changes after the last check and no new explicit check instruction is given, the assistant should ignore those new changes and continue following the previously acknowledged requirements.
11. For Markdown documents containing math, expressions must be written in Notion/Obsidian-friendly delimiters: inline `$...$`, display `$$...$$`; avoid `\(...\)` and `\[...\]`.
12. New experiment settings should be organized into named parameter groups (for example `A1`, `A2`, ...), so previous experiments can be replicated without manual command copy-paste.
13. For each parameter group, include comments that explain:
    - why this group exists (intention),
    - what to observe in results (attention points),
    - expected outcomes.
14. New experiment setting execution should be provided through `.sh` scripts as one-click workflows; preferred usage is switching the configured param group and rerunning the same script.
15. When requested to diagnose experiment problems/fixes, update `result_records.md` with:
    - diagnosis,
    - summary of meaningful results,
    - explicit next-step plans.
16. For new implementations that may affect experiment speed, always evaluate batching-first design as the default approach.
17. When batching is introduced, always include safety nets (for example OOM backoff, fallback path, deterministic checks) so experiments remain robust.
18. Treat batching-specific correctness risks as first-class concerns; explicitly guard and test for batching-introduced bugs (for example padding side, ordering, decode parity).
19. README policy is dual-track:
    - `docs/readme.md` is the public, concise README.
    - `docs/readme_full.md` is the private, detailed operations logbook.
20. Documentation update scope:
    - major code changes must be reflected in both `docs/readme.md` and `docs/readme_full.md`,
    - regular/minor changes should be reflected in `docs/readme_full.md` only.
    - do not directly edit the `readme.md` at the root of the repo, since it is for academic reports, not for dev need.
21. All shell commands provided by the assistant for user reruns must be recorded in `docs/readme_full.md`.
22. Public README environment guidance must remain environment-agnostic and avoid private hardcoded setup assumptions.
23. Documentation style must be beginner-oriented by default:
    - every `py` and `sh` file should begin with a short abstract explaining why the file exists,
    - what responsibilities the file owns,
    - what major functions/classes it contains,
    - how control flows through the file,
    - and how it interacts with other files/modules.
24. Every function and class should have a complete docstring or nearby explanatory comment that covers:
    - purpose,
    - key inputs/outputs,
    - important edge cases or safety behavior,
    - and at least one short usage/example snippet when practical.
25. When modifying existing code, the assistant should raise documentation quality along with code quality instead of leaving new or edited logic under-documented.
26. Future code added by the assistant should follow this documentation style without needing repeated reminders.
27. When generating future experiment commands for the user:
    - prefer `CUDA_VISIBLE_DEVICES` values other than `0` by default because device `0` is usually crowded in the user's environment; if  a few commands come in a series, prefer to use device id: 1 -> 2 -> 3 -> 0 -> 1 -> 2 -> 3 -> ...
    - prefer eval batch sizes of `96` or larger when the task is inference/evaluation and there is no known memory-risk reason to stay smaller.
28. If a command intentionally uses a smaller eval batch size or device `0`, the assistant should explain the reason briefly.
29. All codes delivered through the chat should be run-ready, not missing params or requiring user to manually fill-in some blanks
30. every time, a modified code block or newly generated code file should be filled with Both English and Chinese comments (Bilingual), Chinese followed by English. you are not allowed to write explicit language indicators like "English:" or "中文:". 
31. If the assistant is required to search the web for paper consensus or community repo, the assistant should automatically summarize and write all searched results to docs, working as critical directions for the research.
## Acknowledgement Metadata

- Last explicit user instruction to refresh this file: `2026-03-10`
- Source scope: modified the comment and doc rules, manually by the user
