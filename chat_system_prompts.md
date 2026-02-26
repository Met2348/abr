# Chat System Prompts (User-Controlled)

This file records persistent, system-level operating requirements provided by the user.
Changes to this file are user-authorized only.

## Effective Rules (Summarized from the last two user prompts)

1. Maintain a root file named `progress_detailed.md` as a prepend-only changelog.
2. Whenever any code/file modification is made by the assistant, add a new changelog entry at the top of `progress_detailed.md`.
3. Every changelog entry must begin with a timestamp.
4. Breaking changes must be explicitly marked with `[CAUTION]`.
5. If a turn is pure chat and no code/files are modified, no changelog entry is required.
6. Maintain a root file named `chat_system_prompts.md` for system-level requirements that must be remembered across chats and code modifications.
7. The assistant may add new requirements to `chat_system_prompts.md` only when explicitly instructed by the user.
8. The assistant is **not required** to check `chat_system_prompts.md` at every chat start.
9. The assistant checks `chat_system_prompts.md` only under explicit user instruction.
10. If `chat_system_prompts.md` changes after the last check and no new explicit check instruction is given, the assistant should ignore those new changes and continue following the previously acknowledged requirements.
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

## Acknowledgement Metadata

- Last explicit user instruction to refresh this file: `2026-02-27`
- Source scope: includes math-rendering requirement, experiment-workflow requirements (param groups + one-click shell execution), experiment-diagnosis record maintenance in `result_records.md`, and batching-first implementation policy with safety/correctness guards.
