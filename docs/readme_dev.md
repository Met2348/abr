## [Dev] How to recover the Codex dev context at a new chat?

- Locate `docs/chat_system_prompts.md`, and ask Codex to read it so it can restore the requirements it should follow in each chat.

## [Dev] Where are the previous chats?

- The prompts since the very beginning of the project can be found at `docs/chat_prompts.txt`.

## [Dev] What do the two `TODO` files mean?

- The `docs/TODO.md` assumes the BCR codes are released so that we may build upon their codes.
- However, since no codes are released, we decide to build our own codebase from scratch.
- The `docs/TODO_ours.md` describes what we are going to do at each step, without the assistance of BCR.

## [Dev] What is the current strategic mainline?

- As of `2026-03-10`, do **not** assume StrategyQA is still the primary
  supervised benchmark for value-head training.
- The current repository mainline is:
  1. use PRM-grade supervised datasets to validate the ranking/value method,
  2. then transfer the validated method back to StrategyQA as downstream/OOD
     evidence.
- Source-of-truth docs:
  - `docs/readme_full.md`
  - `docs/phase_D_plan.md`
  - `docs/TODO_ours.md`

## [Dev] What do the two `idea` files mean?
- The `docs/idea_polish.md` contains ideas extracted from the original BCR idea and two following discussions.
- The `docs/idea_formulation.md` explicitly expresses the ideas with math formulas. The format is KaTeX-friendly, so it can be rendered in Obsidian and Notion.
