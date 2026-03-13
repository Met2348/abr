# Phase E / Phase F Web Research Refresh

- generated_at: 2026-03-12 17:33:52 +0800
- scope: current verifier-system and RL best-practice refresh after the hybrid-controller pass

## 1. Source sanity fixes

The repo had one citation conflict that matters for writing and planning:

1. `PRIME` is overloaded in current notes.
2. `PRIME-RL / implicit rewards` is the RL method paper:
   - <https://arxiv.org/abs/2502.01456>
3. `PRIME: A Process-Outcome Alignment Benchmark for Verifiable Reasoning in Mathematics and Engineering` is the verifier benchmark paper:
   - <https://arxiv.org/abs/2602.11570>
4. one newer plan note had incorrectly pointed `PRIME` to `2502.05795`, which is actually `The Curse of Depth in LLMs`:
   - <https://arxiv.org/abs/2502.05795>

Implication:
- when we discuss `Phase E` benchmark/verification redesign, we should mean the `2602.11570` benchmark paper;
- when we discuss `Phase F` implicit online rewards, we should mean `2502.01456`.

## 2. Fresh web-checked takeaways

### 2.1 RL infrastructure

Sources:
1. `TRL GRPOTrainer` docs: <https://huggingface.co/docs/trl/grpo_trainer>
2. `DAPO`: <https://arxiv.org/abs/2503.14476>
3. `Understanding R1-Zero-Like Training`: <https://arxiv.org/abs/2503.20783>
4. `Open Instruct reward modeling`: <https://allenai.github.io/open-instruct/algorithms/reward_modeling/>

What changed relative to older assumptions:

1. modern TRL already exposes the knobs we actually need for reasoning RL canaries:
   - `loss_type`
   - reward scaling (`group` / `batch` / `none`)
   - `mask_truncated_completions`
   - `importance_sampling_level`
   - `beta`
2. TRL now also ships an experimental `GRPOWithReplayBufferTrainer`, explicitly aimed at preserving useful off-policy samples when reward groups degenerate or variance collapses.
3. recent RL papers continue to treat length bias and unstable reward normalization as first-order issues, not minor tuning details.
4. this matches our repo evidence closely: current live RL problems look more like optimization / saturation / variance collapse than “there is no reward signal”.

Practical conclusion for `Phase F`:

1. keep explicit modern GRPO knobs on the CLI;
2. use `clip_delta`-style process shaping plus truncation masking by default;
3. test replay-buffer GRPO as infrastructure, because it targets one of our most plausible real failure modes;
4. do not claim RL is helping until it beats `BC` on a slice that is not already benchmark-saturated.

### 2.2 Verifier-system design

Sources:
1. `VerifyBench`: <https://arxiv.org/abs/2507.09884>
2. `Hard2Verify`: <https://arxiv.org/abs/2510.13744>
3. `When to Trust the Cheap Check`: <https://arxiv.org/abs/2602.17633>
4. `PRIME benchmark`: <https://arxiv.org/abs/2602.11570>

Shared message:

1. weak/cheap verifiers can still be valuable, but only inside a routing system that knows when to escalate.
2. fixed-threshold leaderboard numbers are not enough; robustness under adversarial or generator-shifted errors matters.
3. verifier quality that better matches process-outcome alignment is predictive of downstream RLVR usefulness.

Practical conclusion for `Phase E`:

1. stop treating one scalar head as the final object;
2. promote a `verifier system` design:
   - cheap discriminative head for broad filtering,
   - stronger hybrid / ensemble verifier for escalation,
   - deterministic or benchmark-specific checks where available,
   - controller policy deciding continue / verify / abstain / escalate.
3. data curation should bias toward disagreement buckets, hard negatives, and generator-shift slices rather than just adding more mixed-source pairs.

### 2.3 Process supervision direction

Sources:
1. `GenPRM`: <https://arxiv.org/abs/2504.00891>
2. `ThinkPRM`: <https://arxiv.org/abs/2504.16828>
3. `VPRM`: <https://arxiv.org/abs/2601.17223>
4. `PRIME-RL implicit rewards`: <https://arxiv.org/abs/2502.01456>

Shared message:

1. process supervision does not have to be one discriminative scalar head.
2. generative or structure-aware critics can capture reasoning defects that scalar token pools miss.
3. implicit or online rewards are viable, but they require a trainable policy/backbone and more careful optimization discipline than our older frozen-head setup assumed.

Practical conclusion for this repo:

1. `Phase E` should keep the current hybrid / ensemble path as the mainline.
2. `Phase F` should treat implicit-reward and replay-buffer RL as LoRA-aware follow-up infrastructure, not as a replacement for verifier curation.
3. for paper writing, the strongest narrative is now:
   - `A/B/C/D`: diagnose and stabilize value learning,
   - `E`: redesign the verifier into a system with domain-specific strengths,
   - `F`: use controller / BC / selective RL to operationalize that verifier system.

## 3. Concrete repo actions taken from this refresh

1. `scripts/download_related_papers.py`
   - extended to cover `proceedings.mlr.press` landing pages and parse real PDF links from page metadata.
2. `docs/relatedPapers/`
   - re-synced until all currently referenced paper-like URLs in `docs/**/*.md` are covered by `index.json`.
3. `scripts/phase_f_grpo_lite.py`
   - extended with optional replay-buffer GRPO support.
4. new queue:
   - `assets/artifacts/phase_f_logs/phase_f_replay_grpo_canary_wait_0312_1732/watch.log`

## 4. Updated next experiments

1. Read the standard `Dr.GRPO` canary first.
2. Compare it against the replay-buffer canary once GPU0 frees and the run finishes.
3. If replay buffer reduces collapse or improves stability diagnostics, promote it to the default `Phase F` live-RL wrapper.
4. Keep the main `Phase E` frontier on hybrid / ensemble verifier improvement, especially disagreement-focused data curate and post-preflight candidate narrowing.
