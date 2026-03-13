# Phase E / Phase F Follow-Up Research Note

- generated_at: 2026-03-12 21:40:54 +0800
- scope: current blockers after the semantic-consensus and hybrid-controller passes

## 1. Questions re-checked this pass

1. How should `Phase E` respond to worst-generator / hard-case verifier failures?
2. Which `Phase F` RL stability tricks are actually relevant to our observed failures?
3. Is offline implicit PRM still worth promoting after the latest `pbr38f` failure?

## 2. Web-backed inputs used this pass

1. `TRL GRPOTrainer` docs
   - <https://huggingface.co/docs/trl/grpo_trainer>
2. `Open Instruct reward modeling`
   - <https://allenai.github.io/open-instruct/algorithms/reward_modeling/>
3. `Hard2Verify`
   - <https://arxiv.org/abs/2510.13744>
4. `When to Trust the Cheap Check`
   - <https://arxiv.org/abs/2602.17633>
5. `MathQ-Verify`
   - <https://arxiv.org/abs/2603.03307>
6. `Rewarding Progress` / Hugging Face post
   - <https://huggingface.co/blog/rewarding-progress>

## 3. What these sources change in practice

### 3.1 Phase E data / verifier design

Shared lesson:
1. verifier quality is increasingly bottlenecked by hard, ambiguous, adversarial, or domain-specialized cases;
2. benchmark-tuned verifier tracks are acceptable and often preferable to one universal scalar head;
3. cheap-to-strong routing remains the sane deployment shape.

Repo action:
1. keep `PBR44` semantic-consensus and `PBR45` L2-style clean-data variants as the immediate follow-up frontier;
2. auto-chain their `ProcessBench eval -> Phase F preflight` instead of stopping at pair accuracy.

### 3.2 Phase F RL infrastructure

Shared lesson:
1. reward variance collapse, truncation handling, and length bias are real first-order issues;
2. batch/global reward scaling plus replay-buffer support are more relevant than adding yet another naive RL recipe;
3. live RL should be tested on less saturated reasoning settings before being treated as a headline result.

Repo action:
1. keep the modern `Dr.GRPO` / replay-buffer canaries alive;
2. do not add more from-scratch RL-like controller work until these infra canaries read cleanly.

### 3.3 Offline implicit PRM

Local evidence now dominates here:
1. latest fixed-threshold `pbr38f` Math run produced `F1=0.0000`;
2. step scores are extremely low-dynamic-range and near-zero-centered:
   - step mean about `-0.0033`
   - step std about `0.0079`
   - example-mean max only about `0.0027`
3. this looks much more like a collapsed offline discriminative signal than a thresholding nuisance.

Repo action:
1. demote offline implicit PRM from “near-term promotion candidate” to “diagnostic-only side branch”;
2. if we revisit implicit rewards, do it in online / trainable-policy form rather than further post-hoc fixed-threshold evaluation.

## 4. Concrete actions launched from this pass

1. added follow-up wrapper:
   - `scripts/run_phase_e_candidate_phase_f_followup_suite.sh`
2. launched watcher:
   - `assets/artifacts/phase_e_logs/phase_e_pbr44_pbr45_followup_wait_0312_1750/watch.log`
3. purpose of the watcher:
   - wait for `PBR44` / `PBR45` completion,
   - evaluate both on `ProcessBench GSM` and `Math`,
   - run `Phase F modern preflight` automatically.

## 5. Current recommendation

1. put the next free serious GPU time behind `PBR44/PBR45` verification and promotion.
2. keep RL exploration focused on modern GRPO stability, not on broader algorithm proliferation.
3. keep the paper narrative centered on verifier-system redesign and controller operationalization, not on implicit-PRM optimism.
