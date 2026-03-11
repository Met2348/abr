# Phase E Paradigm Refresh And Targeted Repair Experiments (2026-03-11)

This note has two jobs:

1. explain the newer `2025H2 -> 2026-03` verifier / PRM / RL-facing paradigm in plain language,
2. record one focused repair pass around the current strongest `Math-PRM-7B` frozen-head line.

The ABR core idea is unchanged:

- use local process discrimination instead of only final-answer reward,
- keep Bellman-style / process-faithfulness intuition,
- prefer signals that can survive deployment pressure.

What changes here is the surrounding system design.

## 1. Newer Community Paradigm, Explained First

### 1.1 Verifier

A `verifier` is a model that judges whether a solution, a step, or a final answer should be trusted.

Older repo framing often treated the value head as "the verifier".
Newer community framing treats verification as a first-class problem with its own model families,
data contracts, and benchmarks.

Useful references:

- `CompassVerifier`: https://arxiv.org/abs/2508.03686
- `Hard2Verify`: https://arxiv.org/abs/2510.13744

### 1.2 Outcome Verifier / Answer Verifier

An `outcome verifier` only checks whether the final answer is correct.

Example:

- question: solve a math problem
- model reasoning: many steps
- outcome verifier: only checks whether the final boxed answer is equivalent to the gold answer

Why this matters:

- process supervision catches local reasoning errors,
- outcome supervision protects against a common failure mode in this repo:
  correct full solutions being scored too low at the terminal prefix.

### 1.3 PRM

`PRM` means `Process Reward Model`.

It scores intermediate reasoning prefixes or steps, not just the final answer.
In this repo's language, many current Phase E checkpoints are really lightweight PRM-style verifiers.

Useful references:

- TRL `PRMTrainer`: https://huggingface.co/docs/trl/prm_trainer
- `ThinkPRM`: https://arxiv.org/abs/2504.16828

### 1.4 False Negative

A `false negative` means the verifier says "bad" even though the reasoning or answer is actually correct.

This is especially dangerous for RL:

- the policy may stop exploring correct but unusual reasoning,
- the verifier becomes conservative in the wrong direction,
- optimization pressure then pushes the generator toward verifier-friendly but not truly better behavior.

Useful reference:

- `Online Learnability of Chain-of-Thought Verifiers`: https://arxiv.org/abs/2603.03538

### 1.5 Soundness vs Completeness

These two words are easy to confuse.

- `soundness`: when the verifier accepts something, it is usually truly good
- `completeness`: when something is truly good, the verifier usually accepts it

This repo's current stubborn problem is more a `completeness` problem than a `soundness` problem:

- local bad prefixes are often detected reasonably well,
- but many fully correct traces are still undervalued at the terminal prefix.

### 1.6 Factorized Supervision

`Factorized supervision` means we do not force one scalar to represent every kind of goodness.

Instead, we separate at least some of:

- local step validity
- progress toward solution
- terminal completion correctness
- uncertainty / abstain probability

This is a major shift in the community.
The point is not "more heads because it sounds fancy".
The point is to stop one target from silently fighting another.

Useful references:

- `Reward Reasoning Model`: https://arxiv.org/abs/2505.14674
- `ActPRM`: https://arxiv.org/abs/2504.10559

### 1.7 Teacher-Student Distillation

A `teacher` is a stronger, more expensive verifier or judge.
A `student` is the cheap model actually deployed or used in the loop.

Newer practice often uses:

- expensive teacher for hard slices
- smaller student for bulk scoring

This matters here because the repo's current value head should be viewed as a bounded-support student, not as the whole final verifier stack.

## 2. What The New Paradigm Implies For This Repo

The old risky assumption was:

- if one frozen scalar head gets high pair accuracy, it is close to RL-ready.

The newer and more defensible assumption is:

- a scalar value head can be useful,
- but RL readiness requires:
  - strong local first-bad discrimination,
  - acceptable completeness on all-correct completions,
  - stable behavior under distribution shift,
  - ideally an outcome-verifier backstop.

So the right question is no longer:

- "can one scalar head rank pairs well?"

The right question is:

- "what exact failure mode remains after local ranking is already strong?"

The answer from this repo remains:

- terminal completion ordering.

## 3. Experiment Matrix For This Pass

We used the strongest current baseline family:

- baseline value run:
  - `assets/artifacts/phase_e_runs/phase_e_pbr10_prm7b_dpo8k_s42_value_20260311T110527Z`
- baseline same-family eval:
  - `assets/artifacts/phase_e_samefamily_eval/pbr10_s42_samefamily_fix0311_20260311T114318Z`
- baseline benchmark evals:
  - `assets/artifacts/phase_e_eval/pbr10_dpo8k_gsm_fulleval_20260311T112402Z`
  - `assets/artifacts/phase_e_eval/pbr10_dpo8k_math_fulleval_20260311T112402Z`

### 3.1 Baseline `PBR10`

Purpose:

- current best balanced `Math-PRM-7B` frozen-head line
- strong local ranking, weak terminal completion ordering

### 3.2 Existing Repo Contrast `PBR16`

Purpose:

- stress a more local-heavy / reduced-DPO / termBCE-style recipe
- test whether stronger local geometry alone solves the transfer issue

Relevant artifacts:

- value run:
  - `assets/artifacts/phase_e_runs/phase_e_pbr16_reduced_dpo_termBCE_s42_value_20260311T121704Z`
- benchmark evals:
  - `assets/artifacts/phase_e_eval/pbr16_reduced_dpo_gsm_fulleval_20260311T122504Z`
  - `assets/artifacts/phase_e_eval/pbr16_reduced_dpo_math_fulleval_20260311T122505Z`

### 3.3 New `PRX1_TERMMIX_MLP`

Goal:

- keep the successful `PBR10` local geometry,
- double the same-family terminal anchor mass,
- do not change backbone or head family.

Curated data artifact:

- `assets/artifacts/phase_e_pairs/phase_e_paradigm_refresh_pbr10core_term1024__5f129586db22`

Key mix:

- `4679 sibling_branch`
- `1783 local_first_bad_edge`
- `1024 terminal_completion_anchor`

Run + eval artifacts:

- value run:
  - `assets/artifacts/phase_e_runs/phase_e_prx1_pbr10core_term1024_mlp_0311_20260311T123100Z`
- same-family:
  - `assets/artifacts/phase_e_samefamily_eval/phase_e_prx1_samefamily_0311_20260311T130525Z`
- benchmarks:
  - `assets/artifacts/phase_e_eval/phase_e_prx1_gsm_fulleval_0311_20260311T130544Z`
  - `assets/artifacts/phase_e_eval/phase_e_prx1_math_fulleval_0311_20260311T130924Z`

### 3.4 New `PRX2_TERMMIX_DUAL`

Goal:

- same data as `PRX1`,
- change only the head family,
- test whether explicit local/terminal factorization fixes the terminal problem.

Run + eval artifacts:

- value run:
  - `assets/artifacts/phase_e_runs/phase_e_prx2_pbr10core_term1024_dual_0311_20260311T130045Z`
- same-family:
  - `assets/artifacts/phase_e_samefamily_eval/phase_e_prx2_samefamily_0311_20260311T132959Z`
- benchmarks:
  - `assets/artifacts/phase_e_eval/phase_e_prx2_gsm_fulleval_0311_20260311T133017Z`
  - `assets/artifacts/phase_e_eval/phase_e_prx2_math_fulleval_0311_20260311T133033Z`

### 3.5 LoRA Direction

We also attempted a backbone-adaptation restart, but the first warm-start try immediately exposed a config mismatch:

- the loaded checkpoint used `dropout_prob=0.1`
- the new run defaulted to `dropout_prob=0.0`

Because the data and architecture experiments above already gave a decisive tradeoff picture, and the shared machine was heavily contended, this pass did not spend more GPU time on a low-signal LoRA rerun.

Interpretation:

- representation adaptation still matters,
- but the stronger near-term bottleneck remains objective / data geometry, not just frozen capacity.

## 4. Results

### 4.1 Compact Comparison

| config | heldout_pair_acc | samefamily_top1 | samefamily_local | gsm_auc | math_auc | gsm_terminal_top1 | math_terminal_top1 | gsm_f1 | math_f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `PBR10` | 0.9064 | 0.9098 | 0.8981 | 0.8910 | 0.8651 | 0.1762 | 0.1626 | 0.7334 | 0.6313 |
| `PBR16` | 0.7963 | N/A | N/A | 0.9054 | 0.8725 | 0.1399 | 0.1970 | 0.7641 | 0.6520 |
| `PRX1_TERMMIX_MLP` | 0.8984 | 0.8981 | 0.8767 | 0.8869 | 0.8529 | 0.3316 | 0.4828 | 0.7518 | 0.6482 |
| `PRX2_TERMMIX_DUAL` | 0.8464 | 0.8483 | 0.6438 | 0.8442 | 0.8182 | 0.6218 | 0.6355 | 0.7398 | 0.6228 |

### 4.2 Strict Transfer Diagnosis

Unified diagnosis artifact:

- `assets/artifacts/phase_e_transfer_diag/phase_e_paradigm_refresh_diag_0311_00`

Headline:

- `PBR10`:
  - still fails strict RL-ready because `terminal_top1` is too low
- `PRX1_TERMMIX_MLP`:
  - not RL-ready yet
  - but clearly moves in the right direction on terminal completeness
- `PRX2_TERMMIX_DUAL`:
  - no longer terminal-limited on benchmark
  - but same-family quality collapses too much

## 5. What The New Experiments Actually Teach Us

### 5.1 `PBR10` Was Already A Strong Local Verifier

This was not generic transfer failure.

Evidence:

- strong same-family top1
- strong benchmark AUC
- strong first-error-edge behavior

The residual was narrow:

- complete correct traces were still not ranked high enough at the terminal prefix.

### 5.2 `PBR16` Improves Local Error Detection, But Not The Core Terminal Completeness Problem

`PBR16` is informative because it improves benchmark local metrics while still leaving terminal top1 weak:

- `gsm_terminal_top1 = 0.1399`
- `math_terminal_top1 = 0.1970`

So "make the model more local-error-sensitive" is not enough.

### 5.3 `PRX1` Is The Best New Direction In This Pass

`PRX1` does something important:

- it materially raises terminal completeness
- without collapsing local ranking

Most important changes vs `PBR10`:

- `gsm_terminal_top1: 0.1762 -> 0.3316`
- `math_terminal_top1: 0.1626 -> 0.4828`
- `math_terminal_gap: -0.0960 -> -0.0247`
- `gsm_f1: 0.7334 -> 0.7518`
- `math_f1: 0.6313 -> 0.6482`

But the price is real:

- same-family top1 drops below the `0.90` gate
- benchmark AUC also drops a bit

Interpretation:

- more terminal support helps,
- but current terminal mixing is still too blunt.

### 5.4 `PRX2` Over-Rotates Toward Completion

`PRX2` gives the clearest factorization lesson of this pass.

It makes benchmark terminal behavior much healthier:

- `gsm_terminal_top1 = 0.6218`
- `math_terminal_top1 = 0.6355`
- terminal gap becomes positive on both benchmarks

But it breaks the core verifier too much:

- held-out pair accuracy drops to `0.8464`
- same-family top1 drops to `0.8483`
- same-family local-first-bad drops to `0.6438`

Interpretation:

- explicit factorization is not wrong,
- but the current `dual_head + alpha-mix` implementation is not a good deployment rule.

In plain language:

- it learns to like complete solutions more,
- but it stops being a sharp local mistake detector.

## 6. Research Conclusion

The best current conclusion is:

1. the repo's stubborn issue is real and specific:
   - `terminal completion ordering`
2. stronger terminal data helps:
   - `PRX1` proves this
3. naive structural factorization can overshoot:
   - `PRX2` proves this
4. stronger local geometry alone is insufficient:
   - `PBR16` proves this

So the next serious redesign should be:

1. keep the ABR local discriminator as one component,
2. add an explicit outcome / answer verifier branch,
3. treat terminal completeness as a separate decision channel,
4. use uncertainty / abstain instead of forcing one scalar to do every job.

## 7. Recommended Next Experiments

### 7.1 `PRX3_CALIBRATED_TERMINAL_MIX`

Do not increase terminal anchors blindly.
Instead:

- keep `local_first_bad_edge` ratio near the current `PBR10` regime,
- use a smaller terminal increase than `PRX1`,
- bias terminal anchors toward hard all-correct completions only.

### 7.2 `PRX4_TWO_STAGE_VERIFIER`

Do not deploy one scalar.
Use:

- stage 1: ABR local-process scorer
- stage 2: answer-equivalence / outcome verifier

Only promote a checkpoint if both move in the right direction.

### 7.3 `PRX5_FACTORIZED_WITH_ROUTING`

Keep factorization, but do not infer with one fixed alpha.
Instead:

- use a router or uncertainty gate
- local-error regime -> local head dominates
- complete-solution regime -> terminal head dominates

### 7.4 LoRA Only After Objective Fix

Backbone adaptation should come after the data/objective redesign is cleaner.
Otherwise LoRA just adds degrees of freedom without resolving the supervision conflict.
