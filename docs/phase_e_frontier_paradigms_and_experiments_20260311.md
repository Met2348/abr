# Phase E Frontier Paradigms And Experiments (2026-03-11)

## Purpose

This note does two jobs:

1. explain the newer verifier / PRM paradigms in novice-friendly language,
2. record the current repository-level frontier experiments that were designed
   to test those paradigms against our stubborn Phase E bottleneck.

The current bottleneck is no longer “local ranking is broken”.

The strongest already-finished `PBR10` diagnosis says:

1. local first-error behavior is already strong,
2. same-family utility is already strong,
3. the narrow remaining weakness is:
   - `all-correct terminal completion undervaluation`.

So the question is no longer:

1. “can the head learn anything at all?”

It is now:

1. “what kind of verifier architecture or data-cleaning paradigm is most likely
   to fix the remaining terminal / completion blind spot without destroying the
   strong local geometry we already have?”

## Novice-Friendly Glossary

### 1. scalar verifier

The simplest verifier.

It takes one reasoning prefix or one candidate solution, and outputs one score.

You can think of it as:

1. “how trustworthy does this reasoning look?”

This repository's old Phase E mainline is mostly this kind of model.

### 2. critique / judge verifier

A judge-style verifier does not only emit a score.

It tries to **read the reasoning and explicitly decide whether it is good or
bad**, often with some explanation or structured decision.

Practical use:

1. filter bad training pairs,
2. relabel uncertain cases,
3. audit disagreement slices.

### 3. self-verification

The model does not just solve the problem.

It also learns to verify its own solution or reasoning.

This is important because newer RL reasoning systems increasingly train:

1. generation
2. and verification

together, instead of assuming one small static reward head is enough.

### 4. process-outcome alignment

This asks:

1. does the verifier reward a reasoning process because the process is actually
   good,
2. or only because the final answer happens to look correct?

This matters because a final answer can be correct even when the derivation is
bad or misleading.

### 5. deterministic verifier

Instead of relying only on a neural score, part of the verification is done by
explicit rules or checks.

Examples:

1. equation equivalence,
2. parser-based contract checks,
3. local symbolic consistency checks.

These are often more trustworthy than an opaque score when they are available.

### 6. dual-head verifier

One shared encoder, but two scoring heads.

In our current usage, the idea is:

1. one head emphasizes local process quality,
2. another head emphasizes terminal / completion quality.

This is an explicit attempt to prevent one scalar score from collapsing two
different jobs into one number.

### 7. light backbone adaptation / LoRA

Instead of training only the tiny score head, we allow a small subset of the
backbone to adapt.

`LoRA` is a low-rank update trick that changes the model more cheaply than full
fine-tuning.

The motivation is:

1. maybe the current bottleneck is not the head,
2. maybe the backbone representation itself is missing the right features.

## Newer Community / Literature Direction

The broad 2025H2-2026 direction is now more consistent than it was a year ago.

### 1. benchmark-aligned verifier selection

Recent work increasingly emphasizes that the “best verifier” should be selected
against benchmark slices that actually matter, not only same-source held-out
pair accuracy.

Relevant sources:

1. `PRIME`
   - https://arxiv.org/abs/2602.11570
2. `VerifyBench`
   - https://arxiv.org/abs/2507.09884
3. `CompassVerifier`
   - https://aclanthology.org/2025.acl-long.1102/

Repository implication:

1. same-source win is necessary,
2. but not enough.

### 2. critique / self-verification is becoming first-class

Recent systems do not rely only on a scalar score.

They increasingly incorporate:

1. critique behavior,
2. self-verification,
3. verifier-side reasoning.

Relevant sources:

1. `Trust, But Verify`
   - https://arxiv.org/abs/2505.13445
2. `ThinkPRM`
   - https://arxiv.org/abs/2504.16828

Repository implication:

1. `judge-filter` is a legitimate direction to test,
2. but it must be stable and not destroy the data geometry.

### 3. factorization beats one overloaded scalar

Verifier systems increasingly separate roles:

1. local process signal,
2. terminal correctness,
3. abstention / uncertainty,
4. answer verification.

Relevant sources:

1. `PRIME`
2. `Weaver`
   - https://arxiv.org/abs/2510.18084
3. `When to Trust the Cheap Check`
   - https://arxiv.org/abs/2603.05390

Repository implication:

1. `dual_head` is a principled experiment, not just random architecture search.

### 4. representation still matters

Some recent evidence says verifier quality is often representation-limited, not
only data-limited.

Relevant sources:

1. `Hard2Verify`
   - https://arxiv.org/abs/2510.13744
2. `VPRM`
   - https://arxiv.org/abs/2601.17223

Repository implication:

1. a small LoRA probe is worth doing,
2. but only with safe recipes and controlled expectations.

## Current Frontier Experiment Design

All three experiments deliberately share the same strong base artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_pbr10_prm7b_dpo8k_s42__6184f7e62f65`

Why this artifact:

1. it is the strongest current frozen-head frontier line,
2. it already demonstrates strong local ProcessBench behavior,
3. so it is the right place to test “what fixes the remaining terminal blind
   spot”.

### F1. Judge / critique filtering

Command wrapper:

1. `scripts/run_phase_e_frontier_suite.sh`
2. group:
   - `F1_JUDGE_FILTER_PBR10`

What it does:

1. use `Qwen2.5-Math-7B-Instruct` as a local pair judge,
2. try to keep pairs that the judge can confidently parse and verify,
3. retrain a safe frozen `mlp` on the filtered set.

Purpose:

1. test whether a stronger math judge can remove residual pair noise and help
   the terminal weakness.

Observed result so far:

1. filtered artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_frontier_judge_0311_2031_judgefilter__88888f79419b`
2. all `1783` auditable `local_first_bad_edge` pairs were dropped as
   `parse_failed`
3. training set was reduced to:
   - `4679 sibling_branch`
   - `485 terminal_completion_anchor`
   - `0 local_first_bad_edge`

Interpretation:

1. the current strict judge contract is not a useful denoiser here;
2. it changes the supervision geometry instead of cleaning it.

### F2. Dual-head routing

Wrapper:

1. `scripts/run_phase_e_frontier_suite.sh`
2. group:
   - `F2_DUAL_HEAD_PBR10`

What it does:

1. keep the shared `PBR10` artifact unchanged,
2. replace the single head with `dual_head`,
3. keep the safer recipe:
   - `ranking_target_space=score`
   - `pair_weight_mode=none`
   - `checkpoint_selection_metric=pair_acc`

Purpose:

1. explicitly separate local and terminal pressure,
2. test whether the remaining terminal weakness is caused by objective overload.

### F3. Minimal LoRA probe

Wrapper:

1. `scripts/run_phase_e_frontier_suite.sh`
2. group:
   - `F3_LORA_PBR10`

Compact probe also launched separately on a subset:

1. run name:
   - `phase_e_frontier_lora_probe_0311_2050`

What it does:

1. apply a very small LoRA update,
2. keep the safe recipe,
3. use a smaller train/eval subset for a faster directional answer.

Purpose:

1. test whether the frozen-head ceiling is the main remaining bottleneck,
2. without paying the cost of a full large LoRA sweep.

## How To Read These Frontier Directions

### If judge-filter wins

Then the main issue was:

1. residual label noise / semantic inconsistency.

### If dual-head wins

Then the main issue was:

1. one scalar score being forced to encode both local correctness and terminal
   value.

### If LoRA wins

Then the main issue was:

1. representation bottleneck in the frozen backbone.

### If all three fail

Then the next redesign should probably move toward:

1. answer-verifier split,
2. abstain / escalate behavior,
3. process-outcome alignment slices,
4. or critique-capable verifier branches.

## Current Status

As of this note:

1. judge-filter has already produced a strong negative intermediate result,
2. full `dual-head` and compact `LoRA probe` are still running,
3. completed pilot evidence from earlier in the repo already suggests:
   - tiny-data LoRA can hurt same-family trust badly,
   - brittle strict-JSON judge contracts are dangerous,
   - and benchmark-facing gains usually require stronger geometry or stronger
     representations, not just “more hyperparameter search”.
