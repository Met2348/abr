# Phase E / Phase F Best-Practice Refresh (2026-03-12)

This note has two jobs:

1. reconcile the repo's current implementation reality with several stale local notes,
2. translate the newest verifier / verifier-guided RL lessons into concrete Phase E / F decisions.

## 1. Repo Reality Check

Before importing new ideas from papers, three local facts matter:

1. `phase_e_train_value_lora.py` already supports:
   - `--contrastive-loss-weight`
   - `--contrastive-margin`
   - `--reward-centering-weight`
   - `--head-architecture {mlp,gated_mlp,dual_head}`
2. some local markdown files still say contrastive-on-LoRA is "not yet implemented"
3. the older hand-written `PBR33/34/35` launchers used `set -e` without `pipefail`
   - if `python ... | tee log` crashed mid-train, the shell still kept going
   - that could produce a false "training done" narrative around an incomplete run dir

So the right immediate move is not "design contrastive from scratch again".
It is:

1. package the already-implemented knobs into stronger overnight suites,
2. and harden the launch layer so failed runs are not mislabeled as completed.

## 2. Fresh External Signals

The newest literature points to a fairly consistent pattern.

### 2.1 Harder verifier benchmarks keep exposing the same weakness

`Hard2Verify` argues that many current verifiers look strong on standard math
benchmarks but collapse on harder reasoning distributions and richer error
types, which is directly relevant to our repeated "held-out good / benchmark
still weak" pattern. Source: [Hard2Verify](https://arxiv.org/abs/2510.13744)

`VerifyBench` similarly emphasizes that verifier quality is often
over-estimated unless evaluation includes diverse, realistically difficult
reasoning traces instead of only easy local corruptions. Source:
[VerifyBench](https://arxiv.org/abs/2507.09884)

Implication for us:

1. do not over-trust held-out pair accuracy
2. keep ProcessBench / same-family trust / generator slices as first-class
   selection targets
3. prioritize data geometry over another generic scalar-head tweak

### 2.2 Cheap-vs-strong verification should be explicitly gated

`When to Trust the Cheap Check` studies how weaker, cheaper checks should be
used only when confidence is high and ambiguity is low; harder cases should be
routed to stronger verification. Source:
[When to Trust the Cheap Check](https://arxiv.org/abs/2602.17633)

Implication for us:

1. Phase F should not behave like a single monolithic stop policy forever
2. the more deployable direction is:
   - cheap verifier / heuristic first
   - uncertainty-triggered escalation on hard cases
3. this supports Phase F controller / router work more than more pure RL hype

### 2.3 Backbone adaptation still matters more than head churn

`PRIME` shows that implicit process rewards from a tuned policy/reference gap
can drive strong reasoning improvements without relying on hand-written step
labels alone. Source: [PRIME](https://arxiv.org/abs/2502.01456)

`GenPRM` pushes the same lesson in another form: stronger process reward
behavior often comes from richer generation/backbone modeling, not only from a
small scalar discriminator head. Source: [GenPRM](https://arxiv.org/abs/2504.00891)

Implication for us:

1. the repo should keep pushing LoRA / backbone-adapted verifiers
2. all-layer LoRA on stronger curated pair pools is a more credible frontier
   direction than another frozen-head micro-ablation
3. implicit / generative PRM ideas are medium-term, after LoRA packaging is
   stable

### 2.4 Question-level and completion-level supervision still need repair

`MathQ-Verify` focuses on math-question-level verifier construction and again
highlights that good step-local signals do not automatically imply good
question-level verification. Source: [MathQ-Verify](https://arxiv.org/abs/2603.03307)

`VPRM` also points toward richer process-level supervision contracts rather than
blindly reusing one scalar score for every sub-problem. Source:
[VPRM](https://arxiv.org/abs/2601.17223)

Implication for us:

1. current Phase E weakness is still "local ranking is decent, completion /
   all-correct behavior is not clean enough"
2. this is why dual-head is worth only controlled retries, not broad promotion
3. terminal / all-correct repair must be benchmark-aware, not just head-aware

### 2.5 Verifier-guided RL is real, but still depends on verifier quality

`Trust, but Verify: Self-Improving Reasoning with Synthetic Verifier-Guided RL`
reinforces a familiar lesson: verifier-guided RL can help, but only when the
verifier is already strong and its failure surface is understood. Source:
[Trust, but Verify](https://arxiv.org/abs/2505.13905)

Implication for us:

1. Phase F should still rank:
   - heuristic controller
   - BC-distilled controller
   - RL-like fine-tune
2. not the reverse
3. current RL-from-scratch results in this repo are still best treated as a
   stress test, not the mainline

## 3. Concrete Design Translation

### 3.1 Phase E

Most justified near-term frontier:

1. `PBR26` pair pool as the base
2. all-layer `Math-PRM` LoRA as the default high-ceiling branch
3. mild contrastive + mild reward-centering as the first controlled add-ons
4. `gated_mlp` as the capacity test
5. `dual_head` only as one controlled retry, not a full sweep

Why:

1. this matches the local artifact evidence (`PBR26`, `PBR31`, `PBR32`, `PBR33`)
2. it also matches the broader community lesson that representation/data
   geometry dominate tiny head churn

### 3.2 Phase F

Most justified near-term mainline:

1. stronger verifier slices as controller input (`p32_math`, `p33_gsm`, plus
   validated frozen baselines)
2. shared, interpretable heuristic teachers
3. BC warm-start as the practical distillation route
4. robust-from-scratch RL kept as a smaller control arm

Why:

1. this matches the repo's observed controller reality
2. it also matches current verifier-guided RL literature: RL only pays when the
   verifier and routing contract are already strong

## 4. Immediate Operational Changes

The refreshed implementation plan for this turn is:

1. patch the old LoRA launchers so failed training cannot silently fall through
   into eval
2. add a single-GPU Phase E overnight suite around:
   - all-layer LoRA + mild contrastive + centering
   - all-layer LoRA + stronger contrastive + gated head
   - top-4 LoRA + controlled dual-head retry
3. add a CPU Phase F overnight suite around:
   - stronger-slice controller sweep
   - generator robustness
   - weak-verifier ensemble
   - BC / BC->RL
   - robust-from-scratch RL-like probe

That package is much closer to the actual bottlenecks than re-running broad
legacy sweeps.
