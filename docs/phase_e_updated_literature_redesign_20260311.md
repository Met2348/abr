# Phase E Updated Literature Refresh And Redesign (2026-03-11)

## Purpose

This note refreshes the repository's research assumptions using literature and
community evidence newer than the old `2025-03` cutoff, while preserving the
core ABR idea:

1. a learned process/value utility is still useful,
2. but the surrounding data curation, verifier architecture, and evaluation
   protocol can change substantially.

## Broader 2026-03 refresh

This document remains the concise strategy note.

For a broader literature refresh that extends the evidence window through
**2026-03-11** and adds newer directions such as:

1. `CompassVerifier / VerifyBench / Hard2Verify`
2. `Reward Reasoning Model / Libra / Think Twice`
3. `BiPRM / ActPRM / PACR`
4. `LongRLVR / TinyV / Online Learnability of CoT Verifiers / V1`

see:

1. `docs/phase_e_literature_refresh_20260311.md`

## Newer Evidence That Changes The Design

### 1. PRIME (2026): verifier quality predicts RLVR effectiveness

Source:
1. `PRIME: A Process-Outcome Alignment Benchmark for Verifiable Reasoning in Mathematics and Engineering`
   - https://arxiv.org/abs/2602.11570

What matters:
1. outcome-only verification is not enough;
2. correct final answers can still come from flawed derivations;
3. verifier accuracy on `PRIME` correlates strongly with RLVR effectiveness
   (`R^2 > 0.92`).

Repository consequence:
1. our current "same-source held-out > benchmark later" sequence is still too
   weak if we want RL-ready claims;
2. we need one evaluation layer that explicitly measures
   **process-outcome alignment**, not just local pair ranking.

### 2. Hard2Verify (2025): open-source step verifiers still lag badly

Source:
1. `Hard2Verify: A Step-Level Verification Benchmark for Open-Ended Frontier Math`
   - https://arxiv.org/abs/2510.13744

What matters:
1. even strong open-source verifiers struggle on human-annotated frontier
   step verification;
2. verification quality depends on both model family and verification compute.

Repository consequence:
1. a frozen scalar head should be treated as a bounded-support verifier, not a
   universal verifier;
2. low benchmark scores should not be over-explained as just "bad hyperparams".

### 3. RISE (2025): online self-verification is a first-class training signal

Source:
1. `Trust, But Verify: A Self-Verification Approach to Reinforcement Learning with Verifiable Rewards`
   - https://arxiv.org/abs/2505.13445

What matters:
1. strong RLVR systems increasingly train:
   - solution generation
   - and self-verification
   together;
2. more verification compute can materially help.

Repository consequence:
1. a pure scalar head is likely not the final form;
2. we should add an experimental branch where the model also learns to emit a
   critique / verification decision, not just a score.

### 4. VPRM (2026): deterministic process verification can beat opaque judges

Source:
1. `Beyond Outcome Verification: Verifiable Process Reward Models for Structured Reasoning`
   - https://arxiv.org/abs/2601.17223

What matters:
1. when step correctness can be checked programmatically, process-level
   verification becomes much more trustworthy than opaque neural scoring;
2. step coherence and final-label coherence both improve.

Repository consequence:
1. whenever a benchmark allows deterministic local checking, we should exploit
   it instead of relying only on neural pair heads;
2. our future RL-ready stack should support hybrid rewards:
   - learned verifier score
   - plus deterministic checks where available.

## Widespread Direction Of Travel

Across `PRIME`, `Hard2Verify`, `RISE`, `VPRM`, plus earlier `ProcessBench`,
`PRMBench`, `Lessons`, `R-PRM`, `VersaPRM`, and `ThinkPRM`, the direction is
consistent:

1. better data geometry matters;
2. benchmark-aligned verifier selection matters;
3. process-outcome alignment matters;
4. critique / self-verification behavior matters;
5. pure scalar same-source fit is not enough for RL-ready claims.

## Strategic Diagnosis For This Repository

The persistent local difficulty is now better explained as:

1. **coverage mismatch**
   - training pairs cover local ranking,
   - but not enough all-correct / derivation-flaw contrast;
2. **output-object mismatch**
   - a scalar head is asked to stand in for:
     - local edge detection
     - terminal correctness
     - process-outcome consistency
     - RL-time trustworthy rejection;
3. **evaluation mismatch**
   - same-source held-out still says too little about RL usefulness.

## Redesigned Phase E Mainline

### Track 1: bounded-support scalar verifier

Keep the current strongest scalar path, but scope it honestly:

1. same-source learnability
2. same-family utility
3. bounded-support rerank / rejection only

### Track 2: benchmark-aligned verifier selection

Adopt a `PRIME`-style mindset:

1. select verifiers by benchmark-aligned process-outcome criteria,
2. not by same-source accuracy alone.

Concrete repository move:
1. add one evaluation layer that scores:
   - all-correct trajectories,
   - locally wrong but final-correct trajectories,
   - clearly wrong trajectories.

### Track 3: critique / self-verification branch

Inspired by `RISE` and the broader community trend:

1. add a second experimental head or auxiliary task for:
   - first-error explanation
   - or binary critique generation;
2. compare:
   - score-only
   - score + critique
   inside the same benchmark family.

### Track 4: deterministic verifier hooks

Inspired by `VPRM`:

1. where the benchmark or task permits rule-based checks, use them;
2. treat learned value heads as complements, not sole arbiters.

## Immediate Experiment Redesign

### E-Redesign A: process-outcome alignment split

Construct three evaluation buckets per benchmark:
1. all-correct
2. derivation-flawed-but-final-correct
3. clearly incorrect

Goal:
1. stop over-crediting verifiers that only learn terminal correctness

### E-Redesign B: score vs critique auxiliary

Compare:
1. scalar-only head
2. scalar + first-error binary auxiliary
3. scalar + short critique auxiliary

Goal:
1. test whether RL-facing usefulness improves once verification is partly
   verbalized / structured

### E-Redesign C: deterministic check hybrid

For datasets where local consistency checks are available:
1. learned score
2. deterministic process check
3. blended reward / rejection rule

Goal:
1. reduce reward hacking risk before Phase F / RL

## Bottom Line

The core ABR idea remains viable:
1. learn a process utility and use it for selection.

But the implementation target should change:
1. from "just get a strong scalar head"
2. to "build a verifier stack that is benchmark-aligned, process-outcome aware,
   and eventually critique-capable".
