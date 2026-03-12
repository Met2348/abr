# Phase E/F Verifier-System Refresh (2026-03-12)

## Purpose

This note consolidates one updated judgment:

1. recent verifier / RL-for-reasoning literature no longer supports treating a
   single scalar value head as the final system object;
2. local repository evidence now matches that broader community direction;
3. the next redesign should preserve the core `ABR` idea, but factor the
   verifier stack into clearer roles.

## New external guidance that still matters after the 2025-03 cutoff

Primary sources worth keeping in view:

1. `PRIME (2026)` — process-outcome alignment matters for RLVR
   - https://arxiv.org/abs/2602.11570
2. `Hard2Verify (2025)` — open verifiers still struggle on hard step verification
   - https://arxiv.org/abs/2510.13744
3. `Trust, But Verify / RISE (2025)` — self-verification should be treated as a
   first-class signal, not only a post-hoc scalar label
   - https://arxiv.org/abs/2505.13445
4. `VPRM (2026)` — deterministic / structured verification can beat opaque
   reward modeling when the task permits it
   - https://arxiv.org/abs/2601.17223
5. `VerifyBench` — verifier behavior is highly input-structure-sensitive
   - https://arxiv.org/abs/2507.09884
6. `When to Trust the Cheap Check (2026)` — cheap/strong cascades are valid in
   principle, but only if the cheap verifier is strong enough to save most of
   the strong-verifier budget
   - https://arxiv.org/abs/2603.05390

Repository implication:

1. `same-source held-out acc/auc` remains necessary;
2. but the final deployable object should be a verifier system, not just a
   scalar head checkpoint.

## Local evidence that changes the design

### 1. Cheap-to-strong gate is directionally right, but not yet cost-effective

Artifacts:

1. `assets/artifacts/phase_e_gate_sweeps/phase_e_cheap_strong_gate_0312_20260311T204929Z/summary.md`
2. `assets/artifacts/phase_e_gate_sweeps/phase_e_cheap_strong_gate_ms_0312_20260311T204937Z/summary.md`

Findings:

1. `prm_e46 -> pbr26`
   - to approach `pbr26`, the gate needed about:
     - `95%` strong usage on `ProcessBench Math`
     - `97%` strong usage on `ProcessBench GSM8K`
2. `ms_e43 -> pbr26`
   - slightly better, but still needed about:
     - `91%` strong usage on `Math`
     - `87%` strong usage on `GSM8K`
3. under a more practical budget (`<=50%` strong usage), gains existed but were
   still far from the strong verifier:
   - `prm_e46 -> pbr26 math`: best AUC `0.684`
   - `ms_e43 -> pbr26 math`: best AUC `0.673`

Interpretation:

1. a weak/strong cascade is a good systems direction;
2. but current weak verifiers are not good enough yet to make that cascade
   economically attractive.

### 2. Dual-head is not automatically the fix

Artifact:

1. `assets/artifacts/phase_e_logs/phase_e_probe_f2/final_summary.md`

Finding:

1. the current dual-head implementation was a negative result:
   - same-family routing collapsed,
   - benchmark AUC regressed.

Interpretation:

1. factorization is still conceptually valid;
2. but the current local-vs-terminal routing implementation is not yet the
   correct realization of that idea.

### 3. Benchmark-oriented hybrids can still learn the wrong thing

Artifact:

1. `assets/artifacts/phase_e_logs/phase_e_probe_ph2/final_summary.md`

Finding:

1. held-out pair acc / auc became very high,
2. but benchmark AUC stayed weak.

Interpretation:

1. trainability is not the same as trustworthiness;
2. benchmark-facing repair needs stronger contract design, not just more mixed
   pair types.

## Updated redesign target

The next target should be a **verifier system** with three explicit roles:

1. `local/process verifier`
   - decides whether the reasoning is still locally healthy
   - optimized for:
     - first-bad localization
     - later-bad degradation
     - same-family reranking
2. `terminal/answer verifier`
   - decides whether the current completion is complete / acceptable / correct
   - optimized for:
     - all-correct terminal ranking
     - answer-equivalence or completion acceptance
3. `abstain / escalate gate`
   - decides whether the cheap local system is confident enough,
   - otherwise hands the case to a stronger verifier or deterministic check

This keeps the core `ABR` idea unchanged:

1. a branch controller still needs value-like signals,
2. but the value signal is now produced by a small verifier system instead of
   one overloaded scalar head.

## Suggested pipeline refresh

### Stage R1: safe scalar baseline

1. keep current safe `Phase E` scalar head as the bounded-support baseline
2. never use dangerous recipes as defaults
3. keep `pair_acc` as the default checkpoint selection metric

### Stage R2: explicit local vs terminal dataset split

Curate separate training contracts:

1. local contracts:
   - sibling / fork-point / first-bad / later-bad
2. terminal contracts:
   - all-correct completion acceptance
   - answer-equivalence / completion anchors
3. disagreement slice:
   - samples where local and terminal objectives disagree

### Stage R3: answer-verifier split

Do not keep pushing everything through one scalar process head.

Instead:

1. process verifier handles step/prefix health
2. answer verifier handles final completion acceptability
3. fusion happens downstream in controller logic, not inside one opaque score

### Stage R4: cheap/strong gate after weak verifier improves

Only after a better weak verifier exists should we revisit:

1. cheap-verifier early accept
2. strong-verifier escalation on ambiguous cases

### Stage R5: RL phase only after verifier-system readiness

Required before calling anything `RL-ready`:

1. same-family utility remains strong
2. benchmark AUC / first-edge survive
3. process-outcome alignment slices stop being pathological
4. cheap/strong or abstain gate has a meaningful cost-quality tradeoff

## Bottom line

The repository should now stop asking:

1. "Can one scalar verifier head be tuned a bit more?"

and start asking:

1. "What is the minimal verifier system that preserves ABR while matching the
   actual structure of reasoning trust?"
