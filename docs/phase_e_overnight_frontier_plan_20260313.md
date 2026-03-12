# Phase E Overnight Frontier Plan (2026-03-13)

## Purpose

This note captures the overnight experiment package launched after the latest
literature refresh.

The guiding constraint is simple:

1. do **not** spend the night re-running broad legacy sweeps,
2. instead run a small set of orthogonal experiments that directly test the
   current bottlenecks:
   - terminal blind spot,
   - frozen-backbone ceiling,
   - benchmark-oriented hybrid supervision,
   - curated repair after hybrid,
   - supervision geometry sweep.

## Updated Literature Inputs

Recent verifier / PRM system design papers and community benchmarks suggest
that offline improvement is now less about "one more scalar head tweak" and
more about:

1. factorizing local-vs-terminal verification pressure,
2. using critique / judge systems conservatively,
3. testing whether frozen backbones are already the true ceiling,
4. aligning supervision geometry to benchmark contracts,
5. separating cheap verifier utility from final RL-ready claims.

Key references:

1. `PRIME`
   - https://arxiv.org/abs/2602.11570
2. `Hard2Verify`
   - https://arxiv.org/abs/2510.13744
3. `Trust, But Verify`
   - https://arxiv.org/abs/2505.13445
4. `VPRM`
   - https://arxiv.org/abs/2601.17223
5. `VerifyBench`
   - https://arxiv.org/abs/2507.09884
6. `When to Trust the Cheap Check`
   - https://arxiv.org/abs/2603.05390
7. `MathQ-Verify`
   - https://arxiv.org/abs/2603.03307

## Why These Overnight Jobs

### 1. `F2_DUAL_HEAD_PBR10`

Question:

1. can we repair the known terminal blind spot without destroying the already
   strong local-first-bad geometry on `PBR10`?

Why now:

1. current repo evidence says `PBR10` already has strong local same-family and
   benchmark behavior,
2. so it is the right shared artifact to test whether score factorization
   helps.

### 2. `F3_LORA_PBR10`

Question:

1. is the remaining ceiling now mostly representation-limited rather than
   supervision-limited?

Why now:

1. this is the smallest safe way to probe backbone ceiling,
2. it avoids changing the data contract while still testing the strongest
   remaining hypothesis.

### 3. `PH2_PRM_LOCAL_TA10_MSGRID10_ARCH_SWEEP_SMOKE`

Question:

1. does a benchmark-oriented hybrid of:
   - PRMBench local,
   - light terminal anchors,
   - MS grid support
   improve benchmark geometry more cleanly than naive source transfer?

Why now:

1. literature increasingly supports benchmark-aligned verifier design,
2. the current repo still shows geometry mismatch between same-source success
   and benchmark-facing utility.

### 4. queued `CR1_CURATED_CENTER_GATE_SMOKE`

Question:

1. after hybrid curation, does reward centering + gated-MLP make the geometry
   more stable?

Why now:

1. this is a low-risk repair that can follow the hybrid run on the same GPU,
2. it tests whether some remaining instability is calibration drift rather than
   raw supervision shortage.

### 5. queued terminal ratio sweep

Question:

1. how much terminal-anchor signal is enough before we start damaging broader
   pair ranking?

Why now:

1. this directly maps to the known terminal-blind-spot failure mode,
2. and it is cheap enough to queue overnight after the dual-head run.

## Launcher

Use:

```bash
RUN_PREFIX=phase_e_overnight_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES= \
bash scripts/run_phase_e_overnight_frontier_suite.sh
```

The launcher itself handles GPU assignment and background execution.

Outputs:

1. launcher plan:
   - `assets/artifacts/phase_e_logs/<RUN_PREFIX>/overnight_plan.md`
2. launcher manifest:
   - `assets/artifacts/phase_e_logs/<RUN_PREFIX>/launch_manifest.jsonl`
3. per-job logs:
   - `assets/artifacts/phase_e_logs/<RUN_PREFIX>/*.log`

## Morning Inspection

```bash
RUN_DIR=assets/artifacts/phase_e_logs/<RUN_PREFIX>

cat "$RUN_DIR/launch_manifest.jsonl"
ls -dt "$RUN_DIR"/*.log
tail -n 80 "$RUN_DIR"/*.log
```

The real result summaries will still live under each suite's own output
directory; the launcher manifest is only the top-level coordination record.

## Current Observed Outcomes

### `F2_DUAL_HEAD_PBR10`

Status:

1. completed
2. wrapper summary had to be repaired after a collector bug, but train/eval
   artifacts are complete

Result:

1. held-out pair acc / auc: `0.6384 / 0.5792`
2. same-family:
   - top1 `0.6706`
   - local first-bad `0.4352`
3. ProcessBench:
   - GSM AUC `0.7254`
   - Math AUC `0.7063`

Interpretation:

1. dual-head factorization, in this concrete implementation, is a failed
   repair
2. it degrades the strong `PBR10` geometry instead of fixing the known
   terminal issue

### `PH2_PRM_LOCAL_TA10_MSGRID10_ARCH_SWEEP_SMOKE`

Status:

1. training completed
2. wrapper stopped before eval because of a path-resolution bug
3. post-hoc benchmark eval repair was run successfully

Result:

1. held-out pair acc / auc: `0.9318 / 0.9040`
2. ProcessBench GSM8K:
   - pair acc `0.4625`
   - auc `0.5224`
3. ProcessBench Math:
   - pair acc `0.4360`
   - auc `0.5321`

Interpretation:

1. the hybrid artifact is easy to fit
2. but the learned score is still not benchmark-trustworthy
3. this argues for stricter contract redesign rather than another loose hybrid
   concat

### `CR1_CURATED_CENTER_GATE_SMOKE`

Status:

1. blocked by recipe guard before training

Interpretation:

1. this is the correct behavior
2. the run combined mixed local/terminal semantics with semantic weighting,
   which is now treated as unsafe by default

### `F3_LORA_PBR10`

Status:

1. still running in the latest overnight batch
2. current intermediate signal is positive:
   - epoch-0 eval pair acc `0.7863`
   - epoch-0 eval auc `0.7741`

Interpretation:

1. this is still the most credible frontier direction left from the overnight
   package
2. unlike `F2`, it has not already shown a clear regression
