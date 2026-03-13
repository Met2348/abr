# Phase E/F Overnight Refresh Plan (2026-03-12)

## Purpose

This note records the current overnight research direction while the server is busy and while the paper-writing report is being prepared.

The plan keeps the `ABR/BCR` core idea unchanged:
1. learn a verifier-like utility that can judge reasoning quality;
2. use it later for routing, reranking, rejection, and controller/RL infrastructure;
3. but stop assuming a single frozen scalar head is already enough.

## Why These Overnight Tracks

Recent local evidence says:
1. `same-source` learnability is already established on the better pair sources;
2. the remaining bottlenecks are:
   - benchmark-facing trust,
   - terminal completion valuation,
   - representation ceiling,
   - controller-facing usefulness.

Recent literature/community guidance supports the same direction:
1. `PRIME (2026)` emphasizes process-outcome alignment for RLVR.
2. `Trust, But Verify / RISE (2025)` supports self-verification and verification-aware training.
3. `VerifyBench` and `Hard2Verify` both warn that apparent verifier quality often breaks on harder verification settings.
4. `VPRM (2026)` argues for more structured verification where possible.

## Key Literature References

1. `PRIME (2026)`
   - https://arxiv.org/abs/2602.11570
2. `Trust, But Verify / RISE (2025)`
   - https://arxiv.org/abs/2505.13445
3. `VerifyBench (2025)`
   - https://arxiv.org/abs/2507.09884
4. `Hard2Verify (2025)`
   - https://arxiv.org/abs/2510.13744
5. `VPRM (2026)`
   - https://arxiv.org/abs/2601.17223

## Overnight Tracks

### Track A: PRMBench selected-relabel, benchmark-facing

Script:
- `scripts/run_phase_e_prmbench_selected_relabel_suite.sh`

What changed:
1. default baseline switched from overfit `E78` to balanced `E46`;
2. default trainer geometry switched to the safe family:
   - `score`
   - `none`
   - `pair_acc`
3. the suite now appends:
   - same-family trust eval,
   - `ProcessBench GSM8K` eval,
   - `ProcessBench Math` eval;
4. then it compares the repaired candidate back to `E46`.

Question:
- can conservative selected-relabel improve benchmark-facing trust without simply recreating an overfit same-source classifier?

### Track B: PBR6 LoRA backbone smoke

Script:
- `scripts/run_phase_e_processbench_research_suite.sh`
- group: `PBR6_LORA_BACKBONE_SMOKE`

Question:
- after frozen-head repair lines plateaued, does minimal LoRA on a stronger backbone improve benchmark-facing geometry enough to justify deeper backbone adaptation?

### Track C: Phase F modern preflight on the selected-relabel candidate

Script:
- `scripts/run_phase_f_modern_preflight_suite.sh`

Question:
- even if selected-relabel improves benchmark AUC, does it help fixed-threshold stability and reward-hacking robustness enough to matter for `Phase F`?

## Queue Wrapper

Wrapper:
- `scripts/run_phase_e_phase_f_autonomy_overnight_suite.sh`

It queues two jobs on busy shared GPUs:
1. `PRMBench selected-relabel + Phase F modern preflight`
2. `PBR6 LoRA backbone smoke`

Default GPU order in this wrapper:
1. `GPU_RELABEL=1`
2. `GPU_LORA=3`

## Direct Run Command

```bash
RUN_PREFIX=phase_e_phase_f_auto_$(date +%m%d_%H%M) \
GPU_RELABEL=1 \
GPU_LORA=3 \
bash scripts/run_phase_e_phase_f_autonomy_overnight_suite.sh
```

## Success Criteria

### Selected-relabel
1. held-out should not regress badly versus `E46`;
2. same-family rerank and rejection should remain positive;
3. `ProcessBench GSM8K/Math` should improve or at least become more stable;
4. `Phase F modern preflight` should show no new reward-hacking surface.

### PBR6 LoRA
1. benchmark AUC should improve over the frozen reference without same-family collapse;
2. no recipe collapse;
3. no severe memory blow-up under the current shared-server constraints.

## Immediate Interpretation Rules

1. If selected-relabel only improves held-out, it is not enough.
2. If LoRA improves benchmark AUC but destroys same-family trust, it is only a ceiling probe, not a deployable verifier.
3. If neither line helps, the next redesign should prioritize:
   - answer/terminal verifier split,
   - abstain/escalate,
   - process-outcome alignment slices,
   rather than more small scalar-head variants.
