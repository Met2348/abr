## 0. 17:33 Addendum

1. paper sync is now complete for the currently referenced paper-like URLs in `docs/**/*.md`:
   - local audit count: `113` paper URLs
   - missing after resync: `0`
2. the `PRIME` citation conflict has been corrected conceptually:
   - `2502.01456` = implicit-reward RL method
   - `2602.11570` = process-outcome alignment benchmark
3. based on current TRL docs and local env inspection, replay-buffer GRPO is now part of the autonomous RL infra plan.
4. new queued run:
   - `phase_f_replay_grpo_canary_wait_0312_1732`
   - purpose: test whether replay-buffer GRPO improves stability on the same `PBR32 -> GSM8K` canary slice.

# Phase E / Phase F Autonomous Research Plan

- generated_at: 2026-03-12 17:14:14 +0800
- scope: literature refresh, problem verification, and the next ~10 hours of autonomous experiments
- status: mixes completed work, running work, and queued work

## 1. Web-Checked Inputs Used In This Pass

Primary sources re-checked on the web:

1. `TRL GRPOTrainer` docs
   - https://huggingface.co/docs/trl/grpo_trainer
2. `VerifyBench`
   - https://arxiv.org/abs/2507.09884
3. `Hard2Verify`
   - https://arxiv.org/abs/2510.13744
4. `When to Trust the Cheap Check`
   - https://arxiv.org/abs/2602.17633
5. `PRIME-RL / implicit rewards`
   - https://arxiv.org/abs/2502.01456
6. `PRIME benchmark / process-outcome alignment benchmark`
   - https://arxiv.org/abs/2602.11570
7. `The Curse of Depth in LLMs`
   - https://arxiv.org/abs/2502.05795
8. `ThinkPRM`
   - https://arxiv.org/abs/2504.16828
9. `GenPRM`
   - https://arxiv.org/abs/2504.00891
10. `VPRM`
   - https://arxiv.org/abs/2601.17223
11. `DAPO / Dr. GRPO / VAPO / Is PRM Necessary?`
   - already mirrored in `docs/relatedPapers/` and previously integrated into repo notes

Cross-source takeaways that still match repo evidence:

1. verification quality is slice-dependent; cheap/strong routing is more realistic than one universal verifier;
2. RL with verifiable outcomes remains strong, but process-reward RL needs stronger reward-shaping and stability discipline;
3. controller utility and abstain/escalate behavior are valid product-level targets even before full RL succeeds;
4. modern GRPO stacks should expose `loss_type`, reward scaling, KL, and truncation handling explicitly rather than burying defaults.

## 2. New Problems Verified In This Pass

### 2.1 RL infrastructure problem

Problem:
- `scripts/phase_f_grpo_lite.py` depended on `trl`, but the default repo `python` does not have `trl` installed.

Verification:
- current default `python` env: `trl` import fails
- current `/home/zling/anaconda3/envs/bcr/bin/python3`: `trl 0.29.0` is available and exposes modern `GRPOConfig` fields such as `loss_type`, `scale_rewards`, `beta`, `mask_truncated_completions`, and `use_vllm`

Action taken:
- patched `phase_f_grpo_lite.py` to expose those modern knobs and emit an explicit environment hint when `trl` is missing.

### 2.2 Legacy overnight launcher problem

Problem:
- `scripts/run_phase_f_overnight_suite.sh` had no `pipefail`, used brittle artifact resolution, and encoded an older live-RL interpretation.

Action taken:
- patched it into a safer legacy canary launcher with marker-based artifact lookup and explicit modern GRPO defaults.

## 3. Completed Experiment Additions In This Pass

### 3.1 Hybrid usability suite completed

Artifact root:
- `assets/artifacts/phase_f_logs/phase_f_hybrid_usability_0312_1701_HALL_HYBRID_USABILITY/`

What it answered:

1. `PH1/PH2` hybrid verifiers are already useful controller slices.
2. single best slices are domain-dependent:
   - `PH2-gated` is strongest on Math controller sweep (`0.8764` balanced F1)
   - `PH2-mlp` is strongest on GSM controller sweep (`0.8729` balanced F1)
3. hybrid ensembles can beat single models:
   - `math ph1mlp + ph1gated = 0.8926`
4. `BC->RL` is still selective, and from-scratch RL-like is still weak.

### 3.2 Teacher-aligned BC audit closed a loophole

Artifacts:
- `assets/artifacts/phase_f_bc/phase_f_focus_ph2_guarded_*`
- `assets/artifacts/phase_f_bc/phase_f_focus_ph1_drop_*`

Takeaway:
- the earlier `PH2-mlp Math` RL-positive signal was not robust after teacher alignment and multi-seed follow-up.

## 4. Running And Queued Autonomous Work

### 4.1 Running

1. clean-data LoRA follow-ups: `PBR38`, `PBR39`, `PBR40`
2. `L2` gated/centered LoRA on the noisier pool
3. `PH3` hybrid bridge experiment

### 4.2 Queued

1. `phase_f_hybrid_preflight_wait_0312_1712`
   - command target: `scripts/run_phase_f_hybrid_preflight_suite.sh`
   - purpose: threshold stability + reward-hacking audit for `PH1/PH2`
2. `phase_f_modern_grpo_canary_wait_0312_1712`
   - command target: modernized `scripts/phase_f_grpo_lite.py`
   - purpose: verify that the updated GRPO infra really runs under the `bcr` env with explicit modern settings
3. `phase_f_replay_grpo_canary_wait_0312_1732`
   - command target: modernized `scripts/phase_f_grpo_lite.py` with `--trl-use-replay-buffer`
   - purpose: test whether replay-buffer GRPO improves stability on the same `PBR32 -> GSM8K` canary slice

## 5. 10-Hour Autonomous Plan

### A1. Finish and interpret clean-data LoRA runs

Targets:
1. `PBR38`
2. `PBR39`
3. `PBR40`
4. `L2`

Decision rule:
- only promote a new `Phase E` scalar-head line if it beats `PBR32` on Math without obvious GSM or stability regression.

### A2. Finish and interpret `PH3`

Target:
- decide whether `PH3` actually bridges `PH1` and `PH2` or just duplicates one of them.

Decision rule:
- if `PH3` improves either Math or GSM while keeping controller usability competitive, it becomes the default hybrid mainline.

### A3. Finish hybrid preflight queue

Target:
- check threshold stability and reward-hacking exposure of `PH1/PH2` before any controller or RL promotion.

Decision rule:
- only promote hybrid candidates with both acceptable fixed-threshold tolerance and clearly lower superficial-hack exposure than `PBR31`.

### A4. Finish modern GRPO canary queue

Target:
- confirm the repaired `GRPO` path runs cleanly with explicit `dr_grpo`-style settings and no hidden environment failure.

Decision rule:
- treat this as infra validation, not headline performance.
- if it fails, fix infra before any larger live RL run.

### A5. If a GPU opens after queues finish: hybrid-focused controller continuation

Preferred next experiment:
1. run a second hybrid usability pass centered on the best post-preflight candidates only;
2. only then consider another BC->RL follow-up;
3. keep from-scratch RL-like as a stress-test arm, not the mainline.

## 6. Default Recommendations While The Queues Run

1. `Phase E`: invest more in benchmark-anchored hybrid curation than in more blind source concat.
2. `Phase F`: keep controller/reranker/preflight as the main promotion path.
3. `RL`: do not spend a full overnight on bigger live RL until the canary and hybrid preflight results are read.
