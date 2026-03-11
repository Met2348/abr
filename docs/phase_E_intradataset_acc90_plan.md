# Phase E Intradataset ACC90 Plan

## Why This Branch Exists

The latest Phase E evidence changed the target question.

We no longer assume that:
1. training on one source should directly generalize to a different benchmark,
2. cross-dataset transfer is the first success criterion.

The new question is narrower and stricter:

> On one dataset's own held-out pairs, can we train a value head that reaches very high discrimination accuracy?

This branch therefore ignores transfer and benchmark-native generalization on purpose.
It is a controlled learnability branch.

## Success Criterion

For one dataset-specific suite to count as a success, we require:
1. `mean_heldout_pair_acc >= 0.90`
2. `mean_heldout_auc >= 0.90`
3. `worst_seed_pair_acc >= 0.85`
4. `worst_seed_auc >= 0.85`
5. low seed variance (`std <= 0.05` by default)

The primary metric is `heldout_pair_acc`.

## Covered Datasets

Only the sources that currently qualify as high-quality pair/process supervision are included:
1. `Math-Shepherd`
2. `PRMBench_Preview`
3. `R-PRM`

`PRM800K` is intentionally left out of this ACC90 branch because current evidence still treats it as a weak/noisy source under our pair objective.

## Recipe Families

Each dataset gets a small same-source sweep.

### Math-Shepherd

1. `E40_MS_ACC90_LINEAR_ROBUST_SEED3`
   - linear head
   - ranking-only
   - anti-saturation
   - purpose: strongest simple baseline
2. `E41_MS_ACC90_MLP_RANK_SEED3`
   - MLP head
   - ranking-only
   - purpose: test whether head capacity is the main bottleneck
3. `E42_MS_ACC90_MLP_JOINT_SEED3`
   - MLP head
   - ranking + BCE
   - purpose: test whether same-source binary fitting helps
4. `E43_MS_ACC90_MLP_HIGHCONF_SEED3`
   - MLP head
   - joint objective
   - stricter confidence filter
   - purpose: test whether denoising is the main lever

### PRMBench Preview

1. `E44_PRMBENCH_ACC90_LINEAR_SEED3`
2. `E45_PRMBENCH_ACC90_MLP_RANK_SEED3`
3. `E46_PRMBENCH_ACC90_MLP_JOINT_SEED3`

Purpose: direct process-pair supervision should be the best candidate for crossing 90%.

### R-PRM

1. `E47_RPRM_ACC90_LINEAR_SEED3`
2. `E48_RPRM_ACC90_MLP_RANK_SEED3`
3. `E49_RPRM_ACC90_MLP_JOINT_SEED3`

Purpose: direct chosen/rejected preference supervision may fit the pair objective more cleanly than converted labels.

## Current Empirical Reading (2026-03-10)

The latest completed runs already answer one important model-design question:

> Are current weak same-source results caused by an over-simple head, by
> under-training, or by something else?

### Math-Shepherd

Current evidence:
1. `E40_MS_ACC90_LINEAR_ROBUST_SEED3`
   - `mean_heldout_pair_acc = 0.9172`
   - `mean_heldout_auc = 0.8623`
2. `E41_MS_ACC90_MLP_RANK_SEED3`
   - `mean_heldout_pair_acc = 0.9863`
   - `mean_heldout_auc = 0.9056`
3. `E42_MS_ACC90_MLP_JOINT_SEED3`
   - `mean_heldout_pair_acc = 0.9641`
   - `mean_heldout_auc = 0.9408`
4. `E43_MS_ACC90_MLP_HIGHCONF_SEED3`
   - `mean_heldout_pair_acc = 0.9619`
   - `mean_heldout_auc = 0.9425`

Interpretation:
1. `Math-Shepherd` does **not** support the claim that the linear head is too
   simple to solve the same-source pair task.
2. `MLP` still helps a lot, but here it is an upgrade over an already-working
   baseline, not a rescue from total underfitting.

### PRMBench Preview

Current evidence:
1. `E44_PRMBENCH_ACC90_LINEAR_SEED3`
   - `mean_heldout_pair_acc = 0.7380`
   - `mean_heldout_auc = 0.6782`
2. `E45_PRMBENCH_ACC90_MLP_RANK_SEED3`
   - `mean_heldout_pair_acc = 0.9315`
   - `mean_heldout_auc = 0.8711`
3. `E46_PRMBENCH_ACC90_MLP_JOINT_SEED3`
   - `mean_heldout_pair_acc = 0.9309`
   - `mean_heldout_auc = 0.9057`

Interpretation:
1. `PRMBench_Preview` is the clearest current evidence that the linear head can
   be too simple.
2. For this source, `MLP` is no longer optional for serious same-source
   fitting.

### Under-training control

`E12_MS_TRUST_LOWLR_SEED3` remained weak:
1. `mean_heldout_pair_acc = 0.5853`
2. `mean_heldout_auc = 0.5856`

Interpretation:
1. weak runs cannot be explained by "just train longer / lower the LR";
2. data semantics, objective choice, denoising, and head capacity all matter.

### Current design rule

1. `Math-Shepherd`
   - keep linear as a valid strong baseline;
   - keep `MLP` as the higher-ceiling same-source recipe family.
2. `PRMBench_Preview`
   - default to `MLP`;
   - do not treat linear as the main candidate anymore.
3. `R-PRM`
   - the first completed full matrix is now negative:
     - `E47`: `0.4374 / 0.5016`
     - `E48`: `0.5197 / 0.5123`
     - `E49`: `0.6002 / 0.5885`
   - interpretation:
     - `joint > mlp-rank > linear`, so recipe changes help,
     - but the source still stays far below the `ACC90` gate,
     - and the low seed std means this is systematic rather than a random seed collapse.
   - newer large-artifact probe on the repaired compact contract:
     - train-distribution eval with `MLP + joint + 2048`:
       - `pair_acc = 0.9090`
       - `auc = 0.9131`
     - matching true held-out eval:
       - `pair_acc = 0.6280`
       - `auc = 0.6508`
   - updated interpretation:
     - current `R-PRM compact` is learnable,
     - but the blocker has moved to held-out generalization under the current
       supervision contract,
     - so same-source `ACC90` is not a mere "train longer / bigger head"
       problem on this source.
4. Same-source `>90%` held-out accuracy proves a narrow claim only:
   - the dataset-local ranking problem is learnable under the current
     supervision semantics.
5. It does **not** prove:
   - benchmark transfer,
   - cross-dataset trust,
   - or RL-readiness.

## New Code

### Main value-head enhancement

`src/ours/phase_b/value_head.py`

New options:
1. `architecture=linear`
2. `architecture=mlp`
3. `mlp_hidden_size`
4. `activation`

This keeps old checkpoints compatible while allowing stronger same-source fitting.

### Training entrypoint

`scripts/phase_e_train_value.py`

New CLI knobs:
1. `--head-architecture`
2. `--head-dropout-prob`
3. `--head-init-std`
4. `--head-mlp-hidden-size`
5. `--head-activation`

### Same-source selector

`scripts/phase_e_select_intradataset_candidate.py`

This script only judges held-out same-source metrics and applies the ACC90 gate.

### Wrapper suite

`scripts/run_phase_e_intradataset_suite.sh`

This wrapper bundles dataset-specific ACC90 matrices and writes one candidate report.

## Direct Commands

### Quick smoke

```bash
ACTIVE_PHASE_E_INTRADATASET_GROUP=I1_INTRADATASET_SMOKE \
RUN_PREFIX=phase_e_intra_smoke_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_intradataset_suite.sh
```

### Math-Shepherd full ACC90 matrix

```bash
ACTIVE_PHASE_E_INTRADATASET_GROUP=I2_MS_ACC90_MATRIX \
RUN_PREFIX=phase_e_ms_acc90_full_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_intradataset_suite.sh
```

### PRMBench Preview full ACC90 matrix

```bash
ACTIVE_PHASE_E_INTRADATASET_GROUP=I3_PRMBENCH_ACC90_MATRIX \
RUN_PREFIX=phase_e_prmbench_acc90_full_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_e_intradataset_suite.sh
```

### R-PRM full ACC90 matrix

```bash
ACTIVE_PHASE_E_INTRADATASET_GROUP=I4_RPRM_ACC90_MATRIX \
RUN_PREFIX=phase_e_rprm_acc90_full_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_e_intradataset_suite.sh
```

### All datasets full ACC90 matrix

```bash
ACTIVE_PHASE_E_INTRADATASET_GROUP=I5_ALL_ACC90_MATRIX \
RUN_PREFIX=phase_e_all_acc90_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_intradataset_suite.sh
```

For fair comparison across `I2 / I3 / I4`:
1. do not reuse smoke-only overrides such as:
   - `SEEDS_OVERRIDE=42`
   - reduced pair caps
   - reduced epochs
2. keep each dataset's default recipe family intact,
3. keep the default full-suite batch size unless memory forces a smaller one.

## Selector fix note

The intradataset candidate selector originally mis-read repeated
`--suite-log-dirs` flags from the wrapper and could end up selecting only the
final group. That parsing path is now fixed in:
1. `scripts/phase_e_select_intradataset_candidate.py`

Practical implication:
1. old smoke candidate reports remain historical and invalid,
2. new full-matrix runs can trust the selector output again.

Related hardening:
1. `scripts/phase_e_select_candidate.py`
   - now also supports repeated `--suite-log-dirs` occurrences.
2. Phase E wrapper summaries no longer regex-parse `final_summary.md` for
   means:
   - `scripts/run_phase_e_intradataset_suite.sh`
   - `scripts/run_phase_e_single_source_suite.sh`
   - `scripts/run_phase_e_multisource_math_suite.sh`
3. These wrappers now aggregate directly from:
   - `seed_results.jsonl`

Reason:
1. `seed_results.jsonl` is the structured source-of-truth artifact,
2. `final_summary.md` is a human-facing rendering and should not be treated as
   the primary machine-readable contract.

## Interpretation Rule

This branch is intentionally narrow.

If one dataset reaches `>90%` on its own held-out split, the correct conclusion is:
- the current frozen-feature value-head stack is learnable on that dataset.

It does **not** imply:
- cross-dataset transfer,
- benchmark-native generalization,
- RL trustworthiness across tasks.

Those are later questions.
