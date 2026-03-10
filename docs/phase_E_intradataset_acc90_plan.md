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
RUN_PREFIX=phase_e_ms_acc90_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_intradataset_suite.sh
```

### PRMBench Preview full ACC90 matrix

```bash
ACTIVE_PHASE_E_INTRADATASET_GROUP=I3_PRMBENCH_ACC90_MATRIX \
RUN_PREFIX=phase_e_prmbench_acc90_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_intradataset_suite.sh
```

### R-PRM full ACC90 matrix

```bash
ACTIVE_PHASE_E_INTRADATASET_GROUP=I4_RPRM_ACC90_MATRIX \
RUN_PREFIX=phase_e_rprm_acc90_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_intradataset_suite.sh
```

### All datasets full ACC90 matrix

```bash
ACTIVE_PHASE_E_INTRADATASET_GROUP=I5_ALL_ACC90_MATRIX \
RUN_PREFIX=phase_e_all_acc90_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_intradataset_suite.sh
```

## Interpretation Rule

This branch is intentionally narrow.

If one dataset reaches `>90%` on its own held-out split, the correct conclusion is:
- the current frozen-feature value-head stack is learnable on that dataset.

It does **not** imply:
- cross-dataset transfer,
- benchmark-native generalization,
- RL trustworthiness across tasks.

Those are later questions.
