# OURS Research Project (Independent from external BCR code)

This repository is building an in-house reasoning-faithfulness pipeline from scratch.

Current milestone status (2026-02-28):
- Phase A is concluded.
- Phase B B0 (scope freeze) is completed.
- Phase B B1 code skeleton is implemented (smoke gate pending).
- Project execution focus is officially switched to Phase B.
- Phase C planning is now frozen in `phase_C_plan.md`.
- Phase C C0/C1 artifact-preparation code is implemented.
- Phase C P(IK) diagnostic branch is implemented end-to-end:
  - C1 question-level rollout artifacts (`scripts/phase_c_prepare_pik_data.py`),
  - C2 question-level value-head training (`scripts/phase_c_train_pik.py`),
  - standalone re-eval (`scripts/phase_c_eval_pik.py`),
  - lifecycle suite (`scripts/run_phase_c_pik_suite.sh`).
- Phase A core conclusion:
  - parse-error inflation was mostly protocol/extraction/free-form formatting related,
  - after binary-choice decode, StrategyQA parse error reaches `0.0000`,
  - remaining errors are model/prompt decision-quality gaps (not parser-only noise).
- Current StrategyQA decision-quality baseline (`binary_choice`, n=193):
  - `direct_binchoice`: `accuracy=0.6788`
  - `cot_binchoice`: `accuracy=0.5803`
- Batched freeform inference is now validated after padding fix:
  - StrategyQA direct (`max_new_tokens=32`, n=193):
  - `batch_size=1`: `accuracy=0.5492`, `parse_error_rate=0.1762`, `1.180 sample/s`
  - `batch_size=4`: `accuracy=0.5596`, `parse_error_rate=0.1658`, `4.189 sample/s`
- Phase B should start now as a separate training track (value head + BCR-lite), while keeping Phase A scripts frozen as benchmark references.
- Phase B execution plan and B0 freeze contract:
  - `phase_B_plan.md`

Primary roadmap:
- `TODO_ours.md`
- `phase_B_plan.md`

Context files:
- `idea_polish.md`
- `idea_formulation.md`
- `readme_dev.md`
- `result_records.md` (experiment history + diagnosis)
- `phase_A_report.md` (newcomer-facing Phase A closeout report)
- `phase_A_ppt_reference.md` (PPT-ready supervisor summary + glossary + numeric outcomes)
- `phase_B_plan.md` (Phase B lifecycle and first-run freeze contract)
- `phase_B_report.md` (live Phase B experiment report and diagnosis log)
- `phase_C_plan.md` (Phase C ABR/value-head implementation guidance)
- `foundation_reliability_audit.md` (low-level risk scan + hardening plan before Phase B scale)

## Phase C Entry Points (Current Implemented Scope)

Phase C is the first stage that implements the unique BCR/ABR stack rather than
plain PEFT tuning.

Current implemented scope:
- `C0`: freeze contracts, sequencing, and non-goals
- `C1`: build deterministic step prefixes, corruption artifacts, and optional
  rollout targets

Authoritative guidance:
- `phase_C_plan.md`

Main implementation files:
- `scripts/phase_b_prepare_value_data.py`
- `src/ours/phase_b/value_targets.py`
- `src/ours/phase_b/corruptions.py`

Why this layer exists:
- Phase B training rows are plain `(prompt_text, target_text)` records.
- ABR/BCR needs prefix states `h_t`, corrupted variants, and empirical prefix
  value targets.
- This layer exists specifically to prevent silent data-contract bugs before
  value-head or RL training begins.

Recommended first run: contract smoke only

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_b_prepare_value_data.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl \
  --run-name strategyqa_value_smoke \
  --max-samples 128 \
  --build-corruptions \
  --no-build-rollouts
```

Full prefix/corruption build without GPU rollouts:

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_b_prepare_value_data.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl \
  --run-name strategyqa_value_prefix_full \
  --build-corruptions \
  --no-build-rollouts
```

Rollout-target smoke build:

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/phase_b_prepare_value_data.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl \
  --run-name strategyqa_value_rollouts \
  --max-samples 256 \
  --build-corruptions \
  --build-rollouts \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --batch-size 64 \
  --rollout-count 4 \
  --max-new-tokens 96 \
  --temperature 0.7 \
  --top-p 0.95 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda
```

Rollout-target build on top of a finished StrategyQA PEFT adapter:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_b_prepare_value_data.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl \
  --run-name strategyqa_value_rollouts_r32 \
  --max-samples 256 \
  --build-corruptions \
  --build-rollouts \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --adapter-path assets/artifacts/phase_b_runs/<best_strategyqa_run>/final_model \
  --batch-size 64 \
  --rollout-count 4 \
  --max-new-tokens 96 \
  --temperature 0.7 \
  --top-p 0.95 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda
```

Replace `<best_strategyqa_run>` with the exact finished run directory before
execution.

Output layout:
- `assets/artifacts/phase_c_data/<dataset>/<run_name>__<fingerprint>/step_sequences.jsonl`
- `assets/artifacts/phase_c_data/<dataset>/<run_name>__<fingerprint>/prefixes.jsonl`
- `assets/artifacts/phase_c_data/<dataset>/<run_name>__<fingerprint>/errors.jsonl`
- optional:
  - `corruptions.jsonl`
  - `rollout_predictions.jsonl`
  - `rollout_targets.jsonl`
  - `corruption_rollout_targets.jsonl` (when `--build-pair-quality` is enabled)
  - `pair_quality.jsonl` (when `--build-pair-quality` is enabled)
- plus:
  - `manifest.json`
  - `summary.json`
  - `summary.md`

Contract checks after every run:
1. `errors.jsonl` should be small and understandable.
2. `prefixes.jsonl` should have unique `prefix_id`.
3. `corruptions.jsonl` should actually change the prefix text.
4. `rollout_targets.jsonl` should show non-trivial `success_rate` values when
   rollouts are enabled.
5. if pair-quality mode is enabled, `pair_quality.jsonl` should contain
   `delta_q`, `z_delta`, and `pair_weight` fields.
6. `summary.md` should match file counts on disk.

Validation commands:

```bash
python -m py_compile \
  src/ours/phase_b/value_targets.py \
  src/ours/phase_b/corruptions.py \
  scripts/phase_b_prepare_value_data.py
```

```bash
python -m pytest -q \
  tests/unit/test_phase_c_prepare_value.py \
  tests/unit/test_phase_c_value_components.py \
  tests/unit/test_phase_b_data.py \
  tests/unit/test_step_builder.py
```

## Phase C C2 Entry Points (Now Implemented)

Current implemented scope now includes:
- `C2`: frozen-backbone value-head training and standalone faithfulness eval.

Main implementation files:
- `scripts/phase_b_train_value.py`
- `scripts/phase_b_eval_faithfulness.py`
- `scripts/run_phase_c_value_suite.sh`
- `src/ours/phase_b/value_data.py`
- `src/ours/phase_b/value_head.py`
- `src/ours/phase_b/value_losses.py`
- `src/ours/phase_b/faithfulness_eval.py`
- `src/ours/phase_b/posthoc_calibration.py`

### C2 data prerequisites

You need two C1 artifact directories built with compatible contracts:
1. one for training prefixes/targets
2. one for held-out evaluation prefixes/targets

Typical pair:
- train: `assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts__<train_fp>`
- eval: `assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts_val__<eval_fp>`

### C2 training command

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/phase_b_train_value.py \
  --train-dir assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts__<train_fp> \
  --eval-dir assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts_val__<eval_fp> \
  --run-name strategyqa_value_c2_smoke \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --max-length 1024 \
  --per-device-train-batch-size 256 \
  --per-device-eval-batch-size 256 \
  --learning-rate 1e-3 \
  --num-train-epochs 5 \
  --use-contrastive-loss \
  --lambda-contrastive 1.0 \
  --contrastive-margin 0.1
```

### Top-three C2 tricks (new options)

The following options are now implemented and can be enabled/disabled directly:

1. Trick-1: calibration objective upgrade
- `--calibration-loss {mse,bce,bce_mse}`
- `--calibration-bce-pos-weight`
- `--calibration-mse-weight`, `--calibration-bce-weight` (for mixed mode)

2. Trick-2: post-hoc temperature calibration
- `--posthoc-calibration {none,temperature,isotonic}`
- `--checkpoint-selection-metric {raw_brier,posthoc_brier}`
- `--posthoc-temperature-lr`, `--posthoc-temperature-max-iters`
- `--posthoc-temperature-min`, `--posthoc-temperature-max`
- `--posthoc-isotonic-min-points`

3. Trick-3: adaptive calibration/contrastive balancing
- `--adaptive-loss-balancing {none,uncertainty}`
- `--adaptive-loss-init-log-variance`

4. Trick-4: confidence-aware calibration weighting
- `--calibration-sample-weighting {none,confidence,entropy_inverse,parseable,confidence_parseable,q_weight,q_weight_parseable}`
- `--calibration-weight-floor`
- `--calibration-weight-gamma`

5. Trick-5: contrastive pair filtering
- `--contrastive-pair-filter {none,confidence,parseable,confidence_parseable,label_quality,confidence_parseable_label}`
- `--contrastive-confidence-threshold`
- `--contrastive-parseable-threshold`
- `--contrastive-label-delta-q-min`
- `--contrastive-label-z-min`
- `--contrastive-label-pair-weight-min`
- `--contrastive-require-pair-pass-gate`
- `--contrastive-use-pair-weights`

6. Trick-6: calibration target smoothing
- `--calibration-target-smoothing` (epsilon in `[0, 0.5)`)

7. Trick-7: contrastive score-gap mining
- `--contrastive-score-gap-min`
- `--contrastive-score-gap-max`
- use a narrow band (for example `[0.0, 0.2]`) for hard-negative focus

Example (Trick-1 + Trick-2 together):

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/phase_b_train_value.py \
  --train-dir assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts_k8_train__<train_fp> \
  --eval-dir assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts_k8_val__<eval_fp> \
  --run-name strategyqa_value_c2_bce_temp \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --max-length 1024 \
  --per-device-train-batch-size 256 \
  --per-device-eval-batch-size 256 \
  --learning-rate 1e-4 \
  --num-train-epochs 8 \
  --calibration-loss bce \
  --calibration-bce-pos-weight 1.0 \
  --no-use-contrastive-loss \
  --posthoc-calibration temperature \
  --checkpoint-selection-metric posthoc_brier
```

Runtime note:
- After model shard loading, C2 now prints lightweight feature-cache progress
  (`cache_train_clean`, `cache_eval_clean`, `cache_eval_corruptions`) so long
  cache phases are visible without excessive log noise.

### C2 standalone evaluation command

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_b_eval_faithfulness.py \
  --value-run-dir assets/artifacts/phase_c_runs/strategyqa_value_c2_smoke_<timestamp> \
  --eval-dir assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts_val__<eval_fp> \
  --checkpoint-name best \
  --posthoc-calibration from_run \
  --run-name strategyqa_value_c2_eval
```

### One-command lifecycle suite

Use this wrapper when you want the full C1+C2 lifecycle in one reportable run:
- C1 train artifact build
- C1 eval artifact build
- C2 value-head training
- C2 standalone faithfulness eval
- consolidated suite summary

Smoke lifecycle:

```bash
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_c_strategyqa_smoke \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_value_suite.sh
```

Full lifecycle:

```bash
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_FULL \
RUN_PREFIX=phase_c_strategyqa_full \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_c_value_suite.sh
```

Trick-group lifecycle runs:

```bash
# Trick-1: BCE calibration objective
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_TRICK1_BCE \
RUN_PREFIX=phase_c_trick1_bce \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_value_suite.sh

# Trick-2: Post-hoc temperature scaling
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_TRICK2_POSTHOC_TEMP \
RUN_PREFIX=phase_c_trick2_temp \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_c_value_suite.sh

# Trick-3: Adaptive cal/contrastive balancing
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_TRICK3_ADAPTIVE_BALANCE \
RUN_PREFIX=phase_c_trick3_adaptive \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_c_value_suite.sh

# Trick-4: Isotonic post-hoc calibration
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_TRICK4_ISOTONIC \
RUN_PREFIX=phase_c_trick4_isotonic \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_c_value_suite.sh

# Trick-5: Confidence-weighted calibration
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_TRICK5_WEIGHTED_CAL \
RUN_PREFIX=phase_c_trick5_weighted \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_value_suite.sh

# Trick-6: Contrastive pair filtering
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_TRICK6_PAIR_FILTER \
RUN_PREFIX=phase_c_trick6_pair_filter \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_c_value_suite.sh

# Trick-7: Combined retry
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_TRICK7_COMBINED \
RUN_PREFIX=phase_c_trick7_combined \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_c_value_suite.sh

# Trick-8: Calibration target smoothing
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_TRICK8_LABEL_SMOOTH \
RUN_PREFIX=phase_c_trick8_label_smooth \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_c_value_suite.sh

# Trick-9: Hard-negative pair mining
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_TRICK9_HARD_NEG_MINING \
RUN_PREFIX=phase_c_trick9_hard_neg \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_value_suite.sh

# Trick-10: K16 + combined noise-control
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_TRICK10_K16_COMBINED \
RUN_PREFIX=phase_c_trick10_k16_combined \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_c_value_suite.sh

# Quality-first smoke (Q + pair-quality labels)
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_QUALITY_FIRST \
RUN_PREFIX=phase_c_quality_first \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_c_value_suite.sh

# Quality-first full (train/eval full coverage)
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_QUALITY_FIRST_FULL \
RUN_PREFIX=phase_c_quality_first_full \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_c_value_suite.sh
```

Supported groups:
1. `C2_STRATEGYQA_SMOKE`
2. `C2_STRATEGYQA_FULL`
3. `C2_STRATEGYQA_TRICK1_BCE`
4. `C2_STRATEGYQA_TRICK2_POSTHOC_TEMP`
5. `C2_STRATEGYQA_TRICK3_ADAPTIVE_BALANCE`
6. `C2_STRATEGYQA_TRICK4_ISOTONIC`
7. `C2_STRATEGYQA_TRICK5_WEIGHTED_CAL`
8. `C2_STRATEGYQA_TRICK6_PAIR_FILTER`
9. `C2_STRATEGYQA_TRICK7_COMBINED`
10. `C2_STRATEGYQA_TRICK8_LABEL_SMOOTH`
11. `C2_STRATEGYQA_TRICK9_HARD_NEG_MINING`
12. `C2_STRATEGYQA_TRICK10_K16_COMBINED`
13. `C2_STRATEGYQA_QUALITY_FIRST`
14. `C2_STRATEGYQA_QUALITY_FIRST_FULL`

Useful overrides:
- `TRAIN_MAX_SAMPLES`, `EVAL_MAX_SAMPLES`
- `ROLLOUT_BATCH_SIZE`, `ROLLOUT_COUNT`, `ROLLOUT_MAX_NEW_TOKENS`
- `C2_TRAIN_BATCH_SIZE`, `C2_EVAL_BATCH_SIZE`, `C2_LR`, `C2_EPOCHS`
- `C1_PREP_EXTRA_ARGS_DEFAULT`
- `C2_TRAIN_EXTRA_ARGS_DEFAULT`, `C2_EVAL_EXTRA_ARGS_DEFAULT`
- `PHASE_C_PREP_EXTRA_ARGS`, `PHASE_C_TRAIN_EXTRA_ARGS`, `PHASE_C_EVAL_EXTRA_ARGS`

### C2 outputs

Training run dir:
- `assets/artifacts/phase_c_runs/<run_name>_<timestamp>/`
- key files:
  - `best_value_head.pt` (if enabled)
  - `final_value_head.pt`
  - `best_posthoc_calibration.json` (if enabled)
  - `final_posthoc_calibration.json` (if enabled)
  - `value_head_config.json`
  - `train_metrics.json`
  - `eval_metrics.json`
  - `eval_prefix_scores.jsonl`
  - `eval_corruption_scores.jsonl`
  - `summary.json`
  - `summary.md`

Standalone eval run dir:
- `assets/artifacts/phase_c_eval/<run_name>_<timestamp>/`
- key files:
  - `metrics.json`
  - `prefix_scores.jsonl`
  - `corruption_scores.jsonl`
  - `summary.md`
  - note: `metrics.json` now includes both `calibration` (raw) and
    `calibration_posthoc` (when post-hoc mode is enabled)

### C2 current empirical status (StrategyQA, updated 2026-03-03)

Current completed C2 variants on the same held-out eval artifact:

| run | objective | brier | pearson | pair_acc | auc |
| --- | --- | ---: | ---: | ---: | ---: |
| `strategyqa_value_c2_smoke_20260302T215825Z` | calibration + strong contrastive (`lr=1e-3`, `lambda=1.0`) | 0.3681 | 0.0334 | 0.5082 | 0.5179 |
| `strategyqa_value_c2_cal_only_lr3e4_20260302T220411Z` | calibration only (`lr=3e-4`) | 0.2446 | 0.0232 | 0.4496 | 0.4130 |
| `strategyqa_value_c2_cal_plus_ctr_lr3e4_20260302T220429Z` | calibration + weak contrastive (`lr=3e-4`, `lambda=0.2`) | 0.2540 | -0.0002 | 0.2904 | 0.5460 |
| `strategyqa_value_c2_k8_cal_only_lr1e4_full_20260303T032550Z` | K=8 calibration-first (`lr=1e-4`, wd=0.01, dropout=0.1) | **0.1924** | **0.1915** | 0.4707 | 0.4765 |
| `strategyqa_value_c2_k8_ctr_l005_m002_20260303T064706Z` | K=8 + contrastive (`lambda=0.05`, margin=0.02) | 0.1949 | 0.1847 | **0.4988** | **0.4917** |
| `strategyqa_value_c2_k8_ctr_l002_m001_20260303T064742Z` | K=8 + weaker contrastive (`lambda=0.02`, margin=0.01) | 0.1948 | 0.1859 | 0.4895 | 0.4818 |
| `strategyqa_value_c2_k8_ctr_l001_m001_20260303T065230Z` | K=8 + even weaker (`lambda=0.01`, margin=0.01) | 0.1948 | 0.1862 | 0.4801 | 0.4786 |
| `strategyqa_value_c2_k8_ctr_l005_m0005_20260303T065233Z` | K=8 + tiny margin (`lambda=0.005`, margin=0.005) | 0.1948 | 0.1864 | 0.4801 | 0.4767 |

Reference baseline from the same eval set:
- old K=4 eval set baseline mean-predictor brier: `0.1510`
- K=8 eval set baseline mean-predictor brier: `0.1394`

Current interpretation:
1. C2 engineering pipeline is working end-to-end (train/eval artifacts and metrics are stable).
2. C2 quality is not yet sufficient for BCR/ABR routing:
   - best calibration (`brier=0.1924`) is still worse than baseline (`0.1394`),
   - correlation improved with K=8 (`pearson~0.19`) but corruption discrimination is still near random,
   - contrastive variants slightly improve corruption ordering but consistently degrade calibration.
3. Current Phase C bottleneck is no longer runtime reliability; it is objective/data signal quality.
4. C2 should continue with calibration-preserving improvements before starting O5.

### Why current C2 tries are not yet positive

1. We introduced contrastive because ABR/BCR needs local corruption sensitivity, not only average calibration.
2. Our runs confirm the expected tradeoff:
   - calibration-only gives the best Brier/Pearson,
   - contrastive gives slightly better corruption metrics.
3. However, both families still fail the practical promotion gate:
   - does not beat trivial baseline calibration,
   - corruption metrics remain close to random.
4. Therefore these are informative failed attempts, not invalid experiments:
   - they de-risk implementation,
   - they isolate the next research target to objective/data quality instead of engineering bugs.

### External references and what they imply for our C2 dilemma

- Process supervision is useful but noisy and data-hungry:
  - Let’s Verify Step by Step (PRM800K): https://arxiv.org/abs/2305.20050
  - Solving Math Word Problems With Process- and Outcome-Based Feedback: https://arxiv.org/abs/2211.14275
  - Tree-PLV (EMNLP 2024): https://aclanthology.org/2024.emnlp-main.125/
- Outcome-only labels often underdetermine step quality:
  - Do We Need to Verify Step by Step? (ICML 2025): https://proceedings.mlr.press/v267/jia25f.html
  - Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning: https://openreview.net/pdf?id=QerCdAGjyl
- Confidence/value calibration is non-trivial:
  - Language Models (Mostly) Know What They Know: https://arxiv.org/abs/2207.05221
  - On Calibration of Modern Neural Networks: https://proceedings.mlr.press/v70/guo17a.html
- Implementation levers already wired in this repo:
  - `BCEWithLogitsLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
  - post-hoc calibration basics: https://scikit-learn.org/stable/modules/calibration.html
  - adaptive multi-loss balancing:
    - GradNorm: https://proceedings.mlr.press/v80/chen18a.html
    - uncertainty weighting: https://arxiv.org/abs/1705.07115
  - generative PRM direction (later branch, not yet in C2 baseline path):
    - ThinkPRM: https://arxiv.org/abs/2504.16828

### What these tricks really are in our code path

1. Trick-A: probability-aware calibration objective
   - Replace MSE-only target fit with `BCE` or `BCE+MSE` on rollout-derived soft targets.
   - Goal: reduce overconfident raw scores and improve Brier/ECE before any routing.
2. Trick-B: post-hoc temperature scaling
   - Fit one scalar temperature on held-out logits and report both raw and post-hoc metrics.
   - Goal: improve calibration without changing the backbone/value-head weights.
3. Trick-C: adaptive calibration-vs-contrastive balancing
   - Use uncertainty-based weighting (or GradNorm-style balancing) instead of static `lambda`.
   - Goal: avoid the common failure where contrastive gains on corruption ordering destroy calibration.
4. Trick-D: margin-aware pair mining (next C1 upgrade)
   - Keep only clean/corrupt pairs with enough estimated Q-gap, downweight near-tie noisy pairs.
   - Goal: reduce ranking noise that causes pair metrics to stay near random.
5. Trick-E: staged verifier training
   - First make C2 calibration non-random; only then promote to rerank utility tests and C3/C4.
   - Goal: avoid Bellman/router training on an uninformative value head.

### Full-scale C2 commands (bs=256, GPUs 0/1/2/3)

1) Rebuild C1 train/eval artifacts with `K=8` and `batch_size=256`:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_b_prepare_value_data.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl \
  --run-name strategyqa_value_rollouts_k8_train_full_bs256 \
  --build-corruptions \
  --build-rollouts \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --batch-size 256 \
  --rollout-count 8 \
  --max-new-tokens 96 \
  --temperature 0.7 \
  --top-p 0.95 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda

CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_b_prepare_value_data.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl \
  --run-name strategyqa_value_rollouts_k8_eval_full_bs256 \
  --build-corruptions \
  --build-rollouts \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --batch-size 256 \
  --rollout-count 8 \
  --max-new-tokens 96 \
  --temperature 0.7 \
  --top-p 0.95 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda
```

2) Resolve latest completed artifact dirs:

```bash
TRAIN_DIR="$(python - <<'PY'
from pathlib import Path
base = Path('assets/artifacts/phase_c_data/strategyqa')
for d in sorted(base.glob('strategyqa_value_rollouts_k8_train_full_bs256__*'), reverse=True):
    if (d / 'manifest.json').exists() and (d / 'rollout_targets.jsonl').exists():
        print(d)
        raise SystemExit(0)
raise SystemExit('ERROR: no completed k8 train artifact found (missing manifest/rollout_targets)')
PY
)"

EVAL_DIR="$(python - <<'PY'
from pathlib import Path
base = Path('assets/artifacts/phase_c_data/strategyqa')
for d in sorted(base.glob('strategyqa_value_rollouts_k8_eval_full_bs256__*'), reverse=True):
    if (d / 'manifest.json').exists() and (d / 'rollout_targets.jsonl').exists():
        print(d)
        raise SystemExit(0)
raise SystemExit('ERROR: no completed k8 eval artifact found (missing manifest/rollout_targets)')
PY
)"
```

3) Launch four full-scale C2 variants (one GPU each, `bs=256`):

```bash
# GPU 0: MSE calibration anchor
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_b_train_value.py \
  --train-dir "$TRAIN_DIR" \
  --eval-dir "$EVAL_DIR" \
  --run-name c2_full_bs256_cal_mse \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --max-length 1024 \
  --per-device-train-batch-size 256 \
  --per-device-eval-batch-size 256 \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --dropout-prob 0.1 \
  --num-train-epochs 12 \
  --calibration-loss mse \
  --no-use-contrastive-loss

# GPU 1: BCE calibration
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_b_train_value.py \
  --train-dir "$TRAIN_DIR" \
  --eval-dir "$EVAL_DIR" \
  --run-name c2_full_bs256_cal_bce \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --max-length 1024 \
  --per-device-train-batch-size 256 \
  --per-device-eval-batch-size 256 \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --dropout-prob 0.1 \
  --num-train-epochs 12 \
  --calibration-loss bce \
  --calibration-bce-pos-weight 1.0 \
  --no-use-contrastive-loss

# GPU 2: MSE + post-hoc temperature
CUDA_VISIBLE_DEVICES=2 python -u scripts/phase_b_train_value.py \
  --train-dir "$TRAIN_DIR" \
  --eval-dir "$EVAL_DIR" \
  --run-name c2_full_bs256_cal_mse_posthoc \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --max-length 1024 \
  --per-device-train-batch-size 256 \
  --per-device-eval-batch-size 256 \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --dropout-prob 0.1 \
  --num-train-epochs 12 \
  --calibration-loss mse \
  --no-use-contrastive-loss \
  --posthoc-calibration temperature \
  --checkpoint-selection-metric posthoc_brier \
  --posthoc-temperature-lr 0.05 \
  --posthoc-temperature-max-iters 200

# GPU 3: BCE+MSE + adaptive contrastive + post-hoc
CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_b_train_value.py \
  --train-dir "$TRAIN_DIR" \
  --eval-dir "$EVAL_DIR" \
  --run-name c2_full_bs256_hybrid_adaptive \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --max-length 1024 \
  --per-device-train-batch-size 256 \
  --per-device-eval-batch-size 256 \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --dropout-prob 0.1 \
  --num-train-epochs 12 \
  --calibration-loss bce_mse \
  --calibration-bce-weight 1.0 \
  --calibration-mse-weight 1.0 \
  --use-contrastive-loss \
  --lambda-contrastive 0.05 \
  --contrastive-margin 0.02 \
  --adaptive-loss-balancing uncertainty \
  --adaptive-loss-init-log-variance 0.0 \
  --posthoc-calibration temperature \
  --checkpoint-selection-metric posthoc_brier \
  --posthoc-temperature-lr 0.05 \
  --posthoc-temperature-max-iters 200
```

4) Evaluate best checkpoints (`from_run` for post-hoc-enabled runs):

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_b_eval_faithfulness.py \
  --value-run-dir "$(ls -dt assets/artifacts/phase_c_runs/c2_full_bs256_cal_mse_* | head -n 1)" \
  --eval-dir "$EVAL_DIR" \
  --checkpoint-name best \
  --run-name c2_full_bs256_cal_mse_eval \
  --posthoc-calibration none

CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_b_eval_faithfulness.py \
  --value-run-dir "$(ls -dt assets/artifacts/phase_c_runs/c2_full_bs256_cal_bce_* | head -n 1)" \
  --eval-dir "$EVAL_DIR" \
  --checkpoint-name best \
  --run-name c2_full_bs256_cal_bce_eval \
  --posthoc-calibration none

CUDA_VISIBLE_DEVICES=2 python -u scripts/phase_b_eval_faithfulness.py \
  --value-run-dir "$(ls -dt assets/artifacts/phase_c_runs/c2_full_bs256_cal_mse_posthoc_* | head -n 1)" \
  --eval-dir "$EVAL_DIR" \
  --checkpoint-name best \
  --run-name c2_full_bs256_cal_mse_posthoc_eval \
  --posthoc-calibration from_run

CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_b_eval_faithfulness.py \
  --value-run-dir "$(ls -dt assets/artifacts/phase_c_runs/c2_full_bs256_hybrid_adaptive_* | head -n 1)" \
  --eval-dir "$EVAL_DIR" \
  --checkpoint-name best \
  --run-name c2_full_bs256_hybrid_adaptive_eval \
  --posthoc-calibration from_run
```

## Phase C P(IK) Bootstrap Path (New)

Why we switched to P(IK) now:
1. Prefix-level C2 runs were reproducible but near-random on corruption ranking
   (`pair_accuracy`/`AUC` around chance in most variants).
2. We needed a simpler, lower-noise diagnostic to answer one hard question:
   can the head learn any usable confidence signal at all?
3. P(IK) isolates that question by moving to question-level supervision:
   - input: one prompt,
   - target: empirical success rate from `K` sampled answers.

What this branch adds:
- `scripts/phase_c_prepare_pik_data.py`:
  - writes question-level C1 artifacts under `assets/artifacts/phase_c_pik_data/...`
- `scripts/phase_c_train_pik.py`:
  - trains a frozen-backbone question-level head on `pik_targets.jsonl`
- `scripts/phase_c_eval_pik.py`:
  - standalone re-evaluation with optional post-hoc calibration
- `scripts/run_phase_c_pik_suite.sh`:
  - one-command lifecycle (`C1 train + C1 eval + C2 train + C2 eval`)

One-command smoke run:

```bash
ACTIVE_PHASE_C_PIK_GROUP=PIK_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_c_pik_smoke \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_pik_suite.sh
```

One-command full StrategyQA run:

```bash
ACTIVE_PHASE_C_PIK_GROUP=PIK_STRATEGYQA_FULL \
RUN_PREFIX=phase_c_pik_full \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_c_pik_suite.sh
```

Standalone C2 eval from a finished P(IK) run:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_c_eval_pik.py \
  --value-run-dir assets/artifacts/phase_c_pik_runs/<pik_c2_run_dir> \
  --eval-dir assets/artifacts/phase_c_pik_data/strategyqa/<pik_eval_dir> \
  --checkpoint-name best \
  --posthoc-calibration from_run \
  --run-name strategyqa_pik_eval
```

Quality/safety checks:

```bash
python -m py_compile \
  scripts/phase_c_prepare_pik_data.py \
  scripts/phase_c_train_pik.py \
  scripts/phase_c_eval_pik.py \
  src/ours/phase_b/pik_data.py
```

```bash
python -m pytest -q \
  tests/unit/test_phase_c_pik_components.py \
  tests/unit/test_phase_c_prepare_pik.py
```

Current non-goals:
- Bellman-coupled BCR-lite training,
- ABR router training,
- RL.

Those remain intentionally deferred until C2 metrics are stable.

## Phase A Retrospective (Before You Start Phase B)

Treat these as operational lessons from the full Phase A cycle:
1. Evaluation logic is part of the experiment. Keep evaluator/version fixed when comparing runs.
2. Always read metrics as a tuple:
   - `accuracy`,
   - `parse_error_rate`,
   - `n_parseable`,
   - `acc_parseable`.
3. For StrategyQA, maintain both tracks:
   - `binary_choice` for decision quality,
   - `freeform` for end-to-end compliance.
4. Batching is now the default speed path, but trust it only with correctness guards:
   - left-padding in batch decode for decoder-only models,
   - deterministic parity checks vs batch size 1.
5. For math tasks, watch truncation/extraction diagnostics before interpreting accuracy:
   - cap hit rate,
   - final-answer tag rate,
   - fallback extraction rate.

## Documentation Convention (Active from 2026-02-28)

This repo is now being documented for novice-readability by default.

Required style for maintained runtime files:
- every `py` and `sh` file should begin with a short abstract explaining:
  - why the file exists,
  - what it contains,
  - how control flows through it,
  - how it interacts with other files.
- every function and class should have a docstring or nearby explanatory comment that covers:
  - purpose,
  - important inputs/outputs,
  - safety/edge-case behavior when relevant,
  - at least one short usage example when practical.

Current first-pass coverage focus:
- active Phase B runtime path,
- shared files directly used by current Phase B work.

Completed first-pass runtime coverage so far:
- Phase B:
  - `scripts/phase_b_train_sft.py`
  - `scripts/phase_b_eval.py`
  - `scripts/run_phase_b_training_suite.sh`
  - `src/ours/phase_b/__init__.py`
  - `src/ours/phase_b/contracts.py`
  - `src/ours/phase_b/data.py`
- Phase A:
  - `scripts/phase_a_generate_and_eval.py`
  - `scripts/phase_a_prepare.py`
  - `scripts/phase_a_analyze_instability.py`
  - `scripts/run_phase_a_benchmark_suite.sh`
  - `src/ours/phase_a/__init__.py`
  - `src/ours/phase_a/prompt_builder.py`
  - `src/ours/phase_a/contracts.py`
  - `src/ours/phase_a/answer_extraction.py`
  - `src/ours/phase_a/evaluator.py`
  - `src/ours/phase_a/instability.py`
  - `src/ours/phase_a/splitting.py`
- Shared/support files:
  - `download_datasets.sh`
  - `check_gsm8k.py`
  - `scripts/check_data.py`
  - `scripts/phase_a_eval_predictions.py`
  - `scripts/preprocess_steps.py`
  - `src/ours/__init__.py`
  - `src/ours/data/schema.py`

Useful maintenance commands:

```bash
python -m py_compile \
  scripts/phase_b_train_sft.py \
  scripts/phase_b_eval.py \
  src/ours/phase_b/contracts.py \
  src/ours/phase_b/data.py \
  src/ours/phase_a/prompt_builder.py

bash -n scripts/run_phase_b_training_suite.sh

python -m pytest -q \
  tests/unit/test_phase_b_train_script.py \
  tests/unit/test_phase_b_data.py
```

## 1. Model files

Please manually place the 4 model shard files in:

`assets/models/Qwen2.5-7B-Instruct`

Expected filenames:
- `model-00001-of-00004.safetensors`
- `model-00002-of-00004.safetensors`
- `model-00003-of-00004.safetensors`
- `model-00004-of-00004.safetensors`

## 2. Dataset downloads

Use:

```bash
bash download_datasets.sh
```

or:

```bash
bash download_datasets.sh assets/datasets
```

Why this script exists:
- it uses valid, namespaced HF dataset IDs
- it includes fallbacks for datasets with moved/disabled repos

## 3. Quick Start (Novice-Friendly)

```bash
cd /home/zling/y/bcr/ref
conda activate bcr
export HF_HOME=$PWD/assets/hf_cache
export HF_DATASETS_CACHE=$PWD/assets/hf_cache/datasets
export PYTHONPATH=$PWD/src
```

### 3.1 Run data checks (first priority)

Core datasets first:

```bash
python scripts/check_data.py --datasets gsm8k strategyqa --split train --limit 5
```

Additional datasets:

```bash
python scripts/check_data.py --datasets drop proofwriter --split train --limit 5
python scripts/check_data.py --datasets bbh --split test --limit 5 --bbh-task boolean_expressions
python scripts/check_data.py --datasets hendrycks_math --split train --limit 5 --hendrycks-subset algebra
python scripts/check_data.py --datasets logiqa --split train --limit 5
```

Expected success signal:
- `Result: SUCCESS`
- non-empty `question` and `answer` counts
- sensible sample preview

### 3.2 Run tests

Preferred command:

```bash
python -m pytest -q tests/unit/test_data_schema.py tests/integration/test_data_loaders.py
```

If you see `No module named pytest` in `bcr` environment:

```bash
python -m pip install -U pytest
```

## 4. Implemented Data Pipeline Files

- `src/ours/data/schema.py`: canonical sample contract + validation
- `src/ours/data/loaders.py`: per-dataset normalization to canonical schema
- `src/ours/data/step_builder.py`: deterministic canonical-sample -> step-sequence builder
- `src/ours/phase_a/`: Stage A baseline package (prompting, split policy, extraction, evaluator)
- `scripts/check_data.py`: CLI for readiness checks and sample previews
- `scripts/preprocess_steps.py`: batch preprocessing CLI that writes reusable artifacts
- `scripts/phase_a_prepare.py`: build model-ready Stage A prompt/target records
- `scripts/phase_a_eval_predictions.py`: evaluate model predictions with dataset-aware answer extraction
- `tests/unit/test_data_schema.py`: schema unit tests
- `tests/integration/test_data_loaders.py`: loader smoke tests
- `tests/unit/test_step_builder.py`: step-builder behavior tests
- `tests/unit/test_phase_a_*.py`: Stage A prompt/split/extraction/evaluator tests

## 5. Common Failure Cases

1. `Repository not found` on HF:
- Usually wrong dataset repo ID (use `download_datasets.sh`).

2. `403 DisabledRepoError` for old competition math repo:
- Use `EleutherAI/hendrycks_math` fallback (already in script).

3. `pyarrow` / `datasets` runtime mismatch:
- Activate a clean env (recommended: `bcr`) and reinstall compatible versions.

4. Permission error under default HF cache:
- Ensure `HF_HOME` and `HF_DATASETS_CACHE` point to writable project paths.

## 6. What to do next

Next milestone:
- run step-level preprocessing to generate reusable artifacts,
- then move to baseline SFT/value training.

### 6.1 Build step-level artifacts

Use this command for a first smoke run:

```bash
python scripts/preprocess_steps.py \
  --datasets gsm8k strategyqa \
  --split train \
  --limit 200 \
  --batch-size 64
```

Outputs are written under:

`assets/artifacts/steps/<dataset>/<split__variant__fingerprint>/`

Key files:
- `sample_sequences.jsonl`: one JSON object per sample, containing full step sequence.
- `flat_steps.jsonl`: one JSON object per step (easy for direct model ingestion).
- `summary.json`: aggregate counters and averages for the run.
- `manifest.json`: exact run configuration and deterministic fingerprint.
- `errors.jsonl`: sample-level preprocessing failures (empty if all good).

Resume behavior:
- rerunning the same command will reuse existing artifacts (`--resume` default true).
- use `--overwrite` to force regeneration.

### 6.2 How to read preprocessing outputs

After one run, inspect:

```bash
ls -lah assets/artifacts/steps/strategyqa
ls -lah assets/artifacts/steps/strategyqa/<run_folder>
```

Then verify content:

```bash
sed -n '1,2p' assets/artifacts/steps/strategyqa/<run_folder>/sample_sequences.jsonl
sed -n '1,20p' assets/artifacts/steps/strategyqa/<run_folder>/summary.json
sed -n '1,20p' assets/artifacts/steps/strategyqa/<run_folder>/manifest.json
```

What each file means:
- `sample_sequences.jsonl`: one line per original sample, containing ordered steps.
- `flat_steps.jsonl`: one line per step across all samples (easy for step-level model input).
- `summary.json`: counts (samples, steps, role distribution, averages).
- `manifest.json`: exact run config and fingerprint for reproducibility.
- `errors.jsonl`: per-sample failures (should be empty for clean runs).

## 7. Change History (Novice Digest)

This section summarizes the most recent engineering changes in plain language.

### 7.1 Loader improvements
- DROP and ProofWriter now include context in the canonical `question` field:
  - DROP: `Passage: ...` + `Question: ...`
  - ProofWriter: `Theory: ...` + `Question: ...`
- Hendrycks MATH now separates:
  - `cot`: full worked solution
  - `answer`: extracted final answer (prefer last `\\boxed{...}`)

Why this matters:
- Better training signal quality for reasoning tasks.
- Less information loss from dataset-specific formats.

### 7.2 New step preprocessing pipeline
- Added step builder with deterministic step IDs and configurable splitting:
  - split modes: `auto`, `newline`, `sentence`
  - optional question step 0
  - optional answer terminal step
- Added preprocessing CLI that writes reusable artifacts and supports:
  - `--resume` (default reuse)
  - `--overwrite` (force regeneration)
  - `--strict` (fail fast on bad samples)

Why this matters:
- You preprocess once and reuse outputs across experiments.
- You can compare experiments fairly because run fingerprints are stable.

## 8. Recommended Next Commands

1. Run a tiny smoke test:
```bash
python scripts/preprocess_steps.py --datasets strategyqa --split train --limit 20 --batch-size 8
```

2. Run broader preprocessing:
```bash
python scripts/preprocess_steps.py --datasets gsm8k strategyqa drop proofwriter --split train --limit 2000 --batch-size 128
```

3. Run tests to confirm pipeline stability:
```bash
python -m pytest -q
```

## 9. Stage A Baseline (New)

Stage A objective:
- build a reproducible baseline harness before BCR/ABR-specific training.

This repo now implements Stage A foundations:
- versioned prompt templates,
- deterministic split policy,
- dataset-aware answer extraction,
- robust evaluator that does more than raw `==`.

### 9.1 Prepare Stage A artifacts

Example (StrategyQA, hash split, answer-only target):

```bash
python scripts/phase_a_prepare.py \
  --datasets strategyqa \
  --source-split train \
  --split-policy hash \
  --target-style answer_only \
  --template-id qa_direct \
  --template-version 1.0.0 \
  --limit 200
```

Output folder pattern:
- `assets/artifacts/phase_a_prepared/<dataset>/<run_fingerprint>/`

Files created:
- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `summary.json`
- `manifest.json`

Notes:
- `--resume` is on by default (reuse matching run).
- use `--overwrite` to force regeneration.

### 9.2 Evaluate prediction JSONL files

Input JSONL must include:
- `sample_id`
- `dataset`
- `split`
- `raw_prediction`
- `gold_answer`

Run:

```bash
python scripts/phase_a_eval_predictions.py \
  --predictions <path_to_predictions.jsonl> \
  --run-name my_eval
```

Output:
- `assets/artifacts/phase_a_eval/<run_name_timestamp>/scored_predictions.jsonl`
- `assets/artifacts/phase_a_eval/<run_name_timestamp>/metrics.json`

### 9.3 Which template/target pair to start with

Recommended first baseline:
1. `--template-id qa_direct --target-style answer_only`
2. Then compare against:
`--template-id qa_cot_then_final --target-style cot_then_answer`

This A/B comparison is your Stage A baseline matrix.

Dataset-specific note:
- For GSM8K direct-answer baselines, prefer:
  - `--template-id qa_math_direct_final --target-style answer_only`
  - this avoids weak `last_number` extraction from truncated reasoning outputs.

### 9.4 Why this milestone is critical

- You now have reproducible prompt+target generation.
- You now have deterministic data splitting.
- You now have extraction-aware evaluation for natural-language outputs.
- These are prerequisites before adding value heads or ABR router logic.

### 9.5 One-Script Inference + Eval + Diff

Use:

```bash
python scripts/phase_a_generate_and_eval.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/21095d3c688a/validation.jsonl \
  --run-name qwen_strategyqa_val \
  --require-cuda \
  --no-do-sample \
  --max-new-tokens 64 \
  --log-every 10
```

What this script writes:
- `assets/artifacts/phase_a_runs/<run_name_timestamp>/predictions.jsonl`
- `assets/artifacts/phase_a_runs/<run_name_timestamp>/scored_predictions.jsonl`
- `assets/artifacts/phase_a_runs/<run_name_timestamp>/metrics.json`
- `assets/artifacts/phase_a_runs/<run_name_timestamp>/manifest.json`
- `assets/artifacts/phase_a_runs/<run_name_timestamp>/console.log`

Rerun the same experiment to compare differences:
1. Run the same command again (same `--input-jsonl`, `--run-name`, and generation settings).
2. The script auto-compares with the latest previous run of the same `run_name`.
3. It reports:
   - `delta_accuracy`
   - `delta_parse_err`
   - `changed_samples`

Notes for reproducibility:
- keep `--no-do-sample` for deterministic baseline behavior.
- keep `--seed` fixed (default `42`).
- keep input file and generation parameters unchanged.

Notes for long runs:
- use `python -u ...` to force unbuffered console output.
- `--log-every` controls candidate checkpoint stride.
- `--max-progress-lines` caps generation progress verbosity (default `5` lines including start line).
- use `--require-cuda` for strict benchmarking so the script aborts instead of silently using CPU.
- console output is persisted by default in each run folder as `console.log`.
- disable console-log persistence only if needed: add `--no-persist-console-log`.
- model loading progress bars are suppressed; logs show concise load start + completion time.
- `--strategyqa-decode-mode` controls StrategyQA decoding:
  - `freeform` (default): normal text generation,
  - `binary_choice`: score `yes` vs `no` directly (removes most format-related parse errors).
- `--batch-size` controls free-form decode batching:
  - `1` keeps old behavior (safest baseline),
  - `2/4/8` can improve throughput when VRAM allows.
- `--oom-backoff` (default on) auto-splits a failing batch on OOM:
  - useful for robust sweeps on shared servers,
  - turn off with `--no-oom-backoff` for strict failure behavior.
- `--truncate-chat-markers` (default on) trims leaked next-turn markers such as `[USER]` / `Human:`.
- truncation recovery (default on for math datasets):
  - `--truncation-recovery` / `--no-truncation-recovery`,
  - `--truncation-recovery-rounds`,
  - `--truncation-recovery-extra-tokens`,
  - `--truncation-recovery-datasets`,
  - `--truncation-recovery-require-final-answer-signal`.
  - behavior: if a sample hits token cap and still lacks final-answer signal, script auto-runs continuation rounds.
- For math datasets, script output now includes `math_diag` (extraction reliability signals):
  - `last_number_rate`,
  - `hit_token_limit_rate`,
  - `final_answer_tag_rate`.
- script output now also prints and stores generation throughput:
  - `gen_elapsed_sec`,
  - `gen_sample_rate`,
  - `oom_backoff_evts`.
- script now prints and stores VRAM telemetry (sampled during generation):
  - `vram_mean_gib` (total reserved, sampled),
  - `vram_max_gib` (total reserved, sampled),
  - plus per-device peak stats under `metrics.json -> generation_stats.vram_per_device`.
- while running, you can monitor file progress:

```bash
wc -l assets/artifacts/phase_a_runs/<run_name_timestamp>/predictions.jsonl
```

### 9.6 One-Command Benchmark Suite (Prepare + Inference + Diagnostics)

Use:

```bash
bash scripts/run_phase_a_benchmark_suite.sh
```

The script now supports **param groups** (`A1`~`A12`, plus `A11_*` token-stress variants) for one-click experiment presets.

One-click switching:
1. Open `scripts/run_phase_a_benchmark_suite.sh`.
2. Change `ACTIVE_PARAM_GROUP` (for example from `A1` to `A2`).
3. Save and rerun the same script.

Group intent summary:
- `A1`: core reproduction (`direct_t32` repeat + `cot_t128` + `cot_t256`)
- `A2`: CoT token sweep (truncation/compliance diagnosis)
- `A3`: direct token sweep (speed/accuracy frontier)
- `A4`: determinism check (repeat same config and compare deltas)
- `A5`: strict yes/no compliance fix (`qa_binary_strict`) to reduce parse errors and token waste
- `A6`: binary-choice decode validation (removes free-form format noise for StrategyQA)
- `A7`: StrategyQA prompt-style sweep (3 template styles)
- `A8`: GSM8K prompt-style sweep (3 template styles)
- `A9`: StrategyQA full-data best-setting run (`qa_strategyqa_cot_compact`, `t96`, reproducibility pair)
- `A10`: GSM8K full-data best-setting run (`qa_gsm8k_cot_compact_final`, `t192`, reproducibility pair)
- `A11`: StrategyQA whole-corpus review over train+validation+test with weighted aggregate output
- `A11_128`, `A11_256`, `A11_384`, `A11_512`, `A11_1024`:
  StrategyQA whole-corpus token-stress subgroups (same A11 setup, larger decode budgets)
- `A12`: GSM8K whole-corpus review over train+validation+test with weighted aggregate output

Template styles used in `A7` (StrategyQA):
1. `qa_strategyqa_minimal_binary`
2. `qa_strategyqa_cot_compact`
3. `qa_strategyqa_evidence_verdict`

Template styles used in `A8` (GSM8K):
1. `qa_gsm8k_direct_final_only`
2. `qa_gsm8k_cot_compact_final`
3. `qa_gsm8k_equation_then_final`

One-click shell commands for these new groups:

```bash
# StrategyQA: 3-style prompt sweep
ACTIVE_PARAM_GROUP=A7 \
CUDA_VISIBLE_DEVICES=0 \
RUN_PREFIX=strategyqa_style_sweep \
bash scripts/run_phase_a_benchmark_suite.sh

# GSM8K: 3-style prompt sweep
ACTIVE_PARAM_GROUP=A8 \
CUDA_VISIBLE_DEVICES=0 \
RUN_PREFIX=gsm8k_style_sweep \
bash scripts/run_phase_a_benchmark_suite.sh

# StrategyQA: full-data best setting (A7 winner)
ACTIVE_PARAM_GROUP=A9 \
CUDA_VISIBLE_DEVICES=0 \
RUN_PREFIX=strategyqa_full_best \
bash scripts/run_phase_a_benchmark_suite.sh

# GSM8K: full-data best setting (A8 winner + truncation recovery)
ACTIVE_PARAM_GROUP=A10 \
CUDA_VISIBLE_DEVICES=0 \
RUN_PREFIX=gsm8k_full_best \
bash scripts/run_phase_a_benchmark_suite.sh

# StrategyQA: whole-corpus review (train+validation+test aggregate)
ACTIVE_PARAM_GROUP=A11 \
CUDA_VISIBLE_DEVICES=0 \
RUN_PREFIX=strategyqa_whole_2290 \
bash scripts/run_phase_a_benchmark_suite.sh

# StrategyQA: token-stress subgroup examples on whole corpus
ACTIVE_PARAM_GROUP=A11_128  CUDA_VISIBLE_DEVICES=0 RUN_PREFIX=strategyqa_whole_t128  bash scripts/run_phase_a_benchmark_suite.sh
ACTIVE_PARAM_GROUP=A11_256  CUDA_VISIBLE_DEVICES=0 RUN_PREFIX=strategyqa_whole_t256  bash scripts/run_phase_a_benchmark_suite.sh
ACTIVE_PARAM_GROUP=A11_384  CUDA_VISIBLE_DEVICES=0 RUN_PREFIX=strategyqa_whole_t384  bash scripts/run_phase_a_benchmark_suite.sh
ACTIVE_PARAM_GROUP=A11_512  CUDA_VISIBLE_DEVICES=0 RUN_PREFIX=strategyqa_whole_t512  bash scripts/run_phase_a_benchmark_suite.sh
ACTIVE_PARAM_GROUP=A11_1024 CUDA_VISIBLE_DEVICES=0 RUN_PREFIX=strategyqa_whole_t1024 bash scripts/run_phase_a_benchmark_suite.sh

# GSM8K: whole-corpus review (train+validation+test aggregate)
ACTIVE_PARAM_GROUP=A12 \
CUDA_VISIBLE_DEVICES=0 \
RUN_PREFIX=gsm8k_whole_corpus \
bash scripts/run_phase_a_benchmark_suite.sh
```

For `A11`, the final summary includes `WHOLE-CORPUS AGGREGATE` over the primary
`train/validation/test` runs only (reproducibility reruns are excluded from the aggregate).

For `A11`/`A12`, runtime knobs passed via env (for example `BATCH_SIZE`, `OOM_BACKOFF`,
`TRUNCATION_RECOVERY_*`) take precedence over group defaults.

For long whole-corpus runs, `A11`/`A12` now default to `MAX_PROGRESS_LINES=50`
(unless you set `MAX_PROGRESS_LINES` explicitly via env).

For `A11_*` token-stress variants, default batch size is reduced as token budget grows:
- `A11_128`: batch `64`
- `A11_256`: batch `32`
- `A11_384`: batch `24`
- `A11_512`: batch `16`
- `A11_1024`: batch `8`
You can still override with `BATCH_SIZE=...` when your VRAM allows.

Each group prints:
- intention,
- what to observe,
- expected trend,
- and a summary table with `accuracy`, `parse_error_rate`, `acc_parseable`, plus delta fields when comparison is enabled.

New in-suite artifact diagnostics:
- final summary now includes an `INSTABILITY INDICATORS (ARTIFACT ANALYSIS)` block
  directly under `RESULT TABLE`,
- and a `PAIRWISE FLIP ANALYSIS` block when run pairs share sample IDs.

Indicator meanings:
- `tagged`: fraction of rows containing at least one `Final answer` yes/no tag.
- `multi_tag`: fraction of rows containing 2+ yes/no final-answer tags.
- `first_last_flip`: fraction where first and last final-answer tags disagree.
- `tag_switch`: fraction where tag sequence flips at least once (`yes -> no` or `no -> yes`).
- `mean_tags`: average number of final-answer tags among tagged rows.

Standalone artifact-only analyzer (no new inference required):

```bash
# Analyze one run directory.
python scripts/phase_a_analyze_instability.py \
  --run-dirs assets/artifacts/phase_a_runs/strategyqa_whole_t384_full_train_t384_20260227T164745Z

# Analyze multiple runs and include pairwise flip rates.
python scripts/phase_a_analyze_instability.py \
  --run-dirs \
    assets/artifacts/phase_a_runs/strategyqa_whole_t128_full_train_t128_20260227T164223Z \
    assets/artifacts/phase_a_runs/strategyqa_whole_t256_full_train_t256_20260227T164403Z \
    assets/artifacts/phase_a_runs/strategyqa_whole_t384_full_train_t384_20260227T164745Z

# Save report artifacts for PPT/records.
python scripts/phase_a_analyze_instability.py \
  --run-dirs \
    assets/artifacts/phase_a_runs/strategyqa_whole_t128_full_train_t128_20260227T164223Z \
    assets/artifacts/phase_a_runs/strategyqa_whole_t384_full_train_t384_20260227T164745Z \
  --output-json assets/artifacts/phase_a_logs/instability_compare/summary.json \
  --output-markdown assets/artifacts/phase_a_logs/instability_compare/summary.md
```

Useful overrides (optional):

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_PREFIX=my_suite \
LIMIT=2000 \
COT_SWEEP_TOKENS="128 192 256 320 384" \
DIRECT_SWEEP_TOKENS="16 24 32 48 64" \
STRATEGYQA_DECODE_MODE=freeform \
TRUNCATE_CHAT_MARKERS=1 \
TRUNCATION_RECOVERY=1 \
TRUNCATION_RECOVERY_ROUNDS=2 \
TRUNCATION_RECOVERY_EXTRA_TOKENS=96 \
TRUNCATION_RECOVERY_DATASETS="gsm8k,hendrycks_math" \
TRUNCATION_RECOVERY_REQUIRE_FINAL_SIGNAL=1 \
LOG_EVERY=5 \
MAX_PROGRESS_LINES=5 \
BATCH_SIZE=1 \
OOM_BACKOFF=1 \
bash scripts/run_phase_a_benchmark_suite.sh
```

For full-dataset preparation/eval runs, use:
- `LIMIT=None` or `LIMIT=all` (both are supported).

Notes:
- The script no longer blocks concurrent runs.
- If concurrent runs are detected, it prints a warning and continues.
- JSONL parsing is robust to Unicode line separators (for example `U+2028`):
  - readers now use newline-stream iteration (not `splitlines()`),
  - this prevents false `JSONDecodeError` on valid records containing `U+2028` inside strings.
- `phase_a_prepare.py` now validates resumed JSONL artifacts:
  - if a matching run fingerprint exists but split JSONLs are invalid, the run is auto-regenerated.

If a long run fails after one split with a JSON decode issue, you can safely rerun the same group:

```bash
ACTIVE_PARAM_GROUP=A12 \
CUDA_VISIBLE_DEVICES=0 \
BATCH_SIZE=256 \
RUN_PREFIX=gsm8k_whole_corpus_b256 \
bash scripts/run_phase_a_benchmark_suite.sh
```

Quick JSONL health check (artifact-only):

```bash
python - <<'PY'
from pathlib import Path
import json
path = Path("assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/test.jsonl")
bad = 0
with path.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except Exception as e:
            bad += 1
            print("bad", i, e)
            break
print("status", "ok" if bad == 0 else "invalid")
PY
```

### 9.7 Truncation Recovery: Hands-On Fix Path

If GSM8K/CoT runs show high `hit_cap_rate` or frequent `last_number` fallback, run with stronger recovery:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl assets/artifacts/phase_a_prepared/gsm8k/09d73d23f451/validation.jsonl \
  --run-name gsm8k_cot_t192_trunc_fix \
  --require-cuda --dtype bfloat16 --device-map auto \
  --no-do-sample --seed 42 \
  --max-new-tokens 192 \
  --truncation-recovery \
  --truncation-recovery-rounds 2 \
  --truncation-recovery-extra-tokens 96 \
  --truncation-recovery-datasets gsm8k,hendrycks_math \
  --truncation-recovery-require-final-answer-signal \
  --batch-size 1 --oom-backoff \
  --log-every 5 --max-progress-lines 5 \
  --no-compare-latest-same-name
```

More aggressive variant (slower, stronger truncation guard):

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl assets/artifacts/phase_a_prepared/gsm8k/09d73d23f451/validation.jsonl \
  --run-name gsm8k_cot_t192_trunc_fix_aggr \
  --require-cuda --dtype bfloat16 --device-map auto \
  --no-do-sample --seed 42 \
  --max-new-tokens 192 \
  --truncation-recovery \
  --truncation-recovery-rounds 3 \
  --truncation-recovery-extra-tokens 128 \
  --truncation-recovery-datasets gsm8k,hendrycks_math \
  --truncation-recovery-require-final-answer-signal \
  --batch-size 1 --oom-backoff \
  --log-every 5 --max-progress-lines 5 \
  --no-compare-latest-same-name
```

What to compare in `metrics.json`:
1. `accuracy`,
2. `math_diagnostics.hit_token_limit_rate`,
3. `math_diagnostics.last_number_rate`,
4. `generation_stats.truncation_recovery_rows`,
5. `generation_stats.truncation_recovery_rounds`.

### 9.8 Batching Hands-On: How To See Speed Difference

Run the same config twice, only changing `--batch-size`.

Example (`batch_size=1` baseline):

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl \
  --run-name strat_batch1 \
  --require-cuda --dtype bfloat16 --device-map auto \
  --no-do-sample --seed 42 \
  --max-new-tokens 32 \
  --batch-size 1 \
  --oom-backoff \
  --log-every 10 --max-progress-lines 5 \
  --no-compare-latest-same-name
```

Then run batched:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl \
  --run-name strat_batch4 \
  --require-cuda --dtype bfloat16 --device-map auto \
  --no-do-sample --seed 42 \
  --max-new-tokens 32 \
  --batch-size 4 \
  --oom-backoff \
  --log-every 10 --max-progress-lines 5 \
  --no-compare-latest-same-name
```

Compare these fields in console or `metrics.json`:
1. `gen_sample_rate` (higher is better),
2. `gen_elapsed_sec` (lower is better),
3. `accuracy` / `parse_error_rate` (should stay comparable).
- If you see warning `decoder-only ... right-padding`, stop and fix tokenizer padding before trusting results.
- Current expected stable setup on A100 80G for StrategyQA direct:
  - `--batch-size 4 --oom-backoff`
  - quality parity with significant speedup.
- Default `CUDA_VISIBLE_DEVICES=0` is chosen for stable, simpler 7B benchmarking.
- Suite-level live logs are persisted to `assets/artifacts/phase_a_logs/<RUN_PREFIX>/suite.log`.
- At the end of a group run, it prints a **final summary block** (group + settings + table) and saves it to:
  - `assets/artifacts/phase_a_logs/<RUN_PREFIX>/final_summary.md`
- Disable suite log persistence if needed:

```bash
ENABLE_PERSISTED_LOGS=0 bash scripts/run_phase_a_benchmark_suite.sh
```

## 10. Phase B B1 Quick Start (SFT/PEFT Skeleton)

Phase B plan and lifecycle:
- `phase_B_plan.md`

This project currently uses a PEFT-first B1 training skeleton:
- trainer script: `scripts/phase_b_train_sft.py`
- evaluation bridge: `scripts/phase_b_eval.py`
- starter configs: `configs/phase_b/*.json`

Official transition note:
1. Starting now, Phase B is the default task stream.
2. Phase A runs are baseline/reference only unless explicitly requested for diagnostics.
3. New experiments for milestone progress should use Phase B groups first (`B1_SMOKE` then `B1_FIRST`).

### 10.1 Run a tiny smoke training job

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_b_train_sft.py \
  --config-json configs/phase_b/peft_smoke_strategyqa.json
```

Expected outputs:
- `assets/artifacts/phase_b_runs/<run_name_timestamp>/manifest.json`
- `assets/artifacts/phase_b_runs/<run_name_timestamp>/train_metrics.json`
- `assets/artifacts/phase_b_runs/<run_name_timestamp>/eval_metrics.json` (if validation provided)
- `assets/artifacts/phase_b_runs/<run_name_timestamp>/summary.json`
- `assets/artifacts/phase_b_runs/<run_name_timestamp>/final_model/` (if enabled)

### 10.2 Run first full candidate config

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_b_train_sft.py \
  --config-json configs/phase_b/peft_first_run_strategyqa.json
```

### 10.3 Evaluate a Phase B checkpoint with frozen Phase A protocol

Important:
- `train_metrics.json` and `eval_metrics.json` from Phase B training are trainer-loss
  metrics, not Phase A task-accuracy metrics.
- To measure benchmark gain, re-run the frozen Phase A evaluator on the trained
  Phase B artifact.
- For PEFT runs, do not point Phase A directly at `final_model/` as if it were a
  standalone full model; the bridge now resolves `base model + adapter` correctly.

Recommended StrategyQA evaluation after training:

```bash
# StrategyQA binary-choice decision-quality check
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_b_eval.py \
  --phase-b-run-dir assets/artifacts/phase_b_runs/<run_name_timestamp> \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl \
  --run-name phase_b_eval_strategyqa_binchoice \
  --strategyqa-decode-mode binary_choice \
  --max-new-tokens 16 \
  --batch-size 4

# StrategyQA end-to-end freeform check
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_b_eval.py \
  --phase-b-run-dir assets/artifacts/phase_b_runs/<run_name_timestamp> \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl \
  --run-name phase_b_eval_strategyqa_freeform \
  --strategyqa-decode-mode freeform \
  --max-new-tokens 32 \
  --batch-size 4
```

How the bridge resolves artifacts:
1. if the Phase B run was `peft`:
   - base model comes from `manifest.json:model_path`
   - adapter comes from `<run_dir>/final_model`
2. if the Phase B run was `sft`:
   - the bridge evaluates `<run_dir>/final_model` directly

### 10.4 Safety/Debug Notes

1. Default mode is PEFT; if `peft` import fails and fallback is enabled, script logs warning and continues in SFT mode.
2. Batching-first defaults are used (`per_device_train_batch_size=4`).
3. OOM safety for training is enabled via `--auto-find-batch-size` by default.
4. Keep `--seed` fixed when comparing run behavior.
5. `TrainingArguments` compatibility is version-tolerant:
   - script now filters kwargs by runtime signature (older/newer transformers won’t crash on unknown args).
6. Model loading uses version-aware dtype arg (`dtype` vs `torch_dtype`) to avoid deprecation-only failures.

### 10.4.1 Conda Environment Repair (Phase B)

If you see errors like:
- `No module named 'peft'`
- `TrainingArguments.__init__() got an unexpected keyword argument ...`
- transformers/hub version conflicts

run these in your `bcr` env:

```bash
conda activate bcr
python -m pip install -U \
  "transformers>=4.44,<5" \
  "huggingface-hub>=0.23.2,<1.0" \
  "accelerate>=1.1,<2" \
  "peft>=0.11,<0.14" \
  "datasets>=2.20,<3" \
  "safetensors>=0.4.3"
```

Quick sanity check:

```bash
python - <<'PY'
import transformers, huggingface_hub, accelerate
print("transformers", transformers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("accelerate", accelerate.__version__)
try:
    import peft
    print("peft", peft.__version__)
except Exception as e:
    print("peft import failed:", e)
PY
```

### 10.5 One-Click Phase B Param Groups

Use:

```bash
bash scripts/run_phase_b_training_suite.sh
```

Supported groups:
1. `B1_SMOKE` (default)
2. `B1_FIRST`
3. `B2_STRATEGYQA_FULL`
4. `B2_STRATEGYQA_DIAG_EPOCH_200`
5. `B2_STRATEGYQA_DIAG_EPOCH_300`
6. `B2_STRATEGYQA_DIAG_LORA_R8`
7. `B2_STRATEGYQA_DIAG_LORA_R32`
8. `B2_GSM8K_FULL`
9. `B2_GSM8K_DIAG_LR_5E5`
10. `B2_GSM8K_DIAG_LR_1E4`
11. `B2_GSM8K_DIAG_EPOCH_025`
12. `B2_GSM8K_DIAG_EPOCH_050`
13. `B2_GSM8K_DIAG_DIRECT_STYLE`
14. `B2_GSM8K_DIAG_EQUATION_STYLE`
15. `B2_GSM8K_DIAG_CHECKPOINT_SWEEP`
16. `B2_GSM8K_DIAG_SHORT_COT`
17. `B2_GSM8K_DIAG_ANSWER_WEIGHTED`

Switch groups with env var:

```bash
ACTIVE_PHASE_B_GROUP=B1_FIRST RUN_PREFIX=phase_b_first bash scripts/run_phase_b_training_suite.sh
```

Operational note for heavy groups:
- avoid launching multiple full-dataset Phase B suites on the same GPU at the same time,
- the suite now writes a partial `final_summary.md` with `status: failed` and `failed_stage` if it exits early,
- if you only see `pre_*` Phase A eval runs and no Phase B run directory, that experiment did not reach training and should be treated as aborted.
- user-environment command preference:
  - avoid `CUDA_VISIBLE_DEVICES=0` by default when suggesting future commands,
  - prefer eval batch sizes `>=64` when memory allows and no known risk requires a smaller batch.

Full-dataset gain runs:

```bash
ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_FULL \
RUN_PREFIX=phase_b_strategyqa_full \
CUDA_VISIBLE_DEVICES=1 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh

ACTIVE_PHASE_B_GROUP=B2_GSM8K_FULL \
RUN_PREFIX=phase_b_gsm8k_full \
CUDA_VISIBLE_DEVICES=2 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh
```

StrategyQA scaling suite:

```bash
ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_DIAG_EPOCH_200 \
RUN_PREFIX=strategyqa_diag_e200 \
CUDA_VISIBLE_DEVICES=1 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh

ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_DIAG_EPOCH_300 \
RUN_PREFIX=strategyqa_diag_e300 \
CUDA_VISIBLE_DEVICES=2 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh

ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_DIAG_LORA_R8 \
RUN_PREFIX=strategyqa_diag_r8 \
CUDA_VISIBLE_DEVICES=1 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh

ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_DIAG_LORA_R32 \
RUN_PREFIX=strategyqa_diag_r32 \
CUDA_VISIBLE_DEVICES=2 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh
```

GSM8K diagnostic suite:

```bash
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_LR_5E5 \
RUN_PREFIX=gsm8k_diag_lr5e5 \
CUDA_VISIBLE_DEVICES=1 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh

ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_LR_1E4 \
RUN_PREFIX=gsm8k_diag_lr1e4 \
CUDA_VISIBLE_DEVICES=2 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh

ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_EPOCH_025 \
RUN_PREFIX=gsm8k_diag_e025 \
CUDA_VISIBLE_DEVICES=3 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh

ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_EPOCH_050 \
RUN_PREFIX=gsm8k_diag_e050 \
CUDA_VISIBLE_DEVICES=1 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh

ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_DIRECT_STYLE \
RUN_PREFIX=gsm8k_diag_direct \
CUDA_VISIBLE_DEVICES=2 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh

ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_EQUATION_STYLE \
RUN_PREFIX=gsm8k_diag_equation \
CUDA_VISIBLE_DEVICES=3 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh

ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_CHECKPOINT_SWEEP \
RUN_PREFIX=gsm8k_diag_ckpt_sweep \
CUDA_VISIBLE_DEVICES=1 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh

ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_SHORT_COT \
RUN_PREFIX=gsm8k_diag_short_cot \
CUDA_VISIBLE_DEVICES=2 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh

ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_ANSWER_WEIGHTED \
RUN_PREFIX=gsm8k_diag_answer_weighted \
CUDA_VISIBLE_DEVICES=3 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh

ACTIVE_PHASE_B_GROUP=B2_GSM8K_REPAIR_ANSWER_WEIGHTED_CKPT \
RUN_PREFIX=gsm8k_repair_aw_ckpt \
CUDA_VISIBLE_DEVICES=2 \
PHASE_B_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_b_training_suite.sh
```

Optional extra CLI args passthrough:

```bash
ACTIVE_PHASE_B_GROUP=B1_SMOKE \
PHASE_B_EXTRA_ARGS="--max-train-samples 128 --max-eval-samples 32" \
bash scripts/run_phase_b_training_suite.sh
```

### 10.5.1 What the Full-Dataset `B2_*` Groups Actually Do

These groups are the first Phase B settings that answer the real project question:
"How much benchmark gain does PEFT give compared with the frozen base model?"

Flow:
1. baseline eval on held-out splits with the frozen base model,
2. full PEFT training on the full training split,
3. post-train eval on the same held-out splits,
4. gain analysis that computes per-split and aggregate deltas.

Held-out focus:
1. train split is used for training only,
2. reportable gain is computed from validation + test,
3. this avoids the common novice mistake of reporting training-set memorization as real improvement.

Artifacts written by the suite:
1. suite log:
   - `assets/artifacts/phase_b_logs/<RUN_PREFIX>/suite.log`
2. suite summary:
   - `assets/artifacts/phase_b_logs/<RUN_PREFIX>/final_summary.md`
3. gain summary:
   - `assets/artifacts/phase_b_logs/<RUN_PREFIX>/peft_gain_summary.json`
   - `assets/artifacts/phase_b_logs/<RUN_PREFIX>/peft_gain_summary.md`
4. underlying evaluation runs:
   - `assets/artifacts/phase_a_runs/<run_name>_pre_<split>_*/metrics.json`
   - `assets/artifacts/phase_a_runs/<run_name>_post_<split>_*/metrics.json`
5. training run:
   - `assets/artifacts/phase_b_runs/<run_name>_*/`

Dataset-specific defaults:
1. `B2_STRATEGYQA_FULL`
   - train input: `assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl`
   - held-out eval inputs:
     - `assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl`
     - `assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/test.jsonl`
   - decode mode: `freeform`
   - max new tokens: `96`
2. `B2_GSM8K_FULL`
   - train input: `assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/train.jsonl`
   - held-out eval inputs:
     - `assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/validation.jsonl`
     - `assets/artifacts/phase_a_prepared/gsm8k/e3abe2fb9883/test.jsonl`
   - decode mode: `freeform`
   - max new tokens: `192`

Useful overrides:

```bash
# faster/slower held-out eval batching
PHASE_B_EVAL_BATCH_SIZE=8

# extra eval flags forwarded into scripts/phase_a_generate_and_eval.py
PHASE_B_EVAL_EXTRA_ARGS="--max-progress-lines 20 --log-every 25 --no-compare-latest-same-name"

# extra training flags forwarded into scripts/phase_b_train_sft.py
PHASE_B_EXTRA_ARGS="--num-train-epochs 2.0"
```

Standalone comparison script:

```bash
python -u scripts/phase_b_compare_eval.py \
  --dataset strategyqa \
  --phase-b-run-dir assets/artifacts/phase_b_runs/<phase_b_run_dir> \
  --compare validation before_validation_metrics.json after_validation_metrics.json \
  --compare test before_test_metrics.json after_test_metrics.json
```

### 10.5.2 GSM8K Diagnostic Decision Tree

Use these runs to isolate *why* GSM8K dropped after PEFT.

1. Optimization overshoot check:
   - `B2_GSM8K_DIAG_LR_5E5`
   - `B2_GSM8K_DIAG_LR_1E4`
   - Interpretation:
     - if either materially beats `B2_GSM8K_FULL`, the original `2e-4` LR was too large.

2. Exposure / overtraining check:
   - `B2_GSM8K_DIAG_EPOCH_025`
   - `B2_GSM8K_DIAG_EPOCH_050`
   - Interpretation:
     - if shorter exposure beats the 1.0 epoch run, the adapter is over-learning style patterns.

3. Supervision-target style check:
   - `B2_GSM8K_DIAG_DIRECT_STYLE`
   - `B2_GSM8K_DIAG_EQUATION_STYLE`
   - Interpretation:
     - if direct style holds up better than CoT style, the main issue is CoT-target imitation,
     - if equation style reproduces the same cleaner-but-wrong pattern, equation markup is part of the failure mode.

4. Checkpoint drift check:
   - `B2_GSM8K_DIAG_CHECKPOINT_SWEEP`
   - Interpretation:
     - if an earlier checkpoint beats the final adapter, the GSM8K drop is partly late-run drift rather than purely bad supervision.

5. Long-CoT target-length check:
   - `B2_GSM8K_DIAG_SHORT_COT`
   - Interpretation:
     - if short-CoT supervision recovers accuracy, the long target itself is causing style-over-truth damage.

6. Loss-balance check:
   - `B2_GSM8K_DIAG_ANSWER_WEIGHTED`
   - Interpretation:
     - if answer-weighted supervision recovers accuracy, the original loss is too dominated by rationale tokens.

7. Combined repair check:
   - `B2_GSM8K_REPAIR_ANSWER_WEIGHTED_CKPT`
   - Interpretation:
     - combine the two strongest GSM8K repair signals already found:
       - answer-weighted supervision,
       - and early/best checkpoint selection.
     - If this run matches or exceeds the frozen base model at one retained checkpoint, the GSM8K problem is largely a late-drift + objective-balance issue.

Recommended run order:
1. `B2_GSM8K_DIAG_LR_1E4`
2. `B2_GSM8K_DIAG_EPOCH_050`
3. `B2_GSM8K_DIAG_LR_5E5`
4. `B2_GSM8K_DIAG_EPOCH_025`
5. `B2_GSM8K_DIAG_DIRECT_STYLE`
6. `B2_GSM8K_DIAG_EQUATION_STYLE`
7. `B2_GSM8K_DIAG_SHORT_COT`
8. `B2_GSM8K_DIAG_ANSWER_WEIGHTED`
9. `B2_GSM8K_DIAG_CHECKPOINT_SWEEP`
10. `B2_GSM8K_REPAIR_ANSWER_WEIGHTED_CKPT`

Suggested stable launch form:

```bash
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_LR_1E4 \
RUN_PREFIX=gsm8k_diag_lr1e4 \
CUDA_VISIBLE_DEVICES=1 \
PHASE_B_EVAL_BATCH_SIZE=64 \
PHASE_B_EVAL_EXTRA_ARGS="--max-progress-lines 20 --log-every 25 --no-compare-latest-same-name" \
bash scripts/run_phase_b_training_suite.sh
```

Checkpoint sweep note:
- `B2_GSM8K_DIAG_CHECKPOINT_SWEEP` trains a dedicated full GSM8K CoT run with dense checkpoint saving (`save_steps=100`, `save_total_limit=12`) and then auto-evaluates every retained checkpoint plus the final adapter.
- Output files:
  - `assets/artifacts/phase_b_logs/<RUN_PREFIX>/checkpoint_sweep_summary.md`
  - `assets/artifacts/phase_b_logs/<RUN_PREFIX>/checkpoint_sweep_summary.json`

Combined repair note:
- `B2_GSM8K_REPAIR_ANSWER_WEIGHTED_CKPT` uses the same dense checkpoint sweep flow, but changes the training loss to:
  - reasoning tokens weight `0.5`
  - final-answer line weight `3.0`
- This is the current best-evidence GSM8K repair attempt after the completed Phase B diagnosis.

### 10.5.4 Cross-Task Interference Suite

Why this suite exists:
- the core BCR/ABR question is not only whether PEFT helps on its source task,
- it is also whether a task-specific adapter interferes with other reasoning tasks.

Script:

```bash
bash scripts/run_phase_b_cross_task_suite.sh
```

Supported groups:
1. `B3_XTASK_STRAT_R32_TO_GSM8K`
2. `B3_XTASK_GSM8K_FULL_TO_STRAT`
3. `B3_XTASK_GSM8K_DIRECT_TO_STRAT`
4. `B3_XTASK_GSM8K_EQUATION_TO_STRAT`

Recommended launches:

```bash
ACTIVE_CROSS_TASK_GROUP=B3_XTASK_STRAT_R32_TO_GSM8K \
RUN_PREFIX=xtask_strat_r32_to_gsm8k \
CUDA_VISIBLE_DEVICES=1 \
PHASE_B_EVAL_BATCH_SIZE=64 \
PHASE_B_EVAL_EXTRA_ARGS="--max-progress-lines 20 --log-every 25 --no-compare-latest-same-name" \
bash scripts/run_phase_b_cross_task_suite.sh

ACTIVE_CROSS_TASK_GROUP=B3_XTASK_GSM8K_FULL_TO_STRAT \
RUN_PREFIX=xtask_gsm8k_full_to_strat \
CUDA_VISIBLE_DEVICES=2 \
PHASE_B_EVAL_BATCH_SIZE=64 \
PHASE_B_EVAL_EXTRA_ARGS="--max-progress-lines 20 --log-every 25 --no-compare-latest-same-name" \
bash scripts/run_phase_b_cross_task_suite.sh

ACTIVE_CROSS_TASK_GROUP=B3_XTASK_GSM8K_DIRECT_TO_STRAT \
RUN_PREFIX=xtask_gsm8k_direct_to_strat \
CUDA_VISIBLE_DEVICES=3 \
PHASE_B_EVAL_BATCH_SIZE=64 \
PHASE_B_EVAL_EXTRA_ARGS="--max-progress-lines 20 --log-every 25 --no-compare-latest-same-name" \
bash scripts/run_phase_b_cross_task_suite.sh

ACTIVE_CROSS_TASK_GROUP=B3_XTASK_GSM8K_EQUATION_TO_STRAT \
RUN_PREFIX=xtask_gsm8k_equation_to_strat \
CUDA_VISIBLE_DEVICES=1 \
PHASE_B_EVAL_BATCH_SIZE=64 \
PHASE_B_EVAL_EXTRA_ARGS="--max-progress-lines 20 --log-every 25 --no-compare-latest-same-name" \
bash scripts/run_phase_b_cross_task_suite.sh
```

Outputs:
- `assets/artifacts/phase_b_logs/<RUN_PREFIX>/cross_task_gain_summary.md`
- `assets/artifacts/phase_b_logs/<RUN_PREFIX>/cross_task_gain_summary.json`

Interpretation:
- if StrategyQA PEFT hurts GSM8K, StrategyQA alignment is not task-isolated,
- if GSM8K full-CoT PEFT hurts StrategyQA more than GSM8K short-style PEFT does,
  the cross-task damage is tied to the long-CoT GSM8K supervision pattern.

How to read the result:
1. open `assets/artifacts/phase_b_logs/<RUN_PREFIX>/peft_gain_summary.md`
2. compare:
   - `delta_accuracy`
   - `delta_correct`
   - `delta_parse_error_rate`
3. if needed, inspect the underlying scored outputs under:
   - `assets/artifacts/phase_a_runs/<run_name>_post_validation_*/scored_predictions.jsonl`
   - `assets/artifacts/phase_a_runs/<run_name>_post_test_*/scored_predictions.jsonl`

### 10.5.3 StrategyQA Scaling Decision Tree

Use these runs to test whether StrategyQA can still improve beyond the current 1.0 epoch, rank-16 baseline.

1. Epoch scaling:
   - `B2_STRATEGYQA_DIAG_EPOCH_200`
   - `B2_STRATEGYQA_DIAG_EPOCH_300`
   - Interpretation:
     - if held-out accuracy continues rising, the current run is under-trained,
     - if it flattens or drops, the current StrategyQA setup is already near its useful training limit.

2. LoRA capacity scaling:
   - `B2_STRATEGYQA_DIAG_LORA_R8`
   - `B2_STRATEGYQA_DIAG_LORA_R32`
   - Interpretation:
     - if `r=32` beats baseline, capacity is limiting,
     - if `r=8` matches baseline, the baseline adapter is larger than needed.

Recommended run order:
1. `B2_STRATEGYQA_DIAG_EPOCH_200`
2. `B2_STRATEGYQA_DIAG_LORA_R32`
3. `B2_STRATEGYQA_DIAG_EPOCH_300`
4. `B2_STRATEGYQA_DIAG_LORA_R8`

Suggested stable launch form:

```bash
ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_DIAG_EPOCH_200 \
RUN_PREFIX=strategyqa_diag_e200 \
CUDA_VISIBLE_DEVICES=1 \
PHASE_B_EVAL_BATCH_SIZE=8 \
PHASE_B_EVAL_EXTRA_ARGS="--max-progress-lines 20 --log-every 25 --no-compare-latest-same-name" \
bash scripts/run_phase_b_training_suite.sh
```
