# OURS Research Project (Independent from external BCR code)

This repository is building an in-house reasoning-faithfulness pipeline from scratch.

## 2026-03-13 Breakthrough Verification

Newest verification report:

1. `docs/phase_e_latest_breakthrough_verification_20260312.md`
   - verifies the newest `PBR26` / `PBR31` breakthrough claims
   - corrected status:
     - `PBR26` = best verified frozen benchmark candidate, not RL-ready
     - `PBR31` = best verified LoRA balance so far, not Math-AUC SOTA, not RL-ready

## 2026-03-13 Phase F Controller Sweep

New artifact:

1. [controller sweep summary](/home/zling/y/bcr/ref/assets/artifacts/phase_f_controller_sweep/phase_f_controller_sweep_0312_main_20260311T181216Z/summary.md)

What changed:

1. the earlier `F2 FAIL` conclusion was correct for the old `baseline_immediate` controller;
2. but once we sweep multiple controller families offline, the picture changes sharply.

Best redesigned controller results:

1. `pbr19_math`
   - old baseline: `0.5537`
   - `threshold_only tau=0.38`: `0.8674`
2. `pbr21_math`
   - old baseline: `0.5588`
   - `threshold_only tau=0.35`: `0.8641`
3. `pbr26_math`
   - old baseline: `0.5752`
   - `threshold_only tau=0.38`: `0.8639`
4. `pbr19_gsm`
   - old baseline: `0.6713`
   - `delayed_drop`: `0.9052`
5. `pbr26_gsm`
   - old baseline: `0.6509`
   - `delayed_drop`: `0.9053`

Shared simple policies also transfer well:

1. `threshold_only tau=0.42`
   - mean `balanced_f1 = 0.8552`
2. `delayed_drop tau=0.42 delta=0.25 min_step=4`
   - mean `balanced_f1 = 0.8501`

Interpretation:

1. verifier quality is now strong enough for a real heuristic controller baseline;
2. the immediate next step is **controller redesign + live validation**;
3. RL is no longer the first thing to do.

## 2026-03-13 Eval Provenance Hardening

One more repository-wide audit pass closed a real trust risk in `Phase B/C/D/E`
standalone evaluation:

1. older eval paths could silently replace a requested `best` checkpoint with
   `final`;
2. some `posthoc_calibration=from_run` paths could silently replace the missing
   `best` calibrator with `final`.

Current rule:

1. default = strict `fail`
2. legacy fallback requires explicit opt-in:
   - `--checkpoint-missing-policy fallback_final`
   - `--posthoc-missing-policy fallback_final`
3. any fallback is now persisted into:
   - `metrics.json`
   - `manifest.json`
   - `summary.md`
   - console logs

Affected entrypoints:

1. `scripts/phase_b_eval_faithfulness.py`
2. `scripts/phase_c_eval_pik.py`
3. `scripts/phase_d_eval_external_pairs.py`
4. `scripts/phase_e_eval_benchmark.py`
5. `scripts/phase_e_eval_samefamily_trust.py`

Minimal verification commands:

```bash
python -u scripts/phase_b_eval_faithfulness.py \
  --value-run-dir <phase_c_run_dir> \
  --eval-dir <phase_c_eval_dir> \
  --checkpoint-name best \
  --checkpoint-missing-policy fail

python -u scripts/phase_c_eval_pik.py \
  --value-run-dir <phase_c_pik_run_dir> \
  --eval-dir <phase_c_pik_eval_dir> \
  --checkpoint-name best \
  --checkpoint-missing-policy fail \
  --posthoc-calibration from_run \
  --posthoc-missing-policy fail
```

This audit pass also rechecked benchmark leakage:

1. the primary `Phase E` source-bundle training path does **not** directly ingest
   `ProcessBench` or `PRMBench` benchmark test files;
2. benchmark-facing scripts remain separated as:
   - eval
   - alignment audit
   - curated research utilities

Newest focused report:

1. `docs/phase_e_paradigm_refresh_experiments_20260311.md`
   - plain-language explanation of newer verifier / PRM terminology
   - targeted repair sweep around `PBR10`
   - clean conclusion:
     - `PRX1` helps terminal completeness
   - `PRX2 dual_head` helps terminal ordering but damages same-family verifier quality

## 2026-03-12 Phase F Audit Correction

The current `Phase F` audit is **not fully green**.

What is supported by artifact:

1. `threshold / generator-shift` audit
2. `reward-hacking` probe

What was overstated and is now corrected:

1. the earlier doc wording around `F2 ABR-lite` was too optimistic
2. raw artifact:
   - [summary.json](/home/zling/y/bcr/ref/assets/artifacts/phase_f_simulation/pbr19_math_abr_lite_0312/summary.json)
3. corrected numbers:
   - `balanced_f1=0.3388`
   - `positive_f1=0.7806`
   - `acc_erroneous=0.9882`
   - `acc_correct=0.2044`
   - `mean_step_fraction=0.3524`

Correct interpretation:

1. the offline controller catches errors very aggressively,
2. but it wrongly stops too many all-correct traces,
3. so `Phase F live RL / ABR-lite promotion` is **not yet justified**.

## 2026-03-11 Frontier Suite: Three New Directions On The Strongest Shared Artifact

Newest frontier wrapper:

1. `scripts/run_phase_e_frontier_suite.sh`

It tests three directions that are all motivated by the newer verifier
literature:

1. `F1_JUDGE_FILTER_PBR10`
   - idea:
     - use a stronger math judge as a conservative offline denoiser
   - command:
```bash
RUN_PREFIX=phase_e_frontier_judge_$(date +%m%d_%H%M) \
ACTIVE_PHASE_E_FRONTIER_GROUP=F1_JUDGE_FILTER_PBR10 \
BENCH_MAX_SAMPLES=192 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_frontier_suite.sh
```
2. `F2_DUAL_HEAD_PBR10`
   - idea:
     - split local-vs-terminal scoring pressure with `dual_head`
   - command:
```bash
RUN_PREFIX=phase_e_frontier_dual_$(date +%m%d_%H%M) \
ACTIVE_PHASE_E_FRONTIER_GROUP=F2_DUAL_HEAD_PBR10 \
BENCH_MAX_SAMPLES=192 \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_e_frontier_suite.sh
```
3. `F3_LORA_PBR10`
   - idea:
     - test whether the remaining bottleneck is representation-limited
   - command:
```bash
RUN_PREFIX=phase_e_frontier_lora_$(date +%m%d_%H%M) \
ACTIVE_PHASE_E_FRONTIER_GROUP=F3_LORA_PBR10 \
BENCH_MAX_SAMPLES=192 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_e_frontier_suite.sh
```

Important current intermediate finding:

1. `judge-filter` already shows a strong negative contract result on the shared
   `PBR10` artifact:
   - artifact:
     - `assets/artifacts/phase_e_pairs/phase_e_frontier_judge_0311_2031_judgefilter__88888f79419b`
   - all `1783` auditable `local_first_bad_edge` pairs were dropped as
     `parse_failed`
   - the resulting train set contains:
     - `4679 sibling_branch`
     - `485 terminal_completion_anchor`
     - `0 local_first_bad_edge`
2. interpretation:
   - the current strict judge contract does not act as a useful local denoiser;
   - it changes the data geometry instead.

## 2026-03-11 Latest Mainline State

The repository now has one new benchmark-facing mainline that materially changes
the earlier picture:

1. code audit:
   - A-E implementation paths were re-audited;
   - highest-risk infra issues fixed this round:
     - feature-cache live-writer lock stealing,
     - Phase B silent `best -> final` eval fallback provenance,
     - dtype alias drift across A/B/C entrypoints.
2. method:
   - the new strong path is no longer pure `Math-Shepherd` local supervision;
   - it is `math_step_dpo_v1` curation + `Qwen2.5-Math-PRM-7B` backbone + `mlp` head.
3. benchmark:
   - `PBR10 mlp` reached:
     - `ProcessBench Math`: `AUC=0.863`, `F1_oracle=0.659`, `F1_fixed@0.5=0.654`
     - `ProcessBench GSM8K`: `AUC=0.873`, `F1_oracle=0.724`, `F1_fixed@0.5=0.693`
4. architecture interpretation:
   - `gated_mlp` improves held-out ranking and edge metrics,
   - but hurts fixed-threshold F1,
   - so it remains a secondary probe rather than the main deployment head.
5. infrastructure:
   - all active `run_phase_e*.sh` wrappers now pass explicit `--recipe-risk-policy`;
   - direct `scripts/phase_e_train_value.py` now defaults to `pair_acc` checkpoint selection.
6. research direction:
   - after refreshing the literature to 2026-03, the repository should treat the
     scalar value head as a bounded-support verifier baseline rather than the
     assumed final RL-ready object.
7. broader verifier refresh:
   - the newest `2025H2 -> 2026-03` evidence further says the future mainline
     should not be `single scalar verifier`;
   - it should be:
     - separate answer verifier,
     - `invalid / abstain / escalate` behavior,
     - weak-ensemble or teacher-assisted hard-slice labeling,
     - factorized ABR student outputs,
     - and stricter `process-outcome alignment + format robustness` gates.

Primary reference document for this update:

1. `docs/phase_abcde_impl_audit_and_redesign_20260311.md`
2. `docs/phase_e_literature_refresh_20260311.md`
3. `docs/phase_e_rl_ready_research_redesign_20260311.md`

### Follow-Up Validation: Teammate Breakthrough Report

The teammate report contained one strong claim that needed local validation:

1. `Math-PRM-7B frozen backbone` is the real breakthrough
2. and the newest `PRX1` line may now be the strongest candidate

The local evidence audit now supports the first claim, but not the second.

#### Controlled backbone evidence

`FLB2` comparison on recent ProcessBench evals:

| case | backbone | GSM AUC | GSM F1 | Math AUC | Math F1 |
|---|---|---:|---:|---:|---:|
| `FLB2 math-instruct gated` | `Qwen2.5-Math-7B-Instruct` | `0.7461` | `0.5158` | `0.7423` | `0.4275` |
| `FLB2 prm-backbone mlp` | `Qwen2.5-Math-PRM-7B` | `0.8581` | `0.7148` | `0.8313` | `0.6276` |

This is a real step change. The backbone breakthrough is not speculative.

#### Current strong-candidate comparison

Oracle ProcessBench:

| case | GSM AUC | GSM F1 | Math AUC | Math F1 |
|---|---:|---:|---:|---:|
| `PBR10 mlp` | `0.8730` | `0.7243` | `0.8631` | `0.6589` |
| `PBR12` | `0.9093` | `0.7523` | `0.8874` | `0.6439` |
| `PBR21` | `0.9045` | `0.7565` | `0.8697` | `0.6632` |
| `PRX1` | `0.8869` | `0.7518` | `0.8529` | `0.6482` |

Fixed-threshold `0.5`:

| case | GSM F1@0.5 | Math F1@0.5 |
|---|---:|---:|
| `PBR12` | `0.7023` | `0.6344` |
| `PBR21` | `0.7185` | `0.6051` |
| `PRX1`  | `0.7454` | `0.6040` |

Interpretation:

1. `PRX1` is not a clean global replacement for `PBR12 / PBR21`;
2. it is strongest on the checked `GSM fixed@0.5` slice;
3. but `PBR12` still owns the best `Math AUC`;
4. and `PBR21` still owns the best `Math oracle F1`.

So the correct headline is:

1. the PRM-backbone breakthrough is real;
2. the repo now has a strong candidate triangle:
   - `PBR12`
   - `PBR21`
   - `PRX1`
3. but there is not yet one single globally dominant winner.

## 2026-03-11 Fresh A-E Audit + New ProcessBench Transfer Follow-Up

This pass combined:

1. one fresh A-E implementation audit,
2. one new Phase E transfer-profile sweep,
3. and one new architecture follow-up on the strongest newly discovered data profile.

### Fresh Audit Result

The repository is no longer in a broad “implementation instability” state:

1. `PYTHONPATH=src pytest -q tests/unit`
   - `217 passed, 2 warnings`
2. active `run_phase_*.sh` entrypoints:
   - syntax clean

One newly confirmed high-risk bug was fixed:

1. file:
   - `scripts/phase_a_prepare.py`
2. old behavior:
   - in `official` split mode, prepared rows were written according to the
     requested CLI `--source-split`, even when the dataset loader had already
     resolved that request to another effective split
3. risk:
   - test rows could silently land in `validation.jsonl`
4. fix:
   - official-mode routing now uses the effective loader-provided
     `metadata["source_split"]`
   - emitted rows preserve both:
     - `source_split`
     - `requested_source_split`
5. regression:
   - `tests/unit/test_phase_a_prepare_script.py`

### New Transfer Experiments

#### `NDS5_MS_STRICT_ONLY_SMOKE`

```bash
CUDA_DEVICE=2 \
TARGET_TOTAL_PAIRS=4096 \
BENCH_MAX_SAMPLES=96 \
TRAIN_BATCH_SIZE=64 \
EVAL_BATCH_SIZE=96 \
ACTIVE_PHASE_E_NDS_GROUP=NDS5_MS_STRICT_ONLY_SMOKE \
RUN_PREFIX=phase_e_nds5_strictonly_0311 \
bash scripts/run_phase_e_newdataset_suite.sh
```

Result:

1. `PB GSM`: `pair_acc=0.4048`, `auc=0.4210`, `first_edge=0.5122`
2. `PB Math`: `pair_acc=0.3876`, `auc=0.4321`, `first_edge=0.5957`

Interpretation:

1. removing `fanout/grid` alone does not fix transfer;
2. the old failure is not just length bias.

#### `NDS6_RLHFLOW_STRICT_ONLY_SMOKE`

```bash
CUDA_DEVICE=2 \
TARGET_TOTAL_PAIRS=4096 \
BENCH_MAX_SAMPLES=96 \
TRAIN_BATCH_SIZE=64 \
EVAL_BATCH_SIZE=96 \
ACTIVE_PHASE_E_NDS_GROUP=NDS6_RLHFLOW_STRICT_ONLY_SMOKE \
RUN_PREFIX=phase_e_nds6_rlhstrict_0311 \
bash scripts/run_phase_e_newdataset_suite.sh
```

Result:

1. `PB GSM`: `pair_acc=0.6224`, `auc=0.5022`, `first_edge=0.6585`
2. `PB Math`: `pair_acc=0.4357`, `auc=0.4520`, `first_edge=0.6383`

Interpretation:

1. better step labels alone help some local edge behavior,
2. but strict-only geometry is still not enough.

#### `NDS7_MS_DPO_CALIBRATED_SMOKE`

```bash
CUDA_DEVICE=2 \
TARGET_TOTAL_PAIRS=4096 \
BENCH_MAX_SAMPLES=96 \
TRAIN_BATCH_SIZE=64 \
EVAL_BATCH_SIZE=96 \
ACTIVE_PHASE_E_NDS_GROUP=NDS7_MS_DPO_CALIBRATED_SMOKE \
RUN_PREFIX=phase_e_nds7_dpocal_0311 \
bash scripts/run_phase_e_newdataset_suite.sh
```

Result (`mlp`):

1. `PB GSM`: `pair_acc=0.5374`, `auc=0.5575`, `first_edge=0.5854`
2. `PB Math`: `pair_acc=0.5301`, `auc=0.5594`, `first_edge=0.7234`

Interpretation:

1. `Math-Step-DPO sibling_branch` supervision is clearly more benchmark-aligned
   than strict-only variants;
2. this was the first new smoke in this pass that improved both `Math AUC` and
   `Math first_edge` together.

### Architecture Follow-Up: `NDS7 + gated_mlp`

Shared curated data:

1. `local_first_bad_edge = 1474`
2. `sibling_branch = 1831`
3. `terminal_completion_anchor = 387`

3-seed benchmark summary:

| seed | GSM pair_acc | GSM auc | GSM first_edge | Math pair_acc | Math auc | Math first_edge |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.6565 | 0.6456 | 0.8049 | 0.7992 | 0.7519 | 0.7447 |
| 1  | 0.6531 | 0.6514 | 0.7805 | 0.8052 | 0.7260 | 0.7872 |
| 7  | 0.7177 | 0.6817 | 0.7073 | 0.8253 | 0.7601 | 0.7021 |

Aggregate:

1. `PB GSM`
   - `pair_acc = 0.6757 ± 0.0297`
   - `auc = 0.6596 ± 0.0158`
   - `first_edge = 0.7642 ± 0.0415`
2. `PB Math`
   - `pair_acc = 0.8099 ± 0.0112`
   - `auc = 0.7460 ± 0.0145`
   - `first_edge = 0.7447 ± 0.0347`

Relative to the older `PBR5B ms_prm_align_v1 + gated_mlp` line:

1. `PB GSM AUC`: `~0.589 -> ~0.660`
2. `PB Math AUC`: `~0.613 -> ~0.746`

This is a real step change, not noise-level movement.

### Updated Diagnosis

The repo is no longer stuck in the older failure mode:

1. not “frozen-head ProcessBench transfer is broadly broken”;
2. not “the head cannot learn benchmark-facing ranking”;
3. and not “fanout/grid length bias is the whole problem”.

The new narrower bottleneck is:

1. `DPO sibling_branch` is now the strongest current signal carrier;
2. with `gated_mlp`, local and global benchmark ranking are already strong;
3. the main residual weakness is **all-correct terminal completion ordering**.

That means the next repair stage should stop searching for generic better
sources and instead target terminal completion directly while preserving the new
DPO-driven local gains.

### Terminal-Residual Repair Sweep (`seed=42`, controlled comparison)

To avoid mixing source, objective, and head changes in one opaque jump, a
three-way controlled sweep was run around the `NDS7` baseline:

1. data-level repair:
   - `ms_dpo_terminalboost_v1`
2. loss-level repair:
   - `NDS7 + terminal BCE`
3. architecture-level repair:
   - `NDS7 + dual_head + terminal BCE`

Representative commands:

```bash
# A) data-level repair: terminal-boosted pair curation
python -u scripts/phase_e_curate_processbench_transfer_pairs.py \
  --profile ms_dpo_terminalboost_v1 \
  --run-name phase_e_nds8_termboost_0311_pairs \
  --output-root assets/artifacts/phase_e_pairs \
  --seed 42 \
  --validation-ratio 0.1 \
  --split-granularity source_sample \
  --target-total-pairs 4096 \
  --min-pair-confidence 0.55

# B) train on the terminal-boost artifact
CUDA_VISIBLE_DEVICES=2 python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_nds8_termboost_0311_pairs__480ab06bf8d6/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_nds8_termboost_0311_pairs__480ab06bf8d6/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_nds8_termboost_gated_pilot_value_retry2 \
  --output-root assets/artifacts/phase_e_runs \
  --objective-mode joint \
  --learning-rate 3e-5 \
  --num-train-epochs 5 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 32 \
  --max-length 1024 \
  --lambda-ranking 1.0 \
  --lambda-bce 1.0 \
  --ranking-margin 0.02 \
  --ranking-target-space score \
  --pair-weight-mode none \
  --source-balance none \
  --permutation-mode stable_hash \
  --checkpoint-selection-metric pair_acc \
  --seed 42 \
  --feature-cache-root assets/artifacts/phase_e_feature_cache \
  --feature-cache-mode read_write \
  --head-architecture gated_mlp \
  --head-mlp-hidden-size 1024 \
  --head-dropout-prob 0.05 \
  --head-init-std 0.02 \
  --head-activation gelu \
  --anti-saturation-weight 5e-4 \
  --anti-saturation-logit-threshold 3.5 \
  --max-gpu-memory-gib 24 \
  --max-cpu-memory-gib 256 \
  --require-cuda

# C) same data, loss-level repair
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_nds7_dpocal_0311_nds7_ms_dpo_calibrated_mlp_ms_dpo_calibrated_v1_pairs__ffabba63a6bf/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_nds7_dpocal_0311_nds7_ms_dpo_calibrated_mlp_ms_dpo_calibrated_v1_pairs__ffabba63a6bf/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_nds7_termbce_gated_pilot_value_retry1 \
  --output-root assets/artifacts/phase_e_runs \
  --objective-mode joint \
  --learning-rate 3e-5 \
  --num-train-epochs 5 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 32 \
  --max-length 1024 \
  --lambda-ranking 1.0 \
  --lambda-bce 1.0 \
  --terminal-bce-lambda 0.25 \
  --ranking-margin 0.02 \
  --ranking-target-space score \
  --pair-weight-mode none \
  --source-balance none \
  --permutation-mode stable_hash \
  --checkpoint-selection-metric pair_acc \
  --seed 42 \
  --feature-cache-root assets/artifacts/phase_e_feature_cache \
  --feature-cache-mode read_write \
  --head-architecture gated_mlp \
  --head-mlp-hidden-size 1024 \
  --head-dropout-prob 0.05 \
  --head-init-std 0.02 \
  --head-activation gelu \
  --anti-saturation-weight 5e-4 \
  --anti-saturation-logit-threshold 3.5 \
  --max-gpu-memory-gib 24 \
  --max-cpu-memory-gib 256 \
  --require-cuda

# D) same data, architecture-level repair
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_nds7_dpocal_0311_nds7_ms_dpo_calibrated_mlp_ms_dpo_calibrated_v1_pairs__ffabba63a6bf/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_nds7_dpocal_0311_nds7_ms_dpo_calibrated_mlp_ms_dpo_calibrated_v1_pairs__ffabba63a6bf/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_nds7_dualhead_termbce_pilot_value \
  --output-root assets/artifacts/phase_e_runs \
  --objective-mode joint \
  --learning-rate 3e-5 \
  --num-train-epochs 5 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 32 \
  --max-length 1024 \
  --lambda-ranking 1.0 \
  --lambda-bce 1.0 \
  --terminal-bce-lambda 0.25 \
  --ranking-margin 0.02 \
  --ranking-target-space score \
  --pair-weight-mode none \
  --source-balance none \
  --permutation-mode stable_hash \
  --checkpoint-selection-metric pair_acc \
  --seed 42 \
  --feature-cache-root assets/artifacts/phase_e_feature_cache \
  --feature-cache-mode read_write \
  --head-architecture dual_head \
  --head-inference-alpha 0.65 \
  --head-mlp-hidden-size 1024 \
  --head-dropout-prob 0.05 \
  --head-init-std 0.02 \
  --head-activation gelu \
  --anti-saturation-weight 5e-4 \
  --anti-saturation-logit-threshold 3.5 \
  --max-gpu-memory-gib 24 \
  --max-cpu-memory-gib 256 \
  --require-cuda
```

#### Baseline: `NDS7 + gated_mlp`

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.6565` | `0.6456` | `0.8049` | `0.2174` | `-0.1640` |
| `PB Math`  | `0.7992` | `0.7519` | `0.7447` | `0.1538` | `-0.1461` |

#### Repair A: data-level terminal boost

Artifact / run:

1. pair artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_nds8_termboost_0311_pairs__480ab06bf8d6`
2. value run:
   - `assets/artifacts/phase_e_runs/phase_e_nds8_termboost_gated_pilot_value_retry2_20260311T124818Z`

ProcessBench:

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.6752` | `0.6816` | `0.7118` | `0.2798` | `-0.1038` |
| `PB Math`  | `0.7316` | `0.7046` | `0.6848` | `0.1823` | `-0.1670` |

Interpretation:

1. GSM terminal ordering improved in a real way;
2. Math suffered a local/global tradeoff;
3. therefore "just add more terminal anchors" is not a safe global fix.

#### Repair B: loss-level terminal BCE on original `NDS7`

Run:

1. `assets/artifacts/phase_e_runs/phase_e_nds7_termbce_gated_pilot_value_retry1_20260311T125922Z`

ProcessBench:

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.6053` | `0.6612` | `0.6353` | `0.2073` | `-0.1383` |
| `PB Math`  | `0.7229` | `0.7049` | `0.6555` | `0.1650` | `-0.1842` |

Interpretation:

1. terminal BCE alone does not solve the residual;
2. it mostly hurts local/global behavior while barely helping terminal ordering.

#### Repair C: `dual_head + terminal BCE` on original `NDS7`

Run:

1. `assets/artifacts/phase_e_runs/phase_e_nds7_dualhead_termbce_pilot_value_20260311T130037Z`

ProcessBench:

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.8150` | `0.7489` | `0.7882` | `0.0829` | `-0.2312` |
| `PB Math`  | `0.7820` | `0.7105` | `0.6806` | `0.0739` | `-0.2535` |

Interpretation:

1. `dual_head` creates a much stronger local verifier;
2. but the current inference mixture catastrophically undervalues final correct
   traces;
3. the repository now has hard evidence that:
   - local error detection
   - and terminal completion ordering
   are separable subproblems.

### Updated Takeaway

The next mainline should **not** be:

1. more terminal data by default,
2. or terminal BCE by default,
3. or switching wholesale to `dual_head`.

The safer conclusion is:

1. keep `NDS7 + gated_mlp` as the balanced benchmark-facing baseline;
2. treat terminal completion ordering as a narrow residual subtask;
3. and fix it with a more explicit routing / calibration design rather than a
   naive mixed objective.

## 2026-03-11 Phase E Safety Audit Closure

Static suite audit:

```bash
python -u scripts/phase_e_audit_suite_recipes.py \
  --run-name phase_e_suite_recipe_audit_$(date +%m%d_%H%M)
```

Trainer sanity:

```bash
python -u scripts/phase_e_train_value.py --help | rg "checkpoint-selection-metric|recipe-risk-policy|train-oom-backoff"
```

Current evidence:
1. first pass:
   - `assets/artifacts/phase_e_audits/phase_e_suite_recipe_audit_0311_1931/summary.md`
   - findings = `10`
2. post-fix pass:
   - `assets/artifacts/phase_e_audits/phase_e_suite_recipe_audit_postfix_0311_1933/summary.md`
   - findings = `0`

## 2026 Literature Refresh For Phase E

Newer sources that materially change the design:
1. `PRIME (2026)`:
   - `https://arxiv.org/abs/2602.11570`
   - verifier quality on process-outcome alignment strongly predicts RLVR effectiveness
2. `Hard2Verify (2025)`:
   - `https://arxiv.org/abs/2510.13744`
   - open-source step verifiers still lag on frontier open-ended verification
3. `RISE (2025)`:
   - `https://arxiv.org/abs/2505.13445`
   - self-verification should be trained jointly with reasoning, not bolted on later
4. `VPRM (2026)`:
   - `https://arxiv.org/abs/2601.17223`
   - deterministic process verification can beat pure outcome verification in structured domains

Repository interpretation:
1. same-source held-out accuracy is no longer enough for RL-ready claims;
2. the next experimental redesign should explicitly add:
   - process-outcome alignment evaluation,
   - critique / self-verification auxiliary training,
   - deterministic verifier hooks wherever the task permits.

Primary redesign note:
1. `docs/phase_e_updated_literature_redesign_20260311.md`

## 2026-03-11 Postfix Strict RL Re-Audit

After the broad A-E audit and the Phase E safety closure, one more postfix pass
closed the remaining evaluation-side loopholes and re-ran the newest
`Qwen2.5-Math-PRM-7B` candidates under the stricter RL gate.

What changed:

1. selector hygiene:
   - `scripts/phase_e_select_candidate.py`
   - default `selection_policy=heldout_only`
   - benchmark metrics remain promotion gates / canaries, but no longer rank
     candidates by default
2. intradataset checkpoint publication:
   - `scripts/phase_e_select_intradataset_candidate.py`
   - candidate reports now resolve `best/final` checkpoints explicitly instead
     of publishing a bogus `<run_dir>/best_value_head.pt`
3. samefamily PRM-7B crash handling:
   - `src/ours/phase_e/runtime.py`
   - batched encoding now synchronizes CUDA inside the retry block and treats
     async capacity failures as retryable batch-backoff events

Validation:

1. `PYTHONPATH=src pytest -q tests/unit/test_phase_e_runtime.py tests/unit/test_phase_e_select_candidate.py tests/unit/test_phase_e_select_intradataset_candidate.py`
   - `16 passed`
2. `python -m py_compile src/ours/phase_e/runtime.py scripts/phase_e_select_candidate.py scripts/phase_e_select_intradataset_candidate.py scripts/phase_e_eval_samefamily_trust.py`

Postfix reruns:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -u scripts/phase_e_eval_samefamily_trust.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_pbr10_prm7b_dpo8k_s42_value_20260311T110527Z \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_pbr10_prm7b_dpo8k_s42__6184f7e62f65/validation_pairs.jsonl \
  --run-name pbr10_s42_samefamily_fix0311 \
  --output-root assets/artifacts/phase_e_samefamily_eval \
  --checkpoint-name best \
  --batch-size 96 \
  --feature-cache-mode read_write \
  --require-cuda

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -u scripts/phase_e_eval_samefamily_trust.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_pbr13_pbr6_terminalBCE_s42_value_20260311T112822Z \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_ndsbh_0311_ndsbh2_math_step_dpo_mlp_math_step_dpo_v1_pairs__9e4e39b13902/validation_pairs.jsonl \
  --run-name pbr13_s42_samefamily_fix0311 \
  --output-root assets/artifacts/phase_e_samefamily_eval \
  --checkpoint-name best \
  --batch-size 96 \
  --feature-cache-mode read_write \
  --require-cuda
```

Strict diagnosis:

```bash
PYTHONPATH=src python -u scripts/phase_e_diagnose_transfer.py \
  --run-name phase_e_transfer_diag_postfix0311_pbr10 \
  --output-root assets/artifacts/phase_e_transfer_diag \
  --audit-spec 'pbr10_dpo8k|math_mixed|assets/artifacts/phase_e_samefamily_eval/pbr10_s42_samefamily_fix0311_20260311T114318Z|assets/artifacts/phase_e_eval/pbr10_dpo8k_gsm_fulleval_20260311T112402Z|assets/artifacts/phase_e_eval/pbr10_dpo8k_math_fulleval_20260311T112402Z'

PYTHONPATH=src python -u scripts/phase_e_diagnose_transfer.py \
  --run-name phase_e_transfer_diag_postfix0311_pbr13 \
  --output-root assets/artifacts/phase_e_transfer_diag \
  --audit-spec 'pbr13_terminalbce|math_mixed|assets/artifacts/phase_e_samefamily_eval/pbr13_s42_samefamily_fix0311_20260311T113935Z|assets/artifacts/phase_e_eval/pbr13_terminalBCE_gsm_fulleval_20260311T112921Z|assets/artifacts/phase_e_eval/pbr13_terminalBCE_math_fulleval_20260311T112921Z'
```

Postfix conclusion:

1. `pbr10_dpo8k` is the strongest current frozen-head checkpoint:
   - samefamily `top1=0.9098`
   - samefamily `local_first_bad=0.8981`
   - `ProcessBench GSM F1=0.7334`
   - `ProcessBench Math F1=0.6313`
2. `pbr13_terminalbce` improves benchmark headline scores but still does not fix
   terminal completion ordering, and it weakens samefamily trust.
3. no current checkpoint clears the strict internal RL gate yet.
4. the remaining blocker is no longer generic benchmark transfer collapse, but
   **all-correct terminal completion undervaluation**.

## 2026-03-11 Formal Judge Follow-Up: PRMBench Selected Relabel + ProcessBench Hard-Slice Adjudication

Question:
1. if we promote judge usage from pilot to formal experiments, does it now help on
   `PRMBench_Preview selected relabel`?
2. if we move judge to the safer benchmark side, can it adjudicate the hardest
   `ProcessBench` failure slices?

Short answer:
1. not in the way that matters:
   - formal `selected relabel` improved held-out `PRMBench_Preview`,
   - but the judge itself only preserved `1/64` selected boundary pairs,
   - so the gain is better read as `targeted pruning / truncation cleanup`, not
     successful semantic relabel;
2. no:
   - on `ProcessBench` hard slices, current local judge is effectively unusable as
     a benchmark-side adjudicator.

New scripts:
1. `scripts/phase_e_slice_pairs_by_margin.py`
2. `scripts/phase_e_materialize_selected_relabel_pairs.py`
3. `scripts/phase_e_prepare_processbench_hardslice_pairs.py`
4. `scripts/run_phase_e_prmbench_selected_relabel_suite.sh`
5. `scripts/run_phase_e_processbench_hardslice_adjudication_suite.sh`

Commands actually executed:

Formal `PRMBench_Preview selected relabel`:

```bash
RUN_PREFIX=phase_e_prmbench_selected_relabel_formal_0311 \
SLICE_GPU=0 \
JUDGE_GPU=1 \
TRAIN_GPU=0 \
bash scripts/run_phase_e_prmbench_selected_relabel_suite.sh
```

Formal `ProcessBench` benchmark-side hard-slice adjudication:

```bash
RUN_PREFIX=phase_e_processbench_hardslice_adj_0311_r2 \
PREP_GPU=2 \
JUDGE_GPU=3 \
bash scripts/run_phase_e_processbench_hardslice_adjudication_suite.sh
```

Key results:
1. `PRMBench_Preview selected relabel`
   - selected slice:
     - `64` lowest-abs-margin rows from `5407` train pairs
     - `58/64` selected rows already had negative margin under baseline `E78`
   - judge on selected slice:
     - `both_parse_ok_rate = 0.0469`
     - `pair_acc_majority = 0.0312`
     - `label_preserving_keep_rate = 0.0156`
   - materialized train set:
     - `5344` rows total
     - only `1` judge-kept row from the selected slice
   - retrain vs baseline:
     - `baseline_e78`: `pair_acc = 0.9521`, `auc = 0.9071`
     - `selected_relabel`: `pair_acc = 0.9555`, `auc = 0.9207`
2. `ProcessBench hard-slice adjudication`
   - `math`:
     - `29` pairs
     - `both_parse_ok_rate = 0.0345`
     - `pair_acc_majority = 0.1034`
     - `label_preserving_keep_rate = 0.0000`
   - `gsm8k`:
     - `27` pairs
     - `both_parse_ok_rate = 0.0370`
     - `pair_acc_majority = 0.0000`
     - `label_preserving_keep_rate = 0.0000`

Interpretation:
1. the formal `selected relabel` run does **not** show that local judge can now
   semantically repair hard `PRMBench_Preview` labels;
2. the observed held-out gain is much more plausibly explained by deleting the
   hardest and most truncation-prone boundary rows:
   - baseline train truncation:
     - `frac_pairs_over_limit = 0.0146`
     - `frac_pairs_identical_after_truncation = 0.0074`
   - materialized train truncation:
     - `frac_pairs_over_limit = 0.0067`
     - `frac_pairs_identical_after_truncation = 0.0000`
3. current local judge should therefore remain:
   - an audit / disagreement / selective-pruning tool on judge-friendly pair data,
   not:
   - a trusted relabeler,
   - or a benchmark-side adjudicator for `ProcessBench`.

## 2026-03-11 Phase E Recipe Guard And Collapse Diagnosis

`Phase E` 现在新增了两层基础设施防护：

1. 训练前 recipe 风险检查
   - 默认 `--recipe-risk-policy error`
   - 会直接拦截仓库里已经确认会灾难性失败的组合
2. 训练后 health 诊断
   - 每个新 run 会自动产出：
     - `recipe_risk.json`
     - `training_health.json`
     - `training_health.md`

关键脚本：
1. `scripts/phase_e_train_value.py`
2. `scripts/phase_e_diagnose_training_collapse.py`

直接复现实验：

坏配方会被训练前直接拒绝：

```bash
python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_ndsbh_0311_ndsbh3_ms_align_baseline_mlp_ms_align_v1_pairs__c2631693ee5b/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_ndsbh_0311_ndsbh3_ms_align_baseline_mlp_ms_align_v1_pairs__c2631693ee5b/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_recipe_guard_badprobe \
  --max-train-samples 128 \
  --max-eval-samples 64 \
  --objective-mode joint \
  --learning-rate 3e-5 \
  --num-train-epochs 2 \
  --per-device-train-batch-size 16 \
  --per-device-eval-batch-size 16 \
  --max-length 1024 \
  --ranking-margin 0.02 \
  --ranking-target-space logit \
  --pair-weight-mode confidence_semantic \
  --checkpoint-selection-metric ranking_score \
  --recipe-risk-policy error
```

对历史坏 run 做统一复诊：

```bash
python -u scripts/phase_e_diagnose_training_collapse.py \
  --run-dir assets/artifacts/phase_e_runs/phase_e_ms_trust_smoke_0310_1400_e1_math_shepherd_pair_learn_smoke_e1_math_shepherd_pair_learn_smoke_s42_value_20260310T060115Z
```

## 2026-03-11 Pairwise Judge Benchmark And Filter Pilot

Question:
1. after the negative strict-JSON pointwise judge audit, does a more literature-aligned
   `pairwise + swap-debias + light contract` judge setup become usable?
2. can the local judge now act as a conservative label-preserving Phase E pair filter?

Short answer:
1. partially, but only weakly:
   - `PRMBench_Preview` shows some limited judge utility,
   - `Math-Shepherd local_first_bad_edge` does not;
2. no: the current local judge is still not strong enough to be promoted to a
   bulk label-preserving filter.

New script:
1. `scripts/phase_e_pairwise_judge_benchmark.py`

Unified compare artifact:
1. `assets/artifacts/phase_e_pairwise_judge_compare/judge_pairwise_compare_20260311T084132Z/summary.md`

Commands actually executed:

Held-out `PRMBench_Preview`, `Qwen2.5-Math-7B-Instruct`:

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_e_pairwise_judge_benchmark.py \
  --pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_all_acc90_0310_1915_e44_prmbench_acc90_linear_seed3_e44_prmbench_acc90_linear_seed3_sharedsplit_s42_pairs__66693d3b512f/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-Math-7B-Instruct \
  --run-name pairjudge_qwen_prmbench_anchor32 \
  --dataset-label prmbench_preview_val32_anchor \
  --max-samples 32 \
  --batch-size 4 \
  --max-input-length 2048 \
  --max-new-tokens 48 \
  --assistant-prefix '[FINAL]\nPREFERRED='
```

Held-out `PRMBench_Preview`, `DeepSeek-R1-Distill-Qwen-14B`:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_e_pairwise_judge_benchmark.py \
  --pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_all_acc90_0310_1915_e44_prmbench_acc90_linear_seed3_e44_prmbench_acc90_linear_seed3_sharedsplit_s42_pairs__66693d3b512f/validation_pairs.jsonl \
  --model-path assets/models/DeepSeek-R1-Distill-Qwen-14B \
  --run-name pairjudge_deepseek_prmbench_anchor16 \
  --dataset-label prmbench_preview_val16_anchor \
  --max-samples 16 \
  --batch-size 2 \
  --max-input-length 2048 \
  --max-new-tokens 48 \
  --no-use-system-prompt \
  --assistant-prefix '[FINAL]\nPREFERRED='
```

Held-out `Math-Shepherd local_first_bad_edge`, `Qwen2.5-Math-7B-Instruct`:

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_e_pairwise_judge_benchmark.py \
  --pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_all_acc90_0310_1915_e41_ms_acc90_mlp_rank_seed3_e41_ms_acc90_mlp_rank_seed3_sharedsplit_s42_pairs__a642ebf5fab7/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-Math-7B-Instruct \
  --run-name pairjudge_qwen_ms_val32_anchor \
  --dataset-label math_shepherd_val32_anchor \
  --max-samples 32 \
  --batch-size 4 \
  --max-input-length 2048 \
  --max-new-tokens 48 \
  --assistant-prefix '[FINAL]\nPREFERRED='
```

Train-slice filter pilot, `PRMBench_Preview`:

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_e_pairwise_judge_benchmark.py \
  --pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_all_acc90_0310_1915_e44_prmbench_acc90_linear_seed3_e44_prmbench_acc90_linear_seed3_sharedsplit_s42_pairs__66693d3b512f/train_pairs.jsonl \
  --model-path assets/models/Qwen2.5-Math-7B-Instruct \
  --run-name pairjudge_qwen_prmbench_train64_anchor \
  --dataset-label prmbench_preview_train64_anchor \
  --max-samples 64 \
  --batch-size 4 \
  --max-input-length 2048 \
  --max-new-tokens 48 \
  --assistant-prefix '[FINAL]\nPREFERRED='
```

Train-slice filter pilot, `Math-Shepherd`:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_e_pairwise_judge_benchmark.py \
  --pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_all_acc90_0310_1915_e41_ms_acc90_mlp_rank_seed3_e41_ms_acc90_mlp_rank_seed3_sharedsplit_s42_pairs__a642ebf5fab7/train_pairs.jsonl \
  --model-path assets/models/Qwen2.5-Math-7B-Instruct \
  --run-name pairjudge_qwen_ms_train64_anchor \
  --dataset-label math_shepherd_train64_anchor \
  --max-samples 64 \
  --batch-size 4 \
  --max-input-length 2048 \
  --max-new-tokens 48 \
  --assistant-prefix '[FINAL]\nPREFERRED='
```

Key results:
1. `Qwen` on `PRMBench_Preview` held-out:
   - `both_parse_ok_rate = 0.3125`
   - `pair_acc_majority = 0.3438`
   - `swap_consistency_rate = 0.6000`
   - `label_preserving_keep_rate = 0.1250`
2. `Qwen` on `Math-Shepherd` held-out:
   - `both_parse_ok_rate = 0.2188`
   - `pair_acc_majority = 0.0625`
   - `label_preserving_keep_rate = 0.0312`
3. `Qwen` as train-slice filter:
   - `PRMBench_Preview`: `keep_rate = 0.0469`
   - `Math-Shepherd`: `keep_rate = 0.0000`
4. `DeepSeek` on `PRMBench_Preview` held-out:
   - `both_parse_ok_rate = 0.5625`
   - `pair_acc_majority = 0.0625`
   - `tie_rate = 0.8125`

Interpretation:
1. pairwise judging is indeed a better operational direction than the earlier
   pointwise strict-JSON judge path;
2. but the current local judge models are still too weak to serve as robust
   bulk Phase E pair filters;
3. and `PRMBench_Preview` is much closer to judge-usable semantics than
   `Math-Shepherd local_first_bad_edge`.

## 2026-03-11 Judge LLM Selection And Local Deployment

Question:
1. if Phase E moves toward `LLM-as-a-judge`, which local judge models should we
   actually install and trust on the current server?

Short answer:
1. `Qwen2.5-Math-7B-Instruct` is the best current bulk local judge candidate,
2. `DeepSeek-R1-Distill-Qwen-14B` is installed but not yet operationally stable
   enough for the mainline local judge loop,
3. `QwQ-32B` remains a later optional upgrade, not the right first local
   deployment choice.

## 2026-03-11 Backbone Relaxation + Judge Pilot

Question:
1. if we relax the current frozen-backbone assumption, does a small LoRA pilot
   already beat the stronger frozen Phase E reference?
2. can the newly installed local judge already clean Phase E pair artifacts
   enough to improve value-head learning?

Short answer:
1. no: the current small-data LoRA pilot underperforms the stronger frozen
   reference and is not trustworthy enough to promote,
2. no: the current strict-JSON local judge contract is too brittle to use as a
   hard Phase E pair filter,
3. yes, but only in a narrower sense: the judge models are still useful for
   offline audit / disagreement mining / selective adjudication.

### New detailed note

1. `docs/phase_e_backbone_judge_audit_20260311.md`

### New tooling

1. `scripts/phase_e_train_value_lora.py`
   - minimal online-encoding LoRA trainer for Phase E pairs
2. `scripts/phase_e_judge_filter_pairs.py`
   - conservative prefix-correctness judge filter for auditable semantics
3. `scripts/run_phase_e_backbone_judge_suite.sh`
   - pilot wrapper for frozen vs LoRA and raw vs judge-filtered comparisons

### Commands actually executed

Note:
1. the manual frozen rerun used `CUDA_VISIBLE_DEVICES=0` because at run time it
   was the only GPU with enough free memory for a no-cache 7B backbone load;
2. this was a temporary cluster-load constraint, not a new default preference.

Pilot subset curation:

```bash
PYTHONPATH=src python -u scripts/phase_e_curate_semantic_pairs.py \
  --slice 'pilot=assets/artifacts/phase_e_pairs/phase_e_processbench_research_v2_pbr2_ms_align_gated_ms_align_v1_pairs__79c6e734325c|*|*|96|127' \
  --run-name phase_e_backbone_judge_pilot_small_0311_subset \
  --output-root assets/artifacts/phase_e_pairs \
  --seed 42
```

Judge hard filter:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_e_judge_filter_pairs.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_backbone_judge_pilot_small_0311_subset__9b43b747715c/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_backbone_judge_pilot_small_0311_subset__9b43b747715c/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-Math-7B-Instruct \
  --run-name phase_e_backbone_judge_pilot_small_0311_judgefilter \
  --output-root assets/artifacts/phase_e_pairs \
  --batch-size 16 \
  --max-new-tokens 80 \
  --logging-batches 1 \
  --dtype bfloat16 \
  --device-map auto \
  --min-confidence 0.0
```

Raw frozen subset:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_backbone_judge_pilot_small_0311_subset__9b43b747715c/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_backbone_judge_pilot_small_0311_subset__9b43b747715c/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_backbone_judge_manual_base_frozen_nocache_g0 \
  --output-root assets/artifacts/phase_e_runs \
  --objective-mode joint \
  --learning-rate 3e-5 \
  --num-train-epochs 4 \
  --per-device-train-batch-size 48 \
  --per-device-eval-batch-size 48 \
  --max-length 1024 \
  --lambda-ranking 1.0 \
  --lambda-bce 1.0 \
  --ranking-margin 0.02 \
  --ranking-target-space logit \
  --pair-weight-mode confidence_semantic \
  --source-balance none \
  --permutation-mode stable_hash \
  --checkpoint-selection-metric ranking_score \
  --head-architecture mlp \
  --head-mlp-hidden-size 1024 \
  --head-dropout-prob 0.05 \
  --head-init-std 0.02 \
  --head-activation gelu \
  --anti-saturation-weight 5e-4 \
  --anti-saturation-logit-threshold 3.5 \
  --feature-cache-mode off \
  --max-gpu-memory-gib 45
```

Raw frozen evals:

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/phase_e_eval_samefamily_trust.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_backbone_judge_manual_base_frozen_nocache_g0_20260311T080638Z \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_backbone_judge_pilot_small_0311_subset__9b43b747715c/validation_pairs.jsonl \
  --run-name phase_e_backbone_judge_manual_base_frozen_samefamily \
  --output-root assets/artifacts/phase_e_samefamily_eval \
  --batch-size 48 \
  --max-length 1024 \
  --device-map auto
```

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_e_eval_benchmark.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_backbone_judge_manual_base_frozen_nocache_g0_20260311T080638Z \
  --benchmark-id processbench_gsm8k \
  --run-name phase_e_backbone_judge_manual_base_frozen_pb_gsm8k \
  --output-root assets/artifacts/phase_e_eval \
  --max-samples 64 \
  --batch-size 48 \
  --max-length 1024 \
  --device-map auto
```

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_e_eval_benchmark.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_backbone_judge_manual_base_frozen_nocache_g0_20260311T080638Z \
  --benchmark-id processbench_math \
  --run-name phase_e_backbone_judge_manual_base_frozen_pb_math \
  --output-root assets/artifacts/phase_e_eval \
  --max-samples 64 \
  --batch-size 32 \
  --max-length 1024 \
  --device-map auto
```

Raw LoRA subset:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_e_train_value_lora.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_backbone_judge_pilot_small_0311_subset__9b43b747715c/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_backbone_judge_pilot_small_0311_subset__9b43b747715c/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_backbone_judge_manual_base_lora \
  --output-root assets/artifacts/phase_e_runs \
  --objective-mode joint \
  --learning-rate 2e-5 \
  --num-train-epochs 2 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 8 \
  --gradient-accumulation-steps 16 \
  --max-length 768 \
  --lambda-ranking 1.0 \
  --lambda-bce 1.0 \
  --ranking-margin 0.02 \
  --ranking-target-space logit \
  --pair-weight-mode confidence_semantic \
  --source-balance none \
  --permutation-mode stable_hash \
  --checkpoint-selection-metric ranking_score \
  --head-architecture mlp \
  --head-mlp-hidden-size 1024 \
  --head-dropout-prob 0.05 \
  --head-init-std 0.02 \
  --head-activation gelu \
  --anti-saturation-weight 5e-4 \
  --anti-saturation-logit-threshold 3.5 \
  --lora-rank 4 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,v_proj,gate_proj,up_proj,down_proj \
  --lora-top-k-layers 4 \
  --gradient-checkpointing
```

Raw LoRA evals:

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/phase_e_eval_samefamily_trust.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_backbone_judge_manual_base_lora_20260311T080936Z \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_backbone_judge_pilot_small_0311_subset__9b43b747715c/validation_pairs.jsonl \
  --run-name phase_e_backbone_judge_manual_base_lora_samefamily \
  --output-root assets/artifacts/phase_e_samefamily_eval \
  --batch-size 24 \
  --max-length 768 \
  --device-map auto
```

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_e_eval_benchmark.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_backbone_judge_manual_base_lora_20260311T080936Z \
  --benchmark-id processbench_gsm8k \
  --run-name phase_e_backbone_judge_manual_base_lora_pb_gsm8k \
  --output-root assets/artifacts/phase_e_eval \
  --max-samples 64 \
  --batch-size 24 \
  --max-length 768 \
  --device-map auto
```

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/phase_e_eval_benchmark.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_backbone_judge_manual_base_lora_20260311T080936Z \
  --benchmark-id processbench_math \
  --run-name phase_e_backbone_judge_manual_base_lora_pb_math_1024 \
  --output-root assets/artifacts/phase_e_eval \
  --max-samples 64 \
  --batch-size 12 \
  --max-length 1024 \
  --device-map auto
```

Judge prefix audit:

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_e_audit_judge_prefix_pairs.py \
  --pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_backbone_judge_pilot_small_0311_subset__9b43b747715c/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-Math-7B-Instruct \
  --run-name phase_e_judge_prefix_audit_rawsubset \
  --output-root assets/artifacts/phase_e_judge_audit \
  --max-pairs 32 \
  --max-new-tokens 96
```

### Main results

Reference `PBR2` remains ahead:

1. held-out:
   - `pair_acc=0.8819`, `auc=0.8723`
2. same-family:
   - `top1=0.8947`
   - `local_first_bad=0.9231`
3. `ProcessBench GSM8K`:
   - `auc=0.4713`
   - `first_edge=0.5600`
4. `ProcessBench Math`:
   - `auc=0.5055`
   - `first_edge=0.5357`

Raw frozen subset pilot:

1. held-out:
   - `pair_acc=0.7874`, `auc=0.7343`
2. same-family:
   - `top1=0.6842`
   - `local_first_bad=0.7253`
3. `ProcessBench GSM8K`:
   - `auc=0.4481`
   - `first_edge=0.3200`
   - `f1=0.2133`
4. `ProcessBench Math`:
   - `auc=0.4479`
   - `first_edge=0.2500`
   - `f1=0.2087`

Raw LoRA subset pilot:

1. held-out:
   - `pair_acc=0.6693`, `auc=0.6859`
2. same-family:
   - `top1=0.4947`
   - `local_first_bad=0.0000`
3. `ProcessBench GSM8K`:
   - `auc=0.5390`
   - `first_edge=0.5200`
   - but `f1=0.0745`
4. `ProcessBench Math`:
   - `auc=0.7903`
   - `first_edge=0.0357`
   - `f1=0.0513`
   - this is not a real win; it signals benchmark-misaligned / unstable scoring

Judge hard-filter result:

1. train pairs:
   - `96 -> 16`
2. all 80 auditable pairs were dropped due `parse_failed`
3. only bypassed terminal-anchor pairs survived

Judge prefix audit result:

1. `pair_json_ok_rate=0.0`
2. `pair_agreement_rate=0.0`
3. runtime:
   - `230.5s` for only 32 audited pairs

### Practical reading

1. current `small-data LoRA` is a negative result, not a promotion candidate
2. current `strict-JSON judge hard filter` is also a negative result
3. judge should currently be used as:
   - disagreement mining
   - selective relabel
   - offline audit
4. current benchmark-facing reference remains:
   - `PBR2`

### New detailed note

1. `docs/phase_e_judge_llm_selection_20260311.md`

### Installed local judge models

1. `assets/models/Qwen2.5-Math-7B-Instruct`
2. `assets/models/DeepSeek-R1-Distill-Qwen-14B`
3. existing fallback:
   - `assets/models/Qwen2.5-7B-Instruct`

### New tooling

1. `scripts/phase_e_smoke_judge_llm.py`
   - loads a local instruct model
   - sends a structured step-judge prompt
   - checks JSON contract adherence and tiny good/bad reasoning examples
   - writes artifacts under `assets/artifacts/phase_e_judge_smoke/`

### Smoke verdict

1. `Qwen2.5-7B-Instruct`
   - cheapest baseline only
   - bad-example output can degenerate
2. `Qwen2.5-Math-7B-Instruct`
   - best current candidate for bulk relabel / disagreement mining
   - semantic judging is usable, but formatting still needs tolerant parsing
3. `DeepSeek-R1-Distill-Qwen-14B`
   - loads successfully
   - but current local `transformers + strict JSON prompt` path is unstable
   - should not yet be promoted as the main local judge backend

### Real small-slice benchmark after smoke

The smoke result was not enough, so a second script was added:

1. `scripts/phase_e_benchmark_judge_llm.py`

It benchmarks one local judge on deterministic small `ProcessBench` slices and
records:

1. `parse_ok_rate`
2. `overall_acc`
3. `first_bad_exact_acc`
4. `first_bad_within_one_acc`
5. `mean_step_acc`

Commands actually executed:

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_e_benchmark_judge_llm.py \
  --model-path assets/models/Qwen2.5-Math-7B-Instruct \
  --run-name judge_bench_qwen25_math_7b \
  --benchmark processbench_math \
  --benchmark processbench_gsm8k \
  --max-samples-per-benchmark 6 \
  --max-new-tokens 256
```

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_e_benchmark_judge_llm.py \
  --model-path assets/models/DeepSeek-R1-Distill-Qwen-14B \
  --run-name judge_bench_deepseek_r1_14b \
  --benchmark processbench_math \
  --benchmark processbench_gsm8k \
  --max-samples-per-benchmark 6 \
  --max-new-tokens 256 \
  --no-use-system-prompt \
  --assistant-prefix '<think>\n'
```

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_e_benchmark_judge_llm.py \
  --model-path assets/models/Qwen2.5-Math-7B-Instruct \
  --run-name judge_bench_qwen25_math_7b_fbonly \
  --benchmark processbench_math \
  --benchmark processbench_gsm8k \
  --max-samples-per-benchmark 6 \
  --max-new-tokens 160 \
  --contract-mode first_bad_only
```

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_e_benchmark_judge_llm.py \
  --model-path assets/models/DeepSeek-R1-Distill-Qwen-14B \
  --run-name judge_bench_deepseek_r1_14b_fbonly \
  --benchmark processbench_math \
  --benchmark processbench_gsm8k \
  --max-samples-per-benchmark 6 \
  --max-new-tokens 160 \
  --contract-mode first_bad_only \
  --no-use-system-prompt \
  --assistant-prefix '<think>\n'
```

Unified compare artifact:

1. `assets/artifacts/phase_e_judge_bench_compare/judge_model_compare_20260311T074514Z/summary.md`

Main conclusions from that compare:

1. `Qwen2.5-Math-7B-Instruct`
   - is operationally more stable,
   - but on real `ProcessBench` rows it is strongly biased toward
     `OVERALL=correct` and `FIRST_BAD=none`,
   - simplifying the output contract does not fix the main bias.
2. `DeepSeek-R1-Distill-Qwen-14B`
   - is still unstable under verbose contracts,
   - but under `first_bad_only` it shows real signal on `gsm8k`:
     - `overall_acc=0.6667`
     - `first_bad_exact=0.3333`
   - while still remaining weak on `math`.
3. Practical deployment reading:
   - use `Qwen2.5-Math-7B-Instruct` first for bulk relabel / disagreement mining,
   - keep `DeepSeek-R1-Distill-Qwen-14B` only as a second-stage adjudicator for
     harder, more verbal cases,
   - do not treat either model as a benchmark-ready standalone ProcessBench
     judge yet.

## 2026-03-11 ProcessBench State Audit + Community Gap Review

Question:
1. after the teammate's recent ProcessBench repair work, local PDF uploads, and
   new dataset downloads, what does the repository *actually* support now?

Short answer:
1. some ProcessBench slices are indeed improved,
2. but the newer repair lines still do not replace the older strongest
   benchmark-facing baselines,
3. and the main bottleneck has shifted from "can the value head learn?" to
   "can one training stack jointly preserve local, later-bad, and terminal
   behavior?"

### New synthesis document

1. `docs/processbench_state_and_community_gap_20260311.md`

### Fresh audit commands

Cross-run `ProcessBench Math` review:

```bash
python -u scripts/phase_e_compare_processbench_transfer.py \
  --run-name processbench_state_review_math_0311 \
  --case 'ms_e43=assets/artifacts/phase_e_runs/phase_e_ms_acc90_full_0310_1914_e43_ms_acc90_mlp_highconf_seed3_e43_ms_acc90_mlp_highconf_seed3_s42_value_20260310T111956Z::assets/artifacts/phase_e_eval/phase_e_rltops_0311_1124_ms_e43_processbench_math_20260311T032646Z' \
  --case 'prm_e46=assets/artifacts/phase_e_runs/phase_e_prmbench_acc90_full_0310_1914_e46_prmbench_acc90_mlp_joint_seed3_e46_prmbench_acc90_mlp_joint_seed3_s43_value_20260310T113737Z::assets/artifacts/phase_e_eval/phase_e_rl_readiness_0310_2338_prm_e46_processbench_math_20260310T153553Z' \
  --case 'e79_baseline=assets/artifacts/phase_e_runs/phase_e_pb_transfer_0311_1614_e79_ms_processbench_transfer_baseline_seed42_e79_ms_processbench_transfer_baseline_seed42_s42_value_20260310T160924Z::assets/artifacts/phase_e_eval/phase_e_pb_transfer_0311_1614_e79_ms_processbench_transfer_baseline_seed42_e79_ms_processbench_transfer_baseline_seed42_s42_processbench_math_20260310T164147Z' \
  --case 'e87_repair=assets/artifacts/phase_e_runs/phase_e_processbench_rlrepair_0311_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_s42_value_20260311T032957Z::assets/artifacts/phase_e_eval/phase_e_processbench_rlrepair_0311_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_s42_processbench_math_20260311T033955Z' \
  --case 'c3_curated_gated=assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_value_20260311T045452Z::assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_pb_math_20260311T045635Z' \
  --case 'c4_dual=assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_dual_c4_value_20260311T051319Z::assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_dual_c4_pb_math_20260311T051652Z' \
  --case 'pbr2_align_gated=assets/artifacts/phase_e_runs/phase_e_processbench_research_v2_pbr2_ms_align_gated_value_20260311T043818Z::assets/artifacts/phase_e_eval/phase_e_processbench_research_v2_pbr2_ms_align_gated_processbench_math_20260311T043935Z'
```

Cross-run `ProcessBench GSM8K` review:

```bash
python -u scripts/phase_e_compare_processbench_transfer.py \
  --run-name processbench_state_review_gsm_0311 \
  --case 'ms_e43=assets/artifacts/phase_e_runs/phase_e_ms_acc90_full_0310_1914_e43_ms_acc90_mlp_highconf_seed3_e43_ms_acc90_mlp_highconf_seed3_s42_value_20260310T111956Z::assets/artifacts/phase_e_eval/phase_e_rltops_0311_1124_ms_e43_processbench_gsm8k_20260311T032638Z' \
  --case 'prm_e46=assets/artifacts/phase_e_runs/phase_e_prmbench_acc90_full_0310_1914_e46_prmbench_acc90_mlp_joint_seed3_e46_prmbench_acc90_mlp_joint_seed3_s43_value_20260310T113737Z::assets/artifacts/phase_e_eval/phase_e_rl_readiness_0310_2338_prm_e46_processbench_gsm8k_20260310T153544Z' \
  --case 'e79_baseline=assets/artifacts/phase_e_runs/phase_e_pb_transfer_0311_1614_e79_ms_processbench_transfer_baseline_seed42_e79_ms_processbench_transfer_baseline_seed42_s42_value_20260310T160924Z::assets/artifacts/phase_e_eval/phase_e_pb_transfer_0311_1614_e79_ms_processbench_transfer_baseline_seed42_e79_ms_processbench_transfer_baseline_seed42_s42_processbench_gsm8k_20260310T164130Z' \
  --case 'e87_repair=assets/artifacts/phase_e_runs/phase_e_processbench_rlrepair_0311_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_s42_value_20260311T032957Z::assets/artifacts/phase_e_eval/phase_e_processbench_rlrepair_0311_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_s42_processbench_gsm8k_20260311T033946Z' \
  --case 'c3_curated_gated=assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_value_20260311T045452Z::assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_pb_gsm8k_20260311T045623Z' \
  --case 'c4_dual=assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_dual_c4_value_20260311T051319Z::assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_dual_c4_pb_gsm8k_20260311T051637Z' \
  --case 'pbr2_align_gated=assets/artifacts/phase_e_runs/phase_e_processbench_research_v2_pbr2_ms_align_gated_value_20260311T043818Z::assets/artifacts/phase_e_eval/phase_e_processbench_research_v2_pbr2_ms_align_gated_processbench_gsm8k_20260311T043919Z'
```

Strict RL-promotion review:

```bash
python -u scripts/phase_e_diagnose_rl_promotion.py \
  --run-name rl_promotion_state_review_0311 \
  --audit-spec 'ms_e43|math_shepherd|assets/artifacts/phase_e_samefamily_eval/phase_e_rltops_0311_1124_ms_e43_samefamily_20260311T032630Z|assets/artifacts/phase_e_eval/phase_e_rltops_0311_1124_ms_e43_processbench_gsm8k_20260311T032638Z|assets/artifacts/phase_e_eval/phase_e_rltops_0311_1124_ms_e43_processbench_math_20260311T032646Z' \
  --audit-spec 'prm_e46|prmbench_preview|assets/artifacts/phase_e_samefamily_eval/phase_e_rl_readiness_0310_2338_prm_e46_samefamily_20260310T153536Z|assets/artifacts/phase_e_eval/phase_e_rl_readiness_0310_2338_prm_e46_processbench_gsm8k_20260310T153544Z|assets/artifacts/phase_e_eval/phase_e_rl_readiness_0310_2338_prm_e46_processbench_math_20260310T153553Z' \
  --audit-spec 'e87_repair|math_shepherd|assets/artifacts/phase_e_samefamily_eval/phase_e_rlready_e87_samefamily_0311_20260311T040021Z|assets/artifacts/phase_e_eval/phase_e_processbench_rlrepair_0311_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_s42_processbench_gsm8k_20260311T033946Z|assets/artifacts/phase_e_eval/phase_e_processbench_rlrepair_0311_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_s42_processbench_math_20260311T033955Z' \
  --audit-spec 'c3_curated_gated|mixed_curated|assets/artifacts/phase_e_samefamily_eval/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_samefamily_20260311T045608Z|assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_pb_gsm8k_20260311T045623Z|assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_pb_math_20260311T045635Z' \
  --audit-spec 'c4_dual|mixed_curated|assets/artifacts/phase_e_samefamily_eval/phase_e_curated_rlready_0311_dual_c4_samefamily_20260311T051616Z|assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_dual_c4_pb_gsm8k_20260311T051637Z|assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_dual_c4_pb_math_20260311T051652Z' \
  --audit-spec 'pbr2_align_gated|math_shepherd|assets/artifacts/phase_e_samefamily_eval/phase_e_processbench_research_v2_pbr2_ms_align_gated_samefamily_20260311T043905Z|assets/artifacts/phase_e_eval/phase_e_processbench_research_v2_pbr2_ms_align_gated_processbench_gsm8k_20260311T043919Z|assets/artifacts/phase_e_eval/phase_e_processbench_research_v2_pbr2_ms_align_gated_processbench_math_20260311T043935Z'
```

### Fresh artifacts

1. `assets/artifacts/phase_e_transfer_compare/processbench_state_review_math_0311_20260311T070248Z/summary.md`
2. `assets/artifacts/phase_e_transfer_compare/processbench_state_review_gsm_0311_20260311T070308Z/summary.md`
3. `assets/artifacts/phase_e_rl_promotion_diag/rl_promotion_state_review_0311_20260311T070332Z/summary.md`

### Key conclusions

1. `ms_e43` remains the strongest local/later-bad candidate:
   - `ProcessBench Math: auc=0.6341, good_vs_laterbad=0.7515`
   - RL verdict:
     - `near_rl_ready_but_terminal_gap`
2. `prm_e46` remains the strongest more-balanced benchmark-facing candidate:
   - `ProcessBench GSM8K: auc=0.6264`
   - `ProcessBench Math: auc=0.6053, good_vs_laterbad=0.5501, terminal_top1=0.1970`
3. recent repair lines improve some slices, but still fail strict RL gate:
   - `e87_repair`
   - `c3_curated_gated`
   - `c4_dual`
   - `pbr2_align_gated`
4. the dominant unresolved conflict is now explicit:
   - local/later-bad strong candidates do not solve terminal completion
   - terminal-aware repair candidates often break broader good-vs-bad transfer
5. next mainline should therefore shift from:
   - more head-only churn
   to:
   - stronger downloaded sources
   - equal-budget source-only audits
   - one minimal frozen-vs-LoRA comparison
   after the best source mix is known

## 2026-03-11 Verified Internet Reading + Dual-Head Semantic Routing

Question:
1. after the earlier curated `gated_mlp` smoke, can a more explicit
   local-vs-terminal decomposition do better than "one stronger scalar head"?

Short answer:
1. not as a new mainline,
2. but the failure pattern is informative:
   - `dual_head` improves `first_edge` and terminal slices,
   - while worsening samefamily fit, overall AUC, and `good_vs_laterbad`.

### Verified external reading folded into this round

These sources were re-checked directly and used as the design basis:

1. `ProcessBench: Identifying Process Errors in Mathematical Reasoning`
   - <https://aclanthology.org/2025.acl-long.50/>
2. `Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning`
   - <https://arxiv.org/abs/2410.08146>
3. `Advancing Process Verification for Large Language Models via Tree-Based Preference Learning`
   - <https://arxiv.org/abs/2407.00390>
4. `PRMBench: Can Reward Models Truly Verify Reasoning Processes?`
   - <https://arxiv.org/abs/2501.03124>
5. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   - <https://arxiv.org/abs/2501.07301>
6. `Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning`
   - <https://arxiv.org/abs/2504.15275>
7. `R-PRM` official repo
   - <https://github.com/NJUNLP/R-PRM>
8. `Open Instruct` RM docs
   - <https://allenai.github.io/open-instruct/algorithms/reward_modeling/>

What was extracted:
1. benchmark transfer should be judged by slices, not one scalar
2. decomposed verifier behavior is more defensible than just adding head capacity
3. inside the current frozen-feature infra, the lowest-risk approximation is:
   - explicit `dual_head`
   - explicit semantic routing
   - no backbone unfreeze

### Code changes

Updated:
1. `src/ours/phase_b/value_head.py`
   - added `architecture='dual_head'`
   - added `inference_alpha`
   - forward now returns:
     - blended `logits/scores`
     - `local_logits/local_scores`
     - `terminal_logits/terminal_scores`
2. `src/ours/phase_e/training.py`
   - added semantic route resolution and route-weight tensors
   - `compute_pair_objective()` now supports routed dual-head optimization
3. `scripts/phase_e_train_value.py`
   - added `--head-architecture dual_head`
   - added `--head-inference-alpha`
   - routed branch weights into epoch training

### Validation commands

```bash
python -m py_compile \
  src/ours/phase_b/value_head.py \
  src/ours/phase_e/training.py \
  scripts/phase_e_train_value.py

pytest -q \
  tests/unit/test_value_head.py \
  tests/unit/test_phase_e_training.py \
  tests/unit/test_phase_e_train_script.py
```

Result:
1. `23 passed`

### Exact experiment commands

Train `dual_head` on the same curated artifact used by the earlier `C1/C2/C3`
comparison:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_curated_rlready_0311_retry2_curated_pairs__03ac5eebc8fd/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_curated_rlready_0311_retry2_curated_pairs__03ac5eebc8fd/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_curated_rlready_0311_dual_c4_value \
  --output-root assets/artifacts/phase_e_runs \
  --objective-mode joint \
  --ranking-target-space logit \
  --lambda-ranking 1.0 \
  --lambda-bce 1.0 \
  --ranking-margin 0.02 \
  --learning-rate 3e-5 \
  --num-train-epochs 4 \
  --per-device-train-batch-size 96 \
  --per-device-eval-batch-size 12 \
  --pair-weight-mode confidence_group_balance \
  --source-balance uniform \
  --permutation-mode stable_hash \
  --head-architecture dual_head \
  --head-mlp-hidden-size 1024 \
  --head-dropout-prob 0.05 \
  --head-init-std 0.02 \
  --head-activation gelu \
  --head-inference-alpha 0.5 \
  --anti-saturation-weight 0.0005 \
  --anti-saturation-logit-threshold 3.5 \
  --reward-centering-weight 0.01 \
  --checkpoint-selection-metric ranking_score \
  --feature-cache-root assets/artifacts/phase_e_feature_cache \
  --feature-cache-mode read \
  --max-gpu-memory-gib 40 \
  --max-cpu-memory-gib 96
```

Evaluate samefamily:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -u scripts/phase_e_eval_samefamily_trust.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_dual_c4_value_20260311T051319Z \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_curated_rlready_0311_retry2_curated_pairs__03ac5eebc8fd/validation_pairs.jsonl \
  --run-name phase_e_curated_rlready_0311_dual_c4_samefamily \
  --output-root assets/artifacts/phase_e_samefamily_eval \
  --batch-size 12 \
  --feature-cache-root assets/artifacts/phase_e_feature_cache \
  --feature-cache-mode read \
  --max-gpu-memory-gib 40 \
  --max-cpu-memory-gib 96
```

Evaluate `ProcessBench`:

```bash
CUDA_VISIBLE_DEVICES=1 \
python -u scripts/phase_e_eval_benchmark.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_dual_c4_value_20260311T051319Z \
  --benchmark-id processbench_gsm8k \
  --run-name phase_e_curated_rlready_0311_dual_c4_pb_gsm8k \
  --output-root assets/artifacts/phase_e_eval \
  --batch-size 12 \
  --feature-cache-root assets/artifacts/phase_e_feature_cache \
  --feature-cache-mode read \
  --max-gpu-memory-gib 40 \
  --max-cpu-memory-gib 96 \
  --max-samples 128

CUDA_VISIBLE_DEVICES=1 \
python -u scripts/phase_e_eval_benchmark.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_dual_c4_value_20260311T051319Z \
  --benchmark-id processbench_math \
  --run-name phase_e_curated_rlready_0311_dual_c4_pb_math \
  --output-root assets/artifacts/phase_e_eval \
  --batch-size 12 \
  --feature-cache-root assets/artifacts/phase_e_feature_cache \
  --feature-cache-mode read \
  --max-gpu-memory-gib 40 \
  --max-cpu-memory-gib 96 \
  --max-samples 128
```

Compare against earlier curated baselines:

```bash
python -u scripts/phase_e_compare_processbench_transfer.py \
  --run-name processbench_curated_arch_compare_0311_dual_math \
  --case 'c1_mlp=assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_retry2_c1_curated_mlp_base_value_20260311T044610Z::assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c1_curated_mlp_base_pb_math_20260311T045301Z' \
  --case 'c3_gated=assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_value_20260311T045452Z::assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_pb_math_20260311T045635Z' \
  --case 'c4_dual=assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_dual_c4_value_20260311T051319Z::assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_dual_c4_pb_math_20260311T051652Z'

python -u scripts/phase_e_compare_processbench_transfer.py \
  --run-name processbench_curated_arch_compare_0311_dual_gsm \
  --case 'c1_mlp=assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_retry2_c1_curated_mlp_base_value_20260311T044610Z::assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c1_curated_mlp_base_pb_gsm8k_20260311T045245Z' \
  --case 'c3_gated=assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_value_20260311T045452Z::assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_pb_gsm8k_20260311T045623Z' \
  --case 'c4_dual=assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_dual_c4_value_20260311T051319Z::assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_dual_c4_pb_gsm8k_20260311T051637Z'

python -u scripts/phase_e_diagnose_rl_promotion.py \
  --run-name phase_e_curated_rlpromo_compare_0311_dual \
  --audit-spec 'c3_gated|curated_multisource|assets/artifacts/phase_e_samefamily_eval/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_samefamily_20260311T045608Z|assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_pb_gsm8k_20260311T045623Z|assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_pb_math_20260311T045635Z' \
  --audit-spec 'c4_dual|curated_multisource|assets/artifacts/phase_e_samefamily_eval/phase_e_curated_rlready_0311_dual_c4_samefamily_20260311T051616Z|assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_dual_c4_pb_gsm8k_20260311T051637Z|assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_dual_c4_pb_math_20260311T051652Z'
```

### Key artifacts

1. train:
   - `assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_dual_c4_value_20260311T051319Z`
2. samefamily:
   - `assets/artifacts/phase_e_samefamily_eval/phase_e_curated_rlready_0311_dual_c4_samefamily_20260311T051616Z`
3. `ProcessBench`:
   - GSM8K:
     - `assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_dual_c4_pb_gsm8k_20260311T051637Z`
   - Math:
     - `assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_dual_c4_pb_math_20260311T051652Z`
4. compare summaries:
   - `assets/artifacts/phase_e_transfer_compare/processbench_curated_arch_compare_0311_dual_math_20260311T051707Z/summary.md`
   - `assets/artifacts/phase_e_transfer_compare/processbench_curated_arch_compare_0311_dual_gsm_20260311T051726Z/summary.md`
5. RL-promotion compare:
   - `assets/artifacts/phase_e_rl_promotion_diag/phase_e_curated_rlpromo_compare_0311_dual_20260311T051742Z/summary.md`

### Main findings

`C4 dual_head`:
1. held-out:
   - `pair_acc=0.8608`
   - `auc=0.7390`
2. samefamily:
   - `top1=0.8787`
   - `local_first_bad=0.9408`
3. `ProcessBench GSM8K`:
   - `auc=0.4730`
   - `first_edge=0.5660`
   - `terminal_top1=0.8548`
   - `good_vs_laterbad=0.2756`
4. `ProcessBench Math`:
   - `auc=0.4789`
   - `first_edge=0.5714`
   - `terminal_top1=0.9038`
   - `good_vs_laterbad=0.3672`

Interpretation:
1. `dual_head` is not a pure dead end
   - it moves the expected slices:
     - `first_edge`
     - terminal ranking
2. but the current hard routing is too coarse
   - samefamily top1 drops from `0.9443 -> 0.8787`
   - `good_vs_laterbad` drops badly on both benchmarks
3. so it should not replace `C3 gated_mlp` as the current mainline
4. the most defensible next experiment is:
   - `dual_head` soft routing / alpha sweep / staged repair
   - not more one-shot hard-routed training

## 2026-03-11 Internet-Guided Semantic Curation + Reward Centering Smoke

Question:
1. can a literature-guided semantic curation layer plus reward centering repair
   the current Phase E transfer gap enough to approach RL-ready behavior?

Short answer:
1. no current candidate became RL-ready,
2. reward centering alone did almost nothing,
3. `gated_mlp` improved `ProcessBench Math` somewhat but still failed the strict
   local-transfer gate.

### New code

Added / updated:
1. `docs/phase_e_internet_research_20260311.md`
2. `src/ours/phase_b/value_losses.py`
   - `reward_centering_penalty()`
3. `src/ours/phase_e/training.py`
   - `reward_centering_weight`
4. `scripts/phase_e_train_value.py`
   - `--reward-centering-weight`
5. `scripts/phase_e_curate_semantic_pairs.py`
6. `scripts/run_phase_e_curated_rlready_suite.sh`

### Commands used

Targeted tests:

```bash
PYTHONPATH=src pytest -q \
  tests/unit/test_phase_e_training.py \
  tests/unit/test_phase_e_train_script.py \
  tests/unit/test_phase_e_curate_semantic_pairs.py

bash -n scripts/run_phase_e_curated_rlready_suite.sh
```

First suite attempt that exposed the wrapper/runtime bottleneck:

```bash
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_curated_rlready_suite.sh
```

What happened:
1. the first full-size run exposed two wrapper bugs:
   - shell backtick substitution inside metadata text
   - env override clobbering for `EVAL_BATCH_SIZE`
2. the first fixed retry also showed that `per-device-eval-batch-size=96`
   wastes time in repeated OOM backoff during frozen-feature encoding.

Final successful suite run:

```bash
CUDA_VISIBLE_DEVICES=1 \
RUN_PREFIX=phase_e_curated_rlready_0311_retry2 \
EVAL_BATCH_SIZE=12 \
bash scripts/run_phase_e_curated_rlready_suite.sh
```

Strict transfer diagnosis on the three curated candidates:

```bash
python -u scripts/phase_e_diagnose_transfer.py \
  --run-name phase_e_curated_rlready_0311_retry2_diag \
  --output-root assets/artifacts/phase_e_transfer_diag \
  --audit-spec 'c1_curated|curated_multisource|assets/artifacts/phase_e_samefamily_eval/phase_e_curated_rlready_0311_retry2_c1_curated_mlp_base_samefamily_20260311T045230Z|assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c1_curated_mlp_base_pb_gsm8k_20260311T045245Z|assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c1_curated_mlp_base_pb_math_20260311T045301Z' \
  --audit-spec 'c2_curated|curated_multisource|assets/artifacts/phase_e_samefamily_eval/phase_e_curated_rlready_0311_retry2_c2_curated_mlp_center_samefamily_20260311T045419Z|assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c2_curated_mlp_center_pb_gsm8k_20260311T045433Z|assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c2_curated_mlp_center_pb_math_20260311T045446Z' \
  --audit-spec 'c3_curated|curated_multisource|assets/artifacts/phase_e_samefamily_eval/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_samefamily_20260311T045608Z|assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_pb_gsm8k_20260311T045623Z|assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_retry2_c3_curated_gated_center_pb_math_20260311T045635Z'
```

### Key artifacts

Curated semantic artifact:
1. `assets/artifacts/phase_e_pairs/phase_e_curated_rlready_0311_retry2_curated_pairs__03ac5eebc8fd`

Suite summary:
1. `assets/artifacts/phase_e_logs/phase_e_curated_rlready_0311_retry2/final_summary.md`

Strict transfer diagnosis:
1. `assets/artifacts/phase_e_transfer_diag/phase_e_curated_rlready_0311_retry2_diag_00/summary.md`

### Main findings

Curated pool:
1. `first_bad_fanout_prefix_ranking=1600`
2. `local_modified_process_error_step=1600`
3. `terminal_completion_anchor=320`

Results:
1. `C1_CURATED_MLP_BASE`
   - held-out:
     - `pair_acc=0.9034`
     - `auc=0.8574`
   - samefamily:
     - `top1=0.9541`
     - `local_first_bad=0.9737`
   - ProcessBench:
     - `gsm_auc=0.4892`
     - `math_auc=0.4553`
     - `math_first_edge=0.5079`
2. `C2_CURATED_MLP_CENTER`
   - held-out:
     - `pair_acc=0.8977`
     - `auc=0.8643`
   - samefamily:
     - `top1=0.9508`
     - `local_first_bad=0.9737`
   - ProcessBench:
     - `gsm_auc=0.4928`
     - `math_auc=0.4545`
     - `math_first_edge=0.5079`
3. `C3_CURATED_GATED_CENTER`
   - held-out:
     - `pair_acc=0.9290`
     - `auc=0.8706`
   - samefamily:
     - `top1=0.9443`
     - `local_first_bad=0.9934`
   - ProcessBench:
     - `gsm_auc=0.4861`
     - `math_auc=0.5152`
     - `math_first_edge=0.5397`

Strict diagnosis:
1. all three candidates:
   - `strict_rl_ready=0`
   - `assessment=not_rl_ready_local_transfer_weak`
2. recurring failure tags:
   - `benchmark_local_error_weak`
   - `margin_collapse`
   - `support_length_drift`

Operational reading:
1. reward centering is worth keeping as a stable option, but it is not the main fix
2. semantic curation alone is not enough
3. `gated_mlp` improves the math-side tradeoff a little, but does not solve the
   cross-benchmark local-transfer problem
4. the next repair should target local benchmark transfer explicitly rather than
   adding even more terminal pressure

## 2026-03-11 RL Promotion Infrastructure Hardening

Question:
1. can the current Phase E stack support a defensible RL-promotion decision?

Short answer:
1. infrastructure is materially better after this round,
2. but no audited candidate became RL-ready,
3. and a hidden non-finite pooled-feature bug turned out to be real.

### New code

Added / updated:
1. `scripts/phase_e_diagnose_rl_promotion.py`
2. `scripts/phase_e_prepare_prmbench_terminal_anchor_pairs.py`
   - new `--terminal-anchor-ratio`
3. `scripts/phase_e_train_value.py`
   - fail fast on non-finite loss
   - new `--nonfinite-feature-policy {error,drop}`
4. `src/ours/phase_e/training.py`
   - explicit non-finite pooled-feature validation / filtering

### Commands used

Baseline RL-promotion audit:

```bash
python -u scripts/phase_e_diagnose_rl_promotion.py \
  --run-name phase_e_rlpromo_diag_baselines_0311 \
  --audit-spec 'e80_fanout|math_shepherd|assets/artifacts/phase_e_samefamily_eval/phase_e_rlpromo_0311_e80_samefamily_20260311T033647Z|assets/artifacts/phase_e_eval/phase_e_pb_transfer_repairs_0311_1655_e80_ms_processbench_transfer_fanout_seed42_e80_ms_processbench_transfer_fanout_seed42_s42_processbench_gsm8k_20260310T170319Z|assets/artifacts/phase_e_eval/phase_e_pb_transfer_repairs_0311_1655_e80_ms_processbench_transfer_fanout_seed42_e80_ms_processbench_transfer_fanout_seed42_s42_processbench_math_20260310T170333Z' \
  --audit-spec 'e84_fanout_terminal_heavy|math_shepherd|assets/artifacts/phase_e_samefamily_eval/phase_e_rlpromo_0311_e84_samefamily_20260311T033833Z|assets/artifacts/phase_e_eval/phase_e_processbench_terminal_focus_0311_e84_ms_processbench_transfer_fanout_terminal_seed42_e84_ms_processbench_transfer_fanout_terminal_seed42_s42_processbench_gsm8k_20260311T022312Z|assets/artifacts/phase_e_eval/phase_e_processbench_terminal_focus_0311_e84_ms_processbench_transfer_fanout_terminal_seed42_e84_ms_processbench_transfer_fanout_terminal_seed42_s42_processbench_math_20260311T022320Z' \
  --audit-spec 'e46_prm_local|prmbench_preview|assets/artifacts/phase_e_samefamily_eval/phase_e_rltops_0311_1124_prm_e46_samefamily_20260311T032656Z|assets/artifacts/phase_e_eval/phase_e_rltops_0311_1124_prm_e46_processbench_gsm8k_20260311T032704Z|assets/artifacts/phase_e_eval/phase_e_rltops_0311_1124_prm_e46_processbench_math_20260311T032713Z'
```

Missing same-family audits for the two Math-Shepherd transfer repairs:

```bash
CUDA_VISIBLE_DEVICES=3 \
python -u scripts/phase_e_eval_samefamily_trust.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_pb_transfer_repairs_0311_1655_e80_ms_processbench_transfer_fanout_seed42_e80_ms_processbench_transfer_fanout_seed42_s42_value_20260310T165030Z \
  --run-name phase_e_rlpromo_0311_e80_samefamily \
  --output-root assets/artifacts/phase_e_samefamily_eval \
  --batch-size 96 \
  --edge-weight-mode confidence \
  --rejection-coverages 1.0,0.8,0.6,0.4,0.2 \
  --pressure-sizes 2,4,8 \
  --pressure-repeats 6 \
  --require-cuda

CUDA_VISIBLE_DEVICES=3 \
python -u scripts/phase_e_eval_samefamily_trust.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_processbench_terminal_focus_0311_e84_ms_processbench_transfer_fanout_terminal_seed42_e84_ms_processbench_transfer_fanout_terminal_seed42_s42_value_20260311T021431Z \
  --run-name phase_e_rlpromo_0311_e84_samefamily \
  --output-root assets/artifacts/phase_e_samefamily_eval \
  --batch-size 96 \
  --edge-weight-mode confidence \
  --rejection-coverages 1.0,0.8,0.6,0.4,0.2 \
  --pressure-sizes 2,4,8 \
  --pressure-repeats 6 \
  --require-cuda
```

Prepare light-anchor repair artifacts:

```bash
python -u scripts/phase_e_prepare_mathshepherd_terminal_anchor_pairs.py \
  --run-name phase_e_rlpromo_0311_ms_fanout_ta015 \
  --max-local-pairs 12000 \
  --terminal-anchor-ratio 0.15 \
  --terminal-anchor-prefix-mode penultimate \
  --step-label-pair-mode first_bad_fanout

python -u scripts/phase_e_prepare_prmbench_terminal_anchor_pairs.py \
  --run-name phase_e_rlpromo_0311_prm_ta010 \
  --terminal-anchor-ratio 0.10
```

Math-side diagnostic failures that exposed the hidden bug:

```bash
CUDA_VISIBLE_DEVICES=3 \
python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rlpromo_0311_ms_fanout_ta015__16a79535c2e6/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rlpromo_0311_ms_fanout_ta015__16a79535c2e6/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_rlpromo_0311_ms_fanout_ta015_rankonly_warm_e80 \
  --output-root assets/artifacts/phase_e_runs \
  --objective-mode ranking_only \
  --learning-rate 2e-6 \
  --num-train-epochs 2 \
  --per-device-train-batch-size 64 \
  --per-device-eval-batch-size 64 \
  --max-length 1024 \
  --max-gpu-memory-gib 48 \
  --max-cpu-memory-gib 96 \
  --lambda-ranking 1.0 \
  --ranking-margin 0.02 \
  --ranking-target-space logit \
  --pair-weight-mode none \
  --checkpoint-selection-metric ranking_score \
  --seed 42 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda \
  --head-architecture mlp \
  --head-dropout-prob 0.05 \
  --head-mlp-hidden-size 1024 \
  --anti-saturation-weight 5e-4 \
  --anti-saturation-logit-threshold 3.5 \
  --feature-cache-mode read \
  --init-value-head-path assets/artifacts/phase_e_runs/phase_e_pb_transfer_repairs_0311_1655_e80_ms_processbench_transfer_fanout_seed42_e80_ms_processbench_transfer_fanout_seed42_s42_value_20260310T165030Z/best_value_head.pt
```

Direct cache audit that confirmed non-finite pooled features:

```bash
python - <<'PY'
from pathlib import Path
import torch

for key in [
    'phase_e_pair_features_5b618641826324449fd5fb12',
    'phase_e_pair_features_4b73d7ae477b24ca7b10c5db',
    'phase_e_pair_features_7d90df2f63220370f27ed7f3',
    'phase_e_pair_features_37a30ddecbf706815a14a2a8',
]:
    payload = torch.load(
        Path('assets/artifacts/phase_e_feature_cache') / key[:2] / key / 'payload.pt',
        map_location='cpu',
        weights_only=True,
    )
    features = payload['features']
    print(key, bool(torch.isnan(features).any()), bool(torch.isinf(features).any()))
PY
```

Stable sanitized retry:

```bash
CUDA_VISIBLE_DEVICES=3 \
python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rlpromo_0311_ms_fanout_ta015__16a79535c2e6/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rlpromo_0311_ms_fanout_ta015__16a79535c2e6/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_rlpromo_0311_ms_fanout_ta015_rankonly_warm_e80_dropnf_fix \
  --output-root assets/artifacts/phase_e_runs \
  --objective-mode ranking_only \
  --learning-rate 2e-6 \
  --num-train-epochs 2 \
  --per-device-train-batch-size 64 \
  --per-device-eval-batch-size 64 \
  --max-length 1024 \
  --max-gpu-memory-gib 48 \
  --max-cpu-memory-gib 96 \
  --lambda-ranking 1.0 \
  --ranking-margin 0.02 \
  --ranking-target-space logit \
  --pair-weight-mode none \
  --checkpoint-selection-metric ranking_score \
  --seed 42 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda \
  --head-architecture mlp \
  --head-dropout-prob 0.05 \
  --head-mlp-hidden-size 1024 \
  --anti-saturation-weight 5e-4 \
  --anti-saturation-logit-threshold 3.5 \
  --feature-cache-mode read \
  --nonfinite-feature-policy drop \
  --init-value-head-path assets/artifacts/phase_e_runs/phase_e_pb_transfer_repairs_0311_1655_e80_ms_processbench_transfer_fanout_seed42_e80_ms_processbench_transfer_fanout_seed42_s42_value_20260310T165030Z/best_value_head.pt
```

Same-family + ProcessBench re-eval for the sanitized retry:

```bash
CUDA_VISIBLE_DEVICES=3 \
python -u scripts/phase_e_eval_samefamily_trust.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_rlpromo_0311_ms_fanout_ta015_rankonly_warm_e80_dropnf_fix_20260311T044122Z \
  --run-name phase_e_rlpromo_0311_ms_fanout_ta015_dropnf_samefamily \
  --output-root assets/artifacts/phase_e_samefamily_eval \
  --batch-size 96 \
  --edge-weight-mode confidence \
  --rejection-coverages 1.0,0.8,0.6,0.4,0.2 \
  --pressure-sizes 2,4,8 \
  --pressure-repeats 6 \
  --require-cuda

CUDA_VISIBLE_DEVICES=3 \
python -u scripts/phase_e_eval_benchmark.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_rlpromo_0311_ms_fanout_ta015_rankonly_warm_e80_dropnf_fix_20260311T044122Z \
  --benchmark-id processbench_gsm8k \
  --run-name phase_e_rlpromo_0311_ms_fanout_ta015_dropnf_pb_gsm \
  --output-root assets/artifacts/phase_e_eval \
  --checkpoint-name best \
  --batch-size 96 \
  --require-cuda

CUDA_VISIBLE_DEVICES=0 \
python -u scripts/phase_e_eval_benchmark.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_rlpromo_0311_ms_fanout_ta015_rankonly_warm_e80_dropnf_fix_20260311T044122Z \
  --benchmark-id processbench_math \
  --run-name phase_e_rlpromo_0311_ms_fanout_ta015_dropnf_pb_math \
  --output-root assets/artifacts/phase_e_eval \
  --checkpoint-name best \
  --batch-size 96 \
  --require-cuda

python -u scripts/phase_e_diagnose_rl_promotion.py \
  --run-name phase_e_rlpromo_diag_math_dropnf_0311 \
  --audit-spec 'e80_fanout|math_shepherd|assets/artifacts/phase_e_samefamily_eval/phase_e_rlpromo_0311_e80_samefamily_20260311T033647Z|assets/artifacts/phase_e_eval/phase_e_pb_transfer_repairs_0311_1655_e80_ms_processbench_transfer_fanout_seed42_e80_ms_processbench_transfer_fanout_seed42_s42_processbench_gsm8k_20260310T170319Z|assets/artifacts/phase_e_eval/phase_e_pb_transfer_repairs_0311_1655_e80_ms_processbench_transfer_fanout_seed42_e80_ms_processbench_transfer_fanout_seed42_s42_processbench_math_20260310T170333Z' \
  --audit-spec 'e84_fanout_terminal_heavy|math_shepherd|assets/artifacts/phase_e_samefamily_eval/phase_e_rlpromo_0311_e84_samefamily_20260311T033833Z|assets/artifacts/phase_e_eval/phase_e_processbench_terminal_focus_0311_e84_ms_processbench_transfer_fanout_terminal_seed42_e84_ms_processbench_transfer_fanout_terminal_seed42_s42_processbench_gsm8k_20260311T022312Z|assets/artifacts/phase_e_eval/phase_e_processbench_terminal_focus_0311_e84_ms_processbench_transfer_fanout_terminal_seed42_e84_ms_processbench_transfer_fanout_terminal_seed42_s42_processbench_math_20260311T022320Z' \
  --audit-spec 'e80_dropnf_rankrepair|math_shepherd|assets/artifacts/phase_e_samefamily_eval/phase_e_rlpromo_0311_ms_fanout_ta015_dropnf_samefamily_20260311T044612Z|assets/artifacts/phase_e_eval/phase_e_rlpromo_0311_ms_fanout_ta015_dropnf_pb_gsm_20260311T044635Z|assets/artifacts/phase_e_eval/phase_e_rlpromo_0311_ms_fanout_ta015_dropnf_pb_math_20260311T044656Z'
```

PRMBench light continuation attempted but not completed this round:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rlpromo_0311_prm_ta010__b2281c74f155/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rlpromo_0311_prm_ta010__b2281c74f155/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_rlpromo_0311_prm_ta010_warm_e46_nocache \
  --output-root assets/artifacts/phase_e_runs \
  --objective-mode joint \
  --learning-rate 5e-6 \
  --num-train-epochs 2 \
  --per-device-train-batch-size 64 \
  --per-device-eval-batch-size 64 \
  --max-length 1024 \
  --max-gpu-memory-gib 48 \
  --max-cpu-memory-gib 96 \
  --lambda-ranking 1.0 \
  --lambda-bce 1.0 \
  --ranking-margin 0.02 \
  --ranking-target-space logit \
  --pair-weight-mode none \
  --checkpoint-selection-metric pair_acc \
  --seed 42 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda \
  --head-architecture mlp \
  --head-dropout-prob 0.05 \
  --head-mlp-hidden-size 1024 \
  --anti-saturation-weight 5e-4 \
  --anti-saturation-logit-threshold 3.5 \
  --feature-cache-mode off \
  --init-value-head-path assets/artifacts/phase_e_runs/phase_e_prmbench_acc90_full_0310_1914_e46_prmbench_acc90_mlp_joint_seed3_e46_prmbench_acc90_mlp_joint_seed3_s42_value_20260310T113722Z/best_value_head.pt
```

### Main result

The current best reading is:
1. no audited candidate clears the strict RL-promotion gate,
2. the repository is stronger in *infrastructure integrity* than before,
3. but candidate quality is still not at RL-ready level.

## 2026-03-11 Low-terminal mixed repair retry update

This round added a more targeted RL-facing repair attempt instead of another
generic same-source sweep.

Code / config changes:
1. `src/ours/phase_d/external_pairs_adapters.py`
   - `load_prmbench_preview_pairs()` now records
     - `positive_step_index`
     - `negative_step_index`
   - this fixes the missing local-step metadata contract for PRMBench pairs.
2. `scripts/run_phase_e_suite.sh`
   - added:
     - `E89_MS_PRMBENCH_TRANSFER_MIX_TERMINAL10_CONFWT_WARM_E82_SEED42`
     - `E90_MS_PRMBENCH_TRANSFER_MIX_TERMINAL05_CONFWT_WARM_E82_SEED42`
   - both warm-start from `E82` and probe low terminal-anchor budgets.
3. `tests/unit/test_phase_d_external_pairs.py`
   - added a PRMBench loader contract test covering:
     - step-index normalization
     - `positive_step_index`
     - `negative_step_index`

Key experiment state:
1. rebuilt PRMBench local-diagnostic pair artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_prmbench_localdiag_0311_e46_rebuild_sharedsplit_s42_pairs__f5778317f28b`
2. refreshed `E46` PRMBench same-family eval:
   - `assets/artifacts/phase_e_samefamily_eval/phase_e_prmbench_localdiag_0311_e46_samefamily_20260311T033803Z`
   - `prompt_pool_top1_accuracy = 0.9659`
3. refreshed `E82` same-family eval:
   - `assets/artifacts/phase_e_samefamily_eval/phase_e_rlpush_lowterm_0311_e82_samefamily_20260311T034620Z`
   - `prompt_pool_top1_accuracy = 0.9633`
   - `local_first_bad_edge_accuracy = 0.9841`
4. new repair pair artifacts:
   - `assets/artifacts/phase_e_pairs/phase_e_rlpush_lowterm_0311_e89_e89_ms_prmbench_transfer_mix_terminal10_confwt_warm_e82_seed42_sharedsplit_s42_pairs__9e47a4b941d8`
   - `assets/artifacts/phase_e_pairs/phase_e_rlpush_lowterm_0311_e90_e90_ms_prmbench_transfer_mix_terminal05_confwt_warm_e82_seed42_sharedsplit_s42_pairs__e9ecdd65f92b`

Runtime outcome:
1. `E89` failed during frozen-backbone encoding with:
   - `RuntimeError: CUDA error: unspecified launch failure`
2. the first parallel `E90` attempt was stopped after showing the same unstable
   large-run path.
3. a safer retry was launched with:
   - `feature_cache_mode=off`
   - `max_gpu_memory_gib=48`
   - `max_cpu_memory_gib=96`
   - `per_device_eval_batch_size=48`
   - but it did not finish within this turn, so no benchmark result should be
     promoted from it yet.

Important diagnosis:
1. `PRMBench` local auditing is still incomplete even after metadata repair.
2. reason:
   - current same-family local metrics assume
     `last_safe_prefix vs first_bad_prefix`,
   - PRMBench supervision is
     `same-step correct sibling vs wrong sibling`.
3. so the remaining gap is now clearly a metric-definition issue, not a missing
   metadata issue anymore.

Commands run:
```bash
PYTHONPATH=src pytest -q tests/unit/test_phase_d_external_pairs.py
bash -n scripts/run_phase_e_suite.sh
python -u scripts/phase_e_prepare_pairs.py --source-bundle prmbench_preview --run-name phase_e_prmbench_localdiag_0311_e46_rebuild_sharedsplit_s42_pairs --output-root assets/artifacts/phase_e_pairs --seed 42 --validation-ratio 0.1 --split-granularity pair_id --max-pairs-total 6000 --max-pairs-per-source 6000 --min-pair-confidence 0.8 --step-label-pair-mode first_bad_edge_strict
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_e_eval_samefamily_trust.py --value-run-dir assets/artifacts/phase_e_runs/phase_e_prmbench_acc90_full_0310_1914_e46_prmbench_acc90_mlp_joint_seed3_e46_prmbench_acc90_mlp_joint_seed3_s43_value_20260310T113737Z --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_prmbench_localdiag_0311_e46_rebuild_sharedsplit_s42_pairs__f5778317f28b/validation_pairs.jsonl --run-name phase_e_prmbench_localdiag_0311_e46_samefamily --output-root assets/artifacts/phase_e_samefamily_eval --checkpoint-name best --batch-size 96 --feature-cache-root assets/artifacts/phase_e_feature_cache --feature-cache-mode read_write --feature-cache-lock-timeout-sec 600 --edge-weight-mode confidence --require-cuda
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_e_eval_samefamily_trust.py --value-run-dir assets/artifacts/phase_e_runs/phase_e_pb_transfer_mix_0311_1722_e82_ms_prmbench_transfer_mix_seed42_e82_ms_prmbench_transfer_mix_seed42_s42_value_20260310T171945Z --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_pb_transfer_mix_0311_1722_e82_ms_prmbench_transfer_mix_seed42_e82_ms_prmbench_transfer_mix_seed42_sharedsplit_s42_pairs__ae568fa2f36e/validation_pairs.jsonl --run-name phase_e_rlpush_lowterm_0311_e82_samefamily --output-root assets/artifacts/phase_e_samefamily_eval --checkpoint-name best --batch-size 96 --feature-cache-root assets/artifacts/phase_e_feature_cache --feature-cache-mode read_write --feature-cache-lock-timeout-sec 600 --edge-weight-mode confidence --require-cuda
CUDA_VISIBLE_DEVICES=2 ACTIVE_PHASE_E_GROUP=E89_MS_PRMBENCH_TRANSFER_MIX_TERMINAL10_CONFWT_WARM_E82_SEED42 RUN_PREFIX=phase_e_rlpush_lowterm_0311_e89 bash scripts/run_phase_e_suite.sh
CUDA_VISIBLE_DEVICES=1 ACTIVE_PHASE_E_GROUP=E90_MS_PRMBENCH_TRANSFER_MIX_TERMINAL05_CONFWT_WARM_E82_SEED42 RUN_PREFIX=phase_e_rlpush_lowterm_0311_e90 bash scripts/run_phase_e_suite.sh
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u scripts/phase_e_train_value.py --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rlpush_lowterm_0311_e90_e90_ms_prmbench_transfer_mix_terminal05_confwt_warm_e82_seed42_sharedsplit_s42_pairs__e9ecdd65f92b/train_pairs.jsonl --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rlpush_lowterm_0311_e90_e90_ms_prmbench_transfer_mix_terminal05_confwt_warm_e82_seed42_sharedsplit_s42_pairs__e9ecdd65f92b/validation_pairs.jsonl --model-path assets/models/Qwen2.5-7B-Instruct --run-name phase_e_rlpush_lowterm_0311_e90_retry_nocache --output-root assets/artifacts/phase_e_runs --objective-mode joint --learning-rate 2e-5 --num-train-epochs 4 --per-device-train-batch-size 64 --per-device-eval-batch-size 48 --pair-weight-mode confidence --source-balance uniform --permutation-mode stable_hash --checkpoint-selection-metric pair_acc --seed 42 --dtype bfloat16 --device-map auto --max-gpu-memory-gib 48 --max-cpu-memory-gib 96 --feature-cache-root assets/artifacts/phase_e_feature_cache --feature-cache-mode off --feature-cache-lock-timeout-sec 600 --ranking-target-space logit --lambda-ranking 1.0 --lambda-bce 1.0 --ranking-margin 0.02 --anti-saturation-weight 5e-4 --anti-saturation-logit-threshold 3.5 --init-value-head-path assets/artifacts/phase_e_runs/phase_e_pb_transfer_mix_0311_1722_e82_ms_prmbench_transfer_mix_seed42_e82_ms_prmbench_transfer_mix_seed42_s42_value_20260310T171945Z/best_value_head.pt --head-architecture mlp --head-dropout-prob 0.05 --head-init-std 0.02 --head-mlp-hidden-size 1024 --head-activation gelu --require-cuda --strict-determinism
```

## 2026-03-11 RL-Readiness Candidate Refresh

Latest RL-readiness summaries:
- top candidates:
  - `assets/artifacts/phase_e_logs/phase_e_rltops_0311_1124/final_summary.md`
- repair pilots:
  - `assets/artifacts/phase_e_logs/phase_e_rlrepairs_0311_1124/final_summary.md`

Current bounded-support RL candidates:
1. `ms_e43`
   - source:
     - `Math-Shepherd`
   - same-family:
     - `pool_top1 = 0.9648`
     - `local_first_bad = 0.9702`
     - `rej40_gain = 0.0352`
   - benchmark:
     - `gsm8k_auc = 0.6245`
     - `math_auc = 0.6341`
   - assessment:
     - `provisionally_rl_ready`
2. `prm_e46`
   - source:
     - `PRMBench_Preview`
   - same-family:
     - `pool_top1 = 0.9659`
     - `rej40_gain = 0.0341`
   - benchmark:
     - `gsm8k_auc = 0.6264`
     - `math_auc = 0.6053`
   - assessment:
     - `provisionally_rl_ready`

Most important caution:
1. `prm_e78` still looks very strong on same-family metrics,
2. but falls to:
   - `gsm8k_auc = 0.5398`
   - `math_auc = 0.5117`
3. interpretation:
   - stronger same-source fitting does not automatically produce a safer RL prior.

Best repair direction to scale:
1. `ms_grid_micro`
   - `pool_top1 = 0.9982`
   - `local_first_bad = 0.9914`
   - `gsm8k_auc = 0.5891`
   - `math_auc = 0.5559`
   - weakness:
     - `rej40_gain = 0.0018`
   - interpretation:
     - this is the best current local-geometry repair direction, but it still
       needs an explicit rejection-utility improvement pass.

Direct rerun commands:

```bash
cd /home/zling/y/bcr/ref

ACTIVE_PHASE_E_RL_GROUP=RR5_COMPARE_INTRADATASET_TOPS \
RUN_PREFIX=phase_e_rltops_$(date +%m%d_%H%M) \
RL_AUDIT_BATCH_SIZE=96 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_rl_readiness_suite.sh
```

```bash
cd /home/zling/y/bcr/ref

ACTIVE_PHASE_E_RL_GROUP=RR6_COMPARE_REPAIR_PILOTS \
RUN_PREFIX=phase_e_rlrepairs_$(date +%m%d_%H%M) \
RL_AUDIT_BATCH_SIZE=96 \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_e_rl_readiness_suite.sh
```

## 2026-03-11 R-PRM Compact Train-Fit vs Held-Out Gap Update

This is the newest `R-PRM` conclusion and should now be treated as the
repository baseline reading for this source.

Representative artifacts:
- full compact pair artifact:
  - `assets/artifacts/phase_e_pairs/phase_e_rprm_full_compact_fix__8d69afd6dba5`
- train-distribution fit probe:
  - `assets/artifacts/phase_e_runs/phase_e_rprm_trainfit_probe_0310_s4k_20260310T160730Z/summary.json`
- held-out repair probe:
  - `assets/artifacts/phase_e_runs/phase_e_rprm_heldout_repair_0310_s4k_20260310T164443Z/summary.json`

What changed:
1. a real truncation-gate bug in `src/ours/phase_e/training.py` was fixed;
2. the repaired compact contract was then tested on a larger and cleaner
   same-source setup.

New evidence:
1. train-distribution fit probe:
   - `pair_acc = 0.9090`
   - `auc = 0.9131`
2. matching true held-out probe:
   - `pair_acc = 0.6280`
   - `auc = 0.6508`

Interpretation:
1. current `R-PRM compact_verdict` is not unlearnable;
2. the current head/objective family can fit the train distribution strongly;
3. the remaining blocker is held-out generalization under the present compact
   supervision contract.

Why this matters:
1. this rules out the overly simple explanation:
   - "the head is just too weak"
2. and it also sharpens the previous diagnosis:
   - the main remaining problem is not length,
   - and not just verdict-polarity imbalance,
   - but supervision-contract mismatch.

## 2026-03-11 Phase E Source-Specific Conclusion Update

This repository now has enough same-source evidence to stop treating all
external pair datasets as one generic supervision family.

Three separate conclusions are now supported:

1. `Math-Shepherd` is a strong same-source training source.
2. `PRMBench_Preview` is also a strong same-source training source.
3. `R-PRM` is not failing mainly because of legacy truncation anymore; it is
   now mainly limited by supervision-contract mismatch under the current frozen
   feature scorer.

### `Math-Shepherd`

Representative artifacts:
- `assets/artifacts/phase_e_logs/phase_e_ms_acc90_full_0310_1914_e41_ms_acc90_mlp_rank_seed3/final_summary.md`
- `assets/artifacts/phase_e_logs/phase_e_ms_acc95_push_0310_2146/final_summary.md`

Key reading:
- `E41`
  - `pair_acc=0.9850`
  - `auc=0.9034`
- `E68`
  - `pair_acc=0.9725`
  - `auc=0.9415`

Interpretation:
- same-source `Math-Shepherd` fitting is already solved under the current Phase E stack.
- Remaining errors concentrate on later `first_bad_edge` positions and longer
  step chains, not on generic learnability.

### `PRMBench_Preview`

Representative artifacts:
- `assets/artifacts/phase_e_logs/phase_e_prmbench_acc90_full_0310_1914/final_summary.md`
- `assets/artifacts/phase_e_logs/phase_e_prmbench_acc95_push_0310_2359/final_summary.md`

Key reading:
- `E46`
  - `pair_acc=0.9309`
  - `auc=0.9057`
- `E78`
  - `pair_acc=0.9521`
  - `auc=0.9071`

Interpretation:
- `PRMBench_Preview` can now clear the `95%` same-source pair-accuracy target.
- This matters because it proves the current trainer is not generally weak.
- It is source-sensitive.

### `R-PRM`

Representative artifacts:
- root-cause audit:
  - `assets/artifacts/phase_e_logs/phase_e_rprm_deep_diag_0310_2359/final_summary.md`
- compact length diagnosis:
  - `assets/artifacts/phase_e_logs/phase_e_rprm_diag_0310_2019/final_summary.md`
- polarity-repair runs:
  - `assets/artifacts/phase_e_runs/phase_e_rprm_correctness_diag_0310_2341_bce2048_20260310T154119Z/summary.json`
  - `assets/artifacts/phase_e_runs/phase_e_rprm_correctness_diag_0310_2341_bce2048_vbal_20260310T154500Z/summary.json`
  - `assets/artifacts/phase_e_runs/phase_e_rprm_correctness_diag_0310_2341_joint2048_vbal_logit_20260310T154937Z/summary.json`

Current best compact-contract baseline:
- `C9_MLP_BCE_2048`
  - `pair_acc=0.6694`
  - `auc=0.6571`

What the new polarity-repair runs show:
- `compact_correctness + BCE + 2048`
  - `pair_acc=0.6481`
  - `auc=0.6519`
- `compact_correctness + BCE + verdict_balance`
  - `pair_acc=0.5926`
  - `auc=0.5765`
- `compact_correctness + joint + logit + verdict_balance`
  - `pair_acc=0.5926`
  - `auc=0.6306`

Key interpretation:
- old `compact_verdict` runs had a strong `chosen=no` vs `chosen=yes` asymmetry.
- the new `compact_correctness + joint + verdict_balance` run almost removes
  that asymmetry:
  - chosen=`no`: `0.5968`
  - chosen=`yes`: `0.5870`
- but overall accuracy still stays far below `ACC90`.

This is the critical diagnosis:
- `R-PRM` no longer mainly fails because of truncation.
- `R-PRM` no longer mainly fails because of a simple verdict-polarity bias.
- `R-PRM` now mainly fails because the compact supervision contract is too weak
  for the current frozen feature head.

Therefore:
- `R-PRM` should not currently be treated as a primary same-source
  high-accuracy training source.
- It should be treated as a verifier-style source that likely needs a different
  model contract, not just more tuning.

## 2026-03-11 R-PRM Large-Artifact Verification Commands

These are the exact commands used to tighten the current `R-PRM` diagnosis
without relying on old smoke-scale artifacts.

Prepare the repaired full compact artifact:

```bash
CUDA_VISIBLE_DEVICES=1 \
python -u scripts/phase_e_prepare_pairs.py \
  --source-bundle r_prm_train \
  --run-name phase_e_rprm_full_compact_fix \
  --output-root assets/artifacts/phase_e_pairs \
  --seed 42 \
  --validation-ratio 0.1 \
  --split-granularity source_sample \
  --min-pair-confidence 0.75 \
  --r-prm-pair-mode compact_verdict \
  --r-prm-root assets/external_datasets/kevinpro_r_prm
```

Train-distribution fit ceiling probe:

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rprm_full_compact_fix__8d69afd6dba5/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rprm_full_compact_fix__8d69afd6dba5/train_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_rprm_trainfit_probe_0310_s4k \
  --output-root assets/artifacts/phase_e_runs \
  --max-train-samples 4000 \
  --max-eval-samples 1000 \
  --objective-mode joint \
  --learning-rate 2e-5 \
  --weight-decay 0.0 \
  --num-train-epochs 16 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 12 \
  --gradient-accumulation-steps 1 \
  --warmup-ratio 0.05 \
  --max-grad-norm 1.0 \
  --max-length 2048 \
  --lambda-ranking 1.0 \
  --lambda-bce 1.0 \
  --ranking-margin 0.02 \
  --ranking-target-space logit \
  --pair-weight-mode none \
  --source-balance none \
  --permutation-mode stable_hash \
  --anti-saturation-weight 1e-4 \
  --anti-saturation-logit-threshold 3.0 \
  --checkpoint-selection-metric pair_acc \
  --seed 42 \
  --dtype bfloat16 \
  --device-map auto \
  --strict-determinism \
  --head-architecture mlp \
  --head-dropout-prob 0.0 \
  --head-init-std 0.02 \
  --head-mlp-hidden-size 2048 \
  --head-activation gelu \
  --feature-cache-root assets/artifacts/phase_e_feature_cache \
  --feature-cache-mode read_write \
  --max-truncation-over-limit-fraction 0.01 \
  --require-cuda
```

Matching true held-out probe:

```bash
CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rprm_full_compact_fix__8d69afd6dba5/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rprm_full_compact_fix__8d69afd6dba5/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_rprm_heldout_repair_0310_s4k \
  --output-root assets/artifacts/phase_e_runs \
  --max-train-samples 4000 \
  --max-eval-samples 1000 \
  --objective-mode joint \
  --learning-rate 2e-5 \
  --weight-decay 0.0 \
  --num-train-epochs 16 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 12 \
  --gradient-accumulation-steps 1 \
  --warmup-ratio 0.05 \
  --max-grad-norm 1.0 \
  --max-length 2048 \
  --lambda-ranking 1.0 \
  --lambda-bce 1.0 \
  --ranking-margin 0.02 \
  --ranking-target-space logit \
  --pair-weight-mode none \
  --source-balance none \
  --permutation-mode stable_hash \
  --anti-saturation-weight 1e-4 \
  --anti-saturation-logit-threshold 3.0 \
  --checkpoint-selection-metric pair_acc \
  --seed 42 \
  --dtype bfloat16 \
  --device-map auto \
  --strict-determinism \
  --head-architecture mlp \
  --head-dropout-prob 0.0 \
  --head-init-std 0.02 \
  --head-mlp-hidden-size 2048 \
  --head-activation gelu \
  --feature-cache-root assets/artifacts/phase_e_feature_cache \
  --feature-cache-mode read_write \
  --max-truncation-over-limit-fraction 0.01 \
  --require-cuda
```

## 2026-03-10 RL-Readiness Audit Of Current Value-Head Candidates

Why this block exists:
- The repository had already shown strong same-source held-out fitting on some
  sources.
- But that still did not answer the more practical question:
  - is any current value head good enough for *conservative RL-style use*?
- So this round audited the strongest current checkpoints under:
  - same-family reranking,
  - rejection / abstention,
  - stronger best-of-N pressure,
  - and ProcessBench re-evaluation.

New wrapper added:
- `scripts/run_phase_e_rl_readiness_suite.sh`

Main audit command used:

```bash
CUDA_VISIBLE_DEVICES=2 \
ACTIVE_PHASE_E_RL_GROUP=RR4_COMPARE_CURRENT_TOPS \
RUN_PREFIX=phase_e_rl_readiness_0310_2338 \
bash scripts/run_phase_e_rl_readiness_suite.sh
```

Why `CUDA_VISIBLE_DEVICES=2`:
- `GPU 2` was idle at launch,
- `GPU 1` and `GPU 3` were already busy,
- and this kept the main audit sequential and predictable on the shared server.

Main artifact:
- `assets/artifacts/phase_e_logs/phase_e_rl_readiness_0310_2338/final_summary.md`

Audit targets:
- `ms_e68`
  - strongest current same-source Math-Shepherd winner
- `ms_e14`
  - benchmark-aware Math-Shepherd trust candidate
- `prm_e46`
  - strongest current PRMBench same-source winner

Main results:
- `ms_e68`
  - same-family:
    - `prompt_pool_top1=0.9793`
    - `local_first_bad_acc=0.9779`
  - ProcessBench:
    - `gsm8k_auc=0.5885`
    - `math_auc=0.5547`
  - reading:
    - excellent same-family reranker
    - benchmark behavior is positive but not the cleanest overall RL candidate
- `ms_e14`
  - same-family:
    - `prompt_pool_top1=0.8584`
    - `local_first_bad_acc=0.8664`
  - ProcessBench:
    - `gsm8k_auc=0.5026`
    - `math_auc=0.5138`
  - reading:
    - useful signal
    - not benchmark-safe enough to promote
- `prm_e46`
  - same-family:
    - `prompt_pool_top1=0.9659`
    - `rejection@0.4=1.0000`
    - `pressure@8=0.9375`
  - ProcessBench:
    - `gsm8k_auc=0.6264`
    - `math_auc=0.6053`
  - reading:
    - first current checkpoint that survives both same-family and benchmark-native audit strongly enough to count as a bounded-support RL candidate

Important bug found during this audit:
- the first wrapper summary incorrectly read benchmark AUC from a generic
  `auc` key
- `ProcessBench` actually reports:
  - `pair_auc_good_vs_bad`
- the wrapper was fixed and the final summary was regenerated from the
  completed artifacts without rerunning the full audit

Extra pressure-stress checks:

```bash
CUDA_VISIBLE_DEVICES=2 \
python -u scripts/phase_e_eval_samefamily_trust.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_ms_acc95_push_0310_2146_e68_ms_acc95_joint_logit_seed42_e68_ms_acc95_joint_logit_seed42_s42_value_20260310T151651Z \
  --run-name phase_e_rl_pressure_stress_0310_2340_ms_e68 \
  --output-root assets/artifacts/phase_e_samefamily_eval \
  --batch-size 96 \
  --edge-weight-mode confidence \
  --rejection-coverages 1.0,0.8,0.6,0.4,0.2,0.1 \
  --pressure-sizes 2,4,8,12,16 \
  --pressure-repeats 8

CUDA_VISIBLE_DEVICES=0 \
python -u scripts/phase_e_eval_samefamily_trust.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_prmbench_acc90_full_0310_1914_e46_prmbench_acc90_mlp_joint_seed3_e46_prmbench_acc90_mlp_joint_seed3_s43_value_20260310T113737Z \
  --run-name phase_e_rl_pressure_stress_0310_2340_prm_e46 \
  --output-root assets/artifacts/phase_e_samefamily_eval \
  --batch-size 96 \
  --edge-weight-mode confidence \
  --rejection-coverages 1.0,0.8,0.6,0.4,0.2,0.1 \
  --pressure-sizes 2,4,8,12,16 \
  --pressure-repeats 8
```

Why `CUDA_VISIBLE_DEVICES=0` was acceptable in the second stress command:
- at that moment `GPU 0` and `GPU 2` were both idle,
- the jobs were eval-only and reused cached features,
- parallelizing them shortened total occupancy time instead of extending it.

Stress artifacts:
- `assets/artifacts/phase_e_samefamily_eval/phase_e_rl_pressure_stress_0310_2340_ms_e68_20260310T153847Z/summary.md`
- `assets/artifacts/phase_e_samefamily_eval/phase_e_rl_pressure_stress_0310_2340_prm_e46_20260310T153848Z/summary.md`

Stress takeaways:
- `E68`
  - `rejection@0.1=1.0000`
  - `pressure@12=1.0000`
  - same-family decision pressure is not the bottleneck
- `E46`
  - `rejection@0.1=1.0000`
  - `pressure@4=0.9411`
  - `pressure@8=0.9375`
  - still robust under stronger selection pressure

Bottom-line reading:
- the repository now has one checkpoint family that is usable for:
  - conservative reranking
  - rejection / abstention
  - low-weight reward prior
  - math-family process filtering
- the best current candidate is:
  - `prm_e46`
- but this is still **not** the same as proving:
  - unrestricted high-weight RL reward-model readiness
- missing evidence still includes:
  - true closed-loop RL improvement
  - reward-hacking / exploitation resistance
  - broader cross-domain robustness

## 2026-03-10 Math-Shepherd ACC95 Verification And Push Matrix

Why this block exists:
- The request for this round was to independently verify current
  `Math-Shepherd` performance, try low-risk fixes if necessary, and make sure
  the source reaches `95%` held-out ACC without triggering avoidable OOM on a
  shared server.
- But older repository artifacts already suggested the target had been crossed,
  so the real job here was:
  - verify that the current code state still reproduces `>95%`,
  - test a few focused fixes,
  - and identify the most effective one instead of blindly pushing capacity.

Resource hygiene:
- `nvidia-smi` was checked before launch.
- All four `A100 80GB` devices were idle at the start.
- The run still used only `CUDA_VISIBLE_DEVICES=1` to stay conservative and
  avoid unnecessary multi-GPU memory/cache pressure on a shared machine.

New one-click groups added:
- `E67_MS_ACC95_JOINT_VERIFY_SEED42`
- `E68_MS_ACC95_JOINT_LOGIT_SEED42`
- `E69_MS_ACC95_JOINT_OVERFIT_SEED42`
- wrapper:
  - `I6_MS_ACC95_PUSH_MATRIX`

Command used:

```bash
CUDA_VISIBLE_DEVICES=1 \
ACTIVE_PHASE_E_INTRADATASET_GROUP=I6_MS_ACC95_PUSH_MATRIX \
RUN_PREFIX=phase_e_ms_acc95_push_0310_2146 \
bash scripts/run_phase_e_intradataset_suite.sh
```

Main artifacts:
- suite summary:
  - `assets/artifacts/phase_e_logs/phase_e_ms_acc95_push_0310_2146/final_summary.md`
- candidate report:
  - `assets/artifacts/phase_e_candidates/phase_e_ms_acc95_push_0310_2146_candidate/candidate_report.md`

Results:
- `E67_MS_ACC95_JOINT_VERIFY_SEED42`
  - `pair_acc=0.963267`
  - `auc=0.942383`
  - `ranking_score=0.952825`
- `E68_MS_ACC95_JOINT_LOGIT_SEED42`
  - `pair_acc=0.972450`
  - `auc=0.941545`
  - `ranking_score=0.956998`
- `E69_MS_ACC95_JOINT_OVERFIT_SEED42`
  - `pair_acc=0.966167`
  - `auc=0.945630`
  - `ranking_score=0.955898`

Candidate selection:
- selected group:
  - `E68_MS_ACC95_JOINT_LOGIT_SEED42`
- `trust_score=0.968689`
- checkpoint:
  - `assets/artifacts/phase_e_runs/phase_e_ms_acc95_push_0310_2146_e68_ms_acc95_joint_logit_seed42_e68_ms_acc95_joint_logit_seed42_s42_value_20260310T151651Z/best_value_head.pt`

Interpretation:
- `Math-Shepherd` was already above `95%` before this round.
- This new run confirms the current code state still reproduces that regime.
- The best low-risk improvement is:
  - `joint + ranking_target_space=logit`
- More aggressive "overfit push" changes did not beat the logit-space tweak.
- So on this source, the current best explanation is:
  - the stack already learns the source well,
  - and geometry choice matters more than simply turning down regularization.

## 2026-03-10 R-PRM Root-Cause Diagnosis

Why this block exists:
- `R-PRM` had contradictory signals across old Phase E runs.
- We therefore separated three questions:
  - legacy vs compact pair-contract truncation risk,
  - repaired-contract learnability,
  - and whether invalid legacy settings now fail fast before model load.

Key commands used:

```bash
python -u scripts/phase_e_prepare_pairs.py \
  --source-bundle r_prm_train \
  --run-name phase_e_rprm_contractcmp_0310_2055_legacy_pairs \
  --output-root assets/artifacts/phase_e_pairs \
  --seed 42 \
  --split-granularity pair_id \
  --max-pairs-total 2500 \
  --max-pairs-per-source 2500 \
  --min-pair-confidence 0.75 \
  --r-prm-pair-mode direct_pair_legacy

python -u scripts/phase_e_prepare_pairs.py \
  --source-bundle r_prm_train \
  --run-name phase_e_rprm_contractcmp_0310_2055_compact_pairs \
  --output-root assets/artifacts/phase_e_pairs \
  --seed 42 \
  --split-granularity pair_id \
  --max-pairs-total 2500 \
  --max-pairs-per-source 2500 \
  --min-pair-confidence 0.75 \
  --r-prm-pair-mode compact_verdict

python -u scripts/phase_e_diagnose_truncation.py \
  --pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rprm_contractcmp_0310_2055_legacy_pairs__efc9444c97f8/train_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_rprm_contractcmp_0310_2055_legacy_train_diag \
  --output-root assets/artifacts/phase_e_truncation_diagnostics \
  --batch-size 64 \
  --max-lengths 768 1024 1536 2048

python -u scripts/phase_e_diagnose_truncation.py \
  --pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rprm_contractcmp_0310_2055_compact_pairs__ca03cf0c9aa1/train_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_rprm_contractcmp_0310_2055_compact_train_diag \
  --output-root assets/artifacts/phase_e_truncation_diagnostics \
  --batch-size 64 \
  --max-lengths 768 1024 1536 2048

python -u scripts/phase_e_audit_rprm_contract.py \
  --r-prm-root assets/external_datasets/kevinpro_r_prm \
  --split train \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_rprm_contractaudit_0310_2100 \
  --output-root assets/artifacts/phase_e_rprm_audit \
  --max-rows 4000 \
  --max-lengths 1024 1536 2048

CUDA_VISIBLE_DEVICES=3 \
ACTIVE_PHASE_E_RPRM_DIAG_GROUP=RD2_RPRM_RECIPE_MATRIX_SMOKE \
RUN_PREFIX=phase_e_rprm_recipe_smoke_0310_2105 \
bash scripts/run_phase_e_rprm_diagnostic_suite.sh

python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rprm_contractcmp_0310_2055_legacy_pairs__efc9444c97f8/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_rprm_contractcmp_0310_2055_legacy_pairs__efc9444c97f8/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_rprm_legacy_failfast_0310_2100 \
  --output-root assets/artifacts/phase_e_runs \
  --objective-mode ranking_only \
  --learning-rate 5e-5 \
  --num-train-epochs 4 \
  --per-device-train-batch-size 128 \
  --per-device-eval-batch-size 128 \
  --pair-weight-mode none \
  --source-balance none \
  --permutation-mode stable_hash \
  --checkpoint-selection-metric pair_acc \
  --seed 42 \
  --dtype bfloat16 \
  --device-map auto \
  --feature-cache-root assets/artifacts/phase_e_feature_cache \
  --feature-cache-mode read_write \
  --feature-cache-lock-timeout-sec 600 \
  --max-length 1024 \
  --lambda-ranking 1.0 \
  --lambda-bce 0.25 \
  --ranking-margin 0.05 \
  --head-architecture mlp \
  --head-dropout-prob 0.05 \
  --head-init-std 0.02 \
  --head-mlp-hidden-size 1024 \
  --head-activation gelu \
  --truncation-diagnostics-batch-size 64 \
  --max-truncation-over-limit-fraction 0.10
```

Operator notes:
- `CUDA_VISIBLE_DEVICES=3` was used because devices `0-2` were already busy during this diagnosis window.
- `same-family trust` inside the recipe suite keeps `batch-size=16` on `2048`-token runs because the evaluation path scores candidate pools at full context length; using the larger default eval batch would be unnecessarily risky.
- The dedicated wrapper currently writes correct row artifacts (`artifact_paths.json`, `recipe_rows.jsonl`) even when its final Markdown render step fails; treat `recipe_rows.jsonl` as the source of truth until the wrapper render bug is fully cleaned up.

Main findings from this block:
- `direct_pair_legacy` first becomes fully clean only at `2048`.
- `compact_verdict` first becomes fully clean already at `1536`.
- `legacy@1024` now fails immediately at the truncation gate, before backbone load.
- On repaired compact `R-PRM`, the best smoke recipe was:
  - `mlp + joint @2048`
  - `heldout_pair_acc=0.6694`
  - `heldout_auc=0.6611`
  - `samefamily_top1=0.6829`
- This means repaired `R-PRM` is learnable, but still far from `ACC90`; current interpretation is “usable medium-strength source”, not “anchor-quality source”.

Current milestone status (2026-03-10, Phase E transition):
- Phase A is concluded and stable (full-dataset benchmark contracts are reproducible).
- Phase B core diagnosis is concluded:
  - StrategyQA can gain from PEFT under current pipeline,
  - GSM8K still shows post-PEFT degradation under many settings.
- Phase C infra is complete and runnable end-to-end:
  - C0/C1/C2 value-head path,
  - P(IK) branch (`scripts/phase_c_prepare_pik_data.py`, `scripts/phase_c_train_pik.py`, `scripts/phase_c_eval_pik.py`),
  - CQR quality-first controls and two-stage rollout enrichment.
- Phase D is now the direction-correction and bridge-evidence track:
  - D1 external PRM teacher scoring is implemented and validated,
  - D2 teacher+MC target fusion is implemented in C1,
  - D3 target-source ablation (`mc`, `teacher`, `fused`) is implemented in C2.
- Latest empirical signal (important):
  - Mentor-review methodology correction:
    - promote `MC target + PRM pair gate` as new mainline,
    - treat `q_teacher/q_fused` direct-target runs as ablation.
  - D1 teacher on full StrategyQA C1 train artifact:
    - `num_prefix_scores=7192`, `num_corruption_scores=6111`, `num_errors=0`,
    - `mean_prefix_score=0.9138`, `mean_corrupt_score=0.8881`,
    - clean>corrupt ratio from paired audit: `0.6649`.
  - C2 CQR full control run is currently the strongest internal baseline:
    - `brier=0.1968`, `posthoc_brier=0.1112`,
    - `corr_pair_acc=0.5379`, `corr_auc=0.5141`.
  - D3 teacher path is **mixed, not dead**:
    - in `phase_d_bundle_smoke` and `overnight_d4_conf_fulltrain`, `teacher`
      improves pair metrics over MC with moderate deltas,
    - but gains are still below promotion-level targets.
  - D4 smoke 3-way HQ currently regresses and should not be used as evidence:
    - quality summary confirms HQ pair-consensus path had ineffective teacher
      corruption alignment in that run.
  - D4ABC external-pair chain is currently exploratory-only:
    - A/B/C stages show weak AUC and poor calibration under current config.
  - Several D4 tuning runs failed at D2 eval due teacher-coverage gate
    (`coverage < teacher_min_coverage`), so command-level coverage control is required.
  - D6-T stable DT2 branch has now produced the first strong positive result:
    - `phase_d6t_stable_0306_2211`
    - external held-out pair gate passed with:
      - `mean_pair_acc=0.8154`,
      - `mean_auc=0.7857`,
      - `std_pair_acc=0.0075`,
      - `std_auc=0.0086`.
    - This is the first clean evidence that under high-quality same-question
      triplets, our current value-head training can learn stable ranking.
  - Immediate implication:
    - the old "seed instability blocks the whole D6-T branch" diagnosis is no
      longer true for the stabilized DT2 recipe,
    - the next blockers are now:
      - in-domain transfer back to StrategyQA C1 metrics,
      - mixed-source robustness (`Math-Shepherd + PRM800K`).
  - New bridge result (critical):
    - `DB3_STRATEGYQA_BRIDGE_FULL_RANK_SEED3`
      - `mean_stage2_corr_pair_acc=0.5874`
      - `mean_stage2_corr_auc=0.5461`
      - vs strong C2 reference:
        - `+0.0499 pair_acc`
        - `+0.0303 auc`
    - interpretation:
      - external ranking pretrain is not enough by itself,
      - but `external ranking pretrain -> in-domain ranking continue` is a real
        positive method result.
  - New bridge counterexample (equally important):
    - `DB4_STRATEGYQA_BRIDGE_FULL_JOINT_SEED3`
      - `mean_stage2_corr_pair_acc=0.4957`
      - `mean_stage2_corr_auc=0.4948`
      - while Brier improves relative to DB3/reference.
    - interpretation:
      - scalar calibration objective can conflict with ranking objective,
      - so the main object we need to learn is closer to `process ranking`
        than to `scalar value regression`.
  - Strategic pivot (2026-03-10, authoritative):
    - StrategyQA no longer serves as the primary supervised benchmark for
      value-head / process-ranking validation.
    - Reason:
      - public StrategyQA resources provide decomposition/evidence, but do not
        provide PRM-grade step-quality labels or high-quality process
        preference pairs.
      - continuing to use StrategyQA as the main supervision source risks a
        `garbage in, garbage out` regime.
    - New mainline benchmark policy:
      - training / supervision:
        - `Math-Shepherd`
        - `PRM800K`
      - primary evaluation:
        - `ProcessBench`
        - `PRMBench`
      - optional future non-math logic track:
        - `ProofWriter`
        - `EntailmentBank`
        - `FOLIO`
        - `FoVer`
    - StrategyQA remains important, but only as:
    - bridge continue-training target,
    - downstream transfer benchmark,
    - OOD stress test.
- Phase E is now the active execution track:
  - purpose:
    - validate value-head learnability on high-quality pair/process-supervision
      benchmarks first,
    - then reuse StrategyQA only for transfer/OOD evaluation.
  - implementation status (2026-03-10, E0-E3 delivered):
    - benchmark-native Phase E modules:
      - `src/ours/phase_e/contracts.py`
      - `src/ours/phase_e/pairs.py`
      - `src/ours/phase_e/runtime.py`
      - `src/ours/phase_e/training.py`
      - `src/ours/phase_e/benchmark_eval.py`
    - benchmark-native entrypoints:
      - `scripts/phase_e_prepare_pairs.py`
      - `scripts/phase_e_train_value.py`
      - `scripts/phase_e_eval_benchmark.py`
      - `scripts/run_phase_e_suite.sh`
    - smoke validation already completed:
      - direct `prepare -> train -> ProcessBench eval`
      - one-click suite smoke `E1_MATH_SHEPHERD_PAIR_LEARN_SMOKE`
  - latest Phase E diagnosis:
    - `Math-Shepherd` has real same-source learnability,
    - but seed fragility is still visible (`E2` has one collapsed seed),
    - so the next official task is no longer "run one more baseline",
      but "run the Math-Shepherd trust matrix and then move to same-family
      multi-source math mixture training".
  - runtime reliability hardening (2026-03-11):
    - shared feature cache now tracks actual weight shards referenced by model
      index files, not only metadata files
    - corrupt cache entries now self-heal on rewrite instead of remaining
      sticky forever
    - stale cache lock files are reclaimed after timeout
    - `Phase B` and `Phase C P(IK)` head-training loops now keep cache tensors
      on CPU and move only the current mini-batch to GPU
  - left-padding pooling fix (2026-03-10):
    - `src/ours/phase_b/value_head.py` no longer assumes right padding when
      pooling the last token from frozen backbone hidden states
    - the pooled feature now comes from the last attended token position in
      `attention_mask`, so left-padded and right-padded batches share the same
      semantics
    - `src/ours/phase_b/feature_cache.py` now uses
      `phase_feature_cache_v3` so pre-fix pooled features are treated as stale
      instead of being silently reused
    - verification command:

      ```bash
      PYTHONPATH=src pytest -q tests/unit/test_value_head_pooling.py tests/unit/test_feature_cache.py tests/unit/test_phase_e_benchmark_eval.py
      ```
  - new Phase E multi-source entrypoint:
    - `scripts/run_phase_e_multisource_math_suite.sh`
    - this wrapper now covers:
      - Stage A anchors,
      - Stage B balanced two-source mixtures,
      - Stage C tri-source main mixture,
      - Stage D weak-source PRM800K ablations,
      - Stage E staged curricula.
  - R-PRM data-contract repair (2026-03-10):
    - the old `R-PRM` Phase E path used full chosen/rejected verifier essays
      directly as pair texts
    - this was retained as:
      - `direct_pair_legacy`
    - a new Phase E repair path is now implemented:
      - `compact_verdict`
    - `compact_verdict` rewrites each `R-PRM` DPO row into:
      - compact `Question / Previous Steps / Now Step` prompt
      - short opposite-verdict pair
    - why:
      - the current Phase E learner is a frozen backbone plus tiny scalar head,
        not a generative verifier
      - feeding multi-thousand-token verifier essays into that learner was a
        bad data contract and caused severe truncation
    - real truncation diagnosis on repository artifacts:
      - legacy:
        - `assets/artifacts/phase_e_truncation_diagnostics/rprm_legacy_diag_20260310T122121Z/summary.json`
        - `frac_pairs_over_limit = 1.0`
      - compact:
        - `assets/artifacts/phase_e_truncation_diagnostics/rprm_compact_diag_20260310T122126Z/summary.json`
        - `frac_pairs_over_limit = 0.0`
    - new groups for this repair:
      - `E12_RPRM_COMPACT_VERDICT_SEED3`
      - `E65_STAGEB_MS_RPRM_COMPACT_REPAIRED_LINEAR_SEED3`
      - `E66_STAGEB_MS_RPRM_COMPACT_REPAIRED_MLP_SEED3`
    - new wrapper groups for diagnosis:
      - `R3_RPRM_DATAFIX_SMOKE`
      - `R4_RPRM_DATAFIX_SEED3`
    - same-source compact smoke already ran end-to-end:
      - `assets/artifacts/phase_e_logs/phase_e_rprm_compact_smoke_0310_2038/final_summary.md`
      - tiny seed-42 smoke:
        - `pair_accuracy = 0.666667`
        - `auc = 0.555556`
    - latest dedicated recheck with the repaired contract:
      - `assets/artifacts/phase_e_logs/phase_e_rprm_diag_0310_2019/final_summary.md`
    - important correction:
      - the old statement
        - "`R-PRM` mainly fails because `1024` truncates almost everything"
        is only true for the **legacy verifier-essay contract**
      - under the current `compact_verdict` contract:
        - `1024` is still slightly unsafe,
        - `1536` is the first fully clean cutoff,
        - `2048` is not needed for safety
    - repaired-contract truncation facts:
      - train:
        - `1024`: `over_limit=0.0155`, `collapse=0.0155`
        - `1536`: all truncation-risk metrics `0.0000`
      - validation:
        - `1024`: `over_limit=0.0242`, `collapse=0.0242`
        - `1536`: all truncation-risk metrics `0.0000`
    - repaired-contract same-source fit:
      - `1536`:
        - `pair_acc=0.6129`
        - `auc=0.6031`
      - `2048`:
        - `pair_acc=0.7016`
        - `auc=0.6735`
    - interpretation:
      - removing truncation damage does improve `R-PRM`,
      - but the source still remains far below `ACC90`,
      - so the remaining bottleneck is no longer plain sequence length alone
      - it is now more likely about supervision semantics / head-contract fit
    - config hardening applied after this recheck:
      - `E12_RPRM_COMPACT_VERDICT_SEED3`
      - `E65_STAGEB_MS_RPRM_COMPACT_REPAIRED_LINEAR_SEED3`
      - `E66_STAGEB_MS_RPRM_COMPACT_REPAIRED_MLP_SEED3`
      now default to:
      - `MAX_LENGTH=1536`
  - Phase E source-of-truth:
    - `docs/phase_E_plan.md`
  - Latest strict diagnosis (2026-03-07):
    - `DT4_MIXED_MS_PRM800K_SEED3_STABLE` does pass the external gate, but only
      marginally:
      - `mean_pair_acc=0.6612`
      - `mean_auc=0.6616`
    - source breakdown shows the mixed pass is asymmetric:
      - `math_shepherd`: clearly useful,
      - `prm800k`: near-random under current adapter/training recipe.
    - the old transfer run `phase_d6t_transfer_0306_2236` is **not valid
      evidence** for transfer because its `C1 standalone eval` never actually
      ran (`c1_eval_run_dir=null` for all seeds); that suite has now been fixed.

Primary roadmap:
- `TODO_ours.md`
- `phase_D_plan.md`
- `phase_E_plan.md`

## 2026-03-11 ProcessBench Alignment Audit + Micro Repair Pilots

This round focused on one narrower question:

1. why do strong same-source Phase E heads still transfer weakly to
   `ProcessBench`?
2. can we separate:
   - terminal-completion weakness,
   - local error discrimination weakness,
   - broader prefix-coverage weakness?

### New diagnostic command

```bash
python -u scripts/phase_e_analyze_processbench_failures.py \
  --value-run-dir <PHASE_E_RUN_DIR> \
  --benchmark-eval-dir <PHASE_E_PROCESSBENCH_EVAL_DIR> \
  --run-name <DIAG_NAME>
```

Representative baseline diagnostics:
1. `assets/artifacts/phase_e_processbench_analysis/ms_e68_pb_math_v2_0311_20260310T160909Z/summary.md`
2. `assets/artifacts/phase_e_processbench_analysis/prm_e46_pb_math_v2_0311_20260310T160909Z/summary.md`

Key diagnosis:
1. `ProcessBench` is not only a local first-bad benchmark.
2. It contains a large all-correct block:
   - GSM8K: `0.4825`
   - Math: `0.4060`
3. Both strong baselines had effectively zero terminal-anchor supervision:
   - `E68`: pure `local_first_bad_edge`
   - `E46`: pure `local_modified_process_error_step`

### New repair artifacts

`PRMBench + terminal anchors`:

```bash
python -u scripts/phase_e_prepare_prmbench_terminal_anchor_pairs.py \
  --run-name phase_e_prmbench_terminal_anchor_full_0311 \
  --terminal-anchor-confidence 0.86
```

Artifact:
1. `assets/artifacts/phase_e_pairs/phase_e_prmbench_terminal_anchor_full_0311__192ca71fd301`

`Math-Shepherd + terminal anchors` (capped):

```bash
python -u scripts/phase_e_prepare_mathshepherd_terminal_anchor_pairs.py \
  --run-name phase_e_ms_terminal_anchor_cap20k_diag_0311 \
  --max-local-pairs 20000 \
  --terminal-anchor-ratio 0.50 \
  --terminal-anchor-prefix-mode penultimate \
  --step-label-pair-mode first_bad_edge_strict
```

Artifact:
1. `assets/artifacts/phase_e_pairs/phase_e_ms_terminal_anchor_cap20k_diag_0311__6d57b0d4b490`

`Math-Shepherd grid`:

```bash
python -u scripts/phase_e_prepare_pairs.py \
  --source-bundle math_shepherd \
  --run-name phase_e_ms_grid_cap40k_diag_0311 \
  --split-granularity source_sample \
  --step-label-pair-mode all_good_vs_all_bad \
  --max-pairs-per-sample 4 \
  --max-pairs-total 40000 \
  --max-pairs-per-source 40000 \
  --math-shepherd-path assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl \
  --overwrite
```

Artifact:
1. `assets/artifacts/phase_e_pairs/phase_e_ms_grid_cap40k_diag_0311__4f87d4f4cea6`

### Micro repair pilots

These were intentionally run as server-safe warm-start pilots.

`PRMBench + terminal anchors` from `E46`:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_prmbench_terminal_anchor_full_0311__192ca71fd301/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_prmbench_terminal_anchor_full_0311__192ca71fd301/validation_pairs.jsonl \
  --max-train-samples 512 \
  --max-eval-samples 128 \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_pbta_warm_e46_micro_0311 \
  --objective-mode joint \
  --learning-rate 1e-5 \
  --num-train-epochs 2 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 32 \
  --max-length 1024 \
  --lambda-ranking 1.0 \
  --lambda-bce 1.0 \
  --ranking-margin 0.02 \
  --ranking-target-space score \
  --pair-weight-mode none \
  --checkpoint-selection-metric pair_acc \
  --seed 42 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda \
  --head-architecture mlp \
  --head-dropout-prob 0.05 \
  --head-mlp-hidden-size 1024 \
  --init-value-head-path assets/artifacts/phase_e_runs/phase_e_prmbench_acc90_full_0310_1914_e46_prmbench_acc90_mlp_joint_seed3_e46_prmbench_acc90_mlp_joint_seed3_s43_value_20260310T113737Z/best_value_head.pt
```

ProcessBench results:
1. GSM8K:
   - baseline `E46`: `pair_acc=0.6701`, `auc=0.6264`, `first_edge=0.6706`, `all_correct_last=0.2924`
   - repair pilot: `pair_acc=0.5840`, `auc=0.6014`, `first_edge=0.6471`, `all_correct_last=0.4196`
2. Math:
   - baseline `E46`: `pair_acc=0.5653`, `auc=0.6053`, `first_edge=0.6096`, `all_correct_last=0.2452`
   - repair pilot: `pair_acc=0.5177`, `auc=0.5906`, `first_edge=0.6013`, `all_correct_last=0.3492`

Reading:
1. terminal anchors do raise completion preference,
2. but they soften local ranking when used alone.

`Math-Shepherd + terminal anchors` from `E68`:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_ms_terminal_anchor_cap20k_diag_0311__6d57b0d4b490/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_ms_terminal_anchor_cap20k_diag_0311__6d57b0d4b490/validation_pairs.jsonl \
  --max-train-samples 512 \
  --max-eval-samples 128 \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_msta_warm_e68_micro_0311 \
  --objective-mode joint \
  --learning-rate 1e-5 \
  --num-train-epochs 2 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 32 \
  --max-length 1024 \
  --lambda-ranking 1.0 \
  --lambda-bce 1.0 \
  --ranking-margin 0.02 \
  --ranking-target-space logit \
  --pair-weight-mode none \
  --anti-saturation-weight 0.0005 \
  --anti-saturation-logit-threshold 4.0 \
  --checkpoint-selection-metric pair_acc \
  --seed 42 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda \
  --head-architecture mlp \
  --head-dropout-prob 0.05 \
  --head-mlp-hidden-size 1024 \
  --init-value-head-path assets/artifacts/phase_e_runs/phase_e_ms_acc95_push_0310_2146_e68_ms_acc95_joint_logit_seed42_e68_ms_acc95_joint_logit_seed42_s42_value_20260310T151651Z/best_value_head.pt
```

ProcessBench results:
1. GSM8K:
   - baseline `E68`: `pair_acc=0.6385`, `auc=0.5885`, `first_edge=0.6294`, `all_correct_last=0.5626`
   - repair pilot: `pair_acc=0.5396`, `auc=0.5527`, `first_edge=0.6059`, `all_correct_last=0.7590`
2. Math:
   - baseline `E68`: `pair_acc=0.5809`, `auc=0.5547`, `first_edge=0.5553`, `all_correct_last=0.5895`
   - repair pilot: `pair_acc=0.5252`, `auc=0.5350`, `first_edge=0.5324`, `all_correct_last=0.7663`

Reading:
1. pure terminal-anchor pressure strongly over-corrects toward completion preference.

`Math-Shepherd grid` from `E68`:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_ms_grid_cap40k_diag_0311__4f87d4f4cea6/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_ms_grid_cap40k_diag_0311__4f87d4f4cea6/validation_pairs.jsonl \
  --max-train-samples 512 \
  --max-eval-samples 128 \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_msgrid_warm_e68_micro_0311 \
  --objective-mode joint \
  --learning-rate 1e-5 \
  --num-train-epochs 2 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 32 \
  --max-length 1024 \
  --lambda-ranking 1.0 \
  --lambda-bce 1.0 \
  --ranking-margin 0.02 \
  --ranking-target-space logit \
  --pair-weight-mode none \
  --anti-saturation-weight 0.0005 \
  --anti-saturation-logit-threshold 4.0 \
  --checkpoint-selection-metric pair_acc \
  --seed 42 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda \
  --head-architecture mlp \
  --head-dropout-prob 0.05 \
  --head-mlp-hidden-size 1024 \
  --init-value-head-path assets/artifacts/phase_e_runs/phase_e_ms_acc95_push_0310_2146_e68_ms_acc95_joint_logit_seed42_e68_ms_acc95_joint_logit_seed42_s42_value_20260310T151651Z/best_value_head.pt
```

ProcessBench results:
1. GSM8K:
   - baseline `E68`: `pair_acc=0.6385`, `auc=0.5885`, `first_edge=0.6294`, `all_correct_last=0.5626`
   - repair pilot: `pair_acc=0.6436`, `auc=0.5891`, `first_edge=0.6235`, `all_correct_last=0.5768`
2. Math:
   - baseline `E68`: `pair_acc=0.5809`, `auc=0.5547`, `first_edge=0.5553`, `all_correct_last=0.5895`
   - repair pilot: `pair_acc=0.5839`, `auc=0.5559`, `first_edge=0.5595`, `all_correct_last=0.6011`

Reading:
1. grid supervision helps the broader good-vs-bad side,
2. but it does not solve the terminal-completion gap.

### Current synthesis

The current ProcessBench transfer problem is now much better localized:
1. terminal anchors and grid-style pairs move different parts of the benchmark,
2. neither alone is sufficient,
3. the next non-generic repair should be:
   - local pairs
   - plus limited terminal anchors
   - plus optionally low-weight grid coverage
   in one mixed or staged curriculum.

## 2026-03-11 MCTS Literature Check For The Current Phase E Bottleneck

Question:
1. can `MCTS` directly solve the current `ProcessBench` transfer problem?

Short answer:
1. not as the next mainline fix,
2. but it is still relevant as a later branch for:
   - offline pair/tree construction,
   - or test-time judge/search.

Why this is the correct reading:
1. our current measured bottleneck is a supervision mismatch:
   - local-only supervision under-teaches terminal completion,
   - terminal-anchor supervision over-corrects,
   - grid supervision helps a different benchmark slice.
2. `MCTS` does not remove that mismatch by itself.
3. if the verifier target is already misaligned, search usually amplifies that
   target rather than correcting it.

What the literature supports:
1. `ReST-MCTS*`
   - supports tree search as an offline data-harvesting / self-training method:
     - https://arxiv.org/abs/2406.03816
     - https://github.com/THUDM/ReST-MCTS
2. `Tree-PLV`
   - supports tree-based preference construction for better step-level ranking:
     - https://arxiv.org/abs/2407.00390
     - https://aclanthology.org/2024.emnlp-main.125/
3. `Rewarding Progress`
   - supports search when the target is progress/advantage aware:
     - https://arxiv.org/abs/2410.08146
4. `MCTS-Judge`
   - supports MCTS as test-time judge scaling:
     - https://arxiv.org/abs/2502.12468

What the literature warns against:
1. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   suggests that naive MC-style synthetic PRM labels often generalize poorly:
   - https://arxiv.org/abs/2501.07301
2. therefore a naive "replace the current pipeline with tree search rollouts"
   move would likely create a new noisy-label problem.

Repository-specific conclusion:
1. `MCTS` should not replace the current `local + terminal + optional grid`
   repair path.
2. If later introduced, the defensible forms are:
   - tree-harvested higher-margin local / terminal pairs,
   - or a separate test-time judge/search baseline.
3. So the next mainline work remains supervision-alignment repair, not search
   escalation.

## 2026-03-10 R-PRM Repair Commands

The commands below are the new source-of-truth reruns for the repaired
`R-PRM` Phase E path.

Same-source compact R-PRM seed-3:

```bash
CUDA_VISIBLE_DEVICES=3 \
ACTIVE_PHASE_E_GROUP=E12_RPRM_COMPACT_VERDICT_SEED3 \
RUN_PREFIX=phase_e_rprm_compact_seed3_$(date +%m%d_%H%M) \
bash scripts/run_phase_e_suite.sh
```

R-PRM data-fix smoke comparison:

```bash
CUDA_VISIBLE_DEVICES=3 \
ACTIVE_PHASE_E_REPAIR_GROUP=R3_RPRM_DATAFIX_SMOKE \
RUN_PREFIX=phase_e_rprm_datafix_smoke_$(date +%m%d_%H%M) \
bash scripts/run_phase_e_repair_diagnostics_suite.sh
```

This official smoke now runs only executable fixed groups by default.  It no
longer includes legacy long-analysis `R-PRM` groups that fail the truncation
gate by design.

R-PRM data-fix official seed-3 matrix:

```bash
CUDA_VISIBLE_DEVICES=3 \
ACTIVE_PHASE_E_REPAIR_GROUP=R4_RPRM_DATAFIX_SEED3 \
RUN_PREFIX=phase_e_rprm_datafix_seed3_$(date +%m%d_%H%M) \
bash scripts/run_phase_e_repair_diagnostics_suite.sh
```

This official seed-3 matrix also excludes the legacy long-analysis `R-PRM`
groups.  Use historical artifacts or explicit overrides only when you
intentionally want to rerun the known-bad contract.

Direct truncation comparison for legacy vs compact `R-PRM` contracts:

```bash
python scripts/phase_e_prepare_pairs.py \
  --source-bundle r_prm_train \
  --run-name rprm_compact_diag \
  --max-pairs-total 128 \
  --max-pairs-per-source 128 \
  --min-pair-confidence 0.75 \
  --r-prm-pair-mode compact_verdict
```

Then run:

```bash
python scripts/phase_e_diagnose_truncation.py \
  --pairs-jsonl <TRAIN_PAIRS_JSONL> \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --max-samples 64 \
  --max-lengths 1024
```

## 2026-03-10 Strategic Pivot (Authoritative)

This is now the repository's active methodology baseline.

Why the pivot happened:
1. `DT2 stable` proved high-quality external ranking is learnable.
2. `transfer_fix` proved direct transfer back to StrategyQA stays near random.
3. `DB3` then proved bridge-style ranking-only continue training can help.
4. `DB4` proved adding scalar/joint calibration pressure can destroy that gain.
5. Separate dataset research found no public StrategyQA resource with PRM-grade
   step-quality supervision.

What this means:
1. StrategyQA is still valuable, but no longer as the primary supervised
   benchmark for value-head training.
2. The scientific question should first be answered on datasets whose process
   supervision is genuinely strong.
3. After the method is validated there, StrategyQA becomes the transfer test
   that tells us how much of the learned ranking prior can survive domain shift.
4. The first Phase E question is now:
   - "can we learn stable ranking/value behavior on the same benchmark family?"
   not:
   - "can one source dataset immediately generalize across benchmarks?"

Operational decision:
1. Freeze existing StrategyQA value-head results as:
   - control baselines,
   - transfer evidence,
   - objective-mismatch evidence.
2. Move main validation to:
   - training / supervision:
     - `Math-Shepherd`
     - `PRM800K`
   - evaluation:
     - `ProcessBench`
     - `PRMBench`
3. Keep a second-track plan for non-math logical reasoning:
   - `ProofWriter`
   - `EntailmentBank`
   - `FOLIO`
   - `FoVer`

External references that motivated the benchmark change:
1. StrategyQA dataset card:
   - https://huggingface.co/datasets/voidful/StrategyQA
2. StrategyQA official repository:
   - https://github.com/eladsegal/strategyqa
3. PRM800K:
   - https://github.com/openai/prm800k
4. Math-Shepherd:
   - https://huggingface.co/datasets/peiyi9979/Math-Shepherd
5. ProcessBench:
   - https://huggingface.co/datasets/Qwen/ProcessBench
6. PRMBench:
   - https://prmbench.github.io/
7. ProcessBench paper:
   - https://arxiv.org/abs/2412.06559
8. VersaPRM:
   - https://arxiv.org/abs/2502.06737
9. ThinkPRM:
   - https://arxiv.org/abs/2504.16828
10. ThinkPRM official repo:
   - https://github.com/mukhal/thinkprm
11. R-PRM:
   - https://arxiv.org/abs/2503.21295
12. R-PRM official repo:
   - https://github.com/NJUNLP/R-PRM
13. LMs Mostly Know What They Know:
   - https://arxiv.org/abs/2207.05221

## 2026-03-10 Phase E Re-scope (Same-Benchmark First)

Why this re-scope is necessary:
1. Community evidence does not support the assumption that a PRM/value head
   trained on a single source dataset should automatically generalize across
   benchmarks.
2. `ProcessBench` and `PRMBench` were created because current PRMs remain weak
   on explicit process error detection.
3. Recent follow-up work (`VersaPRM`, `ThinkPRM`, `R-PRM`) points toward:
   - multi-domain training,
   - stronger preference/process constructions,
   - or even generative verifier designs
   when cross-domain generalization is the goal.

What our own newest evidence says:
1. `Math-Shepherd smoke`:
   - source held-out pair metrics are positive:
     - `pair_acc=0.6787`
     - `auc=0.6704`
   - but `ProcessBench GSM8K` stays weak:
     - `pair_acc=0.4661`
     - `auc=0.5155`
2. `PRM800K seed-3`:
   - source held-out mean is weak:
     - `mean_pair_acc=0.4783`
     - `mean_auc=0.4855`
   - `ProcessBench GSM8K` mean is also weak:
     - `mean_pair_acc=0.4737`
     - `mean_auc=0.4943`

Operational consequence:
1. Phase E no longer treats cross-dataset transfer as the first pass/fail gate.
2. New gate order is:
   - source-family held-out learnability first,
   - benchmark-native eval second,
   - StrategyQA transfer third.
3. If later we want stronger cross-benchmark generalization, that becomes a
   method-upgrade task, not the default expectation for the current recipe.

## 2026-03-10 E15 Update (Why Phase E Moves To Multi-Source Math)

Newest result:
1. `E15_MATH_SHEPHERD_TRUST_ROBUST_SEED3`
   - same-source held-out is now clearly positive:
     - `mean_heldout_pair_acc=0.7518`
     - `mean_heldout_auc=0.7280`
   - but benchmark-native metrics remain weak:
     - `ProcessBench GSM8K mean_auc=0.4834`
     - `ProcessBench Math mean_auc=0.4746`

Interpretation:
1. Single-source `Math-Shepherd` is enough to prove learnability.
2. It is not enough to produce a benchmark-trustworthy value head.
3. Therefore the next official Phase E direction is:
   - same-family multi-source math mixture training,
   - not more isolated one-source transfer expectations.

Current planned sources:
1. `Math-Shepherd`
2. `R-PRM`
3. `PRMBench_Preview`
4. `PRM800K` only as a weak-source ablation / low-weight auxiliary source

Detailed plan:
1. `docs/phase_E_multisource_math_plan.md`

## 2026-03-10 Late Phase E Update (Post-Fix Anchor / Mixture Snapshot)

This is the first post-fix snapshot after the `first_bad_edge_strict` repair
and the first multi-source Stage A/B runs.

What is now newly established:
1. Post-fix `Math-Shepherd` trust selection is stronger than the older
   `E15`-only view suggested.
   - `MS2_MATH_SHEPHERD_TRUST_SEED3_MATRIX` now provisionally selects:
     - `E14_MATH_SHEPHERD_TRUST_ANTISAT_SEED3`
   - with:
     - `mean_hold_pair=0.8246`
     - `mean_hold_auc=0.7899`
     - `pb_gsm_auc=0.4893`
     - `pb_math_auc=0.4783`
   - interpretation:
     - the repair did improve same-family learnability,
     - but it still did **not** solve `ProcessBench`.
2. `E20_STAGEA_MS_ANCHOR_SEED3` now gives the cleanest post-fix
   single-source `Math-Shepherd` anchor summary:
   - `mean_heldout_pair_acc=0.6853`
   - `mean_heldout_auc=0.6676`
   - `mean_prmbench_preview_auc=0.5868`
   - `mean_processbench_gsm8k_auc=0.4750`
   - `mean_processbench_math_auc=0.4715`
   - interpretation:
     - `Math-Shepherd` can learn a real source-family ranking signal,
     - and transfers somewhat to `PRMBench_Preview`,
     - but remains weak on `ProcessBench`.
3. `E21_STAGEA_RPRM_ANCHOR_SEED3` provides a different kind of signal:
   - `mean_heldout_pair_acc=0.4381`
   - `mean_heldout_auc=0.4953`
   - `mean_prmbench_preview_auc=0.5623`
   - `mean_processbench_gsm8k_auc=0.4744`
   - `mean_processbench_math_auc=0.4626`
   - one concrete run (`seed 44`) reaches:
     - `PRMBench_Preview pair_acc=0.6001`
     - `PRMBench_Preview auc=0.5937`
   - interpretation:
     - `R-PRM` is not a strong general anchor under our current recipe,
     - but it is more aligned with `PRMBench_Preview` than with
       `ProcessBench`.
4. `E22_STAGEA_PRMBENCH_PREVIEW_ANCHOR_SEED3` is **not yet complete**.
   - pair preparation and training started,
   - but no final group summary is available yet,
   - so no scientific conclusion should be written for `E22` at this point.
5. `E24_STAGEB_MS_RPRM_MIX_SEED3` has not shown a positive mixture result yet.
   - current smoke run failed before finishing the full group summary,
   - partial seed-42 benchmark metrics are weak:
     - `PRMBench_Preview auc=0.5207`
     - `ProcessBench GSM8K auc=0.4682`
     - `ProcessBench Math auc=0.3835`
   - interpretation:
     - there is no evidence yet that simple `MS + R-PRM` mixing is
       automatically complementary.
6. `CUR1_STAGEE_MS_TO_MSRPRM` is still incomplete as a curriculum claim.
   - stage-1 warm-start (`E20`) completed,
   - stage-2 (`E24`) has only partial progress so far,
   - therefore `CUR1` currently provides no final curriculum conclusion.

Project-level reading:
1. The post-fix `Math-Shepherd` repair was necessary and scientifically
   meaningful.
2. The repository now has two different usable source signals:
   - `Math-Shepherd`: stronger same-family anchor
   - `R-PRM`: better `PRMBench_Preview` alignment
3. The repository does **not** yet have evidence that:
   - balanced mixture solves benchmark robustness,
   - or that `ProcessBench` has become reliably positive.
4. The next method question is therefore:
   - not "did the fix solve Phase E?"
   - but:
   - "which source combination and training order best preserves the
     `Math-Shepherd` anchor while adding `PRMBench`-style robustness?"

## 2026-03-10 Literature Boundary Update (Small-Scale Value Head)

This is the current literature-backed reading of the Phase E question:
1. the community does **not** strongly support:
   - small-scale
   - noisy-label
   - scalar-regression
   - benchmark-robust process value heads
2. the community **does** support narrower success cases:
   - large-scale dense process supervision (`PRM800K`, `Math-Shepherd`,
     `OmegaPRM`, `Rewarding Progress`)
   - or smaller but stronger ranking / verifier formulations
     (`ThinkPRM`, `R-PRM`)
   - or coarse uncertainty/value heads at the question level
     (`LMs Mostly Know What They Know`,
     `LLMs Must Be Taught to Know What They Don't Know`)

What this means for our repository:
1. the old hope:
   - "small scratch scalar value head should become a robust benchmark-facing
     PRM"
   is not literature-backed.
2. the narrower hope:
   - "small-to-medium scale same-family ranking verifier / first-bad-edge head
     can learn useful local process discrimination"
   is literature-backed enough to remain worth pursuing.
3. our newest results match that boundary:
   - `MS2` / `E20` show local learnability,
   - `E21` shows preview-aligned preference signal,
   - `E24` still does not show that easy mixture fixes benchmark transfer.

Immediate framing correction:
1. We should describe current Phase E work as:
   - ranking-first same-family verifier learning
2. We should **not** describe it as:
   - already having evidence for a general-purpose small PRM/value model.

## 2026-03-10 Intradataset ACC90 Smoke Diagnostic

First smoke:
1. `assets/artifacts/phase_e_logs/phase_e_top3_acc90_0310_1808/final_summary.md`
   - `E41_MS_ACC90_MLP_RANK_SEED3`
     - `heldout_pair_acc=0.9610`
     - `heldout_auc=0.8908`
   - `E45_PRMBENCH_ACC90_MLP_RANK_SEED3`
     - `heldout_pair_acc=0.9483`
     - `heldout_auc=0.8333`
   - `E48_RPRM_ACC90_MLP_RANK_SEED3`
     - `heldout_pair_acc=0.4379`
     - `heldout_auc=0.4666`

Interpretation:
1. The current same-source ACC90 branch is strongly positive for:
   - `Math-Shepherd`
   - `PRMBench_Preview`
2. It is clearly negative for:
   - `R-PRM` under the current adapter/objective recipe.
3. Therefore same-source fit quality is now clearly source-dependent, not just
   a question of head capacity.

Critical caveat:
1. The smoke intentionally used only `seed=42`.
2. So these numbers are direction checks, not stability evidence.

Important bug discovered during interpretation:
1. the smoke candidate selector currently mis-selects the final group because
   the suite passes repeated `--suite-log-dirs` flags while the selector parser
   consumes only the last repeated occurrence under its current contract.
2. result:
   - the candidate report from this smoke should be treated as invalid,
   - the top-level summary remains usable.

## 2026-03-11 Cache/OOM Reliability Repair

Why this matters:
1. A code audit found a real trust risk in the shared feature-cache layer:
   - cache provenance tracked config/index files,
   - but not the actual weight shards those files referenced.
2. This created a silent failure mode:
   - swap checkpoint weights in-place,
   - keep metadata files unchanged,
   - old feature cache may still look reusable.
3. A second risk was also confirmed in training:
   - `Phase B` and `Phase C P(IK)` could move whole cached feature bundles to
     GPU on cache hit,
   - which made cache-heavy runs more likely to OOM during head-only training.

What is now fixed:
1. `src/ours/phase_b/feature_cache.py`
   - schema bumped to `phase_feature_cache_v2`
   - model directory signatures now include:
     - tracked metadata files
     - index-referenced shard files
     - loose top-level weight files
2. cache self-healing:
   - corrupt payload/meta pairs are purged and rewritten
   - stale `.write.lock` files are reclaimed after timeout
3. `scripts/phase_b_train_value.py`
   - cache payloads remain on CPU
   - training/eval move only current mini-batches to the head device
   - frozen backbone is released after encoding
4. `scripts/phase_c_train_pik.py`
   - same CPU-cache refactor applied
5. `src/ours/phase_e/runtime.py`
   - cached feature payload validation is stricter than before

Operational implication:
1. New cache-sensitive results should be interpreted under the `v2` cache
   semantics.
2. If a run looks suspiciously too fast after checkpoint swaps, treat old cache
   reuse as suspect and rerun with:
   - `FEATURE_CACHE_MODE=off`
   - or a fresh cache root
3. Full recursive OOM backoff still primarily covers:
   - generation
   - rollout generation
   - feature encoding
   - feature scoring
4. Training loops are now safer because they no longer keep whole feature banks
   on GPU, but they are not yet a universal backward-pass OOM splitter.

## 2026-03-10 Data Semantics Risk Update

This risk was real, and the code has now been tightened.

Legacy issue:
1. Old `Math-Shepherd` / `RLHFlow` / `PRM800K` fallback runs used one
   nearest-negative converter over a single labeled trajectory.
2. That produced different-depth prefix pairs and mixed:
   - progress/depth signal,
   - local error signal.
3. Those old results should now be treated as `legacy` and should not be mixed
   into post-fix summaries.

Current fix:
1. Default `step_label_pair_mode` is now `first_bad_edge_strict`.
2. Single-trajectory `+/-` sources now only emit:
   - the last clean prefix before the first negative step,
   - versus the prefix that includes the first bad step.
3. If no positive step exists before the first negative step, the sample is
   dropped instead of being forced into a noisy pair.
4. New artifacts explicitly expose:
   - `pair_build_mode`
   - `pair_semantics`
5. New artifact stages:
   - `phase_e_pairs_v2`
   - `phase_d_external_pairs_v2`

What this means scientifically:
1. `Math-Shepherd` and `RLHFlow` are still not true same-step sibling-pair
   datasets.
2. They are now interpreted more narrowly and more honestly as:
   - strict `first_bad_edge` supervision sources.
3. `PRMBench_Preview` and `R-PRM` remain lower-risk sources.
4. `PRM800K` stays mixed:
   - official `completion_ratings` path is cleaner,
   - fallback `+/-` path now also uses the stricter first-bad-edge builder.

Operational consequence:
1. New `Math-Shepherd` wins may now be narrated as:
   - local first-bad-edge learnability inside the source family.
2. They still should **not** be narrated as:
   - exact same-step branch preference learning.
3. Detailed audit record:
   - `docs/data_semantics_risk_audit_20260310.md`

## 2026-03-10 Intradataset ACC90 Structural Diagnosis

The newest same-source ACC90 runs answer a narrower but important question:

> Are current weak results mainly caused by an over-simple head, by
> under-training, or by something else?

What the repository now supports:
1. `Math-Shepherd`
   - linear robust recipe `E40` already reaches:
     - `mean_heldout_pair_acc = 0.9172`
     - `mean_heldout_auc = 0.8623`
   - therefore the old weak `Math-Shepherd` results cannot be blamed only on
     "the head is too simple".
2. `Math-Shepherd` MLP recipes still help:
   - `E41`
     - `mean_heldout_pair_acc = 0.9863`
     - `mean_heldout_auc = 0.9056`
   - `E42`
     - `mean_heldout_pair_acc = 0.9641`
     - `mean_heldout_auc = 0.9408`
   - `E43`
     - `mean_heldout_pair_acc = 0.9619`
     - `mean_heldout_auc = 0.9425`
   - so higher capacity is a real lever, but here it is an upgrade over a
     working baseline rather than a rescue from total underfitting.
3. `PRMBench_Preview`
   - linear recipe `E44` remains clearly weak:
     - `mean_heldout_pair_acc = 0.7380`
     - `mean_heldout_auc = 0.6782`
   - MLP recipes cross the intended target:
     - `E45`
       - `mean_heldout_pair_acc = 0.9315`
       - `mean_heldout_auc = 0.8711`
     - `E46`
       - `mean_heldout_pair_acc = 0.9309`
       - `mean_heldout_auc = 0.9057`
   - therefore `PRMBench_Preview` is now the clearest evidence that the
     linear head can be too simple on some sources.
4. Weak runs are not explained by under-training alone:
   - `E12_MS_TRUST_LOWLR_SEED3`
     - `mean_heldout_pair_acc = 0.5853`
     - `mean_heldout_auc = 0.5856`
   - so "smaller LR + longer training" is not a universal explanation.

Updated interpretation:
1. head capacity is a source-specific issue:
   - `Math-Shepherd`: linear already works, MLP improves further
   - `PRMBench_Preview`: MLP is required
2. current failures cannot be summarized by one global slogan like:
   - "the head is too simple"
   - or:
   - "the runs were just under-trained"
3. same-source `>90%` held-out accuracy still does **not** prove:
   - benchmark transfer,
   - cross-task trustworthiness,
   - or RL-readiness
4. it proves a narrower claim:
   - the frozen-feature value-head stack is learnable on that dataset under
     the current supervision semantics.

## 2026-03-10 RL Trustworthiness Threshold Reading

Question:

> If cross-dataset transfer is ignored for now, how good does a value utility
> need to be before it is trustworthy enough for RL-style reasoning
> faithfulness experiments?

The literature does **not** support one universal threshold like:
1. `pair_acc > 0.90 => RL-ready`
2. or `auc > 0.90 => safe to optimize`

The stronger consensus is:
1. high same-source held-out discrimination is necessary;
2. it is not sufficient;
3. usefulness under optimization pressure must also be demonstrated.

Why:
1. `ProcessBench` shows that many PRMs which look acceptable on easier data
   still fail explicit process-error identification.
2. `PRMBench` shows that process verification remains multi-dimensional and is
   not captured by one comfortable pair metric.
3. `The Lessons of Developing PRMs` argues that reward-model reliability
   depends heavily on label quality, evaluation design, and optimization
   behavior.
4. `PRM800K` and later PRM work justify process supervision, but they do not
   justify trusting a value head from one held-out metric alone.

Practical rule for this repository:
1. same-source held-out discrimination must be strong:
   - `mean_pair_acc >= 0.90`
   - `mean_auc >= 0.90`
   - low seed variance
2. the head must remain stable:
   - worst seed cannot collapse,
   - margins should not vanish,
   - small recipe changes should not flip the direction completely
3. same-family policy-level benefit must be shown:
   - reranking / rejection / conservative search should improve answer/process
     quality on the same dataset family
4. same-family local-faithfulness checks should also be positive
5. even then, the correct narration is still:
   - bounded-support utility,
   - not universal verifier.

What we already have:
1. `Math-Shepherd`
   - same-source held-out metrics are strong under the best current recipes
2. `PRMBench_Preview`
   - same-source held-out metrics are also strong under `MLP`

What is still missing:
1. same-family policy-improvement evidence
2. stronger same-family local-faithfulness gates
3. optimization-pressure evidence

Current scientific reading:
1. same-source learnability is supported;
2. RL-level trustworthiness is **not yet** established, even after transfer is
   ignored.

Evidence:
1. `assets/artifacts/phase_e_logs/phase_e_all_acc90_0310_1808_e40_ms_acc90_linear_robust_seed3/final_summary.md`
2. `assets/artifacts/phase_e_logs/phase_e_all_acc90_0310_1808_e41_ms_acc90_mlp_rank_seed3/final_summary.md`
3. `assets/artifacts/phase_e_logs/phase_e_all_acc90_0310_1652_e42_ms_acc90_mlp_joint_seed3/final_summary.md`
4. `assets/artifacts/phase_e_logs/phase_e_all_acc90_0310_1808_e43_ms_acc90_mlp_highconf_seed3/final_summary.md`
5. `assets/artifacts/phase_e_logs/phase_e_all_acc90_0310_1652_e44_prmbench_acc90_linear_seed3/final_summary.md`
6. `assets/artifacts/phase_e_logs/phase_e_all_acc90_0310_1652_e45_prmbench_acc90_mlp_rank_seed3/final_summary.md`
7. `assets/artifacts/phase_e_logs/phase_e_all_acc90_0310_1652_e46_prmbench_acc90_mlp_joint_seed3/final_summary.md`
8. `assets/artifacts/phase_e_logs/phase_e_ms_trust_seed3_fix_0310_1659_e12_math_shepherd_trust_lowlr_seed3/final_summary.md`

## 2026-03-10 I4 Full-Matrix Result + Phase E Aggregation Hardening

The first fully completed intradataset full-matrix run is now:
1. `assets/artifacts/phase_e_logs/phase_e_rprm_acc90_full_0310_1914/final_summary.md`
2. group:
   - `I4_RPRM_ACC90_MATRIX`

What it says:
1. `R-PRM` remains weak under the current `ACC90` branch even after expanding
   from:
   - linear
   - to MLP ranking
   - to MLP joint
2. actual group means:
   - `E47_RPRM_ACC90_LINEAR_SEED3`
     - `mean_pair_acc = 0.4374`
     - `mean_auc = 0.5016`
   - `E48_RPRM_ACC90_MLP_RANK_SEED3`
     - `mean_pair_acc = 0.5197`
     - `mean_auc = 0.5123`
   - `E49_RPRM_ACC90_MLP_JOINT_SEED3`
     - `mean_pair_acc = 0.6002`
     - `mean_auc = 0.5885`
3. therefore:
   - `E49 > E48 > E47`
   - so objective/head changes help,
   - but they do not rescue the source under the current repository contract
4. seed std stays small:
   - `E49 std_pair_acc = 0.0098`
   - `E49 std_auc = 0.0106`
5. so this is a systematic weakness, not a seed-collapse story.

Engineering hardening completed in the same pass:
1. `scripts/phase_e_select_candidate.py`
   - now supports repeated `--suite-log-dirs` occurrences, matching the
     already-fixed intradataset selector
2. these wrapper scripts no longer regex-parse `final_summary.md` to build
   top-level comparisons:
   - `scripts/run_phase_e_intradataset_suite.sh`
   - `scripts/run_phase_e_single_source_suite.sh`
   - `scripts/run_phase_e_multisource_math_suite.sh`
3. all now aggregate directly from:
   - `seed_results.jsonl`

Why this matters:
1. `final_summary.md` is a human-readable rendering,
2. `seed_results.jsonl` is the real structured source-of-truth artifact,
3. so Phase E top-level summaries are now less sensitive to Markdown format
   drift.

## 2026-03-07 D6-T Strict Diagnosis

This is the current method-level reading of the latest two Phase D6-T results.

Confirmed runs:
1. `assets/artifacts/phase_d6t_logs/phase_d6t_stable_0306_2211/final_summary.md`
2. `assets/artifacts/phase_d6t_logs/phase_d6t_mixed_stable_0306_2242/final_summary.md`
3. `assets/artifacts/phase_d_triplet_eval/phase_d6t_mixed_stable_0306_2242_dt4_mixed_ms_prm800k_seed3_stable_s42_ext_eval_20260306T150134Z/metrics.json`

What is now solid:
1. `DT2_MATH_SHEPHERD_SEED3_STABLE` is a real success:
   - external held-out ranking is high and seed-stable,
   - so the ranking-first branch is learnable under high-quality same-question pairs.
2. `DT4_MIXED_MS_PRM800K_SEED3_STABLE` is only a partial success:
   - overall gate barely passes,
   - but source breakdown shows `PRM800K` is almost random while `Math-Shepherd`
     still carries the result.
3. Therefore the mixed result should not be narrated as:
   - "multi-source external supervision works",
   - but as:
   - "the current mixed recipe still survives because Math-Shepherd dominates the usable signal".

What is not yet solved:
1. In-domain transfer is still unproven.
2. The previous transfer suite was implementation-invalid:
   - `phase_d6t_transfer_0306_2236`
   - `c1_eval_run_dir=null` for all seeds,
   - root cause: the `DT2_MATH_SHEPHERD_SEED3_STABLE_C1_TRANSFER` group did not
     force-enable `RUN_C1_STANDALONE_EVAL`, so stale environment state could
     silently disable the extra eval stage.
3. The D6 promotion gate report remains blocked by an older failed D6 gated run:
   - `D6_STRATEGYQA_SMOKE_RANKING_PRM_GATE`
   - failure reason:
     - zero optimizer steps after overly strict pair filtering,
     - not a parser bug in the gate report itself.

Method-level conclusion:
1. There is no longer evidence of a core implementation bug in the stable DT2
   learning loop.
2. The remaining risk is methodological:
   - source mismatch,
   - external pair quality mismatch,
   - and lack of transfer from math-process triplets to StrategyQA corruption ranking.
3. Two new strict results sharpened this diagnosis:
   - `DT2_MATH_SHEPHERD_SEED3_STABLE_C1_TRANSFER`
     - external held-out remains strong,
     - but mean in-domain C1 transfer stays near random,
     - therefore "external ranking learnability" does not automatically become
       "target-task corruption ranking utility".
   - `DT3_PRM800K_SEED3_STABLE`
     - stable but weak,
     - therefore `PRM800K` under the current adapter should be treated as a
       control/ablation source, not the main external signal.
4. Immediate project move:
   - stop spending the next cycle on small LR/epoch tuning,
   - move to a bridge experiment:
     - stage-1 external ranking pretrain on Math-Shepherd,
     - stage-2 warm-start on StrategyQA in-domain CQR/C1 pairs,
     - evaluate whether in-domain `corr_pair_acc` / `corr_auc` improves.

## 2026-03-07 Bridge Suite Added

Bridge rationale:
1. `DT2 stable` proved the ranking branch can learn from high-quality external
   triplets.
2. `transfer_fix` proved that direct zero-bridge transfer back to StrategyQA is
   not enough.
3. Therefore the right next method step is not more direct-transfer tuning, but
   a two-stage bridge.

What is implemented:
1. `scripts/phase_b_train_value.py`
   - added `--init-value-head-path`,
   - allows stage-2 to warm-start from a stage-1 `best_value_head.pt`,
   - validates hidden-size/pooling compatibility before loading.
2. `scripts/run_phase_d_bridge_suite.sh`
   - stage-1:
     - build/reuse shared Math-Shepherd pairs,
     - run stable external ranking pretrain,
     - evaluate on held-out external validation pairs.
   - stage-2:
     - continue training on StrategyQA in-domain CQR/C1 artifacts,
     - re-use the stage-1 value-head weights via `--init-value-head-path`,
     - summarize final in-domain corruption metrics and delta vs reference C2 baseline.
3. `scripts/phase_b_train_value.py`
   - fixed a real eval bug:
     - when `--max-eval-samples` truncates eval prefixes, corruption rows are
       now aligned to the surviving clean prefix IDs,
     - avoids `KeyError(clean_prefix_id)` during smoke and bridge runs.

Smoke validation:
1. `phase_d_bridge_devsmoke2`
   - successfully completed:
     - pair preparation,
     - stage-1 ranking pretrain,
     - stage-1 external held-out eval,
     - stage-2 warm-start (`init_value_head_ok` printed),
     - final bridge summary.
2. This was a tiny run for wiring validation only, not for reporting quality.

Directly runnable bridge groups:
```bash
# DB1: smoke, ranking-only bridge
ACTIVE_PHASE_DBR_GROUP=DB1_STRATEGYQA_BRIDGE_SMOKE_RANK \
RUN_PREFIX=phase_d_bridge_smoke_rank_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_d_bridge_suite.sh

# DB2: smoke, joint bridge
ACTIVE_PHASE_DBR_GROUP=DB2_STRATEGYQA_BRIDGE_SMOKE_JOINT \
RUN_PREFIX=phase_d_bridge_smoke_joint_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_d_bridge_suite.sh

# DB3: full seed-3, ranking-only bridge
ACTIVE_PHASE_DBR_GROUP=DB3_STRATEGYQA_BRIDGE_FULL_RANK_SEED3 \
RUN_PREFIX=phase_d_bridge_full_rank_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_d_bridge_suite.sh

# DB4: full seed-3, joint bridge
ACTIVE_PHASE_DBR_GROUP=DB4_STRATEGYQA_BRIDGE_FULL_JOINT_SEED3 \
RUN_PREFIX=phase_d_bridge_full_joint_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_d_bridge_suite.sh
```

Interpretation update after full runs:
1. `DB3` is now frozen as the strongest positive StrategyQA transfer result in
   this repository.
2. `DB4` is frozen as the strongest evidence that ranking and scalar
   calibration are not interchangeable objectives in our setting.
3. These runs should now be cited as methodological evidence, not as proof that
   StrategyQA itself is a sufficient supervised benchmark.

## 2026-03-06 Stable DT2 Breakthrough

This is now the most important new result in the repository.

Confirmed run:
1. `assets/artifacts/phase_d6t_logs/phase_d6t_stable_0306_2211/suite.log`
2. `assets/artifacts/phase_d6t_logs/phase_d6t_stable_0306_2211/seed_results.jsonl`
3. `assets/artifacts/phase_d_external_pairs/phase_d6t_stable_0306_2211_dt2_math_shepherd_seed3_stable_sharedsplit_s42_pairs__b5c1a9ab3d12/summary.json`

Per-seed external held-out ranking:
1. seed42:
   - `pair_acc=0.8050`, `auc=0.7772`
2. seed43:
   - `pair_acc=0.8222`, `auc=0.7975`
3. seed44:
   - `pair_acc=0.8190`, `auc=0.7823`

Aggregated result:
1. `mean_pair_acc=0.8154`
2. `mean_auc=0.7857`
3. `std_pair_acc=0.0075`
4. `std_auc=0.0086`
5. gate interpretation:
   - pass

Why this matters:
1. It resolves the earlier ambiguity around D6-T:
   - the ranking branch is learnable,
   - and it can be made seed-stable with shared split + deterministic ordering + conservative optimization.
2. It narrows the real remaining problem:
   - not "can the head learn ranking at all?",
   - but "can that ranking signal transfer to our in-domain corruption metrics and survive mixed-source data?".
3. Therefore the next project move is no longer more random hyperparameter search on unstable DT2,
   but:
   - transfer verification,
   - mixed-source robustness,
   - then return to mainline D6 promotion logic.

## 2026-03-05 Critical Diagnosis (Project Direction)

This diagnosis is direction-critical and is now reflected in `phase_D_plan.md`.

What is confirmed:
1. D3 `q_teacher` is not a universal failure; it can improve pair metrics over
   MC in some controlled runs.
2. Current D4ABC settings do not deliver usable gains.
3. Low absolute Brier in teacher runs can be misleading under skewed targets;
   baseline-relative calibration checks must be mandatory.

What changes now:
1. Keep D3 path active.
2. Prioritize D2 teacher-corruption alignment integrity before new ablations.
3. Treat current HQ and D4ABC outcomes as exploratory, not promotion evidence.
4. Apply the methodology pivot recorded in `phase_D_plan.md` Section 1.2:
   - stop using calibration gains as primary success signal,
   - switch to ranking-first training and evaluation,
   - enforce high-quality within-question pair construction.

Root-cause diagnosis (newly formalized):
1. This is an objective mismatch:
   - calibration/regression targets can improve while pair discrimination stays weak.
2. External PRM should be used mainly for pair gating/filtering and confidence
   control, not as a direct replacement for ranking supervision.
3. Promotion decisions must be ranking-led (`corr_pair_acc`, `corr_auc`) with
   calibration as a secondary guardrail.

Mentor guidance (actionable summary, 2026-03-05):
1. Our current plateau is a community-known failure mode in PRM/value-head
   training, not an isolated implementation mistake.
2. Blocking point is process-supervision quality:
   - weak outcome-only labels,
   - noisy/low-margin pairs,
   - score-fusion without sufficient pair redesign.
3. Project implication:
   - keep BCR/ABR direction,
   - prioritize D6 ranking-first + strict pair quality path,
   - use D4/D5 as ablation context rather than promotion evidence.
4. New branch added in plan:
   - `D6-T Triplet Validation Branch` (Math-Shepherd/PRM800K),
   - objective is to prove ranking learnability with high-confidence
     same-question triplets before further C1-heavy expansion.
5. Execution update:
   - `phase_D_plan.md` now includes a staged checklist `D6T-0 -> D6T-5`
     (contract freeze, data quality gate, seed-3 stability gate, mixed-source
     robustness gate, and migration decision gate).

## 2026-03-05 Latest Result Snapshot (Historical, Superseded by 2026-03-06 Stable DT2)

This section records the latest overnight reruns and their implications.

### New run outputs (confirmed)

1. D6-T cal-aux substitute reruns:
   - `assets/artifacts/phase_d6t_logs/d6t_dt2_calaux_substitute_0305_0522/final_summary.md`
   - `assets/artifacts/phase_d6t_logs/d6t_dt2_calaux_substitute_0305_0523/final_summary.md`
   - `assets/artifacts/phase_d6t_logs/d6t_dt2_calaux_substitute_0305_0526/final_summary.md`
2. D4 full rerun attempt:
   - `assets/artifacts/phase_d_logs/d4abc_full_0305_0523/suite.log`
3. New D4 3-way smoke summary:
   - `assets/artifacts/phase_d_logs/phase_cd_full_0304_2347_d4_strategyqa_smoke_3way/final_summary.md`

### Key observed numbers

1. Historical status before the stable recipe:
   - `DT2_MATH_SHEPHERD_SEED3` was unstable across seeds,
   - typical pattern in latest reruns:
     - seed42: near-random to moderate (`pair_acc ~0.54-0.77`)
     - seed43: strong (`pair_acc ~0.75-0.79`, `auc ~0.74-0.78`)
     - seed44: collapse (`pair_acc ~0.36-0.37`, `auc ~0.37-0.41`)
   - all reruns fail gate due high variance; `gate_pass=False`.
2. D6-T cal-aux substitute does not improve robustness:
   - no stable lift vs ranking-only baseline;
   - variance remains the dominant failure mode.
3. D4ABC full rerun (`d4abc_full_0305_0523`) did not reach stage summary:
   - log stops in `D4A_train_c2` during large external pair encoding;
   - no `final_summary.md` generated.
4. D4 3-way smoke confirms prior pattern:
   - teacher/fused strongly improve Brier (calibration),
   - but ranking metrics remain only modestly above random (`pair_acc ~0.56`, `auc ~0.52`).

### Critical diagnosis update

1. Current blocker is now clearly "ranking stability", not "single-run learnability":
   - we can get good seeds, but not seed-robust behavior.
2. Calibration auxiliary does not solve this blocker:
   - improves calibration diagnostics in some runs,
   - but does not reliably prevent ranking collapse.
3. Full external-pair chain is still compute-fragile at current scale:
   - pair volume is too large for reliable one-shot full reruns without tighter caps/sharding.

### Immediate next steps (execution order)

1. Keep D6-T as the main promotion gate, but enforce robust-seed criteria:
   - require `mean_pair_acc >= 0.65`, `mean_auc >= 0.65`,
   - plus `std_pair_acc <= 0.05` and `std_auc <= 0.05` before promotion.
2. Run seed-collapse triage before new architecture changes:
   - compare score distributions for strong seed vs collapsed seed from `phase_d_triplet_eval/*/pair_scores.jsonl`;
   - explicitly check saturation (`chosen_score` and `rejected_score` both near 1).
3. For D4 full-chain reruns, cap external pair volume first:
   - do not rerun uncapped D4A full;
   - start from bounded-pair variants and recover end-to-end stage completion.
4. Treat D4 teacher/fused improvements as calibration-side evidence only:
   - do not use current D4 outputs as ranking-promotion evidence.

Evidence files:
1. `assets/artifacts/phase_d_logs/phase_d_bundle_smoke/final_summary.md`
2. `assets/artifacts/phase_d_logs/overnight_d4_conf_fulltrain/final_summary.md`
3. `assets/artifacts/phase_cd_reports/smoke_local/summary.md`
4. `assets/artifacts/phase_d_logs/phase_d4abc_smoke_opt/stage_results.jsonl`

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
- `phase_C_fix_value_head.md` (Phase C diagnosis and external-help rationale)
- `phase_D_plan.md` (Phase D external-PRM-supported execution plan)
- `foundation_reliability_audit.md` (low-level risk scan + hardening plan before Phase B scale)

## 2026-03-05 Reproducibility Command Bundles (Key)

This section is a compact "runbook" of the most important command groups for
reproducing the current project state.

### Common prerequisite

```bash
cd /home/zling/y/bcr/ref
export PYTHONPATH=$PWD/src
```

### Repro Tracking (New)

Raw outputs are already persisted by default in core pipelines:
1. Phase A (`scripts/phase_a_generate_and_eval.py`) writes:
   - `predictions.jsonl` (raw model outputs),
   - `scored_predictions.jsonl`,
   - `metrics.json`,
   - `manifest.json`,
   - `console.log`.
2. Phase C/D runs similarly persist metrics + manifest + summary artifacts.

To auto-maintain command and result docs with compact signal-only summaries:

```bash
# Run any experiment command through the logger wrapper.
bash scripts/run_with_exp_log.sh \
  python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl \
  --run-name phase_a_log_demo \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --no-do-sample \
  --max-new-tokens 16 \
  --strategyqa-decode-mode binary_choice \
  --batch-size 1 \
  --max-samples 20 \
  --no-compare-latest-same-name
```

Control switches (important):

```bash
# 1) Keep logging but skip markdown rewrite (JSONL only).
EXP_LOG_UPDATE_DOCS=0 \
bash scripts/run_with_exp_log.sh \
  python -u scripts/phase_b_eval_faithfulness.py --help

# 2) Temporary bypass logger and run command directly.
EXP_LOG_ENABLE=0 \
bash scripts/run_with_exp_log.sh \
  python -u scripts/phase_c_train_pik.py --help

# 3) Add run tags and a short note for filtering.
EXP_LOG_TAGS="phase_d,smoke,ablation" \
EXP_LOG_NOTE="D4A baseline smoke after cache check" \
bash scripts/run_with_exp_log.sh \
  python -u scripts/phase_b_prepare_value_data.py --help

# 4) Rebuild docs from existing JSONL records only (no new run).
python -u scripts/experiment_command_logger.py \
  --rebuild-docs-only
```

Auto-maintained files:
1. `docs/commands_to_run.md`: command-family catalog (normalized signatures).
2. `docs/command_result.md`: per-run concise outcomes (key metrics only).
3. `assets/artifacts/command_logs/run_records.jsonl`: full machine-readable history.

Rule for repeated runs of the same command:
1. Each run gets a unique `run_id`.
2. A normalized `command_family_id` groups semantically same commands.
3. `family_run_index` increments across retries, so conflicting outcomes remain visible.

### Bundle A: Phase A whole-dataset baselines

```bash
# StrategyQA whole-corpus
ACTIVE_PARAM_GROUP=A11 \
RUN_PREFIX=strategyqa_whole_2290 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_a_benchmark_suite.sh

# GSM8K whole-corpus
ACTIVE_PARAM_GROUP=A12 \
RUN_PREFIX=gsm8k_whole_corpus \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_a_benchmark_suite.sh
```

### Bundle B: Phase B full PEFT baselines

```bash
# StrategyQA full PEFT + frozen-protocol eval
ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_FULL \
RUN_PREFIX=phase_b_strategyqa_full \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_b_training_suite.sh

# GSM8K full PEFT + frozen-protocol eval
ACTIVE_PHASE_B_GROUP=B2_GSM8K_FULL \
RUN_PREFIX=phase_b_gsm8k_full \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_b_training_suite.sh
```

### Bundle C: Phase C value-head controls (smoke + full)

```bash
# Fast smoke (CQR quality controls enabled)
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_CQR_SMOKE \
RUN_PREFIX=phase_c_cqr_smoke \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_value_suite.sh

# Full-scale control run
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_CQR_FULL \
RUN_PREFIX=phase_c_cqr_full \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_c_value_suite.sh
```

### Bundle C-PIK: Question-level P(IK) sanity track

```bash
ACTIVE_PHASE_C_PIK_GROUP=PIK_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_c_pik_smoke \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_c_pik_suite.sh
```

### Bundle D1: External PRM teacher scoring

```bash
# Score one C1 eval artifact with Qwen2.5-Math-PRM-7B
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_c_score_prm_teacher.py \
  --phase-c-dir assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d \
  --teacher-model-path assets/models/Qwen2.5-Math-PRM-7B \
  --batch-size 192 \
  --max-length 2048 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda
```

### Bundle D2+D3: 3-way teacher/MC/fused ablation

```bash
ACTIVE_PHASE_D_GROUP=D4_STRATEGYQA_SMOKE_3WAY_HQ \
RUN_PREFIX=phase_d_bundle_smoke_hq \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_d_teacher_suite.sh
```

### Bundle D6: Ranking-first method-correction (new mainline engineering)

```bash
# D6 control: ranking-first + MC target only (teacher-free C1 path)
ACTIVE_PHASE_D_GROUP=D6_STRATEGYQA_SMOKE_RANKING_CTRL \
RUN_PREFIX=phase_d6_smoke_rank_ctrl \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_d_teacher_suite.sh

# D6 main: ranking-first + PRM pair gate (teacher only for pair consensus)
ACTIVE_PHASE_D_GROUP=D6_STRATEGYQA_SMOKE_RANKING_PRM_GATE \
RUN_PREFIX=phase_d6_smoke_rank_pairgate \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_d_teacher_suite.sh
```

### Bundle D6-T: Triplet validation branch (DT1..DT6, mentor-mandated)

```bash
# Optional prerequisite for DT3/DT4/DT5/DT6:
# clone PRM800K data repo locally (path used by suite default).
git clone https://github.com/openai/prm800k assets/external_datasets/openai_prm800k

# PRM800K uses Git LFS for large JSONL files.
# If `git lfs` is missing in your env, install it first:
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base
conda install -y -c conda-forge git-lfs

# Pull real JSONL payloads (instead of LFS pointer stubs).
git lfs install --skip-repo
git -C assets/external_datasets/openai_prm800k lfs install
git -C assets/external_datasets/openai_prm800k lfs pull

# Quick parser smoke before running DT3/DT4:
python -u scripts/phase_d_prepare_external_pairs.py \
  --run-name prm800k_install_smoke \
  --output-root assets/artifacts/phase_d_external_pairs \
  --prm800k-path assets/external_datasets/openai_prm800k \
  --max-pairs-total 200 \
  --max-pairs-per-source 200 \
  --validation-ratio 0.1 \
  --min-pair-confidence 0.55 \
  --min-chars 12 \
  --max-length-ratio 3.0 \
  --max-token-overlap 0.99 \
  --max-pairs-per-sample 2 \
  --overwrite

# DT1: Math-Shepherd smoke
ACTIVE_PHASE_D6T_GROUP=DT1_MATH_SHEPHERD_SMOKE \
RUN_PREFIX=d6t_dt1_smoke \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<eval_dir> \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_d_triplet_validation_suite.sh

# DT2: Math-Shepherd seed-3 gate
ACTIVE_PHASE_D6T_GROUP=DT2_MATH_SHEPHERD_SEED3 \
RUN_PREFIX=d6t_dt2_seed3 \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<eval_dir> \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_d_triplet_validation_suite.sh

# DT3: PRM800K smoke (requires local PRM800K path)
ACTIVE_PHASE_D6T_GROUP=DT3_PRM800K_SMOKE \
RUN_PREFIX=d6t_dt3_prm800k_smoke \
PRM800K_PATH=assets/external_datasets/openai_prm800k \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<eval_dir> \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_d_triplet_validation_suite.sh

# DT4: mixed Math-Shepherd + PRM800K seed-3
ACTIVE_PHASE_D6T_GROUP=DT4_MIXED_MS_PRM800K_SEED3 \
RUN_PREFIX=d6t_dt4_mixed_seed3 \
PRM800K_PATH=assets/external_datasets/openai_prm800k \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<eval_dir> \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_d_triplet_validation_suite.sh

# DT5: ablation without strict pair filters
ACTIVE_PHASE_D6T_GROUP=DT5_ABLATION_NO_FILTER \
RUN_PREFIX=d6t_dt5_no_filter \
PRM800K_PATH=assets/external_datasets/openai_prm800k \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<eval_dir> \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_d_triplet_validation_suite.sh

# DT6: ablation with calibration auxiliary
ACTIVE_PHASE_D6T_GROUP=DT6_ABLATION_WITH_CAL_AUX \
RUN_PREFIX=d6t_dt6_cal_aux \
PRM800K_PATH=assets/external_datasets/openai_prm800k \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<eval_dir> \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_d_triplet_validation_suite.sh
```

Direct external held-out eval only (no retraining):

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_d_eval_external_pairs.py \
  --value-run-dir assets/artifacts/phase_c_runs/<d6t_c2_run_dir> \
  --external-pair-jsonl assets/artifacts/phase_d_external_pairs/<d6t_pair_run_dir>/validation_pairs.jsonl \
  --run-name d6t_manual_ext_eval \
  --checkpoint-name best \
  --batch-size 64 \
  --require-cuda
```

Triplet-suite safety note:
1. Avoid shrinking eval by setting both
   `--max-eval-samples` and `--max-corruption-variants-eval` inside
   `C2_TRAIN_EXTRA_ARGS`; this can break clean/corruption ID alignment in C2 eval.

Low-resource quick smoke (small batch, one seed):

```bash
ACTIVE_PHASE_D6T_GROUP=DT1_MATH_SHEPHERD_SMOKE \
RUN_PREFIX=d6t_dt1_lowres_smoke \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<eval_dir> \
C2_EPOCHS=1 \
C2_TRAIN_BATCH_SIZE=8 \
C2_EVAL_BATCH_SIZE=8 \
MAX_PAIRS_PER_SOURCE=200 \
MAX_PAIRS_TOTAL=300 \
MIN_PAIR_CONFIDENCE=0.55 \
FEATURE_CACHE_MODE=read_write \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_d_triplet_validation_suite.sh
```

Even lighter fallback (if still OOM / queue pressure):

```bash
ACTIVE_PHASE_D6T_GROUP=DT1_MATH_SHEPHERD_SMOKE \
RUN_PREFIX=d6t_dt1_tiny_smoke \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<eval_dir> \
C2_EPOCHS=1 \
C2_TRAIN_BATCH_SIZE=4 \
C2_EVAL_BATCH_SIZE=4 \
MAX_PAIRS_PER_SOURCE=80 \
MAX_PAIRS_TOTAL=120 \
MIN_PAIR_CONFIDENCE=0.55 \
FEATURE_CACHE_MODE=read_write \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_d_triplet_validation_suite.sh
```

If you hit teacher-coverage failure in D2 (`coverage < teacher_min_coverage`),
first ensure teacher-score files match the current C1 artifact. For exploratory
runs, you may lower the gate:

```bash
ACTIVE_PHASE_D_GROUP=D4_STRATEGYQA_SMOKE_3WAY \
RUN_PREFIX=d4_low_cov_probe \
TEACHER_MIN_COVERAGE=0.55 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_d_teacher_suite.sh
```

### Bundle D6/D4 diagnosis matrix (current active paths, 2026-03-05)

Use this matrix when the main issue is "good seeds exist, but seed stability fails".

Common env:

```bash
cd /home/zling/y/bcr/ref
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0

export PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/phase_d_bundle_smoke_d4_strategyqa_smoke_3way_d2_c1_train__b06602a61215
export PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/phase_d_bundle_smoke_d4_strategyqa_smoke_3way_d2_c1_eval__fd79eb66bbe5
export MATH_SHEPHERD_PATH=assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl
export FEATURE_CACHE_ROOT=assets/artifacts/phase_c_feature_cache
export FEATURE_CACHE_MODE=read_write
```

G1 (baseline, ranking-only, seed-3 gate):

```bash
ACTIVE_PHASE_D6T_GROUP=DT2_MATH_SHEPHERD_SEED3 \
RUN_PREFIX=diag_g1_dt2_rank_seed3_$(date +%m%d_%H%M) \
PHASE_C_TRAIN_DIR=$PHASE_C_TRAIN_DIR \
PHASE_C_EVAL_DIR=$PHASE_C_EVAL_DIR \
MATH_SHEPHERD_PATH=$MATH_SHEPHERD_PATH \
C2_EPOCHS=4 \
C2_LR=1e-4 \
C2_TRAIN_BATCH_SIZE=64 \
C2_EVAL_BATCH_SIZE=64 \
FEATURE_CACHE_ROOT=$FEATURE_CACHE_ROOT \
FEATURE_CACHE_MODE=$FEATURE_CACHE_MODE \
bash scripts/run_phase_d_triplet_validation_suite.sh
```

G2 (repeatability rerun, same config new prefix):

```bash
ACTIVE_PHASE_D6T_GROUP=DT2_MATH_SHEPHERD_SEED3 \
RUN_PREFIX=diag_g2_dt2_rank_repeat_$(date +%m%d_%H%M) \
PHASE_C_TRAIN_DIR=$PHASE_C_TRAIN_DIR \
PHASE_C_EVAL_DIR=$PHASE_C_EVAL_DIR \
MATH_SHEPHERD_PATH=$MATH_SHEPHERD_PATH \
C2_EPOCHS=4 \
C2_LR=1e-4 \
C2_TRAIN_BATCH_SIZE=64 \
C2_EVAL_BATCH_SIZE=64 \
FEATURE_CACHE_ROOT=$FEATURE_CACHE_ROOT \
FEATURE_CACHE_MODE=$FEATURE_CACHE_MODE \
bash scripts/run_phase_d_triplet_validation_suite.sh
```

G3 (calibration-aux substitute ablation):

```bash
ACTIVE_PHASE_D6T_GROUP=DT2_MATH_SHEPHERD_SEED3 \
RUN_PREFIX=diag_g3_dt2_calaux_$(date +%m%d_%H%M) \
PHASE_C_TRAIN_DIR=$PHASE_C_TRAIN_DIR \
PHASE_C_EVAL_DIR=$PHASE_C_EVAL_DIR \
MATH_SHEPHERD_PATH=$MATH_SHEPHERD_PATH \
C2_TRAIN_MODE=joint \
C2_CALIBRATION_LOSS=bce_mse \
C2_TRAIN_EXTRA_ARGS="--calibration-mse-weight 0.2 --calibration-bce-weight 0.2" \
C2_EPOCHS=4 \
C2_LR=1e-4 \
C2_TRAIN_BATCH_SIZE=64 \
C2_EVAL_BATCH_SIZE=64 \
FEATURE_CACHE_ROOT=$FEATURE_CACHE_ROOT \
FEATURE_CACHE_MODE=$FEATURE_CACHE_MODE \
bash scripts/run_phase_d_triplet_validation_suite.sh
```

G4 (stability-tuned DT2, lower lr + ranking_score selection):

```bash
ACTIVE_PHASE_D6T_GROUP=DT2_MATH_SHEPHERD_SEED3 \
RUN_PREFIX=diag_g4_dt2_stability_$(date +%m%d_%H%M) \
PHASE_C_TRAIN_DIR=$PHASE_C_TRAIN_DIR \
PHASE_C_EVAL_DIR=$PHASE_C_EVAL_DIR \
MATH_SHEPHERD_PATH=$MATH_SHEPHERD_PATH \
C2_EPOCHS=4 \
C2_LR=5e-5 \
C2_TRAIN_BATCH_SIZE=64 \
C2_EVAL_BATCH_SIZE=64 \
C2_TRAIN_EXTRA_ARGS="--checkpoint-selection-metric ranking_score --contrastive-margin 0.2 --dropout-prob 0.1" \
FEATURE_CACHE_ROOT=$FEATURE_CACHE_ROOT \
FEATURE_CACHE_MODE=$FEATURE_CACHE_MODE \
bash scripts/run_phase_d_triplet_validation_suite.sh
```

G4b (stability + anti-saturation regularization):

```bash
ACTIVE_PHASE_D6T_GROUP=DT2_MATH_SHEPHERD_SEED3 \
RUN_PREFIX=diag_g4b_dt2_antisat_$(date +%m%d_%H%M) \
PHASE_C_TRAIN_DIR=$PHASE_C_TRAIN_DIR \
PHASE_C_EVAL_DIR=$PHASE_C_EVAL_DIR \
MATH_SHEPHERD_PATH=$MATH_SHEPHERD_PATH \
C2_EPOCHS=4 \
C2_LR=5e-5 \
C2_TRAIN_BATCH_SIZE=64 \
C2_EVAL_BATCH_SIZE=64 \
C2_TRAIN_EXTRA_ARGS="--checkpoint-selection-metric ranking_score --contrastive-margin 0.2 --dropout-prob 0.1 --anti-saturation-weight 0.03 --anti-saturation-logit-threshold 4.0" \
FEATURE_CACHE_ROOT=$FEATURE_CACHE_ROOT \
FEATURE_CACHE_MODE=$FEATURE_CACHE_MODE \
bash scripts/run_phase_d_triplet_validation_suite.sh
```

G5 (D4A-only capacity diagnosis with strict caps, must finish):

```bash
ACTIVE_PHASE_D4_GROUP=D4A_STRATEGYQA_SMOKE \
RUN_PREFIX=diag_g5_d4a_cap_$(date +%m%d_%H%M) \
PHASE_C_TRAIN_DIR=$PHASE_C_TRAIN_DIR \
PHASE_C_EVAL_DIR=$PHASE_C_EVAL_DIR \
R_PRM_ROOT=assets/external_datasets/kevinpro_r_prm \
PRMBENCH_PREVIEW_PATH=assets/external_datasets/hitsmy_prmbench_preview/prmbench_preview.jsonl \
PAIR_PREP_EXTRA_ARGS="--max-pairs-per-source 4000 --max-pairs-total 12000" \
C2_TRAIN_EXTRA_ARGS="--external-pair-max-train-samples 30000" \
C2_EPOCHS=4 \
C2_TRAIN_BATCH_SIZE=32 \
C2_EVAL_BATCH_SIZE=32 \
D4_EVAL_POSTHOC_MODE=none \
FEATURE_CACHE_ROOT=$FEATURE_CACHE_ROOT \
FEATURE_CACHE_MODE=$FEATURE_CACHE_MODE \
bash scripts/run_phase_d_external_pair_suite.sh
```

G6 (D4ABC chain with caps, diagnose stage-by-stage survivability):

```bash
ACTIVE_PHASE_D4_GROUP=D4ABC_STRATEGYQA_SMOKE \
RUN_PREFIX=diag_g6_d4abc_cap_$(date +%m%d_%H%M) \
PHASE_C_TRAIN_DIR=$PHASE_C_TRAIN_DIR \
PHASE_C_EVAL_DIR=$PHASE_C_EVAL_DIR \
R_PRM_ROOT=assets/external_datasets/kevinpro_r_prm \
PRMBENCH_PREVIEW_PATH=assets/external_datasets/hitsmy_prmbench_preview/prmbench_preview.jsonl \
MATH_SHEPHERD_PATH=assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl \
RLHFLOW_MISTRAL_ROOT=assets/external_datasets/rlhflow_mistral_prm \
RLHFLOW_DEEPSEEK_PATH=assets/external_datasets/rlhflow_deepseek_prm/deepseek_instruct_data.jsonl \
PAIR_PREP_EXTRA_ARGS="--max-pairs-per-source 5000 --max-pairs-total 15000" \
C2_TRAIN_EXTRA_ARGS="--external-pair-max-train-samples 40000" \
C2_EPOCHS=4 \
C2_TRAIN_BATCH_SIZE=32 \
C2_EVAL_BATCH_SIZE=32 \
D4_EVAL_POSTHOC_MODE=none \
FEATURE_CACHE_ROOT=$FEATURE_CACHE_ROOT \
FEATURE_CACHE_MODE=$FEATURE_CACHE_MODE \
bash scripts/run_phase_d_external_pair_suite.sh
```

Post-run saturation audit (for any D6-T run prefix):

```bash
python - <<'PY'
import json, statistics as st
from pathlib import Path
import glob

prefix = "diag_g1_dt2_rank_seed3"  # change this
paths = sorted(glob.glob(f"assets/artifacts/phase_d_triplet_eval/{prefix}*_ext_eval_*/pair_scores.jsonl"))
for p in paths:
    chosen, rejected, margin = [], [], []
    for line in Path(p).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        c = float(row.get("chosen_score", 0.0))
        r = float(row.get("rejected_score", 0.0))
        chosen.append(c)
        rejected.append(r)
        margin.append(c - r)
    if not margin:
        continue
    print(Path(p).parent.name)
    print(" chosen_mean/std:", round(sum(chosen)/len(chosen), 6), round(st.pstdev(chosen), 6))
    print(" reject_mean/std:", round(sum(rejected)/len(rejected), 6), round(st.pstdev(rejected), 6))
    print(" margin_mean/std:", round(sum(margin)/len(margin), 6), round(st.pstdev(margin), 6))
    print(" margin_min/max :", round(min(margin), 6), round(max(margin), 6))
    print()
PY
```

### Bundle D4ABC: External pair bootstrap chain

```bash
ACTIVE_PHASE_D4_GROUP=D4ABC_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_d4abc_smoke \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<your_train_c1_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<your_eval_c1_dir> \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_d_external_pair_suite.sh
```

### Bundle E: One-click C-vs-D control + global diagnosis (new)

This bundle is the recommended orchestration entrypoint when GPUs are free:
- runs comparable C baseline,
- runs D teacher ablations,
- runs D4 external-pair chain when C1 dirs are available,
- always emits a unified C/D diagnosis report.

```bash
# Light bundle (fast)
ACTIVE_PHASE_CD_GROUP=CD_LIGHT \
RUN_PREFIX=phase_cd_light \
CUDA_PHASE_C=1 CUDA_PHASE_D=2 CUDA_PHASE_D4=3 \
bash scripts/run_phase_cd_control_suite.sh

# Full bundle (detailed)
ACTIVE_PHASE_CD_GROUP=CD_FULL \
RUN_PREFIX=phase_cd_full \
CUDA_PHASE_C=1 CUDA_PHASE_D=2 CUDA_PHASE_D4=3 \
bash scripts/run_phase_cd_control_suite.sh

# Method-correction bundle (PRM as pair gate, not target)
ACTIVE_PHASE_CD_GROUP=CD_METHOD_FIX_LIGHT \
RUN_PREFIX=phase_cd_fix_light \
CUDA_PHASE_C=1 CUDA_PHASE_D=2 CUDA_PHASE_D4=3 \
bash scripts/run_phase_cd_control_suite.sh

ACTIVE_PHASE_CD_GROUP=CD_METHOD_FIX_FULL \
RUN_PREFIX=phase_cd_fix_full \
CUDA_PHASE_C=1 CUDA_PHASE_D=2 CUDA_PHASE_D4=3 \
bash scripts/run_phase_cd_control_suite.sh

# D6 ranking-first one-click bundle
ACTIVE_PHASE_CD_GROUP=CD_D6_RANKING_LIGHT \
RUN_PREFIX=phase_cd_d6_rank_light \
CUDA_PHASE_C=1 CUDA_PHASE_D=2 CUDA_PHASE_D4=3 \
bash scripts/run_phase_cd_control_suite.sh
```

Standalone diagnosis-only command:

```bash
python -u scripts/phase_cd_compare_report.py \
  --phase-c-logs-root assets/artifacts/phase_c_logs \
  --phase-d-logs-root assets/artifacts/phase_d_logs \
  --output-dir assets/artifacts/phase_cd_reports/manual_diag \
  --dataset strategyqa
```

Standalone promotion-gate decision command (new):

```bash
python -u scripts/phase_d_promotion_gate.py \
  --phase-c-logs-root assets/artifacts/phase_c_logs \
  --phase-d-logs-root assets/artifacts/phase_d_logs \
  --phase-d6t-logs-root assets/artifacts/phase_d6t_logs \
  --output-dir assets/artifacts/phase_d_gate_reports/manual_gate \
  --dataset strategyqa \
  --d6-control-group-id D6_STRATEGYQA_SMOKE_RANKING_CTRL \
  --d6-gated-group-id D6_STRATEGYQA_SMOKE_RANKING_PRM_GATE \
  --required-d6t-groups DT2_MATH_SHEPHERD_SEED3_STABLE,DT4_MIXED_MS_PRM800K_SEED3_STABLE \
  --min-corr-pair-acc 0.55 \
  --min-corr-auc 0.60 \
  --d6t-mean-pair-acc-min 0.65 \
  --d6t-mean-auc-min 0.65 \
  --d6t-std-max 0.05

cat assets/artifacts/phase_d_gate_reports/manual_gate/summary.md
```

Gate-backlog completion commands (for missing D6/D6-T artifacts in gate report):

```bash
# Optional preflight: quickly check required files.
ls -l assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl
ls -ld assets/external_datasets/openai_prm800k
ls -l assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_mcorr4_c2_strategyqa_quality_first_full_c1_train__b7a8789f1974/teacher_prefix_scores.jsonl
ls -l assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d/teacher_prefix_scores.jsonl
```

```bash
# 1) Fill missing D6 control arm artifact.
ACTIVE_PHASE_D_GROUP=D6_STRATEGYQA_SMOKE_RANKING_CTRL \
RUN_PREFIX=phase_d6_smoke_rank_ctrl_fill \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_d_teacher_suite.sh

# 2) Fill missing D6 PRM-gated arm artifact.
ACTIVE_PHASE_D_GROUP=D6_STRATEGYQA_SMOKE_RANKING_PRM_GATE \
RUN_PREFIX=phase_d6_smoke_rank_pairgate_fill \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_d_teacher_suite.sh

# 3) Fill missing DT4 mixed-source seed-3 artifact.
# Set PRM800K_PATH to your local mirror if default path is absent.
ACTIVE_PHASE_D6T_GROUP=DT4_MIXED_MS_PRM800K_SEED3_STABLE \
RUN_PREFIX=phase_d6t_dt4_seed3_fill \
MATH_SHEPHERD_PATH=assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl \
PRM800K_PATH=assets/external_datasets/openai_prm800k \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_d_triplet_validation_suite.sh
```

DT2 stability re-run command (historical; superseded by the stable wrapper below):

```bash
ACTIVE_PHASE_D6T_GROUP=DT2_MATH_SHEPHERD_SEED3 \
RUN_PREFIX=phase_d6t_dt2_seed3_tight \
MATH_SHEPHERD_PATH=assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl \
CUDA_VISIBLE_DEVICES=1 \
MIN_PAIR_CONFIDENCE=0.65 \
MAX_PAIRS_TOTAL=12000 \
MAX_PAIRS_PER_SOURCE=6000 \
MAX_PAIRS_PER_SAMPLE=1 \
C2_LR=5e-5 \
C2_EPOCHS=3 \
C2_TRAIN_EXTRA_ARGS="--lambda-contrastive 1.0 --anti-saturation-weight 0.03 --anti-saturation-logit-threshold 4.0" \
bash scripts/run_phase_d_triplet_validation_suite.sh
```

D6-T seed-stability recommended defaults (new):
1. Use one shared external-pair split across all training seeds to remove split noise.
2. Enable strict deterministic backend + stable external-pair permutation inside C2.

```bash
RUN_PREFIX=phase_d6t_stable_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_d6t_stable_suite.sh
```

Equivalent raw command:

```bash
ACTIVE_PHASE_D6T_GROUP=DT2_MATH_SHEPHERD_SEED3_STABLE \
RUN_PREFIX=phase_d6t_dt2_seed3_stable \
MATH_SHEPHERD_PATH=assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl \
CUDA_VISIBLE_DEVICES=1 \
D6T_PAIR_SPLIT_MODE=shared \
D6T_PAIR_SPLIT_SEED=42 \
D6T_EXTERNAL_PAIR_PERM_MODE=stable_hash \
D6T_STRICT_DETERMINISM=1 \
bash scripts/run_phase_d_triplet_validation_suite.sh
```

Next-step D6-T bundles after the stable DT2 pass:
1. Transfer check:
   - keep the stable DT2 recipe fixed,
   - also run standalone C1 eval after each seed,
   - objective: verify whether external ranking learning improves our in-domain StrategyQA corruption metrics.
2. Mixed-source robustness:
   - keep the stable recipe fixed,
   - only widen data source diversity (`Math-Shepherd + PRM800K`),
   - objective: check whether the new stability survives cross-source mixing.
3. PRM800K source isolation:
   - keep the exact stable DT2 recipe fixed,
   - remove Math-Shepherd entirely,
   - objective: determine whether `PRM800K` itself is usable, or whether it is
     the source dragging DT4 toward the gate floor.

```bash
# DT2 stable + C1 transfer check
ACTIVE_PHASE_D6T_GROUP=DT2_MATH_SHEPHERD_SEED3_STABLE_C1_TRANSFER \
RUN_PREFIX=phase_d6t_transfer_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_d6t_stable_suite.sh

# DT3 stable PRM800K-only source isolation
ACTIVE_PHASE_D6T_GROUP=DT3_PRM800K_SEED3_STABLE \
RUN_PREFIX=phase_d6t_prm800k_stable_$(date +%m%d_%H%M) \
PRM800K_PATH=assets/external_datasets/openai_prm800k \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_d6t_stable_suite.sh

# DT4 stable mixed-source robustness
ACTIVE_PHASE_D6T_GROUP=DT4_MIXED_MS_PRM800K_SEED3_STABLE \
RUN_PREFIX=phase_d6t_mixed_stable_$(date +%m%d_%H%M) \
PRM800K_PATH=assets/external_datasets/openai_prm800k \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_d6t_stable_suite.sh
```

```bash
# 4) Re-run promotion gate after missing groups are filled.
python -u scripts/phase_d_promotion_gate.py \
  --phase-c-logs-root assets/artifacts/phase_c_logs \
  --phase-d-logs-root assets/artifacts/phase_d_logs \
  --phase-d6t-logs-root assets/artifacts/phase_d6t_logs \
  --output-dir assets/artifacts/phase_d_gate_reports/manual_gate_after_fill \
  --dataset strategyqa \
  --d6-control-group-id D6_STRATEGYQA_SMOKE_RANKING_CTRL \
  --d6-gated-group-id D6_STRATEGYQA_SMOKE_RANKING_PRM_GATE \
  --required-d6t-groups DT2_MATH_SHEPHERD_SEED3_STABLE,DT4_MIXED_MS_PRM800K_SEED3_STABLE \
  --min-corr-pair-acc 0.55 \
  --min-corr-auc 0.60 \
  --d6t-mean-pair-acc-min 0.65 \
  --d6t-mean-auc-min 0.65 \
  --d6t-std-max 0.05

cat assets/artifacts/phase_d_gate_reports/manual_gate_after_fill/summary.md
```

Notes:
1. Current default `192` batch sizing in Phase C/P(IK)/D suites is intentional
   (safer for multi-tenant GPUs while keeping throughput high).
2. Read suite outcomes from `assets/artifacts/*_logs/<run_prefix>/final_summary.md`
   first; then inspect `suite.log` only when diagnosing failures.

## Phase D Kickoff (Active Track)

Phase D objective is to fix the remaining value-head learnability bottleneck by
injecting external PRM teacher signal into the existing C1/C2 pipeline.

Phase C closeout summary:
1. C1/C2 infra is stable and reproducible; this is no longer a tooling-failure problem.
2. Best C2 calibration improved versus early runs but still fails the baseline promotion gate.
3. Corruption-ordering metrics remain near-random in most variants.
4. Conclusion: supervision quality is the main blocker, so Phase D shifts to
   external-PRM-supported label quality improvement.

Authoritative Phase D plan:
- `phase_D_plan.md`

## 2026-03-04 Ops Update (Verified Command Set)

This section tracks the latest infrastructure commands that are useful, stable,
and verified in this repository state.

### Prerequisite (recommended for all commands below)

```bash
cd /home/zling/y/bcr/ref
export PYTHONPATH=$PWD/src
```

### A. One-click feature-cache healthcheck (new, recommended)

Fast profile (static checks + cache-module selftest):

```bash
RUN_PREFIX=cache_hc_fast \
CHECK_PROFILE=fast \
bash scripts/run_feature_cache_healthcheck.sh
```

Full profile (includes optional runtime smoke when artifacts/GPU are available):

```bash
RUN_PREFIX=cache_hc_full \
CHECK_PROFILE=full \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_feature_cache_healthcheck.sh
```

Where to read results:

```bash
cat assets/artifacts/healthcheck_logs/cache_hc_fast/final_summary.md
cat assets/artifacts/healthcheck_logs/cache_hc_fast/suite.log
```

### B. Direct static integrity checks (manual fallback)

```bash
python -m py_compile \
  src/ours/phase_b/feature_cache.py \
  scripts/phase_b_train_value.py \
  scripts/phase_b_eval_faithfulness.py \
  scripts/phase_c_train_pik.py \
  scripts/phase_c_eval_pik.py

bash -n \
  scripts/run_phase_c_value_suite.sh \
  scripts/run_phase_c_pik_suite.sh \
  scripts/run_phase_d_external_pair_suite.sh
```

### C. Phase C value suite with safe cache defaults

```bash
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_c_cache_smoke \
FEATURE_CACHE_ROOT=assets/artifacts/phase_c_feature_cache \
FEATURE_CACHE_MODE=read_write \
FEATURE_CACHE_LOCK_TIMEOUT_SEC=600 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_value_suite.sh
```

### D. Phase C P(IK) suite with safe cache defaults

```bash
ACTIVE_PHASE_C_PIK_GROUP=PIK_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_c_pik_cache_smoke \
FEATURE_CACHE_ROOT=assets/artifacts/phase_c_feature_cache \
FEATURE_CACHE_MODE=read_write \
FEATURE_CACHE_LOCK_TIMEOUT_SEC=600 \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_c_pik_suite.sh
```

### E. D4 suite with robust eval-posthoc fallback

Use `D4_EVAL_POSTHOC_MODE=auto` to avoid failure when the run has no saved
post-hoc calibrator payload.

```bash
ACTIVE_PHASE_D4_GROUP=D4A_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_d4a_safe_eval \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<c1_train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<c1_eval_dir> \
FEATURE_CACHE_ROOT=assets/artifacts/phase_c_feature_cache \
FEATURE_CACHE_MODE=read_write \
FEATURE_CACHE_LOCK_TIMEOUT_SEC=600 \
D4_EVAL_POSTHOC_MODE=auto \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_d_external_pair_suite.sh
```

### F. Cache-mode quick policy (operational)

1. `read_write`: default for normal experiments.
2. `read`: strict reproducibility mode (no new cache writes).
3. `off`: debugging mode when suspecting cache mismatch.
4. `write`: warm-up mode to prebuild caches intentionally.

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
  --per-device-train-batch-size 192 \
  --per-device-eval-batch-size 192 \
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
- `--checkpoint-selection-metric {raw_brier,posthoc_brier,corr_pair_acc,corr_auc,ranking_score}`
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
  --per-device-train-batch-size 192 \
  --per-device-eval-batch-size 192 \
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

### C1 quality-first fixes (new, high ROI)

Two data-quality fixes are now implemented for Phase C and integrated into
quality-first suite groups:

1. Primary corruption selection by quality (not ID order)
- implemented in `src/ours/phase_b/value_data.py`
- when `pair_quality.jsonl` exists, primary corruption is selected by:
  - `pair_pass_gate` first
  - higher `pair_weight`
  - larger `delta_q`
  - larger `z_delta`
- deterministic ID ordering is only a final tie-breaker

2. Two-stage rollout enrichment for uncertain prefixes
- implemented in `scripts/phase_b_prepare_value_data.py`
- stage-1: all prefixes at `K = rollout_stage1_count`
- stage-2: only uncertain prefixes are topped up to `K = rollout_stage2_count`
- uncertainty condition:
  - `|q_mean_smoothed - 0.5| < rollout_uncertain_band`
  - or `q_ci_width >= rollout_uncertain_ci_width`

New C1 CLI switches:
- `--rollout-two-stage`
- `--rollout-stage1-count`
- `--rollout-stage2-count`
- `--rollout-uncertain-band`
- `--rollout-uncertain-ci-width`
- `--corruption-selection-policy {legacy,cqr_balanced}`
- `--min-non-step-drop-per-prefix`
- `--max-step-drop-per-prefix`
- `--enable-negation-flip/--no-enable-negation-flip`
- `--enable-comparator-flip/--no-enable-comparator-flip`
- `--enable-condition-reversal/--no-enable-condition-reversal`
- `--enable-entity-substitution/--no-enable-entity-substitution`

The suite groups `C2_STRATEGYQA_QUALITY_FIRST` and
`C2_STRATEGYQA_QUALITY_FIRST_FULL` now enable this two-stage policy by default.

New C2 stratified-sampling switches (CQR-4):
- `--contrastive-stratified-sampling`
- `--contrastive-stratify-step-bucket-size`
- `--contrastive-stratify-include-no-corruption`

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

Default batch sizing for Phase C/P(IK) is now `192` (`ROLLOUT_BATCH_SIZE`, `C2_TRAIN_BATCH_SIZE`, `C2_EVAL_BATCH_SIZE`) so one 80GB GPU can safely host 3 concurrent jobs.

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

# CQR smoke (CQR-1..4 end-to-end)
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_CQR_SMOKE \
RUN_PREFIX=phase_c_cqr_smoke \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_value_suite.sh

# CQR full (CQR-1..4 end-to-end)
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_CQR_FULL \
RUN_PREFIX=phase_c_cqr_full \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_c_value_suite.sh

# Re-eval old Trick-10 recipe on CQR artifacts
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_CQR_RERUN_TRICK10 \
RUN_PREFIX=phase_c_cqr_rerun_trick10 \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_c_value_suite.sh

# Re-eval old Quality-First recipe on CQR artifacts
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_CQR_RERUN_QUALITY_FIRST \
RUN_PREFIX=phase_c_cqr_rerun_quality_first \
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
15. `C2_STRATEGYQA_CQR_SMOKE`
16. `C2_STRATEGYQA_CQR_FULL`
17. `C2_STRATEGYQA_CQR_RERUN_TRICK10`
18. `C2_STRATEGYQA_CQR_RERUN_QUALITY_FIRST`

Useful overrides:
- `TRAIN_MAX_SAMPLES`, `EVAL_MAX_SAMPLES`
- `ROLLOUT_BATCH_SIZE`, `ROLLOUT_COUNT`, `ROLLOUT_MAX_NEW_TOKENS`
- `C2_TRAIN_BATCH_SIZE`, `C2_EVAL_BATCH_SIZE`, `C2_LR`, `C2_EPOCHS`
- `C1_PREP_EXTRA_ARGS_DEFAULT`
- `C2_TRAIN_EXTRA_ARGS_DEFAULT`, `C2_EVAL_EXTRA_ARGS_DEFAULT`
- `PHASE_C_PREP_EXTRA_ARGS`, `PHASE_C_TRAIN_EXTRA_ARGS`, `PHASE_C_EVAL_EXTRA_ARGS`
- two-stage rollout overrides (via `PHASE_C_PREP_EXTRA_ARGS`):
  - `--rollout-two-stage --rollout-stage1-count 8 --rollout-stage2-count 24`
  - `--rollout-uncertain-band 0.2 --rollout-uncertain-ci-width 0.3`

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

## Phase D Entry Point (Active Track)

Phase D is now the official active stage. Theme:
- external-PRM-supported value supervision.

Why this switch:
1. Phase C infra is stable, but value quality still fails promotion gates.
2. The dominant blocker is supervision quality, not implementation reliability.

Authoritative plan:
- `phase_D_plan.md`

Phase D first implementation targets:
1. teacher sidecar scoring on existing C1 artifacts,
2. C1 teacher/MC fusion fields (`q_teacher`, `q_fused`),
3. C2 target-source ablation (`mc`, `teacher`, `fused`),
4. promotion gate before restarting BCR-lite/ABR-lite expansion.

### D1: Teacher Sidecar Scoring (Implemented)

New script:
- `scripts/phase_c_score_prm_teacher.py`

Purpose:
1. score existing C1 prefixes with external PRM teacher,
2. optionally score C1 corruptions,
3. persist sidecar outputs for D2 fusion.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_c_score_prm_teacher.py \
  --phase-c-dir assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d \
  --teacher-model-path assets/models/Qwen2.5-Math-PRM-7B \
  --batch-size 192 \
  --max-length 2048 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda
```

Full-train command (overwrite old sidecars):

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/phase_c_score_prm_teacher.py \
  --phase-c-dir assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_c2_strategyqa_quality_first_full_c1_train__90dcbacfbae1 \
  --teacher-model-path assets/models/Qwen2.5-Math-PRM-7B \
  --batch-size 192 \
  --max-length 2048 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda \
  --overwrite
```

Outputs written into the same C1 directory:
1. `teacher_prefix_scores.jsonl`
2. `teacher_corruption_scores.jsonl` (when `--score-corruptions`)
3. `teacher_errors.jsonl`
4. `teacher_summary.json`
5. `teacher_summary.md`

Contract notes:
1. D1 is sidecar-only and does not mutate existing C1/C2 artifacts.
2. If chat-template calls fail on another tokenizer, D1 falls back to a plain
   `[SYSTEM]/[USER]/[ASSISTANT]` prompt format.
3. Keep `use_cache=False` in PRM forward path for compatibility with remote-code
   model variants.
4. For legacy/incomplete C1 dirs that are missing `manifest.json`, use:
   - `--allow-missing-manifest`
   This keeps scoring usable while marking lineage as partial in `teacher_summary.json`.

### D2: C1 Teacher/MC Fusion (Now Implemented)

`scripts/phase_b_prepare_value_data.py` now supports D2 fields directly in
`rollout_targets.jsonl`:
1. `q_teacher`
2. `q_fused`
3. `teacher_available`
4. `teacher_disagree`
5. `teacher_model_id`

Join/fusion controls:
1. `--teacher-prefix-scores-jsonl`
2. `--teacher-corruption-scores-jsonl`
3. `--teacher-fuse-mode {none,fixed,confidence}`
4. `--teacher-fusion-lambda`
5. `--teacher-confidence-ci-ref`
6. `--teacher-disagree-threshold`
7. `--teacher-min-coverage`
8. `--pair-consensus-enable` and `--pair-consensus-*` thresholds

Hard checks in D2 join:
1. duplicate `prefix_id` in teacher score file -> fail,
2. coverage below `--teacher-min-coverage` -> fail.
3. duplicate `corruption_id` in teacher corruption sidecar -> fail (when provided).

Example (fixed fusion):

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_b_prepare_value_data.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl \
  --run-name phase_d_c1_train_fused \
  --build-corruptions \
  --build-rollouts \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --batch-size 192 \
  --rollout-count 16 \
  --max-new-tokens 128 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda \
  --teacher-prefix-scores-jsonl assets/artifacts/phase_c_data/strategyqa/<c1_train_dir>/teacher_prefix_scores.jsonl \
  --teacher-corruption-scores-jsonl assets/artifacts/phase_c_data/strategyqa/<c1_train_dir>/teacher_corruption_scores.jsonl \
  --teacher-fuse-mode fixed \
  --teacher-fusion-lambda 0.5 \
  --teacher-min-coverage 0.98 \
  --build-pair-quality \
  --pair-consensus-enable
```

Example (confidence fusion):

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/phase_b_prepare_value_data.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl \
  --run-name phase_d_c1_eval_fused \
  --build-corruptions \
  --build-rollouts \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --batch-size 192 \
  --rollout-count 16 \
  --max-new-tokens 128 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda \
  --teacher-prefix-scores-jsonl assets/artifacts/phase_c_data/strategyqa/<c1_eval_dir>/teacher_prefix_scores.jsonl \
  --teacher-corruption-scores-jsonl assets/artifacts/phase_c_data/strategyqa/<c1_eval_dir>/teacher_corruption_scores.jsonl \
  --teacher-fuse-mode confidence \
  --teacher-confidence-ci-ref 0.30 \
  --teacher-disagree-threshold 0.25 \
  --teacher-min-coverage 0.98 \
  --build-pair-quality \
  --pair-consensus-enable
```

### D3: C2 Target-Source Switch (Now Implemented)

`scripts/phase_b_train_value.py` now supports:
1. `--target-source {q_mean_smoothed,q_teacher,q_fused}`
2. `--target-source-missing-policy {fail,fallback_mc}`
3. `--contrastive-max-corruptions-per-prefix`
4. `--train-mode {joint,ranking_only,calibration_only,two_stage}`
5. `--two-stage-ranking-ratio`

`scripts/phase_b_eval_faithfulness.py` now supports:
1. `--target-source {from_run,q_mean_smoothed,q_teacher,q_fused}`
2. `--target-source-missing-policy {from_run,fail,fallback_mc}`

Train on fused targets:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_b_train_value.py \
  --train-dir assets/artifacts/phase_c_data/strategyqa/<phase_d_c1_train_fused_dir> \
  --eval-dir assets/artifacts/phase_c_data/strategyqa/<phase_d_c1_eval_fused_dir> \
  --run-name phase_d_c2_fused \
  --target-source q_fused \
  --target-source-missing-policy fail \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --per-device-train-batch-size 192 \
  --per-device-eval-batch-size 192
```

Standalone eval with run-aligned target source:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_b_eval_faithfulness.py \
  --value-run-dir assets/artifacts/phase_c_runs/<phase_d_c2_fused_run_dir> \
  --eval-dir assets/artifacts/phase_c_data/strategyqa/<phase_d_c1_eval_fused_dir> \
  --checkpoint-name best \
  --target-source from_run \
  --target-source-missing-policy from_run \
  --run-name phase_d_c2_fused_eval
```

### D3 via Lifecycle Suite (Recommended)

Use existing C2 groups and override only target source:

```bash
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_CQR_SMOKE \
RUN_PREFIX=phase_d_ablate_mc \
PHASE_C_TRAIN_EXTRA_ARGS="--target-source q_mean_smoothed --target-source-missing-policy fail" \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_value_suite.sh

ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_CQR_SMOKE \
RUN_PREFIX=phase_d_ablate_teacher \
PHASE_C_TRAIN_EXTRA_ARGS="--target-source q_teacher --target-source-missing-policy fail" \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_c_value_suite.sh

ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_CQR_SMOKE \
RUN_PREFIX=phase_d_ablate_fused \
PHASE_C_TRAIN_EXTRA_ARGS="--target-source q_fused --target-source-missing-policy fail" \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_c_value_suite.sh
```

### D4: Bundled D2+D3 Teacher Suite (New)

New orchestration script:
- `scripts/run_phase_d_teacher_suite.sh`

Why use it:
1. Runs D2 C1 prep (`train` + `eval`) once with teacher fusion enabled.
2. Runs D3 C2 ablation in one bundle:
   - `q_mean_smoothed` (label `mc`)
   - `q_teacher` (label `teacher`)
   - `q_fused` (label `fused`)
3. Writes one consolidated markdown table for fast comparison.

Recommended smoke command (HQ policy enabled):

```bash
ACTIVE_PHASE_D_GROUP=D4_STRATEGYQA_SMOKE_3WAY_HQ \
RUN_PREFIX=phase_d_bundle_smoke_hq \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_d_teacher_suite.sh
```

Promotion-oriented full run:

```bash
ACTIVE_PHASE_D_GROUP=D4_STRATEGYQA_FULL_3WAY_HQ \
RUN_PREFIX=phase_d_bundle_full_hq \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_d_teacher_suite.sh
```

Teacher sidecar overrides (if you scored new artifacts):

```bash
TEACHER_TRAIN_SCORES=<.../teacher_prefix_scores.jsonl> \
TEACHER_TRAIN_CORR_SCORES=<.../teacher_corruption_scores.jsonl> \
TEACHER_EVAL_SCORES=<.../teacher_prefix_scores.jsonl> \
TEACHER_EVAL_CORR_SCORES=<.../teacher_corruption_scores.jsonl> \
ACTIVE_PHASE_D_GROUP=D4_STRATEGYQA_SMOKE_3WAY_HQ \
RUN_PREFIX=phase_d_bundle_smoke_hq_custom \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_d_teacher_suite.sh
```

Optional speed knobs:
1. `TRAIN_MAX_SAMPLES=256`
2. `EVAL_MAX_SAMPLES=128`
3. `ROLLOUT_BATCH_SIZE=192`
4. `C2_TRAIN_BATCH_SIZE=192`
5. `C2_EVAL_BATCH_SIZE=192`

Where to read results:
1. `assets/artifacts/phase_d_logs/<RUN_PREFIX>/suite.log`
2. `assets/artifacts/phase_d_logs/<RUN_PREFIX>/final_summary.md`
3. C2 run dirs are listed per row in `final_summary.md`.

Summary table columns:
1. `label`, `target_source`
2. calibration: `brier`, `pearson`, `posthoc_brier`
3. corruption ordering: `pair_acc`, `auc`
4. D2 join quality: `target_cov`, `teacher_dis`
5. traceability: `c2_train_dir`, `c2_eval_dir`

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

## 11. Phase E Execution

Phase E is now the benchmark-native learnability track.

New code files:
1. `src/ours/phase_e/contracts.py`
2. `src/ours/phase_e/pairs.py`
3. `src/ours/phase_e/runtime.py`
4. `src/ours/phase_e/training.py`
5. `src/ours/phase_e/benchmark_eval.py`
6. `scripts/phase_e_prepare_pairs.py`
7. `scripts/phase_e_train_value.py`
8. `scripts/phase_e_eval_benchmark.py`
9. `scripts/run_phase_e_suite.sh`

What is already verified:
1. external-pair-only training can run without any `Phase C` artifact dir
2. `ProcessBench` benchmark-native evaluation is wired
3. `PRMBench_Preview` benchmark-native evaluation is wired
4. direct smoke and one-click suite smoke both ran through end-to-end
5. `Math-Shepherd` seed-3 has now shown a mixed but important pattern:
   - two strong held-out seeds,
   - one collapsed seed,
   - therefore the next engineering/science target is not "more transfer",
     but "find a trustworthy Math-Shepherd checkpoint family".

### 11.1 Minimal Direct Smoke

```bash
python -u scripts/phase_e_prepare_pairs.py \
  --source-bundle math_shepherd \
  --run-name phase_e_smoke_pairs \
  --max-pairs-total 16 \
  --max-pairs-per-source 16 \
  --min-pair-confidence 0.55

PAIR_DIR="$(ls -dt assets/artifacts/phase_e_pairs/phase_e_smoke_pairs__* | head -n 1)"

CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl "$PAIR_DIR/train_pairs.jsonl" \
  --eval-pairs-jsonl "$PAIR_DIR/validation_pairs.jsonl" \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_smoke_train \
  --objective-mode ranking_only \
  --num-train-epochs 1 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 8 \
  --learning-rate 5e-5 \
  --max-length 512 \
  --pair-weight-mode confidence \
  --permutation-mode stable_hash \
  --checkpoint-selection-metric ranking_score \
  --feature-cache-root assets/artifacts/phase_e_feature_cache \
  --feature-cache-mode read_write \
  --require-cuda \
  --strict-determinism

RUN_DIR="$(ls -dt assets/artifacts/phase_e_runs/phase_e_smoke_train_* | head -n 1)"

CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_e_eval_benchmark.py \
  --value-run-dir "$RUN_DIR" \
  --benchmark-id processbench_gsm8k \
  --run-name phase_e_smoke_eval \
  --max-samples 8 \
  --batch-size 8 \
  --dtype bfloat16 \
  --device-map auto \
  --feature-cache-root assets/artifacts/phase_e_feature_cache \
  --feature-cache-mode read_write \
  --require-cuda
```

### 11.2 One-Click Smoke

```bash
ACTIVE_PHASE_E_GROUP=E1_MATH_SHEPHERD_PAIR_LEARN_SMOKE \
RUN_PREFIX=phase_e_smoke_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_suite.sh
```

### 11.3 First Official Seed-3 Suites

```bash
ACTIVE_PHASE_E_GROUP=E2_MATH_SHEPHERD_PAIR_LEARN_SEED3 \
RUN_PREFIX=phase_e_math_seed3_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_suite.sh

ACTIVE_PHASE_E_GROUP=E4_RPRM_PRMBENCH_PREVIEW_SEED3 \
RUN_PREFIX=phase_e_rprm_seed3_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_e_suite.sh

ACTIVE_PHASE_E_GROUP=E5_PRM800K_PAIR_LEARN_SEED3 \
RUN_PREFIX=phase_e_prm800k_seed3_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_e_suite.sh
```

### 11.4 Math-Shepherd Trust Candidate Search

This is the new Phase E priority if the goal is to obtain one checkpoint that we
may later rely on for RL-style value judgements.

Direct matrix wrapper:

```bash
# Quick recipe comparison before spending overnight GPU time.
ACTIVE_PHASE_E_MS_GROUP=MS1_MATH_SHEPHERD_TRUST_SMOKE \
RUN_PREFIX=phase_e_ms_trust_smoke_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_mathshepherd_trust_suite.sh

# Official candidate-search matrix.
ACTIVE_PHASE_E_MS_GROUP=MS2_MATH_SHEPHERD_TRUST_SEED3_MATRIX \
RUN_PREFIX=phase_e_ms_trust_seed3_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_mathshepherd_trust_suite.sh
```

What the wrapper runs:
1. same-source control:
   - `E7_MATH_SHEPHERD_SAME_SOURCE_SEED3`
2. benchmark-aware baseline:
   - `E2_MATH_SHEPHERD_PAIR_LEARN_SEED3`
3. lower-LR stability probe:
   - `E12_MATH_SHEPHERD_TRUST_LOWLR_SEED3`
4. confidence-weight ablation:
   - `E13_MATH_SHEPHERD_TRUST_UNWEIGHTED_SEED3`
5. anti-saturation probe:
   - `E14_MATH_SHEPHERD_TRUST_ANTISAT_SEED3`
6. conservative robust candidate recipe:
   - `E15_MATH_SHEPHERD_TRUST_ROBUST_SEED3`

What it produces:
1. group-level summary:
   - `assets/artifacts/phase_e_logs/<RUN_PREFIX>/final_summary.md`
2. candidate report:
   - `assets/artifacts/phase_e_candidates/<RUN_PREFIX>_candidate/candidate_report.md`
3. recommended checkpoint:
   - one explicit `best_value_head.pt` path chosen by
     `scripts/phase_e_select_candidate.py`

Trust interpretation rule:
1. do not freeze a checkpoint by one lucky seed,
2. require:
   - strong held-out same-source metrics,
   - acceptable worst-seed behavior,
   - low seed variance,
   - non-trivial `ProcessBench GSM8K/Math` secondary metrics,
3. if no group passes the full gate, the selector still returns one
   **provisional** candidate so later stages can continue with explicit caveats.

How to interpret the outputs:
1. first look at held-out pair metrics inside `assets/artifacts/phase_e_runs/.../summary.md`
2. then inspect benchmark-native metrics under `assets/artifacts/phase_e_eval/.../summary.md`
3. finally compare cross-seed variance in `assets/artifacts/phase_e_logs/<RUN_PREFIX>/final_summary.md`

### 11.5 Phase E Multi-Source Math Mainline

This is the next official Phase E direction after the latest `E15` result.

Direct Stage A-D matrix:

```bash
ACTIVE_PHASE_E_MM_GROUP=MM2_MULTISOURCE_MATH_STAGE_ABCD_SEED3 \
RUN_PREFIX=phase_e_multisource_abcd_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_multisource_math_suite.sh
```

Stage E curriculum matrix:

```bash
ACTIVE_PHASE_E_MM_GROUP=MM3_MULTISOURCE_MATH_STAGEE_CURRICULUM_SEED3 \
RUN_PREFIX=phase_e_multisource_curriculum_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_multisource_math_suite.sh
```

Full overnight program:

```bash
ACTIVE_PHASE_E_MM_GROUP=MM4_MULTISOURCE_MATH_FULL_PROGRAM \
RUN_PREFIX=phase_e_multisource_full_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_multisource_math_suite.sh
```

What the wrapper now compares:
1. Stage A anchors:
   - `E20_STAGEA_MS_ANCHOR_SEED3`
   - `E21_STAGEA_RPRM_ANCHOR_SEED3`
   - `E22_STAGEA_PRMBENCH_PREVIEW_ANCHOR_SEED3`
   - `E23_STAGEA_PRM800K_CTRL_SEED3`
2. Stage B balanced two-source mixtures:
   - `E24_STAGEB_MS_RPRM_MIX_SEED3`
   - `E25_STAGEB_MS_PRMBENCH_MIX_SEED3`
   - `E26_STAGEB_RPRM_PRMBENCH_MIX_SEED3`
3. Stage C main three-source mixture:
   - `E27_STAGEC_MS_RPRM_PRMBENCH_MIX_SEED3`
4. Stage D weak-source ablations:
   - `E28_STAGED_TRIMIX_PLUS_PRM800K_LOWWT_SEED3`
   - `E29_STAGED_MS_PLUS_PRM800K_LOWWT_SEED3`
5. Stage E curricula:
   - `CUR1_STAGEE_MS_TO_MSRPRM`
   - `CUR2_STAGEE_MS_TO_TRIMIX`
   - `CUR3_STAGEE_MS_TO_MSRPRM_TO_TRIMIX`

What each family is supposed to answer:
1. Stage A:
   - which single source deserves anchor status
2. Stage B:
   - whether two complementary sources already beat the anchors
3. Stage C:
   - whether the main same-family tri-mix is the best direct recipe
4. Stage D:
   - whether `PRM800K` helps only under explicit low-weight control, or should
     be dropped from the mainline
5. Stage E:
   - whether warm-started staged training is better than one-shot mixing

### 11.6 Phase E Intradataset ACC90 Branch

This branch deliberately ignores:
1. cross-dataset transfer,
2. benchmark-native generalization,
3. StrategyQA bridge pressure.

It asks only:

> Can one dataset by itself support a value head that reaches `>90%` held-out pair accuracy?

Main wrapper:

```bash
ACTIVE_PHASE_E_INTRADATASET_GROUP=I5_ALL_ACC90_MATRIX \
RUN_PREFIX=phase_e_all_acc90_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_intradataset_suite.sh
```

Per-dataset wrappers:

```bash
ACTIVE_PHASE_E_INTRADATASET_GROUP=I2_MS_ACC90_MATRIX \
RUN_PREFIX=phase_e_ms_acc90_full_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_intradataset_suite.sh

ACTIVE_PHASE_E_INTRADATASET_GROUP=I3_PRMBENCH_ACC90_MATRIX \
RUN_PREFIX=phase_e_prmbench_acc90_full_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_e_intradataset_suite.sh

ACTIVE_PHASE_E_INTRADATASET_GROUP=I4_RPRM_ACC90_MATRIX \
RUN_PREFIX=phase_e_rprm_acc90_full_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_e_intradataset_suite.sh
```

Comparable full-matrix recommendation:
1. keep each dataset's default recipe family intact,
2. do **not** reuse smoke overrides such as:
   - `SEEDS_OVERRIDE=42`
   - reduced pair caps
   - reduced epochs

Focused Math-Shepherd ACC95 push matrix:

```bash
CUDA_VISIBLE_DEVICES=1 \
ACTIVE_PHASE_E_INTRADATASET_GROUP=I6_MS_ACC95_PUSH_MATRIX \
RUN_PREFIX=phase_e_ms_acc95_push_$(date +%m%d_%H%M) \
bash scripts/run_phase_e_intradataset_suite.sh
```

Current top-candidate RL-readiness audit:

```bash
CUDA_VISIBLE_DEVICES=2 \
ACTIVE_PHASE_E_RL_GROUP=RR4_COMPARE_CURRENT_TOPS \
RUN_PREFIX=phase_e_rl_readiness_$(date +%m%d_%H%M) \
bash scripts/run_phase_e_rl_readiness_suite.sh
```

Important clarification after the stricter transfer diagnosis:
1. the older `RR4` audit is still useful as a quick coarse screen,
2. but it is not a sufficient RL gate.

Why:
1. its heuristic can mark `PRMBench E46` as `provisionally_rl_ready`,
2. yet the stricter structure-aware diagnosis shows that no current checkpoint
   satisfies all of:
   - same-family decision strength,
   - benchmark local-error discrimination,
   - and terminal completion safety.

New strict transfer diagnosis script:

```bash
python -u scripts/phase_e_diagnose_transfer.py \
  --run-name phase_e_transfer_diag_manual \
  --audit-spec 'ms_e68|math_shepherd|<samefamily_dir>|<pb_gsm_dir>|<pb_math_dir>' \
  --audit-spec 'prm_e46|prmbench_preview|<samefamily_dir>|<pb_gsm_dir>|<pb_math_dir>'
```

What this stricter diagnosis adds:
1. benchmark margin-collapse ratio vs same-family score gap,
2. all-correct final-prefix top1 / mean-gap,
3. local first-bad edge behavior,
4. length/support drift warnings.

Current strict diagnosis snapshot:
1. `Math-Shepherd E68`
   - same-family utility is excellent
   - but terminal completion is catastrophically under-valued on ProcessBench
   - and Math-side length drift remains large
2. `PRMBench E46`
   - best current benchmark local discrimination baseline
   - but `all-correct final-prefix top1` is still only:
     - `0.2332` on `ProcessBench GSM8K`
     - `0.1970` on `ProcessBench Math`
3. `PRMBench E78`
   - fixes terminal completion strongly
   - but collapses local good-vs-bad benchmark ranking
4. strict conclusion:
   - no current checkpoint is RL-ready

Experimental terminal-anchor repair path:

```bash
python -u scripts/phase_e_prepare_prmbench_terminal_anchor_pairs.py \
  --run-name phase_e_prmbench_terminal_anchor_$(date +%m%d_%H%M) \
  --seed 42
```

This research-only artifact keeps original `PRMBench` local error-step pairs and
adds one synthetic terminal anchor per source row:
1. full correct process,
2. versus a shorter safe prefix near the first modified error step.

Smoke repair command used in the latest diagnosis:

```bash
CUDA_VISIBLE_DEVICES=3 \
python -u scripts/phase_e_train_value.py \
  --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_prmbench_terminal_anchor_0311_0015__95e591d28f80/train_pairs.jsonl \
  --eval-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_prmbench_terminal_anchor_0311_0015__95e591d28f80/validation_pairs.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name phase_e_prmbench_terminal_anchor_joint_logit_smoke_0311_0025 \
  --output-root assets/artifacts/phase_e_runs \
  --max-train-samples 3000 \
  --max-eval-samples 600 \
  --objective-mode joint \
  --learning-rate 3e-5 \
  --num-train-epochs 10 \
  --per-device-train-batch-size 128 \
  --per-device-eval-batch-size 128 \
  --max-length 1024 \
  --lambda-ranking 1.0 \
  --lambda-bce 1.0 \
  --ranking-margin 0.02 \
  --ranking-target-space logit \
  --pair-weight-mode none \
  --source-balance none \
  --permutation-mode stable_hash \
  --checkpoint-selection-metric ranking_score \
  --head-architecture mlp \
  --head-mlp-hidden-size 1024 \
  --head-dropout-prob 0.05 \
  --head-init-std 0.02 \
  --head-activation gelu \
  --feature-cache-root assets/artifacts/phase_e_feature_cache \
  --feature-cache-mode read_write
```

Observed repair-smoke tradeoff:
1. same-family prompt-pool top1 remains positive:
   - `0.9280`
2. terminal completion becomes very strong on ProcessBench:
   - `terminal_top1 = 0.9585 / 0.9433` (`GSM8K / Math`)
3. but local good-vs-bad ranking collapses badly:
   - `auc = 0.4778 / 0.4691`

Interpretation:
1. terminal undervaluation is a real failure mode,
2. but naive terminal-anchor augmentation over-corrects and damages local
   process discrimination,
3. so this repair direction is scientifically useful,
4. yet still not sufficient for RL promotion.

## 2026-03-10 Intradataset Candidate Selector Fix

Bug:
1. `run_phase_e_intradataset_suite.sh` passed repeated
   `--suite-log-dirs ...` flags,
2. but `phase_e_select_intradataset_candidate.py` originally parsed only the
   last repeated occurrence under its argparse contract.

Effect:
1. the old smoke candidate report incorrectly selected
   `E48_RPRM_ACC90_MLP_RANK_SEED3`.

Fix:
1. selector now accepts repeated groups,
2. flattens them in-order,
3. deduplicates repeated directories.

Verification:
1. unit test:
   - `PYTHONPATH=src pytest -q tests/unit/test_phase_e_select_intradataset_candidate.py`
2. replay on the existing smoke suite:
   - selected group is now:
     - `E41_MS_ACC90_MLP_RANK_SEED3`
   - report:
     - `assets/artifacts/phase_e_candidates/phase_e_top3_acc90_0310_1808_candidate_fixcheck/candidate_report.json`

## 2026-03-11 ProcessBench Transfer Repair Plumbing Fixes

Three bugs were fixed before re-running the next ProcessBench transfer sweep:

1. benchmark subset sampling:
   - `src/ours/phase_e/benchmark_eval.py`
   - `ProcessBench` smoke subsets are now stratified across:
     - `all-correct`
     - `error`
2. failure-analysis subset matching:
   - `scripts/phase_e_analyze_processbench_failures.py`
   - the script now filters the benchmark by `scored_rows.jsonl` example ids
     before rebuilding prefix rows
3. terminal-anchor starvation:
   - `src/ours/phase_d/external_pairs_adapters.py`
   - `src/ours/phase_e/pairs.py`
   - `scripts/phase_e_prepare_pairs.py`
   - `scripts/run_phase_e_processbench_transfer_suite.sh`

Critical diagnosis:
1. in the current `Math-Shepherd` mirror, the first all-positive trajectory
   appears only at source row `121569`
2. therefore `max_pairs_per_source=20000` plus naive stream-head loading makes
   `terminal_completion_anchor = 0` inevitable
3. this means an old terminal-anchor config could look correct in the shell
   while still training on zero terminal pairs

Verified fixed artifact:
1. artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_processbench_terminal_focus_0311_e83_ms_processbench_transfer_terminal_seed42_e83_ms_processbench_transfer_terminal_seed42_sharedsplit_s42_pairs__8b75a88516bc`
2. after the fix, its train split contains:
   - `local_first_bad_edge = 3595`
   - `terminal_completion_anchor = 3603`

Recommended reproduction commands:

```bash
PYTHONPATH=src pytest -q \
  tests/unit/test_phase_d_external_pairs.py \
  tests/unit/test_phase_e_pairs.py \
  tests/unit/test_phase_e_benchmark_eval.py \
  tests/unit/test_phase_e_runtime.py
```

```bash
CUDA_VISIBLE_DEVICES=1 \
MAX_GPU_MEMORY_GIB=48 \
MAX_CPU_MEMORY_GIB=96 \
ACTIVE_PHASE_E_TRANSFER_GROUP=PT1_PROCESSBENCH_TRANSFER_SMOKE \
RUN_PREFIX=phase_e_processbench_transfer_smoke_fix_$(date +%m%d_%H%M) \
SUITE_BENCH_MAX_SAMPLES=96 \
SUITE_TRAIN_BATCH_SIZE=64 \
SUITE_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_e_processbench_transfer_suite.sh
```

Terminal-focused subset used for the current repair check:

```bash
CUDA_VISIBLE_DEVICES=1 \
MAX_GPU_MEMORY_GIB=48 \
MAX_CPU_MEMORY_GIB=96 \
ACTIVE_PHASE_E_TRANSFER_GROUP=PT1_PROCESSBENCH_TRANSFER_SMOKE \
PHASE_E_TRANSFER_DIRECT_GROUPS_OVERRIDE='E83_MS_PROCESSBENCH_TRANSFER_TERMINAL_SEED42 E84_MS_PROCESSBENCH_TRANSFER_FANOUT_TERMINAL_SEED42 E85_MS_PRMBENCH_TRANSFER_MIX_TERMINAL_SEED42' \
RUN_PREFIX=phase_e_processbench_terminal_focus_$(date +%m%d_%H%M) \
SUITE_BENCH_MAX_SAMPLES=96 \
SUITE_TRAIN_BATCH_SIZE=64 \
SUITE_EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_e_processbench_transfer_suite.sh
```

Current corrected outcome on the fixed `ProcessBench 96` subset:
1. `E83 terminal` is no longer a fake repair.
   - train artifact:
     - `3595 local_first_bad_edge`
     - `3603 terminal_completion_anchor`
   - benchmark effect:
     - `gsm8k all_correct_top1: 0.1087 -> 0.6304`
     - `math all_correct_top1: 0.0000 -> 0.5641`
     - but `gsm8k pair_acc: 0.6088 -> 0.3163`
     - and `math pair_acc: 0.4558 -> 0.3273`
2. `E84 fanout + terminal` keeps the same terminal improvement and improves
   geometric alignment over `E83`,
   but still remains far below the baseline ranking surface:
   - `gsm8k pair_acc = 0.3299`
   - `math pair_acc = 0.3233`

Current engineering conclusion:
1. terminal supervision is a real missing slice,
2. but the naive `50/50` terminal-heavy repair is over-weighted,
3. so the next recipe should *reduce* terminal-anchor mass instead of treating
   it as half of the training pool.

## 2026-03-11 Low-terminal RL-facing repair experiment

New knob added to `Phase E` pair prep:
1. `--step-label-terminal-anchor-fraction`
2. intent:
   - keep terminal anchors as a bounded auxiliary instead of a co-equal half of
     the training pool

Main new repair run:
```bash
CUDA_VISIBLE_DEVICES=3 \
MAX_GPU_MEMORY_GIB=48 \
MAX_CPU_MEMORY_GIB=96 \
ACTIVE_PHASE_E_GROUP=E87_MS_PROCESSBENCH_TRANSFER_FANOUT_TERMINAL10_CONFWT_SEED42 \
RUN_PREFIX=phase_e_processbench_rlrepair_$(date +%m%d_%H%M) \
BENCH_MAX_SAMPLES=96 \
TRAIN_EPOCHS=4 \
TRAIN_BATCH_SIZE=64 \
EVAL_BATCH_SIZE=64 \
bash scripts/run_phase_e_suite.sh
```

Observed result summary:
1. train artifact semantics:
   - `6487 fanout`
   - `735 terminal`
2. held-out fit:
   - `pair_acc = 0.8823`
   - `auc = 0.8647`
3. benchmark:
   - `ProcessBench GSM8K pair_acc = 0.4932`
   - `ProcessBench Math pair_acc = 0.4217`
4. terminal slice:
   - `gsm8k all_correct_top1 = 0.3696`
   - `math all_correct_top1 = 0.2051`

RL-facing trust audit command used:
```bash
CUDA_VISIBLE_DEVICES=3 \
python -u scripts/phase_e_eval_samefamily_trust.py \
  --value-run-dir assets/artifacts/phase_e_runs/phase_e_processbench_rlrepair_0311_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_s42_value_20260311T032957Z \
  --run-name phase_e_rlready_e87_samefamily_0311 \
  --output-root assets/artifacts/phase_e_samefamily_eval \
  --checkpoint-name best \
  --batch-size 64 \
  --dtype bfloat16 \
  --device-map auto \
  --max-gpu-memory-gib 48 \
  --max-cpu-memory-gib 96 \
  --feature-cache-root assets/artifacts/phase_e_feature_cache \
  --feature-cache-mode read_write \
  --edge-weight-mode confidence \
  --require-cuda
```

Strict transfer diagnosis command used:
```bash
python -u scripts/phase_e_diagnose_transfer.py \
  --run-name phase_e_transfer_diag_e87_0311 \
  --audit-spec 'e87_ms_rlrepair|math_shepherd|assets/artifacts/phase_e_samefamily_eval/phase_e_rlready_e87_samefamily_0311_20260311T040021Z|assets/artifacts/phase_e_eval/phase_e_processbench_rlrepair_0311_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_s42_processbench_gsm8k_20260311T033946Z|assets/artifacts/phase_e_eval/phase_e_processbench_rlrepair_0311_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_s42_processbench_math_20260311T033955Z'
```

Current conclusion:
1. `10%` terminal is better than `50%`
2. but it is still not enough for RL-ready promotion
3. next step should target training strategy:
   - semantics-aware weighting,
   - staged/curriculum repair,
   - and trust-gated checkpoint selection

## 2026-03-11 Internet-Guided ProcessBench Redesign Suite

This round added a more explicit research pipeline for the current
`ProcessBench` transfer blocker.

Why this changed:
1. the latest paper/community check shows that current PRMs usually fail not
   because same-source learning is impossible,
   but because transfer requires support beyond pure local first-bad pairs.
2. at the same time, the literature does **not** support replacing the clean
   local signal with heavy terminal/global supervision.
3. therefore the new design is:
   - keep a strong local core,
   - add broader support conservatively,
   - and weight those auxiliary families explicitly.

Primary external references used in this redesign:
1. `PRM800K / Let's Verify Step by Step`
   - https://arxiv.org/abs/2305.20050
2. `OmegaPRM`
   - https://arxiv.org/abs/2406.06592
3. `ProcessBench`
   - https://arxiv.org/abs/2412.06559
   - https://github.com/QwenLM/ProcessBench
4. `PRMBench`
   - https://arxiv.org/abs/2501.03124
5. `The Lessons of Developing PRMs`
   - https://arxiv.org/abs/2501.07301
6. `PathFinder-PRM`
   - https://arxiv.org/abs/2501.11690
   - https://github.com/Gen-Verse/PathFinder-PRM

New code landed:
1. curated profile builder:
   - `scripts/phase_e_curate_processbench_transfer_pairs.py`
2. trainer weight modes:
   - `semantic`
   - `confidence_semantic`
3. new one-click suite:
   - `scripts/run_phase_e_processbench_research_suite.sh`

Executed smoke command:

```bash
CUDA_VISIBLE_DEVICES=1 \
ACTIVE_PHASE_E_PB_RESEARCH_GROUP=PBR1_PROCESSBENCH_REDESIGN_SMOKE \
RUN_PREFIX=phase_e_processbench_research_v2 \
PHASE_E_PB_RESEARCH_CASES_OVERRIDE='pbr1_ms_align_mlp|one_shot|ms_align_v1|mlp|none pbr2_ms_align_gated|one_shot|ms_align_v1|gated_mlp|none pbr4_ms_curriculum_gated|curriculum|ms_align_v1|gated_mlp|none' \
TARGET_TOTAL_PAIRS=2048 \
BENCH_MAX_SAMPLES=64 \
TRAIN_BATCH_SIZE=64 \
EVAL_BATCH_SIZE=96 \
bash scripts/run_phase_e_processbench_research_suite.sh
```

Artifacts:
1. suite summary:
   - `assets/artifacts/phase_e_logs/phase_e_processbench_research_v2/final_summary.md`
2. per-case rows:
   - `assets/artifacts/phase_e_logs/phase_e_processbench_research_v2/research_results.jsonl`
3. transfer compare:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_processbench_research_v2_gsm_compare_20260311T044545Z/summary.md`
   - `assets/artifacts/phase_e_transfer_compare/phase_e_processbench_research_v2_math_compare_20260311T044545Z/summary.md`
4. RL-promotion diagnosis:
   - `assets/artifacts/phase_e_rl_promotion_diag/phase_e_processbench_research_v2_rl_promotion_20260311T044546Z/summary.md`

Observed result summary:
1. `pbr2_ms_align_gated` is the best new case.
2. compared with `ref_e87`, it improves:
   - same-family `top1: 0.6597 -> 0.8947`
   - same-family `local_first_bad: 0.2914 -> 0.9231`
   - `ProcessBench Math auc: 0.4467 -> 0.5055`
   - terminal-completion `top1` on both benchmark splits
3. it still does **not** clear RL promotion because:
   - `gsm first_edge = 0.5600`
   - `math first_edge = 0.5357`
   - `pb_min_laterbad = 0.4792`
4. `pbr4_ms_curriculum_gated` improved terminal slices further,
   but weakened same-family trust and is not the preferred mainline.
5. current best next-step reading:
   - keep `ms_align_v1`
   - keep `confidence_semantic`
   - keep `gated_mlp`
   - next repair should target `later-bad` transfer while preserving
     `first_edge`

Recommended reruns:

```bash
CUDA_VISIBLE_DEVICES=1 \
TARGET_TOTAL_PAIRS=2048 \
BENCH_MAX_SAMPLES=64 \
TRAIN_BATCH_SIZE=64 \
EVAL_BATCH_SIZE=96 \
bash scripts/run_phase_e_processbench_research_suite.sh
```

```bash
CUDA_VISIBLE_DEVICES=1 \
ACTIVE_PHASE_E_PB_RESEARCH_GROUP=PBR1_PROCESSBENCH_REDESIGN_SMOKE \
RUN_PREFIX=phase_e_processbench_research_v2 \
PHASE_E_PB_RESEARCH_CASES_OVERRIDE='pbr2_ms_align_gated|one_shot|ms_align_v1|gated_mlp|none' \
TARGET_TOTAL_PAIRS=4096 \
BENCH_MAX_SAMPLES=96 \
TRAIN_BATCH_SIZE=64 \
EVAL_BATCH_SIZE=96 \
bash scripts/run_phase_e_processbench_research_suite.sh
```

## 2026-03-11 ProcessBench Hybrid Artifact Result: Tiny Terminal Support Already Over-Corrects

This round tested a stricter benchmark-oriented hybrid recipe based on the
current literature reading:

1. keep `PRMBench` explicit local error-step pairs as the anchor,
2. add only a *very small* terminal-completion auxiliary,
3. compare whether a stronger head (`gated_mlp`) can preserve local ranking
   better than a standard `mlp`.

### New code and infrastructure

1. `scripts/phase_e_mix_pair_artifacts.py`
   - now supports weighted input specs:
     - `LABEL=ARTIFACT_DIR:TRAIN_CAP:VAL_CAP:MIX_WEIGHT`
2. `scripts/run_phase_e_processbench_hybrid_suite.sh`
   - new ProcessBench-oriented hybrid curation wrapper
3. wrapper hardening:
   - fixed baseline parsing for `run::gsm_eval::math_eval`
   - removed shell backtick-substitution hazards
   - separated helper logs from helper outputs
   - switched helper outputs to global result variables to avoid command-substitution state loss

### Executed artifact

Hybrid pair artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_pbhybrid_ph1_0311_1230_pairs__d6fb5a3ec28c`

Artifact caps:

1. `prm_local`
   - train `3072`
   - val `384`
2. `prm_terminal`
   - train `512`
   - val `64`

But the important semantic fact is:

1. failure analysis later showed that only
   - `132 / 3584 = 3.68%`
   of the final train rows are actually `terminal_completion_anchor`.

So this was already a conservative terminal repair, not a heavy one.

### Same-source held-out fit

1. `mlp`
   - run:
     - `assets/artifacts/phase_e_runs/phase_e_pbhybrid_ph1_0311_1230_ph1_prm_local_ta15_arch_sweep_smoke_mlp_20260311T043055Z`
   - held-out:
     - `pair_acc=0.919643`
     - `auc=0.892518`
2. `gated_mlp`
   - run:
     - `assets/artifacts/phase_e_runs/phase_e_pbhybrid_ph1_0311_1230_ph1_prm_local_ta15_arch_sweep_smoke_gated_mlp_20260311T045211Z`
   - held-out:
     - `pair_acc=0.912946`
     - `auc=0.871079`

Reading:

1. same-source learning is still easy,
2. so this is not another "the head cannot learn at all" failure.

### ProcessBench transfer comparison

Comparison tables:

1. GSM8K:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_pbhybrid_ph1_0311_1230_processbench_gsm8k_compare_20260311T045320Z/summary.md`
2. Math:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_pbhybrid_ph1_0311_1230_processbench_math_compare_20260311T045332Z/summary.md`

#### GSM8K

1. baseline `E46`
   - `auc=0.6264`
   - `first_edge=0.6706`
   - `terminal_top1=0.2332`
2. hybrid `mlp`
   - `auc=0.5543`
   - `first_edge=0.4906`
   - `terminal_top1=0.8548`
3. hybrid `gated_mlp`
   - `auc=0.5011`
   - `first_edge=0.5472`
   - `terminal_top1=0.8710`

#### Math

1. baseline `E46`
   - `auc=0.6053`
   - `first_edge=0.6096`
   - `terminal_top1=0.1970`
2. hybrid `mlp`
   - `auc=0.4931`
   - `first_edge=0.4921`
   - `terminal_top1=0.7115`
3. hybrid `gated_mlp`
   - `auc=0.5162`
   - `first_edge=0.4921`
   - `terminal_top1=0.8654`

### Failure-pattern diagnosis

Detailed failure analyses:

1. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_mlp_gsm_diag_20260311T045406Z/summary.md`
2. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_gated_gsm_diag_20260311T045406Z/summary.md`
3. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_mlp_math_diag_20260311T045406Z/summary.md`
4. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_gated_math_diag_20260311T045406Z/summary.md`

Most important facts:

1. `ProcessBench` still has a large `all_correct` block:
   - GSM8K `0.4844`
   - Math `0.4062`
2. the hybrid train set is still overwhelmingly local:
   - `local_modified_process_error_step = 3452`
   - `terminal_completion_anchor = 132`
3. even so, the small terminal anchor support is already enough to
   over-correct terminal behavior:
   - GSM8K `terminal_top1` jumps to `0.85+`
   - Math `terminal_top1` jumps to `0.71~0.87`
4. but local discrimination drops sharply:
   - `first_error_edge_accuracy`
   - `pair_auc_good_vs_bad`
   both regress versus `E46`

Interpretation:

1. naive `local + terminal` mixing is still too blunt,
2. architecture helps only marginally,
3. the main bottleneck is still supervision geometry, not head capacity.

### Updated conclusion

This round rules out one tempting but incorrect next step:

1. we should **not** assume that adding "just a little" terminal support to a
   strong local source will automatically fix `ProcessBench`.
2. we should **not** interpret strong same-source hybrid fit as benchmark
   progress.

What this does support:

1. keep `PRMBench local` as the benchmark-aligned local anchor,
2. reduce terminal ratio even further,
3. or move to a staged / two-objective recipe where terminal repair is
   constrained instead of naively merged into the same ranking pool,
4. and use benchmark-aware selection instead of held-out pair fit alone.

## 2026-03-11 Tiny-Terminal Follow-up: Ratio Matters, But Ratio Alone Does Not Fix ProcessBench

After the first hybrid pilot, the next critical question was:

1. did the new hybrid fail only because terminal support was still too large,
2. or is the deeper problem that terminal supervision should not be naively merged into the same ranking pool at all?

### Tiny-terminal artifact

Artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_pbhybrid_tinyterm_0311__dd5acae29427`

Configured caps:

1. `prm_local`
   - train `3072`
   - val `384`
2. `prm_terminal`
   - train `64`
   - val `8`
   - mix weight `0.10`

Important failure-analysis fact:

1. the resulting artifact contains only:
   - `17 / 3136 = 0.54%`
   terminal-anchor semantics.

So this is not just "smaller than before"; it is genuinely tiny.

### Same-source held-out fit

Run:

1. `assets/artifacts/phase_e_runs/phase_e_pbhybrid_tinyterm_0311_mlp_20260311T050119Z`

Held-out:

1. `pair_acc=0.918367`
2. `auc=0.890293`

This matters because it rules out the easy but wrong story that "the tiny artifact became too weak to learn."

### ProcessBench comparison

Compare tables:

1. GSM8K:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_pbhybrid_terminal_ratio_gsm_compare_20260311T051627Z/summary.md`
2. Math:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_pbhybrid_terminal_ratio_math_compare_20260311T051627Z/summary.md`

#### GSM8K

1. `E46`
   - `auc=0.6264`
   - `first_edge=0.6706`
   - `terminal_top1=0.2332`
2. `hybrid_ta15`
   - `auc=0.5543`
   - `first_edge=0.4906`
   - `terminal_top1=0.8548`
3. `hybrid_tiny`
   - `auc=0.5654`
   - `first_edge=0.5094`
   - `terminal_top1=0.7742`

#### Math

1. `E46`
   - `auc=0.6053`
   - `first_edge=0.6096`
   - `terminal_top1=0.1970`
2. `hybrid_ta15`
   - `auc=0.4931`
   - `first_edge=0.4921`
   - `terminal_top1=0.7115`
3. `hybrid_tiny`
   - `auc=0.4948`
   - `first_edge=0.5238`
   - `terminal_top1=0.6538`

### Failure analysis

Failure-analysis artifact:

1. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_tinyterm_math_diag_20260311T051628Z/summary.md`

Most important interpretation:

1. shrinking terminal supervision from `3.68%` to `0.54%` clearly softens the over-correction,
2. terminal slices fall from the extreme `0.71+` range to `0.65` on Math,
3. `first_error_edge_accuracy` recovers slightly,
4. but benchmark-local ranking still does not return to `E46`.

### Updated scientific reading

This is one of the most informative negative results so far.

It says:

1. terminal ratio is real and matters,
2. but the current problem is not *only* a ratio problem,
3. therefore pure terminal-ratio sweep is unlikely to be the true mainline fix.

The stronger updated conclusion is:

1. terminal supervision likely needs a different optimization role,
2. such as:
   - staged training,
   - two-objective training,
   - or benchmark-aware checkpoint selection,
3. instead of further naive local+terminal blending.

## 2026-03-11 Judge Integration Verdict After Network Check

We re-checked the `LLM-as-a-judge` design against the most relevant literature and community guidance, then validated it with local pairwise judge experiments.

What now appears well-supported:

1. `PRMBench_Preview`-style pairwise data is judge-friendly.
2. `selected relabel` is viable only on narrow, high-value slices.
3. `disagreement mining` is a realistic use of local judges.
4. `benchmark-side adjudication` is safer than training-side bulk filtering.

What is not supported by current evidence:

1. bulk judge filtering over `Math-Shepherd local_first_bad_edge`
2. full-dataset relabel using the current local judge stack
3. treating the local judge as a strong verifier replacement

Key local evidence:

1. `Qwen2.5-Math-7B-Instruct` on `PRMBench_Preview train64`
   - `label_preserving_keep_rate = 0.0469`
2. `Qwen2.5-Math-7B-Instruct` on `Math-Shepherd train64`
   - `label_preserving_keep_rate = 0.0000`
3. Pairwise + swap-debias is directionally better than the earlier pointwise strict-JSON judge path, but still not strong enough for bulk filtering.

Operational decision:

1. keep `selected relabel`
2. keep `disagreement mining`
3. keep `benchmark-side adjudication`
4. do not promote `judge-driven bulk pair filtering` into the Phase E mainline
