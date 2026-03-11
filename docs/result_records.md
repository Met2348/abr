# Result Records

Last updated: 2026-03-11 17:00:00 +0800

## 0AAAB. NDSBH Suite Results — New Dataset Source Ablation (2026-03-11)

### Setup
- Suite: `NDSBH_ALL_SMOKE` via `scripts/run_phase_e_nds_best_hparams_suite.sh`
- Target pairs: 4096 | Epochs: 10 | Batch: 128 | GPU: 2 (single GPU, avoids multi-device bug)
- **Correct hyperparameters**: `--ranking-target-space score` + `--pair-weight-mode none` + `--checkpoint-selection-metric pair_acc`
- `--nonfinite-feature-policy drop` (handles NaN from RLHFlow data)

### Results

| Case | Data Source | Train Pairs | Same-src AUC | GSM8K AUC | Math AUC | GSM8K 1st-edge | Math 1st-edge |
|------|------------|------------|-------------|-----------|----------|----------------|---------------|
| NDS1 | RLHFlow-Deepseek (LLM-judge) | 3037 | 0.743 | 0.571 | 0.552 | 0.679 | 0.609 |
| **NDS2** | **Math-Step-DPO (fork-point)** | **3705** | **0.723** | **0.655** | **0.712** | **0.698** | **0.656** |
| NDS3 | Math-Shepherd ms_align_v1 | ~3000 | 0.836 | 0.477 | 0.470 | 0.509 | 0.461 |
| NDS4 | Mixed (MS+RLHFlow+Math-Step-DPO) | 3093 | 0.744 | 0.605 | 0.647 | 0.594 | 0.672 |
| ms_e43 (ref) | Math-Shepherd ms_acc90 | 14290 | 0.945 | 0.625 | 0.634 | 0.688 | 0.633 |

### Key Findings

1. **Math-Step-DPO (NDS2) achieves Math AUC=0.712 — new best for frozen head** with only 3705 pairs, beating ms_e43 (14290 pairs, 0.634). Fork-point sibling_branch pairs directly match ProcessBench's first-error-edge format.
2. **Math-Shepherd at small scale (NDS3) near-random**: same-source AUC=0.836 but ProcessBench Math=0.470. MS needs ~14K+ pairs for positive transfer (MC labels are noisy, need scale).
3. **Mixed data significantly helps**: NDS4 (MS+RLH+DPO) gives 0.647 vs MS alone (0.470) at same scale.
4. **Data quality hierarchy** (frozen head, ≤4K pairs): Math-Step-DPO > Mixed > RLHFlow > Math-Shepherd
5. **Correct hparams are critical**: Broken (logit+confidence_semantic+ranking_score) → AUC=0.39 (inverted); Correct (score+none+pair_acc) → full recovery.

### Updated Frozen-Head Leaderboard (All Experiments)

| config | Data | Pairs | GSM8K AUC | Math AUC | Status |
|---|---|---|---|---|---|
| **NDS2: Math-Step-DPO + mlp (correct hparams)** | xinlai_math_step_dpo | 3705 | **0.655** | **0.712** | **NEW BEST** |
| NDS4: Mixed + mlp (correct hparams) | ms+rlhflow+dpo | 3093 | 0.605 | 0.647 | OK |
| ms_e43: ms_acc90 + mlp (correct hparams) | math_shepherd | 14290 | 0.625 | 0.634 | OK |
| ms_prm_align_v1 + gated_mlp (broken hparams) | ms+prmbench | 4096 | 0.549 | 0.621 | OK (gated avoids inversion) |
| prm_e46: prmbench + mlp | prmbench | — | 0.626 | 0.605 | OK |
| NDS1: RLHFlow + mlp (correct hparams) | rlhflow | 3037 | 0.571 | 0.552 | OK |
| ms_align_v1 + mlp (s1, broken hparams) | math_shepherd | 16384 | 0.616 | 0.476 | Unstable |
| ms_align_v1 + mlp (s42, broken hparams) | math_shepherd | 16384 | 0.442 | 0.442 | Inverted |
| NDS3: MS + mlp (correct hparams, small) | math_shepherd | ~3000 | 0.477 | 0.470 | Near-random |

### Recommended Next Steps
1. **NDS2-scale-up**: Math-Step-DPO at larger pair count (14K+ pairs, if augmentable) with 10 epochs. Expect further improvement.
2. **NDS2+LoRA**: Math-Step-DPO data with LoRA backbone — may push Math AUC past 0.75+.
3. **PBR5 with correct hparams**: Re-run PBR5 (ms_prm_align_v1 + gated_mlp) with score+none+pair_acc. Current PBR4b (0.621) used broken hparams; correct hparams may improve further.
4. **Investigate NDS3 inversion mechanism**: Why does MS at small scale invert? Understanding this helps explain the data-quality gap.

---

## 0AAAA. PBR2/PBR3/PBR4 Experiment Results — Critical Bug Fix + Full Results (2026-03-11)

### Background

A critical cross-device bug was found and fixed in `src/ours/phase_b/value_head.py:encode_text_features()`.
When `device_map="auto"` split the backbone across multiple GPUs, `attention_mask` (on cuda:0) and `outputs.hidden_states[-1]` (on the last-layer GPU) were on different devices. `torch.where` silently produced all-zero results. The fix: `attn_mask = inputs["attention_mask"].to(last_hidden.device)`.

**All PBR2 seed-42 results from the morning run (06:00 UTC) are INVALID (zero features).**
**Results below are from the corrected run (08:00+ UTC) on single GPUs.**

### PBR2 Full-Scale ms_align_v1+mlp (16384 pairs, 3 seeds)

Groups: `PBR2_FULL_MIXED_MLP_SEED3`

| seed | heldout_pair | heldout_auc | SF_top1 | SF_local | GSM8K_AUC | MATH_AUC | GSM8K_fe | MATH_fe |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.8471 | 0.8332 | 0.867 | 0.947 | 0.442 | 0.442 | 0.610 | 0.638 |
| 1  | 0.8456 | 0.8305 | 0.864 | 0.935 | **0.616** | 0.476 | 0.659 | 0.553 |
| 7  | 0.8484 | 0.8272 | 0.846 | 0.938 | 0.486 | **0.543** | 0.512 | 0.447 |

**Median**: GSM8K=0.486, MATH=0.476. AUC range: GSM8K=0.174, MATH=0.101. **FAILS** 0.58/0.52 targets. **UNSTABLE** (range >> 0.05).

**Key finding**: Extreme inter-seed variance — score inversion flips axis per seed. Seed 42: both axes inverted. Seed 1: GSM8K OK, MATH inverted. Seed 7: GSM8K inverted, MATH OK. Same-source accuracy stable (84.5-84.8%). Plain MLP head randomly latches onto length-bias vs step-correctness feature depending on initialization.

### PBR3 Later-Bad Branch (ms_laterbad_v1+mlp, 16384 pairs, 3 seeds)

Groups: `PBR3_LATER_BAD_BRANCH_SEED3`

| seed | heldout_pair | heldout_auc | SF_top1 | SF_local | GSM8K_AUC | MATH_AUC | GSM8K_fe | MATH_fe |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.8592 | 0.8393 | 0.860 | 0.947 | 0.425 | 0.436 | 0.561 | 0.617 |
| 1  | 0.8481 | 0.8352 | 0.873 | 0.946 | **0.619** | 0.487 | 0.707 | 0.596 |
| 7  | 0.8511 | 0.8330 | 0.854 | 0.952 | 0.515 | **0.548** | 0.537 | 0.468 |

**Median**: GSM8K=0.515, MATH=0.487. AUC range: GSM8K=0.194, MATH=0.112. **UNSTABLE** (variance even worse than PBR2). **FAILS** targets.

**Key finding**: ms_laterbad_v1 provides marginal improvement (+0.029 GSM8K median, +0.011 MATH median) over ms_align_v1 but does NOT fix instability or score inversion with plain MLP head. Targeted later-bad pairs insufficient for stable cross-distribution transfer.

### PBR4 PRMBench Auxiliary + Later-Bad Follow-Up Smoke (4096 pairs)

Groups: `PBR4_PRM_AND_LATERBAD_FOLLOWUP_SMOKE`

| case | profile | head | heldout_pair_acc | heldout_auc | SF_top1 | SF_local | GSM8K_AUC | MATH_AUC | GSM8K_inv? | MATH_inv? |
|---|---|---|---|---|---|---|---|---|---|---|
| pbr4a | ms_prm_align_v1 | mlp | 0.8556 | 0.8419 | 0.856 | 0.925 | 0.490 | 0.458 | **INV** | **INV** |
| pbr4b | ms_prm_align_v1 | gated_mlp | 0.8732 | 0.8833 | 0.872 | 0.940 | **0.549** | **0.621** | **OK** | **OK** |
| pbr4c | ms_laterbad_v1  | gated_mlp | 0.8889 | 0.8780 | 0.884 | 0.941 | 0.472 | 0.598 | **INV** | OK |

**KEY FINDING — gated_mlp breaks score inversion on ProcessBench MATH:**
- `ms_prm_align_v1 + gated_mlp` (pbr4b) achieves **MATH AUC=0.621**, **GSM8K AUC=0.549** — both non-inverted.
- `ms_laterbad_v1 + gated_mlp` (pbr4c) achieves **MATH AUC=0.598**, GSM8K AUC=0.472 (still inverted).
- `ms_prm_align_v1 + mlp` (pbr4a) remains fully inverted — the head architecture is decisive, not just the data.
- `gated_mlp` consistently improves MATH AUC by ~0.15+ over `mlp` with the same profile.
- pbr4c achieves **best samefamily** top1=0.884 of all experiments.

**Interpretation**: The gating mechanism (output = gate × main_path) separates length-bias from step-correctness signal better than plain MLP. PRMBench auxiliary pairs (in ms_prm_align_v1) further help GSM8K, likely because PRMBench examples are GSM8K-domain.

### PBR Experiment Comparison Summary (frozen head baseline)

| config | GSM8K_AUC | MATH_AUC | SF_top1 | Status |
|---|---|---|---|---|
| ms_align_v1 + mlp (s42) | 0.442 | 0.442 | 0.867 | Both inverted |
| ms_align_v1 + mlp (s1)  | 0.616 | 0.476 | 0.864 | GSM8K OK, MATH inv — unstable |
| ms_align_v1 + mlp (s7)  | 0.486 | 0.543 | 0.846 | GSM8K inv, MATH OK — unstable |
| ms_laterbad_v1 + mlp (s42) | 0.425 | 0.436 | 0.860 | Both inverted |
| ms_laterbad_v1 + mlp (s1)  | 0.619 | 0.487 | 0.873 | GSM8K OK — unstable |
| ms_laterbad_v1 + mlp (s7)  | 0.515 | 0.548 | 0.854 | MATH OK — unstable |
| ms_prm_align_v1 + mlp      | 0.490 | 0.458 | 0.856 | Both inverted |
| **ms_prm_align_v1 + gated_mlp** | **0.549** | **0.621** | **0.872** | **Both OK — stable direction** |
| ms_laterbad_v1 + gated_mlp | 0.472 | 0.598 | 0.884 | GSM8K inv, MATH OK |
| ms_prm_align_v1 + mlp | 0.490 | 0.458 | 0.856 | Still inverted |
| **ms_prm_align_v1 + gated_mlp** | **0.549** | **0.621** | **0.872** | **Best overall** |
| ms_laterbad_v1 + gated_mlp | 0.472 | 0.598 | 0.884 | Best samefamily |

### Recommended Next Steps

1. **PBR5 (immediate)**: `ms_prm_align_v1 + gated_mlp` at full scale (16384 pairs), 3 seeds. Target: MATH AUC ≥ 0.65, GSM8K AUC ≥ 0.57.
2. **PBR5b**: `ms_laterbad_v1 + gated_mlp` at full scale, 3 seeds. Compare against PBR5.
3. **PBR6 (LoRA path)**: If PBR5 MATH AUC < 0.65, proceed with LoRA backbone unfreezing. Published PRMs (>75% on ProcessBench) all require backbone fine-tuning.
4. **Architecture Note**: `dual_head` is NOT yet useful without pair_semantics routing in training.py. Blocked pending routing implementation.



## 0ZZZZZ. Pairwise judge benchmark and label-preserving filter pilot (2026-03-11)

This round tested the judge usage pattern that current literature supports more
strongly than our earlier strict-JSON pointwise prefix audit:

1. pairwise comparison,
2. swap-debias (`A/B` plus `B/A`),
3. light output contracts,
4. and using judge as a conservative label-preserving filter rather than as a
   full replacement verifier.

Primary artifacts:
1. script:
   - `scripts/phase_e_pairwise_judge_benchmark.py`
2. compare summary:
   - `assets/artifacts/phase_e_pairwise_judge_compare/judge_pairwise_compare_20260311T084132Z/summary.md`
3. anchored held-out runs:
   - `assets/artifacts/phase_e_pairwise_judge/pairjudge_qwen_prmbench_anchor32_20260311T083726Z`
   - `assets/artifacts/phase_e_pairwise_judge/pairjudge_deepseek_prmbench_anchor16_20260311T083726Z`
   - `assets/artifacts/phase_e_pairwise_judge/pairjudge_qwen_ms_val32_anchor_20260311T083839Z`
4. anchored train-slice filter pilots:
   - `assets/artifacts/phase_e_pairwise_judge/pairjudge_qwen_prmbench_train64_anchor_20260311T083927Z`
   - `assets/artifacts/phase_e_pairwise_judge/pairjudge_qwen_ms_train64_anchor_20260311T083927Z`

### 0ZZZZZ.1 Key results

`Qwen2.5-Math-7B-Instruct` on held-out `PRMBench_Preview` (`32` pairs):
1. `both_parse_ok_rate = 0.3125`
2. `pair_acc_majority = 0.3438`
3. `swap_consistency_rate = 0.6000`
4. `label_preserving_keep_rate = 0.1250`

`Qwen2.5-Math-7B-Instruct` on held-out `Math-Shepherd local_first_bad_edge` (`32` pairs):
1. `both_parse_ok_rate = 0.2188`
2. `pair_acc_majority = 0.0625`
3. `swap_consistency_rate = 1.0000` on the parsable subset
4. `label_preserving_keep_rate = 0.0312`
5. `judge_contradiction_rate = 0.0625`

`Qwen2.5-Math-7B-Instruct` as train-slice filter:
1. `PRMBench_Preview train64`:
   - `label_preserving_keep_rate = 0.0469`
2. `Math-Shepherd train64`:
   - `label_preserving_keep_rate = 0.0000`

`DeepSeek-R1-Distill-Qwen-14B` on held-out `PRMBench_Preview` (`16` pairs):
1. `both_parse_ok_rate = 0.5625`
2. `pair_acc_majority = 0.0625`
3. `tie_rate = 0.8125`

### 0ZZZZZ.2 Interpretation

This round gives three concrete conclusions:

1. pairwise + swap-debias is a better judge setup than the earlier
   strict-JSON pointwise prefix-judge attempt;
2. but the current local judge stack is still too weak to serve as a robust
   bulk pair filter;
3. and the judge is much closer to usable on `PRMBench_Preview` than on
   `Math-Shepherd local_first_bad_edge`.

The most important dataset-level finding is:

1. `Math-Shepherd` local-first-bad pairs are not well matched to naive
   pairwise judge filtering,
2. because the shorter safe prefix is often judged worse than the longer prefix
   that already contains the first wrong step.

So even though the pair label is valid for value-head training, it is not a
natural fit for "which prefix is better so far?" judged by a small local LLM.

### 0ZZZZZ.3 Current recommendation

1. keep judge work in the role of:
   - disagreement mining,
   - selected relabeling,
   - or benchmark-side adjudication,
2. do not promote the current local judge to a bulk Phase E pair filter,
3. and if judge is used for pairwise tasks, prefer sources closer to
   `PRMBench_Preview`-style explicit modified-process pairs over
   `Math-Shepherd local_first_bad_edge`.

## 0ZZZZ. Backbone adaptation and judge/oracle bounded study (2026-03-11)

This round asked three operational questions:

1. if we hold the Phase E head/pair pipeline fixed but swap in an already
   adapted PRM backbone, does benchmark transfer improve,
2. if we keep the backbone fixed but filter pairs through an independent PRM
   oracle, does data quality improve enough to offset the smaller pool,
3. is the newly installed local judge ready to relabel prefix pairs directly.

Primary artifacts:
1. summary note:
   - `docs/phase_e_backbone_judge_experiments_20260311.md`
2. adapted-backbone train run:
   - `assets/artifacts/phase_e_runs/phase_e_backboneproxy_prm_mixedsmall_20260311T074134Z`
3. adapted-backbone same-family eval:
   - `assets/artifacts/phase_e_samefamily_eval/phase_e_backboneproxy_prm_mixedsmall_samefamily_20260311T080555Z`
4. adapted-backbone `ProcessBench Math 50` eval:
   - `assets/artifacts/phase_e_eval/phase_e_backboneproxy_prm_mixedsmall_math50_20260311T080653Z`
5. PRM-oracle filtered artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_prm_oracle_filter_mixedsmall_20260311T081105Z`
6. filtered-base train run:
   - `assets/artifacts/phase_e_runs/phase_e_oraclefilter_base_mixedsmall_20260311T081526Z`
7. filtered-base `ProcessBench Math 50` eval:
   - `assets/artifacts/phase_e_eval/phase_e_oraclefilter_base_mixedsmall_math50_20260311T082011Z`
8. direct prefix-judge audit:
   - `assets/artifacts/phase_e_judge_audit/phase_e_judge_prefix_audit_mixedsmall_20260311T082237Z`

### 0ZZZZ.1 Adapted-backbone proxy result

The backbone proxy used `Qwen2.5-Math-PRM-7B` as the frozen feature extractor
under the ordinary mixed-small Phase E recipe.

Held-out pair fit:
1. `pair_acc = 0.8984`
2. `auc = 0.8946`

Same-family trust:
1. `prompt_pool_top1_accuracy = 0.9160`
2. `local_first_bad_edge_accuracy = 0.9487`

`ProcessBench Math 50`:
1. `pair_acc = 0.6234`
2. `auc = 0.6810`
3. `first_edge = 0.6923`
4. `mean_all_correct_last_score = 0.5749`
5. `processbench_f1 = 0.4800`

Interpretation:
1. benchmark transfer improved strongly,
2. same-family trust did not dominate,
3. so backbone representation is a real ceiling,
4. but naive backbone replacement does not preserve all existing trust geometry.

### 0ZZZZ.2 PRM-oracle filtering result

The oracle filter kept:
1. train: `878 / 2048 = 0.4287`
2. eval: `121 / 256 = 0.4727`

Critical composition finding:
1. `first_bad_fanout_prefix_ranking`: kept `770 / 1536`
2. `local_modified_process_error_step`: kept `107 / 380`
3. `terminal_completion_anchor`: kept only `1 / 132` on train and `0 / 12` on eval

Training on this filtered pool produced:
1. held-out `pair_acc = 0.8678`
2. held-out `auc = 0.8427`

`ProcessBench Math 50` then fell to:
1. `pair_acc = 0.5190`
2. `auc = 0.5535`
3. `mean_all_correct_last_score = 0.1543`
4. `processbench_f1 = 0.2933`

Interpretation:
1. naive global oracle filtering is a negative result,
2. because it silently deletes terminal supervision and breaks the mixed objective.

### 0ZZZZ.3 Direct prefix-judge audit result

Judge:
1. `Qwen2.5-Math-7B-Instruct`

Audit setup:
1. `24` validation pairs from the same mixed-small artifact,
2. prompt explicitly said:
   - judge only the displayed steps,
   - do not punish incompleteness,
   - return JSON.

Result:
1. `pair_json_ok_rate = 0.0000`
2. `pair_agreement_rate = 0.0000`
3. elapsed time: `369.3 sec`

Observed failure:
1. the model mostly produced free-form analysis,
2. not a machine-readable JSON contract,
3. so it is not yet a practical automatic prefix-pair relabeler in the current
   local `transformers + generate` stack.

### 0ZZZZ.4 Current repository-level conclusion

1. if the goal is to move `ProcessBench`, backbone adaptation is the strongest
   lever found this round.
2. if the goal is to keep the mixed supervision geometry healthy, naive global
   oracle filtering is the wrong move.
3. the new judge models are still useful, but their immediate role should be:
   - disagreement audit,
   - terminal / full-solution relabel,
   - or low-volume adjudication,
   not direct prefix-pair auto-filtering.

## 0ZZZ. Judge LLM benchmark check on real ProcessBench slices (2026-03-11)

### 0ZZZ.1 Why the smoke result was not enough

The earlier judge smoke only used tiny toy math examples. That was enough to
check:

1. model loading,
2. basic prompting,
3. and whether structured outputs are even plausible.

It was not enough to answer:

1. how the two candidate judge LLMs behave on real benchmark rows,
2. whether they can locate the first bad step,
3. and whether verbose output contracts are the main bottleneck.

### 0ZZZ.2 New script

1. `scripts/phase_e_benchmark_judge_llm.py`

This script:

1. loads deterministic small ProcessBench slices,
2. runs one local instruct model as a judge,
3. evaluates:
   - parse success,
   - overall correctness,
   - first-bad exact accuracy,
   - first-bad within-1 accuracy,
   - mean step accuracy.

### 0ZZZ.3 Experimental setup

Benchmarks:

1. `ProcessBench math`
2. `ProcessBench gsm8k`

Per-benchmark cap:

1. `6` rows each

Compared contracts:

1. `full_steps`
2. `first_bad_only`

Compared judge models:

1. `Qwen2.5-Math-7B-Instruct`
2. `DeepSeek-R1-Distill-Qwen-14B`

### 0ZZZ.4 Artifacts

Runs:

1. `assets/artifacts/phase_e_judge_bench/judge_bench_qwen25_math_7b_20260311T073507Z`
2. `assets/artifacts/phase_e_judge_bench/judge_bench_deepseek_r1_14b_20260311T073507Z`
3. `assets/artifacts/phase_e_judge_bench/judge_bench_qwen25_math_7b_fbonly_20260311T074138Z`
4. `assets/artifacts/phase_e_judge_bench/judge_bench_deepseek_r1_14b_fbonly_20260311T074138Z`

Unified compare:

1. `assets/artifacts/phase_e_judge_bench_compare/judge_model_compare_20260311T074514Z/summary.md`

### 0ZZZ.5 Main results

The most important rows are:

1. `Qwen2.5-Math-7B-Instruct`, `full_steps`
   - `ProcessBench math`
     - `parse_ok=0.6667`
     - `overall_acc=0.3333`
     - `first_bad_exact=0.0000`
   - `ProcessBench gsm8k`
     - `parse_ok=1.0000`
     - `overall_acc=0.5000`
     - `first_bad_exact=0.0000`

2. `Qwen2.5-Math-7B-Instruct`, `first_bad_only`
   - `ProcessBench math`
     - `parse_ok=0.5000`
     - `overall_acc=0.3333`
     - `first_bad_exact=0.0000`
   - `ProcessBench gsm8k`
     - `parse_ok=1.0000`
     - `overall_acc=0.5000`
     - `first_bad_exact=0.0000`

3. `DeepSeek-R1-Distill-Qwen-14B`, `full_steps`
   - `ProcessBench math`
     - `parse_ok=0.5000`
     - `overall_acc=0.1667`
     - `first_bad_exact=0.0000`
   - `ProcessBench gsm8k`
     - `parse_ok=0.5000`
     - `overall_acc=0.3333`
     - `first_bad_exact=0.0000`
     - `first_bad_within1=0.3333`

4. `DeepSeek-R1-Distill-Qwen-14B`, `first_bad_only`
   - `ProcessBench math`
     - `parse_ok=0.5000`
     - `overall_acc=0.1667`
     - `first_bad_exact=0.0000`
   - `ProcessBench gsm8k`
     - `parse_ok=0.8333`
     - `overall_acc=0.6667`
     - `first_bad_exact=0.3333`
     - `first_bad_within1=0.3333`

### 0ZZZ.6 Failure pattern reading

This round exposed a clear split:

1. `Qwen2.5-Math-7B-Instruct`
   - better operational stability,
   - but on real benchmark rows it is heavily biased toward:
     - `OVERALL=correct`
     - `FIRST_BAD=none`
   - simplifying the contract does not fix that bias.

2. `DeepSeek-R1-Distill-Qwen-14B`
   - less stable under verbose contracts,
   - but once the contract is reduced to `first_bad_only`, it shows real signal
     on `gsm8k`,
   - while remaining weak on `math`.

Therefore:

1. verbosity is not the main blocker for `Qwen2.5-Math-7B-Instruct`,
2. verbosity *is* a material blocker for `DeepSeek-R1-Distill-Qwen-14B`,
3. neither model is currently strong enough to act as a benchmark-ready,
   standalone ProcessBench judge.

### 0ZZZ.7 Operational conclusion

1. Keep `Qwen2.5-Math-7B-Instruct` as the bulk local judge candidate.
2. Do not expect it to act as a precise first-bad-step annotator.
3. Use `DeepSeek-R1-Distill-Qwen-14B` only as a second-stage adjudicator,
   especially on more verbal / GSM-style cases and only under a lighter output
   contract.

## 0ZZ. Judge LLM local selection + deployment check (2026-03-11)

### 0ZZ.1 Why this round was needed

Phase E is about to move toward stronger `LLM-as-a-judge` style supervision.
Before wiring that into relabeling or active learning, we needed to answer:

1. which local judge models are most defensible from the literature,
2. which ones fit the current server budget,
3. which ones actually work in our local inference stack.

### 0ZZ.2 Sources checked

Web / model cards:

1. `ProcessBench`
   - `https://arxiv.org/abs/2412.06559`
2. `ThinkPRM`
   - `https://arxiv.org/abs/2504.16828`
3. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   - `https://arxiv.org/abs/2501.07301`
4. `DeepSeek-R1-Distill-Qwen-14B`
   - `https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
5. `Qwen2.5-Math-7B-Instruct`
   - `https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct`
6. `QwQ-32B`
   - `https://huggingface.co/Qwen/QwQ-32B`

Local documents / PDFs:

1. `docs/relatedPapers/2412.06559_processbench.pdf`
2. `docs/relatedPapers/2504.16828_thinkprm.pdf`
3. `docs/relatedPapers/2501.07301_lessons_developing_prm.pdf`
4. `docs/bcr_synthesis_report_20260311.md`

### 0ZZ.3 Operational selection

The chosen local judge stack is now:

1. bulk math judge:
   - `assets/models/Qwen2.5-Math-7B-Instruct`
2. stronger adjudicator candidate:
   - `assets/models/DeepSeek-R1-Distill-Qwen-14B`
3. cheapest fallback baseline:
   - `assets/models/Qwen2.5-7B-Instruct`

`QwQ-32B` was deliberately not downloaded in this round:

1. it is a strong open-source critic candidate,
2. but it is too expensive for day-to-day local bulk judging at the repo's
   current stage,
3. and a 14B adjudicator is enough to validate the local judge path first.

### 0ZZ.4 New local tooling

New script:

1. `scripts/phase_e_smoke_judge_llm.py`

Purpose:

1. load one local instruct model,
2. send a strict structured step-judge prompt,
3. test tiny good/bad reasoning examples,
4. save artifacts under `assets/artifacts/phase_e_judge_smoke/`.

### 0ZZ.5 Smoke-test results

Artifacts:

1. `assets/artifacts/phase_e_judge_smoke/judge_smoke_qwen25_7b_20260311T071409Z`
2. `assets/artifacts/phase_e_judge_smoke/judge_smoke_qwen25_math_7b_v2_20260311T071746Z`
3. `assets/artifacts/phase_e_judge_smoke/judge_smoke_deepseek_r1_qwen14b_v2_20260311T072126Z`

Observed behavior:

1. `Qwen2.5-7B-Instruct`
   - `json_ok = 1 / 2`
   - correct sample works
   - bad sample degenerates into `!!!!!!!!...`
   - useful only as the cheapest baseline
2. `Qwen2.5-Math-7B-Instruct`
   - `json_ok = 1 / 2`
   - semantics on the bad math example are actually good
   - format adherence is imperfect, but tolerant parsing can recover some cases
   - best current candidate for bulk local judge use
3. `DeepSeek-R1-Distill-Qwen-14B`
   - model loads successfully
   - but current local `transformers + strict JSON judge prompt` path is not stable
   - both the generic prompt and the model-card-aligned prompt produced unusable outputs
   - adding recommended sampling then triggered `inf/nan/<0` probability failure

### 0ZZ.6 Final judgement

The important outcome is not "14B is stronger than 7B".

The important outcome is:

1. the repository now has one locally installed judge path that is cheap enough
   and operational enough to pilot:
   - `Qwen2.5-Math-7B-Instruct`
2. the stronger 14B reasoning model should not yet be promoted to the main
   Phase E judge loop without a different serving setup or more specialized
   prompting / decoding control.

### 0ZZ.7 Immediate next step

Use `Qwen2.5-Math-7B-Instruct` first for:

1. low-confidence pair relabeling,
2. disagreement mining,
3. judge-confidence artifact logging.

## 0Z. ProcessBench state audit + community gap review (2026-03-11)

### 0Z.1 Why this audit was needed

A teammate reported three things at once:

1. many new local PDFs and survey notes had been added,
2. multiple external datasets were now downloaded locally,
3. some newer experiments had "fixed part of `ProcessBench`".

The practical question was therefore no longer:

1. "what do we think the repo is doing?"

but:

1. "what do the current code, current artifacts, and current paper set jointly
   support?"

### 0Z.2 What was checked

Code / wrappers:

1. `scripts/run_phase_e_processbench_research_suite.sh`
2. `scripts/phase_e_curate_processbench_transfer_pairs.py`
3. `scripts/phase_e_curate_semantic_pairs.py`
4. `src/ours/phase_b/value_head.py`
5. `src/ours/phase_e/training.py`
6. `src/ours/phase_e/processbench_alignment.py`

Local papers / notes:

1. `docs/research_survey_processverifier_20260311.md`
2. `docs/bcr_feasibility_review_20260311.md`
3. `docs/dataset_survey_stepwise_pairs_20260311.md`
4. `docs/new_dataset_experiment_plan_20260311.md`
5. `docs/phase_e_rl_ready_research_redesign_20260311.md`
6. PDFs under `docs/relatedPapers/`

Fresh audit artifacts produced in this round:

1. `assets/artifacts/phase_e_transfer_compare/processbench_state_review_math_0311_20260311T070248Z/summary.md`
2. `assets/artifacts/phase_e_transfer_compare/processbench_state_review_gsm_0311_20260311T070308Z/summary.md`
3. `assets/artifacts/phase_e_rl_promotion_diag/rl_promotion_state_review_0311_20260311T070332Z/summary.md`

New synthesis note:

1. `docs/processbench_state_and_community_gap_20260311.md`

### 0Z.3 Fresh state-of-repo conclusions

The teammate claim "some `ProcessBench` behavior is fixed" is only partially
correct.

What is true:

1. recent repairs do improve some benchmark slices:
   - `terminal_top1`
   - some `first_edge`
2. pair-geometry alignment is somewhat better in the newer curated /
   ProcessBench-aware artifacts than in pure local baselines

What is not true:

1. there is still no new mainline candidate that beats the strongest older runs
   on overall benchmark utility while also preserving same-family trust

The strongest current evidence:

1. `ms_e43` remains the strongest local / later-bad candidate
   - `ProcessBench Math`:
     - `auc=0.6341`
     - `good_vs_laterbad=0.7515`
     - `terminal_top1=0.0099`
2. `prm_e46` remains the strongest more-balanced benchmark-facing candidate
   - `ProcessBench GSM8K`:
     - `auc=0.6264`
   - `ProcessBench Math`:
     - `auc=0.6053`
     - `good_vs_laterbad=0.5501`
     - `terminal_top1=0.1970`
3. recent repair candidates still fail the strict RL gate:
   - `c3_curated_gated`
     - `assessment=not_rl_ready_laterbad_generalization_weak`
   - `c4_dual`
     - `assessment=not_rl_ready_laterbad_generalization_weak`
   - `pbr2_align_gated`
     - `assessment=not_rl_ready_laterbad_generalization_weak`
   - `e87_repair`
     - `assessment=terminal_and_local_tradeoff_unresolved`

### 0Z.4 Community-vs-repo gap

The paper set is increasingly consistent on four points:

1. strong process verification uses richer supervision than single-trajectory
   local first-bad edges
2. MC-only synthetic labels are noisy for step correctness
3. strong current systems increasingly rely on:
   - consensus filtering / judge filtering
   - tree / sibling-branch preferences
   - backbone adaptation or generative verification
4. evaluation must separate:
   - local first-edge
   - later-bad
   - all-correct terminal behavior

The current repo mainline still does:

1. frozen backbone
2. `last_token` pooled frozen features
3. small scalar head
4. pairwise ranking/BCE on converted pairs

This means the repo is still strongest as:

1. a diagnosis and ablation platform

not yet as:

1. a community-level competitive verifier stack

### 0Z.5 Main anti-patterns now made explicit

1. do not treat `heldout_pair_acc` or same-family top1 as the main success
   proxy
2. do not narrate one repaired slice (`terminal_top1`, `first_edge`) as
   "ProcessBench fixed"
3. do not treat `fanout/grid` as true tree / sibling-branch supervision
4. do not keep adding new scalar heads before using the stronger downloaded
   sources already present locally

### 0Z.6 Decision for next experiments

The next mainline should be:

1. stop architecture churn for one round
2. first debug / mainline the stronger local datasets:
   - `PRM800K`
   - `MATH-APS`
   - `EurusPRM-Stage2`
   - `Math-Step-DPO-10K`
3. run equal-budget source-only ProcessBench slice audits
4. only then do one minimal frozen-vs-LoRA comparison on the strongest source
   mixture

Therefore the current research reading changes from:

1. "keep repairing ProcessBench mostly by pair geometry and head design"

to:

1. "use the current stack to isolate which missing ingredient matters most:
   better source, better supervision geometry, or minimal backbone adaptation"

## 0Y. Verified internet reading + `dual_head` semantic-routing smoke (2026-03-11)

### 0Y.1 Why this follow-up existed

After the earlier semantic-curation + `gated_mlp` smoke, the remaining open
question was:

1. if the real bottleneck is "one scalar head is compressing multiple verifier
   subtasks",
2. can a lightweight decomposed head help inside the existing frozen-feature
   Phase E stack,
3. without yet paying the cost of a full critique/generative verifier?

### 0Y.2 Verified external reading

The following sources were re-checked directly on 2026-03-11 and used to guide
this round:

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
8. `Open Instruct` reward-modeling docs
   - <https://allenai.github.io/open-instruct/algorithms/reward_modeling/>

The practical synthesis from those sources was:

1. benchmark difficulty is broader than one local edge
2. richer intermediate preference relations matter
3. evaluation should separate local, later-bad, and terminal behavior
4. if we stay inside a scalar-verifier regime, "decompose the objective" is more
   defensible than "blindly add more head capacity"

### 0Y.3 New code for the decomposition smoke

Updated:

1. `src/ours/phase_b/value_head.py`
   - added `architecture='dual_head'`
   - added `inference_alpha`
   - dual head returns:
     - blended `logits/scores`
     - branch-specific `local_*`
     - branch-specific `terminal_*`
2. `src/ours/phase_e/training.py`
   - added semantic route resolution and per-pair route weights
   - current routing:
     - `terminal_completion_anchor -> terminal`
     - `local_* / first_bad_fanout_prefix_ranking -> local`
     - `good_bad_prefix_grid -> both`
   - `compute_pair_objective()` now supports routed dual-head loss
3. `scripts/phase_e_train_value.py`
   - added `--head-architecture dual_head`
   - added `--head-inference-alpha`
   - threaded route weights through epoch training

Tests:

1. `python -m py_compile src/ours/phase_b/value_head.py src/ours/phase_e/training.py scripts/phase_e_train_value.py`
2. `pytest -q tests/unit/test_value_head.py tests/unit/test_phase_e_training.py tests/unit/test_phase_e_train_script.py`
3. result:
   - `23 passed`

### 0Y.4 Experiment design

Controlled comparison:

1. keep the exact same curated artifact as the previous `CR1` smoke
2. keep the same optimizer / balancing / centering settings as `C3`
3. replace only the head:
   - old: `gated_mlp`
   - new: `dual_head`

Artifact and run paths:

1. curated pairs:
   - `assets/artifacts/phase_e_pairs/phase_e_curated_rlready_0311_retry2_curated_pairs__03ac5eebc8fd`
2. new value run:
   - `assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_dual_c4_value_20260311T051319Z`
3. new samefamily eval:
   - `assets/artifacts/phase_e_samefamily_eval/phase_e_curated_rlready_0311_dual_c4_samefamily_20260311T051616Z`
4. new ProcessBench eval:
   - GSM8K:
     - `assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_dual_c4_pb_gsm8k_20260311T051637Z`
   - Math:
     - `assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_dual_c4_pb_math_20260311T051652Z`

### 0Y.5 Main results

#### `C4_CURATED_DUAL_CENTER`

1. held-out:
   - `pair_acc=0.8608`
   - `auc=0.7390`
2. samefamily:
   - `top1=0.8787`
   - `local_first_bad=0.9408`
3. ProcessBench GSM8K:
   - `auc=0.4730`
   - `first_edge=0.5660`
   - `terminal_top1=0.8548`
   - `good_vs_laterbad=0.2756`
4. ProcessBench Math:
   - `auc=0.4789`
   - `first_edge=0.5714`
   - `terminal_top1=0.9038`
   - `good_vs_laterbad=0.3672`

Cross-run slice comparison artifacts:

1. Math:
   - `assets/artifacts/phase_e_transfer_compare/processbench_curated_arch_compare_0311_dual_math_20260311T051707Z/summary.md`
2. GSM8K:
   - `assets/artifacts/phase_e_transfer_compare/processbench_curated_arch_compare_0311_dual_gsm_20260311T051726Z/summary.md`
3. RL-promotion compare:
   - `assets/artifacts/phase_e_rl_promotion_diag/phase_e_curated_rlpromo_compare_0311_dual_20260311T051742Z/summary.md`

### 0Y.6 Interpretation

Compared with `C3_CURATED_GATED_CENTER`, the new `dual_head` does improve some
of the *right* slices:

1. `first_edge`
   - GSM8K:
     - `0.4906 -> 0.5660`
   - Math:
     - `0.5397 -> 0.5714`
2. terminal behavior
   - GSM8K terminal top1:
     - `0.7097 -> 0.8548`
   - Math terminal top1:
     - `0.8654 -> 0.9038`

But it degrades the broader things we still need:

1. samefamily top1:
   - `0.9443 -> 0.8787`
2. overall benchmark AUC:
   - GSM8K:
     - `0.4861 -> 0.4730`
   - Math:
     - `0.5152 -> 0.4789`
3. `good_vs_laterbad`
   - GSM8K:
     - `0.4267 -> 0.2756`
   - Math:
     - `0.4738 -> 0.3672`

This means:

1. the decomposition hypothesis is not empty
   - the model *did* move the expected slices
2. but the current hard semantic routing is too blunt
   - it over-specializes local-first-edge and terminal behavior
   - while under-preserving broader good-vs-bad ranking geometry

### 0Y.7 Final decision from this round

1. do **not** promote current `dual_head` as the new mainline
2. keep `C3 gated_mlp` as the stronger current curated baseline
3. if the decomposition direction continues, the next version should test:
   - softer route weights
   - inference `alpha` sweep
   - or staged curriculum (`single-head/gated` warm start -> dual-head repair)
4. current RL-promotion verdict remains unchanged:
   - `gate=0`
   - main failure label:
     - `not_rl_ready_laterbad_generalization_weak`

## 0X. Internet-guided semantic-curation + reward-centering smoke (2026-03-11)

### 0X.1 What this round asked

This round turned the latest paper/community reading into one concrete
repository question:

1. if transfer failure is partly caused by supervision geometry mismatch,
2. can we first curate a small semantic-balanced pool,
3. then add reward centering as a low-risk reward-model regularizer,
4. and finally test whether a more flexible scalar head (`gated_mlp`) improves
   benchmark transfer over a plain `mlp`?

### 0X.2 New code and docs

External-reading artifact:

1. `docs/phase_e_internet_research_20260311.md`

New / updated code:

1. `src/ours/phase_b/value_losses.py`
   - added `reward_centering_penalty()`
2. `src/ours/phase_e/training.py`
   - threaded `reward_centering_weight` into train/eval objective computation
3. `scripts/phase_e_train_value.py`
   - added `--reward-centering-weight`
4. `scripts/phase_e_curate_semantic_pairs.py`
   - deterministic curation by semantic bucket and source filter
5. `scripts/run_phase_e_curated_rlready_suite.sh`
   - one-click curated smoke suite

Wrapper hardening during execution:

1. fixed shell backtick substitution in suite metadata
2. fixed env override clobbering so `EVAL_BATCH_SIZE=12` really takes effect
3. fixed ProcessBench terminal extraction in the suite script so future reruns
   compute `all-correct final-prefix top1` from `scored_rows.jsonl`

### 0X.3 Curated artifact contract

Artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_curated_rlready_0311_retry2_curated_pairs__03ac5eebc8fd`

Composition:

1. `Math-Shepherd local fanout`
   - train `1600`
   - val `160`
2. `PRMBench local sibling`
   - train `1600`
   - val `160`
3. `PRMBench terminal anchor`
   - train `320`
   - val `32`

Total:

1. train `3520`
2. val `352`

Interpretation:

1. this is a deliberately bounded terminal regime
2. terminal supervision is about `9.1%` of the train pool
3. this is exactly the regime we wanted to test after the earlier
   terminal-overcorrection failures

### 0X.4 Smoke suite result

Suite summary:

1. `assets/artifacts/phase_e_logs/phase_e_curated_rlready_0311_retry2/final_summary.md`

#### `C1_CURATED_MLP_BASE`

1. held-out:
   - `pair_acc=0.9034`
   - `auc=0.8574`
2. samefamily:
   - `top1=0.9541`
   - `local_first_bad=0.9737`
3. ProcessBench:
   - GSM8K:
     - `auc=0.4892`
     - `first_edge=0.4906`
   - Math:
     - `auc=0.4553`
     - `first_edge=0.5079`

#### `C2_CURATED_MLP_CENTER`

1. held-out:
   - `pair_acc=0.8977`
   - `auc=0.8643`
2. samefamily:
   - `top1=0.9508`
   - `local_first_bad=0.9737`
3. ProcessBench:
   - GSM8K:
     - `auc=0.4928`
     - `first_edge=0.4906`
   - Math:
     - `auc=0.4545`
     - `first_edge=0.5079`

Reading:

1. reward centering alone did almost nothing for benchmark transfer
2. it slightly changed held-out calibration, but not the actual transfer regime

#### `C3_CURATED_GATED_CENTER`

1. held-out:
   - `pair_acc=0.9290`
   - `auc=0.8706`
2. samefamily:
   - `top1=0.9443`
   - `local_first_bad=0.9934`
3. ProcessBench:
   - GSM8K:
     - `auc=0.4861`
     - `first_edge=0.4906`
   - Math:
     - `auc=0.5152`
     - `first_edge=0.5397`

Reading:

1. this is the only config that improved the hard `ProcessBench Math` slice
2. but it still failed badly on GSM8K
3. therefore it changes the tradeoff, but does not clear the RL-facing gate

### 0X.5 Strict transfer diagnosis

Diagnostic artifact:

1. `assets/artifacts/phase_e_transfer_diag/phase_e_curated_rlready_0311_retry2_diag_00/summary.md`

All three candidates were classified as:

1. `not_rl_ready_local_transfer_weak`

Shared failure tags:

1. `benchmark_local_error_weak`
2. `margin_collapse`
3. `support_length_drift`

Important implication:

1. in this curated regime, the main blocker is **not** terminal completion
2. the main blocker is that local ProcessBench discrimination still does not
   survive transfer strongly enough

### 0X.6 Comparison to the previous best mixed baseline

Reference baseline:

1. `E82`
   - samefamily:
     - `top1=0.9633`
     - `local_first_bad=0.9841`
   - ProcessBench GSM8K:
     - `auc=0.5738`
     - `first_edge=0.6038`
   - ProcessBench Math:
     - `auc=0.4937`
     - `first_edge=0.4603`

Comparison:

1. `C3` improved `ProcessBench Math`
   - `auc 0.4937 -> 0.5152`
   - `first_edge 0.4603 -> 0.5397`
2. but `C3` hurt `ProcessBench GSM8K`
   - `auc 0.5738 -> 0.4861`
3. and `C3` still lost a little samefamily top1
   - `0.9633 -> 0.9443`

Conclusion:

1. the new pipeline found a **directional math-side improvement**
2. but it did **not** produce a universally better or RL-ready candidate

### 0X.7 Next step

This round narrows the next principled repair:

1. stop treating terminal completion as the main missing piece in this curated regime
2. focus on why local benchmark transfer still collapses:
   - benchmark length/support drift
   - possible need for staged objectives instead of naive joint mixture
   - explicit dual-objective or dual-head local-vs-terminal training
3. keep reward centering available, but do not expect it to be a primary fix by
   itself

## 0W. Latest Diagnosis Update (2026-03-11, ProcessBench Hybrid Artifact + Architecture Comparison)

### 0W.1 What this round asked

This round turned the latest literature reading into one concrete question:

1. if `ProcessBench` transfer fails because current training is too local,
2. can we keep `PRMBench` local error-step pairs as the anchor,
3. add only a very small terminal auxiliary,
4. and recover benchmark behavior without destroying local discrimination?

At the same time, it asked a second question:

1. if this repair still fails,
2. is the main blocker:
   - data contract mismatch,
   - or insufficient head architecture?

### 0W.2 New pipeline / code

New curation + training support:

1. `scripts/phase_e_mix_pair_artifacts.py`
   - now supports weighted source mixing:
     - `LABEL=DIR:TRAIN_CAP:VAL_CAP:MIX_WEIGHT`
2. `scripts/run_phase_e_processbench_hybrid_suite.sh`
   - new benchmark-oriented wrapper:
     - `PRMBench local` anchor
     - bounded terminal auxiliary
     - `mlp` vs `gated_mlp`
3. wrapper hardening:
   - fixed baseline triplet parsing for:
     - `name=run_dir::gsm_eval_dir::math_eval_dir`
   - removed shell backtick substitution hazards in suite metadata
   - separated helper logs from helper return values
   - switched helper outputs to global result variables instead of command substitution

### 0W.3 Executed artifact

Hybrid artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_pbhybrid_ph1_0311_1230_pairs__d6fb5a3ec28c`

Composition:

1. `prm_local`
   - train `3072`
   - val `384`
2. `prm_terminal`
   - train `512`
   - val `64`

Total:

1. train `3584`
2. val `448`

Important detail from later failure analysis:

1. the *effective* terminal-anchor semantics inside this hybrid are only:
   - `132 / 3584 = 0.0368`
2. so this was already a very conservative terminal repair.

### 0W.4 Same-source held-out result

Runs:

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

Interpretation:

1. the hybrid artifact is easy to fit on same-source held-out pairs,
2. so this round is **not** another "the head cannot learn at all" result.

### 0W.5 ProcessBench transfer result

Comparison artifacts:

1. GSM8K compare:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_pbhybrid_ph1_0311_1230_processbench_gsm8k_compare_20260311T045320Z/summary.md`
2. Math compare:
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

### 0W.6 Failure pattern

Failure-analysis artifacts:

1. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_mlp_gsm_diag_20260311T045406Z/summary.md`
2. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_gated_gsm_diag_20260311T045406Z/summary.md`
3. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_mlp_math_diag_20260311T045406Z/summary.md`
4. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_gated_math_diag_20260311T045406Z/summary.md`

Key reading:

1. benchmark structure:
   - GSM8K `all_correct_ratio=0.4844`
   - Math `all_correct_ratio=0.4062`
2. hybrid training semantics:
   - `local_modified_process_error_step = 3452`
   - `terminal_completion_anchor = 132`
3. despite the terminal fraction being only `3.68%`,
   - terminal completion is repaired extremely aggressively,
   - but local ranking and first-error discrimination collapse.

Concrete example:

1. GSM8K `mlp`
   - `all_correct terminal top1 = 0.8548`
   - `first_edge = 0.4906`
2. Math `gated_mlp`
   - `all_correct terminal top1 = 0.8654`
   - `late_error pair_acc = 0.2131`

This means:

1. the model is learning "full correct completion should score high,"
2. but it loses the relative geometry needed for bad-prefix discrimination.

### 0W.7 Scientific conclusion

This round is a strong negative result against the simple hybrid recipe:

1. `PRMBench local + tiny terminal auxiliary` is **not** enough.
2. `gated_mlp` changes the tradeoff slightly,
   - but architecture alone does not repair the benchmark mismatch.
3. the main blocker is still supervision geometry:
   - terminal anchors are powerful even at very small ratios,
   - and naive mixture is still too blunt.

So the next mainline should **not** be:

1. more of the same mixture with larger caps,
2. or pure head-capacity tuning.

The next mainline should be:

1. smaller terminal ratios than the current `3.68%`,
2. benchmark-aware checkpoint selection,
3. or explicit staged / two-objective training where:
   - local ranking remains primary,
   - terminal-completion repair is constrained instead of merged naively.

## 0V. Latest Diagnosis Update (2026-03-11, RL Promotion Infrastructure Hardening + Nonfinite Feature Audit)

### 0V.1 What this round answered

This round asked a stricter question than the earlier "same-family strong" or
"ProcessBench somewhat positive" reads:

1. can the current Phase E infrastructure actually support a defensible
   RL-promotion decision,
2. and if not, is the blocker:
   - benchmark semantics,
   - shared-server infrastructure,
   - or hidden numerical corruption?

The answer is now:

1. the infrastructure is materially better after this round,
2. but no current candidate became RL-ready,
3. and a previously hidden numerical corruption bug was real.

### 0V.2 New infrastructure

Primary additions:

1. `scripts/phase_e_diagnose_rl_promotion.py`
   - new slice-aware RL-promotion gate
2. `scripts/phase_e_prepare_prmbench_terminal_anchor_pairs.py`
   - new `--terminal-anchor-ratio`
3. `scripts/phase_e_train_value.py`
   - fail fast on non-finite loss
   - new `--nonfinite-feature-policy {error,drop}`
4. `src/ours/phase_e/training.py`
   - explicit non-finite pooled-feature validation / filtering

Why this matters:

1. the repository can now distinguish:
   - local-first-bad weakness
   - later-bad weakness
   - terminal-completion weakness
   - outright numerical corruption
2. this is much closer to what an RL-promotion pipeline actually needs than a
   single `AUC` threshold.

### 0V.3 Baseline RL-promotion audit

Artifact:

1. `assets/artifacts/phase_e_rl_promotion_diag/phase_e_rlpromo_diag_baselines_0311_20260311T040150Z/summary.md`

Main reading:

1. `E80 fanout`
   - `samefamily_top1=0.9954`
   - `pb_min_auc=0.5106`
   - `pb_min_laterbad=0.5167`
   - `pb_min_terminal_top1=0.0577`
   - assessment:
     - `terminal_and_local_tradeoff_unresolved`
2. `E84 heavy terminal`
   - `samefamily_top1=0.8249`
   - `pb_min_auc=0.3906`
   - `pb_min_laterbad=0.2448`
   - `pb_min_terminal_top1=0.5897`
   - assessment:
     - `not_rl_ready_laterbad_generalization_weak`
3. `E46 PRM local`
   - `samefamily_top1=0.9659`
   - `pb_min_auc=0.6053`
   - `pb_min_laterbad=0.5501`
   - `pb_min_terminal_top1=0.1970`
   - assessment:
     - `terminal_and_local_tradeoff_unresolved`

Interpretation:

1. `E80` is still local-strong but terminal-poor.
2. `E84` repairs the terminal slice but destroys broader ranking.
3. `E46` remains the closest current baseline, but terminal completion is still
   the main blocker.

### 0V.4 Hidden numerical bug: pooled feature tensors themselves contained `NaN`

Math-side repair artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_rlpromo_0311_ms_fanout_ta015__16a79535c2e6`

Observed behavior:

1. warm-start `joint` continuation from `E80`
   - `loss=nan`
2. more conservative `ranking_only` continuation from `E80`
   - still failed on batch 1
   - with:
     - `rejected_logit_abs_max=nan`

Direct cache inspection then showed:

1. chosen pooled-feature cache had `13` bad rows
2. rejected pooled-feature cache had `31` bad rows
3. the bad rows were mainly `first_bad_fanout_prefix_ranking` examples, not
   only terminal anchors

This is the main technical result of the round:

1. the earlier NaN failure was not just a weak recipe,
2. it was an infrastructure-level corruption risk in:
   - `backbone -> pooled feature -> cache -> head`

### 0V.5 What the new `drop` policy achieved

Stable repaired run:

1. `assets/artifacts/phase_e_runs/phase_e_rlpromo_0311_ms_fanout_ta015_rankonly_warm_e80_dropnf_fix_20260311T044122Z`

What happened:

1. train bad rows dropped:
   - `44 / 9921`
2. eval bad rows dropped:
   - `3 / 1164`
3. held-out repair metrics:
   - `pair_acc=0.7700`
   - `auc=0.7235`
   - `ranking_score=0.7468`
4. same-family trust:
   - `prompt_pool_top1=0.7802`
   - `local_first_bad_acc=0.8492`
5. `ProcessBench`:
   - GSM8K:
     - `auc=0.5832`
     - `first_edge=0.6176`
   - Math:
     - `auc=0.5206`
     - `first_edge=0.5491`

Comparison artifact:

1. `assets/artifacts/phase_e_rl_promotion_diag/phase_e_rlpromo_diag_math_dropnf_0311_20260311T044717Z/summary.md`

Interpretation:

1. the infrastructure fix worked:
   - the candidate now trains and evaluates cleanly
2. but the scientific promotion result is still negative:
   - same-family utility dropped far below `E80`
   - `ProcessBench Math` stayed weak
   - the RL-promotion gate still returns `gate=0`

This is the right kind of negative result:

1. "numerical corruption removed"
2. but "candidate still not good enough"

### 0V.6 PRM-side light repair status

Prepared artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_rlpromo_0311_prm_ta010__b2281c74f155`

What was tried:

1. light `PRMBench` terminal-anchor continuation from `E46`
2. first with shared cache
3. then with `--feature-cache-mode off`

Outcome this round:

1. no completed final artifact was produced before this round closed
2. the shared-cache run was blocked by lock contention
3. the `off` run did remove the lock wait
4. but shared-machine feature extraction throughput was still too slow, so the
   run was manually stopped

Interpretation:

1. this is not a negative scientific result yet
2. it is an infrastructure / throughput limitation under a crowded server

### 0V.7 Updated practical conclusion

The repository is now closer to RL-ready *in infrastructure* than it was at
the start of the round:

1. promotion gating is stricter and more interpretable
2. non-finite losses no longer silently continue
3. non-finite pooled features are now explicit and optionally filterable

But the repository is still **not** RL-ready in candidate quality:

1. no audited candidate passes the strict RL-promotion gate
2. the best current benchmark-facing baseline remains `E46 PRM local`
3. math-side light repair proved:
   - infrastructure hardening can rescue a broken run
   - but that does not by itself produce an RL-usable head

The next sensible follow-up is:

1. re-run the light `PRMBench` continuation under an isolated feature-cache root
   or lower-cap smoke budget,
2. keep the new `error/drop` non-finite policy on by default in all repair work,
3. treat "candidate quality" and "infrastructure integrity" as two separate
   gates from now on.

## 0U. Internet-guided ProcessBench redesign suite (`pbr1/pbr2/pbr4`) (2026-03-11)

This round converted the latest internet research scan into one concrete local
redesign and then tested it end-to-end.

Primary code added:
1. `scripts/phase_e_curate_processbench_transfer_pairs.py`
2. `scripts/run_phase_e_processbench_research_suite.sh`
3. `src/ours/phase_e/training.py`
   - new `semantic / confidence_semantic` pair-weight modes

Executed smoke command:
1. `CUDA_VISIBLE_DEVICES=1 ACTIVE_PHASE_E_PB_RESEARCH_GROUP=PBR1_PROCESSBENCH_REDESIGN_SMOKE RUN_PREFIX=phase_e_processbench_research_v2 PHASE_E_PB_RESEARCH_CASES_OVERRIDE='pbr1_ms_align_mlp|one_shot|ms_align_v1|mlp|none pbr2_ms_align_gated|one_shot|ms_align_v1|gated_mlp|none pbr4_ms_curriculum_gated|curriculum|ms_align_v1|gated_mlp|none' TARGET_TOTAL_PAIRS=2048 BENCH_MAX_SAMPLES=64 TRAIN_BATCH_SIZE=64 EVAL_BATCH_SIZE=96 bash scripts/run_phase_e_processbench_research_suite.sh`

Primary artifacts:
1. suite summary:
   - `assets/artifacts/phase_e_logs/phase_e_processbench_research_v2/final_summary.md`
2. transfer compare:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_processbench_research_v2_gsm_compare_20260311T044545Z/summary.md`
   - `assets/artifacts/phase_e_transfer_compare/phase_e_processbench_research_v2_math_compare_20260311T044545Z/summary.md`
3. RL-promotion diagnosis:
   - `assets/artifacts/phase_e_rl_promotion_diag/phase_e_processbench_research_v2_rl_promotion_20260311T044546Z/summary.md`

Main result table:

| case | samefamily_top1 | samefamily_local | gsm_auc | gsm_first_edge | math_auc | math_first_edge | assessment |
|---|---:|---:|---:|---:|---:|---:|---|
| `ref_e87` | `0.6597` | `0.2914` | `0.4410` | `0.5854` | `0.4467` | `0.5957` | `terminal_and_local_tradeoff_unresolved` |
| `pbr1_ms_align_mlp` | `0.8316` | `0.8681` | `0.4754` | `0.3600` | `0.4692` | `0.2857` | `terminal_and_local_tradeoff_unresolved` |
| `pbr2_ms_align_gated` | `0.8947` | `0.9231` | `0.4713` | `0.5600` | `0.5055` | `0.5357` | `not_rl_ready_laterbad_generalization_weak` |
| `pbr4_ms_curriculum_gated` | `0.8316` | `0.8791` | `0.4947` | `0.4400` | `0.4743` | `0.4286` | `not_rl_ready_laterbad_generalization_weak` |

Conclusion:
1. `pbr2_ms_align_gated` is the strongest new candidate.
2. it materially improves same-family utility and terminal-completion behavior.
3. it still fails the strict RL-facing gate because `later-bad` and
   `first-edge` slices stay below threshold.
4. the current curriculum variant is not worth promoting.
5. the next repair target is now better localized:
   - improve `good_vs_laterbad` transfer
   - while keeping `first_bad` edge accuracy.

## 0T. Literature-guided mixed-supervision redesign on `ProcessBench Math 50` (2026-03-11)

This round asked a narrower but more actionable question than the earlier
repair smokes:

1. can we combine the useful part of `fanout`-style local supervision with the
   useful part of `terminal-anchor` supervision,
2. can training stay balanced enough that one branch does not erase the other,
3. and does head complexity matter once the data contract is improved?

Primary artifacts:
1. research/design note:
   - `docs/phase_e_rl_ready_research_redesign_20260311.md`
2. mixed artifact builder:
   - `scripts/phase_e_mix_pair_artifacts.py`
3. small mixed artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_pb_repair_0311_mixed_small_fanout_terminal__99976bcc33a8`
4. mixed-MLP run/eval:
   - `assets/artifacts/phase_e_runs/phase_e_pb_repair_0311_e75_mixed_small_mlp_20260311T042122Z`
   - `assets/artifacts/phase_e_eval/phase_e_pb_repair_0311_e75_mixed_small_mlp_math50_20260311T042808Z`
5. mixed-`gated_mlp` run/eval:
   - `assets/artifacts/phase_e_runs/phase_e_pb_repair_0311_e76_mixed_small_gated_20260311T042757Z`
   - `assets/artifacts/phase_e_eval/phase_e_pb_repair_0311_e76_mixed_small_gated_math50_20260311T043534Z`
6. comparison summaries:
   - `assets/artifacts/phase_e_transfer_compare/processbench_mixed_mlp_math50_compare_0311_20260311T043029Z/summary.md`
   - `assets/artifacts/phase_e_transfer_compare/processbench_mixed_arch_compare_0311_20260311T043656Z/summary.md`

### 0T.1 Research takeaway that directly changed the design

The literature and strong open-source baselines converge on one point:

1. `ProcessBench` is not only a local `last-safe > first-bad` edge task,
2. useful PRM supervision should preserve both:
   - local process discrimination
   - and broader progress / completion ordering,
3. naive PRM improvement is not yet the same thing as safe RL credit
   assignment.

That made the previous single-family repair logic too narrow.

### 0T.2 What changed locally

Pipeline changes:
1. new mixed-artifact balancing in `src/ours/phase_e/training.py`
   and `scripts/phase_e_train_value.py`
2. new pair-weight modes:
   - `group_balance`
   - `confidence_group_balance`
3. balancing label priority:
   - `artifact_mix_source_label`
   - `pair_semantics`
   - `source_tag`

Architecture changes:
1. added `gated_mlp` to `src/ours/phase_b/value_head.py`
2. kept plain `mlp` as the control

Artifact design:
1. mixed small artifact contains:
   - `fanout 1536`
   - `terminal-anchor 512`
2. intent:
   - retain local first-bad geometry,
   - add terminal-completion preference,
   - and explicitly avoid unbalanced optimization.

### 0T.3 Main results on identical `ProcessBench Math 50` slice

| case | auc | pair_acc | anygood_vs_firstbad | good_vs_laterbad | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|---:|
| `E46 baseline` | `0.5335` | `0.4778` | `0.4762` | `0.4783` | `0.2000` | `-0.2688` |
| `PRM terminal-anchor` | `0.5207` | `0.4272` | `0.4921` | `0.4111` | `0.5500` | `-0.0382` |
| `mixed MLP` | `0.5400` | `0.5000` | `0.4921` | `0.5020` | `0.3500` | `-0.1751` |
| `mixed gated_mlp` | `0.4263` | `0.4873` | `0.4762` | `0.4901` | `0.0500` | `-0.2213` |

### 0T.4 What we can trust from this round

1. `mixed MLP` is the best tradeoff tested so far on this benchmark slice.
2. it improves over `E46 baseline` on:
   - overall `auc`
   - overall `pair_acc`
   - broader `good_vs_laterbad`
3. it also preserves some of the terminal repair:
   - `terminal_top1 0.20 -> 0.35`
4. `terminal-anchor only` still wins the pure terminal slice,
   but breaks too much of the broader ranking surface.
5. `gated_mlp` did not help; the problem still looks more like
   data-contract / balancing mismatch than insufficient head complexity.

### 0T.5 Current conclusion

1. this repo is still not fully `RL-ready` in the strict sense.
2. however, the strongest next mainline is now much clearer:
   - mixed local + terminal supervision
   - explicit group-balanced training
   - simple `mlp` head
3. the next scaling move should be:
   - larger mixed artifact
   - optional extra later-bad branch
   - full same-family trust
   - full `ProcessBench GSM8K/Math`
   - and only then conservative RL-readiness claims
4. do **not** scale `gated_mlp` as the default mainline based on current
   evidence.

## 0S. Low-terminal mixed repair retry and runtime blocker audit (2026-03-11)

This round targeted the most plausible next RL-facing repair, not another
generic sweep:

1. keep the strongest confirmed mixed local baseline `E82`,
2. add only a small terminal-completion auxiliary signal,
3. and test whether the repo can move benchmark completion safety without
   breaking the local ranking geometry that already works.

Primary artifacts:
1. rebuilt PRMBench local-diagnostic pair artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_prmbench_localdiag_0311_e46_rebuild_sharedsplit_s42_pairs__f5778317f28b`
2. `E82` same-family baseline:
   - `assets/artifacts/phase_e_samefamily_eval/phase_e_rlpush_lowterm_0311_e82_samefamily_20260311T034620Z`
3. `E46` PRMBench same-family rerank refresh:
   - `assets/artifacts/phase_e_samefamily_eval/phase_e_prmbench_localdiag_0311_e46_samefamily_20260311T033803Z`
4. new repair pair artifacts:
   - `assets/artifacts/phase_e_pairs/phase_e_rlpush_lowterm_0311_e89_e89_ms_prmbench_transfer_mix_terminal10_confwt_warm_e82_seed42_sharedsplit_s42_pairs__9e47a4b941d8`
   - `assets/artifacts/phase_e_pairs/phase_e_rlpush_lowterm_0311_e90_e90_ms_prmbench_transfer_mix_terminal05_confwt_warm_e82_seed42_sharedsplit_s42_pairs__e9ecdd65f92b`

### 0S.1 What was added

Two new named groups were introduced:
1. `E89_MS_PRMBENCH_TRANSFER_MIX_TERMINAL10_CONFWT_WARM_E82_SEED42`
2. `E90_MS_PRMBENCH_TRANSFER_MIX_TERMINAL05_CONFWT_WARM_E82_SEED42`

Their purpose is narrow and controlled:
1. keep `Math-Shepherd fanout`,
2. keep `PRMBench local` support,
3. warm-start from `E82`,
4. and vary only the `terminal anchor` budget.

### 0S.2 Pair-budget diagnosis

Both new repair artifacts were built successfully under shared
`source_sample` splits and `balanced_support_bucket` capping.

1. `E89`
   - total pairs: `10000`
   - train semantics:
     - `first_bad_fanout_prefix_ranking = 4156`
     - `local_modified_process_error_step = 4112`
     - `terminal_completion_anchor = 735`
2. `E90`
   - total pairs: `10000`
   - train semantics:
     - `first_bad_fanout_prefix_ranking = 4331`
     - `local_modified_process_error_step = 4295`
     - `terminal_completion_anchor = 377`

Interpretation:
1. this is a clean terminal-budget probe,
2. the main local support stays almost fixed,
3. only terminal mass changes materially.

### 0S.3 Same-family baseline refresh

Before trusting any repair outcome, two baselines were pinned down:

1. `E82` same-family trust
   - `prompt_pool_top1_accuracy = 0.9633`
   - `prompt_pool_mean_regret = 0.0367`
   - `local_last_safe_top1_accuracy = 0.9124`
   - `local_first_bad_edge_accuracy = 0.9841`
2. `E46` PRMBench same-family rerank refresh
   - `prompt_pool_top1_accuracy = 0.9659`
   - `prompt_pool_mean_regret = 0.0341`

Important metric-contract finding:
1. restoring `positive_step_index / negative_step_index` into PRMBench pairs
   was necessary,
2. but `local_first_bad_edge_accuracy` still stays `N/A`.
3. this is now understood as a metric-definition gap:
   - current same-family local metrics assume
     `last_safe_prefix vs first_bad_prefix`,
   - PRMBench local supervision is
     `same-step correct sibling vs wrong sibling`.

Interpretation:
1. PRMBench local observability improved,
2. but there is still one missing audit metric for same-step sibling correctness.

### 0S.4 Runtime outcome of the new repair attempt

The scientific direction stayed plausible, but runtime robustness did not clear:

1. `E89` full repair run
   - pair construction succeeded,
   - training failed during frozen-backbone feature encoding with:
     - `RuntimeError: CUDA error: unspecified launch failure`
2. initial parallel `E90` attempt
   - showed the same large-run instability pattern,
   - and was intentionally stopped instead of being treated as a valid result
3. safer `E90` retry
   - reused the built pair artifact,
   - disabled feature-cache writes,
   - set:
     - `max_gpu_memory_gib = 48`
     - `max_cpu_memory_gib = 96`
     - `per_device_eval_batch_size = 48`
   - removed the immediate launch failure,
   - but still did not finish within this turn, so no benchmark claim should be
     promoted from it yet.

Interpretation:
1. the repo is still not runtime-stable enough for larger RL-facing warm-start
   repair runs to be treated as routine,
2. so there is still an infrastructure gate before the scientific RL gate.

### 0S.5 Current conclusion after this round

1. `E82` remains the best confirmed mixed local baseline.
2. low-terminal mixed repair remains the right scientific next move.
3. this round did **not** produce a new promoted RL-ready checkpoint.
4. the current blockers are now explicitly twofold:
   - local-vs-terminal scientific tradeoff
   - large-run runtime instability

### 0S.6 Explicit next-step plan

1. add a PRMBench-compatible same-family local metric for
   `same-step sibling correctness`,
2. make large Phase E repair runs robust by default:
   - no-cache mode for one-off sweeps,
   - lower initial encode batch for warm-start repairs,
   - explicit retry/fallback on non-OOM CUDA launch failures,
3. only then rerun `E90`-style low-terminal mixed repair and re-audit it
   against:
   - `E82` same-family trust
   - `ProcessBench GSM8K`
   - `ProcessBench Math`
   - strict transfer diagnosis

## 0R. RL-readiness bounded-support audit refresh (2026-03-11)

This round asked a more operational question than earlier `ACC90` work:

1. which current same-source winners are plausible bounded-support RL priors,
2. and which repair pilot is most worth scaling next?

Primary artifacts:
1. top-candidate audit:
   - `assets/artifacts/phase_e_logs/phase_e_rltops_0311_1124/final_summary.md`
2. repair-pilot audit:
   - `assets/artifacts/phase_e_logs/phase_e_rlrepairs_0311_1124/final_summary.md`

### 0R.1 Top-candidate audit

Audited checkpoints:
1. `ms_e68`
2. `ms_e43`
3. `prm_e46`
4. `prm_e78`

Key results:
1. `ms_e43`
   - `pool_top1 = 0.9648`
   - `local_first_bad = 0.9702`
   - `rej40_gain = 0.0352`
   - `pb_gsm_auc = 0.6245`
   - `pb_math_auc = 0.6341`
   - assessment:
     - `provisionally_rl_ready`
2. `prm_e46`
   - `pool_top1 = 0.9659`
   - `rej40_gain = 0.0341`
   - `pb_gsm_auc = 0.6264`
   - `pb_math_auc = 0.6053`
   - assessment:
     - `provisionally_rl_ready`
3. `ms_e68`
   - same-family geometry remains excellent,
   - benchmark remains positive,
   - but rejection gain is only:
     - `0.0207`
   - assessment:
     - `useful_signal_but_not_rl_ready`
4. `prm_e78`
   - same-family metrics remain very strong,
   - but benchmark safety falls to:
     - `gsm8k_auc = 0.5398`
     - `math_auc = 0.5117`
   - assessment:
     - `samefamily_only_not_benchmark_safe`

Interpretation:
1. current bounded-support RL candidates are:
   - `ms_e43`
   - `prm_e46`
2. `prm_e78` is now the clearest warning that stronger same-source fitting can
   still make a checkpoint less benchmark-safe.

### 0R.2 Repair-pilot audit

Audited repair pilots:
1. `ms_grid_micro`
2. `ms_ta_micro`
3. `prm_ta_smoke`

Key results:
1. `ms_grid_micro`
   - `pool_top1 = 0.9982`
   - `local_first_bad = 0.9914`
   - `pb_gsm_auc = 0.5891`
   - `pb_math_auc = 0.5559`
   - weakness:
     - `rej40_gain = 0.0018`
   - assessment:
     - `useful_signal_but_not_rl_ready`
2. `ms_ta_micro`
   - abstention utility improves:
     - `rej40_gain = 0.0571`
   - but same-family ordering degrades:
     - `pool_top1 = 0.7307`
     - `local_first_bad = 0.7904`
   - assessment:
     - `useful_signal_but_not_rl_ready`
3. `prm_ta_smoke`
   - `rej40_gain = 0.0627`
   - but benchmark ranking collapses:
     - `gsm8k_auc = 0.4778`
     - `math_auc = 0.4691`
   - assessment:
     - `samefamily_only_not_benchmark_safe`

Interpretation:
1. `ms_grid_micro` is the best current repair direction to scale.
2. terminal-anchor style repairs improve abstention more than they improve
   trustworthy ranking geometry.
3. the next repair stage should preserve the `ms_grid_micro` local geometry
   while explicitly raising rejection utility.

## 0Q. ProcessBench transfer engineering corrections and terminal-anchor audit (2026-03-11)

This round fixed the evaluation plumbing first, before trusting any new repair result.

Completed corrections:
1. `ProcessBench` subset eval is now stratified instead of raw first-`N`.
   - file:
     - `src/ours/phase_e/benchmark_eval.py`
2. `phase_e_analyze_processbench_failures.py` now analyzes the exact scored
   subset rather than rebuilding from the full benchmark file.
3. `Phase E` pair artifacts now record:
   - `global_cap_mode`
   - `global_cap_summary`
   - `overall_summary_before_global_cap`
4. `Math-Shepherd` terminal-anchor runs no longer silently lose all-positive
   repairs under `max_pairs_per_source`.
   - file:
     - `src/ours/phase_d/external_pairs_adapters.py`

Critical source-order finding:
1. in the current `Math-Shepherd` mirror, the first all-positive trajectory
   appears only at source row `121569`
2. total all-positive rows observed:
   - `160920`
3. consequence:
   - any naive stream-head source cap such as `max_pairs_per_source=20000`
     guarantees `terminal_completion_anchor = 0`
     even when the config nominally enables terminal anchors

Verified post-fix pair artifact:
1. artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_processbench_terminal_focus_0311_e83_ms_processbench_transfer_terminal_seed42_e83_ms_processbench_transfer_terminal_seed42_sharedsplit_s42_pairs__8b75a88516bc`
2. semantics before final cap:
   - `local_first_bad_edge = 9985`
   - `terminal_completion_anchor = 9992`
3. semantics after final cap:
   - `local_first_bad_edge = 4000`
   - `terminal_completion_anchor = 4000`
4. train split semantics:
   - `local_first_bad_edge = 3595`
   - `terminal_completion_anchor = 3603`

Interpretation:
1. the terminal-anchor recipe is now *actually present* in the train artifact
2. any earlier negative read on `E83`-style terminal-anchor repairs is not
   trustworthy unless that run's artifact explicitly shows non-zero
   `terminal_completion_anchor`

Corrected rerun outcome on the fixed `ProcessBench 96` subset:
1. baseline `E79`:
   - `gsm8k pair_acc = 0.6088`
   - `math pair_acc = 0.4558`
   - `gsm8k all_correct_top1 = 0.1087`
   - `math all_correct_top1 = 0.0000`
2. corrected `E83 terminal`:
   - held-out `pair_acc = 0.7731`
   - `gsm8k pair_acc = 0.3163`
   - `math pair_acc = 0.3273`
   - `gsm8k all_correct_top1 = 0.6304`
   - `math all_correct_top1 = 0.5641`
3. corrected `E84 fanout + terminal`:
   - held-out `pair_acc = 0.7808`
   - `gsm8k pair_acc = 0.3299`
   - `math pair_acc = 0.3233`
   - `gsm8k all_correct_top1 = 0.6739`
   - `math all_correct_top1 = 0.5897`

Interpretation:
1. the corrected terminal-anchor repair is now real and very strong on the
   all-correct slice
2. but at the current `50/50` mixture weight it over-corrects and destroys the
   broader good-vs-bad ranking surface
3. `fanout + terminal` improves geometric alignment over `terminal` alone, but
   still fails badly on overall `ProcessBench` ranking
4. next mainline should lower terminal-anchor mass rather than increase it

## 0R. Low-terminal RL-facing repair probe (`E87`, 2026-03-11)

New infrastructure added:
1. `step_label_terminal_anchor_fraction`
   - added to:
     - `PairBuildConfig`
     - `scripts/phase_e_prepare_pairs.py`
     - `scripts/run_phase_e_suite.sh`
2. purpose:
   - keep terminal anchors as a bounded auxiliary signal instead of a `50/50`
     co-equal training pool

Main new run:
1. group:
   - `E87_MS_PROCESSBENCH_TRANSFER_FANOUT_TERMINAL10_CONFWT_SEED42`
2. train artifact semantics:
   - `first_bad_fanout_prefix_ranking = 6487`
   - `terminal_completion_anchor = 735`
3. held-out fit:
   - `pair_acc = 0.8823`
   - `auc = 0.8647`
4. `ProcessBench GSM8K 96`:
   - `pair_acc = 0.4932`
   - `auc = 0.4410`
   - `all_correct_top1 = 0.3696`
   - `all_correct_gap = -0.0819`
5. `ProcessBench Math 96`:
   - `pair_acc = 0.4217`
   - `auc = 0.4467`
   - `all_correct_top1 = 0.2051`
   - `all_correct_gap = -0.1314`
6. same-family trust:
   - `prompt_pool_top1 = 0.6597`
   - `local_first_bad_acc = 0.2914`

Strict diagnosis:
1. `assets/artifacts/phase_e_transfer_diag/phase_e_transfer_diag_e87_0311_00/summary.md`
2. result:
   - `strict_rl_ready = 0`
   - `assessment = not_rl_ready_terminal_completion_risk`

Interpretation:
1. reducing terminal mass from `50%` to `10%` is directionally correct
2. it gives the best current benchmark tradeoff
3. but same-family decision utility is still far too weak for `Phase F`
4. therefore the repository is still **not RL-ready**
5. the next repair axis must be:
   - semantics-aware weighting / curriculum,
   - not just more pair-mix tuning

## 0P. Latest Diagnosis Update (2026-03-11, R-PRM Compact Train-Fit vs Held-Out Gap)

### 0P.1 What this round asked

Earlier `R-PRM` diagnosis had already established three negative facts:
1. legacy long-form truncation is no longer the main blocker,
2. simple verdict-polarity rebalance is not enough,
3. the source still stays far below the `ACC90` gate.

The missing question was narrower and more decisive:
1. is current `R-PRM compact_verdict` actually unlearnable under the present
   frozen-feature value-head stack,
2. or can it fit its own pair distribution well and the remaining problem is
   held-out generalization?

### 0P.2 Contract and infrastructure state used in this check

This round used the repaired compact contract on the full dataset:
1. pair artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_rprm_full_compact_fix__8d69afd6dba5`
2. key counts:
   - total pairs: `83159`
   - train: `74842`
   - validation: `8317`
3. important duplication audit:
   - train:
     - rows: `74842`
     - unique prompts: `74501`
     - mean pairs per prompt: `1.0046`
   - validation:
     - rows: `8317`
     - unique prompts: `8313`
     - mean pairs per prompt: `1.0005`

Interpretation:
1. prompt duplication exists but is tiny,
2. same-source `>90%` on this source cannot be waved away as simple prompt
   memorization.

### 0P.3 Important bug fixed before the probes

One real training-gate bug was fixed in:
1. `src/ours/phase_e/training.py`

Old behavior:
1. any nonzero
   - `frac_pairs_identical_after_truncation`
   - or `frac_pairs_first_diff_after_cutoff`
   caused a hard failure,
2. even if the fraction was only around `0.0015`.

New behavior:
1. these diagnostics now respect the same configured tolerance as
   `frac_pairs_over_limit`.

Why this matters:
1. it removes false negatives where a healthy pair artifact was being rejected
   because of a tiny truncation tail.

### 0P.4 Train-distribution fit ceiling probe

Run:
1. `assets/artifacts/phase_e_runs/phase_e_rprm_trainfit_probe_0310_s4k_20260310T160730Z`

Configuration summary:
1. train pairs:
   - `4000`
2. eval pairs:
   - `1000`
3. eval source:
   - sampled from the training distribution itself
4. head:
   - `MLP`
   - hidden size `2048`
5. objective:
   - `joint`
6. max length:
   - `2048`

Result:
1. `pair_acc = 0.9090`
2. `auc = 0.9131`
3. `mean_margin = 0.2932`

Interpretation:
1. current `R-PRM compact_verdict` is **not** fundamentally unlearnable.
2. the current feature extractor + head + objective stack can already fit the
   train-distribution pair relation to `>90%`.

### 0P.5 Matching true held-out probe

Run:
1. `assets/artifacts/phase_e_runs/phase_e_rprm_heldout_repair_0310_s4k_20260310T164443Z`

Configuration:
1. intentionally matched the train-fit probe on:
   - source
   - head
   - objective
   - length
   - optimization
2. only the eval split changed:
   - true validation pairs instead of train-distribution pairs

Result:
1. `pair_acc = 0.6280`
2. `auc = 0.6508`
3. `mean_margin = 0.1063`

Interpretation:
1. the `R-PRM` problem is now sharply localized:
   - it is **not** mainly head capacity,
   - **not** mainly training duration,
   - **not** mainly residual truncation.
2. the remaining blocker is:
   - held-out generalization under the current `compact_verdict` contract.

### 0P.6 Operational side note: OOM behavior

During the train-fit probe:
1. feature encoding hit real GPU pressure on the crowded server,
2. OOM backoff triggered:
   - `bs=12 -> 6 -> 3`,
3. and the run completed successfully.

Interpretation:
1. current Phase E encoding-layer OOM backoff is functioning,
2. the main scientific conclusion here is therefore not confounded by a
   crashed or partially skipped run.

### 0P.7 Updated repository conclusion for `R-PRM`

The strongest current statement is now:
1. `R-PRM compact_verdict` is learnable,
2. but it is not currently a held-out `ACC90` source,
3. and the bottleneck is a supervision-contract / generalization mismatch.

This is a much stronger diagnosis than either of the earlier weaker claims:
1. "it just needs more length",
2. "it just needs more head capacity",
3. or "it is simply random / unusable."

## 0O. Latest Diagnosis Update (2026-03-11, ProcessBench Smoke Repairs Split The Failure Modes)

### 0O.1 What this round answered

This round asked a narrower and more operational question:
1. on `ProcessBench`, is the bottleneck mainly:
   - missing non-local good-vs-bad geometry,
   - or missing terminal-completion supervision?

The answer is now:
1. **both matter**,
2. but they repair **different slices**,
3. and neither one alone is sufficient.

### 0O.2 New diagnostic layer

New script:
1. `scripts/phase_e_compare_processbench_transfer.py`

Why it matters:
1. a single benchmark AUC was hiding which part of transfer was broken.
2. the new compare view puts these side by side:
   - `anygood_vs_firstbad`
   - `good_vs_laterbad`
   - `terminal_top1`
   - `terminal_gap`
   - training-geometry fractions

### 0O.3 `Math-Shepherd` geometry repairs: useful but not sufficient

Main smoke artifact:
1. `assets/artifacts/phase_e_transfer_compare/processbench_math50_compare_all_0311_20260310T170538Z/summary.md`

On the same `ProcessBench Math 50` subset:
1. baseline `E68`
   - `auc=0.4251`
   - `first_edge=0.4615`
   - `terminal_top1=0.0500`
2. `Math-Shepherd grid` smoke
   - `auc=0.4262`
   - `first_edge=0.4615`
   - `terminal_top1=0.0500`
3. `Math-Shepherd fanout` smoke
   - `auc=0.4199`
   - `first_edge=0.5000`
   - `terminal_top1=0.0500`

Interpretation:
1. `fanout` helps exactly the local slice it should help:
   - `anygood_vs_firstbad`
   - and slightly `first_edge`
2. `grid` reduces pair-type mismatch on paper,
   but does not deliver benchmark lift on this subset.
3. neither geometry repair changes terminal collapse.

### 0O.4 `PRMBench terminal-anchor` repair: first strong positive terminal signal

Main artifact:
1. `assets/artifacts/phase_e_transfer_compare/processbench_prm_math50_compare_0311_20260310T171808Z/summary.md`

On the same `ProcessBench Math 50` subset:
1. baseline `E46`
   - `terminal_top1=0.2000`
   - `terminal_gap=-0.2688`
   - `pair_acc=0.4778`
   - `auc=0.5335`
2. `PRMBench terminal-anchor` smoke
   - `terminal_top1=0.5500`
   - `terminal_gap=-0.0382`
   - `pair_acc=0.4272`
   - `auc=0.5207`

Interpretation:
1. terminal-anchor supervision is the first repair that clearly moves the
   all-correct terminal slice in the desired direction.
2. but it damages broader good-vs-bad ranking, especially:
   - `good_vs_laterbad`
3. so the terminal undervaluation diagnosis was correct,
   but terminal anchors alone are not the full fix.

### 0O.5 Updated practical conclusion

The repository now has a much sharper ProcessBench diagnosis:
1. `Math-Shepherd` geometry repairs mostly improve the local first-bad
   neighborhood.
2. `PRMBench` terminal anchors strongly improve the terminal all-correct slice.
3. the next mainline repair should therefore be a **mixed supervision recipe**:
   - terminal anchors
   - plus later-bad / broader good-vs-bad coverage
   - with careful weighting

This is a stronger and more useful conclusion than:
1. "try more benchmark-like pairs",
2. or "just add terminal anchors",
because the smoke runs now show exactly what each one fixes and what each one breaks.

## 0N. Latest Diagnosis Update (2026-03-11, Should MCTS Be The Next Mainline Fix?)

### 0N.1 Question

After the `ProcessBench` alignment audit, the next natural question is:
1. can `MCTS` directly solve the current Phase E transfer problem?

### 0N.2 Literature-guided answer

Short answer:
1. **not as the next mainline fix**,
2. but **possibly yes as a later data-construction or test-time branch**.

The main literature pattern is:
1. `ReST-MCTS*`
   - supports tree search as a way to harvest better traces / step targets for
     later training.
2. `Tree-PLV`
   - supports tree-based pair construction for better step-level ranking
     supervision.
3. `Rewarding Progress`
   - supports progress-aware process targets, which makes search more useful
     than naive correctness-only rollouts.
4. `MCTS-Judge`
   - supports MCTS as test-time judge/search scaling.

What this does **not** imply:
1. that adding `MCTS` to the current misaligned local/terminal/grid setup will
   automatically fix benchmark transfer.

The main warning signal from the literature is:
1. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   shows that naive synthetic MC-style PRM labels can generalize poorly.
2. so an uncontrolled tree-search branch can easily become a new noisy-label
   pipeline.

### 0N.3 Repository-specific diagnosis

For this repository, the current measured bottleneck is:
1. **supervision semantics mismatch**
   - local-only supervision under-teaches terminal completion,
   - terminal-only repair over-corrects,
   - grid-only repair helps a different side of the benchmark.

Therefore:
1. the present problem is not primarily "lack of search budget",
2. it is "training supervision and benchmark semantics are not yet aligned".

### 0N.4 Practical conclusion

The correct reading is:
1. `MCTS` should **not** replace the current `local + terminal + optional grid`
   repair path.
2. If introduced later, the two defensible forms are:
   - offline tree harvesting for higher-margin local / terminal pairs
   - or a separate test-time judge/search baseline
3. So the current mainline remains:
   - repair the supervision contract first,
   - and keep `MCTS` as a branch experiment rather than a reset of Phase E.

Primary references:
1. `ReST-MCTS*`
   - https://arxiv.org/abs/2406.03816
   - https://github.com/THUDM/ReST-MCTS
2. `Advancing Process Verification for Large Language Models via Tree-Based Preference Learning`
   - https://arxiv.org/abs/2407.00390
   - https://aclanthology.org/2024.emnlp-main.125/
3. `Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning`
   - https://arxiv.org/abs/2410.08146
4. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   - https://arxiv.org/abs/2501.07301
5. `MCTS-Judge: Test-Time Scaling in LLM-as-a-Judge for Code Correctness Evaluation`
   - https://arxiv.org/abs/2502.12468

## 0M. Latest Diagnosis Update (2026-03-11, ProcessBench Alignment Audit + Micro Repair Pilots)

### 0M.1 What this round was trying to answer

This round focused on one narrower but more important question than generic
same-source learnability:
1. why do strong same-source value heads still transfer weakly to `ProcessBench`?
2. is the failure mainly caused by:
   - missing terminal-completion supervision,
   - missing broader good-vs-bad prefix coverage,
   - or both?

### 0M.2 New diagnosis artifact

New script:
1. `scripts/phase_e_analyze_processbench_failures.py`

It compares:
1. training-pair semantics
2. ProcessBench structure
3. bucketed benchmark behavior

Representative outputs:
1. `assets/artifacts/phase_e_processbench_analysis/ms_e68_pb_math_v2_0311_20260310T160909Z/summary.md`
2. `assets/artifacts/phase_e_processbench_analysis/prm_e46_pb_math_v2_0311_20260310T160909Z/summary.md`

What it established:
1. `ProcessBench` is not only a local first-bad benchmark.
2. It contains a large all-correct block:
   - GSM8K: `0.4825`
   - Math: `0.4060`
3. Both current strong baselines had effectively zero terminal-anchor supervision:
   - `E68`: pure `local_first_bad_edge`
   - `E46`: pure `local_modified_process_error_step`
4. Therefore the previous qualitative diagnosis is now hard evidence:
   - both baselines are structurally under-supervising the
     `all-correct terminal completion` part of ProcessBench.

### 0M.3 New repair artifacts

New scripts / artifacts:
1. `scripts/phase_e_prepare_prmbench_terminal_anchor_pairs.py`
   - artifact:
     - `assets/artifacts/phase_e_pairs/phase_e_prmbench_terminal_anchor_full_0311__192ca71fd301`
2. `scripts/phase_e_prepare_mathshepherd_terminal_anchor_pairs.py`
   - capped artifact:
     - `assets/artifacts/phase_e_pairs/phase_e_ms_terminal_anchor_cap20k_diag_0311__6d57b0d4b490`
3. benchmark-aligned Math-Shepherd grid artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_ms_grid_cap40k_diag_0311__4f87d4f4cea6`

Their intended semantics are now clearly separated:
1. `PRMBench + terminal anchors`
   - keep local explicit error-step supervision
   - add full-correct vs shorter-safe prefix anchors
2. `Math-Shepherd + terminal anchors`
   - keep local first-bad-edge supervision
   - add all-positive full-completion anchors
3. `Math-Shepherd grid`
   - expose broader good-vs-bad prefix relations
   - but still no explicit terminal-anchor signal

### 0M.4 Micro repair pilot results

These were intentionally run as micro warm-start pilots, not promotion runs.
The goal was to test directionality under shared-server constraints.

#### Baselines

1. `E46` baseline on `ProcessBench`
   - GSM8K:
     - `pair_acc = 0.6701`
     - `auc = 0.6264`
     - `first_edge = 0.6706`
     - `all_correct_last = 0.2924`
   - Math:
     - `pair_acc = 0.5653`
     - `auc = 0.6053`
     - `first_edge = 0.6096`
     - `all_correct_last = 0.2452`

2. `E68` baseline on `ProcessBench`
   - GSM8K:
     - `pair_acc = 0.6385`
     - `auc = 0.5885`
     - `first_edge = 0.6294`
     - `all_correct_last = 0.5626`
   - Math:
     - `pair_acc = 0.5809`
     - `auc = 0.5547`
     - `first_edge = 0.5553`
     - `all_correct_last = 0.5895`

#### Pilot A: `PRMBench + terminal anchors`, warm-start from `E46`

Artifacts:
1. train run:
   - `assets/artifacts/phase_e_runs/phase_e_pbta_warm_e46_micro_0311_20260310T162646Z`
2. benchmark eval:
   - `assets/artifacts/phase_e_eval/phase_e_pbta_warm_e46_micro_pb_gsm8k_0311_20260310T163036Z/metrics.json`
   - `assets/artifacts/phase_e_eval/phase_e_pbta_warm_e46_micro_pb_math_0311_20260310T163037Z/metrics.json`

Result:
1. terminal completion improved clearly:
   - GSM8K `0.2924 -> 0.4196`
   - Math `0.2452 -> 0.3492`
2. but local/global ranking softened:
   - GSM8K `auc 0.6264 -> 0.6014`
   - Math `auc 0.6053 -> 0.5906`
   - GSM8K `first_edge 0.6706 -> 0.6471`
   - Math `first_edge 0.6096 -> 0.6013`

Interpretation:
1. terminal anchors are doing the intended thing,
2. but they trade against local error discrimination when used alone.

#### Pilot B: `Math-Shepherd + terminal anchors`, warm-start from `E68`

Artifacts:
1. train run:
   - `assets/artifacts/phase_e_runs/phase_e_msta_warm_e68_micro_0311_20260310T163102Z`
2. benchmark eval:
   - `assets/artifacts/phase_e_eval/phase_e_msta_warm_e68_micro_pb_gsm8k_0311_20260310T163337Z/metrics.json`
   - `assets/artifacts/phase_e_eval/phase_e_msta_warm_e68_micro_pb_math_0311_20260310T163336Z/metrics.json`

Result:
1. terminal completion improved dramatically:
   - GSM8K `0.5626 -> 0.7590`
   - Math `0.5895 -> 0.7663`
2. but local / good-vs-bad discrimination degraded materially:
   - GSM8K `auc 0.5885 -> 0.5527`
   - Math `auc 0.5547 -> 0.5350`
   - GSM8K `first_edge 0.6294 -> 0.6059`
   - Math `first_edge 0.5553 -> 0.5324`

Interpretation:
1. pure terminal-anchor repair is too strong here,
2. and it over-corrects toward completion preference.

#### Pilot C: `Math-Shepherd all_good_vs_all_bad` grid, warm-start from `E68`

Artifacts:
1. train run:
   - `assets/artifacts/phase_e_runs/phase_e_msgrid_warm_e68_micro_0311_20260310T163400Z`
2. benchmark eval:
   - `assets/artifacts/phase_e_eval/phase_e_msgrid_warm_e68_micro_pb_gsm8k_0311_20260310T163559Z/metrics.json`
   - `assets/artifacts/phase_e_eval/phase_e_msgrid_warm_e68_micro_pb_math_0311_20260310T163559Z/metrics.json`

Result:
1. local / late-bad side held up better than the terminal-anchor-only repair:
   - GSM8K `pair_acc 0.6385 -> 0.6436`
   - GSM8K `first_edge 0.6294 -> 0.6235`
   - Math `pair_acc 0.5809 -> 0.5839`
   - Math `first_edge 0.5553 -> 0.5595`
2. but terminal gap remained largely unresolved:
   - Math `all_correct terminal top1` stayed poor:
     - baseline `0.0517`
     - grid pilot `0.0591`
   - mean terminal score only moved slightly:
     - Math `0.5895 -> 0.6011`

Interpretation:
1. the grid repair mainly targets the good-vs-bad prefix side,
2. but it does not fix terminal completion on its own.

### 0M.5 Updated conclusion

This round establishes a much sharper ProcessBench diagnosis:
1. `terminal anchors` fix the terminal side,
2. `good_bad_prefix_grid` helps the broader prefix-ranking side,
3. but neither alone solves both.

So the next credible Phase E repair should be:
1. **mixed supervision or staged curriculum**
   - local error-step pairs
   - plus limited terminal anchors
   - plus optional grid-style broader good/bad coverage
2. not another generic LR / dropout / batch-size sweep.

That is the main new scientific information from this round.

## 0L. Latest Diagnosis Update (2026-03-11, Phase E Same-Source ACC90/95 And R-PRM Root Cause)

### 0L.1 What this round answered

This round tightened three separate questions that should not be mixed:
1. can `Math-Shepherd` reach stable same-source `>95%` pair accuracy?
2. can `PRMBench_Preview` reach same-source `>95%` pair accuracy with a dataset-specific recipe?
3. if `R-PRM` still underperforms, is the bottleneck still truncation, or has it moved to a deeper contract/objective mismatch?

### 0L.2 Same-source ACC results now clearly separate the sources

#### `Math-Shepherd`

Artifacts:
1. `assets/artifacts/phase_e_logs/phase_e_ms_acc90_full_0310_1914_e41_ms_acc90_mlp_rank_seed3/final_summary.md`
2. `assets/artifacts/phase_e_logs/phase_e_ms_acc95_push_0310_2146/final_summary.md`

Main reading:
1. same-source `Math-Shepherd` is already solved under the current Phase E trainer family.
2. representative high-water marks:
   - `E41`
     - `pair_acc=0.9850`
     - `auc=0.9034`
   - `E68`
     - `pair_acc=0.9725`
     - `auc=0.9415`
3. pair-error analysis on the best same-source run shows that remaining failures are concentrated on:
   - later `first_bad_edge` positions,
   - longer step chains,
   rather than a generic inability to rank.

#### `PRMBench_Preview`

Artifacts:
1. `assets/artifacts/phase_e_logs/phase_e_prmbench_acc90_full_0310_1914/final_summary.md`
2. `assets/artifacts/phase_e_logs/phase_e_prmbench_acc95_push_0310_2359/final_summary.md`

Main reading:
1. `PRMBench_Preview` also supports strong same-source learning.
2. the best current single-seed push already crosses the `95%` pair-accuracy target:
   - `E78_PRMBENCH_ACC95_JOINT_OVERFIT_SEED42`
     - `pair_acc=0.9521`
     - `auc=0.9071`
3. therefore the current Phase E stack is not generically weak.
4. it is source-sensitive.

### 0L.3 `R-PRM` deep diagnosis: truncation is no longer the main blocker

Artifacts:
1. deep-diagnostic summary:
   - `assets/artifacts/phase_e_logs/phase_e_rprm_deep_diag_0310_2359/final_summary.md`
2. compact length/root-cause audit:
   - `assets/artifacts/phase_e_logs/phase_e_rprm_diag_0310_2019/final_summary.md`

Deep-diagnostic findings:
1. on the repaired small compact artifact, `1536` and `2048` are already truncation-clean.
2. the best old compact recipe remains:
   - `C9_MLP_BCE_2048`
     - `pair_acc=0.6694`
     - `auc=0.6571`
3. objective ordering is now clear:
   - `pair_bce_only` > `joint` > `ranking_only`
4. this means compact `R-PRM` behaves more like a binary verdict-fitting source than a generic ranking-pair source.

### 0L.4 New polarity-repair experiments on `R-PRM`

New artifacts:
1. repaired `compact_correctness` pair artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_rprm_correctness_diag_0310_2341_pairs__b835692e7df6`
2. runs:
   - `assets/artifacts/phase_e_runs/phase_e_rprm_correctness_diag_0310_2341_bce2048_20260310T154119Z`
   - `assets/artifacts/phase_e_runs/phase_e_rprm_correctness_diag_0310_2341_bce2048_vbal_20260310T154500Z`
   - `assets/artifacts/phase_e_runs/phase_e_rprm_correctness_diag_0310_2341_joint2048_vbal_logit_20260310T154937Z`
3. error profiles:
   - `assets/artifacts/phase_e_error_analysis/rprm_correctness_bce2048_profile_20260310T155010Z/summary.md`
   - `assets/artifacts/phase_e_error_analysis/rprm_correctness_bce2048_vbal_profile_20260310T155010Z/summary.md`
   - `assets/artifacts/phase_e_error_analysis/rprm_correctness_joint2048_vbal_profile_20260310T155010Z/summary.md`

What these runs established:
1. `compact_correctness + BCE + 2048`
   - `pair_acc=0.6481`
   - `auc=0.6519`
   - this does **not** beat the old `compact_verdict + BCE + 2048` baseline.
2. `compact_correctness + BCE + verdict_balance`
   - `pair_acc=0.5926`
   - `auc=0.5765`
   - naive verdict balancing hurts overall performance.
3. `compact_correctness + joint + logit + verdict_balance`
   - `pair_acc=0.5926`
   - `auc=0.6306`
   - but importantly, it almost removes the old polarity asymmetry:
     - chosen=`no`: `0.5968`
     - chosen=`yes`: `0.5870`

Interpretation:
1. the repository has now separated two different `R-PRM` issues:
   - **issue A: polarity bias**
   - **issue B: weak compact supervision contract**
2. `compact_correctness + verdict_balance + joint/logit` largely fixes issue A.
3. but overall accuracy still remains far below `ACC90`.
4. so the main remaining problem is no longer truncation or simple verdict imbalance.
5. it is that the current compact contract does not expose enough useful signal to the frozen-head scorer.

### 0L.5 Updated conclusion

The current Phase E evidence now supports:
1. `Math-Shepherd`
   - strong same-source source
2. `PRMBench_Preview`
   - strong same-source source
3. `R-PRM`
   - not a primary same-source high-accuracy source under the current compact contract
   - useful mainly as a source-specific diagnosis case

So the project should stop framing `R-PRM` as "one more generic pair dataset" and instead treat it as:
1. a compact-verdict / verifier-style source with a different supervision geometry,
2. one that likely needs a different model contract than the current frozen feature scorer.

## 0K. Latest Diagnosis Update (2026-03-10, RL-Readiness Audit Of Current Value-Head Candidates)

### 0K.1 What this round was trying to answer

The repository had already shown that some sources support very strong
same-source held-out pair fitting.

But that still did not answer the more important practical question:
1. is any current value head good enough for *conservative RL-style use*?

So this round asked:
1. among the repository's strongest current checkpoints,
2. which ones survive stronger decision-style offline checks,
3. and has the project now reached a level that is usable for bounded-support
   RL, even if not for unrestricted reward optimization?

### 0K.2 Audit design

A new wrapper was added:
1. `scripts/run_phase_e_rl_readiness_suite.sh`

Main command:
1. `CUDA_VISIBLE_DEVICES=2 ACTIVE_PHASE_E_RL_GROUP=RR4_COMPARE_CURRENT_TOPS RUN_PREFIX=phase_e_rl_readiness_0310_2338 bash scripts/run_phase_e_rl_readiness_suite.sh`

This compared three current top candidates:
1. `ms_e68`
   - strongest current same-source Math-Shepherd winner
2. `ms_e14`
   - benchmark-aware Math-Shepherd trust candidate
3. `prm_e46`
   - strongest current PRMBench same-source winner

Each candidate was tested on:
1. same-family prompt-pool reranking
2. rejection / abstention utility
3. best-of-N pressure
4. `ProcessBench GSM8K`
5. `ProcessBench Math`

Then the two most promising candidates got an extra higher-pressure recheck.

### 0K.3 One summary-layer bug was discovered and fixed during the audit

The first generated wrapper summary incorrectly read benchmark AUC from a
generic `auc` key.

But `ProcessBench` summary files store:
1. `pair_auc_good_vs_bad`

So the wrapper was fixed, and the final summary was regenerated from the
already-finished benchmark artifacts without rerunning the whole audit.

This matters because the corrected benchmark values materially change the
interpretation of the strongest candidates.

### 0K.4 Main audit results

Artifact:
1. `assets/artifacts/phase_e_logs/phase_e_rl_readiness_0310_2338/final_summary.md`

#### `ms_e68` : same-family excellent, benchmark acceptable but not dominant

Same-family trust:
1. `prompt_pool_top1_accuracy = 0.9793`
2. `random_top1_baseline = 0.4995`
3. `top1_lift_over_random = 0.4798`
4. `local_first_bad_edge_accuracy = 0.9779`
5. `pressure@8 top1 = 1.0000`

ProcessBench:
1. `gsm8k_auc = 0.5885`
2. `math_auc = 0.5547`

Interpretation:
1. this checkpoint is already very strong as a same-family reranker,
2. and its benchmark behavior is clearly above random,
3. but it is not the cleanest overall RL-facing candidate because its full
   audit profile is less balanced than `prm_e46`.

#### `ms_e14` : benchmark-aware Math-Shepherd candidate still fails benchmark safety

Same-family trust:
1. `prompt_pool_top1_accuracy = 0.8584`
2. `local_first_bad_edge_accuracy = 0.8664`
3. `rejection@0.4 top1 = 0.9913`

ProcessBench:
1. `gsm8k_auc = 0.5026`
2. `math_auc = 0.5138`

Interpretation:
1. this run is useful signal,
2. but it is not benchmark-safe enough,
3. so it should not be promoted as the repository's main RL-facing candidate.

#### `prm_e46` : first current checkpoint that clears a conservative bounded-support gate

Same-family trust:
1. `prompt_pool_top1_accuracy = 0.9659`
2. `random_top1_baseline = 0.4959`
3. `top1_lift_over_random = 0.4701`
4. `rejection@0.4 top1 = 1.0000`
5. `pressure@8 top1 = 0.9375`

ProcessBench:
1. `gsm8k_auc = 0.6264`
2. `math_auc = 0.6053`

Interpretation:
1. this is the first current checkpoint family that looks strong both:
   - inside its own source family,
   - and on the repository's stronger benchmark-native recheck
2. under the wrapper's conservative internal heuristic, this row becomes:
   - `provisionally_rl_ready`

### 0K.5 Extra pressure-stress recheck

Artifacts:
1. `assets/artifacts/phase_e_samefamily_eval/phase_e_rl_pressure_stress_0310_2340_ms_e68_20260310T153847Z/summary.md`
2. `assets/artifacts/phase_e_samefamily_eval/phase_e_rl_pressure_stress_0310_2340_prm_e46_20260310T153848Z/summary.md`

#### `E68` stress result

1. `rejection@0.4 = 1.0000`
2. `rejection@0.1 = 1.0000`
3. `pressure@12 = 1.0000`

Interpretation:
1. `E68` is not fragile under stronger same-family decision pressure;
2. its weaker overall RL audit result is therefore **not** because the
   checkpoint collapses when selection gets stronger;
3. it is mainly because its benchmark-native behavior is only moderate, not
   because same-family utility is weak.

#### `E46` stress result

1. `rejection@0.4 = 1.0000`
2. `rejection@0.1 = 1.0000`
3. `pressure@4 = 0.9411`
4. `pressure@8 = 0.9375`

Interpretation:
1. the current best PRMBench checkpoint keeps strong selection utility under
   higher pressure,
2. and it does so without collapsing into near-random behavior.

### 0K.6 Final judgement

The right answer is **not** a flat yes/no.

The most accurate reading is:
1. the repository has **not** yet proved that current value-head training is
   safe for aggressive, open-ended, high-weight RL optimization;
2. but it **has** now reached a level where at least one checkpoint family
   (`prm_e46`) is usable for **bounded-support RL-style control**:
   - conservative reranking
   - rejection / abstention
   - low-weight reward prior
   - math-family process search / filtering

So the current status should be narrated as:
1. **provisionally RL-usable under narrow constraints**
2. **not yet generally RL-ready as a trusted reward model**

### 0K.7 Why the claim must still stay bounded

Even after this positive result, several things are still missing:
1. no true closed-loop RL experiment has yet been run with this candidate
2. no direct reward-hacking / shortcut-exploitation test has yet been passed
3. robustness is still shown only on nearby benchmark families, not across
   broader domains
4. `Math-Shepherd` and `PRMBench` still disagree on what the best checkpoint
   family is, so source-specificity remains real

### 0K.8 Immediate next-step plan

1. Promote `prm_e46` as the current **bounded-support RL candidate**.
2. Use it first only in conservative modes:
   - rerank
   - rejection gate
   - clipped / low-weight reward shaping
3. Before calling anything "fully RL-ready", require at least one new stage:
   - actual closed-loop conservative search / RL improvement test
   - plus reward-scale clipping / exploitation audit
4. Keep `ms_e68` as the strongest Math-Shepherd local-faithfulness control,
   but not as the main RL-facing promoted checkpoint.

## 0J. Latest Diagnosis Update (2026-03-10, Math-Shepherd ACC95 Verification And Push Matrix)

### 0J.1 What this round was actually trying to answer

The user request was framed as:
1. re-check current `Math-Shepherd` performance,
2. try repairs if needed,
3. and reach `95%` held-out pair accuracy,
4. while staying careful about shared-server OOM risk.

But the repository's own recent artifacts already showed that
`Math-Shepherd` had crossed this target:
1. `E42_MS_ACC90_MLP_JOINT_SEED3`
   - `mean_pair_acc = 0.963106`
2. `E43_MS_ACC90_MLP_HIGHCONF_SEED3`
   - `mean_pair_acc = 0.961268`

So the real question for this round became:
1. does the current code state still reproduce a `>95%` same-source regime
   cleanly,
2. and among low-risk fixes, which one improves the already-strong baseline
   the most?

### 0J.2 Resource hygiene mattered because the server is shared

Before launching the new matrix, GPU occupancy was checked with `nvidia-smi`.

Observed state at launch:
1. all four `A100 80GB` devices were idle
2. none had active memory pressure

Even so, the run was executed on only:
1. `CUDA_VISIBLE_DEVICES=1`

Reason:
1. keep the experiment sequential and conservative,
2. avoid unnecessary multi-GPU cache churn,
3. and reduce the chance of transient OOM caused by overlapping users joining
   later during the run window.

### 0J.3 New dedicated Math-Shepherd ACC95 matrix

New parameter groups were added:
1. `E67_MS_ACC95_JOINT_VERIFY_SEED42`
   - full-data verify control
2. `E68_MS_ACC95_JOINT_LOGIT_SEED42`
   - same control, but `ranking_target_space = logit`
3. `E69_MS_ACC95_JOINT_OVERFIT_SEED42`
   - same control, but with a more aggressive fit-oriented recipe
4. wrapper:
   - `I6_MS_ACC95_PUSH_MATRIX`

Executed command:
1. `CUDA_VISIBLE_DEVICES=1 ACTIVE_PHASE_E_INTRADATASET_GROUP=I6_MS_ACC95_PUSH_MATRIX RUN_PREFIX=phase_e_ms_acc95_push_0310_2146 bash scripts/run_phase_e_intradataset_suite.sh`

Artifacts:
1. suite summary:
   - `assets/artifacts/phase_e_logs/phase_e_ms_acc95_push_0310_2146/final_summary.md`
2. candidate report:
   - `assets/artifacts/phase_e_candidates/phase_e_ms_acc95_push_0310_2146_candidate/candidate_report.md`

### 0J.4 Results: the target was not only met, it was improved

Held-out same-source results:
1. `E67_MS_ACC95_JOINT_VERIFY_SEED42`
   - `pair_acc = 0.963267`
   - `auc = 0.942383`
   - `ranking_score = 0.952825`
2. `E68_MS_ACC95_JOINT_LOGIT_SEED42`
   - `pair_acc = 0.972450`
   - `auc = 0.941545`
   - `ranking_score = 0.956998`
3. `E69_MS_ACC95_JOINT_OVERFIT_SEED42`
   - `pair_acc = 0.966167`
   - `auc = 0.945630`
   - `ranking_score = 0.955898`

Candidate selector output:
1. selected group:
   - `E68_MS_ACC95_JOINT_LOGIT_SEED42`
2. `trust_score = 0.968689`
3. best checkpoint:
   - `assets/artifacts/phase_e_runs/phase_e_ms_acc95_push_0310_2146_e68_ms_acc95_joint_logit_seed42_e68_ms_acc95_joint_logit_seed42_s42_value_20260310T151651Z/best_value_head.pt`

### 0J.5 Interpretation: what actually helped

This round gives a clean answer.

The best improvement did **not** come from stronger overfit pressure.

Instead, the best improvement came from changing the ranking geometry:
1. `score-space joint` verify control:
   - `pair_acc = 0.963267`
2. `logit-space joint`:
   - `pair_acc = 0.972450`

So the updated interpretation is:
1. `Math-Shepherd` already has enough same-source signal to clear `95%`
   comfortably,
2. the current codebase is not blocked on this source,
3. and the most useful low-risk improvement here is
   `ranking_target_space = logit`,
4. not extra denoising removal or extra overfit pressure by itself.

### 0J.6 What this means for the broader repository diagnosis

This result narrows the failure surface elsewhere.

Because `Math-Shepherd` can now repeatedly reach:
1. `0.96+` under multiple current recipes,
2. and `0.97245` under the best current low-risk push,

the repository's remaining weaknesses should not be narrated as:
1. "the trainer cannot fit a strong process-style source"

They are better narrated as:
1. source-specific data/contract issues
   - as seen in old and repaired `R-PRM`
2. transfer / benchmark generalization issues
   - same-source success still does not imply `ProcessBench` success
3. recipe sensitivity across sources
   - the best Math-Shepherd setting is not automatically the best `R-PRM`
     setting

### 0J.7 Immediate next-step plan

1. Treat `E68_MS_ACC95_JOINT_LOGIT_SEED42` as the current same-source
   Math-Shepherd reference checkpoint family.
2. If seed-stability matters for promotion, rerun the same recipe as a
   multi-seed group instead of inventing a new recipe family first.
3. Keep using Math-Shepherd as a "can the current stack learn a clean source?"
   control, not as proof of benchmark-grade verification.
4. Focus future repair effort on the sources that still fail after contract
   cleanup, rather than continuing to over-optimize an already-solved
   Math-Shepherd intradataset objective.

## 0I. Latest Diagnosis Update (2026-03-10, R-PRM Root-Cause Matrix After Contract Repair)

### 0I.1 What this new round tried to answer

The earlier dedicated `R-PRM` recheck already showed that the current
`compact_verdict` contract is much healthier than the historical
`direct_pair_legacy` path.

But two more questions still mattered:
1. exactly how big is the gap between `legacy` and `compact` when we build
   canonical pair artifacts side by side from the same source?
2. after compact repair removes most truncation damage, is the remaining
   weakness mainly about recipe choice, or is the source still fundamentally
   unusable?

This round therefore combined four evidence surfaces:
1. legacy vs compact pair-artifact truncation comparison,
2. raw compact-contract audit directly on `R-PRM` DPO rows,
3. repaired-contract recipe matrix with same-family trust evaluation,
4. explicit `legacy@1024` fail-fast validation.

### 0I.2 Contract comparison: legacy vs compact are not in the same regime

Artifacts:
1. legacy artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_rprm_contractcmp_0310_2055_legacy_pairs__efc9444c97f8`
2. compact artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_rprm_contractcmp_0310_2055_compact_pairs__ca03cf0c9aa1`

Pair counts:
1. legacy:
   - total `2500`
   - train `2212`
   - validation `288`
2. compact:
   - total `910`
   - train `807`
   - validation `103`

Train truncation diagnostics:
1. legacy:
   - `768`: `over_limit=1.0000`, `collapse=0.4132`, `hidden_diff=0.4132`
   - `1024`: `over_limit=0.9313`, `collapse=0.1279`, `hidden_diff=0.1279`
   - `1536`: `over_limit=0.3639`, `collapse=0.0122`, `hidden_diff=0.0122`
   - `2048`: `over_limit=0.0696`, `collapse=0.0000`, `hidden_diff=0.0000`
2. compact:
   - `768`: `over_limit=0.0607`, `collapse=0.0607`, `hidden_diff=0.0607`
   - `1024`: `over_limit=0.0186`, `collapse=0.0186`, `hidden_diff=0.0186`
   - `1536`: all `0.0000`
   - `2048`: all `0.0000`

Validation truncation diagnostics tell the same story:
1. legacy first becomes clean at `2048`
2. compact first becomes clean at `1536`

So the updated hard fact is:
1. `legacy` and `compact` are not two nearby variants of the same contract,
2. they live in qualitatively different truncation regimes,
3. and any result that mixes them together is methodologically invalid.

### 0I.3 Raw compact audit: the repaired source still has real usable signal

Raw audit artifact:
1. `assets/artifacts/phase_e_rprm_audit/phase_e_rprm_contractaudit_0310_2100_20260310T124905Z/summary.json`

Key facts from `4000` audited raw rows:
1. `accepted_rows = 4000`
2. `acceptance_rate = 1.0000`
3. chosen verdict balance:
   - `yes = 1857`
   - `no = 2143`
4. compact prompt token stats:
   - mean `353.36`
   - `p95 = 739`
   - `p99 = 1064`
5. first-difference token stats:
   - mean `346.36`
   - `p95 = 732`
   - `p99 = 1057`
6. cutoff risk:
   - `1024`: `over_limit = 0.0125`, `hidden_diff_after_cutoff = 0.0125`
   - `1536`: both `0.0000`
   - `2048`: both `0.0000`

Interpretation:
1. compact repair is not a tiny cosmetic rewrite; it moves the source into a
   tractable token-length regime,
2. but `1024` is still not completely safe,
3. so repaired `R-PRM` should be treated as a `1536+` source by default.

### 0I.4 Legacy 1024 is now explicitly blocked before model load

Validation run:
1. `phase_e_rprm_legacy_failfast_0310_2100`

Observed behavior:
1. the run prints truncation diagnostics immediately,
2. then aborts with:
   - `over_limit_fraction=0.9313 exceeds 0.1000`
   - `collapse_after_cut_fraction=0.1279`
   - `hidden_diff_after_cut_fraction=0.1279`

This matters because the training entrypoint was hardened so unsafe settings
now fail before loading the backbone.

So this is both:
1. an experimental result:
   - `legacy@1024` is invalid
2. and an infrastructure result:
   - the repository now rejects this invalid configuration cheaply and early

### 0I.5 Repaired-contract recipe matrix: R-PRM compact is usable, but not easy

Recipe-matrix artifact:
1. `assets/artifacts/phase_e_logs/phase_e_rprm_recipe_smoke_0310_2105/final_summary.md`
2. raw rows:
   - `assets/artifacts/phase_e_logs/phase_e_rprm_recipe_smoke_0310_2105/recipe_rows.jsonl`

Shared setup:
1. compact pair artifact:
   - total `1092`
   - train `968`
   - validation `124`
2. validation prompt pools:
   - `123`
3. all recipe rows used:
   - `max_length = 2048`
   - safe truncation diagnostics:
     - train `0.0000`
     - validation `0.0000`

Results:
1. `linear + ranking_only`
   - `pair_acc = 0.6129`
   - `auc = 0.5728`
   - `samefamily_top1 = 0.6179`
   - `reject@0.6_top1 = 0.5946`
2. `mlp + ranking_only`
   - `pair_acc = 0.6452`
   - `auc = 0.5928`
   - `samefamily_top1 = 0.6585`
   - `reject@0.6_top1 = 0.7162`
3. `mlp + joint`
   - `pair_acc = 0.6694`
   - `auc = 0.6611`
   - `samefamily_top1 = 0.6829`
   - `reject@0.6_top1 = 0.7703`
4. `mlp + BCE_only`
   - `pair_acc = 0.6613`
   - `auc = 0.6439`
   - `samefamily_top1 = 0.6667`
   - `reject@0.6_top1 = 0.7297`

This matrix gives three non-trivial conclusions:
1. repaired `R-PRM compact` is genuinely learnable inside its own family
   - random-like failure is no longer the right description
2. head capacity matters
   - `MLP > linear`
3. unlike some earlier math-source results, `joint` is currently the best
   recipe on this source
   - so `R-PRM compact` is source-specific enough that recipe conclusions from
     `Math-Shepherd / PRMBench` cannot be copied blindly

### 0I.6 What remains weak

Even after repair, this is still not an `ACC90` source.

The best repaired smoke row (`mlp + joint @2048`) reaches:
1. `pair_acc = 0.6694`
2. `auc = 0.6611`
3. `samefamily_top1 = 0.6829`

So the new diagnosis is:
1. old `R-PRM` weakness was partly a contract/truncation artifact,
2. but not entirely,
3. because once the contract is repaired and length is safe, performance
   improves materially but still plateaus far below the easy-anchor regime

That means the remaining bottleneck is now better described as:
1. verdict-style supervision may be narrower / noisier than the other strong
   math sources,
2. the current frozen-scalar head can exploit some of it but not saturate it,
3. and `R-PRM` should currently be treated as a usable but medium-strength
   source, not a gold-standard anchor.

### 0I.7 Engineering warning that also matters scientifically

A legacy `2048` control run was attempted after it cleared the truncation gate.

What happened:
1. feature encoding immediately triggered repeated OOM backoff:
   - `64 -> 32 -> 16`
2. the full run reached only `1104 / 2212` chosen texts after about `239.7s`
   before being stopped
3. even a smaller matched-size subset still triggered repeated OOM backoff
   during feature encoding and was also stopped

This does not prove legacy `2048` can never train.
But it does prove something important:
1. even after becoming statistically executable, the legacy contract is still
   much more expensive and unstable operationally
2. so it is not a peer competitor to compact in practical research iteration

### 0I.8 Immediate next-step plan

1. Promote repaired `R-PRM` defaults to `1536+` everywhere; never silently
   allow `1024`.
2. Treat `compact_verdict` as the only valid Phase E mainline `R-PRM` contract;
   keep `direct_pair_legacy` only for historical comparison.
3. Use `mlp + joint` as the current repaired-source reference recipe.
4. Before mixing `R-PRM` into multi-source curricula again, explicitly test:
   - verdict-balanced resampling,
   - prompt-length bucket breakdowns,
   - and whether `samefamily_top1` stays stable under `source_sample` splitting.

## 0H. Latest Diagnosis Update (2026-03-10, Dedicated R-PRM Length-Sweep Recheck)

### 0H.1 What this new diagnostic was for

Earlier `I4_RPRM_ACC90_MATRIX` results were built on the historical
`direct_pair_legacy` contract, where `R-PRM` rows were still treated as long
verifier-essay pairs.

That meant one key question was still unresolved:
1. is current `R-PRM` weak because the source itself is poor,
2. or because the old long-text contract damaged the signal before training?

So a new dedicated diagnostic group was added and executed:
1. `scripts/run_phase_e_rprm_diagnostic_suite.sh`
2. group:
   - `RD1_RPRM_LENGTH_SWEEP_SMOKE`

### 0H.2 What was actually run

Artifacts:
1. pair artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_rprm_diag_0310_2019_pairs__2dd6a39365d8`
2. diagnostic summary:
   - `assets/artifacts/phase_e_logs/phase_e_rprm_diag_0310_2019/final_summary.md`

This run intentionally used the **current repaired** `R-PRM` contract:
1. `r_prm_pair_mode = compact_verdict`
2. prompt becomes compact:
   - `Question / Previous Steps / Now Step / Verification`
3. chosen/rejected become short opposite verdicts:
   - `Final answer: Yes/No`

### 0H.3 What the truncation diagnostics now say

This is the most important correction to the earlier understanding:

For the **current compact-verdict contract**, `R-PRM` is no longer a
"95%+ pairs are broken at 1024" dataset.

Train split:
1. `1024`
   - `over_limit = 0.0155`
   - `collapse_after_cut = 0.0155`
   - `hidden_diff_after_cut = 0.0155`
2. `1280`
   - `over_limit = 0.0031`
   - `collapse_after_cut = 0.0031`
   - `hidden_diff_after_cut = 0.0031`
3. `1536+`
   - all three become `0.0000`

Validation split:
1. `1024`
   - `over_limit = 0.0242`
   - `collapse_after_cut = 0.0242`
   - `hidden_diff_after_cut = 0.0242`
2. `1280`
   - `over_limit = 0.0161`
   - `collapse_after_cut = 0.0161`
   - `hidden_diff_after_cut = 0.0161`
3. `1536+`
   - all three become `0.0000`

So the new hard fact is:
1. `1024` is still slightly unsafe for compact `R-PRM`,
2. but `1536` is already fully clean,
3. and `2048` is not needed for safety.

### 0H.4 Same-source training results after removing truncation

Using the same repaired pair artifact and the strongest current `R-PRM`
same-source recipe family (`joint + MLP` style), we reran length-sweep training.

Results:
1. `1536`
   - run:
     - `assets/artifacts/phase_e_runs/phase_e_rprm_diag_0310_2019_len1536_20260310T121925Z`
   - held-out:
     - `pair_acc = 0.6129`
     - `auc = 0.6031`
2. `2048`
   - run:
     - `assets/artifacts/phase_e_runs/phase_e_rprm_diag_0310_2019_len2048_20260310T122338Z`
   - held-out:
     - `pair_acc = 0.7016`
     - `auc = 0.6735`

Reference old legacy same-source run:
1. `E49` seed-42 legacy result:
   - `pair_acc = 0.5901`
   - `auc = 0.5800`

### 0H.5 Interpretation

This new run changes the diagnosis in a precise way:

1. The old statement
   - "`R-PRM` mainly fails because `1024` destroys almost all pairs"
   is **true only for the legacy long-essay contract**.

2. For the **current repaired compact-verdict contract**:
   - truncation is still a real issue at `1024`,
   - but it is already fully solved at `1536`.

3. Once truncation is removed:
   - performance does improve,
   - especially from legacy `E49` to repaired `2048`,
   - but it still remains far below `ACC90`.

4. Therefore the current bottleneck is no longer "input got destroyed before
   learning started".
   It is now much more likely to be one of:
   - `R-PRM compact` supervision semantics are too narrow,
   - the frozen scalar head cannot fully exploit this verdict-style signal,
   - or same-source `R-PRM` simply is not a very strong source under this
     repository contract.

### 0H.6 Immediate engineering actions taken

To prevent future silent regressions:
1. `scripts/run_phase_e_rprm_diagnostic_suite.sh`
   - now explicitly passes `--r-prm-pair-mode compact_verdict`
2. `scripts/run_phase_e_suite.sh`
   - `E12_RPRM_COMPACT_VERDICT_SEED3`
   - `E65_STAGEB_MS_RPRM_COMPACT_REPAIRED_LINEAR_SEED3`
   - `E66_STAGEB_MS_RPRM_COMPACT_REPAIRED_MLP_SEED3`
   now default to:
   - `MAX_LENGTH = 1536`

This is important because the new diagnostic proved:
1. `1024` is not clean enough,
2. `1536` is the first clean cutoff,
3. so `compact R-PRM` groups should not silently fall back to `1024`.

## 0G. Latest Diagnosis Update (2026-03-10, R-PRM Dataset Contract Repair)

### 0G.1 What was fixed

The `R-PRM` data problem was not treated as a pure optimizer issue.

The concrete fix is now:
1. keep `direct_pair_legacy` for historical reproducibility,
2. add a new `compact_verdict` adapter mode for Phase E,
3. rewrite each `R-PRM` DPO row from:
   - one long verifier instruction
   - plus two long chosen/rejected verifier essays
   into:
   - one compact `Question / Previous Steps / Now Step` prompt
   - plus one short opposite-verdict pair.

Implementation surface:
1. `src/ours/phase_d/external_pairs_adapters.py`
2. `scripts/phase_e_prepare_pairs.py`
3. `scripts/run_phase_e_suite.sh`
4. `scripts/run_phase_e_repair_diagnostics_suite.sh`

New Phase E repair groups:
1. `E12_RPRM_COMPACT_VERDICT_SEED3`
2. `E65_STAGEB_MS_RPRM_COMPACT_REPAIRED_LINEAR_SEED3`
3. `E66_STAGEB_MS_RPRM_COMPACT_REPAIRED_MLP_SEED3`

New wrapper groups:
1. `R3_RPRM_DATAFIX_SMOKE`
2. `R4_RPRM_DATAFIX_SEED3`

### 0G.2 Why this is the right fix

The old contract forced the frozen-feature value head to score:
1. the full rubric-heavy verifier prompt,
2. plus multi-thousand-token chosen/rejected analyses.

That contract is mismatched with the current Phase E learner:
1. the model is a frozen backbone plus small scalar head,
2. not a generative verifier trained to consume and produce long analyses.

So the repaired contract now asks a cleaner question:
1. given the compact `Now Step` verification context,
2. does the head prefer the correct short verdict over the wrong short verdict?

### 0G.3 Real artifact-level validation

Using real `R-PRM` rows and the repository's own truncation diagnostic:

Legacy artifact:
1. `assets/artifacts/phase_e_truncation_diagnostics/rprm_legacy_diag_20260310T122121Z/summary.json`
2. `num_pairs = 64`
3. `frac_pairs_over_limit = 1.0`
4. `chosen_p95 = 1678`
5. `first_diff_p95 = 957`

Compact artifact:
1. `assets/artifacts/phase_e_truncation_diagnostics/rprm_compact_diag_20260310T122126Z/summary.json`
2. `num_pairs = 44`
3. `frac_pairs_over_limit = 0.0`
4. `chosen_p95 = 533`
5. `first_diff_p95 = 526`

Meaning:
1. the old `R-PRM` Phase E contract was fully over the `1024` limit on this real sample,
2. the repaired compact contract removes that truncation burden completely on the same diagnostic.

### 0G.4 End-to-end smoke confirmation

The new compact contract now runs through the real Phase E training path:
1. suite group:
   - `E12_RPRM_COMPACT_VERDICT_SEED3`
2. smoke run:
   - `assets/artifacts/phase_e_logs/phase_e_rprm_compact_smoke_0310_2038/final_summary.md`
3. seed-42 tiny smoke result:
   - `pair_accuracy = 0.666667`
   - `auc = 0.555556`
4. truncation diagnostics inside training:
   - `train over_limit = 0.0000`
   - `eval over_limit = 0.0000`

This smoke is intentionally tiny and is **not** scientific evidence of final
quality. It only proves:
1. the repaired `R-PRM` contract is wired through prepare/train/summarize,
2. the new compact path no longer fails because of immediate truncation.

### 0G.5 Follow-up hardening after the first user reruns

Two wrapper/runtime issues were then fixed:

1. `E12_RPRM_COMPACT_VERDICT_SEED3`
   - now defaults to:
     - `MAX_LENGTH=1536`
     - `EVAL_BATCH_SIZE=16`
   - reason:
     - the repaired compact contract should use the already-audited safe
       cutoff,
     - and Phase E currently reuses eval batch size for frozen-backbone
       feature caching, so a smaller value is the safest OOM guard.
2. `R3_RPRM_DATAFIX_SMOKE` and `R4_RPRM_DATAFIX_SEED3`
   - no longer include legacy long-analysis `R-PRM` groups by default
   - reason:
     - those legacy groups fail the truncation gate by design,
     - so including them in the official wrapper aborts the suite before the
       repaired groups can run.

## 0F. Latest Diagnosis Update (2026-03-10, I4 Full-Matrix Result + Phase E Aggregation Hardening)

### 0F.1 What the new completed run says

The first fully completed intradataset full-matrix run is now:
1. `assets/artifacts/phase_e_logs/phase_e_rprm_acc90_full_0310_1914/final_summary.md`
2. group:
   - `I4_RPRM_ACC90_MATRIX`

Its three recipe-family results are:
1. `E47_RPRM_ACC90_LINEAR_SEED3`
   - `mean_pair_acc = 0.4374`
   - `mean_auc = 0.5016`
2. `E48_RPRM_ACC90_MLP_RANK_SEED3`
   - `mean_pair_acc = 0.5197`
   - `mean_auc = 0.5123`
3. `E49_RPRM_ACC90_MLP_JOINT_SEED3`
   - `mean_pair_acc = 0.6002`
   - `mean_auc = 0.5885`

### 0F.2 Interpretation

This run adds a much stronger conclusion than the earlier smoke:
1. `R-PRM` is currently weak under the repository's same-source `ACC90`
   branch even after the recipe family is expanded from:
   - linear
   - to MLP ranking
   - to MLP joint
2. the direction `E49 > E48 > E47` shows that:
   - architecture and objective do matter,
   - but they do not rescue the source under the current adapter/contract.
3. the seed std is small:
   - `E49 std_pair_acc = 0.0098`
   - `E49 std_auc = 0.0106`
4. therefore this is **not** a seed-collapse story.

Current scientific reading:
1. `Math-Shepherd` and `PRMBench_Preview` remain positive same-source
   learnability sources.
2. `R-PRM` is not a trustworthy `ACC90` anchor under the current repository
   contract.

### 0F.3 Similar hidden risk that was fixed

After the earlier intradataset selector bug, a second class of structural
fragility was found:
1. several Phase E wrapper scripts were aggregating top-level suite results by
   regex-parsing `final_summary.md`;
2. this makes result aggregation depend on Markdown formatting rather than the
   real structured artifact.

This is now hardened:
1. `scripts/run_phase_e_intradataset_suite.sh`
2. `scripts/run_phase_e_single_source_suite.sh`
3. `scripts/run_phase_e_multisource_math_suite.sh`

All three now read:
1. `seed_results.jsonl`

instead of reverse-parsing:
1. `final_summary.md`

Meaning:
1. top-level suite summaries now trace back to the structured source-of-truth
   artifact,
2. mild Markdown formatting edits will no longer silently corrupt group-level
   comparisons.

### 0F.4 Candidate-selector hardening

For consistency with the already-fixed intradataset selector, the benchmark-
aware selector was also hardened:
1. `scripts/phase_e_select_candidate.py`

New behavior:
1. repeated `--suite-log-dirs DIR` occurrences are accepted,
2. directory groups are flattened in first-seen order,
3. duplicates are removed.

This prevents future wrapper/selector contract drift from reproducing the same
class of error.

## 0E. Latest Diagnosis Update (2026-03-10, Why Some Phase E Sources Reach High ACC While Others Do Not)

### 0E.1 Core finding

The current evidence does **not** support the claim that the whole `Phase E`
trainer is fundamentally broken.

Reason:
1. the same training stack can already reach very high same-source held-out
   performance on some datasets:
   - `E41_MS_ACC90_MLP_RANK_SEED3`
     - `heldout_pair_acc = 0.9610`
     - `heldout_auc = 0.8908`
   - `E45_PRMBENCH_ACC90_MLP_RANK_SEED3`
     - `heldout_pair_acc = 0.9483`
     - `heldout_auc = 0.8333`
2. but `E48_RPRM_ACC90_MLP_RANK_SEED3` stays near random:
   - `heldout_pair_acc = 0.4379`
   - `heldout_auc = 0.4666`

Therefore:
1. this is not one universal low-level trainer failure,
2. but there **is** a source-specific bottom-layer implementation / contract
   mismatch.

### 0E.2 Strongest implementation-level issue found

The largest concrete issue is the interaction between:
1. `R-PRM` direct-pair adapter format,
2. the repo-wide fixed `max_length = 1024`,
3. and silent tokenizer truncation during frozen-feature encoding.

What the code currently does:
1. `src/ours/phase_d/external_pairs_adapters.py`
   - loads `R-PRM` as full `instruction + chosen + rejected` long-form direct
     verifier texts.
2. `src/ours/phase_b/value_head.py`
   - encodes all sources with `truncation=True` and one shared `max_length`.
3. the failing `R-PRM` ACC90 run manifest confirms:
   - `max_length = 1024`.

What the audit measured on the actual training artifact:
1. `Math-Shepherd`
   - mean token length about `122`
   - `0%` exceed `1024`
2. `PRMBench_Preview`
   - mean token length about `288`
   - about `1.2%` exceed `1024`
3. `R-PRM`
   - mean token length about `1425.6`
   - `94.1%` exceed `1024`
   - `11.7%` of pairs become **identical after truncation**

This is the strongest code-level reason why one source can fail while others
train well under the same recipe.

### 0E.3 Why this matters scientifically

For `R-PRM`, the problem is not only sequence length.

There is also a source-contract mismatch:
1. `Math-Shepherd` and `PRMBench_Preview`
   - are converted into local prefix / local-error pairs
   - first difference appears early in the token sequence
2. `R-PRM`
   - is loaded as long-form verifier analyses
   - first difference is often much later

Measured first-difference position under the current tokenizer:
1. `Math-Shepherd`
   - median first difference at token `113`
2. `PRMBench_Preview`
   - median first difference at token `171`
3. `R-PRM`
   - median first difference at token `740`
   - 90th percentile first difference at token `1056`

Meaning:
1. the shared `1024`-token recipe is almost neutral for `Math-Shepherd`,
2. mostly acceptable for `PRMBench_Preview`,
3. but systematically lossy for `R-PRM`.

### 0E.4 Practical conclusion

Current judgment:
1. the repo does have a real bottom-layer issue,
2. but it is **source-specific**, not a universal trainer bug.

The issue is:
1. one unified input contract is being applied to sources with very different
   sequence lengths and supervision semantics,
2. and the current pipeline provides no truncation diagnostics or source-aware
   max-length policy.

So the safe conclusion is:
1. `Math-Shepherd` and `PRMBench_Preview` high-ACC results are credible as
   evidence that the trainer can learn,
2. `R-PRM` failure should **not** be read as pure scientific failure yet,
3. because the current implementation likely destroys part of its supervision
   signal before the head ever sees it.

### 0E.5 Next-step plan

1. Add explicit truncation diagnostics to `Phase E` pair preparation / training
   summaries:
   - per-source token-length quantiles,
   - fraction over `max_length`,
   - fraction whose chosen/rejected first difference lies after the cutoff.
2. Re-run `R-PRM` same-source ACC with at least:
   - `max_length = 1536`
   - `max_length = 2048`
3. If long-context reruns still fail badly, then treat:
   - `R-PRM` source semantics,
   - not truncation,
   as the main blocker.
4. Do not compare source quality using one fixed `max_length` recipe unless the
   truncation burden is shown to be comparable across sources.

## 0D. Latest Diagnosis Update (2026-03-10, Phase E Value-Head Structure / Recipe Audit)

### 0D.1 What was checked

This audit focused on the current `Phase E` value-head implementation from
three angles:
1. network structure
2. learning mechanism
3. newest single-source vs multi-source experimental evidence

The question was not just "did one run fail?", but:
1. is the current head intrinsically too weak or too strange,
2. or is the present training recipe the bigger problem?

### 0D.2 Reliable structure facts from the current code and artifacts

1. The current `Phase E` learner is a **frozen-feature scalar-head** design.
   - The transformer backbone is run once to extract pooled prefix features.
   - Those features are cached.
   - Training then updates only the small head, not the whole backbone.
2. The head implementation exposes both:
   - raw `logits`
   - `sigmoid(logits)` scores
3. The active failing mixture still uses `objective_mode=ranking_only`.
4. The run artifacts inspected here only store a minimal
   `value_head_config.json` payload:
   - `hidden_size`
   - `dropout_prob`
   - `init_std`
   - `pooling`
5. Therefore the exact historical `linear` vs `mlp` branch is not fully
   recoverable from artifact metadata alone.
6. What is recoverable is the broader family:
   - lightweight frozen-feature head,
   - not full-backbone joint fine-tuning.

### 0D.3 Most informative experiment contrast

| Run | Train recipe snapshot | Held-out result | External / benchmark-facing result | Interpretation |
|---|---|---|---|---|
| `E2_MATH_SHEPHERD_PAIR_LEARN_SEED3` seed-42 | `5358` train pairs; `batch=32`; `epochs=4`; `lr=5e-5`; `pair_weight_mode=confidence`; about `672` optimizer steps | `pair_acc=0.8480`; `auc=0.8218`; `ranking_score=0.8349` | strong same-family held-out signal; trust-matrix `ProcessBench` transfer still weak | Confirms that the current value-head family can learn a clean same-source ranking task. |
| `E24_STAGEB_MS_RPRM_MIX_SEED3` seed-42 | `1774` train pairs; `batch=128`; `epochs=6`; `lr=2e-5`; `pair_weight_mode=none`; `source_balance=uniform`; about `84` optimizer steps | `pair_acc=0.4381`; `auc=0.4922`; `ranking_score=0.4651` | `PRMBench_Preview auc=0.5207`; partial smoke summary also remained weak on `PB-GSM8K` and `PB-MATH` | Negative result for the current balanced two-source recipe, but not evidence that value-head learning itself is impossible. |

### 0D.4 Diagnosis

The main conclusion is:
1. the current head family is **not** yet proven intrinsically broken,
2. the current mixture recipe is the more credible failure point.

Most important risks:
1. **Loss-design risk**
   - ranking is currently applied on bounded sigmoid scores
   - this compresses margin geometry and makes saturation easier
2. **Source-semantics mismatch**
   - `Math-Shepherd` contributes strict local `first_bad_edge` pairs
   - `R-PRM` contributes direct chosen/rejected preference pairs
   - these are related but not identical supervision targets
3. **Under-training in effective update count**
   - the failing `E24` run gets about `84` optimizer steps
   - the strong `Math-Shepherd` run gets about `672`
4. **Weighting is disabled in the heterogeneous setting**
   - `pair_weight_mode=none`
   - `source_balance=uniform`
5. **Split granularity is weaker than ideal**
   - current splitting is pair-id based, not problem-id based
6. **Capacity may still matter, but it is not the first suspect**
   - current evidence still points more strongly to objective/data mismatch
     than to head complexity alone

### 0D.5 What we should conclude scientifically

1. `Phase E` has already demonstrated **same-source learnability**.
2. The open question is now:
   - not whether a value head can learn anything,
   - but whether heterogeneous math-process sources can be combined without
     destroying the useful signal.
3. The current `MS + R-PRM` failure should therefore be interpreted as:
   - a failure of the present Stage B recipe,
   - not a decisive failure of the value-head family.

### 0D.6 Explicit next-step plan

1. repair the pairwise objective first
2. stop treating local `first_bad_edge` supervision and direct preference
   supervision as a trivially interchangeable pool
3. increase effective optimizer steps for Stage B mixtures
4. restore confidence/source weighting for heterogeneous mixtures
5. consider sample/problem-level splitting where one raw example can emit
   multiple related pairs
6. only after the above, revisit head-capacity upgrades as a first-order
   experiment

## 0C. Latest Diagnosis Update (2026-03-10, Repository Audit + Phase E Reliability)

### 0C.1 What the repository is actually trying to prove

The repository's real claim is now narrower than the old `Phase C / Phase D`
phrasing might suggest.

What the pipeline is actually trying to establish:
1. freeze a strong backbone,
2. train a small prefix-level value / ranking head,
3. prove same-family process discrimination first,
4. and only after that ask whether the signal is trustworthy enough for
   benchmark-facing verification or later control.

What the current blocker really is:
1. not "can a tiny head fit any signal at all?",
2. but:
   - whether supervision semantics are clean enough,
   - whether the objective matches the scientific claim,
   - and whether the evaluation pipeline itself can be trusted.

### 0C.2 Most meaningful experimental evidence reviewed in this audit

1. `MS2_MATH_SHEPHERD_TRUST_SEED3_MATRIX`
   - strongest reviewed group:
     - `E14_MATH_SHEPHERD_TRUST_ANTISAT_SEED3`
   - metrics:
     - `mean_hold_pair = 0.8246`
     - `mean_hold_auc = 0.7899`
     - `pb_gsm_auc = 0.4893`
     - `pb_math_auc = 0.4783`
   - meaning:
     - same-source learnability is clearly real,
     - benchmark-native robustness is still not established.
2. `CUR1_STAGEE_MS_TO_MSRPRM` benchmark eval on `processbench_math`
   - metrics:
     - `pair_accuracy_good_vs_bad = 0.5105`
     - `pair_auc_good_vs_bad = 0.4812`
     - `first_error_edge_accuracy = 0.4593`
   - meaning:
     - the current multi-source curriculum evidence is still near-random on the
       benchmark-facing task that matters most.
3. Combined reading of earlier `Phase D` and current `Phase E` evidence:
   - external same-family ranking learnability is a real positive result,
   - but the repository still does **not** have clean evidence for a robust
     benchmark verifier.

### 0C.3 Strict code-audit findings

This audit explicitly separated "research limitation" from "implementation bug".

Confirmed real bug:
1. `src/ours/phase_b/value_head.py`
   - the newer head refactor removed the old `.proj` access pattern used by
     older warm-start / debug code,
   - this was already breaking
     `tests/unit/test_phase_c_train_value.py::test_initialize_value_head_from_checkpoint_loads_matching_weights`,
   - the compatibility alias has now been restored and regression-tested.

Confirmed high-risk behaviors that can silently contaminate experiment reading:
1. `src/ours/phase_e/runtime.py`
   - asking for checkpoint `"best"` silently falls back to `"final"` when
     `best_value_head.pt` is missing.
   - risk:
     - an operator can believe a benchmark report used the best checkpoint when
       it actually used the final one.
2. `src/ours/phase_b/feature_cache.py`
   - cache-lock staleness is inferred only from file `mtime`.
   - risk:
     - a live but slow writer can be treated as dead and have its lock stolen,
       allowing overlapping writers.
3. `src/ours/phase_e/benchmark_eval.py`
   - `PRMBench_Preview` error-step normalization assumes 1-based indexing and
     subtracts one from positive indices.
   - current status:
     - the audited snapshot in this repo appears consistent with 1-based
       indexing,
     - but the loader is still schema-heuristic rather than schema-verified.

Audit verification status:
1. targeted audit tests were added for:
   - value-head backward compatibility,
   - checkpoint fallback behavior,
   - feature-cache live-lock overlap,
   - PRMBench preview 1-based normalization.
2. full test status after the audit:
   - `142 passed, 2 skipped`.

### 0C.4 Updated evaluation of the repository strategy

Current judgment:
1. the scientific direction is mostly correct,
2. the current implementation and evidence are not yet sufficient for the
   stronger "trustworthy small verifier" claim.

What is correct:
1. moving from `StrategyQA-first` to benchmark-native process data is correct,
2. moving from scalar-calibration-first to ranking-first is correct,
3. treating same-family learnability as a prerequisite before transfer claims
   is correct.

What is not yet effective:
1. the project has not solved benchmark-facing verification,
2. the current positive evidence is still local:
   - source-family learnability,
   - not trustworthy cross-benchmark process discrimination.

Therefore the scientifically safe narrative is:
1. "small frozen-feature ranking heads can learn some same-family local process
   discrimination",
2. not:
   - "the repository has already produced a reliable small PRM / verifier."

### 0C.5 Explicit next-step plan

1. For official promotion runs, fail hard if `"best"` is requested but
   `best_value_head.pt` does not exist.
2. For feature-cache writes, either:
   - enforce one-writer operational policy,
   - or upgrade the lock to a heartbeat / owner-checked design.
3. Keep `PRMBench_Preview` schema pinned and re-verify index semantics whenever
   the dataset file changes.
4. Continue `Phase E` under the narrow claim:
   - same-family ranking-first verifier learning,
   - not benchmark-robust process verification.
5. If benchmark metrics stay near random, the next scientific lever should be:
   - supervision quality,
   - pair semantics,
   - or verifier-style objective design,
   not just increasing head complexity.

## 0B. Latest Diagnosis Update (2026-03-10, Phase E intradataset ACC90 structural reading)

### 0B.1 Core question

After the new same-source `ACC90` suites landed, the diagnosis question became:
1. Is the current head too simple?
2. Are weak runs just under-trained?
3. Or do the answers depend on the source dataset?

### 0B.2 Strong evidence

#### Math-Shepherd

| Group | Head / Objective | mean held-out pair_acc | mean held-out auc | Reading |
|---|---|---:|---:|---|
| `E40_MS_ACC90_LINEAR_ROBUST_SEED3` | linear + ranking | 0.9172 | 0.8623 | Linear head already works strongly |
| `E41_MS_ACC90_MLP_RANK_SEED3` | MLP + ranking | 0.9863 | 0.9056 | Higher capacity improves further |
| `E42_MS_ACC90_MLP_JOINT_SEED3` | MLP + joint | 0.9641 | 0.9408 | Joint objective boosts AUC strongly |
| `E43_MS_ACC90_MLP_HIGHCONF_SEED3` | MLP + joint + denoise | 0.9619 | 0.9425 | Cleaner pairs also help |

#### PRMBench Preview

| Group | Head / Objective | mean held-out pair_acc | mean held-out auc | Reading |
|---|---|---:|---:|---|
| `E44_PRMBENCH_ACC90_LINEAR_SEED3` | linear + ranking | 0.7380 | 0.6782 | Linear head underfits |
| `E45_PRMBENCH_ACC90_MLP_RANK_SEED3` | MLP + ranking | 0.9315 | 0.8711 | MLP crosses ACC90 |
| `E46_PRMBENCH_ACC90_MLP_JOINT_SEED3` | MLP + joint | 0.9309 | 0.9057 | MLP + joint is strongest balanced path |

#### Under-training control

| Group | mean held-out pair_acc | mean held-out auc | Reading |
|---|---:|---:|---|
| `E12_MS_TRUST_LOWLR_SEED3` | 0.5853 | 0.5856 | Lower LR / longer training alone does not rescue the run |

### 0B.3 Diagnosis

1. `Math-Shepherd`
   - current evidence does **not** support the claim that the linear head is
     too simple;
   - the linear head already exceeds `0.90` same-source held-out accuracy;
   - MLP is an upgrade, not a rescue.
2. `PRMBench_Preview`
   - current evidence strongly supports the claim that the linear head is too
     simple;
   - here, MLP capacity is a real bottleneck remover.
3. Old weak runs cannot be explained only by under-training.
   - `E12` should have improved much more if that were the main issue.
   - therefore pair semantics, objective choice, denoising, and head capacity
     all matter.
4. Very high same-source held-out accuracy should still be interpreted
   narrowly.
   - it proves same-source learnability;
   - it does not yet prove benchmark transfer or future RL trustworthiness.

### 0B.4 Next plan

1. Freeze source-specific defaults:
   - `Math-Shepherd`: keep linear as strong baseline, compare against MLP.
   - `PRMBench_Preview`: default to MLP.
2. Finish the missing `R-PRM` same-source matrix before making a general claim
   about direct preference supervision.
3. Use same-source winners as bounded-support candidates only; do not narrate
   them as cross-task RL-ready value functions.

## 0C. Latest Diagnosis Update (2026-03-10, RL trustworthiness threshold reading)

### 0C.1 Core question

If we ignore cross-dataset transfer for now, what level of value-utility
quality is enough before we should trust it inside RL-style reasoning
faithfulness experiments?

### 0C.2 Literature-guided answer

The strongest consensus is:
1. there is no single universal scalar threshold such as:
   - `pair_acc > 0.90 => RL-ready`
2. strong held-out discrimination is necessary,
3. but process reward / value models must also survive optimization pressure.

Operational reading from the literature:
1. `ProcessBench` and `PRMBench` both show that apparently decent PRMs can
   still fail explicit process-error identification.
2. `PRM800K` and later PRM work support process supervision as a viable signal,
   but not as a one-metric guarantee of trust.
3. reward/value models should also demonstrate policy-level utility and
   resistance to shortcut exploitation.

### 0C.3 Current repository status

What we already have:
1. same-source held-out discrimination is now strong on:
   - `Math-Shepherd`
   - `PRMBench_Preview`
2. therefore same-source learnability is no longer the main uncertainty.

What is still missing before RL-level trust can be claimed:
1. same-family policy-improvement evidence
   - reranking / rejection / conservative search should improve final process
     quality inside the same dataset family
2. same-family local-faithfulness evidence
   - not just pair discrimination, but stronger local error-ordering checks
3. robustness under stronger optimization pressure
   - high-scoring processes should not merely exploit dataset-specific
     shortcuts

### 0C.4 Diagnosis

1. The current supported claim is:
   - same-source value learning is established on some sources.
2. The current unsupported claim is:
   - these value heads are already trustworthy enough to heavily drive RL.
3. Therefore the next useful experiments should shift from:
   - "can the head fit held-out pairs?"
   to:
   - "does the head improve same-family process selection under stronger
     decision pressure?"

### 0B.5 Important caveat from the first smoke run

The first smoke run:
1. `assets/artifacts/phase_e_logs/phase_e_top3_acc90_0310_1808/final_summary.md`

was still intentionally narrow:
1. it forced `seed=42` only,
2. reduced pair counts to `3000`,
3. and should be interpreted as a quick direction check, not a stability
   result.

Meaning:
1. the strong `E41` / `E45` numbers in that smoke are useful,
2. but they are not enough on their own to claim seed-stable `ACC90`
   promotion.

### 0B.6 Candidate-selection script issue from the smoke and its fix

The original auto-selected candidate in that smoke run was invalid, but the
selector bug has now been fixed.

Observed symptom:
1. the candidate report selected `E48_RPRM_ACC90_MLP_RANK_SEED3`,
2. even though `E41` and `E45` were numerically much stronger.

Original root cause:
1. `scripts/run_phase_e_intradataset_suite.sh`
   - passes multiple repeated `--suite-log-dirs ...` flags,
2. `scripts/phase_e_select_intradataset_candidate.py`
   - originally defined `--suite-log-dirs` as one `nargs=\"+\"` argument,
3. so the selector ends up consuming only the last repeated flag in this call
   pattern.

Fix:
1. the selector now accepts repeated `--suite-log-dirs` groups,
2. flattens and deduplicates them before scoring.

Verification:
1. dedicated unit test now covers repeated-flag parsing and strongest-group
   selection,
2. re-running the selector on the existing smoke suite now selects:
   - `E41_MS_ACC90_MLP_RANK_SEED3`
   instead of the incorrect old `E48` result.

Consequence:
1. the old smoke-level `candidate_report.json/.md` remains invalid history,
2. but the selector code path is now fixed for future ACC90 suite runs.

## 0A. Latest Diagnosis Update (2026-03-10, Phase E Small-Scale Value-Head Viability)

### 0A.1 Core literature diagnosis

Recent literature does **not** strongly support the following target:
1. small-scale
2. noisy or weakly synthesized labels
3. scalar value regression
4. benchmark-robust step-level quality prediction

Recent literature **does** support narrower variants:
1. large-scale dense process supervision:
   - `Let's Verify Step by Step` / `PRM800K`
   - `Math-Shepherd`
   - `OmegaPRM`
   - `Rewarding Progress`
2. smaller but stronger ranking / verifier formulations:
   - `ThinkPRM`
   - `R-PRM`
3. coarse-grained confidence/value-style heads:
   - `Language Models (Mostly) Know What They Know`
   - `Large Language Models Must Be Taught to Know What They Don't Know`

Operational reading:
1. the literature does not justify expecting our old scalar `MC / q_teacher /
   q_fused` value-head formulation to become a robust process verifier at small
   scale;
2. the literature does justify testing:
   - same-family
   - ranking-first
   - local process discrimination
   under stronger pair/process supervision.

### 0A.2 New meaningful experimental results

1. `MS2_MATH_SHEPHERD_TRUST_SEED3_MATRIX`
   - provisional best:
     - `E14_MATH_SHEPHERD_TRUST_ANTISAT_SEED3`
   - metrics:
     - `mean_hold_pair=0.8246`
     - `mean_hold_auc=0.7899`
     - `pb_gsm_auc=0.4893`
     - `pb_math_auc=0.4783`
   - meaning:
     - post-fix `Math-Shepherd` same-family learnability is real,
     - but benchmark robustness is still weak.
2. `E20_STAGEA_MS_ANCHOR_SEED3`
   - `mean_heldout_pair_acc=0.6853`
   - `mean_heldout_auc=0.6676`
   - `mean_prmbench_preview_auc=0.5868`
   - `mean_processbench_gsm8k_auc=0.4750`
   - `mean_processbench_math_auc=0.4715`
   - meaning:
     - best clean post-fix same-family anchor,
     - still not a trustworthy `ProcessBench` verifier.
3. `E21_STAGEA_RPRM_ANCHOR_SEED3`
   - `mean_heldout_pair_acc=0.4381`
   - `mean_heldout_auc=0.4953`
   - `mean_prmbench_preview_auc=0.5623`
   - `mean_processbench_gsm8k_auc=0.4744`
   - `mean_processbench_math_auc=0.4626`
   - one informative run:
     - `PRMBench_Preview pair_acc=0.6001`
     - `PRMBench_Preview auc=0.5937`
   - meaning:
     - `R-PRM` is not the strongest general anchor,
     - but it is more aligned with `PRMBench_Preview`.
4. `E24_STAGEB_MS_RPRM_MIX_SEED3` smoke
   - currently only partial seed-42 evidence:
     - `PRMBench_Preview auc=0.5207`
     - `ProcessBench GSM8K auc=0.4682`
     - `ProcessBench Math auc=0.3835`
   - meaning:
     - no positive evidence yet that simple balanced mixing fixes the gap.

### 0A.3 Updated diagnosis

1. Our current experiments are now consistent with the literature:
   - local same-family learnability exists,
   - benchmark-facing process verification remains unresolved.
2. The strongest positive evidence is no longer:
   - "small scalar value head works"
   but:
   - "small-to-medium same-family ranking verifier learning works to some
     extent."
3. The current unresolved problem is:
   - how to combine `Math-Shepherd`'s same-family anchor strength with
     `R-PRM` / `PRMBench_Preview` style benchmark alignment.

### 0A.4 Explicit next-step plan

1. Finish `E22_STAGEA_PRMBENCH_PREVIEW_ANCHOR_SEED3` before making any source
   ranking claim about the Stage A matrix.
2. Do not narrate `E24` or `CUR1` as positive until a full completed summary is
   available.
3. Keep the current Phase E scientific claim narrow:
   - same-family ranking-first verifier learning,
   - not general-purpose small PRM/value modeling.
4. If the project wants a stronger claim later, prioritize one of:
   - stronger judge-generated supervision
   - generative verifier training
   - or a larger process-label pipeline

### 0A.5 Key references

1. `Let's Verify Step by Step`
   - https://arxiv.org/abs/2305.20050
2. `Rewarding Progress`
   - https://arxiv.org/abs/2410.08146
3. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   - https://arxiv.org/abs/2501.07301
4. `ProcessBench`
   - https://arxiv.org/abs/2412.06559
5. `R-PRM`
   - https://arxiv.org/abs/2503.21295
6. `ThinkPRM`
   - https://arxiv.org/abs/2504.16828
7. `Language Models (Mostly) Know What They Know`
   - https://arxiv.org/abs/2207.05221
8. `Large Language Models Must Be Taught to Know What They Don't Know`
   - https://arxiv.org/abs/2406.08391

## 0. Latest Diagnosis Update (2026-02-27, A8 GSM8K Prompt Style Sweep)

### 0.1 New A8 Results (n=172, freeform decode)

| Label | Template | Max New Tokens | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---|---:|---:|---:|---:|---:|
| `style_direct_final_t32` | `qa_gsm8k_direct_final_only` | 32 | 0.3895 | 0.0000 | 172 | 0.3895 |
| `style_cot_compact_t192` | `qa_gsm8k_cot_compact_final` | 192 | 0.7616 | 0.0000 | 172 | 0.7616 |
| `style_equation_t64` | `qa_gsm8k_equation_then_final` | 64 | 0.3895 | 0.0000 | 172 | 0.3895 |

### 0.2 Comparison With Previous GSM8K Baselines

| Track | Accuracy | Parse Error |
|---|---:|---:|
| Previous direct baseline (re-eval) | 0.3721 | 0.0000 |
| Previous CoT baseline (`t256`, re-eval) | 0.7035 | 0.0000 |
| A8 direct-final (`t32`) | 0.3895 | 0.0000 |
| A8 CoT-compact (`t192`) | 0.7616 | 0.0000 |
| A8 equation (`t64`) | 0.3895 | 0.0000 |

Observed deltas:
1. A8 CoT-compact improves over previous CoT baseline by `+0.0581` (0.7616 vs 0.7035).
2. A8 direct-final improves over previous direct baseline by `+0.0174` (0.3895 vs 0.3721).
3. Equation-style gives no quality gain vs direct-final in this setup.

### 0.3 Runtime and Throughput Tradeoff (A8 runs)

| Label | Elapsed Seconds | Samples/sec |
|---|---:|---:|
| `style_direct_final_t32` | 146.54 | 1.174 |
| `style_cot_compact_t192` | 883.72 | 0.195 |
| `style_equation_t64` | 288.53 | 0.596 |

Interpretation:
1. CoT-compact is strongest on quality but about `6x` slower than direct-final (`0.195` vs `1.174` sample/s).
2. Equation-style is slower than direct-final and does not improve accuracy, so it is currently dominated.

### 0.4 Extraction-Method Diagnostics (Why CoT still misses some items)

1. `style_cot_compact_t192`:
   - `final_answer_tag`: 162 samples, method-level acc `0.8086`
   - `last_number`: 10 samples, method-level acc `0.0000`
2. `style_equation_t64`:
   - mostly `final_answer_tag` but low method-level accuracy (`0.3988`)
3. `style_direct_final_t32`:
   - all samples `final_answer_tag`, method-level accuracy `0.3895`

Conclusion from extraction view:
1. Remaining CoT errors are concentrated where model fails to emit clean final-answer format and falls to `last_number`.
2. The main bottleneck is no longer parse coverage (already `0.0000` parse error), but reasoning correctness and answer-format consistency under long-form outputs.

### 0.5 A8 Conclusions

1. For GSM8K Phase A, `qa_gsm8k_cot_compact_final@t192` is the best current quality baseline.
2. `qa_gsm8k_direct_final_only@t32` is the practical fast baseline.
3. `qa_gsm8k_equation_then_final@t64` should not be prioritized further unless reformulated.
4. A8 narrows the gap to expected public-model ceiling and gives a stronger launch point for Phase B.

### 0.6 Next Plan

1. Freeze two GSM8K reporting baselines:
   - quality: `cot_compact_t192`
   - speed: `direct_final_t32`
2. Run one reproducibility repeat for `cot_compact_t192`.
3. Add a focused follow-up to reduce `last_number` fallback cases in CoT outputs (format tightening + extraction guards).
4. Carry these two baselines into Phase B pre/post training comparison.

### 0.7 A8 Error Pattern Deep-Dive (Sample-Level)

Focus run: `style_cot_compact_t192` (best GSM8K quality in A8).

Error composition (41 wrong samples):
1. `final_answer_tag` wrong: `31`
2. `last_number` fallback wrong: `10`

Observed error patterns:
1. **Truncation / unfinished outputs** (`10/41`):
   - No clean final-answer line, evaluator falls back to `last_number`.
   - Typical signature: reasoning is mid-derivation and output ends abruptly near token cap.
   - Example IDs: `gsm8k:main:train:140`, `gsm8k:main:train:198`, `gsm8k:main:train:581`.
2. **Reasoning arithmetic/sign mistakes** (majority of remaining errors):
   - Model emits parseable final answer but applies wrong operation/sign or rate conversion.
   - Example IDs: `gsm8k:main:train:1406` (subtracts missed count instead of adding), `gsm8k:main:train:692`, `gsm8k:main:train:1033`.
3. **Extraction edge cases with expression-style final lines** (small but critical):
   - Some outputs contain `Final answer: <expression> = <number>` and/or multiple `Final answer:` lines.
   - Current extraction can pick the first numeric token from the expression line, undercounting true correctness.
   - Confirmed undercount IDs: `gsm8k:main:train:838`, `gsm8k:main:train:1418`.
4. **Self-correction / repeated final-answer lines**:
   - Multi-final-answer outputs appeared in error rows and can interact with extraction heuristics.
   - Count in CoT errors: `11`.

Cross-style hardness signal:
1. `31` samples were wrong in **all three** A8 styles (direct, CoT-compact, equation).
2. These shared-hard samples are dominated by multi-stage word problems with money/time/rate/percentage structure.

Improvement implications:
1. Keep improving prompt quality, but also harden evaluator extraction for math final-answer lines with expressions/repeated tags.
2. Reduce truncation pressure on CoT outputs (format tightening / better stop behavior / budget tuning).
3. Build a targeted “hard subset” regression list from the 31 shared-hard IDs for future Phase B comparisons.

## 1. Previous Diagnosis Update (2026-02-27, A7 StrategyQA Prompt Style Sweep)

### 0.1 New A7 Results (n=193, freeform decode)

| Label | Template | Max New Tokens | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---|---:|---:|---:|---:|---:|
| `style_minimal_t16` | `qa_strategyqa_minimal_binary` | 16 | 0.6632 | 0.0000 | 193 | 0.6632 |
| `style_cot_compact_t96` | `qa_strategyqa_cot_compact` | 96 | 0.6943 | 0.0311 | 187 | 0.7166 |
| `style_evidence_verdict_t32` | `qa_strategyqa_evidence_verdict` | 32 | 0.3782 | 0.4663 | 103 | 0.7087 |

### 0.2 Diagnosis

1. `qa_strategyqa_cot_compact` is currently the best total-accuracy style in freeform mode.
   - It improves absolute accuracy over `qa_strategyqa_minimal_binary` by `+0.0311` (0.6943 vs 0.6632).
2. `qa_strategyqa_minimal_binary` is the best compliance style.
   - Parse error is exactly `0.0000`, making it the safest format baseline.
3. `qa_strategyqa_evidence_verdict` likely has a **format/truncation bottleneck**, not purely low reasoning quality.
   - Parseable-only accuracy is high (`0.7087`), close to the best parseable track (`0.7166`),
   - but parse failures are very high (`0.4663`), collapsing total accuracy.

### 0.3 What We Learn

1. Prompt style matters as much as token budget for StrategyQA freeform performance.
2. There is a practical tradeoff:
   - best compliance (`minimal_binary`) vs best current total accuracy (`cot_compact`).
3. For verdict-style prompts, current output protocol is under-specified for short budgets (`t32`).
   - the model often produces partially formatted outputs, causing non-parseable predictions.

### 0.4 Reliability Notes

1. This comparison is protocol-consistent (same dataset split/policy/seed/model family), so the ranking is meaningful for this setup.
2. Sample size is still modest (`n=193`); small deltas should be reconfirmed with reruns and/or larger validation sets before freezing final claims.

### 0.5 Next Plan

1. Keep `qa_strategyqa_cot_compact` as the freeform quality candidate for Phase A.
2. Keep `qa_strategyqa_minimal_binary` as compliance/efficiency baseline.
3. Run a targeted verdict-style follow-up with higher token budgets and/or stricter output contract before discarding the style.
4. For fair model-quality comparison independent of parse noise, continue reporting binary-choice results in parallel.

## 2. Previous Diagnosis Update (2026-02-26, sweeper2 + A5)

### 0.1 What Problem We Tried To Fix

The active issue was poor output compliance in StrategyQA yes/no evaluation:
- CoT runs produced long explanations without extractable final yes/no.
- Parse errors dominated total accuracy at mid token budgets.
- We introduced strict yes/no prompting (`A5`) to force parseable outputs.

### 0.2 New Results Added

#### 0.2.1 CoT Sweeper2 (`qa_cot_then_final`, n=193)

| Run Dir | Max New Tokens | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---:|---:|---:|---:|---:|
| `strat_cot_sweep_t128_20260226T103204Z` | 128 | 0.1036 | 0.8497 | 29 | 0.6897 |
| `strat_cot_sweep_t192_20260226T104650Z` | 192 | 0.2591 | 0.6684 | 64 | 0.7812 |
| `strat_cot_sweep_t256_20260226T110535Z` | 256 | 0.4974 | 0.3212 | 131 | 0.7328 |
| `strat_cot_sweep_t320_20260226T112825Z` | 320 | 0.6684 | 0.1140 | 171 | 0.7544 |
| `strat_cot_sweep_t384_20260226T115617Z` | 384 | 0.6839 | 0.0518 | 183 | 0.7213 |

#### 0.2.2 A5 Strict Compliance Fix (`qa_binary_strict`, n=193)

| Run Dir | Max New Tokens | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---:|---:|---:|---:|---:|
| `my_run_baseline_direct_t16_20260226T125946Z` | 16 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `my_run_strict_t4_20260226T130128Z` | 4 | 0.5078 | 0.2383 | 147 | 0.6667 |
| `my_run_strict_t8_20260226T130206Z` | 8 | 0.5078 | 0.2383 | 147 | 0.6667 |
| `my_run_strict_t16_20260226T130259Z` | 16 | 0.5285 | 0.0363 | 186 | 0.5484 |
| `my_run_strict_t16_20260226T130431Z` | 16 | 0.5285 | 0.0363 | 186 | 0.5484 |

### 0.3 Updated Diagnosis

1. CoT quality was suppressed by parse failures, not only by reasoning quality.
   - Evidence: parseable-only accuracy is already high at low token budgets (`0.6897` at `t128`) despite very low total accuracy.
2. CoT with enough token budget becomes the strongest quality setup.
   - `t320/t384` clearly outperform direct baseline total accuracy.
3. Strict prompt is an effective compliance control, but not a full quality fix.
   - `strict_t16` sharply reduces parse errors (`0.0363`) but lowers parseable-only correctness (`0.5484`).
4. Extremely short strict budgets (`t4/t8`) over-truncate.
   - They do not improve either total accuracy or parse errors vs direct baseline.
5. Reproducibility is confirmed for strict runs.
   - `strict_t16` repeated with identical metrics and zero changed samples.

### 0.4 Fix Verdict By Problem

| Problem | Status | Notes |
|---|---|---|
| Parse/compliance failures | Partially fixed | Strongly improved by strict template or larger CoT budget. |
| Overall answer quality | Mixed | Best with large CoT budget; strict short format alone hurts quality. |
| Inference efficiency | Unfixed for high-quality mode | CoT `t320/t384` is accurate but expensive. |
| Deterministic reproducibility | Fixed | Repeat runs match exactly under no-sample seed-fixed settings. |

### 0.5 Next Plan (Actionable)

1. Keep two official baselines:
   - Quality baseline: CoT `t320` and `t384`.
   - Efficiency baseline: direct `t16` or `t32`.
2. Tighten strict-mode decoding boundaries:
   - stop sequences for `Human:`, `User:`, `[SYSTEM]`, `[END]`.
   - keep strict runs at small token limits (`16-32`) after stop fix.
3. Add extractor robustness checks:
   - test explicit variants like `yes.`, `no.` and leakage strings (`noHuman:`).
4. Run a targeted strict follow-up sweep after stop/extractor fix:
   - tokens: `16, 24, 32`.
   - compare against `baseline_direct_t16` and `strat_cot_sweep_t320`.
5. Freeze one benchmark profile for Phase A reporting:
   - deterministic, no-sampling, fixed artifact fingerprint, fixed evaluator version.

### 0.6 Parser/Decode Remedy Update (2026-02-26 late)

We implemented a targeted remedy for parse/compliance failures:
1. stronger StrategyQA extraction rules,
2. chat-leak truncation (`[USER]`, `Human:` etc.),
3. optional binary-choice decode mode (`yes` vs `no` scoring) to remove free-form formatting variance.

Quick re-evaluation on existing predictions (same generation outputs, new parser):

| Run Dir | Old Accuracy | Old Parse Error | New Accuracy | New Parse Error | New Accuracy on Parseable |
|---|---:|---:|---:|---:|---:|
| `strat_cot_sweep_t128_20260226T103204Z` | 0.1036 | 0.8497 | 0.1036 | 0.8446 | 0.6667 |
| `strat_cot_sweep_t320_20260226T112825Z` | 0.6684 | 0.1140 | 0.6684 | 0.1088 | 0.7500 |
| `strat_cot_sweep_t384_20260226T115617Z` | 0.6839 | 0.0518 | 0.6995 | 0.0518 | 0.7377 |
| `my_run_baseline_direct_t16_20260226T125946Z` | 0.5492 | 0.1762 | 0.5492 | 0.1762 | 0.6667 |
| `my_run_strict_t16_20260226T130259Z` | 0.5285 | 0.0363 | 0.6580 | 0.0000 | 0.6580 |

Interpretation:
- A non-trivial share of previous parse errors came from evaluation/code-side extraction brittleness (especially strict leakage forms like `noHuman:`).
- Remaining CoT parse errors at low token budgets are still largely model/protocol behavior (model does not emit an explicit binary decision often enough).
- To isolate true model decision quality from formatting noise, use `strategyqa_decode_mode=binary_choice` in A6 runs.

### 0.7 A6 Confirmation + Freeform-vs-Binary Diagnosis (2026-02-26)

New A6 run (`strategyqa_decode_mode=binary_choice`) result:

| label | n | accuracy | parse_error_rate | acc_parseable |
|---|---:|---:|---:|---:|
| `direct_binchoice` | 193 | 0.6788 | 0.0000 | 0.6788 |
| `cot_binchoice` | 193 | 0.5803 | 0.0000 | 0.5803 |
| `direct_binchoice_repro` | 193 | 0.6788 | 0.0000 | 0.6788 |

CoT A2 under binary-choice decode (token sweep):
- `t128/t192/t256/t320/t384` all reported exactly `acc=0.5803`, `parse_err=0.0000`.

Root-cause conclusion:
1. **Parse-error problem is mostly protocol/extraction/generation-format**, not pure model incapability.
   - Binary-choice decode eliminates parse errors to `0.0000`.
2. **Model-quality problem is still real after removing formatting noise**.
   - Direct binary-choice reaches `0.6788`, while CoT-prompt binary-choice is lower (`0.5803`).
3. **“Accuracy on parseable subset” can be misleading alone**.
   - Freeform CoT `t128` has `acc_parseable=0.6667` but only `n_parseable=30/193`.
   - Binary-choice gives full coverage (`193/193`) and is the fairer decision-quality view.

Important interpretation note:
- Under current implementation, StrategyQA `binary_choice` mode scores yes/no directly and does not depend on `max_new_tokens`.
- Therefore identical A2 token-sweep accuracies in binary mode are expected behavior, not a runtime bug.

Plan update:
1. Use two official StrategyQA metrics tracks:
   - **Decision quality track**: `binary_choice` (coverage fixed at 100%).
   - **End-to-end format track**: `freeform` (includes compliance burden).
2. Keep direct prompt as binary-choice baseline (`~0.6788`) for Phase A.
3. For CoT prompt, improve question framing/system instruction before attributing gap to model limits.

### 0.8 Phase A Closeout (Concise)

Conclusion:
- Phase A goals are met for benchmark infrastructure and diagnosis quality.
- We now have a reliable split between:
  - decision-quality evaluation (`binary_choice`), and
  - end-to-end formatting/compliance evaluation (`freeform`).

Achievements:
1. Deterministic one-click benchmark suite with param groups (`A1..A6`) and persisted logs.
2. Stable prepared-data pipeline and evaluation pipeline for StrategyQA.
3. Parse-error root cause identified and partially fixed in extraction/decoding.
4. Reproducibility validated (repeat runs produce identical results under fixed settings).

Problems probed and fixed:
1. Prompt-leak parse failures (`noHuman: ...`) fixed via extraction hardening and marker truncation.
2. Misleading evaluation from parseable-only subset mitigated by always reporting:
   - `n_parseable`,
   - `acc_parseable`,
   - and total-coverage metrics.
3. Free-form format variance isolated using `binary_choice` mode.

Open (unsolved) problems:
1. CoT prompt underperforms direct prompt on decision quality (`0.5803` vs `0.6788`).
2. Free-form CoT still has severe formatting burden at small budgets.
3. Phase A is inference-only; value learning and faithfulness-training components are not implemented yet.

To-do next (immediate):
1. Freeze Phase A benchmark protocol (artifact fingerprint + decode mode + reporting schema).
2. Start Phase B in separate scripts/modules:
   - model wrapper + value head,
   - BCR-lite losses,
   - first training smoke runs.
3. Keep Phase A scripts unchanged as regression baselines.

Re-planned route to final ABR/BCR goal:
1. Stage B: BCR-lite implementation and training stability.
2. Stage C: faithfulness calibration/corruption metrics and robust ablations.
3. Stage D: ABR heuristic router (`gen/ver/fin`) with fixed budget.
4. Stage E: ABR learned router (RL) only after Stage D is stable and reproducible.

### 0.9 GSM8K Smoke Criticality Scan and Fix (2026-02-26)

Observed critical symptom (direct baseline smoke, 20 samples):
- run: `gsm8k_direct_smoke_20260226T150012Z`
- `accuracy=0.0500`, `parse_error_rate=0.0000`
- extraction methods: 20/20 were `last_number`.

Diagnosis:
1. This was not a parser crash, but a weak extraction regime.
2. Model outputs were partial reasoning text under token cap; evaluator fell back to `last_number`.
3. Therefore parse-error stayed zero while answer quality collapsed.

Fixes implemented:
1. Added new prompt template `qa_math_direct_final@1.0.0`:
   - forces one-line `Final answer: <number>` format.
2. Extended math extraction:
   - supports both `Final answer: ...` and `Final answer is ...`.
3. Added math diagnostics in `phase_a_generate_and_eval.py`:
   - `last_number_rate`,
   - `hit_token_limit_rate`,
   - `final_answer_tag_rate`,
   - warning text when extraction looks unreliable.

Post-fix smoke validation:
1. run: `gsm8k_math_direct_smoke_20260226T150913Z` (`max_new_tokens=64`)
   - `accuracy=0.6000`, `parse_error_rate=0.0000`,
   - `last_number_rate=0.0000`,
   - `final_answer_tag_rate=1.0000`.
2. run: `gsm8k_math_direct_smoke_t16_20260226T151029Z` (`max_new_tokens=16`)
   - `accuracy=0.6000`, `parse_error_rate=0.0000`,
   - same smoke accuracy with much faster runtime.

Practical guidance:
1. For GSM8K direct baseline in Phase A, use `qa_math_direct_final`.
2. Start with `max_new_tokens=16` for fast baseline iteration.
3. Treat heavy `last_number` reliance as a quality warning, not as a healthy parse signal.

### 0.10 GSM8K Sweep Re-Diagnosis (2026-02-26 late)

New finding from your full GSM8K runs:
1. Direct sweep (`t16/t32/t64/t96`) stayed exactly at `accuracy=0.3663` with:
   - `parse_error_rate=0.0000`,
   - `hit_cap_rate=1.0000`,
   - `final_tag_rate=1.0000`.
2. CoT run (`t256`) looked worse at `accuracy=0.2733`, but diagnostics showed:
   - `final_answer_tag=119`,
   - `last_number=53`,
   - many tagged answers contained units/currency text (for example: `10 meters.`, `$140.`),
   - strict numeric comparison undercounted true correctness.

Code-side remedy:
1. Math evaluator now supports relaxed numeric equivalence when gold is numeric:
   - strict parse first,
   - fallback to first numeric token from predicted text (for example `75% ... -> 75`).
2. Math final-tag extraction now normalizes numeric answers from tagged text directly.

Re-evaluation (same prediction files, new evaluator):
1. `gsm8k_cot_t256_20260226T151603Z`:
   - old `accuracy=0.2733` -> new `accuracy=0.7035`.
2. `gsm8k_math_direct_t16_20260226T152107Z`:
   - old `accuracy=0.3663` -> new `accuracy=0.3721`.

Interpretation update:
1. CoT was strongly underestimated due to extraction strictness, not only model failure.
2. Direct baseline remains relatively stable around ~0.37 in this setup.
3. CoT now appears much stronger than direct on GSM8K under corrected numeric matching.
4. `hit_cap_rate=1.0` still indicates throughput waste; speed optimization remains required.

### 0.11 Foundation Reliability Hardening Gate (2026-02-27)

Reason:
1. We observed that low-level evaluation and parsing details can drastically shift conclusions.
2. To avoid Phase B/C being misled, we ran a base-component reliability hardening pass.

Completed hardening:
1. Binary-choice generation path now has branch-safe metadata handling.
2. Empty model outputs are treated as parse-error cases (no hard evaluator crash).
3. Loader split normalization now fails fast on split typos.
4. Metrics now carry evaluator version and comparison warns on version mismatch.
5. Prepared-input loader now rejects duplicate `sample_id`.

Artifacts:
1. Audit document: `foundation_reliability_audit.md`.
2. Execution checklist: `TODO_ours.md` section `18`.

### 0.12 Batched Inference Fix Validation (2026-02-27)

Problem:
1. Early batched freeform runs produced severe quality regression and repeated warnings:
   - decoder-only model warned about right-padding,
   - parse errors spiked and accuracy dropped vs `batch_size=1`.

Fix:
1. In batched freeform tokenization path, force `padding_side='left'` and restore tokenizer state after call.
2. Keep deterministic ordering and existing OOM backoff safety net.

Validation runs (StrategyQA, direct prompt, `max_new_tokens=32`, no-sample, seed=42, n=193):

| Run | Batch Size | Accuracy | Parse Error Rate | n_parseable | acc_parseable | gen_sample_rate | gen_elapsed_sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `strat_batch1_20260226T164044Z` | 1 | 0.5492 | 0.1762 | 159 | 0.6667 | 1.180 | 163.53 |
| `strat_batch4_fix_20260226T165830Z` | 4 | 0.5596 | 0.1658 | 161 | 0.6708 | 4.189 | 46.08 |

Inference:
1. Throughput gain is substantial (`~3.55x` sample/s, `~72%` less wall time).
2. Quality is not degraded after fix; metrics are slightly improved.
3. Current batching implementation is trustworthy for Phase A benchmarking at `batch_size=4` on this setup.

Next checks:
1. Run one reproducibility pair for `strat_batch4_fix` (same run-name, same seed) to verify `changed_samples=0`.
2. Repeat the same pair on GSM8K direct path to confirm cross-dataset batching parity.

### 0.13 Phase A Retrospective Lessons (Full-Period Consolidation)

This section consolidates what we learned across the full Phase A cycle, beyond single-run snapshots.

#### 0.13.1 Evaluation semantics can dominate conclusions

1. Parser/extractor design can move apparent accuracy dramatically without any model-weight change.
2. Never compare old/new metrics across evaluator logic changes unless predictions are rescored under the same evaluator version.
3. Keep `evaluator_version` in metrics and treat version mismatch as a hard caution.

#### 0.13.2 Always separate coverage from decision quality

1. `accuracy` alone is insufficient when parse failures are possible.
2. Always report this set together:
   - `accuracy`,
   - `parse_error_rate`,
   - `n_parseable`,
   - `acc_parseable`.
3. For StrategyQA, maintain two official tracks:
   - binary-choice (decision quality),
   - freeform (end-to-end format compliance).

#### 0.13.3 Prompt/protocol choices can outweigh token-budget tweaks

1. StrategyQA freeform CoT was highly sensitive to protocol compliance.
2. Binary-choice mode showed that much of the previous gap was format/channel noise.
3. Token budget sweeps are meaningful only after decode/eval protocol is stable.

#### 0.13.4 Throughput optimizations require explicit correctness guards

1. Batching initially improved speed but introduced a decoder-only padding bug that degraded quality.
2. Batch rollout must include:
   - padding-side correctness checks,
   - ordering parity checks,
   - deterministic rerun checks.
3. Throughput gains are valid only when metric parity is preserved.

#### 0.13.5 Truncation pressure is a real research confound

1. High hit-cap rates can hide weak extraction behavior and understate model quality.
2. Math runs need extraction diagnostics (`last_number_rate`, `final_answer_tag_rate`, cap rate) before interpretation.
3. Short-token baselines are useful for speed, but should not be treated as quality ceilings.

#### 0.13.6 Phase A final operational rule set (for Phase B carry-over)

1. Freeze baseline protocol before Phase B sweeps:
   - dataset fingerprint,
   - template/version,
   - decode mode,
   - evaluator version,
   - seed and no-sample setting.
2. Any infra change (parser/prompt/decode/batching) must be followed by:
   - replay on at least one stored prediction file,
   - one reproducibility pair run.
3. Treat infra regressions as P0, because they can invalidate all higher-level claims.

## 1. Purpose

This file records meaningful experiment outcomes so new teammates can quickly understand:
- what was run,
- what settings were used,
- what happened,
- and what conclusions we can trust.

Scope: Phase A inference/evaluation experiments on StrategyQA with local `Qwen2.5-7B-Instruct`.

## 2. Prepared Artifact Sets (Inputs)

These artifact folders were generated by `scripts/phase_a_prepare.py` and reused by inference runs.

| Fingerprint | Created (UTC) | Template | Target Style | Source Split | Split Policy | Limit | Validation File |
|---|---|---|---|---|---|---:|---|
| `21095d3c688a` | 2026-02-25T19:09:43.283792+00:00 | `qa_direct` | `answer_only` | `train` | `hash` | 200 | `assets/artifacts/phase_a_prepared/strategyqa/21095d3c688a/validation.jsonl` |
| `b0f373610f96` | 2026-02-26T07:13:22.068668+00:00 | `qa_direct` | `answer_only` | `train` | `hash` | 2000 | `assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl` |
| `f3e476b514c3` | 2026-02-26T07:18:52.297402+00:00 | `qa_cot_then_final` | `cot_then_answer` | `train` | `hash` | 2000 | `assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl` |

## 3. Early Smoke Runs (Small Validation Set)

These were used to verify pipeline wiring and deterministic rerun behavior.

| Run Dir | Created (UTC) | n | Max New Tokens | Accuracy | Parse Error Rate | Notes |
|---|---|---:|---:|---:|---:|---|
| `qwen_strategyqa_val_20260226T065337Z` | 2026-02-26T06:55:19.075627+00:00 | 2 | 32 | 1.0000 | 0.0000 | smoke |
| `qwen_strategyqa_val_20260226T065605Z` | 2026-02-26T06:56:41.906460+00:00 | 1 | 16 | 1.0000 | 0.0000 | smoke |
| `qwen_strategyqa_val_20260226T065859Z` | 2026-02-26T06:59:51.576498+00:00 | 17 | 64 | 0.6471 | 0.1176 | first stable mini run |
| `qwen_strategyqa_val_20260226T070029Z` | 2026-02-26T07:01:14.792043+00:00 | 17 | 64 | 0.6471 | 0.1176 | deterministic repeat |
| `qwen_strategyqa_val_20260226T070404Z` | 2026-02-26T07:04:50.085621+00:00 | 17 | 64 | 0.6471 | 0.1176 | deterministic repeat |

## 4. Main StrategyQA Runs (n=193)

All runs below are deterministic (`--no-do-sample`, seed=42).

| Run Dir | Created (UTC) | Input Prepared Set | Max New Tokens | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---|---|---:|---:|---:|---:|---:|
| `strat_vA_direct_20260226T084052Z` | 2026-02-26T09:05:30.421698+00:00 | `f3e476b514c3` (CoT prompt/target) | 128 | 0.1036 | 0.8497 | 29 | 0.6897 |
| `strat_vA_direct_20260226T090829Z` | 2026-02-26T09:28:41.357635+00:00 | `f3e476b514c3` (CoT prompt/target) | 128 | 0.1036 | 0.8497 | 29 | 0.6897 |
| `strat_direct_t32_20260226T093714Z` | 2026-02-26T09:40:12.219698+00:00 | `b0f373610f96` (Direct prompt/target) | 32 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_direct_t32_20260226T095322Z` | 2026-02-26T09:56:16.570779+00:00 | `b0f373610f96` (Direct prompt/target) | 32 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_direct_t32_20260226T100949Z` | 2026-02-26T10:12:38.609192+00:00 | `b0f373610f96` (Direct prompt/target) | 32 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_cot_t128_20260226T095113Z` | 2026-02-26T10:02:08.907577+00:00 | `f3e476b514c3` (CoT prompt/target) | 128 | 0.1036 | 0.8497 | 29 | 0.6897 |
| `strat_cot_t256_20260226T095906Z` | 2026-02-26T10:21:17.394167+00:00 | `f3e476b514c3` (CoT prompt/target) | 256 | 0.4974 | 0.3212 | 131 | 0.7328 |

## 5. Operational Incidents (Important Context)

### 5.1 GPU unavailable period (resolved later)
- During one period, CUDA/NVML was unavailable in runtime; jobs fell back to CPU and ran very slowly.
- Later checks showed GPU access restored (`torch.cuda.is_available() == True`, `nvidia-smi -L` normal).

### 5.2 CPU offload caused severe slowdown
- Run directory: `assets/artifacts/phase_a_runs/strat_cot_t128_20260226T094126Z`
- Symptom: `device_map=auto` offloaded tail layers to CPU (`Some parameters are on the meta device...`), causing very slow decode speed.
- Status: interrupted/incomplete run; only partial `predictions.jsonl` exists (no `metrics.json`).

## 6. Key Findings We Can Trust

1. Determinism is stable for fixed config.
   - Repeated runs with same settings produced identical metrics (and in key cases identical predictions).

2. Main bottleneck in CoT setting is format/completion, not core correctness.
   - At CoT `t128`, parse error is very high (`0.8497`) while accuracy on parseable subset remains much higher (`0.6897`).

3. Increasing CoT budget helps a lot but does not fully solve formatting issues.
   - `t128 -> t256` improved:
     - accuracy: `0.1036 -> 0.4974`
     - parse error rate: `0.8497 -> 0.3212`

4. Direct answer setup is currently the strongest robust baseline.
   - `strat_direct_t32` is fast and stable:
     - accuracy `0.5492`
     - parse error `0.1762`

## 7. Suggested Next Experiments (Queued)

To finish diagnosing truncation/compliance tradeoffs, run:
- CoT token sweep: `128, 192, 256, 320, 384`
- Direct token sweep: `16, 24, 32, 48, 64`
- Determinism repeat at one key CoT point (`t256`) to ensure run-to-run stability.

Keep fixed for fair comparisons:
- `--no-do-sample`
- `--seed 42`
- same input artifact file per variant
- no CPU offload (ensure model fully fits on selected GPU)

## 8. Direct Token Sweep Result (Completed)

Run family: `strat_direct_sweep_t*` on prepared input
`assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl`

| Run Dir | Max New Tokens | Elapsed (from console) | Throughput (sample/s) | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---:|---:|---:|---:|---:|---:|---:|
| `strat_direct_sweep_t16_20260226T103213Z` | 16 | 00:01:52 | 1.709 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_direct_sweep_t24_20260226T103417Z` | 24 | 00:02:47 | 1.155 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_direct_sweep_t32_20260226T103717Z` | 32 | 00:03:42 | 0.866 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_direct_sweep_t48_20260226T104125Z` | 48 | 00:05:25 | 0.592 | 0.5492 | 0.1710 | 160 | 0.6625 |
| `strat_direct_sweep_t64_20260226T104715Z` | 64 | 00:07:21 | 0.437 | 0.5544 | 0.1658 | 161 | 0.6646 |

### 8.1 Direct Sweep Analysis

1. Accuracy is mostly flat from `t16` to `t48`; only a small increase appears at `t64`.
2. Parse error improves only slightly with larger budgets (`0.1762 -> 0.1658`).
3. Runtime cost increases strongly as token budget grows; quality gain is small.
4. Outputs are still almost always near token cap even in direct mode, suggesting generation tends to keep talking until cutoff.
5. Prediction bias remains strong toward `no` on parseable outputs (roughly 85% `no`, 15% `yes`), so class-bias is still a key issue.

Operational inference:
- For current direct template, `t16`/`t24` are much more efficient and nearly as accurate as larger budgets.
- Purely increasing token budget is not enough to fix format/bias issues.

## 9. New Param Block Added to Address Current Problems

New one-click group in `scripts/run_phase_a_benchmark_suite.sh`:
- `ACTIVE_PARAM_GROUP=A5`

Goal of `A5`:
- improve answer-format compliance,
- reduce token waste,
- check whether stricter binary prompting helps yes/no behavior.

`A5` run plan:
1. baseline direct reference: `baseline_direct_t16`
2. strict binary prompt runs: `strict_t4`, `strict_t8`, `strict_t16`
3. strict determinism check: `strict_t16` repeated (run1 vs run2 with comparison enabled)

Template used by strict runs:
- `qa_binary_strict@1.0.0` (added in prompt builder)
- target style: `answer_only`

Recommended execution command:

```bash
ACTIVE_PARAM_GROUP=A5 \
CUDA_VISIBLE_DEVICES=3 \
RUN_PREFIX=strat_strict_fix_20260226 \
bash scripts/run_phase_a_benchmark_suite.sh
```

What to focus on for A5 results:
1. parse error rate vs `baseline_direct_t16`
2. total accuracy change
3. parseable-only accuracy
4. determinism (`delta_accuracy=0`, `changed_samples=0` on strict repeat)

## 0X. Follow-up Diagnosis (2026-03-11, Tiny-Terminal Hybrid Pilot)

### 0X.1 Why this follow-up mattered

After the first hybrid pilot, the key uncertainty was:

1. is `terminal_completion_anchor` still too large,
2. or is naive local+terminal mixture fundamentally too blunt even at tiny ratios?

### 0X.2 Tiny-terminal artifact

Artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_pbhybrid_tinyterm_0311__dd5acae29427`

Configured caps:

1. `prm_local`: train `3072`, val `384`
2. `prm_terminal`: train `64`, val `8`, mix weight `0.10`

Failure analysis later showed:

1. actual terminal semantics are only:
   - `17 / 3136 = 0.54%`

### 0X.3 Same-source fit

Run:

1. `assets/artifacts/phase_e_runs/phase_e_pbhybrid_tinyterm_0311_mlp_20260311T050119Z`

Held-out:

1. `pair_acc=0.918367`
2. `auc=0.890293`

So reducing terminal support does not damage source-family learnability.

### 0X.4 ProcessBench result

Comparison tables:

1. GSM8K:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_pbhybrid_terminal_ratio_gsm_compare_20260311T051627Z/summary.md`
2. Math:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_pbhybrid_terminal_ratio_math_compare_20260311T051627Z/summary.md`

Main comparison:

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

### 0X.5 What this changed in the diagnosis

This follow-up is highly informative:

1. terminal ratio clearly matters.
2. shrinking terminal semantics from `3.68%` to `0.54%` softens the over-correction.
3. but the result is still far below `E46` on benchmark-local discrimination.

Therefore the updated reading is:

1. the current failure is **not only** a ratio problem,
2. it is also an objective / supervision-contract problem,
3. so the next mainline should move toward:
   - staged training,
   - two-objective training,
   - or benchmark-aware checkpoint selection,
   not just more terminal-ratio sweeps.

## 0ZZZZZZ. Network-backed judge integration verdict (2026-03-11)

We re-checked the relevant judge literature and community best practices against our local runs. The most relevant references remain:

1. `JudgeLM`
2. `Prometheus / Prometheus 2`
3. `G-Eval`
4. `ProcessBench`
5. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`

The consolidated conclusion is:

1. `PRMBench_Preview`-style pairwise data is the only currently supported judge-facing training/data interface among the sources we tested.
2. `selected relabel` is supported in principle, but only on narrow slices; current evidence does **not** support full-dataset relabel.
3. `disagreement mining` remains one of the best uses of the local judge stack.
4. `benchmark-side adjudication` is safer and more realistic than `training-side bulk filtering`.

Local evidence:

1. Pairwise + swap-debias improved the *direction* of the judge setup, but did not produce a strong enough bulk filter.
2. `Qwen2.5-Math-7B-Instruct` on `PRMBench_Preview train64` only kept `4.69%` of pairs under the strict label-preserving criterion.
3. The same pipeline on `Math-Shepherd train64` kept `0%`.
4. Therefore:
   - keep `judge-assisted selected relabel`
   - keep `disagreement mining`
   - keep `benchmark-side adjudication`
   - drop `judge-driven bulk filtering` from the current mainline.

## 0YYYYYY. PRM-7B Backbone Breakthrough (2026-03-11 evening)

### Key finding: Backbone is the bottleneck, not training config

Multiple experiments on 2026-03-11 led to a clear conclusion:
using `Qwen2.5-Math-PRM-7B` as the backbone instead of `Qwen2.5-7B-Instruct`
gives **~2× improvement** on ProcessBench F1, regardless of training config.

### Experiment summary

| Run | Backbone | Val pair_acc | PB Math pb_f1 | PB GSM8K pb_f1 |
|---|---|---|---|---|
| PBR2a (ms_align_v1, joint+logit+conf_sem) | 7B-Instruct | ~0.51 | 0.186 | N/A |
| gated_mlp (dual head smoke, same config) | 7B-Instruct | 0.511, margin≈0 | (same config, same fail) | - |
| NDS ndsbh1 (rlhflow_align_v1, score space) | 7B-Instruct | 0.814 | 0.220 | 0.240 |
| Judge-1 (oracle filter + E87 config) | 7B-Instruct | 0.931 | 0.240 | 0.281 |
| **backboneproxy_prm_mixedsmall (2048 pairs)** | **PRM-7B** | **0.898** | **0.378** | **0.524** |

### Key discoveries

1. **Training config failure (Anti-pattern G)**: `joint + logit + confidence_semantic + lr=3e-5`
   causes near-zero gradient updates. The hinge loss at initialization = 0.713 (= margin 0.02 + BCE 0.693),
   mean_margin ≈ 5.7e-5. Fixed by: `ranking_only + score + confidence + lr=5e-5 + pair_acc selection`.

2. **Diag-E87config NaN failure**: ms_align_v1 raw data has terminal_completion_anchor pairs with very
   long texts (full solutions). These produce NaN/Inf features from the Instruct backbone.
   65% (6144/9478) of training pairs are affected. Oracle filter removes most terminal pairs → fixed.

3. **PRM-7B backbone superiority**: The `<extra_0>` token hidden states in PRM-7B carry step-level
   correctness assessments learned during PRM pretraining. `last_token` pooling extracts this.
   No amount of pair training on Instruct backbone can teach this from scratch.

4. **Oracle filter confirmation**: terminal_completion_anchor pairs pass filter at 12.6% rate
   (191/1511), proving the length-bias bug: both chosen and rejected are correct steps.
   After filter: terminal fraction drops 15.9% → 4.7%.

5. **ProcessBench F1 achieved** (best so far):
   - Math: 0.378 (target: 0.90+ for RL-ready, current SOTA from Qwen: 0.735)
   - GSM8K: 0.524

### Artifacts

- `assets/artifacts/phase_e_runs/phase_e_backboneproxy_prm_mixedsmall_20260311T074134Z/` — PRM-7B, 0.898 val
- `assets/artifacts/phase_e_eval/phase_e_eval_benchmark_20260311T084521Z/` — PB Math 0.378
- `assets/artifacts/phase_e_eval/phase_e_eval_benchmark_20260311T084946Z/` — PB GSM8K 0.524
- `assets/artifacts/phase_e_runs/phase_e_judge1_oracle_filter_e87config_20260311T080505Z/` — 7B-Instruct, 0.931 val

### Next experiment: PBR5-PRM7B (launched 2026-03-11 ~18:50)

- Backbone: Qwen2.5-Math-PRM-7B
- Pairs: oracle_filter_ms_align_v1 (4054 train, 462 val) — 2× more data than backboneproxy
- Config: ranking_only + score + confidence + lr=5e-5 + epochs=5
- Expected: val pair_acc > 0.93, PB Math pb_f1 > 0.40

## 0YYYYY. NDS Suite Results: math_step_dpo_v1 is competitive with PRM-7B (2026-03-11 evening)

### NDS ndsbh2 (math_step_dpo_v1 profile, 7B-Instruct)

Unexpected strong result: 7B-Instruct backbone with PURE LOCAL pairs (no terminal) achieves
comparable ProcessBench F1 to PRM-7B backbone.

| Run | Backbone | Pairs | Val pair_acc | PB Math pb_f1 | PB GSM8K pb_f1 |
|---|---|---|---|---|---|
| ndsbh1 rlhflow_align_v1 | 7B-Instruct | 3037 | 0.814 | 0.220 | 0.240 |
| ndsbh2 math_step_dpo_v1 | 7B-Instruct | 3705 | 0.742 | **0.379** | **0.363** |
| ndsbh3 ms_align_v1 (score config) | 7B-Instruct | 4096 | ~0.56 | 0.209 | 0.232 |
| backboneproxy PRM-7B | PRM-7B | 2048 | 0.898 | 0.378 | 0.524 |

### Key insight: pair semantics matter more than training config for 7B-Instruct

ndsbh2 uses `math_step_dpo_v1` profile = pure `ms_strict` (first_bad_edge_strict only),
NO terminal_completion_anchor pairs. Compared to ndsbh3 (ms_align_v1 = includes terminal):
- pb_f1: 0.379 vs 0.209 (+81% improvement!)
- Despite ndsbh2 having LOWER val pair_acc (0.742 vs ~0.56)

This proves: terminal pairs create a length bias that INVERTS ProcessBench pair_acc.
Removing terminal pairs entirely (ndsbh2) fixes the issue.

### Revised root cause understanding

Anti-pattern G (terminal length bias) is **dominant**:
- ms_align_v1 includes ~16% terminal anchor pairs → length bias → inverted PB pair_acc
- math_step_dpo_v1 (0% terminal) + same 7B-Instruct backbone → pb_f1=0.379
- This matches PRM-7B backbone (0.378) despite PRM-7B having much stronger representation

### When does PRM-7B backbone help?

- PB GSM8K: PRM-7B 0.524 vs ndsbh2 7B-Instruct 0.363 (+44%)
- PB Math: PRM-7B 0.378 vs ndsbh2 7B-Instruct 0.379 (parity!)

PRM-7B backbone gives larger gain on GSM8K (simpler arithmetic problems) where
the PRM's learned step quality signal is more directly applicable.

For complex Math problems, local pair discrimination is the bottleneck, not backbone.

### Current best recipe (as of 2026-03-11 ~19:00)

Option A: 7B-Instruct + pure local pairs (math_step_dpo_v1):
- pb_f1 Math: 0.379, GSM8K: 0.363
- No PRM-7B needed, fast training, caching works

Option B: PRM-7B + oracle filter (PBR5, running):
- Expected pb_f1 Math: 0.40+, GSM8K: 0.52+
- Better representation but slower (PRM-7B as backbone)

---

## 0ZZZZZZ — PBR5 实验结果：Oracle Filter + PRM-7B Backbone 突破（2026-03-11）

**实验**：PBR5 = oracle filter ms_align_v1（4054 train + 462 val）+ PRM-7B backbone + ranking_only + score space + lr=5e-5 + confidence weighting + pair_acc checkpoint selection

**训练 val 结果**（5 epochs）：
| Epoch | Pair Acc | AUC | Mean Margin |
|---|---|---|---|
| ep0 | 0.9848 | 0.9671 | 0.653 |
| ep1 | 0.9892 | 0.9716 | 0.669 |
| ep2 | **0.9935** | 0.9724 | 0.664 |
| ep3 | 0.9935 | 0.9786 | 0.686 ← best margin |
| ep4 | 0.9870 | 0.9783 | 0.680 |

Best checkpoint: ep2 (pair_acc=0.9935)

**ProcessBench 结果（256 样本，oracle threshold sweep）**：
| Benchmark | Pair Acc | Pair AUC | pb_f1 | pb_acc_err | pb_acc_cor | first_edge_acc |
|---|---|---|---|---|---|---|
| PB Math | 0.873 | 0.850 | **0.609** | 0.500 | 0.779 | 0.859 |
| PB GSM8K | 0.901 | 0.876 | **0.704** | 0.591 | 0.871 | 0.896 |

**完整实验对比表（7B-Instruct vs PRM-7B backbone）**：
| 实验 | Backbone | 数据 | Config | PB Math F1 | PB GSM8K F1 |
|---|---|---|---|---|---|
| PBR2a | 7B-Instruct | ms_align_v1 raw | joint+logit+lr=3e-5 | 0.186 | — |
| PBR3a (s42) | 7B-Instruct | ms_laterbad_v1 | joint+logit+lr=3e-5 | 0.207 | 0.221 |
| PBR3b (s1) | 7B-Instruct | ms_laterbad_v1 | joint+logit+lr=3e-5 | 0.176 | 0.206 |
| Judge-1 | 7B-Instruct | oracle filter | ranking_only+score | 0.240 | 0.281 |
| ndsbh2 | 7B-Instruct | math_step_dpo | ranking_only+score | **0.379** | 0.363 |
| backboneproxy | PRM-7B | mixed_small | group_bal+score | 0.378 | 0.524 |
| **PBR5** | **PRM-7B** | **oracle filter** | **ranking_only+score** | **0.609** | **0.704** |

**关键结论**：
1. PRM-7B backbone × oracle 过滤数据 = 协同增益（非线性叠加）
   - PRM-7B alone（mixed_small raw）: Math 0.378
   - oracle filter alone（7B-Instruct）: Math 0.240
   - PRM-7B + oracle filter（PBR5）: Math **0.609** = 2.5× over 7B-Instruct oracle
2. PBR5 val pair_acc=0.9935 是迄今最高，表明 oracle filter 给 PRM-7B 提供了近乎完美的训练信号
3. first_error_edge_accuracy（Math: 0.859，GSM8K: 0.896）远超之前所有方案
4. 下一步：PBR6（PRM-7B + 纯 local pairs math_step_dpo）对比 PBR5，确认 oracle filter vs zero-terminal 哪个更优

### 最优 Recipe（截止 2026-03-11）

**PBR5 Recipe**:
```
backbone: Qwen2.5-Math-PRM-7B
data: oracle_filter(ms_align_v1) → 4054 train pairs (12.6% pass rate)
pooling: last_token
objective: ranking_only
ranking_target_space: score
pair_weight_mode: confidence
lr: 5e-5, epochs: 5, batch_size: 96
checkpoint_selection: pair_acc
```
PB Math F1: **0.609** | PB GSM8K F1: **0.704**

