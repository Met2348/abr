# ProcessBench Transfer Diagnosis (2026-03-11)

## Goal

This note tracks one specific question:

1. why do current same-source Phase E winners still transfer only moderately to `ProcessBench`,
2. which parts of the mismatch look like training/eval geometry misalignment rather than generic optimization failure,
3. which minimal repair experiments are worth running next.

## Baseline Evidence

### Cross-run comparison

Artifact:

1. `assets/artifacts/phase_e_transfer_compare/processbench_baseline_compare_0311_20260310T161214Z/summary.md`

Current baseline table:

| case | benchmark | auc | pair_acc | first_edge | anygood_vs_firstbad | good_vs_laterbad | terminal_top1 | terminal_gap | pair_type_l1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ms_e68_math` | `processbench_math` | `0.5547` | `0.5809` | `0.5553` | `0.5768` | `0.5822` | `0.0517` | `-0.2900` | `1.8280` |
| `ms_e68_gsm` | `processbench_gsm8k` | `0.5885` | `0.6385` | `0.6294` | `0.6935` | `0.6117` | `0.1140` | `-0.2505` | `1.7101` |
| `ms_e14_math` | `processbench_math` | `0.5138` | `0.5523` | `0.4718` | `0.4734` | `0.5772` | `0.0936` | `-0.2346` | `1.8280` |
| `prm_e46_math` | `processbench_math` | `0.6053` | `0.5653` | `0.6096` | `0.6135` | `0.5501` | `0.1970` | `-0.2529` | `1.0000` |
| `prm_e46_gsm` | `processbench_gsm8k` | `0.6264` | `0.6701` | `0.6706` | `0.6987` | `0.6561` | `0.2332` | `-0.1895` | `1.0000` |

### Training vs benchmark structure

`ProcessBench Math` topology:

1. `40.6%` of examples are all-correct trajectories.
2. only `8.6%` of good-vs-bad pairs are `lastsafe_vs_firstbad`.
3. `76.0%` of good-vs-bad pairs involve either:
   - earlier-good vs first-bad,
   - or any good vs later-bad.
4. `48.1%` of good-vs-bad pairs are in `gap5p`.

Current strong training sources do not look like that:

1. `Math-Shepherd E68`
   - pure `lastsafe_vs_firstbad`
   - pure `gap1`
   - `terminal_anchor_fraction = 0`
2. `Math-Shepherd E14`
   - same local-only geometry
   - `terminal_anchor_fraction = 0`
3. `PRMBench E46`
   - local modified-error supervision only
   - `terminal_anchor_fraction = 0`
   - no step-metadata coverage for ProcessBench pair-type slices

### Current diagnosis

The baseline evidence points to two different transfer bottlenecks:

1. local-only pair geometry:
   - `Math-Shepherd` winners are almost entirely trained on the one easiest ProcessBench relation:
     - `lastsafe_vs_firstbad`
   - this explains the large pair-type distance on both `math` and `gsm8k`.
2. missing terminal supervision:
   - all current baseline sources have `terminal_anchor_fraction = 0`,
   - while `ProcessBench` contains a large all-correct slice,
   - and all current baselines show severe terminal collapse:
     - `terminal_top1` only `0.05` to `0.23`,
     - `terminal_gap` always negative.

Interpretation:

1. current Phase E is not generically incapable of ranking.
2. it is under-supervised for:
   - non-local good-vs-bad prefix comparisons,
   - and complete-correct terminal preference.

## Repair Matrix

### Prepared pair artifacts

1. `Math-Shepherd fanout`
   - artifact:
     - `assets/artifacts/phase_e_pairs/phase_e_pb_repair_0311_ms_fanout__005d9a0f02ac`
   - geometry:
     - `pair_type_l1 = 1.5206`
     - adds `earlygood_vs_firstbad`
     - still lacks later-bad and terminal support
2. `Math-Shepherd grid`
   - artifact:
     - `assets/artifacts/phase_e_pairs/phase_e_pb_repair_0311_ms_grid__8b322b320180`
   - geometry:
     - `pair_type_l1 = 0.9799`
     - exposes all four major ProcessBench pair types
     - still under-covers long-gap buckets relative to `ProcessBench Math`
3. `PRMBench terminal-anchor`
   - artifact:
     - `assets/artifacts/phase_e_pairs/phase_e_pb_repair_0311_prm_terminal__f5242c95f8f4`
   - train semantics:
     - `local_modified_process_error_step = 10962`
     - `terminal_completion_anchor = 3643`
   - explicit purpose:
     - repair all-correct terminal collapse directly

### Active experiment intent

1. `E71_MS_FANOUT`
   - warm-start from `E68`
   - ask whether broader `any-good vs first-bad` coverage alone lifts ProcessBench.
2. `E72_MS_GRID`
   - warm-start from `E68`
   - ask whether benchmark-like `good-vs-bad` grid supervision helps more than narrow fanout.
3. `E73_PRM_TERMINAL`
   - warm-start from `E46`
   - ask whether explicit terminal anchors improve all-correct ProcessBench behavior without destroying the already-strong first-edge behavior.

## Status

1. baseline diagnosis complete
2. repair artifacts prepared
3. `Math-Shepherd` smoke repairs completed
4. `PRMBench terminal-anchor` smoke completed

## Smoke Results

### `Math-Shepherd grid` smoke

Training artifact:

1. `assets/artifacts/phase_e_runs/phase_e_pb_repair_0311_e72b_ms_grid_smoke_20260310T162436Z`

Held-out pair fit:

1. `pair_acc = 0.9961`
2. `auc = 0.9733`

`ProcessBench Math 50`:

1. eval artifact:
   - `assets/artifacts/phase_e_eval/phase_e_pb_repair_0311_e72b_ms_grid_math50_20260310T170141Z`
2. direct metrics:
   - `pair_acc = 0.5854`
   - `auc = 0.4262`
   - `first_edge = 0.4615`

Interpretation:

1. same-source fit stays excellent.
2. but benchmark transfer is effectively unchanged versus `E68` on the same `Math 50` subset.
3. wider good-vs-bad grid supervision alone does **not** fix terminal collapse.

### `Math-Shepherd fanout` smoke

Training artifact:

1. `assets/artifacts/phase_e_runs/phase_e_pb_repair_0311_e71b_ms_fanout_smoke_20260310T165258Z`

Held-out pair fit:

1. `pair_acc = 0.9902`
2. `auc = 0.9593`

`ProcessBench Math 50`:

1. eval artifact:
   - `assets/artifacts/phase_e_eval/phase_e_pb_repair_0311_e71b_ms_fanout_math50_20260310T170437Z`
2. direct metrics:
   - `pair_acc = 0.5886`
   - `auc = 0.4199`
   - `first_edge = 0.5000`

Interpretation:

1. fanout slightly helps the local `first_bad` slice.
2. but overall benchmark AUC becomes worse.
3. this is consistent with fanout teaching more `any-good vs first-bad`,
   without addressing later-bad or terminal behavior.

### Same-subset comparison

Artifact:

1. `assets/artifacts/phase_e_transfer_compare/processbench_math50_compare_all_0311_20260310T170538Z/summary.md`

Key comparison on identical `Math 50` subset:

| case | auc | pair_acc | first_edge | anygood_vs_firstbad | good_vs_laterbad | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `E68 baseline` | `0.4251` | `0.5918` | `0.4615` | `0.5873` | `0.5929` | `0.0500` | `-0.2632` |
| `MS grid smoke` | `0.4262` | `0.5854` | `0.4615` | `0.5873` | `0.5850` | `0.0500` | `-0.2494` |
| `MS fanout smoke` | `0.4199` | `0.5886` | `0.5000` | `0.6032` | `0.5850` | `0.0500` | `-0.2420` |

What this establishes:

1. `fanout` improves exactly the slice it was designed to target:
   - `anygood_vs_firstbad`
   - and modestly `first_edge`
2. `grid` reduces pair-type distance more aggressively than `fanout`,
   but still does not improve terminal ranking in practice on this subset.
3. both repairs leave `terminal_top1 = 0.05`.
4. therefore the current bottleneck is not just missing benchmark-like pair geometry.
5. missing terminal-completion supervision remains the main unresolved gap.

### `PRMBench terminal-anchor` smoke

Training artifact:

1. `assets/artifacts/phase_e_runs/phase_e_pb_repair_0311_e73c_prm_terminal_smoke_20260310T170602Z`

Held-out pair fit:

1. `pair_acc = 0.9375`
2. `auc = 0.8526`

`ProcessBench Math 50` baseline and repair comparison:

1. baseline eval:
   - `assets/artifacts/phase_e_eval/phase_e_pb_repair_0311_e46_math50_20260310T170837Z`
2. repair eval:
   - `assets/artifacts/phase_e_eval/phase_e_pb_repair_0311_e73c_prm_terminal_math50_20260310T171708Z`
3. slice compare:
   - `assets/artifacts/phase_e_transfer_compare/processbench_prm_math50_compare_0311_20260310T171808Z/summary.md`

Key numbers:

| case | auc | pair_acc | first_edge | anygood_vs_firstbad | good_vs_laterbad | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `E46 baseline` | `0.5335` | `0.4778` | `0.5000` | `0.4762` | `0.4783` | `0.2000` | `-0.2688` |
| `PRM terminal smoke` | `0.5207` | `0.4272` | `0.5000` | `0.4921` | `0.4111` | `0.5500` | `-0.0382` |

Interpretation:

1. terminal anchors do the thing they were designed to do.
2. they dramatically improve the all-correct terminal slice:
   - `terminal_top1: 0.20 -> 0.55`
   - `terminal_gap: -0.2688 -> -0.0382`
3. but they also hurt the broader good-vs-bad ranking surface:
   - especially `good_vs_laterbad`
   - and total pair accuracy
4. this means the current repair is not “wrong”.
5. it is incomplete:
   - terminal supervision can be moved,
   - but it must be mixed with stronger later-bad ranking coverage.

### Final smoke-scale conclusion

Smoke-scale results now separate the failure modes cleanly:

1. `Math-Shepherd fanout`
   - helps `anygood_vs_firstbad` / `first_edge`
   - does not touch terminal collapse
2. `Math-Shepherd grid`
   - reduces geometric mismatch on paper
   - but still does not improve terminal behavior in practice
3. `PRMBench terminal-anchor`
   - strongly improves terminal behavior
   - but weakens broader good-vs-bad discrimination

Therefore the most defensible next mainline experiment is not:

1. more fanout alone
2. or more grid alone
3. or more terminal-anchor alone

It is a mixed recipe that combines:

1. terminal-anchor supervision,
2. explicit later-bad / good-vs-bad coverage,
3. and careful weighting so terminal gains do not erase the broader ranking surface.

## Infrastructure Follow-up

To support that next-step recipe directly, this round also added:

1. `scripts/phase_e_mix_pair_artifacts.py`
   - mixes existing canonical pair artifacts into one new deterministic artifact
   - avoids going back through raw-dataset adapters for every combination experiment
2. first mixed artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_pb_repair_0311_mixed_fanout_terminal__ada0800c3d71`
   - current ratio:
     - `fanout 3072`
     - `prm_terminal 1024`

This mixed artifact is the concrete starting point for the next mainline run:

1. preserve the local/later-bad help from `fanout`,
2. keep some explicit terminal anchors,
3. then tune weighting instead of treating either source as the whole answer.

## 2026-03-11 Engineering Corrections Before the Next Repair Sweep

The first `ProcessBench transfer` smoke pass exposed three engineering bugs that
could quietly invalidate the interpretation of small repair runs:

1. `ProcessBench` subset evaluation was previously a raw first-`N` slice.
   - This could accidentally remove the `all-correct` slice entirely.
   - Fix:
     - `src/ours/phase_e/benchmark_eval.py` now uses deterministic stratified subsampling across:
       - `all-correct`
       - `error`
2. `phase_e_analyze_processbench_failures.py` previously rebuilt prefix rows from
   the full benchmark file instead of the exact scored subset.
   - This could produce a false length-mismatch or silently analyze a different set.
   - Fix:
     - the script now filters the benchmark examples by `scored_rows.jsonl` example ids first.
3. terminal-anchor repairs were being silently dropped before training.
   - There were actually **two** truncation points:
     - `max_pairs_per_source` in `load_math_shepherd_pairs()`
     - `max_pairs_total` in `prepare_phase_e_pair_artifact()`
   - For `Math-Shepherd`, all-positive trajectories start much later in file order:
     - first observed all-positive row index: `121569`
     - total all-positive rows in the current mirror: `160920`
   - So a naive stream-head cap such as `max_pairs_per_source=20000` guarantees
     zero terminal anchors, even when `step_label_terminal_anchor_mode=all_positive_fanout`.

### New cap policy added in code

To make the repair experiments honest rather than nominal:

1. `load_math_shepherd_pairs()` now reinterprets `max_pairs_per_source` as a
   balanced per-semantics source budget when terminal anchors are enabled.
   - for the current `E83`-style recipe, this means:
     - `local_first_bad_edge`
     - `terminal_completion_anchor`
     each receive half of the source budget deterministically
2. `prepare_phase_e_pair_artifact()` now supports:
   - `global_cap_mode=pair_id_head`
   - `global_cap_mode=balanced_support_bucket`
3. the transfer wrapper `scripts/run_phase_e_processbench_transfer_suite.sh`
   now defaults its smoke mode to:
   - `PAIR_GLOBAL_CAP_MODE=balanced_support_bucket`

### Verified pair-artifact effect

Real pair-prep check after the fix:

1. artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_processbench_terminal_focus_0311_e83_ms_processbench_transfer_terminal_seed42_e83_ms_processbench_transfer_terminal_seed42_sharedsplit_s42_pairs__8b75a88516bc`
2. summary:
   - `overall_before_global_cap`
     - `local_first_bad_edge = 9985`
     - `terminal_completion_anchor = 9992`
   - `overall_after_global_cap`
     - `local_first_bad_edge = 4000`
     - `terminal_completion_anchor = 4000`
   - `train_after_split`
     - `local_first_bad_edge = 3595`
     - `terminal_completion_anchor = 3603`

This is the first time the `E83` recipe is confirmed to be training on a real
50/50 local-vs-terminal artifact rather than a fake terminal-augmented config
that still contained only local pairs.

## 2026-03-11 Corrected terminal-focused reruns (`E83` / `E84`)

With the plumbing fixed, we reran the terminal-heavy repairs on the corrected
`ProcessBench 96` smoke subset:

1. baseline control:
   - value run:
     - `assets/artifacts/phase_e_runs/phase_e_processbench_transfer_debug_e79_0311_e79_ms_processbench_transfer_baseline_seed42_s42_value_20260310T164516Z`
2. corrected terminal-only:
   - `E83_MS_PROCESSBENCH_TRANSFER_TERMINAL_SEED42`
   - value run:
     - `assets/artifacts/phase_e_runs/phase_e_processbench_terminal_focus_0311_e83_ms_processbench_transfer_terminal_seed42_e83_ms_processbench_transfer_terminal_seed42_s42_value_20260311T020535Z`
3. corrected fanout + terminal:
   - `E84_MS_PROCESSBENCH_TRANSFER_FANOUT_TERMINAL_SEED42`
   - value run:
     - `assets/artifacts/phase_e_runs/phase_e_processbench_terminal_focus_0311_e84_ms_processbench_transfer_fanout_terminal_seed42_e84_ms_processbench_transfer_fanout_terminal_seed42_s42_value_20260311T021431Z`

### Same-source held-out fit

| case | heldout pair_acc | heldout auc |
|---|---:|---:|
| `E79 baseline` | `0.9761` | `0.9251` |
| `E83 terminal` | `0.7731` | `0.7694` |
| `E84 fanout+terminal` | `0.7808` | `0.7756` |

### `ProcessBench GSM8K 96`

| case | pair_acc | auc | first_edge | all_correct_top1 | all_correct_gap |
|---|---:|---:|---:|---:|---:|
| `E79 baseline` | `0.6088` | `0.4999` | `0.6098` | `0.1087` | `-0.2821` |
| `E83 terminal` | `0.3163` | `0.3973` | `0.4878` | `0.6304` | `0.0788` |
| `E84 fanout+terminal` | `0.3299` | `0.4019` | `0.5122` | `0.6739` | `0.1097` |

### `ProcessBench Math 96`

| case | pair_acc | auc | first_edge | all_correct_top1 | all_correct_gap |
|---|---:|---:|---:|---:|---:|
| `E79 baseline` | `0.4558` | `0.4606` | `0.5745` | `0.0000` | `-0.3323` |
| `E83 terminal` | `0.3273` | `0.3864` | `0.5532` | `0.5641` | `0.0048` |
| `E84 fanout+terminal` | `0.3233` | `0.3906` | `0.4894` | `0.5897` | `0.0329` |

### Interpretation

These reruns settle the previously ambiguous question:

1. terminal anchors **do** attack the intended failure mode.
   - `all_correct_top1` moves from near-zero / near-random to `0.56` to `0.67`
   - `all_correct_gap` flips from strongly negative to slightly positive
2. but the current repair is badly over-weighted.
   - broad good-vs-bad `pair_acc` collapses on both `gsm8k` and `math`
   - same-source held-out fit also drops from `0.976` to about `0.78`
3. adding `fanout` on top of terminal anchors improves the training geometry:
   - `pair_type_l1_distance` drops from:
     - `1.7101 -> 1.3436` on `gsm8k`
     - `1.8280 -> 1.5206` on `math`
   - but it does **not** recover the lost benchmark ranking surface

Conclusion:

1. the old terminal-anchor hypothesis was scientifically correct,
2. but the naive `50/50` terminal-heavy mixture is not a viable mainline recipe,
3. so the next repair should reduce terminal-anchor mass and treat terminal
   supervision as a bounded auxiliary signal rather than a co-equal half of the
   training pool.

## 2026-03-11 Low-terminal RL-facing repair probe (`E87`)

To test whether the terminal signal was useful but simply overweighted, we
added a new repair axis:

1. keep `fanout` as the main error-side geometry,
2. cut terminal anchors down to `10%` of the source budget,
3. and respect pair confidence during optimization.

Artifact and run:

1. pair artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_processbench_rlrepair_0311_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_sharedsplit_s42_pairs__87812754052f`
2. value run:
   - `assets/artifacts/phase_e_runs/phase_e_processbench_rlrepair_0311_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_e87_ms_processbench_transfer_fanout_terminal10_confwt_seed42_s42_value_20260311T032957Z`
3. train semantics:
   - `first_bad_fanout_prefix_ranking = 6487`
   - `terminal_completion_anchor = 735`
   - terminal share:
     - `10.18%`

### Same-source / same-family readout

1. held-out pair fit:
   - `pair_acc = 0.8823`
   - `auc = 0.8647`
2. same-family trust eval:
   - artifact:
     - `assets/artifacts/phase_e_samefamily_eval/phase_e_rlready_e87_samefamily_0311_20260311T040021Z`
   - `prompt_pool_top1 = 0.6597`
   - `local_first_bad_acc = 0.2914`

### `ProcessBench GSM8K 96`

1. direct metrics:
   - `pair_acc = 0.4932`
   - `auc = 0.4410`
   - `first_edge = 0.5854`
2. failure slices:
   - `all_correct_top1 = 0.3696`
   - `all_correct_gap = -0.0819`
   - `late_error_pair_acc = 0.5979`

### `ProcessBench Math 96`

1. direct metrics:
   - `pair_acc = 0.4217`
   - `auc = 0.4467`
   - `first_edge = 0.5957`
2. failure slices:
   - `all_correct_top1 = 0.2051`
   - `all_correct_gap = -0.1314`
   - `late_error_pair_acc = 0.6125`

### Interpretation

This is the best current *tradeoff* repair, but it is still not RL-ready.

1. compared with `E84`, it clearly recovers much more of the benchmark ranking
   surface:
   - `gsm8k pair_acc: 0.3299 -> 0.4932`
   - `math pair_acc: 0.3233 -> 0.4217`
2. compared with the strict baseline, it still remains below the benchmark
   frontier:
   - `gsm8k pair_acc: 0.6088 -> 0.4932`
   - `math pair_acc: 0.4558 -> 0.4217`
3. terminal collapse is improved relative to the baseline, but not solved:
   - `gsm8k all_correct_top1: 0.1087 -> 0.3696`
   - `math all_correct_top1: 0.0000 -> 0.2051`
4. the same-family trust signal is too weak for any RL claim:
   - `prompt_pool_top1 = 0.6597`
   - `local_first_bad_acc = 0.2914`

Strict transfer diagnosis:

1. artifact:
   - `assets/artifacts/phase_e_transfer_diag/phase_e_transfer_diag_e87_0311_00/summary.md`
2. result:
   - `strict_rl_ready = 0`
   - `assessment = not_rl_ready_terminal_completion_risk`

Operational conclusion:

1. lowering terminal-anchor mass is the right direction,
2. but data-ratio repair alone is insufficient,
3. the next mainline should move from "change the pair mix" to
   "change the training strategy":
   - staged/curriculum repair,
   - semantics-aware loss weighting,
   - and same-family trust gating during model selection.

## 2026-03-11 Internet-guided redesign + new smoke suite

The latest internet scan sharpened the design rule for this repo.

Primary sources checked:
1. `Let's Verify Step by Step (PRM800K)`:
   - https://arxiv.org/abs/2305.20050
2. `OmegaPRM`:
   - https://arxiv.org/abs/2406.06592
3. `ProcessBench`:
   - https://arxiv.org/abs/2412.06559
   - https://github.com/QwenLM/ProcessBench
4. `PRMBench`:
   - https://arxiv.org/abs/2501.03124
5. `The Lessons of Developing PRMs`:
   - https://arxiv.org/abs/2501.07301
6. `PathFinder-PRM`:
   - https://arxiv.org/abs/2501.11690
   - https://github.com/Gen-Verse/PathFinder-PRM

### Literature-backed design implications

The sources above point to one coherent reading:

1. process/reward heads do learn on high-quality local supervision,
2. but transfer fails when the evaluation geometry contains relations that the
   training support barely covers,
3. and the right fix is usually not "replace local supervision" but
   "keep the local core and add bounded auxiliary support for harder slices".

That changed the local design rule from:
1. repair terminal collapse by adding more terminal anchors

to:
1. keep local first-error discrimination as the primary anchor,
2. add broader `fanout / grid / terminal` support conservatively,
3. and weight those auxiliary families explicitly in the loss.

### New local implementation

Code added in this round:
1. `scripts/phase_e_curate_processbench_transfer_pairs.py`
   - fixed profiles:
     - `ms_core_v1`
     - `ms_align_v1`
     - `ms_prm_align_v1`
2. `src/ours/phase_e/training.py`
   - new pair-weight modes:
     - `semantic`
     - `confidence_semantic`
3. `scripts/run_phase_e_processbench_research_suite.sh`
   - one-click pipeline:
     - curate
     - train
     - same-family eval
     - `ProcessBench` eval
     - RL-promotion diagnosis

The key methodological change is explicit `semantic_weight` metadata.

Interpretation:
1. `local_first_bad_edge` remains the primary trust anchor,
2. `good_bad_prefix_grid` and `terminal_completion_anchor` are kept,
   but are down-weighted instead of being treated as equal-mass truth.

### Smoke run actually executed

Executed command:

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

Primary artifacts:
1. suite summary:
   - `assets/artifacts/phase_e_logs/phase_e_processbench_research_v2/final_summary.md`
2. raw per-case rows:
   - `assets/artifacts/phase_e_logs/phase_e_processbench_research_v2/research_results.jsonl`
3. transfer compare:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_processbench_research_v2_gsm_compare_20260311T044545Z/summary.md`
   - `assets/artifacts/phase_e_transfer_compare/phase_e_processbench_research_v2_math_compare_20260311T044545Z/summary.md`
4. RL-promotion diagnosis:
   - `assets/artifacts/phase_e_rl_promotion_diag/phase_e_processbench_research_v2_rl_promotion_20260311T044546Z/summary.md`

### Result summary versus `ref_e87`

| case | heldout_pair | heldout_auc | samefamily_top1 | samefamily_local | gsm_auc | gsm_first_edge | gsm_terminal_top1 | math_auc | math_first_edge | math_terminal_top1 | RL diagnosis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ref_e87` | `0.8823` | `0.8647` | `0.6597` | `0.2914` | `0.4410` | `0.5854` | `0.3696` | `0.4467` | `0.5957` | `0.2051` | `terminal_and_local_tradeoff_unresolved` |
| `pbr1_ms_align_mlp` | `0.8504` | `0.8344` | `0.8316` | `0.8681` | `0.4754` | `0.3600` | `0.4839` | `0.4692` | `0.2857` | `0.1154` | `terminal_and_local_tradeoff_unresolved` |
| `pbr2_ms_align_gated` | `0.8819` | `0.8723` | `0.8947` | `0.9231` | `0.4713` | `0.5600` | `0.5806` | `0.5055` | `0.5357` | `0.6154` | `not_rl_ready_laterbad_generalization_weak` |
| `pbr4_ms_curriculum_gated` | `0.8661` | `0.8572` | `0.8316` | `0.8791` | `0.4947` | `0.4400` | `0.7419` | `0.4743` | `0.4286` | `0.6538` | `not_rl_ready_laterbad_generalization_weak` |

### What this establishes

1. `pbr2_ms_align_gated` is the best new candidate.
2. compared with `ref_e87`, it clearly improves:
   - same-family utility,
   - `ProcessBench Math` AUC,
   - terminal-completion behavior on both splits.
3. however it still does **not** clear the RL-facing gate, because:
   - `gsm first_edge = 0.5600`
   - `math first_edge = 0.5357`
   - `pb_min_laterbad = 0.4792`
4. `pbr4_ms_curriculum_gated` pushes terminal completion even harder,
   but it damages same-family trust and does not beat `pbr2` on the more useful
   transfer metrics.
5. `pbr1_ms_align_mlp` confirms that the new curate profile alone is not
   enough; architecture/training still matter.

### Updated conclusion

The repository's current best reading is now:

1. the old blocker description was incomplete.
2. it is no longer enough to say:
   - terminal completion is missing.
3. after the new run, the next main blocker is better described as:
   - `later-bad generalization remains weak while first-edge must still be preserved`.
4. therefore the next mainline should be:
   - keep `ms_align_v1`,
   - keep `confidence_semantic`,
   - keep `gated_mlp` as the strongest current candidate,
   - and specifically target better `good_vs_laterbad` transfer without losing
     the current `first_bad` edge.
5. the current curriculum variant should **not** be promoted.
6. none of the current cases are RL-ready yet.
7. engineering note: even on `A100 80GB`, benchmark-native eval and the curriculum core stage hit OOM backoff more than once, so runtime stability is still below the standard we want before any Phase F claim.
