# Phase E Plan: High-Quality Pair Benchmark Validation

This file defines the official **Phase E** execution plan.

Date baseline: 2026-03-10.

## 0.0 Infrastructure Hardening Rule (2026-03-11 update)

Phase E now treats certain recipe families as infrastructure risk, not just as
"bad experiment outcomes".

Current operational rule:
1. the default Phase E trainer path uses `--recipe-risk-policy error`;
2. known catastrophic combinations should be blocked before backbone load;
3. every new Phase E run should emit a structured training-health artifact,
   rather than relying on manual log reading alone.

Why this was added:
1. recent `NDSBH / PBR` evidence showed that mixed-semantics math artifacts can
   catastrophically collapse under:
   - `ranking_target_space = logit`
   - semantic-style pair weighting
   - `checkpoint_selection_metric = ranking_score`
2. if these recipes are allowed to run silently, future source-quality
   comparisons become methodologically polluted.

Operational consequence:
1. all active `run_phase_e*.sh` wrappers should pass explicit `--recipe-risk-policy`;
2. direct trainer use should assume:
   - `checkpoint_selection_metric = pair_acc`
   unless the run is a controlled diagnostic;
3. evidence collected through stale wrappers should be treated as lower-trust until rerun.

Phase transition rule:
1. Phase D is not discarded.
2. Phase D is now treated as the methodology-correction and bridge-evidence stage.
3. Phase E becomes the new active mainline:
   - validate value-head learnability on datasets that genuinely provide
     high-quality process/pair supervision,
   - then transfer the validated method back to StrategyQA.

Reference redesign note:
1. `docs/phase_e_pipeline_redesign_20260311.md`
2. `docs/phase_e_updated_literature_redesign_20260311.md`

## 0.3 2026 Literature Refresh

Newer evidence beyond the older `2025-03` assumption set now matters:
1. `PRIME (2026)` says verifier quality on process-outcome alignment strongly
   predicts RLVR effectiveness.
2. `Hard2Verify (2025)` shows open-source step verifiers still lag badly on
   frontier open-ended verification.
3. `RISE (2025)` treats self-verification as a first-class online training
   signal, not just an offline scalar label.
4. `VPRM (2026)` shows deterministic process verification can outperform
   outcome-only or opaque-neural-only approaches in structured settings.

Operational interpretation:
1. same-source pair fit is necessary but no longer sufficient;
2. Phase E should now develop along two parallel tracks:
   - bounded-support scalar verifier,
   - benchmark-aligned process / critique verifier.
3. a pure `cheap verifier -> rare strong verifier escalation` system is also
   not yet supported by local evidence:
   - current `ms_e43 -> pbr26` and `prm_e46 -> pbr26` gate sweeps only approach
     strong-verifier AUC when strong usage rises to roughly `86%-97%`.
4. therefore the nearer-term architectural target should be:
   - local/process verifier,
   - terminal/answer verifier,
   - abstain/escalate gate,
   not one scalar verifier with a thin escalation wrapper.

---

## 0. Why Phase E Starts Now

Recent evidence changed the project scope materially.

What Phase D already established:
1. `DT2_MATH_SHEPHERD_SEED3_STABLE`
   - proved the ranking branch is learnable on high-quality external triplets.
2. `DT2_MATH_SHEPHERD_SEED3_STABLE_C1_TRANSFER`
   - proved direct transfer back to StrategyQA stays near-random.
3. `DT3_PRM800K_SEED3_STABLE`
   - proved `PRM800K` under the current adapter is a stable weak source.
4. `DB3_STRATEGYQA_BRIDGE_FULL_RANK_SEED3`
   - proved external ranking pretrain + in-domain ranking continue training can
     improve StrategyQA corruption ranking.
5. `DB4_STRATEGYQA_BRIDGE_FULL_JOINT_SEED3`
   - proved scalar/joint calibration objectives can actively harm ranking even
     when Brier improves.

The decisive strategic diagnosis:
1. The current learning object is better described as `process ranking` than as
   `scalar value regression`.
2. StrategyQA does not provide public PRM-grade step-quality supervision.
3. Therefore StrategyQA should no longer be the primary supervised benchmark for
   validating the value-head idea.

This is the exact trigger for Phase E.

## 0.1 Community / Literature Diagnosis

Community evidence is consistent with our current direction change.

What current papers and benchmarks imply:
1. `Let's Verify Step by Step (PRM800K)` established the modern empirical case
   for process supervision:
   - step-level supervision can beat outcome-only supervision on mathematical
     reasoning,
   - so a reward/value-style head is not inherently doomed if the supervision
     is strong enough.
2. `Math-Shepherd` further showed that step-level good/bad process labels can
   train a practically useful verifier/reward model that improves downstream
   math reasoning.
3. `OmegaPRM` showed that large-scale automated process supervision can still
   produce strong gains when the collection pipeline preserves first-error
   localization and balanced positive/negative signal.
4. `SVPO` and `R-PRM` both support the narrower claim we care about:
   - value/preference learning on step-quality data can be effective inside the
     source family,
   - and better pair construction materially affects downstream quality.
5. `ProcessBench` was introduced because existing PRMs often do not generalize
   well to harder explicit process-error detection settings.
6. `PRMBench` likewise exists because fine-grained process verification remains
   weak for many current reward/value models.
7. `The Lessons of Developing PRMs` argues that data construction quality is a
   first-order factor, and that weaker synthetic labels can underperform
   stronger judge/human-style supervision.
8. `VersaPRM` responds to poor cross-domain behavior by moving to broader
   multi-domain process supervision.
9. `ThinkPRM` suggests that stronger process verification may require a
   different model family (generative verifier), not just a small
   discriminative scalar head.
10. `Do We Need to Verify Step by Step?` provides an important theoretical
    caution:
    - process supervision is not automatically superior by statistical law,
    - so empirical success depends heavily on algorithm and data pipeline
      quality, not on naming the signal a "process reward".

Operational implication:
1. Cross-dataset transfer should be treated as difficult by default.
2. Therefore the first Phase E success criterion should not be "train on one
   source and immediately generalize elsewhere".
3. The primary scientific question is narrower:
   - can the value/ranking head be learned reliably inside one
     high-quality benchmark family?
4. For the current repository, the relevant claim is therefore:
   - prove `same-family learnability` first,
   - downgrade `cross-family generalization` from near-term objective to future
     research.
5. If `Math-Shepherd` succeeds and we still stay in the same family for the
   next stage, the correct use of that result is not "assume we solved general
   value estimation".
6. The correct interpretation is narrower:
   - we have obtained one bounded-support process reward model that may guide
     conservative RL inside the same distribution.

## 0.2 Empirical Diagnosis: Head Capacity vs Training Sufficiency

The intradataset ACC90 branch already produced a useful structural diagnosis.

What is now supported by evidence:
1. `Math-Shepherd`
   - `E40_MS_ACC90_LINEAR_ROBUST_SEED3`
   - `mean_heldout_pair_acc = 0.9172`
   - `mean_heldout_auc = 0.8623`
   - therefore the linear head is already sufficient to solve the same-source
     held-out discrimination task.
2. `Math-Shepherd` still benefits from a stronger head:
   - `E41/E42/E43` move same-source performance into the `0.96 ~ 0.99` pair
     accuracy range.
   - therefore higher capacity helps, but here it is an improvement over a
     working baseline, not a rescue from complete underfitting.
3. `PRMBench_Preview`
   - `E44_PRMBENCH_ACC90_LINEAR_SEED3`
     - `mean_heldout_pair_acc = 0.7380`
   - `E45_PRMBENCH_ACC90_MLP_RANK_SEED3`
     - `mean_heldout_pair_acc = 0.9315`
   - `E46_PRMBENCH_ACC90_MLP_JOINT_SEED3`
     - `mean_heldout_pair_acc = 0.9309`
   - therefore this source does provide strong evidence that the linear head
     can be too simple.
4. Weak runs cannot be explained by under-training alone:
   - `E12_MS_TRUST_LOWLR_SEED3`
   - `mean_heldout_pair_acc = 0.5853`
   - `mean_heldout_auc = 0.5856`
   - so "lower LR + more epochs" is not a universal explanation.

Methodological implication:
1. head capacity is a source-specific issue, not one global yes/no diagnosis;
2. future Phase E design should treat architecture choice as
   dataset-conditioned;
3. high same-source held-out ACC still does **not** imply cross-task
   trustworthiness.

## 0.3 RL Trustworthiness: What Is Still Missing?

The current literature does **not** support one universal rule such as:
1. `pair_acc > 0.90` means a value head is already ready for RL,
2. or `auc > 0.90` means the utility is already trustworthy.

The more defensible consensus is operational:
1. strong same-source held-out discrimination is necessary;
2. it is not sufficient;
3. reward/value models that will be used inside optimization loops must also
   remain meaningful under stronger selection pressure.

### Literature-guided reading

1. `PRM800K / Let's Verify Step by Step`
   - process supervision can produce useful reward signals,
   - but usefulness is demonstrated through downstream policy gains, not one
     isolated held-out scalar metric.
2. `ProcessBench`
   - many apparently competent PRMs still fail explicit process-error
     identification.
3. `PRMBench`
   - process verification quality is multi-dimensional and harder than simple
     pair classification.
4. `The Lessons of Developing PRMs`
   - label quality, evaluation design, and optimization behavior dominate
     naive comfort metrics.
5. `ThinkPRM`, `R-PRM`, `VersaPRM`
   - stronger verifier formulations and richer supervision are often needed
     when the final goal is robust process judgement under optimization.

### Practical trust rule for this repository

For the current RL-faithfulness objective, a head should only be called
`trustworthy enough for bounded same-source RL use` if:
1. same-source held-out discrimination is strong:
   - `mean_pair_acc >= 0.90`
   - `mean_auc >= 0.90`
   - low seed variance
2. the behavior is stable:
   - no worst-seed collapse,
   - no obvious margin collapse,
   - no brittle flip under a small recipe change
3. same-family policy-level benefit is shown:
   - reranking / rejection / conservative search should improve answer or
     process quality inside that same dataset family
4. same-family local-faithfulness checks are positive
5. even then, the correct narration remains:
   - bounded-support utility,
   - not universal verifier.

### What we already have

1. `Math-Shepherd`
   - same-source held-out metrics are already strong:
     - `E40`: `0.9172 / 0.8623`
     - `E41`: `0.9863 / 0.9056`
     - `E42`: `0.9641 / 0.9408`
     - `E43`: `0.9619 / 0.9425`
2. `PRMBench_Preview`
   - same-source held-out metrics are strong once `MLP` is used:
     - `E45`: `0.9315 / 0.8711`
     - `E46`: `0.9309 / 0.9057`

### What is still missing

1. same-family policy-improvement evidence
   - we have not yet shown that these heads reliably improve reranking,
     branching, or conservative search on their own source family
2. same-family local-faithfulness evidence
   - we have pair discrimination,
   - but not yet a stronger same-family local-error gate for all candidate
     heads
3. robustness under optimization pressure
   - we have not yet shown that high-scoring processes are not merely
     exploiting source-specific shortcuts

Therefore the current correct statement is:
1. same-source learnability is now established on some sources;
2. RL-level trustworthiness is **not yet** established, even after ignoring
   cross-dataset transfer.

### 0.1A Data-Semantics Risk Update

Mentor audit identified one adapter-level 一级风险, and this has now been
repaired in code.

Previous risk:
1. `Math-Shepherd` / `RLHFlow` / `PRM800K` fallback `+/-` all reused one
   nearest-negative converter.
2. That converter built different-depth prefix pairs from a single trajectory.
3. So old artifacts mixed:
   - progress/depth signal,
   - local error signal.

Current remediation:
1. Default `step_label_pair_mode` is now `first_bad_edge_strict`.
2. For single-trajectory `+/-` sources, we now only keep:
   - the last clean prefix before the first negative step,
   - versus the prefix that includes the first bad step.
3. Samples with no positive step before the first negative step are dropped.
4. New artifacts explicitly record:
   - `pair_build_mode`
   - `pair_semantics`
5. Artifact stages are bumped:
   - `phase_e_pairs_v2`
   - `phase_d_external_pairs_v2`

Interpretation rule after the fix:
1. `Math-Shepherd` / `RLHFlow` are still **not** true same-step sibling-pair
   sources.
2. They are now treated as:
   - strict local `first_bad_edge` supervision sources.
3. Therefore new results may be narrated as:
   - local first-bad-edge ranking learnability,
   but still not as:
   - exact same-step branch preference learning.
4. Pre-fix artifacts/results should be treated as `legacy` and should not be
   pooled with post-fix runs.
5. Detailed audit record:
   - `docs/data_semantics_risk_audit_20260310.md`

### 0.1B Internet-Guided ProcessBench Redesign Update (2026-03-11)

The newest internet literature/community scan changed the practical reading of
the current `ProcessBench` blocker.

Sources checked in this round:
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

What these sources jointly imply for this repo:
1. same-source learnability is real,
   but benchmark transfer remains structurally hard;
2. preserving local first-error supervision is still necessary;
3. wider `later-bad` / completion support should be added as bounded auxiliary
   supervision, not as an equal-mass replacement;
4. hierarchy-aware or expert-style heads can be worth trying,
   but they only matter after the training support is re-curated coherently.

That directly led to a new local redesign:
1. new curated profiles:
   - `ms_core_v1`
   - `ms_align_v1`
   - `ms_prm_align_v1`
2. new trainer weighting modes:
   - `semantic`
   - `confidence_semantic`
3. new research wrapper:
   - `scripts/run_phase_e_processbench_research_suite.sh`

Current smoke outcome on `2026-03-11`:
1. the best new case is `pbr2_ms_align_gated`
   - same-family:
     - `top1 = 0.8947`
     - `local_first_bad = 0.9231`
   - `ProcessBench GSM8K`:
     - `auc = 0.4713`
     - `first_edge = 0.5600`
     - `terminal_top1 = 0.5806`
   - `ProcessBench Math`:
     - `auc = 0.5055`
     - `first_edge = 0.5357`
     - `terminal_top1 = 0.6154`
2. compared with `E87`:
   - AUC improved on both `gsm8k` and `math`,
   - terminal completion improved strongly on both,
   - same-family trust improved materially,
   - but `first_edge` and `good_vs_laterbad` still stay below the current
     RL-facing gate.
3. current diagnosis therefore changed from:
   - `terminal signal missing`
   to:
   - `later-bad generalization and benchmark-first-edge preservation remain the
     next blocking canaries`.

Operational conclusion:
1. keep the conservative aligned curate profile and semantic weighting;
2. keep `gated_mlp` as the strongest current research candidate;
3. do not promote the current curriculum variant to the mainline;
4. the next repair should target:
   - `later-bad` generalization,
   - while explicitly preserving `first_edge`,
   - instead of continuing to push terminal anchors harder.

### 0.1C Second Internet Scan And Experiment Design (2026-03-11 — second pass)

Additional papers found in the second internet scan (beyond the first-round coverage):

1. `ThinkPRM` (generative step verifier, +7.2% F1 on ProcessBench vs scalar PRM):
   - https://arxiv.org/abs/2504.09418
2. `GenPRM` (AAAI 2026, generative PRM with visible rationale):
   - https://arxiv.org/abs/2504.00891
3. `PathFinder-PRM` (hierarchical error-typing, PRMBench SOTA 67.7):
   - https://arxiv.org/abs/2501.11690
4. `PRIME` (online implicit PRM via DPO log-ratio reward):
   - https://arxiv.org/abs/2502.01456
5. `VersaPRM` (multi-domain self-filtering):
   - https://arxiv.org/abs/2504.08481
6. `OmegaPRM` (MCTS sibling-branch pairs, binary-search first-error):
   - https://arxiv.org/abs/2406.06592
7. `Full-Step-DPO` (DPO applied at every step, not just outcome):
   - related to step-level DPO work
8. `BiPRM` (bidirectional backbone for richer local features):
   - see VersaPRM discussion

Key joint lesson from this round:
1. the strongest frozen-head scalar PRMs on ProcessBench all rely on
   richer supervision geometry than local-first-bad only;
2. `all_good_vs_all_bad` grid supervision is the closest in-dataset
   approximation to MC-step estimation without additional rollouts;
3. inter-seed variance at smoke scale (4096 pairs) is expected to be
   high; scaling to 16384 pairs is the minimum for stable estimates.

New data profile added:
1. `ms_laterbad_v1` (in `scripts/phase_e_curate_processbench_transfer_pairs.py`):
   - `ms_strict` 40% (unchanged vs ms_align_v1), semantic_weight 1.0
   - `ms_fanout` 20% (unchanged vs ms_align_v1), semantic_weight 0.9
   - `ms_laterbad` 30% — `all_good_vs_all_bad` filtered via `pair_type_allowlist`
     to only `lastsafe_vs_laterbad` + `earlygood_vs_laterbad` pair types, semantic_weight 0.85
   - `ms_terminal` 10% (unchanged), semantic_weight 0.40
   - Key: the `pair_type_allowlist` filter (via `classify_step_label_pair_type`
     in `src/ours/phase_e/processbench_alignment.py`) ensures only later-bad
     pair types are included, avoiding overlap with strict/fanout coverage.
   - Required new infrastructure: `CurateComponent.pair_type_allowlist` field,
     `_filter_rows_by_pair_type` function in the curate script.

New experiment groups added to `scripts/run_phase_e_processbench_research_suite.sh`:
1. `PBR2_FULL_MIXED_MLP_SEED3`:
   - profile: `ms_align_v1`, head: `mlp`, 3 seeds (42/1/7)
   - target pairs: 16384 (via `GROUP_TARGET_PAIRS=16384`)
   - success criterion: median pb_math_auc >= 0.58, pb_gsm_auc >= 0.52
   - failure criterion: inter-seed AUC range > 0.05 (recipe unstable)
2. `PBR3_LATER_BAD_BRANCH_SEED3`:
   - profile: `ms_laterbad_v1`, head: `mlp`, 3 seeds (42/1/7)
   - target pairs: 16384
   - success criterion: pb_math_auc improves vs PBR2, first_edge regression <= 0.05
   - failure criterion: first_edge < 0.45 (grid fraction too heavy)

Infrastructure changes (2026-03-11):
1. extended case spec to 6-field format: `case_id|mode|profile|head|balance|seed`
2. added `GROUP_TARGET_PAIRS` to allow per-group artifact scale override
3. `run_curate` and `run_train` now accept per-case seed argument

Research note:
- `docs/research_survey_processverifier_20260311.md` contains the full
  literature survey and experiment hypothesis chain from both internet rounds.

Architecture research findings (2026-03-11 second round):
1. **All PRMs achieving >75% ProcessBench fine-tune the backbone** (at minimum LoRA).
   Frozen backbone is a representation gap, not a capacity gap. No head change can fix it.
2. `dual_head` architecture is already implemented in `src/ours/phase_b/value_head.py`,
   but the training loop has no `pair_semantics` routing logic yet.
   Without routing, `dual_head` ≈ wider `gated_mlp` with same expected outcome.
3. Step-position metadata injection (append `step_frac + is_terminal` to pooled vector)
   is a low-cost experiment that can give the gate explicit structural context.
   Blocked on: `training.py` and `phase_e_train_value.py` not plumbing aux metadata yet.
4. mean_pool is NOT recommended for causal LMs (confirmed by all published PRM papers).
   Do not implement mean_pool ablation — it is a known worse design.
5. LoRA on last 4 Qwen2.5-7B layers = the highest-impact architecture change.
   Blocked on: feature cache invalidation and on-the-fly encoding fallback.
   Priority: after PBR2/PBR3 ceiling is confirmed.
Full analysis: `docs/research_survey_processverifier_20260311.md` section 9.

Execution order (after this design):
1. run `PBR2_FULL_MIXED_MLP_SEED3` first to establish full-scale baseline;
2. if PBR2 is stable (seed variance < 0.05), run `PBR3_LATER_BAD_BRANCH_SEED3`;
3. if PBR3 first_edge holds (>= 0.45), it becomes the new mainline profile;
4. if frozen-backbone ProcessBench AUC ceiling is confirmed below 0.65 after PBR3,
   enter the LoRA backbone path (new Phase E-LoRA suite, separate design);
5. `dual_head` with pair_semantics routing is a parallel medium-priority option
   if the data geometry is confirmed stable but local-vs-terminal objective conflict
   is confirmed as the remaining gap.

---

## 1. Phase E Objective

Validate whether the value head is **genuinely learnable** on high-quality pair
and process-supervision datasets.

Primary question:
1. On a single high-quality benchmark family, does the ranking branch become
   stable, strong, and reproducible?

Secondary question:
1. Once validated there, how much of that capability survives:
   - benchmark-native evaluation,
   - cross-benchmark evaluation,
   - transfer back to StrategyQA?

Gate-order rule:
1. Same-benchmark learnability is the first gate.
2. Benchmark-native/cross-benchmark evidence is the second gate.
3. StrategyQA transfer is the third gate.

---

## 2. Code Readiness Assessment

This section is the code-level inventory, not a wish list.

### 2.1 What is now implemented

Phase E is now **benchmark-native and runnable** for `E0-E3`.

Implemented modules:
1. Benchmark/source contract layer:
   - `src/ours/phase_e/contracts.py`
2. Pair artifact builder:
   - `src/ours/phase_e/pairs.py`
   - `scripts/phase_e_prepare_pairs.py`
3. Benchmark-native training runtime:
   - `src/ours/phase_e/runtime.py`
   - `src/ours/phase_e/training.py`
   - `scripts/phase_e_train_value.py`
4. Benchmark-native evaluation layer:
   - `src/ours/phase_e/benchmark_eval.py`
   - `scripts/phase_e_eval_benchmark.py`
5. One-click experiment suite:
   - `scripts/run_phase_e_suite.sh`
6. Multi-source math wrapper:
   - `scripts/run_phase_e_multisource_math_suite.sh`
7. Unit tests:
   - `tests/unit/test_phase_e_contracts.py`
   - `tests/unit/test_phase_e_pairs.py`
   - `tests/unit/test_phase_e_benchmark_eval.py`
   - `tests/unit/test_phase_e_train_script.py`
   - `tests/unit/test_feature_cache.py`

Benchmark/source support already wired:
1. Training/supervision bundles:
   - `Math-Shepherd`
   - `PRM800K`
   - `R-PRM`
   - `PRMBench_Preview`
   - mixed bundles:
     - `r_prm_prmbench_preview`
     - `math_shepherd_prm800k`
     - `math_shepherd_r_prm`
     - `math_shepherd_prmbench_preview`
     - `math_shepherd_r_prm_prmbench_preview`
     - `math_shepherd_r_prm_prmbench_preview_prm800k`
2. Benchmark-native eval:
   - `ProcessBench GSM8K`
   - `ProcessBench Math`
   - `PRMBench_Preview`

Smoke validation already completed:
1. Direct script chain:
   - pair prepare -> train -> ProcessBench eval
2. Full suite smoke:
   - `E1_MATH_SHEPHERD_PAIR_LEARN_SMOKE`
3. Validation commands passed:
   - `python -m py_compile ...`
   - `pytest` on all Phase E unit tests
   - `bash -n scripts/run_phase_e_suite.sh`
   - `bash -n scripts/run_phase_e_multisource_math_suite.sh`

Conclusion:
1. We do **not** need to invent a new training stack for Phase E.
2. `E0-E3` have moved from plan to implementation.
3. Runtime trust improved on `2026-03-11`:
   - feature-cache provenance is stricter,
   - cache corruption is self-healing,
   - head-only training no longer assumes whole cache tensors stay on GPU.

### 2.2 What is still not ready yet

Key remaining gaps are no longer in the E0-E3 backbone.

Current limitations:
1. Promotion gate is not yet rewritten around Phase E benchmark metrics.
2. StrategyQA transfer remains a separate downstream step, not yet integrated
   into a unified Phase E -> transfer promotion flow.
3. Full benchmark result matrices have not yet been populated; only smoke
   validation has completed in the new stack.
4. `ProcessBench / PRMBench` are wired as benchmark evaluators, but the
   repository still needs official threshold decisions for promotion.
5. Current interpretation rules still need to distinguish:
   - source-family learnability,
   - benchmark-native evaluation,
   - downstream transfer,
   instead of collapsing them into one pass/fail score.

Conclusion:
1. Phase E core infra is ready.
2. The next work is experimental validation and gate-setting, not more basic
   framework decoupling.
3. The first official multi-source math experiment families are now wired:
   - direct Stage A-D groups in `scripts/run_phase_e_suite.sh`
   - staged curriculum Stage E in `scripts/run_phase_e_multisource_math_suite.sh`

### 2.3 Risk assessment

If we continue using the current stack without that decoupling:
1. We may accidentally keep using StrategyQA artifacts as hidden scaffolding.
2. We may over-interpret transfer improvements as primary-benchmark success.
3. We may fail to distinguish:
   - source-benchmark learnability
   - target-benchmark transfer

So the first engineering goal of Phase E is **not** more tuning.
It is **benchmark separation**.

---

## 3. Phase E Scope

### In scope

1. Benchmark-native learnability validation on high-quality pair/process datasets.
2. Source-separated training/evaluation reports.
3. Promotion gates defined in layers:
   - source-family held-out first,
   - benchmark-native eval second,
   - transfer third.
4. StrategyQA only as transfer / bridge / OOD check.

### Out of scope

1. Router RL.
2. Joint LM/value/router optimization.
3. Returning to “StrategyQA first” as the main supervised story.
4. Treating better Brier as sufficient success evidence.
5. Treating cross-dataset transfer as the first pass/fail gate.

---

## 4. Selected Phase E Benchmarks

### 4.1 Main training / supervision sources

These are the immediate Phase E sources because the repository already has
either code support or local copies.

1. `Math-Shepherd`
   - current best external ranking signal in our experiments.
   - local path:
     - `assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl`
2. `PRM800K`
   - still important because it is the most canonical public PRM dataset.
   - local path:
     - `assets/external_datasets/openai_prm800k`
3. `R-PRM`
   - direct chosen/rejected pairs.
   - local path:
     - `assets/external_datasets/kevinpro_r_prm`
4. `PRMBench_Preview`
   - useful as high-quality direct/converted process-pair data.
   - local path:
     - `assets/external_datasets/hitsmy_prmbench_preview/prmbench_preview.jsonl`

### 4.2 Main evaluation benchmarks

These should become the **secondary Phase E gates**, after same-family
learnability is established.

1. `ProcessBench`
   - local path:
     - `assets/external_datasets/qwen_processbench`
   - current status:
     - dataset is present locally,
     - repo does not yet have benchmark-native evaluator wiring.
2. `PRMBench`
   - current repo status:
     - `PRMBench_Preview` pair data is available,
     - full benchmark-native evaluation wiring is not yet implemented.

### 4.3 Transfer / downstream target

1. `StrategyQA`
   - keep as:
     - bridge continue-training target,
     - downstream transfer benchmark,
     - OOD stress test.

---

## 5. Phase E Work Packages

## E0. Benchmark Contract Freeze

Status:
1. Implemented.

What shipped:
1. Canonical benchmark/source registry:
   - `src/ours/phase_e/contracts.py`
2. New script/module naming contract:
   - `phase_e_*`
3. Docs updated so that:
   - StrategyQA is transfer-only,
   - Phase E is the active mainline for learnability validation.

Pass condition:
1. Satisfied.

## E1. Trainer Decoupling

Status:
1. Implemented.

What shipped:
1. New benchmark-native trainer:
   - `scripts/phase_e_train_value.py`
2. New shared runtime layer:
   - `src/ours/phase_e/runtime.py`
   - `src/ours/phase_e/training.py`
3. Pure external-pair training now starts from only:
   - model path
   - train pair artifact
   - eval pair artifact
   - no `Phase C` dir required

Pass condition:
1. Satisfied by direct smoke and suite smoke.

## E2. Benchmark-Native Learnability Suite

Status:
1. Implemented.

Shipped groups:
1. `E1_MATH_SHEPHERD_PAIR_LEARN_SMOKE`
2. `E2_MATH_SHEPHERD_PAIR_LEARN_SEED3`
3. `E3_RPRM_PRMBENCH_PREVIEW_SMOKE`
4. `E4_RPRM_PRMBENCH_PREVIEW_SEED3`
5. `E5_PRM800K_PAIR_LEARN_SEED3`
6. `E6_MATH_SHEPHERD_SAME_SOURCE_SMOKE`
7. `E7_MATH_SHEPHERD_SAME_SOURCE_SEED3`
8. `E8_PRMBENCH_PREVIEW_SAME_SOURCE_SMOKE`
9. `E9_PRMBENCH_PREVIEW_SAME_SOURCE_SEED3`
10. `E12_MATH_SHEPHERD_TRUST_LOWLR_SEED3`
11. `E13_MATH_SHEPHERD_TRUST_UNWEIGHTED_SEED3`
12. `E14_MATH_SHEPHERD_TRUST_ANTISAT_SEED3`
13. `E15_MATH_SHEPHERD_TRUST_ROBUST_SEED3`
14. wrapper bundles:
   - `scripts/run_phase_e_single_source_suite.sh`
   - `scripts/run_phase_e_mathshepherd_trust_suite.sh`
15. checkpoint promotion helper:
   - `scripts/phase_e_select_candidate.py`

Measured outputs:
1. held-out pair accuracy
2. held-out pair AUC
3. train/eval pair loss and ranking score
4. seed mean/std in suite final summary

Pass condition:
1. Implementation complete.
2. Scientific pass/fail should first be judged on:
   - held-out pair quality inside the same source family,
   - seed stability inside that same source family.

## E2.1 Math-Shepherd Trust Track

Why this sub-track was added:
1. `Math-Shepherd` is currently the only source family that has already shown
   clearly positive same-source held-out learnability in our own runs.
2. Later RL-heavy stages will need one concrete `best_value_head.pt`, not just
   a loose statement that "the source is somewhat learnable".
3. Therefore we now need a dedicated candidate-search matrix inside the same
   source family.

New trust-matrix recipes:
1. Baseline benchmark-aware control:
   - `E2_MATH_SHEPHERD_PAIR_LEARN_SEED3`
2. Same-source-only control:
   - `E7_MATH_SHEPHERD_SAME_SOURCE_SEED3`
3. Lower-LR stability probe:
   - `E12_MATH_SHEPHERD_TRUST_LOWLR_SEED3`
4. Confidence-weight ablation:
   - `E13_MATH_SHEPHERD_TRUST_UNWEIGHTED_SEED3`
5. Logit anti-saturation probe:
   - `E14_MATH_SHEPHERD_TRUST_ANTISAT_SEED3`
6. Conservative robust candidate recipe:
   - `E15_MATH_SHEPHERD_TRUST_ROBUST_SEED3`

What each recipe is trying to answer:
1. `E2`:
   - current best baseline with benchmark-native pressure.
2. `E7`:
   - does the source family itself remain learnable without any extra benchmark
     confound?
3. `E12`:
   - is seed collapse mainly an optimization aggressiveness problem?
4. `E13`:
   - are our heuristic confidence weights themselves introducing instability?
5. `E14`:
   - is saturation damaging ranking geometry and benchmark behavior?
6. `E15`:
   - what is the most conservative recipe we would currently trust for later
     RL-facing promotion?

Trust-candidate policy:
1. Do not promote by single-run peak score.
2. Use:
   - source-family held-out mean,
   - worst-seed held-out behavior,
   - held-out seed std,
   - `ProcessBench GSM8K/Math` secondary metrics,
   - then select one explicit checkpoint file.
3. Promotion is scripted by:
   - `scripts/phase_e_select_candidate.py`

## E3. Benchmark-Native Evaluation Layer

Status:
1. Implemented.

What shipped:
1. New evaluator:
   - `scripts/phase_e_eval_benchmark.py`
2. New benchmark layer:
   - `src/ours/phase_e/benchmark_eval.py`
3. Current benchmark-native reports:
   - `ProcessBench`:
     - good-vs-bad pair accuracy
     - good-vs-bad pair AUC
     - first-error edge accuracy
   - `PRMBench_Preview`:
     - pair accuracy
     - pair AUC

Pass condition:
1. Satisfied in smoke runs.
2. Benchmark-native scores are interpreted as secondary evidence of external
   validity, not as the first proof that learnability exists.

## E4. Cross-Benchmark Evaluation Layer

Goal:
1. Measure how much a source-trained value head survives on another benchmark.

Execution rule:
1. Do not require strong cross-benchmark transfer as the initial proof that the
   method works.
2. Use this layer to diagnose:
   - source specificity,
   - benchmark mismatch,
   - whether a stronger cross-domain method is needed.

Pass condition:
1. Cross-benchmark metrics are reported explicitly as secondary evidence, never
   as the primary proof of learnability.

## E5. StrategyQA Transfer Layer

Goal:
1. Only after E2/E3 pass, measure transfer back to StrategyQA.

Execution rule:
1. Use StrategyQA only after the source benchmark has already shown value-head
   learnability.
2. Use bridge-style continue training only as a secondary/downstream check.

Pass condition:
1. StrategyQA transfer metrics are reported explicitly as transfer, never as the
   primary proof of learnability.

## E6. Promotion Gate

A Phase E method is promotable only if:
1. It passes source-family learnability gates first.
2. It passes benchmark-native evaluation second.
3. StrategyQA transfer is then used as bonus evidence, not prerequisite proof.

---

## 6. Immediate Code Judgment

This is the direct answer to "is the repo ready?".

Yes, but only partially.

More precise judgment:
1. The repository is now ready for **Phase E execution**:
   - benchmark-native pair prep
   - benchmark-native training
   - benchmark-native eval
   - suite orchestration
   are all present.
2. The repository is not yet done with **Phase E science**:
   - full source-family comparisons still need to be run,
   - promotion thresholds still need to be frozen,
   - StrategyQA transfer should still wait until source-family success is
     reproduced.

Therefore the right next move is:
1. stop further base-framework refactoring,
2. run the official Phase E suites,
3. compare source families on held-out pair metrics first,
4. inspect benchmark-native metrics as second-layer evidence,
5. only then resume transfer experiments.

---

## 7. Immediate Next Actions

Ordered strictly:

1. Run:
   - `MS2_MATH_SHEPHERD_TRUST_SEED3_MATRIX`
   - `E4_RPRM_PRMBENCH_PREVIEW_SEED3`
   - `E5_PRM800K_PAIR_LEARN_SEED3`
2. Compare:
   - held-out pair learnability
   - seed stability
   - benchmark-native eval as secondary evidence
3. For `Math-Shepherd`, freeze one provisional checkpoint candidate using:
   - `scripts/phase_e_select_candidate.py`
   - trust summary from `scripts/run_phase_e_mathshepherd_trust_suite.sh`
4. Freeze first Phase E promotion thresholds with the new gate order:
   - source-family first
   - benchmark-native second
   - transfer third
5. Only after that, reopen StrategyQA transfer as downstream validation.

## 7.1 Current Empirical Evidence (Why The Gate Order Changed)

Our own newest Phase E results already support this ordering.

1. `Math-Shepherd smoke`:
   - same-source held-out is clearly positive:
     - `pair_acc=0.6787`
     - `auc=0.6704`
   - but `ProcessBench GSM8K` remains weak:
     - `pair_acc=0.4661`
     - `auc=0.5155`
2. `Math-Shepherd seed-3`:
   - two seeds are very strong on same-source held-out:
     - `s42: pair_acc=0.8480, auc=0.8218`
     - `s43: pair_acc=0.8511, auc=0.8394`
   - one seed collapses:
     - `s44: pair_acc=0.3495, auc=0.4449`
   - interpretation:
     - learnability exists,
     - but the recipe is not yet trustworthy enough to freeze as an RL-facing
       checkpoint family without a stronger trust-selection layer.
3. `PRM800K seed-3`:
   - same-source held-out mean is weak:
     - `mean_pair_acc=0.4783`
     - `mean_auc=0.4855`
   - `ProcessBench GSM8K` mean is also weak:
     - `mean_pair_acc=0.4737`
     - `mean_auc=0.4943`
4. `E15_MATH_SHEPHERD_TRUST_ROBUST_SEED3`:
   - same-source held-out is now clearly positive and reasonably stable:
     - `mean_heldout_pair_acc=0.7518`
     - `mean_heldout_auc=0.7280`
   - but benchmark-native metrics are still weak:
     - `ProcessBench GSM8K mean_auc=0.4834`
     - `ProcessBench Math mean_auc=0.4746`
5. Therefore source-family learnability is already a stronger discriminator than
   cross-benchmark transfer under the current method.

### 7.1A Post-Fix Update (2026-03-10 Late)

After the `first_bad_edge_strict` repair and the first Stage A/B multi-source
runs, the evidence is now more specific.

1. Post-fix `Math-Shepherd` trust matrix:
   - `MS2_MATH_SHEPHERD_TRUST_SEED3_MATRIX`
   - provisional best family:
     - `E14_MATH_SHEPHERD_TRUST_ANTISAT_SEED3`
   - summary:
     - `mean_hold_pair=0.8246`
     - `mean_hold_auc=0.7899`
     - `pb_gsm_auc=0.4893`
     - `pb_math_auc=0.4783`
   - reading:
     - the repair improved same-family learnability,
     - but still did not unlock trustworthy `ProcessBench` behavior.
2. Stage A single-source anchors now answer distinct questions:
   - `E20_STAGEA_MS_ANCHOR_SEED3`
     - `mean_heldout_auc=0.6676`
     - `mean_prmbench_preview_auc=0.5868`
     - `mean_processbench_gsm8k_auc=0.4750`
     - `mean_processbench_math_auc=0.4715`
     - reading:
       - strongest post-fix same-family anchor,
       - still weak on `ProcessBench`.
   - `E21_STAGEA_RPRM_ANCHOR_SEED3`
     - `mean_heldout_auc=0.4953`
     - `mean_prmbench_preview_auc=0.5623`
     - `mean_processbench_gsm8k_auc=0.4744`
     - `mean_processbench_math_auc=0.4626`
     - reading:
       - weaker as a general anchor,
       - but materially aligned with `PRMBench_Preview`.
3. `E22_STAGEA_PRMBENCH_PREVIEW_ANCHOR_SEED3`
   - currently has pair artifact + training start,
   - but no complete final group summary yet,
   - so it remains a pending question, not evidence.
4. `E24_STAGEB_MS_RPRM_MIX_SEED3`
   - current smoke evidence is negative / incomplete, not promotable:
     - `PRMBench_Preview auc=0.5207`
     - `ProcessBench GSM8K auc=0.4682`
     - `ProcessBench Math auc=0.3835`
   - reading:
     - simple balanced mixture has not yet shown complementarity.
5. `CUR1_STAGEE_MS_TO_MSRPRM`
   - stage-1 warm-start exists,
   - but stage-2 is still incomplete,
   - so curriculum should not yet be narrated as a successful stabilization
     result.

Immediate interpretation:
1. The `Math-Shepherd` semantics repair was necessary and real.
2. The current repository now has:
   - one better same-family anchor (`Math-Shepherd`)
   - one better preview-aligned auxiliary source (`R-PRM`)
3. What is still missing is a demonstrated recipe that combines these without
   losing benchmark behavior.

## 7.2 Literature-Supported Decision For The Next Stage

The literature supports three claims simultaneously.

Claim A: same-source learnability is real.
1. `PRM800K`, `Math-Shepherd`, `OmegaPRM`, `SVPO`, and `R-PRM` all support the
   idea that a value/reward-like head can learn a meaningful signal from
   high-quality step labels or good/bad process pairs.

Claim B: cross-domain generalization is genuinely difficult.
1. `ProcessBench` and `PRMBench` were created precisely because current PRMs
   still fail on harder and more fine-grained verification.
2. `VersaPRM` exists because single-domain math PRMs do not generalize well by
   default.
3. `ThinkPRM` shows that fixing this may require a qualitatively stronger
   verifier class, not just intuitive retuning of the current scalar head.

Claim C: RL amplifies reward-model mistakes.
1. `InstructGPT` already treats the reward model as a critical component and
   uses KL regularization during policy optimization.
2. `Dense Reward for Free in RLHF` supports using denser shaping signals because
   sparse terminal reward is hard to optimize.
3. `Scaling Laws for Reward Model Overoptimization` and `Reward Model Ensembles
   Help Mitigate Overoptimization` both warn that optimizing a proxy reward too
   aggressively can worsen true quality.

Project-level decision:
1. We explicitly do **not** require cross-domain generalization for the next
   stage.
2. We explicitly do **not** claim that an intuitive `Math-Shepherd`-trained
   value head will become a robust cross-benchmark verifier.
3. If Phase E succeeds, the next stage should instead ask:
   - how do we use one trusted `Math-Shepherd` value head as a conservative,
     same-family RL guidance signal without letting RL exploit its blind spots?

## 7.2A Mixed Math Strategy Update

The new question is no longer:
1. "Can one source train a universally transferable verifier?"

The new question is:
1. "Can same-family **multi-source math** supervision convert source-family
   learnability into benchmark-facing robustness?"

Official decision:
1. The next Phase E mainline should move from single-source trust selection to
   same-family multi-source math mixture training.
2. The first-wave core sources should be:
   - `Math-Shepherd`
   - `R-PRM`
   - `PRMBench_Preview`
3. `PRM800K` should be retained only as:
   - weak-source ablation,
   - or low-weight auxiliary source,
   not as a dominant first-wave source.
4. `ProcessBench` remains eval-only.
5. `PRMBench` official benchmark remains eval-only; only
   `PRMBench_Preview` stays on the training side.

Primary reason:
1. Single-source `Math-Shepherd` already proves learnability.
2. It does **not** yet prove trustworthy benchmark-facing verification.
3. Therefore the next valid method question is coverage and source quality, not
   more isolated one-source tuning.

Detailed design:
1. `docs/phase_E_multisource_math_plan.md`
2. Direct suite entrypoint:
   - `scripts/run_phase_e_multisource_math_suite.sh`

## 7.3 Recommended RL Use Of A Successful Math-Shepherd Value Head

Assumption for this section:
1. `Math-Shepherd` single-source training passes the trust matrix.
2. The next stage still stays in the same math/process family.
3. Cross-dataset transfer is temporarily out of scope.

Under that assumption, the value head should be used as a **frozen process
reward model**, not as an all-purpose online critic.

### 7.3.1 What to optimize first

Recommended first RL target:
1. Freeze the base LM used for Phase E.
2. Freeze the selected `best_value_head.pt`.
3. Optimize only a small downstream policy/controller first:
   - router,
   - verifier-calling policy,
   - branch/retry policy,
   - or candidate-selection policy.
4. Do **not** start with full joint LM + value + router online RL.

Reason:
1. This keeps the policy close to the support where the value head was trained.
2. It reduces credit-assignment noise.
3. It prevents immediate reward hacking through large-distribution drift.

### 7.3.2 How reward should be formed

Recommended reward design:
1. Keep final task success in the loop:
   - final correctness or verifier-approved terminal success should remain the
     anchor reward.
2. Use the value head only as dense shaping or ranking guidance:
   - step delta reward,
   - prefix improvement bonus,
   - candidate ranking bonus,
   - early-stop / continue / backtrack guidance.
3. Clip and normalize the shaping component.
4. Add an explicit compute cost term if the action space includes rethinking,
   branching, or verification calls.

Practical rule:
1. Do not optimize raw `V(prefix)` directly as the sole reward.
2. Prefer conservative forms such as:
   - clipped `V(prefix_t) - V(prefix_{t-1})`,
   - pairwise preference bonus between sampled branches,
   - or a mixed reward:
     - terminal correctness
     - plus bounded dense shaping
     - minus compute penalty

Reason:
1. The literature on dense reward says shaping helps optimization.
2. The literature on reward overoptimization says raw proxy maximization is
   dangerous.

### 7.3.3 What policy class is safest

Safest early policy class:
1. low-dimensional decision policy over a fixed generator,
2. short-horizon controller,
3. candidate reranker / selector,
4. router-only RL before any full-token policy RL.

Unsafe early policy class:
1. unrestricted token-level PPO on the whole LM,
2. joint updating of LM and reward head,
3. long-horizon search with no KL or support control.

Reason:
1. A narrow policy class constrains the ways RL can exploit reward-model
   artifacts.
2. It also makes failure attribution tractable:
   - controller problem,
   - reward problem,
   - or generator problem.

### 7.3.4 How to guard against reward hacking

Minimum safeguards:
1. Keep a strong KL penalty or equivalent reference-policy constraint.
2. Maintain one held-out `Math-Shepherd` validation split that is never used for
   RL updates.
3. Re-run `ProcessBench` and same-source held-out evaluation as canary checks,
   even if they are not the main optimization target.
4. Use more than one trusted value-head checkpoint when feasible:
   - ensemble average,
   - worst-case score,
   - or uncertainty penalty from trust-matrix candidates.
5. Stop training if policy reward rises while:
   - held-out pair accuracy falls,
   - benchmark metrics fall,
   - or generated trajectories become obviously longer/weirder without better
     final correctness.

Why this is necessary:
1. Reward-model overoptimization is a standard failure mode.
2. Distribution shift between training prefixes and RL-generated prefixes is the
   easiest path to reward hacking.

### 7.3.5 What should remain frozen

For the first RL stage, freeze:
1. the reward/value head,
2. the reward calibration constants,
3. the reference policy,
4. the held-out evaluation protocol.

Do not:
1. continue online training of the reward model while also optimizing the
   policy,
2. change the reward normalization every few runs,
3. select checkpoints by ad-hoc manual reading of logs.

Reason:
1. If reward semantics drift during RL, we lose the ability to interpret gains.
2. Phase E exists precisely to freeze one trustworthy checkpoint family before
   downstream optimization.

### 7.3.6 Recommended order of downstream experiments

Strict recommended order:
1. `Offline scoring sanity`
   - verify the frozen value head improves same-source candidate ranking on
     held-out trajectories.
2. `ABR-lite / deterministic controller`
   - convert the value head into a non-RL decision rule first.
3. `Router-only RL`
   - optimize only the controller/router under a frozen generator and frozen
     reward model.
4. `Limited generator adaptation`
   - only if router-only RL is stable and genuinely improves final correctness.

This order matters because:
1. if a deterministic controller cannot use the signal, RL will not fix it
   cleanly,
2. if router-only RL fails, full-policy RL will only add confounding factors.

### 7.3.7 Pitfalls we should explicitly avoid

Do not do the following in the immediate next stage:
1. claim cross-domain robustness from `Math-Shepherd` alone,
2. use the Phase E head as the only reward with no terminal anchor,
3. jointly fine-tune LM + value head + router with online RL from the start,
4. remove KL/support regularization because same-source metrics look strong,
5. promote a checkpoint by one lucky seed,
6. use the same reward head as both optimizer and sole final evaluator,
7. interpret higher average reward as success if external canaries degrade.

Immediate success criterion for the next stage:
1. better final math correctness inside the same family,
2. no collapse on held-out Math-Shepherd ranking metrics,
3. no major degradation on benchmark canaries,
4. gains reproduced across more than one trust-matrix candidate or seed.

---

## 8. References / Local Anchors

Current key code files:
1. `src/ours/phase_e/contracts.py`
2. `src/ours/phase_e/pairs.py`
3. `src/ours/phase_e/runtime.py`
4. `src/ours/phase_e/training.py`
5. `src/ours/phase_e/benchmark_eval.py`
6. `scripts/phase_e_prepare_pairs.py`
7. `scripts/phase_e_train_value.py`
8. `scripts/phase_e_eval_benchmark.py`
9. `scripts/run_phase_e_suite.sh`

Current key local datasets:
1. `assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl`
2. `assets/external_datasets/openai_prm800k`
3. `assets/external_datasets/kevinpro_r_prm`
4. `assets/external_datasets/hitsmy_prmbench_preview/prmbench_preview.jsonl`
5. `assets/external_datasets/qwen_processbench`

Community references that justify the revised gate order:
1. Let's Verify Step by Step:
   - https://arxiv.org/abs/2305.20050
2. Math-Shepherd:
   - https://arxiv.org/abs/2312.08935
3. Improve Mathematical Reasoning in Language Models by Automated Process Supervision (`OmegaPRM`):
   - https://arxiv.org/abs/2406.06592
4. Step-level Value Preference Optimization (`SVPO`):
   - https://arxiv.org/abs/2406.10858
5. ProcessBench paper:
   - https://arxiv.org/abs/2412.06559
6. ProcessBench official repo:
   - https://github.com/QwenLM/ProcessBench
7. PRMBench paper:
   - https://arxiv.org/abs/2501.03124
8. PRMBench project page:
   - https://prmbench.github.io/
9. PRMBench official repo:
   - https://github.com/ssmisya/PRMBench
10. The Lessons of Developing Process Reward Models in Mathematical Reasoning:
    - https://arxiv.org/abs/2501.07301
11. Do We Need to Verify Step by Step? Rethinking Process Supervision from a Theoretical Perspective:
    - https://arxiv.org/abs/2502.10581
12. VersaPRM paper:
   - https://arxiv.org/abs/2502.06737
13. R-PRM paper:
    - https://arxiv.org/abs/2503.21295
14. R-PRM official repo:
    - https://github.com/NJUNLP/R-PRM
15. ThinkPRM paper:
   - https://arxiv.org/abs/2504.16828
16. ThinkPRM official repo:
   - https://github.com/mukhal/thinkprm
17. Training language models to follow instructions with human feedback (`InstructGPT`):
    - https://arxiv.org/abs/2203.02155
18. Scaling Laws for Reward Model Overoptimization:
    - https://arxiv.org/abs/2210.10760
19. Reward Model Ensembles Help Mitigate Overoptimization:
    - https://arxiv.org/abs/2310.02743
20. Dense Reward for Free in Reinforcement Learning from Human Feedback:
    - https://arxiv.org/abs/2402.00782
21. LMs Mostly Know What They Know:
   - https://arxiv.org/abs/2207.05221

---

## 9. Intradataset ACC90 Branch

Phase E now has one additional narrow branch focused only on same-source learnability.

Purpose:
1. ignore cross-dataset transfer,
2. ignore benchmark-native generalization,
3. train one value head per high-quality dataset,
4. test whether the dataset's own held-out pair accuracy can exceed `0.90`.

Plan and commands:
- `docs/phase_E_intradataset_acc90_plan.md`

Main wrapper:
- `scripts/run_phase_e_intradataset_suite.sh`

This branch exists because recent evidence showed:
1. same-source learnability can be positive,
2. but cross-dataset generalization remains weak,
3. so the first-principles question should be separated from the transfer question.
