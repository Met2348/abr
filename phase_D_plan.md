# Phase D Plan: External-PRM-Supported Value Learning

This file defines the official Phase D execution plan.

Date baseline: 2026-03-03.

## 0. Document Governance (Source of Truth)

Primary execution document:
- `phase_D_plan.md` (this file)

Supporting diagnosis/technical references:
1. `phase_C_fix_value_head.md` (deep diagnosis, external-aid rationale, deployment caveats)
2. `phase_B_report.md` (empirical historical baseline and cross-task interference evidence)
3. `TODO_ours.md` (task-level checklist and milestone tracking)

Governance rule:
1. Use this file for scope, order, and promotion gates.
2. Use supporting files for evidence and implementation detail.
3. If conflict exists, this file wins for execution sequencing.

## 1. Why Phase D Starts Now

Phase C proved the full local pipeline is operational, but not yet decision-useful.

Phase C conclusions from completed runs:
1. C1/C2 engineering is stable and reproducible (artifacts, manifests, eval schemas, suite wrappers).
2. Calibration improved from early failures but still did not consistently beat the trivial calibration baseline.
3. Corruption discrimination stayed near random in most runs.
4. Objective-only trick sweeps (loss mix, post-hoc calibration, adaptive weighting, label smoothing, hard-negative mining) changed metrics but did not cross promotion gates.
5. Therefore the bottleneck is now supervision quality, not runtime reliability.

This is the precise trigger for Phase D: introduce high-quality external PRM signals as a controlled teacher, without replacing our method.

## 2. Phase C Closeout (What Is Done, What Remains)

### 2.1 Completed in Phase C

1. Prefix-level data contract (`step_sequences`, `prefixes`, `corruptions`, `rollout_targets`) is implemented and stable.
2. Value-head training/evaluation stack is implemented with diagnostics:
   - calibration losses (`mse`, `bce`, `bce_mse`),
   - post-hoc calibration (`temperature`, `isotonic`),
   - contrastive filtering/mining,
   - uncertainty-aware weighting.
3. Quality-first C1 upgrades are implemented:
   - uncertainty-aware Q fields (`q_mean_smoothed`, `q_std_error`, `q_ci_width`, `q_weight`),
   - pair-quality artifacts (`pair_quality.jsonl`),
   - primary-corruption selection by quality,
   - two-stage uncertain-prefix rollout enrichment.
4. P(IK) branch is implemented as a simplified learnability gate.

### 2.2 Remaining Phase C Remnants (Unresolved)

1. Weak supervision remains the dominant issue.
   - Monte Carlo labels are still noisy for fine-grained prefix ranking.
2. Calibration-vs-ranking tension remains unresolved.
   - Better Brier does not imply better corruption ordering.
3. Pair signal is often too weak.
   - Many clean/corrupt pairs are low-margin and hard to separate.
4. No external teacher signal is yet integrated in the training data path.
5. P(IK) smoke did not show robust separability, so prefix-level learning is unlikely to improve without stronger labels.

Interpretation: Phase C is not a dead end; it is a successful infrastructure phase with negative-but-informative model results.

## 3. Phase D Objective

Use external PRM teachers to improve value supervision quality while preserving the existing in-house C1/C2 pipeline and ABR research identity.

Core objective:
- improve label quality first,
- then re-evaluate whether value head becomes routing-usable.

## 4. Phase D Scope and Non-Goals

### In scope

1. External PRM teacher scoring on existing C1 artifacts.
2. Teacher + Monte Carlo label fusion and disagreement logging.
3. C2 target-source switching (`mc`, `teacher`, `fused`).
4. Controlled ablations and promotion gates.

### Out of scope

1. Replacing our method with third-party PRM inference-only system.
2. Router RL expansion before value-head promotion gate passes.
3. Multi-teacher ensembling as first implementation.

## 5. Selected External Teacher Baseline

Primary teacher:
- `assets/models/Qwen2.5-Math-PRM-7B`

Control teacher (optional ablation):
- `assets/models/Qwen2.5-Math-7B-PRM800K`

Backbone generator remains:
- `assets/models/Qwen2.5-7B-Instruct`

## 6. Workstreams

### D0: Environment and teacher readiness gate

Deliverables:
1. deterministic local smoke script for teacher scoring,
2. environment compatibility record,
3. model availability checks in manifests.

Gate:
- teacher model can score step sequences reproducibly on GPU.

### D1: Teacher scoring sidecar (new)

Planned new script:
- `scripts/phase_c_score_prm_teacher.py`

Inputs:
1. `prefixes.jsonl`
2. `corruptions.jsonl` (optional)
3. `--teacher-model-path`

Outputs:
1. `teacher_prefix_scores.jsonl`
2. `teacher_corruption_scores.jsonl`

Required fields per record:
1. `prefix_id` or `corruption_id`
2. `teacher_score_mean`
3. `teacher_score_min`
4. `teacher_num_steps`
5. `teacher_model_id`

### D2: Label fusion in C1 (new)

Extend C1 build path to ingest teacher scores and emit:
1. `q_teacher`
2. `q_fused`
3. `teacher_available`
4. `teacher_disagree`

Fusion baseline:
- $q_{fused} = \lambda q_{mc} + (1-\lambda) q_{teacher}$

Defaults:
1. fixed $\lambda=0.5$ baseline,
2. confidence-weighted $\lambda$ variant using `q_ci_width`.

### D3: C2 target-source switch (new)

Add C2 option:
- `--target-source {q_mean_smoothed,q_teacher,q_fused}`

Promotion-oriented runs must log:
1. target source,
2. teacher coverage,
3. teacher disagreement ratio.

### D4: Controlled ablation matrix

Required first matrix (same split/protocol):
1. MC-only target
2. Teacher-only target
3. Fixed fusion target
4. Confidence-weighted fusion target

For each run report:
1. raw/posthoc Brier
2. raw/posthoc Pearson
3. `corr_pair_acc`
4. `corr_auc`
5. calibration baseline delta

### D5: Promotion gate to ABR/BCR continuation

Only proceed to router/BCR-lite expansion if both pass:
1. calibration gate:
   - selected Brier beats trivial baseline with stable repeats,
2. ranking gate:
   - corruption ordering metrics clearly above random and reproducible.

If gate fails:
1. upgrade pair construction (sibling-style high-margin pairs),
2. increase selective rollout budget,
3. try control teacher variant before architectural expansion.

## 7. Experiment Protocol (Phase D)

### Fixed protocol

1. Keep dataset and split protocol fixed per group.
2. Keep evaluation extractor/settings fixed.
3. Compare only one axis change at a time where possible.

### Required diagnostics per run

1. label histograms (`q_mc`, `q_teacher`, `q_fused`),
2. disagreement histogram,
3. pair margin distribution,
4. coverage stats by corruption type and step index,
5. teacher-serialization sanity report:
   - sampled raw text,
   - `<extra_0>` count,
   - parsed `teacher_num_steps`,
   - score vector length consistency,
6. id-join integrity report:
   - raw record count,
   - scored record count,
   - joined record count,
   - dropped-id count and examples,
7. scale-alignment report:
   - `q_mc/q_teacher/q_fused` mean/std/quantiles,
   - whether a monotonic mapping is used and fitted on which split,
8. effective-pair report:
   - retained pair count,
   - retained pair ratio versus pre-filter pairs,
   - retained pair ratio versus MC-only baseline on same split/budget,
9. corruption-mutation quality report:
   - corruption type mix ratio,
   - index-reference mutation profile (for example `#k` reference changes),
   - degenerate/self-contradictory mutation rate (for example `#2` vs `#2` style artifacts),
   - top repeated mutation templates by frequency.

## 8. Risks and Controls

### 8.1 Hard-fail validity checks (must pass)

1. Teacher input-serialization contract:
   - scoring must use the exact chat-template/step-separator contract (`<extra_0>`),
   - `teacher_num_steps` must match parsed step count.
   - Hard fail if step-count mismatch rate exceeds `1%`.
2. Teacher scoring numerical contract:
   - no NaN/Inf in teacher scores,
   - score vector length must match step markers.
   - Hard fail on any NaN/Inf or systematic length mismatch.
3. Cache compatibility contract:
   - enforce `use_cache=False` for teacher forward scoring in this repo env.
   - Hard fail if cache-mode runtime mismatch reappears.
4. ID-join contract:
   - `prefix_id/corruption_id` must be unique and joinable,
   - no silent dropping.
   - Hard fail if `teacher_available_ratio < 0.98` or duplicate IDs exist.
5. Fusion scale contract:
   - `q_teacher` must be mapped into a comparable `[0,1]` range before fusion,
   - any mapping/calibration must be fitted on train split only.
   - Hard fail on eval-split leakage.
6. Pair-signal sufficiency contract:
   - pair filtering cannot collapse training signal.
   - Hard fail if retained-pair ratio is `< 0.30` on full runs.
7. Fair-ablation contract:
   - `mc/teacher/fused_fixed/fused_confidence` must use equal split, equal seed policy,
     and equal decode/eval settings for promotable conclusions.
   - Runs with changed budget/settings are exploratory only and cannot decide promotion.
8. Corruption-design validity contract:
   - corruption artifacts must not collapse into one dominant mutation pattern.
   - Hard fail for promotion decisions if:
     - one corruption type dominates more than `70%`, or
     - effective active corruption types `< 3` (types with at least `5%` share), or
     - degenerate/self-contradictory mutation rate exceeds `5%`.
   - Violating runs are allowed for debugging, but cannot be used as promotion evidence.

### 8.2 Core research risks (tracked after validity passes)

Risk 1: teacher domain mismatch (math PRM vs StrategyQA style).
- Control: run teacher-only ablation and disagreement audit.

Risk 2: teacher over-dominates and hides our signal.
- Control: always keep MC-only and fusion ablations.

Risk 3: silent contract drift in JSONL artifacts.
- Control: explicit schema checks and manifest signature checks.

Risk 4: metric overfitting to one eval set.
- Control: require repeatability and split-level reporting.

Risk 5: corruption scheme bias (surface pattern over-learning).
- Control: enforce corruption-type diversity and mutation-quality diagnostics before
  interpreting ranking gains.

### 8.3 Corruption diagnostic snapshot (2026-03-03, StrategyQA eval artifact)

Observed from:
- `assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_c2_strategyqa_quality_first_c1_eval__f608255f810d/corruptions.jsonl`

Snapshot:
1. total corruptions: `255`
2. type distribution:
   - `numeric_perturb`: `178` (~`69.8%`)
   - `step_drop`: `76` (~`29.8%`)
   - `binary_flip`: `1` (~`0.4%`)
3. many mutations are index-reference shifts (for example `#1 -> #2`),
   including degenerate patterns such as `#2` vs `#2`.

Interpretation:
- this setup is useful as a deterministic cold-start corruption baseline, but
  it is not sufficient as the sole promoted corruption scheme for faithful
  reasoning evaluation.

### 8.4 Upstream omitted-risk checklist (must be controlled before promotion)

These are upstream risks that can silently contaminate Phase D results even if
teacher-side code is correct.

Risk A: prompt serialization mismatch with backbone family.
- Description:
  - some upstream runs use repo-defined role markers (`[SYSTEM]/[USER]/[ASSISTANT]`)
    instead of model-native chat template serialization.
  - this may create hidden formatting drift and unstable parse behavior.
- Control:
  1. freeze one explicit prompt-serialization contract per dataset family,
  2. run one A/B sanity test (`plain-marker` vs `chat-template`) on same split,
  3. if material delta exists, lock one contract and record it in manifest.

Risk B: split-policy comparability drift (`hash` vs `official`).
- Description:
  - mixing local hash split and official split weakens cross-run and external-baseline comparability.
- Control:
  1. every D-stage result table must include `source_split` + `split_policy`,
  2. promotion decisions must be based on one fixed protocol only,
  3. external/SOTA comparisons are valid only when split protocol matches.

Risk C: prepared-artifact staleness under resume mode.
- Description:
  - parameter fingerprinting can miss upstream raw-data changes when args are unchanged.
- Control:
  1. add provenance digests (raw-source hash + prepared JSONL hash) in manifest,
  2. disable resume or force provenance revalidation for promotion-grade runs,
  3. hard-fail if provenance changes but prepared artifact is reused.

Risk D: style conclusions confounded by unequal decode budgets.
- Description:
  - style A/B/C conclusions are not causal when `max_new_tokens` or decode mode differs.
- Control:
  1. unequal-budget style runs are exploratory only,
  2. require equal-budget/equal-decode confirmation run for promotable style claims,
  3. reject style-causality conclusions without this confirmation.

Risk E: extractor-template mismatch inflates parse-error metrics.
- Description:
  - template output contract (for example `Verdict:`) may be partially misaligned with evaluator extraction rules.
- Control:
  1. add template-parser compatibility tests before D4 full runs,
  2. report parse-error by extraction-method bucket,
  3. promotion evidence must be parser-compatible or parser-change-versioned.

## 9. Phase D Deliverables

1. New plan file (`phase_D_plan.md`) and updated top-level docs.
2. Teacher sidecar scoring script and tests.
3. C1 fusion support and tests.
4. C2 target-source switch and tests.
5. Phase D experiment report with ablation table and pass/fail recommendation.

## 10. Immediate Next Steps

1. D1 is implemented:
   - `scripts/phase_c_score_prm_teacher.py`
2. D2 is implemented in C1:
   - `scripts/phase_b_prepare_value_data.py` now emits
     `q_teacher/q_fused/teacher_available/teacher_disagree/teacher_model_id`
     when teacher sidecar scores are provided.
3. D3 is implemented in C2:
   - `scripts/phase_b_train_value.py` now supports
     `--target-source {q_mean_smoothed,q_teacher,q_fused}`.
   - `scripts/phase_b_eval_faithfulness.py` supports `--target-source from_run`
     for evaluation consistency with training.
4. Run the first D4 four-way ablation on StrategyQA smoke, then full.
5. Publish `phase_D_report.md` as the new live report file.

### 10.1 Implementation Status Snapshot (2026-03-04)

Already implemented and validated by unit tests:
1. C1 pair-consensus gate (teacher clean-vs-corrupt delta checks):
   - new controls in `scripts/phase_b_prepare_value_data.py`:
     - `--teacher-corruption-scores-jsonl`
     - `--pair-consensus-*`
2. C2 top-k contrastive candidates per clean prefix:
   - data loader supports ranked corruption candidates,
   - training cache encodes candidate banks instead of single corruption only.
3. C2 stage scheduler:
   - `--train-mode {joint,ranking_only,calibration_only,two_stage}`
   - `--two-stage-ranking-ratio`
4. Bundled Phase D suite upgraded:
   - `scripts/run_phase_d_teacher_suite.sh`
   - new groups:
     - `D4_STRATEGYQA_SMOKE_3WAY_HQ`
     - `D4_STRATEGYQA_FULL_3WAY_HQ`

Remaining highest-priority execution item:
1. run full HQ ablation matrix and decide D5 promotion against Section 11 gates.

## 11. Quantitative Promotion Gates (ABR-Oriented)

Phase D must pass all gates before restarting BCR-lite / ABR-lite expansion.

### 11.1 Data-quality gate (C1)

1. Teacher coverage:
   - `teacher_available_ratio >= 0.98` on both train/eval C1 artifacts.
2. Pair quality:
   - `positive_delta_ratio` improves materially versus MC-only baseline.
   - `mean_pair_weight` improves and remains stable across repeats.
3. Disagreement control:
   - disagreement histogram must be bounded and explainable (no silent collapse).
4. Join integrity:
   - duplicate ID count must be `0`,
   - dropped-id count must be explicitly reported with causes.
5. Corruption-design quality:
   - dominant corruption-type ratio must be `<= 0.70`,
   - at least `3` corruption types must each have `>= 0.05` share,
   - degenerate/self-contradictory mutation rate must be `<= 0.05`.

### 11.2 Model-quality gate (C2)

1. Calibration:
   - selected run beats trivial calibration baseline on Brier, with repeated runs.
2. Ranking:
   - `corr_pair_acc >= 0.55` and `corr_auc >= 0.60` on held-out eval.
3. Stability:
   - metrics remain directionally consistent across at least 2 reruns.
4. Anti-illusion rule:
   - post-hoc-only improvement is insufficient,
   - promotion requires raw ranking signal and utility signal to move in same direction.

### 11.3 Utility gate (ABR relevance)

1. Downstream rerank/router utility shows net gain under equal budget.
2. If utility gain is absent, Phase D is not promoted even if single metrics improve.
3. Utility claims from unequal-budget runs are informational only, not promotable.

## 12. Operational Runbook (Minimal Commands)

This section integrates the practical strengths from Phase C diagnosis docs so
Phase D can be executed by newcomers without re-deriving setup details.

### 12.1 Teacher environment quick check

```bash
conda activate bcr
export HF_HOME=$PWD/assets/hf_cache
export HF_DATASETS_CACHE=$PWD/assets/hf_cache/datasets
export PYTHONPATH=$PWD/src
python - <<'PY'
import torch, transformers, accelerate
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('accelerate', accelerate.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_count', torch.cuda.device_count())
PY
```

### 12.2 Teacher model deployment

```bash
huggingface-cli login
huggingface-cli download Qwen/Qwen2.5-Math-PRM-7B \
  --local-dir assets/models/Qwen2.5-Math-PRM-7B \
  --local-dir-use-symlinks False
```

### 12.3 Teacher smoke scoring compatibility rule

When scoring with Qwen PRM in this repo environment:
1. use `dtype=bfloat16`,
2. keep `use_cache=False` in forward calls to avoid cache API mismatch in some
   `transformers + remote-code` combinations.

## 13. ABR Handoff Policy

### 13.1 What counts as “Phase D success”

1. External teacher materially improves C1 supervision quality.
2. C2 reaches the promotion gates in Section 11.
3. Improvement is reproducible and not a one-off metric spike.

### 13.2 If success: next stage order (fixed)

1. Restart BCR-lite (`L_sft + lambda_B * L_Bellman`) with promoted target source.
2. Run ABR-lite heuristic router on top of promoted value head.
3. Consider router RL only after ABR-lite frontier improvement is verified.

### 13.3 If failure: fallback order (fixed)

1. Improve pair construction first (higher-margin, better sibling-style pairs).
2. Increase selective rollout budget only for uncertain prefixes.
3. Run control teacher (`PRM800K`) before architecture expansion.
4. Do not start ABR router work on an unpromoted value head.

## 14. Integration Summary (Why This Plan Is the Main Driver)

`phase_D_plan.md` is selected as the execution backbone because it is:
1. sequencing-first (workstreams D0–D5),
2. promotion-gate-first (prevents premature ABR expansion),
3. ablation-first (protects scientific attribution).

Integrated strengths from `phase_C_fix_value_head.md` now included here:
1. explicit quantitative promotion gates,
2. teacher deployment/compatibility runbook,
3. ABR handoff criteria and no-go policy.

## 15. Final Carry-Out Version (Locked Before Implementation)

This section freezes the execution order so implementation does not drift.

### 15.1 Phase D implementation order (must follow)

1. D0 gate first:
   - verify `bcr` env, CUDA visibility, and PRM load+forward smoke.
2. D1 script:
   - implement `scripts/phase_c_score_prm_teacher.py` and unit tests.
3. D2 fusion:
   - add `q_teacher/q_fused/teacher_available/teacher_disagree` into C1 output,
   - keep backward compatibility when no teacher path is provided.
4. D3 training target source:
   - add `--target-source` to C2 and propagate into summaries/manifests.
5. D4 smoke ablations:
   - run `mc/teacher/fused_fixed/fused_confidence` on StrategyQA smoke.
6. D4 full ablations:
   - run the same four-way matrix on full StrategyQA.
7. D5 decision:
   - apply Section 11 gates, then choose promote or fallback path.

No ABR router work may start before step 7 completes.

### 15.2 Minimal artifact contract additions (locked)

C1 additional fields (prefix target record):
1. `q_teacher` (nullable float in `[0,1]`)
2. `q_fused` (nullable float in `[0,1]`)
3. `teacher_available` (bool)
4. `teacher_disagree` (bool)
5. `teacher_model_id` (nullable string)

C2 additional summary fields:
1. `target_source`
2. `teacher_coverage_ratio`
3. `teacher_disagreement_ratio`

### 15.3 Final promotion decision template (locked)

1. If both calibration + ranking gates pass:
   - mark `Phase D: promoted`,
   - proceed to BCR-lite restart sequence.
2. If either gate fails:
   - mark `Phase D: not promoted`,
   - execute fallback order in Section 13.3.
3. All decisions must be attached to a reproducible run table and manifest IDs.

### 15.4 Pre-implementation no-go checklist (locked)

Implementation may start only after the following are explicitly acknowledged:
1. Teacher serialization contract is fixed and testable (`<extra_0>` handling).
2. `use_cache=False` rule is enforced in teacher scoring path.
3. ID-join hard checks are implemented (duplicate/drop detection).
4. Fusion mapping is train-only and leakage-safe.
5. Equal-budget ablation protocol is fixed for promotable conclusions.
6. Corruption-type mix and mutation-validity checks are implemented for C1 artifacts.
7. Prompt serialization contract is frozen (`plain-marker` vs `chat-template` decision logged).
8. Split comparability contract is frozen (`source_split` + `split_policy` fixed for promotion runs).
9. Prepared-artifact provenance digest checks are enabled (raw-source + prepared JSONL hashes).
10. Template-parser compatibility checks are green for promoted template families.

## 16. PRM Primer and Community Lessons (Phase D Appendix)

This appendix captures the practical consensus used by Phase D design.

### 16.1 PRM vs LM (operational difference)

1. Base LM:
   - objective: next-token generation,
   - output: text continuation quality.
2. PRM (process reward model):
   - objective: score reasoning process quality at step/prefix level,
   - output: step/prefix scores used as supervision or ranking signal.
3. Execution consequence:
   - LM tolerates prompt variance better;
   - PRM is highly sensitive to input contract (chat template, step separators, score extraction positions).

### 16.2 Known community conclusions relevant to us

1. Weak outcome-only supervision often underdetermines step quality.
2. PRM/value-head signal quality is commonly bottlenecked by noisy labels.
3. Better calibration numbers alone do not guarantee better process ranking.
4. Pair/ranking construction quality frequently dominates objective-trick gains.
5. Teacher signal can help cold-start, but only with strict contract + ablation controls.

### 16.3 Phase-D best practices (must follow)

1. Keep teacher scoring as sidecar first (`D1`), then fuse (`D2`), then retrain (`D3`).
2. Enforce stable serialization:
   - fixed system prompt,
   - fixed step separator token,
   - deterministic row-to-score mapping.
3. Run equal-budget four-way ablations for promotable claims:
   - `mc`,
   - `teacher`,
   - `fused_fixed`,
   - `fused_confidence`.
4. Always report calibration + ranking + utility together.
5. Audit disagreement buckets (teacher vs MC) before interpreting gains.

### 16.4 Anti-patterns to avoid

1. Promoting based on post-hoc calibration gains only.
2. Mixing teacher/MC score scales without train-only mapping.
3. Drawing conclusions from unequal-budget comparisons.
4. Letting pair filtering collapse effective sample size.
5. Treating one-off metric spikes as stage-level success.

### 16.5 External references used by this appendix

1. OpenAI PRM / process supervision:
   - https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/
2. Let’s Verify Step by Step (PRM800K context):
   - https://arxiv.org/abs/2305.20050
3. Qwen teacher models:
   - https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B
   - https://huggingface.co/Qwen/Qwen2.5-Math-7B-PRM800K
4. ProcessBench / PRMBench / lessons papers:
   - https://arxiv.org/abs/2412.06559
   - https://arxiv.org/abs/2501.03124
   - https://arxiv.org/abs/2501.07301

## 17. Corruption-Quality Repair Plan (Priority: Immediate)

This section is added to directly address the currently observed risk:
- corruption mix is dominated by `step_drop`,
- semantic mutation coverage is too weak,
- pair ranking signal is diluted.

This plan is the highest-priority blocker before interpreting D2/D3 gains.

### 17.1 Goal

Upgrade C1 corruption quality so that value-head training receives:
1. diverse mutation types,
2. stronger clean-vs-corrupt margin,
3. stable pair coverage without collapsing sample count.

### 17.2 Workstream CQR (Corruption-Quality Repair)

#### CQR-1: Corruption generator expansion

Add semantic corruption operators beyond local numeric edits:
1. negation polarity flip (`is` <-> `is not`, `can` <-> `cannot`),
2. entity substitution (replace a key entity span with another entity from context),
3. condition reversal (`if A then B` -> `if not A then B`),
4. comparator reversal (`more` <-> `less`, `before` <-> `after`).

Implementation constraints:
1. keep local-edit principle (change one step at a time),
2. preserve JSONL schema compatibility,
3. log `corruption_type`, `corruption_span`, and mutation metadata for diagnostics.

#### CQR-2: Per-prefix quota and type balancing

When building corruptions for each prefix:
1. reserve at most one `step_drop` variant,
2. reserve at least one non-`step_drop` variant when available,
3. sample up to `max_corruptions_per_prefix` with type-balanced priority.

Selection priority:
1. semantic operators,
2. numeric/operator perturbation,
3. `step_drop` fallback.

#### CQR-3: Pair-quality filtering by label margin

Use label-side margin first (not model-score margin):
1. compute `delta_q = q_clean - q_corrupt`,
2. retain pairs with `delta_q >= delta_q_min`,
3. downweight high-uncertainty pairs using `q_ci_width`.

Default starting thresholds:
1. `delta_q_min = 0.15` (smoke),
2. `delta_q_min = 0.20` (full),
3. uncertainty-weighting enabled.

#### CQR-4: Stratified sampling for training pairs

During C2 training batch assembly, stratify by:
1. `corruption_type`,
2. `prefix_step_index` bucket.

Reason:
1. prevent one mutation family from dominating,
2. improve robustness of ranking signal across step positions.

#### CQR-5: Two-tier gates (debug vs promotion)

This refines Section 11.1 item 5 into staged gates:

Debug gate (must pass before continuing experiments):
1. dominant corruption-type ratio `<= 0.85`,
2. at least `2` corruption types each with `>= 0.02` share,
3. degenerate/self-contradictory mutation rate `<= 0.10`.

Promotion gate (must pass for final claims; unchanged strict target):
1. dominant corruption-type ratio `<= 0.70`,
2. at least `3` corruption types each with `>= 0.05` share,
3. degenerate/self-contradictory mutation rate `<= 0.05`.

### 17.3 Experiment matrix after CQR

Run this order:
1. CQR baseline replay (current corruption policy) for control,
2. CQR-1/2 enabled, no teacher (MC-only),
3. CQR-1/2 enabled + teacher (`q_teacher`),
4. CQR-1/2 enabled + fused (`q_fused_fixed`, `q_fused_confidence`).

Report for each:
1. corruption mix table,
2. pair margin histogram (`delta_q`),
3. retained pair ratio,
4. C2 metrics (`Brier`, `Pearson`, `corr_pair_acc`, `corr_auc`),
5. utility signal under equal budget.

### 17.4 Decision rule

1. If CQR improves pair diagnostics but C2 metrics still fail promotion gates:
   - conclude supervision remains insufficient even after corruption repair,
   - escalate to stronger external supervision path (multi-teacher or PRM-labeled data).
2. If CQR improves both pair diagnostics and C2 ranking/calibration:
   - keep repaired corruption policy as Phase D default,
   - continue with ABR-oriented rerank/router tests.

### 17.5 D3 Plateau Diagnosis (2026-03-04 Smoke Snapshot)

This subsection records the current best-effort diagnosis for why Phase D
(`teacher` / `fused`) shows only limited ranking gains.

Observed snapshot (from `phase_d_bundle_smoke`):
1. ranking moved slightly (`pair_acc` from ~0.547 to ~0.566) but `corr_auc`
   stayed around `0.52` and did not approach the promotion gate `0.60`,
2. teacher coverage is high (`teacher_available_ratio=1.0`), so the bottleneck is
   not missing teacher rows,
3. disagreement is non-trivial (~0.34 on eval), indicating MC and teacher signals
   are not fully aligned.

Likely bottlenecks:
1. **Target saturation under teacher/fused labels**:
   - `q_teacher`/`q_fused` are concentrated near high values on StrategyQA,
   - this weakens calibration learnability and can make Brier-baseline comparison
     structurally unfavorable.
2. **Fusion dominance**:
   - confidence fusion currently keeps `q_fused` close to `q_teacher` in practice,
   - effective diversity between `teacher` and `fused` supervision is limited.
3. **Ranking branch remains pair-quality limited**:
   - D2 mainly augments clean-prefix targets; contrastive ranking still depends on
     corruption pair quality and filtering gates,
   - if pair margins are weak/noisy, PRM injection at clean-target level cannot
     fully fix ranking.
4. **Frozen-backbone capacity constraint**:
   - C2 only trains a small value head; representational updates are limited,
   - noisy supervision or weak pair signal is harder to absorb in this regime.

Action items (must be treated as Phase D-critical):
1. **Label-shape diagnostics (mandatory)**:
   - report distribution and entropy of `q_mc`, `q_teacher`, `q_fused` per split,
   - include spread metrics (std, IQR, quantiles) and not only means.
2. **Fusion sensitivity sweep (mandatory)**:
   - run fixed-lambda and confidence-fusion variants under equal budget,
   - explicitly compare how much `q_fused` departs from `q_teacher`.
3. **Teacher-for-ranking extension (priority)**:
   - add optional corruption-side teacher scoring path and evaluate teacher-derived
     pair margins, not only clean-prefix scalar targets.
4. **Pair-signal strengthening (priority)**:
   - tighten `delta_q`/`z_delta` gates and monitor retained-pair ratio,
   - prioritize high-margin pairs before increasing architecture complexity.
5. **Capacity fallback protocol**:
   - if gates still fail after improved pair quality, run controlled unfreeze/adapter
     ablation to test whether the bottleneck is model capacity rather than labels.

Interpretation rule:
1. Do not claim “PRM ineffective” from one smoke table.
2. The current evidence supports a narrower conclusion:
   - **PRM-as-clean-target augmentation alone is insufficient for promotion-level
     ranking gains in the current C2 setup.**

### 17.6 External Evidence for Similar Plateaus and What Worked

This subsection records external evidence that this type of D2/D3 plateau is
common, plus interventions that repeatedly helped.

Observed in prior work:
1. **Process signal is often weak/noisy without strong supervision density**:
   - *Let’s Verify Step by Step* required large-scale step supervision to train a
     strong PRM, rather than relying on sparse outcome-derived signals.
2. **Process feedback can outperform outcome feedback, but only with careful setup**:
   - *Solving Math Word Problems With Process- and Outcome-Based Feedback* reports
     strong gains from process supervision in math settings, reinforcing that label
     quality/pipeline design is the deciding factor.
3. **Benchmark evidence that process-level checking is hard**:
   - ProcessBench reports many strong LLM/RM variants still struggle on explicit
     process-level verification tasks.
4. **PRM can be gamed if used naively**:
   - *The Lessons of Developing Process Reward Models in Mathematical Reasoning*
     highlights reward-hacking patterns (e.g., verbosity/BoN effects) and argues
     for stricter anti-gaming protocols.
5. **Theory reminder**:
   - *Do We Need to Verify Step by Step?* indicates process supervision is not a
     free lunch; when assumptions hold, outcome supervision can be similarly
     learnable, so implementation/data quality dominate practical outcomes.

What prior work typically does to recover signal:
1. **Improve label quality first**:
   - increase effective supervision density,
   - keep high-margin/hard pairs, down-weight ambiguous pairs,
   - control label saturation and distribution collapse.
2. **Use ranking-centric objectives for decision boundaries**:
   - prefer pair/listwise formulations for pairwise selection behavior,
   - avoid relying only on pointwise calibration loss when ranking is the target.
3. **Use PRM as a teacher/search signal, not only a scalar replacement target**:
   - use step-wise scores in rerank/selection loops,
   - evaluate with process-specific benchmarks, not just aggregate calibration.
4. **Add anti-gaming checks**:
   - length sensitivity, format sensitivity, and BoN inflation tests are required
     before concluding a true faithfulness gain.

Direct implications for our D2/D3 stack:
1. Current pattern (small pair gain, AUC plateau) is consistent with known weak
   signal regimes; this is not an isolated failure mode.
2. The most likely high-impact next step is not “more fusion variants” alone, but:
   - teacher-informed **pair construction**,
   - stronger pair filtering/margin controls,
   - explicit anti-gaming diagnostics,
   - and only then controlled capacity increase if needed.
3. Promotion interpretation must remain strict:
   - we only claim D-stage success after both ranking and calibration pass gates
     under robustness checks.

Reference links (primary sources and official code):
1. Let’s Verify Step by Step (paper): https://arxiv.org/abs/2305.20050
2. PRM800K (official repo): https://github.com/openai/prm800k
3. Process- vs Outcome-Based Feedback (paper): https://arxiv.org/abs/2211.14275
4. ProcessBench (paper): https://arxiv.org/abs/2412.06559
5. ProcessBench (official repo): https://github.com/QwenLM/ProcessBench
6. PRMBench (paper): https://arxiv.org/abs/2504.16828
7. PRMBench (official repo): https://github.com/ssmisya/PRMBench
8. The Lessons of Developing PRMs (paper): https://arxiv.org/abs/2501.07301
9. Do We Need to Verify Step by Step? (paper): https://arxiv.org/abs/2502.10581
10. ThinkPRM (official repo): https://github.com/mukhal/thinkprm
11. ImplicitPRM (official repo): https://github.com/PRIME-RL/ImplicitPRM

### 17.7 Pair Data Bootstrap Decision and Available Resources

This subsection captures the project-direction decision from recent discussion:
we should not rely on a 7B model to bootstrap pair supervision from scratch.

Decision:
1. **Do not use 7B self-bootstrapping as the primary pair-data source**.
2. **Use external pair/step-supervision resources + teacher filtering as the
   primary path**, and keep 7B-generated pairs only as a late-stage supplement.

Why:
1. Current D2/D3 evidence already shows weak-signal behavior (small pair gain,
   AUC plateau).
2. Prior PRM literature repeatedly reports high noise when process labels are
   sparse or weakly grounded.
3. In this setting, pure self-generated pairs are likely to amplify bias and
   annotation noise rather than fix ranking quality.

Available resources (high priority first):
1. **R-PRM dataset** (direct chosen/rejected preference pairs):
   - https://huggingface.co/datasets/kevinpro/R-PRM
   - https://github.com/NJUNLP/R-PRM
2. **PRMBench_Preview** (original vs modified process trajectories):
   - https://huggingface.co/datasets/hitsmy/PRMBench_Preview
3. **PRM800K** (step-level ratings, convertible to pair data):
   - https://github.com/openai/prm800k
4. **Math-Shepherd** (step-level +/- supervision):
   - https://huggingface.co/datasets/peiyi9979/Math-Shepherd
5. **RLHFlow PRM data** (step-level process labels):
   - https://huggingface.co/datasets/RLHFlow/Mistral-PRM-Data
   - https://huggingface.co/datasets/RLHFlow/Deepseek-PRM-Data
   - https://github.com/RLHFlow/RLHF-Reward-Modeling

Evaluation-only resources (not primary training source):
1. ProcessBench:
   - https://arxiv.org/abs/2412.06559
   - https://github.com/QwenLM/ProcessBench
2. PRMBench:
   - https://arxiv.org/abs/2504.16828
   - https://github.com/ssmisya/PRMBench

Execution plan (Phase D alignment):
1. Build a **pair-source mix**: external curated pairs + converted step-level pairs.
2. Normalize into current C1/C2 schema with explicit `source_tag` and confidence.
3. Apply teacher-side filtering:
   - keep high-margin pairs first,
   - down-weight ambiguous pairs,
   - reject style/length-only artifacts.
4. Train C2 ranking with source-balanced sampling.
5. Keep 7B-generated pairs as optional augmentation only after baseline quality
   gates are met.

Quality gates before promotion:
1. Pair-level:
   - retained high-quality pair ratio above threshold,
   - margin distribution not collapsed.
2. Model-level:
   - `corr_pair_acc` and `corr_auc` both improve under equal budget,
   - robustness checks (length/style/BoN anti-gaming) pass.
3. Claim-level:
   - no D-stage promotion claim if gains depend on one narrow source only.

### 17.8 Task-Specific Availability (GSM8K vs StrategyQA) and Installation Guide

Key availability finding:
1. **GSM8K**:
   - has multiple community preference/pair datasets (chosen/rejected format),
   - plus several step-level process datasets convertible to pair supervision.
2. **StrategyQA**:
   - official data provides question/answer/decomposition/evidence,
   - no widely adopted official chosen/rejected pair corpus.
3. Project implication:
   - for GSM8K, external pair resources can be plugged in directly,
   - for StrategyQA, pair supervision still relies on our corruption + teacher
     scoring pipeline.

Verified dataset links (as of 2026-03-04):
1. GSM8K official:
   - https://huggingface.co/datasets/openai/gsm8k
   - https://github.com/openai/grade-school-math
2. StrategyQA official:
   - https://huggingface.co/datasets/voidful/StrategyQA
3. Direct pair / preference resources:
   - https://huggingface.co/datasets/kevinpro/R-PRM
   - https://huggingface.co/datasets/hitsmy/PRMBench_Preview
   - https://huggingface.co/datasets/Rudra-ai/ai-responses-gsm8k-405b-dpo
   - https://huggingface.co/datasets/Rudra-ai/ai-responses-gsm8k-70b-update-dpo
   - https://huggingface.co/datasets/Genesis-AI-Labs/GAIL-gsm8k-preference-small
4. Step-level resources (convertible to pair):
   - https://huggingface.co/datasets/peiyi9979/Math-Shepherd
   - https://huggingface.co/datasets/RLHFlow/Mistral-PRM-Data
   - https://huggingface.co/datasets/RLHFlow/Deepseek-PRM-Data
5. Evaluation/diagnostic:
   - https://huggingface.co/datasets/Qwen/ProcessBench
   - https://github.com/ssmisya/PRMBench

Install/download instructions (copy-paste friendly):

1. Environment setup:
```bash
conda activate bcr
pip install -U datasets huggingface_hub pyarrow
huggingface-cli --version
```

2. Optional auth (for rate limits/private mirrors):
```bash
huggingface-cli login
```

3. Create local storage root:
```bash
mkdir -p assets/external_datasets
```

4. Fast raw download (recommended first pass):
```bash
huggingface-cli download openai/gsm8k --repo-type dataset --local-dir assets/external_datasets/openai_gsm8k --local-dir-use-symlinks False
huggingface-cli download voidful/StrategyQA --repo-type dataset --local-dir assets/external_datasets/voidful_strategyqa --local-dir-use-symlinks False
huggingface-cli download kevinpro/R-PRM --repo-type dataset --local-dir assets/external_datasets/kevinpro_r_prm --local-dir-use-symlinks False
huggingface-cli download hitsmy/PRMBench_Preview --repo-type dataset --local-dir assets/external_datasets/hitsmy_prmbench_preview --local-dir-use-symlinks False
huggingface-cli download peiyi9979/Math-Shepherd --repo-type dataset --local-dir assets/external_datasets/peiyi_math_shepherd --local-dir-use-symlinks False
huggingface-cli download RLHFlow/Mistral-PRM-Data --repo-type dataset --local-dir assets/external_datasets/rlhflow_mistral_prm --local-dir-use-symlinks False
huggingface-cli download RLHFlow/Deepseek-PRM-Data --repo-type dataset --local-dir assets/external_datasets/rlhflow_deepseek_prm --local-dir-use-symlinks False
huggingface-cli download Qwen/ProcessBench --repo-type dataset --local-dir assets/external_datasets/qwen_processbench --local-dir-use-symlinks False
```

5. Optional community GSM8K pair datasets:
```bash
huggingface-cli download Rudra-ai/ai-responses-gsm8k-405b-dpo --repo-type dataset --local-dir assets/external_datasets/rudra_gsm8k_405b_dpo --local-dir-use-symlinks False
huggingface-cli download Rudra-ai/ai-responses-gsm8k-70b-update-dpo --repo-type dataset --local-dir assets/external_datasets/rudra_gsm8k_70b_dpo --local-dir-use-symlinks False
huggingface-cli download Genesis-AI-Labs/GAIL-gsm8k-preference-small --repo-type dataset --local-dir assets/external_datasets/gail_gsm8k_preference_small --local-dir-use-symlinks False
```

6. Validate schemas quickly before integration:
```bash
python - <<'PY'
from datasets import (
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)

repos = [
    "openai/gsm8k",
    "voidful/StrategyQA",
    "kevinpro/R-PRM",
    "hitsmy/PRMBench_Preview",
    "peiyi9979/Math-Shepherd",
    "RLHFlow/Mistral-PRM-Data",
    "RLHFlow/Deepseek-PRM-Data",
    "Qwen/ProcessBench",
]

for repo in repos:
    cfgs = get_dataset_config_names(repo)
    cfg = cfgs[0] if cfgs else None
    split_kwargs = {"name": cfg} if cfg else {}
    splits = get_dataset_split_names(repo, **split_kwargs)
    split = "train" if "train" in splits else splits[0]
    load_kwargs = {"split": split}
    if cfg:
        load_kwargs["name"] = cfg
    ds = load_dataset(repo, **load_kwargs)
    row = ds[0]
    print("\\n==", repo)
    print("config:", cfg)
    print("split:", split)
    print("rows:", len(ds))
    print("keys:", list(row.keys())[:12])
PY
```

Integration caution:
1. Treat community pair datasets as weakly trusted until schema + quality checks.
2. Keep `source_tag` and provenance hashes in manifests.
3. Do not mix StrategyQA with math-only pair data without explicit domain tagging.

## 18. Phase D4: External Pair Bootstrap and Integration

Objective:
1. Inject reliable external pair signal into C2 ranking training.
2. Keep StrategyQA in-domain supervision path (corruption + teacher) as the anchor.
3. Raise ranking metrics (`corr_pair_acc`, `corr_auc`) without sacrificing calibration robustness.

### 18.1 Scope and non-goals

In scope:
1. external pair ingestion and normalization,
2. step-label -> pair conversion,
3. pair quality filtering and source-balanced training integration,
4. controlled ablation matrix and promotion gates.

Out of scope (for D4):
1. full RL policy optimization,
2. replacing C1/C2 contracts end-to-end,
3. claiming cross-domain generalization without dedicated tests.

### 18.2 Data source roles

Role A (direct pair warm start):
1. `R-PRM (dpo)`:
   - strongest immediate chosen/rejected signal,
   - use first for ranking warm start.
2. `PRMBench_Preview`:
   - smaller but high-quality modified-process errors,
   - use as hard negative supplement.

Role B (converted pair expansion):
1. `Math-Shepherd`:
   - convert step `+/-` labels into pair candidates.
2. `RLHFlow Mistral/Deepseek PRM data`:
   - convert conversation-level `+/-` supervision into pair candidates.

Role C (in-domain anchor):
1. StrategyQA C1 artifacts:
   - keep corruption + teacher path as the primary in-domain signal.

### 18.3 New artifacts and contracts

Add one normalized external-pair artifact family:
1. output root:
   - `assets/artifacts/phase_d_external_pairs/<run_name>__<fingerprint>/`
2. required files:
   - `train_pairs.jsonl`
   - `validation_pairs.jsonl`
   - `summary.json`
   - `manifest.json`
3. canonical row schema:
   - `pair_id`
   - `source_tag` (e.g., `r_prm`, `prmbench_preview`, `math_shepherd`, `rlhflow_mistral`)
   - `domain_tag` (e.g., `gsm8k_math`, `strategyqa_like`, `general_math`)
   - `prompt_text`
   - `chosen_text`
   - `rejected_text`
   - `pair_confidence` in `[0, 1]`
   - `quality_flags` (length_ratio, overlap_ratio, parse_ok, etc.)
   - `metadata` (upstream ids/split/source file hashes)

### 18.4 Implementation tasks (code-level)

Task D4-1: external-pair preparation script
1. add `scripts/phase_d_prepare_external_pairs.py`:
   - load all configured sources,
   - map each source to canonical row schema,
   - deduplicate by normalized hash,
   - split train/validation deterministically,
   - write artifact + manifest.

Task D4-2: source adapters
1. add adapter module: `src/ours/phase_d/external_pairs_adapters.py`:
   - `load_r_prm_pairs(...)`
   - `load_prmbench_preview_pairs(...)`
   - `load_math_shepherd_step_pairs(...)`
   - `load_rlhflow_step_pairs(...)`
2. source-specific notes:
   - `R-PRM`: direct `instruction/chosen/rejected`.
   - `PRMBench_Preview`: chosen=`original_process`, rejected=`modified_process`,
     confidence boosted by `error_steps` count and explicit classification tags.
   - `Math-Shepherd` and `RLHFlow`: build intra-question step pairs from `+/-`
     labels; keep only high-margin candidates.

Task D4-3: pair quality filters
1. add module: `src/ours/phase_d/pair_filters.py`:
   - minimum/maximum length ratio,
   - lexical overlap sanity checks,
   - malformed/empty step rejection,
   - optional teacher rescoring gate.
2. write per-source rejection stats to summary.

Task D4-4: C2 training integration
1. extend `scripts/phase_b_train_value.py` with optional args:
   - `--external-pair-jsonl`
   - `--external-pair-weight`
   - `--external-pair-max-train-samples`
   - `--external-pair-source-balance` (`none|uniform|custom`)
   - `--external-pair-domain-filter`
2. training behavior:
   - keep existing C1 calibration + contrastive path,
   - add external pair ranking loss branch using the same value head,
   - combine loss with explicit weights.

Task D4-5: data loaders
1. add `src/ours/phase_d/external_pairs_data.py`:
   - canonical row validator,
   - source-balanced sampler,
   - optional domain-aware sampling caps.

### 18.5 Suggested execution order

Phase D4-A (minimum viable):
1. prepare pairs from `R-PRM` + `PRMBench_Preview`,
2. integrate into C2 as ranking warm start only,
3. evaluate vs current D3 baseline.

Phase D4-B (expansion):
1. add converted pairs from `Math-Shepherd` + `RLHFlow`,
2. apply stricter quality gates + lower initial weight,
3. rerun ablations.

Phase D4-C (in-domain stabilization):
1. combine external pairs with StrategyQA in-domain pair-quality branch,
2. tune source-balanced sampling,
3. run robustness/anti-gaming checks before promotion claim.

### 18.6 Experiment matrix (required)

E0 (baseline):
1. current D3 (`q_fused`, no external pair branch).

E1 (direct pair warm start):
1. add `R-PRM` only.

E2 (high-quality supplement):
1. E1 + `PRMBench_Preview`.

E3 (step-converted expansion):
1. E2 + `Math-Shepherd` + `RLHFlow` converted pairs.

E4 (domain-aware balancing):
1. E3 + source/domain balanced sampling + stricter filters.

For each experiment, report:
1. retained pair counts by source,
2. pair margin/quality distributions,
3. `corr_pair_acc`, `corr_auc`, calibration metrics,
4. anti-gaming checks (length/style/BoN sensitivity).

### 18.7 Promotion gates for D4

Primary gates:
1. `corr_auc` improves materially vs E0 and remains stable across at least two seeds.
2. `corr_pair_acc` improves with non-collapsed retained-pair ratio.
3. no severe degradation in calibration metrics.

Reliability gates:
1. gains persist under source-ablation (remove any single external source).
2. no single-source dominance > 70% in effective training pairs after filtering.
3. robustness checks pass (no obvious length/style reward hacking).

### 18.8 Risks and mitigations

Risk 1: domain shift from math-heavy external pairs.
1. Mitigation:
   - enforce `domain_tag`,
   - cap out-of-domain sampling ratio,
   - prioritize StrategyQA in-domain branch during late epochs.

Risk 2: noisy converted step pairs.
1. Mitigation:
   - apply strict pair-quality gates,
   - start with low external weight,
   - inspect rejection summaries each run.

Risk 3: shortcut learning from formatting artifacts.
1. Mitigation:
   - normalize templates,
   - add length/style perturbation checks,
   - block promotion if shortcut sensitivity is high.

### 18.9 Minimal command template (D4-A)

1. Prepare external pairs:
```bash
python -u scripts/phase_d_prepare_external_pairs.py \
  --run-name d4a_direct_pairs \
  --r-prm-root assets/external_datasets/kevinpro_r_prm \
  --prmbench-preview-path assets/external_datasets/hitsmy_prmbench_preview/prmbench_preview.jsonl \
  --output-root assets/artifacts/phase_d_external_pairs
```

2. Train C2 with external pair branch:
```bash
python -u scripts/phase_b_train_value.py \
  --train-dir <phase_c_train_dir> \
  --eval-dir <phase_c_eval_dir> \
  --target-source q_fused \
  --external-pair-jsonl assets/artifacts/phase_d_external_pairs/d4a_direct_pairs__<fp>/train_pairs.jsonl \
  --external-pair-weight 0.5 \
  --external-pair-source-balance uniform \
  --run-name d4a_with_external_pairs
```

3. Evaluate and compare with E0:
```bash
python -u scripts/phase_b_eval_faithfulness.py \
  --value-run-dir <d4a_run_dir> \
  --eval-dir <phase_c_eval_dir> \
  --checkpoint best \
  --run-name d4a_eval
```

### 18.10 Implemented entrypoints (D4A/B/C)

Implemented scripts:
1. `scripts/phase_d_prepare_external_pairs.py`
   - normalizes external sources into canonical pair schema.
2. `scripts/run_phase_d_external_pair_suite.sh`
   - one-command runner for D4A / D4B / D4C.
3. `scripts/phase_b_train_value.py`
   - supports external pair branch via:
     - `--external-pair-jsonl`
     - `--external-pair-weight`
     - `--external-pair-source-balance`
     - `--external-pair-min-confidence`
     - related controls.

Supported suite groups:
1. `D4A_STRATEGYQA_SMOKE`
2. `D4B_STRATEGYQA_SMOKE`
3. `D4C_STRATEGYQA_SMOKE`
4. `D4ABC_STRATEGYQA_SMOKE`
5. `D4ABC_STRATEGYQA_FULL`

Quick smoke run:
```bash
ACTIVE_PHASE_D4_GROUP=D4ABC_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_d4abc_smoke \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<eval_dir> \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_d_external_pair_suite.sh
```
