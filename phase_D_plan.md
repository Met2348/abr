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
