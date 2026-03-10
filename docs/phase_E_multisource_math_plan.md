# Phase E Multi-Source Math Mixture Plan

This file defines the next official `Phase E` experimental direction after the
latest `Math-Shepherd` trust runs.

Date baseline: 2026-03-10.

---

## 0. Why This Document Exists

The latest single-source result is now clear enough to change the next
experiment priority.

Newest repository evidence:
1. `E15_MATH_SHEPHERD_TRUST_ROBUST_SEED3`
   - source-family held-out is clearly positive:
     - `mean_heldout_pair_acc = 0.7518`
     - `mean_heldout_auc = 0.7280`
   - but benchmark-native transfer is still weak:
     - `ProcessBench GSM8K mean_auc = 0.4834`
     - `ProcessBench Math mean_auc = 0.4746`
2. Therefore:
   - single-source process supervision is sufficient to prove learnability,
   - but insufficient to justify a trustworthy benchmark-facing value head.

Operational conclusion:
1. We should stop expecting:
   - "train on one math process dataset, then directly generalize to unseen
     benchmarks".
2. We should instead move to:
   - **same-family multi-source mixture training**
   - with explicit source balancing, held-out-by-source evaluation, and
     benchmark-native evaluation as separate gates.
3. Data-semantics constraint after the 2026-03-10 fix:
   - `Math-Shepherd` / `RLHFlow` / `PRM800K` fallback must now be interpreted as
     strict `first_bad_edge` sources,
   - not as true same-step sibling-branch sources.
4. Any pre-fix nearest-negative artifact should be treated as `legacy`.

---

## 1. Literature / Community Guidance

This section summarizes the most relevant external guidance for the exact
question we now face.

### 1.1 Strongest signals from current papers

1. `ProcessBench`
   - Existing PRMs often fail to generalize beyond easier math distributions.
   - This is exactly why `ProcessBench` was created.
   - Practical implication:
     - benchmark failure after single-source training is not surprising,
     - and should not be treated as a code bug by default.
2. `PRMBench`
   - Fine-grained process evaluation must go beyond simple step correctness.
   - PRMs need sensitivity to multiple error dimensions, not just one scalar.
   - Practical implication:
     - optimizing one source-family pair metric is not enough to trust a head.
3. `VersaPRM`
   - Single-domain PRMs generalize poorly outside their training domain.
   - Multi-domain synthetic reasoning data was introduced exactly to address
     this weakness.
   - Practical implication:
     - if we want broader benchmark performance, multi-source supervision is a
       principled next step.
4. `R-PRM`
   - Stronger pair construction and preference optimization can outperform much
     larger PRM datasets.
   - Practical implication:
     - source quality and preference formulation matter more than just adding a
       huge weak source.
5. `The Lessons of Developing Process Reward Models`
   - Monte Carlo label synthesis often generalizes worse than stronger
     judge/human-style annotation.
   - Best-of-N evaluation can also hide process-verification failure.
   - Practical implication:
     - weak or noisy sources must not dominate a mixed pool,
     - and promotion cannot rely on response-level gains alone.
6. `OmegaPRM`
   - Automated process supervision can scale if first-error localization and
     positive/negative balance are preserved.
   - Practical implication:
     - error-localized, balanced pair construction should be preferred over
       coarse outcome-derived supervision.
7. `Qwen2.5-Math Technical Report`
   - Reward models were used in an iterative, staged self-improvement loop,
     not as a one-shot isolated component.
   - Practical implication:
     - staged mixture / curriculum is more defensible than one-shot source
       concatenation.
8. `ImplicitPRM`
   - Task-relevant instructions matter more than merely increasing response
     diversity.
   - Practical implication:
     - source coverage should be benchmark-family relevant, not just bigger.

### 1.2 What these sources jointly imply

The combined lesson is:
1. We do need broader supervision than one source.
2. But we do **not** want "uncontrolled bigger mixture".
3. The correct direction is:
   - multi-source,
   - same-family,
   - source-balanced,
   - benchmark-clean,
   - ranking-first,
   - with source-wise ablations and leave-one-source-out checks.

---

## 2. Main Pitfalls To Avoid

These are the main failure modes that the literature and our own results jointly
warn about.

### 2.1 Weak-source domination

Example from our own runs:
1. `PRM800K` remains weak under the current formulation.
2. Therefore adding it naively at full volume can drown stronger sources.

Rule:
1. Do not let the largest dataset determine the mixture automatically.
2. Use per-source caps or source-balanced sampling.

### 2.2 Label-semantics mismatch

Not all "process supervision" sources mean the same thing:
1. `Math-Shepherd`:
   - step reward / process-wise labels
2. `R-PRM`:
   - direct chosen/rejected preference signal
3. `PRMBench_Preview`:
   - original vs modified process traces
4. `PRM800K`:
   - public PRM step supervision, but currently weak under our adapter

Rule:
1. Mixed training must normalize these into a consistent pairwise objective.
2. Source tags must be preserved in logs and evaluation.

### 2.3 Benchmark contamination

If we train on the same benchmark examples later used for external evaluation,
the benchmark becomes meaningless.

Rule:
1. `ProcessBench` remains eval-only.
2. `PRMBench` official benchmark remains eval-only unless a distinct train/dev
   split is explicitly available and isolated.
3. `PRMBench_Preview` can be used as supervision only because it is already a
   separate preview artifact in our repo contract.

### 2.4 Over-trusting one metric

We have already seen this failure mode:
1. better Brier does not imply better ranking,
2. higher same-source score does not imply benchmark robustness.

Rule:
1. A candidate must be judged by:
   - source-family held-out,
   - benchmark-native metrics,
   - seed stability,
   together.

### 2.5 One-shot mixture overfitting

Large one-shot concatenation is convenient but methodologically weak.

Rule:
1. We should compare:
   - direct mixed training,
   - staged training,
   - and low-weight weak-source augmentation,
   instead of assuming one-shot concatenation is optimal.

---

## 3. Source Policy for the Next Phase E Matrix

### 3.1 Core sources

These are the first-wave sources we should trust enough to build the new matrix.

1. `Math-Shepherd`
   - strongest current same-source learnability evidence
   - role:
     - strong anchor source
2. `R-PRM`
   - literature-backed strong process-preference source
   - role:
     - direct preference supervision
3. `PRMBench_Preview`
   - closer to benchmark-style process modification
   - role:
     - benchmark-aligned auxiliary source

### 3.2 Weak-source policy

1. `PRM800K` should not be removed entirely from science.
2. But it should not be a first-wave dominant source.
3. Its role should be:
   - weak-source ablation,
   - low-weight auxiliary source,
   - or later-stage stress test.

---

## 4. Official Experiment Design

This is the proposed next matrix.

### 4.1 Stage A: Single-source anchors

Purpose:
1. Establish clean anchor points before mixture.

Runs:
1. `MS only`
2. `R-PRM only`
3. `PRMBench_Preview only`
4. `PRM800K only` as weak-source control

Expected use:
1. Calibrate source difficulty and source quality.
2. Freeze per-source held-out ranges before any mixing claim.

### 4.2 Stage B: Balanced two-source mixtures

Purpose:
1. Test whether complementary supervision helps benchmark-native performance.

Runs:
1. `MS + R-PRM`
2. `MS + PRMBench_Preview`
3. `R-PRM + PRMBench_Preview`

Rules:
1. Use source-balanced sampling.
2. Cap each source to the same effective contribution.
3. Keep `ranking_only` as the default objective first.

### 4.3 Stage C: Three-source same-family mixture

Purpose:
1. Build the most plausible same-family generalization candidate.

Run:
1. `MS + R-PRM + PRMBench_Preview`

Why this is the main target:
1. `Math-Shepherd` provides strong same-source learnability.
2. `R-PRM` provides strong preference-style training signal.
3. `PRMBench_Preview` is the best currently available benchmark-aligned
   auxiliary source in the repo.

### 4.4 Stage D: Weak-source ablation

Purpose:
1. Test whether `PRM800K` helps, harms, or is neutral when added with low
   influence.

Runs:
1. `MS + R-PRM + PRMBench_Preview + low-weight PRM800K`
2. `MS + low-weight PRM800K`

Rule:
1. Never let `PRM800K` dominate by sample count in the first wave.

### 4.5 Stage E: Staged curriculum

Purpose:
1. Test whether staged training works better than one-shot mixing.

Recommended sequence:
1. warm-start on `Math-Shepherd`
2. continue on balanced `MS + R-PRM`
3. optional final continue on `MS + R-PRM + PRMBench_Preview`

Why:
1. This is more consistent with staged RM-guided post-training practice.
2. It also reduces the chance that weaker/heterogeneous sources destabilize the
   first representation learned from the strongest source.

---

## 5. Evaluation Design

### 5.1 Required metrics

Every candidate should report:
1. source-family held-out:
   - pair accuracy
   - AUC
   - ranking score
2. benchmark-native:
   - `ProcessBench GSM8K`
   - `ProcessBench Math`
   - `PRMBench_Preview`
3. seed stability:
   - mean
   - std
   - worst seed

### 5.2 Additional diagnosis

For mixture runs, also report:
1. per-source held-out metrics
2. source mixture composition
3. source contribution caps / weights
4. whether one source dominates the best checkpoint

### 5.3 Promotion rule

The next candidate for later RL-facing use should satisfy all three:
1. strong same-family held-out
2. non-trivial benchmark-native gains above random
3. acceptable seed stability

If one model wins only on same-source held-out but stays random on benchmarks:
1. it remains a source-family result,
2. but does not become a trustworthy promoted head.

---

## 6. Immediate Repository Work Plan

This is the exact next implementation plan implied by the strategy above.

### 6.1 Implemented bundle ids

The repository now contains these official Phase E mixture bundles:
1. `math_shepherd_r_prm`
2. `math_shepherd_prmbench_preview`
3. `math_shepherd_r_prm_prmbench_preview`
4. `math_shepherd_r_prm_prmbench_preview_prm800k`

### 6.2 Implemented direct experiment groups

These groups are now wired into `scripts/run_phase_e_suite.sh`.

| Stage | Group ID | What it does | Why it exists |
|---|---|---|---|
| A | `E20_STAGEA_MS_ANCHOR_SEED3` | `Math-Shepherd` single-source robust anchor | Freeze the strongest converted-label anchor and provide Stage E warm-start checkpoints |
| A | `E21_STAGEA_RPRM_ANCHOR_SEED3` | `R-PRM` single-source robust anchor | Test whether direct chosen/rejected supervision is a better anchor than converted pairs |
| A | `E22_STAGEA_PRMBENCH_PREVIEW_ANCHOR_SEED3` | `PRMBench_Preview` single-source robust anchor | Measure whether benchmark-aligned preview data can stand on its own |
| A | `E23_STAGEA_PRM800K_CTRL_SEED3` | `PRM800K` single-source control | Keep a weak-source anchor in the same matrix for fair comparison |
| B | `E24_STAGEB_MS_RPRM_MIX_SEED3` | balanced `MS + R-PRM` | Test whether converted + direct preference supervision are complementary |
| B | `E25_STAGEB_MS_PRMBENCH_MIX_SEED3` | balanced `MS + PRMBench_Preview` | Test whether benchmark alignment helps more than generic preference diversity |
| B | `E26_STAGEB_RPRM_PRMBENCH_MIX_SEED3` | balanced `R-PRM + PRMBench_Preview` | Test whether Math-Shepherd is actually necessary as anchor |
| C | `E27_STAGEC_MS_RPRM_PRMBENCH_MIX_SEED3` | balanced three-source tri-mix | Main same-family mixture candidate |
| D | `E28_STAGED_TRIMIX_PLUS_PRM800K_LOWWT_SEED3` | tri-mix plus low-weight `PRM800K` | Weak-source ablation: does `PRM800K` help when not allowed to dominate? |
| D | `E29_STAGED_MS_PLUS_PRM800K_LOWWT_SEED3` | `MS + low-weight PRM800K` | Cleanest weak-source delta relative to the Math-Shepherd anchor |

Common training recipe for these groups:
1. `ranking_only`
2. `learning_rate=2e-5`
3. `pair_weight_mode=none`
4. `anti_saturation_weight=1e-3`
5. `checkpoint_selection_metric=auc`
6. `source_balance=uniform` for Stage B/C/D mixtures

### 6.3 Implemented staged curriculum groups

These are wired into `scripts/run_phase_e_multisource_math_suite.sh`.

| Stage | Curriculum ID | Sequence | Purpose |
|---|---|---|---|
| E | `CUR1_STAGEE_MS_TO_MSRPRM` | `E20 -> E24` | Test whether warm-starting from the Math-Shepherd anchor stabilizes the first balanced two-source mixture |
| E | `CUR2_STAGEE_MS_TO_TRIMIX` | `E20 -> E27` | Test whether direct jump from anchor to tri-mix is better than one-shot tri-mix |
| E | `CUR3_STAGEE_MS_TO_MSRPRM_TO_TRIMIX` | `E20 -> E24 -> E27` | Full staged curriculum matching the preferred literature-guided sequence |

### 6.4 Implemented suite entrypoints

1. Main direct suite:
   - `scripts/run_phase_e_suite.sh`
2. New multi-source wrapper:
   - `scripts/run_phase_e_multisource_math_suite.sh`

Wrapper groups:
1. `MM1_MULTISOURCE_MATH_SMOKE`
   - quick smoke across representative Stage A/B/C/D/E paths
2. `MM2_MULTISOURCE_MATH_STAGE_ABCD_SEED3`
   - official direct Stage A-D matrix
3. `MM3_MULTISOURCE_MATH_STAGEE_CURRICULUM_SEED3`
   - official curriculum matrix
4. `MM4_MULTISOURCE_MATH_FULL_PROGRAM`
   - full overnight bundle

### 6.5 Direct runnable commands

Direct Stage A-D matrix:

```bash
ACTIVE_PHASE_E_MM_GROUP=MM2_MULTISOURCE_MATH_STAGE_ABCD_SEED3 \
RUN_PREFIX=phase_e_multisource_abcd_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_multisource_math_suite.sh
```

Curriculum matrix:

```bash
ACTIVE_PHASE_E_MM_GROUP=MM3_MULTISOURCE_MATH_STAGEE_CURRICULUM_SEED3 \
RUN_PREFIX=phase_e_multisource_curriculum_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_multisource_math_suite.sh
```

Full overnight bundle:

```bash
ACTIVE_PHASE_E_MM_GROUP=MM4_MULTISOURCE_MATH_FULL_PROGRAM \
RUN_PREFIX=phase_e_multisource_full_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_multisource_math_suite.sh
```

Quick smoke:

```bash
ACTIVE_PHASE_E_MM_GROUP=MM1_MULTISOURCE_MATH_SMOKE \
RUN_PREFIX=phase_e_multisource_smoke_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_multisource_math_suite.sh
```

### 6.6 Analysis outputs

The new wrapper renders:
1. one summary table for direct Stage A-D groups
2. one summary table for Stage E curricula
3. explicit links to each sub-suite `final_summary.md`

### 6.7 Latest Result Snapshot (2026-03-10 Late)

This table records what the newest completed or partially completed groups
actually tell us.

| Group | Current result | What question it answered | Current interpretation |
|---|---|---|---|
| `MS2_MATH_SHEPHERD_TRUST_SEED3_MATRIX` | provisional best is `E14`; `mean_hold_pair=0.8246`, `mean_hold_auc=0.7899`, but `pb_gsm_auc=0.4893`, `pb_math_auc=0.4783` | Did the post-fix `Math-Shepherd` trust matrix improve the single-source anchor? | Yes for same-family ranking; no for `ProcessBench` robustness. |
| `E20_STAGEA_MS_ANCHOR_SEED3` | `mean_heldout_pair_acc=0.6853`, `mean_heldout_auc=0.6676`, `mean_prmbench_preview_auc=0.5868`, `mean_processbench_gsm8k_auc=0.4750`, `mean_processbench_math_auc=0.4715` | What is the clean post-fix `Math-Shepherd` anchor? | Best current same-family anchor; still weak on `ProcessBench`. |
| `E21_STAGEA_RPRM_ANCHOR_SEED3` | `mean_heldout_pair_acc=0.4381`, `mean_heldout_auc=0.4953`, `mean_prmbench_preview_auc=0.5623`, `mean_processbench_gsm8k_auc=0.4744`, `mean_processbench_math_auc=0.4626` | Is direct chosen/rejected supervision a better first anchor than converted `Math-Shepherd` pairs? | Not as a general anchor, but it is more aligned with `PRMBench_Preview`. |
| `E21` seed-44 on `PRMBench_Preview` | `pair_acc=0.6001`, `auc=0.5937`, `mean_margin=0.0500`, `median_margin=0.0025` | Is the `R-PRM -> PRMBench_Preview` signal definitely above random? | Yes, moderately above random; but confidence margins remain weak. |
| `E22_STAGEA_PRMBENCH_PREVIEW_ANCHOR_SEED3` | pair artifact prepared; training started; no final group summary yet | Can benchmark-aligned preview data stand on its own? | Still pending. Do not claim success or failure yet. |
| `E24_STAGEB_MS_RPRM_MIX_SEED3` smoke | partial seed-42 only: `PRMBench_Preview auc=0.5207`, `PB-GSM8K auc=0.4682`, `PB-MATH auc=0.3835`; full group failed before completion | Are `Math-Shepherd` and `R-PRM` complementary under a simple balanced mixture? | No positive evidence yet; may even interfere under the current recipe. |
| `CUR1_STAGEE_MS_TO_MSRPRM` | stage-1 warm-start exists; stage-2 is still incomplete | Does staged warm-start stabilize the first two-source mixture? | Not answered yet. Current evidence is insufficient. |

Current scientific read:
1. The post-fix semantic repair was necessary and meaningful.
2. `Math-Shepherd` and `R-PRM` now play different roles:
   - `Math-Shepherd`: strongest same-family anchor
   - `R-PRM`: strongest currently observed `PRMBench_Preview`-aligned source
3. The unresolved problem is no longer:
   - whether either source has any signal,
   but:
   - how to combine them without losing the strengths of each.

---

## 7. Final Scientific Position

The current position is now:

1. Single-source math PRM training is enough to prove **learnability**.
2. It is not enough to prove **trustworthy benchmark-facing verification**.
3. The right next step is not:
   - keep tuning one source forever,
   - or expect direct cross-benchmark transfer.
4. The right next step is:
   - structured same-family multi-source mixture training,
   - with source balancing, benchmark hygiene, and staged evaluation.

---

## References

1. `Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations`
   - https://arxiv.org/abs/2312.08935
2. `Improve Mathematical Reasoning in Language Models by Automated Process Supervision` (`OmegaPRM`)
   - https://arxiv.org/abs/2406.06592
3. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   - https://arxiv.org/abs/2501.07301
4. `R-PRM: Reasoning-Driven Process Reward Modeling`
   - https://arxiv.org/abs/2503.21295
   - https://github.com/NJUNLP/R-PRM
5. `ProcessBench: Identifying Process Errors in Mathematical Reasoning`
   - https://arxiv.org/abs/2412.06559
   - https://github.com/QwenLM/ProcessBench
6. `PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models`
   - https://arxiv.org/abs/2501.03124
   - https://github.com/ssmisya/PRMBench
7. `VersaPRM: Multi-Domain Process Reward Model via Synthetic Reasoning Data`
   - https://arxiv.org/abs/2502.06737
8. `Process Reward Models That Think` (`ThinkPRM`)
   - https://arxiv.org/abs/2504.16828
   - https://github.com/mukhal/thinkprm
9. `Qwen2.5-Math Technical Report`
   - https://arxiv.org/abs/2409.12122
   - https://github.com/QwenLM/Qwen2.5-Math
10. `Free Process Rewards without Process Labels` (`ImplicitPRM`)
   - https://arxiv.org/abs/2412.01981
