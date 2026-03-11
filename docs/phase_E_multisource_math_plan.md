# Phase E Multi-Source Math Mixture Plan

This file defines the next official `Phase E` experimental direction after the
latest `Math-Shepherd` trust runs.

Date baseline: 2026-03-10.

---

## 0C. ProcessBench Alignment Audit And Repair-Pilot Update (2026-03-11)

This section records the first explicit benchmark-mismatch audit after the
single-source `ACC90/95` branch became strong enough to separate
same-source learnability from benchmark transfer.

### 0C.1 What the new audit measured

New script:
1. `scripts/phase_e_analyze_processbench_failures.py`

The audit now compares:
1. training-pair semantics
2. ProcessBench structure
3. bucketed benchmark behavior

This matters because a single benchmark AUC cannot tell us whether a failure is
coming from:
1. weak local first-error discrimination,
2. missing full-completion preference,
3. or broader support mismatch.

### 0C.2 What the audit established

Representative baseline diagnostics:
1. `ms_e68`:
   - `assets/artifacts/phase_e_processbench_analysis/ms_e68_pb_math_v2_0311_20260310T160909Z/summary.md`
2. `prm_e46`:
   - `assets/artifacts/phase_e_processbench_analysis/prm_e46_pb_math_v2_0311_20260310T160909Z/summary.md`

Hard findings:
1. `ProcessBench` has a large `all-correct` block:
   - GSM8K: `0.4825`
   - Math: `0.4060`
2. the strongest current baselines had zero terminal-anchor supervision:
   - `E68` was pure `local_first_bad_edge`
   - `E46` was pure `local_modified_process_error_step`
3. so the old verbal diagnosis is now a measured contract mismatch:
   - strong local supervision does not automatically teach
     `all-correct terminal completion > shorter safe prefix`.

### 0C.3 New targeted repair artifacts

Three benchmark-alignment repair artifacts were built:

1. `PRMBench + terminal anchors`
   - script:
     - `scripts/phase_e_prepare_prmbench_terminal_anchor_pairs.py`
   - artifact:
     - `assets/artifacts/phase_e_pairs/phase_e_prmbench_terminal_anchor_full_0311__192ca71fd301`

2. `Math-Shepherd + terminal anchors` (capped, source-aware)
   - script:
     - `scripts/phase_e_prepare_mathshepherd_terminal_anchor_pairs.py`
   - artifact:
     - `assets/artifacts/phase_e_pairs/phase_e_ms_terminal_anchor_cap20k_diag_0311__6d57b0d4b490`

3. `Math-Shepherd all_good_vs_all_bad` grid
   - artifact:
     - `assets/artifacts/phase_e_pairs/phase_e_ms_grid_cap40k_diag_0311__4f87d4f4cea6`

These are deliberately not the same repair:
1. terminal-anchor variants target the completion side of ProcessBench,
2. the grid variant targets broader good-vs-bad prefix relations.

### 0C.4 Micro repair pilots and what they mean

All three were run as warm-start micro pilots under server-safe settings.
These are direction tests, not promotion candidates.

#### Pilot A: `PRMBench + terminal anchors` from `E46`

Artifacts:
1. train run:
   - `assets/artifacts/phase_e_runs/phase_e_pbta_warm_e46_micro_0311_20260310T162646Z`
2. benchmark eval:
   - `assets/artifacts/phase_e_eval/phase_e_pbta_warm_e46_micro_pb_gsm8k_0311_20260310T163036Z/metrics.json`
   - `assets/artifacts/phase_e_eval/phase_e_pbta_warm_e46_micro_pb_math_0311_20260310T163037Z/metrics.json`

Directional result:
1. terminal completion improved:
   - GSM8K `0.2924 -> 0.4196`
   - Math `0.2452 -> 0.3492`
2. but local/global ranking softened:
   - GSM8K `auc 0.6264 -> 0.6014`
   - Math `auc 0.6053 -> 0.5906`

Reading:
1. the repair is semantically on-target,
2. but terminal anchors alone are not enough.

#### Pilot B: `Math-Shepherd + terminal anchors` from `E68`

Artifacts:
1. train run:
   - `assets/artifacts/phase_e_runs/phase_e_msta_warm_e68_micro_0311_20260310T163102Z`
2. benchmark eval:
   - `assets/artifacts/phase_e_eval/phase_e_msta_warm_e68_micro_pb_gsm8k_0311_20260310T163337Z/metrics.json`
   - `assets/artifacts/phase_e_eval/phase_e_msta_warm_e68_micro_pb_math_0311_20260310T163336Z/metrics.json`

Directional result:
1. terminal completion improved strongly:
   - GSM8K `0.5626 -> 0.7590`
   - Math `0.5895 -> 0.7663`
2. but local / first-edge behavior worsened.

Reading:
1. terminal-anchor pressure is powerful,
2. and by itself it over-corrects.

#### Pilot C: `Math-Shepherd grid` from `E68`

Artifacts:
1. train run:
   - `assets/artifacts/phase_e_runs/phase_e_msgrid_warm_e68_micro_0311_20260310T163400Z`
2. benchmark eval:
   - `assets/artifacts/phase_e_eval/phase_e_msgrid_warm_e68_micro_pb_gsm8k_0311_20260310T163559Z/metrics.json`
   - `assets/artifacts/phase_e_eval/phase_e_msgrid_warm_e68_micro_pb_math_0311_20260310T163559Z/metrics.json`

Directional result:
1. broader prefix discrimination improved or held:
   - GSM8K `pair_acc 0.6385 -> 0.6436`
   - Math `pair_acc 0.5809 -> 0.5839`
2. but terminal completion stayed weak:
   - Math terminal top1 remained near zero in bucket analysis.

Reading:
1. the grid variant helps the good-vs-bad side,
2. but does not solve the completion side.

### 0C.5 Updated operational conclusion

This audit changes the next-repair priority.

The evidence now says:
1. `terminal anchors` and `grid pairs` move different parts of the benchmark,
2. and the real missing recipe is a **mixed or staged repair**, not a single
   universal replacement.

Therefore the next principled experiment is:
1. keep strong local supervision,
2. add a limited amount of terminal anchors,
3. optionally add low-weight grid coverage,
4. and evaluate whether both:
   - `first_error_edge_accuracy`
   - and `all-correct terminal behavior`
   can improve together.

## 0A. Intradataset ACC90 Update

The later single-dataset ACC90 runs changed one assumption behind the mixture
plan.

What changed:
1. `Math-Shepherd` under the fixed semantics is now stronger than we previously
   thought:
   - linear robust recipe already crosses `0.90` held-out pair accuracy;
   - MLP variants push it into the `0.96 ~ 0.99` range.
2. `PRMBench_Preview` behaves differently:
   - linear head underfits materially;
   - MLP is required to cross `0.90`.
3. Therefore a single universal head recipe is no longer the right design
   assumption for future multi-source math training.

Operational implication:
1. the multi-source plan should continue to normalize supervision semantics and
   source weights;
2. but source-specific anchor selection must now remember that head capacity is
   itself source-dependent;
3. in practice:
   - `Math-Shepherd` can still justify a strong linear anchor,
   - `PRMBench_Preview` should default to `MLP`,
   - `R-PRM` still needs fuller same-source evidence before we freeze one
     definitive architecture.

## 0B. Audit-Constrained Reading Of The Current State

The repository now has enough evidence to separate three different questions
that were previously too mixed together:
1. scientific limitation,
2. supervision / semantics limitation,
3. infrastructure-induced result contamination.

### 0B.1 What has actually been established

1. The strongest current positive result is still local same-source
   learnability, not benchmark robustness.
2. Best reviewed same-source evidence in this read:
   - `E14_MATH_SHEPHERD_TRUST_ANTISAT_SEED3`
   - `mean_hold_pair = 0.8246`
   - `mean_hold_auc = 0.7899`
3. Best reviewed benchmark-negative evidence in this read:
   - `CUR1_STAGEE_MS_TO_MSRPRM` on `processbench_math`
   - `pair_accuracy_good_vs_bad = 0.5105`
   - `pair_auc_good_vs_bad = 0.4812`
   - `first_error_edge_accuracy = 0.4593`

Operational meaning:
1. the move to same-family multi-source training is still justified,
2. but no current multi-source result should be narrated as "problem solved"
   until benchmark-native metrics improve materially.

### 0B.2 What the current strategy is good at

The present strategy is strongest as:
1. a diagnosis pipeline,
2. a supervision-semantics cleanup pipeline,
3. and a ranking-first verifier warm-start study.

It is not yet strong evidence for:
1. a robust small PRM,
2. stable benchmark-facing verification,
3. or direct RL / control deployment.

### 0B.3 New reliability constraints from the code audit

Recent audit work found that some failure modes can distort experiment reading
even when training itself completes normally.

High-risk reliability constraints:
1. checkpoint selection:
   - requesting `"best"` can silently fall back to `"final"` if the best
     checkpoint file is missing.
2. feature-cache concurrency:
   - the current stale-lock rule can let a slow live writer be treated as dead
     and overlapped by another writer.
3. `PRMBench_Preview` loader semantics:
   - current code assumes 1-based error-step indices and normalizes by
     subtracting one.
   - this matches the currently audited repo snapshot, but it is still a schema
     assumption rather than an auto-verified contract.

Plan implication:
1. future `Phase E` promotions must distinguish:
   - scientific failure,
   - data-semantics mismatch,
   - and infrastructure-induced metric contamination.

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

### 1.3 What The Literature Does And Does Not Support For Small-Scale Training

This is the specific literature question that matters for the repository now:
1. Is there evidence that a small-scale, from-scratch value head can work?

The answer is: **yes, but only under a narrower formulation than our original
Phase C/D wording implied.**

What the literature does **not** strongly support:
1. small-scale
2. noisy or outcome-only labels
3. scalar value regression
4. step-level general-purpose quality prediction
5. immediate cross-benchmark transfer

This is the weakly supported version of the problem, and it is exactly the
version that most closely matches the old `MC / q_teacher / q_fused` logic.

What the literature **does** support more clearly:
1. large-scale dense process supervision:
   - `Let's Verify Step by Step` / `PRM800K`
   - strong positive evidence, but under very large step-level human labels
2. automated process supervision at scale:
   - `Math-Shepherd`
   - `OmegaPRM`
   - `Rewarding Progress`
   - positive evidence, but with much heavier data construction than a small
     scratch setup
3. smaller-data verifier or ranking-style learning:
   - `ThinkPRM`
   - `R-PRM`
   - these works support the idea that **small(ish) high-quality pair/process
     data can work**, but usually with:
     - a strong backbone
     - ranking / preference or generative verification objectives
     - not plain scalar regression
4. coarse-grained confidence / uncertainty heads:
   - `Language Models (Mostly) Know What They Know`
   - `Large Language Models Must Be Taught to Know What They Don't Know`
   - these give weaker but still relevant support that small feature-based
     uncertainty/value-style heads can work for coarse question-level
     confidence,
   - but they are not direct evidence for fine-grained step-quality PRMs.

Practical reading for this repository:
1. The literature does **not** justify claiming that a small scratch run should
   learn a robust, benchmark-facing scalar process-value function.
2. The literature **does** justify testing a narrower claim:
   - a small-to-medium scale, same-family, ranking-first verifier/value head
     can learn useful local process discrimination.
3. Therefore the scientifically safe framing is:
   - not "general-purpose small PRM/value head"
   - but:
   - "same-family ranking verifier warm-start" or
   - "local first-bad-edge discriminator".

### 1.4 Joint Reading Of Literature And Our Newest Results

Our current repository evidence is consistent with that literature boundary.

1. `MS2` and `E20`:
   - clearly support **same-family local learnability**
   - do not support benchmark-robust verification
2. `E21`:
   - supports the narrower claim that direct preference-style supervision can
     align with a benchmark-adjacent preview source
   - but still does not support general benchmark robustness
3. `E24`:
   - currently provides no evidence that simple source concatenation solves the
     gap
4. therefore:
   - our current evidence matches the literature pattern:
     - local ranking/verifier learning is plausible,
     - small scalar value generalization is not established,
     - stronger judge/verifier-style supervision remains the dominant SOTA
       pattern.

Operational conclusion:
1. The repository should continue to evaluate small-scale training,
2. but only under the narrower scientific claim that the literature actually
   supports.
3. If we later want a stronger headline claim, we likely need one of:
   - stronger judge-generated supervision,
   - a generative verifier formulation,
   - or a much larger process-label pipeline.

### 1.5 What The Literature Suggests About MCTS For The Current Bottleneck

This is the next natural question after the latest `ProcessBench` mismatch
audit:
1. can `MCTS` directly rescue the current Phase E verifier/value-head problem?

Short answer:
1. **not as the next mainline fix**,
2. but **possibly yes as a later data-construction or test-time branch**.

Why the answer is not simply "add MCTS":
1. our current measured bottleneck is a **supervision contract mismatch**:
   - local-only supervision under-teaches terminal completion,
   - terminal-anchor supervision over-corrects,
   - grid supervision helps a different side of the benchmark.
2. `MCTS` does not remove that mismatch by itself.
3. if the node reward / verifier signal is already misaligned, tree search will
   usually amplify that weakness rather than fix it.

What the literature does support:
1. `ReST-MCTS*`
   - uses search to collect better traces and per-step values for later
     training,
   - so this is evidence for **offline tree-based data construction** rather
     than a cheap direct repair of an existing weak scalar head.
2. `Tree-PLV`
   - uses search/tree expansion to construct step-level paired preferences,
   - which is much closer to our current need for better `local + terminal`
     pair construction.
3. `Rewarding Progress`
   - supports search when the verifier measures **progress / advantage**, not
     just naive absolute correctness,
   - which again suggests that search quality depends on the quality of the
     verifier target.
4. `MCTS-Judge`
   - supports MCTS as a **test-time judge/search** framework,
   - not as proof that our current training mismatch disappears.

What the literature warns against:
1. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   shows that naive MC-estimated synthetic PRM labels often generalize worse
   than stronger judge or human-supervised signals.
2. therefore, replacing the current pipeline with a naive tree-search rollout
   pipeline would likely create a new noisy-label problem instead of solving
   the measured contract mismatch.

Repository-specific reading:
1. The current Phase E problem is:
   - not "we have too little search",
   - but "the training supervision and benchmark semantics are misaligned".
2. Therefore `MCTS` should **not** replace the current `local + terminal +
   optional grid` repair path.
3. The most defensible uses of `MCTS` later would be:
   - offline tree harvesting of higher-margin `first_bad_edge` or
     `terminal-anchor` pairs,
   - or a separate test-time judge/search baseline for comparison against the
     current frozen-head verifier.

Operational conclusion:
1. keep `MCTS` as a **branch experiment**, not the next main Phase E fix;
2. finish the current supervision-alignment repairs first;
3. if we later introduce `MCTS`, use it in one of two narrow forms:
   - data construction for better training pairs / trees,
   - or test-time verifier scaling baseline.

References:
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
5. `Process Reward Models That Think`
   - https://arxiv.org/abs/2504.16828
   - https://github.com/mukhal/thinkprm
6. `MCTS-Judge: Test-Time Scaling in LLM-as-a-Judge for Code Correctness Evaluation`
   - https://arxiv.org/abs/2502.12468

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

### 2.6 Infrastructure-induced metric contamination

We now have explicit evidence that some infra paths can distort experiment
reading without producing an obvious crash.

Examples:
1. `"best"` checkpoint request can silently evaluate the final checkpoint.
2. Feature-cache lock timeout can overlap two live writers.
3. `PRMBench_Preview` step-index normalization is currently snapshot-specific.

Rule:
1. Do not treat benchmark numbers as promotable unless the resolved checkpoint
   path is recorded explicitly.
2. Avoid concurrent writers on the same feature-cache namespace unless the lock
   protocol is strengthened.
3. Re-verify dataset schema assumptions whenever the external dataset file is
   updated.

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

Current wired training recipe for these groups
(descriptive, not yet endorsed as best practice):
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

### 6.8 Value-Head Implementation Audit and Failure Analysis (2026-03-10 Late)

This section records a code-and-artifact audit of the current `Phase E`
value-head setup, so future work does not confuse "head cannot learn" with
"current mixture recipe is flawed."

#### 6.8.1 What the current learner actually is

Code-structure facts:
1. `Phase E` does **not** fine-tune the full transformer backbone in the
   current design.
2. The backbone is run once to produce pooled prefix features, those features
   are cached, and only a small scalar head is trained afterward.
3. The head implementation returns both:
   - raw `logits`
   - `sigmoid(logits)` scores in `[0, 1]`
4. The historical run artifacts inspected here only record the minimal
   `value_head_config.json` payload
   (`hidden_size`, `dropout_prob`, `init_std`, `pooling`), so the exact
   historical `linear` vs `mlp` branch is not fully recoverable from artifact
   metadata alone.
5. The reliable statement is therefore:
   - these runs belong to the lightweight frozen-feature scalar-head family,
   - not to joint backbone fine-tuning.
6. The active training objective in the failing multi-source run is still
   `ranking_only`.

Operational meaning:
1. A failed run here does **not** immediately imply that the backbone features
   are useless.
2. It can also mean the head/loss/data recipe is mismatched.

#### 6.8.2 Why `Math-Shepherd + R-PRM` is not a trivial mixture

The two sources currently mixed in `E24` are not semantically identical:
1. `Math-Shepherd`
   - is converted into strict local `first_bad_edge` pairs
   - meaning:
     - compare the last clean prefix against the prefix that includes the first
       bad step
2. `R-PRM`
   - is already a direct chosen/rejected preference source
   - meaning:
     - compare two full candidate reasoning traces or trace prefixes already
       expressed as preference supervision

Therefore:
1. `E24` is not just "more process data."
2. It is a mixture of:
   - local first-error supervision
   - direct preference supervision
3. A simple balanced concatenation should be treated as a hard scientific
   setting, not as a default baseline that is expected to work immediately.

#### 6.8.3 Most informative experiment contrast so far

| Run | Training recipe snapshot | Held-out result | Benchmark-facing result | Interpretation |
|---|---|---|---|---|
| `E2_MATH_SHEPHERD_PAIR_LEARN_SEED3` seed-42 | `5358` train pairs; `batch=32`; `epochs=4`; `lr=5e-5`; `pair_weight_mode=confidence`; `source_balance=none`; about `672` optimizer steps | `pair_acc=0.8480`; `auc=0.8218`; `ranking_score=0.8349` | still weak on `ProcessBench` in the trust matrix, but clearly positive on same-family held-out | Strong evidence that the current frozen-feature value-head family can learn a clean same-source ranking task. |
| `E24_STAGEB_MS_RPRM_MIX_SEED3` seed-42 | `1774` train pairs; `batch=128`; `epochs=6`; `lr=2e-5`; `pair_weight_mode=none`; `source_balance=uniform`; about `84` optimizer steps | `pair_acc=0.4381`; `auc=0.4922`; `ranking_score=0.4651`; source margins are weak for both `math_shepherd` and `r_prm` | `PRMBench_Preview auc=0.5207`; partial smoke summary also stayed weak on `PB-GSM8K` and `PB-MATH` | Negative result for the **current Stage B recipe**, but not proof that value-head learning itself has failed. |

What this comparison rules out:
1. The failure is unlikely to be explained by "the head is too powerful and
   obviously overfitting."
2. The failure is also not strong evidence that "frozen-feature value heads
   cannot learn at all," because the single-source anchor already learned
   strongly.

What it does suggest:
1. The current multi-source recipe is likely under-testing or mis-specifying
   the problem.

#### 6.8.4 Main design risks identified by the audit

1. Ranking loss is applied on bounded sigmoid scores instead of directly on
   logits.
   - Practical risk:
     - score geometry is compressed into `[0, 1]`,
     - margins are easier to saturate,
     - the later anti-saturation term looks more like a patch than a clean
       primary design.
2. Mixed-source supervision semantics are not aligned tightly enough.
   - `Math-Shepherd` local first-bad-edge labels and `R-PRM` direct preference
     labels are currently optimized under one shared scalar ranking recipe.
   - This may collapse two different notions of "better reasoning" into one
     insufficiently specified target.
3. The current failing mixture looks under-trained in effective optimizer-step
   budget.
   - `E24` seed-42 only receives about `84` optimizer steps,
   - while the strong `Math-Shepherd` run receives about `672`.
   - Therefore "6 epochs" here is misleading; the useful update budget is still
     small.
4. Mixed-source runs currently disable the weighting mechanisms that should be
   most useful in heterogeneous pools.
   - `pair_weight_mode=none`
   - `source_balance=uniform`
   - This discards confidence structure that the code already preserves.
5. Train/validation splitting is currently pair-id based, not
   sample/problem-id based.
   - For sources that can emit multiple related pairs from one underlying raw
     example, this is weaker than a true problem-level split.
6. Head capacity might still be a secondary limitation for heterogeneous
   mixtures.
   - But current evidence is not strong enough to blame head capacity first.
   - Objective/data mismatch is the higher-priority suspect.
7. Checkpoint selection can become misaligned with the training objective.
   - The weak mixture run selects by `auc`,
   - while the stronger single-source run selected by `ranking_score`.
   - This is probably secondary, but it can matter on fragile runs.

#### 6.8.5 Scientific conclusion for the current mainline

The main conclusion is:
1. We do **not** currently have evidence that the `Phase E` value-head design
   is intrinsically incapable of learning.
2. We **do** have evidence that:
   - single-source `Math-Shepherd` learnability is real,
   - the current `MS + R-PRM` Stage B recipe is not yet a trustworthy test of
     source complementarity.

Therefore the next repair priority should be:
1. fix the pairwise objective before blaming the backbone,
2. tighten source-semantic handling before scaling mixture size,
3. increase effective optimizer steps for mixture runs,
4. restore confidence/source weighting where appropriate,
5. only then revisit head-capacity changes as a first-order research question.

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
5. However, the newest audit adds an important refinement:
   - the current negative `MS + R-PRM` result should be treated as a failure of
     the present mixture recipe,
   - not as final evidence against the value-head family itself.

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
11. `Let's Verify Step by Step`
   - https://arxiv.org/abs/2305.20050
12. `Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning`
   - https://arxiv.org/abs/2410.08146
13. `Language Models (Mostly) Know What They Know`
   - https://arxiv.org/abs/2207.05221
14. `Large Language Models Must Be Taught to Know What They Don't Know`
   - https://arxiv.org/abs/2406.08391

## 6. ProcessBench-Oriented Hybrid Pilot (2026-03-11)

### 6.1 Why this pilot was added

The latest literature reading sharpened the current diagnosis:

1. `ProcessBench`-style transfer usually fails because supervision geometry is misaligned,
   not because same-source learning is impossible.
2. Prior work supports keeping strong local/process supervision as the core,
   while adding broader signals conservatively.
3. Therefore the next repair was:
   - keep `PRMBench` local error-step pairs as the anchor,
   - add only a very small terminal-completion auxiliary,
   - compare `mlp` vs `gated_mlp`.

Relevant external references that informed this design:

1. `ProcessBench`
   - https://arxiv.org/abs/2412.06559
2. `PRMBench`
   - https://arxiv.org/abs/2501.03124
3. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   - https://arxiv.org/abs/2501.07301
4. `Rewarding Progress`
   - https://arxiv.org/abs/2410.08146
5. `Tree-PLV`
   - https://arxiv.org/abs/2407.00390

### 6.2 New artifact and code

Artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_pbhybrid_ph1_0311_1230_pairs__d6fb5a3ec28c`

Code:

1. `scripts/phase_e_mix_pair_artifacts.py`
   - supports `MIX_WEIGHT`
2. `scripts/run_phase_e_processbench_hybrid_suite.sh`
   - new benchmark-oriented wrapper
   - hardened during this round:
     - fixed baseline triplet parsing
     - removed shell backtick hazards
     - separated logs from helper outputs
     - switched helper outputs to global result variables

### 6.3 Hybrid composition

Configured caps:

1. `prm_local`
   - train `3072`
   - val `384`
2. `prm_terminal`
   - train `512`
   - val `64`

Important semantic reading from later failure analysis:

1. the final artifact still contains only:
   - `132 / 3584 = 3.68%`
   terminal anchors
2. so this pilot already tested a *small* terminal ratio.

### 6.4 Same-source held-out fit

1. `mlp`
   - run: `assets/artifacts/phase_e_runs/phase_e_pbhybrid_ph1_0311_1230_ph1_prm_local_ta15_arch_sweep_smoke_mlp_20260311T043055Z`
   - held-out:
     - `pair_acc=0.919643`
     - `auc=0.892518`
2. `gated_mlp`
   - run: `assets/artifacts/phase_e_runs/phase_e_pbhybrid_ph1_0311_1230_ph1_prm_local_ta15_arch_sweep_smoke_gated_mlp_20260311T045211Z`
   - held-out:
     - `pair_acc=0.912946`
     - `auc=0.871079`

Reading:

1. hybrid same-source fit is strong.
2. therefore the failure is again on benchmark transfer, not source-family learnability.

### 6.5 Benchmark transfer result

Transfer compare tables:

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

### 6.6 Failure pattern

Failure-analysis outputs:

1. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_mlp_gsm_diag_20260311T045406Z/summary.md`
2. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_gated_gsm_diag_20260311T045406Z/summary.md`
3. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_mlp_math_diag_20260311T045406Z/summary.md`
4. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_gated_math_diag_20260311T045406Z/summary.md`

Key reading:

1. `ProcessBench` still contains a large `all_correct` block:
   - GSM8K `0.4844`
   - Math `0.4062`
2. even a `3.68%` terminal auxiliary is enough to massively raise terminal slices.
3. but it simultaneously damages:
   - `pair_auc_good_vs_bad`
   - `first_error_edge_accuracy`
4. `gated_mlp` changes the tradeoff slightly but does not remove it.

### 6.7 Updated diagnosis

This pilot strengthens the current main diagnosis:

1. the main blocker is still supervision geometry mismatch,
   not insufficient head capacity.
2. `local + tiny terminal` naive mixing is still too blunt.
3. next mainline should be one of:
   - even smaller terminal ratios,
   - benchmark-aware checkpoint selection,
   - or staged / two-objective training where terminal repair is constrained.
4. what should **not** be concluded:
   - "we just need more terminal anchors"
   - or "gated_mlp alone fixes ProcessBench transfer"

## 6.8 Tiny-Terminal Follow-up Pilot (2026-03-11)

### 6.8.1 Why this follow-up was necessary

The first hybrid pilot already showed that a small terminal auxiliary can cause a very strong tradeoff:

1. terminal-completion slices improve sharply,
2. but local ProcessBench discrimination drops sharply.

The next question was:

1. is the problem simply that the terminal ratio is still too large,
2. or is terminal supervision itself something that should not be naively mixed into the same ranking pool?

### 6.8.2 New tiny-terminal artifact

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

### 6.8.3 Same-source fit

Run:

1. `assets/artifacts/phase_e_runs/phase_e_pbhybrid_tinyterm_0311_mlp_20260311T050119Z`

Held-out:

1. `pair_acc=0.918367`
2. `auc=0.890293`

Reading:

1. reducing terminal support to a tiny level does not harm same-source learning.

### 6.8.4 ProcessBench transfer

Eval artifacts:

1. GSM8K:
   - `assets/artifacts/phase_e_eval/phase_e_pbhybrid_tinyterm_0311_mlp_processbench_gsm8k_20260311T051607Z/metrics.json`
2. Math:
   - `assets/artifacts/phase_e_eval/phase_e_pbhybrid_tinyterm_0311_mlp_processbench_math_20260311T051607Z/metrics.json`
3. compare tables:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_pbhybrid_terminal_ratio_gsm_compare_20260311T051627Z/summary.md`
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

### 6.8.5 Failure-analysis reading

Artifact:

1. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_tinyterm_math_diag_20260311T051628Z/summary.md`

Key fact:

1. the true terminal-anchor semantics are now only:
   - `17 / 3136 = 0.54%`

Reading:

1. shrinking terminal ratio from `3.68%` to `0.54%` does soften the over-correction.
2. terminal `top1` drops from `0.71` to `0.65` on ProcessBench Math.
3. `first_edge` recovers slightly (`0.4921 -> 0.5238`).
4. but overall benchmark ranking is still far below baseline `E46`.

### 6.8.6 Updated conclusion after the tiny-terminal test

This follow-up is important because it sharpens the diagnosis:

1. terminal ratio does matter.
2. smaller terminal support is strictly better than the earlier `ta15` recipe.
3. however, even `0.54%` terminal semantics still do not recover the benchmark gap.

So the current best reading is:

1. the problem is **not only** that terminal ratio was too large,
2. the deeper problem is that terminal supervision should probably not be naively merged into the same ranking objective.
3. next mainline should therefore prioritize:
   - staged training,
   - two-objective training,
   - or benchmark-aware selection,
   rather than a pure terminal-ratio sweep alone.
