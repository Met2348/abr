# TODO (OURS): Independent ABR-Style Research Codebase

This plan assumes **no access to external BCR code** and treats the project as a fully independent implementation effort.
It is explicit, test-driven, and structured for senior engineering review.

---

## 0. Project Positioning

### Objective

Build and validate **our own method** (step-level adaptive reasoning + value consistency), inspired by but not dependent on external BCR implementation.

### Hard Constraints

- No reliance on unreleased BCR repository.
- No hidden assumptions about external training scripts/checkpoints.
- All claims must be supported by our own reproducible experiments.

### Primary Deliverables

1. End-to-end training/evaluation pipeline from raw datasets to reports.
2. Baseline systems implemented in-house for fair comparison.
3. Our method implementation (ABR-style: `gen/ver/fin` with step-level verification).
4. Reproducible faithfulness + efficiency + accuracy results.

---

## 0.1 Phase A Closeout Status (2026-02-26)

### What Is Done

1. Data and preprocessing foundation:
   - canonical schema/loaders,
   - deterministic step builder,
   - preprocessing artifacts and tests.
2. Phase A benchmark harness:
   - prompt preparation,
   - inference + evaluation,
   - grouped one-click benchmark suites.
3. Diagnosis milestones:
   - parse-error root cause split established,
   - binary-choice track isolates decision quality from free-form format noise.

### What Is Not Done Yet

1. No training loop for value head / BCR losses yet.
2. No ABR router training or verification policy yet.
3. No full faithfulness metric suite (calibration/corruption AUC) wired into training-time evaluations.

### Phase A Freeze Rule

Keep `scripts/phase_a_*` as benchmark references.
New BCR/ABR implementation work should go to new Phase B+ scripts/modules.

---

## 0.2 Current Real Priorities (2026-03-03, Phase D Kickoff)

This is the operational priority list now. Earlier generic checklists below should
be read in this order.

1. Freeze Phase B PEFT conclusions:
   - StrategyQA PEFT is stable and positive.
   - GSM8K full long-CoT PEFT suffers late-run drift and should use checkpoint selection, not final-checkpoint reporting.
2. Phase D (official active track): external-PRM-supported value supervision.
   - add teacher sidecar scoring on existing C1 artifacts,
   - fuse `q_mc` and `q_teacher` into `q_fused`,
   - re-train C2 with explicit target-source ablations (`mc`, `teacher`, `fused`).
3. Promote only if both gates pass:
   - calibration beats trivial baseline reproducibly,
   - corruption ordering becomes clearly above random.
4. Only after promotion gate:
   - restart BCR-lite (`L_sft + lambda_B * L_Bellman`),
   - then ABR-lite router,
   - then router-only RL.

Immediate no-go rules:
1. No router RL on GSM8K first.
2. No joint LM/value/router RL as the first RL experiment.
3. No naive clipped short-CoT supervision reuse for GSM8K.
4. No final-checkpoint reporting on GSM8K when checkpoint sweep data exists.

---

## 0.3 Phase C C1/C2 Closeout and Remnants (2026-03-03)

### Newly Completed

1. C1 uncertainty-aware target schema is implemented:
   - `q_mean_smoothed`, `q_std_error`, `q_ci_width`, `q_weight`.
2. C1 label-side pair-quality artifacts are implemented:
   - `corruption_rollout_targets.jsonl`,
   - `pair_quality.jsonl`.
3. C2 now consumes C1 quality fields:
   - q-weighted calibration modes (`q_weight`, `q_weight_parseable`),
   - label-quality pair filters (`label_quality`, `confidence_parseable_label`),
   - label-threshold controls (`delta_q`, `z_delta`, `pair_weight`, pass-gate),
   - optional pair-weighted contrastive loss.
4. Phase C suite groups added for quality-first runs:
   - `C2_STRATEGYQA_QUALITY_FIRST`,
   - `C2_STRATEGYQA_QUALITY_FIRST_FULL`.

### Final Conclusion from Phase C Runs

1. The implementation layer is stable:
   - artifacts, manifests, and suites are reproducible.
2. The model-quality layer is still blocked:
   - best C2 calibration remains inconsistent versus trivial baseline,
   - corruption ordering is near-random in most variants.
3. Loss-side tricks alone are insufficient:
   - objective sweeps changed metrics but did not cross promotion gates.
4. Root issue is now treated as supervision quality, not engineering failure.

### Phase C Remnants (Carried into Phase D)

1. Weak Monte Carlo supervision for fine-grained prefix ranking.
2. Low-margin clean/corrupt pair quality in many prefixes.
3. Calibration-vs-contrastive tradeoff unresolved under current labels.
4. P(IK) branch did not yet prove robust separability.

### Immediate Phase D To-Do

1. Implement teacher sidecar scoring (`Qwen2.5-Math-PRM-7B`) on C1 artifacts.
2. Add C1 fusion fields (`q_teacher`, `q_fused`, disagreement flags).
3. Add C2 target-source switch and run four-way ablation (`mc/teacher/fused`).
4. Promote to BCR-lite/ABR-lite only if gates pass on repeat runs.

---

## 0.4 Phase B Lifecycle Status (Authoritative)

Source of truth:
- `phase_B_plan.md`

Official execution status:
- [x] Project active phase switched to Phase B (`2026-02-28`).
- [x] Phase A scripts are now treated as frozen benchmark references for comparison only.

Lifecycle checklist:
- [x] `B0` Scope Freeze (completed):
  - PEFT-first path selected,
  - first milestone task and artifact set frozen,
  - evaluation/reporting contract frozen.
- [x] `B1` Train Pipeline Skeleton
- [x] `B2` Data Contract Wiring
- [x] `B3` Smoke Stability
- [x] `B4` Development Tuning
- [x] `B5` First Official SFT/PEFT Run
- [ ] `B6` Handoff to next stage (value-head/BCR-lite expansion)

Phase B first-run target (frozen in B0):
1. StrategyQA direct-answer training on prepared artifact set:
   - `assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/`
2. Base model:
   - `assets/models/Qwen2.5-7B-Instruct`
3. First path:
   - PEFT (LoRA) default, full SFT fallback only if blocked.

---

## 0.5 Phase D Execution Checklist (Authoritative, 2026-03-03)

Source of truth:
- `phase_D_plan.md`

Workstream checklist:
- [ ] D0: Teacher environment gate and reproducible smoke scoring
- [x] D1: Add teacher sidecar scoring script (`scripts/phase_c_score_prm_teacher.py`)
- [ ] D2: Add C1 teacher+MC fusion fields and disagreement logging
- [ ] D3: Add C2 target-source switch (`mc/teacher/fused`)
- [ ] D4: Run four-way ablation on StrategyQA smoke then full
- [ ] D5: Evaluate promotion gates and decide whether to resume BCR-lite/ABR-lite

Promotion gates:
1. calibration gate:
   - selected Brier must beat trivial baseline on repeat runs
2. corruption-order gate:
   - pair/AUC metrics must be clearly above random and stable

If either gate fails:
1. increase pair quality (higher-margin pairs, selective rollout top-up),
2. run control teacher ablation before architecture expansion,
3. do not start router RL.

---

## 1. Method Scope Freeze (Do This First)

Before coding, freeze a minimal v1 method to prevent scope drift.

### v1 Method Definition (Locked)

1. Step-level reasoning units, not token-level consistency checking.
2. Base LM with value head `V(h_t)`.
3. Router actions: `gen`, `ver`, `fin`.
4. Verification uses historical anchor selection (TSS-style).
5. No gold-token interpolation reward for intermediate steps (avoid leakage).

### v1 Exclusions

1. Multi-agent orchestration.
2. Complex frequency-domain objectives as primary training signals.
3. Heavy RL complexity before heuristic router baseline is stable.

### Acceptance Criteria

- [ ] One-page method spec exists in `docs/method_v1.md`.
- [ ] Terms/symbols aligned with `idea_formulation.md`.
- [ ] All team members agree on v1 boundaries.

---

## 2. Target Repository Architecture

```text
ref/
  assets/
    datasets/
    models/
  configs/
    data/
    model/
    train/
    eval/
    experiments/
  scripts/
    check_data.py
    preprocess_steps.py
    phase_a_*.py
    phase_b_train_sft.py
    phase_b_eval.py
    phase_b_compare_eval.py
    phase_b_checkpoint_sweep.py
    phase_b_prepare_value_data.py
    phase_b_train_value.py          # next
    phase_b_eval_faithfulness.py    # next
    phase_b_train_bcr_lite.py       # next
    phase_b_run_abr_lite.py         # next
    phase_b_train_abr_rl.py         # later
  src/
    ours/
      data/
        schema.py
        loaders.py
        step_builder.py
      phase_a/
        prompt_builder.py
        evaluator.py
        answer_extraction.py
      phase_b/
        contracts.py
        data.py
        supervision.py
        value_head.py               # next
        value_targets.py
        value_losses.py             # next
        corruptions.py
        faithfulness_eval.py        # next
        action_space.py             # next
        heuristic_router.py         # next
        tss.py                      # next
        router_model.py             # later
        rewards.py                  # later
  tests/
    unit/
  docs/
    method_v1.md
  TODO.md
  TODO_ours.md
```

Definition of done:
- [ ] Import check passes: `python -c "import ours"`.
- [x] Current Phase A/Phase B scripts parse CLI args.
- [ ] Next-phase value / BCR / ABR scripts exist and parse CLI args.

---

## 3. Engineering Rules (Project-Wide)

1. Config-first: all hyperparameters must live in config files.
2. Reproducibility: every run logs seed, config, versions, hardware, outputs.
3. Test gates before scale-up: no large runs before smoke + integration tests pass.
4. Layered rollout: data -> model -> trainer -> eval -> optimization.
5. Fail loud: explicit exceptions with actionable messages.
6. Batching-first mindset: for throughput-sensitive paths, design batch execution first, then add safe fallback paths.
7. Any batching rollout must include dedicated correctness guards (padding, ordering, decode parity, reproducibility checks).

---

## 4. Phase O0: Environment and Infrastructure

### Tasks

- [ ] Create pinned environment file (`requirements.txt` or `pyproject.toml`).
- [ ] Add `scripts/check_env.py` with checks:
  - Python/torch/cuda versions
  - HF authentication
  - dataset/model path existence
  - disk availability
- [ ] Standardize run output directory (`runs/<run_id>/...`).
- [ ] Implement unified logger utility.
- [ ] Implement checkpoint utility with safe atomic writes.

### Acceptance Criteria

- [ ] `python scripts/check_env.py` exits 0.
- [ ] A dummy run writes `config.yaml`, `metrics.json`, `train.log`.

---

## 5. Phase O1: Data and Step Construction

### Tasks

- [ ] Implement canonical sample schema in `src/ours/data/schema.py`.
- [ ] Implement dataset adapters:
  - GSM8K
  - StrategyQA
  - DROP
  - LogiQA
  - ProofWriter
  - BBH
  - Hendrycks Math (fallback source)
- [ ] Implement split normalization.
- [ ] Implement `step_builder.py`:
  - parse CoT into step units
  - deterministic step IDs
  - preserve mapping to original text
- [ ] Implement `scripts/preprocess_steps.py`.
- [ ] Implement corruption generator for faithfulness tests.

### Acceptance Criteria

- [ ] All enabled datasets convert to canonical schema.
- [ ] Stepization is deterministic under fixed seed/config.
- [ ] Corruption pipeline produces valid and traceable perturbed samples.

### Test Gates

- [ ] Unit tests for each adapter.
- [ ] Unit tests for step segmentation edge cases.
- [ ] Integration test for full dataset registry load.

---

## 6. Phase O2: Base Model + Value Stack

### Tasks

- [ ] Implement local model loading (`lm_backbone.py`).
- [ ] Implement value head (`value_head.py`) with bounded output.
- [ ] Implement joint forward object (`reasoning_model.py`):
  - token logits
  - step-level value estimates
- [ ] Add explicit dtype/precision controls (bf16/fp16/fp32).
- [ ] Add gradient checkpointing support.

### Acceptance Criteria

- [ ] Forward pass on mini-batch succeeds without NaN.
- [ ] Value tensor dimensions align with step boundaries.
- [ ] Model save/load roundtrip is lossless for shapes and keys.

### Test Gates

- [ ] Shape and dtype unit tests.
- [ ] Value range tests.

---

## 7. Phase O3: Baseline Layer (Current Reality)

This layer is already largely implemented.

### What Is Done

- [x] Phase A benchmark/eval stack exists and is frozen as reference.
- [x] Phase B SFT/PEFT trainer exists:
  - `scripts/phase_b_train_sft.py`
- [x] Phase B eval bridge exists:
  - `scripts/phase_b_eval.py`
- [x] Before/after gain comparison exists:
  - `scripts/phase_b_compare_eval.py`
- [x] GSM8K checkpoint sweep exists:
  - `scripts/phase_b_checkpoint_sweep.py`
- [x] StrategyQA and GSM8K PEFT baselines have been run and diagnosed.

### What Still Matters Here

- [ ] Freeze one official StrategyQA PEFT checkpoint policy:
  - `rank 32` as best-quality,
  - `rank 16` as efficiency default.
- [ ] Freeze one official GSM8K policy:
  - held-out checkpoint selection instead of final-checkpoint reporting.
- [ ] Record the combined GSM8K repair result after it finishes:
  - `B2_GSM8K_REPAIR_ANSWER_WEIGHTED_CKPT`

---

## 8. Phase O4: Value-Head Bootstrap (Next Priority)

This is the real start of the unique BCR/ABR work.

### Core principle

Do **not** start with Bellman-only self-supervision and do **not** start with RL.
Bootstrap the value head from empirical prefix targets first.

### Why we temporarily switched to P(IK)

Prefix-level C2 runs were stable but showed weak discrimination (often near-random
clean-vs-corrupt ordering). Before adding more complex losses, we inserted a
question-level diagnostic branch:

1. Input is one full question prompt (not a step prefix).
2. Label is empirical success rate from `K` sampled answers.
3. Objective is calibration-first (`BCE`/`MSE` family), no corruption branch by default.

This branch answers a critical gating question:
- can the head learn any confidence signal at all under lower-noise supervision?

### Training order

1. Freeze the LM backbone initially.
2. Build step prefixes from `src/ours/data/step_builder.py`.
3. For each prefix `h_t`, estimate value target with rollout success:
   - sample `K` continuations,
   - compute final-answer correctness rate,
   - use that empirical rate as `v_hat(h_t)`.
4. Add corrupted-prefix contrastive supervision:
   - clean prefix should score higher than minimally corrupted prefix.
5. Only after the value head is non-random, add Bellman-style regularization.

### Concrete implementation tasks

- [x] Add `phase_C_plan.md`
  - Phase C lifecycle, training order, and risk controls are now frozen in one guidance file.
- [x] Add `scripts/phase_b_prepare_value_data.py`
  - deterministic C0/C1 entrypoint for:
    - step sequences,
    - prefixes,
    - corruptions,
    - optional rollout targets.
- [x] Add `src/ours/phase_b/value_head.py`
  - bounded scalar head `V(h_t) in [0,1]`
- [x] Add `src/ours/phase_b/value_targets.py`
  - rollout-based prefix target generation
- [x] Add `src/ours/phase_b/corruptions.py`
  - minimal semantic perturbations on step prefixes
- [x] Add `src/ours/phase_b/value_losses.py`
  - calibration MSE
  - contrastive margin loss
  - Bellman loss with stop-gradient target
- [x] Add `scripts/phase_b_train_value.py`
  - train value head with frozen backbone
- [x] Add `scripts/phase_b_eval_faithfulness.py`
  - calibration
  - corruption AUC
  - value-drop localization
- [x] Add `scripts/run_phase_c_value_suite.sh`
  - one-command C1 train/eval prep + C2 train + C2 standalone eval
- [x] Add `scripts/phase_c_prepare_pik_data.py`
  - question-level C1 artifacts (`questions`, `rollout_predictions`, `pik_targets`)
- [x] Add `src/ours/phase_b/pik_data.py`
  - strict contracts/loaders for P(IK) artifacts and compatibility checks
- [x] Add `scripts/phase_c_train_pik.py`
  - question-level C2 training (calibration-first)
- [x] Add `scripts/phase_c_eval_pik.py`
  - standalone P(IK) checkpoint evaluation
- [x] Add `scripts/run_phase_c_pik_suite.sh`
  - one-command P(IK) lifecycle suite

### Dataset priority

1. StrategyQA first
2. GSM8K second

Reason:
- StrategyQA PEFT backbone is stable.
- GSM8K still has long-CoT drift and checkpoint sensitivity.

### Exit gates

- [ ] Value head beats trivial baselines on calibration.
- [ ] Corruption AUC is clearly above random.
- [ ] Value drops localize around corrupted steps better than untrained value head.

### C2 status snapshot (2026-03-03 runs, StrategyQA)

- Current best `brier_score`: `0.1924` (`strategyqa_value_c2_k8_cal_only_lr1e4_full`)
- K=8 trivial baseline `brier_score`: `0.1394`
- Current best `pearson`: `0.1915`
- Corruption metrics across tested variants:
  - `pair_accuracy`: `0.4707` to `0.5082`
  - `auc_clean_vs_corrupt`: `0.4765` to `0.5460`
- Interpretation:
  - engineering is stable and reproducible,
  - C2 is still below deployment gate for BCR/ABR routing because calibration does not yet beat trivial baseline.

### C2a P(IK) status snapshot (2026-03-03)

- Status:
  - implementation complete and test-validated,
  - ready for smoke and full-suite runs.
- What this gives us:
  - a lower-noise calibration benchmark that isolates head learnability from
    prefix corruption/ranking complexity.
- Promotion criteria back to prefix-heavy branch:
  - [ ] P(IK) `known_auc` clearly above random on held-out eval,
  - [ ] P(IK) Brier meaningfully beats constant baseline,
  - [ ] gains are reproducible across reruns.

### Immediate C2 recovery tasks (ranked top-to-try list)

These tasks are prioritized from highest expected impact to lowest, based on current runs plus external references in `phase_C_plan.md`.

- [ ] `P0` Promote calibration objective from MSE-only to probability-aware calibration:
  - add `BCEWithLogitsLoss` option (optionally mixed with current MSE),
  - add per-example weighting by rollout support/hardness,
  - keep contrastive off in the first pass.
- [ ] `P1` Add post-hoc calibration layer for deployment scoring:
  - temperature scaling on validation logits first,
  - optional isotonic fallback if monotonic nonlinearity improves brier/reliability.
- [ ] `P2` Replace static contrastive lambda sweep with adaptive balancing:
  - implement GradNorm-style or uncertainty-based dynamic weighting for `L_cal` and `L_ctr`,
  - keep contrastive as auxiliary signal instead of dominant objective.
- [ ] `P3` Upgrade C1 supervision quality instead of only expanding quantity:
  - keep K=8, add disagreement-aware target confidence (high-entropy prefixes get lower target weight),
  - prioritize hard negatives where clean/corrupt are currently inseparable.
- [ ] `P4` Add utility-level gate before O5:
  - run best-of-N rerank with value score on held-out StrategyQA,
  - verify value ranking improves final answer quality or sample efficiency.

### Updated gate check for O5 enable

- [ ] `brier_score < 0.1394` on the K=8 held-out eval set
- [ ] `pearson >= 0.20`
- [ ] `auc_clean_vs_corrupt >= 0.60`
- [ ] `pair_accuracy >= 0.55`
- [ ] Positive utility signal on rerank test (best-of-N by value score)

---

## 9. Phase O5: BCR-Lite (Joint LM + Value, No Router RL)

This is the first real BCR-style training baseline.

### Objective

Use:
- `L_sft`
- `+ lambda_B * L_Bellman`
- optionally later:
  - calibration loss
  - corruption contrastive loss

### Implementation tasks

- [ ] Add `scripts/phase_b_train_bcr_lite.py`
- [ ] Add config family for:
  - StrategyQA smoke
  - StrategyQA full
  - GSM8K smoke only after StrategyQA stabilizes
- [ ] Add Bellman target mode with stop-gradient
- [ ] Add optional target-network / EMA support if Bellman training is unstable
- [ ] Add before/after BCR-lite eval bridge using frozen Phase A protocol plus faithfulness metrics

### Comparison set

- [ ] PEFT baseline
- [ ] value-head-only baseline
- [ ] BCR-lite

### Exit gates

- [ ] BCR-lite trains end-to-end on StrategyQA subset without NaN/divergence.
- [ ] BCR-lite preserves answer accuracy within acceptable range.
- [ ] BCR-lite improves at least one faithfulness metric:
  - calibration,
  - corruption AUC,
  - or localization.

---

## 10. Phase O6: ABR-Lite (Heuristic Router, No RL)

This stage adds the unique step-level control behavior without RL instability.

### Action semantics

- `gen`: generate next reasoning step
- `ver`: perform verification against a historical anchor
- `fin`: stop and answer

### Implementation tasks

- [ ] Add `src/ours/phase_b/action_space.py`
- [ ] Add `src/ours/phase_b/heuristic_router.py`
  - uncertainty threshold
  - value-drop threshold
  - max-step rule
  - finish rule
- [ ] Add `src/ours/phase_b/tss.py`
  - target-step / anchor selection
- [ ] Add `scripts/phase_b_run_abr_lite.py`
- [ ] Log action traces per sample
- [ ] Add fixed-verify baseline for comparison

### Exit gates

- [ ] Hard samples trigger `ver` more often than easy samples.
- [ ] Verification rate stays within explicit budget.
- [ ] Heuristic ABR shows a better faithfulness/efficiency frontier than fixed verification schedule.

---

## 11. Phase O7: OURS-RL (Learned Router)

Only start after O6 is stable.

### Non-negotiable training rule

Start with **router-only RL**.
Freeze LM and value head initially.

Do not start with:
- joint LM + value + router RL
- GSM8K as the first RL task

### Implementation tasks

- [ ] Add `src/ours/phase_b/router_model.py`
- [ ] Add `src/ours/phase_b/rewards.py`
  - correctness reward
  - verify penalty
  - token/compute penalty
- [ ] Add `scripts/phase_b_train_abr_rl.py`
- [ ] Start with REINFORCE / simple actor-critic, not heavy PPO by default
- [ ] Add anti-hacking diagnostics:
  - always-gen detector
  - always-ver detector
  - premature-fin detector

### First RL target

- [ ] StrategyQA first
- [ ] GSM8K only after router learning is stable on StrategyQA

### Exit gates

- [ ] Router policy trains without collapse on StrategyQA smoke/full subset.
- [ ] Improves at least one frontier:
  - accuracy vs compute
  - faithfulness vs compute
- [ ] Repeats across at least 2 seeds.

---

## 12. Evaluation System (Unique-Method Credibility)

### Required metrics

- [ ] Accuracy
- [ ] Calibration:
  - Brier
  - optional ECE
- [ ] Corruption detection AUC
- [ ] Value-drop localization
- [ ] Efficiency:
  - tokens
  - reasoning steps
  - verify count
  - wall time

### Tasks

- [ ] Add `src/ours/phase_b/faithfulness_eval.py`
- [ ] Standardize JSON + Markdown reporting for:
  - baseline PEFT
  - value-head-only
  - BCR-lite
  - ABR-lite
  - ABR-RL
- [ ] Add one command that evaluates all methods under the same protocol

### Exit gates

- [ ] Same evaluation interface across all methods
- [ ] Same seeds / decode settings / split definitions across comparisons

---

## 13. Testing and CI Policy

### Must-have unit tests

- [ ] Prefix target generation
- [ ] Corruption generation
- [ ] Value loss functions
- [ ] TSS anchor selection
- [ ] Router action constraints

### Must-have integration tests

- [ ] value-head smoke run
- [ ] BCR-lite smoke run
- [ ] ABR-lite smoke run

### Existing useful checks

- [x] `python -m py_compile` is already used regularly
- [x] targeted Phase A/Phase B tests already exist and should remain green

---

## 14. Real Next-To-Do Sequence

Do these in order.

1. [ ] Finish the combined GSM8K repair run:
   - `B2_GSM8K_REPAIR_ANSWER_WEIGHTED_CKPT`
2. [ ] Freeze final Phase B baseline policy:
   - StrategyQA best PEFT checkpoint
   - GSM8K best-checkpoint policy
3. [x] Implement prefix-artifact generation for value training:
   - step prefixes
   - rollout targets
   - corruption variants
4. [x] Implement `phase_b_train_value.py`
5. [x] Implement `phase_b_eval_faithfulness.py`
6. [ ] Run first StrategyQA value-head smoke experiment
7. [ ] Add Bellman loss and run first BCR-lite smoke experiment
8. [ ] Implement heuristic ABR-lite router
9. [ ] Only after that, implement router RL

If a decision is needed mid-way:
- prefer StrategyQA over GSM8K for the first value/RL experiments
- prefer frozen-backbone value training over joint training
- prefer explicit metrics over “looks plausible” judgment

---

## 15. Explicit Out-of-Scope Until Value Head Is Stable

- Joint LM + value + router RL
- GSM8K-first RL
- Complex PPO-style router optimization
- Multi-agent variants
- Advanced spectral smoothness objectives as primary loss
- Mixed-data RL before single-task router behavior is understood

---

## 16. Milestone Success Criteria

### Next milestone

- [ ] Value-head training works on StrategyQA and produces non-trivial calibration/corruption signals

### BCR-lite milestone

- [ ] BCR-lite preserves answer accuracy while improving at least one faithfulness metric

### ABR-lite milestone

- [ ] Heuristic router beats fixed verification schedule on one efficiency-faithfulness frontier

### RL milestone

- [ ] Learned router beats heuristic router on at least one constrained frontier

---

## 17. Scaling and Throughput Boost Plan (A100 Cluster)

This section is the performance plan before large Phase B sweeps.

Observed runtime context:
1. Inference jobs are slow for CoT variants despite A100 GPUs.
2. Per-process VRAM usage has been observed around `4-15 GiB` on A100 80GB.
3. This indicates memory headroom exists; current bottlenecks are mostly decode throughput and software design.

### 17.1 Bottleneck Diagnosis

Code-level bottlenecks:
1. Single-sample decode loop (`batch_size=1`) in inference script.
2. Token-by-token autoregressive generation with high `max_new_tokens` for CoT.
3. Weak/late stopping criteria; many samples run to cap.
4. Model reload per run/sweep item adds repeated overhead.
5. Single-process-per-run does not leverage all 4 GPUs.

Hardware/runtime bottlenecks:
1. Decoder inference is often memory-bandwidth bound rather than pure FLOP bound.
2. PCIe/CPU offload events can collapse throughput if device map spills.
3. One active GPU with three idle GPUs wastes cluster capacity.

### 17.2 Best Strategy for This Project (Recommended)

Given current phase (research correctness first, low ops complexity, strong reproducibility needs), the best plan is:
1. Keep HF/Transformers path (no immediate engine migration).
2. Add deterministic batching + input bucketing.
3. Shard evaluation data and run one process per GPU (4-way parallel).
4. Tighten stopping and prompt contracts to reduce unnecessary token generation.

Why this is best now:
1. Highest speedup per implementation risk.
2. Preserves metric continuity with current Phase A artifacts.
3. Minimal infrastructure complexity for a small research team.

### 17.3 Phased Acceleration Roadmap

#### Phase S1 (Quick Wins, 1-2 days)

1. [ ] Add stronger stop behavior:
   - explicit stop markers for prompt leakage (`[USER]`, `Human:`).
   - dataset-specific short-output contracts where appropriate.
2. [ ] Add token-budget policy by task/template:
   - direct math: small cap,
   - CoT reasoning: larger cap only when needed.
3. [ ] Add run warnings for cap overuse:
   - high `hit_token_limit_rate` should fail CI/smoke gates.

Target gain:
- 1.2x to 1.6x wall-clock improvement with near-zero methodological risk.

#### Phase S2 (Core Throughput Upgrade, 2-4 days)

1. [x] Implement batched generation in `phase_a_generate_and_eval.py`:
   - new `--batch-size` (default 1),
   - OOM auto-backoff option (`--oom-backoff`),
   - deterministic output ordering and per-sample metadata.
2. [x] Fix decoder-only padding correctness in batch path:
   - force left-padding in batched tokenization,
   - restore tokenizer state after call,
   - regression test for padding + restore behavior.
3. [ ] Add prompt-length bucketing to reduce padding waste.
4. [ ] Keep deterministic ordering and row-index mapping for exact reproducibility.

Target gain:
- 1.5x to 3x depending on sequence lengths and batch-size stability.

#### Phase S3 (Multi-GPU Parallel Evaluation, 2-3 days)

1. [ ] Add dataset sharding utility:
   - split input JSONL into N shards with stable sample-id hashing.
2. [ ] Launch one process per GPU with fixed shard:
   - `CUDA_VISIBLE_DEVICES=0..3`.
3. [ ] Merge shard predictions and run one unified evaluator.

Target gain:
- near-4x throughput scaling at cluster level (minus merge overhead).

#### Phase S4 (Optional Engine Upgrade, later)

1. [ ] Evaluate vLLM backend for inference-only sweeps.
2. [ ] Validate equivalence protocol:
   - same prompts, same seeds/settings,
   - compare answer distribution + metrics drift.
3. [ ] Adopt only if speedup is substantial and metric drift is acceptable.

Target gain:
- potentially large throughput gain for long-sequence decoding.

Risk:
- runtime stack complexity and possible behavioral drift.

### 17.4 Performance Experiment Groups (Planned)

Use dedicated shell param groups (same one-click philosophy):
1. `S1`: stop-rule ablation (same model/settings, different stop strategy).
2. `S2`: batch-size sweep (`1,2,4,8`).
3. `S3`: single-GPU vs 4-GPU sharded wall-clock comparison.
4. `S4`: optional backend comparison (HF vs vLLM) with drift checks.

Each group must report:
1. accuracy metrics (same as baseline),
2. throughput (`sample/s`),
3. tokens/sec if available,
4. peak VRAM usage per process,
5. reproducibility deltas.

### 17.5 Go/No-Go Criteria for Scaling Work

Go:
1. Speedup >= 2x aggregate for CoT-heavy benchmarks.
2. Accuracy drift <= 0.5 percentage points on fixed benchmark set.
3. Reproducibility checks pass on repeat runs.

No-Go:
1. Any optimization that changes core metric semantics without explicit versioning.
2. Any acceleration path that introduces unstable or opaque run artifacts.

---

## 18. Foundation Reliability Hardening Gate (Before Large Phase B Runs)

Reference audit:
- `foundation_reliability_audit.md`

### 18.1 Completed Hardening (P0)

1. [x] Binary-choice generation metadata safety (no branch-unsafe token dependency).
2. [x] Empty raw prediction accepted as parse-error case (no hard evaluator crash).
3. [x] Split typo fail-fast in loader split normalization.
4. [x] Evaluator-version tracking and comparison mismatch caution.
5. [x] Duplicate `sample_id` detection for prepared input JSONL.
6. [x] Regression tests added for above fixes.

### 18.2 Required Before Phase B Scale (P1)

1. [ ] Add adversarial numeric-equivalence fixture pack (should-pass/should-fail).
2. [ ] Add artifact schema-version validator module and fail-fast checks.
3. [ ] Add `scripts/check_foundation_reliability.sh` gate script:
   - critical unit tests,
   - schema checks,
   - evaluator-version presence check.
4. [ ] Document a strict "no-compare-across-evaluator-version" rule in benchmark SOP.

### 18.3 Entry Rule

1. [ ] Do not start large Phase B sweeps until all 18.2 items are complete.
