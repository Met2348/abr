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

## 0.2 Phase A -> Phase B Handoff Gate (Commit-Time)

Use this as the final pre-Phase-B gate before committing Phase A baseline.

### Baseline Freeze Items

- [ ] Freeze and record StrategyQA baselines in docs:
  - binary-choice decision-quality baseline,
  - freeform end-to-end baseline.
- [ ] Freeze and record GSM8K baselines in docs:
  - direct math baseline,
  - CoT math baseline (with current evaluator semantics).
- [ ] Confirm evaluator/template/decode versions are explicitly listed in baseline notes.

### Reproducibility Gate

- [ ] Run at least one reproducibility pair for StrategyQA direct baseline.
- [ ] Run at least one reproducibility pair for batched StrategyQA baseline.
- [ ] Run at least one reproducibility pair for GSM8K direct baseline.
- [ ] Ensure run diffs show deterministic parity (`changed_samples=0`) where expected.

### Artifact Integrity Gate

- [ ] Every frozen baseline run has:
  - `manifest.json`,
  - `metrics.json`,
  - persisted `console.log`.
- [ ] Metrics include evaluator version information.
- [ ] Prepared artifacts have no duplicate sample IDs.

### Change-Control Gate

- [ ] Mark `scripts/phase_a_*` as frozen reference scripts for early Phase B.
- [ ] Route new BCR/ABR training logic to Phase B modules/scripts only.

---

## 0.3 Phase B Lifecycle Status (Authoritative)

Source of truth:
- `phase_B_plan.md`

Lifecycle checklist:
- [x] `B0` Scope Freeze (completed):
  - PEFT-first path selected,
  - first milestone task and artifact set frozen,
  - evaluation/reporting contract frozen.
- [ ] `B1` Train Pipeline Skeleton
- [ ] `B1` Train Pipeline Skeleton (code implemented; smoke exit gate pending)
- [ ] `B2` Data Contract Wiring
- [ ] `B3` Smoke Stability
- [ ] `B4` Development Tuning
- [ ] `B5` First Official SFT/PEFT Run
- [ ] `B6` Handoff to next stage (value-head/BCR-lite expansion)

Phase B first-run target (frozen in B0):
1. StrategyQA direct-answer training on prepared artifact set:
   - `assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/`
2. Base model:
   - `assets/models/Qwen2.5-7B-Instruct`
3. First path:
   - PEFT (LoRA) default, full SFT fallback only if blocked.

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
    check_env.py
    check_data.py
    preprocess_steps.py
    train_sft.py
    train_value.py
    train_ours_lite.py
    train_ours_rl.py
    eval_all.py
    run_ablation.py
  src/
    ours/
      data/
        schema.py
        registry.py
        loaders.py
        step_builder.py
        corruptions.py
      models/
        lm_backbone.py
        value_head.py
        reasoning_model.py
      router/
        action_space.py
        heuristic_policy.py
        router_model.py
        tss.py
        rewards.py
      losses/
        sft.py
        value_temporal.py
        value_calibration.py
        contrastive.py
      training/
        common.py
        sft_trainer.py
        value_trainer.py
        ours_lite_trainer.py
        ours_rl_trainer.py
      eval/
        accuracy.py
        calibration.py
        corruption_auc.py
        efficiency.py
        reporting.py
      utils/
        seed.py
        logging.py
        io.py
        checkpoint.py
  tests/
    unit/
    integration/
    regression/
  docs/
    method_v1.md
    setup.md
    experiments.md
    troubleshooting.md
  TODO.md
  TODO_ours.md
```

Definition of done:
- [ ] Import check passes: `python -c "import ours"`.
- [ ] Script skeletons exist and parse CLI args.

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

## 7. Phase O3: In-House Baselines (No External Code)

We must build reference baselines ourselves for fair comparison.

### Baseline B0: SFT Only

- [ ] Implement SFT trainer (`sft_trainer.py`).
- [ ] Train/eval on GSM8K smoke split.

### Baseline B1: SFT + Value Temporal Regularization

- [ ] Implement temporal consistency loss in `value_temporal.py`.
- [ ] Add optional stop-gradient target mode.
- [ ] Integrate as weighted auxiliary loss.

### Baseline B2: Fixed Verify Schedule

- [ ] Implement fixed strategy (`gen,gen,gen,ver` pattern).
- [ ] Use as non-learned control baseline.

### Acceptance Criteria

- [ ] B0/B1/B2 all runnable via scripts + configs.
- [ ] Outputs comparable metrics in common report format.

---

## 8. Phase O4: OURS-LITE (Heuristic Router First)

This is the first official implementation of our method family.

### Tasks

- [ ] Define action space in `action_space.py`.
- [ ] Implement heuristic router in `heuristic_policy.py`:
  - uncertainty threshold
  - value-change threshold
  - max-step and stop rules
- [ ] Implement TSS in `tss.py`:
  - choose anchor from prior steps
  - compute verification consistency against anchor
- [ ] Integrate action loop into trainer/inference.

### Acceptance Criteria

- [ ] Action traces are logged per sample.
- [ ] Hard samples use `ver` more frequently than easy samples.
- [ ] Better efficiency-faithfulness tradeoff than B2 on smoke benchmark.

---

## 9. Phase O5: OURS-RL (Learned Router)

Only start after O4 is stable.

### Tasks

- [ ] Implement router model (`router_model.py`).
- [ ] Implement reward module (`rewards.py`):
  - correctness reward
  - verify penalty
  - token/compute penalty
  - optional smoothness term
- [ ] Implement RL trainer (`ours_rl_trainer.py`), start simple.
- [ ] Add constrained optimization mode (verification budget).
- [ ] Add anti-hacking diagnostics (always-gen and always-ver detectors).

### Acceptance Criteria

- [ ] Converges on smoke tasks without policy collapse.
- [ ] Improves at least one frontier point (accuracy vs cost or faithfulness vs cost).
- [ ] Results replicate across at least 2 seeds.

---

## 10. Phase O6: Evaluation System (Core to Paper Credibility)

### Required Metrics

- [ ] Accuracy
- [ ] Calibration (Brier, optional ECE)
- [ ] Corruption detection AUC
- [ ] Value-drop localization
- [ ] Efficiency (tokens, steps, verify count, wall time)

### Tasks

- [ ] Implement metric modules in `src/ours/eval/`.
- [ ] Implement unified reporting (`reporting.py`) producing:
  - JSON for machine parsing
  - Markdown summary table
- [ ] Add `scripts/eval_all.py` for one-command benchmark.

### Acceptance Criteria

- [ ] Identical eval pipeline across all baselines and ours.
- [ ] Report reproducible under fixed seed.

---

## 11. Phase O7: Ablations and Sensitivity

### Required Ablations

- [ ] Router off vs heuristic vs RL
- [ ] TSS on/off
- [ ] Verify penalty sweep
- [ ] Temporal loss weight sweep
- [ ] Dataset transfer (math -> commonsense)

### Tasks

- [ ] Implement experiment matrix configs.
- [ ] Implement `run_ablation.py`.
- [ ] Add summarizer for cross-run comparison.

### Acceptance Criteria

- [ ] Every claim in docs maps to at least one ablation result.
- [ ] All plots/tables can be regenerated from run artifacts.

---

## 12. Testing and CI Policy

### Unit Tests

- [ ] Data adapters
- [ ] Step builder
- [ ] Loss functions
- [ ] Router action logic
- [ ] Metrics correctness on synthetic fixtures

### Integration Tests

- [ ] End-to-end smoke: data -> train -> eval (SFT baseline)
- [ ] End-to-end smoke: ours-lite

### Regression Tests

- [ ] Metric schema compatibility checks
- [ ] Checkpoint load compatibility

### CI Commands

- [ ] `python -m py_compile src scripts`
- [ ] `pytest -q tests/unit`
- [ ] optional nightly integration suite

---

## 13. Run Artifact Contract (Must Persist Per Run)

- [ ] `config.yaml`
- [ ] `metrics.json`
- [ ] `train.log`
- [ ] checkpoint(s)
- [ ] environment summary (package versions + CUDA)
- [ ] random seed(s)

Recommended layout:

```text
runs/
  <run_id>/
    config.yaml
    env.json
    metrics.json
    train.log
    checkpoints/
    report.md
```

---

## 14. Re-Planned Route After Phase A (Actionable)

### Stage B: BCR-Lite Build (Next Priority)

1. [ ] Create `src/ours/models/`:
   - `lm_backbone.py`,
   - `value_head.py`,
   - `reasoning_model.py`.
2. [ ] Create `src/ours/losses/`:
   - `sft.py`,
   - `value_temporal.py` (Bellman-style),
   - optional `value_calibration.py`.
3. [ ] Create `src/ours/training/`:
   - `common.py`,
   - `sft_trainer.py`,
   - `value_trainer.py`.
4. [ ] Add new scripts:
   - `scripts/phase_b_train_sft.py`,
   - `scripts/phase_b_train_bcr_lite.py`,
   - `scripts/phase_b_eval.py`.
5. [ ] Run smoke training on small StrategyQA/GSM8K subsets and verify stability.

### Stage C: Faithfulness Evaluation Completion

1. [ ] Implement calibration metrics (Brier/ECE).
2. [ ] Implement corruption generator + corruption AUC pipeline.
3. [ ] Add a single eval report format for:
   - accuracy,
   - parse/compliance,
   - calibration,
   - corruption sensitivity,
   - compute cost.

### Stage D: ABR-Lite (Heuristic Router)

1. [ ] Implement `gen/ver/fin` action loop with deterministic policy.
2. [ ] Add verification budget controls and traces.
3. [ ] Compare against BCR-lite under fixed compute budgets.

### Stage E: ABR-RL (Final)

1. [ ] Add learned router only after Stage D is stable.
2. [ ] Use constrained reward design to prevent reward hacking.
3. [ ] Require multi-seed replication before claiming gains.

### Go/No-Go Gates

1. Stage B cannot start large runs until smoke runs are stable and reproducible.
2. Stage D cannot start until Stage C metrics are automated.
3. Stage E cannot start until Stage D shows non-trivial frontier gains.

---

## 15. Explicit Out-of-Scope Until O4 Complete

- Complex RL variants and advanced policy algorithms.
- Multi-node/distributed scaling.
- Over-optimization of long-tail metrics without baseline stability.
- Heavy paper polishing before reproducible core results.

---

## 16. Milestone Success Criteria

### MVP Success

- [ ] In-house baselines and ours-lite all run end-to-end.
- [ ] Ours-lite shows better faithfulness signal than SFT-only baseline.
- [ ] Outputs are reproducible and reviewable.

### Extended Success

- [ ] Learned router (ours-RL) beats fixed schedule on one frontier.
- [ ] Multi-dataset evidence supports generality claim.
- [ ] Artifact package is paper-ready (scripts + configs + reports).

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
