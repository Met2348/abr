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

## 14. Immediate Next Actions (Next 7 Days)

1. [ ] Create `src/ours/` package skeleton and config folders.
2. [ ] Write `docs/method_v1.md` and freeze v1 scope.
3. [ ] Implement `scripts/check_env.py`.
4. [ ] Implement `src/ours/data/schema.py` and `registry.py`.
5. [ ] Implement dataset adapters for GSM8K and StrategyQA first.
6. [ ] Implement `step_builder.py` and deterministic stepization tests.
7. [ ] Implement base model + value head smoke forward.
8. [ ] Implement `train_sft.py` smoke training on GSM8K subset.
9. [ ] Implement temporal consistency loss and `train_ours_lite.py`.
10. [ ] Implement `eval_all.py` with accuracy + Brier + corruption AUC.

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

