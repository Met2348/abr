# TODO: BCR / ABR Implementation Roadmap

This document is the execution plan for turning this repo into a complete research codebase.
It is written to be explicit, testable, and reviewable from a senior software engineer perspective.

---

## 0. Project Objective

Build a reproducible codebase for:

1. Baseline SFT reasoning model.
2. BCR training (SFT + Bellman-consistency value regularization).
3. Faithfulness evaluation (calibration, corruption sensitivity, value trajectory diagnostics).
4. Optional ABR extension (adaptive generate/verify/finish routing).

Primary deliverable: clear evidence that BCR (and later ABR) improves faithfulness metrics without unacceptable accuracy or compute regressions.

---

## 1. Engineering Principles (Non-Negotiable)

- Single source of truth for configs; avoid hardcoded hyperparameters in scripts.
- Every experiment run must log: config, seed, model checkpoint, git commit hash (if available), metrics.
- Every module must have at least one smoke test before scaling.
- Add features in layers: data -> model -> trainer -> eval -> optimization.
- No ABR-RL work before BCR baseline is stable and reproducible.

---

## 2. Target Repository Layout

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
    train_sft.py
    train_bcr.py
    eval_bcr.py
    train_abr.py
    run_ablation.py
  src/
    bcr/
      data/
      models/
      losses/
      training/
      router/
      eval/
      utils/
  tests/
    unit/
    integration/
  docs/
    setup.md
    experiments.md
    troubleshooting.md
  TODO.md
  readme.md
```

Definition of done for this section:
- [ ] Directories created.
- [ ] Python package import works: `python -c "import bcr"`.

---

## 3. Phase P0: Environment and Bootstrap

### Tasks

- [ ] Create environment spec (`requirements.txt` or `pyproject.toml`).
- [ ] Pin core libs (transformers, datasets, torch, accelerate, huggingface_hub, evaluate, scikit-learn).
- [ ] Add `scripts/check_env.py` to validate:
  - Python version
  - torch CUDA availability
  - HF auth status
  - cache paths and free disk
- [ ] Ensure dataset downloader is operational (`download_datasets.sh`).
- [ ] Add basic logging utility (`src/bcr/utils/logging.py`).

### Acceptance Criteria

- [ ] `python scripts/check_env.py` exits 0 on target server.
- [ ] `bash download_datasets.sh assets/datasets` runs successfully or clearly reports gated/disabled repos.

### Risks and Controls

- Risk: binary conflicts (pyarrow/datasets).
  - Control: fresh conda env, pinned versions.
- Risk: internet/auth failures.
  - Control: explicit `hf auth whoami` gate in `check_env.py`.

---

## 4. Phase P1: Data Layer (Canonical Schema First)

### Tasks

- [ ] Implement `src/bcr/data/schema.py` with canonical sample schema:
  - `id: str`
  - `dataset: str`
  - `question: str`
  - `answer: str`
  - `cot: str | None`
  - `metadata: dict`
- [ ] Implement dataset adapters in `src/bcr/data/loaders.py`:
  - GSM8K
  - DROP
  - LogiQA
  - StrategyQA
  - ProofWriter
  - BBH
  - Hendrycks Math fallback
- [ ] Implement split normalization (`train/validation/test`) regardless of source naming.
- [ ] Implement deterministic sampling utility (`seed`, `limit`, `stratify`).
- [ ] Implement `scripts/check_data.py`:
  - prints per-split counts
  - prints one normalized sample
  - validates required fields

### Acceptance Criteria

- [ ] All target datasets can be loaded into canonical schema.
- [ ] `scripts/check_data.py` passes for each enabled dataset.
- [ ] Field validation errors are explicit and actionable.

### Test Gates

- [ ] Unit tests for each adapter (`tests/unit/test_data_*.py`).
- [ ] Integration test loading all configured datasets (`tests/integration/test_data_registry.py`).

---

## 5. Phase P2: Model Layer (Base LM + Value Head)

### Tasks

- [ ] Implement `src/bcr/models/base_lm.py`:
  - load local HF causal LM and tokenizer
  - support bf16/fp16 flags
  - support gradient checkpointing
- [ ] Implement `src/bcr/models/value_head.py`:
  - input: hidden state
  - output: scalar value in `[0, 1]` via sigmoid
- [ ] Implement `src/bcr/models/bcr_wrapper.py`:
  - returns token logits + value outputs
  - supports token-level and step-level value extraction modes
- [ ] Add explicit model config dataclass for all hyperparameters.

### Acceptance Criteria

- [ ] Forward pass works on a mini batch.
- [ ] Value output shape matches sequence/step mode.
- [ ] No NaN in logits/value on smoke batch.

### Test Gates

- [ ] Unit tests for tensor shapes and dtype behavior.
- [ ] Unit test for value range bounds.

---

## 6. Phase P3: Trainer V1 (SFT Only Baseline)

### Tasks

- [ ] Implement `src/bcr/losses/sft_loss.py`.
- [ ] Implement `src/bcr/training/sft_trainer.py`:
  - gradient accumulation
  - mixed precision
  - checkpoint save/resume
  - periodic eval hooks
- [ ] Add entry script `scripts/train_sft.py`.
- [ ] Add tiny experiment config (`configs/experiments/sft_smoke.yaml`).

### Acceptance Criteria

- [ ] One full smoke run on GSM8K subset completes.
- [ ] Checkpoint and metrics JSON are written.
- [ ] Resume-from-checkpoint works.

### Quality Bar

- [ ] Training throughput and memory logged each eval interval.
- [ ] Crash-safe checkpointing (atomic save pattern or temp + rename).

---

## 7. Phase P4: BCR Core (Bellman Loss Integration)

### Tasks

- [ ] Implement `src/bcr/losses/bellman_loss.py`.
- [ ] Add optional stop-gradient target mode.
- [ ] Integrate into trainer (`src/bcr/training/bcr_trainer.py`):
  - total loss: `L_sft + lambda_b * L_bellman`
- [ ] Expose loss weights and gamma in config.
- [ ] Add debug metrics:
  - bellman mean/std
  - value drift per sequence
  - ratio `L_bellman / L_sft`

### Acceptance Criteria

- [ ] BCR training runs end-to-end on subset.
- [ ] Bellman loss decreases from initialization (at least in smoke regime).
- [ ] No persistent instability (NaN, exploding gradients).

### Test Gates

- [ ] Unit tests for Bellman target correctness on synthetic sequences.
- [ ] Integration test ensuring gradients flow to value head and shared layers.

---

## 8. Phase P5: Faithfulness Evaluation Framework

### Tasks

- [ ] Implement `src/bcr/eval/answer_eval.py`:
  - exact match / task-specific normalization
- [ ] Implement `src/bcr/eval/value_eval.py`:
  - Brier score
  - optional ECE
- [ ] Implement `src/bcr/data/corruptions.py`:
  - arithmetic sign flip
  - premise deletion
  - substitution noise
- [ ] Implement `src/bcr/eval/corruption_eval.py`:
  - corruption detection AUC
  - value-drop localization around corruption step
- [ ] Implement `src/bcr/eval/reports.py` to output:
  - machine-readable JSON
  - compact markdown report

### Acceptance Criteria

- [ ] One command evaluates SFT and BCR checkpoints on same test set.
- [ ] Report compares accuracy + faithfulness metrics side by side.
- [ ] Corruption pipeline is deterministic given seed.

---

## 9. Phase P6: Multi-Dataset and Ablation Matrix

### Tasks

- [ ] Add dataset registry config (`configs/data/*.yaml`).
- [ ] Add experiment matrix configs:
  - SFT baseline
  - BCR lambda sweep
  - value head variants
  - sequence length variants
- [ ] Implement `scripts/run_ablation.py` runner.
- [ ] Add summary table generator across runs.

### Acceptance Criteria

- [ ] Ablation runner can execute N experiments and consolidate results.
- [ ] All run artifacts include config snapshot and random seed.

---

## 10. Phase P7: ABR-Lite (Heuristic Routing Before RL)

### Tasks

- [ ] Implement router interface (`src/bcr/router/action_policy.py`) with actions:
  - `gen`
  - `ver`
  - `fin`
- [ ] Implement heuristic trigger policy:
  - uncertainty threshold
  - value curvature threshold
- [ ] Implement target-step selection (`src/bcr/router/tss.py`), initially simple.
- [ ] Integrate verify action into inference/training loop.

### Acceptance Criteria

- [ ] Router action traces are logged and inspectable.
- [ ] Verify actions occur more on harder examples than trivial ones.
- [ ] Efficiency/accuracy tradeoff can be plotted.

---

## 11. Phase P8: ABR-RL (Only After Stable ABR-Lite)

### Tasks

- [ ] Implement lightweight router model (`src/bcr/router/router_model.py`).
- [ ] Implement reward module (`src/bcr/router/rewards.py`):
  - correctness reward
  - verify penalty
  - optional token-cost penalty
- [ ] Implement RL trainer (`src/bcr/training/abr_trainer.py`), start simple.
- [ ] Add budget-constrained training mode to reduce reward hacking.

### Acceptance Criteria

- [ ] ABR-RL training converges without collapse.
- [ ] Outperforms fixed schedule baseline on at least one compute-faithfulness frontier point.
- [ ] Reproducible across >= 2 seeds.

---

## 12. Validation, CI, and Test Strategy

### Required Tests

- [ ] Unit tests:
  - data adapters
  - losses
  - model wrapper outputs
  - router policy primitives
- [ ] Integration tests:
  - mini SFT train/eval
  - mini BCR train/eval
- [ ] Regression tests:
  - key metrics do not silently break on code changes

### CI Targets

- [ ] `python -m py_compile` on scripts and `src/`.
- [ ] `pytest -q` on unit tests.
- [ ] Optional nightly integration job.

---

## 13. Experiment Tracking and Artifact Contract

For every run, persist:

- [ ] run id
- [ ] timestamp
- [ ] seed
- [ ] model checkpoint path
- [ ] full resolved config
- [ ] git commit hash (if repo available)
- [ ] train/val/test metrics
- [ ] hardware and CUDA metadata

Suggested output layout:

```text
runs/
  <run_id>/
    config.yaml
    metrics.json
    train.log
    checkpoints/
    report.md
```

---

## 14. Immediate Next 10 Tasks (Actionable This Week)

1. [ ] Create `src/bcr/` package skeleton and config folders.
2. [ ] Implement canonical data schema and dataset registry.
3. [ ] Implement `scripts/check_env.py`.
4. [ ] Implement `scripts/check_data.py`.
5. [ ] Implement base LM wrapper with local model loading.
6. [ ] Implement value head and wrapper forward path.
7. [ ] Implement SFT trainer and `scripts/train_sft.py`.
8. [ ] Implement Bellman loss and `scripts/train_bcr.py`.
9. [ ] Implement minimal eval script (`scripts/eval_bcr.py`) with accuracy + Brier.
10. [ ] Add tests for data schema + Bellman loss + wrapper tensor shapes.

---

## 15. Explicit Out-of-Scope Until P5 Completes

- Large-scale hyperparameter search.
- Complex ABR-RL variants.
- Frequency-domain diagnostics as primary optimization objective.
- Multi-node distributed training.

Reason: these increase complexity and hide baseline correctness issues.

---

## 16. Definition of Success (Milestone-Level)

### MVP Success

- [ ] SFT and BCR both train and evaluate end-to-end on GSM8K.
- [ ] BCR shows improved faithfulness metric on corruption eval with controlled accuracy change.
- [ ] Runs are reproducible and documented.

### Extended Success

- [ ] Multi-dataset validation completed.
- [ ] ABR-lite demonstrates useful efficiency-faithfulness control.
- [ ] ABR-RL shows measurable gain over fixed verification schedule.

