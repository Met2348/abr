# Phase E: LoRA and Advanced Techniques Research Notes

**Date**: 2026-03-12
**Status**: Implementation complete; smoke experiments running

---

## 1. LoRA Backbone Fine-tuning

### Motivation

Frozen-backbone Phase E ceiling: **MATH AUC=0.749** (DPO 8K + gated_mlp, seed 42).
Community evidence suggests LoRA adds +5–10 AUC points over frozen feature extraction.

### Implementation (2026-03-12)

LoRA was implemented across three files:

- **`src/ours/phase_e/runtime.py`**: `apply_lora_to_backbone()` function
  - Uses PEFT 0.18.x `LoraConfig` + `get_peft_model`
  - Default target modules: `q_proj, v_proj`
  - `num_top_layers` parameter: attach only to top-N layers (memory saving)
  - **Gradient checkpointing enabled automatically** — critical for memory efficiency

- **`src/ours/phase_e/training.py`**: Two new functions
  - `build_pair_tokenized_cache()`: Tokenizes pairs → raw input_ids (no backbone)
  - `encode_tokenized_cache_with_backbone()`: Encodes tokenized cache with current backbone state

- **`scripts/phase_e_train_value.py`**: New args + LoRA-aware training loop
  - `--lora-rank`, `--lora-alpha`, `--lora-target-modules`, `--lora-num-top-layers`
  - `_run_one_epoch_lora()`: Per-batch backbone forward → gradients flow through LoRA layers
  - Eval cache rebuilt from current backbone at end of each epoch
  - Feature disk cache disabled for training (backbone changes each step)

### Key Design Insight: Why Gradient Checkpointing is Required

Without gradient checkpointing, LoRA backward pass through a 7B model requires:
```
28 layers × batch=4 × seq=1024 × hidden=3584 × bfloat16 = ~784 MB activation storage (hidden states only)
+ attention matrices + MLP activations ≈ 3-5× multiplier
+ optimizer states (Adam 2nd moment) for 7.6B params ≈ 14 GB
= Total: ~55-65 GB on top of the 14 GB model → exceeds A100-80GB
```

With gradient checkpointing (`use_reentrant=False`):
- Intermediate activations are NOT stored during forward pass
- During backward, the activations are **recomputed** on demand for each layer
- Memory: ~14 GB (model) + ~4 GB (recomputed activations at any one time) + ~200 MB (LoRA grads) = ~20 GB total
- Cost: ~30% slower due to double forward passes

**Observed in practice**: `CUDA_VISIBLE_DEVICES=2` with batch=4 runs at ~20 GB VRAM (within A100-80GB budget). Without checkpointing: OOM at batch=4.

### LoRA Smoke Experiment Configurations

| Config | rank | alpha | modules | top_layers | LR | Expected memory |
|---|---:|---:|---|---:|---|---|
| LORA_S1 | 8 | 16 | q+v | all | 1e-4 | ~18 GB |
| LORA_S2 | 16 | 32 | q+v+k+o | top-8 | 1e-4 | ~20 GB |
| LORA_S3 | 32 | 64 | q+v+k+o | all | 5e-5 | ~22 GB |
| LORA_S4 | 16 | 32 | q+v | all | 3e-5 | ~20 GB |

LORA_S4 is the cleanest ablation vs frozen SOTA (same LR=3e-5, same recipe).

### Expected Results (Community Benchmarks)

Based on LoRA literature for PRM-style tasks:
- LoRA (r=16) typically adds **+5–10 AUC points** over frozen backbone
- Full fine-tune (SFT) on DPO data adds ~15–20 AUC, but risks forgetting step-quality signals
- Frozen-feature approach is simpler but represents a significant upper bound before LoRA
- Key risk: **overfitting** — DPO 8K is small for LoRA; need careful LR and dropout tuning

### Run Command

```bash
ACTIVE_LORA_GROUP=LORA_S4_RANK16_QV_CONSERVATIVE \
CUDA_DEVICE=<free_gpu> \
PYTHON_BIN=/home/zling/anaconda3/envs/bcr/bin/python3 \
  bash scripts/run_phase_e_lora_smoke.sh
```

---

## 2. Contrastive Loss on Hidden States (Scale AI Technique)

### Overview (from internet research)

Paper: Scale AI blog post and related work on PRM fine-tuning.
Technique: Add a **contrastive loss** on hidden state representations alongside the usual ranking loss.

**Reported improvement**: +0.09 AUROC on MATH and GSM8K step-level verification.

### Mechanism

```
Loss = L_ranking + lambda_contrast * L_contrastive
L_contrastive = InfoNCE or SimCSE-style loss:
  - Positive pairs: chosen steps from same solution
  - Negative pairs: error steps from different solutions
  - Objective: pull "good step" representations together, push "error step" apart
```

### Implementation Status

**Not yet implemented**. `training.py` already imports `contrastive_margin_loss` from `phase_b.value_losses` but it's not exposed as a training objective.

**To implement**:
1. Add `--contrastive-loss-weight` arg to `phase_e_train_value.py`
2. Modify `compute_pair_objective()` to optionally apply contrastive term on hidden features
3. Need to pass hidden features (pre-value-head) instead of just logits

**Priority**: Medium. Currently frozen-backbone SOTA is 0.749; contrastive loss would require modifying the feature→head pipeline. Better to first establish LoRA baseline, then add contrastive on top.

---

## 3. BiRM: Balanced Process Reward Modeling

### Overview (arXiv:2503.04618)

BiRM addresses the **terminal/local tradeoff** problem:
- Pure local training: good step discrimination but can't calibrate "all correct" solutions
- Pure terminal (outcome) training: calibrates final answers but weakens step discrimination

**BiRM solution**: Joint training with:
- Local pairs (same prefix, next step differs)
- Terminal completion anchors (chosen=correct completion, rejected=wrong completion)
- `c=1.0` balance factor between local and terminal losses

**Key finding from BiRM paper**:
- Soft MC labels (probability, not hard 0/1) for terminal BCE avoid instability
- Hard labels: terminal BCE can saturate and destabilize local pair gradients
- Optimal terminal ratio: 10–20% of total pairs

### Current Status in our Implementation

Terminal BCE is already implemented (`--terminal-bce-lambda`). BiRM-style soft labels are not.

**Phase E results at various terminal ratios**:
- 0% terminal: good step separation, but all_correct_last < good_prefix (not calibrated)
- 50% terminal: pair_acc collapses to 0.43 (too many low-signal terminal pairs)
- **10–20% terminal**: optimal tradeoff (from ratio sweep experiments)

---

## 4. Math-Step-DPO: Score Inversion Root Cause

### Summary

Math-Step-DPO-10K dataset provides sibling_branch pairs (same prefix, different next step). These pairs correctly calibrate step-level quality because:
1. Chosen/rejected have **equal prefix length** (no length bias)
2. The first token difference is at the next step boundary (not mid-step)
3. Clean "good step > bad step" signal

In contrast, Math-Shepherd fanout/grid pairs can introduce length bias:
- Rejected chains are often longer (more branching steps)
- Training the model to prefer shorter chains → score inversion on ProcessBench
- ProcessBench bad_prefix is always longer than good_prefix → model assigns lower score to bad_prefix due to length, not quality

**Conclusion**: DPO sibling_branch pairs are the correct signal source for ProcessBench transfer. Math-Shepherd pairs require careful filtering.

---

## 5. FLB2 Experiment Strategy

### FLB2A: Full-Scale DPO (9,687 pairs)

Goal: Test if more DPO data pushes MATH AUC from 0.749 toward 0.76+.
Config: gated_mlp, 3 seeds (42, 1, 7).
Expectation: moderate improvement (+1–3 AUC) due to additional training signal.

### FLB2B: DPO 8K, Seeds 1 and 7

Goal: Confirm that gated_mlp's advantage over MLP is consistent across seeds.
Context: Fix_C gated (s42) = 0.749, Fix_C MLP (s7) = 0.737. Is the 0.012 gap real?
Expectation: seeds 1 and 7 with gated_mlp should both exceed 0.737.

### FLB2_BACKBONE_SMOKE: Alternative Backbones

Testing whether specialized math backbones improve feature quality:

| Backbone | Pretraining | Expected effect |
|---|---|---|
| Qwen2.5-7B-Instruct | General instruction | Current SOTA (0.749) |
| Qwen2.5-Math-7B-Instruct | Math-domain instruction | +2–5 AUC (math-specific features) |
| Qwen2.5-Math-PRM-7B | PRM-specific training | +10–20 AUC (already trained for step quality) |

Qwen2.5-Math-PRM-7B is the most promising: it already achieves ProcessBench F1=73.5% with its own head, vs our frozen-backbone F1≈40%.

---

## 6. ProcessBench F1 as Primary Metric

**Why pair_auc_good_vs_bad is insufficient**:
- AUC measures relative ranking only
- ProcessBench's actual task is: "find the first error step in a chain"
- F1 measures both precision and recall of error detection across an actual threshold

**Formulae**:
```
F1 = 2 × Acc_error × Acc_correct / (Acc_error + Acc_correct)
Acc_error = fraction of error chains where first_bad_step was detected
Acc_correct = fraction of correct chains that were NOT flagged as errors
```

**Goal**: ProcessBench F1 >= 0.60 (community baseline: published models at 0.60–0.73)
**Current**: F1 not yet measured on FLB1/FLB2 runs (eval script adds it automatically via `compute_processbench_f1()`).

---

## 7. Recommended Next Steps After FLB2

1. **LoRA smoke results** → if +5 AUC, launch full LoRA sweep (ranks 8/16/32, target_modules, seeds)
2. **Backbone comparison** → if Math-PRM-7B gives +10 AUC, pivot all experiments to that backbone
3. **Contrastive loss** → add after LoRA baseline is established, test +lambda_contrastive experiment
4. **Full 1000-sample ProcessBench eval** → current evals use 256 samples; official uses 500/3000
5. **RL preparation** → once Phase E frozen/LoRA SOTA is ≥0.80 MATH AUC, begin RL experiments
