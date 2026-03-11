# Phase E Research Update — 2026-03-12

**Source**: Background research on state-of-the-art PRM techniques, ProcessBench community numbers, and data efficiency insights.

---

## 1. ProcessBench Community Baselines (Official F1)

The official ProcessBench metric is F1 of erroneous/correct detection accuracy — **not** AUC. Our current AUC=0.749 maps roughly to F1 in the 60–70 range for a 7B model.

| Model | GSM8K F1 | MATH F1 | OlyBench F1 | OmniMATH F1 |
|---|---:|---:|---:|---:|
| Qwen2.5-Math-7B-PRM800K | 68.2 | 62.6 | 50.7 | 44.3 |
| Skywork-PRM-7B | 70.8 | 53.6 | 22.9 | 21.0 |
| RLHFlow-PRM-Mistral-8B | 50.4 | 33.4 | 13.8 | 15.8 |
| R-PRM-7B-DPO | ~70.4 (overall) | — | — | — |
| **Qwen2.5-Math-PRM-7B** | **73.5 (overall)** | — | — | — |
| Qwen2.5-Math-PRM-72B | ~78.3 (overall) | — | — | — |
| EDU-PRM-7B | 95.0 | 88.2 | 76.2 | — |

**Our PBR19 result**: Math F1=0.683, GSM F1=0.778.

**Gap to Qwen2.5-Math-PRM-7B**: ~5 F1 points on MATH. This is achievable; the architecture is the main gap.

**Why Skywork-PRM-7B has high GSM8K but terrible OlyBench/OmniMATH**: It overfits to in-distribution problems. Generalizable PRMs (like Qwen2.5-Math-PRM-7B and EDU-PRM) use consensus filtering + entropy step boundaries.

---

## 2. Data Quality Bottleneck: MC Estimation vs. Consensus Filtering

**Key finding from arXiv:2501.07301**: MC-estimated PRMs (Math-Shepherd-style) show "significantly lower erroneous step localization" than human-annotated or consensus-filtered PRMs. Their diagnosis:
- MC labels teach outcome-based evaluation (does it reach the right final answer?) rather than process evaluation (is this specific step correct?).
- "A significant proportion of minimum scores are concentrated on final answer steps" — PRMs trained on MC labels degrade to ORM behavior.

**Qwen2.5-Math-PRM-7B's advantage**: Consensus filtering — MC estimate AND LLM-as-judge must agree before a label is used. This removes ambiguous/noisy labels.

**Implication for us**: Our `min_pair_confidence=0.55` is an approximation of consensus filtering. The gap between 0.683 and 0.735 F1 is likely primarily a data quality issue, not architecture.

---

## 3. Implicit PRM — New Technique (arXiv:2412.01981)

**Key insight**: Training only an ORM (outcome reward model) with DPO automatically yields a PRM for free via the log-likelihood ratio:
```
implicit_reward(step_t) = β · log[π_θ(step_t | prefix) / π_ref(step_t | prefix)]
```

**Results**: Outperforms Math-Shepherd-PRM-7B using <1/38 the training FLOPs. Training on Math-Shepherd step labels provides no further improvement over implicit rewards.

**Why this matters for us**: This is an alternative to our frozen-backbone PRM. It requires a fine-tuned backbone (the reward is the *change* in log-likelihood relative to reference), which is exactly what our LoRA implementation enables.

**To implement**: After LoRA training stabilizes, extract implicit rewards from the LoRA-tuned backbone (π_θ) vs. the frozen reference (π_ref) as an additional scoring signal or ensemble component.

---

## 4. EDU-PRM — Entropy-Anchored Step Boundaries (arXiv:2503.22233)

**Key insight**: Instead of defining step boundaries at `\n\n` or `Step N:` markers, define them at **high-entropy token positions** — points where the model is most uncertain.

**Why this works**: Each "step" corresponds to a natural uncertainty junction. The PRM signal per token is sharper because boundaries fall at meaningful decision points, not arbitrary whitespace.

**Results**: Achieved MATH accuracy matching Qwen2.5-Math-PRM-72B (88.4%) using only 1.5% of comparable training data.
- 7,500 problems, ~1.42M instances total
- LR=1e-6, 5 epochs, cosine annealing
- Standard cross-entropy on step-end tokens (no exotic loss)

**Applicability for us**:
- Preprocessing change only (no architecture change needed)
- Requires tokenizing solutions and computing token entropy from the backbone
- Our current step boundaries are `\n\n`-based — this would be a significant improvement
- **Priority**: Medium (after LoRA baseline established)

---

## 5. Contrastive Loss — Scale AI Technique (Confirmed Details)

From arXiv (contrastive RM, EMNLP 2024):
- **Gain**: Up to +0.09 AUROC on MATH/GSM8K step-level verification
- **Mechanism**: InfoNCE/SimCSE-style loss on backbone hidden states
  - Positive pairs: hidden states of correct-prefix steps from the same solution
  - Negative pairs: hidden states of error steps from different solutions
  - Applied alongside the standard ranking loss: `L = L_ranking + λ_contrast · L_contrastive`

**Current status in our code**: `contrastive_margin_loss` is imported in `training.py` but not wired up. `--contrastive-loss-weight` arg not yet added.

**To implement**:
1. Pass hidden features (pre-value-head) out of `encode_tokenized_cache_with_backbone()`
2. Add `--contrastive-loss-weight` to `PhaseETrainConfig`
3. Modify `compute_pair_objective()` to apply contrastive term on hidden features

**Priority**: After LoRA baseline. Contrastive on a LoRA-tuned backbone is the most powerful combination.

---

## 6. BiRM Soft Labels — Implementation Update

**BiRM paper (arXiv:2503.04618) specifics**:
- `L_BiRM = L_PRM + 1.0 × L_VM` where L_VM uses **MSE loss** (not BCE) on soft terminal probability labels
- 8 MC rollouts per step for soft value label estimation: `p = fraction_of_rollouts_reaching_correct_answer`
- **Why MSE, not BCE**: BCE with hard 0/1 terminal labels saturates and destabilizes local pair gradients. MSE with soft probabilities avoids batch normalization instability.

**Gap in our implementation**: Our `--terminal-bce-lambda` uses **hard** 0/1 labels (chosen=1, rejected=0). BiRM's finding is that this can destabilize training at high terminal ratios, which matches our observation that 50% terminal mix destroys pair_acc.

**Suggested fix**: Add a `--terminal-bce-soft` flag that uses MC-rollout-derived probabilities as the terminal BCE target instead of hard labels. Since we don't have MC rollouts, use the pair confidence score as a proxy for the soft label.

---

## 7. BiPRM Bidirectional Architecture (arXiv:2508.01682)

New architecture: two heads running L2R (left-to-right) and R2L (right-to-left) over the token sequence, combined.

**Results**: Average +10.6% relative gain over unidirectional PRM across 54 configurations. +37.7% improvement in step-level error detection.

**Complexity**: Requires two forward passes per sample (L2R and R2L). Higher priority than contrastive loss but lower than LoRA.

**Status**: Not planned yet. Architecture change needed in `phase_b/value_head.py`.

---

## 8. Summary: Priority Stack After FLB2/LoRA

| Priority | Technique | Expected Gain | Effort |
|---|---|---|---|
| 1 | LoRA backbone (rank≥8, q+v+k+o) | +5–10 F1 | Done |
| 2 | Data consensus filtering (stricter min_conf or LLM judge) | +3–5 F1 | Medium |
| 3 | Contrastive loss on LoRA hidden states | +0.09 AUROC (~+3 F1) | Low (wiring only) |
| 4 | BiRM soft terminal labels | +2–4 F1 | Low (training change) |
| 5 | Entropy step boundaries (EDU-PRM style) | +3–8 F1 | High (data preprocessing) |
| 6 | Implicit PRM from LoRA-tuned backbone | Unknown | Medium |
| 7 | BiPRM bidirectional architecture | +10% relative | High |

**Target**: ProcessBench MATH F1 ≥ 0.73 (match Qwen2.5-Math-PRM-7B community baseline)
**Stretch**: MATH F1 ≥ 0.80 (using LoRA + contrastive + consensus filtering)
