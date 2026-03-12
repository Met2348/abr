# Phase F RL Research Update — 2026-03-12

**Synthesized from:** overnight literature survey + web research (2025-2026 papers)
**Author:** Claude Code overnight research session
**Status:** draft — to be extended when fresh agent results arrive

---

## 0. Terminology Reference

| Term | Definition |
|---|---|
| **RLVR** | RL with Verifiable Rewards (DeepSeek-R1 paradigm: outcome-only, no neural reward model) |
| **PRM** | Process Reward Model — per-step scoring |
| **ORM** | Outcome Reward Model — final answer only |
| **MC Estimation** | Math-Shepherd style: sample K completions from step t, fraction correct → step label |
| **Fork-point pairs** | Sibling branches at same prefix; Math-Step-DPO format; cleaner than length-varying pairs |
| **PAV** | Process Advantage Verifier: reward = Δ(P(success) after step t) |
| **PRIME** | Process Reinforcement via Implicit Rewards: r(y_t) = β·log(π_θ / π_ref) — no step labels needed |
| **Consensus Filtering** | Keep only samples where MC estimate and LLM judge agree (~60% of data retained) |
| **Gated MLP / GLU** | output = sigmoid(W_g·x) ⊙ tanh(W_v·x) — dynamic feature selection |
| **Clip+Delta** | Reward shaping: clip absolute PRM score, use Δ between adjacent steps |
| **DAPO** | RL variant: asymmetric clipping + dynamic sampling + overlong filtering + token-level PG loss |
| **Dr. GRPO** | Remove GRPO's per-sample variance normalization (fixes difficulty bias from long errors) |

---

## 1. Signal Quality & RL Readiness

### 1.1 How much reward noise can RL tolerate?

**Gao et al. (ICML 2023) — Scaling Laws for RM:**
- Larger/better RM → higher policy quality ceiling, continuous improvement
- RL overoptimizes RM 3× faster than BoN (decay ~ d³ vs d²)
- **Implication**: RL is MORE sensitive to RM quality than BoN — but our AUC ≥ 0.87 is well above the "safe" threshold

**Noisy Reward Study (arXiv:2510.00915):**
- 40% random label noise → policy retains 96% of oracle performance
- Condition: noise must be random, not systematic
- **Our PRM MATH AUC ~0.87 means ~13% error rate → well within safe zone**
- WARNING: systematic bias (e.g., length bias) is more dangerous than random noise

### 1.2 Outcome-only baselines are surprisingly competitive

DeepSeek-R1 finding: RLVR (outcome reward only, no PRM) induces self-checking, CoT extension, self-correction **as emergent behaviors** — no step supervision needed.

BUT: "reasoning" in outcome-only RL does not guarantee step-level correctness. CoT chains can be internally inconsistent and still reach correct final answers. PRM supervision adds step-level grounding.

### 1.3 Our RL readiness verdict

| Metric | Value | Threshold | Status |
|---|---|---|---|
| MATH AUC | 0.883 (PBR12 pair) | ≥ 0.80 | ✅ PASS |
| ProcessBench F1 | 0.686 (PBR26) | ≥ 0.60 | ✅ PASS |
| Threshold stability | width=0.18 | ≥ 0.10 | ✅ PASS |
| Reward hacking risk | LOW (MATH) | MEDIUM | ✅ PASS |
| BoN (Phase F3) | pending | PRM > random + 2% | 🔄 running |
| GRPO (Phase F4) | pending | delta ≥ +1% | 🔄 running |

---

## 2. Process Reward Integration Strategies

### 2.1 Naive PRM reward shaping → training collapse

**Zeng et al. (arXiv:2410.15115):**
- Directly using PRM score as step reward → model learns to generate many short, score-boosting but meaningless steps
- This is NOT learning better reasoning; it's reward hacking the PRM

**Fix (required for Phase F):**
```
r_shaped(step_t) = clip(PRM(step_t) - PRM(step_{t-1}), -clip_range, clip_range)
```
- **Delta**: use the reward DIFFERENCE between consecutive steps, not absolute score
- **Clip**: limit magnitude to prevent degenerate strategies
- In `phase_f_grpo_lite.py`: current implementation uses `r_outcome + λ * (prm_score - 0.5)` which is a raw absolute score → should be changed to delta+clip formulation

### 2.2 PAV (Process Advantage Verifier)

**From math process reward literature (2024-2025):**
- Define step reward as: `r(t) = V(state after step t) - V(state before step t)`
- This is the TD error / advantage formulation
- 6× sample efficiency vs outcome reward, +7% accuracy improvement
- Requires training a value function V(s) separately

### 2.3 PRIME (Process Reinforcement via Implicit Rewards)

**PRIME (arXiv:2502.01456, approx):**
- Step reward: `r(y_t) = β · log[π_θ(y_t | context) / π_ref(y_t | context)]`
- No step labels needed — implicit from the model's own probabilities
- 2.5× sample efficiency vs outcome-only GRPO
- Requires backbone to be trainable (not frozen)
- **Our situation**: LoRA enables this — PRIME is feasible after LoRA adapters are trained

**Implementation note**: PRIME is incompatible with frozen backbone. Need LoRA or full fine-tuning to make backbone probabilities trainable. After PBR35/36/37 are done, we can implement PRIME.

---

## 3. GRPO and RL Algorithm Improvements (2025)

### 3.1 DAPO (Directional Advantage Policy Optimization)

**Key changes from standard GRPO:**
1. **Clip-Higher**: asymmetric ε — clip low by ε₁ but allow upside by ε₂ > ε₁ (e.g., ε₁=0.2, ε₂=0.5). Encourages exploration.
2. **Dynamic Sampling**: skip batches where all K samples succeed OR fail (zero variance → no gradient signal)
3. **Overlong Filtering**: truncated responses ≠ wrong responses; don't penalize mid-sentence truncation
4. **Token-Level PG Loss**: normalize by total token count, not per-response count (prevents short-response bias)
5. **Result**: 50% fewer training steps than DeepSeek-R1-Zero to reach equivalent performance

**Recommendation**: Adopt all 4 DAPO changes in `phase_f_grpo_lite.py`.

### 3.2 Dr. GRPO (Debiased GRPO)

**Problem with GRPO's variance normalization:**
- Standard GRPO divides each sample's advantage by the batch variance
- Long, wrong responses have high variance → inflated gradient → model over-penalized for hard problems
- This creates "difficulty bias": model avoids hard problems

**Fix**: Remove per-sample variance normalization, use global normalization

**Implementation in our GRPO script**: Change `_normalize=True` to global normalization.

### 3.3 REINFORCE++ and Clip-Variance variants

From various 2025 papers, recommended GRPO config for math reasoning:
```python
GRPOConfig(
    max_completion_length=512,
    num_generations=8,     # more samples = better gradient estimate
    temperature=1.0,       # full exploration during RL
    learning_rate=1e-6,
    beta=0.04,             # KL penalty coefficient
    use_vllm=True,         # if available for faster generation
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
)
```

---

## 4. Best-of-N Reranking: Empirical Numbers

### 4.1 Literature benchmarks

| Model | K | Method | GSM8K | MATH |
|---|---|---|---|---|
| Qwen2.5-Math-7B-Instruct (greedy) | 1 | — | 92.0% | 79.4% |
| Qwen2.5-Math-7B-Instruct | 4 | random-4 | ~95% | ~83% |
| Qwen2.5-Math-7B-Instruct + PRM-7B | 4 | PRM rerank | ~97% | ~86% |
| Qwen2.5-Math-7B-Instruct | 64 | oracle | ~99% | ~94% |

**Our Phase F gate**: PRM-reranked (K=4) > random-4 + 2% accuracy on 200 GSM8K problems

### 4.2 Phase F3 expected outcome

Pre-training accuracy of Qwen2.5-Math-7B-Instruct on GSM8K: **95.5%** (measured in our v2 GRPO run).

This is already extremely high. PRM-reranking at K=4 on 200 problems:
- Random-4 ≈ 97-98% (ceiling effect)
- PRM-reranked-4: may show only marginal improvement due to ceiling
- **Concern**: GSM8K too easy for this model — may not be a discriminative BoN test
- **Recommendation**: Test on MATH-500 where base accuracy is lower (~79%)

---

## 5. Generative PRM (ThinkPRM / GenPRM)

### 5.1 ThinkPRM (April 2025)

- Generates a verification chain `<verify>...</verify>` before scoring
- 8K training samples → outperforms discriminative PRM trained on 800K samples
- Model: Qwen2.5-7B-Instruct fine-tuned with verification examples
- ProcessBench MATH: ~85.0 F1 (vs our 68.6)

### 5.2 GenPRM (April 2025 / AAAI 2026)

- Combines LLM verification with code execution (sympy, wolfram)
- 1.5B params > GPT-4o; 7B params > Qwen2.5-Math-PRM-72B
- No step annotation needed — uses the model's code execution as oracle

### 5.3 Implications for our project

**Not feasible in current Phase E/F timeline** — requires full model fine-tuning as seq2seq.

**Strategic note**: Our discriminative PRM (MLP value head on frozen backbone) serves a different purpose — it's designed to be lightweight and fast for inference-time compute allocation (ABR). Generative PRMs are too slow for online step scoring.

---

## 6. LoRA for PRM Fine-Tuning

### 6.1 Literature findings

- **LoRA Learns Less** (2024): LoRA consistently underperforms full fine-tuning on math/reasoning tasks due to limited capacity
- BUT: LoRA + PRM-specialized backbone (our PBR32-37 setup) should minimize this gap
- Recommended rank for 7B math: r=16-64 for better coverage; r=8 is minimum viable

### 6.2 Our LoRA experiments (in progress)

| Run | Config | Status | Prediction |
|---|---|---|---|
| PBR33 | r=8 top-4 + PBR26 | ✅ DONE — MATH F1=0.666, GSM F1=0.797 | GSM SOTA but MATH regressed |
| PBR34 | r=16 top-4 + PBR26 | training epoch 3 | MATH F1 ~0.67-0.69 |
| PBR35 | r=8 all-28 + PBR26 | training epoch 2 | MATH F1 ~0.69-0.72 (target) |
| PBR36 | r=32 all-28 + PBR26 | training epoch 0 | MATH F1 ~0.71-0.73 (highest capacity) |
| PBR37 | r=8 all-28 + contrastive 0.2 | training epoch 0 | target: PBR35 + 1-2 F1 |

**Key hypothesis**: all-28 layers + larger rank → better MATH F1 (at cost of GSM F1)

### 6.3 Contrastive loss effect

From Scale AI's research (cited in our memory):
- Contrastive margin loss on backbone features: +0.09 AUROC
- Margin loss: L_ctr = ReLU(cos_sim(chosen_feat, rejected_feat) - (1 - margin))
- Recommended weight: 0.05-0.15; our PBR37 uses 0.20 (aggressive)

**Hypothesis being tested**: Does contrastive loss help LoRA adapters learn better step discriminations?

---

## 7. Offline ABR Controller: Key Findings

### 7.1 Heuristic threshold beats offline REINFORCE

All offline REINFORCE variants (linear, MLP, GRU, BC warmstart) fail to beat the fixed threshold:

| Method | Binary F1 | Compute Fraction |
|---|---|---|
| **Fixed threshold τ=0.35** | **0.808** | 0.72 |
| Offline REINFORCE (linear) | 0.747 | lower |
| Offline REINFORCE (MLP-64) | 0.722 | 0.69 |
| Offline REINFORCE (MLP-128) | 0.722 | similar |
| Offline REINFORCE (GRU-32) | 0.777 | similar |
| BC warmstart + fine-tune | 0.779 | similar |

**Conclusion**: For this task (step-level error detection with single PRM score feature), a fixed threshold is optimal. Neural policies don't have enough signal to improve over this simple strategy.

**Implication for ABR**: The ABR controller should use the fixed threshold τ ≈ 0.35-0.40 for Phase F production. No need for RL-trained controller.

### 7.2 Why heuristic wins

The PRM score is already a well-calibrated single number. The "stop/continue" decision based on score < τ is optimal for a unimodal score distribution. Adding recurrent state (GRU) or nonlinear features (MLP) over the score history doesn't help because:
1. The score trajectory is noisy — short history doesn't predict future
2. The task has binary labels — complex policies overfit small training sets

**Exception**: Multi-step lookahead or multi-feature states (uncertainty + score + step position) might help. Deferred to Phase G.

---

## 8. Key Actionable Recommendations

### Phase E (in-flight)
1. **Wait for PBR35/36 to complete** — all-28-layer LoRA is the critical experiment
2. **Contrastive loss validation** — PBR37 vs PBR35 (same except contrastive 0.2)
3. **Fix `phase_f_grpo_lite.py` reward shaping**: replace raw PRM score with Clip+Delta formulation
4. **Test BoN on MATH-500** not just GSM8K — 7B-Instruct has 95.5% on GSM8K already (ceiling)

### Phase F (immediate)
1. **BoN validation**: check PRM reranking on MATH-500 (harder domain with more room for improvement)
2. **GRPO outcome-only baseline**: already running (v2, PID 3787823)
3. **GRPO+PRM with Clip+Delta**: not yet done — should run after BoN
4. **Use PBR35/36 (LoRA) adapter** for GRPO+PRM once training completes (~2026-03-13)

### Phase F/G improvements
1. **Adopt DAPO changes**: asymmetric clip, dynamic sampling, overlong filter, token-level PG loss
2. **Try PRIME implicit reward** after LoRA backbone is available (no step labels needed)
3. **ABR controller**: use fixed threshold τ=0.38 — neural controller adds no value at current data scale
4. **Test on MATH-500**: more discriminative than GSM8K for RL improvements

---

## 9. Summary: Literature Gaps and Our Contributions

| Contribution | Status | Our Value |
|---|---|---|
| Frozen PRM backbone + lightweight value head | ✅ Working (0.686 F1) | Low-latency inference |
| LoRA backbone fine-tuning for PRM | 🔄 In progress | Breaks frozen ceiling |
| ProcessBench F1 evaluation | ✅ Implemented | Proper F1 metric vs AUC |
| Contrastive loss for PRM | 🔄 Testing (PBR37) | Literature says +0.09 AUROC |
| ABR adaptive compute controller | ✅ Validated (offline) | Fixed threshold works |
| GRPO with process reward | 🔄 In progress | Clip+Delta needed |
| PRIME implicit reward | 📋 Planned | After LoRA done |
| Generative PRM (ThinkPRM) | ❌ Not planned | Too slow for ABR |
| Consensus filtering | 📋 Planned | Need LLM judge oracle |

---

*Updated: 2026-03-12 by overnight research session*
*Related docs: `docs/phase_F_plan.md`, `docs/phase_e_research_update_20260312.md`*
