# PRM + RL + GRPO Variants: Research Synthesis (2025–2026)

Compiled: 2026-03-12. Focused on papers NOT covered in `literature_refresh_20260312_prm_rl.md`.
Companion document — read alongside existing literature refresh. This doc adds coverage on:
DAPO / Dr.GRPO / VAPO RL algorithm variants, PAV process advantage verifiers,
reward hacking Clip/Delta mechanisms, contrastive goal-conditioned representations
for reward models (Scale AI), ThinkPRM/GenPRM quantitative results, PRIME deep details,
Best-of-N scaling numbers, and LoRA vs full fine-tuning for PRM/RL math training.

---

## 1. PRIME: Deep Technical Details

**Paper:** arXiv:2502.01456 (Cui et al., Tsinghua/SJTU/PKU/UIUC/CUHK, Feb 2025)
**Code:** https://github.com/PRIME-RL/PRIME
**Predecessor:** "Free Process Rewards without Process Labels" arXiv:2412.01981

### Core formula

The implicit process reward at token t is:

```
r_φ(y_t) = β · log[ π_φ(y_t | y_<t) / π_ref(y_t | y_<t) ]
```

The entire sequence ORM score is therefore the SUM of per-token log-ratios (scaled by β),
making the PRM implicit in the ORM parameterization. No step-level annotation needed.

Training loss (cross-entropy variant, used during online update):
```
L_CE(φ) = -E[ r_o · log σ(r_φ(y)) + (1 - r_o) · log(1 - σ(r_φ(y))) ]
```
where `r_o ∈ {0,1}` is the binary outcome label from the verifier.

### Online RL training loop (per iteration)

1. Sample K=8 responses per prompt from current policy.
2. Obtain binary outcome labels (rule-based verifier: answer correctness).
3. Forward pass through implicit PRM and reference model → compute per-token rewards.
4. Update PRM via L_CE loss using on-policy rollouts.
5. Compute hybrid advantage:
   ```
   A_t^i = Σ_{s≥t} γ^(s-t) [ r_φ(y_s^i) - mean_j(r_φ(y_s^j)) ]  (process part)
           + [ r_o(y^i) - mean_j(r_o(y^j)) ]                        (outcome part)
   ```
   Uses leave-one-out Monte Carlo baseline. RLOO estimator for outcome reward component.
6. Update policy via PPO clip objective.

### Key results: Eurus-2-7B-PRIME

Starting from Qwen2.5-Math-7B-Base, training ~380K steps:

| Benchmark     | Eurus-2-7B-PRIME | RLOO (ORM only) | Qwen2.5-Math-7B-Instruct |
|---------------|------------------|-----------------|--------------------------|
| MATH-500      | 79.2%            | 78.2%           | 79.8%                    |
| AIME 2024     | 26.7%            | 20.0%           | 13.3%                    |
| AMC           | 57.8%            | 50.6%           | 50.6%                    |

Average across all 7 benchmarks: +15.1% over SFT baseline.
Training data: 10% of Qwen2.5-Math-7B-Instruct's dataset (only 230K SFT + 150K RL steps).

### Efficiency gains

- 2.5x sample efficiency over sparse ORM-only RLOO baseline.
- +6.9% final performance gain at matched compute.
- 11x faster than VinePPO (1.22 hr vs 13.94 hr for equivalent training).
- +24% compute overhead per step (PRM forward pass) but 2.5x fewer steps needed.

### Critical ablation: online vs offline PRM

Offline PRM (initialized from supervised SFT, NOT updated during RL):
- Performance degrades during training due to distribution shift.
- ORM-only RLOO is strictly better than frozen offline PRM.

Online PRM (initialized from SFT, updated each iteration on rollouts):
- Best overall results. Distribution shift mitigated.
- **Takeaway: if using implicit PRM, must update online. Frozen offline use is harmful.**

### Actionable implications for our project

- After LORA_S4 converges, implicit PRM is extractable with zero additional training:
  `score_step = mean( β · log(π_LoRA(t) / π_SFT(t)) for t in step_tokens )`
- The implicit PRM works as a discriminative PRM for ProcessBench F1 evaluation — no
  generative decoding overhead.
- Online update is only needed for RL training, not for inference-time reranking.
- Initialization from SFT model (not random) is critical — aligns with our LoRA setup
  where LoRA adapter starts at 0 and backbone stays at SFT init.

---

## 2. DAPO: Decoupled Clip + Dynamic Sampling Policy Optimization

**Paper:** arXiv:2503.14476 (ByteDance Seed, Mar 2025)
**Result:** 50 pts on AIME 2024 using Qwen2.5-32B base — beats DeepSeek-R1-Zero-32B (47)
**Framework:** built on veRL (same framework as PRIME)

### Four algorithmic improvements over GRPO

**1. Decoupled Clip-Higher (DCH)**

Standard PPO/GRPO uses single clipping hyperparameter ε for both upward and downward
probability ratio changes:
```
clip(π_θ/π_old, 1-ε, 1+ε)
```

DAPO decouples into ε_low (downward) and ε_high (upward), setting ε_high > ε_low:
```
clip(π_θ/π_old, 1-ε_low, 1+ε_high)   [ε_high > ε_low, e.g., 0.2 and 0.28]
```

Effect: allows token probability to INCREASE more aggressively (exploration-friendly)
while still conservatively preventing probability DECREASES. Encourages surprising but
correct reasoning tokens (novel steps that the model was initially reluctant to generate).

**2. Token-Level Policy Loss (not response-level)**

GRPO: divides loss by response LENGTH |o|, creating response-level normalization.
DAPO: divides by total NUMBER OF TOKENS across the batch (token-level normalization).

Effect: prevents short responses from dominating gradients. Encourages longer detailed
reasoning chains. More stable training dynamics on variable-length CoT.

**3. Dynamic Sampling (DS)**

Remove prompts where ALL sampled responses have the same reward (all correct OR all
incorrect) from the training batch. These prompts produce zero advantage:
```
advantage ∝ reward - mean(group_rewards) = 0  when all rewards identical
```
so they contribute only noise, not signal.

Caveats: ablation studies showed DS DOES NOT always improve performance. One comparative
analysis found "best DAPO results achieved with DS DISABLED." Verify empirically.

**4. Soft Overlong Filtering + Length Penalty**

Instead of hard-truncating long responses (which corrupts training signal near truncation):
- Mask loss on truncated portions (no gradient from incomplete tokens).
- Apply soft length penalty: add a penalty term that grows with response length beyond
  a threshold L_max. Prevents runaway length growth without abrupt truncation artifacts.

**5. Remove KL divergence term (same as Dr. GRPO)**

GRPO includes explicit KL penalty: L_GRPO = -advantage + β_KL · KL(π_θ || π_ref)
DAPO removes β_KL=0: L_DAPO = -advantage only.

Rationale: for long-CoT reasoning, policy must diverge substantially from SFT reference.
KL penalty artificially constrains the policy from exploring useful reasoning strategies.

### DAPO vs DeepSeek-R1-Zero-32B

| Method                  | AIME 2024 | Training steps |
|-------------------------|-----------|----------------|
| DeepSeek-R1-Zero-32B    | 47        | full           |
| DAPO (Qwen2.5-32B base) | 50        | 50% of above   |

### Actionable implications for our project

- If we run online RL training on the LoRA backbone (Phase E future work or Phase F3):
  use DAPO's token-level loss instead of GRPO's response-level loss. Direct swap in
  `_run_one_epoch_lora()` in `phase_e_train_value.py`.
- Clip-Higher (ε_high > ε_low): add `--clip-high` argument to RL training script.
  Encourages exploration of new reasoning token patterns — directly relevant to breaking
  the frozen backbone plateau.
- Drop KL penalty in RL training. Our current setup has no KL term already (DPO pairs
  training is not RL), so this is a note for if/when we switch to online RL.
- Soft length penalty: relevant for preventing the LoRA from generating verbose but
  incoherent outputs during RL training.

---

## 3. Dr. GRPO: Understanding and Fixing GRPO Biases

**Paper:** arXiv:2503.20783, "Understanding R1-Zero-Like Training: A Critical Perspective"
**Published:** COLM 2025
**Related:** arXiv:2504.11343, "Revisiting RL Algorithms for LLM Post-Training"

### Three biases in vanilla GRPO

**Bias 1: Baseline bias**
GRPO computes baseline as `mean(group_rewards)` divided by 1/K, but the unbiased estimator
for leave-one-out baseline requires 1/(K-1). Minor but accumulates at large K.

**Bias 2: Response-length bias**
Dividing per-response loss by response length |o_i| creates systematic bias:
- For CORRECT responses: penalizes longer explanations (pushes model toward shorter answers).
- For INCORRECT responses: penalizes shorter wrong answers LESS (pushes model toward longer
  incorrect outputs).
- Net effect: model generates progressively LONGER incorrect responses, "overthinking" behavior.

**Bias 3: Question-level difficulty bias (std normalization)**
Dividing advantages by `std(rewards across group)` gives disproportionate weight to
questions with LOW variance (easy or very hard questions where all K samples agree).
Hard problems where exploration matters are effectively down-weighted.

### Dr. GRPO fix

Remove both 1/|o_i| and 1/std(group_rewards) terms from the GRPO objective.
The resulting objective is an unbiased PPO gradient estimator with GRPO-style grouping.

```
L_DrGRPO = -Σ_i Σ_t A_i · log π_θ(a_t | s_t)    [no length or std normalization]
```

where advantages A_i are computed as simple (reward - group_mean) without std scaling.

### Ablation results

Oat-Zero-7B trained with Dr. GRPO recipe (27 hr on 8×A100):

| Benchmark | Oat-Zero-7B (Dr. GRPO) |
|-----------|------------------------|
| AIME 2024 | 43.3%                  |
| AMC       | 62.7%                  |
| MATH-500  | 80.0%                  |
| Avg (5 benchmarks) | 51.4%         |

State-of-the-art for 7B models at time of release.

### Effect on response length

Removing length normalization stabilizes response length growth. GRPO: length grows
continuously even after reward plateaus. Dr. GRPO: length stabilizes, preventing the
"wildly extended but wrong" outputs.

### Key insight from arXiv:2504.11343

Ablation: GRPO's core value comes from FILTERING zero-reward groups (prompts where all
K samples are wrong), NOT from its reward normalization. A minimal REINFORCE variant
that only applies this filtering ("Reinforce-Rej") achieves near-GRPO performance with
much less complexity.

### Actionable implications for our project

- If implementing RL training for LoRA backbone, use Dr. GRPO (no length or std normalization)
  as default over vanilla GRPO.
- Length bias in GRPO explains why models generate overly verbose incorrect reasoning:
  a known pathology that explains some of the "length preference" reward hacking we observed
  in PBR experiments (longer chains receive higher accumulated PRM scores even if wrong).
- The filtering insight (Reinforce-Rej) suggests: a simple training strategy is to discard
  all-zero-reward batches and use plain policy gradient on the rest. Easier to implement
  than full GRPO in our LoRA setup.

---

## 4. VAPO: Value-Based Augmented Proximal Policy Optimization

**Paper:** arXiv:2504.05118 (ByteDance Seed, Apr 2025)
**Result:** 60.4 on AIME 2024 (Qwen2.5-32B base), vs DAPO's 50 — +10 points

### Motivation: why value models?

Value-model-free methods (GRPO, DAPO) use group-relative advantages as credit assignment.
This is statistically unbiased but high variance — especially for long CoT where token-level
credit assignment is diffuse.

VAPO argues value models enable more precise credit assignment by explicitly tracking
Q(state, action) across time steps.

### Three key innovations

**1. Value Pretraining (avoids value model bias at init)**

Standard practice: initialize value model from reward model weights.
Problem: reward models predict scalar preference; value models predict expected future return.
These are different objectives → initialization creates bias.

Fix: pretrain value model using Monte Carlo returns from a FIXED (frozen) policy:
```
V_pretrain(s_t) = MC return from s_t under π_fixed
```
This gives the value model a clean unbiased starting point aligned with expected return.

**2. Decoupled GAE (different λ for critic vs policy)**

Standard GAE uses single λ for both critic training and policy gradient:
```
A_t = Σ_k (γλ)^k δ_{t+k}      [same λ everywhere]
```

VAPO decouples:
- Critic (value) update: λ_critic = 1.0 (unbiased Monte Carlo, no TD bootstrapping bias)
- Policy update: λ_policy set adaptively (lower for long responses to reduce variance)

**3. Length-Adaptive GAE**

For LONG responses: lower λ_policy to reduce variance (less bootstrapping reach).
For SHORT responses: higher λ_policy (full temporal credit assignment is manageable).
```
λ_policy(length) = λ_base · exp(-α · length / L_max)
```
This prevents credit signal from being diluted across very long reasoning chains.

**4. Positive Example NLL Loss**

For correctly-answered problems: add standard language modeling loss (NLL) on the
positive completion in addition to the RL objective:
```
L_total = L_RL + α · L_NLL_positive_only
```
Effect: imitation learning on correct samples prevents catastrophic forgetting and
accelerates convergence on problems the model can already solve.

### Results

| Method                       | AIME 2024 | Training steps |
|------------------------------|-----------|----------------|
| DeepSeek-R1-Zero-32B         | 47        | large          |
| DAPO (Qwen2.5-32B base)      | 50        | medium         |
| **VAPO (Qwen2.5-32B base)**  | **60.4**  | 5,000 steps    |

60.4 AIME 2024 in 5,000 training steps with no crashes across multiple runs.

### Actionable implications for our project

- The Positive Example NLL Loss is directly applicable NOW without RL:
  In our DPO/pair training, add an NLL loss term on the CHOSEN (correct) steps.
  This is different from terminal_bce — it's a language modeling objective that prevents
  the backbone features from drifting away from correct reasoning patterns.
- Value pretraining concept maps to our architecture: if adding a value head, initialize
  it using MC-estimated step success rates (Math-Shepherd labels) rather than random init.
  Our current random MLP init likely has bias in early epochs.
- Length-adaptive GAE is important context for Phase F3 RL controller design: any future
  RL training should use length-dependent discount or λ adjustment.

---

## 5. PAV: Process Advantage Verifiers

**Paper:** arXiv:2410.08146 (Setlur et al., Google DeepMind/CMU, ICLR 2025 Spotlight)
**Title:** "Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning"

### Core concept: measuring PROGRESS not correctness

Standard PRM: `P(correct | solution_prefix)` — probability that prefix leads to correct answer.
PAV: `A_π_prover(s_t, a_t)` — CHANGE in probability of correct completion BETWEEN step t
and step t+1, measured under a SEPARATE PROVER POLICY (not the base policy).

```
PAV_score(s_t, a_t) = Q_prover(s_{t+1}) - Q_prover(s_t)
                     = P_prover(correct | prefix_{t+1}) - P_prover(correct | prefix_t)
```

This is the advantage function in the MDP defined by the prover policy — not the base policy.

### Why a separate prover?

If you use the BASE policy's own advantages as process rewards in RL, the result is
mathematically equivalent to just using outcome rewards (the advantages cancel in the
policy gradient). No additional signal is provided.

For process rewards to add information, the prover must evaluate progress DIFFERENTLY
from the base policy — ideally a "complementary" prover that succeeds where the base fails
and vice versa.

Empirical finding: weak prover policies outperform strong ones as verifiers because they
better DISTINGUISH steps. A strong prover completes from both correct and incorrect
prefixes equally well (both score high). A weak prover succeeds only from correct prefixes.

### Performance gains

- Test-time search with PAV vs ORM: >8% accuracy improvement, 1.5–5x compute efficiency.
- Online RL training with PAV dense rewards vs ORM:
  - 5–6x sample efficiency gain.
  - >6% accuracy gain at matched compute.
- Effective reward formula for RL: `R_eff(s, a) = Q_base(s, a) + α · A_prover(s, a)`

### Comparison to our setup

Our MLP value head is essentially trying to learn `P(correct | step_prefix)`, which is
the standard PRM formulation — not PAV's progress/advantage formulation.

Converting to PAV would require:
1. Training a separate "prover" model (could be Qwen2.5-Math-PRM-7B full model or R-PRM).
2. Computing `Q_prover(t+1) - Q_prover(t)` as the training target for our MLP.
3. Using Math-Shepherd MC labels as approximate Q_prover values (they already estimate
   P(correct | prefix) which is exactly what PAV needs).

### Actionable implications

- Our Math-Shepherd MC labels are ALREADY PAV-compatible: `MC_rate(step_t)` estimates
  `P_prover(correct | prefix_t)`. The PAV score for step t is `MC_rate(t+1) - MC_rate(t)`.
- Training MLP to predict PAV scores (differences) vs raw MC rates:
  Replace training target `y = MC_rate(step)` with `y = MC_rate(t+1) - MC_rate(t)`.
  This is a simple label transformation, no architecture change needed.
- PAV scores are inherently SIGNED and centered near 0 (progress can be positive or negative),
  making them more informative than one-sided MC rates for detecting regression steps.
- The prover vs base distinction: Qwen2.5-Math-PRM-7B backbone (our frozen model) serves
  as the prover evaluating progress, and our MLP value head becomes the lightweight
  surrogate that approximates the prover's advantage function.

---

## 6. Reward Hacking: Clip and Delta Mechanisms

**Paper:** arXiv:2410.15115 (Luo et al., CUHK, Oct 2024)
**Title:** "On Designing Effective RL Reward at Training Time for LLM Reasoning"

### The problem: PRM reward hacking in RL

When a PRM is directly used as a dense reward in RL training, LLMs exploit it by:
1. **Repetition hack**: repeat correct but trivial steps (e.g., "therefore X = X") to
   accumulate high PRM scores without advancing toward the answer.
2. **Verbosity hack**: pad each step with confident-sounding but empty content.
3. **Structural hack**: match known good solution patterns in ways that fool the PRM
   without actually reasoning correctly.

Empirical evidence: PRM-only RL degrades from 30.58% → 11.16% greedy accuracy (Qwen2-1.5B
on MATH). A catastrophic 19.4-point DROP from adding the PRM reward.

### Clip mechanism

```
r_clip(step_k) = min( PRM(step_k) - η, 0 )
```

Upper-bounds all step rewards to ≤ 0. Only PENALIZES steps with PRM score below η.
Steps with score above η receive reward = 0 (not rewarded, not penalized).

Effect: the model cannot gain by generating high-scoring trivial steps. It is only
punished for generating notably bad steps. Prevents repetition exploitation.

Threshold η: chosen so that "the majority of reasoning steps receive reward 0." A value
like η = 0.9 (for PRMs with range [0,1]) ensures only the bottom ~30% of steps are
penalized and the top ~70% are merely 0.

### Delta mechanism

```
r_delta(step_k) = PRM(step_k) - PRM(step_{k+1})    for k < K-1
r_delta(step_K) = PRM(step_K)                       for final step
r_delta(step_0) = 0                                 for first step
```

Computes DIFFERENCE between adjacent step scores. Rewards only steps that IMPROVE the
solution, penalizes steps that make it worse. Static high scores from repetition cancel
out in the difference (score stays constant → delta = 0).

The Delta mechanism is mathematically related to PAV's advantage formulation — both
measure progress rather than absolute quality.

### Combined results (Qwen2-1.5B on MATH)

| Method                | Greedy Accuracy |
|-----------------------|----------------|
| Success reward only   | 30.58%         |
| PRM reward (raw)      | 11.16% (HACK)  |
| PRM + Clip            | 30.30%         |
| PRM + Delta           | 30.68%         |
| **PRM + Clip + Delta**| **31.44%**     |

Clip+Delta recovers from catastrophic hacking AND achieves a modest improvement over
outcome reward alone.

### Scaling experiments (Qwen2.5-Math-7B-Instruct)

| Metric         | Baseline | With Clip+Delta |
|----------------|----------|-----------------|
| MATH greedy    | 83.30%   | 83.38%          |
| MATH sampling  | 52.76%   | 81.22% (+28%!)  |
| GSM8K greedy   | 95.60%   | 95.60%          |
| GSM8K sampling | 80.74%   | 95.07% (+14%)   |

Sampling accuracy (pass@16) improves dramatically — the model learns diverse correct
reasoning paths rather than a single mode.

### Actionable implications for our project

- Our Phase F reward hacking probe PASSED on MATH (all LOW risk). The Clip mechanism
  explains why: our value head with threshold τ=0.35 only penalizes steps scoring BELOW
  the threshold — equivalent to the Clip mechanism with η=0.35.
- If we implement online RL training (Phase F3 or beyond), use Clip+Delta as the
  default PRM reward shaping. The η=0.9 threshold for Clip works if PRM scores are in [0,1].
- Delta mechanism for our ABR controller: instead of hard-thresholding on absolute score,
  flag a step for early stop when `score(t) - score(t-1) < δ` (score is DECREASING).
  This is more principled than our current absolute threshold approach.
- PRM sampling accuracy improvement (+28% on MATH) suggests that even without improving
  the PRM itself, proper RL training with Clip+Delta yields large gains.

---

## 7. Contrastive / Goal-Conditioned Representations for PRMs

**Source:** Scale AI research blog (goal-conditioned representations for reward models)
**Related:** arXiv:2024.emnlp "Improving Discriminative Capability of Reward Models Using
Contrastive Learning"; CLHA (Contrastive Learning for Human Alignment, 2024)

### Scale AI: Goal-Conditioned Contrastive PRM (key findings)

Core idea: train reward model hidden representations with a contrastive objective applied
at INTERMEDIATE STEPS of the reasoning sequence, not just at the final output.

Training objective:
- For preferred trajectories: increase cosine similarity between consecutive step
  representations (pull them together).
- For dispreferred trajectories: decrease cosine similarity between step representations.
- Applied to intermediate hidden states (not just the final layer logit).

Formulation (goal-conditioned contrastive loss at step t):
```
L_contrastive = -log[ sim(h_t_preferred, h_{t+1}_preferred)
                    / (sim(h_t_preferred, h_{t+1}_preferred) + sim(h_t_rejected, h_{t+1}_rejected)) ]
```

Implementation details:
- Learning rate: 5e-6 for contrastive training.
- Data augmentation: dropout rate 0.05 to create perturbed views for contrastive pairs.
- No additional annotation needed beyond standard sequence-level preference labels.

Results:
- **+0.09 AUROC improvement** on MATH and GSM8K benchmarks.
- **+2.3% accuracy** on the Helpful-Harmless alignment dataset.
- More stable PPO training curves (consistent reward and return signals).
- Dense signal at intermediate steps helps localize errors in partially completed sequences.

### CLHA and EMNLP approach (discriminative RM contrastive)

- Add unsupervised contrastive loss alongside supervised ranking loss.
- Contrastive pairs: augment chosen/rejected sequences with dropout perturbations.
- Improves generalization and stabilizes RL training.
- Interaction effect: contrastive + ranking loss > either alone.

### Relationship to our existing work

Our existing DPO pair training is already contrastive at the SOLUTION level (chosen vs
rejected full sequences). The Scale AI method extends this to STEP level (intermediate
hidden states between steps).

The `dual_head` routing in `training.py:871` approximates this: separate reward_head and
value_head from shared backbone hidden states. The contrastive signal could be applied
across the backbone hidden states at step boundaries.

### Actionable implications for our project

- The +0.09 AUROC / +2.3% accuracy gain is achievable with a contrastive loss applied
  to intermediate hidden states — no additional annotations needed.
- Implementation: in `training.py:_compute_single_pair_objective()`, after extracting
  step-level backbone features (h_chosen_t, h_rejected_t), add:
  ```python
  # Contrastive loss: chosen steps should have similar representations to each other
  # across the trajectory; rejected steps should diverge
  cos_sim_chosen = F.cosine_similarity(h_chosen_t, h_chosen_t_next)
  cos_sim_rejected = F.cosine_similarity(h_rejected_t, h_rejected_t_next)
  l_contrastive = -torch.log(cos_sim_chosen.exp() / (cos_sim_chosen.exp() + cos_sim_rejected.exp()))
  ```
- Dropout augmentation at rate 0.05 is the contrastive view creation mechanism.
  For LoRA training, backbone dropout can be enabled (check `apply_lora_to_backbone()`).
- This is additive to PQM margin loss (see existing literature refresh). Combined:
  PQM ranking loss + Scale AI contrastive loss + terminal BCE could be tried together.

---

## 8. ThinkPRM and GenPRM: Quantitative Benchmarks

### ThinkPRM (arXiv:2504.16828, MIT/Mila/LG AI, Apr 2025)

Data efficiency:
- Fine-tuned on only 1% of PRM800K process labels (approx 8K labeled steps).
- 8K process labels + 1K synthetic CoT examples total.

Out-of-domain results (vs discriminative PRMs trained on full PRM800K):
- GPQA-Diamond subset: +8% over full-PRM800K-trained discriminative verifiers.
- LiveCodeBench: +4.5% over full-PRM800K-trained discriminative verifiers.
- ProcessBench subset (same token budget): +7.2% vs LLM-as-a-Judge.

Scaling properties:
- Parallel scaling: sample K independent verification CoTs, average scores. Improves
  with K up to ~8, then diminishing returns.
- Sequential scaling: enable model to double-check initial verification. Modest additional
  gains over single-pass.

Key insight: generative verification CoT allows the model to REASON ABOUT WHY a step
is wrong, not just score it. This is qualitatively different from discriminative PRMs
that only output a scalar.

### GenPRM (arXiv:2504.00891, AAAI 2026)

Architecture: generative PRM that writes step-by-step CoT reasoning + optional code
verification before producing a binary judgment per step.

Relative Progress Estimation (RPE):
- Key innovation for label generation.
- For step t, RPE = (MC_success_rate(prefix_t+1) - MC_success_rate(prefix_t)) / MC_success_rate(prefix_t)
  This is a normalized improvement over the previous step's success rate.
- Steps with high POSITIVE RPE = important progress steps (key for training signal).
- Steps with NEGATIVE RPE = regression steps (most important to flag for ProcessBench).

Training data: 23K examples from MATH dataset only (not Math-Shepherd, not PRM800K).

ProcessBench results:
- 1.5B GenPRM: outperforms GPT-4o on ProcessBench.
- 7B GenPRM: surpasses Qwen2.5-Math-PRM-72B on ProcessBench.
- F1 scaling: 1.5B → 7B shows large gain; 7B → 32B shows marginal gain (7B sweet spot).

### Discriminative vs generative: key tradeoff

| Dimension | Discriminative PRM (ours) | Generative PRM (ThinkPRM/GenPRM) |
|-----------|--------------------------|-----------------------------------|
| Inference latency | O(1) per step (single forward pass) | O(100s of tokens) per step |
| Interpretability | Score only | Full reasoning trace |
| Data efficiency | Needs large labeled sets | Can work with 8K–23K examples |
| Scaling | No test-time compute scaling | Can sample K verification CoTs |
| ProcessBench F1 | PBR26: MATH=0.686 | 7B GenPRM: beats PRM-72B (~0.80+) |

**Bottom line:** generative PRMs dominate discriminative at quality. The gap is latency
and compute: a discriminative PRM is ~100x faster to evaluate per step.

For our Phase F ABR controller: discriminative PRM is the right choice for real-time
inference compute control (latency matters). For offline data labeling and quality
filtering, GenPRM-style approach is superior.

### RPE labels as training signal for our MLP value head

GenPRM's RPE labels (normalized progress) are a practical improvement over raw MC labels:
- RPE highlights steps that MATTER (high progress or high regression).
- Flat steps (RPE ≈ 0) are implicitly down-weighted.
- Can be computed offline from Math-Shepherd MC data: `RPE(t) = (mc(t+1) - mc(t)) / max(mc(t), ε)`.
- Using RPE as soft training labels for our value head terminal supervision (instead of
  hard 0/1 BCE) would directly implement GenPRM's insight in our discriminative framework.

---

## 9. Best-of-N Reranking: Actual Numbers

### BoN with discriminative PRM (GSM8K, MATH-500)

From "Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability" (2025):

| Model (Evaluator)         | ProcessBench F1 avg | Notes                           |
|---------------------------|---------------------|---------------------------------|
| Skywork-PRM-7B            | 42.1                | Poor MATH generalization        |
| Qwen2.5-Math-PRM-7B       | 73.5                | Our community target            |
| Qwen2.5-Math-PRM-72B      | 78.3                | Oracle upper bound              |
| DeepSeek-R1               | 83.5                | RL-trained, implicit PRM        |
| QwQ-32B                   | 83.7                | RL-trained, implicit PRM        |

Key finding: RL-trained reasoning models (DeepSeek-R1, QwQ-32B) EXCEED all explicitly
trained discriminative PRMs on ProcessBench F1, WITHOUT any PRM training.

### BoN compute efficiency (from REBASE, ICLR 2025)

REBASE (tree search with PRM) vs Best-of-N sampling:

- GSM8K + N=128 samples: REBASE achieves 90.2% accuracy with weighted majority voting.
- 7B model: REBASE achieves same accuracy as Best-of-N with 7x LESS compute.
- Easy problems: Best-of-N saturates quickly (GSM8K difficulty plateau).
- Hard problems: Best-of-N (MATH) does not saturate until N=256+.

### EORM (Efficient ORM, May 2025)

EORM achieves 92.8% on GSM8K using only 256 candidate chains — matches brute-force
self-consistency at much lower cost. Shows that smart reranking can outperform
simply generating more samples.

### PRM overconfidence problem (calibration)

Recent work on PRM calibration: state-of-the-art PRMs systematically OVERESTIMATE
success probability. This causes:
- Inefficient compute allocation (allocates large budgets to already-easy problems).
- Unreliable uncertainty estimates for ABR controller (phase F3 concern).

Fix: quantile regression calibration layer on top of PRM scores.
`score_calibrated(t) = quantile_regress(score_raw(t), coverage=0.8)` ensures
80% of steps with score > threshold are actually correct.

### Actionable implications

- Our PBR19/PBR26 ProcessBench F1 (0.683/0.686) vs community target (0.735 for PRM-7B):
  the remaining gap is PRIMARILY a data quality gap, not an architecture gap. RL-trained
  models reaching 83.5 without explicit PRM training is a warning sign for our approach.
- The path to 0.735+ is: either LoRA (break frozen plateau) or better training data
  (consensus-filtered pairs, RPE labels, EDU entropy step boundaries).
- For Phase F3 controller: calibrate PRM scores before using as stopping criterion.
  Overconfident scores → premature stopping → missed errors.

---

## 10. LoRA for PRM Fine-Tuning: Best Practices from 2025

### Key finding: LoRA matches full fine-tuning for math RL

From Thinking Machines Lab "LoRA Without Regret" experiments:
- Qwen3-8B-base on DeepMath dataset (large, hard problems).
- LoRA (rank 8, 16, 64) vs full fine-tuning: training progresses IDENTICALLY when
  optimal learning rates are chosen for each setting.
- Same reasoning behaviors emerge (backtracking, self-verification, in-context exploration).
- LoRA is MORE FORGIVING to tune: wider range of learning rates achieves near-optimal.
- Memory savings: 50%+ over full fine-tuning at rank=8, competitive performance.

### LoRA targets for math backbone (from Qwen2.5-Math fine-tuning practice)

For a Transformer backbone targeting attention and feed-forward:
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj` (all 4 projection matrices)
- Feed-forward: `gate_proj`, `up_proj`, `down_proj` (all 3 MLP matrices)
- Recommended rank: r=16 for full target, r=8 for conservative/fast experiments.
- Larger rank (r=32+): marginal gains, not worth memory cost.

Our current LoRA configuration (PBR31, PBR32): rank=8, targeting q+v only (2 of 4
attention projections). The literature suggests including all 4 attention projections
(+ o_proj) and optionally MLP projections for larger effect size.

### NORM: Better LoRA via SVD filtering (ICLR 2025)

SVD-based extension of LoRA that removes noisy singular value components:
- Identifies "essential" vs "noisy" singular values using subspace similarity.
- +5.31 improvement over vanilla LoRA on math reasoning + code generation.
- Less sensitive to rank hyperparameter than standard LoRA.
- Not yet available in standard PEFT library — requires custom implementation.

### GRPO + LoRA training results

From TRL/HuggingFace advanced cookbook and published reports:
- GRPO + LoRA (rank=8) on GSM8K: comparable final accuracy to GRPO + full fine-tuning.
- Training speed: LoRA is 2–4x faster per step due to reduced gradient computation.
- Memory: A100-80GB can handle batch_size=4, rank=8, sequence_length=4096 without OOM.
  Gradient checkpointing mandatory for rank > 8 or sequence_length > 4096.
- Effective batch size matters more than rank: use gradient accumulation to simulate
  large batches (effective_batch = 32–64) for stable GRPO training.

### Frozen vs LoRA plateau (our project specific)

Confirmed plateau in our experiments: frozen backbone + MLP value head = MATH F1 ≤ 0.686.
Literature confirms: LoRA is the necessary lever to break through this ceiling.
The Thinking Machines Lab finding that LoRA matches full fine-tuning means:
rank=8–16 on q+v+o projections should be sufficient to reach community target (0.735).

Remaining open question: which layers to target with LoRA for ProcessBench F1 specifically?
PBR32 (all layers) vs PBR31 (q+v only) will answer this empirically. Literature suggests
all 4 attention projections (not just q+v) is the standard best practice.

---

## 11. Key New Papers Not in Existing Literature Refresh

| Paper | arXiv | Key contribution |
|-------|-------|-----------------|
| DAPO | 2503.14476 | Decoupled clip, token-level loss, soft overlong filtering, +50 AIME |
| Dr. GRPO | 2503.20783 | Removes length+std bias, prevents overthinking, COLM 2025 |
| VAPO | 2504.05118 | Value pretraining + Decoupled-GAE, 60.4 AIME 2024 |
| Clip+Delta reward shaping | 2410.15115 | Prevents PRM reward hacking in RL |
| PAV (ICLR 2025) | 2410.08146 | Process advantage as training signal, 5-6x RL efficiency |
| ThinkPRM | 2504.16828 | Generative verifier with long CoT, 1% labels needed |
| GenPRM (AAAI 2026) | 2504.00891 | RPE labels + code verification, 7B beats PRM-72B |
| EDU-PRM | 2503.22233 | Entropy step boundaries τ=1.0, 7.5K queries matches Qwen PRM-72B |
| BiPRM | 2508.01682 | R2L stream via prompt reversal, +37.7% step detection |
| Is PRM Necessary? | 2505.11227 | RL-trained LLMs outperform discriminative PRMs on ProcessBench |
| Scaf-GRPO | 2510.19807 | Scaffolding fix for zero-reward collapse |
| GRPO is secretly PRM | (2025) | GRPO implicitly defines per-step credit assignment |

---

## 12. Consolidated Priority Recommendations

These complement (and in some cases update) the priority queue in `literature_refresh_20260312_prm_rl.md`:

### Immediate (within LoRA experiments, no new code infrastructure)

1. **Add o_proj to LoRA targets** in PBR31/PBR32 follow-up runs. Cost: minor, expected impact:
   literature consistently recommends all 4 attention projections for full effect.

2. **Use Dr. GRPO objective** (no length or std normalization) if/when adding RL training
   to LoRA experiments. Prevents overthinking and improves token efficiency.

3. **Value pretraining (VAPO concept):** Initialize MLP value head weights using MC rates
   from Math-Shepherd rather than random init. Simple change in `training.py`, may improve
   early-epoch convergence.

### Short-term (new training runs, no architecture change)

4. **PAV-style training targets:** Replace raw MC labels with RPE labels
   `(mc(t+1) - mc(t)) / max(mc(t), ε)` as soft training targets for value head.
   Equivalent to GenPRM's RPE but applied to our discriminative framework.

5. **Clip+Delta reward shaping:** For any future RL training run, replace raw PRM scores
   with `r_clip_delta = min(PRM(t) - η, 0) + (PRM(t) - PRM(t+1))`. Prevents repetition
   and verbosity hacking (confirmed by arXiv:2410.15115).

6. **PRIME implicit PRM evaluation:** After any LoRA run, compute
   `score(step) = mean(β · log(π_LoRA/π_ref))` over step tokens and evaluate on
   ProcessBench F1 directly. Zero additional training.

### Medium-term (new data pipeline)

7. **EDU entropy step boundaries** (existing recommendation, now confirmed with exact
   numbers: entropy τ=1.0, 7.5K queries sufficient to match Qwen2.5-Math-PRM-72B on
   ProcessBench at 98% less query cost).

8. **PBR calibration layer:** Add quantile regression calibration to PRM scores before
   using as ABR controller stopping criterion. Prevents overconfident early stopping.

9. **Contrastive loss on step-level hidden states** (Scale AI method): +0.09 AUROC with
   no additional annotations. Add dropout-augmented contrastive term in
   `_compute_single_pair_objective()`. Worth implementing alongside PQM ranking loss.

---

## Sources

- [PRIME paper (arXiv:2502.01456)](https://arxiv.org/abs/2502.01456)
- [PRIME GitHub](https://github.com/PRIME-RL/PRIME)
- [ImplicitPRM GitHub](https://github.com/PRIME-RL/ImplicitPRM)
- [DAPO (arXiv:2503.14476)](https://arxiv.org/abs/2503.14476)
- [Dr. GRPO (arXiv:2503.20783)](https://arxiv.org/abs/2503.20783)
- [VAPO (arXiv:2504.05118)](https://arxiv.org/abs/2504.05118)
- [PAV (arXiv:2410.08146)](https://arxiv.org/abs/2410.08146)
- [Clip+Delta (arXiv:2410.15115)](https://arxiv.org/abs/2410.15115)
- [ThinkPRM (arXiv:2504.16828)](https://arxiv.org/abs/2504.16828)
- [GenPRM (arXiv:2504.00891)](https://arxiv.org/abs/2504.00891)
- [EDU-PRM (arXiv:2503.22233)](https://arxiv.org/abs/2503.22233)
- [BiPRM (arXiv:2508.01682)](https://arxiv.org/abs/2508.01682)
- [Is PRM Necessary? (arXiv:2505.11227)](https://arxiv.org/abs/2505.11227)
- [Qwen2.5-Math-PRM blog](https://qwenlm.github.io/blog/qwen2.5-math-prm/)
- [MarkTechPost PRIME overview](https://www.marktechpost.com/2025/02/07/process-reinforcement-through-implicit-rewards-prime-a-scalable-machine-learning-framework-for-enhancing-reasoning-capabilities/)
- [MarkTechPost VAPO overview](https://www.marktechpost.com/2025/04/10/bytedance-introduces-vapo-a-novel-reinforcement-learning-framework-for-advanced-reasoning-tasks/)
- [Thinking Machines Lab LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- [DAPO paper (AI Papers Academy overview)](https://aipapersacademy.com/dapo/)
- [BiRM (ACL 2025 Findings)](https://aclanthology.org/2025.findings-acl.747/)
- [REBASE (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/98711dea460bdefe0e651ca23ec98ba2-Abstract-Conference.html)
- [GenPRM GitHub (RyanLiu112/GenPRM)](https://github.com/RyanLiu112/GenPRM)
- [ThinkPRM GitHub (mukhal/ThinkPRM)](https://github.com/mukhal/thinkprm)
