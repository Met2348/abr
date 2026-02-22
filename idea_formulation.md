# Idea Formulation: Bellman-Consistent Reasoning (BCR) and Adaptive Extension (ABR)

> Math compatibility note: all formulas use standard `$...$` (inline) and `$$...$$` (display) delimiters with KaTeX-friendly syntax.

## 1. Goal and Scope

We consider reasoning tasks where a model must produce both:
- a final answer, and
- a reasoning trajectory (Chain-of-Thought, CoT).

The core objective is to improve:
- **outcome quality** (final answer correctness), and
- **process faithfulness** (the reasoning path is internally consistent and not post-hoc fabrication).

---

## 2. Notation and Task Definition

Let:
- $x \in \mathcal{X}$ be an input problem.
- $y^\star \in \mathcal{Y}$ be the ground-truth final answer.
- $\tau = (s_1, s_2, \dots, s_T)$ be a reasoning trajectory (token-level or step-level sequence).
- $h_t = (x, s_{1:t})$ be the partial history at time $t$.
- $\pi_\theta(s_{t+1} \mid h_t)$ be the policy (base LLM).

We define a terminal prediction $\hat{y} = g(\tau)$ via an answer extractor $g(\cdot)$.

Terminal correctness signal:

$$
r_T = \mathbf{1}[\hat{y} = y^\star], \quad r_t = 0 \;\; \text{for } t < T.
$$

We introduce a value head:

$$
V_\phi(h_t) \in [0,1],
$$

interpreted as estimated probability of eventual correctness from state $h_t$.

---

## 3. Faithfulness via Temporal Consistency

### 3.1 Martingale-Style Consistency Principle

Under a faithful local reasoning transition (no hidden external information), expected future value should align with current value:

$$
V(h_t) \approx \mathbb{E}_{s_{t+1}\sim\pi_\theta(\cdot \mid h_t)}[V(h_{t+1})].
$$

Large unjustified value jumps are treated as potential indicators of unfaithful transitions.

### 3.2 Bellman Consistency Error

Using TD-style targets with discount $\gamma \in (0,1]$:

$$
\delta_t = V_\phi(h_t) - \left(r_t + \gamma \,\bar{V}_\phi(h_{t+1})\right),
$$

where $\bar{V}_\phi$ indicates stop-gradient on target value.

Bellman consistency loss:

$$
\mathcal{L}_{\text{Bellman}}(\theta,\phi)
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_{t=0}^{T-1} \delta_t^2
\right].
$$

---

## 4. Baseline BCR Objective

Given supervised reasoning traces $\tau^\star = (s_1^\star,\dots,s_T^\star)$, the SFT loss is:

$$
\mathcal{L}_{\text{SFT}}(\theta)
=
-\mathbb{E}_{(x,\tau^\star)}
\left[
\sum_{t=1}^{T}
\log \pi_\theta(s_t^\star \mid h_{t-1}^\star)
\right].
$$

Joint objective:

$$
\mathcal{L}_{\text{BCR}}(\theta,\phi)
=
\mathcal{L}_{\text{SFT}}(\theta)
+ \lambda_B \,\mathcal{L}_{\text{Bellman}}(\theta,\phi),
$$

with $\lambda_B > 0$ controlling process-consistency strength.

---

## 5. Leakage-Aware Reward Design

A key risk is using intermediate rewards directly tied to gold next-token likelihood
$\log \pi_\theta(s_t^\star \mid h_{t-1}^\star)$,
which may cause supervision leakage into value learning.

To avoid this, keep process targets label-light:
- terminal correctness only, or
- rollout-based empirical success targets from partial prefixes.

Prefix rollout target example:

$$
\hat{v}(h_t)
=
\frac{1}{K}
\sum_{k=1}^{K}
\mathbf{1}\!\left[g(\tau^{(k)}_{t\rightarrow T}) = y^\star\right],
$$

where $\tau^{(k)}_{t\rightarrow T}$ are $K$ sampled continuations from prefix $h_t$.

Calibration loss:

$$
\mathcal{L}_{\text{cal}}(\phi)
=
\mathbb{E}_{h_t}
\left[
\left(V_\phi(h_t)-\hat{v}(h_t)\right)^2
\right].
$$

This can be used as a replacement or supplement to TD targets.

---

## 6. Contrastive Faithfulness Signal (Optional but Recommended)

Construct corrupted prefixes $\tilde{h}_t$ by minimal semantic perturbations
(e.g., sign flip, premise drop, invalid substitution).

Margin loss:

$$
\mathcal{L}_{\text{ctr}}(\phi)
=
\mathbb{E}
\left[
\max\left(0, m - V_\phi(h_t) + V_\phi(\tilde{h}_t)\right)
\right],
$$

where $m>0$ is a margin.

Interpretation: clean partial reasoning should score higher than corrupted partial reasoning.

---

## 7. ABR: Step-Level Adaptive Verification

ABR introduces a router policy $\pi_\psi(a_t \mid S_t)$ over discrete actions:

$$
a_t \in \{\texttt{gen}, \texttt{ver}, \texttt{fin}\},
$$

with state summary $S_t$ built from query and reasoning history embeddings.

### 7.1 Action Semantics
- $\texttt{gen}$: generate next reasoning step.
- $\texttt{ver}$: perform value-consistency check and update shared representation.
- $\texttt{fin}$: stop and output final answer.

### 7.2 Target-Step Selection (TSS)

When $a_t=\texttt{ver}$, choose anchor index $k<t$ from attention weights:

$$
\alpha_{t,i}
=
\frac{\exp(e_t^\top u_i)}{\sum_{j < t}\exp(e_t^\top u_j)},
\quad
k \sim \alpha_t.
$$

Here $e_t$ is router state embedding and $u_i$ is history-step embedding.

Anchor consistency loss (step-level form):

$$
\mathcal{L}_{\text{ver}}
=
\left(V_\phi(h_t)-V_\phi(h_k)\right)^2.
$$

This is not standard one-step TD; it is a long-range consistency constraint.

### 7.3 Router Optimization

Define episode reward:

$$
R
=
\mathbf{1}[\hat{y}=y^\star]
- \beta \,N_{\texttt{ver}}
- \eta \,C_{\text{tok}},
$$

where:
- $N_{\texttt{ver}}$ is number of verify actions,
- $C_{\text{tok}}$ is token/compute cost,
- $\beta,\eta \ge 0$.

Router RL objective:

$$
\max_{\psi}\; \mathbb{E}_{\pi_\psi,\pi_\theta}[R].
$$

Practical note: use constrained tuning or budget scheduling to avoid reward hacking
(always skip verification).

---

## 8. Unified Training Objective (One Practical Form)

A practical combined loss can be:

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{SFT}}
+ \lambda_B \mathcal{L}_{\text{Bellman}}
+ \lambda_C \mathcal{L}_{\text{cal}}
+ \lambda_M \mathcal{L}_{\text{ctr}}
+ \lambda_V \mathcal{L}_{\text{ver}},
$$

with RL optimization for router $\psi$ in alternating updates.

Recommended staged optimization:
1. train $\theta,\phi$ without router RL,
2. enable router with fixed verification budget,
3. then tune $\psi$ with RL under explicit constraints.

---

## 9. Evaluation Protocol

### 9.1 Core Metrics
- Final accuracy:

$$
\text{Acc} = \frac{1}{N}\sum_{n=1}^{N}\mathbf{1}[\hat{y}^{(n)}=y^{\star(n)}].
$$

- Value calibration (e.g., Brier score):

$$
\text{Brier}
=
\frac{1}{M}\sum_{i=1}^{M}\left(V_\phi(h_i)-z_i\right)^2,
\quad z_i \in \{0,1\}.
$$

- Corruption detection AUC:
distinguish clean vs perturbed steps using value-drop features.

### 9.2 Secondary Diagnostics
- First-difference smoothness:

$$
S_1(V) = \frac{1}{T-1}\sum_{t=1}^{T-1}|V_{t+1}-V_t|.
$$

- Second-difference spike sensitivity:

$$
S_2(V) = \frac{1}{T-2}\sum_{t=1}^{T-2}|V_{t+2}-2V_{t+1}+V_t|.
$$

These are diagnostics, not sole faithfulness definitions.

---

## 10. Main Design Principles (Concise)

1. **No leakage first**: avoid value targets that directly expose gold next-token supervision.
2. **Calibration before control**: build reliable value estimates before training adaptive router policies.
3. **Faithfulness is multi-signal**: combine correctness, calibration, and corruption sensitivity.
4. **Efficiency under constraints**: evaluate compute-faithfulness tradeoff with explicit budgets.

---

## 11. Minimal Formal Claim Template (for future paper draft)

If value estimator calibration error is bounded and long-range consistency error is reduced, then:
- trajectory-level faithfulness indicators improve (lower corruption AUC error / better localization),
- while maintaining or improving final-answer accuracy under fixed compute budget.

This is an empirical claim framework to validate via ablation, not yet a theorem.
