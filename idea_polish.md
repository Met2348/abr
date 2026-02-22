# Idea Polish: BCR / ABR Early-Stage Notes

> Math compatibility note: formulas use standard `$...$` (inline) and `$$...$$` (display) delimiters.

## 1. Basic Idea

### 1.1 Problem Statement
Chain-of-Thought (CoT) models can output a correct final answer with an unfaithful reasoning process (post-hoc rationalization, brittle jumps, or hidden hallucinations).  
The target is not only higher answer accuracy, but also a reasoning trajectory whose internal confidence/value is behaviorally consistent with true progress.

### 1.2 Core Hypothesis
If reasoning is faithful, the estimated success value along the trajectory should evolve consistently.  
Large unjustified spikes/drops in value often indicate broken logic or hallucination-like transitions.

## 2. Baseline Method Understanding (BCR)

### 2.1 Setup
- Policy model: LLM generates reasoning tokens/steps.
- Value head: $V_\phi(h_t) \in [0,1]$, where $h_t$ is prompt + partial reasoning up to step $t$.
- Reward: sparse terminal reward (correct final answer), with optional shaping variants discussed in slides.

### 2.2 Training Objective
- Standard SFT loss:
  $$
  \mathcal{L}_{SFT}
  $$
- Bellman-consistency regularization:
  $$
  \mathcal{L}_{Bellman}=\sum_t \left(V_\phi(h_t) - (r_t + \gamma \,\bar V_\phi(h_{t+1}))\right)^2
  $$
- Joint objective:
  $$
  \mathcal{L}_{total} = \mathcal{L}_{SFT} + \lambda \,\mathcal{L}_{Bellman}
  $$

### 2.3 Intended Effect
The value trajectory becomes a self-supervised process signal:
- faithful chain: smoother/consistent value evolution
- unfaithful chain: stronger local inconsistency

## 3. Discussion Summary from PPTs

### 3.1 Strong Points in Current Discussion
- Good focus on **faithfulness gap** rather than only final accuracy.
- Good concern about **supervision leakage** when intermediate rewards directly use gold-token likelihood.
- Good observation that token-level Bellman checks may be inefficient/noisy.
- Good direction toward step-level adaptive control and explicit verification decisions.

### 3.2 Major Risks Already Identified (and important)
- Reward design can be gamed (e.g., minimize verification actions and still pass easy questions).
- If value sees strong token-level supervision, it may learn token matching rather than true future-success estimation.
- Full coupling of base model + value head + RL router can be unstable at early stage.

## 4. ABR Extension Understanding (Step-Level Adaptive Reasoning)

### 4.1 Key Design
Use a lightweight router to choose actions at each reasoning step:
- `a_gen`: continue generating reasoning step
- `a_ver`: verify consistency (invoke value-based check)
- `a_fin`: finish and answer

When verify is triggered, select a target historical step (TSS-like mechanism) and apply consistency regularization against that anchor.

### 4.2 Claimed Benefit
- Reduces always-on token-level checking cost
- Allows explicit “when to verify” control
- Potentially better efficiency-faithfulness tradeoff

### 4.3 Main Concern
Introducing RL routing too early may create training instability and confounded attribution (router failure vs value failure vs generator failure).

## 5. My Comments and Suggested Polishing

### 5.1 What I Agree With
- The project direction is meaningful and novel enough if faithfulness is operationalized carefully.
- Early focus on GSM8K + StrategyQA is appropriate.
- ABR-style step-level control is promising as a later phase.

### 5.2 What I Would Change
- Do not make raw smoothness the primary faithfulness target. Legitimate reasoning can include sharp value updates.
- Do not over-invest early in FFT/Hurst/SPC metrics for core claims; keep them as advanced diagnostics/appendix.
- Avoid gold-token interpolation reward in core method due to leakage risk.

### 5.3 Better Early-Stage Path (Pragmatic)
1. Build a stable BCR-lite baseline first (SFT + value head + weak temporal consistency).
2. Add corruption-based faithfulness supervision:
   - create minimally perturbed steps (sign flip, deleted premise, wrong substitution)
   - enforce margin: clean prefix value > corrupted prefix value.
3. Use rollout-based prefix targets for value calibration:
   - from prefix $h_t$, sample multiple continuations and use empirical success rate as value target.
4. Add ABR routing only after value calibration works, with constrained verification budget.

## 6. Metrics Recommendation (for Early Experiments)

### 6.1 Must-Have
- Final answer accuracy
- Value calibration (Brier / ECE)
- Corruption detection AUC (can value detect injected wrong step?)
- Value-drop localization around corruption point

### 6.2 Efficiency
- Tokens generated
- Number/rate of verify actions (for ABR phase)
- Accuracy under fixed compute budget

### 6.3 Optional Diagnostics
- First/second-order smoothness
- Frequency-domain and change-point analyses as supportive evidence, not primary objective

## 7. Suggested Development Stages
1. `v0`: SFT + value head + robust evaluation harness.
2. `v1`: Value calibration via prefix rollouts; corruption contrastive loss.
3. `v2`: ABR-lite with heuristic verify trigger.
4. `v3`: Learned router (PPO/REINFORCE) under explicit verification budget constraints.

## 8. Bottom Line
The ideas are promising. The highest-risk failure mode is not model size or datasets, but **objective leakage and over-coupled training**.  
A staged build with strong faithfulness diagnostics will make the project much more likely to produce publishable, defensible results.
