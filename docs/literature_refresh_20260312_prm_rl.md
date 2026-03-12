# PRM + RL Adaptive Compute: Literature Refresh (2024–2026)

Compiled 2026-03-12. Covers 8 topics with papers from 2024–2025, focused on Process Reward
Models, RL training, and adaptive compute for a Qwen2.5-Math-PRM-7B frozen backbone + MLP
value head setup targeting ProcessBench F1 improvement.

---

## 1. GRPO / Group Relative Policy Optimization for Math Reasoning

### Core papers

**DeepSeekMath** (Shao et al., Feb 2024). arXiv:2402.03300.
- Introduced GRPO as a memory-efficient PPO variant. Eliminates critic/value model.
  Instead estimates baseline from group of G sampled responses (z-normalizes rewards
  within the group: subtract mean, divide by std).
- Two modes: *outcome supervision* (final-answer reward only) and *process supervision*
  (step-level rewards). Process supervision variant modifies advantage: normalized reward
  relative to all step rewards across all group trajectories; advantage for step t =
  sum of normalized rewards from t onward.
- GRPO+PS > GRPO+OS in DeepSeekMath experiments.
- No separate critic model → ~50% memory savings vs PPO.
- KL divergence added directly to the loss (not to reward), unlike PPO.

**DeepSeek-R1** (Guo et al., Jan 2025). arXiv:2501.12948.
- Uses GRPO with outcome-only rewards (RLVR). Deliberately avoids PRM and MCTS.
- Reasoning emerges spontaneously from RL without SFT warmup (R1-Zero).
- AIME 2024: 15.6% → 77.9% pass@1 during RL training.
- Avoided neural reward models entirely to prevent reward hacking at scale.

**"GRPO Is Secretly a Process Reward Model"** (Sullivan, Sep 2025).
arXiv / OpenReview PDF.
- Theoretical proof: trajectory-level GRPO advantage implicitly defines per-step credit.
- Proposes two-level GRPO: trajectory-level + step-level advantage; step-level groups
  formed by similarity of steps across trajectories.
- Only a handful of papers in the literature use GRPO with explicit step-level PRM
  signals (Shao 2024, Yang 2025, Feng 2025). All require modified advantage computation.

**Scaf-GRPO** (Zhang et al., Oct 2025). arXiv:2510.19807.
- Addresses "learning cliff": when a model consistently fails hard problems, all rewards
  are 0 and GRPO advantage collapses (no gradient signal).
- Fix: inject minimal ground-truth reasoning scaffolding only when learning has plateaued.
- Qwen2.5-Math-7B on AIME24: +44.3% relative gain over vanilla GRPO.

### Actionable takeaways for our project
- If training LoRA backbone with GRPO, use process supervision mode (step rewards from
  value head). Advantage = sum of normalized future step rewards from position t onward.
- GRPO eliminates the need for a separate critic model — ideal for our setup where the
  MLP value head already serves as the critic.
- Use verifiable math rewards for RL training, not a neural reward model, to prevent
  reward hacking at scale (DeepSeek-R1's lesson).
- If zero-reward collapse on hard problems: Scaf-GRPO scaffolding is the fix.

---

## 2. Implicit PRM

### Core papers

**"Free Process Rewards without Process Labels"** (Yuan et al., Dec 2024).
arXiv:2412.01981. GitHub: https://github.com/PRIME-RL/ImplicitPRM

- Core insight: parameterize outcome reward as `r_θ(y) = β · log(π_θ(y) / π_ref(y))`.
  Training an ORM on response-level labels only with this parameterization automatically
  yields an implicit PRM — token-level log-ratios decompose as step scores.
- Any loss function works (DPO, NCA, KTO, cross-entropy). DPO achieves best overall
  performance.
- Critical finding: adding Math-Shepherd step labels on top of outcome-only training
  brings NO additional gain. Step annotation budget is not necessary.
- Works with 1 response per instruction (extreme imbalance regime).
- Evaluation: MATH500 Best-of-N vs 6 competing PRMs/ORMs. Implicit PRM (DPO) wins.
  Claimed SOTA PRM for Llama-3.1 backbone family at time of release.

**PRIME: Process Reinforcement through Implicit Rewards** (Cui et al., Feb 2025).
arXiv:2502.01456. GitHub: https://github.com/PRIME-RL/PRIME
HuggingFace: PRIME-RL/Eurus-2-7B-PRIME

- Extends implicit PRM to online RL: PRM updated during rollouts using only outcome
  labels (no step annotation).
- PRM initialized from SFT model (same as policy). Reference log-probs from SFT
  retained throughout training.
- Advantage estimation with RLOO: outcome rewards use RLOO directly; process rewards
  apply leave-one-out baseline then discounted return.
- Results (starting from Qwen2.5-Math-7B-Base):
  - +15.1% average across reasoning benchmarks vs SFT model.
  - Eurus-2-7B-PRIME surpasses Qwen2.5-Math-7B-Instruct on 7 benchmarks using only
    10% of training data (230K SFT + 150K RL).
  - AIME24 pass@1: 26.7%, beating GPT-4o.
  - 2.5x faster training vs sparse-reward baselines.
  - +16.7% average improvement, +20% on AMC/AIME.
- EurusPRM (same team): SOTA open-source PRM for Best-of-N, two-stage pipeline on
  Qwen2.5-Math-7B-Instruct.
- Implemented in veRL framework (in veRL main branch as of 2025-03-12).

**iStar** (Agentic implicit step rewards, Sep 2025). arXiv:2509.19199.
- Applies implicit PRM to multi-turn agent tasks: derives step credit without annotations
  from a trajectory-based DPO objective.
- Uses step-level (not token-level) implicit rewards for stability in multi-turn RL.
- Up to +48% goal completion in open-ended social interaction benchmarks.

### Actionable takeaways for our project
- After LoRA run stabilizes (LORA_S4), compute step scores as:
    `score(step) = mean(β · log(π_LoRA(t) / π_ref(t)) for t in step_tokens)`
  This IS an implicit PRM requiring no additional training.
- Compare implicit PRM vs MLP value head on ProcessBench F1. If competitive, eliminates
  the value head entirely.
- Use DPO as training objective (beats NCA, KTO, CE in Yuan et al.).
- Do NOT annotate step labels — outcome labels suffice per the empirical finding.

---

## 3. ABR / Adaptive Branching / Early Exit for LLM Inference

### Core papers

**AB-MCTS: "Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive
Branching Tree Search"** (Sakana AI, Mar 2025). arXiv:2503.04412.
GitHub: https://github.com/SakanaAI/treequest (TreeQuest library)

- Core problem: fixed-branching MCTS wastes compute on already-solved branches.
  AB-MCTS introduces a "GEN node" at every tree node: selects between expanding a new
  child vs. refining an existing one via Thompson Sampling.
- Two variants:
  - AB-MCTS-M (Mixed): Bayesian mixed-effects model estimating quality of new vs.
    refinement actions; each child is its own random intercept group.
  - AB-MCTS-A (Aggregation): maintains separate CONT node (exploitation) and GEN node
    (exploration) explicitly.
- Consistently outperforms repeated sampling and standard MCTS. Gains scale with larger
  generation budgets.
- Multi-LLM extension (Jul 2025): o4-mini + Gemini-2.5-Pro + DeepSeek-R1-0528
  cooperative inference; strong results on ARC-AGI-2.

**Bandit-Based Budget Allocation** (Zuo et al., Jun 2025). arXiv:2506.12721.
- Frames per-query compute allocation as a bandit problem. Estimates query difficulty
  on the fly and allocates more compute to instances likely to benefit.
- Uses Qwen2.5-Math-PRM-7B (our exact backbone) as the reward oracle in experiments.
- +11.10% absolute improvement (15.04% relative) on MATH-500 by adaptive vs. uniform.
- Key insight: on easy problems, sequential revision > parallel resampling; on hard
  problems, resampling + tree search > sequential revision.

**"Scaling LLM Test-Time Compute Optimally"** (Snell et al., 2024). ICLR 2025.
https://proceedings.iclr.cc/paper_files/paper/2025/file/1b623663fd9b874366f3ce019fdfdd44-Paper-Conference.pdf
- Compute-optimal adaptive budget allocation: prescribes problem-dependent budgeting
  based on PRM scores.
- Adaptive strategy improves compute efficiency 2-4x over uniform allocation.

**Survey: "Reasoning on a Budget"** (2025). arXiv:2507.02076.
- Comprehensive taxonomy: dynamic iterative reasoning, bandit-based scheduling,
  verifier-guided control, difficulty-aware routing, latent convergence / early exit.

**AdaInfer** (IJCAI 2025). **HELIOS** (2025).
- AdaInfer: adaptive layer skipping at inference without modifying parameters.
  Top-1 accuracy within <1% of dense model.
- HELIOS: EE-LLMs that skip later layers when confident early; targets latency without
  accuracy loss. Distinct from our ABR controller use case but relevant for step-early-exit.

### Actionable takeaways for our project
- Our ABR-lite offline simulation (F2, binary F1=0.863) is directly aligned with the
  bandit paradigm validated in Zuo et al. — and uses our exact backbone (Qwen2.5-Math-
  PRM-7B). This validates our F2 approach against the literature.
- Thompson Sampling over step score distributions is a principled alternative to our
  hard-threshold stop policy (current τ=0.35). Worth A/B testing in F3.
- Bandit difficulty → compute budget mapping: Zuo et al. achieved +11-15% on MATH-500
  with our backbone. Direct reference for F3/F4 controller design.
- AB-MCTS GEN-node idea: treat each solution step as a tree node; decide whether to
  continue sampling (go deeper) or emit verdict (terminate) based on value head
  confidence. This maps cleanly to our ABR controller's binary continue/stop decision.

---

## 4. Contrastive Loss for PRMs

### Core papers

**PQM: "Process Reward Model with Q-Value Rankings"** (Li & Li, Oct 2024).
arXiv:2410.11287. GitHub: https://github.com/WindyLee0822/Process_Q_Model. ICLR 2025.

- Reframes PRM as Markov Decision Process. Q(s,a) = probability of reaching correct
  final answer from state s.
- Theorem 3.5: correct steps must have strictly higher Q-value than the first incorrect
  step — motivates a comparative ranking loss.
- Margin-based ranking loss enforces `Q(correct_step) - Q(first_incorrect_step) > margin`.
  Unlike BCE which treats each step independently.
- Supports 4 loss modes: `rank` (comparative), `orm`, `mse`, `bce`. Comparative wins.
- Results: BCE 39.8% → PQM rank 51.4% accuracy on MATH500, +11.6% absolute.
  Consistent gains across Llama-3-70B-Instruct and other backbones.
- Trained on PRM800K and Math-Shepherd. HuggingFace checkpoints released.

**BiPRM: "The Bidirectional Process Reward Model"** (Aug 2025). arXiv:2508.01682.
- Parallel R2L (right-to-left) evaluation stream via prompt reversal + learned gating
  mechanism (0.3% parameter overhead, 5% latency overhead).
- Gating: adaptive fusion of L2R (forward) and R2L (backward) scores.
- +10.6% average relative gain across 54 solution-level evaluation configurations.
- +37.7% across 12 step-level error detection scenarios.
- R2L catches errors where L2R is deceived by superficial coherence but logical fallacy
  is visible retrospectively.

**Coarse-to-Fine PRM** (Jan 2025). arXiv:2501.13622.
- Direct comparison of BCE (ShepHerd), MSE (ReSTMCTS*), Q-ranking (PQM).
  Q-ranking wins across all settings.
- Adds coarse-grained (solution-level) and fine-grained (step-level) joint training.

### Actionable takeaways for our project
- Replace BCE in MLP value head training with margin-based ranking loss (PQM-style).
  Wire into `training.py:_compute_single_pair_objective()`. For each pair:
  enforce `score(chosen_step) - score(rejected_step) > margin`.
  +11.6% absolute on MATH500 is a large reported gain.
- BiPRM R2L trick is inference-only (zero training cost): run value head on reversed
  solution and average with forward scores. Test on ProcessBench F1 directly.
  +37.7% on step-level error detection is the largest single gain in this section.
- PQM code is open source (GitHub above) — reference implementation for margin loss.

---

## 5. BiRM / Bidirectional Reward Modeling

### Core papers

**BiRM: "Better Process Supervision with Bi-directional Rewarding Signals"**
(Chen et al., Mar 2025). arXiv:2503.04618. ACL 2025 Findings.

- Motivated by A*: PRM estimates h(n) (future cost heuristic), VM estimates g(n)
  (cost so far). A* uses f(n) = g(n) + h(n). BiRM estimates both simultaneously.
- Observation: standard PRM is better at detecting errors in early steps; VM is better
  in later steps. BiRM outperforms both at all stages.
- Architecture: dual output head from shared backbone hidden states:
    backbone → hidden
      ├── reward_head (MLP) → r_score   [correctness label, BCE loss]
      └── value_head  (MLP) → v_score   [future success probability, MSE loss]
    Inference: weighted combination of r_score and v_score.
- Training data: GSM8K + MATH, ~225K solutions per base model. Reward labels from
  DeepSeek-V3; value labels from Math-Shepherd soft-label (MC completion rate).
- Applications: solution re-ranking and trajectory searching.

**BiPRM** (separate work, see Topic 4): bidirectionality = L2R + R2L streams, not
process + terminal. Different concept, same name prefix.

**Our existing terminal_bce is BiRM-like.**
`training.py:1309` applies extra BCE on terminal steps via `lambda_terminal_bce`.
PBR19 (λ=0.25, MATH F1=0.683, GSM F1=0.778) and PBR26 (λ=0.25, MATH F1=0.686)
confirm the combination helps.

### Actionable takeaways for our project
- The gap between our terminal_bce and BiRM: we use hard labels (0/1 BCE) for terminal
  supervision; BiRM uses soft MC-estimated labels for the value head (MSE loss).
- Soft terminal labels fix: replace `BCE(chosen→1, rejected→0)` in `training.py:1309`
  with `MSE(chosen→MC_prob, rejected→MC_prob)`. Use pair confidence as proxy for MC
  probability. This is the "BiRM soft terminal labels" technique already flagged in
  MEMORY.md.
- Chen et al. separate reward labels (DeepSeek-V3 correctness) from value labels
  (Math-Shepherd MC rate). Our current setup uses DPO pairs for both — consider
  sourcing label types separately.
- Our `dual_head` routing in `training.py:871` matches BiRM architecture exactly.
  The remaining gap is solely the soft label signal for the value head.

---

## 6. Step-Level Verification + RL (R-PRM, OmegaPRM, MCTS)

### Core papers

**OmegaPRM** (Google DeepMind, Jun 2024). arXiv:2406.06592. ICLR 2025.

- Divide-and-conquer MCTS + binary search to find first error in CoT efficiently.
  Balances positive/negative examples automatically.
- Collects 1.5M+ process annotations without human intervention.
- Results (Gemini Pro): MATH500: 51% → 69.4%; GSM8K: 86.4% → 93.6%.
- Outperforms PRM800K (human-annotated) and Math-Shepherd (MC sampling).

**R-PRM: "Reasoning-Driven Process Reward Modeling"** (She et al., Mar 2025).
arXiv:2503.21295. GitHub: https://github.com/NJUNLP/R-PRM

- Two-stage: SFT on PRM800K samples with chain-of-thought step analysis (prompted from
  stronger LLMs), then DPO on preference pairs from multi-path sampling.
- Generative PRM: produces reasoning critiques AND binary decisions per step.
- ProcessBench F1: +8.7 over Qwen2.5-Math-7B-PRM800K at same data scale.
  F1 improves 62.8 (N=2 rollouts) → 67.6 (N=4) → wider gap at N=16.
  At N=16, beats LLaMA-3.3-70B-Instruct by +13.1 F1.
- Best-of-N: +8.6 accuracy pts on 6 math benchmarks. Guided Search: +8.4.

**Dyve: "Thinking Fast and Slow for Dynamic Process Verification"** (Feb 2025).
arXiv:2502.11157. ACL 2025.

- System 1 (fast): immediate token-level confirmation for straightforward steps.
- System 2 (slow): comprehensive analysis for complex steps.
- Data pipeline: OmegaPRM → 1.2M rollouts → DeepSeek-V3 consensus filtering (removes
  ~50% noisy rollouts) → 117K high-quality balanced examples.
- Outperforms existing process verifiers on ProcessBench and MATH.
- Key contribution: consensus filtering (MC + LLM judge agreement) is the data quality
  lever — same technique Qwen PRM uses.

**ThinkPRM: "Process Reward Models That Think"** (MIT/Mila/LG, Apr 2025).
arXiv:2504.16828.

- Generative PRM: verifies each step by generating a long verification chain-of-thought.
- Fine-tuned on 8K process labels (1% of PRM800K) or 1K synthetic CoT examples.
- ThinkPRM-14B beats discriminative PRMs trained on 100x more data on ProcessBench.
- Out-of-domain: +8% on GPQA-Diamond, +4.5% on LiveCodeBench vs PRM800K-trained models.
- Under same token budget: +7.2% on ProcessBench subset vs LLM-as-a-Judge.

**GenPRM** (Apr 2025). arXiv:2504.00891.

- Generative PRM with CoT reasoning + code verification before step judgment.
- Proposes Relative Progress Estimation (RPE) for process supervision labels and a
  rationale synthesis framework.
- 23K training examples from MATH only.
- 1.5B GenPRM beats GPT-4o on ProcessBench; 7B GenPRM beats Qwen2.5-Math-PRM-72B.
- F1 scaling: 1.5B→7B: 57.3→75.2 and 63.4→80.5 (large); 7B→32B marginal (7B is
  the sweet spot).
- Enables critic model role (policy refinement), not just scoring.

### Actionable takeaways for our project
- OmegaPRM consensus filtering (confirmed by Dyve): MC labels alone are noisy. Running
  DeepSeek-V3 or Qwen-72B as judge to filter disagreements → 50% data reduction but
  higher quality. This is the primary DATA QUALITY lever for our ProcessBench plateau.
- GenPRM 7B beats Qwen2.5-Math-PRM-72B using generative CoT (our community target is
  ~73.5 MATH F1). Not directly applicable to our discriminative MLP, but RPE labels
  from GenPRM could replace Math-Shepherd MC labels in our training data.
- R-PRM N-scaling (F1 improves with N rollouts per step) suggests step-level multi-path
  aggregation is more effective than seed-level ensembling. Our 4-seed ensemble
  underperforms cold seed=42 — likely because we aggregate at the run level rather than
  the step level.
- ThinkPRM data efficiency (8K labels beats 100x more): argues for quality of DPO pairs
  over raw quantity — consistent with our observation that DPO data is the minimum
  requirement for correct ProcessBench transfer.

---

## 7. Offline RL for Verifier / Controller Training

### Core papers

**"Offline Reinforcement Learning for LLM Multi-step Reasoning"** (Wang et al.).
ACL 2025 Findings. https://aclanthology.org/2025.findings-acl.464.pdf

- Frames multi-step reasoning as sequential decision process. Simple rejection sampling
  over offline datasets fails on failure traces (wastes negative signal). Offline RL
  propagates negative rewards backward via Q-learning style updates.
- Direct offline RL over LLM multi-step traces improves over rejection sampling.

**PCL-Reasoner-V1.5** (2025). arXiv:2601.14716.

- 32B Qwen2.5-based math model: SFT then novel offline RL stage.
- Offline RL avoids instabilities and compute costs of online RL.
- AIME 2024: 90.9%; AIME 2025: 85.6%.
- Key argument: distribution mismatch in offline RL is small when starting from strong
  SFT models — standard concern is reduced in practice.

**"Bridging Offline and Online RL for LLMs"** (Jun 2025). arXiv:2506.21495.

- Systematic comparison. Key offline failure modes: length collapse and entropy blow-up
  when importance ratios are used in large action spaces.
- DPO = offline RL with binary rewards. GRPO = online RL with group rewards.
- Recommended hybrid: start offline (DPO) → transition to online (GRPO) for stability.

**Our Phase F ABR-lite simulation IS offline RL** over pre-scored PBR19 traces:
- Stop policy trained/evaluated on fixed rollouts with PRM score as state, step index
  as time, correctness as terminal reward.
- Binary detection F1=0.863 vs fixed-schedule best F1=0.545. Compute savings: 71%
  steps processed on average (29% savings). τ=0.35.
- Results: `assets/artifacts/phase_f_simulation/pbr19_math_abr_lite_0312/`

**Strategic Bandit Compute Allocation** (Zuo et al., see Topic 3) also qualifies as
offline learning: learns difficulty-to-budget mapping from pre-scored examples.

### Actionable takeaways for our project
- F2 offline simulation is the correct foundation. F3 can extend to online RL: deploy
  ABR controller in live generation loop, update via GRPO with step-level value head
  reward as process signal.
- Stop policy offline Q-learning: given pre-scored PBR19 artifacts, fit
  Q(step_index, value_score) → {continue, stop} via tabular fitted Q-iteration.
  This is simpler and more stable than policy gradient for the binary decision.
- GSM8K generator-shift vulnerability (worst_logo_f1=0.0 at F1) is a distribution
  mismatch problem. Mitigate by including generator-shifted traces in offline training
  data before F3 deployment.
- Avoid length collapse: add minimum-step constraint in stop policy reward (never
  terminate before step 2, regardless of score).
- Importance-ratio corrections for distribution shift cause variance explosion in large
  action spaces (Bridging paper). Prefer offline Q-iteration over importance-weighted
  policy gradient for the stop policy.
- PCL-Reasoner confirms: starting from strong SFT (our PBR19 is strong), distribution
  mismatch in offline RL is manageable.

---

## 8. EDU-PRM / Entropy-Based Step Boundaries

### Core papers

**EDU-PRM: "More Bang for the Buck: Process Reward Modeling with Entropy-Driven
Uncertainty"** (Mar 2025). arXiv:2503.22233.

- Core mechanism: use token-level predictive entropy (threshold = 1.0 nats empirically)
  to identify high-uncertainty tokens as natural step boundaries.
  High-entropy tokens: "however", "thus", "because", "suppose" — logical connectors
  where reasoning genuinely branches.
  Low-entropy tokens: affixes, code snippets, math expression components — construction
  tokens that do not define reasoning paths.
- Key finding: only a few high-entropy tokens guide reasoning paths. Training on 80% of
  low-entropy tokens only degrades performance significantly.
- Step boundary algorithm:
    1. Compute logit entropy H = -Σ p·log(p) at each token during generation.
    2. Flag tokens where H > 1.0 as branching points (step boundaries).
    3. Augment with symbol whitelist to avoid punctuation artifacts.
    4. At branching points, sample both top-1 and top-2 tokens to explore diverging paths.
    5. Hierarchical path exploration: greedy continuation + top-k at critical nodes.
- Results on ProcessBench:
    - Outperforms Math-Shepherd PRM and OmegaPRM baselines.
    - With 7,500 training queries (1.5% of data Qwen uses): accuracy 71.1%
      vs 71.6% for Qwen2.5-72B-PRM trained on 500K queries.
    - 98% reduction in query cost vs prior methods.
    - 32% reduction in token usage on generative tasks.
    - Accuracy boost: 64.7% → 67.3%.
- Entropy threshold 1.0 is the empirically validated default.

**Entropy-Regularized PRM (ER-PRM)** (Dec 2024). arXiv:2412.11006.
- Distinct from EDU-PRM: KL-regularized MDP for PRM training stability (prevents
  policy shift during reward optimization — not step boundary detection).
- MATH: +2-3% under Best-of-N; GSM8K: +1%. Complementary to EDU-PRM.

### Our current setup vs EDU-PRM

| Step boundary method          | Granularity         | Quality note                                     |
|-------------------------------|---------------------|--------------------------------------------------|
| `\n\n` (Math-Shepherd default) | Fixed structural   | Unequal semantic weight per step                 |
| EDU entropy threshold=1.0     | Semantic/logical   | Aligned with actual logical transitions          |

### Implementation plan for entropy boundaries

```python
# In training.py: build_pair_tokenized_cache() variant
# During backbone forward pass in generation mode:
with torch.no_grad():
    logits = backbone(input_ids).logits          # [seq, vocab]
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * probs.log()).sum(-1)     # [seq]
    boundary_mask = entropy > 1.0               # [seq], bool
    # Augment with symbol whitelist to suppress punctuation artifacts
    # Use boundary_mask to define step spans instead of \n\n splits
```

### ProcessBench F1 context

| Model                               | MATH F1 | Notes                            |
|-------------------------------------|---------|----------------------------------|
| Math-Shepherd PRM                   | <0.686  | EDU-PRM outperforms              |
| OmegaPRM                            | <0.686  | EDU-PRM outperforms              |
| Our PBR26 (MLP value head, current) | 0.686   | Current best                     |
| Qwen2.5-Math-PRM-7B (community)     | ~0.735  | Target; uses 500K queries        |
| EDU-PRM (7.5K queries, 72B model)   | ~0.711  | 1.5% of Qwen's data budget       |

### Actionable takeaways for our project
- Highest-priority data preprocessing change. Replace `\n\n` boundaries with entropy-
  threshold boundaries in `training.py:build_pair_tokenized_cache()`. Store boundary
  token indices alongside input_ids in the tokenized cache.
- EDU sampling generates cheap training pairs: at high-entropy branch points, sample
  top-1 and top-2 to create natural contrastive pairs without MCTS rollouts. This
  produces DPO-style pairs at near-zero marginal cost vs. existing Math-Shepherd MC.
- 32% token savings reduces feature cache size and LoRA training cost downstream.
- Implementation detail: add symbol whitelist (skip `(`, `)`, `,`, `.`, etc.) to
  prevent punctuation positions from being flagged as boundaries.
- Entropy threshold 1.0 is the empirically validated default; may need tuning if
  backbone tokenizer differs from EDU-PRM's (both Qwen-family, should transfer).
- Highest expected impact for closing the remaining 5 F1-point gap (0.686 → 0.735).

---

## Summary: Priority Implementation Queue

Expected ProcessBench MATH F1 impact × implementation cost:

| Pri | Technique                                     | Expected MATH F1 Gain | Implementation Location                              |
|-----|-----------------------------------------------|-----------------------|------------------------------------------------------|
| 1   | EDU entropy step boundaries (data prepro)     | +3–5 pts              | `training.py:build_pair_tokenized_cache()`           |
| 2   | Implicit PRM score from LoRA run              | +2–4 pts              | `runtime.py` post-LORA_S4                            |
| 3   | PQM margin/ranking loss (replace BCE)         | +2–3 pts              | `training.py:_compute_single_pair_objective()`       |
| 4   | BiRM soft terminal labels (MSE not BCE)       | +1–2 pts              | `training.py:1309`                                   |
| 5   | BiPRM R2L inference trick (zero training)     | +1–2 pts              | `benchmark_eval.py` (no training needed)             |
| 6   | Bandit ABR controller (saves compute)         | saves ~29% compute    | `scripts/phase_f_abr_lite_simulation.py`             |
| 7   | Dyve-style consensus filtering (data quality) | data quality lever    | new data pipeline script                             |
| 8   | GRPO process supervision for LoRA             | unknown               | if LORA_S4 stabilizes                                |

Notes:
- Items 1–4 work within the frozen backbone + MLP value head paradigm.
- Item 2 (Implicit PRM) requires LORA_S4 to complete first.
- Items 6–7 are Phase F / data pipeline work independent of value head training.
- Items 3–5 can be run in parallel as ablations on the PBR26 base configuration.

---

## Key Paper References

| Topic | Paper | arXiv / URL |
|-------|-------|-------------|
| GRPO | DeepSeekMath (Shao et al., 2024) | https://arxiv.org/abs/2402.03300 |
| GRPO | DeepSeek-R1 (Guo et al., 2025) | https://arxiv.org/abs/2501.12948 |
| GRPO | Scaf-GRPO (Zhang et al., 2025) | https://arxiv.org/abs/2510.19807 |
| Implicit PRM | Free Process Rewards (Yuan et al., 2024) | https://arxiv.org/abs/2412.01981 |
| Implicit PRM | PRIME (Cui et al., 2025) | https://arxiv.org/abs/2502.01456 |
| ABR | AB-MCTS Wider or Deeper (Sakana AI, 2025) | https://arxiv.org/abs/2503.04412 |
| ABR | Bandit Compute Allocation (Zuo et al., 2025) | https://arxiv.org/abs/2506.12721 |
| ABR | Compute-Optimal Scaling (Snell et al., ICLR 2025) | https://proceedings.iclr.cc/paper_files/paper/2025/file/1b623663fd9b874366f3ce019fdfdd44-Paper-Conference.pdf |
| ABR | Survey: Reasoning on a Budget (2025) | https://arxiv.org/pdf/2507.02076 |
| Contrastive | PQM Q-Value Rankings (Li & Li, 2024) | https://arxiv.org/abs/2410.11287 |
| Contrastive | BiPRM Bidirectional L2R+R2L (2025) | https://arxiv.org/abs/2508.01682 |
| Contrastive | Coarse-to-Fine PRM (2025) | https://arxiv.org/abs/2501.13622 |
| BiRM | BiRM Process+Terminal (Chen et al., 2025) | https://arxiv.org/abs/2503.04618 |
| Step RL | OmegaPRM (Google DeepMind, 2024) | https://arxiv.org/abs/2406.06592 |
| Step RL | R-PRM (She et al., 2025) | https://arxiv.org/abs/2503.21295 |
| Step RL | Dyve Fast+Slow Verification (2025) | https://arxiv.org/abs/2502.11157 |
| Step RL | ThinkPRM (MIT/Mila/LG, 2025) | https://arxiv.org/abs/2504.16828 |
| Step RL | GenPRM (2025) | https://arxiv.org/abs/2504.00891 |
| Offline RL | Offline RL Multi-step Reasoning (ACL 2025) | https://aclanthology.org/2025.findings-acl.464.pdf |
| Offline RL | Bridging Offline/Online RL (2025) | https://arxiv.org/abs/2506.21495 |
| EDU-PRM | EDU-PRM Entropy Boundaries (2025) | https://arxiv.org/abs/2503.22233 |
| EDU-PRM | Entropy-Regularized PRM (2024) | https://arxiv.org/abs/2412.11006 |

### GitHub repositories

| Repo | URL |
|------|-----|
| PRIME-RL/PRIME | https://github.com/PRIME-RL/PRIME |
| PRIME-RL/ImplicitPRM | https://github.com/PRIME-RL/ImplicitPRM |
| SakanaAI/treequest (AB-MCTS) | https://github.com/SakanaAI/treequest |
| WindyLee0822/Process_Q_Model (PQM) | https://github.com/WindyLee0822/Process_Q_Model |
| NJUNLP/R-PRM | https://github.com/NJUNLP/R-PRM |
