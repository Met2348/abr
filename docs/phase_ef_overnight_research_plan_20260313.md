# Phase E/F Research & Experiment Planning — 2026-03-13

**Context**: PBR26 (frozen, MATH F1=0.686) and PBR32 (LoRA r=8 all-28, PBR12 data, MATH F1=0.689) are the current ceiling. LoRA on PBR26 data plateaus at 0.656-0.666, below frozen. All LoRA variants (PBR33-37) confirm that data quality, not architecture, is the bottleneck.

**10-hour budget**: 4× A100 80GB. Plan covers 5 experiments with clear hypotheses, expected timings, and go/no-go gates.

---

## Section 1: Literature Review — Consensus Filtering for PRM Training

### 1.1 Qwen2.5-Math-PRM Consensus Filtering

**Source**: Zhang et al. (2025), the Qwen2.5-Math-PRM-7B technical report.

The core mechanism: label every training pair using *two independent methods* (LLM-as-judge + MathShepherd MC estimation), then **discard pairs where the two methods disagree**. For the Qwen2.5 data pipeline:

- 860K raw samples labeled by both methods.
- 40% discarded (label disagreement).
- Remaining 516K filtered samples are cleaner.
- Key finding: **after filtering, hard labels substantially outperform soft labels** on ProcessBench. Before filtering, the two are nearly equivalent — the noise masked the hard/soft distinction entirely.
- Community Qwen2.5-Math-PRM-7B achieves MATH F1 ≈ 0.735 on ProcessBench with this recipe.

**Application to our setup**: We have DPO pairs (from arXiv data) and MS pairs (from Math-Shepherd). The two sources play the role of the two independent labelers. A pair where DPO labels a step as good but MS labels it as bad (or vice versa) is a *noise-conflicted pair* — exactly what consensus filtering removes.

**Practical implementation**:
- For each MS pair, assign a `confidence` score = MC probability of the chosen step.
- Filter to the top 50% by `|confidence - 0.5|` (i.e., keep high-certainty pairs).
- Alternative: keep only pairs where *both* DPO and MS agree on directionality (DPO preferred > MS preferred = both say "chosen is better").
- Expected: removing ~40% of the noisiest MS pairs should improve ProcessBench MATH F1 for the frozen MLP head by ~0.01-0.02.

### 1.2 PROF: Process Consistency Filter (arXiv:2509.03403)

**Source**: "Beyond Correctness: Harmonizing Process and Outcome Rewards through RL Training" (PROF).

PROF addresses the online RL version of the same problem: in GRPO training, the PRM assigns high process scores to some incorrect rollouts (reward hacking). Instead of mixing PRM + ORM naively in the loss, PROF performs **consistency-driven sample selection**:

- Keep correct rollouts that *also* have high mean PRM score (consistent positive signal).
- Keep incorrect rollouts that *also* have low mean PRM score (consistent negative signal).
- Discard rollouts where PRM and ORM contradict.

PROF improves final RL accuracy by **4%+ vs naive PRM+ORM blending**, showing that *agreeing on which samples to train on* matters more than how to weight the two losses. This is directly relevant to Phase F GRPO+PRM experiments: rather than a fixed `lambda_process`, use PROF-style selection to gate which rollouts the PRM process reward is applied to.

### 1.3 ActPRM: Active Learning for PRM Data (arXiv:2504.10559)

**Source**: "Efficient Process Reward Model Training via Active Learning".

ActPRM achieves SOTA on ProcessBench (75.0% F1) and PRMBench (65.5%) using only 20% of annotation costs. The key insight: instead of labeling all steps uniformly, use *epistemic uncertainty* to identify which unlabeled steps would most improve the model if labeled.

**Practical relevance**: We cannot run full ActPRM without a secondary oracle. However, the passive filtering analog is to keep pairs where the current value head is *most uncertain* (score near 0.5) — these are the hardest and most informative examples. This inverts the Qwen consensus filter: instead of keeping easy high-confidence pairs, it keeps hard uncertain pairs.

**When to use which**:
- Consensus filtering (keep high-confidence): use when current model is unreliable (early training, noisy backbone).
- ActPRM-style uncertainty sampling (keep uncertain): use when current model is already decent and needs to learn harder decision boundaries.
- Our PBR26 frozen model at ~0.686 MATH F1 is "decent" — uncertainty sampling may be better for the next data generation round.

---

## Section 2: PAV — Process Advantage Verifiers (arXiv:2410.08146, ICLR 2025)

**Source**: "Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning", Lightman et al., ICLR 2025.

### 2.1 Core Concept

PAV defines the training target for step `t` as the *advantage*:

```
PAV(s_t, a_t) = Q(s_t, a_t) - V(s_t)
              = MC_prover(s_{t+1}) - MC_prover(s_t)
```

Where `MC_prover(s_t)` is the probability that a *separate prover policy* (different from the training base policy) reaches a correct final answer from state `s_t`.

The critical subtlety: the prover policy must be **complementary** to the base policy — strong enough to complete from correct prefixes, but not so strong it can recover from wrong ones. A prover that is too strong scores 1.0 from both good and bad prefixes (no signal). A prover that is too weak scores 0.0 from both.

### 2.2 Training Targets

Using PAV differences as training targets:
```
target_t = MC_prover(after step t) - MC_prover(before step t)
         = p(correct | s_{t+1}) - p(correct | s_t)
```

This can be positive (good step) or negative (bad step), and specifically encodes *how much progress the step makes*, not just whether it is correct.

### 2.3 Empirical Results

- PAVs are **5× more compute efficient** than ORMs at the same test-time compute budget.
- **10% more accurate** in best-of-N search.
- **6× more sample efficient** for online RL.
- The key to PAV's gains vs standard PRM: standard PRMs train on hard 0/1 labels (is this step correct?), while PAV trains on soft progress labels (how much better is the next state than the current state?).

### 2.4 Implementation for Our Setting

Our pair format is already compatible with a version of PAV. Each pair `(chosen_prefix, rejected_prefix)` at step `t` corresponds to:
- `chosen_prefix` = solution up to and including a correct step at `t`.
- `rejected_prefix` = solution with an error at step `t`.

The PAV target for the chosen/rejected split:
```
chosen_PAV  = MC(chosen_{t+1}) - MC(chosen_t)
rejected_PAV = MC(rejected_{t+1}) - MC(rejected_t)
```

In practice, training the value head with a **regression loss on PAV scores** (instead of binary classification `chosen=1, rejected=0`) would remove the hard-label noise. We have the MC estimates in the Math-Shepherd data — `mc_prob` is the field to use.

**Experiment PBR38C (see Section 6.3) will test PAV-style regression targets on existing pairs.**

---

## Section 3: RPE Labels — GenPRM (arXiv:2504.00891, AAAI 2026)

**Source**: "GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning", AAAI 2026.

### 3.1 Relative Progress Estimation (RPE)

GenPRM introduces two RPE variants:

**RPE variant 1 (difference)**:
```
P_t = MC(s_t, a_t) - MC(s_t)   (same as PAV difference)
```

**RPE (original, ratio-normalized)**:
```
RPE(t) = (MC(s_{t+1}) - MC(s_t)) / max(MC(s_t), ε)
```

The denominator `max(MC(s_t), ε)` normalizes the progress by the *current correctness level*. This prevents a step at `MC=0.9` from being unfairly penalized for a small absolute gain (going from 0.9 to 0.95 is meaningful even though `Δ=0.05`), and prevents a step at `MC=0.1` from getting excessive credit for a small absolute gain.

Label binarization: a step is labeled `correct` if `RPE(t) ≥ ε_threshold`. Best ε_threshold = 0.8 in GenPRM experiments.

### 3.2 Implementation Details

From the GenPRM GitHub (https://github.com/RyanLiu112/GenPRM):
- MC estimation uses QwQ-32B as the prover for GenPRM's own data. For our setting, Qwen2.5-Math-7B-Instruct is already available as the prover.
- Step boundaries: GenPRM uses `\n\n` boundaries (identical to our current implementation in `phase_f_implicit_prm_eval.py`).
- Base models: DeepSeek-R1-Distill 1.5B/7B/32B. For our frozen backbone, we can apply RPE at the pair-selection level without changing the backbone.

### 3.3 Key Difference from Standard BCE Labels

| Label type | Chosen target | Rejected target |
|---|---|---|
| Hard binary (current) | 1.0 | 0.0 |
| Soft MC (current via confidence) | MC(s_{t+1}) | MC(s_{t+1}) |
| PAV difference | MC_after - MC_before | MC_after - MC_before |
| RPE | (MC_after - MC_before) / max(MC_before, ε) | same |

RPE with regression (MSE loss) vs binary threshold (BCE) is a key ablation. GenPRM shows **RPE with ε=0.8 outperforms hard labels** and the PAV difference variant on their benchmarks.

### 3.4 Note on BiRM Soft Terminal Labels

Related to RPE: the BiRM paper (arXiv:2410.08146, same as PAV, plus the BiRM supplement) recommends soft terminal labels — instead of `terminal_chosen=1`, use `soft_terminal = MC_prover(terminal_step)`. This reduces batch instability when terminal_frac is high. The `terminal_bce_lambda=0.25` currently used could be replaced with a regression target `MSE(score, MC_prover(terminal))`.

---

## Section 4: GRPO on Harder Benchmarks — Dr. GRPO (arXiv:2503.20783)

**Source**: "Understanding R1-Zero-Like Training: A Critical Perspective" (Dr. GRPO), COLM 2025.

### 4.1 The GSM8K Saturation Problem

Our F3 GRPO experiment failed because Qwen2.5-Math-7B-Instruct starts at 95.5% on GSM8K — essentially saturated. GRPO needs a *meaningful reward signal*, which requires the generator to be at ~40-70% baseline accuracy. At 95.5%, the model almost always generates the correct answer and the advantage signal collapses.

**Correct benchmark choice**: MATH Level 3-5 problems (AMC/AIME difficulty). The Dr. GRPO paper uses exactly this:
- Training set: MATH levels 3-5 (roughly 7,500 problems).
- Starting accuracy: Qwen2.5-Math-7B base on MATH L3-5 ≈ 40-55%.
- After Dr. GRPO (27 hours on 8×A100): 43.3% on AIME 2024 (new SOTA for 7B at time of writing).

### 4.2 Dr. GRPO vs Standard GRPO Objective

Standard GRPO loss (DeepSeekMath):
```
L_GRPO = -Σ_t [ (A_i / std(R)) * (1/|o_i|) * min(π_θ/π_old * 1, clip) ]
```

Two bias sources:
1. **Length normalization `1/|o_i|`**: divides by response length, which under-penalizes long incorrect responses.
2. **Std normalization `/ std(R)`**: gives disproportionate weight to problems with low reward variance (very easy or very hard problems where all K completions have the same outcome).

Dr. GRPO fix (two changes only):
```
L_DrGRPO = -Σ_t [ A_i * (1/L_max) * min(π_θ/π_old * 1, clip) ]
```

Where `L_max` is the maximum completion length (a fixed constant, not per-sample length). This removes both biases.

**Empirical result**: Dr. GRPO prevents the model from generating progressively longer incorrect responses, improving token efficiency and final accuracy. With MATH L3-5 training, achieves 43.3% AIME 2024 from a 7B model.

### 4.3 Why Qwen2.5-Math Models are Special for GRPO

The interconnects.ai post (2025) documents that Qwen2.5-Math models respond positively to GRPO even with random/broken rewards. The underlying mechanism: Qwen2.5-Math pretraining already contains latent code-reasoning strategies that GRPO "unlocks" via exploration. This means even noisy PRM signals might yield gains, as long as the baseline accuracy is not already saturated.

**For our GRPO experiment (PBR38D)**:
- Use MATH L3-5 as the training distribution (starting accuracy ~45% for Qwen2.5-Math-7B-Instruct).
- Apply Dr. GRPO objective (fixed `L_max` normalization, no std normalization).
- Outcome reward only first to establish baseline, then add PBR32 PRM process reward.
- Evaluate on MATH500 held-out (not the training distribution).

### 4.4 AIME vs MATH500 as Evaluation Target

For our 10-hour budget with 4×A100, full Dr. GRPO (27h on 8×A100) is not feasible. A 10-hour smoke test on 4×A100 with:
- 500 MATH L3-5 training problems
- 100 gradient steps
- Batch size 8 (4 per GPU), K=4 completions per problem
- Evaluate on MATH500 (500 problems) at end

This would tell us whether the PRM process reward improves accuracy beyond outcome-only GRPO on hard problems, not whether we can match AIME SOTA.

---

## Section 5: PRIME Implicit PRM (arXiv:2502.01456)

**Source**: "Process Reinforcement through IMplicit rEwards" (PRIME-RL), arXiv:2502.01456.

### 5.1 The Beta Log-Ratio Formula

PRIME's implicit PRM score at token `t` is:
```
r_φ(y_t | context) = β · log[ π_φ(y_t | context) / π_ref(y_t | context) ]
```

Step-level score = mean of token-level scores over the step's tokens:
```
score(step_t) = mean_{y in step_t} β · log[ π_LoRA(y | prefix) / π_base(y | prefix) ]
```

### 5.2 Why This is Free

The implicit PRM is derived from any LoRA adapter trained with DPO or cross-entropy on *response-level* labels. No step-level annotation is required. The adapter already exists (PBR32, PBR33, etc.). The only cost is two forward passes per step prefix: one through the LoRA model and one through the frozen base.

### 5.3 Offline Evaluation Setup

The `scripts/phase_f_implicit_prm_eval.py` script already implements the full pipeline:
- Loads PBR32 LoRA adapter via `attach_peft_adapter_for_inference()`.
- Loads the base model separately as reference.
- Splits each ProcessBench solution into cumulative prefixes at `\n\n` boundaries.
- Computes `β·log(π_LoRA/π_ref)` per token, averages per step.
- Runs the standard `_compute_processbench_f1()` with threshold sweep.

**Key parameter to sweep**: `β`. PRIME paper uses β=0.5 as default. For our LoRA checkpoint which was trained with DPO + ranking loss (not pure DPO), the optimal β may differ.

**Expected performance range**: PRIME on Llama-3.1 checkpoints achieves competitive ProcessBench F1 vs dedicated PRMs. For our PBR32 checkpoint (MATH F1=0.689 with explicit value head), the implicit variant without any additional training should land at **~0.60-0.67 MATH F1** depending on β. If it exceeds 0.689, that would be a strong signal that the LoRA backbone itself has learned richer discriminative features than the value head is exploiting.

### 5.4 Memory Requirement

Two copies of Qwen2.5-Math-PRM-7B (base + LoRA) = 2 × ~14GB at bfloat16 = ~28GB. This fits comfortably on a single A100 80GB. Batch size is forced to 1 by the current implementation for simplicity (forward pass per step prefix).

**Estimated runtime**: ProcessBench MATH = 1024 examples × ~5 steps × 2 forward passes = ~10,240 forward passes. At ~0.1s/pass on A100 (1024 tokens, bfloat16) = ~1024s ≈ **17 minutes** per β value.

---

## Section 6: Experiment Plan — 10-Hour Autonomous Session

### Resource Allocation

```
GPU 0: PBR38A (PBR12 data + LoRA + contrastive 0.2)      ~3.5 hours
GPU 1: PBR38B (consensus-filtered PBR26 + frozen MLP)     ~1.5 hours
GPU 2: PBR38D (GRPO on MATH L3-5, Dr. GRPO, 100 steps)   ~2.5 hours
GPU 3: PBR38E (implicit PRM eval, β sweep)                ~1.5 hours
GPU 0 (after PBR38A): PBR38C (PAV regression labels)      ~2.5 hours
```

Total wall time: max(3.5+2.5, 2.5, 1.5, 1.5) = **6 hours** (well within 10h), with time for evals.

---

### 6.1 PBR38A: PBR12 Data + LoRA r=8 All-28 + Contrastive 0.2

**Hypothesis**: PBR32 (LoRA r=8 all-28 + PBR12 data) already breaks the frozen ceiling at MATH F1=0.689. Contrastive loss pushes chosen/rejected representations apart in LoRA feature space, potentially improving acc_erroneous (the main failure mode of all LoRA experiments). PBR12 data quality + contrastive regularization may synergize in a way PBR26 data + contrastive does not.

**Why different from PBR37 (contrastive 0.2 + PBR26 data)**: PBR12 data has fewer but cleaner MS pairs (ms_strict_only, 3307 pairs). PBR26 has more but noisier MS pairs (ms_full, 4968 pairs). The contrastive loss should be more stable on the cleaner PBR12 geometry.

**Command**:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/phase_e_train_value_lora.py \
    --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_pbr12_dpo_plus_ms_strict_pairs__*/train_pairs.jsonl \
    --eval-pairs-jsonl  assets/artifacts/phase_e_pairs/phase_e_pbr12_dpo_plus_ms_strict_pairs__*/validation_pairs.jsonl \
    --model-path assets/models/Qwen2.5-Math-PRM-7B \
    --run-name phase_e_pbr38a_lora_r8_all28_ctr02_pbr12data_s42 \
    --objective-mode joint \
    --lambda-bce 0.5 \
    --lambda-ranking 1.0 \
    --terminal-bce-lambda 0.25 \
    --learning-rate 3e-5 \
    --num-train-epochs 5 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 24 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --lora-target-modules q_proj,v_proj \
    --lora-top-k-layers 0 \
    --ranking-target-space score \
    --pair-weight-mode none \
    --checkpoint-selection-metric pair_acc \
    --head-architecture mlp \
    --head-mlp-hidden-size 512 \
    --head-dropout-prob 0.05 \
    --anti-saturation-weight 5e-4 \
    --contrastive-loss-weight 0.20 \
    --contrastive-margin 0.15 \
    --seed 42 \
    --require-cuda
```

Then evaluate:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/phase_e_eval_benchmark.py \
    --value-run-dir <PBR38A_RUN_DIR> \
    --benchmark-id processbench_math \
    --feature-cache-mode off --batch-size 32 --require-cuda

CUDA_VISIBLE_DEVICES=0 python scripts/phase_e_eval_benchmark.py \
    --value-run-dir <PBR38A_RUN_DIR> \
    --benchmark-id processbench_gsm8k \
    --feature-cache-mode off --batch-size 32 --require-cuda
```

**Expected MATH F1 range**: 0.685-0.700
**Expected GSM F1 range**: 0.765-0.790
**What result tells us**:
- MATH F1 > 0.689: contrastive + good data synergize. Proceed to PBR39 (PAV + contrastive + PBR12).
- MATH F1 in 0.680-0.689: contrastive does not hurt but does not help much; investigate acc_erroneous vs acc_correct decomposition.
- MATH F1 < 0.680: contrastive hurts on good data (same as PBR26 data). Pivot away from contrastive entirely.
**Timing**: ~3 hours training + 0.5 hours eval = **3.5 hours total**

---

### 6.2 PBR38B: Consensus-Filtered PBR26 Data + Frozen MLP

**Hypothesis**: The ~0.686 plateau for frozen MLP on PBR26 data is caused by noisy MS pairs (PBR26 uses ms_full with 4968 pairs including fanout/grid). Filtering to top-50% confidence MS pairs removes the noisiest pairs, providing cleaner gradient signal. With only 50% of the MS data, effective training set shrinks but quality improves.

**Data filtering strategy**: Use the existing `confidence` field in the pair JSONL records. Filter out any MS pair where `|chosen_score - rejected_score|` (from the Qwen2.5-Math-PRM backbone) falls below a threshold (i.e., the backbone itself is not confident on this pair). Keep all DPO pairs (they are already clean).

**Filtering script** (one-time data preprocessing, not a training run — run before this experiment):
```python
# For each pair in train_pairs.jsonl:
# If source == "math_shepherd":
#     score_gap = |pair.chosen_score - pair.rejected_score|
#     keep if score_gap >= percentile_50 among all MS pairs
# If source == "dpo":
#     keep always
```

This is a ~5 minute preprocessing step on CPU.

**Command**:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/phase_e_train_value.py \
    --train-pairs-jsonl <FILTERED_PAIR_DIR>/train_pairs_consensus50.jsonl \
    --eval-pairs-jsonl  assets/artifacts/phase_e_pairs/phase_e_pbr26_dpo_plus_ms_full_pairs__*/validation_pairs.jsonl \
    --model-path assets/models/Qwen2.5-Math-PRM-7B \
    --run-name phase_e_pbr38b_consensus50_frozen_mlp_s42 \
    --objective-mode joint \
    --lambda-bce 0.5 \
    --lambda-ranking 1.0 \
    --terminal-bce-lambda 0.25 \
    --learning-rate 1e-4 \
    --num-train-epochs 10 \
    --per-device-train-batch-size 64 \
    --gradient-accumulation-steps 1 \
    --checkpoint-selection-metric pair_acc \
    --head-architecture mlp \
    --head-mlp-hidden-size 512 \
    --feature-cache-mode read_write \
    --seed 42 \
    --require-cuda
```

**Expected MATH F1 range**: 0.686-0.702
**Expected GSM F1 range**: 0.770-0.790
**What result tells us**:
- MATH F1 > 0.686: consensus filtering helps on frozen backbone. **This is the most important finding** — it would mean we can exceed the frozen plateau purely via data curation, without LoRA.
- MATH F1 ≈ 0.686: filtering doesn't help, plateau is truly architectural (features are the bottleneck).
- MATH F1 < 0.686: filtering removes important coverage (less data hurts more than noise helped).
**Failure mode**: pair_acc might increase while ProcessBench F1 stays flat (overfitting to cleaner pairs). Check acc_erroneous decomposition.
**Timing**: ~1.5 hours total (frozen MLP is fast with feature cache)

---

### 6.3 PBR38C: PAV-Style Regression Targets on Existing PBR12 Pairs

**Hypothesis**: The hard labels (chosen=1, rejected=0) in our current BCE loss are noisy because a "correct" step may have `MC=0.6` (not `1.0`). Using PAV soft regression targets `(MC_after - MC_before)` captures the actual progress signal without hard binarization. Combined with PBR12 data (which has more reliable MC estimates from ms_strict), this may improve acc_erroneous.

**Implementation note**: This requires modifying the training objective to use MSE on `(score - PAV_target)^2` instead of BCE, where `PAV_target` is read from the pair metadata's `mc_prob` fields. The current training code in `scripts/phase_e_train_value.py` and `src/ours/phase_e/training.py` uses BCE/ranking loss — a new objective mode `"pav_regression"` would need to be added, OR we can approximate PAV by using `lambda_bce` with soft targets via the existing `confidence` weighting.

**Approximate PAV without code change**: Use `--pair-weight-mode confidence` with high `min_confidence` cutoff, which already down-weights low-margin pairs (effectively a soft version of PAV). This is already supported in the codebase.

**Command** (using confidence weighting as PAV approximation):
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/phase_e_train_value.py \
    --train-pairs-jsonl assets/artifacts/phase_e_pairs/phase_e_pbr12_dpo_plus_ms_strict_pairs__*/train_pairs.jsonl \
    --eval-pairs-jsonl  assets/artifacts/phase_e_pairs/phase_e_pbr12_dpo_plus_ms_strict_pairs__*/validation_pairs.jsonl \
    --model-path assets/models/Qwen2.5-Math-PRM-7B \
    --run-name phase_e_pbr38c_pav_approx_pbr12_frozen_mlp_s42 \
    --objective-mode joint \
    --lambda-bce 0.5 \
    --lambda-ranking 1.0 \
    --terminal-bce-lambda 0.25 \
    --learning-rate 1e-4 \
    --num-train-epochs 10 \
    --per-device-train-batch-size 64 \
    --gradient-accumulation-steps 1 \
    --ranking-target-space score \
    --pair-weight-mode confidence \
    --checkpoint-selection-metric pair_acc \
    --head-architecture mlp \
    --head-mlp-hidden-size 512 \
    --feature-cache-mode read_write \
    --seed 42 \
    --require-cuda
```

**Expected MATH F1 range**: 0.685-0.700
**What result tells us**:
- If MATH F1 ≥ 0.689 (matches or beats PBR32 frozen): confidence weighting is enough to recover PAV gains. Combine with LoRA in next session.
- If MATH F1 < 0.686: PAV approximation doesn't work via confidence weighting. Need true regression loss.
**Timing**: ~2 hours training + 0.5 hours eval = **2.5 hours total** (after PBR38A finishes on GPU 0)

---

### 6.4 PBR38D: GRPO on MATH L3-5 with Dr. GRPO Objective

**Hypothesis**: GSM8K is saturated (95.5% baseline). MATH L3-5 provides a meaningful RL signal (~45% baseline). Dr. GRPO (no length normalization, no std normalization) prevents reward hacking and length inflation. 100 gradient steps with outcome-only reward should show measurable improvement on MATH500.

**Reference**: Dr. GRPO (arXiv:2503.20783). Code at `github.com/sail-sg/understand-r1-zero`. Their recipe: Qwen2.5-Math-7B base, MATH L3-5 train, 27h on 8×A100. We run a 10× shorter smoke test.

**Data needed**: MATH training problems level 3-5. These are available in standard HuggingFace `lighteval/MATH` or from `hendrycks/competition_math`. Need to filter by `level` field.

**Dr. GRPO modification**: The `scripts/phase_f_grpo_lite.py` uses standard GRPO. To implement Dr. GRPO:
- Replace per-token normalization `(1 / len(completion))` with `(1 / max_length)`.
- Remove `/ std(group_rewards)` from advantage computation.
- Set `--reward-shaping clip_delta` (already implemented) to prevent reward hacking.

**Command** (requires the two Dr. GRPO modifications above):
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/phase_f_grpo_lite.py \
    --policy-model-path assets/models/Qwen2.5-Math-7B-Instruct \
    --value-run-dir assets/artifacts/phase_e_runs/phase_e_pbr32_*/  \
    --train-dataset math_level345 \
    --num-problems 500 \
    --num-rollouts 4 \
    --max-steps 100 \
    --batch-size 8 \
    --learning-rate 1e-6 \
    --lambda-process 0.3 \
    --reward-shaping clip_delta \
    --dr-grpo \
    --eval-dataset math500 \
    --eval-every 25 \
    --run-name pbr38d_grpo_math345_drgrpo_pbr32prm \
    --require-cuda
```

**Expected MATH500 accuracy**:
- Outcome-only baseline (100 steps): +1-3% over starting accuracy.
- With PBR32 PRM process reward: +2-5% if PRM helps, 0-1% if PRM is neutral/harmful.
**What result tells us**:
- Delta (PRM) > Delta (outcome-only): process reward provides useful dense signal on hard problems. Proceed to Phase F4.
- Delta (PRM) ≈ Delta (outcome-only): PRM is neutral. PBR32's process reward is not calibrated well enough for hard MATH problems.
- Delta (PRM) < Delta (outcome-only): PRM is harmful (reward hacking). Review `PRMScorer.score_clip_delta()` calibration.
**Failure mode**: `phase_f_grpo_lite.py` currently uses GSM8K data path. Needs MATH L3-5 data loading and Dr. GRPO normalization change.
**Timing**: ~2.5 hours for 100 steps × 500 problems × 4 rollouts with eval

---

### 6.5 PBR38E: PRIME Implicit PRM Evaluation from PBR32 LoRA Adapter

**Hypothesis**: The PBR32 LoRA adapter (trained on PBR12 data with DPO+MS ranking loss) has implicitly learned to produce log-likelihood ratios `β·log(π_LoRA/π_ref)` that correlate with step correctness. This "free" PRM requires no value head — only two forward passes through the backbone. If it achieves MATH F1 ≥ 0.65, it's a strong alternative signal source.

**Implementation**: `scripts/phase_f_implicit_prm_eval.py` already fully implements this. Simply run with PBR32 run directory.

**Beta sweep** (run sequentially, each ~17 minutes):
```bash
for BETA in 0.1 0.3 0.5 1.0 2.0; do
    CUDA_VISIBLE_DEVICES=3 python scripts/phase_f_implicit_prm_eval.py \
        --lora-run-dir assets/artifacts/phase_e_runs/phase_e_pbr32_*/ \
        --base-model-path assets/models/Qwen2.5-Math-PRM-7B \
        --benchmark-jsonl assets/external_datasets/processbench_math.jsonl \
        --beta $BETA \
        --run-name pbr38e_implicit_prm_pbr32_beta${BETA/./p} \
        --require-cuda
done
```

Total for β sweep: 5 × ~20min = **~1.7 hours total**. Also run GSM for best β:
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/phase_f_implicit_prm_eval.py \
    --lora-run-dir assets/artifacts/phase_e_runs/phase_e_pbr32_*/ \
    --base-model-path assets/models/Qwen2.5-Math-PRM-7B \
    --benchmark-jsonl assets/external_datasets/processbench_gsm8k.jsonl \
    --beta <BEST_BETA_FROM_MATH> \
    --run-name pbr38e_implicit_prm_pbr32_gsm_bestbeta \
    --require-cuda
```

**Expected MATH F1 range**: 0.60-0.68 depending on β
**What result tells us**:
- Implicit F1 > PBR32 explicit (0.689): the LoRA backbone representation is richer than the value head — consider replacing value head with implicit scorer in GRPO loop.
- Implicit F1 ≈ 0.65-0.689: implicit PRM is useful as a secondary signal source but not a replacement.
- Implicit F1 < 0.60: LoRA log-ratio is not a reliable step-level signal; backbone needs more explicit step-level supervision.
**Timing**: ~1.5 hours including GSM eval at best β

---

## Section 7: BCR/ABR Validation Roadmap

### 7.1 What is the BCR-Style Controller?

The original BCR (Bellman Consistency Reward) concept in this project was designed to identify *which rollout steps are value-consistent* vs *value-inconsistent*. In standard RL, Bellman consistency requires:

```
V(s_t) ≈ r(s_t, a_t) + γ · V(s_{t+1})
```

For our PRM setting, the analog is:
```
V(prefix_t) ≈ reward(step_t) + V(prefix_{t+1})
```

A *Bellman-consistent* step is one where the value head's prediction at step `t` is consistent with the prediction at step `t+1` (after accounting for the step's reward). A *Bellman-inconsistent* step is one where the value head makes a large prediction error — indicating the step was surprising in some way (either unusually good or unusually bad).

**BCR-as-controller**: Instead of using Bellman consistency as a training signal (which is circular for a discriminative head), use it as a *routing decision*:
- If `|V(s_t) - V(s_{t+1})| > δ_stop`: stop scoring, this is the error location.
- If `|V(s_t) - V(s_{t+1})| ≤ δ_continue`: continue to next step.

This is exactly what the ABR-lite simulation (F2) implements with `delta_drop=0.15` in `simulate_abr_lite()`.

### 7.2 How BCR Differs from the ABR-lite Simulation

The current ABR-lite controller uses a threshold on the *absolute score* `V(prefix_t) < τ`. The BCR-style controller instead uses a threshold on the *first difference* `V(prefix_t) - V(prefix_{t+1})`:

| Controller | Stop condition | Signal source |
|---|---|---|
| ABR-lite (current F2) | `V(t) < τ` | Absolute score drops below threshold |
| BCR-style | `|V(t+1) - V(t)| > δ` | Consecutive score changes exceed threshold |
| PAV-based | `V(t+1) - V(t) < -δ` | Only negative transitions trigger stop |

The PAV-based version (one-directional BCR) is most principled: only significant *negative* transitions (score drops) trigger a stop signal.

F2 achieved binary detection F1=0.863 using ABR-lite (absolute threshold). The BCR (difference-based) controller may achieve higher precision with lower compute savings, or similar F1 with higher compute efficiency.

### 7.3 Validation Experiment: BCR-Style Controller on PBR19 Scored Rows

The `assets/artifacts/phase_e_eval/pbr19_dpo_mathms_joint_math_fulleval_20260311T123421Z/scored_rows.jsonl` already contains per-step scores from PBR19. Run a modified `phase_f_abr_lite_simulation.py` with the BCR difference-based stop condition.

**Proposed modification** (to `phase_f_abr_lite_simulation.py`):
```python
def simulate_bcr_controller(example, *, tau_delta=0.10, ...):
    scores = [step.score for step in example.steps]
    for t in range(1, len(scores)):
        delta = scores[t-1] - scores[t]  # drop in score
        if delta > tau_delta:
            return {"stopped_at_step": t, "predicted_erroneous": True, ...}
    return {"stopped_at_step": None, "predicted_erroneous": False, ...}
```

**τ_delta sweep**: 0.05, 0.08, 0.10, 0.12, 0.15, 0.20

**Expected improvement over ABR-lite**:
- BCR (difference-based) should have higher *precision* on error detection: it fires only when the score drops significantly, not just when it falls below a threshold.
- ABR-lite fires on globally low-scoring (easy-to-score) problems too early.
- BCR may have lower *recall* on problems that start with a bad step at step 1 (no previous step to compare).
- Combined BCR+ABR-lite (fire if EITHER condition met) is likely optimal.

### 7.4 Connection to F2 Results and Next Steps

F2 (ABR-lite) passed with binary detection F1=0.863. The next gate (F3) requires:
1. **F3A**: Confirm BCR controller achieves binary F1 ≥ 0.863 (same as ABR-lite) or improves on it.
2. **F3B**: Confirm compute efficiency: BCR processes fewer steps per example than ABR-lite (expected: stops earlier on problems with abrupt transitions).
3. **F3C**: Test BCR on PBR32 scored rows (LoRA, MATH F1=0.689) — is the controller still calibrated when the backbone has been fine-tuned?

**F4 gate** (after F3A-C pass): Enable GRPO with BCR-gated process rewards. Only apply process reward to steps that the BCR controller flags as *uncertain transitions* (high `|ΔV|`). This avoids applying process reward to trivially correct/incorrect steps where the signal is already clear from the outcome reward.

### 7.5 BCR Controller vs RL Controller (F-phase)

The `phase_f_train_rl_controller.py` and `run_phase_f_rl_sweep.sh` train a small neural controller (linear/MLP/GRU with hidden=32-128) on top of the scored rows. This is an *adaptive* BCR-style controller where the stopping rule is learned rather than hand-crafted. The RL sweep tests:
- Architecture: linear vs MLP vs GRU.
- Efficiency penalty: `efficiency_alpha=0.0/0.1/0.2/0.3`.
- Window: 2-8 steps of history.

**F4 priority ordering**:
1. Run BCR deterministic simulation (add `simulate_bcr_controller()` to `phase_f_abr_lite_simulation.py`) — 30 minutes CPU only.
2. If BCR F1 ≥ 0.863: proceed to RL controller sweep (already implemented in `run_phase_f_rl_sweep.sh`).
3. If BCR F1 < ABR-lite: understand why (check whether difference-based stopping hurts problems with gradual error accumulation).

---

## Section 8: 10-Hour Experiment Schedule

```
T+0:00  Start all 4 GPUs simultaneously
  GPU 0: Launch PBR38A training     (PBR12 data + LoRA + contrastive 0.2)
  GPU 1: Launch PBR38B preprocessing + training  (consensus-filtered frozen)
  GPU 2: Launch PBR38D GRPO smoke   (MATH L3-5, Dr. GRPO, 100 steps)
  GPU 3: Launch PBR38E β sweep      (implicit PRM, β=0.1..2.0)

T+1:30  PBR38B training completes
  GPU 1: Eval PBR38B on ProcessBench MATH + GSM → check if > 0.686
  GPU 1: If PBR38B ≥ 0.688: start PBR38B-v2 (consensus 30% threshold, tighter filter)
  GPU 1: If PBR38B < 0.685: start BCR simulation (CPU) + RL sweep (GPU 1)

T+2:00  PBR38E β sweep completes (5 × ~20min)
  GPU 3: Select best β, run GSM eval (~20min)
  GPU 3: Run implicit PRM from PBR33 (PBR26 data LoRA) for comparison (~30min)
  GPU 3: Start BCR deterministic simulation (CPU, ~5min)

T+2:30  PBR38D first eval checkpoint (25 steps)
  GPU 2: Log MATH500 accuracy. If <+0.5% over baseline: abort GRPO (likely no signal)
  GPU 2: If +1%+: continue to 100 steps

T+3:30  PBR38A training completes
  GPU 0: Eval PBR38A ProcessBench MATH + GSM (~30min)
  GPU 0: Start PBR38C (PAV-approx, frozen, PBR12 data) — runs ~2.5h

T+4:00  PBR38B eval complete, PBR38D final eval (~2.5h in)
  All: Compare interim results. Reprioritize GPU 1 based on PBR38B outcome.

T+6:00  PBR38C completes (on GPU 0)
  GPU 0: Eval PBR38C ProcessBench MATH + GSM

T+6:30  PBR38D completes (2.5h run)
  GPU 2: Check MATH500 accuracy with vs without PRM reward

T+7:00-10:00  Buffer for follow-up experiments based on results
  If any PBR38 run shows MATH F1 > 0.695:
    - Launch 5-epoch version of the winning recipe on all available GPUs
  If implicit PRM > 0.68 MATH F1:
    - Launch implicit PRM eval on PBR35/PBR37 adapters for comparison
  If GRPO PRM delta > +2%:
    - Run GRPO PRM phase F4 with 500 training steps
```

---

## Section 9: Priority Ranking if Experiments Fail Early

**Tier 1 (must complete)**:
1. **PBR38E** (implicit PRM eval): cheapest experiment, no new training required, provides direct insight into whether LoRA representation is richer than value head. Always worth running.
2. **PBR38B** (consensus-filtered frozen): tests the core hypothesis that data quality is the bottleneck without architecture change. Most actionable if successful.

**Tier 2 (run if GPUs available)**:
3. **PBR38A** (PBR12 + LoRA + contrastive 0.2): tests contrastive on good data. May break through 0.689 ceiling. Requires 3.5h GPU time.
4. **BCR deterministic simulation**: CPU only, 5 minutes, directly validates BCR as a controller.

**Tier 3 (run only if Tier 1 and 2 complete before T+7:00)**:
5. **PBR38C** (PAV-approx frozen): useful research insight but approximate implementation.
6. **PBR38D** (GRPO Dr. GRPO): requires code modifications to `phase_f_grpo_lite.py`.

**If PBR38A, PBR38B, PBR38C all fail to improve over PBR26 (0.686)**:
- The bottleneck is confirmed to be beyond data quality: likely the frozen backbone architecture (score pooling on Qwen2ForProcessRewardModel is not expressive enough).
- Next session pivot: GenPRM-style generative PRM (CoT verification chain before scalar judgment), using a smaller base model to save memory.

---

## Section 10: Key Metrics Table to Track

After each experiment completes, populate this table:

| Run | MATH F1 | GSM F1 | acc_erroneous | acc_correct | Improvement over PBR32 | Notes |
|---|---|---|---|---|---|---|
| PBR26 (baseline frozen) | 0.686 | 0.768 | — | — | -0.003 | Current frozen SOTA |
| PBR32 (baseline LoRA) | 0.689 | — | — | — | 0.000 | Current overall SOTA |
| PBR38A | TBD | TBD | TBD | TBD | TBD | PBR12+LoRA+ctr0.2 |
| PBR38B | TBD | TBD | TBD | TBD | TBD | consensus50+frozen |
| PBR38C | TBD | TBD | TBD | TBD | TBD | PAV-approx+frozen |
| PBR38D outcome-only | TBD | — | — | — | — | GRPO MATH L3-5 |
| PBR38D + PRM | TBD | — | — | — | — | GRPO + PBR32 PRM |
| PBR38E (β=0.5) | TBD | TBD | — | — | — | Implicit PRM β=0.5 |
| PBR38E (best β) | TBD | TBD | — | — | — | Implicit PRM best β |

---

## Section 11: Reference Summary

| Topic | Source | Key Formula/Insight |
|---|---|---|
| Consensus filtering | Qwen2.5-Math-PRM (Zhang et al., 2025) | Discard pairs where MC-label ≠ LLM-judge label; 40% filtered; hard labels >> soft labels after filtering |
| PROF (RL) | arXiv:2509.03403 | Consistency-driven sample selection in GRPO: keep rollouts where PRM and ORM agree; +4% accuracy vs naive mixing |
| ActPRM | arXiv:2504.10559 | Active learning PRM: 20% annotation cost, SOTA at 75.0% ProcessBench F1 |
| PAV | arXiv:2410.08146, ICLR 2025 | `PAV = MC(s_{t+1}) - MC(s_t)` as training target; 5× more compute efficient than ORM |
| RPE (GenPRM) | arXiv:2504.00891, AAAI 2026 | `RPE(t) = (MC_after - MC_before) / max(MC_before, ε)`; RPE ε=0.8 > hard labels > PAV difference |
| Dr. GRPO | arXiv:2503.20783, COLM 2025 | Remove length normalization + std normalization from GRPO; prevents length inflation; 43.3% AIME2024 at 7B |
| PRIME implicit PRM | arXiv:2502.01456 | `r = β·log(π_LoRA/π_ref)`; free PRM from any ORM/DPO adapter; 2.5× sample efficiency vs RLOO |
| PQM contrastive | arXiv:2410.11287 | Comparative (contrastive) loss vs BCE: +11.6% accuracy on MATH500 |
| NAIT noise-aware | arXiv:2601.12748 | Iterative noise correction for PRM labels; reflection-aware re-labeling |

---

## Appendix: Known Issues and Constraints

1. **PBR38D requires code change**: `phase_f_grpo_lite.py` is currently hardcoded for GSM8K data loading. The Dr. GRPO normalization change requires two small modifications (replace `len(completion)` with `max_length` as denominator, remove std normalization). These changes are described precisely in Section 4.2.

2. **PBR38C PAV approximation limitation**: True PAV requires `mc_prob` metadata in pair records. The current `confidence` field in PBR12 pairs encodes the backbone's score on the chosen/rejected items, not the prover's MC estimate. The approximation is reasonable but not exact.

3. **Implicit PRM and Qwen2ForProcessRewardModel**: `_get_lm_logits()` in `phase_f_implicit_prm_eval.py` has special handling for `Qwen2ForProcessRewardModel` — it extracts `hidden_states[-1]` and applies `lm_head` manually. This path is exercised when `model_out.logits` is None. Verify this path works for both the LoRA model and the reference model before the overnight run.

4. **PBR38A pair directory**: Confirm the PBR12 pair directory path. From the MEMORY.md, PBR12 uses `DPO 2398 + MS-strict 3307 pairs`. The directory should match `phase_e_pairs/phase_e_pbr12_*`. Run `ls assets/artifacts/phase_e_pairs/` to confirm.

5. **GPU memory for PBR38D**: `phase_f_grpo_lite.py` with PBR32 LoRA adapter as PRM requires loading both the policy model (Qwen2.5-Math-7B-Instruct, ~14GB) and the PRM model (Qwen2.5-Math-PRM-7B + LoRA adapter, ~14GB) on one GPU. Total ~28GB — fits on A100 80GB. But with 4 rollouts per problem and batch_size=8, peak memory may approach 60-70GB. Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.

---

Sources:
- [Qwen2.5-Math-PRM consensus filtering (Hugging Face papers)](https://huggingface.co/papers?q=Process+Reward+Models)
- [Towards Robust Process Reward Modeling via Noise-aware Learning (arXiv:2601.12748)](https://arxiv.org/html/2601.12748)
- [Beyond Correctness: Harmonizing Process and Outcome Rewards (PROF, arXiv:2509.03403)](https://arxiv.org/pdf/2509.03403)
- [Efficient Process Reward Model Training via Active Learning (ActPRM, arXiv:2504.10559)](https://arxiv.org/html/2504.10559v1)
- [Rewarding Progress: Scaling Automated Process Verifiers (PAV, arXiv:2410.08146, ICLR 2025)](https://arxiv.org/abs/2410.08146)
- [GenPRM: Scaling Test-Time Compute of PRMs via Generative Reasoning (arXiv:2504.00891, AAAI 2026)](https://arxiv.org/abs/2504.00891)
- [GenPRM GitHub](https://github.com/RyanLiu112/GenPRM)
- [Process Reinforcement through Implicit Rewards (PRIME, arXiv:2502.01456)](https://arxiv.org/abs/2502.01456)
- [PRIME-RL GitHub (ImplicitPRM)](https://github.com/PRIME-RL/ImplicitPRM)
- [Understanding R1-Zero-Like Training: Dr. GRPO (arXiv:2503.20783, COLM 2025)](https://arxiv.org/abs/2503.20783)
- [Dr. GRPO GitHub](https://github.com/sail-sg/understand-r1-zero)
- [Process Reward Model with Q-Value Rankings (PQM, arXiv:2410.11287)](https://arxiv.org/html/2410.11287v2)
- [The Lessons of Developing Process Reward Models (arXiv:2501.07301)](https://arxiv.org/pdf/2501.07301)
