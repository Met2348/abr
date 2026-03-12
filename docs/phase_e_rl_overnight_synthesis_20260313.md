# Phase E/F Overnight Synthesis: LoRA Expansion + RL Feasibility + Literature

**Date**: 2026-03-13 (overnight session)
**Prepared by**: Claude Code (automated overnight research + experiment chain)

---

## 1. Executive Summary

This overnight session achieved the following:

1. **Phase E LoRA expansion**: Launched PBR34 (r=16 top-4), PBR35 (r=8 all-28 + PBR26 data),
   PBR36 (r=32 all-28), PBR37 (r=8 all-28 + contrastive loss). Expected to push MATH F1 > 0.689.

2. **RL controller evaluation**: Confirmed heuristic tau=0.35 (F1=0.822) is optimal for offline
   error detection on PBR32 scored traces. All RL-learned controllers underperform.

3. **GRPO feasibility assessment**: PBR32 achieves separation=0.404 MATH / 0.467 GSM8K with
   step AUC 0.976/0.982. **STRONG PASS** — PBR32 is ready as GRPO process reward signal.

4. **GRPO-lite baseline training**: Outcome-only GRPO running on GPU3 (Qwen2.5-Math-7B-Instruct
   policy, 500 GSM8K problems, k=4 samples). This is the Phase F RL baseline experiment.

5. **Literature synthesis**: Comprehensive survey of 20+ papers identified contrastive loss,
   LoRA rank scaling, and implicit PRM as highest-priority Phase E improvements.

---

## 2. Phase E: LoRA Experiment Chain Status

### 2.1 Completed Experiments

| Run | Config | MATH F1 | GSM F1 | Key Finding |
|-----|--------|---------|--------|-------------|
| PBR32 | r=8 all-28, PBR12 | **0.689** | 0.776 | **CURRENT MATH SOTA** |
| PBR33 | r=8 top-4, PBR26 | 0.666 | **0.797** | **CURRENT GSM SOTA** |
| PBR31 | r=8 top-4, PBR12 | 0.676 | 0.788 | Top-4 vs all-28 baseline |
| PBR26 (frozen) | None, PBR26 | 0.686 | 0.768 | Best frozen |

Key axis insights:
- **Depth > Width**: All-28 layers (2.52M params) consistently beats top-4 (360K) for MATH F1
- **Data volume > Depth for GSM**: PBR26 (7366 pairs) + top-4 gives best GSM despite weaker MATH
- **LoRA vs frozen for MATH**: PBR32 (+0.003 F1 vs PBR26) — modest gain, room for improvement
- **Community gap**: 0.689 vs Qwen2.5-Math-PRM-7B ~0.735 — 4.6 F1 points remaining

### 2.2 Running Experiments

| Run | GPU | Config | Epoch 0 pair_acc | Expected MATH F1 |
|-----|-----|--------|-----------------|------------------|
| PBR34 | 2 | r=16 top-4 + PBR26 | 0.856→0.879 | 0.66-0.68 (top-4 range) |
| PBR35 | 1 | r=8 all-28 + PBR26 | 0.847 | 0.690-0.700 (hypothesis best) |
| PBR36 | 3 | r=32 all-28 + PBR26 | starting | 0.695-0.705 (rank boost) |
| PBR37 | 0 | r=8 all-28 + contrastive=0.2 + PBR26 | starting | 0.692-0.710 (contrastive boost) |

All evals auto-queued after training completes.

### 2.3 Hypotheses to Test

| Hypothesis | Experiment | Expected Outcome |
|------------|-----------|-----------------|
| More data helps all-28 LoRA | PBR35 vs PBR32 | PBR35 MATH F1 > PBR32 (0.689) |
| r=32 beats r=8 for MATH | PBR36 vs PBR35 | +1-3 F1 from rank scaling |
| Contrastive loss (+0.09 AUROC) | PBR37 vs PBR35 | +2-3 F1 from feature separation |
| r=32 + contrastive = additive? | PBR38 (not yet launched) | +3-5 F1 combined? |

### 2.4 Next Planned: PBR38 (r=32 + contrastive + PBR26)

If both PBR36 (rank scaling) and PBR37 (contrastive) show positive results, launch:
- PBR38: r=32 all-28, alpha=64, contrastive_weight=0.2, PBR26 data
- Expected: combines both improvements, may reach MATH F1 ≥ 0.700

---

## 3. Phase F: RL Feasibility Assessment

### 3.1 Controller Sweep (Previous Session)

The best offline controllers for PBR19/PBR32 MATH traces:

| Controller | balanced_F1 | Notes |
|-----------|------------|-------|
| `threshold_only tau=0.38` | **0.8674** | pbr19_math |
| `threshold_only tau=0.35` | 0.8641 | pbr21_math |
| `threshold_only tau=0.42` | 0.8552 (unified) | all 8 slices |
| `delayed_drop tau=0.42 δ=0.25 min=4` | 0.8501 (unified) | all 8 slices |

**Recommendation**: Use `threshold_only tau=0.35-0.42` as the live Phase F controller.

### 3.2 RL Controller Training (This Session)

Offline REINFORCE on PBR32 MATH scored traces (n=1000, 70/15/15 split):

| Method | Val F1 | Test F1 |
|--------|--------|---------|
| **Heuristic tau=0.35** | **0.848** | **0.822** |
| MLP BC-warmstart (fp=5) | 0.818 (post-BC) → 0.781 (post-RL) | 0.779 |
| GRU REINFORCE | 0.72-0.75 | 0.777 |
| Supervised MLP (fp=3) | 0.734 | 0.770 |

**Conclusion**: RL fine-tuning consistently hurts BC-initialized policies. Use heuristic.

### 3.3 GRPO Feasibility: STRONG PASS

Using PBR32 LoRA adapter as process reward signal:

| Domain | Separation | Step AUC (late) | Verdict |
|--------|-----------|----------------|---------|
| MATH | 0.404 | 0.976 (step 15) | PASS |
| GSM8K | 0.467 | 0.982 (step 8) | PASS |

**Interpretation**:
- Separation > 0.30 is sufficient for positive GRPO gradient signal
- Step AUC > 0.95 means PRM can reliably identify error positions at late steps
- PBR32 is **ready for live GRPO RL training**

### 3.4 GRPO-lite Training: Running

**Outcome-only baseline** (PID 3773602, GPU3):
- Policy: Qwen2.5-Math-7B-Instruct
- 500 GSM8K train problems, 200 eval, k=4 samples, 1 epoch, lr=1e-6
- Still in pre-training eval (generation of 200×4=800 solutions takes ~30-60 min)
- This establishes the GRPO outcome-only baseline accuracy

**Known blocker for process-reward GRPO**: PEFT version incompatibility
- bcr env (Python 3.11, peft 0.11) cannot load LoRA adapters saved by base env (Python 3.12, peft 0.14+)
- Fix: run GRPO with process reward using base env python3 and upgrading peft, OR save adapter
  in older format compatible with both versions
- Workaround: use `--ranking-target-space score` with the adapter loaded in base env

---

## 4. Literature Synthesis: Closing the Community F1 Gap

### 4.1 Immediate Opportunities (Low Effort)

**Contrastive Loss (Scale AI arXiv:2407.13887)**: +0.09 AUROC on math/GSM8K
- Already implemented in `phase_e_train_value_lora.py` via `--contrastive-loss-weight`
- Uses cosine-margin loss to push chosen/rejected backbone hidden states apart
- PBR37 tests weight=0.2; PBR38 will combine with r=32 if PBR37 succeeds

**LoRA rank r=32 (literature consensus)**: +1-3 F1 expected
- Requires alpha=64 (2× rank) for stable training
- 10.1M trainable params vs 2.52M for r=8 — 4× increase
- Higher rank = better capacity to adapt math step-level features
- PBR36 tests this directly

### 4.2 Medium-Effort Opportunities

**EDU-PRM Entropy Step Boundaries (arXiv:2503.22233)**:
- Replace `\n\n`-based step boundaries with high-entropy token positions (cognitive transitions)
- Achieved Qwen2.5-72B-PRM quality with only 7,500 training queries (38× data reduction)
- For us: reprocess DPO+MS pairs using entropy-detected boundaries
- Data preprocessing only, no training algorithm change
- Implementation: `build_entropy_step_boundaries()` in preprocessing pipeline

**Multi-scale supervision (step + solution)**:
- Joint step-level ranking + solution-level BCE already partially implemented via `--terminal-bce-lambda`
- Could add a separate ORM head on the `[EOS]` token for full-solution outcome
- AutoPRM (NAACL 2024) shows +2% precision with joint objectives

### 4.3 High-Effort / Architecture Changes

**Implicit PRM / PRIME (arXiv:2502.01456)**:
- Reward = β·log[π_LoRA(y) / π_ref(y)] — token-level log-ratio between fine-tuned and frozen backbone
- Eliminates need for step annotations entirely
- With LoRA active (PBR32/35/36), π_ref = frozen checkpoint, π_LoRA = adapted checkpoint
- Very strong evidence: outperforms Math-Shepherd trained PRM using <1/38 data
- Risk: β=0.05, our r=8 LoRA may have insufficient departure from π_ref to be discriminative
- Implementation: after each forward pass, compute token-level log-ratio as alternative reward head

**GenPRM Generative Verification (arXiv:2504.00891)**:
- Fine-tune model to generate CoT verification reasoning before binary judgment
- 7B GenPRM achieves ProcessBench MATH F1 ~80.5 — our clear ceiling target
- Very high effort: complete architecture change, requires 23K training examples with generated critiques
- Not feasible in current Phase E scope; relevant for Phase G if needed

### 4.4 RL Integration Strategy (Phase F)

**GRPO with step-level PRM rewards (key insight from "GRPO is Secretly a PRM")**:
- GRPO's group advantage = sum of leave-one-out normalized rewards
- Plugging in PRM step scores as per-step rewards requires only modifying advantage computation
- Formula: `advantage_t = PRM_score(step_t) - mean(PRM_scores across k rollouts for step t)`
- Implementable with any GRPO framework (TRL, veRL, OpenRLHF)

**rStar-Math self-evolution loop (ICML 2025)**:
- 4 rounds: MCTS → PRM training → policy training → better MCTS
- Pushed Qwen2.5-Math-7B from 58.8% → 90.0% on MATH500
- Minimum smoke test: 1000 problems × 8 rollouts × 1 round ≈ 8000 generations
- Our PBR32 is sufficient as the initial PRM for round 1 MCTS guidance

**PRIME online PRM update (arXiv:2502.01456)**:
- Update PRM online alongside policy to prevent reward hacking from distribution shift
- Prevents static PRM from becoming stale as policy improves
- For us: update LoRA adapter parameters each training batch using outcome labels
- Critical for long Phase F training runs (reward hacking risk after >100 gradient steps)

---

## 5. Recommended Experiment Roadmap (Post-Overnight)

### Phase E Completion (Expected by 09:00-10:00)

| Priority | Experiment | Why |
|----------|-----------|-----|
| **Highest** | Wait for PBR35 eval results | Tests all-28 + more data combination |
| **High** | Wait for PBR36 eval results | Tests rank scaling (r=32) |
| **High** | Wait for PBR37 eval results | Tests contrastive loss |
| **Medium** | Launch PBR38 if PBR36+PBR37 both positive | r=32 + contrastive combined |

### Phase F Live RL (Next Steps)

| Step | Action | Blocker |
|------|--------|---------|
| F3a | Evaluate GRPO baseline (outcome-only) | Training in progress |
| F3b | Fix peft compatibility, run GRPO+PRM | Minor code fix |
| F3c | Compare GRPO+PRM vs GRPO-outcome-only | Need F3a+F3b |
| F4 | MCTS smoke test (1000 problems, PBR32 as PRM) | GPU time, 24h |

### Quick Wins (Can Do Immediately)

1. Fix PEFT compatibility for GRPO+PRM (`--peft-compat-mode` flag or base env)
2. Launch PBR38 (r=32 + contrastive) after PBR36/37 results arrive
3. Design entropy-based step boundary preprocessing script

---

## 6. Summary of Key Decisions

| Decision | Rationale |
|----------|-----------|
| Focus on LoRA r=32 first (PBR36) | Literature: r=8 underpowers 7B model for math tasks |
| Test contrastive loss (PBR37) | Already implemented, high evidence quality (+0.09 AUROC) |
| Use heuristic tau=0.35-0.42 for Phase F controller | RL doesn't beat heuristic with current data size |
| GRPO-lite baseline first (outcome-only) | Need baseline before PRM process reward comparison |
| PBR32 as Phase F RL primary candidate | Best MATH F1=0.689, STRONG PASS on feasibility |
| Don't use TD learning with frozen backbone | No convergence guarantee; use MC targets only |

---

*Artifact directory*: `docs/phase_e_rl_overnight_synthesis_20260313.md`
*Related*: `docs/phase_F_plan.md`, `docs/result_records.md`, `memory/MEMORY.md`
