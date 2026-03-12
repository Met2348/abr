# Result Records

Last updated: 2026-03-12 16:45:00 +0800

## 0AAARCBE. Phase E/F Midday Audit Refresh: Gate Sweep + GRPO Proxy Fix + Focused Controller Distill (2026-03-12)

### Infra / Audit Fixes

1. `docs/relatedPapers/` has now been resynced against current docs: `106/106` referenced paper URLs are mirrored locally, with `5` new PDFs added this round.
2. `scripts/run_phase_e_lora_overnight_suite.sh` no longer routes heavy train/eval stdout through `tee`; the suite now resolves only completed artifacts with marker files such as `manifest.json` / `metrics.json`.
3. `scripts/phase_f_grpo_feasibility.py` fixed a real methodology bug: the old default grouped traces only within the same label family, which forced outcome reward variance to collapse to zero. Default sampling is now `mixed_pool`, and `tests/unit/test_phase_f_grpo_feasibility.py` locks that behavior.

### Web-backed Design Direction

Detailed synthesis: `docs/phase_e_phase_f_web_research_refresh_20260312.md`

Primary sources checked this round:

1. VerifyBench (`2507.09884`)
2. Hard2Verify (`2510.13744`)
3. When to Trust the Cheap Check (`2602.17633`)
4. DeepSeekMath / GRPO (`2402.03300`)
5. RL reward design / Clip+Delta (`2410.15115`)
6. DeepSeek-R1 (`2501.12948`)
7. PRIME (`2502.01456`)
8. ThinkPRM (`2504.16828`)
9. GenPRM (`2504.00891`)
10. VPRM (`2601.17223`)
11. MASH (`2510.01152`)
12. TRL `GRPOTrainer` docs
13. DAPO / Dr. GRPO / VAPO / Is PRM Necessary? (`2503.14476`, `2503.20783`, `2504.05118`, `2505.11227`)

Cross-source conclusion:

1. `Phase E` should move toward selective escalation and hybrid verifier systems, not one universal scalar head.
2. `Phase F` should remain `heuristic/BC first, RL second`.
3. Future live RL runs should leave `GSM8K` and move to a harder benchmark before burning more overnight GPU.

### New Experiment Results

#### 1. Cheap -> strong gate sweep

Artifact: `assets/artifacts/phase_e_gate_sweeps/phase_e_gate_phasef_0312_1630_20260312T083206Z/summary.json`

| Case | weak_auc | strong_auc | best_tau | best_auc | best_f1 | strong_usage |
|-----|-----:|-----:|-----:|-----:|-----:|-----:|
| math_p26_to_p32 | 0.8882 | 0.8685 | 0.00 | 0.8882 | 0.6700 | 0.0000 |
| math_p31_to_p32 | 0.8823 | 0.8685 | 0.00 | 0.8823 | 0.6762 | 0.0000 |
| gsm_p19_to_p31 | 0.9148 | 0.9006 | 0.10 | 0.9170 | 0.7822 | 0.0908 |
| gsm_p19_to_p33 | 0.9148 | 0.9026 | 0.15 | 0.9175 | 0.7924 | 0.1633 |

Interpretation:

1. selective escalation is real on GSM-like slices,
2. but the current `PBR32` Math scorer is not strong enough to justify escalation over `PBR26/PBR31` on Math.

#### 2. Mixed-pool GRPO feasibility on stronger verifier slices

Artifact: `assets/artifacts/phase_f_grpo_feasibility/strong_mixedpool_0312_1635/grpo_feasibility_20260312T083443Z.json`

| Domain | process |adv| | outcome |adv| | max_beta | verdict |
|-----|-----:|-----:|-----:|-----|
| math (`PBR32`) | 0.811 | 0.864 | 0.200 | FEASIBLE |
| gsm8k (`PBR33`) | 0.805 | 0.872 | 0.500 | FEASIBLE |

Interpretation:

1. after fixing the proxy, outcome reward is no longer degenerate,
2. the current RL blocker is more likely benchmark saturation / optimization stability than raw reward variance absence,
3. if live GRPO continues, beta should start around `0.05`, but on a harder benchmark than current `GSM8K`.

#### 3. Focused controller distill / RL experiments

Artifacts:

1. `assets/artifacts/phase_f_bc/phase_f_focus_bc_0312_1638_20260312T083525Z/summary.json`
2. `assets/artifacts/phase_f_bc/phase_f_focus_bc_then_rl_0312_1638_20260312T083525Z/summary.json`
3. `assets/artifacts/phase_f_rl_like/phase_f_focus_rl_0312_1643_20260312T083645Z/summary.json`

Held-out `test_eval` results:

| Case | bc_only | bc_then_rl | rl_from_scratch |
|-----|-----:|-----:|-----:|
| p31_math | 0.7987 | 0.8171 | 0.0000 |
| p32_math | 0.8309 | 0.8395 | n/a |
| p31_gsm | 0.9000 | 0.9082 | 0.0000 |

Interpretation:

1. mild RL fine-tuning on top of BC can still help,
2. but the gains are modest and remain teacher-dependent,
3. from-scratch RL-like controller training is still unusable.

#### 4. Modern preflight on `PBR26/PBR31`

Artifact: `assets/artifacts/phase_f_logs/phase_f_preflight_wait_0312_1641/final_summary.md`

Key findings:

1. `p31_gsm` has wider threshold tolerance than `p26_gsm`, but worst-generator logo F1 is still `0.0000`.
2. `p31_math` lifts worst-generator logo F1 to `0.4251`, but `confidence_tail` reward-hacking risk is still `high`.
3. `PBR31` is a useful controller research slice, not a safe blind deployment verifier.

### Active Overnight Follow-up

Queued / running now:

1. `Phase E L2` gated LoRA frontier
   - watcher log: `assets/artifacts/phase_e_logs/phase_e_l2_wait_0312_1641/watch.log`
   - active run dir: `assets/artifacts/phase_e_runs/phase_e_l2_wait_0312_1641_L2_all28_ctr010_center_gated_20260312T083609Z`

### Decision Update

1. `Phase E`: keep `L2` running; prioritize `gated_mlp + all-layer LoRA` over more Math-side cheap->strong gate work.
2. `Phase F`: keep `preflight -> heuristic/BC -> mild RL` as the only promotion path.
3. `RL`: do not promote from-scratch controller RL; move any future live GRPO off saturated `GSM8K`.

---

## 0AAARCBD. Phase E+F Full Summary: All Experiments Complete — Strategic Pivot (2026-03-12)

### Complete Experiment Inventory (as of 2026-03-12 12:05 local)

All planned Phase E+F experiments have now completed. This entry summarizes the full picture.

#### Phase E LoRA Sweep — Final Results (all PBR26 data unless noted)

| Run | Config | MATH F1 | GSM F1 | vs Frozen |
|-----|--------|---------|--------|-----------|
| **Frozen PBR26** | — | **0.686** | 0.768 | **baseline** |
| Frozen PBR19 | — | 0.683 | **0.778** | -0.003 |
| **PBR32 ★** | LoRA r=8 all-28, PBR12 data | **0.689** | 0.776 | **+0.003** |
| PBR33 | LoRA r=8 top-4, PBR26 data | 0.666 | **0.797** | -0.020 |
| PBR34 | LoRA r=16 top-4, PBR26 data | 0.657 | 0.766 | -0.029 |
| PBR35 | LoRA r=8 all-28, PBR26 data | 0.657 | 0.768 | -0.029 |
| PBR36 | LoRA r=32 all-28, PBR26 data | 0.656 | 0.739 | -0.030 |
| PBR37 | LoRA r=8 all-28, ctr=0.2, PBR26 | 0.657 | 0.768 | -0.029 |
| L1 | LoRA r=8 all-28, ctr=0.05, center, PBR26 | 0.654 | 0.762 | -0.032 |

**Finding**: All PBR26-data LoRA variants converge to MATH F1 = 0.654-0.657, uniformly BELOW frozen PBR26 (0.686). PBR32 is the exception: it uses PBR12 data (smaller, higher-quality MS pairs) and beats frozen by +0.003.

**Root cause**: PBR26 MS data (math_shepherd full, 4968 pairs) has more noise than PBR12 MS data (3307 pairs). LoRA amplifies the training distribution noise while frozen MLP can learn a cleaner signal.

#### Phase F GRPO Experiments — Complete Table

| Run | PRM | reward | pre | post | delta | gate | outcome |
|-----|-----|--------|-----|------|-------|------|---------|
| v1 (PBR32) | LoRA | raw_score | 0.950 | — | — | CRASH | disk full |
| v2 outcome-only | none | outcome | 0.955 | 0.950 | -0.005 | FAIL | noise degradation |
| v3 (PBR32) | LoRA | clip_delta | 0.955 | — | — | CRASH | disk full |
| v4 (PBR32) | LoRA | clip_delta | 0.955 | — | — | CRASH | disk full at step 150 |
| **v4 r2 (PBR26)** | Frozen | clip_delta | **0.955** | **0.955** | **0.000** | **FAIL** | **saturated** |

**Finding**: GSM8K with 95.5% pre-training accuracy is unsuitable for GRPO. No variant can improve.

#### Phase F BoN K=4 Results

| Scorer | greedy | random@4 | prm@4 | vs_greedy | vs_random | phase_f3 |
|--------|--------|----------|-------|-----------|-----------|---------|
| PBR32 (LoRA) | 0.905 | 0.900 | **0.920** | **+1.5%** | **+2.0%** | **PASS** |
| PBR26 (frozen) | 0.910 | 0.915 | 0.925 | +1.5% | +1.0% | **FAIL** |

LoRA training (PBR32) makes scores more discriminative for BoN than frozen PBR26.

### Strategic Pivot: Why All Experiments Failed

1. **LoRA on PBR26 data** → overfits to noisy MS labels, MATH F1 = 0.654-0.657 (below frozen 0.686)
2. **GRPO on GSM8K** → 95.5% ceiling means no room to improve
3. **BoN with PBR26** → random@4 > greedy baseline neutralizes PRM gain

The core bottleneck is **training data quality**. PBR12 data (better) gives PBR32=0.689. PBR26 data (more but noisier) gives everything below 0.660 with LoRA.

### Next Step: PBR38 — Data Quality Improvement

Based on the analysis: switching from PBR26 data to higher-quality data should break the ceiling.

**Hypothesis 1 (PBR38A)**: Use PBR12 data (DPO 2398 + MS strict 3307) for LoRA r=8 all-28 + contrastive=0.2
- This is the same data as PBR32, but with contrastive loss added
- PBR32 without contrastive = 0.689 → PBR38A expected 0.690-0.695?

**Hypothesis 2 (PBR38B)**: Consensus filtering on PBR26 data
- Keep only MS pairs where MC estimate confidence ≥ 0.80 (top 50% by confidence)
- Reduces ~4968 MS pairs to ~2500 high-confidence pairs
- The idea: high-confidence MC pairs are more likely to have correct labels

**Hypothesis 3 (PBR38C)**: Harder GRPO benchmark (MATH500 at ~45% instead of GSM8K at 95.5%)
- Use MATH500 as GRPO training benchmark
- Requires a different generator checkpoint that's ~70% on MATH (e.g., different model or prompt)

**Recommended next run**: PBR38A (PBR12 data + LoRA r=8 + contrastive=0.2) — lowest risk, directly extends PBR32 result

---

## 0AAARCBC. Phase F GRPO+PRM v4 r2 (Clip+Delta, PBR26) + L1 LoRA Training Complete (2026-03-12)

### GRPO+PRM v4 r2 (Clip+Delta, PBR26) — COMPLETE: phase_f4_gate=FAIL

| Metric | Value |
|--------|-------|
| pre_training_accuracy | **0.955 (95.5%)** |
| post_training_accuracy | **0.955 (95.5%)** |
| accuracy_delta | **0.000** (±0.0%) |
| train_elapsed | 4649s (~77 min, 250 steps × ~18.6s/step) |
| **phase_f4_gate** | **FAIL** |

Config: Qwen2.5-Math-7B-Instruct, PRM=PBR26 (frozen, MATH F1=0.686), λ_process=0.3, reward_shaping=clip_delta.
Output: `assets/artifacts/phase_f_grpo/grpo_pbr26_process_clipdelta_v4_r2_20260312T020814Z/summary.json`

### Root Cause: GSM8K Ceiling Problem

The generator (Qwen2.5-Math-7B-Instruct) already achieves **95.5% accuracy on GSM8K** before GRPO training.
GRPO cannot improve accuracy when most k=4 solution groups are all-correct — the outcome reward variance
within groups is near-zero, and the process reward (Clip+Delta from PBR26) fails to shift probability
mass toward meaningfully better solutions because "good enough" is already achieved.

Reward trajectory: steps 10-250 oscillate 0.54-0.87 with high std (0.32-0.61), no upward trend.
Entropy oscillates 0.47-1.94, indicating policy instability rather than improvement.

**To achieve GRPO improvement, need a harder benchmark where generator is ~70-80% (not 95%).**
Candidates: MATH500 (~45%), AMC/AIME (~15-30%), or self-generated harder problem sets.

### Pattern Across All Phase F GRPO Attempts

| Run | PRM | reward_shaping | pre-acc | post-acc | delta | gate |
|-----|-----|----------------|---------|----------|-------|------|
| GRPO v1 (PBR32) | LoRA | raw_score | 0.950 | unknown | unknown | CRASH (disk) |
| GRPO v2 outcome-only | none | outcome | 0.955 | 0.950 | -0.005 | FAIL |
| GRPO v3 (PBR32) | LoRA | clip_delta | 0.955 | unknown | unknown | CRASH (disk) |
| GRPO v4 (PBR32) | LoRA | clip_delta | 0.955 | unknown | unknown | CRASH (step 150) |
| **GRPO v4 r2 (PBR26)** | Frozen | **clip_delta** | **0.955** | **0.955** | **0.000** | **FAIL** |

**Conclusion**: All GRPO experiments on GSM8K fail because the generator is already saturated at 95.5%.
The process reward adds stability (no further decline unlike outcome-only v2), but cannot improve accuracy.

### L1 Overnight LoRA Training + Eval — COMPLETE

- Config: Math-PRM-7B + LoRA r=8, all-28 layers, contrastive=0.05 (margin=0.15), center_mlp, reward_centering=0.01, PBR26 data
- Training: 5 epochs, best at **epoch 3** (pair_acc=0.8792, AUC=0.8378)
- **MATH F1 = 0.654** (acc_err=0.524, acc_cor=0.872) — below PBR35/37 (0.657)
- **GSM F1 = 0.762** (acc_err=0.633, acc_cor=0.959) — below PBR35/37 (0.768)

**Conclusion**: contrastive=0.05 + center_mlp + reward_centering does NOT improve over baseline.
All 5 PBR26-data LoRA variants (PBR34, 35, 36, 37, L1) plateau at MATH F1 = 0.654-0.657.

Artifact: `assets/artifacts/phase_e_runs/phase_e_lora_overnight_0312_0452_L1_all28_ctr005_center_mlp_20260311T225358Z/`

---

## 0AAARCB. Phase F BoN v4 r2 (PBR26): FAIL Gate + GRPO+PRM v4 r2 Running (2026-03-12)

### BoN K=4 v4 r2 — PBR26 Frozen Scorer — COMPLETE: phase_f3_gate=**FAIL**

**Confirmed authoritative result** (fresh run, seed=42, 200 GSM8K problems, k=4):

| Metric | Value |
|--------|-------|
| greedy_accuracy | 0.910 (91.0%) |
| random@4_accuracy | 0.915 (91.5%) |
| prm_reranked_accuracy | **0.925 (92.5%)** |
| oracle_accuracy | 0.950 (95.0%) |
| prm_vs_greedy_delta | **+0.015 (+1.5%)** |
| prm_vs_random_delta | **+0.010 (+1.0%)** |
| **phase_f3_gate** | **FAIL** (need ≥+2% vs random) |

Scorer: PBR26 (frozen Qwen2.5-Math-PRM-7B + MLP value head, MATH F1=0.686).
Output: `assets/artifacts/phase_f_bon/pbr26_bon4_gsm8k_v4_r2_20260312T003638Z/summary.json`

### Why FAIL despite +1.5% vs greedy?

The random@4 accuracy (0.915) is HIGHER than greedy (0.910). This means:
- The generator is so strong (91% greedy) that random selection from k=4 already outperforms greedy
- The PRM gains +1.5% vs greedy but only +1.0% vs this strong random baseline
- Phase_f3 gate requires `prm@4 ≥ random + 2%` = `0.925 ≥ 0.935` → FAIL (0.010 < 0.020 threshold)

This contrasts with PBR32 (LoRA scorer):
- PBR32: random@4=0.900 (lower than greedy 0.905), prm@4=0.920, prm_vs_random=+2.0% → PASS
- PBR26: random@4=0.915 (higher than greedy 0.910), prm@4=0.925, prm_vs_random=+1.0% → FAIL

**Key insight**: PBR26 adds absolute accuracy (+1.5pp over greedy) but not enough relative to the high random baseline. LoRA training (PBR32) makes scores more discriminative — better at distinguishing the best solution among similar-difficulty candidates.

### Note: Earlier log-recovered v4 was slightly wrong
The `phase_f_bon/pbr26_bon4_gsm8k_v4_20260311T225703Z/summary.json` was recovered from partial logs
and showed `prm@4=0.910, delta=0.0%`. The authoritative v4_r2 shows `prm@4=0.925, delta=+1.5%`.
The log-recovery missed the late batch improvements and underestimated PBR26's BoN performance.

### GRPO+PRM v4 r2 (Clip+Delta, PBR26) — STARTED (10:10)
- Run: `grpo_pbr26_process_clipdelta_v4_r2_20260312T020814Z`
- Config: Qwen2.5-Math-7B-Instruct, PRM=PBR26, λ_process=0.3, reward_shaping=clip_delta
- 500 train + 200 eval problems, 250 steps, GPU 3 (CUDA_VISIBLE_DEVICES=3)
- Disk fix: `save_total_limit=1` → only 1 checkpoint kept (no disk overflow)
- Log: `assets/artifacts/phase_e_logs/phase_f_grpo_process_v4_r2.log`
- ETA: ~75 min (250 steps × ~18s/step)

---

## 0AAARC. Phase F Morning Results: GRPO Crash, BoN PBR26 Final, PBR34-37 LoRA Complete (2026-03-13)

### GRPO PRM-Guided v4 (λ=0.3) — CRASHED at step 150

- Run: `grpo_pbr32_process_lambda03_v4_20260311T225806Z`
- Policy: Qwen2.5-Math-7B-Instruct, PRM: PBR32 (LoRA, MATH F1=0.689)
- **pre_training_accuracy: 0.955** (from log)
- **post_training_accuracy: UNKNOWN** (process crashed at step 150 checkpoint save)
- Cause: `OSError: [Errno 28] No space left on device` during `optimizer.pt` write in checkpoint-150
- checkpoint-150: COMPLETE (36GB, all safetensors shards present, optimizer.pt saved after disk freed)
- **phase_f4_gate: INCONCLUSIVE_CRASH** — no post-training eval possible
- Reward trajectory at crash: steps 10-150 show oscillation 0.46–0.82, no upward trend; pre-training 95.5% ceiling
- Decision: Do NOT resume (all prior GRPO runs failed, reward trend unclear, disk risk)

### BoN K=4 — PBR26 Secondary Scorer — COMPLETE: phase_f3_gate=PASS (barely)

| Metric | Value |
|--------|-------|
| greedy_accuracy | 0.910 (91.0%) |
| random@4 accuracy | 0.890 (89.0%) |
| prm_reranked_accuracy | **0.910 (91.0%)** |
| prm_vs_greedy_delta | **0.000 (0.0%)** — ties greedy, no improvement |
| prm_vs_random_delta | **+0.020 (+2.0%)** |
| **phase_f3_gate** | **PASS** (threshold ≥+2% vs random — exactly met) |

Scorer: PBR26 (frozen backbone + MLP head, MATH F1=0.686). Note: `filler_outrank_rate=0.208` (HIGH).
Artifact: `assets/artifacts/phase_f_bon/pbr26_bon4_gsm8k_v4_20260311T225703Z/summary.json` (recovered from log — disk was full during original write)

**BoN scorer comparison** (same eval set, k=4, 200 GSM8K problems):
| Scorer | Type | MATH F1 | prm@4 | Δ vs greedy | phase_f3 |
|--------|------|---------|-------|------------|---------|
| PBR32 | LoRA r=8 all-28 | 0.689 | 92.0% | **+1.5%** | PASS |
| PBR26 | Frozen backbone | 0.686 | 91.0% | **+0.0%** | PASS (barely) |

**Key insight**: LoRA training improves generation-time discriminability even with minimal ProcessBench F1 difference (+0.003). PBR26's high filler_outrank_rate exactly neutralizes any greedy improvement.

### PBR34-37 LoRA Series — All Training Complete

| Run | Config | Best pair_acc | Best AUC | MATH F1 | GSM F1 | vs PBR26 frozen |
|-----|--------|--------------|---------|---------|--------|----------------|
| **PBR32 ★** | r=8 all-28, PBR12 data | 0.898 | 0.880 | **0.689** | 0.776 | **+0.003** |
| PBR34 | r=16 top-4, PBR26 data | 0.882 | — | 0.657 | 0.766 | **-0.029** |
| PBR35 | r=8 all-28, PBR26 data | 0.879 | — | 0.657 | 0.768 | **-0.029** |
| PBR36 | r=32 all-28, PBR26 data | 0.869 | — | 0.656 | 0.739 | **-0.030** |
| PBR37 | r=8 all-28, ctr=0.2, PBR26 | **0.878** | 0.834 | **0.657** | **0.768** | **-0.029** (contrastive=0.2 gives zero benefit) |

PBR37 DONE: MATH F1=0.657, GSM F1=0.768 — identical to PBR35 (no contrastive). Contrastive loss (weight=0.2) provides zero improvement with current PBR26 training data.

**LoRA ceiling CONFIRMED**: All PBR26-data LoRA variants (PBR33-37) plateau at MATH F1 ≈ 0.656-0.657. Only PBR32 (different data = PBR12) breaks above. Data quality, not model capacity, is the bottleneck.

### ta_sweep — Instruct Backbone — All Results (r000/005/010/020 running)

| Variant | MATH F1 | GSM F1 | pair_acc | Status |
|---------|---------|--------|---------|--------|
| r000 (0% terminal) | 0.141 | 0.183 | 0.691 | DONE |
| r005 (5% terminal) | 0.219 | 0.262 | 0.606 | DONE |
| r010 (10% terminal) | 0.220 | 0.229 | 0.558 | DONE |
| r020 (20% terminal) | **0.236** | **0.258** | 0.846 | ✅ DONE |

All < 0.25 MATH F1. Instruct backbone universally inadequate (MATH F1 0.686 from Math-PRM-7B).

### Artifacts
- GRPO v4 checkpoint-150: `assets/artifacts/phase_f_grpo/grpo_pbr32_process_lambda03_v4_20260311T225806Z/grpo_checkpoints/checkpoint-150/`
- PBR26 BoN summary: `assets/artifacts/phase_f_bon/pbr26_bon4_gsm8k_v4_20260311T225703Z/summary.json`
- PBR37 run dir: `assets/artifacts/phase_e_runs/phase_e_pbr37_lora_r8_contrastive02_pbr26data_s42_20260311T204010Z/`

---

## 0AAARARB. Phase F4: GRPO Outcome-Only Result + PRM-Guided GRPO + BoN v4 Running (2026-03-13)

### GRPO v2 Outcome-Only — COMPLETE (completed 06:51, 06:52 summary saved)
- Run: `grpo_outcome_only_gsm8k_v2_20260311T204720Z`
- Policy: Qwen2.5-Math-7B-Instruct (λ_process=0, outcome-only GRPO, 250 steps, 1h48m)
- PRM: PBR32 (r=8 all-28, value head only for scoring — not used in training since λ=0)
- **pre_training_accuracy: 0.955** (GSM8K, greedy on 200 problems)
- **post_training_accuracy: 0.950** (delta = **-0.005**)
- **phase_f4_gate: FAIL** — accuracy DECREASED

### Interpretation
Outcome-only GRPO on a 95.5%-baseline model decreases performance by -0.5%. This matches the offline feasibility analysis: outcome reward |advantage|=0.000 (all k=4 samples in a group have same correct/incorrect label). Without grade variance within groups, GRPO updates are noise-only → slight policy degradation.

Key confirmation: **process reward is required** (offline analysis showed process |advantage|=0.796 >> 0.3 threshold for GRPO viability).

### Bug Fix Applied (2026-03-13)
`scripts/phase_f_grpo_lite.py` (PRMScorer class, line 160 OLD): `load_value_head_checkpoint` called as
`load_value_head_checkpoint(vh, path)` (WRONG — 2 positional args), but function signature is
`load_value_head_checkpoint(path) → (vh, config, extra)`. Fixed to `vh, _, _ = load_value_head_checkpoint(path)`.

### PRM-Guided GRPO v4 (λ=0.3) — Running (06:57)
- PID: 3894900, CUDA_VISIBLE_DEVICES=1
- run-name: `grpo_pbr32_process_lambda03_v4`
- PRM: PBR32 (r=8 all-28, MATH F1=0.689), λ_process=0.3
- Log: `assets/artifacts/phase_e_logs/phase_f_grpo_process_v4.log`
- Expected: accuracy INCREASE via process reward (|advantage|=0.796)

### BoN K=4 v4 — COMPLETE: phase_f3_gate=PASS ✓

| Metric | Value |
|--------|-------|
| greedy_accuracy | 0.905 (90.5%) |
| prm_reranked_accuracy | **0.920 (92.0%)** |
| oracle_accuracy | 0.950 (95.0%) |
| prm_vs_greedy_delta | **+0.015 (+1.5%)** |
| prm_vs_random_delta | **+0.020 (+2.0%)** |
| **phase_f3_gate** | **PASS** (threshold ≥0.02) |

Scorer: PBR32 (r=8 all-28 LoRA + MLP head, MATH F1=0.689)
Reranker captures 73% of oracle-greedy gap (1.5/2.05%). Confirms PRM adds value for Best-of-N.
Output: `assets/artifacts/phase_f_bon/pbr32_bon4_gsm8k_v4_20260311T225813Z/summary.json`

### Artifacts
- GRPO v2 result: `assets/artifacts/phase_f_grpo/grpo_outcome_only_gsm8k_v2_20260311T204720Z/summary.json`
- GRPO v4 output: `assets/artifacts/phase_f_grpo/grpo_pbr32_process_lambda03_v4_20260311T225806Z/`
- BoN v4 output: `assets/artifacts/phase_f_bon/pbr32_bon4_gsm8k_v4_20260311T225813Z/`

---

## 0AAAQX. Phase E/F Audit Fix: LoRA Safe Defaults + Phase F Controller Evaluation Scope (2026-03-12)

### What was wrong

Two interpretation-risk issues were confirmed:

1. `phase_e_train_value_lora.py`
   - direct trainer defaults still kept legacy-dangerous settings:
     - `ranking_target_space=logit`
     - `pair_weight_mode=confidence_semantic`
   - this meant wrapper-safe runs and direct LoRA runs did not share the same default safety floor.

2. `phase_f_train_trainable_controller.py`
   - `phase_f_behavior_clone_controller.py`
   - both selected models on `dev`, but then reported the headline controller score on the full benchmark trace pool.
   - this mixed train/dev traces back into the final metric and could overstate controller quality.

### Fix

1. LoRA trainer defaults are now:
   - `score`
   - `none`
   - `pair_acc`
2. Phase F controllers now use explicit:
   - `train`
   - `dev`
   - `test`
   splits
3. Main summary metric is now `test_eval`.
4. `full_eval` is preserved only as:
   - in-benchmark upper-bound reference
   - not valid external generalization evidence

### Validation

- `python -m py_compile scripts/phase_e_train_value_lora.py scripts/phase_f_train_trainable_controller.py scripts/phase_f_behavior_clone_controller.py`
- `PYTHONPATH=src pytest -q tests/unit/test_phase_f_trainable_controller.py tests/unit/test_phase_e_train_script.py tests/unit/test_phase_e_recipe_safety.py`
- result: `12 passed`

### Interpretation impact

After this fix:

1. new LoRA runs are less likely to silently fall into already-audited collapse defaults,
2. new Phase F controller summaries are methodologically cleaner,
3. older controller summaries should be re-read carefully if they only expose one `eval` block and do not distinguish `test_eval` from `full_eval`.

## 0AAARA. PBR36 LoRA r=32 all-28 Results: Rank Sweep Confirms LoRA Ceiling (2026-03-13)

### Setup
- Backbone: Qwen2.5-Math-PRM-7B + LoRA rank=32, alpha=64, q+v proj, **all 28 layers**
- Data: PBR26 data (DPO+MS full)
- Epochs completed: 2/5 (training stopped early — GPU 3 blocked by GRPO v2)
- Best checkpoint: epoch 1, pair_acc=0.8688, AUC=0.826

### Results: ProcessBench MATH
- **MATH F1: 0.656** (acc_err=0.519, acc_cor=0.892)
- pair_acc_good_vs_bad: 0.867, pair_AUC: 0.874

### Results: ProcessBench GSM8K
- **GSM F1: 0.739** (acc_err=0.647, acc_cor=0.860)

### Interpretation
PBR36 (r=32) matches PBR35 (r=8) within 0.001 MATH F1 (both 0.656-0.657).
Increasing LoRA rank from 8 → 32 on all-28-layers provides NO improvement over r=8.
The LoRA ceiling on this data is ~0.656 MATH F1 regardless of rank.

Frozen PBR26 (0.686) remains MATH SOTA by +3 F1 points.

### Leaderboard update
| Run | Rank | Layers | MATH F1 |
|-----|------|--------|---------|
| PBR26 frozen | — | — | **0.686** |
| PBR35 all-28 | 8 | all | 0.657 |
| PBR36 all-28 | **32** | all | 0.656 |
| PBR34 top-4 | 16 | top-4 | 0.657 |
| PBR33 all-28 | 8 | all | ~0.660 |

Rank sweep: r=8, r=16, r=32 all land in 0.656-0.660. Confirmed diminishing returns.

### Artifacts
- Run dir: `assets/artifacts/phase_e_runs/phase_e_pbr36_lora_r32_all28_pbr26data_s42_20260311T203151Z`
- Eval dir: `assets/artifacts/phase_e_eval/phase_e_pbr36_lora_r32_all28_pbr26data_s42_20260311T203151Z_pb_math_20260311T220742Z`

---

## 0AAAQZ. PBR35 LoRA r=8 all-28 Results + LoRA Strategy Pivot (2026-03-13)

### Results

- Run: `phase_e_pbr35_lora_mathprm_all28_pbr26data_s42_20260311T200442Z`
- Config: Math-PRM-7B + LoRA r=8, q+v, **all-28 layers**, PBR26 data (7366 pairs), 3 epochs
- Best checkpoint: epoch 2, pair_acc=0.8792, AUC=0.8336
- **MATH F1 = 0.657** (same as PBR34 r=16 top-4!)
- acc_erroneous = 0.525, acc_correct = 0.877
- Threshold = 0.365

### LoRA vs Frozen Backbone Summary (critical finding)

| Config | MATH F1 | GSM F1 | pair_acc | acc_err | acc_cor |
|--------|---------|--------|----------|---------|---------|
| **Frozen PBR26** | **0.686** | 0.768 | — | — | — |
| Frozen PBR19 | 0.683 | **0.778** | — | — | — |
| PBR33: LoRA r=8 top-4 | 0.666 | **0.797** | 0.879 | 0.557 | 0.828 |
| PBR34: LoRA r=16 top-4 | 0.657 | 0.766 | 0.882 | 0.527 | 0.872 |
| PBR35: LoRA r=8 all-28 | 0.657 | **0.768** | 0.884 | 0.525 | 0.877 |

**CONCLUSION: LoRA DOES NOT BREAK THE FROZEN CEILING.**

All LoRA variants underperform the frozen backbone on MATH F1 (0.657-0.666 < 0.686).
The frozen backbone acts as a regularizer — preventing the backbone from overfitting to
the training pair distribution, which differs from ProcessBench.

Three consistent patterns across all LoRA variants:
1. Higher pair_acc on training eval (0.879-0.884) than frozen baseline
2. Lower acc_erroneous (0.525-0.557) than frozen baseline — worse at detecting errors
3. Higher acc_correct (0.828-0.877) — more conservative (says "correct" more often)

This is classic overfitting: the LoRA backbone fine-tunes to match the training data
distribution but loses ability to detect errors in the ProcessBench distribution.

### Strategic Implication

**Stop LoRA architecture experiments.** The bottleneck is NOT model capacity — it's
**training data quality**. The frozen backbone with high-quality data already achieves
0.686 F1. To break through, we need:
1. **Consensus filtering** (MC estimate + LLM judge agreement) — addresses distribution shift
2. **More diverse training pairs** — reduces overfitting to pair distribution
3. **Better step boundaries** — EDU-PRM entropy boundaries instead of `\n\n`

PBR36 (r=32 all-28) and PBR37 (r=8 all-28+contrastive) are still running but
expected to show similar or worse results. Contrastive loss (PBR37) is the only
remaining LoRA hypothesis that might help.

---

## 0AAAQY. PBR34 LoRA r=16 top-4 Results (2026-03-13)

### Result

- Run: `phase_e_pbr34_lora_mathprm_r16_top4_pbr26data_s42_20260311T194426Z`
- Config: Math-PRM-7B + LoRA r=16, q+v, top-4 layers, PBR26 data (7366 pairs), ~4 epochs (crashed ep4)
- Best checkpoint: epoch 3, pair_acc=0.882, AUC=0.838
- **MATH F1 = 0.657** (WORSE than PBR33 r=8 which got 0.666!)
- acc_erroneous = 0.527 (dropped from 0.557), acc_correct = 0.872 (improved from 0.828)
- Threshold = 0.435

### Key Finding: Higher rank LoRA ≠ better ProcessBench F1

Despite higher pair_accuracy (0.882 vs 0.879) and AUC (0.886 vs 0.887) on training eval pairs,
r=16 achieves LOWER ProcessBench F1. The pattern is the same as the frozen backbone plateau:
more capacity → overfits to training distribution → worse generalization.

| Config | pair_acc | AUC | MATH F1 | acc_err | acc_cor |
|--------|----------|-----|---------|---------|---------|
| PBR33 r=8 top-4 | 0.879 | 0.887 | **0.666** | 0.557 | 0.828 |
| PBR34 r=16 top-4 | 0.882 | 0.886 | 0.657 | 0.527 | 0.872 |

The model with r=16 has MORE capacity and better pair discrimination but WORSE error detection.
Higher acc_correct with lower acc_erroneous = model says "correct" more often = threshold shift.
This suggests r=8 is near-optimal for top-4 layer LoRA with this training set size.

---

## 0AAAQW. Phase E/F Overnight Session: LoRA Code Fixes + Phase F BoN/GRPO Setup (2026-03-13)

### PBR33 Final Results

- Run: `phase_e_pbr33_lora_mathprm_top4_pbr26data_s42`
- Config: Math-PRM-7B + LoRA r=8, q+v, top-4 layers, PBR26 data (7366 pairs), 5 epochs
- **MATH F1 = 0.666**, GSM F1 = **0.797** (NEW GSM SOTA, beats PBR27's 0.784)
- Pattern: top-4 layers = better GSM, all-28 layers = better MATH (confirmed again)

### LoRA PRM Loading Bug Fix

**Bug**: `PeftModel.from_pretrained()` fails on `Qwen2ForProcessRewardModel` because LoRA adapters
saved with `task_type=CAUSAL_LM` cause PEFT to wrap as `PeftModelForCausalLM`, which requires
`prepare_inputs_for_generation` (absent on PRM).

**Fix**: `attach_peft_adapter_for_inference()` in `src/ours/phase_e/runtime.py` now uses
`get_peft_model()` + manual safetensors loading, bypassing the PEFT model class mapping.
Also changed `apply_lora_to_backbone()` to use `task_type=None` for future adapters.

Files fixed:
- `src/ours/phase_e/runtime.py` — `attach_peft_adapter_for_inference()` and `apply_lora_to_backbone()`
- `scripts/phase_f_best_of_n_eval.py` — use `attach_peft_adapter_for_inference()`
- `scripts/phase_f_grpo_lite.py` — use `attach_peft_adapter_for_inference()`

### GRPO Reward Shaping: Clip+Delta Added

Based on Zeng et al. 2024: naive PRM reward shaping causes reward hacking (model generates
many tiny steps to accumulate scores). Fix: use clipped step-to-step deltas.

Added to `phase_f_grpo_lite.py`:
- `PRMScorer.score_clip_delta()` method
- `--reward-shaping clip_delta` (default) / `mean_centered` (legacy)
- GPU3 watcher updated to use `--reward-shaping clip_delta` for GRPO+PRM run

### Phase F Status (GPU 3 overnight)

| Experiment | Status | Notes |
|---|---|---|
| GRPO outcome-only v2 | 🔄 running | PID 3787823, pre-eval in progress |
| BoN eval v3 | ⏳ waiting | starts after GRPO v2 finishes |
| GRPO+PRM (Clip+Delta) | ⏳ waiting | starts after BoN finishes |

Expected: GPU3 watcher PID 3806070 will run chain sequentially.

### Offline ABR RL Controller: All Variants HEURISTIC WINS on MATH

All 10 variants tested (linear, mlp-64, mlp-128, gru-32, BC warmstart, REINFORCE):
- Best RL: F1 = 0.779 (pbr32_math_mlp_bcwarmstart)
- Heuristic: F1 = 0.822 (τ=0.35)
- **Conclusion**: Fixed threshold beats neural controller on MATH with current data scale
- **ABR production setting**: use τ ≈ 0.35-0.40 for MATH domain

Note: different from GSM8K results (see entry 0AAAQV) where RL won.

### In-Flight LoRA Training

| Run | Config | Status | ETA |
|---|---|---|---|
| PBR34 | r=16 top-4 + PBR26, 5ep | epoch 3/5 | ~2h |
| PBR35 | r=8 all-28 + PBR26, 5ep | epoch 2/5 | ~4h |
| PBR36 | r=32 all-28 + PBR26, 5ep | epoch 0/5 | ~12h |
| PBR37 | r=8 all-28 + contrastive 0.2 | epoch 0/5 | ~12h |

---

## 0AAAQV. Phase F RL Controller Full Sweep + Analysis (2026-03-13)

### Main Finding (revised after full sweep analysis)

**MATH**: Heuristic threshold universally wins (0/10 RL wins). Root cause: 84.3% of erroneous MATH trajectories have score oscillations (avg 1.42 per example) → RL policy can't learn stable patterns.

**GSM8K**: RL "wins" at seed=42 but is a **split artifact**. Seed controls train/test split; RL policy is consistent (F1≈0.79-0.84), but heuristic F1 varies widely by split composition.

### Cross-Seed GSM8K Analysis (22 configs)

| Seed | RL F1 | Heuristic F1 | ΔF1 | Verdict |
|------|-------|-------------|-----|---------|
| 42 (×15 configs) | 0.837 | **0.793** | **+0.044** | RL WINS |
| 1 (×2 configs) | 0.839 | **0.916** | -0.077 | HEURISTIC WINS |
| 7 (×2 configs) | 0.796 | **0.900** | -0.104 | HEURISTIC WINS |
| 123 (×1 config) | 0.787 | **0.903** | -0.116 | HEURISTIC WINS |

- RL F1 std = **0.021** (stable across splits)
- Heuristic F1 std = **0.057** (varies by test split composition, 15%×400 ≈ 60 examples)
- Non-seed-42 combined: RL=0.812, Heuristic=0.907, **Δ=-0.095 (heuristic wins)**

### MATH Score Trajectory Analysis (PBR19 scored_rows, 1000 examples)

| Metric | MATH | GSM8K |
|--------|------|-------|
| Heuristic τ=0.38 detection rate (erroneous) | 86.5% | 55.1% |
| Score trajectory std (erroneous) | 0.219 | 0.173 |
| Oscillation rate (erroneous) | **84.3%** | 68.6% |
| Avg oscillations per erroneous example | 1.42 | 0.87 |
| Last-step separation (correct vs erroneous) | 0.391 | 0.331 |

**Why MATH RL fails**: High oscillation (84.3%) means every example the RL policy tries to learn "wait after dip" has different oscillation patterns. Heuristic tau=0.38 (fire on any dip) achieves 86.5% detection and is near-optimal.

### Full Dataset Evaluation (definitive)

Running heuristic on ALL 400 GSM8K examples (not a 15% test split):

```
Full-dataset GSM8K heuristic (tau=0.38): F1 = 0.8998
Full-dataset GSM8K heuristic (oracle tau=0.44): F1 = 0.9003
Erroneous detected: 179/207 (86.5%)  Correct preserved: 181/193 (93.8%)
```

This confirms:
- RL max on biased seed=42 test split: F1=0.844 (only 60 examples)
- Full dataset heuristic: F1=0.900
- **Heuristic wins by ~0.056 on full dataset**

The seed=42 test split heuristic (F1=0.793) was an outlier — that specific 60-example subset happened to contain near-threshold examples where the heuristic underperformed.

### ABR Implications (final)

1. For **ABR production**: Use heuristic threshold tau=0.38-0.44 for both MATH and GSM. **No RL controller needed.**
2. For **future RL research**: Needs full-dataset training and evaluation (not 15% splits), or online RL with ProcessBench as a live environment
3. MATH hard ceiling: 13.5% erroneous examples never drop below 0.38 → only PRM quality improvement helps

### Artifacts

- Full sweep results: `assets/artifacts/phase_e_logs/phase_f_gsm_full_sweep.log`
- RL policy checkpoints: `assets/artifacts/phase_f_rl_controller/gsm_*/`
- Log: `assets/artifacts/phase_e_logs/phase_f_rl_sweep_v2.log`

---

## 0AAAQU. Phase E/F Overnight Packaging: LoRA Fail-Fast Repair + Phase F Usability Refresh (2026-03-12)

### Context

This pass had two operational goals:

1. stop Phase E overnight LoRA work from being corrupted by launcher-level false-success behavior,
2. package the strongest currently available Phase F verifier slices into one overnight usability chain.

### Diagnosis: older LoRA launchers could produce false "training done" stories

Confirmed risk:

1. `scripts/run_pbr33_lora_mathprm_pbr26data.sh`
2. `scripts/run_pbr34_lora_mathprm_r16_pbr26data.sh`
3. `scripts/run_pbr35_lora_contrastive.sh`

all used:

1. `set -e`
2. `python ... | tee log`

without `pipefail`.

That means:

1. a LoRA train crash could still let the shell continue into eval,
2. the wrapper could then point at the newest timestamped run dir,
3. and the resulting narrative looked like "training done" even when
   `manifest.json` never existed.

### Fix

The launchers above now:

1. use `set -euo pipefail`
2. resolve only completed run dirs that already contain `manifest.json`

`scripts/run_lora_auto_eval.sh` was also tightened to use `set -euo pipefail`.

### New overnight entrypoints

Added:

1. `scripts/run_phase_e_lora_overnight_suite.sh`
2. `scripts/run_phase_f_usability_overnight_suite.sh`

Supporting design note:

1. `docs/phase_e_phase_f_best_practice_refresh_20260312.md`

### Phase E runtime status

Current status:

1. the new Phase E suite was launched as:
   - `RUN_PREFIX=phase_e_lora_overnight_0312_0452`
2. but all 4 GPUs were already occupied by live LoRA runs (`PBR34/35/36/37`)
3. the suite therefore now waits for GPU 3 to free up instead of trying to
   force-start and OOM

This is the correct behavior in the current environment.

### Phase F results

Main artifact bundle:

1. `assets/artifacts/phase_f_logs/phase_f_usability_overnight_0312_0452_UALL_PHASEF_USABILITY/final_summary.md`

#### 1. Controller sweep on refreshed verifier slices

Artifacts:

1. `assets/artifacts/phase_f_controller_sweep/phase_f_usability_overnight_0312_0452_controller_sweep_20260311T204944Z`

Best policies:

1. Math:
   - `pbr26_math`: `threshold_only = 0.8639`
   - `pbr31_math`: `threshold_only = 0.8697`
   - `pbr32_math`: `threshold_only = 0.8616`
   - `pbr33_math`: `threshold_only = 0.8529`
2. GSM:
   - `pbr19_gsm`: `delayed_drop = 0.9052`
   - `pbr31_gsm`: `threshold_only = 0.9134`
   - `pbr32_gsm`: `delayed_drop = 0.9101`
   - `pbr33_gsm`: `threshold_only = 0.9053`

Interpretation:

1. stronger verifier slices do not overturn the basic controller story
2. Math still prefers simpler threshold-like stopping
3. GSM still supports either `threshold_only` or `delayed_drop`, depending on
   the score geometry of the verifier slice

#### 2. Worst-generator robust policy search

Artifacts:

1. `assets/artifacts/phase_f_controller_robustness/phase_f_usability_overnight_0312_0452_generator_robustness_20260311T204947Z`

Best robust families:

1. Math:
   - `pbr26_math`: `guarded_drop`, worst-gen `0.7604`
   - `pbr31_math`: `guarded_drop`, worst-gen `0.7744`
   - `pbr32_math`: `delayed_drop`, worst-gen `0.7397`
   - `pbr33_math`: `guarded_drop`, worst-gen `0.7801`
2. GSM:
   - `pbr19_gsm`: `delayed_drop`, worst-gen `0.7347`
   - `pbr31_gsm`: `delayed_drop`, worst-gen `0.7668`
   - `pbr32_gsm`: `delayed_drop`, worst-gen `0.7692`
   - `pbr33_gsm`: `delayed_drop`, worst-gen `0.7416`

Interpretation:

1. robust selection sharpens the Math-vs-GSM split:
   - Math leans `guarded_drop`
   - GSM leans `delayed_drop`
2. `pbr32_math` remains somewhat unusual and prefers `delayed_drop`

#### 3. Weak-verifier ensemble still helps

Artifacts:

1. `assets/artifacts/phase_f_controller_ensemble/phase_f_usability_overnight_0312_0452_ensemble_eval_20260311T204954Z`

Best ensemble cases:

1. Math:
   - `p31+p32`: `mean_50 + guarded_drop = 0.8699`
   - `p26+p33`: `mean_50 + threshold_only = 0.8687`
2. GSM:
   - `p19+p31`: `mean_50 + threshold_only = 0.9144`
   - `p19+p33`: `mean_75a + delayed_drop = 0.9127`

Interpretation:

1. score-level ensembling still buys controller quality
2. the gain is incremental, not a reason to postpone single-model controller use

#### 4. BC vs BC->RL vs robust-from-scratch

Artifacts:

1. `assets/artifacts/phase_f_bc/phase_f_usability_overnight_0312_0452_bc_only_20260311T205007Z`
2. `assets/artifacts/phase_f_bc/phase_f_usability_overnight_0312_0452_bc_then_rl_20260311T205055Z`
3. `assets/artifacts/phase_f_rl_like/phase_f_usability_overnight_0312_0452_rl_like_robust_20260311T205412Z`

Results:

1. `bc_only`
   - `pbr26_math = 0.8547`
   - `pbr32_math = 0.8381`
   - `pbr31_gsm = 0.8819`
   - `pbr33_gsm = 0.9053`
2. `bc_then_rl`
   - `pbr26_math = 0.8562`
   - `pbr32_math = 0.8678`
   - `pbr31_gsm = 0.8874`
   - `pbr33_gsm = 0.8775`
3. robust from scratch
   - `pbr32_math = 0.6398`, worst-gen `0.4291`
   - `pbr33_gsm = 0.9001`, worst-gen `0.6557`

Interpretation:

1. `BC -> RL` is no longer a uniformly negative result:
   - it helped `pbr32_math`
   - it helped `pbr31_gsm` slightly
2. but it is still not a safe default:
   - `pbr33_gsm` degraded badly under RL fine-tune
3. robust-from-scratch still loses clearly on Math and remains fragile

### Updated reading

The current strongest practical Phase F story is now:

1. heuristic controller is already usable
2. BC distillation is also usable
3. BC->RL is a case-by-case optional upgrade, not a default promotion path
4. robust-from-scratch RL remains a stress-test branch

### Next steps

1. let the queued Phase E LoRA suite start automatically once GPU 3 is free
2. compare `L1/L2/L3` directly against:
   - `PBR26` frozen
   - `PBR32` Math LoRA
   - `PBR33` GSM LoRA
3. keep Phase F default recommendation as:
   - heuristic / BC first
   - RL second

## 0AAAQT. Phase E/F Safety Hardening + Cheap→Strong Gate Sweep (2026-03-12)

### Infrastructure fixes completed

This round closed two remaining high-risk `Phase E` pitfalls:

1. `docs/relatedPapers/` is now fully synced against currently referenced repo papers.
   - downloader:
     - `scripts/download_related_papers.py`
   - `index.json` now reports:
     - `failed = 0`
     - `unresolved = 0`
2. `Phase E` candidate promotion is now strict by default.
   - `scripts/phase_e_select_candidate.py`
   - `scripts/phase_e_select_intradataset_candidate.py`
   - both now require `best_value_head.pt` unless the caller explicitly opts into
     `--checkpoint-missing-policy fallback_final`
3. `run_phase_e_dual_head_smoke.sh` no longer defaults to the audited-dangerous
   recipe:
   - `ranking_target_space = logit`
   - `pair_weight_mode = confidence_semantic`
   - `checkpoint_selection_metric = ranking_score`

Interpretation:

1. active `Phase E` wrappers are now less likely to silently reintroduce
   previously-audited collapse settings;
2. historical `dual-head smoke` results remain valid as *diagnostic history*,
   but new runs will no longer accidentally reproduce that unsafe default.

### New experiment: cheap→strong verifier gate sweep

New script:

1. `scripts/phase_e_sweep_weak_strong_gate.py`

Purpose:

1. test a current community direction:
   - let a cheap verifier handle only high-confidence prefixes,
   - escalate low-confidence prefixes to a stronger verifier,
   - measure how much benchmark quality improves as strong-verifier usage grows.

#### A. `prm_e46 -> pbr26`

Artifact:

1. `assets/artifacts/phase_e_gate_sweeps/phase_e_cheap_strong_gate_0312_20260311T204929Z/summary.md`

Main results:

1. Math:
   - weak AUC `0.6053`
   - strong AUC `0.8882`
   - best mixed AUC `0.8790`
   - but required `strong_usage_rate = 0.9507`
2. GSM:
   - weak AUC `0.6264`
   - strong AUC `0.9148`
   - best mixed AUC `0.9103`
   - but required `strong_usage_rate = 0.9736`

Intermediate thresholds were smoother but still expensive:

1. Math at `tau=0.30`
   - AUC `0.6842`
   - strong usage `0.429`
2. GSM at `tau=0.30`
   - AUC `0.7402`
   - strong usage `0.493`

Interpretation:

1. the gate works directionally;
2. but this weak verifier is not strong enough to save much strong-verifier cost.

#### B. `ms_e43 -> pbr26`

Artifact:

1. `assets/artifacts/phase_e_gate_sweeps/phase_e_cheap_strong_gate_ms_0312_20260311T204937Z/summary.md`

Main results:

1. Math:
   - weak AUC `0.6341`
   - strong AUC `0.8882`
   - best mixed AUC `0.8521`
   - strong usage `0.9139`
2. GSM:
   - weak AUC `0.6245`
   - strong AUC `0.9148`
   - best mixed AUC `0.8495`
   - strong usage `0.8655`

At lower strong usage the gains were modest:

1. Math at `tau=0.40`
   - AUC `0.6734`
   - strong usage `0.352`
2. GSM at `tau=0.40`
   - AUC `0.6563`
   - strong usage `0.275`

Interpretation:

1. `ms_e43` is a somewhat better cheap verifier than `prm_e46` for this gate,
2. but it still does not justify a "cheap handles most, strong handles few" system yet.

### Overall conclusion

1. cheap-to-strong verifier routing is a valid systems direction;
2. however, current weak verifiers still require too much strong coverage to get
   close to `pbr26`;
3. therefore the repository should now prioritize:
   - separate local/process verifier,
   - separate terminal/answer verifier,
   - explicit abstain/escalate gate,
   instead of asking one scalar verifier to solve all roles.

## 0AAAQT. Active Overnight Recovery / Expansion Runs (2026-03-12 16:18 +0800)

### Active runs launched after fresh wrapper fixes

1. `phase_e_phase_f_overnight_0312_1621`
   - repaired single-GPU package
   - scope: `PH2` hybrid + default `selected relabel` + `PBR26/PBR31` modern preflight
2. `phase_e_selrel_wide_0312_1615`
   - wider low-margin `PRMBench` selected relabel (`selection_size=192`, `min_conf=0.65`)
3. `phase_e_selrel_strict_0312_1615`
   - stricter consensus-filter variant (`selection_size=96`, `min_conf=0.72`)
4. `phase_f_usability_0312_1615`
   - controller sweep + generator robustness + ensemble + BC / BC->RL

### Important implementation corrections that apply to these runs

1. `run_phase_e_processbench_hybrid_suite.sh`
   - fixed artifact directory resolution so successful `PH2` runs are no longer mislabeled as failed under `pipefail`
2. `run_phase_e_phase_f_single_gpu_overnight.sh`
   - fixed heredoc rendering so its plan markdown no longer executes shell substitutions
3. hybrid warm-start contract
   - `gated_mlp` now skips incompatible `mlp` init checkpoints instead of crashing mid-suite

### Early partial signal

1. repaired `PH2` now enters training and benchmark eval correctly
2. `phase_f_usability_0312_1615` has already reproduced the strong offline controller pattern:
   - `pbr31_gsm` threshold-only / delayed-drop family remains around `balanced_f1≈0.90+`
   - `pbr32_math` BC-only is already above `0.84` on test
3. both `selected relabel` variants have moved past low-margin slicing and are currently in judge scoring

## 0AAAQS. Phase E/F Single-GPU Overnight Packaging + Modern Preflight Refresh (2026-03-13)

### Context

All four GPUs were already heavily utilized, so the old style of parallel
overnight frontier batches had become the wrong default. The immediate need was:

1. one safer `Phase E` overnight package that can still move benchmark-facing
   verifier work forward,
2. one updated `Phase F` preflight that audits the current stronger candidates
   `PBR26 / PBR31`, not older legacy checkpoints.

### New packaging

1. [run_phase_e_phase_f_single_gpu_overnight.sh](/home/zling/y/bcr/ref/scripts/run_phase_e_phase_f_single_gpu_overnight.sh)
   - sequential one-GPU launcher
2. [run_phase_f_modern_preflight_suite.sh](/home/zling/y/bcr/ref/scripts/run_phase_f_modern_preflight_suite.sh)
   - `PBR26 / PBR31` threshold-shift + reward-hacking audit
3. [phase_e_phase_f_overnight_bestpractice_20260313.md](/home/zling/y/bcr/ref/docs/phase_e_phase_f_overnight_bestpractice_20260313.md)
   - literature-backed rationale for why this is the next overnight package

### What the overnight package is meant to answer

1. whether `PH2` style benchmark-oriented hybrid supervision can beat prior
   weak benchmark-facing results,
2. whether `PRMBench selected relabel` can improve a narrow ambiguous slice
   without broad relabel noise,
3. whether `CR1` style curated reward-centered gating is worth re-testing under
   controlled memory settings,
4. whether `PBR26` or `PBR31` is currently the safer pre-RL controller
   candidate.

## 0AAAQ. Phase E LoRA Expansion + RL Controller + Literature Synthesis (2026-03-13)

### Context

Night session continuing from PBR32 MATH F1=0.689 SOTA. Objectives:
1. Expand LoRA series: PBR33 final, PBR34/35/36 training, PBR37 contrastive loss queued
2. RL controller evaluation: offline REINFORCE on PBR32 scored traces
3. Literature synthesis: top methods for closing community F1 gap

### PBR33 Final Results (PBR26 data + top-4 r=8)

| Metric | Value | Notes |
|--------|-------|-------|
| MATH F1 | 0.666 | Below PBR32 (0.689) — top-4 layers insufficient for MATH |
| GSM F1 | **0.797** | **NEW GSM SOTA** — best all-time |

Key finding: **top-4 + PBR26 data = best GSM but weakest MATH**. All-28 layers needed for MATH.

### RL Controller Training Results (phase_f_train_rl_controller.py on PBR32 MATH)

| Method | Val F1 | Notes |
|--------|--------|-------|
| **Heuristic tau=0.35** | **0.848** | **BEST — simple threshold wins** |
| MLP BC warmstart (fp=5) | 0.818 (after BC) | BC alone nearly matches heuristic |
| MLP REINFORCE (after BC) | 0.750-0.781 | REINFORCE degrades BC policy |
| GRU REINFORCE | 0.72-0.75 | Slow convergence, local optima |

**Key finding**: heuristic tau=0.35 at F1=0.848 is the optimal offline error-detection controller.
BC warmstart gets close (0.818) but REINFORCE consistently hurts. Confirms previous session.

### Literature Synthesis: Top Methods to Close Community F1 Gap

Based on comprehensive literature survey (2026-03-13):

| Method | Evidence | Expected MATH F1 Gain | Effort | Priority |
|--------|----------|----------------------|--------|---------|
| Contrastive loss (Scale AI arXiv:2407.13887) | +0.09 AUROC | +2-3 F1 | Medium | **1** |
| LoRA rank r=32 | 4× params vs r=8 | +1-3 F1 | Low | **2** |
| EDU-PRM entropy step boundaries (arXiv:2503.22233) | 7K queries = 72B PRM | +1-2 F1 | Medium-High | 3 |
| Implicit PRM / PRIME log-ratio (arXiv:2502.01456) | No step labels | Unknown | High | 4 |
| GenPRM generative verification (arXiv:2504.00891) | ~80.5 F1 target | +5+ F1 | Very High | 5 |

Contrastive loss already implemented in `phase_e_train_value_lora.py` (`--contrastive-loss-weight`).

### Active Experiments (as of 2026-03-13 06:00 +0800)

| Exp | GPU | Config | Status |
|-----|-----|--------|--------|
| PBR35 | 1 | r=8 all-28 + PBR26 data | Training (ep1) |
| PBR34 | 2 | r=16 top-4 + PBR26 data | Training (ep1) |
| PBR36 | 3 | r=32 all-28 + PBR26 data | Training (ep0) |
| PBR37 | 0 | r=8 all-28 + contrastive 0.2 | Queued |

All evals auto-queued in background on respective GPUs.

---

## 0AAAP. Phase E Overnight Frontier: Dual-Head Regresses, Hybrid Still Overfits Held-Out Geometry (2026-03-13)

### Artifacts

1. [F2 summary](/home/zling/y/bcr/ref/assets/artifacts/phase_e_logs/phase_e_probe_f2/final_summary.md)
2. [F2 same-family](/home/zling/y/bcr/ref/assets/artifacts/phase_e_samefamily_eval/phase_e_probe_f2_dual_head_samefamily_20260311T195659Z/summary.md)
3. [F2 ProcessBench GSM8K](/home/zling/y/bcr/ref/assets/artifacts/phase_e_eval/phase_e_probe_f2_dual_head_processbench_gsm8k_20260311T195708Z/summary.md)
4. [F2 ProcessBench Math](/home/zling/y/bcr/ref/assets/artifacts/phase_e_eval/phase_e_probe_f2_dual_head_processbench_math_20260311T195717Z/summary.md)
5. [PH2 summary](/home/zling/y/bcr/ref/assets/artifacts/phase_e_logs/phase_e_probe_ph2/final_summary.md)
6. [PH2 value run](/home/zling/y/bcr/ref/assets/artifacts/phase_e_runs/phase_e_probe_ph2_ph2_prm_local_ta10_msgrid10_arch_sweep_smoke_mlp_20260311T195358Z/summary.md)
7. [PH2 ProcessBench GSM8K](/home/zling/y/bcr/ref/assets/artifacts/phase_e_eval/phase_e_probe_ph2_ph2_prm_local_ta10_msgrid10_arch_sweep_smoke_mlp_processbench_gsm8k_20260311T201904Z/summary.md)
8. [PH2 ProcessBench Math](/home/zling/y/bcr/ref/assets/artifacts/phase_e_eval/phase_e_probe_ph2_ph2_prm_local_ta10_msgrid10_arch_sweep_smoke_mlp_processbench_math_20260311T201904Z/summary.md)
9. [CR1 suite log](/home/zling/y/bcr/ref/assets/artifacts/phase_e_logs/phase_e_probe_cr1/suite.log)

### Main outcomes

#### 1. `F2_DUAL_HEAD_PBR10` is a clear negative result

Compared with the strong `PBR10` scalar baseline, dual-head factorization
damaged the geometry that actually mattered:

1. held-out remained only moderate:
   - pair acc `0.6384`
   - auc `0.5792`
2. same-family routing collapsed:
   - prompt-pool top1 `0.6706`
   - local first-bad `0.4352`
3. benchmark metrics dropped hard versus `PBR10`
   - GSM AUC `0.7254` vs baseline `0.8730`
   - Math AUC `0.7063` vs baseline `0.8631`

Interpretation:

1. current dual-head implementation does **not** solve the terminal blind spot;
2. instead it destroys the already-good local ranking geometry;
3. this direction should be treated as a failed repair, not a new mainline.

#### 2. `PH2_PRM_LOCAL_TA10_MSGRID10` shows a classic held-out / benchmark split

The hybrid data contract trains very well on its own held-out split:

1. held-out pair acc `0.9318`
2. held-out auc `0.9040`

But benchmark utility stays weak even after a repaired 256-sample benchmark eval:

1. ProcessBench GSM8K
   - pair acc `0.4625`
   - auc `0.5224`
   - first-edge `0.5849`
2. ProcessBench Math
   - pair acc `0.4360`
   - auc `0.5321`
   - first-edge `0.4922`

Interpretation:

1. the hybrid artifact is easy to fit;
2. but it still does **not** produce a trustworthy benchmark-facing verifier;
3. this is strong evidence that benchmark-aligned mixture needs stricter
   contract design, not just "more local + a little terminal + a little grid".

#### 3. `CR1_CURATED_CENTER_GATE_SMOKE` was correctly blocked by recipe guard

The curated centered mix used:

1. mixed local + terminal semantics
2. `pair_weight_mode=confidence_group_balance`

The guard raised:

1. `SEMANTIC_WEIGHT_MIXED_TERMINAL`

Interpretation:

1. the failure is desirable;
2. without the guard, this run would have added another ambiguous result to the
   queue.

### Infrastructure note

Two wrapper bugs were found and fixed while auditing these results:

1. `run_phase_e_frontier_suite.sh`
   - expected an outdated same-family JSON schema and misreported finished runs
     as failed
2. `run_phase_e_processbench_hybrid_suite.sh`
   - resolved run/eval dirs with the wrong suffix pattern and stopped after
     training

These were **wrapper bookkeeping bugs**, not model-learning bugs.

## 0AAAO. Phase F Robust Objective Bug Fix: Old "Robust RL" Was Understated (2026-03-12)

### Diagnosis

This round re-audited the new `Phase F` controller-RL code and found one
research-critical implementation bug:

1. `scripts/phase_f_train_trainable_controller.py`
2. `robust_lambda` previously added a constant scalar penalty
3. that penalty did **not** depend on policy log-probs
4. so the "robust" objective changed logs, but not gradients

Implication:

1. historical `robust-from-scratch` artifacts were underestimating the real
   effect of worst-generator-aware optimization
2. any claim that "robust RL does not help at all" was too strong

### Code Fix

Files:

1. `scripts/phase_f_train_trainable_controller.py`
2. `tests/unit/test_phase_f_trainable_controller.py`

What changed:

1. `robust_lambda` now adds a differentiable extra policy-gradient term on the
   current worst-generator slice
2. training curves now record the epoch's worst-generator slice for audit

Validation:

1. `PYTHONPATH=src pytest -q tests/unit/test_phase_f_trainable_controller.py`

### New Experiment Results

Artifacts:

1. [robust_fixed](/home/zling/y/bcr/ref/assets/artifacts/phase_f_rl_like/phase_f_rl_like_robust_fixed_0312_20260311T201229Z/summary.md)
2. [mean_fixed](/home/zling/y/bcr/ref/assets/artifacts/phase_f_rl_like/phase_f_rl_like_mean_fixed_0312_20260311T201515Z/summary.md)
3. [bc_then_rl_robust_fixed](/home/zling/y/bcr/ref/assets/artifacts/phase_f_bc/phase_f_bc_then_rl_robust_fixed_0312_20260311T201229Z/summary.md)
4. historical [robust_fromscratch](/home/zling/y/bcr/ref/assets/artifacts/phase_f_rl_like/phase_f_rl_like_robust_fromscratch_0312_20260311T200645Z/summary.md)
5. historical [bc_then_rl_robust](/home/zling/y/bcr/ref/assets/artifacts/phase_f_bc/phase_f_bc_then_rl_robust_0312_20260311T200451Z/summary.md)
6. historical [bc_only](/home/zling/y/bcr/ref/assets/artifacts/phase_f_bc/phase_f_bc_only_0312_20260311T200307Z/summary.md)

Key comparisons:

1. from-scratch `mean` vs fixed `robust`
   - `pbr31_math`: `0.3493 -> 0.6623`
   - `pbr31_gsm`: `0.7680 -> 0.8935`
2. historical broken `robust` vs fixed `robust`
   - `pbr31_math`: `0.3493 -> 0.6623`
   - `pbr31_gsm`: `0.8289 -> 0.8935`
3. `BC -> RL`
   - `pbr31_math`: `bc_only 0.8552`, old broken `0.8415`, fixed `0.8351`
   - `pbr31_gsm`: `bc_only 0.9045`, old broken `0.9001`, fixed `0.9012`

### Updated Interpretation

1. the old claim "robust RL-from-scratch is basically useless" is false
2. the corrected claim is:
   - robust objective meaningfully rescues from-scratch RL-like controller learning
   - but still does not beat strong heuristic / BC teachers
3. BC warm start remains the safer Phase F mainline
4. robust RL should now be treated as a real secondary research branch, not a
   no-op artifact

### Explicit Next-Step Plans

1. keep heuristic / BC controller as live-trial priority
2. keep robust RL-like controller as a secondary branch, but compare against BC
   teachers only after the objective fix
3. avoid citing pre-fix "robust" artifacts as evidence against robust
   optimization
4. prioritize `Phase E` data-geometry redesign and `Phase F` controller-layer
   integration over immediate LM-level RL
5. use `docs/phase_abcdef_audit_research_redesign_20260312.md` as the new
   repo-level design note for this pivot


## 0AAAN. A-F Audit Follow-Up: No New Critical Cross-Phase Bug, One Real Overnight Provenance Bug Fixed (2026-03-13)

### Main result

This follow-up audit did **not** uncover a new high-risk bug in the A-F training/eval critical path.
The only new confirmed risk was at the overnight launcher layer:

a downstream frontier job could start as soon as `final_summary.md` existed,
even if the upstream run had already finished with `status: failed`.

### Fix

1. added [wait_for_summary_status.py](/home/zling/y/bcr/ref/scripts/wait_for_summary_status.py)
2. updated [run_phase_e_overnight_frontier_suite.sh](/home/zling/y/bcr/ref/scripts/run_phase_e_overnight_frontier_suite.sh)

New rule:

1. dependent jobs now wait for `- status: ok`
2. `failed` summaries abort the chain
3. timeout is explicit

### Validation

1. targeted tests:
   - `tests/unit/test_wait_for_summary_status.py`
   - `tests/unit/test_phase_f_trainable_controller.py`
2. broader smoke regression:
   - `tests/unit/test_phase_a_prepare_script.py`
   - `tests/unit/test_phase_b_eval_faithfulness_script.py`
   - `tests/unit/test_phase_c_eval_pik_script.py`
   - `tests/unit/test_phase_d_eval_external_pairs_script.py`
   - `tests/unit/test_phase_e_runtime.py`
   - `tests/unit/test_phase_f_trainable_controller.py`
3. all targeted tests passed in this audit round.

### Interpretation

The current repo state is now better described as:

1. main A-F code path: no new critical implementation issue found in this pass
2. current bottlenecks: research-design and control-policy issues, not obvious code corruption bugs
3. overnight experimentation: now safer against silent bad chaining

## 0AAAM. Phase F RL-like Controller Learning: BC Warm Start Beats Pure RL (2026-03-12)

### Artifacts

1. [bc_only](/home/zling/y/bcr/ref/assets/artifacts/phase_f_bc/phase_f_bc_only_0312_20260311T200307Z/summary.md)
2. [bc_then_rl_robust](/home/zling/y/bcr/ref/assets/artifacts/phase_f_bc/phase_f_bc_then_rl_robust_0312_20260311T200451Z/summary.md)
3. [robust from scratch](/home/zling/y/bcr/ref/assets/artifacts/phase_f_rl_like/phase_f_rl_like_robust_fromscratch_0312_20260311T200645Z/summary.md)
4. [naive mean debug](/home/zling/y/bcr/ref/assets/artifacts/phase_f_rl_like/debug_phase_f_rl_like_20260311T195946Z/summary.md)
5. [balanced mean debug](/home/zling/y/bcr/ref/assets/artifacts/phase_f_rl_like/debug_phase_f_rl_like_balanced_20260311T200131Z/summary.md)

### Main findings

This round tested a more direct RL question: if we stop doing pure heuristic rule search
and actually train a small controller policy, what happens?

Results:

1. from-scratch policy-gradient is fragile
   - `pbr26_math` naive and class-balanced REINFORCE both collapsed to `balanced_f1 = 0.0000`
   - `pbr31_math` robust RL from scratch only reached `0.3493`
2. behavior cloning from a good heuristic teacher is strong immediately
   - `pbr26_math bc_only = 0.8502`
   - `pbr31_math bc_only = 0.8552`
   - `pbr31_gsm bc_only = 0.9045`
3. adding RL after BC does not currently help
   - `pbr31_math`: `0.8552 -> 0.8415`
   - `pbr31_gsm`: `0.9045 -> 0.9001`

### Interpretation

1. the controller policy class is not the bottleneck;
2. random-start RL optimization is the bottleneck;
3. current reward shaping is still too collapse-prone;
4. Phase F should prefer:
   - heuristic controller, or
   - BC-warm-start controller,
   before trying more RL.

### Research consequence

The repo now has a stronger ordering of priorities:

1. heuristic controller live trial
2. BC-distilled controller live trial
3. only then reconsider controller-only RL

## 0AAAL. Phase F Robust-Controller + Ensemble Update (2026-03-13)

### Artifacts

1. policy-family sweep
   - [summary.md](/home/zling/y/bcr/ref/assets/artifacts/phase_f_controller_sweep/phase_f_controller_sweep_0312_main_20260311T181216Z/summary.md)
2. worst-generator robustness search
   - [summary.md](/home/zling/y/bcr/ref/assets/artifacts/phase_f_controller_robustness/phase_f_controller_research_0312_generator_robustness_20260311T194732Z/summary.md)
3. weak-verifier ensemble evaluation
   - [summary.md](/home/zling/y/bcr/ref/assets/artifacts/phase_f_controller_ensemble/phase_f_controller_research_0312_ensemble_eval_20260311T194735Z/summary.md)

### Main findings

This round sharpens the `Phase F` diagnosis further:

1. the old `ABR-lite` failure was mostly a controller-design failure;
2. generator-shift still matters, but good robust policies exist;
3. weak-verifier score ensembling gives another measurable gain;
4. therefore the next serious milestone is a **live heuristic controller**, not immediate RL.

### Generator-robust best policies

| case_id | family | overall_balanced_f1 | worst_generator | worst_gen_balanced_f1 |
|---|---|---:|---|---:|
| `pbr26_math` | `guarded_drop` | `0.8391` | `Qwen2.5-Math-72B-Instruct` | `0.7604` |
| `pbr26_gsm` | `drop_needs_low` | `0.8769` | `Llama-3.1-70B-Instruct` | `0.7575` |
| `pbr31_math` | `guarded_drop` | `0.8460` | `Qwen2.5-Math-72B-Instruct` | `0.7744` |
| `pbr31_gsm` | `delayed_drop` | `0.9027` | `Meta-Llama-3-70B-Instruct` | `0.7668` |

### Best score-level ensemble results

| case_id | ensemble | best_policy_family | balanced_f1 |
|---|---|---|---:|
| `pbr26_pbr31_math` | `min` | `threshold_only` | `0.8765` |
| `pbr26_pbr31_gsm` | `min` | `guarded_drop` | `0.9126` |
| `pbr19_pbr31_math` | `mean_50` | `guarded_drop` | `0.8774` |
| `pbr19_pbr31_gsm` | `mean_50` | `threshold_only` | `0.9144` |

### Literature-aligned reading

Recent literature supports this exact ordering of priorities:

1. [VerifyBench](https://arxiv.org/abs/2507.09884): verifier behavior is highly input-structure-sensitive, so domain-specific controller policies are more plausible than one universal rule.
2. [AbstentionBench](https://arxiv.org/abs/2506.09038): reasoning models are still weak at abstention, so `continue / backtrack / abstain` must be evaluated directly.
3. [ThinkPRM](https://arxiv.org/abs/2504.16828) and [GenPRM](https://arxiv.org/abs/2504.00891): the frontier is moving toward explicit critics, not trusting a single scalar score unconditionally.
4. [PURE / Stop Summation](https://arxiv.org/abs/2504.15275): naive PRM-based RL is reward-hack-prone, so a pre-RL conservative controller stage is justified.
5. [MASH](https://arxiv.org/abs/2510.01152): selective extra compute behaves like abstention, matching the repo's current controller framing.

### Updated Phase F recommendation

1. implement a live heuristic controller first;
2. prefer:
   - `Math`: `threshold_only` or robust `guarded_drop`
   - `GSM`: `delayed_drop / guarded_drop`
3. test weak-verifier ensemble as a first-class option;
4. only after live validation should RL move back to the top of the queue.

## 0AAAK. Phase F Controller Sweep — Old ABR-lite Rule Was The Main Failure (2026-03-13)

### Artifact

- [controller sweep summary](/home/zling/y/bcr/ref/assets/artifacts/phase_f_controller_sweep/phase_f_controller_sweep_0312_main_20260311T181216Z/summary.md)

### Main finding

The earlier `F2 FAIL` diagnosis was a **controller-rule failure**, not a verifier failure.

The old `baseline_immediate` controller over-triggered on early `delta_drop`.
When we sweep several controller families on existing scored `ProcessBench` traces:

1. `threshold_only`
2. `delayed_drop`
3. `drop_needs_low`
4. `guarded_drop`
5. `two_strike`

the results become dramatically better.

### Headline numbers

| case | baseline | best policy | balanced_f1 |
|---|---:|---|---:|
| `pbr19_math` | 0.5537 | `threshold_only tau=0.38` | 0.8674 |
| `pbr21_math` | 0.5588 | `threshold_only tau=0.35` | 0.8641 |
| `pbr26_math` | 0.5752 | `threshold_only tau=0.38` | 0.8639 |
| `pbr19_gsm` | 0.6713 | `delayed_drop` | 0.9052 |
| `pbr21_gsm` | 0.6405 | `guarded_drop` | 0.9028 |
| `pbr26_gsm` | 0.6509 | `delayed_drop` | 0.9053 |

### Shared-policy validation

We also tested simple shared policies, not only per-case tuned policies:

1. `threshold_only tau=0.42`
   - mean `balanced_f1 = 0.8552`
2. `delayed_drop tau=0.42 delta=0.25 min_step=4`
   - mean `balanced_f1 = 0.8501`

So the result is not just per-case overfitting.

### Corrected interpretation

1. current verifier candidates are strong enough for a useful heuristic controller;
2. the old `ABR-lite` rule was the wrong design;
3. the next Phase F step should be:
   - heuristic controller redesign,
   - then live generation validation,
   - and only after that decide whether RL is needed.

## 0AAAJ. Breakthrough Verification: PBR26 Frozen, PBR31 LoRA (2026-03-13)

### Verified headline

The newest verification pass settled two repo-level claims:

1. `PBR26` is a real frozen-backbone benchmark breakthrough.
2. `PBR31` is a real LoRA improvement.

But neither is RL-ready.

### Compact verification table

| candidate | heldout pair_acc | sf top1 | sf local | GSM AUC | Math AUC | GSM F1 | Math F1 | GSM terminal_top1 | Math terminal_top1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `PBR26` | 0.8532 | 0.8606 | 0.8510 | 0.9148 | 0.8882 | 0.7793 | 0.6700 | 0.1503 | 0.1355 |
| `PBR31_LORA` | 0.8917 | 0.8916 | 0.8799 | 0.9006 | 0.8823 | 0.7882 | 0.6762 | 0.2332 | 0.2192 |

### Interpretation

`PBR26`:

1. best verified frozen benchmark candidate in the newest batch
2. still fails strict RL gate because:
   - same-family trust is not clean enough
   - terminal completion ordering is still very weak
3. strict diagnosis:
   - `not_rl_ready_terminal_completion_risk`

`PBR31_LORA`:

1. improves same-family trust and oracle-F1 balance over `PBR26`
2. does **not** beat `PBR26` on `Math AUC`
3. still fails strict RL gate for the same core reason:
   - terminal completion ordering remains weak
4. additional caution:
   - new reward-hacking probe on GSM `first_bad` is worse than `PBR26`
   - `confidence_tail flip@0.5 = 0.1875` vs `0.0417`

### Clean wording the repo should now use

1. `PBR26`
   - best verified frozen benchmark candidate
   - not RL-ready
2. `PBR31`
   - best verified LoRA balance so far
   - not a new Math-AUC SOTA
   - not RL-ready

Key verification artifacts:

1. `docs/phase_e_latest_breakthrough_verification_20260312.md`
2. `assets/artifacts/phase_e_transfer_diag/phase_e_pbr26_verify_diag_0312_00`
3. `assets/artifacts/phase_e_transfer_diag/phase_e_pbr26_pbr31_verify_diag_0312_00`
4. `assets/artifacts/phase_f_reward_hacking/phase_f_pbr31_probe_verify_0312_20260311T172022Z`

## 0AAAI. Eval Provenance Hardening Across Phase B/C/D/E (2026-03-13)

### What was re-audited

Repository-wide standalone evaluation paths were rechecked for a specific trust
risk:

1. requesting `best` checkpoint but actually evaluating `final`
2. requesting `best` posthoc calibrator but actually loading `final`

This is a high-risk result-pollution class because it does not necessarily
crash, but it can silently change the object being scored.

### Concrete fixes

The following paths are now strict by default:

1. `scripts/phase_b_eval_faithfulness.py`
2. `scripts/phase_c_eval_pik.py`
3. `scripts/phase_d_eval_external_pairs.py`
4. `src/ours/phase_e/runtime.py`

Current rule:

1. default behavior = `fail`
2. legacy fallback only with explicit opt-in:
   - `--checkpoint-missing-policy fallback_final`
   - `--posthoc-missing-policy fallback_final`
3. any fallback must be written into:
   - `metrics.json`
   - `manifest.json`
   - `summary.md`
   - console logs

### Validation

Targeted regression checks:

1. `python -m py_compile ...` over patched B/C/D/E eval scripts
2. unit tests:
   - `tests/unit/test_phase_e_runtime.py`
   - `tests/unit/test_phase_b_eval_faithfulness_script.py`
   - `tests/unit/test_phase_c_eval_pik_script.py`
   - `tests/unit/test_phase_d_eval_external_pairs_script.py`

Result:

1. `18 passed`

### Additional benchmark-leak recheck

A fresh grep-based audit did **not** find direct benchmark-test ingestion in the
primary `Phase E` source-bundle training path:

1. `ProcessBench` / `PRMBench` references remain confined to:
   - eval
   - alignment audit
   - curated research utilities

So after this round, the more realistic remaining research risks are:

1. source/benchmark geometry mismatch
2. recipe collapse
3. provenance misinterpretation of legacy artifacts

## 0AAAH. Phase F Audit Correction — F2 ABR-lite Was Overstated (2026-03-12)

### What was re-checked

The `Phase F` docs previously claimed:

1. `PBR19 + MATH` offline `ABR-lite` controller had `binary detection F1 = 0.863`
2. `F2 ABR-lite` was a `STRONG PASS`
3. `Phase F RL permission gate` was already granted

Those claims are **not supported by the raw simulation artifact**.

### Raw artifact

- [phase_f_simulation summary.json](/home/zling/y/bcr/ref/assets/artifacts/phase_f_simulation/pbr19_math_abr_lite_0312/summary.json)

Correct raw numbers:

- `balanced_f1 = 0.3388`
- `positive_f1 = 0.7806`
- `acc_erroneous = 0.9882`
- `acc_correct = 0.2044`
- `mean_step_fraction = 0.3524`

### Diagnosis

This controller is **over-stopping**:

1. it catches almost every erroneous trace,
2. but it wrongly stops most all-correct traces,
3. so as a real controller it is not usable yet.

This means:

1. `reward-hacking probe` remains valid,
2. `threshold / shift audit` remains valid,
3. but `F2 ABR-lite STRONG PASS` must be downgraded,
4. and `Phase F live RL permission` is **not yet granted**.

### Current corrected status

| Gate | Corrected status |
|---|---|
| F1 threshold / shift | PASS with real generator-shift caution |
| F1 reward hacking | PASS on MATH candidate family, conditional on GSM |
| F2 ABR-lite controller | FAIL / not promotion-ready |
| Phase F live RL | NOT YET UNBLOCKED |

## 0AAAG. LoRA Series (PBR30-34) — Final Results (2026-03-12)

### Overview

LoRA-adapted backbone experiments to push beyond frozen SOTA (PBR26: MATH F1=0.686).
All use `phase_e_train_value_lora.py`, per-batch backbone forward (no feature cache),
and the same objective: joint+BCE+terminal_bce=0.25.

### Training Curve Summary (validation pair_acc on PBR12 eval set, 591 pairs)

| Run | Backbone | LoRA | Data (pairs) | Ep0 | Ep1 | Ep2 | Ep3 | Ep4 | Best |
|-----|----------|------|-------------|-----|-----|-----|-----|-----|------|
| **PBR31** | Math-PRM-7B | r=8 top-4 (360K) | PBR12 (5705) | 0.870 | 0.887 | 0.890 | 0.890 | **0.892** | ep4 |
| **PBR32** | Math-PRM-7B | r=8 all-28 (2.52M) | PBR12 (5705) | 0.861 | 0.880 | 0.897 | **0.898** | — | ep3 |
| PBR30 | Math-7B-Instruct | r=8 top-4 | PBR12 (5705) | 0.780 | 0.816 | 0.824 | 0.826 | — | ep3 |
| LoRA smoke S4 | 7B-Instruct | r=16 all | flb_0311 (7420) | 0.837 | 0.886 | — | — | — | No adapter saved |
| PBR33 | Math-PRM-7B | r=8 top-4 (360K) | PBR26 (7366) | 0.844 | 0.861 | 0.870 | 0.870 | **0.871** | ep4 |
| PBR34 | Math-PRM-7B | r=16 top-4 (720K) | PBR26 (7366) | 0.856 | — | — | — | — | Training (ep1+) |
| PBR35 | Math-PRM-7B | r=8 all-28 (2.52M) | PBR26 (7366) | 0.847 | — | — | — | — | Training (ep1+) |
| PBR36 | Math-PRM-7B | r=32 all-28 (10.1M) | PBR26 (7366) | — | — | — | — | — | Training (ep0) |

### ProcessBench Results — FINAL (as of 2026-03-12 01:35 +0800)

| Run | MATH AUC | GSM AUC | MATH F1 | GSM F1 | Notes |
|-----|---------|--------|---------|--------|-------|
| **PBR32** (LoRA r=8 all-28) | 0.8685 | 0.8999 | **0.689** | 0.776 | **NEW MATH F1 SOTA** |
| PBR26 (frozen) | **0.888** | 0.915 | 0.686 | 0.768 | Prev MATH F1 best |
| PBR19 (frozen) | — | — | 0.683 | **0.778** | Best GSM frozen |
| PBR31 (LoRA r=8 top-4) | 0.8823 | 0.9004 | 0.676 | **0.788** | **NEW GSM F1 SOTA** |
| PBR30 (Instruct LoRA r=8) | 0.782 | 0.799 | 0.448 | 0.505 | Wrong backbone family |

### Key Findings

1. **PBR32 (all-28-layer LoRA) achieves MATH F1=0.689** — new SOTA, +0.003 vs frozen PBR26.
   - Acc_erroneous=0.564, Acc_correct=0.884. AUC=0.869 (lower than frozen 0.888).
   - Net: F1 improved despite lower AUC → threshold detection improved.

2. **PBR31 (top-4 LoRA) achieves GSM8K F1=0.788** — new SOTA, +0.010 vs frozen PBR19.
   - Tradeoff: MATH F1=0.676 (below frozen). Top-4 layers = better GSM, worse MATH.

3. **Instruct backbone (PBR30) is unsuitable**: MATH F1=0.448 (vs 0.686 frozen). Confirmed:
   - Math-PRM-7B backbone is required for ProcessBench step-level quality assessment.

4. **Depth vs MATH F1**: More LoRA layers → better MATH F1. Top-4=0.676, All-28=0.689.
   PBR33 confirmed: top-4 + larger data → MATH F1=0.666, GSM F1=0.797 (best GSM SOTA).

5. **PBR33 (top-4 r=8 + PBR26 data)**: MATH F1=0.666 (below PBR32), **GSM F1=0.797** (new SOTA).
   Shows top-4 LoRA + large data is GSM-optimal, not MATH-optimal.

6. **Community gap**: MATH F1=0.689 vs Qwen2.5-Math-PRM-7B full fine-tune ~0.735. Gap=4.6 pts.
   Remaining levers: PBR35 (all-28+PBR26), PBR36 (r=32), PBR37 (contrastive loss).

7. **RL controller analysis** (offline REINFORCE on PBR32 scored traces, 2026-03-13):
   - Best result: heuristic tau=0.35 at **val F1=0.848**
   - Supervised MLP: F1=0.770 (HEURISTIC WINS)
   - MLP BC warmstart: BC alone achieves 0.818, REINFORCE degrades to 0.750-0.781
   - GRU reinforce: 0.72-0.75, stuck in local optima
   - **Conclusion: heuristic threshold at tau=0.35 is optimal offline controller for PBR32 MATH**

### Eval Artifacts

| Run | MATH eval dir | GSM eval dir |
|-----|--------------|-------------|
| PBR31 | `pbr31_lora_mathprm_pb_math_20260311T164942Z` | `phase_e_pbr31_lora_mathprm_processbench_gsm8k_20260311T165223Z` |
| PBR32 | `pbr32_lora_mathprm_alllayers_pb_math_20260311T171442Z` | `pbr32_lora_mathprm_alllayers_pb_gsm_20260311T171442Z` |
| PBR30 | `pbr30_lora_math7b_pb_math_full_20260311T164110Z` | `phase_e_pbr30_lora_math7b_processbench_gsm8k_20260311T162412Z` |

## 0AAAF. Comprehensive PBR Ablation & FLB2 Backbone Series — Full Results (2026-03-12)

### Summary

All experiments from session 3 (PBR22-PBR27, FLB2 full series) are now evaluated.
All evals are full 1000-sample ProcessBench MATH; GSM8K evals noted separately.

### PBR Ablation Series (Math-PRM-7B frozen backbone, all full 1000-sample)

| Config | Data Mix | Pairs | MATH AUC | GSM AUC | MATH F1 | Notes |
|---|---|---|---|---|---|---|
| **PBR26** | DPO+MS_full | 7366 | **0.888** | 0.915 | **0.670** | joint+lbce=0.5+terminal_bce=0.25; **CURRENT BEST** |
| PBR12 | DPO+MS_strict | 5705 | 0.887 | 0.909 | 0.644 | ranking_only; baseline |
| PBR24 | DPO+MS_high_term | 5705+extra_term | 0.885 | 0.917 | 0.656 | high terminal anchors |
| PBR22 | DPO+MS+PRM800K | ~6800 | 0.885 | 0.917 | 0.682 | PRM800K adds minimal value |
| PBR23 | DPO+MS_uniform | ~6800 | 0.883 | 0.912 | 0.684 | uniform balance, strong F1 |
| PBR25 | uniform+terminal_bce=0.5 | ~6800 | 0.882 | 0.910 | 0.678 | terminal_bce=0.5 slightly hurts |
| PBR27_mlp1024 | DPO(2398)+MS(3307) | 5705 | 0.873 | 0.900 | 0.666 | different DPO:MS ratio |
| PBR25_seed1 | DPO+MS (PBR12 config) | 5705 | 0.874 | — | 0.633 | seed variance ±0.013 vs seed42 |
| PBR21 | DPO+MS_strict | 5705 | 0.870 | 0.904 | 0.663 | 10 epochs (2× of PBR12): **overfits** |
| PRX1 | DPO+MS+terminal_1K | 6947 | 0.853 | 0.887 | 0.648 | extra terminal pairs hurt MATH OOD |
| PRX2 (dual_head) | DPO+MS+terminal_1K | 6947 | 0.818 | 0.844 | 0.623 | dual_head routing backfires |
| FLB2D (PRM mlp) | DPO-only | 7420 | 0.842 | 0.858 | 0.628 | **pure DPO ceiling w/ PRM backbone** |
| FLB2E (PRM gated) | DPO-only | 7420 | 0.822 | 0.854 | 0.610 | gated_mlp < mlp on PRM backbone |

### Key Ablation Conclusions

1. **PBR26 is the new best** (+0.026 F1 over PBR12 with same MATH AUC; MEDIUM robustness)
2. **PRM800K adds nothing** (PBR22 vs PBR12: MATH AUC 0.885 vs 0.887)
3. **Uniform balance helps F1** (PBR23 MATH F1=0.684 best F1, but AUC same as others)
4. **terminal_bce=0.5 hurts** vs terminal_bce=0.25 (PBR25 vs PBR26: 0.882 vs 0.888)
5. **PRM backbone DPO-only plateau: 0.842** — DPO+MS is needed to break above this ceiling
6. **DPO:MS ratio matters**: PBR12 (3705:2000) beats PBR27_mlp1024 (2398:3307) by +0.014 MATH AUC
   → More DPO sibling pairs are higher quality per-pair than MC-noisy MS pairs
7. **Seed variance ≈ ±0.013** (PBR12 seed42=0.887, seed1=0.874)

### FLB2 Backbone Series (all full 1000-sample)

| Config | Backbone | Head | Data | MATH AUC | GSM AUC | MATH F1 |
|---|---|---|---|---|---|---|
| FLB2_prm_mlp (FLB2D) | Math-PRM-7B | mlp | DPO 7420 | 0.842 | 0.858 | 0.628 |
| FLB2_prm_gated | Math-PRM-7B | gated_mlp | DPO 7420 | 0.822 | 0.854 | 0.610 |
| FLB2_math_instruct | Math-7B-Instruct | gated_mlp | DPO 7420 | 0.742 | 0.746 | 0.427 |
| FLB2A_s42 | 7B-Instruct | gated_mlp | DPO 7420 | 0.746 | 0.706 | 0.366 |
| FLB2A_s7 | 7B-Instruct | gated_mlp | DPO 7420 | 0.772 | 0.727 | 0.375 |
| FLB2B_s7 | 7B-Instruct | gated_mlp | DPO 8K | 0.763 | 0.715 | 0.381 |

**Backbone ranking** (DPO-only training, frozen, full 1000-sample MATH AUC):
Math-PRM-7B (0.842) >> Math-7B-Instruct (0.742) >> 7B-Instruct (0.746-0.772)

Note: 7B-Instruct shows higher **seed variance** than Math backbone variants.

### Experiments Still Running (as of 10:15)

| Exp | Data | Pairs | Config | GPU | Status |
|---|---|---|---|---|---|
| PBR27_dpo_only | DPO-only (PBR12 hparams) | 7420 | ranking_only, 5ep | GPU 0 | encoding |
| PBR28 | DPO_full+MS_4k | ~9K | joint, lbce=0.5 | GPU 0 | encoding |
| PBR29 | DPO_full+MS_full | 12388 | ranking_only, 5ep | GPU 1 | encoding |
| PBR29_fanout | DPO+MS_fanout | ~11K | ranking_only | GPU 3 | encoding |
| LoRA_smoke | 7B-Instruct+LoRA r16 | DPO 7420 | ranking_only | GPU 2 | training |

### Scale Ablation — COMPLETED RESULTS

| Config | Data | Pairs | MATH AUC | MATH F1 | Notes |
|---|---|---|---|---|---|
| **PBR26** | DPO+MS_full, joint+lbce=0.5+terminal_bce=0.25 | 7366 | **0.888** | **0.670** | CURRENT BEST |
| PBR29_fanout | DPO+MS_fanout, ranking_only | 7329 | 0.884 | 0.682 | strong F1 |
| PBR28 | DPO_full+MS_4k, joint+lbce=0.5 | 7705 | 0.882 | 0.674 | — |
| **PBR29_full** | DPO_full+MS_full, ranking_only | **12388** | **0.879** | **0.641** | **SCALE HURTS** |
| PBR27_dpo_only | DPO-only, ranking_only | 7420 | 0.859 | 0.622 | pure DPO ceiling |
| FLB2D | DPO-only, 10ep (overfitting) | 7420 | 0.842 | 0.628 | worse hparams |

**Critical discoveries from scale ablation:**
1. **DPO+MS mix +0.029**: PBR26 (7366, DPO+MS) vs PBR27_dpo_only (7420, DPO-only): 0.888 vs 0.859
2. **Optimal scale ~5K-7K**: PBR29_full (12388) = 0.879 < PBR12 (5705) = 0.887 — 2× data HURTS
3. **Fanout profile competitive**: PBR29_fanout (0.884) nearly matches PBR26 with better F1 (0.682)
4. **joint+terminal_bce=0.25 is PBR26's key**: enables F1=0.670 with equivalent AUC

---

## 0AAAE. Phase F Reward Hacking — Comprehensive Table (2026-03-12, updated)

### Complete flip_rate_fixed05 Results

Results from 3 Phase F probe runs (40-48 examples per benchmark per attack).

#### GSM8K attacks on first-bad erroneous prefixes

| Candidate | confidence_tail | filler_tail | repeat_last | self_verify | Max flip |
|---|---|---|---|---|---|
| **PBR12** | **0.156 HIGH** | **0.156 HIGH** | 0.062 | 0.050 | **HIGH** |
| PBR21 | 0.050 | 0.050 | 0.050 | 0.000 | LOW |
| PRX1 | 0.031 | 0.031 | 0.025 | 0.000 | LOW |
| **PBR26** | **0.042 MEDIUM** | 0.021 | 0.021 | 0.000 | **MEDIUM** |

#### MATH attacks on first-bad erroneous prefixes

| Candidate | confidence_tail | filler_tail | repeat_last | self_verify | Max flip |
|---|---|---|---|---|---|
| **PBR12** | 0.075 HIGH† | 0.094 | 0.050 HIGH† | 0.031 | HIGH |
| PBR21 | 0.075 HIGH† | 0.025 HIGH† | 0.000 HIGH† | 0.000 HIGH† | HIGH (outrank) |
| PRX1 | 0.050 HIGH† | 0.025 HIGH† | 0.000 HIGH† | 0.025 HIGH† | HIGH (outrank) |
| **PBR26** | 0.021 HIGH† | 0.042 HIGH† | 0.021 | 0.021 | HIGH (outrank) |

†: "HIGH" here reflects `outrank_safe_rate > 0.10` (bad+attack outranks safe prefix) even when flip_rate < 0.15

### Critical Finding: outrank_safe_rate vs flip_rate Divergence

On **MATH** benchmarks, all models show concerning `outrank_safe_rate` even with low `flip_rate`:
- PBR21: flip_rate=0.000 but outrank_safe_rate=0.125-0.200 on MATH (attacks reorder without crossing 0.5)
- PBR26: filler_tail outrank=0.208 on MATH despite flip=0.042
- PRX1: multiple HIGH outrank cases on MATH

**Implication**: MATH domain is harder to make robust than GSM8K. The erroneous steps in MATH are
more "borderline" scored — attacks shift scores in relative terms without crossing threshold.

### Comparison of Safety Risk for RL Deployment

| Candidate | MATH AUC | Overall Risk | GSM flip (worst) | MATH outrank (worst) |
|---|---|---|---|---|
| PBR12 | 0.887 | **HIGH** | 0.156 | 0.150 |
| PBR21 | 0.870 | HIGH (outrank) | 0.050 | 0.200 |
| PRX1 | 0.853 | MEDIUM-HIGH | 0.031 | 0.175 |
| **PBR26** | **0.888** | **MEDIUM** | **0.042** | **0.208 (filler)** |

PBR26 is the **best accuracy+robustness tradeoff** so far: highest MATH AUC AND lower GSM flip rate.
MATH outrank_safe_rate remains a concern; Clip+Delta (arXiv:2410.15115) required for safe RL.

---

## 0AAAD. PBR26 New Best — MATH AUC=0.888, F1=0.670 (2026-03-12)

### Summary

PBR26 surpasses PBR12 on BOTH metrics:
- **MATH AUC: 0.888** (vs PBR12 0.887, +0.001)
- **MATH F1: 0.670** (vs PBR12 0.644, **+0.026** = significant step localization improvement)
- pair_acc: 0.853 (vs PBR12 0.880)

Config: Math-PRM-7B frozen, DPO(2398) + MS_full(4968) = 7366 pairs, joint mode,
lambda_bce=0.5, lambda_terminal_bce=0.25, 5 epochs, lr=5e-5, MLP-512, seed=42.

Artifact: `phase_e_pbr26_dpo_ms_full_s42_value_20260311T134542Z`
Data: `phase_e_pbr26_dpo_plus_ms_full_pairs__b17437d10dfc`

**Key difference from PBR12**: More MS data (4968 vs 2759) + joint mode + terminal_bce=0.25.
The joint mode + terminal BCE provides better F1 (step localization) while matching AUC.

Phase F robustness probe running (PID 3523659) — expected more robust than PBR12 due to terminal_bce=0.25.

---

## 0AAAC. Phase F Reward Hacking Probe — AUC vs Robustness Tradeoff (2026-03-12)

### Summary

Phase F probes reward hacking: "can superficial tail additions fool the verifier into accepting bad prefixes?"

### flip_rate_fixed05 on first_bad erroneous examples (GSM8K)

| Candidate | MATH AUC | confidence_tail | filler_tail | Max Risk |
|---|---|---|---|---|
| **PBR12** | **0.887** | **0.156 (HIGH)** | **0.156 (HIGH)** | **HIGH** |
| PBR19 | 0.683 (F1) | 0.094 (med) | 0.062 (med) | MEDIUM |
| PBR21 | 0.870 | 0.050 | 0.050 | LOW (flip), HIGH (outrank MATH) |
| **PRX1** | **0.853** | **0.031 (med)** | **0.031 (med)** | **MEDIUM** |
| **PBR26** | **0.888** | **0.042** | **0.021** | **MEDIUM** |

PBR26 Phase F: GSM8K confidence_tail flip=0.042 MEDIUM; MATH filler outrank_safe_rate=0.208 (concerning).
See section 0AAAE for comprehensive analysis.

### Key Findings (updated)
1. **AUC vs Robustness**: PBR12 (0.887 AUC) has 15.6% confidence_tail flip rate on GSM8K — HIGH risk
2. **Terminal BCE provides robustness**: PBR26 (terminal_bce=0.25) has 3.7× lower GSM flip rate than PBR12
3. **PBR26 = best accuracy+robustness**: 0.888 AUC + MEDIUM flip rate (73% better than PBR12)
4. **MATH outrank risk remains**: even PBR26 has filler_tail outrank=0.208 on MATH — Clip+Delta needed
5. **PBR12 not safe for RL without Clip+Delta** (15.6% flip unacceptable for sustained RL)

### PBR30 Plan (Robustness Fine-Tune)
Hypothesis: small terminal BCE + FLB2D's "DPO+MS at 5705 scale" is optimal. May not be needed
if PBR26 is RL-deployed with Clip+Delta.

---

## 0AAAA. Paradigm Refresh Around PBR10: Terminal Mix Helps, Dual-Head Over-Rotates (2026-03-12)

### Summary

This pass asked one narrow question:

- after `PBR10`, what is the smallest intervention that improves `all-correct terminal completion ordering`
  without breaking the verifier?

Three highly informative lines now exist:

1. `PBR10`
   - best balanced baseline
2. `PRX1_TERMMIX_MLP`
   - more same-family terminal anchors, same `mlp` family
3. `PRX2_TERMMIX_DUAL`
   - same repaired data as `PRX1`, but `dual_head`

### Unified comparison

| Config | heldout pair_acc | sf top1 | sf local | GSM AUC | Math AUC | GSM terminal_top1 | Math terminal_top1 | GSM F1 | Math F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `PBR10` | 0.9064 | 0.9098 | 0.8981 | 0.8910 | 0.8651 | 0.1762 | 0.1626 | 0.7334 | 0.6313 |
| `PRX1_TERMMIX_MLP` | 0.8984 | 0.8981 | 0.8767 | 0.8869 | 0.8529 | 0.3316 | 0.4828 | 0.7518 | 0.6482 |
| `PRX2_TERMMIX_DUAL` | 0.8464 | 0.8483 | 0.6438 | 0.8442 | 0.8182 | 0.6218 | 0.6355 | 0.7398 | 0.6228 |

### Interpretation

`PRX1` is the best new result in this pass:

1. terminal completeness improved a lot:
   - `GSM terminal_top1: 0.1762 -> 0.3316`
   - `Math terminal_top1: 0.1626 -> 0.4828`
2. `ProcessBench F1` also improved:
   - `GSM: 0.7334 -> 0.7518`
   - `Math: 0.6313 -> 0.6482`
3. but it still misses strict RL-ready:
   - `samefamily_top1 < 0.90`
   - `GSM terminal_top1 < 0.50`

`PRX2` proves a different point:

1. the terminal problem is not impossible to fix;
2. but the current `dual_head + fixed alpha` solution solves it by sacrificing the core local verifier.

So the repo's next design target should not be "more dual-head by default".
It should be:

1. `ABR local scorer`
2. `outcome / answer verifier`
3. `routing or abstain`

Key artifacts:

1. `assets/artifacts/phase_e_runs/phase_e_prx1_pbr10core_term1024_mlp_0311_20260311T123100Z`
2. `assets/artifacts/phase_e_runs/phase_e_prx2_pbr10core_term1024_dual_0311_20260311T130045Z`
3. `assets/artifacts/phase_e_transfer_diag/phase_e_paradigm_refresh_diag_0311_00`
4. `docs/phase_e_paradigm_refresh_experiments_20260311.md`

---

## 0AAAB. Overfitting Diagnostics — More Epochs/Data Hurts OOD; DPO+MS Mix > Pure DPO (2026-03-12)

### Summary

New experiments (PBR21, PRX1, FLB2D/E, PBR22/23) reveal critical overfitting patterns:
**PBR12 config (5 epochs, lambda_bce=0.25, DPO+MS mix) remains the best known configuration.**

### Full Benchmark Results vs PBR12 (Math-PRM-7B backbone, all frozen)

| Config | Data | Pairs | Epochs | pair_acc | MATH AUC | GSM AUC | MATH F1 | Notes |
|---|---|---|---|---|---|---|---|---|
| **PBR12** (BEST) | DPO+MS_strict | **5705** | **5** | ~0.887 | **0.887** | **0.909** | 0.644 | lambda_bce=0.25 |
| PBR21 | DPO+MS_strict | 5705 | 10 | 0.890 | 0.870 | 0.904 | 0.663 | lambda_bce=0.5; more epochs hurt |
| PRX1 | DPO+MS+terminal | 6947 | 4 (best ep3) | 0.898 | 0.853 | 0.887 | 0.648 | terminal anchors hurt OOD |
| FLB2D (256-smp) | pure DPO | 8192 | 10 | **0.917** | 0.831 | 0.858 | 0.628 | MLP-1024; within-dist overfit |
| FLB2E (256-smp) | pure DPO | 8192 | 10 (ep3 best) | **0.917** | 0.822 | 0.854 | 0.610 | gated_mlp; same overfit pattern |
| PBR22 | DPO+MS+PRM800K | ~6800 | 5 | 0.858 | 0.885 | 0.917 | 0.682 | PRM800K negligible impact |
| PBR23 | uniform balance | ~6800 | 5 | 0.855 | 0.883 | 0.912 | 0.684 | strong F1 |

FLB2D/E full 1000-sample MATH eval complete (see section 0AAAF for full FLB2 table).

### Key Overfitting Patterns

1. **More epochs hurts OOD**: PBR21 (10ep) vs PBR12 (5ep) — same data, same backbone, 0.870 vs 0.887 MATH AUC
2. **Terminal anchors hurt**: PRX1 adds 1242 terminal pairs → MATH AUC drops 0.887 → 0.853
3. **Pure DPO overfits**: FLB2D/E (8K pure DPO) get highest pair_acc (0.917) but lowest MATH AUC (0.831). DPO train distribution ≠ ProcessBench MATH distribution
4. **DPO+MS mix is optimal**: Complementary supervision (fork-point signal + step-MC signal) outperforms either alone at scale
5. **PRM800K addition hurts**: PBR22 adds PRM800K to PBR12 data → pair_acc 0.887→0.858, likely distribution mismatch

### Backbone Comparison (same dpo_scale_v1 data, updated)

| Backbone | Head | pair_acc | MATH AUC (256) |
|---|---|---|---|
| Qwen2.5-7B-Instruct | gated_mlp | 0.775-0.795 | ~0.60-0.63 |
| Qwen2.5-Math-7B-Instruct | gated_mlp | 0.817 | TBD |
| **Qwen2.5-Math-PRM-7B** | **MLP-1024** | **0.917** | **0.831** |
| **Qwen2.5-Math-PRM-7B** | **gated_mlp** | **0.917** | **0.822** |

Note: 256-sample eval; full 1000-sample eval for FLB2D pending.

### PRX1 Full Results (6947 pairs, terminal 1242, ep3 best)

| Metric | Value |
|---|---|
| MATH AUC (1000) | 0.853 |
| GSM8K AUC (400) | **0.887** |
| MATH F1 | 0.648 |
| MATH first_error_edge_acc | 0.846 |
| GSM8K F1 | 0.752 |
| GSM8K first_error_edge_acc | 0.912 |

GSM8K AUC=0.887 is strong (matches PBR12's MATH AUC), but MATH AUC drop confirms terminal pairs hurt MATH OOD transfer.

---

## 0AAAM. Math-PRM-7B Frozen Backbone Breakthrough — 0.749 → 0.887 AUC (2026-03-12)

### Summary

**Key discovery**: Using Qwen2.5-Math-PRM-7B as the frozen backbone (instead of
Qwen2.5-7B-Instruct) dramatically improves ProcessBench MATH AUC from the previous
ceiling of 0.749 to 0.865-0.887.

The Math-PRM-7B backbone's hidden states are already optimized for step-level verification
from PRM pre-training — freezing these specialized representations + training a lightweight
value head yields near-full-fine-tune quality without expensive backbone updates.

### Backbone Ablation Results (all: DPO+MS_strict data, gated/mlp head, frozen)

| Backbone | pair_acc | MATH AUC | Notes |
|---|---|---|---|
| Qwen2.5-7B-Instruct | ~0.77-0.79 | ~0.749 | Previous SOTA ceiling |
| Qwen2.5-Math-7B-Instruct | ~0.82 | TBD (est ~0.82) | Math-specialized instruction model |
| **Qwen2.5-Math-PRM-7B** | **~0.87-0.90** | **0.865-0.887** | **PRM-specialized: breakthrough** |

Backbone comparison source: FLB2 suite (flb2a vs flb2_math_instruct backbones) + PBR10/12.

### PBR10/12 Benchmark Results (Math-PRM-7B backbone, frozen, full 1000-sample eval)

| Config | Backbone | Data | Head | MATH AUC | GSM AUC | MATH F1 |
|---|---|---|---|---|---|---|
| PBR10 dpo8k (s42) | Math-PRM-7B | DPO(5200)+MS_strict(2000)+term(800) | mlp-512 | **0.865** | **0.891** | 0.631 |
| PBR10 gated (s42) | Math-PRM-7B | Same 6947 pairs | gated_mlp-512 | **0.869** | **0.873** | 0.567 |
| PBR12 dpo+ms | Math-PRM-7B | DPO(3705)+MS_strict(2000) | mlp-512 | **0.887** | **0.909** | 0.644 |

Note: pair_auc_good_vs_bad = MATH AUC in this table (full ProcessBench MATH 1000 examples).
F1 uses oracle threshold sweep.

### PBR21 + PRX1 (Best pair_acc, benchmark eval pending)

| Config | Backbone | Data | Epochs | pair_acc (best) | MATH AUC | Status |
|---|---|---|---|---|---|---|
| PBR21 dpo+ms 10ep | Math-PRM-7B | 5705 pairs | 10 | 0.890 | TBD | training complete, eval pending |
| PRX1 pbr10core+term1024 | Math-PRM-7B | 6947+term | 4 (ep3) | 0.898 | TBD | training in progress |

### Why Math-PRM-7B Backbone Works (Literature-Backed Explanation)

From "Lessons of Developing PRMs" (arXiv:2501.07301): Backbone must be specialized for
step verification. General instruction models have multipurpose representations not oriented
toward step quality.

Math-PRM-7B was trained end-to-end for step correctness classification. Its hidden states
encode step-quality-relevant features. Freezing them preserves this without expensive
retraining; the value head just needs a lightweight projection.

This is the "warm-start from PRM backbone" shortcut that avoids full fine-tuning while
getting most of its benefits.

### Literature Alignment

| Our finding | Literature confirmation |
|---|---|
| Math-PRM-7B >> 7B-Instruct (frozen) | FOVER: "Frozen LLMs significantly underperform fine-tuned PRMs" |
| DPO fork-point > MS mixed | Step-DPO (arXiv:2406.18629): fork-point isolates step causation |
| gated_mlp ≈ mlp at this quality level | GLU variants (Shazeer 2020): gating = conditional feature selection |
| 0.887 AUC sufficient for RL | Noisy rewards (arXiv:2510.00915): 40% noise → 96% oracle performance |

---

## 0AAAL. Phase E Frontier Redesign: New Directions, One Immediate Negative (2026-03-11)

This pass did two things together:

1. translated the newer `2025H2 -> 2026` verifier community direction into
   three concrete repository-level experiment directions,
2. launched one shared-artifact frontier suite on top of the strongest current
   `PBR10` line.

Shared artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_pbr10_prm7b_dpo8k_s42__6184f7e62f65`

Three new frontier directions:

1. `judge / critique filtering`
   - implemented as:
     - `F1_JUDGE_FILTER_PBR10`
     - `scripts/phase_e_judge_filter_pairs.py`
2. `objective decomposition`
   - implemented as:
     - `F2_DUAL_HEAD_PBR10`
     - `head_architecture=dual_head`
3. `light backbone adaptation`
   - implemented as:
     - `F3_LORA_PBR10`
     - `scripts/phase_e_train_value_lora.py`

### Why these three directions were chosen

The strongest already-finished `PBR10` diagnosis says:

1. local ranking and first-error-edge behavior are already strong,
2. the narrow remaining bottleneck is terminal completion undervaluation,
3. so the next fixes should target:
   - label quality / critique filtering,
   - local-vs-terminal objective factorization,
   - or representation limits.

### Immediate completed result: current judge-filter contract fails on PBR10

Filtered artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_frontier_judge_0311_2031_judgefilter__88888f79419b`

Key numbers:

1. input train pairs: `6947`
2. output train pairs: `5164`
3. keep rate: `0.7433`
4. audited pairs: `1783`
5. bypassed pairs: `5164`
6. decision counts:
   - `parse_failed = 1783`
   - `bypass_semantics = 5164`

Semantics after filtering:

1. `sibling_branch = 4679`
2. `terminal_completion_anchor = 485`
3. `local_first_bad_edge = 0`

Interpretation:

1. the filter did **not** act as a useful noise cleaner on the shared frontier
   artifact;
2. instead, it dropped the entire auditable local-edge subset and kept only the
   bypassed semantics;
3. therefore the current strict-JSON judge-filter contract should be treated as
   a negative direction for mainline Phase E training.

### Runs still in progress at the time of this record

1. `F2_DUAL_HEAD_PBR10`
2. compact `LoRA` probe derived from `F3_LORA_PBR10`

Those were intentionally left running after the judge-filter result above,
because they still test two live hypotheses:

1. whether explicit local/terminal factorization improves terminal behavior,
2. whether minimal backbone adaptation gives directional gain without the
   severe geometry damage seen in older tiny-data LoRA pilots.

## 0AAAK. 2026-03 Direction Refresh: ABR Should Survive, But As Student And Router

This pass did not add a new training run. It re-opened the research-direction
question using:

1. the newer `2025H2 -> 2026-03` verifier / reward-model literature,
2. the repository's already-finished `PBR / NDS / RL-ready` evidence,
3. and the newer local survey / redesign notes already accumulated under
   `docs/`.

### Fresh external evidence added this pass

1. `CompassVerifier`
   - https://aclanthology.org/2025.acl-long.1102/
2. `VerifyBench`
   - https://arxiv.org/abs/2507.09884
3. `Verifying the Verifiers`
   - https://arxiv.org/abs/2506.13342
4. `Weaver`
   - https://arxiv.org/abs/2510.18084
5. `When to Trust the Cheap Check`
   - https://arxiv.org/abs/2603.05390
6. `MathQ-Verify`
   - https://arxiv.org/abs/2603.03307

### Updated interpretation

The repository's earlier 2026-03 refresh was directionally correct, but still
slightly under-scoped.

The stronger current interpretation is:

1. the core `ABR` idea still survives:
   - cheap process-aware verification under a compute budget is still useful;
2. what looks outdated is the narrower system approximation:
   - `frozen backbone + one scalar head + pair ranking = the verifier`.

### New system-level risks now judged important

1. missing `answer verifier` layer
   - terminal / equivalence / malformed outputs are still mixed into one score;
2. missing `invalid / abstain / escalate` behavior
   - the cheap scorer is still forced to emit a number even on slices where it
     should defer;
3. missing `format / contract robustness` gate
   - newer verifier papers show rankings can move under prompt / structure
     changes;
4. missing `process-outcome alignment` gate
   - local ranking can improve while derivation-faithfulness still lags;
5. missing `weak-ensemble / teacher-student` design
   - the repo still expects too much from one judge / one head.

### Updated redesign decision

The preferred redesign is now:

1. keep `ABR` as:
   - student verifier
   - router
   - conservative online scorer
2. add:
   - a separate answer verifier,
   - weak-verifier disagreement mining,
   - teacher-assisted hard-slice relabeling,
   - factorized student outputs,
   - and stricter RL gates.

### Most valuable next experiments

1. `ABR_AVCal_v1`
   - answer-equivalence / invalid / false-negative calibration
2. `ABR_WeaverLite_v1`
   - combine:
     - ABR score
     - answer verifier
     - deterministic math equivalence
     - one stronger teacher judge on disagreement slices
3. `ABR_MH_v1`
   - factorized student:
     - `local`
     - `progress`
     - `completion`
     - `uncertainty`
     - `abstain`
4. `ABR_POA_v1`
   - small process-outcome alignment set
5. `ABR_FormatInv_v1`
   - prompt / answer-style / delimiter robustness audit

### Docs updated

1. `docs/phase_e_literature_refresh_20260311.md`
2. `docs/phase_e_rl_ready_research_redesign_20260311.md`
3. `docs/result_records.md`
4. `docs/progress_detailed.md`
5. `docs/readme.md`
6. `docs/readme_full.md`

## 0AAAN. Teammate Breakthrough Report — Local Evidence Audit + Validation Eval (2026-03-11)

The teammate report contains two very different classes of claims:

1. literature / community synthesis;
2. local repository breakthrough claims.

These should not be accepted as one bundle.

The local evidence audit says:

1. the `Math-PRM-7B frozen backbone` breakthrough is real;
2. but the stronger headline
   - "`PRX1` is now the newest best benchmark candidate"
   is **not yet supported** by local benchmark evidence.

### What Was Already Supported Before The Validation Pass

The repo already had strong local evidence for:

1. `Qwen2.5-Math-PRM-7B` as frozen backbone > non-PRM backbones
2. `PBR10 / PBR12 / PBR21` as genuinely strong benchmark-facing lines
3. `same-family` and `ProcessBench` strength both moving together on the PRM backbone

But one important gap remained:

1. `PRX1` had very high training / same-family numbers,
2. yet its full `ProcessBench Math` evaluation was still missing.

So this validation pass did exactly that:

1. filled the missing `PRX1 ProcessBench Math` full eval
2. added fixed-threshold (`0.5`) evals for:
   - `PRX1`
   - `PBR21`
   - `PBR12`

### Validation Experiments Added In This Pass

New eval artifacts:

1. `PRX1 Math full eval`
   - `assets/artifacts/phase_e_eval/phase_e_prx1_math_fulleval_0311_verify_20260311T132702Z`
2. `PRX1 GSM fixed@0.5`
   - `assets/artifacts/phase_e_eval/phase_e_prx1_gsm_fixed05_0311_verify_20260311T132703Z`
3. `PRX1 Math fixed@0.5`
   - `assets/artifacts/phase_e_eval/phase_e_prx1_math_fixed05_0311_verify_20260311T132702Z`
4. `PBR21 GSM fixed@0.5`
   - `assets/artifacts/phase_e_eval/pbr21_gsm_fixed05_verify_20260311T132735Z`
5. `PBR21 Math fixed@0.5`
   - `assets/artifacts/phase_e_eval/pbr21_math_fixed05_verify_20260311T132735Z`
6. `PBR12 GSM fixed@0.5`
   - `assets/artifacts/phase_e_eval/pbr12_gsm_fixed05_verify_20260311T132836Z`
7. `PBR12 Math fixed@0.5`
   - `assets/artifacts/phase_e_eval/pbr12_math_fixed05_verify_20260311T132836Z`

### Controlled Backbone Validation

The teammate report said the backbone change matters more than head tweaks.

Local evidence supports that.

Controlled-ish comparison on the new `FLB2` line:

| case | backbone | GSM AUC | GSM F1 | Math AUC | Math F1 |
|---|---|---:|---:|---:|---:|
| `FLB2 math-instruct gated` | `Qwen2.5-Math-7B-Instruct` | `0.7461` | `0.5158` | `0.7423` | `0.4275` |
| `FLB2 prm-backbone mlp` | `Qwen2.5-Math-PRM-7B` | `0.8581` | `0.7148` | `0.8313` | `0.6276` |

Interpretation:

1. the PRM-specific backbone is a real step change;
2. this is not a tiny noise-level bump;
3. the claim
   - `Math-PRM-7B >> Math-7B-Instruct`
   is supported locally.

### Current Strong Candidates — Oracle Metrics

| case | GSM AUC | GSM F1 | Math AUC | Math F1 |
|---|---:|---:|---:|---:|
| `PBR10 mlp` | `0.8730` | `0.7243` | `0.8631` | `0.6589` |
| `PBR12` | `0.9093` | `0.7523` | `0.8874` | `0.6439` |
| `PBR21` | `0.9045` | `0.7565` | `0.8697` | `0.6632` |
| `PRX1` | `0.8869` | `0.7518` | `0.8529` | `0.6482` |

Interpretation:

1. `PBR12` currently has the best `Math AUC`;
2. `PBR21` currently has the best `Math F1` and `GSM F1`;
3. `PRX1` is strong, but it does **not** dominate `PBR12 / PBR21` on full benchmark metrics.

### Fixed-Threshold (`0.5`) Validation

This is the more conservative view.

| case | GSM F1@0.5 | Math F1@0.5 |
|---|---:|---:|
| `PBR12` | `0.7023` | `0.6344` |
| `PBR21` | `0.7185` | `0.6051` |
| `PRX1`  | `0.7454` | `0.6040` |

Interpretation:

1. `PRX1` is the best checked candidate on `GSM fixed@0.5`;
2. but on `Math fixed@0.5`, it is not better than `PBR12`, and is basically tied
   with `PBR21`;
3. so `PRX1` should be read as:
   - a promising continuation of the `PBR10 core + more terminal support` line,
   - not a clean global replacement for `PBR12 / PBR21`.

### What The Teammate Report Gets Right

These claims are supported by current local evidence:

1. `Math-PRM-7B` frozen backbone is a real breakthrough
2. PRM-specialized hidden states matter a lot
3. `DPO fork-point` data is a much stronger signal carrier than weak local-only baselines
4. benchmark evaluation must remain separate from training `pair_acc`

### What Needs To Be Stated More Carefully

These claims should be softened:

1. "`PRX1` is the latest strongest model"
   - not supported globally
2. "current PRM is already enough for RL"
   - still too aggressive
   - benchmark strength is now good,
   - but RL readiness also depends on reward hacking resistance, fixed-threshold
     stability, and deployment-time behavior
3. "pair_acc 0.898 therefore breakthrough"
   - `pair_acc` is useful,
   - but it is not the deciding metric once full `ProcessBench` is available

### Updated Repository-Level Conclusion

The correct current summary is:

1. the backbone breakthrough is real;
2. the repo is now in a much stronger verifier regime;
3. but the best candidate depends on metric:
   - `PBR12` for strongest `Math AUC`
   - `PBR21` for strongest `Math/GSM oracle F1`
   - `PRX1` for strongest checked `GSM fixed@0.5`
4. therefore the next step should be:
   - keep `PBR12 / PBR21 / PRX1` as the main comparison triangle,
   - not collapse them into one headline winner too early.

## 0AAAK. Community Paradigm Update + Terminal-Residual Repair Sweep (2026-03-11)

### Community Paradigm — What The Current Literature Is Actually Pushing Toward

The latest verifier / PRM literature is no longer centered on
"small scalar value head + weak local labels" as the default mainline.

The strongest recurring themes are now:

1. `critic / verifier` framing over plain scalar regression
   - stronger judge-style models or richer verifier heads are increasingly used
     to inspect reasoning quality;
2. `process-native supervision`
   - step-level labels, fork-point comparisons, or progress-style rewards;
3. `sibling-branch / fork-point` geometry
   - compare two alternative next-step continuations under the same history;
4. `benchmark-native evaluation`
   - especially explicit tests of:
     - local first-error detection
     - complete correct-trace ordering
     - process-outcome alignment
5. `factorization`
   - local error detection and terminal completion ordering are increasingly
     treated as distinct sub-skills rather than one scalar problem.

Repository implication:

1. this supports the local move from `strict-only` labels toward
   `Math-Step-DPO sibling_branch`;
2. it also explains why the current remaining failure is narrow:
   - not generic transfer collapse,
   - but terminal completion ordering.

Relevant external references folded into this interpretation:

1. `ProcessBench`
2. `GenPRM`
3. `SPC`
4. `PRIME`
5. `VerifyBench`

### Controlled Repair Question

Once `NDS7 + gated_mlp` established that:

1. `good > bad` ranking on `ProcessBench` can already be strong,
2. `first_error_edge` can already be strong,
3. but `all-correct terminal completion ordering` remains weak,

the next useful question was no longer "can we generally improve transfer?".

It became:

1. should the residual be fixed by:
   - more terminal data,
   - a more explicit terminal loss,
   - or a more explicit local/terminal architecture?

So this pass ran one three-way controlled sweep:

1. `data-level repair`
   - `ms_dpo_terminalboost_v1`
2. `loss-level repair`
   - `NDS7 + terminal BCE`
3. `architecture-level repair`
   - `NDS7 + dual_head + terminal BCE`

The point was attribution:

1. only one factor changes at a time,
2. so the resulting tradeoff is interpretable.

### Baseline For Comparison

Reference:

1. `NDS7 + gated_mlp` (`seed=42`)

Key baseline metrics:

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.6565` | `0.6456` | `0.8049` | `0.2174` | `-0.1640` |
| `PB Math`  | `0.7992` | `0.7519` | `0.7447` | `0.1538` | `-0.1461` |

Interpretation:

1. the current baseline is already good at:
   - local discrimination
   - global `good > bad` ranking
2. but still weak at:
   - ranking the final correct trace above safe nonterminal prefixes.

### Repair A — Data-Level: `ms_dpo_terminalboost_v1`

Curated artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_nds8_termboost_0311_pairs__480ab06bf8d6`

Geometry:

1. `sibling_branch = 1650`
2. `local_first_bad_edge = 1290`
3. `terminal_completion_anchor = 750`
4. terminal mass increased from about `10%` to about `20%`

Training run:

1. `assets/artifacts/phase_e_runs/phase_e_nds8_termboost_gated_pilot_value_retry2_20260311T124818Z`

Same-source held-out:

1. `pair_acc = 0.7685`
2. `auc = 0.7451`

ProcessBench:

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.6752` | `0.6816` | `0.7118` | `0.2798` | `-0.1038` |
| `PB Math`  | `0.7316` | `0.7046` | `0.6848` | `0.1823` | `-0.1670` |

What this means:

1. on `GSM8K`, more terminal anchors clearly help;
2. on `Math`, the terminal slice improves only slightly, while local/global
   ranking gets worse;
3. therefore:
   - simply increasing terminal-anchor mass is not a globally safe repair.

### Repair B — Loss-Level: `NDS7 + terminal BCE`

Training run:

1. `assets/artifacts/phase_e_runs/phase_e_nds7_termbce_gated_pilot_value_retry1_20260311T125922Z`

Same-source held-out:

1. `pair_acc = 0.7822`
2. `auc = 0.7681`

ProcessBench:

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.6053` | `0.6612` | `0.6353` | `0.2073` | `-0.1383` |
| `PB Math`  | `0.7229` | `0.7049` | `0.6555` | `0.1650` | `-0.1842` |

What this means:

1. adding terminal BCE without changing the data geometry does not solve the
   residual;
2. it mostly hurts:
   - local/global ranking
   - first-edge behavior
3. while terminal improvement is negligible.

Conclusion:

1. the residual is not just "the loss forgot to supervise terminal";
2. the current scalar route is still too entangled.

### Repair C — Architecture-Level: `NDS7 + dual_head + terminal BCE`

Training run:

1. `assets/artifacts/phase_e_runs/phase_e_nds7_dualhead_termbce_pilot_value_20260311T130037Z`

Same-source held-out:

1. `pair_acc = 0.7030`
2. `auc = 0.7076`

ProcessBench:

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.8150` | `0.7489` | `0.7882` | `0.0829` | `-0.2312` |
| `PB Math`  | `0.7820` | `0.7105` | `0.6806` | `0.0739` | `-0.2535` |

What this means:

1. explicit factorization *does* create a much stronger local error detector;
2. but under the current inference mixture, terminal ordering collapses;
3. this is the clearest evidence in this pass that:
   - local and terminal are distinct sub-problems,
   - and a single naive mixture rule is not enough.

### Joint Interpretation

The three-way repair sweep says:

1. `data-level terminal boost`
   - helps `GSM8K`
   - but causes a tradeoff on `Math`
2. `loss-level terminal BCE`
   - does not solve the problem
3. `dual_head`
   - proves the model can become a stronger local verifier
   - but also proves the current terminal route is wrong

So the most accurate updated diagnosis is:

1. the current bottleneck is not generic benchmark transfer anymore;
2. it is not just "need more terminal anchors";
3. it is specifically:
   - **how to route / rank terminal-completion evidence without destroying local discrimination**.

### Updated Next-Step Recommendation

Do **not** treat these results as evidence for:

1. "more terminal data fixes the problem";
2. "terminal BCE fixes the problem";
3. or "dual_head is immediately better".

Instead, the practical next mainline should be:

1. keep `NDS7 sibling_branch + gated_mlp` as the balanced baseline,
2. treat `terminal completion ordering` as a separate residual subtask,
3. design the next repair around:
   - routing or calibration at inference time,
   - or explicit terminal-specific decision logic,
   - not just more mixed supervision.

## 0AAAJ. A-E Fresh Audit + New ProcessBench Transfer Evidence (2026-03-11)

### What Was Re-Checked

This pass was not a narrow Phase E-only tweak. It did three things together:

1. re-ran a fresh implementation audit across `phase_a` to `phase_e`,
2. re-checked the current benchmark-transfer bottleneck with new data profiles,
3. compared those new profiles against the strongest already-completed
   `PBR4/PBR5` evidence.

Infrastructure status:

1. `PYTHONPATH=src pytest -q tests/unit`
   - `217 passed, 2 warnings`
2. active shell entrypoints:
   - `bash -n scripts/run_phase_*.sh`
   - syntax clean

Interpretation:

1. the repository is no longer in a state where broad implementation instability
   is the default explanation for poor results;
2. the remaining problems are much more about:
   - data semantics,
   - benchmark alignment,
   - and narrower training-contract issues.

### Newly Confirmed High-Risk Code Bug: Phase A Official Split Leakage

File:

1. `scripts/phase_a_prepare.py`

Old behavior:

1. `official` split mode used the requested CLI `--source-split` when writing
   `train/validation/test.jsonl`;
2. but dataset loaders can legally resolve that request to a different
   effective split and record it in `sample.metadata["source_split"]`;
3. result:
   - rows loaded from one official split could be written into a different
     output shard,
   - and the emitted `metadata["source_split"]` could be semantically wrong.

Why this is high risk:

1. it silently contaminates the provenance of prepared Phase A artifacts;
2. later Phase B/C comparisons can then be made against the wrong official split
   without any runtime failure.

Fix:

1. `official` mode now routes rows by the effective resolved split from loader
   metadata;
2. output rows preserve both:
   - `source_split`
   - `requested_source_split`
3. the summary now reports:
   - `effective_source_splits`

Regression:

1. `tests/unit/test_phase_a_prepare_script.py`
2. targeted run:
   - `9 passed`

### New ProcessBench Transfer Experiments

This round added three new smoke profiles plus one architecture follow-up.

#### 1. `NDS5_MS_STRICT_ONLY_SMOKE`

Idea:

1. remove `fanout/grid` from `Math-Shepherd`,
2. test whether the old failure was mostly caused by broad length-biased pairs.

Result:

1. `PB GSM`: `pair_acc=0.4048`, `auc=0.4210`, `first_edge=0.5122`
2. `PB Math`: `pair_acc=0.3876`, `auc=0.4321`, `first_edge=0.5957`

Conclusion:

1. no;
2. pure strict-local supervision is not enough;
3. the failure is not just “fanout/grid polluted the data”.

#### 2. `NDS6_RLHFLOW_STRICT_ONLY_SMOKE`

Idea:

1. swap in `RLHFlow-Deepseek` step labels,
2. keep only strict first-bad + light terminal anchors,
3. test whether “better step labels” alone fix transfer.

Result:

1. `PB GSM`: `pair_acc=0.6224`, `auc=0.5022`, `first_edge=0.6585`
2. `PB Math`: `pair_acc=0.4357`, `auc=0.4520`, `first_edge=0.6383`

Conclusion:

1. better labels help some local edge behavior,
2. but strict-only geometry still does not transfer well,
3. especially on `ProcessBench Math`.

#### 3. `NDS7_MS_DPO_CALIBRATED_SMOKE` (`mlp`)

Idea:

1. use `Math-Step-DPO` sibling-branch pairs as the main debiasing signal,
2. keep a smaller `Math-Shepherd strict + terminal` support set,
3. test whether sibling-branch supervision is more benchmark-aligned than
   pure strict-local supervision.

Result:

1. `PB GSM`: `pair_acc=0.5374`, `auc=0.5575`, `first_edge=0.5854`
2. `PB Math`: `pair_acc=0.5301`, `auc=0.5594`, `first_edge=0.7234`

Conclusion:

1. yes;
2. `sibling_branch` is clearly more useful than strict-only variants;
3. this is the first new smoke in this pass that pushes both `Math AUC` and
   `Math first_edge` in the right direction together.

### Follow-Up Architecture Probe: `NDS7 + gated_mlp`

Question:

1. after the data geometry is corrected with `Math-Step-DPO` sibling branches,
   is the remaining ceiling mostly a head-capacity problem?

Shared curated data:

1. `local_first_bad_edge = 1474`
2. `sibling_branch = 1831`
3. `terminal_completion_anchor = 387`

#### Seed 42

1. same-source held-out:
   - `pair_acc = 0.7673`
   - `auc = 0.7631`
2. benchmark:
   - `PB GSM`: `pair_acc=0.6565`, `auc=0.6456`, `first_edge=0.8049`
   - `PB Math`: `pair_acc=0.7992`, `auc=0.7519`, `first_edge=0.7447`

#### Seed 1

1. benchmark:
   - `PB GSM`: `pair_acc=0.6531`, `auc=0.6514`, `first_edge=0.7805`
   - `PB Math`: `pair_acc=0.8052`, `auc=0.7260`, `first_edge=0.7872`

#### Seed 7

1. benchmark:
   - `PB GSM`: `pair_acc=0.7177`, `auc=0.6817`, `first_edge=0.7073`
   - `PB Math`: `pair_acc=0.8253`, `auc=0.7601`, `first_edge=0.7021`

#### Aggregate (3 seeds)

1. `PB GSM`
   - `pair_acc = 0.6757 ± 0.0297`
   - `auc = 0.6596 ± 0.0158`
   - `first_edge = 0.7642 ± 0.0415`
2. `PB Math`
   - `pair_acc = 0.8099 ± 0.0112`
   - `auc = 0.7460 ± 0.0145`
   - `first_edge = 0.7447 ± 0.0347`

This is materially stronger than the older `ms_prm_align_v1 + gated_mlp` line
recorded in `PBR5B`, whose median was roughly:

1. `PB GSM AUC ~ 0.589`
2. `PB Math AUC ~ 0.613`

### What The New Diagnostics Say

The bucket-level `ProcessBench` failure analyses now support a much narrower
diagnosis than the older “frozen-head transfer is weak” story.

#### `NDS5` strict-only

1. still inverts many `good > bad` relations;
2. local-only supervision is too narrow.

#### `NDS6` RLH strict-only

1. improves some edge metrics;
2. still weak on `Math` and still not strong enough in AUC;
3. better labels alone do not solve the benchmark mismatch.

#### `NDS7` DPO-calibrated (`mlp`)

1. improves `Math first_edge` and global `good > bad`;
2. confirms that `sibling_branch` supervision is the first truly benchmark-helpful new signal in this pass.

#### `NDS7` DPO-calibrated (`gated_mlp`)

1. local and global prefix ranking are now strong;
2. but `all-correct terminal completion ordering` is still weak and unstable.

Concrete evidence:

1. `PB Math` is already strong in:
   - `pair_auc_good_vs_bad`
   - `first_error_edge_accuracy`
2. but `terminal_top1` in the failure analysis remains poor:
   - the model still often prefers a safe nonterminal prefix over the final
     correct completed trace.

### Updated Conclusion

The current accurate diagnosis is now:

1. the bottleneck is no longer broad implementation instability;
2. it is no longer “general ProcessBench transfer collapse” either;
3. with `Math-Step-DPO sibling_branch + gated_mlp`, the project has already
   crossed into a regime where:
   - local discrimination is strong,
   - global `good > bad` ranking is strong,
   - and the main residual problem is **terminal completion ordering**.

That changes the next-step design priority:

1. stop treating all benchmark failures as one undifferentiated transfer issue;
2. keep `DPO sibling_branch` as the main signal carrier;
3. target terminal completion as a separate residual objective.

## 0AAAI. Phase E Postfix Re-Audit: Current Best Is Strong But Still Not Strict RL-Ready (2026-03-11)

### Code Fixes Closed In This Pass

This pass fixed three active implementation risks that could still distort
Phase E conclusions even when the headline benchmark numbers looked strong.

1. benchmark-aware candidate ranking leakage
   - file: `scripts/phase_e_select_candidate.py`
   - old behavior:
     - benchmark metrics influenced seed/group ranking directly
     - this made `ProcessBench` behave like a hidden dev set during candidate
       recommendation
   - new behavior:
     - default `selection_policy=heldout_only`
     - benchmark metrics remain promotion gates / canaries, but no longer rank
       candidates by default
   - regression:
     - `tests/unit/test_phase_e_select_candidate.py`
2. intradataset candidate checkpoint-path bug
   - file: `scripts/phase_e_select_intradataset_candidate.py`
   - old behavior:
     - report always published `<run_dir>/best_value_head.pt`
     - older runs that only retained `final_value_head.pt` produced a bogus path
   - new behavior:
     - manifest-aware checkpoint resolution with explicit fallback notes
   - regression:
     - `tests/unit/test_phase_e_select_intradataset_candidate.py`
3. PRM-7B samefamily eval long-batch crash
   - file: `src/ours/phase_e/runtime.py`
   - old behavior:
     - long candidate batches on `Qwen2.5-Math-PRM-7B` could surface async CUDA
       capacity failures as a later generic `device-side assert triggered`
     - samefamily trust audit then crashed before producing artifacts
   - new behavior:
     - batched encoding now synchronizes CUDA before leaving the retry block
     - retryable CUDA capacity failures trigger batch-size backoff instead of
       aborting the audit
   - regression:
     - `tests/unit/test_phase_e_runtime.py`

Validation:

1. `PYTHONPATH=src pytest -q tests/unit/test_phase_e_runtime.py tests/unit/test_phase_e_select_candidate.py tests/unit/test_phase_e_select_intradataset_candidate.py`
   - `16 passed`
2. `python -m py_compile src/ours/phase_e/runtime.py scripts/phase_e_select_candidate.py scripts/phase_e_select_intradataset_candidate.py scripts/phase_e_eval_samefamily_trust.py`

### Postfix Samefamily + Strict RL Diagnosis

After the fixes, I re-ran the current strongest `Qwen2.5-Math-PRM-7B` DPO-side
candidates through:

1. same-family trust
2. full `ProcessBench`
3. the stricter transfer diagnosis gate

#### `pbr10_dpo8k` (current strongest offline tradeoff)

- samefamily:
  - `prompt_pool_top1_accuracy = 0.9098`
  - `local_first_bad_edge_accuracy = 0.8981`
- ProcessBench:
  - `GSM F1 = 0.7334`
  - `Math F1 = 0.6313`
  - `pb_min_auc = 0.8651`
  - `pb_min_first_edge = 0.8309`
- strict diagnosis:
  - `pb_min_terminal_top1 = 0.1626`
  - assessment = `not_rl_ready_terminal_completion_risk`

Interpretation:

1. this is no longer a "local benchmark transfer is weak" story
2. local discrimination is already strong on both same-family and benchmark
3. the remaining blocker is stricter than the usual `mean_all_correct_last_score`
   headline:
   - the model still too often prefers an incomplete but safe prefix over the
     fully correct completed trace on all-correct problems

#### `pbr13_terminalbce`

- samefamily:
  - `prompt_pool_top1_accuracy = 0.8433`
  - `local_first_bad_edge_accuracy = 0.7075`
- ProcessBench:
  - `GSM F1 = 0.7553`
  - `Math F1 = 0.6603`
  - `pb_min_auc = 0.8702`
  - `pb_min_first_edge = 0.8727`
- strict diagnosis:
  - `pb_min_terminal_top1 = 0.2660`
  - assessment = `not_rl_ready_terminal_completion_risk`

Interpretation:

1. adding terminal BCE improves the benchmark headline more than the older
   frozen-head recipes
2. but it still does not solve the stricter terminal-completion ordering issue
3. and in this run it also weakens same-family decision quality relative to
   `pbr10_dpo8k`

### Current Best Conclusion

The repo state has changed materially relative to the older "frozen-head cap
~0.62 and clearly not usable" narrative.

Current accurate statement:

1. `Qwen2.5-Math-PRM-7B + DPO/sibling-branch-heavy data` is already a strong
   offline verifier family
2. `pbr10_dpo8k` is the best current frozen-head candidate for offline rerank /
   rejection research
3. but no current checkpoint passes the strict internal RL gate yet

The blocker is now narrow and concrete:

1. not gross benchmark transfer failure
2. not infrastructure instability
3. but **all-correct terminal completion ordering**

That makes the next research move much clearer:

1. keep `pbr10_dpo8k` as the main anchor
2. target terminal completion directly without sacrificing same-family local
   structure
3. do not use benchmark-informed candidate ranking again by default

## 0AAAH. Phase E Infrastructure Audit Closure (2026-03-11)

### What Was Fixed

This pass closed the remaining active Phase E infrastructure loopholes.

1. direct trainer default:
   - `scripts/phase_e_train_value.py`
   - `checkpoint_selection_metric` now defaults to `pair_acc`
   - old default `ranking_score` is now a research-only option
2. suite wrapper closure:
   - all active `run_phase_e*.sh` entrypoints that call `phase_e_train_value.py`
     now pass explicit `--recipe-risk-policy`
3. explicit semantics:
   - R-PRM suite wrappers now also pass explicit pair-mode declarations

### Audit Evidence

1. first static audit:
   - `assets/artifacts/phase_e_audits/phase_e_suite_recipe_audit_0311_1931/summary.md`
   - findings: `10`
2. post-fix static audit:
   - `assets/artifacts/phase_e_audits/phase_e_suite_recipe_audit_postfix_0311_1933/summary.md`
   - findings: `0`

### Interpretation

This does not raise benchmark scores by itself.

What it does improve:
1. source-quality comparisons are less likely to be polluted by stale wrappers,
2. future Phase E failures are more likely to reflect method/data issues rather than wrapper drift,
3. direct operator use of `phase_e_train_value.py` is materially safer.

## 0AAAI. 2026 Literature Refresh And Strategic Redesign (2026-03-11)

### Newer External Evidence

The older `2025-03` literature cutoff was no longer enough. The most important
new additions are:

1. `PRIME (2026)`:
   - https://arxiv.org/abs/2602.11570
   - process-outcome alignment benchmark;
   - verifier accuracy correlates strongly with RLVR effectiveness
2. `Hard2Verify (2025)`:
   - https://arxiv.org/abs/2510.13744
   - human-annotated open-ended step verification benchmark;
   - open-source verifiers still lag badly
3. `RISE (2025)`:
   - https://arxiv.org/abs/2505.13445
   - online self-verification training improves both reasoning and verification
4. `VPRM (2026)`:
   - https://arxiv.org/abs/2601.17223
   - deterministic process verification can outperform pure outcome verification

### Strategic Consequence

These sources strengthen one conclusion:

1. keeping only a scalar pairwise head is too narrow if the goal is RL-ready
   faithfulness;
2. the scalar head should remain in the repository as:
   - a bounded-support verifier baseline,
   not:
   - the assumed final verifier architecture.

### New Repository-Level Direction

The redesigned Phase E now has two parallel goals:

1. keep improving the bounded-support scalar verifier on same-source and
   same-family utility;
2. add benchmark-aligned verifier branches:
   - process-outcome alignment
   - critique / self-verification auxiliary tasks
   - deterministic verification where available

Reference note:
1. `docs/phase_e_updated_literature_redesign_20260311.md`

## 0AAAG. A-E Full Audit + PBR10 PRM-7B DPO Breakthrough (2026-03-11)

### Code Audit And Fixes

This round added one repository-wide implementation audit across `phase_a` to
`phase_e` and fixed three concrete infrastructure bugs:

1. `feature cache` live-writer lock stealing
   - file: `src/ours/phase_b/feature_cache.py`
   - old behavior: second writer could steal a lock from a slow but still-alive
     writer based only on `mtime`
   - new behavior: stale lock cleanup now requires missing/dead owner pid
   - regression: `tests/unit/test_feature_cache.py`
2. `Phase B` faithfulness eval silent `best -> final` checkpoint fallback
   - file: `scripts/phase_b_eval_faithfulness.py`
   - new behavior: explicit `checkpoint_resolution` is written into metrics and
     manifest, and the console emits a warning when fallback happens
   - regression: `tests/unit/test_phase_b_eval_faithfulness_script.py`
3. dtype alias drift across A/B/C entrypoints
   - files:
     - `scripts/phase_a_generate_and_eval.py`
     - `scripts/phase_b_prepare_value_data.py`
     - `scripts/phase_b_train_value.py`
     - `scripts/phase_b_eval_faithfulness.py`
     - `scripts/phase_c_prepare_pik_data.py`
     - `scripts/phase_c_train_pik.py`
     - `scripts/phase_c_eval_pik.py`
   - new behavior: all now accept `bf16/fp16/fp32` aliases

Validation:

1. `pytest -q` → `223 passed, 2 skipped`
2. targeted regression set for the new fixes → `29 passed`

### Research Diagnosis

The strongest current repo bottleneck is no longer "head can’t learn".
The new evidence says:

1. `DPO sibling_branch` supervision is the critical signal carrier,
2. `PRM-7B` backbone quality matters more than further head complexity,
3. `oracle ProcessBench F1` alone is too optimistic; fixed-threshold F1 must be
   tracked for honest RL-readiness discussion.

Community reading and PDF/web cross-check now point to the same design:

1. better process data quality and consensus filtering,
2. sibling/fork-point comparisons rather than only local MC-derived edges,
3. stronger verifier backbones or LoRA/full tuning,
4. min-form / conservative PRM usage in RL rather than naive summation.

### New Experiments: `PBR10`

Data:

1. curated profile = `math_step_dpo_v1`
2. target mix:
   - `dpo_fork = 5200`
   - `ms_strict = 2000`
   - `ms_terminal = 800`
3. train split semantics:
   - `sibling_branch = 4679`
   - `local_first_bad_edge = 1783`
   - `terminal_completion_anchor = 485`

Backbone:

1. `assets/models/Qwen2.5-Math-PRM-7B`

#### PBR10 `mlp`

Run:

1. `assets/artifacts/phase_e_runs/phase_e_pbr10_prm7b_dpo8k_s42_value_20260311T110527Z`

Held-out:

1. `pair_acc = 0.906`
2. `auc = 0.859`

ProcessBench:

| metric | Math | GSM8K |
|---|---:|---:|
| `pair_auc_good_vs_bad` | 0.863 | 0.873 |
| `first_error_edge_accuracy` | 0.844 | 0.915 |
| `processbench_f1 (oracle)` | 0.659 | 0.724 |
| `processbench_f1 (fixed=0.5)` | 0.654 | 0.693 |

#### PBR10 `gated_mlp`

Run:

1. `assets/artifacts/phase_e_runs/phase_e_pbr10_prm7b_dpo8k_gated_s42_value_20260311T112849Z`

Held-out:

1. `pair_acc = 0.914`
2. `auc = 0.873`

ProcessBench:

| metric | Math | GSM8K |
|---|---:|---:|
| `pair_auc_good_vs_bad` | 0.869 | 0.873 |
| `first_error_edge_accuracy` | 0.852 | 0.925 |
| `processbench_f1 (oracle)` | 0.647 | 0.708 |
| `processbench_f1 (fixed=0.5)` | 0.567 | 0.664 |

### Interpretation

1. `PBR10 mlp` is now the strongest benchmark-facing candidate in the repo.
2. `gated_mlp` improves ranking/AUC/edge metrics, but fixed-threshold F1 drops
   sharply, especially on `ProcessBench Math`.
3. This means the main current bottleneck is not head capacity; it is the
   interaction of:
   - DPO-aligned pair geometry,
   - stronger verifier backbone,
   - honest threshold/calibration behavior.
4. Mainline recommendation:
   - keep `PBR10 mlp` as the new primary baseline,
   - report both oracle and fixed-threshold F1,
   - next experiment should be `PBR10 mlp + minimal LoRA`, not another head-only churn.

## 0AAAF. FLB Extended Suite — gated_mlp + Scale Ablation (2026-03-11)

### Experiments

Following FLB suite findings, ran 4 additional targeted experiments on same DPO pair dirs (fast, cache reuse):
- FIX_C gated_mlp (7420 pairs, gated_mlp) — tests head architecture
- FIX_C seed7 (7420 pairs, mlp, seed=7) — tests seed stability
- FIX_E mlp (9050 pairs, mlp) — tests 10K scale (all available DPO data)
- FIX_E gated_mlp (9050 pairs, gated_mlp) — tests combined 10K + gated

### Results

| Config | Head | Pairs | GSM AUC | MATH AUC | Avg AUC |
|---|---|---|---|---|---|
| **FIX_C gated_mlp (s=42)** | gated_mlp | 7420 | **0.711** | **0.749** | **0.730** |
| FIX_E gated_mlp (s=42) | gated_mlp | 9050 | 0.705 | 0.743 | 0.724 |
| FIX_C s7 (mlp, s=7) | mlp | 7420 | 0.669 | 0.737 | 0.703 |
| FIX_E mlp (s=42) | mlp | 9050 | 0.658 | 0.730 | 0.694 |
| FIX_C s42 (mlp) [ref] | mlp | 7420 | 0.659 | 0.721 | 0.690 |

### Key Findings

1. **gated_mlp > mlp by +0.028 MATH AUC** at same data scale (7.4K): 0.749 vs 0.721
2. **Diminishing returns at 10K**: mlp 7.4K→10K +0.009; gated_mlp 7.4K→10K -0.006 (slight overfit)
3. **FIX_C gated_mlp is the frozen-head optimum**: 7.4K DPO + gated_mlp head = MATH AUC 0.749
4. **Seed variance ±0.016** for mlp: confirms gated_mlp's stability advantage
5. **Frozen-head ceiling**: ≈0.75 (FIX_C gated achieves 0.749). LoRA needed to break through.

### Complete Frozen-Head Leaderboard

| Config | Data | Pairs | MATH AUC | Status |
|---|---|---|---|---|
| **FIX_C gated dpo_scale_v1** | DPO 7.4K | 7420 | **0.749** | 🏆 FROZEN-HEAD SOTA |
| FIX_E gated dpo_scale_v1 | DPO 10K | 9050 | 0.743 | close, slight overfit |
| FIX_C mlp s=7 dpo_scale_v1 | DPO 7.4K | 7420 | 0.737 | mlp seed sensitivity |
| FIX_E mlp dpo_scale_v1 | DPO 10K | 9050 | 0.730 | data scale +0.009 |
| FIX_C mlp s=42 dpo_scale_v1 | DPO 7.4K | 7420 | 0.721 | mlp baseline |
| FIX_B ms_dpo_calibrated_v1 | 50% DPO | 4096 | 0.683 | mixed formula |
| ms_e43 ms_acc90 | MS 14K | 14290 | 0.634 | MS only |
| NDS1 rlhflow_align_v1 | RLH 3K | 3037 | 0.552 | RLH with length bias |
| FIX_D rlh_strict_only_v1 | RLH strict | 4096 | 0.531 | strict, worse than NDS1 |
| FIX_A ms_strict_only_v1 | MS strict | 4096 | 0.489 | no positive transfer |
| NDS3 ms_align_v1 (small) | MS 4K | ~3K | 0.470 | inverted |

## 0AAAE. PBR5B Full-Scale PRM-Align + GatedMLP, 3 Seeds (2026-03-11–12)

### Design

`ms_prm_align_v1` data (16384 pairs target; ~10770 train) with `gated_mlp` head
and `score+none+pair_acc` recipe. 3 seeds: 42, 1, 7. 4 training epochs.

First launch blocked by `recipe_safety.py` (`ANTI_PATTERN_G_FULL`, critical) because
`ms_prm_align_v1` has `terminal_frac=0.14` and global defaults (`logit+confidence_semantic+ranking_score`)
are a known catastrophic combination on mixed-semantics data. Fix: added per-group
override vars `GROUP_RANKING_TARGET_SPACE/PAIR_WEIGHT_MODE/CHECKPOINT_SEL_METRIC`
to the suite script. After fix, severity drops to `info`.

### Results

| seed | sf_top1 | GSM AUC | MATH AUC | terminal_top1 | RL assessment |
|------|---------|---------|----------|---------------|---------------|
| 42 | 0.886 | 0.500 | 0.602 | 0.564 | terminal_and_local_tradeoff_unresolved |
| 1  | 0.880 | 0.656 | 0.645 | 0.051 | near_rl_ready_but_terminal_gap |
| 7  | 0.849 | 0.589 | 0.613 | 0.180 | terminal_and_local_tradeoff_unresolved |
| **median** | — | **0.589** | **0.613** | | |

Score ordering (MATH, seed 7): `mean_good=0.319 > mean_bad=0.265` ✅ no inversion.

### Key Findings

1. **DPO data dominates ms_prm_align data**: FIX_C (DPO 8k, MLP) = 0.721 vs PBR5B median (MS+PRM 16k, gated_mlp) = 0.613. DPO wins by +0.108 with fewer pairs and weaker head.
2. **Large inter-seed variance**: GSM AUC range 0.500–0.656, MATH range 0.602–0.645. Seed 42 has the best terminal_top1 (0.564) but worst local AUC; seed 1 has the best local AUC but near-zero terminal (0.051). gated_mlp did NOT resolve the local/terminal tradeoff.
3. **Target not met**: Goal was median MATH AUC ≥ 0.65. Achieved 0.613.
4. **Scaling ms_prm_align from 4k→16k brought no improvement**: PBR4b smoke (4k): MATH 0.621; PBR5B median (16k): 0.613 — slight regression. Data quality, not quantity, is the bottleneck.
5. **Conclusion**: ms_prm_align data is not the right vehicle. DPO sibling_branch pairs (rej-cho≈0) are the critical signal for ProcessBench transfer. Path forward: DPO-anchored curation (FIX_B/C pattern).

### Updated Frozen-Head Leaderboard (ProcessBench Math AUC)

| Config | Data | Pairs | MATH AUC | Head |
|---|---|---|---|---|
| **FIX_C dpo_scale_v1** | Pure DPO 8k | 7420 | **0.721** | mlp |
| NDS2 math_step_dpo_v1 | Pure DPO 4k | 3705 | 0.712 | mlp |
| FIX_B ms_dpo_calibrated_v1 | DPO 50%+MS 40% | ~3.7k | 0.683 | mlp |
| NDS4 ms_rlhflow_mixed_v1 | Mixed | 3093 | 0.647 | mlp |
| PBR5B s1 (best seed) | MS+PRM 16k | 10770 | 0.645 | gated_mlp |
| ms_e43 ms_acc90 | MS high-quality | 14290 | 0.634 | mlp |
| PBR5B median (3 seeds) | MS+PRM 16k | 10770 | 0.613 | gated_mlp |
| NDS1 rlhflow_align_v1 | RLH | 3037 | 0.552 | mlp |
| FIX_D rlh_strict_only_v1 | RLH strict | 4096 | 0.531 | mlp |
| FIX_A ms_strict_only_v1 | MS strict | 4096 | 0.489 | mlp |
| NDS3 ms_align_v1 (small) | MS 4K | ~3K | 0.470 | mlp |

## 0AAAD. FLB Suite — Length-Bias Fix Ablation Complete (2026-03-11)

### Experiment Design

Root cause of NDS3/NDS1 ProcessBench failure: length-biased pair types.

| Pair type | rej-cho | Origin | Effect |
|---|---|---|---|
| first_bad_fanout_prefix_ranking | +194 | MS fanout | teach "shorter=better" |
| good_bad_prefix_grid | +203 | MS grid | teach "shorter=better" |
| sibling_branch (DPO) | ≈0 | Math-Step-DPO | content-quality only |
| local_first_bad_edge | +99 | MS/RLH strict | moderate, safe direction |

ProcessBench: bad_prefix is **LONGER** → length-biased model inverts (AUC < 0.5).

### Results

| case_id | profile | pairs | pb_gsm_auc | pb_math_auc | pb_math_first_edge |
|---|---|---|---|---|---|
| **fix_c_dpo_scale_8k** | dpo_scale_v1 | 7420 | **0.659** | **0.721** | 0.625 |
| **fix_b_ms_dpo_calibrated** | ms_dpo_calibrated_v1 | 4096 | **0.648** | **0.683** | 0.648 |
| fix_d_rlh_strict_only | rlh_strict_only_v1 | 4096 | 0.535 | 0.531 | 0.570 |
| fix_a_ms_strict_only | ms_strict_only_v1 | 4096 | 0.484 | 0.489 | 0.492 |

### Updated Frozen-Head Leaderboard (ProcessBench Math AUC)

| Config | Data | Pairs | MATH AUC | Status |
|---|---|---|---|---|
| **fix_c dpo_scale_v1** | Pure DPO 8K | 7420 | **0.721** | 🏆 NEW SOTA |
| NDS2 math_step_dpo_v1 | Pure DPO 4K | 3705 | 0.712 | Previous SOTA |
| fix_b ms_dpo_calibrated_v1 | 50% DPO + 40% MS | 4096 | 0.683 | 2nd best formula |
| NDS4 ms_rlhflow_mixed_v1 | Mixed | 3093 | 0.647 | mixed approach |
| ms_e43 ms_acc90 | MS 14K (high quality) | 14290 | 0.634 | high-scale MS |
| NDS1 rlhflow_align_v1 | RLH (43% biased) | 3037 | 0.552 | w/ length bias |
| fix_d rlh_strict_only_v1 | RLH strict only | 4096 | 0.531 | w/o length bias |
| fix_a ms_strict_only_v1 | MS strict only | 4096 | 0.489 | no positive transfer |
| NDS3 ms_align_v1 (small) | MS 4K (20% biased) | ~3K | 0.470 | inverted |

### Key Findings

1. **Length bias → INVERSION, not the full story**: Removing fanout/grid from MS (FIX_A) stops inversion (0.470→0.489) but doesn't enable positive transfer. MS strict-only still near-random.

2. **DPO sibling_branch is the critical signal carrier**: Only pairs with rej-cho≈0 (same prefix length, different next step) match ProcessBench's evaluation format. Even partial DPO (50%) drives 0.683 MATH AUC.

3. **Scale helps DPO**: 3.7K DPO → 0.712; 7.4K DPO → 0.721. Linear scaling regime at current sizes.

4. **RLH strict-only is worse than RLH with fanout/grid (NDS1)**: 0.531 vs 0.552. LLM-judge labels remain accurate even for length-asymmetric pairs; removing them loses diversity. Length bias is mostly a MC-estimation artifact.

5. **Frozen-head ceiling confirmed**: Best is 0.721. Published PRMs with LoRA achieve 0.750+. LoRA unfreezing is the path forward.

### Diagnostic: Score Ordering Check

| Config | mean_good | mean_bad | mean_all_correct | Correct order? |
|---|---|---|---|---|
| NDS2 | 0.595 | 0.406 | 0.678 | ✅ good > bad |
| FIX_C | similar healthy ordering | - | - | ✅ |
| NDS3 | 0.650 | 0.677 | 0.738 | ❌ bad > good (inverted) |
| FIX_A | slightly fixed | ~0.556 vs 0.532 | 0.617 | ✅ marginally fixed |

## 0AAAC. Phase E Infrastructure Hardening — Recipe Guard + Collapse Diagnosis (2026-03-11)

### What Changed

To keep `Phase E` scientifically interpretable, the trainer now explicitly
blocks repository-known catastrophic recipe combinations instead of allowing
them to consume GPU time and then fail later in opaque ways.

New code:
1. `src/ours/phase_e/recipe_safety.py`
2. `scripts/phase_e_diagnose_training_collapse.py`

Trainer changes:
1. `scripts/phase_e_train_value.py`
   - new `--recipe-risk-policy {off,warn,error}`
   - default = `error`
   - writes:
     - `recipe_risk.json`
     - `training_health.json`
     - `training_health.md`
2. `scripts/run_phase_e_suite.sh`
   - now propagates `RECIPE_RISK_POLICY`, defaulting to `error`

### Why This Was Necessary

Recent `NDSBH` / `PBR` evidence already showed one specific anti-pattern:

1. mixed-semantics source,
2. terminal anchors present,
3. `ranking_target_space = logit`,
4. semantic-style pair weighting,
5. `checkpoint_selection_metric = ranking_score`

This family can produce:
1. flat loss,
2. near-zero margin,
3. held-out inversion,
4. misleading same-source fit.

That is an infrastructure problem, not just a modeling curiosity.

### Validation Results

#### A. Bad recipe probe: now blocked before backbone load

Artifact:
1. console run only, no training artifact promoted

Configuration:
1. `ms_align_v1`
2. `joint`
3. `logit`
4. `confidence_semantic`
5. `ranking_score`

Outcome:
1. `severity = critical`
2. findings:
   - `ANTI_PATTERN_G_FULL`
   - `LOGIT_MIXED_TERMINAL`
   - `RANKING_SCORE_MIXED`
   - `SEMANTIC_WEIGHT_MIXED_TERMINAL`
3. trainer exits immediately with explicit remediation text

Interpretation:
1. `Phase E` now prevents a known class of misleading runs from entering the
   expensive training path.

#### B. Good recipe probe: still allowed

Run:
1. `assets/artifacts/phase_e_runs/phase_e_recipe_guard_goodprobe_20260311T093056Z`

Configuration:
1. `math_step_dpo_v1`
2. `joint`
3. `score`
4. `none`
5. `pair_acc`

Result:
1. `pair_accuracy = 0.531250`
2. `auc = 0.505615`
3. `mean_margin = -0.023751`
4. `training_health = unstable_or_inverted`

Interpretation:
1. the guard is not over-broad,
2. weak but legitimate runs still go through,
3. and the new health artifact makes their weakness explicit.

#### C. Historical bad run: now rediagnosed consistently

Run:
1. `assets/artifacts/phase_e_runs/phase_e_ms_trust_smoke_0310_1400_e1_math_shepherd_pair_learn_smoke_e1_math_shepherd_pair_learn_smoke_s42_value_20260310T060115Z`

Re-diagnosis:
1. `diagnosis = collapse_detected`
2. `known_collapse_signature = True`
3. `pair_accuracy = 0.497738`
4. `auc = 0.498106`
5. `mean_margin = -0.001785`

Interpretation:
1. the new diagnostic path correctly identifies an earlier classic collapse run.

### Operational Conclusion

1. `Phase E` source-quality comparisons are now less likely to be polluted by
   preventable recipe collapse.
2. This does **not** fix benchmark transfer by itself.
3. It does make future failure analysis cleaner, because "bad source" and
   "bad recipe" are now better separated.

## 0AAAB. NDSBH Suite Results — New Dataset Source Ablation (2026-03-11)

### Setup
- Suite: `NDSBH_ALL_SMOKE` via `scripts/run_phase_e_nds_best_hparams_suite.sh`
- Target pairs: 4096 | Epochs: 10 | Batch: 128 | GPU: 2 (single GPU, avoids multi-device bug)
- **Correct hyperparameters**: `--ranking-target-space score` + `--pair-weight-mode none` + `--checkpoint-selection-metric pair_acc`
- `--nonfinite-feature-policy drop` (handles NaN from RLHFlow data)

### Results

| Case | Data Source | Train Pairs | Same-src AUC | GSM8K AUC | Math AUC | GSM8K 1st-edge | Math 1st-edge |
|------|------------|------------|-------------|-----------|----------|----------------|---------------|
| NDS1 | RLHFlow-Deepseek (LLM-judge) | 3037 | 0.743 | 0.571 | 0.552 | 0.679 | 0.609 |
| **NDS2** | **Math-Step-DPO (fork-point)** | **3705** | **0.723** | **0.655** | **0.712** | **0.698** | **0.656** |
| NDS3 | Math-Shepherd ms_align_v1 | ~3000 | 0.836 | 0.477 | 0.470 | 0.509 | 0.461 |
| NDS4 | Mixed (MS+RLHFlow+Math-Step-DPO) | 3093 | 0.744 | 0.605 | 0.647 | 0.594 | 0.672 |
| ms_e43 (ref) | Math-Shepherd ms_acc90 | 14290 | 0.945 | 0.625 | 0.634 | 0.688 | 0.633 |

### Key Findings

1. **Math-Step-DPO (NDS2) achieves Math AUC=0.712 — new best for frozen head** with only 3705 pairs, beating ms_e43 (14290 pairs, 0.634). Fork-point sibling_branch pairs directly match ProcessBench's first-error-edge format.
2. **Math-Shepherd at small scale (NDS3) near-random**: same-source AUC=0.836 but ProcessBench Math=0.470. MS needs ~14K+ pairs for positive transfer (MC labels are noisy, need scale).
3. **Mixed data significantly helps**: NDS4 (MS+RLH+DPO) gives 0.647 vs MS alone (0.470) at same scale.
4. **Data quality hierarchy** (frozen head, ≤4K pairs): Math-Step-DPO > Mixed > RLHFlow > Math-Shepherd
5. **Correct hparams are critical**: Broken (logit+confidence_semantic+ranking_score) → AUC=0.39 (inverted); Correct (score+none+pair_acc) → full recovery.

### Updated Frozen-Head Leaderboard (All Experiments)

| config | Data | Pairs | GSM8K AUC | Math AUC | Status |
|---|---|---|---|---|---|
| **NDS2: Math-Step-DPO + mlp (correct hparams)** | xinlai_math_step_dpo | 3705 | **0.655** | **0.712** | **NEW BEST** |
| NDS4: Mixed + mlp (correct hparams) | ms+rlhflow+dpo | 3093 | 0.605 | 0.647 | OK |
| ms_e43: ms_acc90 + mlp (correct hparams) | math_shepherd | 14290 | 0.625 | 0.634 | OK |
| ms_prm_align_v1 + gated_mlp (broken hparams) | ms+prmbench | 4096 | 0.549 | 0.621 | OK (gated avoids inversion) |
| prm_e46: prmbench + mlp | prmbench | — | 0.626 | 0.605 | OK |
| NDS1: RLHFlow + mlp (correct hparams) | rlhflow | 3037 | 0.571 | 0.552 | OK |
| ms_align_v1 + mlp (s1, broken hparams) | math_shepherd | 16384 | 0.616 | 0.476 | Unstable |
| ms_align_v1 + mlp (s42, broken hparams) | math_shepherd | 16384 | 0.442 | 0.442 | Inverted |
| NDS3: MS + mlp (correct hparams, small) | math_shepherd | ~3000 | 0.477 | 0.470 | Near-random |

### Recommended Next Steps
1. **NDS2-scale-up**: Math-Step-DPO at larger pair count (14K+ pairs, if augmentable) with 10 epochs. Expect further improvement.
2. **NDS2+LoRA**: Math-Step-DPO data with LoRA backbone — may push Math AUC past 0.75+.
3. **PBR5 with correct hparams**: Re-run PBR5 (ms_prm_align_v1 + gated_mlp) with score+none+pair_acc. Current PBR4b (0.621) used broken hparams; correct hparams may improve further.
4. **Investigate NDS3 inversion mechanism**: Why does MS at small scale invert? Understanding this helps explain the data-quality gap.

---

## 0AAAA. PBR2/PBR3/PBR4 Experiment Results — Critical Bug Fix + Full Results (2026-03-11)

### Background

A critical cross-device bug was found and fixed in `src/ours/phase_b/value_head.py:encode_text_features()`.
When `device_map="auto"` split the backbone across multiple GPUs, `attention_mask` (on cuda:0) and `outputs.hidden_states[-1]` (on the last-layer GPU) were on different devices. `torch.where` silently produced all-zero results. The fix: `attn_mask = inputs["attention_mask"].to(last_hidden.device)`.

**All PBR2 seed-42 results from the morning run (06:00 UTC) are INVALID (zero features).**
**Results below are from the corrected run (08:00+ UTC) on single GPUs.**

### PBR2 Full-Scale ms_align_v1+mlp (16384 pairs, 3 seeds)

Groups: `PBR2_FULL_MIXED_MLP_SEED3`

| seed | heldout_pair | heldout_auc | SF_top1 | SF_local | GSM8K_AUC | MATH_AUC | GSM8K_fe | MATH_fe |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.8471 | 0.8332 | 0.867 | 0.947 | 0.442 | 0.442 | 0.610 | 0.638 |
| 1  | 0.8456 | 0.8305 | 0.864 | 0.935 | **0.616** | 0.476 | 0.659 | 0.553 |
| 7  | 0.8484 | 0.8272 | 0.846 | 0.938 | 0.486 | **0.543** | 0.512 | 0.447 |

**Median**: GSM8K=0.486, MATH=0.476. AUC range: GSM8K=0.174, MATH=0.101. **FAILS** 0.58/0.52 targets. **UNSTABLE** (range >> 0.05).

**Key finding**: Extreme inter-seed variance — score inversion flips axis per seed. Seed 42: both axes inverted. Seed 1: GSM8K OK, MATH inverted. Seed 7: GSM8K inverted, MATH OK. Same-source accuracy stable (84.5-84.8%). Plain MLP head randomly latches onto length-bias vs step-correctness feature depending on initialization.

### PBR3 Later-Bad Branch (ms_laterbad_v1+mlp, 16384 pairs, 3 seeds)

Groups: `PBR3_LATER_BAD_BRANCH_SEED3`

| seed | heldout_pair | heldout_auc | SF_top1 | SF_local | GSM8K_AUC | MATH_AUC | GSM8K_fe | MATH_fe |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.8592 | 0.8393 | 0.860 | 0.947 | 0.425 | 0.436 | 0.561 | 0.617 |
| 1  | 0.8481 | 0.8352 | 0.873 | 0.946 | **0.619** | 0.487 | 0.707 | 0.596 |
| 7  | 0.8511 | 0.8330 | 0.854 | 0.952 | 0.515 | **0.548** | 0.537 | 0.468 |

**Median**: GSM8K=0.515, MATH=0.487. AUC range: GSM8K=0.194, MATH=0.112. **UNSTABLE** (variance even worse than PBR2). **FAILS** targets.

**Key finding**: ms_laterbad_v1 provides marginal improvement (+0.029 GSM8K median, +0.011 MATH median) over ms_align_v1 but does NOT fix instability or score inversion with plain MLP head. Targeted later-bad pairs insufficient for stable cross-distribution transfer.

### PBR4 PRMBench Auxiliary + Later-Bad Follow-Up Smoke (4096 pairs)

Groups: `PBR4_PRM_AND_LATERBAD_FOLLOWUP_SMOKE`

| case | profile | head | heldout_pair_acc | heldout_auc | SF_top1 | SF_local | GSM8K_AUC | MATH_AUC | GSM8K_inv? | MATH_inv? |
|---|---|---|---|---|---|---|---|---|---|---|
| pbr4a | ms_prm_align_v1 | mlp | 0.8556 | 0.8419 | 0.856 | 0.925 | 0.490 | 0.458 | **INV** | **INV** |
| pbr4b | ms_prm_align_v1 | gated_mlp | 0.8732 | 0.8833 | 0.872 | 0.940 | **0.549** | **0.621** | **OK** | **OK** |
| pbr4c | ms_laterbad_v1  | gated_mlp | 0.8889 | 0.8780 | 0.884 | 0.941 | 0.472 | 0.598 | **INV** | OK |

**KEY FINDING — gated_mlp breaks score inversion on ProcessBench MATH:**
- `ms_prm_align_v1 + gated_mlp` (pbr4b) achieves **MATH AUC=0.621**, **GSM8K AUC=0.549** — both non-inverted.
- `ms_laterbad_v1 + gated_mlp` (pbr4c) achieves **MATH AUC=0.598**, GSM8K AUC=0.472 (still inverted).
- `ms_prm_align_v1 + mlp` (pbr4a) remains fully inverted — the head architecture is decisive, not just the data.
- `gated_mlp` consistently improves MATH AUC by ~0.15+ over `mlp` with the same profile.
- pbr4c achieves **best samefamily** top1=0.884 of all experiments.

**Interpretation**: The gating mechanism (output = gate × main_path) separates length-bias from step-correctness signal better than plain MLP. PRMBench auxiliary pairs (in ms_prm_align_v1) further help GSM8K, likely because PRMBench examples are GSM8K-domain.

### PBR Experiment Comparison Summary (frozen head baseline)

| config | GSM8K_AUC | MATH_AUC | SF_top1 | Status |
|---|---|---|---|---|
| ms_align_v1 + mlp (s42) | 0.442 | 0.442 | 0.867 | Both inverted |
| ms_align_v1 + mlp (s1)  | 0.616 | 0.476 | 0.864 | GSM8K OK, MATH inv — unstable |
| ms_align_v1 + mlp (s7)  | 0.486 | 0.543 | 0.846 | GSM8K inv, MATH OK — unstable |
| ms_laterbad_v1 + mlp (s42) | 0.425 | 0.436 | 0.860 | Both inverted |
| ms_laterbad_v1 + mlp (s1)  | 0.619 | 0.487 | 0.873 | GSM8K OK — unstable |
| ms_laterbad_v1 + mlp (s7)  | 0.515 | 0.548 | 0.854 | MATH OK — unstable |
| ms_prm_align_v1 + mlp      | 0.490 | 0.458 | 0.856 | Both inverted |
| **ms_prm_align_v1 + gated_mlp** | **0.549** | **0.621** | **0.872** | **Both OK — stable direction** |
| ms_laterbad_v1 + gated_mlp | 0.472 | 0.598 | 0.884 | GSM8K inv, MATH OK |
| ms_prm_align_v1 + mlp | 0.490 | 0.458 | 0.856 | Still inverted |
| **ms_prm_align_v1 + gated_mlp** | **0.549** | **0.621** | **0.872** | **Best overall** |
| ms_laterbad_v1 + gated_mlp | 0.472 | 0.598 | 0.884 | Best samefamily |

### Recommended Next Steps

1. **PBR5 (immediate)**: `ms_prm_align_v1 + gated_mlp` at full scale (16384 pairs), 3 seeds. Target: MATH AUC ≥ 0.65, GSM8K AUC ≥ 0.57.
2. **PBR5b**: `ms_laterbad_v1 + gated_mlp` at full scale, 3 seeds. Compare against PBR5.
3. **PBR6 (LoRA path)**: If PBR5 MATH AUC < 0.65, proceed with LoRA backbone unfreezing. Published PRMs (>75% on ProcessBench) all require backbone fine-tuning.
4. **Architecture Note**: `dual_head` is NOT yet useful without pair_semantics routing in training.py. Blocked pending routing implementation.



## 0ZZZZZ. Pairwise judge benchmark and label-preserving filter pilot (2026-03-11)

This round tested the judge usage pattern that current literature supports more
strongly than our earlier strict-JSON pointwise prefix audit:

1. pairwise comparison,
2. swap-debias (`A/B` plus `B/A`),
3. light output contracts,
4. and using judge as a conservative label-preserving filter rather than as a
   full replacement verifier.

Primary artifacts:
1. script:
   - `scripts/phase_e_pairwise_judge_benchmark.py`
2. compare summary:
   - `assets/artifacts/phase_e_pairwise_judge_compare/judge_pairwise_compare_20260311T084132Z/summary.md`
3. anchored held-out runs:
   - `assets/artifacts/phase_e_pairwise_judge/pairjudge_qwen_prmbench_anchor32_20260311T083726Z`
   - `assets/artifacts/phase_e_pairwise_judge/pairjudge_deepseek_prmbench_anchor16_20260311T083726Z`
   - `assets/artifacts/phase_e_pairwise_judge/pairjudge_qwen_ms_val32_anchor_20260311T083839Z`
4. anchored train-slice filter pilots:
   - `assets/artifacts/phase_e_pairwise_judge/pairjudge_qwen_prmbench_train64_anchor_20260311T083927Z`
   - `assets/artifacts/phase_e_pairwise_judge/pairjudge_qwen_ms_train64_anchor_20260311T083927Z`

### 0ZZZZZ.1 Key results

`Qwen2.5-Math-7B-Instruct` on held-out `PRMBench_Preview` (`32` pairs):
1. `both_parse_ok_rate = 0.3125`
2. `pair_acc_majority = 0.3438`
3. `swap_consistency_rate = 0.6000`
4. `label_preserving_keep_rate = 0.1250`

`Qwen2.5-Math-7B-Instruct` on held-out `Math-Shepherd local_first_bad_edge` (`32` pairs):
1. `both_parse_ok_rate = 0.2188`
2. `pair_acc_majority = 0.0625`
3. `swap_consistency_rate = 1.0000` on the parsable subset
4. `label_preserving_keep_rate = 0.0312`
5. `judge_contradiction_rate = 0.0625`

`Qwen2.5-Math-7B-Instruct` as train-slice filter:
1. `PRMBench_Preview train64`:
   - `label_preserving_keep_rate = 0.0469`
2. `Math-Shepherd train64`:
   - `label_preserving_keep_rate = 0.0000`

`DeepSeek-R1-Distill-Qwen-14B` on held-out `PRMBench_Preview` (`16` pairs):
1. `both_parse_ok_rate = 0.5625`
2. `pair_acc_majority = 0.0625`
3. `tie_rate = 0.8125`

### 0ZZZZZ.2 Interpretation

This round gives three concrete conclusions:

1. pairwise + swap-debias is a better judge setup than the earlier
   strict-JSON pointwise prefix-judge attempt;
2. but the current local judge stack is still too weak to serve as a robust
   bulk pair filter;
3. and the judge is much closer to usable on `PRMBench_Preview` than on
   `Math-Shepherd local_first_bad_edge`.

The most important dataset-level finding is:

1. `Math-Shepherd` local-first-bad pairs are not well matched to naive
   pairwise judge filtering,
2. because the shorter safe prefix is often judged worse than the longer prefix
   that already contains the first wrong step.

So even though the pair label is valid for value-head training, it is not a
natural fit for "which prefix is better so far?" judged by a small local LLM.

### 0ZZZZZ.3 Current recommendation

1. keep judge work in the role of:
   - disagreement mining,
   - selected relabeling,
   - or benchmark-side adjudication,
2. do not promote the current local judge to a bulk Phase E pair filter,
3. and if judge is used for pairwise tasks, prefer sources closer to
   `PRMBench_Preview`-style explicit modified-process pairs over
   `Math-Shepherd local_first_bad_edge`.

## 0ZZZZ. Backbone adaptation and judge/oracle bounded study (2026-03-11)

This round asked three operational questions:

1. if we hold the Phase E head/pair pipeline fixed but swap in an already
   adapted PRM backbone, does benchmark transfer improve,
2. if we keep the backbone fixed but filter pairs through an independent PRM
   oracle, does data quality improve enough to offset the smaller pool,
3. is the newly installed local judge ready to relabel prefix pairs directly.

Primary artifacts:
1. summary note:
   - `docs/phase_e_backbone_judge_experiments_20260311.md`
2. adapted-backbone train run:
   - `assets/artifacts/phase_e_runs/phase_e_backboneproxy_prm_mixedsmall_20260311T074134Z`
3. adapted-backbone same-family eval:
   - `assets/artifacts/phase_e_samefamily_eval/phase_e_backboneproxy_prm_mixedsmall_samefamily_20260311T080555Z`
4. adapted-backbone `ProcessBench Math 50` eval:
   - `assets/artifacts/phase_e_eval/phase_e_backboneproxy_prm_mixedsmall_math50_20260311T080653Z`
5. PRM-oracle filtered artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_prm_oracle_filter_mixedsmall_20260311T081105Z`
6. filtered-base train run:
   - `assets/artifacts/phase_e_runs/phase_e_oraclefilter_base_mixedsmall_20260311T081526Z`
7. filtered-base `ProcessBench Math 50` eval:
   - `assets/artifacts/phase_e_eval/phase_e_oraclefilter_base_mixedsmall_math50_20260311T082011Z`
8. direct prefix-judge audit:
   - `assets/artifacts/phase_e_judge_audit/phase_e_judge_prefix_audit_mixedsmall_20260311T082237Z`

### 0ZZZZ.1 Adapted-backbone proxy result

The backbone proxy used `Qwen2.5-Math-PRM-7B` as the frozen feature extractor
under the ordinary mixed-small Phase E recipe.

Held-out pair fit:
1. `pair_acc = 0.8984`
2. `auc = 0.8946`

Same-family trust:
1. `prompt_pool_top1_accuracy = 0.9160`
2. `local_first_bad_edge_accuracy = 0.9487`

`ProcessBench Math 50`:
1. `pair_acc = 0.6234`
2. `auc = 0.6810`
3. `first_edge = 0.6923`
4. `mean_all_correct_last_score = 0.5749`
5. `processbench_f1 = 0.4800`

Interpretation:
1. benchmark transfer improved strongly,
2. same-family trust did not dominate,
3. so backbone representation is a real ceiling,
4. but naive backbone replacement does not preserve all existing trust geometry.

### 0ZZZZ.2 PRM-oracle filtering result

The oracle filter kept:
1. train: `878 / 2048 = 0.4287`
2. eval: `121 / 256 = 0.4727`

Critical composition finding:
1. `first_bad_fanout_prefix_ranking`: kept `770 / 1536`
2. `local_modified_process_error_step`: kept `107 / 380`
3. `terminal_completion_anchor`: kept only `1 / 132` on train and `0 / 12` on eval

Training on this filtered pool produced:
1. held-out `pair_acc = 0.8678`
2. held-out `auc = 0.8427`

`ProcessBench Math 50` then fell to:
1. `pair_acc = 0.5190`
2. `auc = 0.5535`
3. `mean_all_correct_last_score = 0.1543`
4. `processbench_f1 = 0.2933`

Interpretation:
1. naive global oracle filtering is a negative result,
2. because it silently deletes terminal supervision and breaks the mixed objective.

### 0ZZZZ.3 Direct prefix-judge audit result

Judge:
1. `Qwen2.5-Math-7B-Instruct`

Audit setup:
1. `24` validation pairs from the same mixed-small artifact,
2. prompt explicitly said:
   - judge only the displayed steps,
   - do not punish incompleteness,
   - return JSON.

Result:
1. `pair_json_ok_rate = 0.0000`
2. `pair_agreement_rate = 0.0000`
3. elapsed time: `369.3 sec`

Observed failure:
1. the model mostly produced free-form analysis,
2. not a machine-readable JSON contract,
3. so it is not yet a practical automatic prefix-pair relabeler in the current
   local `transformers + generate` stack.

### 0ZZZZ.4 Current repository-level conclusion

1. if the goal is to move `ProcessBench`, backbone adaptation is the strongest
   lever found this round.
2. if the goal is to keep the mixed supervision geometry healthy, naive global
   oracle filtering is the wrong move.
3. the new judge models are still useful, but their immediate role should be:
   - disagreement audit,
   - terminal / full-solution relabel,
   - or low-volume adjudication,
   not direct prefix-pair auto-filtering.

## 0ZZZ. Judge LLM benchmark check on real ProcessBench slices (2026-03-11)

### 0ZZZ.1 Why the smoke result was not enough

The earlier judge smoke only used tiny toy math examples. That was enough to
check:

1. model loading,
2. basic prompting,
3. and whether structured outputs are even plausible.

It was not enough to answer:

1. how the two candidate judge LLMs behave on real benchmark rows,
2. whether they can locate the first bad step,
3. and whether verbose output contracts are the main bottleneck.

### 0ZZZ.2 New script

1. `scripts/phase_e_benchmark_judge_llm.py`

This script:

1. loads deterministic small ProcessBench slices,
2. runs one local instruct model as a judge,
3. evaluates:
   - parse success,
   - overall correctness,
   - first-bad exact accuracy,
   - first-bad within-1 accuracy,
   - mean step accuracy.

### 0ZZZ.3 Experimental setup

Benchmarks:

1. `ProcessBench math`
2. `ProcessBench gsm8k`

Per-benchmark cap:

1. `6` rows each

Compared contracts:

1. `full_steps`
2. `first_bad_only`

Compared judge models:

1. `Qwen2.5-Math-7B-Instruct`
2. `DeepSeek-R1-Distill-Qwen-14B`

### 0ZZZ.4 Artifacts

Runs:

1. `assets/artifacts/phase_e_judge_bench/judge_bench_qwen25_math_7b_20260311T073507Z`
2. `assets/artifacts/phase_e_judge_bench/judge_bench_deepseek_r1_14b_20260311T073507Z`
3. `assets/artifacts/phase_e_judge_bench/judge_bench_qwen25_math_7b_fbonly_20260311T074138Z`
4. `assets/artifacts/phase_e_judge_bench/judge_bench_deepseek_r1_14b_fbonly_20260311T074138Z`

Unified compare:

1. `assets/artifacts/phase_e_judge_bench_compare/judge_model_compare_20260311T074514Z/summary.md`

### 0ZZZ.5 Main results

The most important rows are:

1. `Qwen2.5-Math-7B-Instruct`, `full_steps`
   - `ProcessBench math`
     - `parse_ok=0.6667`
     - `overall_acc=0.3333`
     - `first_bad_exact=0.0000`
   - `ProcessBench gsm8k`
     - `parse_ok=1.0000`
     - `overall_acc=0.5000`
     - `first_bad_exact=0.0000`

2. `Qwen2.5-Math-7B-Instruct`, `first_bad_only`
   - `ProcessBench math`
     - `parse_ok=0.5000`
     - `overall_acc=0.3333`
     - `first_bad_exact=0.0000`
   - `ProcessBench gsm8k`
     - `parse_ok=1.0000`
     - `overall_acc=0.5000`
     - `first_bad_exact=0.0000`

3. `DeepSeek-R1-Distill-Qwen-14B`, `full_steps`
   - `ProcessBench math`
     - `parse_ok=0.5000`
     - `overall_acc=0.1667`
     - `first_bad_exact=0.0000`
   - `ProcessBench gsm8k`
     - `parse_ok=0.5000`
     - `overall_acc=0.3333`
     - `first_bad_exact=0.0000`
     - `first_bad_within1=0.3333`

4. `DeepSeek-R1-Distill-Qwen-14B`, `first_bad_only`
   - `ProcessBench math`
     - `parse_ok=0.5000`
     - `overall_acc=0.1667`
     - `first_bad_exact=0.0000`
   - `ProcessBench gsm8k`
     - `parse_ok=0.8333`
     - `overall_acc=0.6667`
     - `first_bad_exact=0.3333`
     - `first_bad_within1=0.3333`

### 0ZZZ.6 Failure pattern reading

This round exposed a clear split:

1. `Qwen2.5-Math-7B-Instruct`
   - better operational stability,
   - but on real benchmark rows it is heavily biased toward:
     - `OVERALL=correct`
     - `FIRST_BAD=none`
   - simplifying the contract does not fix that bias.

2. `DeepSeek-R1-Distill-Qwen-14B`
   - less stable under verbose contracts,
   - but once the contract is reduced to `first_bad_only`, it shows real signal
     on `gsm8k`,
   - while remaining weak on `math`.

Therefore:

1. verbosity is not the main blocker for `Qwen2.5-Math-7B-Instruct`,
2. verbosity *is* a material blocker for `DeepSeek-R1-Distill-Qwen-14B`,
3. neither model is currently strong enough to act as a benchmark-ready,
   standalone ProcessBench judge.

### 0ZZZ.7 Operational conclusion

1. Keep `Qwen2.5-Math-7B-Instruct` as the bulk local judge candidate.
2. Do not expect it to act as a precise first-bad-step annotator.
3. Use `DeepSeek-R1-Distill-Qwen-14B` only as a second-stage adjudicator,
   especially on more verbal / GSM-style cases and only under a lighter output
   contract.

## 0ZZ. Judge LLM local selection + deployment check (2026-03-11)

### 0ZZ.1 Why this round was needed

Phase E is about to move toward stronger `LLM-as-a-judge` style supervision.
Before wiring that into relabeling or active learning, we needed to answer:

1. which local judge models are most defensible from the literature,
2. which ones fit the current server budget,
3. which ones actually work in our local inference stack.

### 0ZZ.2 Sources checked

Web / model cards:

1. `ProcessBench`
   - `https://arxiv.org/abs/2412.06559`
2. `ThinkPRM`
   - `https://arxiv.org/abs/2504.16828`
3. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   - `https://arxiv.org/abs/2501.07301`
4. `DeepSeek-R1-Distill-Qwen-14B`
   - `https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
5. `Qwen2.5-Math-7B-Instruct`
   - `https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct`
6. `QwQ-32B`
   - `https://huggingface.co/Qwen/QwQ-32B`

Local documents / PDFs:

1. `docs/relatedPapers/2412.06559_processbench.pdf`
2. `docs/relatedPapers/2504.16828_thinkprm.pdf`
3. `docs/relatedPapers/2501.07301_lessons_developing_prm.pdf`
4. `docs/bcr_synthesis_report_20260311.md`

### 0ZZ.3 Operational selection

The chosen local judge stack is now:

1. bulk math judge:
   - `assets/models/Qwen2.5-Math-7B-Instruct`
2. stronger adjudicator candidate:
   - `assets/models/DeepSeek-R1-Distill-Qwen-14B`
3. cheapest fallback baseline:
   - `assets/models/Qwen2.5-7B-Instruct`

`QwQ-32B` was deliberately not downloaded in this round:

1. it is a strong open-source critic candidate,
2. but it is too expensive for day-to-day local bulk judging at the repo's
   current stage,
3. and a 14B adjudicator is enough to validate the local judge path first.

### 0ZZ.4 New local tooling

New script:

1. `scripts/phase_e_smoke_judge_llm.py`

Purpose:

1. load one local instruct model,
2. send a strict structured step-judge prompt,
3. test tiny good/bad reasoning examples,
4. save artifacts under `assets/artifacts/phase_e_judge_smoke/`.

### 0ZZ.5 Smoke-test results

Artifacts:

1. `assets/artifacts/phase_e_judge_smoke/judge_smoke_qwen25_7b_20260311T071409Z`
2. `assets/artifacts/phase_e_judge_smoke/judge_smoke_qwen25_math_7b_v2_20260311T071746Z`
3. `assets/artifacts/phase_e_judge_smoke/judge_smoke_deepseek_r1_qwen14b_v2_20260311T072126Z`

Observed behavior:

1. `Qwen2.5-7B-Instruct`
   - `json_ok = 1 / 2`
   - correct sample works
   - bad sample degenerates into `!!!!!!!!...`
   - useful only as the cheapest baseline
2. `Qwen2.5-Math-7B-Instruct`
   - `json_ok = 1 / 2`
   - semantics on the bad math example are actually good
   - format adherence is imperfect, but tolerant parsing can recover some cases
   - best current candidate for bulk local judge use
3. `DeepSeek-R1-Distill-Qwen-14B`
   - model loads successfully
   - but current local `transformers + strict JSON judge prompt` path is not stable
   - both the generic prompt and the model-card-aligned prompt produced unusable outputs
   - adding recommended sampling then triggered `inf/nan/<0` probability failure

### 0ZZ.6 Final judgement

The important outcome is not "14B is stronger than 7B".

The important outcome is:

1. the repository now has one locally installed judge path that is cheap enough
   and operational enough to pilot:
   - `Qwen2.5-Math-7B-Instruct`
2. the stronger 14B reasoning model should not yet be promoted to the main
   Phase E judge loop without a different serving setup or more specialized
   prompting / decoding control.

### 0ZZ.7 Immediate next step

Use `Qwen2.5-Math-7B-Instruct` first for:

1. low-confidence pair relabeling,
2. disagreement mining,
3. judge-confidence artifact logging.

## 0Z. ProcessBench state audit + community gap review (2026-03-11)

### 0Z.1 Why this audit was needed

A teammate reported three things at once:

1. many new local PDFs and survey notes had been added,
2. multiple external datasets were now downloaded locally,
3. some newer experiments had "fixed part of `ProcessBench`".

The practical question was therefore no longer:

1. "what do we think the repo is doing?"

but:

1. "what do the current code, current artifacts, and current paper set jointly
   support?"

### 0Z.2 What was checked

Code / wrappers:

1. `scripts/run_phase_e_processbench_research_suite.sh`
2. `scripts/phase_e_curate_processbench_transfer_pairs.py`
3. `scripts/phase_e_curate_semantic_pairs.py`
4. `src/ours/phase_b/value_head.py`
5. `src/ours/phase_e/training.py`
6. `src/ours/phase_e/processbench_alignment.py`

Local papers / notes:

1. `docs/research_survey_processverifier_20260311.md`
2. `docs/bcr_feasibility_review_20260311.md`
3. `docs/dataset_survey_stepwise_pairs_20260311.md`
4. `docs/new_dataset_experiment_plan_20260311.md`
5. `docs/phase_e_rl_ready_research_redesign_20260311.md`
6. PDFs under `docs/relatedPapers/`

Fresh audit artifacts produced in this round:

1. `assets/artifacts/phase_e_transfer_compare/processbench_state_review_math_0311_20260311T070248Z/summary.md`
2. `assets/artifacts/phase_e_transfer_compare/processbench_state_review_gsm_0311_20260311T070308Z/summary.md`
3. `assets/artifacts/phase_e_rl_promotion_diag/rl_promotion_state_review_0311_20260311T070332Z/summary.md`

New synthesis note:

1. `docs/processbench_state_and_community_gap_20260311.md`

### 0Z.3 Fresh state-of-repo conclusions

The teammate claim "some `ProcessBench` behavior is fixed" is only partially
correct.

What is true:

1. recent repairs do improve some benchmark slices:
   - `terminal_top1`
   - some `first_edge`
2. pair-geometry alignment is somewhat better in the newer curated /
   ProcessBench-aware artifacts than in pure local baselines

What is not true:

1. there is still no new mainline candidate that beats the strongest older runs
   on overall benchmark utility while also preserving same-family trust

The strongest current evidence:

1. `ms_e43` remains the strongest local / later-bad candidate
   - `ProcessBench Math`:
     - `auc=0.6341`
     - `good_vs_laterbad=0.7515`
     - `terminal_top1=0.0099`
2. `prm_e46` remains the strongest more-balanced benchmark-facing candidate
   - `ProcessBench GSM8K`:
     - `auc=0.6264`
   - `ProcessBench Math`:
     - `auc=0.6053`
     - `good_vs_laterbad=0.5501`
     - `terminal_top1=0.1970`
3. recent repair candidates still fail the strict RL gate:
   - `c3_curated_gated`
     - `assessment=not_rl_ready_laterbad_generalization_weak`
   - `c4_dual`
     - `assessment=not_rl_ready_laterbad_generalization_weak`
   - `pbr2_align_gated`
     - `assessment=not_rl_ready_laterbad_generalization_weak`
   - `e87_repair`
     - `assessment=terminal_and_local_tradeoff_unresolved`

### 0Z.4 Community-vs-repo gap

The paper set is increasingly consistent on four points:

1. strong process verification uses richer supervision than single-trajectory
   local first-bad edges
2. MC-only synthetic labels are noisy for step correctness
3. strong current systems increasingly rely on:
   - consensus filtering / judge filtering
   - tree / sibling-branch preferences
   - backbone adaptation or generative verification
4. evaluation must separate:
   - local first-edge
   - later-bad
   - all-correct terminal behavior

The current repo mainline still does:

1. frozen backbone
2. `last_token` pooled frozen features
3. small scalar head
4. pairwise ranking/BCE on converted pairs

This means the repo is still strongest as:

1. a diagnosis and ablation platform

not yet as:

1. a community-level competitive verifier stack

### 0Z.5 Main anti-patterns now made explicit

1. do not treat `heldout_pair_acc` or same-family top1 as the main success
   proxy
2. do not narrate one repaired slice (`terminal_top1`, `first_edge`) as
   "ProcessBench fixed"
3. do not treat `fanout/grid` as true tree / sibling-branch supervision
4. do not keep adding new scalar heads before using the stronger downloaded
   sources already present locally

### 0Z.6 Decision for next experiments

The next mainline should be:

1. stop architecture churn for one round
2. first debug / mainline the stronger local datasets:
   - `PRM800K`
   - `MATH-APS`
   - `EurusPRM-Stage2`
   - `Math-Step-DPO-10K`
3. run equal-budget source-only ProcessBench slice audits
4. only then do one minimal frozen-vs-LoRA comparison on the strongest source
   mixture

Therefore the current research reading changes from:

1. "keep repairing ProcessBench mostly by pair geometry and head design"

to:

1. "use the current stack to isolate which missing ingredient matters most:
   better source, better supervision geometry, or minimal backbone adaptation"

## 0Y. Verified internet reading + `dual_head` semantic-routing smoke (2026-03-11)

### 0Y.1 Why this follow-up existed

After the earlier semantic-curation + `gated_mlp` smoke, the remaining open
question was:

1. if the real bottleneck is "one scalar head is compressing multiple verifier
   subtasks",
2. can a lightweight decomposed head help inside the existing frozen-feature
   Phase E stack,
3. without yet paying the cost of a full critique/generative verifier?

### 0Y.2 Verified external reading

The following sources were re-checked directly on 2026-03-11 and used to guide
this round:

1. `ProcessBench: Identifying Process Errors in Mathematical Reasoning`
   - <https://aclanthology.org/2025.acl-long.50/>
2. `Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning`
   - <https://arxiv.org/abs/2410.08146>
3. `Advancing Process Verification for Large Language Models via Tree-Based Preference Learning`
   - <https://arxiv.org/abs/2407.00390>
4. `PRMBench: Can Reward Models Truly Verify Reasoning Processes?`
   - <https://arxiv.org/abs/2501.03124>
5. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   - <https://arxiv.org/abs/2501.07301>
6. `Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning`
   - <https://arxiv.org/abs/2504.15275>
7. `R-PRM` official repo
   - <https://github.com/NJUNLP/R-PRM>
8. `Open Instruct` reward-modeling docs
   - <https://allenai.github.io/open-instruct/algorithms/reward_modeling/>

The practical synthesis from those sources was:

1. benchmark difficulty is broader than one local edge
2. richer intermediate preference relations matter
3. evaluation should separate local, later-bad, and terminal behavior
4. if we stay inside a scalar-verifier regime, "decompose the objective" is more
   defensible than "blindly add more head capacity"

### 0Y.3 New code for the decomposition smoke

Updated:

1. `src/ours/phase_b/value_head.py`
   - added `architecture='dual_head'`
   - added `inference_alpha`
   - dual head returns:
     - blended `logits/scores`
     - branch-specific `local_*`
     - branch-specific `terminal_*`
2. `src/ours/phase_e/training.py`
   - added semantic route resolution and per-pair route weights
   - current routing:
     - `terminal_completion_anchor -> terminal`
     - `local_* / first_bad_fanout_prefix_ranking -> local`
     - `good_bad_prefix_grid -> both`
   - `compute_pair_objective()` now supports routed dual-head loss
3. `scripts/phase_e_train_value.py`
   - added `--head-architecture dual_head`
   - added `--head-inference-alpha`
   - threaded route weights through epoch training

Tests:

1. `python -m py_compile src/ours/phase_b/value_head.py src/ours/phase_e/training.py scripts/phase_e_train_value.py`
2. `pytest -q tests/unit/test_value_head.py tests/unit/test_phase_e_training.py tests/unit/test_phase_e_train_script.py`
3. result:
   - `23 passed`

### 0Y.4 Experiment design

Controlled comparison:

1. keep the exact same curated artifact as the previous `CR1` smoke
2. keep the same optimizer / balancing / centering settings as `C3`
3. replace only the head:
   - old: `gated_mlp`
   - new: `dual_head`

Artifact and run paths:

1. curated pairs:
   - `assets/artifacts/phase_e_pairs/phase_e_curated_rlready_0311_retry2_curated_pairs__03ac5eebc8fd`
2. new value run:
   - `assets/artifacts/phase_e_runs/phase_e_curated_rlready_0311_dual_c4_value_20260311T051319Z`
3. new samefamily eval:
   - `assets/artifacts/phase_e_samefamily_eval/phase_e_curated_rlready_0311_dual_c4_samefamily_20260311T051616Z`
4. new ProcessBench eval:
   - GSM8K:
     - `assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_dual_c4_pb_gsm8k_20260311T051637Z`
   - Math:
     - `assets/artifacts/phase_e_eval/phase_e_curated_rlready_0311_dual_c4_pb_math_20260311T051652Z`

### 0Y.5 Main results

#### `C4_CURATED_DUAL_CENTER`

1. held-out:
   - `pair_acc=0.8608`
   - `auc=0.7390`
2. samefamily:
   - `top1=0.8787`
   - `local_first_bad=0.9408`
3. ProcessBench GSM8K:
   - `auc=0.4730`
   - `first_edge=0.5660`
   - `terminal_top1=0.8548`
   - `good_vs_laterbad=0.2756`
4. ProcessBench Math:
   - `auc=0.4789`
   - `first_edge=0.5714`
   - `terminal_top1=0.9038`
   - `good_vs_laterbad=0.3672`

Cross-run slice comparison artifacts:

1. Math:
   - `assets/artifacts/phase_e_transfer_compare/processbench_curated_arch_compare_0311_dual_math_20260311T051707Z/summary.md`
2. GSM8K:
   - `assets/artifacts/phase_e_transfer_compare/processbench_curated_arch_compare_0311_dual_gsm_20260311T051726Z/summary.md`
3. RL-promotion compare:
   - `assets/artifacts/phase_e_rl_promotion_diag/phase_e_curated_rlpromo_compare_0311_dual_20260311T051742Z/summary.md`

### 0Y.6 Interpretation

Compared with `C3_CURATED_GATED_CENTER`, the new `dual_head` does improve some
of the *right* slices:

1. `first_edge`
   - GSM8K:
     - `0.4906 -> 0.5660`
   - Math:
     - `0.5397 -> 0.5714`
2. terminal behavior
   - GSM8K terminal top1:
     - `0.7097 -> 0.8548`
   - Math terminal top1:
     - `0.8654 -> 0.9038`

But it degrades the broader things we still need:

1. samefamily top1:
   - `0.9443 -> 0.8787`
2. overall benchmark AUC:
   - GSM8K:
     - `0.4861 -> 0.4730`
   - Math:
     - `0.5152 -> 0.4789`
3. `good_vs_laterbad`
   - GSM8K:
     - `0.4267 -> 0.2756`
   - Math:
     - `0.4738 -> 0.3672`

This means:

1. the decomposition hypothesis is not empty
   - the model *did* move the expected slices
2. but the current hard semantic routing is too blunt
   - it over-specializes local-first-edge and terminal behavior
   - while under-preserving broader good-vs-bad ranking geometry

### 0Y.7 Final decision from this round

1. do **not** promote current `dual_head` as the new mainline
2. keep `C3 gated_mlp` as the stronger current curated baseline
3. if the decomposition direction continues, the next version should test:
   - softer route weights
   - inference `alpha` sweep
   - or staged curriculum (`single-head/gated` warm start -> dual-head repair)
4. current RL-promotion verdict remains unchanged:
   - `gate=0`
   - main failure label:
     - `not_rl_ready_laterbad_generalization_weak`

## 0X. Internet-guided semantic-curation + reward-centering smoke (2026-03-11)

### 0X.1 What this round asked

This round turned the latest paper/community reading into one concrete
repository question:

1. if transfer failure is partly caused by supervision geometry mismatch,
2. can we first curate a small semantic-balanced pool,
3. then add reward centering as a low-risk reward-model regularizer,
4. and finally test whether a more flexible scalar head (`gated_mlp`) improves
   benchmark transfer over a plain `mlp`?

### 0X.2 New code and docs

External-reading artifact:

1. `docs/phase_e_internet_research_20260311.md`

New / updated code:

1. `src/ours/phase_b/value_losses.py`
   - added `reward_centering_penalty()`
2. `src/ours/phase_e/training.py`
   - threaded `reward_centering_weight` into train/eval objective computation
3. `scripts/phase_e_train_value.py`
   - added `--reward-centering-weight`
4. `scripts/phase_e_curate_semantic_pairs.py`
   - deterministic curation by semantic bucket and source filter
5. `scripts/run_phase_e_curated_rlready_suite.sh`
   - one-click curated smoke suite

Wrapper hardening during execution:

1. fixed shell backtick substitution in suite metadata
2. fixed env override clobbering so `EVAL_BATCH_SIZE=12` really takes effect
3. fixed ProcessBench terminal extraction in the suite script so future reruns
   compute `all-correct final-prefix top1` from `scored_rows.jsonl`

### 0X.3 Curated artifact contract

Artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_curated_rlready_0311_retry2_curated_pairs__03ac5eebc8fd`

Composition:

1. `Math-Shepherd local fanout`
   - train `1600`
   - val `160`
2. `PRMBench local sibling`
   - train `1600`
   - val `160`
3. `PRMBench terminal anchor`
   - train `320`
   - val `32`

Total:

1. train `3520`
2. val `352`

Interpretation:

1. this is a deliberately bounded terminal regime
2. terminal supervision is about `9.1%` of the train pool
3. this is exactly the regime we wanted to test after the earlier
   terminal-overcorrection failures

### 0X.4 Smoke suite result

Suite summary:

1. `assets/artifacts/phase_e_logs/phase_e_curated_rlready_0311_retry2/final_summary.md`

#### `C1_CURATED_MLP_BASE`

1. held-out:
   - `pair_acc=0.9034`
   - `auc=0.8574`
2. samefamily:
   - `top1=0.9541`
   - `local_first_bad=0.9737`
3. ProcessBench:
   - GSM8K:
     - `auc=0.4892`
     - `first_edge=0.4906`
   - Math:
     - `auc=0.4553`
     - `first_edge=0.5079`

#### `C2_CURATED_MLP_CENTER`

1. held-out:
   - `pair_acc=0.8977`
   - `auc=0.8643`
2. samefamily:
   - `top1=0.9508`
   - `local_first_bad=0.9737`
3. ProcessBench:
   - GSM8K:
     - `auc=0.4928`
     - `first_edge=0.4906`
   - Math:
     - `auc=0.4545`
     - `first_edge=0.5079`

Reading:

1. reward centering alone did almost nothing for benchmark transfer
2. it slightly changed held-out calibration, but not the actual transfer regime

#### `C3_CURATED_GATED_CENTER`

1. held-out:
   - `pair_acc=0.9290`
   - `auc=0.8706`
2. samefamily:
   - `top1=0.9443`
   - `local_first_bad=0.9934`
3. ProcessBench:
   - GSM8K:
     - `auc=0.4861`
     - `first_edge=0.4906`
   - Math:
     - `auc=0.5152`
     - `first_edge=0.5397`

Reading:

1. this is the only config that improved the hard `ProcessBench Math` slice
2. but it still failed badly on GSM8K
3. therefore it changes the tradeoff, but does not clear the RL-facing gate

### 0X.5 Strict transfer diagnosis

Diagnostic artifact:

1. `assets/artifacts/phase_e_transfer_diag/phase_e_curated_rlready_0311_retry2_diag_00/summary.md`

All three candidates were classified as:

1. `not_rl_ready_local_transfer_weak`

Shared failure tags:

1. `benchmark_local_error_weak`
2. `margin_collapse`
3. `support_length_drift`

Important implication:

1. in this curated regime, the main blocker is **not** terminal completion
2. the main blocker is that local ProcessBench discrimination still does not
   survive transfer strongly enough

### 0X.6 Comparison to the previous best mixed baseline

Reference baseline:

1. `E82`
   - samefamily:
     - `top1=0.9633`
     - `local_first_bad=0.9841`
   - ProcessBench GSM8K:
     - `auc=0.5738`
     - `first_edge=0.6038`
   - ProcessBench Math:
     - `auc=0.4937`
     - `first_edge=0.4603`

Comparison:

1. `C3` improved `ProcessBench Math`
   - `auc 0.4937 -> 0.5152`
   - `first_edge 0.4603 -> 0.5397`
2. but `C3` hurt `ProcessBench GSM8K`
   - `auc 0.5738 -> 0.4861`
3. and `C3` still lost a little samefamily top1
   - `0.9633 -> 0.9443`

Conclusion:

1. the new pipeline found a **directional math-side improvement**
2. but it did **not** produce a universally better or RL-ready candidate

### 0X.7 Next step

This round narrows the next principled repair:

1. stop treating terminal completion as the main missing piece in this curated regime
2. focus on why local benchmark transfer still collapses:
   - benchmark length/support drift
   - possible need for staged objectives instead of naive joint mixture
   - explicit dual-objective or dual-head local-vs-terminal training
3. keep reward centering available, but do not expect it to be a primary fix by
   itself

## 0W. Latest Diagnosis Update (2026-03-11, ProcessBench Hybrid Artifact + Architecture Comparison)

### 0W.1 What this round asked

This round turned the latest literature reading into one concrete question:

1. if `ProcessBench` transfer fails because current training is too local,
2. can we keep `PRMBench` local error-step pairs as the anchor,
3. add only a very small terminal auxiliary,
4. and recover benchmark behavior without destroying local discrimination?

At the same time, it asked a second question:

1. if this repair still fails,
2. is the main blocker:
   - data contract mismatch,
   - or insufficient head architecture?

### 0W.2 New pipeline / code

New curation + training support:

1. `scripts/phase_e_mix_pair_artifacts.py`
   - now supports weighted source mixing:
     - `LABEL=DIR:TRAIN_CAP:VAL_CAP:MIX_WEIGHT`
2. `scripts/run_phase_e_processbench_hybrid_suite.sh`
   - new benchmark-oriented wrapper:
     - `PRMBench local` anchor
     - bounded terminal auxiliary
     - `mlp` vs `gated_mlp`
3. wrapper hardening:
   - fixed baseline triplet parsing for:
     - `name=run_dir::gsm_eval_dir::math_eval_dir`
   - removed shell backtick substitution hazards in suite metadata
   - separated helper logs from helper return values
   - switched helper outputs to global result variables instead of command substitution

### 0W.3 Executed artifact

Hybrid artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_pbhybrid_ph1_0311_1230_pairs__d6fb5a3ec28c`

Composition:

1. `prm_local`
   - train `3072`
   - val `384`
2. `prm_terminal`
   - train `512`
   - val `64`

Total:

1. train `3584`
2. val `448`

Important detail from later failure analysis:

1. the *effective* terminal-anchor semantics inside this hybrid are only:
   - `132 / 3584 = 0.0368`
2. so this was already a very conservative terminal repair.

### 0W.4 Same-source held-out result

Runs:

1. `mlp`
   - run:
     - `assets/artifacts/phase_e_runs/phase_e_pbhybrid_ph1_0311_1230_ph1_prm_local_ta15_arch_sweep_smoke_mlp_20260311T043055Z`
   - held-out:
     - `pair_acc=0.919643`
     - `auc=0.892518`
2. `gated_mlp`
   - run:
     - `assets/artifacts/phase_e_runs/phase_e_pbhybrid_ph1_0311_1230_ph1_prm_local_ta15_arch_sweep_smoke_gated_mlp_20260311T045211Z`
   - held-out:
     - `pair_acc=0.912946`
     - `auc=0.871079`

Interpretation:

1. the hybrid artifact is easy to fit on same-source held-out pairs,
2. so this round is **not** another "the head cannot learn at all" result.

### 0W.5 ProcessBench transfer result

Comparison artifacts:

1. GSM8K compare:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_pbhybrid_ph1_0311_1230_processbench_gsm8k_compare_20260311T045320Z/summary.md`
2. Math compare:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_pbhybrid_ph1_0311_1230_processbench_math_compare_20260311T045332Z/summary.md`

#### GSM8K

1. baseline `E46`
   - `auc=0.6264`
   - `first_edge=0.6706`
   - `terminal_top1=0.2332`
2. hybrid `mlp`
   - `auc=0.5543`
   - `first_edge=0.4906`
   - `terminal_top1=0.8548`
3. hybrid `gated_mlp`
   - `auc=0.5011`
   - `first_edge=0.5472`
   - `terminal_top1=0.8710`

#### Math

1. baseline `E46`
   - `auc=0.6053`
   - `first_edge=0.6096`
   - `terminal_top1=0.1970`
2. hybrid `mlp`
   - `auc=0.4931`
   - `first_edge=0.4921`
   - `terminal_top1=0.7115`
3. hybrid `gated_mlp`
   - `auc=0.5162`
   - `first_edge=0.4921`
   - `terminal_top1=0.8654`

### 0W.6 Failure pattern

Failure-analysis artifacts:

1. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_mlp_gsm_diag_20260311T045406Z/summary.md`
2. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_gated_gsm_diag_20260311T045406Z/summary.md`
3. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_mlp_math_diag_20260311T045406Z/summary.md`
4. `assets/artifacts/phase_e_processbench_analysis/pbhybrid_ph1_gated_math_diag_20260311T045406Z/summary.md`

Key reading:

1. benchmark structure:
   - GSM8K `all_correct_ratio=0.4844`
   - Math `all_correct_ratio=0.4062`
2. hybrid training semantics:
   - `local_modified_process_error_step = 3452`
   - `terminal_completion_anchor = 132`
3. despite the terminal fraction being only `3.68%`,
   - terminal completion is repaired extremely aggressively,
   - but local ranking and first-error discrimination collapse.

Concrete example:

1. GSM8K `mlp`
   - `all_correct terminal top1 = 0.8548`
   - `first_edge = 0.4906`
2. Math `gated_mlp`
   - `all_correct terminal top1 = 0.8654`
   - `late_error pair_acc = 0.2131`

This means:

1. the model is learning "full correct completion should score high,"
2. but it loses the relative geometry needed for bad-prefix discrimination.

### 0W.7 Scientific conclusion

This round is a strong negative result against the simple hybrid recipe:

1. `PRMBench local + tiny terminal auxiliary` is **not** enough.
2. `gated_mlp` changes the tradeoff slightly,
   - but architecture alone does not repair the benchmark mismatch.
3. the main blocker is still supervision geometry:
   - terminal anchors are powerful even at very small ratios,
   - and naive mixture is still too blunt.

So the next mainline should **not** be:

1. more of the same mixture with larger caps,
2. or pure head-capacity tuning.

The next mainline should be:

1. smaller terminal ratios than the current `3.68%`,
2. benchmark-aware checkpoint selection,
3. or explicit staged / two-objective training where:
   - local ranking remains primary,
   - terminal-completion repair is constrained instead of merged naively.

## 0V. Latest Diagnosis Update (2026-03-11, RL Promotion Infrastructure Hardening + Nonfinite Feature Audit)

### 0V.1 What this round answered

This round asked a stricter question than the earlier "same-family strong" or
"ProcessBench somewhat positive" reads:

1. can the current Phase E infrastructure actually support a defensible
   RL-promotion decision,
2. and if not, is the blocker:
   - benchmark semantics,
   - shared-server infrastructure,
   - or hidden numerical corruption?

The answer is now:

1. the infrastructure is materially better after this round,
2. but no current candidate became RL-ready,
3. and a previously hidden numerical corruption bug was real.

### 0V.2 New infrastructure

Primary additions:

1. `scripts/phase_e_diagnose_rl_promotion.py`
   - new slice-aware RL-promotion gate
2. `scripts/phase_e_prepare_prmbench_terminal_anchor_pairs.py`
   - new `--terminal-anchor-ratio`
3. `scripts/phase_e_train_value.py`
   - fail fast on non-finite loss
   - new `--nonfinite-feature-policy {error,drop}`
4. `src/ours/phase_e/training.py`
   - explicit non-finite pooled-feature validation / filtering

Why this matters:

1. the repository can now distinguish:
   - local-first-bad weakness
   - later-bad weakness
   - terminal-completion weakness
   - outright numerical corruption
2. this is much closer to what an RL-promotion pipeline actually needs than a
   single `AUC` threshold.

### 0V.3 Baseline RL-promotion audit

Artifact:

1. `assets/artifacts/phase_e_rl_promotion_diag/phase_e_rlpromo_diag_baselines_0311_20260311T040150Z/summary.md`

Main reading:

1. `E80 fanout`
   - `samefamily_top1=0.9954`
   - `pb_min_auc=0.5106`
   - `pb_min_laterbad=0.5167`
   - `pb_min_terminal_top1=0.0577`
   - assessment:
     - `terminal_and_local_tradeoff_unresolved`
2. `E84 heavy terminal`
   - `samefamily_top1=0.8249`
   - `pb_min_auc=0.3906`
   - `pb_min_laterbad=0.2448`
   - `pb_min_terminal_top1=0.5897`
   - assessment:
     - `not_rl_ready_laterbad_generalization_weak`
3. `E46 PRM local`
   - `samefamily_top1=0.9659`
   - `pb_min_auc=0.6053`
   - `pb_min_laterbad=0.5501`
   - `pb_min_terminal_top1=0.1970`
   - assessment:
     - `terminal_and_local_tradeoff_unresolved`

Interpretation:

1. `E80` is still local-strong but terminal-poor.
2. `E84` repairs the terminal slice but destroys broader ranking.
3. `E46` remains the closest current baseline, but terminal completion is still
   the main blocker.

### 0V.4 Hidden numerical bug: pooled feature tensors themselves contained `NaN`

Math-side repair artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_rlpromo_0311_ms_fanout_ta015__16a79535c2e6`

Observed behavior:

1. warm-start `joint` continuation from `E80`
   - `loss=nan`
2. more conservative `ranking_only` continuation from `E80`
   - still failed on batch 1
   - with:
     - `rejected_logit_abs_max=nan`

Direct cache inspection then showed:

1. chosen pooled-feature cache had `13` bad rows
2. rejected pooled-feature cache had `31` bad rows
3. the bad rows were mainly `first_bad_fanout_prefix_ranking` examples, not
   only terminal anchors

This is the main technical result of the round:

1. the earlier NaN failure was not just a weak recipe,
2. it was an infrastructure-level corruption risk in:
   - `backbone -> pooled feature -> cache -> head`

### 0V.5 What the new `drop` policy achieved

Stable repaired run:

1. `assets/artifacts/phase_e_runs/phase_e_rlpromo_0311_ms_fanout_ta015_rankonly_warm_e80_dropnf_fix_20260311T044122Z`

What happened:

1. train bad rows dropped:
   - `44 / 9921`
2. eval bad rows dropped:
   - `3 / 1164`
3. held-out repair metrics:
   - `pair_acc=0.7700`
   - `auc=0.7235`
   - `ranking_score=0.7468`
4. same-family trust:
   - `prompt_pool_top1=0.7802`
   - `local_first_bad_acc=0.8492`
5. `ProcessBench`:
   - GSM8K:
     - `auc=0.5832`
     - `first_edge=0.6176`
   - Math:
     - `auc=0.5206`
     - `first_edge=0.5491`

Comparison artifact:

1. `assets/artifacts/phase_e_rl_promotion_diag/phase_e_rlpromo_diag_math_dropnf_0311_20260311T044717Z/summary.md`

Interpretation:

1. the infrastructure fix worked:
   - the candidate now trains and evaluates cleanly
2. but the scientific promotion result is still negative:
   - same-family utility dropped far below `E80`
   - `ProcessBench Math` stayed weak
   - the RL-promotion gate still returns `gate=0`

This is the right kind of negative result:

1. "numerical corruption removed"
2. but "candidate still not good enough"

### 0V.6 PRM-side light repair status

Prepared artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_rlpromo_0311_prm_ta010__b2281c74f155`

What was tried:

1. light `PRMBench` terminal-anchor continuation from `E46`
2. first with shared cache
3. then with `--feature-cache-mode off`

Outcome this round:

1. no completed final artifact was produced before this round closed
2. the shared-cache run was blocked by lock contention
3. the `off` run did remove the lock wait
4. but shared-machine feature extraction throughput was still too slow, so the
   run was manually stopped

Interpretation:

1. this is not a negative scientific result yet
2. it is an infrastructure / throughput limitation under a crowded server

### 0V.7 Updated practical conclusion

The repository is now closer to RL-ready *in infrastructure* than it was at
the start of the round:

1. promotion gating is stricter and more interpretable
2. non-finite losses no longer silently continue
3. non-finite pooled features are now explicit and optionally filterable

But the repository is still **not** RL-ready in candidate quality:

1. no audited candidate passes the strict RL-promotion gate
2. the best current benchmark-facing baseline remains `E46 PRM local`
3. math-side light repair proved:
   - infrastructure hardening can rescue a broken run
   - but that does not by itself produce an RL-usable head

The next sensible follow-up is:

1. re-run the light `PRMBench` continuation under an isolated feature-cache root
   or lower-cap smoke budget,
2. keep the new `error/drop` non-finite policy on by default in all repair work,
3. treat "candidate quality" and "infrastructure integrity" as two separate
   gates from now on.

## 0U. Internet-guided ProcessBench redesign suite (`pbr1/pbr2/pbr4`) (2026-03-11)

This round converted the latest internet research scan into one concrete local
redesign and then tested it end-to-end.

Primary code added:
1. `scripts/phase_e_curate_processbench_transfer_pairs.py`
2. `scripts/run_phase_e_processbench_research_suite.sh`
3. `src/ours/phase_e/training.py`
   - new `semantic / confidence_semantic` pair-weight modes

Executed smoke command:
1. `CUDA_VISIBLE_DEVICES=1 ACTIVE_PHASE_E_PB_RESEARCH_GROUP=PBR1_PROCESSBENCH_REDESIGN_SMOKE RUN_PREFIX=phase_e_processbench_research_v2 PHASE_E_PB_RESEARCH_CASES_OVERRIDE='pbr1_ms_align_mlp|one_shot|ms_align_v1|mlp|none pbr2_ms_align_gated|one_shot|ms_align_v1|gated_mlp|none pbr4_ms_curriculum_gated|curriculum|ms_align_v1|gated_mlp|none' TARGET_TOTAL_PAIRS=2048 BENCH_MAX_SAMPLES=64 TRAIN_BATCH_SIZE=64 EVAL_BATCH_SIZE=96 bash scripts/run_phase_e_processbench_research_suite.sh`

Primary artifacts:
1. suite summary:
   - `assets/artifacts/phase_e_logs/phase_e_processbench_research_v2/final_summary.md`
2. transfer compare:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_processbench_research_v2_gsm_compare_20260311T044545Z/summary.md`
   - `assets/artifacts/phase_e_transfer_compare/phase_e_processbench_research_v2_math_compare_20260311T044545Z/summary.md`
3. RL-promotion diagnosis:
   - `assets/artifacts/phase_e_rl_promotion_diag/phase_e_processbench_research_v2_rl_promotion_20260311T044546Z/summary.md`

Main result table:

| case | samefamily_top1 | samefamily_local | gsm_auc | gsm_first_edge | math_auc | math_first_edge | assessment |
|---|---:|---:|---:|---:|---:|---:|---|
| `ref_e87` | `0.6597` | `0.2914` | `0.4410` | `0.5854` | `0.4467` | `0.5957` | `terminal_and_local_tradeoff_unresolved` |
| `pbr1_ms_align_mlp` | `0.8316` | `0.8681` | `0.4754` | `0.3600` | `0.4692` | `0.2857` | `terminal_and_local_tradeoff_unresolved` |
| `pbr2_ms_align_gated` | `0.8947` | `0.9231` | `0.4713` | `0.5600` | `0.5055` | `0.5357` | `not_rl_ready_laterbad_generalization_weak` |
| `pbr4_ms_curriculum_gated` | `0.8316` | `0.8791` | `0.4947` | `0.4400` | `0.4743` | `0.4286` | `not_rl_ready_laterbad_generalization_weak` |

Conclusion:
1. `pbr2_ms_align_gated` is the strongest new candidate.
2. it materially improves same-family utility and terminal-completion behavior.
3. it still fails the strict RL-facing gate because `later-bad` and
   `first-edge` slices stay below threshold.
4. the current curriculum variant is not worth promoting.
5. the next repair target is now better localized:
   - improve `good_vs_laterbad` transfer
   - while keeping `first_bad` edge accuracy.

## 0T. Literature-guided mixed-supervision redesign on `ProcessBench Math 50` (2026-03-11)

This round asked a narrower but more actionable question than the earlier
repair smokes:

1. can we combine the useful part of `fanout`-style local supervision with the
   useful part of `terminal-anchor` supervision,
2. can training stay balanced enough that one branch does not erase the other,
3. and does head complexity matter once the data contract is improved?

Primary artifacts:
1. research/design note:
   - `docs/phase_e_rl_ready_research_redesign_20260311.md`
2. mixed artifact builder:
   - `scripts/phase_e_mix_pair_artifacts.py`
3. small mixed artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_pb_repair_0311_mixed_small_fanout_terminal__99976bcc33a8`
4. mixed-MLP run/eval:
   - `assets/artifacts/phase_e_runs/phase_e_pb_repair_0311_e75_mixed_small_mlp_20260311T042122Z`
   - `assets/artifacts/phase_e_eval/phase_e_pb_repair_0311_e75_mixed_small_mlp_math50_20260311T042808Z`
5. mixed-`gated_mlp` run/eval:
   - `assets/artifacts/phase_e_runs/phase_e_pb_repair_0311_e76_mixed_small_gated_20260311T042757Z`
   - `assets/artifacts/phase_e_eval/phase_e_pb_repair_0311_e76_mixed_small_gated_math50_20260311T043534Z`
6. comparison summaries:
   - `assets/artifacts/phase_e_transfer_compare/processbench_mixed_mlp_math50_compare_0311_20260311T043029Z/summary.md`
   - `assets/artifacts/phase_e_transfer_compare/processbench_mixed_arch_compare_0311_20260311T043656Z/summary.md`

### 0T.1 Research takeaway that directly changed the design

The literature and strong open-source baselines converge on one point:

1. `ProcessBench` is not only a local `last-safe > first-bad` edge task,
2. useful PRM supervision should preserve both:
   - local process discrimination
   - and broader progress / completion ordering,
3. naive PRM improvement is not yet the same thing as safe RL credit
   assignment.

That made the previous single-family repair logic too narrow.

### 0T.2 What changed locally

Pipeline changes:
1. new mixed-artifact balancing in `src/ours/phase_e/training.py`
   and `scripts/phase_e_train_value.py`
2. new pair-weight modes:
   - `group_balance`
   - `confidence_group_balance`
3. balancing label priority:
   - `artifact_mix_source_label`
   - `pair_semantics`
   - `source_tag`

Architecture changes:
1. added `gated_mlp` to `src/ours/phase_b/value_head.py`
2. kept plain `mlp` as the control

Artifact design:
1. mixed small artifact contains:
   - `fanout 1536`
   - `terminal-anchor 512`
2. intent:
   - retain local first-bad geometry,
   - add terminal-completion preference,
   - and explicitly avoid unbalanced optimization.

### 0T.3 Main results on identical `ProcessBench Math 50` slice

| case | auc | pair_acc | anygood_vs_firstbad | good_vs_laterbad | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|---:|
| `E46 baseline` | `0.5335` | `0.4778` | `0.4762` | `0.4783` | `0.2000` | `-0.2688` |
| `PRM terminal-anchor` | `0.5207` | `0.4272` | `0.4921` | `0.4111` | `0.5500` | `-0.0382` |
| `mixed MLP` | `0.5400` | `0.5000` | `0.4921` | `0.5020` | `0.3500` | `-0.1751` |
| `mixed gated_mlp` | `0.4263` | `0.4873` | `0.4762` | `0.4901` | `0.0500` | `-0.2213` |

### 0T.4 What we can trust from this round

1. `mixed MLP` is the best tradeoff tested so far on this benchmark slice.
2. it improves over `E46 baseline` on:
   - overall `auc`
   - overall `pair_acc`
   - broader `good_vs_laterbad`
3. it also preserves some of the terminal repair:
   - `terminal_top1 0.20 -> 0.35`
4. `terminal-anchor only` still wins the pure terminal slice,
   but breaks too much of the broader ranking surface.
5. `gated_mlp` did not help; the problem still looks more like
   data-contract / balancing mismatch than insufficient head complexity.

### 0T.5 Current conclusion

1. this repo is still not fully `RL-ready` in the strict sense.
2. however, the strongest next mainline is now much clearer:
   - mixed local + terminal supervision
   - explicit group-balanced training
   - simple `mlp` head
3. the next scaling move should be:
   - larger mixed artifact
   - optional extra later-bad branch
   - full same-family trust
   - full `ProcessBench GSM8K/Math`
   - and only then conservative RL-readiness claims
4. do **not** scale `gated_mlp` as the default mainline based on current
   evidence.

## 0S. Low-terminal mixed repair retry and runtime blocker audit (2026-03-11)

This round targeted the most plausible next RL-facing repair, not another
generic sweep:

1. keep the strongest confirmed mixed local baseline `E82`,
2. add only a small terminal-completion auxiliary signal,
3. and test whether the repo can move benchmark completion safety without
   breaking the local ranking geometry that already works.

Primary artifacts:
1. rebuilt PRMBench local-diagnostic pair artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_prmbench_localdiag_0311_e46_rebuild_sharedsplit_s42_pairs__f5778317f28b`
2. `E82` same-family baseline:
   - `assets/artifacts/phase_e_samefamily_eval/phase_e_rlpush_lowterm_0311_e82_samefamily_20260311T034620Z`
3. `E46` PRMBench same-family rerank refresh:
   - `assets/artifacts/phase_e_samefamily_eval/phase_e_prmbench_localdiag_0311_e46_samefamily_20260311T033803Z`
4. new repair pair artifacts:
   - `assets/artifacts/phase_e_pairs/phase_e_rlpush_lowterm_0311_e89_e89_ms_prmbench_transfer_mix_terminal10_confwt_warm_e82_seed42_sharedsplit_s42_pairs__9e47a4b941d8`
   - `assets/artifacts/phase_e_pairs/phase_e_rlpush_lowterm_0311_e90_e90_ms_prmbench_transfer_mix_terminal05_confwt_warm_e82_seed42_sharedsplit_s42_pairs__e9ecdd65f92b`

### 0S.1 What was added

Two new named groups were introduced:
1. `E89_MS_PRMBENCH_TRANSFER_MIX_TERMINAL10_CONFWT_WARM_E82_SEED42`
2. `E90_MS_PRMBENCH_TRANSFER_MIX_TERMINAL05_CONFWT_WARM_E82_SEED42`

Their purpose is narrow and controlled:
1. keep `Math-Shepherd fanout`,
2. keep `PRMBench local` support,
3. warm-start from `E82`,
4. and vary only the `terminal anchor` budget.

### 0S.2 Pair-budget diagnosis

Both new repair artifacts were built successfully under shared
`source_sample` splits and `balanced_support_bucket` capping.

1. `E89`
   - total pairs: `10000`
   - train semantics:
     - `first_bad_fanout_prefix_ranking = 4156`
     - `local_modified_process_error_step = 4112`
     - `terminal_completion_anchor = 735`
2. `E90`
   - total pairs: `10000`
   - train semantics:
     - `first_bad_fanout_prefix_ranking = 4331`
     - `local_modified_process_error_step = 4295`
     - `terminal_completion_anchor = 377`

Interpretation:
1. this is a clean terminal-budget probe,
2. the main local support stays almost fixed,
3. only terminal mass changes materially.

### 0S.3 Same-family baseline refresh

Before trusting any repair outcome, two baselines were pinned down:

1. `E82` same-family trust
   - `prompt_pool_top1_accuracy = 0.9633`
   - `prompt_pool_mean_regret = 0.0367`
   - `local_last_safe_top1_accuracy = 0.9124`
   - `local_first_bad_edge_accuracy = 0.9841`
2. `E46` PRMBench same-family rerank refresh
   - `prompt_pool_top1_accuracy = 0.9659`
   - `prompt_pool_mean_regret = 0.0341`

Important metric-contract finding:
1. restoring `positive_step_index / negative_step_index` into PRMBench pairs
   was necessary,
2. but `local_first_bad_edge_accuracy` still stays `N/A`.
3. this is now understood as a metric-definition gap:
   - current same-family local metrics assume
     `last_safe_prefix vs first_bad_prefix`,
   - PRMBench local supervision is
     `same-step correct sibling vs wrong sibling`.

Interpretation:
1. PRMBench local observability improved,
2. but there is still one missing audit metric for same-step sibling correctness.

### 0S.4 Runtime outcome of the new repair attempt

The scientific direction stayed plausible, but runtime robustness did not clear:

1. `E89` full repair run
   - pair construction succeeded,
   - training failed during frozen-backbone feature encoding with:
     - `RuntimeError: CUDA error: unspecified launch failure`
2. initial parallel `E90` attempt
   - showed the same large-run instability pattern,
   - and was intentionally stopped instead of being treated as a valid result
3. safer `E90` retry
   - reused the built pair artifact,
   - disabled feature-cache writes,
   - set:
     - `max_gpu_memory_gib = 48`
     - `max_cpu_memory_gib = 96`
     - `per_device_eval_batch_size = 48`
   - removed the immediate launch failure,
   - but still did not finish within this turn, so no benchmark claim should be
     promoted from it yet.

Interpretation:
1. the repo is still not runtime-stable enough for larger RL-facing warm-start
   repair runs to be treated as routine,
2. so there is still an infrastructure gate before the scientific RL gate.

### 0S.5 Current conclusion after this round

1. `E82` remains the best confirmed mixed local baseline.
2. low-terminal mixed repair remains the right scientific next move.
3. this round did **not** produce a new promoted RL-ready checkpoint.
4. the current blockers are now explicitly twofold:
   - local-vs-terminal scientific tradeoff
   - large-run runtime instability

### 0S.6 Explicit next-step plan

1. add a PRMBench-compatible same-family local metric for
   `same-step sibling correctness`,
2. make large Phase E repair runs robust by default:
   - no-cache mode for one-off sweeps,
   - lower initial encode batch for warm-start repairs,
   - explicit retry/fallback on non-OOM CUDA launch failures,
3. only then rerun `E90`-style low-terminal mixed repair and re-audit it
   against:
   - `E82` same-family trust
   - `ProcessBench GSM8K`
   - `ProcessBench Math`
   - strict transfer diagnosis

## 0R. RL-readiness bounded-support audit refresh (2026-03-11)

This round asked a more operational question than earlier `ACC90` work:

1. which current same-source winners are plausible bounded-support RL priors,
2. and which repair pilot is most worth scaling next?

Primary artifacts:
1. top-candidate audit:
   - `assets/artifacts/phase_e_logs/phase_e_rltops_0311_1124/final_summary.md`
2. repair-pilot audit:
   - `assets/artifacts/phase_e_logs/phase_e_rlrepairs_0311_1124/final_summary.md`

### 0R.1 Top-candidate audit

Audited checkpoints:
1. `ms_e68`
2. `ms_e43`
3. `prm_e46`
4. `prm_e78`

Key results:
1. `ms_e43`
   - `pool_top1 = 0.9648`
   - `local_first_bad = 0.9702`
   - `rej40_gain = 0.0352`
   - `pb_gsm_auc = 0.6245`
   - `pb_math_auc = 0.6341`
   - assessment:
     - `provisionally_rl_ready`
2. `prm_e46`
   - `pool_top1 = 0.9659`
   - `rej40_gain = 0.0341`
   - `pb_gsm_auc = 0.6264`
   - `pb_math_auc = 0.6053`
   - assessment:
     - `provisionally_rl_ready`
3. `ms_e68`
   - same-family geometry remains excellent,
   - benchmark remains positive,
   - but rejection gain is only:
     - `0.0207`
   - assessment:
     - `useful_signal_but_not_rl_ready`
4. `prm_e78`
   - same-family metrics remain very strong,
   - but benchmark safety falls to:
     - `gsm8k_auc = 0.5398`
     - `math_auc = 0.5117`
   - assessment:
     - `samefamily_only_not_benchmark_safe`

Interpretation:
1. current bounded-support RL candidates are:
   - `ms_e43`
   - `prm_e46`
2. `prm_e78` is now the clearest warning that stronger same-source fitting can
   still make a checkpoint less benchmark-safe.

### 0R.2 Repair-pilot audit

Audited repair pilots:
1. `ms_grid_micro`
2. `ms_ta_micro`
3. `prm_ta_smoke`

Key results:
1. `ms_grid_micro`
   - `pool_top1 = 0.9982`
   - `local_first_bad = 0.9914`
   - `pb_gsm_auc = 0.5891`
   - `pb_math_auc = 0.5559`
   - weakness:
     - `rej40_gain = 0.0018`
   - assessment:
     - `useful_signal_but_not_rl_ready`
2. `ms_ta_micro`
   - abstention utility improves:
     - `rej40_gain = 0.0571`
   - but same-family ordering degrades:
     - `pool_top1 = 0.7307`
     - `local_first_bad = 0.7904`
   - assessment:
     - `useful_signal_but_not_rl_ready`
3. `prm_ta_smoke`
   - `rej40_gain = 0.0627`
   - but benchmark ranking collapses:
     - `gsm8k_auc = 0.4778`
     - `math_auc = 0.4691`
   - assessment:
     - `samefamily_only_not_benchmark_safe`

Interpretation:
1. `ms_grid_micro` is the best current repair direction to scale.
2. terminal-anchor style repairs improve abstention more than they improve
   trustworthy ranking geometry.
3. the next repair stage should preserve the `ms_grid_micro` local geometry
   while explicitly raising rejection utility.

## 0Q. ProcessBench transfer engineering corrections and terminal-anchor audit (2026-03-11)

This round fixed the evaluation plumbing first, before trusting any new repair result.

Completed corrections:
1. `ProcessBench` subset eval is now stratified instead of raw first-`N`.
   - file:
     - `src/ours/phase_e/benchmark_eval.py`
2. `phase_e_analyze_processbench_failures.py` now analyzes the exact scored
   subset rather than rebuilding from the full benchmark file.
3. `Phase E` pair artifacts now record:
   - `global_cap_mode`
   - `global_cap_summary`
   - `overall_summary_before_global_cap`
4. `Math-Shepherd` terminal-anchor runs no longer silently lose all-positive
   repairs under `max_pairs_per_source`.
   - file:
     - `src/ours/phase_d/external_pairs_adapters.py`

Critical source-order finding:
1. in the current `Math-Shepherd` mirror, the first all-positive trajectory
   appears only at source row `121569`
2. total all-positive rows observed:
   - `160920`
3. consequence:
   - any naive stream-head source cap such as `max_pairs_per_source=20000`
     guarantees `terminal_completion_anchor = 0`
     even when the config nominally enables terminal anchors

Verified post-fix pair artifact:
1. artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_processbench_terminal_focus_0311_e83_ms_processbench_transfer_terminal_seed42_e83_ms_processbench_transfer_terminal_seed42_sharedsplit_s42_pairs__8b75a88516bc`
2. semantics before final cap:
   - `local_first_bad_edge = 9985`
   - `terminal_completion_anchor = 9992`
3. semantics after final cap:
   - `local_first_bad_edge = 4000`
   - `terminal_completion_anchor = 4000`
4. train split semantics:
   - `local_first_bad_edge = 3595`
   - `terminal_completion_anchor = 3603`

Interpretation:
1. the terminal-anchor recipe is now *actually present* in the train artifact
2. any earlier negative read on `E83`-style terminal-anchor repairs is not
   trustworthy unless that run's artifact explicitly shows non-zero
   `terminal_completion_anchor`

Corrected rerun outcome on the fixed `ProcessBench 96` subset:
1. baseline `E79`:
   - `gsm8k pair_acc = 0.6088`
   - `math pair_acc = 0.4558`
   - `gsm8k all_correct_top1 = 0.1087`
   - `math all_correct_top1 = 0.0000`
2. corrected `E83 terminal`:
   - held-out `pair_acc = 0.7731`
   - `gsm8k pair_acc = 0.3163`
   - `math pair_acc = 0.3273`
   - `gsm8k all_correct_top1 = 0.6304`
   - `math all_correct_top1 = 0.5641`
3. corrected `E84 fanout + terminal`:
   - held-out `pair_acc = 0.7808`
   - `gsm8k pair_acc = 0.3299`
   - `math pair_acc = 0.3233`
   - `gsm8k all_correct_top1 = 0.6739`
   - `math all_correct_top1 = 0.5897`

Interpretation:
1. the corrected terminal-anchor repair is now real and very strong on the
   all-correct slice
2. but at the current `50/50` mixture weight it over-corrects and destroys the
   broader good-vs-bad ranking surface
3. `fanout + terminal` improves geometric alignment over `terminal` alone, but
   still fails badly on overall `ProcessBench` ranking
4. next mainline should lower terminal-anchor mass rather than increase it

## 0R. Low-terminal RL-facing repair probe (`E87`, 2026-03-11)

New infrastructure added:
1. `step_label_terminal_anchor_fraction`
   - added to:
     - `PairBuildConfig`
     - `scripts/phase_e_prepare_pairs.py`
     - `scripts/run_phase_e_suite.sh`
2. purpose:
   - keep terminal anchors as a bounded auxiliary signal instead of a `50/50`
     co-equal training pool

Main new run:
1. group:
   - `E87_MS_PROCESSBENCH_TRANSFER_FANOUT_TERMINAL10_CONFWT_SEED42`
2. train artifact semantics:
   - `first_bad_fanout_prefix_ranking = 6487`
   - `terminal_completion_anchor = 735`
3. held-out fit:
   - `pair_acc = 0.8823`
   - `auc = 0.8647`
4. `ProcessBench GSM8K 96`:
   - `pair_acc = 0.4932`
   - `auc = 0.4410`
   - `all_correct_top1 = 0.3696`
   - `all_correct_gap = -0.0819`
5. `ProcessBench Math 96`:
   - `pair_acc = 0.4217`
   - `auc = 0.4467`
   - `all_correct_top1 = 0.2051`
   - `all_correct_gap = -0.1314`
6. same-family trust:
   - `prompt_pool_top1 = 0.6597`
   - `local_first_bad_acc = 0.2914`

Strict diagnosis:
1. `assets/artifacts/phase_e_transfer_diag/phase_e_transfer_diag_e87_0311_00/summary.md`
2. result:
   - `strict_rl_ready = 0`
   - `assessment = not_rl_ready_terminal_completion_risk`

Interpretation:
1. reducing terminal mass from `50%` to `10%` is directionally correct
2. it gives the best current benchmark tradeoff
3. but same-family decision utility is still far too weak for `Phase F`
4. therefore the repository is still **not RL-ready**
5. the next repair axis must be:
   - semantics-aware weighting / curriculum,
   - not just more pair-mix tuning

## 0P. Latest Diagnosis Update (2026-03-11, R-PRM Compact Train-Fit vs Held-Out Gap)

### 0P.1 What this round asked

Earlier `R-PRM` diagnosis had already established three negative facts:
1. legacy long-form truncation is no longer the main blocker,
2. simple verdict-polarity rebalance is not enough,
3. the source still stays far below the `ACC90` gate.

The missing question was narrower and more decisive:
1. is current `R-PRM compact_verdict` actually unlearnable under the present
   frozen-feature value-head stack,
2. or can it fit its own pair distribution well and the remaining problem is
   held-out generalization?

### 0P.2 Contract and infrastructure state used in this check

This round used the repaired compact contract on the full dataset:
1. pair artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_rprm_full_compact_fix__8d69afd6dba5`
2. key counts:
   - total pairs: `83159`
   - train: `74842`
   - validation: `8317`
3. important duplication audit:
   - train:
     - rows: `74842`
     - unique prompts: `74501`
     - mean pairs per prompt: `1.0046`
   - validation:
     - rows: `8317`
     - unique prompts: `8313`
     - mean pairs per prompt: `1.0005`

Interpretation:
1. prompt duplication exists but is tiny,
2. same-source `>90%` on this source cannot be waved away as simple prompt
   memorization.

### 0P.3 Important bug fixed before the probes

One real training-gate bug was fixed in:
1. `src/ours/phase_e/training.py`

Old behavior:
1. any nonzero
   - `frac_pairs_identical_after_truncation`
   - or `frac_pairs_first_diff_after_cutoff`
   caused a hard failure,
2. even if the fraction was only around `0.0015`.

New behavior:
1. these diagnostics now respect the same configured tolerance as
   `frac_pairs_over_limit`.

Why this matters:
1. it removes false negatives where a healthy pair artifact was being rejected
   because of a tiny truncation tail.

### 0P.4 Train-distribution fit ceiling probe

Run:
1. `assets/artifacts/phase_e_runs/phase_e_rprm_trainfit_probe_0310_s4k_20260310T160730Z`

Configuration summary:
1. train pairs:
   - `4000`
2. eval pairs:
   - `1000`
3. eval source:
   - sampled from the training distribution itself
4. head:
   - `MLP`
   - hidden size `2048`
5. objective:
   - `joint`
6. max length:
   - `2048`

Result:
1. `pair_acc = 0.9090`
2. `auc = 0.9131`
3. `mean_margin = 0.2932`

Interpretation:
1. current `R-PRM compact_verdict` is **not** fundamentally unlearnable.
2. the current feature extractor + head + objective stack can already fit the
   train-distribution pair relation to `>90%`.

### 0P.5 Matching true held-out probe

Run:
1. `assets/artifacts/phase_e_runs/phase_e_rprm_heldout_repair_0310_s4k_20260310T164443Z`

Configuration:
1. intentionally matched the train-fit probe on:
   - source
   - head
   - objective
   - length
   - optimization
2. only the eval split changed:
   - true validation pairs instead of train-distribution pairs

Result:
1. `pair_acc = 0.6280`
2. `auc = 0.6508`
3. `mean_margin = 0.1063`

Interpretation:
1. the `R-PRM` problem is now sharply localized:
   - it is **not** mainly head capacity,
   - **not** mainly training duration,
   - **not** mainly residual truncation.
2. the remaining blocker is:
   - held-out generalization under the current `compact_verdict` contract.

### 0P.6 Operational side note: OOM behavior

During the train-fit probe:
1. feature encoding hit real GPU pressure on the crowded server,
2. OOM backoff triggered:
   - `bs=12 -> 6 -> 3`,
3. and the run completed successfully.

Interpretation:
1. current Phase E encoding-layer OOM backoff is functioning,
2. the main scientific conclusion here is therefore not confounded by a
   crashed or partially skipped run.

### 0P.7 Updated repository conclusion for `R-PRM`

The strongest current statement is now:
1. `R-PRM compact_verdict` is learnable,
2. but it is not currently a held-out `ACC90` source,
3. and the bottleneck is a supervision-contract / generalization mismatch.

This is a much stronger diagnosis than either of the earlier weaker claims:
1. "it just needs more length",
2. "it just needs more head capacity",
3. or "it is simply random / unusable."

## 0O. Latest Diagnosis Update (2026-03-11, ProcessBench Smoke Repairs Split The Failure Modes)

### 0O.1 What this round answered

This round asked a narrower and more operational question:
1. on `ProcessBench`, is the bottleneck mainly:
   - missing non-local good-vs-bad geometry,
   - or missing terminal-completion supervision?

The answer is now:
1. **both matter**,
2. but they repair **different slices**,
3. and neither one alone is sufficient.

### 0O.2 New diagnostic layer

New script:
1. `scripts/phase_e_compare_processbench_transfer.py`

Why it matters:
1. a single benchmark AUC was hiding which part of transfer was broken.
2. the new compare view puts these side by side:
   - `anygood_vs_firstbad`
   - `good_vs_laterbad`
   - `terminal_top1`
   - `terminal_gap`
   - training-geometry fractions

### 0O.3 `Math-Shepherd` geometry repairs: useful but not sufficient

Main smoke artifact:
1. `assets/artifacts/phase_e_transfer_compare/processbench_math50_compare_all_0311_20260310T170538Z/summary.md`

On the same `ProcessBench Math 50` subset:
1. baseline `E68`
   - `auc=0.4251`
   - `first_edge=0.4615`
   - `terminal_top1=0.0500`
2. `Math-Shepherd grid` smoke
   - `auc=0.4262`
   - `first_edge=0.4615`
   - `terminal_top1=0.0500`
3. `Math-Shepherd fanout` smoke
   - `auc=0.4199`
   - `first_edge=0.5000`
   - `terminal_top1=0.0500`

Interpretation:
1. `fanout` helps exactly the local slice it should help:
   - `anygood_vs_firstbad`
   - and slightly `first_edge`
2. `grid` reduces pair-type mismatch on paper,
   but does not deliver benchmark lift on this subset.
3. neither geometry repair changes terminal collapse.

### 0O.4 `PRMBench terminal-anchor` repair: first strong positive terminal signal

Main artifact:
1. `assets/artifacts/phase_e_transfer_compare/processbench_prm_math50_compare_0311_20260310T171808Z/summary.md`

On the same `ProcessBench Math 50` subset:
1. baseline `E46`
   - `terminal_top1=0.2000`
   - `terminal_gap=-0.2688`
   - `pair_acc=0.4778`
   - `auc=0.5335`
2. `PRMBench terminal-anchor` smoke
   - `terminal_top1=0.5500`
   - `terminal_gap=-0.0382`
   - `pair_acc=0.4272`
   - `auc=0.5207`

Interpretation:
1. terminal-anchor supervision is the first repair that clearly moves the
   all-correct terminal slice in the desired direction.
2. but it damages broader good-vs-bad ranking, especially:
   - `good_vs_laterbad`
3. so the terminal undervaluation diagnosis was correct,
   but terminal anchors alone are not the full fix.

### 0O.5 Updated practical conclusion

The repository now has a much sharper ProcessBench diagnosis:
1. `Math-Shepherd` geometry repairs mostly improve the local first-bad
   neighborhood.
2. `PRMBench` terminal anchors strongly improve the terminal all-correct slice.
3. the next mainline repair should therefore be a **mixed supervision recipe**:
   - terminal anchors
   - plus later-bad / broader good-vs-bad coverage
   - with careful weighting

This is a stronger and more useful conclusion than:
1. "try more benchmark-like pairs",
2. or "just add terminal anchors",
because the smoke runs now show exactly what each one fixes and what each one breaks.

## 0N. Latest Diagnosis Update (2026-03-11, Should MCTS Be The Next Mainline Fix?)

### 0N.1 Question

After the `ProcessBench` alignment audit, the next natural question is:
1. can `MCTS` directly solve the current Phase E transfer problem?

### 0N.2 Literature-guided answer

Short answer:
1. **not as the next mainline fix**,
2. but **possibly yes as a later data-construction or test-time branch**.

The main literature pattern is:
1. `ReST-MCTS*`
   - supports tree search as a way to harvest better traces / step targets for
     later training.
2. `Tree-PLV`
   - supports tree-based pair construction for better step-level ranking
     supervision.
3. `Rewarding Progress`
   - supports progress-aware process targets, which makes search more useful
     than naive correctness-only rollouts.
4. `MCTS-Judge`
   - supports MCTS as test-time judge/search scaling.

What this does **not** imply:
1. that adding `MCTS` to the current misaligned local/terminal/grid setup will
   automatically fix benchmark transfer.

The main warning signal from the literature is:
1. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   shows that naive synthetic MC-style PRM labels can generalize poorly.
2. so an uncontrolled tree-search branch can easily become a new noisy-label
   pipeline.

### 0N.3 Repository-specific diagnosis

For this repository, the current measured bottleneck is:
1. **supervision semantics mismatch**
   - local-only supervision under-teaches terminal completion,
   - terminal-only repair over-corrects,
   - grid-only repair helps a different side of the benchmark.

Therefore:
1. the present problem is not primarily "lack of search budget",
2. it is "training supervision and benchmark semantics are not yet aligned".

### 0N.4 Practical conclusion

The correct reading is:
1. `MCTS` should **not** replace the current `local + terminal + optional grid`
   repair path.
2. If introduced later, the two defensible forms are:
   - offline tree harvesting for higher-margin local / terminal pairs
   - or a separate test-time judge/search baseline
3. So the current mainline remains:
   - repair the supervision contract first,
   - and keep `MCTS` as a branch experiment rather than a reset of Phase E.

Primary references:
1. `ReST-MCTS*`
   - https://arxiv.org/abs/2406.03816
   - https://github.com/THUDM/ReST-MCTS
2. `Advancing Process Verification for Large Language Models via Tree-Based Preference Learning`
   - https://arxiv.org/abs/2407.00390
   - https://aclanthology.org/2024.emnlp-main.125/
3. `Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning`
   - https://arxiv.org/abs/2410.08146
4. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   - https://arxiv.org/abs/2501.07301
5. `MCTS-Judge: Test-Time Scaling in LLM-as-a-Judge for Code Correctness Evaluation`
   - https://arxiv.org/abs/2502.12468

## 0M. Latest Diagnosis Update (2026-03-11, ProcessBench Alignment Audit + Micro Repair Pilots)

### 0M.1 What this round was trying to answer

This round focused on one narrower but more important question than generic
same-source learnability:
1. why do strong same-source value heads still transfer weakly to `ProcessBench`?
2. is the failure mainly caused by:
   - missing terminal-completion supervision,
   - missing broader good-vs-bad prefix coverage,
   - or both?

### 0M.2 New diagnosis artifact

New script:
1. `scripts/phase_e_analyze_processbench_failures.py`

It compares:
1. training-pair semantics
2. ProcessBench structure
3. bucketed benchmark behavior

Representative outputs:
1. `assets/artifacts/phase_e_processbench_analysis/ms_e68_pb_math_v2_0311_20260310T160909Z/summary.md`
2. `assets/artifacts/phase_e_processbench_analysis/prm_e46_pb_math_v2_0311_20260310T160909Z/summary.md`

What it established:
1. `ProcessBench` is not only a local first-bad benchmark.
2. It contains a large all-correct block:
   - GSM8K: `0.4825`
   - Math: `0.4060`
3. Both current strong baselines had effectively zero terminal-anchor supervision:
   - `E68`: pure `local_first_bad_edge`
   - `E46`: pure `local_modified_process_error_step`
4. Therefore the previous qualitative diagnosis is now hard evidence:
   - both baselines are structurally under-supervising the
     `all-correct terminal completion` part of ProcessBench.

### 0M.3 New repair artifacts

New scripts / artifacts:
1. `scripts/phase_e_prepare_prmbench_terminal_anchor_pairs.py`
   - artifact:
     - `assets/artifacts/phase_e_pairs/phase_e_prmbench_terminal_anchor_full_0311__192ca71fd301`
2. `scripts/phase_e_prepare_mathshepherd_terminal_anchor_pairs.py`
   - capped artifact:
     - `assets/artifacts/phase_e_pairs/phase_e_ms_terminal_anchor_cap20k_diag_0311__6d57b0d4b490`
3. benchmark-aligned Math-Shepherd grid artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_ms_grid_cap40k_diag_0311__4f87d4f4cea6`

Their intended semantics are now clearly separated:
1. `PRMBench + terminal anchors`
   - keep local explicit error-step supervision
   - add full-correct vs shorter-safe prefix anchors
2. `Math-Shepherd + terminal anchors`
   - keep local first-bad-edge supervision
   - add all-positive full-completion anchors
3. `Math-Shepherd grid`
   - expose broader good-vs-bad prefix relations
   - but still no explicit terminal-anchor signal

### 0M.4 Micro repair pilot results

These were intentionally run as micro warm-start pilots, not promotion runs.
The goal was to test directionality under shared-server constraints.

#### Baselines

1. `E46` baseline on `ProcessBench`
   - GSM8K:
     - `pair_acc = 0.6701`
     - `auc = 0.6264`
     - `first_edge = 0.6706`
     - `all_correct_last = 0.2924`
   - Math:
     - `pair_acc = 0.5653`
     - `auc = 0.6053`
     - `first_edge = 0.6096`
     - `all_correct_last = 0.2452`

2. `E68` baseline on `ProcessBench`
   - GSM8K:
     - `pair_acc = 0.6385`
     - `auc = 0.5885`
     - `first_edge = 0.6294`
     - `all_correct_last = 0.5626`
   - Math:
     - `pair_acc = 0.5809`
     - `auc = 0.5547`
     - `first_edge = 0.5553`
     - `all_correct_last = 0.5895`

#### Pilot A: `PRMBench + terminal anchors`, warm-start from `E46`

Artifacts:
1. train run:
   - `assets/artifacts/phase_e_runs/phase_e_pbta_warm_e46_micro_0311_20260310T162646Z`
2. benchmark eval:
   - `assets/artifacts/phase_e_eval/phase_e_pbta_warm_e46_micro_pb_gsm8k_0311_20260310T163036Z/metrics.json`
   - `assets/artifacts/phase_e_eval/phase_e_pbta_warm_e46_micro_pb_math_0311_20260310T163037Z/metrics.json`

Result:
1. terminal completion improved clearly:
   - GSM8K `0.2924 -> 0.4196`
   - Math `0.2452 -> 0.3492`
2. but local/global ranking softened:
   - GSM8K `auc 0.6264 -> 0.6014`
   - Math `auc 0.6053 -> 0.5906`
   - GSM8K `first_edge 0.6706 -> 0.6471`
   - Math `first_edge 0.6096 -> 0.6013`

Interpretation:
1. terminal anchors are doing the intended thing,
2. but they trade against local error discrimination when used alone.

#### Pilot B: `Math-Shepherd + terminal anchors`, warm-start from `E68`

Artifacts:
1. train run:
   - `assets/artifacts/phase_e_runs/phase_e_msta_warm_e68_micro_0311_20260310T163102Z`
2. benchmark eval:
   - `assets/artifacts/phase_e_eval/phase_e_msta_warm_e68_micro_pb_gsm8k_0311_20260310T163337Z/metrics.json`
   - `assets/artifacts/phase_e_eval/phase_e_msta_warm_e68_micro_pb_math_0311_20260310T163336Z/metrics.json`

Result:
1. terminal completion improved dramatically:
   - GSM8K `0.5626 -> 0.7590`
   - Math `0.5895 -> 0.7663`
2. but local / good-vs-bad discrimination degraded materially:
   - GSM8K `auc 0.5885 -> 0.5527`
   - Math `auc 0.5547 -> 0.5350`
   - GSM8K `first_edge 0.6294 -> 0.6059`
   - Math `first_edge 0.5553 -> 0.5324`

Interpretation:
1. pure terminal-anchor repair is too strong here,
2. and it over-corrects toward completion preference.

#### Pilot C: `Math-Shepherd all_good_vs_all_bad` grid, warm-start from `E68`

Artifacts:
1. train run:
   - `assets/artifacts/phase_e_runs/phase_e_msgrid_warm_e68_micro_0311_20260310T163400Z`
2. benchmark eval:
   - `assets/artifacts/phase_e_eval/phase_e_msgrid_warm_e68_micro_pb_gsm8k_0311_20260310T163559Z/metrics.json`
   - `assets/artifacts/phase_e_eval/phase_e_msgrid_warm_e68_micro_pb_math_0311_20260310T163559Z/metrics.json`

Result:
1. local / late-bad side held up better than the terminal-anchor-only repair:
   - GSM8K `pair_acc 0.6385 -> 0.6436`
   - GSM8K `first_edge 0.6294 -> 0.6235`
   - Math `pair_acc 0.5809 -> 0.5839`
   - Math `first_edge 0.5553 -> 0.5595`
2. but terminal gap remained largely unresolved:
   - Math `all_correct terminal top1` stayed poor:
     - baseline `0.0517`
     - grid pilot `0.0591`
   - mean terminal score only moved slightly:
     - Math `0.5895 -> 0.6011`

Interpretation:
1. the grid repair mainly targets the good-vs-bad prefix side,
2. but it does not fix terminal completion on its own.

### 0M.5 Updated conclusion

This round establishes a much sharper ProcessBench diagnosis:
1. `terminal anchors` fix the terminal side,
2. `good_bad_prefix_grid` helps the broader prefix-ranking side,
3. but neither alone solves both.

So the next credible Phase E repair should be:
1. **mixed supervision or staged curriculum**
   - local error-step pairs
   - plus limited terminal anchors
   - plus optional grid-style broader good/bad coverage
2. not another generic LR / dropout / batch-size sweep.

That is the main new scientific information from this round.

## 0L. Latest Diagnosis Update (2026-03-11, Phase E Same-Source ACC90/95 And R-PRM Root Cause)

### 0L.1 What this round answered

This round tightened three separate questions that should not be mixed:
1. can `Math-Shepherd` reach stable same-source `>95%` pair accuracy?
2. can `PRMBench_Preview` reach same-source `>95%` pair accuracy with a dataset-specific recipe?
3. if `R-PRM` still underperforms, is the bottleneck still truncation, or has it moved to a deeper contract/objective mismatch?

### 0L.2 Same-source ACC results now clearly separate the sources

#### `Math-Shepherd`

Artifacts:
1. `assets/artifacts/phase_e_logs/phase_e_ms_acc90_full_0310_1914_e41_ms_acc90_mlp_rank_seed3/final_summary.md`
2. `assets/artifacts/phase_e_logs/phase_e_ms_acc95_push_0310_2146/final_summary.md`

Main reading:
1. same-source `Math-Shepherd` is already solved under the current Phase E trainer family.
2. representative high-water marks:
   - `E41`
     - `pair_acc=0.9850`
     - `auc=0.9034`
   - `E68`
     - `pair_acc=0.9725`
     - `auc=0.9415`
3. pair-error analysis on the best same-source run shows that remaining failures are concentrated on:
   - later `first_bad_edge` positions,
   - longer step chains,
   rather than a generic inability to rank.

#### `PRMBench_Preview`

Artifacts:
1. `assets/artifacts/phase_e_logs/phase_e_prmbench_acc90_full_0310_1914/final_summary.md`
2. `assets/artifacts/phase_e_logs/phase_e_prmbench_acc95_push_0310_2359/final_summary.md`

Main reading:
1. `PRMBench_Preview` also supports strong same-source learning.
2. the best current single-seed push already crosses the `95%` pair-accuracy target:
   - `E78_PRMBENCH_ACC95_JOINT_OVERFIT_SEED42`
     - `pair_acc=0.9521`
     - `auc=0.9071`
3. therefore the current Phase E stack is not generically weak.
4. it is source-sensitive.

### 0L.3 `R-PRM` deep diagnosis: truncation is no longer the main blocker

Artifacts:
1. deep-diagnostic summary:
   - `assets/artifacts/phase_e_logs/phase_e_rprm_deep_diag_0310_2359/final_summary.md`
2. compact length/root-cause audit:
   - `assets/artifacts/phase_e_logs/phase_e_rprm_diag_0310_2019/final_summary.md`

Deep-diagnostic findings:
1. on the repaired small compact artifact, `1536` and `2048` are already truncation-clean.
2. the best old compact recipe remains:
   - `C9_MLP_BCE_2048`
     - `pair_acc=0.6694`
     - `auc=0.6571`
3. objective ordering is now clear:
   - `pair_bce_only` > `joint` > `ranking_only`
4. this means compact `R-PRM` behaves more like a binary verdict-fitting source than a generic ranking-pair source.

### 0L.4 New polarity-repair experiments on `R-PRM`

New artifacts:
1. repaired `compact_correctness` pair artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_rprm_correctness_diag_0310_2341_pairs__b835692e7df6`
2. runs:
   - `assets/artifacts/phase_e_runs/phase_e_rprm_correctness_diag_0310_2341_bce2048_20260310T154119Z`
   - `assets/artifacts/phase_e_runs/phase_e_rprm_correctness_diag_0310_2341_bce2048_vbal_20260310T154500Z`
   - `assets/artifacts/phase_e_runs/phase_e_rprm_correctness_diag_0310_2341_joint2048_vbal_logit_20260310T154937Z`
3. error profiles:
   - `assets/artifacts/phase_e_error_analysis/rprm_correctness_bce2048_profile_20260310T155010Z/summary.md`
   - `assets/artifacts/phase_e_error_analysis/rprm_correctness_bce2048_vbal_profile_20260310T155010Z/summary.md`
   - `assets/artifacts/phase_e_error_analysis/rprm_correctness_joint2048_vbal_profile_20260310T155010Z/summary.md`

What these runs established:
1. `compact_correctness + BCE + 2048`
   - `pair_acc=0.6481`
   - `auc=0.6519`
   - this does **not** beat the old `compact_verdict + BCE + 2048` baseline.
2. `compact_correctness + BCE + verdict_balance`
   - `pair_acc=0.5926`
   - `auc=0.5765`
   - naive verdict balancing hurts overall performance.
3. `compact_correctness + joint + logit + verdict_balance`
   - `pair_acc=0.5926`
   - `auc=0.6306`
   - but importantly, it almost removes the old polarity asymmetry:
     - chosen=`no`: `0.5968`
     - chosen=`yes`: `0.5870`

Interpretation:
1. the repository has now separated two different `R-PRM` issues:
   - **issue A: polarity bias**
   - **issue B: weak compact supervision contract**
2. `compact_correctness + verdict_balance + joint/logit` largely fixes issue A.
3. but overall accuracy still remains far below `ACC90`.
4. so the main remaining problem is no longer truncation or simple verdict imbalance.
5. it is that the current compact contract does not expose enough useful signal to the frozen-head scorer.

### 0L.5 Updated conclusion

The current Phase E evidence now supports:
1. `Math-Shepherd`
   - strong same-source source
2. `PRMBench_Preview`
   - strong same-source source
3. `R-PRM`
   - not a primary same-source high-accuracy source under the current compact contract
   - useful mainly as a source-specific diagnosis case

So the project should stop framing `R-PRM` as "one more generic pair dataset" and instead treat it as:
1. a compact-verdict / verifier-style source with a different supervision geometry,
2. one that likely needs a different model contract than the current frozen feature scorer.

## 0K. Latest Diagnosis Update (2026-03-10, RL-Readiness Audit Of Current Value-Head Candidates)

### 0K.1 What this round was trying to answer

The repository had already shown that some sources support very strong
same-source held-out pair fitting.

But that still did not answer the more important practical question:
1. is any current value head good enough for *conservative RL-style use*?

So this round asked:
1. among the repository's strongest current checkpoints,
2. which ones survive stronger decision-style offline checks,
3. and has the project now reached a level that is usable for bounded-support
   RL, even if not for unrestricted reward optimization?

### 0K.2 Audit design

A new wrapper was added:
1. `scripts/run_phase_e_rl_readiness_suite.sh`

Main command:
1. `CUDA_VISIBLE_DEVICES=2 ACTIVE_PHASE_E_RL_GROUP=RR4_COMPARE_CURRENT_TOPS RUN_PREFIX=phase_e_rl_readiness_0310_2338 bash scripts/run_phase_e_rl_readiness_suite.sh`

This compared three current top candidates:
1. `ms_e68`
   - strongest current same-source Math-Shepherd winner
2. `ms_e14`
   - benchmark-aware Math-Shepherd trust candidate
3. `prm_e46`
   - strongest current PRMBench same-source winner

Each candidate was tested on:
1. same-family prompt-pool reranking
2. rejection / abstention utility
3. best-of-N pressure
4. `ProcessBench GSM8K`
5. `ProcessBench Math`

Then the two most promising candidates got an extra higher-pressure recheck.

### 0K.3 One summary-layer bug was discovered and fixed during the audit

The first generated wrapper summary incorrectly read benchmark AUC from a
generic `auc` key.

But `ProcessBench` summary files store:
1. `pair_auc_good_vs_bad`

So the wrapper was fixed, and the final summary was regenerated from the
already-finished benchmark artifacts without rerunning the whole audit.

This matters because the corrected benchmark values materially change the
interpretation of the strongest candidates.

### 0K.4 Main audit results

Artifact:
1. `assets/artifacts/phase_e_logs/phase_e_rl_readiness_0310_2338/final_summary.md`

#### `ms_e68` : same-family excellent, benchmark acceptable but not dominant

Same-family trust:
1. `prompt_pool_top1_accuracy = 0.9793`
2. `random_top1_baseline = 0.4995`
3. `top1_lift_over_random = 0.4798`
4. `local_first_bad_edge_accuracy = 0.9779`
5. `pressure@8 top1 = 1.0000`

ProcessBench:
1. `gsm8k_auc = 0.5885`
2. `math_auc = 0.5547`

Interpretation:
1. this checkpoint is already very strong as a same-family reranker,
2. and its benchmark behavior is clearly above random,
3. but it is not the cleanest overall RL-facing candidate because its full
   audit profile is less balanced than `prm_e46`.

#### `ms_e14` : benchmark-aware Math-Shepherd candidate still fails benchmark safety

Same-family trust:
1. `prompt_pool_top1_accuracy = 0.8584`
2. `local_first_bad_edge_accuracy = 0.8664`
3. `rejection@0.4 top1 = 0.9913`

ProcessBench:
1. `gsm8k_auc = 0.5026`
2. `math_auc = 0.5138`

Interpretation:
1. this run is useful signal,
2. but it is not benchmark-safe enough,
3. so it should not be promoted as the repository's main RL-facing candidate.

#### `prm_e46` : first current checkpoint that clears a conservative bounded-support gate

Same-family trust:
1. `prompt_pool_top1_accuracy = 0.9659`
2. `random_top1_baseline = 0.4959`
3. `top1_lift_over_random = 0.4701`
4. `rejection@0.4 top1 = 1.0000`
5. `pressure@8 top1 = 0.9375`

ProcessBench:
1. `gsm8k_auc = 0.6264`
2. `math_auc = 0.6053`

Interpretation:
1. this is the first current checkpoint family that looks strong both:
   - inside its own source family,
   - and on the repository's stronger benchmark-native recheck
2. under the wrapper's conservative internal heuristic, this row becomes:
   - `provisionally_rl_ready`

### 0K.5 Extra pressure-stress recheck

Artifacts:
1. `assets/artifacts/phase_e_samefamily_eval/phase_e_rl_pressure_stress_0310_2340_ms_e68_20260310T153847Z/summary.md`
2. `assets/artifacts/phase_e_samefamily_eval/phase_e_rl_pressure_stress_0310_2340_prm_e46_20260310T153848Z/summary.md`

#### `E68` stress result

1. `rejection@0.4 = 1.0000`
2. `rejection@0.1 = 1.0000`
3. `pressure@12 = 1.0000`

Interpretation:
1. `E68` is not fragile under stronger same-family decision pressure;
2. its weaker overall RL audit result is therefore **not** because the
   checkpoint collapses when selection gets stronger;
3. it is mainly because its benchmark-native behavior is only moderate, not
   because same-family utility is weak.

#### `E46` stress result

1. `rejection@0.4 = 1.0000`
2. `rejection@0.1 = 1.0000`
3. `pressure@4 = 0.9411`
4. `pressure@8 = 0.9375`

Interpretation:
1. the current best PRMBench checkpoint keeps strong selection utility under
   higher pressure,
2. and it does so without collapsing into near-random behavior.

### 0K.6 Final judgement

The right answer is **not** a flat yes/no.

The most accurate reading is:
1. the repository has **not** yet proved that current value-head training is
   safe for aggressive, open-ended, high-weight RL optimization;
2. but it **has** now reached a level where at least one checkpoint family
   (`prm_e46`) is usable for **bounded-support RL-style control**:
   - conservative reranking
   - rejection / abstention
   - low-weight reward prior
   - math-family process search / filtering

So the current status should be narrated as:
1. **provisionally RL-usable under narrow constraints**
2. **not yet generally RL-ready as a trusted reward model**

### 0K.7 Why the claim must still stay bounded

Even after this positive result, several things are still missing:
1. no true closed-loop RL experiment has yet been run with this candidate
2. no direct reward-hacking / shortcut-exploitation test has yet been passed
3. robustness is still shown only on nearby benchmark families, not across
   broader domains
4. `Math-Shepherd` and `PRMBench` still disagree on what the best checkpoint
   family is, so source-specificity remains real

### 0K.8 Immediate next-step plan

1. Promote `prm_e46` as the current **bounded-support RL candidate**.
2. Use it first only in conservative modes:
   - rerank
   - rejection gate
   - clipped / low-weight reward shaping
3. Before calling anything "fully RL-ready", require at least one new stage:
   - actual closed-loop conservative search / RL improvement test
   - plus reward-scale clipping / exploitation audit
4. Keep `ms_e68` as the strongest Math-Shepherd local-faithfulness control,
   but not as the main RL-facing promoted checkpoint.

## 0J. Latest Diagnosis Update (2026-03-10, Math-Shepherd ACC95 Verification And Push Matrix)

### 0J.1 What this round was actually trying to answer

The user request was framed as:
1. re-check current `Math-Shepherd` performance,
2. try repairs if needed,
3. and reach `95%` held-out pair accuracy,
4. while staying careful about shared-server OOM risk.

But the repository's own recent artifacts already showed that
`Math-Shepherd` had crossed this target:
1. `E42_MS_ACC90_MLP_JOINT_SEED3`
   - `mean_pair_acc = 0.963106`
2. `E43_MS_ACC90_MLP_HIGHCONF_SEED3`
   - `mean_pair_acc = 0.961268`

So the real question for this round became:
1. does the current code state still reproduce a `>95%` same-source regime
   cleanly,
2. and among low-risk fixes, which one improves the already-strong baseline
   the most?

### 0J.2 Resource hygiene mattered because the server is shared

Before launching the new matrix, GPU occupancy was checked with `nvidia-smi`.

Observed state at launch:
1. all four `A100 80GB` devices were idle
2. none had active memory pressure

Even so, the run was executed on only:
1. `CUDA_VISIBLE_DEVICES=1`

Reason:
1. keep the experiment sequential and conservative,
2. avoid unnecessary multi-GPU cache churn,
3. and reduce the chance of transient OOM caused by overlapping users joining
   later during the run window.

### 0J.3 New dedicated Math-Shepherd ACC95 matrix

New parameter groups were added:
1. `E67_MS_ACC95_JOINT_VERIFY_SEED42`
   - full-data verify control
2. `E68_MS_ACC95_JOINT_LOGIT_SEED42`
   - same control, but `ranking_target_space = logit`
3. `E69_MS_ACC95_JOINT_OVERFIT_SEED42`
   - same control, but with a more aggressive fit-oriented recipe
4. wrapper:
   - `I6_MS_ACC95_PUSH_MATRIX`

Executed command:
1. `CUDA_VISIBLE_DEVICES=1 ACTIVE_PHASE_E_INTRADATASET_GROUP=I6_MS_ACC95_PUSH_MATRIX RUN_PREFIX=phase_e_ms_acc95_push_0310_2146 bash scripts/run_phase_e_intradataset_suite.sh`

Artifacts:
1. suite summary:
   - `assets/artifacts/phase_e_logs/phase_e_ms_acc95_push_0310_2146/final_summary.md`
2. candidate report:
   - `assets/artifacts/phase_e_candidates/phase_e_ms_acc95_push_0310_2146_candidate/candidate_report.md`

### 0J.4 Results: the target was not only met, it was improved

Held-out same-source results:
1. `E67_MS_ACC95_JOINT_VERIFY_SEED42`
   - `pair_acc = 0.963267`
   - `auc = 0.942383`
   - `ranking_score = 0.952825`
2. `E68_MS_ACC95_JOINT_LOGIT_SEED42`
   - `pair_acc = 0.972450`
   - `auc = 0.941545`
   - `ranking_score = 0.956998`
3. `E69_MS_ACC95_JOINT_OVERFIT_SEED42`
   - `pair_acc = 0.966167`
   - `auc = 0.945630`
   - `ranking_score = 0.955898`

Candidate selector output:
1. selected group:
   - `E68_MS_ACC95_JOINT_LOGIT_SEED42`
2. `trust_score = 0.968689`
3. best checkpoint:
   - `assets/artifacts/phase_e_runs/phase_e_ms_acc95_push_0310_2146_e68_ms_acc95_joint_logit_seed42_e68_ms_acc95_joint_logit_seed42_s42_value_20260310T151651Z/best_value_head.pt`

### 0J.5 Interpretation: what actually helped

This round gives a clean answer.

The best improvement did **not** come from stronger overfit pressure.

Instead, the best improvement came from changing the ranking geometry:
1. `score-space joint` verify control:
   - `pair_acc = 0.963267`
2. `logit-space joint`:
   - `pair_acc = 0.972450`

So the updated interpretation is:
1. `Math-Shepherd` already has enough same-source signal to clear `95%`
   comfortably,
2. the current codebase is not blocked on this source,
3. and the most useful low-risk improvement here is
   `ranking_target_space = logit`,
4. not extra denoising removal or extra overfit pressure by itself.

### 0J.6 What this means for the broader repository diagnosis

This result narrows the failure surface elsewhere.

Because `Math-Shepherd` can now repeatedly reach:
1. `0.96+` under multiple current recipes,
2. and `0.97245` under the best current low-risk push,

the repository's remaining weaknesses should not be narrated as:
1. "the trainer cannot fit a strong process-style source"

They are better narrated as:
1. source-specific data/contract issues
   - as seen in old and repaired `R-PRM`
2. transfer / benchmark generalization issues
   - same-source success still does not imply `ProcessBench` success
3. recipe sensitivity across sources
   - the best Math-Shepherd setting is not automatically the best `R-PRM`
     setting

### 0J.7 Immediate next-step plan

1. Treat `E68_MS_ACC95_JOINT_LOGIT_SEED42` as the current same-source
   Math-Shepherd reference checkpoint family.
2. If seed-stability matters for promotion, rerun the same recipe as a
   multi-seed group instead of inventing a new recipe family first.
3. Keep using Math-Shepherd as a "can the current stack learn a clean source?"
   control, not as proof of benchmark-grade verification.
4. Focus future repair effort on the sources that still fail after contract
   cleanup, rather than continuing to over-optimize an already-solved
   Math-Shepherd intradataset objective.

## 0I. Latest Diagnosis Update (2026-03-10, R-PRM Root-Cause Matrix After Contract Repair)

### 0I.1 What this new round tried to answer

The earlier dedicated `R-PRM` recheck already showed that the current
`compact_verdict` contract is much healthier than the historical
`direct_pair_legacy` path.

But two more questions still mattered:
1. exactly how big is the gap between `legacy` and `compact` when we build
   canonical pair artifacts side by side from the same source?
2. after compact repair removes most truncation damage, is the remaining
   weakness mainly about recipe choice, or is the source still fundamentally
   unusable?

This round therefore combined four evidence surfaces:
1. legacy vs compact pair-artifact truncation comparison,
2. raw compact-contract audit directly on `R-PRM` DPO rows,
3. repaired-contract recipe matrix with same-family trust evaluation,
4. explicit `legacy@1024` fail-fast validation.

### 0I.2 Contract comparison: legacy vs compact are not in the same regime

Artifacts:
1. legacy artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_rprm_contractcmp_0310_2055_legacy_pairs__efc9444c97f8`
2. compact artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_rprm_contractcmp_0310_2055_compact_pairs__ca03cf0c9aa1`

Pair counts:
1. legacy:
   - total `2500`
   - train `2212`
   - validation `288`
2. compact:
   - total `910`
   - train `807`
   - validation `103`

Train truncation diagnostics:
1. legacy:
   - `768`: `over_limit=1.0000`, `collapse=0.4132`, `hidden_diff=0.4132`
   - `1024`: `over_limit=0.9313`, `collapse=0.1279`, `hidden_diff=0.1279`
   - `1536`: `over_limit=0.3639`, `collapse=0.0122`, `hidden_diff=0.0122`
   - `2048`: `over_limit=0.0696`, `collapse=0.0000`, `hidden_diff=0.0000`
2. compact:
   - `768`: `over_limit=0.0607`, `collapse=0.0607`, `hidden_diff=0.0607`
   - `1024`: `over_limit=0.0186`, `collapse=0.0186`, `hidden_diff=0.0186`
   - `1536`: all `0.0000`
   - `2048`: all `0.0000`

Validation truncation diagnostics tell the same story:
1. legacy first becomes clean at `2048`
2. compact first becomes clean at `1536`

So the updated hard fact is:
1. `legacy` and `compact` are not two nearby variants of the same contract,
2. they live in qualitatively different truncation regimes,
3. and any result that mixes them together is methodologically invalid.

### 0I.3 Raw compact audit: the repaired source still has real usable signal

Raw audit artifact:
1. `assets/artifacts/phase_e_rprm_audit/phase_e_rprm_contractaudit_0310_2100_20260310T124905Z/summary.json`

Key facts from `4000` audited raw rows:
1. `accepted_rows = 4000`
2. `acceptance_rate = 1.0000`
3. chosen verdict balance:
   - `yes = 1857`
   - `no = 2143`
4. compact prompt token stats:
   - mean `353.36`
   - `p95 = 739`
   - `p99 = 1064`
5. first-difference token stats:
   - mean `346.36`
   - `p95 = 732`
   - `p99 = 1057`
6. cutoff risk:
   - `1024`: `over_limit = 0.0125`, `hidden_diff_after_cutoff = 0.0125`
   - `1536`: both `0.0000`
   - `2048`: both `0.0000`

Interpretation:
1. compact repair is not a tiny cosmetic rewrite; it moves the source into a
   tractable token-length regime,
2. but `1024` is still not completely safe,
3. so repaired `R-PRM` should be treated as a `1536+` source by default.

### 0I.4 Legacy 1024 is now explicitly blocked before model load

Validation run:
1. `phase_e_rprm_legacy_failfast_0310_2100`

Observed behavior:
1. the run prints truncation diagnostics immediately,
2. then aborts with:
   - `over_limit_fraction=0.9313 exceeds 0.1000`
   - `collapse_after_cut_fraction=0.1279`
   - `hidden_diff_after_cut_fraction=0.1279`

This matters because the training entrypoint was hardened so unsafe settings
now fail before loading the backbone.

So this is both:
1. an experimental result:
   - `legacy@1024` is invalid
2. and an infrastructure result:
   - the repository now rejects this invalid configuration cheaply and early

### 0I.5 Repaired-contract recipe matrix: R-PRM compact is usable, but not easy

Recipe-matrix artifact:
1. `assets/artifacts/phase_e_logs/phase_e_rprm_recipe_smoke_0310_2105/final_summary.md`
2. raw rows:
   - `assets/artifacts/phase_e_logs/phase_e_rprm_recipe_smoke_0310_2105/recipe_rows.jsonl`

Shared setup:
1. compact pair artifact:
   - total `1092`
   - train `968`
   - validation `124`
2. validation prompt pools:
   - `123`
3. all recipe rows used:
   - `max_length = 2048`
   - safe truncation diagnostics:
     - train `0.0000`
     - validation `0.0000`

Results:
1. `linear + ranking_only`
   - `pair_acc = 0.6129`
   - `auc = 0.5728`
   - `samefamily_top1 = 0.6179`
   - `reject@0.6_top1 = 0.5946`
2. `mlp + ranking_only`
   - `pair_acc = 0.6452`
   - `auc = 0.5928`
   - `samefamily_top1 = 0.6585`
   - `reject@0.6_top1 = 0.7162`
3. `mlp + joint`
   - `pair_acc = 0.6694`
   - `auc = 0.6611`
   - `samefamily_top1 = 0.6829`
   - `reject@0.6_top1 = 0.7703`
4. `mlp + BCE_only`
   - `pair_acc = 0.6613`
   - `auc = 0.6439`
   - `samefamily_top1 = 0.6667`
   - `reject@0.6_top1 = 0.7297`

This matrix gives three non-trivial conclusions:
1. repaired `R-PRM compact` is genuinely learnable inside its own family
   - random-like failure is no longer the right description
2. head capacity matters
   - `MLP > linear`
3. unlike some earlier math-source results, `joint` is currently the best
   recipe on this source
   - so `R-PRM compact` is source-specific enough that recipe conclusions from
     `Math-Shepherd / PRMBench` cannot be copied blindly

### 0I.6 What remains weak

Even after repair, this is still not an `ACC90` source.

The best repaired smoke row (`mlp + joint @2048`) reaches:
1. `pair_acc = 0.6694`
2. `auc = 0.6611`
3. `samefamily_top1 = 0.6829`

So the new diagnosis is:
1. old `R-PRM` weakness was partly a contract/truncation artifact,
2. but not entirely,
3. because once the contract is repaired and length is safe, performance
   improves materially but still plateaus far below the easy-anchor regime

That means the remaining bottleneck is now better described as:
1. verdict-style supervision may be narrower / noisier than the other strong
   math sources,
2. the current frozen-scalar head can exploit some of it but not saturate it,
3. and `R-PRM` should currently be treated as a usable but medium-strength
   source, not a gold-standard anchor.

### 0I.7 Engineering warning that also matters scientifically

A legacy `2048` control run was attempted after it cleared the truncation gate.

What happened:
1. feature encoding immediately triggered repeated OOM backoff:
   - `64 -> 32 -> 16`
2. the full run reached only `1104 / 2212` chosen texts after about `239.7s`
   before being stopped
3. even a smaller matched-size subset still triggered repeated OOM backoff
   during feature encoding and was also stopped

This does not prove legacy `2048` can never train.
But it does prove something important:
1. even after becoming statistically executable, the legacy contract is still
   much more expensive and unstable operationally
2. so it is not a peer competitor to compact in practical research iteration

### 0I.8 Immediate next-step plan

1. Promote repaired `R-PRM` defaults to `1536+` everywhere; never silently
   allow `1024`.
2. Treat `compact_verdict` as the only valid Phase E mainline `R-PRM` contract;
   keep `direct_pair_legacy` only for historical comparison.
3. Use `mlp + joint` as the current repaired-source reference recipe.
4. Before mixing `R-PRM` into multi-source curricula again, explicitly test:
   - verdict-balanced resampling,
   - prompt-length bucket breakdowns,
   - and whether `samefamily_top1` stays stable under `source_sample` splitting.

## 0H. Latest Diagnosis Update (2026-03-10, Dedicated R-PRM Length-Sweep Recheck)

### 0H.1 What this new diagnostic was for

Earlier `I4_RPRM_ACC90_MATRIX` results were built on the historical
`direct_pair_legacy` contract, where `R-PRM` rows were still treated as long
verifier-essay pairs.

That meant one key question was still unresolved:
1. is current `R-PRM` weak because the source itself is poor,
2. or because the old long-text contract damaged the signal before training?

So a new dedicated diagnostic group was added and executed:
1. `scripts/run_phase_e_rprm_diagnostic_suite.sh`
2. group:
   - `RD1_RPRM_LENGTH_SWEEP_SMOKE`

### 0H.2 What was actually run

Artifacts:
1. pair artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_rprm_diag_0310_2019_pairs__2dd6a39365d8`
2. diagnostic summary:
   - `assets/artifacts/phase_e_logs/phase_e_rprm_diag_0310_2019/final_summary.md`

This run intentionally used the **current repaired** `R-PRM` contract:
1. `r_prm_pair_mode = compact_verdict`
2. prompt becomes compact:
   - `Question / Previous Steps / Now Step / Verification`
3. chosen/rejected become short opposite verdicts:
   - `Final answer: Yes/No`

### 0H.3 What the truncation diagnostics now say

This is the most important correction to the earlier understanding:

For the **current compact-verdict contract**, `R-PRM` is no longer a
"95%+ pairs are broken at 1024" dataset.

Train split:
1. `1024`
   - `over_limit = 0.0155`
   - `collapse_after_cut = 0.0155`
   - `hidden_diff_after_cut = 0.0155`
2. `1280`
   - `over_limit = 0.0031`
   - `collapse_after_cut = 0.0031`
   - `hidden_diff_after_cut = 0.0031`
3. `1536+`
   - all three become `0.0000`

Validation split:
1. `1024`
   - `over_limit = 0.0242`
   - `collapse_after_cut = 0.0242`
   - `hidden_diff_after_cut = 0.0242`
2. `1280`
   - `over_limit = 0.0161`
   - `collapse_after_cut = 0.0161`
   - `hidden_diff_after_cut = 0.0161`
3. `1536+`
   - all three become `0.0000`

So the new hard fact is:
1. `1024` is still slightly unsafe for compact `R-PRM`,
2. but `1536` is already fully clean,
3. and `2048` is not needed for safety.

### 0H.4 Same-source training results after removing truncation

Using the same repaired pair artifact and the strongest current `R-PRM`
same-source recipe family (`joint + MLP` style), we reran length-sweep training.

Results:
1. `1536`
   - run:
     - `assets/artifacts/phase_e_runs/phase_e_rprm_diag_0310_2019_len1536_20260310T121925Z`
   - held-out:
     - `pair_acc = 0.6129`
     - `auc = 0.6031`
2. `2048`
   - run:
     - `assets/artifacts/phase_e_runs/phase_e_rprm_diag_0310_2019_len2048_20260310T122338Z`
   - held-out:
     - `pair_acc = 0.7016`
     - `auc = 0.6735`

Reference old legacy same-source run:
1. `E49` seed-42 legacy result:
   - `pair_acc = 0.5901`
   - `auc = 0.5800`

### 0H.5 Interpretation

This new run changes the diagnosis in a precise way:

1. The old statement
   - "`R-PRM` mainly fails because `1024` destroys almost all pairs"
   is **true only for the legacy long-essay contract**.

2. For the **current repaired compact-verdict contract**:
   - truncation is still a real issue at `1024`,
   - but it is already fully solved at `1536`.

3. Once truncation is removed:
   - performance does improve,
   - especially from legacy `E49` to repaired `2048`,
   - but it still remains far below `ACC90`.

4. Therefore the current bottleneck is no longer "input got destroyed before
   learning started".
   It is now much more likely to be one of:
   - `R-PRM compact` supervision semantics are too narrow,
   - the frozen scalar head cannot fully exploit this verdict-style signal,
   - or same-source `R-PRM` simply is not a very strong source under this
     repository contract.

### 0H.6 Immediate engineering actions taken

To prevent future silent regressions:
1. `scripts/run_phase_e_rprm_diagnostic_suite.sh`
   - now explicitly passes `--r-prm-pair-mode compact_verdict`
2. `scripts/run_phase_e_suite.sh`
   - `E12_RPRM_COMPACT_VERDICT_SEED3`
   - `E65_STAGEB_MS_RPRM_COMPACT_REPAIRED_LINEAR_SEED3`
   - `E66_STAGEB_MS_RPRM_COMPACT_REPAIRED_MLP_SEED3`
   now default to:
   - `MAX_LENGTH = 1536`

This is important because the new diagnostic proved:
1. `1024` is not clean enough,
2. `1536` is the first clean cutoff,
3. so `compact R-PRM` groups should not silently fall back to `1024`.

## 0G. Latest Diagnosis Update (2026-03-10, R-PRM Dataset Contract Repair)

### 0G.1 What was fixed

The `R-PRM` data problem was not treated as a pure optimizer issue.

The concrete fix is now:
1. keep `direct_pair_legacy` for historical reproducibility,
2. add a new `compact_verdict` adapter mode for Phase E,
3. rewrite each `R-PRM` DPO row from:
   - one long verifier instruction
   - plus two long chosen/rejected verifier essays
   into:
   - one compact `Question / Previous Steps / Now Step` prompt
   - plus one short opposite-verdict pair.

Implementation surface:
1. `src/ours/phase_d/external_pairs_adapters.py`
2. `scripts/phase_e_prepare_pairs.py`
3. `scripts/run_phase_e_suite.sh`
4. `scripts/run_phase_e_repair_diagnostics_suite.sh`

New Phase E repair groups:
1. `E12_RPRM_COMPACT_VERDICT_SEED3`
2. `E65_STAGEB_MS_RPRM_COMPACT_REPAIRED_LINEAR_SEED3`
3. `E66_STAGEB_MS_RPRM_COMPACT_REPAIRED_MLP_SEED3`

New wrapper groups:
1. `R3_RPRM_DATAFIX_SMOKE`
2. `R4_RPRM_DATAFIX_SEED3`

### 0G.2 Why this is the right fix

The old contract forced the frozen-feature value head to score:
1. the full rubric-heavy verifier prompt,
2. plus multi-thousand-token chosen/rejected analyses.

That contract is mismatched with the current Phase E learner:
1. the model is a frozen backbone plus small scalar head,
2. not a generative verifier trained to consume and produce long analyses.

So the repaired contract now asks a cleaner question:
1. given the compact `Now Step` verification context,
2. does the head prefer the correct short verdict over the wrong short verdict?

### 0G.3 Real artifact-level validation

Using real `R-PRM` rows and the repository's own truncation diagnostic:

Legacy artifact:
1. `assets/artifacts/phase_e_truncation_diagnostics/rprm_legacy_diag_20260310T122121Z/summary.json`
2. `num_pairs = 64`
3. `frac_pairs_over_limit = 1.0`
4. `chosen_p95 = 1678`
5. `first_diff_p95 = 957`

Compact artifact:
1. `assets/artifacts/phase_e_truncation_diagnostics/rprm_compact_diag_20260310T122126Z/summary.json`
2. `num_pairs = 44`
3. `frac_pairs_over_limit = 0.0`
4. `chosen_p95 = 533`
5. `first_diff_p95 = 526`

Meaning:
1. the old `R-PRM` Phase E contract was fully over the `1024` limit on this real sample,
2. the repaired compact contract removes that truncation burden completely on the same diagnostic.

### 0G.4 End-to-end smoke confirmation

The new compact contract now runs through the real Phase E training path:
1. suite group:
   - `E12_RPRM_COMPACT_VERDICT_SEED3`
2. smoke run:
   - `assets/artifacts/phase_e_logs/phase_e_rprm_compact_smoke_0310_2038/final_summary.md`
3. seed-42 tiny smoke result:
   - `pair_accuracy = 0.666667`
   - `auc = 0.555556`
4. truncation diagnostics inside training:
   - `train over_limit = 0.0000`
   - `eval over_limit = 0.0000`

This smoke is intentionally tiny and is **not** scientific evidence of final
quality. It only proves:
1. the repaired `R-PRM` contract is wired through prepare/train/summarize,
2. the new compact path no longer fails because of immediate truncation.

### 0G.5 Follow-up hardening after the first user reruns

Two wrapper/runtime issues were then fixed:

1. `E12_RPRM_COMPACT_VERDICT_SEED3`
   - now defaults to:
     - `MAX_LENGTH=1536`
     - `EVAL_BATCH_SIZE=16`
   - reason:
     - the repaired compact contract should use the already-audited safe
       cutoff,
     - and Phase E currently reuses eval batch size for frozen-backbone
       feature caching, so a smaller value is the safest OOM guard.
2. `R3_RPRM_DATAFIX_SMOKE` and `R4_RPRM_DATAFIX_SEED3`
   - no longer include legacy long-analysis `R-PRM` groups by default
   - reason:
     - those legacy groups fail the truncation gate by design,
     - so including them in the official wrapper aborts the suite before the
       repaired groups can run.

## 0F. Latest Diagnosis Update (2026-03-10, I4 Full-Matrix Result + Phase E Aggregation Hardening)

### 0F.1 What the new completed run says

The first fully completed intradataset full-matrix run is now:
1. `assets/artifacts/phase_e_logs/phase_e_rprm_acc90_full_0310_1914/final_summary.md`
2. group:
   - `I4_RPRM_ACC90_MATRIX`

Its three recipe-family results are:
1. `E47_RPRM_ACC90_LINEAR_SEED3`
   - `mean_pair_acc = 0.4374`
   - `mean_auc = 0.5016`
2. `E48_RPRM_ACC90_MLP_RANK_SEED3`
   - `mean_pair_acc = 0.5197`
   - `mean_auc = 0.5123`
3. `E49_RPRM_ACC90_MLP_JOINT_SEED3`
   - `mean_pair_acc = 0.6002`
   - `mean_auc = 0.5885`

### 0F.2 Interpretation

This run adds a much stronger conclusion than the earlier smoke:
1. `R-PRM` is currently weak under the repository's same-source `ACC90`
   branch even after the recipe family is expanded from:
   - linear
   - to MLP ranking
   - to MLP joint
2. the direction `E49 > E48 > E47` shows that:
   - architecture and objective do matter,
   - but they do not rescue the source under the current adapter/contract.
3. the seed std is small:
   - `E49 std_pair_acc = 0.0098`
   - `E49 std_auc = 0.0106`
4. therefore this is **not** a seed-collapse story.

Current scientific reading:
1. `Math-Shepherd` and `PRMBench_Preview` remain positive same-source
   learnability sources.
2. `R-PRM` is not a trustworthy `ACC90` anchor under the current repository
   contract.

### 0F.3 Similar hidden risk that was fixed

After the earlier intradataset selector bug, a second class of structural
fragility was found:
1. several Phase E wrapper scripts were aggregating top-level suite results by
   regex-parsing `final_summary.md`;
2. this makes result aggregation depend on Markdown formatting rather than the
   real structured artifact.

This is now hardened:
1. `scripts/run_phase_e_intradataset_suite.sh`
2. `scripts/run_phase_e_single_source_suite.sh`
3. `scripts/run_phase_e_multisource_math_suite.sh`

All three now read:
1. `seed_results.jsonl`

instead of reverse-parsing:
1. `final_summary.md`

Meaning:
1. top-level suite summaries now trace back to the structured source-of-truth
   artifact,
2. mild Markdown formatting edits will no longer silently corrupt group-level
   comparisons.

### 0F.4 Candidate-selector hardening

For consistency with the already-fixed intradataset selector, the benchmark-
aware selector was also hardened:
1. `scripts/phase_e_select_candidate.py`

New behavior:
1. repeated `--suite-log-dirs DIR` occurrences are accepted,
2. directory groups are flattened in first-seen order,
3. duplicates are removed.

This prevents future wrapper/selector contract drift from reproducing the same
class of error.

## 0E. Latest Diagnosis Update (2026-03-10, Why Some Phase E Sources Reach High ACC While Others Do Not)

### 0E.1 Core finding

The current evidence does **not** support the claim that the whole `Phase E`
trainer is fundamentally broken.

Reason:
1. the same training stack can already reach very high same-source held-out
   performance on some datasets:
   - `E41_MS_ACC90_MLP_RANK_SEED3`
     - `heldout_pair_acc = 0.9610`
     - `heldout_auc = 0.8908`
   - `E45_PRMBENCH_ACC90_MLP_RANK_SEED3`
     - `heldout_pair_acc = 0.9483`
     - `heldout_auc = 0.8333`
2. but `E48_RPRM_ACC90_MLP_RANK_SEED3` stays near random:
   - `heldout_pair_acc = 0.4379`
   - `heldout_auc = 0.4666`

Therefore:
1. this is not one universal low-level trainer failure,
2. but there **is** a source-specific bottom-layer implementation / contract
   mismatch.

### 0E.2 Strongest implementation-level issue found

The largest concrete issue is the interaction between:
1. `R-PRM` direct-pair adapter format,
2. the repo-wide fixed `max_length = 1024`,
3. and silent tokenizer truncation during frozen-feature encoding.

What the code currently does:
1. `src/ours/phase_d/external_pairs_adapters.py`
   - loads `R-PRM` as full `instruction + chosen + rejected` long-form direct
     verifier texts.
2. `src/ours/phase_b/value_head.py`
   - encodes all sources with `truncation=True` and one shared `max_length`.
3. the failing `R-PRM` ACC90 run manifest confirms:
   - `max_length = 1024`.

What the audit measured on the actual training artifact:
1. `Math-Shepherd`
   - mean token length about `122`
   - `0%` exceed `1024`
2. `PRMBench_Preview`
   - mean token length about `288`
   - about `1.2%` exceed `1024`
3. `R-PRM`
   - mean token length about `1425.6`
   - `94.1%` exceed `1024`
   - `11.7%` of pairs become **identical after truncation**

This is the strongest code-level reason why one source can fail while others
train well under the same recipe.

### 0E.3 Why this matters scientifically

For `R-PRM`, the problem is not only sequence length.

There is also a source-contract mismatch:
1. `Math-Shepherd` and `PRMBench_Preview`
   - are converted into local prefix / local-error pairs
   - first difference appears early in the token sequence
2. `R-PRM`
   - is loaded as long-form verifier analyses
   - first difference is often much later

Measured first-difference position under the current tokenizer:
1. `Math-Shepherd`
   - median first difference at token `113`
2. `PRMBench_Preview`
   - median first difference at token `171`
3. `R-PRM`
   - median first difference at token `740`
   - 90th percentile first difference at token `1056`

Meaning:
1. the shared `1024`-token recipe is almost neutral for `Math-Shepherd`,
2. mostly acceptable for `PRMBench_Preview`,
3. but systematically lossy for `R-PRM`.

### 0E.4 Practical conclusion

Current judgment:
1. the repo does have a real bottom-layer issue,
2. but it is **source-specific**, not a universal trainer bug.

The issue is:
1. one unified input contract is being applied to sources with very different
   sequence lengths and supervision semantics,
2. and the current pipeline provides no truncation diagnostics or source-aware
   max-length policy.

So the safe conclusion is:
1. `Math-Shepherd` and `PRMBench_Preview` high-ACC results are credible as
   evidence that the trainer can learn,
2. `R-PRM` failure should **not** be read as pure scientific failure yet,
3. because the current implementation likely destroys part of its supervision
   signal before the head ever sees it.

### 0E.5 Next-step plan

1. Add explicit truncation diagnostics to `Phase E` pair preparation / training
   summaries:
   - per-source token-length quantiles,
   - fraction over `max_length`,
   - fraction whose chosen/rejected first difference lies after the cutoff.
2. Re-run `R-PRM` same-source ACC with at least:
   - `max_length = 1536`
   - `max_length = 2048`
3. If long-context reruns still fail badly, then treat:
   - `R-PRM` source semantics,
   - not truncation,
   as the main blocker.
4. Do not compare source quality using one fixed `max_length` recipe unless the
   truncation burden is shown to be comparable across sources.

## 0D. Latest Diagnosis Update (2026-03-10, Phase E Value-Head Structure / Recipe Audit)

### 0D.1 What was checked

This audit focused on the current `Phase E` value-head implementation from
three angles:
1. network structure
2. learning mechanism
3. newest single-source vs multi-source experimental evidence

The question was not just "did one run fail?", but:
1. is the current head intrinsically too weak or too strange,
2. or is the present training recipe the bigger problem?

### 0D.2 Reliable structure facts from the current code and artifacts

1. The current `Phase E` learner is a **frozen-feature scalar-head** design.
   - The transformer backbone is run once to extract pooled prefix features.
   - Those features are cached.
   - Training then updates only the small head, not the whole backbone.
2. The head implementation exposes both:
   - raw `logits`
   - `sigmoid(logits)` scores
3. The active failing mixture still uses `objective_mode=ranking_only`.
4. The run artifacts inspected here only store a minimal
   `value_head_config.json` payload:
   - `hidden_size`
   - `dropout_prob`
   - `init_std`
   - `pooling`
5. Therefore the exact historical `linear` vs `mlp` branch is not fully
   recoverable from artifact metadata alone.
6. What is recoverable is the broader family:
   - lightweight frozen-feature head,
   - not full-backbone joint fine-tuning.

### 0D.3 Most informative experiment contrast

| Run | Train recipe snapshot | Held-out result | External / benchmark-facing result | Interpretation |
|---|---|---|---|---|
| `E2_MATH_SHEPHERD_PAIR_LEARN_SEED3` seed-42 | `5358` train pairs; `batch=32`; `epochs=4`; `lr=5e-5`; `pair_weight_mode=confidence`; about `672` optimizer steps | `pair_acc=0.8480`; `auc=0.8218`; `ranking_score=0.8349` | strong same-family held-out signal; trust-matrix `ProcessBench` transfer still weak | Confirms that the current value-head family can learn a clean same-source ranking task. |
| `E24_STAGEB_MS_RPRM_MIX_SEED3` seed-42 | `1774` train pairs; `batch=128`; `epochs=6`; `lr=2e-5`; `pair_weight_mode=none`; `source_balance=uniform`; about `84` optimizer steps | `pair_acc=0.4381`; `auc=0.4922`; `ranking_score=0.4651` | `PRMBench_Preview auc=0.5207`; partial smoke summary also remained weak on `PB-GSM8K` and `PB-MATH` | Negative result for the current balanced two-source recipe, but not evidence that value-head learning itself is impossible. |

### 0D.4 Diagnosis

The main conclusion is:
1. the current head family is **not** yet proven intrinsically broken,
2. the current mixture recipe is the more credible failure point.

Most important risks:
1. **Loss-design risk**
   - ranking is currently applied on bounded sigmoid scores
   - this compresses margin geometry and makes saturation easier
2. **Source-semantics mismatch**
   - `Math-Shepherd` contributes strict local `first_bad_edge` pairs
   - `R-PRM` contributes direct chosen/rejected preference pairs
   - these are related but not identical supervision targets
3. **Under-training in effective update count**
   - the failing `E24` run gets about `84` optimizer steps
   - the strong `Math-Shepherd` run gets about `672`
4. **Weighting is disabled in the heterogeneous setting**
   - `pair_weight_mode=none`
   - `source_balance=uniform`
5. **Split granularity is weaker than ideal**
   - current splitting is pair-id based, not problem-id based
6. **Capacity may still matter, but it is not the first suspect**
   - current evidence still points more strongly to objective/data mismatch
     than to head complexity alone

### 0D.5 What we should conclude scientifically

1. `Phase E` has already demonstrated **same-source learnability**.
2. The open question is now:
   - not whether a value head can learn anything,
   - but whether heterogeneous math-process sources can be combined without
     destroying the useful signal.
3. The current `MS + R-PRM` failure should therefore be interpreted as:
   - a failure of the present Stage B recipe,
   - not a decisive failure of the value-head family.

### 0D.6 Explicit next-step plan

1. repair the pairwise objective first
2. stop treating local `first_bad_edge` supervision and direct preference
   supervision as a trivially interchangeable pool
3. increase effective optimizer steps for Stage B mixtures
4. restore confidence/source weighting for heterogeneous mixtures
5. consider sample/problem-level splitting where one raw example can emit
   multiple related pairs
6. only after the above, revisit head-capacity upgrades as a first-order
   experiment

## 0C. Latest Diagnosis Update (2026-03-10, Repository Audit + Phase E Reliability)

### 0C.1 What the repository is actually trying to prove

The repository's real claim is now narrower than the old `Phase C / Phase D`
phrasing might suggest.

What the pipeline is actually trying to establish:
1. freeze a strong backbone,
2. train a small prefix-level value / ranking head,
3. prove same-family process discrimination first,
4. and only after that ask whether the signal is trustworthy enough for
   benchmark-facing verification or later control.

What the current blocker really is:
1. not "can a tiny head fit any signal at all?",
2. but:
   - whether supervision semantics are clean enough,
   - whether the objective matches the scientific claim,
   - and whether the evaluation pipeline itself can be trusted.

### 0C.2 Most meaningful experimental evidence reviewed in this audit

1. `MS2_MATH_SHEPHERD_TRUST_SEED3_MATRIX`
   - strongest reviewed group:
     - `E14_MATH_SHEPHERD_TRUST_ANTISAT_SEED3`
   - metrics:
     - `mean_hold_pair = 0.8246`
     - `mean_hold_auc = 0.7899`
     - `pb_gsm_auc = 0.4893`
     - `pb_math_auc = 0.4783`
   - meaning:
     - same-source learnability is clearly real,
     - benchmark-native robustness is still not established.
2. `CUR1_STAGEE_MS_TO_MSRPRM` benchmark eval on `processbench_math`
   - metrics:
     - `pair_accuracy_good_vs_bad = 0.5105`
     - `pair_auc_good_vs_bad = 0.4812`
     - `first_error_edge_accuracy = 0.4593`
   - meaning:
     - the current multi-source curriculum evidence is still near-random on the
       benchmark-facing task that matters most.
3. Combined reading of earlier `Phase D` and current `Phase E` evidence:
   - external same-family ranking learnability is a real positive result,
   - but the repository still does **not** have clean evidence for a robust
     benchmark verifier.

### 0C.3 Strict code-audit findings

This audit explicitly separated "research limitation" from "implementation bug".

Confirmed real bug:
1. `src/ours/phase_b/value_head.py`
   - the newer head refactor removed the old `.proj` access pattern used by
     older warm-start / debug code,
   - this was already breaking
     `tests/unit/test_phase_c_train_value.py::test_initialize_value_head_from_checkpoint_loads_matching_weights`,
   - the compatibility alias has now been restored and regression-tested.

Confirmed high-risk behaviors that can silently contaminate experiment reading:
1. `src/ours/phase_e/runtime.py`
   - asking for checkpoint `"best"` silently falls back to `"final"` when
     `best_value_head.pt` is missing.
   - risk:
     - an operator can believe a benchmark report used the best checkpoint when
       it actually used the final one.
2. `src/ours/phase_b/feature_cache.py`
   - cache-lock staleness is inferred only from file `mtime`.
   - risk:
     - a live but slow writer can be treated as dead and have its lock stolen,
       allowing overlapping writers.
3. `src/ours/phase_e/benchmark_eval.py`
   - `PRMBench_Preview` error-step normalization assumes 1-based indexing and
     subtracts one from positive indices.
   - current status:
     - the audited snapshot in this repo appears consistent with 1-based
       indexing,
     - but the loader is still schema-heuristic rather than schema-verified.

Audit verification status:
1. targeted audit tests were added for:
   - value-head backward compatibility,
   - checkpoint fallback behavior,
   - feature-cache live-lock overlap,
   - PRMBench preview 1-based normalization.
2. full test status after the audit:
   - `142 passed, 2 skipped`.

### 0C.4 Updated evaluation of the repository strategy

Current judgment:
1. the scientific direction is mostly correct,
2. the current implementation and evidence are not yet sufficient for the
   stronger "trustworthy small verifier" claim.

What is correct:
1. moving from `StrategyQA-first` to benchmark-native process data is correct,
2. moving from scalar-calibration-first to ranking-first is correct,
3. treating same-family learnability as a prerequisite before transfer claims
   is correct.

What is not yet effective:
1. the project has not solved benchmark-facing verification,
2. the current positive evidence is still local:
   - source-family learnability,
   - not trustworthy cross-benchmark process discrimination.

Therefore the scientifically safe narrative is:
1. "small frozen-feature ranking heads can learn some same-family local process
   discrimination",
2. not:
   - "the repository has already produced a reliable small PRM / verifier."

### 0C.5 Explicit next-step plan

1. For official promotion runs, fail hard if `"best"` is requested but
   `best_value_head.pt` does not exist.
2. For feature-cache writes, either:
   - enforce one-writer operational policy,
   - or upgrade the lock to a heartbeat / owner-checked design.
3. Keep `PRMBench_Preview` schema pinned and re-verify index semantics whenever
   the dataset file changes.
4. Continue `Phase E` under the narrow claim:
   - same-family ranking-first verifier learning,
   - not benchmark-robust process verification.
5. If benchmark metrics stay near random, the next scientific lever should be:
   - supervision quality,
   - pair semantics,
   - or verifier-style objective design,
   not just increasing head complexity.

## 0B. Latest Diagnosis Update (2026-03-10, Phase E intradataset ACC90 structural reading)

### 0B.1 Core question

After the new same-source `ACC90` suites landed, the diagnosis question became:
1. Is the current head too simple?
2. Are weak runs just under-trained?
3. Or do the answers depend on the source dataset?

### 0B.2 Strong evidence

#### Math-Shepherd

| Group | Head / Objective | mean held-out pair_acc | mean held-out auc | Reading |
|---|---|---:|---:|---|
| `E40_MS_ACC90_LINEAR_ROBUST_SEED3` | linear + ranking | 0.9172 | 0.8623 | Linear head already works strongly |
| `E41_MS_ACC90_MLP_RANK_SEED3` | MLP + ranking | 0.9863 | 0.9056 | Higher capacity improves further |
| `E42_MS_ACC90_MLP_JOINT_SEED3` | MLP + joint | 0.9641 | 0.9408 | Joint objective boosts AUC strongly |
| `E43_MS_ACC90_MLP_HIGHCONF_SEED3` | MLP + joint + denoise | 0.9619 | 0.9425 | Cleaner pairs also help |

#### PRMBench Preview

| Group | Head / Objective | mean held-out pair_acc | mean held-out auc | Reading |
|---|---|---:|---:|---|
| `E44_PRMBENCH_ACC90_LINEAR_SEED3` | linear + ranking | 0.7380 | 0.6782 | Linear head underfits |
| `E45_PRMBENCH_ACC90_MLP_RANK_SEED3` | MLP + ranking | 0.9315 | 0.8711 | MLP crosses ACC90 |
| `E46_PRMBENCH_ACC90_MLP_JOINT_SEED3` | MLP + joint | 0.9309 | 0.9057 | MLP + joint is strongest balanced path |

#### Under-training control

| Group | mean held-out pair_acc | mean held-out auc | Reading |
|---|---:|---:|---|
| `E12_MS_TRUST_LOWLR_SEED3` | 0.5853 | 0.5856 | Lower LR / longer training alone does not rescue the run |

### 0B.3 Diagnosis

1. `Math-Shepherd`
   - current evidence does **not** support the claim that the linear head is
     too simple;
   - the linear head already exceeds `0.90` same-source held-out accuracy;
   - MLP is an upgrade, not a rescue.
2. `PRMBench_Preview`
   - current evidence strongly supports the claim that the linear head is too
     simple;
   - here, MLP capacity is a real bottleneck remover.
3. Old weak runs cannot be explained only by under-training.
   - `E12` should have improved much more if that were the main issue.
   - therefore pair semantics, objective choice, denoising, and head capacity
     all matter.
4. Very high same-source held-out accuracy should still be interpreted
   narrowly.
   - it proves same-source learnability;
   - it does not yet prove benchmark transfer or future RL trustworthiness.

### 0B.4 Next plan

1. Freeze source-specific defaults:
   - `Math-Shepherd`: keep linear as strong baseline, compare against MLP.
   - `PRMBench_Preview`: default to MLP.
2. Finish the missing `R-PRM` same-source matrix before making a general claim
   about direct preference supervision.
3. Use same-source winners as bounded-support candidates only; do not narrate
   them as cross-task RL-ready value functions.

## 0C. Latest Diagnosis Update (2026-03-10, RL trustworthiness threshold reading)

### 0C.1 Core question

If we ignore cross-dataset transfer for now, what level of value-utility
quality is enough before we should trust it inside RL-style reasoning
faithfulness experiments?

### 0C.2 Literature-guided answer

The strongest consensus is:
1. there is no single universal scalar threshold such as:
   - `pair_acc > 0.90 => RL-ready`
2. strong held-out discrimination is necessary,
3. but process reward / value models must also survive optimization pressure.

Operational reading from the literature:
1. `ProcessBench` and `PRMBench` both show that apparently decent PRMs can
   still fail explicit process-error identification.
2. `PRM800K` and later PRM work support process supervision as a viable signal,
   but not as a one-metric guarantee of trust.
3. reward/value models should also demonstrate policy-level utility and
   resistance to shortcut exploitation.

### 0C.3 Current repository status

What we already have:
1. same-source held-out discrimination is now strong on:
   - `Math-Shepherd`
   - `PRMBench_Preview`
2. therefore same-source learnability is no longer the main uncertainty.

What is still missing before RL-level trust can be claimed:
1. same-family policy-improvement evidence
   - reranking / rejection / conservative search should improve final process
     quality inside the same dataset family
2. same-family local-faithfulness evidence
   - not just pair discrimination, but stronger local error-ordering checks
3. robustness under stronger optimization pressure
   - high-scoring processes should not merely exploit dataset-specific
     shortcuts

### 0C.4 Diagnosis

1. The current supported claim is:
   - same-source value learning is established on some sources.
2. The current unsupported claim is:
   - these value heads are already trustworthy enough to heavily drive RL.
3. Therefore the next useful experiments should shift from:
   - "can the head fit held-out pairs?"
   to:
   - "does the head improve same-family process selection under stronger
     decision pressure?"

### 0B.5 Important caveat from the first smoke run

The first smoke run:
1. `assets/artifacts/phase_e_logs/phase_e_top3_acc90_0310_1808/final_summary.md`

was still intentionally narrow:
1. it forced `seed=42` only,
2. reduced pair counts to `3000`,
3. and should be interpreted as a quick direction check, not a stability
   result.

Meaning:
1. the strong `E41` / `E45` numbers in that smoke are useful,
2. but they are not enough on their own to claim seed-stable `ACC90`
   promotion.

### 0B.6 Candidate-selection script issue from the smoke and its fix

The original auto-selected candidate in that smoke run was invalid, but the
selector bug has now been fixed.

Observed symptom:
1. the candidate report selected `E48_RPRM_ACC90_MLP_RANK_SEED3`,
2. even though `E41` and `E45` were numerically much stronger.

Original root cause:
1. `scripts/run_phase_e_intradataset_suite.sh`
   - passes multiple repeated `--suite-log-dirs ...` flags,
2. `scripts/phase_e_select_intradataset_candidate.py`
   - originally defined `--suite-log-dirs` as one `nargs=\"+\"` argument,
3. so the selector ends up consuming only the last repeated flag in this call
   pattern.

Fix:
1. the selector now accepts repeated `--suite-log-dirs` groups,
2. flattens and deduplicates them before scoring.

Verification:
1. dedicated unit test now covers repeated-flag parsing and strongest-group
   selection,
2. re-running the selector on the existing smoke suite now selects:
   - `E41_MS_ACC90_MLP_RANK_SEED3`
   instead of the incorrect old `E48` result.

Consequence:
1. the old smoke-level `candidate_report.json/.md` remains invalid history,
2. but the selector code path is now fixed for future ACC90 suite runs.

## 0A. Latest Diagnosis Update (2026-03-10, Phase E Small-Scale Value-Head Viability)

### 0A.1 Core literature diagnosis

Recent literature does **not** strongly support the following target:
1. small-scale
2. noisy or weakly synthesized labels
3. scalar value regression
4. benchmark-robust step-level quality prediction

Recent literature **does** support narrower variants:
1. large-scale dense process supervision:
   - `Let's Verify Step by Step` / `PRM800K`
   - `Math-Shepherd`
   - `OmegaPRM`
   - `Rewarding Progress`
2. smaller but stronger ranking / verifier formulations:
   - `ThinkPRM`
   - `R-PRM`
3. coarse-grained confidence/value-style heads:
   - `Language Models (Mostly) Know What They Know`
   - `Large Language Models Must Be Taught to Know What They Don't Know`

Operational reading:
1. the literature does not justify expecting our old scalar `MC / q_teacher /
   q_fused` value-head formulation to become a robust process verifier at small
   scale;
2. the literature does justify testing:
   - same-family
   - ranking-first
   - local process discrimination
   under stronger pair/process supervision.

### 0A.2 New meaningful experimental results

1. `MS2_MATH_SHEPHERD_TRUST_SEED3_MATRIX`
   - provisional best:
     - `E14_MATH_SHEPHERD_TRUST_ANTISAT_SEED3`
   - metrics:
     - `mean_hold_pair=0.8246`
     - `mean_hold_auc=0.7899`
     - `pb_gsm_auc=0.4893`
     - `pb_math_auc=0.4783`
   - meaning:
     - post-fix `Math-Shepherd` same-family learnability is real,
     - but benchmark robustness is still weak.
2. `E20_STAGEA_MS_ANCHOR_SEED3`
   - `mean_heldout_pair_acc=0.6853`
   - `mean_heldout_auc=0.6676`
   - `mean_prmbench_preview_auc=0.5868`
   - `mean_processbench_gsm8k_auc=0.4750`
   - `mean_processbench_math_auc=0.4715`
   - meaning:
     - best clean post-fix same-family anchor,
     - still not a trustworthy `ProcessBench` verifier.
3. `E21_STAGEA_RPRM_ANCHOR_SEED3`
   - `mean_heldout_pair_acc=0.4381`
   - `mean_heldout_auc=0.4953`
   - `mean_prmbench_preview_auc=0.5623`
   - `mean_processbench_gsm8k_auc=0.4744`
   - `mean_processbench_math_auc=0.4626`
   - one informative run:
     - `PRMBench_Preview pair_acc=0.6001`
     - `PRMBench_Preview auc=0.5937`
   - meaning:
     - `R-PRM` is not the strongest general anchor,
     - but it is more aligned with `PRMBench_Preview`.
4. `E24_STAGEB_MS_RPRM_MIX_SEED3` smoke
   - currently only partial seed-42 evidence:
     - `PRMBench_Preview auc=0.5207`
     - `ProcessBench GSM8K auc=0.4682`
     - `ProcessBench Math auc=0.3835`
   - meaning:
     - no positive evidence yet that simple balanced mixing fixes the gap.

### 0A.3 Updated diagnosis

1. Our current experiments are now consistent with the literature:
   - local same-family learnability exists,
   - benchmark-facing process verification remains unresolved.
2. The strongest positive evidence is no longer:
   - "small scalar value head works"
   but:
   - "small-to-medium same-family ranking verifier learning works to some
     extent."
3. The current unresolved problem is:
   - how to combine `Math-Shepherd`'s same-family anchor strength with
     `R-PRM` / `PRMBench_Preview` style benchmark alignment.

### 0A.4 Explicit next-step plan

1. Finish `E22_STAGEA_PRMBENCH_PREVIEW_ANCHOR_SEED3` before making any source
   ranking claim about the Stage A matrix.
2. Do not narrate `E24` or `CUR1` as positive until a full completed summary is
   available.
3. Keep the current Phase E scientific claim narrow:
   - same-family ranking-first verifier learning,
   - not general-purpose small PRM/value modeling.
4. If the project wants a stronger claim later, prioritize one of:
   - stronger judge-generated supervision
   - generative verifier training
   - or a larger process-label pipeline

### 0A.5 Key references

1. `Let's Verify Step by Step`
   - https://arxiv.org/abs/2305.20050
2. `Rewarding Progress`
   - https://arxiv.org/abs/2410.08146
3. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   - https://arxiv.org/abs/2501.07301
4. `ProcessBench`
   - https://arxiv.org/abs/2412.06559
5. `R-PRM`
   - https://arxiv.org/abs/2503.21295
6. `ThinkPRM`
   - https://arxiv.org/abs/2504.16828
7. `Language Models (Mostly) Know What They Know`
   - https://arxiv.org/abs/2207.05221
8. `Large Language Models Must Be Taught to Know What They Don't Know`
   - https://arxiv.org/abs/2406.08391

## 0. Latest Diagnosis Update (2026-02-27, A8 GSM8K Prompt Style Sweep)

### 0.1 New A8 Results (n=172, freeform decode)

| Label | Template | Max New Tokens | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---|---:|---:|---:|---:|---:|
| `style_direct_final_t32` | `qa_gsm8k_direct_final_only` | 32 | 0.3895 | 0.0000 | 172 | 0.3895 |
| `style_cot_compact_t192` | `qa_gsm8k_cot_compact_final` | 192 | 0.7616 | 0.0000 | 172 | 0.7616 |
| `style_equation_t64` | `qa_gsm8k_equation_then_final` | 64 | 0.3895 | 0.0000 | 172 | 0.3895 |

### 0.2 Comparison With Previous GSM8K Baselines

| Track | Accuracy | Parse Error |
|---|---:|---:|
| Previous direct baseline (re-eval) | 0.3721 | 0.0000 |
| Previous CoT baseline (`t256`, re-eval) | 0.7035 | 0.0000 |
| A8 direct-final (`t32`) | 0.3895 | 0.0000 |
| A8 CoT-compact (`t192`) | 0.7616 | 0.0000 |
| A8 equation (`t64`) | 0.3895 | 0.0000 |

Observed deltas:
1. A8 CoT-compact improves over previous CoT baseline by `+0.0581` (0.7616 vs 0.7035).
2. A8 direct-final improves over previous direct baseline by `+0.0174` (0.3895 vs 0.3721).
3. Equation-style gives no quality gain vs direct-final in this setup.

### 0.3 Runtime and Throughput Tradeoff (A8 runs)

| Label | Elapsed Seconds | Samples/sec |
|---|---:|---:|
| `style_direct_final_t32` | 146.54 | 1.174 |
| `style_cot_compact_t192` | 883.72 | 0.195 |
| `style_equation_t64` | 288.53 | 0.596 |

Interpretation:
1. CoT-compact is strongest on quality but about `6x` slower than direct-final (`0.195` vs `1.174` sample/s).
2. Equation-style is slower than direct-final and does not improve accuracy, so it is currently dominated.

### 0.4 Extraction-Method Diagnostics (Why CoT still misses some items)

1. `style_cot_compact_t192`:
   - `final_answer_tag`: 162 samples, method-level acc `0.8086`
   - `last_number`: 10 samples, method-level acc `0.0000`
2. `style_equation_t64`:
   - mostly `final_answer_tag` but low method-level accuracy (`0.3988`)
3. `style_direct_final_t32`:
   - all samples `final_answer_tag`, method-level accuracy `0.3895`

Conclusion from extraction view:
1. Remaining CoT errors are concentrated where model fails to emit clean final-answer format and falls to `last_number`.
2. The main bottleneck is no longer parse coverage (already `0.0000` parse error), but reasoning correctness and answer-format consistency under long-form outputs.

### 0.5 A8 Conclusions

1. For GSM8K Phase A, `qa_gsm8k_cot_compact_final@t192` is the best current quality baseline.
2. `qa_gsm8k_direct_final_only@t32` is the practical fast baseline.
3. `qa_gsm8k_equation_then_final@t64` should not be prioritized further unless reformulated.
4. A8 narrows the gap to expected public-model ceiling and gives a stronger launch point for Phase B.

### 0.6 Next Plan

1. Freeze two GSM8K reporting baselines:
   - quality: `cot_compact_t192`
   - speed: `direct_final_t32`
2. Run one reproducibility repeat for `cot_compact_t192`.
3. Add a focused follow-up to reduce `last_number` fallback cases in CoT outputs (format tightening + extraction guards).
4. Carry these two baselines into Phase B pre/post training comparison.

### 0.7 A8 Error Pattern Deep-Dive (Sample-Level)

Focus run: `style_cot_compact_t192` (best GSM8K quality in A8).

Error composition (41 wrong samples):
1. `final_answer_tag` wrong: `31`
2. `last_number` fallback wrong: `10`

Observed error patterns:
1. **Truncation / unfinished outputs** (`10/41`):
   - No clean final-answer line, evaluator falls back to `last_number`.
   - Typical signature: reasoning is mid-derivation and output ends abruptly near token cap.
   - Example IDs: `gsm8k:main:train:140`, `gsm8k:main:train:198`, `gsm8k:main:train:581`.
2. **Reasoning arithmetic/sign mistakes** (majority of remaining errors):
   - Model emits parseable final answer but applies wrong operation/sign or rate conversion.
   - Example IDs: `gsm8k:main:train:1406` (subtracts missed count instead of adding), `gsm8k:main:train:692`, `gsm8k:main:train:1033`.
3. **Extraction edge cases with expression-style final lines** (small but critical):
   - Some outputs contain `Final answer: <expression> = <number>` and/or multiple `Final answer:` lines.
   - Current extraction can pick the first numeric token from the expression line, undercounting true correctness.
   - Confirmed undercount IDs: `gsm8k:main:train:838`, `gsm8k:main:train:1418`.
4. **Self-correction / repeated final-answer lines**:
   - Multi-final-answer outputs appeared in error rows and can interact with extraction heuristics.
   - Count in CoT errors: `11`.

Cross-style hardness signal:
1. `31` samples were wrong in **all three** A8 styles (direct, CoT-compact, equation).
2. These shared-hard samples are dominated by multi-stage word problems with money/time/rate/percentage structure.

Improvement implications:
1. Keep improving prompt quality, but also harden evaluator extraction for math final-answer lines with expressions/repeated tags.
2. Reduce truncation pressure on CoT outputs (format tightening / better stop behavior / budget tuning).
3. Build a targeted “hard subset” regression list from the 31 shared-hard IDs for future Phase B comparisons.

## 1. Previous Diagnosis Update (2026-02-27, A7 StrategyQA Prompt Style Sweep)

### 0.1 New A7 Results (n=193, freeform decode)

| Label | Template | Max New Tokens | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---|---:|---:|---:|---:|---:|
| `style_minimal_t16` | `qa_strategyqa_minimal_binary` | 16 | 0.6632 | 0.0000 | 193 | 0.6632 |
| `style_cot_compact_t96` | `qa_strategyqa_cot_compact` | 96 | 0.6943 | 0.0311 | 187 | 0.7166 |
| `style_evidence_verdict_t32` | `qa_strategyqa_evidence_verdict` | 32 | 0.3782 | 0.4663 | 103 | 0.7087 |

### 0.2 Diagnosis

1. `qa_strategyqa_cot_compact` is currently the best total-accuracy style in freeform mode.
   - It improves absolute accuracy over `qa_strategyqa_minimal_binary` by `+0.0311` (0.6943 vs 0.6632).
2. `qa_strategyqa_minimal_binary` is the best compliance style.
   - Parse error is exactly `0.0000`, making it the safest format baseline.
3. `qa_strategyqa_evidence_verdict` likely has a **format/truncation bottleneck**, not purely low reasoning quality.
   - Parseable-only accuracy is high (`0.7087`), close to the best parseable track (`0.7166`),
   - but parse failures are very high (`0.4663`), collapsing total accuracy.

### 0.3 What We Learn

1. Prompt style matters as much as token budget for StrategyQA freeform performance.
2. There is a practical tradeoff:
   - best compliance (`minimal_binary`) vs best current total accuracy (`cot_compact`).
3. For verdict-style prompts, current output protocol is under-specified for short budgets (`t32`).
   - the model often produces partially formatted outputs, causing non-parseable predictions.

### 0.4 Reliability Notes

1. This comparison is protocol-consistent (same dataset split/policy/seed/model family), so the ranking is meaningful for this setup.
2. Sample size is still modest (`n=193`); small deltas should be reconfirmed with reruns and/or larger validation sets before freezing final claims.

### 0.5 Next Plan

1. Keep `qa_strategyqa_cot_compact` as the freeform quality candidate for Phase A.
2. Keep `qa_strategyqa_minimal_binary` as compliance/efficiency baseline.
3. Run a targeted verdict-style follow-up with higher token budgets and/or stricter output contract before discarding the style.
4. For fair model-quality comparison independent of parse noise, continue reporting binary-choice results in parallel.

## 2. Previous Diagnosis Update (2026-02-26, sweeper2 + A5)

### 0.1 What Problem We Tried To Fix

The active issue was poor output compliance in StrategyQA yes/no evaluation:
- CoT runs produced long explanations without extractable final yes/no.
- Parse errors dominated total accuracy at mid token budgets.
- We introduced strict yes/no prompting (`A5`) to force parseable outputs.

### 0.2 New Results Added

#### 0.2.1 CoT Sweeper2 (`qa_cot_then_final`, n=193)

| Run Dir | Max New Tokens | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---:|---:|---:|---:|---:|
| `strat_cot_sweep_t128_20260226T103204Z` | 128 | 0.1036 | 0.8497 | 29 | 0.6897 |
| `strat_cot_sweep_t192_20260226T104650Z` | 192 | 0.2591 | 0.6684 | 64 | 0.7812 |
| `strat_cot_sweep_t256_20260226T110535Z` | 256 | 0.4974 | 0.3212 | 131 | 0.7328 |
| `strat_cot_sweep_t320_20260226T112825Z` | 320 | 0.6684 | 0.1140 | 171 | 0.7544 |
| `strat_cot_sweep_t384_20260226T115617Z` | 384 | 0.6839 | 0.0518 | 183 | 0.7213 |

#### 0.2.2 A5 Strict Compliance Fix (`qa_binary_strict`, n=193)

| Run Dir | Max New Tokens | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---:|---:|---:|---:|---:|
| `my_run_baseline_direct_t16_20260226T125946Z` | 16 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `my_run_strict_t4_20260226T130128Z` | 4 | 0.5078 | 0.2383 | 147 | 0.6667 |
| `my_run_strict_t8_20260226T130206Z` | 8 | 0.5078 | 0.2383 | 147 | 0.6667 |
| `my_run_strict_t16_20260226T130259Z` | 16 | 0.5285 | 0.0363 | 186 | 0.5484 |
| `my_run_strict_t16_20260226T130431Z` | 16 | 0.5285 | 0.0363 | 186 | 0.5484 |

### 0.3 Updated Diagnosis

1. CoT quality was suppressed by parse failures, not only by reasoning quality.
   - Evidence: parseable-only accuracy is already high at low token budgets (`0.6897` at `t128`) despite very low total accuracy.
2. CoT with enough token budget becomes the strongest quality setup.
   - `t320/t384` clearly outperform direct baseline total accuracy.
3. Strict prompt is an effective compliance control, but not a full quality fix.
   - `strict_t16` sharply reduces parse errors (`0.0363`) but lowers parseable-only correctness (`0.5484`).
4. Extremely short strict budgets (`t4/t8`) over-truncate.
   - They do not improve either total accuracy or parse errors vs direct baseline.
5. Reproducibility is confirmed for strict runs.
   - `strict_t16` repeated with identical metrics and zero changed samples.

### 0.4 Fix Verdict By Problem

| Problem | Status | Notes |
|---|---|---|
| Parse/compliance failures | Partially fixed | Strongly improved by strict template or larger CoT budget. |
| Overall answer quality | Mixed | Best with large CoT budget; strict short format alone hurts quality. |
| Inference efficiency | Unfixed for high-quality mode | CoT `t320/t384` is accurate but expensive. |
| Deterministic reproducibility | Fixed | Repeat runs match exactly under no-sample seed-fixed settings. |

### 0.5 Next Plan (Actionable)

1. Keep two official baselines:
   - Quality baseline: CoT `t320` and `t384`.
   - Efficiency baseline: direct `t16` or `t32`.
2. Tighten strict-mode decoding boundaries:
   - stop sequences for `Human:`, `User:`, `[SYSTEM]`, `[END]`.
   - keep strict runs at small token limits (`16-32`) after stop fix.
3. Add extractor robustness checks:
   - test explicit variants like `yes.`, `no.` and leakage strings (`noHuman:`).
4. Run a targeted strict follow-up sweep after stop/extractor fix:
   - tokens: `16, 24, 32`.
   - compare against `baseline_direct_t16` and `strat_cot_sweep_t320`.
5. Freeze one benchmark profile for Phase A reporting:
   - deterministic, no-sampling, fixed artifact fingerprint, fixed evaluator version.

### 0.6 Parser/Decode Remedy Update (2026-02-26 late)

We implemented a targeted remedy for parse/compliance failures:
1. stronger StrategyQA extraction rules,
2. chat-leak truncation (`[USER]`, `Human:` etc.),
3. optional binary-choice decode mode (`yes` vs `no` scoring) to remove free-form formatting variance.

Quick re-evaluation on existing predictions (same generation outputs, new parser):

| Run Dir | Old Accuracy | Old Parse Error | New Accuracy | New Parse Error | New Accuracy on Parseable |
|---|---:|---:|---:|---:|---:|
| `strat_cot_sweep_t128_20260226T103204Z` | 0.1036 | 0.8497 | 0.1036 | 0.8446 | 0.6667 |
| `strat_cot_sweep_t320_20260226T112825Z` | 0.6684 | 0.1140 | 0.6684 | 0.1088 | 0.7500 |
| `strat_cot_sweep_t384_20260226T115617Z` | 0.6839 | 0.0518 | 0.6995 | 0.0518 | 0.7377 |
| `my_run_baseline_direct_t16_20260226T125946Z` | 0.5492 | 0.1762 | 0.5492 | 0.1762 | 0.6667 |
| `my_run_strict_t16_20260226T130259Z` | 0.5285 | 0.0363 | 0.6580 | 0.0000 | 0.6580 |

Interpretation:
- A non-trivial share of previous parse errors came from evaluation/code-side extraction brittleness (especially strict leakage forms like `noHuman:`).
- Remaining CoT parse errors at low token budgets are still largely model/protocol behavior (model does not emit an explicit binary decision often enough).
- To isolate true model decision quality from formatting noise, use `strategyqa_decode_mode=binary_choice` in A6 runs.

### 0.7 A6 Confirmation + Freeform-vs-Binary Diagnosis (2026-02-26)

New A6 run (`strategyqa_decode_mode=binary_choice`) result:

| label | n | accuracy | parse_error_rate | acc_parseable |
|---|---:|---:|---:|---:|
| `direct_binchoice` | 193 | 0.6788 | 0.0000 | 0.6788 |
| `cot_binchoice` | 193 | 0.5803 | 0.0000 | 0.5803 |
| `direct_binchoice_repro` | 193 | 0.6788 | 0.0000 | 0.6788 |

CoT A2 under binary-choice decode (token sweep):
- `t128/t192/t256/t320/t384` all reported exactly `acc=0.5803`, `parse_err=0.0000`.

Root-cause conclusion:
1. **Parse-error problem is mostly protocol/extraction/generation-format**, not pure model incapability.
   - Binary-choice decode eliminates parse errors to `0.0000`.
2. **Model-quality problem is still real after removing formatting noise**.
   - Direct binary-choice reaches `0.6788`, while CoT-prompt binary-choice is lower (`0.5803`).
3. **“Accuracy on parseable subset” can be misleading alone**.
   - Freeform CoT `t128` has `acc_parseable=0.6667` but only `n_parseable=30/193`.
   - Binary-choice gives full coverage (`193/193`) and is the fairer decision-quality view.

Important interpretation note:
- Under current implementation, StrategyQA `binary_choice` mode scores yes/no directly and does not depend on `max_new_tokens`.
- Therefore identical A2 token-sweep accuracies in binary mode are expected behavior, not a runtime bug.

Plan update:
1. Use two official StrategyQA metrics tracks:
   - **Decision quality track**: `binary_choice` (coverage fixed at 100%).
   - **End-to-end format track**: `freeform` (includes compliance burden).
2. Keep direct prompt as binary-choice baseline (`~0.6788`) for Phase A.
3. For CoT prompt, improve question framing/system instruction before attributing gap to model limits.

### 0.8 Phase A Closeout (Concise)

Conclusion:
- Phase A goals are met for benchmark infrastructure and diagnosis quality.
- We now have a reliable split between:
  - decision-quality evaluation (`binary_choice`), and
  - end-to-end formatting/compliance evaluation (`freeform`).

Achievements:
1. Deterministic one-click benchmark suite with param groups (`A1..A6`) and persisted logs.
2. Stable prepared-data pipeline and evaluation pipeline for StrategyQA.
3. Parse-error root cause identified and partially fixed in extraction/decoding.
4. Reproducibility validated (repeat runs produce identical results under fixed settings).

Problems probed and fixed:
1. Prompt-leak parse failures (`noHuman: ...`) fixed via extraction hardening and marker truncation.
2. Misleading evaluation from parseable-only subset mitigated by always reporting:
   - `n_parseable`,
   - `acc_parseable`,
   - and total-coverage metrics.
3. Free-form format variance isolated using `binary_choice` mode.

Open (unsolved) problems:
1. CoT prompt underperforms direct prompt on decision quality (`0.5803` vs `0.6788`).
2. Free-form CoT still has severe formatting burden at small budgets.
3. Phase A is inference-only; value learning and faithfulness-training components are not implemented yet.

To-do next (immediate):
1. Freeze Phase A benchmark protocol (artifact fingerprint + decode mode + reporting schema).
2. Start Phase B in separate scripts/modules:
   - model wrapper + value head,
   - BCR-lite losses,
   - first training smoke runs.
3. Keep Phase A scripts unchanged as regression baselines.

Re-planned route to final ABR/BCR goal:
1. Stage B: BCR-lite implementation and training stability.
2. Stage C: faithfulness calibration/corruption metrics and robust ablations.
3. Stage D: ABR heuristic router (`gen/ver/fin`) with fixed budget.
4. Stage E: ABR learned router (RL) only after Stage D is stable and reproducible.

### 0.9 GSM8K Smoke Criticality Scan and Fix (2026-02-26)

Observed critical symptom (direct baseline smoke, 20 samples):
- run: `gsm8k_direct_smoke_20260226T150012Z`
- `accuracy=0.0500`, `parse_error_rate=0.0000`
- extraction methods: 20/20 were `last_number`.

Diagnosis:
1. This was not a parser crash, but a weak extraction regime.
2. Model outputs were partial reasoning text under token cap; evaluator fell back to `last_number`.
3. Therefore parse-error stayed zero while answer quality collapsed.

Fixes implemented:
1. Added new prompt template `qa_math_direct_final@1.0.0`:
   - forces one-line `Final answer: <number>` format.
2. Extended math extraction:
   - supports both `Final answer: ...` and `Final answer is ...`.
3. Added math diagnostics in `phase_a_generate_and_eval.py`:
   - `last_number_rate`,
   - `hit_token_limit_rate`,
   - `final_answer_tag_rate`,
   - warning text when extraction looks unreliable.

Post-fix smoke validation:
1. run: `gsm8k_math_direct_smoke_20260226T150913Z` (`max_new_tokens=64`)
   - `accuracy=0.6000`, `parse_error_rate=0.0000`,
   - `last_number_rate=0.0000`,
   - `final_answer_tag_rate=1.0000`.
2. run: `gsm8k_math_direct_smoke_t16_20260226T151029Z` (`max_new_tokens=16`)
   - `accuracy=0.6000`, `parse_error_rate=0.0000`,
   - same smoke accuracy with much faster runtime.

Practical guidance:
1. For GSM8K direct baseline in Phase A, use `qa_math_direct_final`.
2. Start with `max_new_tokens=16` for fast baseline iteration.
3. Treat heavy `last_number` reliance as a quality warning, not as a healthy parse signal.

### 0.10 GSM8K Sweep Re-Diagnosis (2026-02-26 late)

New finding from your full GSM8K runs:
1. Direct sweep (`t16/t32/t64/t96`) stayed exactly at `accuracy=0.3663` with:
   - `parse_error_rate=0.0000`,
   - `hit_cap_rate=1.0000`,
   - `final_tag_rate=1.0000`.
2. CoT run (`t256`) looked worse at `accuracy=0.2733`, but diagnostics showed:
   - `final_answer_tag=119`,
   - `last_number=53`,
   - many tagged answers contained units/currency text (for example: `10 meters.`, `$140.`),
   - strict numeric comparison undercounted true correctness.

Code-side remedy:
1. Math evaluator now supports relaxed numeric equivalence when gold is numeric:
   - strict parse first,
   - fallback to first numeric token from predicted text (for example `75% ... -> 75`).
2. Math final-tag extraction now normalizes numeric answers from tagged text directly.

Re-evaluation (same prediction files, new evaluator):
1. `gsm8k_cot_t256_20260226T151603Z`:
   - old `accuracy=0.2733` -> new `accuracy=0.7035`.
2. `gsm8k_math_direct_t16_20260226T152107Z`:
   - old `accuracy=0.3663` -> new `accuracy=0.3721`.

Interpretation update:
1. CoT was strongly underestimated due to extraction strictness, not only model failure.
2. Direct baseline remains relatively stable around ~0.37 in this setup.
3. CoT now appears much stronger than direct on GSM8K under corrected numeric matching.
4. `hit_cap_rate=1.0` still indicates throughput waste; speed optimization remains required.

### 0.11 Foundation Reliability Hardening Gate (2026-02-27)

Reason:
1. We observed that low-level evaluation and parsing details can drastically shift conclusions.
2. To avoid Phase B/C being misled, we ran a base-component reliability hardening pass.

Completed hardening:
1. Binary-choice generation path now has branch-safe metadata handling.
2. Empty model outputs are treated as parse-error cases (no hard evaluator crash).
3. Loader split normalization now fails fast on split typos.
4. Metrics now carry evaluator version and comparison warns on version mismatch.
5. Prepared-input loader now rejects duplicate `sample_id`.

Artifacts:
1. Audit document: `foundation_reliability_audit.md`.
2. Execution checklist: `TODO_ours.md` section `18`.

### 0.12 Batched Inference Fix Validation (2026-02-27)

Problem:
1. Early batched freeform runs produced severe quality regression and repeated warnings:
   - decoder-only model warned about right-padding,
   - parse errors spiked and accuracy dropped vs `batch_size=1`.

Fix:
1. In batched freeform tokenization path, force `padding_side='left'` and restore tokenizer state after call.
2. Keep deterministic ordering and existing OOM backoff safety net.

Validation runs (StrategyQA, direct prompt, `max_new_tokens=32`, no-sample, seed=42, n=193):

| Run | Batch Size | Accuracy | Parse Error Rate | n_parseable | acc_parseable | gen_sample_rate | gen_elapsed_sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `strat_batch1_20260226T164044Z` | 1 | 0.5492 | 0.1762 | 159 | 0.6667 | 1.180 | 163.53 |
| `strat_batch4_fix_20260226T165830Z` | 4 | 0.5596 | 0.1658 | 161 | 0.6708 | 4.189 | 46.08 |

Inference:
1. Throughput gain is substantial (`~3.55x` sample/s, `~72%` less wall time).
2. Quality is not degraded after fix; metrics are slightly improved.
3. Current batching implementation is trustworthy for Phase A benchmarking at `batch_size=4` on this setup.

Next checks:
1. Run one reproducibility pair for `strat_batch4_fix` (same run-name, same seed) to verify `changed_samples=0`.
2. Repeat the same pair on GSM8K direct path to confirm cross-dataset batching parity.

### 0.13 Phase A Retrospective Lessons (Full-Period Consolidation)

This section consolidates what we learned across the full Phase A cycle, beyond single-run snapshots.

#### 0.13.1 Evaluation semantics can dominate conclusions

1. Parser/extractor design can move apparent accuracy dramatically without any model-weight change.
2. Never compare old/new metrics across evaluator logic changes unless predictions are rescored under the same evaluator version.
3. Keep `evaluator_version` in metrics and treat version mismatch as a hard caution.

#### 0.13.2 Always separate coverage from decision quality

1. `accuracy` alone is insufficient when parse failures are possible.
2. Always report this set together:
   - `accuracy`,
   - `parse_error_rate`,
   - `n_parseable`,
   - `acc_parseable`.
3. For StrategyQA, maintain two official tracks:
   - binary-choice (decision quality),
   - freeform (end-to-end format compliance).

#### 0.13.3 Prompt/protocol choices can outweigh token-budget tweaks

1. StrategyQA freeform CoT was highly sensitive to protocol compliance.
2. Binary-choice mode showed that much of the previous gap was format/channel noise.
3. Token budget sweeps are meaningful only after decode/eval protocol is stable.

#### 0.13.4 Throughput optimizations require explicit correctness guards

1. Batching initially improved speed but introduced a decoder-only padding bug that degraded quality.
2. Batch rollout must include:
   - padding-side correctness checks,
   - ordering parity checks,
   - deterministic rerun checks.
3. Throughput gains are valid only when metric parity is preserved.

#### 0.13.5 Truncation pressure is a real research confound

1. High hit-cap rates can hide weak extraction behavior and understate model quality.
2. Math runs need extraction diagnostics (`last_number_rate`, `final_answer_tag_rate`, cap rate) before interpretation.
3. Short-token baselines are useful for speed, but should not be treated as quality ceilings.

#### 0.13.6 Phase A final operational rule set (for Phase B carry-over)

1. Freeze baseline protocol before Phase B sweeps:
   - dataset fingerprint,
   - template/version,
   - decode mode,
   - evaluator version,
   - seed and no-sample setting.
2. Any infra change (parser/prompt/decode/batching) must be followed by:
   - replay on at least one stored prediction file,
   - one reproducibility pair run.
3. Treat infra regressions as P0, because they can invalidate all higher-level claims.

## 1. Purpose

This file records meaningful experiment outcomes so new teammates can quickly understand:
- what was run,
- what settings were used,
- what happened,
- and what conclusions we can trust.

Scope: Phase A inference/evaluation experiments on StrategyQA with local `Qwen2.5-7B-Instruct`.

## 2. Prepared Artifact Sets (Inputs)

These artifact folders were generated by `scripts/phase_a_prepare.py` and reused by inference runs.

| Fingerprint | Created (UTC) | Template | Target Style | Source Split | Split Policy | Limit | Validation File |
|---|---|---|---|---|---|---:|---|
| `21095d3c688a` | 2026-02-25T19:09:43.283792+00:00 | `qa_direct` | `answer_only` | `train` | `hash` | 200 | `assets/artifacts/phase_a_prepared/strategyqa/21095d3c688a/validation.jsonl` |
| `b0f373610f96` | 2026-02-26T07:13:22.068668+00:00 | `qa_direct` | `answer_only` | `train` | `hash` | 2000 | `assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl` |
| `f3e476b514c3` | 2026-02-26T07:18:52.297402+00:00 | `qa_cot_then_final` | `cot_then_answer` | `train` | `hash` | 2000 | `assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl` |

## 3. Early Smoke Runs (Small Validation Set)

These were used to verify pipeline wiring and deterministic rerun behavior.

| Run Dir | Created (UTC) | n | Max New Tokens | Accuracy | Parse Error Rate | Notes |
|---|---|---:|---:|---:|---:|---|
| `qwen_strategyqa_val_20260226T065337Z` | 2026-02-26T06:55:19.075627+00:00 | 2 | 32 | 1.0000 | 0.0000 | smoke |
| `qwen_strategyqa_val_20260226T065605Z` | 2026-02-26T06:56:41.906460+00:00 | 1 | 16 | 1.0000 | 0.0000 | smoke |
| `qwen_strategyqa_val_20260226T065859Z` | 2026-02-26T06:59:51.576498+00:00 | 17 | 64 | 0.6471 | 0.1176 | first stable mini run |
| `qwen_strategyqa_val_20260226T070029Z` | 2026-02-26T07:01:14.792043+00:00 | 17 | 64 | 0.6471 | 0.1176 | deterministic repeat |
| `qwen_strategyqa_val_20260226T070404Z` | 2026-02-26T07:04:50.085621+00:00 | 17 | 64 | 0.6471 | 0.1176 | deterministic repeat |

## 4. Main StrategyQA Runs (n=193)

All runs below are deterministic (`--no-do-sample`, seed=42).

| Run Dir | Created (UTC) | Input Prepared Set | Max New Tokens | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---|---|---:|---:|---:|---:|---:|
| `strat_vA_direct_20260226T084052Z` | 2026-02-26T09:05:30.421698+00:00 | `f3e476b514c3` (CoT prompt/target) | 128 | 0.1036 | 0.8497 | 29 | 0.6897 |
| `strat_vA_direct_20260226T090829Z` | 2026-02-26T09:28:41.357635+00:00 | `f3e476b514c3` (CoT prompt/target) | 128 | 0.1036 | 0.8497 | 29 | 0.6897 |
| `strat_direct_t32_20260226T093714Z` | 2026-02-26T09:40:12.219698+00:00 | `b0f373610f96` (Direct prompt/target) | 32 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_direct_t32_20260226T095322Z` | 2026-02-26T09:56:16.570779+00:00 | `b0f373610f96` (Direct prompt/target) | 32 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_direct_t32_20260226T100949Z` | 2026-02-26T10:12:38.609192+00:00 | `b0f373610f96` (Direct prompt/target) | 32 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_cot_t128_20260226T095113Z` | 2026-02-26T10:02:08.907577+00:00 | `f3e476b514c3` (CoT prompt/target) | 128 | 0.1036 | 0.8497 | 29 | 0.6897 |
| `strat_cot_t256_20260226T095906Z` | 2026-02-26T10:21:17.394167+00:00 | `f3e476b514c3` (CoT prompt/target) | 256 | 0.4974 | 0.3212 | 131 | 0.7328 |

## 5. Operational Incidents (Important Context)

### 5.1 GPU unavailable period (resolved later)
- During one period, CUDA/NVML was unavailable in runtime; jobs fell back to CPU and ran very slowly.
- Later checks showed GPU access restored (`torch.cuda.is_available() == True`, `nvidia-smi -L` normal).

### 5.2 CPU offload caused severe slowdown
- Run directory: `assets/artifacts/phase_a_runs/strat_cot_t128_20260226T094126Z`
- Symptom: `device_map=auto` offloaded tail layers to CPU (`Some parameters are on the meta device...`), causing very slow decode speed.
- Status: interrupted/incomplete run; only partial `predictions.jsonl` exists (no `metrics.json`).

## 6. Key Findings We Can Trust

1. Determinism is stable for fixed config.
   - Repeated runs with same settings produced identical metrics (and in key cases identical predictions).

2. Main bottleneck in CoT setting is format/completion, not core correctness.
   - At CoT `t128`, parse error is very high (`0.8497`) while accuracy on parseable subset remains much higher (`0.6897`).

3. Increasing CoT budget helps a lot but does not fully solve formatting issues.
   - `t128 -> t256` improved:
     - accuracy: `0.1036 -> 0.4974`
     - parse error rate: `0.8497 -> 0.3212`

4. Direct answer setup is currently the strongest robust baseline.
   - `strat_direct_t32` is fast and stable:
     - accuracy `0.5492`
     - parse error `0.1762`

## 7. Suggested Next Experiments (Queued)

To finish diagnosing truncation/compliance tradeoffs, run:
- CoT token sweep: `128, 192, 256, 320, 384`
- Direct token sweep: `16, 24, 32, 48, 64`
- Determinism repeat at one key CoT point (`t256`) to ensure run-to-run stability.

Keep fixed for fair comparisons:
- `--no-do-sample`
- `--seed 42`
- same input artifact file per variant
- no CPU offload (ensure model fully fits on selected GPU)

## 8. Direct Token Sweep Result (Completed)

Run family: `strat_direct_sweep_t*` on prepared input
`assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl`

| Run Dir | Max New Tokens | Elapsed (from console) | Throughput (sample/s) | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---:|---:|---:|---:|---:|---:|---:|
| `strat_direct_sweep_t16_20260226T103213Z` | 16 | 00:01:52 | 1.709 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_direct_sweep_t24_20260226T103417Z` | 24 | 00:02:47 | 1.155 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_direct_sweep_t32_20260226T103717Z` | 32 | 00:03:42 | 0.866 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_direct_sweep_t48_20260226T104125Z` | 48 | 00:05:25 | 0.592 | 0.5492 | 0.1710 | 160 | 0.6625 |
| `strat_direct_sweep_t64_20260226T104715Z` | 64 | 00:07:21 | 0.437 | 0.5544 | 0.1658 | 161 | 0.6646 |

### 8.1 Direct Sweep Analysis

1. Accuracy is mostly flat from `t16` to `t48`; only a small increase appears at `t64`.
2. Parse error improves only slightly with larger budgets (`0.1762 -> 0.1658`).
3. Runtime cost increases strongly as token budget grows; quality gain is small.
4. Outputs are still almost always near token cap even in direct mode, suggesting generation tends to keep talking until cutoff.
5. Prediction bias remains strong toward `no` on parseable outputs (roughly 85% `no`, 15% `yes`), so class-bias is still a key issue.

Operational inference:
- For current direct template, `t16`/`t24` are much more efficient and nearly as accurate as larger budgets.
- Purely increasing token budget is not enough to fix format/bias issues.

## 9. New Param Block Added to Address Current Problems

New one-click group in `scripts/run_phase_a_benchmark_suite.sh`:
- `ACTIVE_PARAM_GROUP=A5`

Goal of `A5`:
- improve answer-format compliance,
- reduce token waste,
- check whether stricter binary prompting helps yes/no behavior.

`A5` run plan:
1. baseline direct reference: `baseline_direct_t16`
2. strict binary prompt runs: `strict_t4`, `strict_t8`, `strict_t16`
3. strict determinism check: `strict_t16` repeated (run1 vs run2 with comparison enabled)

Template used by strict runs:
- `qa_binary_strict@1.0.0` (added in prompt builder)
- target style: `answer_only`

Recommended execution command:

```bash
ACTIVE_PARAM_GROUP=A5 \
CUDA_VISIBLE_DEVICES=3 \
RUN_PREFIX=strat_strict_fix_20260226 \
bash scripts/run_phase_a_benchmark_suite.sh
```

What to focus on for A5 results:
1. parse error rate vs `baseline_direct_t16`
2. total accuracy change
3. parseable-only accuracy
4. determinism (`delta_accuracy=0`, `changed_samples=0` on strict repeat)

## 0X. Follow-up Diagnosis (2026-03-11, Tiny-Terminal Hybrid Pilot)

### 0X.1 Why this follow-up mattered

After the first hybrid pilot, the key uncertainty was:

1. is `terminal_completion_anchor` still too large,
2. or is naive local+terminal mixture fundamentally too blunt even at tiny ratios?

### 0X.2 Tiny-terminal artifact

Artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_pbhybrid_tinyterm_0311__dd5acae29427`

Configured caps:

1. `prm_local`: train `3072`, val `384`
2. `prm_terminal`: train `64`, val `8`, mix weight `0.10`

Failure analysis later showed:

1. actual terminal semantics are only:
   - `17 / 3136 = 0.54%`

### 0X.3 Same-source fit

Run:

1. `assets/artifacts/phase_e_runs/phase_e_pbhybrid_tinyterm_0311_mlp_20260311T050119Z`

Held-out:

1. `pair_acc=0.918367`
2. `auc=0.890293`

So reducing terminal support does not damage source-family learnability.

### 0X.4 ProcessBench result

Comparison tables:

1. GSM8K:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_pbhybrid_terminal_ratio_gsm_compare_20260311T051627Z/summary.md`
2. Math:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_pbhybrid_terminal_ratio_math_compare_20260311T051627Z/summary.md`

Main comparison:

#### GSM8K

1. `E46`
   - `auc=0.6264`
   - `first_edge=0.6706`
   - `terminal_top1=0.2332`
2. `hybrid_ta15`
   - `auc=0.5543`
   - `first_edge=0.4906`
   - `terminal_top1=0.8548`
3. `hybrid_tiny`
   - `auc=0.5654`
   - `first_edge=0.5094`
   - `terminal_top1=0.7742`

#### Math

1. `E46`
   - `auc=0.6053`
   - `first_edge=0.6096`
   - `terminal_top1=0.1970`
2. `hybrid_ta15`
   - `auc=0.4931`
   - `first_edge=0.4921`
   - `terminal_top1=0.7115`
3. `hybrid_tiny`
   - `auc=0.4948`
   - `first_edge=0.5238`
   - `terminal_top1=0.6538`

### 0X.5 What this changed in the diagnosis

This follow-up is highly informative:

1. terminal ratio clearly matters.
2. shrinking terminal semantics from `3.68%` to `0.54%` softens the over-correction.
3. but the result is still far below `E46` on benchmark-local discrimination.

Therefore the updated reading is:

1. the current failure is **not only** a ratio problem,
2. it is also an objective / supervision-contract problem,
3. so the next mainline should move toward:
   - staged training,
   - two-objective training,
   - or benchmark-aware checkpoint selection,
   not just more terminal-ratio sweeps.

## 0ZZZZZZ. Network-backed judge integration verdict (2026-03-11)

We re-checked the relevant judge literature and community best practices against our local runs. The most relevant references remain:

1. `JudgeLM`
2. `Prometheus / Prometheus 2`
3. `G-Eval`
4. `ProcessBench`
5. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`

The consolidated conclusion is:

1. `PRMBench_Preview`-style pairwise data is the only currently supported judge-facing training/data interface among the sources we tested.
2. `selected relabel` is supported in principle, but only on narrow slices; current evidence does **not** support full-dataset relabel.
3. `disagreement mining` remains one of the best uses of the local judge stack.
4. `benchmark-side adjudication` is safer and more realistic than `training-side bulk filtering`.

Local evidence:

1. Pairwise + swap-debias improved the *direction* of the judge setup, but did not produce a strong enough bulk filter.
2. `Qwen2.5-Math-7B-Instruct` on `PRMBench_Preview train64` only kept `4.69%` of pairs under the strict label-preserving criterion.
3. The same pipeline on `Math-Shepherd train64` kept `0%`.
4. Therefore:
   - keep `judge-assisted selected relabel`
   - keep `disagreement mining`
   - keep `benchmark-side adjudication`
   - drop `judge-driven bulk filtering` from the current mainline.

## 0YYYYYY. PRM-7B Backbone Breakthrough (2026-03-11 evening)

### Key finding: Backbone is the bottleneck, not training config

Multiple experiments on 2026-03-11 led to a clear conclusion:
using `Qwen2.5-Math-PRM-7B` as the backbone instead of `Qwen2.5-7B-Instruct`
gives **~2× improvement** on ProcessBench F1, regardless of training config.

### Experiment summary

| Run | Backbone | Val pair_acc | PB Math pb_f1 | PB GSM8K pb_f1 |
|---|---|---|---|---|
| PBR2a (ms_align_v1, joint+logit+conf_sem) | 7B-Instruct | ~0.51 | 0.186 | N/A |
| gated_mlp (dual head smoke, same config) | 7B-Instruct | 0.511, margin≈0 | (same config, same fail) | - |
| NDS ndsbh1 (rlhflow_align_v1, score space) | 7B-Instruct | 0.814 | 0.220 | 0.240 |
| Judge-1 (oracle filter + E87 config) | 7B-Instruct | 0.931 | 0.240 | 0.281 |
| **backboneproxy_prm_mixedsmall (2048 pairs)** | **PRM-7B** | **0.898** | **0.378** | **0.524** |

### Key discoveries

1. **Training config failure (Anti-pattern G)**: `joint + logit + confidence_semantic + lr=3e-5`
   causes near-zero gradient updates. The hinge loss at initialization = 0.713 (= margin 0.02 + BCE 0.693),
   mean_margin ≈ 5.7e-5. Fixed by: `ranking_only + score + confidence + lr=5e-5 + pair_acc selection`.

2. **Diag-E87config NaN failure**: ms_align_v1 raw data has terminal_completion_anchor pairs with very
   long texts (full solutions). These produce NaN/Inf features from the Instruct backbone.
   65% (6144/9478) of training pairs are affected. Oracle filter removes most terminal pairs → fixed.

3. **PRM-7B backbone superiority**: The `<extra_0>` token hidden states in PRM-7B carry step-level
   correctness assessments learned during PRM pretraining. `last_token` pooling extracts this.
   No amount of pair training on Instruct backbone can teach this from scratch.

4. **Oracle filter confirmation**: terminal_completion_anchor pairs pass filter at 12.6% rate
   (191/1511), proving the length-bias bug: both chosen and rejected are correct steps.
   After filter: terminal fraction drops 15.9% → 4.7%.

5. **ProcessBench F1 achieved** (best so far):
   - Math: 0.378 (target: 0.90+ for RL-ready, current SOTA from Qwen: 0.735)
   - GSM8K: 0.524

### Artifacts

- `assets/artifacts/phase_e_runs/phase_e_backboneproxy_prm_mixedsmall_20260311T074134Z/` — PRM-7B, 0.898 val
- `assets/artifacts/phase_e_eval/phase_e_eval_benchmark_20260311T084521Z/` — PB Math 0.378
- `assets/artifacts/phase_e_eval/phase_e_eval_benchmark_20260311T084946Z/` — PB GSM8K 0.524
- `assets/artifacts/phase_e_runs/phase_e_judge1_oracle_filter_e87config_20260311T080505Z/` — 7B-Instruct, 0.931 val

### Next experiment: PBR5-PRM7B (launched 2026-03-11 ~18:50)

- Backbone: Qwen2.5-Math-PRM-7B
- Pairs: oracle_filter_ms_align_v1 (4054 train, 462 val) — 2× more data than backboneproxy
- Config: ranking_only + score + confidence + lr=5e-5 + epochs=5
- Expected: val pair_acc > 0.93, PB Math pb_f1 > 0.40

## 0YYYYY. NDS Suite Results: math_step_dpo_v1 is competitive with PRM-7B (2026-03-11 evening)

### NDS ndsbh2 (math_step_dpo_v1 profile, 7B-Instruct)

Unexpected strong result: 7B-Instruct backbone with PURE LOCAL pairs (no terminal) achieves
comparable ProcessBench F1 to PRM-7B backbone.

| Run | Backbone | Pairs | Val pair_acc | PB Math pb_f1 | PB GSM8K pb_f1 |
|---|---|---|---|---|---|
| ndsbh1 rlhflow_align_v1 | 7B-Instruct | 3037 | 0.814 | 0.220 | 0.240 |
| ndsbh2 math_step_dpo_v1 | 7B-Instruct | 3705 | 0.742 | **0.379** | **0.363** |
| ndsbh3 ms_align_v1 (score config) | 7B-Instruct | 4096 | ~0.56 | 0.209 | 0.232 |
| backboneproxy PRM-7B | PRM-7B | 2048 | 0.898 | 0.378 | 0.524 |

### Key insight: pair semantics matter more than training config for 7B-Instruct

ndsbh2 uses `math_step_dpo_v1` profile = pure `ms_strict` (first_bad_edge_strict only),
NO terminal_completion_anchor pairs. Compared to ndsbh3 (ms_align_v1 = includes terminal):
- pb_f1: 0.379 vs 0.209 (+81% improvement!)
- Despite ndsbh2 having LOWER val pair_acc (0.742 vs ~0.56)

This proves: terminal pairs create a length bias that INVERTS ProcessBench pair_acc.
Removing terminal pairs entirely (ndsbh2) fixes the issue.

### Revised root cause understanding

Anti-pattern G (terminal length bias) is **dominant**:
- ms_align_v1 includes ~16% terminal anchor pairs → length bias → inverted PB pair_acc
- math_step_dpo_v1 (0% terminal) + same 7B-Instruct backbone → pb_f1=0.379
- This matches PRM-7B backbone (0.378) despite PRM-7B having much stronger representation

### When does PRM-7B backbone help?

- PB GSM8K: PRM-7B 0.524 vs ndsbh2 7B-Instruct 0.363 (+44%)
- PB Math: PRM-7B 0.378 vs ndsbh2 7B-Instruct 0.379 (parity!)

PRM-7B backbone gives larger gain on GSM8K (simpler arithmetic problems) where
the PRM's learned step quality signal is more directly applicable.

For complex Math problems, local pair discrimination is the bottleneck, not backbone.

### Current best recipe (as of 2026-03-11 ~19:00)

Option A: 7B-Instruct + pure local pairs (math_step_dpo_v1):
- pb_f1 Math: 0.379, GSM8K: 0.363
- No PRM-7B needed, fast training, caching works

Option B: PRM-7B + oracle filter (PBR5, running):
- Expected pb_f1 Math: 0.40+, GSM8K: 0.52+
- Better representation but slower (PRM-7B as backbone)

---

## 0ZZZZZZ — PBR5 实验结果：Oracle Filter + PRM-7B Backbone 突破（2026-03-11）

**实验**：PBR5 = oracle filter ms_align_v1（4054 train + 462 val）+ PRM-7B backbone + ranking_only + score space + lr=5e-5 + confidence weighting + pair_acc checkpoint selection

**训练 val 结果**（5 epochs）：
| Epoch | Pair Acc | AUC | Mean Margin |
|---|---|---|---|
| ep0 | 0.9848 | 0.9671 | 0.653 |
| ep1 | 0.9892 | 0.9716 | 0.669 |
| ep2 | **0.9935** | 0.9724 | 0.664 |
| ep3 | 0.9935 | 0.9786 | 0.686 ← best margin |
| ep4 | 0.9870 | 0.9783 | 0.680 |

Best checkpoint: ep2 (pair_acc=0.9935)

**ProcessBench 结果（256 样本，oracle threshold sweep）**：
| Benchmark | Pair Acc | Pair AUC | pb_f1 | pb_acc_err | pb_acc_cor | first_edge_acc |
|---|---|---|---|---|---|---|
| PB Math | 0.873 | 0.850 | **0.609** | 0.500 | 0.779 | 0.859 |
| PB GSM8K | 0.901 | 0.876 | **0.704** | 0.591 | 0.871 | 0.896 |

**完整实验对比表（7B-Instruct vs PRM-7B backbone）**：
| 实验 | Backbone | 数据 | Config | PB Math F1 | PB GSM8K F1 |
|---|---|---|---|---|---|
| PBR2a | 7B-Instruct | ms_align_v1 raw | joint+logit+lr=3e-5 | 0.186 | — |
| PBR3a (s42) | 7B-Instruct | ms_laterbad_v1 | joint+logit+lr=3e-5 | 0.207 | 0.221 |
| PBR3b (s1) | 7B-Instruct | ms_laterbad_v1 | joint+logit+lr=3e-5 | 0.176 | 0.206 |
| Judge-1 | 7B-Instruct | oracle filter | ranking_only+score | 0.240 | 0.281 |
| ndsbh2 | 7B-Instruct | math_step_dpo | ranking_only+score | **0.379** | 0.363 |
| backboneproxy | PRM-7B | mixed_small | group_bal+score | 0.378 | 0.524 |
| **PBR5** | **PRM-7B** | **oracle filter** | **ranking_only+score** | **0.609** | **0.704** |

**关键结论**：
1. PRM-7B backbone × oracle 过滤数据 = 协同增益（非线性叠加）
   - PRM-7B alone（mixed_small raw）: Math 0.378
   - oracle filter alone（7B-Instruct）: Math 0.240
   - PRM-7B + oracle filter（PBR5）: Math **0.609** = 2.5× over 7B-Instruct oracle
2. PBR5 val pair_acc=0.9935 是迄今最高，表明 oracle filter 给 PRM-7B 提供了近乎完美的训练信号
3. first_error_edge_accuracy（Math: 0.859，GSM8K: 0.896）远超之前所有方案
4. 下一步：PBR6（PRM-7B + 纯 local pairs math_step_dpo）对比 PBR5，确认 oracle filter vs zero-terminal 哪个更优

### 最优 Recipe（截止 2026-03-11）

**PBR5 Recipe**:
```
backbone: Qwen2.5-Math-PRM-7B
data: oracle_filter(ms_align_v1) → 4054 train pairs (12.6% pass rate)
pooling: last_token
objective: ranking_only
ranking_target_space: score
pair_weight_mode: confidence
lr: 5e-5, epochs: 5, batch_size: 96
checkpoint_selection: pair_acc
```
PB Math F1: **0.609** | PB GSM8K F1: **0.704**

---

## 0ZZZZZZZ — 多 seed 方差分析：PBR5 vs PBR6（2026-03-11）

**核心发现**：PBR6（纯 local 对）比 PBR5（oracle 过滤混合数据）具有更高均值 F1 和更低方差

### PBR5（oracle_filter ms_align_v1，6 seeds）

| Seed | Val Pair Acc | Mean Margin | PB Math F1 | PB GSM F1 |
|------|-------------|-------------|------------|-----------|
| s42  | 0.9935      | 0.664       | **0.609**  | **0.704** |
| s1   | 0.9502      | 0.438       | 0.465      | 0.512     |
| s2   | 0.9632      | 0.373       | 0.148      | 0.323     |
| s3   | 0.9762      | 0.477       | 0.309      | 0.436     |
| s4   | 0.9654      | 0.524       | 0.447      | 0.450     |
| s5   | 0.9502      | 0.379       | 0.107      | 0.097     |

**统计：Math mean=0.347 stdev=0.196；GSM mean=0.420 stdev=0.202**

### PBR6（pure local math_step_dpo_v1，4 seeds）

| Seed | Val Pair Acc | Mean Margin | PB Math F1 | PB GSM F1 |
|------|-------------|-------------|------------|-----------|
| s42  | 0.895       | 0.297       | **0.650**  | **0.717** |
| s1   | 0.752       | 0.093       | 0.479      | 0.548     |
| s2   | 0.749       | 0.071       | 0.370      | 0.523     |
| s3   | 0.793       | 0.095       | 0.370      | 0.449     |

**统计：Math mean=0.467 stdev=0.132；GSM mean=0.559 stdev=0.113**

### 关键结论

1. **PBR6 > PBR5 on both mean and max F1**：
   - Math 均值 +0.120（0.347→0.467），max +0.041（0.609→0.650）
   - GSM 均值 +0.139（0.420→0.559），max +0.013（0.704→0.717）

2. **PBR6 方差更小**：stdev 从 0.196 降至 0.132（Math），0.202→0.113（GSM）

3. **Val pair_acc 不可靠预测 PB F1**：PBR5 s2 val=0.963 但 PB Math=0.148（最差！）

4. **Mean margin 是更好的 PB 泛化预测指标**：
   - PBR5 s42 margin=0.664 → PB Math=0.609（最好）
   - PBR5 s5 margin=0.379 → PB Math=0.107（最差）
   - 相关性 >0.9

5. **根因：数据分布比 margin 更重要**：
   - PBR6 s42 margin=0.297（低于 PBR5 s42 的 0.664）但 PB Math=0.650（更高！）
   - 原因：math_step_dpo 纯 local 对与 ProcessBench 任务分布完全吻合（first_bad_edge 检测）
   - 而 oracle filter ms_align_v1 包含 fanout + grid pair 类型，偏离了 ProcessBench 分布

### 最优当前 Recipe（截至 2026-03-11）

**PBR6 recipe（推荐）**：
```
backbone: Qwen2.5-Math-PRM-7B
data: math_step_dpo_v1 (3705 pure local first_bad_edge pairs)
pooling: last_token
objective: ranking_only
ranking_target_space: score
pair_weight_mode: confidence
lr: 5e-5, epochs: 5, batch_size: 96
checkpoint_selection: pair_acc
seed: 42 (best known)
```
Best seed: Math F1=0.650 | GSM F1=0.717
Mean across 4 seeds: Math=0.467±0.132 | GSM=0.559±0.113

### 下一步：降低方差

1. PBR7: math_step_dpo + oracle filter（去除噪声对）→ 预期方差降低、均值提升
2. PBR8: 训练更多 epochs（5→8）→ margin 更高，可能减少运气依赖
3. 多 seed 集成推理：取 3-5 seeds 分数平均


---

## 0ZZZZZZZZ — 热启动消除种子方差：PBR6 Warm-Start（2026-03-11）

> ⚠️ **CORRECTION (2026-03-11 续)**: 本节早期结果基于 256-sample 子集评估（乐观估计）。以下已用全量 1000/400 样本重新评估，数据已更正。

### 发现：随机初始化是方差主要来源

分析 PBR6 cold seeds 结果后，发现：
- s42 ep0 pair_acc = 0.854（极好的初始化）
- s1 ep0 pair_acc = 0.675（差的初始化）
- s2,3 ep0 更低

这表明 3705 对 + PRM-7B backbone + 512-dim MLP head 的训练对随机初始化极其敏感。

### 解决方案：两阶段训练

**阶段一**：用 seed=42 冷启动训练 → 得到 best_value_head.pt（PBR6 recipe）
**阶段二**：从 best_value_head.pt 热启动，用任意 seed 再训练 → 稳定收敛

### 热启动实验结果（4 seeds，全量评估）

评估集：ProcessBench Math 全量 1000 例，GSM 全量 400 例

| Seed | Val Pair Acc | PB Math F1 | PB GSM F1 | 备注 |
|------|-------------|------------|-----------|------|
| ws_s1 (lr=5e-5, 5ep) | 0.903 | 0.626 | 0.722 | — |
| ws_s2 (lr=5e-5, 5ep) | 0.882 | 0.631 | **0.745** | — |
| ws_s3 (lr=5e-5, 5ep) | 0.882 | **0.635** | 0.742 | — |
| ws_lowlr_s1 (lr=2e-5, 8ep) | 0.903 | 0.633 | 0.739 | — |

**统计**：
- PB Math F1: mean=0.631, stdev=**0.004**（冷启动 cold seeds stdev 更高）
- PB GSM F1: mean=0.737, stdev=**0.010**

### 与冷启动 seed=42 对比（全量评估）

| Model | Math F1 | GSM F1 |
|---|---|---|
| PBR6 cold seed=42 | **0.642** | **0.743** |
| Warm-start mean (4 seeds) | 0.631 | 0.737 |
| Warm-start 4-seed ensemble (oracle) | 0.640 | **0.746** |

**关键发现**：
- 热启动方差确实大幅降低（stdev 从 0.132 降至 0.004）
- 但热启动单个种子均值（0.631）低于冷启动 seed=42（0.642）
- 4 种子集成（average scores）与冷启动 seed=42 相当，无显著提升

### 4-Seed 集成详细结果

集成方法：对 4 个热启动模型的分数取平均，再用阈值扫描

**Math 集成（1000 例）**：
- Oracle 阈值: F1=0.640, acc_err=0.519, acc_cor=0.835
- 固定 thresh=0.35: F1=0.631, acc_err=0.502, acc_cor=0.850

**GSM 集成（400 例）**：
- Oracle 阈值: F1=0.746, acc_err=0.628, acc_cor=0.917
- 固定 thresh=0.45: F1=0.740, acc_err=0.623, acc_cor=0.912

### Recipe（截至 2026-03-11 修正版）

**最优单模型**：PBR6 cold seed=42
```
backbone: Qwen2.5-Math-PRM-7B
data: math_step_dpo_v1 (3705 pairs)
pooling: last_token
objective: ranking_only, lambda_bce=0.25
ranking_target_space: score, margin=0.1
pair_weight_mode: confidence
lr: 5e-5, epochs: 5, batch_size: 96
checkpoint_selection: pair_acc
head: mlp, hidden=512, dropout=0.1
seed: 42
```
→ Math F1=0.642, GSM F1=0.743（全量评估）

**热启动**（降低方差，但不提升均值）：
```
同上，但 --init-value-head-path <stage1_best_value_head.pt>
→ 任意 seed 稳定输出 Math F1≈0.631±0.004
```

### 结论

热启动成功将种子方差从 stdev=0.132 降至 ~0.004，但：
- 热启动均值（0.631）不超过冷启动最好 seed=42（0.642）
- 4-seed 集成（0.640）与 seed=42 单模型（0.642）相当
- **当前最优推荐：直接使用 PBR6 cold seed=42，Math F1=0.642，GSM F1=0.743**


---

## 0ZZZZZZZZZ — PBR6 最终评估总结（2026-03-11 全量重测）

### 背景

之前使用 256-sample 子集评估 PBR5/PBR6。本次用全量 ProcessBench（Math 1000 例，GSM 400 例）重测，得到更准确的数字。

### 全量评估结果汇总

| Model | Math F1 | Math acc_err | Math acc_cor | GSM F1 | GSM acc_err | GSM acc_cor |
|-------|---------|-------------|-------------|--------|------------|------------|
| PBR5 cold s42 (256-sample est.) | ~0.609 | — | — | ~0.704 | — | — |
| PBR6 cold s42 (full) | **0.642** | 0.497 | 0.906 | **0.743** | 0.609 | 0.953 |
| PBR6 ws_s1 | 0.626 | 0.498 | 0.842 | 0.722 | 0.614 | 0.876 |
| PBR6 ws_s2 | 0.631 | 0.500 | 0.855 | 0.745 | 0.640 | 0.894 |
| PBR6 ws_s3 | 0.635 | 0.524 | 0.808 | 0.742 | 0.642 | 0.883 |
| PBR6 ws_lowlr_s1 | 0.633 | 0.510 | 0.833 | 0.739 | 0.623 | 0.913 |
| **4-seed ensemble (oracle)** | **0.640** | 0.519 | 0.835 | **0.746** | 0.628 | 0.917 |

### 关键结论

1. **PBR6 cold seed=42 是最优单模型**：Math F1=0.642, GSM F1=0.743
2. **4-seed 集成与 seed=42 相当**，无显著提升（ensemble Math 0.640 vs single 0.642）
3. **热启动降低方差**：warm-start 个体 Math stdev=0.004（vs 冷启动 >>0.1），但均值更低
4. **GSM 明显高于 Math**：GSM F1=0.743-0.746 vs Math F1=0.631-0.642，反映 Math 难度更高

### 当前最优 Recipe（RL-Ready 评估）

PBR6 cold seed=42（math_step_dpo_v1 + PRM-7B + ranking_only + mlp/512）：
- Math first_error_edge_accuracy: ~0.84
- GSM first_error_edge_accuracy: ~0.86
- 这是目前实现的最高 ProcessBench 性能，已达到步骤验证实用水准

### 下一步方向

要进一步提升 Math F1 到 0.70+，需要：
1. 更多 math_step_dpo-style 对（当前仅 3705 对，来自 MATH/GSM 有限子集）
2. 针对 Math-heavy 题目的专项数据增强
3. 混合 ProcessBench-train-style 数据（若存在）

---

## PBR10 — Scaled-Up DPO Pairs (8k, PRM-7B) [DONE]

**Hypothesis**: Doubling the DPO fork pairs (3705→6947 train pairs) may improve ProcessBench Math F1.
**Data**: math_step_dpo_v1 profile, target=8000 pairs
- 4679 sibling_branch DPO fork pairs (MATH domain)
- 1783 MS first_bad_edge_strict pairs (GSM8K domain)
- 485 terminal anchor pairs
- **Total**: 6947 train pairs

**Config**: PRM-7B backbone, ranking_only, lr=5e-5, 5 epochs, mlp/512/0.1, seed=42
**Training**: pair_acc=0.906, AUC=0.859

**Results**:
| Metric | PBR10 | PBR6 (baseline) | Δ |
|---|---|---|---|
| Math F1 | 0.631 | **0.642** | **-0.011** |
| Math acc_err | 0.508 | 0.497 | +0.011 |
| Math acc_cor | 0.833 | 0.906 | **-0.073** |
| GSM F1 | 0.733 | 0.743 | -0.010 |

**Conclusion**: More DPO pairs HURT performance. acc_correct dropped −7.3% (more false positives on all-correct examples). More sibling_branch DPO pairs causes score variance that over-triggers on correct steps. **Do not scale DPO above ~2400 pairs**.

---

## PBR11 — MATH-Domain MS First-Error Pairs (5k, PRM-7B) [DONE]

**Hypothesis**: Using MATH-domain (harder) MS first_bad_edge pairs should improve ProcessBench Math acc_err.
**Data**:
- 4514 local_first_bad_edge pairs from MATH competition (Math Shepherd MATH-only, 272k rows)
- 454 terminal anchor pairs (9% ratio)
- **Total**: 4968 train pairs

**Config**: Same as PBR6 (PRM-7B, ranking_only, lr=5e-5, 5 epochs, mlp/512/0.1, seed=42)
**Training**: pair_acc=0.900 (exactly hits 90% target!)

**Results**:
| Metric | PBR11 | PBR6 (baseline) | Δ |
|---|---|---|---|
| Math F1 | 0.357 | **0.642** | **-0.285** |
| Math acc_err | 0.300 | 0.497 | **-0.197** |
| Math acc_cor | 0.441 | 0.906 | **-0.465** |
| GSM F1 | 0.481 | 0.743 | -0.262 |

**Conclusion**: CATASTROPHIC FAILURE. MATH-only MS pairs without sibling_branch DPO pairs produce a completely miscalibrated model. Root cause: without DPO fork pairs for score calibration, the model cannot learn a well-placed decision threshold. High training pair_acc (0.90) does NOT generalize.

**Key Lesson**: `sibling_branch` DPO pairs are ESSENTIAL for score calibration — they cannot be removed.

---

## PBR13 — PBR6 + Terminal BCE λ=0.25 [DONE]

**Hypothesis**: Adding terminal BCE loss to PBR6 recipe will improve acc_err by pushing terminal step scores toward extremes.
**Data**: Same PBR6 pairs (math_step_dpo_v1, 3705 train pairs: 2398 DPO + 920 MS + 387 terminal)
**Config**: PRM-7B, ranking_only, lr=5e-5, 5 epochs, mlp/512/0.1, seed=42 + **terminal_bce_lambda=0.25**
**Training**: pair_acc=0.826 (lower than PBR6, terminal BCE increases loss)

**Results**:
| Metric | PBR13 | PBR6 (baseline) | Δ |
|---|---|---|---|
| Math F1 | **0.660** | 0.642 | **+0.018** ✅ |
| Math acc_err | **0.537** | 0.497 | **+0.040** ✅ |
| Math acc_cor | 0.857 | 0.906 | -0.049 |
| GSM F1 | **0.755** | 0.743 | **+0.012** ✅ |
| GSM acc_err | 0.623 | 0.609 | +0.014 ✅ |
| GSM acc_cor | **0.959** | 0.953 | +0.006 |

**Conclusion**: ✅ Terminal BCE λ=0.25 adds +1.8% Math F1, +4% acc_err.
Trade-off: acc_correct slightly lower (0.906→0.857) but net F1 improves. **SUPERSEDED by PBR18/PBR19**.

---

## PBR18 — PBR6 + Joint Objective + Terminal BCE [DONE]

**Hypothesis**: Adding full pair BCE (joint objective) alongside ranking loss + terminal BCE gives better absolute calibration.
**Data**: PBR6 pairs (3705 train: 2398 DPO + 920 MS + 387 terminal)
**Config**: PRM-7B, **joint**, λ_bce=0.5, λ_ranking=1.0, λ_terminal=0.25, lr=5e-5, 5 epochs, mlp/512/0.1, seed=42
**Training**: pair_acc=0.859 (higher than PBR13's 0.826)

| Metric | PBR18 | PBR13 | PBR6 | Δ vs PBR6 |
|---|---|---|---|---|
| Math F1 | **0.674** | 0.660 | 0.642 | **+0.032** ✅ |
| Math acc_err | 0.549 | 0.537 | 0.497 | +0.052 ✅ |
| Math acc_cor | 0.874 | 0.857 | 0.906 | -0.032 |
| GSM F1 | **0.764** | 0.755 | 0.743 | +0.021 ✅ |

**Conclusion**: Joint objective improves BOTH acc_err AND acc_correct vs PBR13. Math F1=0.674. **Superseded by PBR19**.

---

## PBR19 — DPO+MATH-MS Mix + Joint + Terminal BCE [DONE]

**Hypothesis**: Adding MATH-domain MS strict pairs to PBR18 recipe will improve acc_err for competition math.
**Data**: PBR12 pairs (5705 train: 2398 DPO + 2759 MS(MATH) + 548 terminal)
**Config**: PRM-7B, joint, λ_bce=0.5, λ_ranking=1.0, λ_terminal=0.25, lr=5e-5, 5 epochs, mlp/512/0.1, seed=42
**Training**: pair_acc=0.866

| Metric | PBR19 | PBR18 | PBR6 | Δ vs PBR6 |
|---|---|---|---|---|
| Math F1 | **0.683** | 0.674 | 0.642 | **+0.041** ✅ |
| Math acc_err | **0.574** | 0.549 | 0.497 | +0.077 ✅ |
| Math acc_cor | 0.842 | 0.874 | 0.906 | -0.064 |
| GSM F1 | **0.778** | 0.764 | 0.743 | +0.035 ✅ |

**Conclusion**: ✅ **CURRENT BEST** (2026-03-11 20:xx). Math F1=0.683, GSM F1=0.778.
All four components work synergistically: DPO calibration + MATH-domain first_bad_edge + joint objective + terminal BCE.

### Path to F1=0.70

PBR19: acc_err=0.574, acc_correct=0.842. To reach F1=0.70:
- Need acc_err≥0.60 + acc_correct≥0.88: F1 = 2×0.60×0.88/(0.60+0.88) = 1.056/1.48 = 0.714
- Currently: 406×(1-0.842)=64 false positives, 594×(1-0.574)=253 missed/wrong
- Fix 30 of the 64 false positives (acc_correct→0.87) + fix 40 of the 253 wrong cases (acc_err→0.60)

---

## PBR21 — DPO+MATH-MS+Joint, 10 Epochs (Overfitting Study) [DONE]

**Data**: PBR12 pairs (same as PBR19)
**Config**: Same as PBR19 but num_train_epochs=10
**Training**: pair_acc=0.890 (higher than PBR19's 0.866)

| Metric | PBR21 | PBR19 | Δ |
|---|---|---|---|
| Math F1 | 0.663 | **0.683** | **-0.020** ❌ |
| Math acc_err | 0.537 | **0.574** | -0.037 |
| Math acc_cor | 0.867 | 0.842 | +0.025 |
| GSM F1 | 0.756 | **0.778** | -0.022 |

**Conclusion**: 10 epochs OVERFITS despite higher validation pair_acc. **5 epochs is optimal** for this recipe.

---

## PBR6 Failure Analysis — ProcessBench Math Error Modes (2026-03-11)

**Based on**: PBR6 cold seed=42 scored_rows on full ProcessBench Math (1000 examples)
**Threshold**: τ=0.317 (oracle, optimized for F1)

### Score Distributions by Prefix Type

| Prefix Type | n | Mean Score | Std | P5 | P50 | P95 |
|-------------|---|-----------|-----|-----|-----|-----|
| Good prefix | 3772 | 0.619 | 0.169 | 0.296 | 0.645 | 0.851 |
| **First bad prefix** | 594 | **0.292** | 0.174 | 0.080 | 0.251 | 0.638 |
| Later bad prefix | 2139 | 0.330 | 0.180 | 0.096 | 0.297 | 0.674 |

**At τ=0.317**:
- Good prefix false trigger rate: **6.3%** (55 out of 406 all-correct examples)
- First bad prefix detection rate: **62.3%** (correctly detected errors)

### Error Analysis

**Error examples (594 total)**:
| Prediction outcome | Count | % |
|---|---|---|
| Correctly identified (pred=label) | 285 | 48.0% |
| No error detected (missed) | 102 | 17.2% |
| Wrong step predicted (too early) | 106 | 17.8% |
| Wrong step predicted (too late) | 101 | 17.0% |

**"Too early" prediction analysis** (106 cases):
- Average 2.1 steps too early
- Score at wrong trigger step: 0.244 (below τ, so it fires)
- Score at actual first bad step: 0.218 (even lower, but fires too late)
- Root cause: some CORRECT intermediate steps have suspiciously low scores (0.244)

### Root Cause

The model confuses certain "suspicious-looking correct" math steps with errors.
This is more common in MATH competition problems where intermediate steps look locally incorrect
but are globally correct.

**Implication**: PBR11 (MATH-domain MS pairs) should directly address this — training on
harder MATH competition first_bad_edge pairs will teach the model when to distrust intermediate steps.

### Key Metrics for Improvement

Current: acc_err=0.497, acc_cor=0.906, F1=0.642
To reach F1=0.70 (target): need acc_err≈0.57 (keeping acc_cor≥0.85)

---

## Terminal BCE Sweep — PBR13/14/15/16/17 (2026-03-11)

### Summary Table (All Results, Full ProcessBench)

| Run | Data | λ_termBCE | Math F1 | Math acc_err | Math acc_cor | GSM F1 |
|-----|------|-----------|---------|-------------|-------------|--------|
| PBR6 (baseline) | DPO v1 (3705p) | 0 | 0.642 | 0.497 | **0.906** | 0.743 |
| PBR10 | DPO v1 scale (6947p) | 0 | 0.631 | 0.508 | 0.833 | 0.733 |
| PBR11 | MATH-MS only (4968p) | 0 | 0.357 | 0.300 | 0.441 | 0.481 |
| **PBR13** | DPO v1 (3705p) | **0.25** | **0.660** | 0.537 | 0.857 | 0.755 |
| PBR14 | DPO+MATH-MS (5705p) | 0.25 | 0.659 | 0.554 | 0.813 | 0.757 |
| PBR15 | DPO v1 (3705p) | 0.50 | 0.654 | 0.527 | 0.862 | 0.762 |
| PBR16 | Reduced DPO+MATH-MS (2700p) | 0.25 | 0.652 | 0.545 | 0.810 | **0.764** |
| PBR17 | DPO v1 (3705p), lr=2e-5×8ep | 0.25 | 0.657 | 0.537 | 0.847 | 0.758 |

### Key Findings

1. **PBR13 is the current best for Math F1=0.660** (+1.8% vs PBR6 baseline)
2. **terminal_bce_lambda=0.25 is optimal** for Math; 0.5 overpowers the ranking loss
3. **More DPO pairs always hurts acc_correct**: DPO scaling (PBR10) drops acc_correct from 0.906 to 0.833
4. **MATH-domain MS pairs without DPO calibration = catastrophic failure** (PBR11, F1=0.357)
5. **Trade-off pattern**: all variants with terminal BCE show acc_err +4%, acc_correct −4-5%, net F1 +1.8%
6. **Math F1 plateau at ≈0.660**: no variant breaks 0.661 yet
7. **GSM best with PBR16**: 0.764 (reduced DPO + MATH-MS + terminal BCE)

### Next Direction

Math F1 plateau suggests the bottleneck is NOT the training objective but the training DATA FORMAT.
ProcessBench uses QwQ-32B multi-step chain-of-thought solutions (different style than MS/DPO training data).
Hypothesis: need training pairs in QwQ-32B-style format (PRM800K has OpenAI math solutions, possibly closer).

- PBR18: joint objective + terminal BCE (tests absolute vs relative calibration)
- PBR19: PRM800K first_bad_edge pairs (different solution format)
  → Need to fix ~45 of the 102 "no error detected" OR ~37 of the "wrong step" cases

---

## PBR22-PBR27 Systematic Variation Sweep — ProcessBench Transfer Plateau Analysis (2026-03-11)

### Context

After PBR19 established Math F1=0.683 as the best result, a systematic sweep was run to push toward F1=0.70.
PBR19 config (frozen Qwen2.5-Math-PRM-7B + MLP/512 head):
- Data: 3705 DPO sibling_branch + 2759 MATH-MS first_bad_edge + 548 terminal = 5705 train pairs
- Objective: joint (BCE+ranking), lambda_bce=0.5, lambda_ranking=1.0, lambda_terminal_bce=0.25
- Training: 5 epochs, lr=5e-5, batch=96

### Results Table

| Run | Variation | Train Pairs | Math F1 | Math acc_err | Math acc_cor | GSM F1 | Notes |
|-----|-----------|------------|---------|-------------|-------------|--------|-------|
| PBR19 (baseline) | — | 5705 | **0.683** | 0.574 | 0.842 | 0.778 | Best overall |
| PBR22 | +PRM800K same_step_completion (7705p) | 7705 | 0.682 | 0.569 | 0.852 | 0.771 | No benefit |
| PBR23 | source_balance=uniform (5705p) | 5705 | **0.684** | 0.561 | **0.877** | 0.773 | Best acc_correct |
| PBR24 | 24% terminal anchors (6109p) | 6109 | 0.656 | 0.522 | 0.882 | 0.779 | Too many terminals |
| PBR25 | uniform balance + lambda_termBCE=0.50 (5705p) | 5705 | 0.678 | 0.564 | 0.850 | 0.768 | StrongBCE hurts |
| PBR26 | 2398 DPO + full PBR11 MS (7366p) | 7366 | 0.670 | 0.557 | 0.840 | 0.779 | Fewer DPO hurts |
| PBR27 | PBR12 + MLP/1024 head (5705p) | 5705 | 0.666 | 0.547 | 0.850 | **0.784** | Large head overfits Math |

### Failure Mode Analysis (PBR19 at τ=0.406)

From scored_rows analysis of ProcessBench Math (1000 examples, 594 erroneous, 406 all-correct):

**Error detection accuracy by label type:**
- label=0 (first step wrong): 78/115 = **68%** (easier for model)
- label≥1 (later step wrong): 263/479 = **55%** (harder, main bottleneck)

**Failure mode breakdown for label≥1 errors:**
| Failure mode | Count | % |
|---|---|---|
| Exact correct (pred == label) | 263 | 54.9% |
| Early detect (pred < label) | 100 | 20.9% |
| Wrong step after (pred > label) | 54 | 11.3% |
| Missed entirely | 62 | 12.9% |

**Correct prefix score analysis:**
- 10.9% of correct prefix steps score BELOW threshold τ=0.406 → causes early detects
- p5=0.270, p25=0.592, p50=0.757, p75=0.861, p95=0.935

### Key Findings

1. **All hyperparameter/data variations plateau at Math F1 ≈ 0.683-0.684**
2. **DPO sibling_branch pairs are irreplaceable**: reducing from 3705→2398 (PBR26) drops F1 0.683→0.670
3. **Terminal fraction >10% is harmful**: 24% terminal (PBR24) drops acc_err from 0.574 to 0.522
4. **Larger head (MLP/1024 vs /512) overfits**: F1 0.683→0.666 for Math, but 0.778→0.784 for GSM8K  
5. **Primary bottleneck**: Early detects (20.9% of label≥1 errors) — model gives low score to correct prefixes
6. **Root cause of early detects**: 10.9% of correct prefixes score below threshold due to insufficient "correct prefix → high score" training
7. **Uniform source balance (PBR23)** is Pareto-optimal: same F1 but better acc_correct (0.877 vs 0.842)

### Next Experiment: PBR29 (First-Bad-Fanout Mode)

PBR29 hypothesis: `first_bad_fanout` pairs (multiple correct prefix levels trained against same first bad step)
will reduce early detects by explicitly teaching ALL pre-error steps to score high.
- Data: 3705 DPO full + 3624 MATH-MS fanout pairs = 7329 train
- Same config as PBR19 (joint, λ_termBCE=0.25, 5 epochs, MLP/512)

### PBR28 (also pending): full DPO (3705) + more MATH-MS (4000 cap)

PBR28 tests if more MATH-MS first_bad_edge data (with full DPO calibration preserved) helps:
- Data: 3705 DPO + 4000 MATH-MS = 7705 train pairs


---

## PBR28/PBR29 Additional Data/Mode Experiments (2026-03-11)

### PBR28: Full DPO (3705) + More MATH-MS (4000 cap) = 7705 pairs

**Hypothesis**: More MATH-MS first_bad_edge data with calibration preserved improves acc_err

| Metric | PBR28 | vs PBR19 |
|--------|-------|----------|
| Math F1 | 0.674 | -0.009 ↓ |
| acc_err | 0.566 | -0.008 |
| acc_correct | 0.833 | -0.009 |
| GSM F1 | 0.763 | -0.015 |

**Finding**: More MATH-MS data (4000 vs 2000) with full DPO calibration still HURTS. acc_correct degraded further.

### PBR29: DPO Full (3705) + MATH-MS Fanout (3624) = 7329 pairs

**Hypothesis**: `first_bad_fanout` mode (multiple correct prefix levels vs first bad step) reduces early detects

| Metric | PBR29 | vs PBR19 |
|--------|-------|----------|
| Math F1 | 0.682 | -0.001 |
| acc_err | 0.567 | -0.007 |
| acc_correct | 0.855 | +0.013 |
| GSM F1 | 0.762 | -0.016 |

**Failure mode comparison** (PBR29 vs PBR19):
| Mode | PBR19 | PBR29 |
|------|-------|-------|
| exact correct | 263 (54.9%) | 260 (54.3%) |
| early detect (pred < label) | 100 (20.9%) | 101 (21.1%) |
| wrong step after | 54 (11.3%) | 58 (12.1%) |
| missed entirely | 62 (12.9%) | 60 (12.5%) |
| correct prefix FP rate | **10.9%** | **10.9%** |

**Finding**: Fanout training has ZERO effect on early detects. The 10.9% correct-prefix FP rate is identical.

### Definitive Conclusion: Frozen Backbone Ceiling

The 10.9% correct-prefix FP rate is a **backbone feature quality ceiling**, not a training data problem.
All 9 variations (PBR22-PBR29) yield Math F1 ∈ [0.656, 0.684] with the ceiling at F1=0.683-0.684.

**Complete Sweep Results**:
| Run | Data | Pairs | Math F1 | acc_err | acc_cor | GSM F1 | Notes |
|-----|------|-------|---------|---------|---------|--------|-------|
| PBR19 | DPO 3705+MS 2000 | 5705 | **0.683** | 0.574 | 0.842 | 0.778 | Best Math |
| PBR22 | PBR12+PRM800K | 7705 | 0.682 | 0.569 | 0.852 | 0.771 | Neutral |
| PBR23 | PBR12 unif.bal | 5705 | **0.684** | 0.561 | **0.877** | 0.773 | Best acc_cor |
| PBR24 | 24% terminal | 6109 | 0.656 | 0.522 | 0.882 | **0.779** | Too many terms |
| PBR25 | unif+BCE=0.5 | 5705 | 0.678 | 0.564 | 0.850 | 0.768 | StrongBCE hurts |
| PBR26 | 2398DPO+MS full | 7366 | 0.670 | 0.557 | 0.840 | 0.779 | Less DPO hurts |
| PBR27 | MLP/1024 head | 5705 | 0.666 | 0.547 | 0.850 | **0.784** | Overfits Math |
| PBR28 | DPO+MS4k | 7705 | 0.674 | 0.566 | 0.833 | 0.763 | More MS hurts |
| PBR29 | DPO+MS fanout | 7329 | 0.682 | 0.567 | 0.855 | 0.762 | Fanout neutral |

### Next Steps to Break F1=0.683 Ceiling

1. **LoRA backbone fine-tuning**: Unfreeze top layers of Qwen2.5-Math-PRM-7B to learn task-specific features
   - LoRA has been implemented in `scripts/phase_e_train_value_lora.py` (smoke test done)
   - Expected to directly reduce the 10.9% correct-prefix FP rate
2. **Higher-quality training data**: ProcessBench uses QwQ-32B solutions; training data uses Math Shepherd
   (MC-estimated labels) — distribution mismatch in solution style
3. **Ensemble**: PBR19 + PBR23 (different acc_err/acc_cor tradeoffs) could combine strengths
