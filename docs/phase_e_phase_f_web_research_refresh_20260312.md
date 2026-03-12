# Phase E / Phase F Web Research Refresh (2026-03-12)

## Source Set

Primary sources checked this round:

1. VerifyBench: <https://arxiv.org/abs/2507.09884>
2. Hard2Verify: <https://arxiv.org/abs/2510.13744>
3. When to Trust the Cheap Check: <https://arxiv.org/abs/2602.17633>
4. DeepSeekMath / GRPO: <https://arxiv.org/abs/2402.03300>
5. On Designing Effective RL Reward at Training Time for LLM Reasoning: <https://arxiv.org/abs/2410.15115>
6. DeepSeek-R1: <https://arxiv.org/abs/2501.12948>
7. PRIME: <https://arxiv.org/abs/2502.01456>
8. ThinkPRM: <https://arxiv.org/abs/2504.16828>
9. GenPRM: <https://arxiv.org/abs/2504.00891>
10. VPRM: <https://arxiv.org/abs/2601.17223>
11. MASH: <https://arxiv.org/abs/2510.01152>
12. TRL GRPOTrainer docs: <https://huggingface.co/docs/trl/grpo_trainer>
13. DAPO: <https://arxiv.org/abs/2503.14476>
14. Dr. GRPO: <https://arxiv.org/abs/2503.20783>
15. VAPO: <https://arxiv.org/abs/2504.05118>
16. Is PRM Necessary?: <https://arxiv.org/abs/2505.11227>

New papers mirrored into `docs/relatedPapers/` this round:

1. `docs/relatedPapers/2025.findings-acl.747.pdf`
2. `docs/relatedPapers/2503.14476.pdf`
3. `docs/relatedPapers/2503.20783.pdf`
4. `docs/relatedPapers/2504.05118.pdf`
5. `docs/relatedPapers/2505.11227.pdf`

## Cross-Paper Takeaways

### 1. Selective verification is now a first-class system design, not a minor threshold tweak

`VerifyBench`, `Hard2Verify`, and `When to Trust the Cheap Check` all point in the same direction:

1. verifier quality is highly slice-dependent,
2. cheap checks are often good enough on easy/high-confidence cases,
3. but low-confidence and adversarial slices need escalation rather than blind trust in one scalar score.

Repo implication:

1. `Phase E` should stop treating one scalar head as the sole end-state architecture.
2. A better deployment target is `cheap verifier -> abstain/escalate gate -> stronger verifier/judge`.
3. The value of this design must be measured per domain, not assumed universal.

### 2. RL bottlenecks are as much about reward geometry and benchmark choice as optimizer choice

`DeepSeekMath`, the TRL `GRPOTrainer` docs, `On Designing Effective RL Reward...`, `DAPO`, `Dr. GRPO`, and `VAPO` jointly reinforce a few practical constraints:

1. GRPO-style updates need real within-group reward variance.
2. KL needs to start conservative; large beta can dominate reward before policy learning stabilizes.
3. Reward shaping matters. Raw per-step scores are easy to exploit; clipped deltas / bounded progress rewards are safer.
4. If the generator is already near the benchmark ceiling, RL often turns into policy noise rather than improvement.

Repo implication:

1. `GSM8K` at ~95.5% is a poor main GRPO arena for the current policy.
2. `Phase F` should treat harder reasoning benchmarks as the primary RL proving ground.
3. Any future live RL run should keep explicit reward-hacking probes and worst-generator monitoring in the loop.

### 3. Frontier verifier work is moving toward critic-style or generative verification, not blind scalar trust

`PRIME`, `ThinkPRM`, `GenPRM`, `VPRM`, and `MASH` all suggest the same trend:

1. richer critic behavior beats a lone scalar in hard settings,
2. selective extra compute / abstention is part of the solution,
3. process signals are most useful when they help routing, search, or critique rather than being treated as an oracle reward everywhere.

Repo implication:

1. `Phase E` should invest in hybrid verifier systems and ambiguous-slice escalation.
2. `Phase F` should remain `heuristic/BC first, RL second`.
3. from-scratch RL controller promotion is still unjustified unless it beats strong heuristic/BC baselines on held-out test splits.

## Repo-Local Validation Against Those Ideas

### 1. Cheap -> strong gating only helps on the current GSM-like slices

Run artifact:

1. `assets/artifacts/phase_e_gate_sweeps/phase_e_gate_phasef_0312_1630_20260312T083206Z/summary.json`

Observed:

1. `math_p26_to_p32`: best threshold is `0.00`; strong usage `0.0%`; weak AUC `0.8882` stays better than strong AUC `0.8685`.
2. `math_p31_to_p32`: best threshold is `0.00`; strong usage `0.0%`; weak AUC `0.8823` stays better than strong AUC `0.8685`.
3. `gsm_p19_to_p31`: best threshold `0.10`; AUC improves from `0.9148 -> 0.9170` with only `9.1%` strong usage.
4. `gsm_p19_to_p33`: best threshold `0.15`; AUC improves from `0.9148 -> 0.9175` with `16.3%` strong usage.

Interpretation:

1. The cheap-to-strong design is real, but it is not globally beneficial in the current repo.
2. On Math, the current stronger verifier candidate (`PBR32`) is not actually stronger on ProcessBench than `PBR26/PBR31`.
3. On GSM, selective escalation works and is budget-feasible.

### 2. The previous offline GRPO proxy had a real methodological pitfall

Code fix landed in `scripts/phase_f_grpo_feasibility.py`.

Old problem:

1. the script formed bootstrap groups only from traces with the same label family,
2. which made outcome reward constant within the group,
3. which forced outcome advantage to `0` and overstated the gap between process reward and outcome reward.

Fix:

1. default group sampling is now `mixed_pool`,
2. `label_matched` remains available only as an explicit diagnostic mode,
3. a unit test was added in `tests/unit/test_phase_f_grpo_feasibility.py`.

### 3. With the corrected mixed-pool proxy, RL signal is usable but not the main bottleneck

Run artifact:

1. `assets/artifacts/phase_f_grpo_feasibility/strong_mixedpool_0312_1635/grpo_feasibility_20260312T083443Z.json`

Observed:

1. Math (`PBR32` rows): process `|adv|=0.811`, outcome `|adv|=0.864`, recommended max beta `0.200`.
2. GSM (`PBR33` rows): process `|adv|=0.805`, outcome `|adv|=0.872`, recommended max beta `0.500`.
3. Both domains remain `FEASIBLE` under the script's signal tests.

Interpretation:

1. The current RL blocker is not simply “process reward has no variance”.
2. Benchmark saturation and optimization stability are more plausible bottlenecks than raw signal absence.
3. Future GRPO runs should start around `beta=0.05`, but they should move to harder benchmarks before spending more overnight GPU on `GSM8K`.

### 4. Stronger controllers still need teacher structure; from-scratch RL collapses

Run artifacts:

1. `assets/artifacts/phase_f_bc/phase_f_focus_bc_0312_1638_20260312T083525Z/summary.json`
2. `assets/artifacts/phase_f_bc/phase_f_focus_bc_then_rl_0312_1638_20260312T083525Z/summary.json`
3. `assets/artifacts/phase_f_rl_like/phase_f_focus_rl_0312_1643_20260312T083645Z/summary.json`

Observed:

1. `bc_only`:
   - `p31_math = 0.7987`
   - `p32_math = 0.8309`
   - `p31_gsm = 0.9000`
2. `bc_then_rl` with milder RL settings:
   - `p31_math = 0.8171`
   - `p32_math = 0.8395`
   - `p31_gsm = 0.9082`
3. from-scratch RL-like (`hidden_dim=32`, `robust_lambda=0.2`) collapsed to:
   - `p31_math = 0.0000`
   - `p31_gsm = 0.0000`

Interpretation:

1. warm-started RL can still add a modest gain over BC on some held-out splits,
2. but teacher structure remains essential,
3. and from-scratch RL is still far too unstable to promote.

### 5. Modern preflight says P31 is useful but not deployment-safe by default

Run artifact:

1. `assets/artifacts/phase_f_logs/phase_f_preflight_wait_0312_1641/final_summary.md`

Observed:

1. `p31_gsm` has wider near-best threshold window (`0.220`) than `p26_gsm` (`0.140`), but worst-generator logo F1 remains `0.0000`.
2. `p31_math` improves worst-generator logo F1 to `0.4251`, but `confidence_tail` reward-hacking risk is still `high`.
3. `p31` is better as a research/useability slice than as a blind production verifier.

Interpretation:

1. `Phase F` should keep preflight probes mandatory.
2. Controllers should optimize around both threshold width and adversarial-tail behavior.
3. A strong scalar verifier is still not a safe single-point deployment target.

## Current Design Decisions

### Phase E

1. Keep the running `L2` experiment (`gated_mlp`, all-layer LoRA, stronger contrastive) alive; it is the highest-value active training run.
2. Use cheap-to-strong gating only on GSM-like slices for now.
3. Do not spend new math budget on `p26/p31 -> p32` escalation until a truly stronger Math verifier exists.

### Phase F

1. Default controller stack remains: `heuristic -> BC -> optional mild RL`.
2. from-scratch RL-like controller training stays a control arm only.
3. Keep reward-hacking preflight and worst-generator metrics as promotion gates.

### RL-Specific Guardrails

1. Prefer harder training benchmarks over saturated `GSM8K`.
2. Start KL around `0.05`, not at the upper end of the feasible band.
3. Keep reward shaping bounded (`clip_delta`, clipped progress, or equivalent).
4. Abort promotion if worst-generator held-out F1 collapses, even if mean reward improves.
5. Treat offline mixed-pool GRPO feasibility as a sanity check, not as proof that live RL will help.
