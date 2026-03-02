# Phase C Plan: ABR-Centric Implementation and Training Guidance

This file is the implementation guidance for **Phase C**, the stage where the
project moves from Phase B PEFT baselines into the **unique BCR/ABR method**
work.

Phase C is not "more PEFT tuning". Its purpose is to build, stabilize, and
evaluate the first usable version of:
- a prefix-level **value head**,
- a **BCR-lite** process-consistency training path,
- an **ABR-lite** step-level adaptive verification controller,
- and the first safe **router RL** formulation.

The plan below is grounded in:
- `idea_formulation.md`
- `idea_polish.md`
- `TODO_ours.md`
- `phase_B_plan.md`
- local source materials in `assets/`:
  - `assets/BCR draft_20251211.pdf`
  - `assets/BCR on step level.pptx`
  - `assets/BCR_discussion_20251219.pptx`

## 1. Why Phase C Exists

Phase B answered the baseline question:
- can PEFT improve target tasks?
- where does it help?
- where does it fail?

Current Phase B conclusions:
1. StrategyQA is a stable positive PEFT track.
2. GSM8K long-CoT PEFT is sensitive to objective design and checkpoint drift.
3. Cross-task interference is real and sometimes severe.

That means the project has now reached the point where "plain fine-tuning"
alone is no longer the unique contribution. The real research contribution must
come from **process-aware reasoning control**.

So Phase C exists to answer:
1. Can we estimate prefix-level future success in a useful way?
2. Can we use that signal to make reasoning more faithful?
3. Can we verify only when needed instead of checking every token or step?
4. Can we do this without collapsing training stability?

## 2. Restating the ABR Idea Clearly

ABR stands for **Adaptive Bellman Reasoning**.

The motivating criticism of vanilla BCR is:
1. token-level Bellman regularization is too fine-grained and expensive,
2. it applies consistency checking everywhere, even when a step is easy,
3. it gives no explicit decision mechanism for "generate more" vs "verify now".

ABR changes this by moving to **step-level** reasoning and introducing an
explicit controller.

### 2.1 Core objects

Let:
- `x` be the question,
- `tau = (s_1, ..., s_T)` be the reasoning steps,
- `h_t = (x, s_1:t)` be the partial reasoning state after step `t`,
- `V_phi(h_t)` be the value head estimate of eventual success from that prefix.

ABR adds a router policy over:
- `gen`: generate the next reasoning step,
- `ver`: verify consistency using history,
- `fin`: stop and answer.

### 2.2 Meaning of each action

`gen`
- Continue the reasoning process by emitting the next step.

`ver`
- Do **not** emit new user-visible reasoning text.
- Instead, inspect the current state against historical reasoning anchors and
  update the internal meta-state used by the controller.
- In training, this is where consistency-based supervision is applied.

`fin`
- Stop the step loop and produce the final answer.

### 2.3 Why `ver` is still a real action

This is important because it may be challenged conceptually.

`ver` is a real action because:
1. it changes the agent state,
2. it changes future decisions,
3. it incurs compute cost,
4. it affects final reward.

It is a **state-update action**, not a text-emission action.

### 2.4 TSS: Target-Step Selection

When the router chooses `ver`, ABR must decide **what to verify against**.

Instead of checking only the immediately previous step, ABR uses a
history-selection mechanism:
- select an anchor step `k < t`,
- compare the current prefix against that anchor.

This is the project-specific adaptation inspired by the step-level ABR slides:
- the selected historical step is not just "extra context",
- it is an **attribution anchor** for consistency checking.

## 3. Phase C Scope

Phase C should cover:
1. value-head bootstrap,
2. BCR-lite,
3. ABR-lite,
4. first router-only RL.

Phase C should **not** cover:
1. joint LM + value + router RL from scratch,
2. GSM8K-first RL,
3. multi-agent extensions,
4. heavy PPO-first optimization,
5. speculative advanced smoothness/spectral objectives as primary training.

## 4. Non-Negotiable Strategic Decisions

These choices are required to avoid the chicken-and-egg failure mode.

### 4.1 StrategyQA first

Phase C should start on **StrategyQA**, not GSM8K.

Reason:
1. StrategyQA Phase B PEFT is stable and positive.
2. GSM8K still has long-CoT drift and checkpoint-selection issues.
3. If Phase C starts on GSM8K, failures will be underdetermined:
   - value problem,
   - router problem,
   - or unstable source model.

### 4.2 Frozen-backbone first

Train the first value head with:
- frozen LM backbone,
- no router RL,
- explicit empirical prefix targets.

This avoids the chicken-and-egg loop:
- bad value head harms router learning,
- bad router hides useful value supervision,
- joint updates make attribution impossible.

### 4.3 Step-level, not token-level

All Phase C logic should be step-level first.

The current `src/ours/data/step_builder.py` already gives the correct starting
abstraction:
- `question` step,
- `reasoning` steps,
- optional `answer` step.

Phase C should build on that instead of inventing a token-level control system
first.

### 4.4 Best-checkpoint policy, not final-checkpoint policy

Where Phase C depends on a Phase B backbone:
- prefer the **best held-out checkpoint**,
- not the final checkpoint, especially for GSM8K.

## 5. Phase C Inputs and Outputs

### 5.1 Required inputs

From Phase A:
- prepared train/validation/test JSONL
- frozen evaluator protocol
- step builder and canonical schema

From Phase B:
- best StrategyQA PEFT checkpoint
- best StrategyQA PEFT config
- best GSM8K checkpoint policy

### 5.2 Required outputs

Each official Phase C run should produce:
1. config snapshot,
2. manifest with source artifacts and versions,
3. train/eval logs,
4. value metrics,
5. answer metrics,
6. faithfulness metrics,
7. router/action traces where applicable,
8. JSON + Markdown summaries.

## 6. Phase C Architecture

Recommended new modules:

```text
src/ours/phase_b/
  value_head.py
  value_targets.py
  value_losses.py
  corruptions.py
  faithfulness_eval.py
  action_space.py
  heuristic_router.py
  tss.py
  router_model.py
  rewards.py

scripts/
  phase_b_train_value.py
  phase_b_eval_faithfulness.py
  phase_b_train_bcr_lite.py
  phase_b_run_abr_lite.py
  phase_b_train_abr_rl.py
```

The code should remain under `phase_b` paths for continuity with the existing
repo structure, but conceptually this is now **Phase C work**.

## 7. Data Contracts for Phase C

Phase C needs more than plain `(prompt, completion)` training pairs.

### 7.1 Step sequence artifact

Base object:
- output of `src/ours/data/step_builder.py`

Per sample:
- `sample_id`
- `dataset`
- ordered steps
- step roles
- original source mapping

### 7.2 Prefix artifact

For every step sequence, produce all prefixes:
- `h_0 = question`
- `h_1 = question + step_1`
- ...
- `h_t = question + step_1:t`

Recommended record fields:
- `prefix_id`
- `sample_id`
- `dataset`
- `step_index`
- `prefix_text`
- `prefix_roles`
- `gold_answer`
- `full_num_steps`
- `source_step_ids`

### 7.3 Rollout target artifact

For each prefix:
- sample `K` continuations from the current frozen model,
- score final correctness,
- compute empirical success rate.

Recommended fields:
- `prefix_id`
- `k_rollouts`
- `n_correct`
- `success_rate`
- optional rollout lengths / answer distribution diagnostics

### 7.4 Corruption artifact

For selected prefixes, create minimally corrupted variants.

Examples:
- arithmetic sign flip,
- invalid substitution,
- premise deletion,
- entity swap,
- yes/no flip cue,
- dropped bridge sentence.

Recommended fields:
- `clean_prefix_id`
- `corruption_id`
- `corruption_type`
- `corrupted_text`
- `corruption_step_index`

### 7.5 Router trace artifact

Needed later for ABR-lite and RL.

Fields:
- `sample_id`
- `time_index`
- `state_features`
- `chosen_action`
- `anchor_index` if `ver`
- `value_before`
- `value_after`
- `delta`
- `cost`
- `done`

## 8. Training Scheme: The Critical Part

This is the main Phase C guidance.

## 8.1 Stage C0: Contract Freeze

Before implementation:
1. freeze action semantics,
2. freeze step-level state definition,
3. freeze first dataset and checkpoint,
4. freeze first metrics.

Deliverable:
- this file plus a short `docs/method_v1.md` if needed later.

## 8.2 Stage C1: Prefix Artifact Pipeline

Implement:
- step sequence to prefixes,
- prefix rollout target generation,
- corruption generation.

This stage is purely data/target preparation.

No RL here.
No joint LM/value updates here.

Exit gates:
1. deterministic prefix generation,
2. rollout target generation completes on smoke subset,
3. corruption pipeline produces traceable outputs.

## 8.3 Stage C2: Value-Head Bootstrap

This is the first actual training stage.

### Objective

Train only the value head first, with a frozen backbone.

Primary target:
- empirical rollout success `v_hat(h_t)`

Primary loss:

`L_cal = MSE(V_phi(h_t), v_hat(h_t))`

Optional companion loss:

`L_ctr = max(0, m - V(h_clean) + V(h_corrupt))`

Total v1 value loss:

`L_value = L_cal + lambda_M * L_ctr`

### Why this is the first stage

Because it breaks the chicken-and-egg loop:
- the value head is trained from empirical outcomes,
- not from its own predictions,
- and not from an untrained router.

### Implementation details

Backbone:
- start from best StrategyQA PEFT backbone
- freeze all shared transformer layers initially

Value head:
- simple linear head + sigmoid
- scalar output in `[0,1]`

Batching:
- batch prefixes aggressively
- keep rollout generation separate from value training

Recommended initial settings:
- dataset: StrategyQA
- backbone: best StrategyQA PEFT checkpoint
- rollout `K`: small first, e.g. `4` or `8`
- train only value head parameters first

Exit gates:
1. calibration better than trivial baseline,
2. corruption AUC above random,
3. value drop localizes corruption better than untrained head.

## 8.4 Stage C3: BCR-Lite

Only after C2 is stable.

### Objective

Add Bellman-style consistency, but do it conservatively.

Recommended total loss:

`L_total = L_sft + lambda_B * L_Bellman + lambda_C * L_cal + lambda_M * L_ctr`

with:
- `lambda_B` small initially,
- `L_cal` still present,
- `L_ctr` optional but recommended,
- stop-gradient on the next-step target.

### Important training rule

Do not let Bellman consistency be the only supervision signal.

Bellman-only training is unsafe early because:
1. it can reinforce a bad value function,
2. it can drive self-consistent but wrong internal states,
3. it hides whether the value head is actually predictive.

### Update policy

Recommended order:
1. start with frozen backbone + trainable value head,
2. then unfreeze a small PEFT/LoRA subset or shared layers,
3. monitor answer accuracy and faithfulness together.

Optional stabilization:
- EMA target network for value targets
- delayed LM unfreezing

Exit gates:
1. no NaN/divergence,
2. no unacceptable answer-quality collapse,
3. at least one faithfulness metric improves.

## 8.5 Stage C4: ABR-Lite (Heuristic Router)

ABR-lite should be implemented before RL.

### Why

This tests the action semantics and control loop without RL confounds.

### Router state

At step `t`, the router state should include:
- query embedding summary,
- current step embedding,
- history summary,
- current value `V(h_t)`,
- recent value deltas,
- remaining step budget,
- remaining verification budget,
- optional corruption-like anomaly indicators later.

### Heuristic actions

`gen`
- default action when confidence is stable and uncertainty is low.

`ver`
- trigger when:
  - uncertainty is high,
  - value drops sharply,
  - value spikes suspiciously,
  - step count passes a threshold,
  - contradiction cue or anomaly flag appears.

`fin`
- trigger when:
  - confidence is high,
  - budget is low,
  - answer-ready pattern is detected,
  - or the maximum step count is reached.

### TSS in heuristic mode

Before a learned anchor selector exists, use:
- top-1 attention over historical steps,
- or deterministic heuristics:
  - previous key step,
  - most semantically similar historical step,
  - most recent high-value-drop step.

### Verification signal

For ABR-lite, `ver` should at minimum log:
- selected anchor,
- value difference,
- whether the controller changed later behavior.

Exit gates:
1. hard samples use `ver` more often than easy samples,
2. verification stays inside budget,
3. heuristic ABR beats fixed verification schedule on at least one
   faithfulness-efficiency frontier.

## 8.6 Stage C5: Router-Only RL

Only after ABR-lite is stable.

### Non-negotiable rule

Start with **router-only RL**.

Freeze:
- backbone LM,
- value head,
- TSS machinery except the router if needed.

Do not jointly optimize everything at once.

### Reward

Recommended initial reward:

`R = 1[correct] - beta * N_ver - eta * token_cost - zeta * invalid_action_penalty`

Where:
- `N_ver` penalizes excessive verification,
- `token_cost` penalizes long reasoning,
- `invalid_action_penalty` handles bad transitions such as impossible `fin`.

### Algorithm

Use the simplest stable method first:
- REINFORCE with baseline,
- or light actor-critic.

Do not start with PPO unless variance makes it necessary.

### Warm start

Initialize the router from:
- heuristic traces,
- or supervised imitation of heuristic decisions.

This is directly motivated by the ABR stability discussion in the slides:
- do not ask the router to discover everything from scratch.

### Anti-hacking checks

Track from day 1:
- always-gen collapse,
- always-ver collapse,
- premature-fin collapse,
- budget exhaustion hacks,
- reward inflation with low true accuracy.

Exit gates:
1. router does not collapse,
2. action distribution is sensible,
3. at least one constrained frontier improves over heuristic ABR.

## 9. Tricky Failure Modes and How to Avoid Them

### 9.1 Chicken-and-egg coupling

Problem:
- bad value head -> bad router reward
- bad router -> poor value supervision

Fix:
- frozen-backbone value bootstrap first,
- heuristic router first,
- router-only RL only after both are stable.

### 9.2 Objective leakage

Problem:
- if the value head is trained from gold next-token likelihood, it may learn
  teacher-forcing imitation rather than true future success.

Fix:
- use terminal correctness,
- rollout success,
- corruption contrast,
- Bellman only as a regularizer later.

### 9.3 Self-consistent but wrong value

Problem:
- Bellman consistency can be minimized even by a wrong but self-consistent value.

Fix:
- always keep calibration against empirical prefix targets,
- monitor corruption AUC,
- do not rely on Bellman loss alone.

### 9.4 Router learns to avoid verification

Problem:
- `ver` has cost,
- router may learn to skip it always.

Fix:
- use constrained reward design,
- imitation warm start,
- monitor action histograms,
- compare against fixed-verify baselines.

### 9.5 Router over-verifies

Problem:
- if verification appears useful early, router may spam it.

Fix:
- explicit verification budget,
- per-verify penalty,
- step budget,
- action entropy monitoring.

### 9.6 TSS instability

Problem:
- full soft attention over all steps may be noisy or expensive.

Fix:
- start with lightweight attention,
- optionally top-k truncate,
- begin with deterministic top-1 for smoke runs.

### 9.7 GSM8K contamination of early conclusions

Problem:
- GSM8K backbone behavior is not stable enough yet for first Phase C claims.

Fix:
- do StrategyQA first,
- use GSM8K only as a later stress test after StrategyQA is stable.

## 10. Module-Level Implementation Plan

## 10.1 `src/ours/phase_b/value_head.py`

Responsibilities:
- define bounded scalar head,
- save/load cleanly,
- expose forward API for per-prefix values.

## 10.2 `src/ours/phase_b/value_targets.py`

Responsibilities:
- convert prefixes to rollout targets,
- cache rollout results,
- support smoke/full modes.

## 10.3 `src/ours/phase_b/corruptions.py`

Responsibilities:
- dataset-aware corruption transforms,
- corruption metadata,
- deterministic seed behavior.

## 10.4 `src/ours/phase_b/value_losses.py`

Responsibilities:
- calibration MSE,
- contrastive margin loss,
- Bellman loss with stop-gradient,
- combined weighted objective.

## 10.5 `src/ours/phase_b/faithfulness_eval.py`

Responsibilities:
- Brier / optional ECE,
- corruption AUC,
- localization metrics,
- efficiency summaries.

## 10.6 `src/ours/phase_b/action_space.py`

Responsibilities:
- action enums,
- valid transition rules,
- budget accounting helpers.

## 10.7 `src/ours/phase_b/tss.py`

Responsibilities:
- anchor scoring,
- top-k selection,
- deterministic and sampled modes.

## 10.8 `src/ours/phase_b/heuristic_router.py`

Responsibilities:
- threshold-based control,
- value-delta heuristics,
- finish logic,
- trace emission.

## 10.9 `src/ours/phase_b/router_model.py`

Responsibilities:
- learned router policy,
- state encoder,
- action logits,
- optional anchor-head later.

## 10.10 `src/ours/phase_b/rewards.py`

Responsibilities:
- reward shaping,
- cost penalties,
- anti-hacking diagnostics.

## 10.11 Script entrypoints

`scripts/phase_b_train_value.py`
- smoke/full value-head training

`scripts/phase_b_eval_faithfulness.py`
- offline faithfulness evaluation

`scripts/phase_b_train_bcr_lite.py`
- first joint BCR-style training

`scripts/phase_b_run_abr_lite.py`
- heuristic router execution + reporting

`scripts/phase_b_train_abr_rl.py`
- router-only RL

## 11. Configuration Plan

Recommended future config families:

```text
configs/phase_c/
  value_smoke_strategyqa.json
  value_full_strategyqa.json
  bcr_lite_smoke_strategyqa.json
  bcr_lite_full_strategyqa.json
  abr_lite_smoke_strategyqa.json
  abr_lite_full_strategyqa.json
  abr_rl_smoke_strategyqa.json
  abr_rl_full_strategyqa.json
```

Later:

```text
configs/phase_c/
  value_smoke_gsm8k.json
  bcr_lite_smoke_gsm8k.json
  abr_lite_smoke_gsm8k.json
```

## 12. Experiment Matrix

### 12.1 Value-head matrix

Must compare:
1. random/untrained value head,
2. rollout-calibrated only,
3. rollout + corruption contrastive,
4. rollout + corruption + Bellman.

### 12.2 BCR-lite matrix

Must compare:
1. PEFT baseline,
2. value-head-only baseline,
3. BCR-lite.

### 12.3 ABR-lite matrix

Must compare:
1. no verification,
2. fixed verification schedule,
3. heuristic ABR-lite.

### 12.4 Router RL matrix

Must compare:
1. heuristic ABR-lite,
2. router imitation-only,
3. router RL.

## 13. Evaluation Protocol for Phase C

All methods must be evaluated under one unified interface.

### 13.1 Accuracy metrics

Reuse frozen Phase A metrics:
- accuracy,
- parse error rate where relevant,
- parseable accuracy.

### 13.2 Faithfulness metrics

Primary:
- Brier score
- optional ECE
- corruption AUC
- value-drop localization near corruption

Secondary:
- value trajectory variance,
- change-point diagnostics,
- optional smoothness plots

### 13.3 Efficiency metrics

- generated tokens
- reasoning steps
- number of `ver` actions
- wall-clock cost

### 13.4 Frontier reporting

Every ABR report should include:
- accuracy vs verify-count
- accuracy vs token cost
- faithfulness vs token cost

## 14. Validation and Test Gates

### 14.1 Unit tests

Required:
1. prefix generation
2. rollout target serialization
3. corruption validity
4. value loss correctness
5. action validity rules
6. TSS anchor selection behavior

### 14.2 Integration tests

Required:
1. StrategyQA value-head smoke
2. StrategyQA BCR-lite smoke
3. StrategyQA ABR-lite smoke
4. StrategyQA router-RL smoke

### 14.3 Reproducibility checks

Required:
1. same seed -> same prefix artifact hashes
2. same config -> same manifest signature
3. same eval settings -> stable report schema

## 15. Recommended Execution Order

This is the concrete Phase C order.

1. Freeze Phase B baseline policies.
2. Build prefix artifact generator.
3. Build rollout target generator.
4. Build corruption generator.
5. Implement and train value head on StrategyQA smoke.
6. Add faithfulness evaluator.
7. Scale value head to StrategyQA full.
8. Add Bellman loss and run first BCR-lite smoke.
9. Run first StrategyQA full BCR-lite.
10. Implement heuristic ABR-lite.
11. Compare heuristic ABR-lite to fixed verification.
12. Implement router imitation warm start.
13. Implement router-only RL.
14. Only after all above are stable, port to GSM8K.

## 16. What Counts as Success for Phase C

Phase C is successful if all of the following are true:

1. A value head trained on StrategyQA produces non-trivial calibration and
   corruption sensitivity.
2. BCR-lite preserves usable answer quality while improving at least one
   faithfulness metric.
3. ABR-lite beats a fixed verification schedule on at least one
   faithfulness-efficiency frontier.
4. Router-only RL improves on heuristic ABR-lite on at least one constrained
   frontier without collapsing.
5. The full Phase C stack remains reproducible and diagnosable from run
   artifacts.

## 17. Immediate Next Step After Writing This Plan

The first implementation item should be:

1. prefix artifact generation,
2. rollout target generation,
3. corruption generation,
4. then `scripts/phase_b_train_value.py`.

Do not start Phase C by writing the RL trainer first.
