# Phase F Plan: Conservative RL on a Frozen Same-Family Value Head

This file defines the official conditional **Phase F** plan.

Date baseline: 2026-03-10.

Phase transition rule:
1. `Phase E` remains the active execution stage until its trust gate passes.
2. `Phase F` is **conditional** on a successful `Phase E` outcome.
3. The trigger is narrow and explicit:
   - one `Math-Shepherd` checkpoint family passes the `Phase E` trust matrix,
   - one concrete `best_value_head.pt` is selected by fixed rules,
   - same-family held-out metrics stay positive and stable enough for promotion,
   - and that checkpoint must come from the post-fix
     `phase_e_pairs_v2 / step_label_pair_mode=first_bad_edge_strict` pipeline,
     not from legacy nearest-negative artifacts.
4. `Phase F` does **not** assume cross-dataset generalization.
5. `Phase F` starts from the opposite assumption:
   - the promoted head is only trustworthy inside a bounded same-family support.

---

## 0. Why Phase F Exists

If `Phase E` succeeds, the next scientific question is no longer:
1. "can the value head learn anything at all?"

That question belongs to `Phase E`.

The next question becomes:
1. "how should we use one trusted same-family value head to improve sequential
   decision-making without letting RL exploit its blind spots?"

This matters because:
1. RL assumes the reward/value signal is usable.
2. If the reward/value model is poor, RL amplifies that error.
3. Current PRM literature shows that even useful process reward models remain
   vulnerable to domain shift, proxy hacking, and overoptimization.
4. Therefore `Phase F` is not a generic RL expansion stage.
5. It is a **conservative deployment stage** for one frozen and bounded-support
   process reward model.

Project-level interpretation:
1. A successful `Phase E` result does **not** mean we solved general value
   estimation.
2. It means we may have one process reward model that is usable inside the same
   family under strong constraints.
3. `Phase F` is the stage that tests exactly that claim.

---

## 1. Phase F Objective

Primary objective:
1. Use one trusted `Math-Shepherd`-trained value head to improve same-family
   math reasoning control under explicit compute budgets.

Secondary objective:
1. Determine whether the value head is strong enough to support:
   - deterministic routing,
   - then router-only RL,
   - without reward hacking or benchmark collapse.

Phase F is successful only if it improves a constrained downstream frontier:
1. final correctness,
2. correctness vs compute,
3. or faithfulness vs compute,
4. while preserving same-family held-out value quality.

---

## 2. Non-Negotiable Assumptions

These assumptions are part of the plan, not optional interpretations.

1. Cross-dataset generalization is hard and is **not** required for `Phase F`.
2. The selected `Math-Shepherd` value head is treated as a **bounded-support
   process reward model**, not a universal critic.
3. The first RL stage must remain inside the same math/process family as the
   promoted value head.
4. The value head must remain **frozen** during initial RL.
5. The base LM must remain **frozen** during initial RL unless a later gate
   explicitly authorizes limited adaptation.
6. The first policy class must be small and interpretable:
   - router,
   - branch selector,
   - retry policy,
   - or verifier-calling controller.
7. `Phase F` does not begin with full-token policy RL.
8. `Phase F` does not begin with joint LM + value + router optimization.

---

## 3. What Phase F Is And Is Not

### In scope

1. Freeze one promoted `best_value_head.pt` from `Phase E`.
2. Use that head as dense shaping or ranking guidance inside the same family.
3. Build a deterministic `ABR-lite` controller before RL.
4. Warm-start a router from heuristic traces if needed.
5. Run router-only RL on top of a frozen generator and frozen value head.
6. Measure correctness/faithfulness/efficiency frontiers under explicit budget
   constraints.
7. Add anti-hacking diagnostics and promotion gates.

### Out of scope

1. Cross-domain reward-model claims.
2. StrategyQA-first RL.
3. GSM8K-first RL.
4. Joint LM + value + router online RL from the beginning.
5. Online reward-model training during policy optimization.
6. Treating the Phase E head as the sole final evaluator.
7. Using reward increase alone as a success metric.

---

## 4. Phase F Scientific Position

What we are trying to prove:
1. a same-family process reward model can guide a narrow policy to make better
   sequential control decisions than fixed heuristics,
2. without destroying held-out reward-model validity,
3. and without obvious reward hacking.

What we are **not** trying to prove:
1. that the value head generalizes broadly across benchmark families,
2. that the value head is an accurate Bellman-consistent environment value
   function,
3. that unrestricted policy optimization on the whole LM is already safe.

Therefore the correct framing is:
1. `Phase E` proves bounded-support learnability,
2. `Phase F` proves bounded-support controllability.

---

## 5. Inputs From Phase E

`Phase F` may start only after these artifacts exist.

Required artifacts:
1. one promoted `best_value_head.pt`,
2. the corresponding run manifest and config,
3. the trust-matrix summary showing why this checkpoint family was selected,
4. frozen held-out same-family pair artifacts,
5. benchmark canary reports:
   - `ProcessBench` and/or other same-family external canaries,
6. one fixed reference LM checkpoint.

Required trust evidence:
1. source-family held-out mean is clearly above random,
2. seed instability is bounded enough that promotion is defensible,
3. no benchmark canary shows catastrophic collapse,
4. the selected checkpoint is chosen by scripted policy, not manual log reading.

If any of the above is missing:
1. `Phase F` must not start.

---

## 6. Core Design Rule: Use The Head As Frozen Reward Shaping, Not As A Free Critic

The promoted `Phase E` head should be used in a constrained way.

Allowed uses:
1. prefix improvement shaping,
2. branch ranking,
3. candidate reranking,
4. continue / verify / stop guidance,
5. uncertainty-aware gating when multiple trusted heads are available.

Disallowed uses in the first RL stage:
1. optimizing raw `V(prefix)` as the only reward,
2. treating the head as exact ground-truth future return,
3. online re-fitting the head while policy optimization is running,
4. using the same head as both optimizer and sole final judge.

Interpretation rule:
1. the head is a proxy signal,
2. but the optimization target must still be anchored by final task success and
   explicit cost penalties.

---

## 7. Reward Design

Recommended initial reward:

`R = R_terminal + lambda_s * R_shape - beta_ver * N_ver - beta_tok * token_cost - beta_inv * invalid_action_penalty`

Where:
1. `R_terminal` is the anchor term:
   - final correctness,
   - verifier-approved final success,
   - or another terminal same-family task success signal.
2. `R_shape` is a bounded shaping term derived from the frozen value head.
3. `N_ver` penalizes unnecessary verification calls.
4. `token_cost` penalizes excessive reasoning length or compute.
5. `invalid_action_penalty` penalizes impossible or degenerate controller moves.

### 7.1 Preferred shaping forms

Prefer conservative shaping forms such as:
1. clipped `V(prefix_t) - V(prefix_{t-1})`,
2. pairwise branch preference bonus,
3. bonus for selecting the higher-ranked branch among local alternatives,
4. thresholded improvement bonus rather than raw unbounded score maximization.

### 7.2 Reward rules

1. Always keep a terminal anchor.
2. Clip and normalize the shaping term.
3. Keep shaping smaller than the terminal reward scale.
4. Penalize compute explicitly.
5. Log each reward component separately.
6. Never report only the summed reward.

### 7.3 Why this design is mandatory

1. Dense shaping helps optimization.
2. Raw proxy maximization is the easiest path to reward hacking.
3. The value head was trained on ranked process quality, not on unrestricted
   online RL returns.

---

## 8. Policy Class Progression

The policy class must widen only after earlier gates pass.

### F-safe-1: deterministic controller

1. Fixed generator.
2. Frozen value head.
3. Heuristic rules over value-head outputs and simple uncertainty/cost signals.
4. No RL.

### F-safe-2: imitated controller

1. Learn a small controller from heuristic traces or scripted decisions.
2. Still keep generator and reward head frozen.
3. Use this only as a warm start, not as scientific proof.

### F-safe-3: router-only RL

1. Train only the controller/router.
2. Freeze generator.
3. Freeze value head.
4. Keep action space short-horizon and interpretable.

### F-unsafe-for-now

1. full-token PPO over the entire LM,
2. joint LM + reward-model online training,
3. long-horizon unrestricted search without support control,
4. generator adaptation before router-only RL is stable.

---

## 9. Action Space

The first `Phase F` action space should remain minimal.

Recommended actions:
1. `gen`: continue one reasoning step,
2. `ver`: call verification / judge a prefix or branch,
3. `backtrack`: return to a safer branch or earlier state,
4. `fin`: stop and emit the answer.

Optional later actions:
1. `branch`: explicitly fork a new candidate,
2. `rerank`: score and select among already-generated branches.

Action-space rule:
1. keep the first RL action space small enough that collapse modes are easy to
   diagnose.

---

## 10. Work Packages

## F0. Promotion Freeze

Goal:
1. freeze the exact `Phase E` artifact family that is allowed into RL.

Deliverables:
1. one explicit promoted checkpoint path,
2. one promotion manifest,
3. one immutable reward configuration,
4. one immutable held-out validation split,
5. one canary evaluation command set.

Pass condition:
1. every later `Phase F` run can be traced back to the same frozen reward-model
   artifact family.

## F1. Offline Scoring Sanity

Goal:
1. confirm that the promoted head still produces useful same-family ranking on
   held-out trajectories not used in RL.

Required checks:
1. candidate reranking quality,
2. branch ordering quality,
3. value-drop localization around bad steps,
4. stability of reward normalization.

Pass condition:
1. offline use of the head is still clearly better than random and clearly more
   useful than an untrained or mismatched head.

## F2. ABR-Lite Deterministic Controller

Goal:
1. convert the frozen head into a non-RL control policy.

Implementation direction:
1. threshold-based verify trigger,
2. value-drop trigger,
3. budget-aware finish rule,
4. optional branch reranking.

Comparison set:
1. fixed verification schedule,
2. no-verification baseline,
3. deterministic value-guided controller.

Pass condition:
1. deterministic control improves at least one constrained frontier over fixed
   schedules.

Interpretation rule:
1. if deterministic control cannot use the signal, router RL should not start.

## F3. Router Warm Start

Goal:
1. initialize the router from heuristic traces before RL.

Allowed warm starts:
1. imitation of heuristic controller actions,
2. behavior cloning on safe deterministic traces,
3. supervised branch-selection labels.

Pass condition:
1. the warm-started controller reproduces sensible action frequencies and does
   not immediately collapse.

## F4. Router-Only RL Smoke

Goal:
1. test whether RL can improve the controller without obvious reward hacking.

Training rule:
1. freeze LM,
2. freeze value head,
3. train router only,
4. use the simplest stable algorithm first:
   - REINFORCE with baseline,
   - or light actor-critic.
5. do not start with heavy PPO unless variance forces it.

Pass condition:
1. no immediate always-gen / always-ver / premature-fin collapse,
2. policy reward improves together with at least one external frontier,
3. held-out same-family reward-model metrics do not collapse.

## F5. Seeded Router RL Validation

Goal:
1. show that `F4` is not a lucky seed artifact.

Required comparisons:
1. multiple controller seeds,
2. at least one alternative promoted value-head candidate if available,
3. heuristic controller vs RL controller.

Pass condition:
1. the frontier improvement repeats across more than one seed or trusted reward
   candidate.

## F6. Limited Generator Adaptation (Optional)

Goal:
1. test whether minimal policy-side generator adaptation adds further benefit.

This stage is optional and gated by `F5`.

Non-negotiable restrictions:
1. start with tiny updates only,
2. keep strong KL/reference constraints,
3. keep the reward model frozen,
4. stop immediately if canaries degrade.

Pass condition:
1. limited generator adaptation improves the frontier without degrading held-out
   reward-model validity.

---

## 11. Training Algorithms

Recommended algorithm order:
1. deterministic controller,
2. imitation warm start,
3. REINFORCE with baseline,
4. light actor-critic,
5. PPO only if simpler methods are too noisy.

Reason:
1. early `Phase F` is a validity stage, not a benchmark race.
2. simple algorithms make reward failures easier to identify.

---

## 12. Anti-Hacking Diagnostics

These diagnostics are mandatory from the first RL smoke run.

Controller collapse modes:
1. always-gen collapse,
2. always-ver collapse,
3. premature-fin collapse,
4. budget-exhaustion behavior,
5. repetitive branch-spam behavior.

Reward failure signatures:
1. reward rises while final correctness does not,
2. reward rises while same-family held-out pair accuracy falls,
3. reward rises while canary benchmark metrics fall,
4. reward rises because trajectories become longer or stranger rather than more
   correct,
5. reward rises only for one promoted checkpoint but not another trusted
   candidate.

Mitigations:
1. KL/reference-policy control,
2. explicit compute penalties,
3. early stopping on canary degradation,
4. ensemble or uncertainty-penalized reward when feasible,
5. reward-component logging,
6. same-family held-out validation never used for online updates.

---

## 13. Evaluation System

Every `Phase F` result must report all of the following.

### Outcome metrics

1. final correctness,
2. exact match or task-appropriate success score,
3. success vs compute frontier.

### Reward-model preservation metrics

1. held-out same-family pair accuracy,
2. held-out same-family AUC,
3. value-drop localization,
4. calibration or score-distribution stability.

### Canary metrics

1. same-family held-out offline reranking,
2. `ProcessBench` and/or other benchmark canaries,
3. degradation relative to the frozen pre-RL checkpoint.

### Efficiency metrics

1. total generated tokens,
2. reasoning steps,
3. verification count,
4. branch count,
5. wall-clock cost.

### Policy diagnostics

1. action histogram,
2. average trajectory length,
3. invalid action rate,
4. collapse detector summary.

---

## 14. Promotion Gates

### Gate F-start: permission to begin Phase F

Required:
1. one `Phase E` trust-selected checkpoint,
2. positive same-family held-out evidence,
3. benchmark canaries without catastrophic failure,
4. frozen reward configuration and evaluation protocol.

### Gate F2->F4: permission to begin RL

Required:
1. deterministic `ABR-lite` controller is useful,
2. offline scoring sanity is positive,
3. reward normalization is stable,
4. collapse diagnostics are wired.

If this gate fails:
1. do not start RL,
2. return to deterministic controller design or reward shaping redesign.

### Gate F4->F5: permission to scale RL validation

Required:
1. router-only RL improves at least one constrained frontier,
2. same-family held-out reward-model metrics do not collapse,
3. no obvious reward-hacking signature appears.

### Gate F5->F6: permission for limited generator adaptation

Required:
1. the RL gain repeats across seeds,
2. the gain is not specific to one lucky reward checkpoint,
3. canary degradation remains bounded.

---

## 15. Literature-Flagged Risk Notes

The current `Phase F` design is directionally aligned with the literature, but
the following risk points remain important because they are either only
partially addressed or not yet operationalized enough.

### Risk 1: single-head reward uncertainty is still too optional

Current `Phase F` stance:
1. ensemble or uncertainty-penalized reward is mentioned,
2. but only as an optional enhancement when feasible.

Why this is a literature-level risk:
1. `Scaling Laws for Reward Model Overoptimization` shows that proxy reward
   optimization can diverge from true quality as optimization pressure rises.
2. `Reward Model Ensembles Help Mitigate Overoptimization` and `UP-RLHF` both
   argue that uncertainty-aware conservative optimization is not cosmetic; it is
   one of the main practical defenses against reward hacking.

Risk note:
1. If `Phase F` runs with only one frozen reward head and no uncertainty proxy,
   KL alone may not be enough.
2. This is especially risky once the controller starts discovering narrow
   behaviors that systematically exploit blind spots in one checkpoint.

Recommended note for implementation:
1. If more than one trust-worthy `Phase E` checkpoint is available, treating the
   reward ensemble as optional is weaker than current best practice.
2. The default should become:
   - worst-case reward,
   - uncertainty-penalized reward,
   - or at minimum dual-checkpoint disagreement logging.

### Risk 2: "same-family" is not yet a strong enough support-control definition

Current `Phase F` stance:
1. stay in the same family,
2. freeze LM,
3. keep KL/support control.

Why this is a literature-level risk:
1. `BSPO` argues that overoptimization is driven by out-of-distribution reward
   evaluation, and that explicit behavior-supported regularization is needed.
2. `ProcessBench` shows that PRMs often fail when evaluation moves only modestly
   outside the conditions they were effectively trained for.

Risk note:
1. "same-family" can still hide substantial distribution drift.
2. RL-generated prefixes may leave the actual support of `Math-Shepherd` reward
   training even when the problem domain still looks like math reasoning.

Recommended note for implementation:
1. `Phase F` should define a concrete support audit:
   - on-policy prefix length/profile drift,
   - action-pattern drift,
   - reward-score distribution drift,
   - and distance from heuristic/behavior-cloned traces.
2. Without that audit, "same-family RL" may still become unsupported RL.

### Risk 3: strong non-RL baselines are not yet mandatory enough

Current `Phase F` stance:
1. deterministic `ABR-lite` is required before RL.

Why this is a literature-level risk:
1. `Math-Shepherd` reports large gains from verification/reranking.
2. `Llama 2` relied heavily on rejection sampling / reward-guided sample
   selection in practice.
3. Regularized `Best-of-N` style methods remain strong and often safer than
   full RL when the reward model is imperfect.

Risk note:
1. If router-only RL is compared only against weak fixed schedules, we may
   over-credit RL for gains that a simpler reward-guided selection baseline
   could already achieve.

Recommended note for implementation:
1. Before promoting any `Phase F` RL result, compare against:
   - reward-guided reranking,
   - rejection-sampling style selection,
   - and deterministic `ABR-lite`.
2. If those baselines match the RL gain, RL is not yet justified.

### Risk 4: independent external auditing is still too weakly specified

Current `Phase F` stance:
1. do not use the same head as optimizer and sole final evaluator,
2. keep canary benchmarks.

Why this is a literature-level risk:
1. `ProcessBench` finds that critic-style LLMs can outperform many existing
   PRMs on difficult process-error identification.
2. `The Lessons of Developing PRMs` argues that response-level metrics alone can
   inflate perceived progress and that step-level evaluation remains necessary.
3. `ThinkPRM` points toward stronger verifier families than a small
   discriminative scalar head.

Risk note:
1. Offline canary benchmarks are necessary but may still miss policy-specific
   exploit patterns created during RL.
2. If we never ask an independent critic/judge to audit RL-generated
   trajectories, we risk accepting reward-shaped artifacts as genuine progress.

Recommended note for implementation:
1. Add a fixed external audit path for a subset of RL trajectories:
   - stronger critic LLM,
   - judge model,
   - or manually inspected trace slice.
2. Final `Phase F` promotion should require both:
   - reward-model-preservation metrics,
   - and independent trajectory audit.

### Risk 5: process-vs-outcome misalignment can still leak through the reward

Current `Phase F` stance:
1. keep a terminal correctness anchor,
2. use bounded shaping from the value head.

Why this is a literature-level risk:
1. `The Lessons of Developing PRMs` shows that BoN-style gains can be inflated
   by responses with correct answers but flawed processes.
2. `PRMBench` and `ProcessBench` both emphasize earliest-error detection and
   fine-grained process sensitivity, not just final-answer quality.

Risk note:
1. A controller can learn to produce answers that stay terminally competitive
   while hiding bad intermediate reasoning patterns that the frozen head scores
   incorrectly.
2. This is especially plausible if RL discovers verbose or cosmetically
   structured traces that look "safe" to the reward head.

Recommended note for implementation:
1. During `Phase F`, keep step-level canaries alive:
   - earliest-error localization,
   - bad-step sensitivity,
   - same-family pairwise branch ranking.
2. Do not let final correctness alone certify the controller.

### Risk 6: a frozen reward model is good for interpretability but bad for long-horizon exploitation

Current `Phase F` stance:
1. freeze reward model first,
2. freeze LM first,
3. widen policy class later only if gates pass.

Why this is a literature-level risk:
1. The overoptimization literature consistently shows that static proxy rewards
   become easier to exploit as optimization proceeds.
2. `Math-Shepherd` demonstrates PPO gains, but that does not imply unlimited
   stability under prolonged optimization against one fixed reward head.

Risk note:
1. Freezing the reward model is still the right first move for `Phase F`.
2. But if training horizons grow, the same decision becomes a liability rather
   than a safeguard.

Recommended note for implementation:
1. Keep initial `Phase F` optimization budget intentionally short.
2. Use frequent canary checkpoints and stop early on divergence.
3. Do not silently evolve `Phase F` from a short conservative RL stage into a
   long-run reward-maximization stage without redesigning the audit stack.

### Risk 7: the plan could over-attribute success to explicit process shaping

Current `Phase F` stance:
1. start from the successful `Phase E` process/value head,
2. use it as the main shaping source.

Why this is a literature-level risk:
1. `Do We Need to Verify Step by Step?` argues that process supervision does not
   enjoy an inherent statistical advantage by default.
2. `Free Process Rewards without Process Labels` shows that strong implicit
   process rewards can sometimes emerge from cheaper outcome-style training.

Risk note:
1. If `Phase F` never compares against simpler outcome-anchored or implicit
   reward baselines, we may overstate the necessity of explicit step-shaped
   reward design.

Recommended note for implementation:
1. Keep at least one simpler baseline in the experimental matrix:
   - terminal-only controller reward,
   - verifier-only reward,
   - or outcome-guided reranking.
2. This is not to replace the `Phase E` head, but to prevent false scientific
   attribution.

### Risk 8: training objective may drift away from the eventual inference procedure

Current `Phase F` stance:
1. deterministic controller and router-only RL are the main focus,
2. branch/rerank actions may be added later.

Why this is a literature-level risk:
1. `InfAlign` shows that standard RLHF objectives can become sub-optimal when
   the real deployment procedure uses inference-time selection such as
   `Best-of-N`, controlled decoding, or tree search.
2. `MBR-BoN` and later regularized BoN work further show that reward-guided
   inference is itself an optimization problem with its own hacking risks.

Risk note:
1. If `Phase F` trains one controller objective but deploys another decoding
   stack, we may create a train/deploy mismatch that hides true policy quality.

Recommended note for implementation:
1. Freeze one explicit deployment protocol per experiment:
   - plain controller rollout,
   - rerank after branching,
   - or other fixed inference stack.
2. Evaluate and optimize against the same protocol whenever possible.
3. Do not mix "policy quality" and "inference procedure quality" in one
   uninterpretable number.

Summary judgment:
1. The current `Phase F` philosophy is largely consistent with the literature:
   - freeze first,
   - narrow policy first,
   - dense shaping with terminal anchor,
   - canaries before scaling.
2. The main remaining gap is operational:
   - uncertainty handling,
   - support auditing,
   - stronger non-RL baselines,
   - and independent post-hoc auditing need to be treated as default parts of
     the phase, not merely as good ideas if time permits.

---

## 16. Failure Interpretations

If `Phase F` fails, interpret the failure narrowly.

Case A: `F1` fails.
1. The promoted `Phase E` head is not robust enough for downstream use.
2. Return to `Phase E` trust selection or reward calibration.

Case B: `F2` fails but `F1` passes.
1. The signal exists but the controller design is poor.
2. Improve deterministic control before RL.

Case C: `F4` fails but `F2` passes.
1. The signal is usable offline and heuristically, but RL is exploiting it.
2. Tighten support control, reward clipping, or controller class.
3. Do not escalate to full-policy RL.

Case D: `F5` fails.
1. The RL gain is not trustworthy.
2. Treat it as seed-sensitive and non-promotable.

Case E: canaries degrade while policy reward rises.
1. This is reward hacking until proven otherwise.
2. Stop the run and do not reinterpret it as success.

---

## 17. Immediate Implementation Direction

This section does not mean the code already exists.
It defines the intended `Phase F` code surface.

Suggested new modules/scripts:
1. `src/ours/phase_f/contracts.py`
2. `src/ours/phase_f/rewards.py`
3. `src/ours/phase_f/router.py`
4. `src/ours/phase_f/heuristic_controller.py`
5. `src/ours/phase_f/eval.py`
6. `scripts/phase_f_prepare_controller_data.py`
7. `scripts/phase_f_run_abr_lite.py`
8. `scripts/phase_f_train_router_rl.py`
9. `scripts/run_phase_f_suite.sh`

Expected first implementation order:
1. artifact freeze / config freeze,
2. offline scoring sanity,
3. deterministic controller,
4. controller warm start,
5. router-only RL smoke,
6. seed validation.

## 17.1 Existing RL Framework Survey And Build-vs-Buy Decision

We should not assume that `Phase F` requires building a full RL stack from
scratch.

Current open-source options already cover most of the generic machinery we need.

The important engineering distinction is:
1. `Phase F` is currently **not** standard token-level LLM post-training,
2. it is a custom controller-RL problem on top of:
   - a frozen LM,
   - a frozen value/reward head,
   - a small discrete action space,
   - and strong same-family canary constraints.

Therefore the framework survey must separate two categories.

### A. Environment-centric RL toolchains

These are the best fit for the current `Phase F` shape.

1. `Gymnasium / OpenEnv` style environment interface
   - `OpenEnv` explicitly provides Gymnasium-style `reset()` / `step()` APIs for
     RL and agentic workflows.
   - This matches our intended `gen / ver / backtrack / fin` controller design
     much better than a pure token-level RLHF trainer.
2. `TorchRL`
   - strong fit for custom PyTorch-first RL,
   - environment specs, `reset/step`, transforms, collectors, and training
     primitives are already available,
   - suitable when we want to keep the controller and reward logic highly
     custom while reusing rollout/training infrastructure.
3. `Tianshou`
   - full compatibility with Gymnasium environments,
   - good fit for custom environments and standard policy-gradient style
     algorithms,
   - lighter operational burden than large distributed LLM-RL stacks.
4. `RLlib`
   - viable if later we need larger-scale vectorized sampling and Ray-native
     orchestration,
   - but current external-environment support on the new API stack is still
     under development,
   - so it is not the cleanest first choice for a minimal `Phase F`.
5. `CleanRL`
   - not a modular dependency framework,
   - but a strong reference implementation for PPO / actor-critic details,
   - useful if we want readable algorithm logic without inventing the optimizer
     loop ourselves.

### B. LLM-RLHF / post-training frameworks

These are valuable, but they are a better fit for a later, larger, more
token-level RL stage than for the immediate `Phase F` plan.

1. `TRL`
   - full-stack library integrated with `transformers`,
   - already ships `GRPOTrainer`, `RLOOTrainer`, `PPOTrainer`,
     `RewardTrainer`, and `PRMTrainer`,
   - and now supports `OpenEnv` integration.
   - However:
     - its `OpenEnv` path lives under experimental docs,
     - and the center of gravity is still transformer post-training rather than
       a tiny router policy on top of a frozen LM.
2. `OpenRLHF`
   - strong distributed RLHF/agentic stack on top of `Ray + vLLM + DeepSpeed`,
   - supports `PPO`, `REINFORCE++`, `GRPO`, `RLOO`,
   - supports both custom reward functions and multi-turn `reset/step`
     execution.
   - However:
     - this stack is heavier than we currently need,
     - and is better justified once `Phase F` grows into large-scale or
       multi-node LLM RL.
3. `verl`
   - flexible and production-ready RL training library for LLMs,
   - strong hybrid-controller dataflow model,
   - integrates with `FSDP`, `Megatron-LM`, `vLLM`, `SGLang`, and HF models.
   - However:
     - its main value appears once we are doing large LLM post-training dataflow
       engineering, not minimal controller RL.
4. `SkyRL`
   - modular full-stack RL library for LLMs,
   - includes `skyrl-gym` with Gymnasium environments and `skyrl-agent` for
     long-horizon real-world agent pipelines.
   - However:
     - it is oriented toward longer-horizon agent training and a broader system
       stack than the current frozen-LM router problem.

### Engineering conclusion for this repository

For the current `Phase F`, the best engineering choice is:
1. do **not** build generic RL machinery from scratch,
2. do **not** immediately adopt a heavy token-level LLM-RLHF stack as the core
   abstraction,
3. treat `Phase F` as a custom environment RL problem first.

Recommended immediate stack:
1. environment layer:
   - `Gymnasium`-style env,
   - or `OpenEnv` if we want a cleaner future path to agentic environment
     hosting.
2. trainer layer:
   - `TorchRL` as the first recommendation,
   - `Tianshou` as the second recommendation.
3. algorithm reference:
   - `CleanRL` PPO / actor-critic implementation style as a readable reference,
     even if not imported as a library.
4. model/reward layer:
   - keep our existing frozen LM inference,
   - keep our promoted `best_value_head.pt`,
   - keep custom reward shaping, support-drift checks, and canary evaluation in
     repo-local code.

What we should still implement ourselves:
1. `Phase F` action semantics,
2. reward shaping from frozen value-head outputs,
3. same-family support-drift auditing,
4. canary evaluation and promotion gates,
5. deployment-time inference protocol.

What we should explicitly avoid re-implementing if one of the recommended
frameworks is adopted:
1. rollout collection,
2. return / advantage computation,
3. policy-gradient optimizer loop,
4. vectorized environment plumbing,
5. seeding/logging boilerplate.

Escalation rule:
1. If `Phase F` later expands into token-level LM updates, multi-node training,
   or broader agentic RL with tool-use environments,
2. then re-evaluate `TRL`, `OpenRLHF`, `verl`, or `SkyRL` as the primary
   training stack.
3. Until then, the repository should optimize for minimality and interpretability
   rather than adopting a heavyweight RLHF platform too early.

---

## 18. Final Rule

`Phase F` is not allowed to become a generic RL playground.

Its discipline is the whole point.

If `Phase E` succeeds, the correct next step is:
1. freeze one trusted same-family value head,
2. prove deterministic control can use it,
3. prove router-only RL can improve a constrained frontier,
4. stop immediately if reward-model canaries degrade.

Anything more aggressive than that belongs to a later phase, not to `Phase F`.

---

## 19. References / Local Anchors

Immediate local anchors:
1. `docs/phase_E_plan.md`
2. `docs/TODO_ours.md`
3. `docs/phase_C_plan.md`
4. `scripts/run_phase_e_mathshepherd_trust_suite.sh`
5. `scripts/phase_e_select_candidate.py`

Community references motivating this design:
1. Let's Verify Step by Step:
   - https://arxiv.org/abs/2305.20050
2. Math-Shepherd:
   - https://arxiv.org/abs/2312.08935
3. Improve Mathematical Reasoning in Language Models by Automated Process Supervision (`OmegaPRM`):
   - https://arxiv.org/abs/2406.06592
4. Step-level Value Preference Optimization (`SVPO`):
   - https://arxiv.org/abs/2406.10858
5. ProcessBench:
   - https://arxiv.org/abs/2412.06559
6. PRMBench:
   - https://arxiv.org/abs/2501.03124
7. The Lessons of Developing Process Reward Models in Mathematical Reasoning:
   - https://arxiv.org/abs/2501.07301
8. VersaPRM:
   - https://arxiv.org/abs/2502.06737
9. R-PRM:
   - https://arxiv.org/abs/2503.21295
10. ThinkPRM:
    - https://arxiv.org/abs/2504.16828
11. Training language models to follow instructions with human feedback (`InstructGPT`):
    - https://arxiv.org/abs/2203.02155
12. Scaling Laws for Reward Model Overoptimization:
    - https://arxiv.org/abs/2210.10760
13. Reward Model Ensembles Help Mitigate Overoptimization:
    - https://arxiv.org/abs/2310.02743
14. Dense Reward for Free in Reinforcement Learning from Human Feedback:
    - https://arxiv.org/abs/2402.00782
15. Llama 2: Open Foundation and Fine-Tuned Chat Models:
    - https://arxiv.org/abs/2307.09288
16. Free Process Rewards without Process Labels:
    - https://arxiv.org/abs/2412.01981
17. Behavior-Supported Policy Optimization for RLHF:
    - https://arxiv.org/abs/2501.02446
18. UP-RLHF: Uncertainty Penalized RLHF:
    - https://arxiv.org/abs/2501.15407
19. Regularized Best-of-N Selection for Large Language Models:
    - https://arxiv.org/abs/2503.18013
20. Inference-Aware Alignment:
    - https://arxiv.org/abs/2505.24814
21. TRL documentation:
    - https://huggingface.co/docs/trl/main/index
22. TRL OpenEnv integration:
    - https://huggingface.co/docs/trl/main/openenv
23. OpenEnv:
    - https://github.com/meta-pytorch/OpenEnv
24. TorchRL environment tutorial:
    - https://docs.pytorch.org/rl/stable/tutorials/pendulum.html
25. Tianshou environment documentation:
    - https://tianshou.org/en/stable/02_deep_dives/L3_Environments.html
26. RLlib environment documentation:
    - https://docs.ray.io/en/latest/rllib/package_ref/env.html
27. OpenRLHF:
    - https://github.com/OpenRLHF/OpenRLHF
28. verl:
    - https://github.com/verl-project/verl
29. SkyRL:
    - https://github.com/NovaSky-AI/SkyRL
30. CleanRL:
    - https://github.com/vwxyzjn/cleanrl
