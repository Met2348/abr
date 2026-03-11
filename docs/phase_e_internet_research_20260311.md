# Phase E Internet Research Notes (2026-03-11)

This note records paper and community guidance that is most relevant to the
current repository bottleneck:

1. same-source held-out ranking is already strong,
2. but ProcessBench transfer remains weak,
3. especially on the tradeoff between local first-error detection and
   all-correct terminal preference.

The goal of this note is not to summarize the whole field. It is to extract the
small set of external lessons that are actionable for the current codebase.

## 0. Verified source addendum (re-checked on 2026-03-11)

To keep this note grounded, the following sources were re-checked directly
before the latest redesign iteration:

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

What these verified sources supported most strongly for this repository:

1. benchmark-native process verification is broader than local pair ranking
2. richer intermediate preference structures matter
3. offline evaluation must separate:
   - local first-error detection
   - later-bad ranking
   - all-correct terminal preference
4. before moving to RL, any offline candidate should be judged conservatively
   because naive reward aggregation can still reward-hack

Latest local follow-up informed by this addendum:

1. we implemented a small `dual_head` approximation inside the frozen-feature
   Phase E stack
2. it improved `first_edge` and terminal slices somewhat
3. but it still hurt samefamily fit and `good_vs_laterbad`
4. so the next decomposition attempt should prefer softer routing or staged
   repair, not one-shot hard routing

## 1. What the outside literature says

### 1.1 Process supervision helps, but only when the supervision semantics are aligned

- OpenAI `PRM800K` / `Let's Verify Step by Step` showed that step-level process
  supervision can outperform outcome-only supervision on math reasoning.
  Link: <https://arxiv.org/abs/2305.20050>
- `Math-Shepherd` pushed that idea further with automatically constructed
  step-level supervision, which is directly relevant to this repo because much
  of the current pipeline already depends on converted step labels.
  Link: <https://arxiv.org/abs/2312.08935>

Implication for this repo:

- The current bottleneck is unlikely to be "process supervision does not work".
- It is much more likely to be "the repository is teaching the wrong process
  relations relative to what ProcessBench actually tests".

### 1.2 The benchmark itself is harder than same-source held-out ranking

- `PRMBench` explicitly targets fine-grained process-level verification and is
  designed to expose failures that simple pairwise discrimination can hide.
  Link: <https://arxiv.org/abs/2501.03124>

This matches the repository's own diagnosis:

- same-source pair accuracy can already be very high,
- but benchmark-native evaluation still collapses,
- especially when the model must simultaneously:
  - prefer a fully correct final completion over shorter safe prefixes,
  - and detect the first local mistake reliably.

### 1.3 Several recent papers argue that scalar discriminative PRMs generalize poorly

- `ThinkPRM` argues that pointwise discriminative process reward models have
  limited generalization and proposes generative process supervision instead.
  Link: <https://arxiv.org/abs/2501.07301>
- `GenRM` similarly argues that generative reward modeling is a stronger route
  than plain scalar discrimination for reasoning evaluation.
  Link: <https://arxiv.org/abs/2502.20307>

Implication for this repo:

- A frozen-feature scalar head can still be useful as a conservative verifier
  warm-start.
- But if the local-vs-terminal tradeoff remains unresolved, the correct
  long-term answer may be a richer verifier architecture rather than more
  hyperparameter sweep on one scalar probe.

### 1.4 Decomposed verification is now a common theme

- `Dyve` proposes dynamic verification rather than treating long-chain
  verification as one flat scoring problem.
  Link: <https://arxiv.org/abs/2502.18048>
- `Error Typing for Smarter Rewards` shows that explicitly modeling error types
  can improve reasoning reward behavior.
  Link: <https://arxiv.org/abs/2502.11379>
- `PathFinder-PRM` proposes separate subtask prediction to improve reasoning
  verification.
  Link: <https://arxiv.org/abs/2505.13617>

Implication for this repo:

- The current failure pattern is consistent with "one scalar head is being asked
  to compress at least two subtasks":
  - local error discrimination,
  - terminal completion preference.
- Therefore a lightweight approximation inside the current infra should prefer
  semantically curated training plus multi-regime heads over one undifferentiated
  supervision pool.

### 1.5 Reward-model training practice recommends explicit centering / calibration

- Hugging Face `TRL RewardTrainer` documents a `center_rewards_coefficient`
  regularizer because reward values are only defined up to an affine shift.
  Link: <https://huggingface.co/docs/trl/reward_trainer>

Implication for this repo:

- The repository's transfer problem may partly be score-space drift rather than
  pure ordering failure.
- Adding reward centering is a low-risk change that fits the current head-only
  training setup.

## 2. What community and official tooling say about engineering stability

- PyTorch documents `CUDA_LAUNCH_BLOCKING=1` as a debugging aid for asynchronous
  CUDA failures.
  Link: <https://docs.pytorch.org/docs/stable/cuda_environment_variables.html>
- Hugging Face `Accelerate` big-model guidance emphasizes explicit CPU/GPU
  offload budgets for oversized models.
  Link: <https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference>

Implication for this repo:

- Large frozen-backbone feature encoding is not just a convenience detail; it is
  part of the experimental contract.
- OOM-safe backoff, memory caps, and reproducible cache behavior should be
  treated as first-class infrastructure, not as optional operator workarounds.

## 3. Design principles extracted for this repository

### 3.1 Redesign the data curation layer around semantics, not only raw source bundles

The current repo already knows multiple supervision geometries:

- `first_bad_fanout_prefix_ranking`
- `local_modified_process_error_step`
- `terminal_completion_anchor`
- `good_bad_prefix_grid`

The outside literature strongly suggests these should not be left to emerge from
one mixed pool passively. They should be explicitly curated and quota-controlled.

Decision:

- Introduce a curated semantic-mix artifact layer that samples from existing
  artifacts by `(semantic bucket, source)` with fixed quotas.

### 3.2 Keep local and terminal supervision both present, but bounded

Current repo evidence already shows:

- pure local supervision under-teaches terminal completion,
- heavy terminal repair over-corrects and hurts local ranking.

Decision:

- Use terminal anchors as a bounded auxiliary bucket instead of letting them
  dominate the pool.

### 3.3 Add low-risk logit centering before inventing heavier objectives

Decision:

- Add reward-centering regularization to the current Phase E trainer.

Why:

- It is directly supported by reward-model training practice.
- It is easy to test cleanly.
- It targets score drift without rewriting the whole architecture.

### 3.4 Architecture search should be narrow and hypothesis-driven

Near-term architecture candidates inside current infra:

1. `mlp`
   - baseline high-capacity scalar head
2. `gated_mlp`
   - feature-conditioned two-regime scalar head

Longer-term candidates if the current smoke still fails:

1. explicit dual-head local-vs-terminal scorer
2. error-type-aware verifier
3. generative verifier / critique model in the `ThinkPRM` or `GenRM` direction

## 4. Repository changes and experiment matrix derived from this reading

This turn adopts the following concrete plan:

1. add reward-centering regularization to `phase_e_train_value.py` and the
   underlying loss helper,
2. add a new semantic-curation script so existing artifacts can be remixed by
   supervision bucket instead of only by raw dataset,
3. run one controlled smoke suite on a curated pool:
   - `Math-Shepherd local fanout`
   - `PRMBench local sibling pairs`
   - `PRMBench lightweight terminal anchors`
4. compare:
   - curated `mlp` baseline,
   - curated `mlp` + reward centering,
   - curated `gated_mlp` + reward centering

The actual artifact paths and results of that suite are recorded separately in
the experiment logs and will be summarized into `docs/result_records.md`.

## 5. Outcome of the first implementation round

The first implementation round based on this note has now been executed.

Reference artifacts:

1. suite summary:
   - `assets/artifacts/phase_e_logs/phase_e_curated_rlready_0311_retry2/final_summary.md`
2. strict transfer diagnosis:
   - `assets/artifacts/phase_e_transfer_diag/phase_e_curated_rlready_0311_retry2_diag_00/summary.md`

What happened:

1. semantic curation plus reward centering did **not** make the head RL-ready
2. `reward centering` by itself had almost no transfer effect
3. `gated_mlp` was the only configuration that improved `ProcessBench Math`
   materially, but it still failed the strict local-transfer gate

Updated reading:

1. terminal completion is not the main missing piece in this curated regime
2. the sharper bottleneck is still local benchmark transfer under support drift
3. therefore the next literature-guided repair should move toward:
   - staged objectives,
   - explicit dual local-vs-terminal heads,
   - or a richer verifier family beyond one scalar discriminative probe
