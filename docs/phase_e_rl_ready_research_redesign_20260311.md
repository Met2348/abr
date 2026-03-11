# Phase E RL-Ready Research, Redesign, And Smoke Results (2026-03-11)

## Purpose

This note answers four linked questions:

1. what do recent papers and open-source community implementations suggest about `PRM -> RL-ready` failure modes,
2. how do those lessons map onto this repository's current `ProcessBench` bottleneck,
3. how should the local data-curation and training pipeline be redesigned,
4. which redesigns already show positive signal in local smoke experiments.

## External Research And Community Reading

### 1. `ProcessBench` says the benchmark is not just local first-bad discrimination

Sources:

1. `ProcessBench: Identifying Process Errors in Mathematical Reasoning`
   - https://aclanthology.org/2025.acl-long.50.pdf
2. official repo
   - https://github.com/QwenLM/ProcessBench

Relevant takeaways:

1. `ProcessBench` explicitly evaluates earliest-error localization and also includes a critic-style evaluation prompt.
2. the paper's appendix shows the prompt template for critic-model evaluation, not just scalar reward scoring.
3. the benchmark is therefore structurally broader than:
   - `last-safe > first-bad`
   - on one local edge.

Repository implication:

1. our old pure local `Math-Shepherd` recipe was always going to under-cover the benchmark.
2. "fit same-source held-out pairs" is not enough.

### 2. `Rewarding Progress` argues that useful process rewards should reflect progress, not just local labels

Source:

1. `Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning`
   - https://arxiv.org/abs/2410.08146

Relevant takeaway:

1. the reward for a step should measure *progress*, meaning how much that step changes the future likelihood of a correct solution.
2. this is closer to a step-level advantage signal than a static binary label.

Repository implication:

1. purely local `first_bad` supervision is too narrow.
2. purely terminal-anchor supervision is too outcome-heavy.
3. mixed supervision should preserve:
   - local edge detection,
   - later-bad degradation,
   - terminal completion value.

### 3. `Tree-PLV` supports step-level preference pairs over binary-only supervision

Source:

1. `Advancing Process Verification for Large Language Models via Tree-Based Preference Learning`
   - https://arxiv.org/abs/2407.00390

Relevant takeaway:

1. step-level preference data captures relative merit between intermediate steps better than binary-labeled paths alone.
2. tree-derived comparisons expose richer local-to-global reasoning relations.

Repository implication:

1. this directly supports our move from:
   - strict local edge pairs only
2. toward:
   - fanout pairs
   - broader good-vs-bad grids
   - mixed local + terminal supervision

### 4. `The Lessons of Developing PRMs` warns against weak synthetic supervision and outcome-heavy evaluation

Source:

1. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   - https://arxiv.org/abs/2501.07301

Relevant takeaways:

1. Monte-Carlo-based synthetic labels often generalize worse than LLM-judge or human annotations.
2. evaluation should combine response-level and step-level metrics.
3. many PRMs drift toward outcome-style scoring, especially on final steps.

Repository implication:

1. our diagnosis should not collapse to one AUC number.
2. terminal gains must be monitored together with later-bad discrimination.
3. this directly motivated:
   - `scripts/phase_e_compare_processbench_transfer.py`

### 5. `PURE / Stop Summation` is a warning for the actual RL phase

Source:

1. `Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning`
   - https://arxiv.org/abs/2504.15275

Relevant takeaway:

1. PRM-based RL can reward-hack badly under naive sum-form credit assignment.
2. min-form credit assignment and a small amount of verifiable reward mitigate this.

Repository implication:

1. even after we improve offline ProcessBench behavior, we should not call the stack "RL-ready" under naive reward summation.
2. the realistic target is:
   - conservative rerank/rejection readiness first,
   - then min-form or blended verifiable RL usage.

### 6. Open-source community trend: stronger PRMs increasingly look like critique systems, not just scalar heads

Sources:

1. `R-PRM` repo
   - https://github.com/NJUNLP/R-PRM
2. `Open Instruct` RM docs
   - https://allenai.github.io/open-instruct/algorithms/reward_modeling/

Relevant takeaways:

1. `R-PRM` reports stronger ProcessBench performance by combining:
   - supervised cold start,
   - preference optimization,
   - inference-time scaling,
   - and fine-grained error analysis
2. `Open Instruct` reward-modeling docs highlight stable RM details like:
   - explicit reward margins
   - dropout handling
   - disciplined RM metrics

Repository implication:

1. our current frozen-feature scalar head is a useful debugging regime,
   but probably not the final ceiling.
2. still, before moving to a full critique model, we should first fix the data geometry.

## Local Diagnosis Before Redesign

The repository's strongest pre-redesign signal was:

1. `Math-Shepherd` local recipes:
   - good same-source fit
   - poor terminal ranking on `ProcessBench`
2. `PRMBench terminal-anchor`:
   - strong terminal improvement
   - damage to broader good-vs-bad ranking

So the real problem was not:

1. "need more local pairs"
2. or "need more terminal anchors"

It was:

1. we need both,
2. and we need them mixed in a way that does not let one supervision family erase the other.

## Redesign

### Data-Curation Redesign

What changed:

1. retain branch-specific artifacts instead of collapsing everything into one raw-source recipe
   - `fanout`
   - `grid`
   - `terminal-anchor`
2. add an artifact mixer:
   - `scripts/phase_e_mix_pair_artifacts.py`
3. tag rows with their mixed-artifact source label
   - `artifact_mix_source_label`

New mixed artifacts created this round:

1. `phase_e_pb_repair_0311_mixed_fanout_terminal__ada0800c3d71`
2. `phase_e_pb_repair_0311_mixed_small_fanout_terminal__99976bcc33a8`

Design intent:

1. local/fanout rows preserve first-bad neighborhood discrimination
2. terminal-anchor rows preserve all-correct final-prefix preference
3. later we can add a third branch for broader later-bad coverage if needed

### Training-Pipeline Redesign

What changed:

1. added new pair-weight modes:
   - `group_balance`
   - `confidence_group_balance`
2. group balancing resolves the balancing label in this order:
   - `artifact_mix_source_label`
   - `pair_semantics`
   - `source_tag`
3. existing `source_balance=uniform` is now actually useful in mixed-artifact training,
   because it interleaves groups instead of letting the biggest branch dominate whole stretches of an epoch

Why this matters:

1. mixed supervision without balancing still lets the largest branch dominate optimization.
2. balancing the mixed groups is the smallest intervention that moves us closer to "multi-objective RM training".

### Architecture Redesign

Two architectures were tested this round on the same mixed artifact:

1. `mlp`
   - existing strong baseline head
2. `gated_mlp`
   - newly added lightweight gated two-expert head in `src/ours/phase_b/value_head.py`
   - idea:
     - one expert can fit local ranking structure
     - another can absorb harder/global calibration patterns

## Experiments

### Baselines used for comparison

Same `ProcessBench Math 50` subset:

1. `E46 baseline`
   - eval: `assets/artifacts/phase_e_eval/phase_e_pb_repair_0311_e46_math50_20260310T170837Z`
2. `PRMBench terminal-anchor smoke`
   - eval: `assets/artifacts/phase_e_eval/phase_e_pb_repair_0311_e73c_prm_terminal_math50_20260310T171708Z`

### Experiment A: mixed small artifact + `mlp`

Training run:

1. `assets/artifacts/phase_e_runs/phase_e_pb_repair_0311_e75_mixed_small_mlp_20260311T042122Z`

Config:

1. mixed artifact:
   - `fanout 1536`
   - `terminal-anchor 512`
2. `pair_weight_mode=group_balance`
3. `source_balance=uniform`
4. `head_architecture=mlp`

Held-out pair fit:

1. `pair_acc = 0.9141`
2. `auc = 0.8784`

ProcessBench eval:

1. `assets/artifacts/phase_e_eval/phase_e_pb_repair_0311_e75_mixed_small_mlp_math50_20260311T042808Z`

### Experiment B: mixed small artifact + `gated_mlp`

Training run:

1. `assets/artifacts/phase_e_runs/phase_e_pb_repair_0311_e76_mixed_small_gated_20260311T042757Z`

Config:

1. same mixed artifact as Experiment A
2. same balancing settings
3. `head_architecture=gated_mlp`

Held-out pair fit:

1. `pair_acc = 0.9023`
2. `auc = 0.8804`

ProcessBench eval:

1. `assets/artifacts/phase_e_eval/phase_e_pb_repair_0311_e76_mixed_small_gated_math50_20260311T043534Z`

## Results

Main comparison artifact:

1. `assets/artifacts/phase_e_transfer_compare/processbench_mixed_arch_compare_0311_20260311T043656Z/summary.md`

Key numbers on identical `ProcessBench Math 50` subset:

| case | auc | pair_acc | anygood_vs_firstbad | good_vs_laterbad | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|---:|
| `E46 baseline` | `0.5335` | `0.4778` | `0.4762` | `0.4783` | `0.2000` | `-0.2688` |
| `PRM terminal-anchor` | `0.5207` | `0.4272` | `0.4921` | `0.4111` | `0.5500` | `-0.0382` |
| `mixed MLP` | `0.5400` | `0.5000` | `0.4921` | `0.5020` | `0.3500` | `-0.1751` |
| `mixed gated_mlp` | `0.4263` | `0.4873` | `0.4762` | `0.4901` | `0.0500` | `-0.2213` |

Interpretation:

1. `mixed MLP` is the best overall tradeoff tested this round.
2. it beats `E46 baseline` on:
   - `auc`
   - `pair_acc`
   - `good_vs_laterbad`
3. and it keeps a meaningful portion of the terminal gain:
   - `terminal_top1 0.20 -> 0.35`
4. `terminal-anchor only` still wins the terminal slice,
   but pays too much in broader ranking quality.
5. `gated_mlp` did not help.
6. on this small benchmark slice, the current bottleneck is better addressed by:
   - data geometry
   - mixing
   - balancing
   than by a more complex head.

## Practical Conclusion

The repository is still not fully RL-ready in the strict sense, because:

1. these are smoke-scale `ProcessBench` checks, not full same-family + full benchmark + actual RL stability tests,
2. and the literature warns that naive RL credit assignment can still reward-hack even with a better PRM.

But this round *did* produce a stronger mainline candidate design:

1. mixed supervision,
2. group-balanced training,
3. simple `mlp` head,
4. conservative evaluation by slice.

So the next mainline experiment should be:

1. scale up `mixed MLP` on the full artifact,
2. add one extra later-bad branch if needed,
3. run full same-family trust + full `ProcessBench GSM8K/Math`,
4. and only then revisit RL-readiness under conservative credit assignment.
