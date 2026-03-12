# Phase E RL-Ready Research, Redesign, And Smoke Results (2026-03-11)

## 2026-03-12 再补充：cheap->strong verifier gate 与 paper mirror 已落地

新增基础设施：

1. `docs/relatedPapers/`
   - 现已作为本仓库论文 PDF 镜像目录；
   - 目前文档中提到且能稳定解析的论文，已全部同步到本地。
2. `scripts/phase_e_sweep_weak_strong_gate.py`
   - 用现有 `ProcessBench scored_rows` 直接做 cheap/strong verifier 门控 sweep，
   - 不需要再训练新模型。

新增直接证据：

1. `prm_e46 -> pbr26`
   - 想把 benchmark AUC 拉到接近 `pbr26`，需要：
     - `Math strong usage ≈ 95%`
     - `GSM strong usage ≈ 97%`
2. `ms_e43 -> pbr26`
   - 情况略好，但仍然需要：
     - `Math strong usage ≈ 91%`
     - `GSM strong usage ≈ 87%`

这条结果非常关键，因为它说明：

1. `cheap verifier + 少量升级给 strong verifier`
   - 在当前仓库还不能成立；
2. `cheap verifier + 大量升级`
   - 可以工作，但这更接近“直接用 strong verifier”，而不是一个真正节省成本的两级系统。

因此后文关于 RL-ready redesign 的建议，需要再进一步收紧：

1. 不是只做 `weak/strong routing` 就够；
2. 还需要先提升 weak verifier 本体，或者把问题拆成：
   - local/process verifier
   - terminal/answer verifier
   - abstain/escalate gate
3. `When to Trust the Cheap Check` 这类思路在本仓库当前更像：
   - 未来结构目标，
   - 不是现阶段已可直接依赖的部署方案。

## 2026-03-11 21:40 补充更新

在本文成稿后，又完成了一轮 `Phase E` 基础设施安全收口：

1. 活跃 Phase E wrapper 已统一显式传递 `--recipe-risk-policy`；
2. 直接 trainer 的默认 checkpoint 选择口径已切换到：
   - `pair_acc`
3. 这意味着本文后续关于 RL-readiness 的方法学判断，今后可以更少担心“旧 wrapper 误把危险 recipe 带回主线”的实现噪音。

新增参考：
1. `docs/phase_e_pipeline_redesign_20260311.md`
2. `docs/phase_e_literature_refresh_20260311.md`

## 2026-03-11 23:35 再次补充更新

又完成了一轮更广的 `2025-03 -> 2026-03` 联网 refresh，重点不再是“又多了几篇 PRM paper”，而是 verifier system design 的共识已经更清楚：

1. `ABR` 的核心先验仍然成立；
2. 但 `single scalar verifier` 的角色应进一步收缩为：
   - cheap student
   - router
   - conservative scorer
3. 当前仓库最缺的不是再换一个 head，而是四个系统层：
   - 独立 `answer verifier`
   - `invalid / abstain / escalate` 机制
   - weak-ensemble / teacher-assisted hard-slice 标注
   - `process-outcome alignment + format robustness` gate

这轮新增纳入的更广证据包括：

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

新的主判断：

1. `ProcessBench` 迁移困难，不再应只理解为“local vs terminal supervision 没混好”；
2. 它更像是：
   - answer-equivalence / completeness 未显式拆开，
   - verifier 对输入 contract 可能过敏，
   - cheap verifier 本该 abstain 的地方被迫给分，
   - 以及 teacher/student 角色没有分离。

因此本文后续关于 RL-ready redesign 的建议，应再补上四条：

1. `ABR_AVCal_v1`
   - 先做 answer-equivalence / invalid / false-negative calibration
2. `ABR_WeaverLite_v1`
   - 先做弱 verifier 组合和 disagreement distillation
3. `ABR_POA_v1`
   - 加 process-outcome alignment slice
4. `ABR_FormatInv_v1`
   - 把 prompt / answer-style / delimiter robustness 变成正式 gate

## Purpose

This note answers four linked questions:

1. what do recent papers and open-source community implementations suggest about `PRM -> RL-ready` failure modes,
2. how do those lessons map onto this repository's current `ProcessBench` bottleneck,
3. how should the local data-curation and training pipeline be redesigned,
4. which redesigns already show positive signal in local smoke experiments.

## External Research And Community Reading

### 0. Newer evidence after the old 2025-03 cutoff

The repository now explicitly incorporates four later sources that materially
change the design space:

1. `PRIME (2026)` — process-outcome alignment benchmark
   - https://arxiv.org/abs/2602.11570
2. `Hard2Verify (2025)` — human-annotated frontier step-verification benchmark
   - https://arxiv.org/abs/2510.13744
3. `RISE (2025)` — online self-verification inside RLVR
   - https://arxiv.org/abs/2505.13445
4. `VPRM (2026)` — deterministic verifiable process reward models
   - https://arxiv.org/abs/2601.17223

New interpretation:

1. 旧结论“不要过早宣称 RL-ready”仍然成立；
2. 但到 `2026-03`，更重要的更新已经不是单条新 benchmark，而是 verifier
   system design 本身发生了变化：
   - answer verifier 独立化，
   - factorized supervision，
   - uncertainty gating，
   - generative / reasoning reward model，
   - generator-verifier co-adaptation。
3. 更完整的晚近综述与重设计建议见：
   - `docs/phase_e_literature_refresh_20260311.md`
4. 因此当前 scalar verifier 主线应被理解为：
   - bounded-support baseline，
   - 不是最终 RL-ready object。
5. 后续 RL-ready redesign 需要显式补上：
   - process-outcome alignment，
   - critique / self-verification behavior，
   - deterministic checks where available。

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

## 2026-03-11 Follow-Up: New Data-Geometry Probes

This note originally stopped at the mixed-artifact repair pilots.
The newest follow-up extended that redesign logic with three more targeted
profiles and one architecture sweep.

### 1. `NDS5_MS_STRICT_ONLY_SMOKE`

Question:

1. if we remove fanout/grid and keep only `strict first-bad + terminal`,
   does transfer recover automatically?

Observed result:

1. `PB GSM`: `pair_acc=0.4048`, `auc=0.4210`
2. `PB Math`: `pair_acc=0.3876`, `auc=0.4321`

Reading:

1. no;
2. this rules out the simplistic explanation that the earlier failure was only
   caused by `Math-Shepherd` fanout/grid length shortcuts.

### 2. `NDS6_RLHFLOW_STRICT_ONLY_SMOKE`

Question:

1. if we replace `Math-Shepherd` with stronger `RLHFlow` step labels but keep
   the same strict-local geometry, does that fix transfer?

Observed result:

1. `PB GSM`: `pair_acc=0.6224`, `auc=0.5022`, `first_edge=0.6585`
2. `PB Math`: `pair_acc=0.4357`, `auc=0.4520`, `first_edge=0.6383`

Reading:

1. label quality helps,
2. but the geometry is still too narrow,
3. especially on `ProcessBench Math`.

### 3. `NDS7_MS_DPO_CALIBRATED_SMOKE`

Question:

1. if we make `Math-Step-DPO sibling_branch` the dominant new signal and keep
   only a smaller `Math-Shepherd strict + terminal` support set, does benchmark
   transfer move in the right direction?

Observed result (`mlp`):

1. `PB GSM`: `pair_acc=0.5374`, `auc=0.5575`, `first_edge=0.5854`
2. `PB Math`: `pair_acc=0.5301`, `auc=0.5594`, `first_edge=0.7234`

Reading:

1. yes;
2. `sibling_branch` is the first clearly effective new signal in this pass.

### 4. `NDS7 + gated_mlp`

The same curated `NDS7` data was then re-run with `gated_mlp`.

3-seed benchmark summary:

1. `PB GSM`
   - `pair_acc = 0.6757 ± 0.0297`
   - `auc = 0.6596 ± 0.0158`
   - `first_edge = 0.7642 ± 0.0415`
2. `PB Math`
   - `pair_acc = 0.8099 ± 0.0112`
   - `auc = 0.7460 ± 0.0145`
   - `first_edge = 0.7447 ± 0.0347`

Interpretation:

1. once data geometry is corrected with `sibling_branch`, head capacity does
   matter;
2. but bucket diagnostics show the remaining weakness is still concentrated in:
   - `all-correct terminal completion ordering`
3. in other words:
   - local / global ranking is now strong,
   - terminal completion is the residual gap.

## Updated Redesign Judgment

The redesign recommendation is now more specific than it was in the first draft.

What should remain:

1. `sibling_branch` as the main new supervision family,
2. bounded `strict first-bad` support,
3. small `terminal anchor` support,
4. `gated_mlp` as a serious benchmark-facing probe rather than only a side
   curiosity.

What should not be treated as the next mainline:

1. pure `strict-only` recipes,
2. pure `RLHFlow strict-only` recipes,
3. generic “find an even better source” exploration without explicitly
   targeting terminal completion.

What the next stage should target:

1. terminal-completion-aware objectives that preserve the new DPO-driven local
   gains,
2. not another round of undirected source churn.

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

## Late Web Addendum (2026-03-11)

Two newer external references further support the repository's current redesign
direction.

### 1. `R-PRM` reinforces that data geometry and preference-style optimization matter

Source:

1. `R-PRM` repository
   - https://github.com/NJUNLP/R-PRM

Relevant takeaway:

1. their open training recipe is explicitly staged:
   - supervised cold start,
   - preference optimization,
   - inference-time scaling
2. their public README reports meaningful `ProcessBench` gains from the DPO
   stage, not from a scalar-head-only setup.

Repository implication:

1. this supports the local move toward DPO/fork-point supervision as the main
   signal carrier,
2. and it further weakens the case for staying on "small frozen scalar head +
   weak local labels" as the only mainline.

### 2. `TRL PRMTrainer` reflects current community defaults for stable PRM training

Source:

1. Hugging Face TRL PRM trainer docs
   - https://huggingface.co/docs/trl/prm_trainer

Relevant takeaway:

1. the public trainer defaults emphasize stability primitives:
   - `disable_dropout=True`
   - `gradient_checkpointing=True`
   - `max_length` controls
   - explicit best-checkpoint metric plumbing

Repository implication:

1. this is broadly aligned with the repo's current direction:
   - safer checkpoint selection defaults,
   - more explicit OOM handling,
   - and more explicit training-health diagnostics.
2. it does not prove the local pipeline is optimal,
   but it does support the decision to treat training stability and selection
   hygiene as first-class infrastructure rather than "optional cleanup".

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

## New Controlled Residual Sweep: Data vs Loss vs Architecture (2026-03-11)

After the later `NDS7 + gated_mlp` result, the main residual narrowed further:

1. local discrimination was already strong;
2. global `good > bad` ranking was already strong;
3. the persistent gap was terminal completion ordering.

That suggested a more surgical comparison:

1. repair the data geometry,
2. or repair the loss,
3. or repair the head / routing.

### Shared Baseline

Reference:

1. `NDS7 + gated_mlp`, `seed=42`

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.6565` | `0.6456` | `0.8049` | `0.2174` | `-0.1640` |
| `PB Math`  | `0.7992` | `0.7519` | `0.7447` | `0.1538` | `-0.1461` |

### A. Data-Level Repair: `ms_dpo_terminalboost_v1`

Intent:

1. keep `NDS7`'s successful `sibling_branch + strict` skeleton,
2. increase `terminal_completion_anchor` mass from about `10%` to about `20%`.

Artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_nds8_termboost_0311_pairs__480ab06bf8d6`

Run:

1. `assets/artifacts/phase_e_runs/phase_e_nds8_termboost_gated_pilot_value_retry2_20260311T124818Z`

Results:

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.6752` | `0.6816` | `0.7118` | `0.2798` | `-0.1038` |
| `PB Math`  | `0.7316` | `0.7046` | `0.6848` | `0.1823` | `-0.1670` |

Interpretation:

1. this is a real GSM-side terminal improvement;
2. but it introduces a Math-side tradeoff;
3. terminal mass alone is not a globally safe repair knob.

### B. Loss-Level Repair: `NDS7 + terminal BCE`

Run:

1. `assets/artifacts/phase_e_runs/phase_e_nds7_termbce_gated_pilot_value_retry1_20260311T125922Z`

Results:

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.6053` | `0.6612` | `0.6353` | `0.2073` | `-0.1383` |
| `PB Math`  | `0.7229` | `0.7049` | `0.6555` | `0.1650` | `-0.1842` |

Interpretation:

1. terminal BCE by itself is not enough;
2. it mostly hurts the already-good local/global side;
3. terminal behavior barely improves.

### C. Architecture-Level Repair: `dual_head + terminal BCE`

Run:

1. `assets/artifacts/phase_e_runs/phase_e_nds7_dualhead_termbce_pilot_value_20260311T130037Z`

Results:

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.8150` | `0.7489` | `0.7882` | `0.0829` | `-0.2312` |
| `PB Math`  | `0.7820` | `0.7105` | `0.6806` | `0.0739` | `-0.2535` |

Interpretation:

1. explicit factorization does create a stronger local error detector;
2. but with the current inference mixture, it makes terminal ordering much worse;
3. so the repo now has direct evidence that:
   - local error detection
   - and terminal completion ordering
   are separable subskills.

### Updated Practical Takeaway

The current evidence argues against three naive next steps:

1. "just add more terminal anchors",
2. "just add terminal BCE",
3. "just switch to dual-head".

The better next-step interpretation is:

1. keep `NDS7 + gated_mlp` as the strongest balanced offline baseline;
2. treat terminal completion ordering as a narrow residual problem;
3. and fix that residual with more explicit routing / calibration logic rather
   than heavier mixed supervision.
