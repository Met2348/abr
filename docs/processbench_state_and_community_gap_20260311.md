# ProcessBench State Audit, Community Gap Review, And Next Steps (2026-03-11)

## 0. Executive Summary

这轮结论先说清楚：

1. 队友所说“修复了 `ProcessBench` 的一些表现”只 **部分成立**。
2. 最近的新路线确实修好了一些 slice，尤其是：
   - `terminal_top1`
   - 部分 `first_edge`
3. 但截至当前仓库状态，还 **没有** 一条新路线同时守住：
   - overall `AUC`
   - `good_vs_laterbad`
   - same-family utility
   - RL promotion gate
4. 当前全仓最强的 `ProcessBench` 单跑仍然主要是旧强基线，而不是最新的
   ProcessBench-specific repair smoke：
   - `ms_e43`
   - `prm_e46`

本轮新生成的横向审计 artifact：

1. `ProcessBench Math` state review
   - `assets/artifacts/phase_e_transfer_compare/processbench_state_review_math_0311_20260311T070248Z/summary.md`
2. `ProcessBench GSM8K` state review
   - `assets/artifacts/phase_e_transfer_compare/processbench_state_review_gsm_0311_20260311T070308Z/summary.md`
3. strict RL-promotion state review
   - `assets/artifacts/phase_e_rl_promotion_diag/rl_promotion_state_review_0311_20260311T070332Z/summary.md`

从这三份 fresh audit 看，当前最准确的仓库级诊断是：

1. `ms_e43` 是当前最强的 local / later-bad 候选，但 terminal 几乎完全不会。
2. `prm_e46` 是当前更平衡的 benchmark-facing 候选，但 `later-bad + terminal`
   仍未同时过线。
3. `e87 / C3 / C4 / PBR2` 这些修补路线都验证了“局部修补是有效信号”，但还
   没有产生新的主线候选。

更重要的是，社区现在的强方法已经明显不再是“冻结 backbone + frozen feature
scalar head + pair ranking”这一种范式。当前仓库最像一个**高质量诊断平台**，
还不是一个已接近社区上限的 verifier 系统。

---

## 1. This Audit Checked What

### 1.1 New local code / docs / artifacts checked

重点核对了这些入口：

1. `scripts/run_phase_e_processbench_research_suite.sh`
2. `scripts/phase_e_curate_processbench_transfer_pairs.py`
3. `scripts/phase_e_curate_semantic_pairs.py`
4. `src/ours/phase_b/value_head.py`
5. `src/ours/phase_e/training.py`
6. `src/ours/phase_e/processbench_alignment.py`

重点核对了这些新旧文档：

1. `docs/research_survey_processverifier_20260311.md`
2. `docs/bcr_feasibility_review_20260311.md`
3. `docs/dataset_survey_stepwise_pairs_20260311.md`
4. `docs/new_dataset_experiment_plan_20260311.md`
5. `docs/phase_e_rl_ready_research_redesign_20260311.md`
6. `docs/phase_E_plan.md`

重点核对了这些实验线：

1. 旧强基线：
   - `ms_e43`
   - `prm_e46`
2. ProcessBench repair 线：
   - `e79`
   - `e87`
   - `pbr2_align_gated`
3. curated / architectural repair 线：
   - `c3_curated_gated`
   - `c4_dual`

### 1.2 Local PDFs and downloaded datasets checked

本轮直接阅读或复核的关键论文 PDF：

1. `PRM800K / Let's Verify Step by Step`
2. `ProcessBench`
3. `Tree-PLV`
4. `OmegaPRM`
5. `The Lessons of Developing PRMs`
6. `VersaPRM`
7. `R-PRM`
8. `GenPRM`
9. `ThinkPRM`

已在本地下载、但当前主线尚未真正用起来的数据源包括：

1. `assets/external_datasets/openai_prm800k`
2. `assets/external_datasets/trl_prm800k_formatted`
3. `assets/external_datasets/openreasoner_math_aps`
4. `assets/external_datasets/prime_rl_eurus_prm_stage2`
5. `assets/external_datasets/xinlai_math_step_dpo`
6. `assets/external_datasets/genprm_math_data`
7. `assets/external_datasets/openbmb_ultrainteract_pair`

这意味着当前仓库的瓶颈已经不是“没有数据”，而是：

1. 适配器没补全，
2. 高质量源没有主线化，
3. 训练范式还停留在 frozen-feature scalar head 阶段。

---

## 2. What The Community Is Actually Doing

## 2.1 `PRM800K`: human step labels, full verifier training, explicit step scoring

`PRM800K` 的核心不是 pair ranking，而是：

1. 人工步骤级标签，
2. 在 step 末 token 上训练 verifier，
3. 再把 step score 聚合成 response-level score。

这对当前仓库的意义很直接：

1. 社区最经典的强监督源不是“自动 first-bad 边界 pair”，
2. 而是**逐步 correctness 标签**。

同时，`ProcessBench` 原文也明确给出了一个重要实践细节：

1. 他们训练 `Qwen2.5-Math-7B-PRM800K` 时，把 `PRM800K` 里的 `1` 和 `0`
   都当正类，
2. 只有 `-1` 当负类。

这对本仓库非常关键，因为这正是当前 `PRM800K adapter near-random` 最需要核对
的地方。

## 2.2 `ProcessBench`: benchmark target is broader than one local edge

`ProcessBench` 要求模型：

1. 找到**最早错误步**，
2. 或判断整条解法**全都正确**。

而且原论文明确指出：

1. 现有 PRM 对更难数学问题泛化差，
2. critic-style prompted models 往往能打败不少现有 PRM，
3. 一个 straightforward `PRM800K` fine-tuned baseline 已经比许多公开 PRM 强。

这意味着：

1. benchmark 本身不是“只测 first_bad_edge”，
2. 当前仓库如果主要靠 local pair 学习，然后拿同源高 ACC 当主进展指标，
   就天然会乐观。

## 2.3 `Tree-PLV` / `OmegaPRM`: true branch preference matters

`Tree-PLV` 的关键论点是：

1. verifier 在实际用途上更像一个 ranker，
2. 所以 binary classification 并不总是和目标一致，
3. step-level preference learning 往往比 binary supervision 更对齐。

`OmegaPRM` 则更进一步：

1. 直接构造 reasoning tree，
2. 通过 tree / MCTS / binary search 找第一错误，
3. 用 sibling-branch 关系产生更干净的 step supervision。

这和当前仓库的差别非常大：

1. 当前主线大多数训练对仍来自 single-trajectory conversion，
2. `fanout` / `grid` 虽然比 strict local 更宽，
3. 但它们仍然不是 paper 里那种“同 prefix，不同 continuation”的真 branch pair。

## 2.4 `Lessons`, `GenPRM`: MC labels are noisy; consensus filtering matters

`The Lessons of Developing PRMs` 给出的最重要提醒有三条：

1. MC estimation 做 step correctness 往往噪声很高，
2. PRM 容易从 process verifier 滑向 outcome scorer，
3. 评估必须同时看 response-level 和 step-level，而不是只看一个指标。

`GenPRM` 则把这件事工程化了：

1. MC 只是第一步，
2. 后面还做 Relative Progress Estimation，
3. 再做 rationale generation + code verification + LLM judge consensus filtering，
4. 过滤掉约一半数据以后再训练。

这和当前仓库的差距在于：

1. 我们已经有 `pair_semantics` / slice diagnostics，
2. 但训练数据质量控制还远没有走到 consensus-filtered 那一步，
3. 仍主要停留在“pair 几何修补”。

## 2.5 `VersaPRM`: data diversity + self-filtering + adaptation

`VersaPRM` 给出的经验不是“数学 PRM 直接跨域就好”，而是：

1. 显式造多域 CoT 数据，
2. 自己再做 self-filtering，
3. 并且训练中使用 LoRA / full fine-tuning。

因此，当前仓库如果未来要重新追求原始 BCR 论文里那种跨域 claim，
更像应该走：

1. 新数据源扩展，
2. 数据过滤，
3. backbone adaptation，

而不是继续把 StrategyQA 当主要证明场。

## 2.6 `R-PRM`, `GenPRM`, `ThinkPRM`: strongest recent systems are increasingly critique-style

最近最强的公开路线越来越像：

1. 生成式 verifier，
2. critique / reasoning-driven verification，
3. preference optimization / SFT / inference-time scaling 的组合。

这三篇论文共同传达的信号是：

1. 社区上限已经明显超出“冻结 backbone 后接一个标量头”的 regime，
2. 很强的 ProcessBench 表现往往依赖：
   - backbone finetuning / LoRA
   - 生成式 verification
   - inference-time scaling
   - 高质量过滤过的数据

所以当前仓库路线的正确定位应当是：

1. 不是“已经和社区 SOTA 同层竞争”，
2. 而是“在更便宜、更可诊断的判别式 regime 中摸清失败结构”。

---

## 3. What Our Repo Is Actually Doing

当前主线实现，本质上是：

1. 冻结 `Qwen2.5-7B-Instruct` backbone，
2. 用 `last_token` pooling 抽一个特征，
3. 只训练很小的 scalar head：
   - `linear`
   - `mlp`
   - `gated_mlp`
   - `dual_head`
4. 目标是 pairwise ranking / BCE / terminal BCE / centering 等组合。

对应代码入口：

1. `src/ours/phase_b/value_head.py`
2. `src/ours/phase_e/training.py`
3. `scripts/phase_e_train_value.py`

数据侧主线做的是：

1. 把外部 step-label 或 preference 数据转成 `ExternalPairRecord`
2. 再围绕这些 pair 做：
   - strict local
   - fanout
   - grid
   - terminal anchor
   - PRMBench local auxiliary
3. 然后靠：
   - semantic weighting
   - group balancing
   - source balancing
   - staged wrappers
   来修 benchmark transfer。

这套设计的优点：

1. 便宜，
2. 容易切片诊断，
3. 能很快暴露 supervision geometry 问题。

但它的硬边界也很清楚：

1. backbone 完全不适配 verifier 任务，
2. pooled feature 只看最后 token，
3. 训练监督仍大多不是 tree/sibling structure，
4. 只能在“固定表示上再学一个 scorer”。

---

## 4. Evidence-Backed Gaps In The Current Repo

## 4.1 Teammate claim is only partially correct

最新 repair 线不是没有效果，而是效果类型被说大了。

fresh state review 给出的事实是：

1. `PBR2` 的 `pair_type_l1` 比老 `ms_e43` 更低，
   说明训练对的几何确实更像 benchmark 了；
2. `C3/C4/PBR2` 的 `terminal_top1` 也明显高于旧 local 强基线；
3. 但这些 repair 线并没有产生新的 overall best benchmark run。

最有代表性的对比：

1. `ms_e43` on `ProcessBench Math`
   - `auc=0.6341`
   - `good_vs_laterbad=0.7515`
   - `terminal_top1=0.0099`
2. `c4_dual` on `ProcessBench Math`
   - `auc=0.4789`
   - `good_vs_laterbad=0.3672`
   - `terminal_top1=0.9038`
3. `pbr2_align_gated` on `ProcessBench Math`
   - `auc=0.5055`
   - `good_vs_laterbad=0.5000`
   - `terminal_top1=0.6154`

所以更准确的叙述应该是：

1. 新 repair 线修了 terminal / some first-edge slices，
2. 但还没有超越老强基线的 overall benchmark utility。

## 4.2 The dominant unresolved conflict is now explicit

当前仓库已经把冲突暴露得很清楚：

1. `ms_e43`
   - local / later-bad 很强
   - terminal 极弱
2. `c3/c4`
   - terminal 很强
   - later-bad 很弱
3. `prm_e46`
   - benchmark overall 更平衡
   - 但 terminal 和 math later-bad 仍不过线

最新 RL promotion 审计结论非常直接：

1. `ms_e43`
   - `assessment=near_rl_ready_but_terminal_gap`
2. `prm_e46`
   - `assessment=terminal_and_local_tradeoff_unresolved`
3. `c3/c4/pbr2`
   - `assessment=not_rl_ready_laterbad_generalization_weak`

这说明：

1. 当前 repo 的主问题已经不是“会不会学”，
2. 而是“两个 verifier 子能力还没有被同一训练栈同时学稳”。

## 4.3 Frozen-feature regime is now the main structural bottleneck

根据论文和当前结果一起看，当前 frozen-feature 主线的问题不是 capacity 太小，而是：

1. backbone representation 没有按 verifier 任务适配，
2. scalar head 只能在固定表示上做折中，
3. 所以一旦想把 terminal 补起来，就容易伤 local / later-bad。

这不是纯猜测，当前证据链是闭合的：

1. 社区强方法普遍用：
   - LoRA / finetuning
   - 或直接 generative verifier
2. 本仓库当前所有 repair 仍停留在 frozen-feature 头部改造；
3. 结果就是：
   - slice 能被推来推去，
   - 但很难同时过 gate。

## 4.4 High-quality downloaded sources are still not mainlined

当前最大的研究组织问题之一是：

1. 本地已经下载了：
   - `PRM800K`
   - `MATH-APS`
   - `EurusPRM-Stage2`
   - `Math-Step-DPO`
   - `GenPRM-MATH`
2. 但真正主线使用的，仍主要是：
   - `Math-Shepherd`
   - `PRMBench_Preview`
   - `R-PRM` converted pairs

这意味着当前 ProcessBench 路线在工程上过度依赖：

1. 本地修配方，
2. 小规模 curate，
3. 头部改造，

而没有把社区更强的数据源真正打进主线比较矩阵。

---

## 5. Anti-Patterns To Stop Repeating

## 5.1 Do not use same-family high ACC as the main success signal

`heldout_pair_acc` 和 `samefamily_top1` 仍然重要，但只能当：

1. learnability check，
2. utility safety canary。

不能再把它们当成：

1. `ProcessBench` readiness proxy，
2. 更不能当 RL-ready proxy。

## 5.2 Do not read one repaired slice as “benchmark fixed”

当前最容易产生误判的是：

1. `terminal_top1` 大涨，
2. 或 `first_edge` 微涨，
3. 就把这解释成 benchmark transfer 已经修好。

fresh compare 正好证明这很危险：

1. `c4_dual` 的 terminal 最好，
2. 但 `good_vs_laterbad` 是这批候选里最差的一档。

## 5.3 Do not treat synthetic grid/fanout as true branch-preference data

`fanout` / `grid` 是仓库里很有价值的过渡性诊断手段，
但它们不是：

1. `Tree-PLV` 的 sibling-node preference，
2. 也不是 `OmegaPRM` 的 MCTS tree supervision。

所以这类 pair 只能被表述成：

1. benchmark-aware approximation，
2. 不能被表述成 community-style tree verifier data。

## 5.4 Do not keep inventing new heads before opening the stronger data sources

当前已经有：

1. `linear`
2. `mlp`
3. `gated_mlp`
4. `dual_head`

而且结论已经足够清楚：

1. 头结构会改变 slice tradeoff，
2. 但不会凭空造出缺失的数据语义和 backbone 表示。

所以再继续只做 head 变体 smoke，边际价值已经明显下降。

## 5.5 Do not overstate the literature as if it gives a theorem

更严谨的说法应当是：

1. 社区当前强结果**大多**依赖 backbone adaptation 或 generative verification，
2. 这让它们比当前 frozen-feature 路线更可信地接近上限。

但不应把它写成：

1. “所有 >75% 结果都必然如此，因此 LoRA 一定足够”。

更稳妥的研究表达应该是：

1. `LoRA` 是最值得做的下一个最小结构跃迁，
2. 不是已经被当前仓库内部实验证明的定理。

---

## 6. Corrected Research Reading Of Our Idea

当前项目的可行主张，应该修正成下面这个版本：

1. 仓库当前已经证明：
   - frozen-feature value head 可以可靠学习某些 process slices；
2. 但它尚未证明：
   - 单一 frozen-feature verifier 足以解决 benchmark-native process verification。

更准确的 research claim 应该是：

1. single-trajectory converted pairs 足以学出 local error boundary detection；
2. all-correct completion preference 是另一项不同子任务；
3. later-bad generalization 又是第三项更难子任务；
4. 当前主线的真正科学问题，不是“value head 能不能学”，而是：
   - 这三项能力怎样被同一 verifier 一起学到，
   - 且不过度牺牲彼此。

因此，BCR/ABR 相关 idea 的近期现实版本应当表述为：

1. 先把 verifier 当一个多切片 process scorer，
2. 不急着把它 narrate 成 Bellman-faithful value estimator，
3. 更不急着 narrate 成已可直接主导 RL 的 reward model。

---

## 7. Concrete Next Experiments

## 7.1 Priority A: stop architecture churn; open the stronger sources

下一阶段第一优先级不应再是新 head，而应是新 source。

### A1. Fix `PRM800K` adapter first

目标：

1. 把 `PRM800K` 从“已有但 near-random”变成正式主线候选。

最关键检查点：

1. `1/0 -> positive`, `-1 -> negative`
2. neutral / ambiguous step 的处理
3. step split 与 current pair construction 的对应

成功判据：

1. source-only smoke 在 equal budget 下至少不差于 `Math-Shepherd`
2. 如果 `ProcessBench` 更好，则 `PRM800K` 应进入主线 mix

### A2. Add `MATH-APS` adapter

目标：

1. 用更接近 true branch / tree supervision 的 source 专打 `later-bad`。

原因：

1. 当前最薄弱、最稳定重复出现的 failure tag 就是
   `benchmark_good_laterbad_weak`。

### A3. Add `EurusPRM-Stage2` and `Math-Step-DPO-10K`

目标：

1. 让主线第一次真正比较：
   - MC-ish source
   - human source
   - judge-style source
   - clean DPO-style first-error source

这一步做完以后，当前 repo 才算真正开始和社区数据策略接轨。

## 7.2 Priority B: run source-only and equal-budget comparisons before mixing

建议固定一个小矩阵：

1. equal pair budget
2. same `mlp` head
3. same training epochs
4. same `ProcessBench` slice audit

先比较：

1. `Math-Shepherd`
2. `PRM800K`
3. `MATH-APS`
4. `EurusPRM-Stage2`
5. `Math-Step-DPO-10K`

目的不是立刻拼最强，而是回答：

1. 谁最擅长 local
2. 谁最擅长 later-bad
3. 谁最擅长 terminal

## 7.3 Priority C: only after that, do one minimal LoRA jump

如果 source-only 比较确认了最强 source 组合，下一步最值得做的是：

1. 保留最稳的 `mlp` 或 `gated_mlp`，
2. 对最后 4 层做 `LoRA`,
3. 禁用 feature cache，
4. 跑一轮与 frozen 版本完全同 artifact 的正面对照。

目标不是追 SOTA，而是验证：

1. 当前 repo 的主 ceiling 到底是不是 frozen representation。

## 7.4 Priority D: generative verifier should be teacher path first, not immediate replacement

`R-PRM / GenPRM / ThinkPRM` 的结论已经很明确：

1. 长期上限更像 critique / generative verifier。

但对当前仓库最务实的落地方式是：

1. 先让本地 stronger model 生成 verification rationale / step judge label，
2. 当 teacher 或 data filter，
3. 而不是马上把整个 Phase E 主线改成生成式 verifier。

这样可以先回答一个更便宜的问题：

1. 数据质量一旦上去，现有 discriminative stack 能提升多少？

---

## 8. Final Bottom Line

当前仓库最该停止的一件事是：

1. 继续把“pair geometry 小修小补 + 新 head smoke”当主进展。

当前最该开始的一件事是：

1. 把已经下载到本地的高价值数据源真正拉进 equal-budget matrix，
2. 再用最小 LoRA 对照确认 frozen-feature ceiling。

一句话总结这轮审计：

1. 社区已经在用更强数据、更强适配、更强 verifier 范式解决问题；
2. 我们的代码现在最擅长的是把失败结构看清楚；
3. 下一步不该继续“猜 head”，而该去验证：
   - `better source`
   - `better supervision geometry`
   - `minimal backbone adaptation`
   这三件事到底哪件才是真正的主瓶颈。
