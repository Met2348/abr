# Process Verifier Research Survey: Solutions to Current Bottlenecks

**Date:** 2026-03-11
**Purpose:** 系统整理社区与论文经验，对标当前 Phase E 瓶颈，指导下一阶段数据策划与架构设计。

---

## 0. Executive Summary（执行摘要）

当前项目最核心的三个瓶颈：

| 瓶颈 | 当前表现 | 社区是否有解法 |
|---|---|---|
| **Local→Terminal gap** | 同源 held-out 95%+，但 ProcessBench AUC ~0.62 | ✅ 有：terminal anchor + MC 标注 + 混合课程 |
| **Single-trajectory pair 语义过窄** | `first_bad_edge` 只给一种信号，缺全局视角 | ✅ 有：OmegaPRM MCTS / MC 多步估计 / Full-Step-DPO |
| **Frozen head 表示力天花板** | MLP 在 PRMBench 93%，但跨域泛化弱 | ✅ 有：BiPRM 双向、LoRA partial unfreeze、DPO 隐式 PRM |

**关键结论（先读这个）：**
1. 同源高 ACC 是必要条件，不是充分条件。社区共识和我们一致。
2. 真正的 ProcessBench 突破来自以下任一路径：
   - **生成式验证器**（R-PRM/GenPRM/ThinkPRM），用 chain-of-thought reasoning 验证每步 → 直接 F1 70+
   - **数据策划修复**（terminal anchors + MC step 估计 + 混合课程）→ 部分弥合 local/terminal gap
   - **隐式 PRM（PRIME 框架）**：用 DPO log-ratio 作为 step reward，不需要独立的 PRM 训练
3. 在不扩展到生成式路径的前提下，下一轮最有价值的实验是：**数据修复（terminal anchor 混合比例调参）+ 架构修复（BiPRM 风格双向 / mean pooling / 双头）**。

---

## 1. 当前瓶颈详细分析

### 1.1 ProcessBench 结构问题

ProcessBench 有 **~48% (GSM8K) / ~41% (Math) 的 all-correct 样本**。
评测时，all-correct 样本要求模型打出"所有前缀都应高分"的分布。
当前训练只有 `first_bad_edge` 监督，对应的语义是：
- `chosen` = 最后一个干净前缀（长度 $t-1$）
- `rejected` = 包含第一个错误步的前缀（长度 $t$）

这个监督可以教会模型"在错误发生边界处降分"，但**无法教会**：
- 完整正确解法应比截断的正确前缀得更高分
- 越接近终点的正确前缀应该得分递增
- 完整解法 vs 中途停下但都正确的前缀之间的偏好

因此 ProcessBench 的 `all-correct terminal completion` 指标天然低不是因为模型差，而是**监督契约本身缺失这个维度**。

### 1.2 单轨迹 pair 的覆盖局限

`Math-Shepherd` 每条样本只有一条轨迹（good/bad 步序列）。
`first_bad_edge` 从中取出的一对 pair 的语义：
- 对比的是"前 $t-1$ 步"和"前 $t$ 步"
- 两者路径相同，只有最后一步不同
- **不是** "同一个 prefix 有两种不同的后续选择"

社区称这为 **single-trajectory local edge**，区别于：
- **sibling-branch pair**（OmegaPRM/MCTS 风格）：同一前缀，不同 rollout，一个到达正确答案，一个不能
- **full-MC pair**（Qwen2.5-Math-PRM 风格）：对每一步独立估计，用 8 次 rollout 的成功率打软标签

### 1.3 Frozen backbone 表示力上限

当前设计：冻结的 Qwen2.5-7B-Instruct → `last_token` 池化 → 线性/MLP 头。
问题：
1. `last_token` 只看序列末尾，对中间步骤的 local error 不敏感
2. 冻结 backbone 的 feature 是按语言建模任务优化的，不一定能最优表达"step quality"
3. 单向 causal attention 看不到未来步骤，判断当前步好坏缺乏全局上下文

---

## 2. 社区/论文关键经验（按相关度排序）

### 2.1 R-PRM（最相关，直接解决 ProcessBench gap）

**论文：** [R-PRM: Reasoning-Driven Process Reward Modeling (arXiv:2503.21295)](https://arxiv.org/abs/2503.21295)
**ProcessBench 成绩：** F1=70.4（DPO 阶段后）vs 我们的 ~0.62 AUC
**关键做法：**
1. **SFT 冷启动**：用 LLaMA3.3-70B-Instruct 对 PRM800K 样本生成带推理链的步骤分析（289K SFT 样本）
2. **DPO 自我提升**：用 SFT 模型本身生成 chosen/rejected 推理链对，不需要额外标注（269K DPO 样本）
3. **推理时 scaling**：采样 N=4 条推理链，多数投票聚合 → F1 从 62.8 涨到 67.6

**对我们的启示：**
- 真正的 ProcessBench 突破需要模型"思考"每一步是否正确，不只是打分数
- 如果不走生成式路线，DPO 自我提升是更轻量的路径
- DPO 在"没有额外标注数据"前提下能明显提升 F1

**可借鉴的设计（短期可行）：**
- 对 Math-Shepherd 训完的 checkpoint，用 SFT+DPO 继续训练，不用生成式验证器也能涨点
- 用 Qwen2.5-7B-Instruct（已有本地模型）对训练样本生成 chosen/rejected 对（替代 LLaMA3.3-70B）

---

### 2.2 OmegaPRM（数据构造最相关）

**论文：** [OmegaPRM: Automated Process Supervision via Divide-and-Conquer MCTS (arXiv:2406.06592)](https://arxiv.org/abs/2406.06592)
**关键做法：**
1. **二分搜索定位第一错误**：对一条解法，用 MCTS 二分查找第一个使成功率明显下降的步骤
2. **平衡正负样本**：在选择阶段优先选"预期正确但最终答错"的 rollout（因为单纯错误 rollout 过多）
3. 构造的 pair 是真正的 **sibling-branch pair**：同一 prefix，两条不同 completion，一对一错

**对我们的启示：**
- 我们当前的 pair 缺少 "同 prefix，不同 branch" 这一维度
- 正负样本平衡是关键：只有 bad rollout 会导致模型只学到"截断/错误"信号，缺少"正确 completion 比截断更好"信号
- MCTS 构造的树形结构天然包含 terminal completion（从根到叶子的完整路径 = 完整正确解法）

**可借鉴的设计（中期可行）：**
- 对现有 Math-Shepherd 数据做 MCTS 二分搜索，生成 sibling-branch pairs
- 如果没有 MCTS 资源，用 MC 采样（每步 4-8 次 rollout）估计步骤质量，替代 `first_bad_edge`

---

### 2.3 ThinkPRM / GenPRM（生成式路线的天花板）

**ThinkPRM:** [arXiv:2504.16828](https://arxiv.org/abs/2504.16828)
**GenPRM:** [arXiv:2504.00891](https://arxiv.org/abs/2504.00891)
**共同点：**
- 不打标量分数，而是**生成 verification chain-of-thought**，最后输出 correct/incorrect
- 可以利用 test-time scaling（采样 K 条验证链，多数投票）
- 天然处理 terminal completion：生成式推理本身会判断"这是完整正确解"还是"这一步有错"

**ThinkPRM 具体数据：**
- 只需 8K process labels + QwQ-32B-Preview 生成合成验证链
- 对 ProcessBench 比 discriminative PRM 高 7.2%
- 对 GPQA-Physics（OOD）比 discriminative PRM 高 8%

**对我们的启示：**
- 生成式路线跨域泛化更强，因为它依赖的是 reasoning ability，而不是 feature distribution
- 短期成本高（需要强大生成模型，推理代价大），但长期是 SOTA 路线
- 如果项目目标是"方法可用于 RL"，生成式验证器可以先作为离线 teacher，为 discriminative head 提供更高质量的训练信号

---

### 2.4 PathFinder-PRM（架构最相关）

**论文：** [PathFinder-PRM: Error-Aware Hierarchical Supervision (GitHub: declare-lab/PathFinder-PRM)](https://github.com/declare-lab/PathFinder-PRM)
**PRMBench 成绩：** PRMScore=67.7（SOTA，超过上一名 65.5，且只用 1/3 数据）
**关键架构：**
1. **两阶段前向**：
   - 第一次前向：预测 Math error + Consistency error（两个独立分类任务）
   - 第二次前向：基于 Math+Consistency 的结果，预测 step Optimality
2. **误差去耦**：Math 和 Consistency 独立预测，避免 autoregressive 生成中一个标签影响另一个

**对我们的启示：**
- 简单的 scalar head 混合了多种错误类型，会产生 blurry gradients
- 显式分离 error type 能提升 data efficiency（同样数据出更强 PRM）
- 可以设计一个**双头版本**：一个头预测 local step quality（当前步是否有错），另一个头预测 terminal completion preference（这条链最终是否正确）

---

### 2.5 PRIME（隐式 PRM，最相关于我们的训练目标设计）

**论文：** [Process Reinforcement through Implicit Rewards (arXiv:2502.01456)](https://arxiv.org/abs/2502.01456)
**关键做法：**
- 不单独训练 PRM，而是利用 DPO 学到的 Q 函数隐式提供 step-level reward
- Reward 公式：$r(x, s_t) = \beta \log \frac{\pi_\theta(s_t|h_{t-1})}{\pi_{ref}(s_t|h_{t-1})}$
- 每条 rollout 按 outcome 分 positive/negative，DPO loss 反向传播给出隐式 process reward
- 可在线更新（online RL），避免 distribution shift 和 reward hacking

**对我们的启示：**
- 我们当前的 ranking loss 在数学上接近 DPO 的一个变体
- PRIME 证明：用 outcome labels 驱动 DPO 就能得到有效的 step-level reward
- 对我们的改进方向：可以把当前的 pair ranking loss 升级为 PRIME 风格的 log-ratio reward，同时允许 backbone 部分解冻做 LoRA 适配

---

### 2.6 VersaPRM（多源混训最相关）

**论文：** [VersaPRM: Multi-Domain PRM via Synthetic Reasoning Data (arXiv:2502.06737)](https://arxiv.org/abs/2502.06737)
**关键做法：**
1. 合成多域 CoT 数据（每题 16 条 CoT）+ 自动标注步骤标签
2. **Self-filtering**：用训练好的 PRM 本身评分，删掉预测分数与自动标签偏差 > 0.4 的 CoT（过滤了约 37%）
3. 只训一个 epoch，batch size 32

**对我们的启示：**
- 混源训练中，数据自过滤（self-filtering）是防止噪声源主导训练的关键手段
- 当前 PRM800K 的"弱源"问题可能部分来自标注噪声，self-filtering 后质量可能改善
- 单 epoch + 小 batch 的训练策略在多域场景下可能比多 epoch 更稳定

---

### 2.7 Qwen2.5-Math-PRM（工程实现参考）

**Blog：** [Towards Effective Process Supervision in Mathematical Reasoning](https://qwenlm.github.io/blog/qwen2.5-math-prm/)
**关键设计：**
- 每步标注用 **MC estimation（8次 rollout）+ LLM-judge + 共识过滤**（三者一致才保留）
- 产品最终分数 = **所有步骤分数的乘积**（product aggregation，而不是 min 或 mean）
- 约 500K 问题 × 6-8 条回答 = 约 3-4M 步骤级标注
- 既有硬标签也有软标签变体

**对我们的启示：**
- 三重共识过滤能大幅提升标注质量，降低噪声来源的影响
- Product aggregation 对完整解法自然给出更高分（所有步正确时乘积 ≈ 1）——这天然处理 terminal completion 偏好
- 我们当前只用 `first_bad_edge` 是一种极端的 hard label，改为 MC 软标签可能更稳定

---

## 3. 问题根因分解

根据上述调研，当前失败的根因可以拆成三层：

```
Layer 1: Data-side root cause
  └── 训练 pair 语义缺失 terminal completion 信号
      └── only first_bad_edge, no positive terminal anchor
      └── no MC-estimated step scores (soft labels)
      └── no sibling-branch pairs

Layer 2: Architecture-side root cause
  └── 单向 causal last_token 特征无法捕获全局信号
      └── 不能"看到"当前步之后的推理是否成功
      └── pooling 策略（last_token）对中间步信号不敏感
      └── 单头设计混淆了 local quality 和 global terminal preference

Layer 3: Training-side root cause
  └── Ranking loss 仅优化 pair 的相对顺序
      └── 没有绝对锚点（完整正确解法 = 接近 1.0）
      └── 冻结 backbone 使 feature 无法适配 process error 判断
```

---

## 4. 新数据策划 Pipeline 设计

### 4.1 三级 Pair 构造体系

**Level 1: First-Bad-Edge Pairs（当前已有，保留）**
- 语义：局部错误边界判别
- 构造：`chosen` = prefix[0..t-1]，`rejected` = prefix[0..t]（t = 第一个错误步索引）
- 强度：最简单，可学习，当前 Math-Shepherd 上已达 95%+ ACC

**Level 2: Terminal Anchor Pairs（部分已有，需扩展）**
- 语义：完整正确解法 > 截断（但正确的）前缀
- 构造：`chosen` = 完整正确解法，`rejected` = 随机截断的正确前缀（步骤数 < 总步数）
- 目的：教会模型"完成"比"未完成"更好 → 直接修复 ProcessBench all-correct gap
- 建议比例：占总训练 pair 的 15-30%

**Level 3: MC-Estimated Step Pairs（新增，核心改进）**
- 语义：基于 MC rollout 成功率的软标签 step 偏好
- 构造：对每条轨迹的每个 step，用 4-8 次 MC rollout 估计该 step 之后的成功率 $p_t$
  - `chosen` = step with higher $p_t$（from same question, different rollouts）
  - `rejected` = step with lower $p_t$
- 目的：提供 sibling-branch 语义，而不仅仅是 single-trajectory edge
- 实现路径：用本地已有 Qwen2.5-7B-Instruct 做 MC rollout（或用 Qwen2.5-Math-PRM-7B 打步骤分）

### 4.2 数据混合策略

**Phase 1 (立即可行):** 调整 terminal anchor 比例
- 当前：`step_label_terminal_anchor_fraction = 0.5`（已有 `all_positive_fanout` 模式）
- 建议对比组：0.2 / 0.35 / 0.5 三组比例
- 同时对比：mixed `first_bad_edge` + `all_good_vs_all_bad` grid

**Phase 2 (中期，需新 adapter):** 引入 MC 软标签
- 用 Qwen2.5-Math-PRM-7B（本地已有）对 Math-Shepherd 每步打软标签
- 构造 sibling-branch pairs：同一问题，score_high vs score_low 的步骤对
- 过滤条件：两步分数差 > 0.3（避免标注噪声主导）

**Phase 3 (中期，新 adapter):** VersaPRM 风格自过滤
- 训练一个初步 value head 后，用它对训练数据重新打分
- 删掉预测分数与训练标签偏差 > 0.3 的样本
- 然后重新训练（self-distillation 效果）

**Phase 4 (长期):** MCTS-based sibling pair 构造
- 对 Math-Shepherd 原始问题做 MCTS 二分搜索，生成 OmegaPRM 风格的 tree 数据
- 这需要能运行较强推理模型（如 Qwen2.5-7B-Instruct 本地）做 MC rollout

### 4.3 新增 `PairBuildConfig` 参数设计

在现有 `PairBuildConfig` 基础上扩展：
```python
# 新增字段
step_label_terminal_anchor_fraction: float = 0.25  # terminal pair 占比目标（调参维度）
terminal_anchor_chosen_text: str = "full_positive"  # 完整正确解法
terminal_anchor_rejected_mode: str = "random_truncation"  # 截断前缀作为 rejected

# 新增 pair_semantics 类型
# "first_bad_edge"（已有）
# "terminal_anchor_positive"（已有）
# "mc_sibling_branch"（新增：MC 软标签 sibling pair）
# "all_good_vs_all_bad_grid"（已有）
```

---

## 5. 架构变体设计

### 5.1 Variant A: DualHead（双头分离，推荐优先尝试）

**动机：** PathFinder-PRM 证明解耦 local 和 global 信号能提升 data efficiency
**架构：**
```
backbone → hidden_state[last_token]
    ├── local_head (MLP) → local_score ∈ [0,1]  # 当前步局部质量
    └── terminal_head (MLP) → terminal_score ∈ [0,1]  # 链条最终完成偏好

inference_score = α * local_score + (1-α) * terminal_score
```
**训练损失：**
- `first_bad_edge` pairs → 主要更新 `local_head`
- `terminal_anchor` pairs → 主要更新 `terminal_head`
- `all_good_vs_all_bad_grid` pairs → 两个头都更新

**实现复杂度：** 中等。需在 `SigmoidValueHead` 中增加一个 `architecture="dual_head"` 分支

### 5.2 Variant B: MeanPool（均值池化替代 last_token）

**动机：** `last_token` 对中间步 error 不敏感；`mean_pool` 包含整个前缀的平均表示
**架构：**
```
backbone → hidden_states[all_tokens] → mean_pool → MLP head
```
**预期效果：** 对 local step error 检测可能无收益（因为 error 在中间某步），但对 terminal completion 可能有改善（因为包含了全序列信息）
**实现复杂度：** 低。只需改 pooling 策略，`ValueHeadConfig.pooling` 增加 `"mean_token"` 支持
**注意：** 需要在 `runtime.py` 的 feature encoding 中支持 mean pooling 存储

### 5.3 Variant C: LoRA Partial Unfreeze（最后 N 层 LoRA 适配）

**动机：** 冻结 backbone 的 feature 是通用语言建模优化的，对 process error 判断可能不是最优
**架构：**
```
backbone (last 2-4 layers: LoRA, rank=16)
    ↓
last_token pooling
    ↓
MLP head
```
**预期效果：** backbone 最后几层能适配到"step quality"特征空间，提升跨源和跨 benchmark 泛化
**实现复杂度：** 高（需要 PEFT 集成）。但可以在 Phase B 已有的 PEFT 框架基础上改造
**成本：** 显存增加（需要存 LoRA 梯度），训练时间增加约 3-5 倍

### 5.4 Variant D: PrefixSuffix Two-Tower（前缀+后缀双塔，新颖）

**动机：** ProcessBench 要求区分"前缀到此为止"和"前缀 + 完整后续"——两者的区别在于有无 completion
**架构：**
```
输入 A: prompt + prefix 到步骤 t
输入 B: 完整解法（或 NULL 补位）

tower_A = backbone(A) → last_token → feature_A
tower_B = backbone(B) → last_token → feature_B

score = MLP([feature_A; feature_B - feature_A])
```
**直觉：** `feature_B - feature_A` 捕获"还差什么"，类似 completion advantage
**实现复杂度：** 高（需要每条样本做两次前向）
**注意：** 训练时只需要 `first_bad_edge` 已有数据，不需要额外标注（完整 positive 解法已有）

---

## 6. 实验设计矩阵

实验统一命名规则：`F<tier><idx>_<描述>`
- `F1` = 数据修复类（Fix Data）
- `F2` = 架构修复类（Fix Architecture）
- `F3` = 训练策略修复类（Fix Training）
- `F4` = 长期/生成式路线（Future）

### 6.1 F1 系列：数据修复（直接修复监督契约）

#### F1A: Terminal Anchor 比例调参
```
F1A_TA_FRAC20_MS_SEED3:
  - 意图：测试 terminal anchor 20% 比例下 ProcessBench 是否改善
  - 数据：Math-Shepherd + terminal_anchor_fraction=0.20 (first_bad_edge 80%)
  - 观测：ProcessBench all-correct 分项，同源 held-out pair_acc
  - 预期：比纯 first_bad_edge 的 ms_e43 ProcessBench 提升 3-5 AUC 点

F1B_TA_FRAC35_MS_SEED3:
  - 意图：更高 terminal anchor 比例是否带来更多改善（或损害同源判别）
  - 数据：Math-Shepherd + terminal_anchor_fraction=0.35
  - 观测：对比 F1A，看 local discrimination vs terminal completion 的 tradeoff

F1C_TA_FRAC35_PRMBENCH_SEED3:
  - 意图：验证 PRMBench_Preview 上也能从 terminal anchor 获益
  - 数据：PRMBench_Preview + terminal_anchor_fraction=0.35
  - 观测：ProcessBench 指标，PRMBench pair_acc
```

#### F1D: Mixed Grid + Terminal
```
F1D_GRID_TA_MS_SEED3:
  - 意图：同时引入 all_good_vs_all_bad grid + terminal anchor，看是否互补
  - 数据：Math-Shepherd, pair_mode=first_bad_edge + grid 20% + terminal 20%
  - 观测：ProcessBench 各分项，重点 all-correct 和 first-bad 分开看
```

#### F1E: Self-Filtering（VersaPRM 风格）
```
F1E_SELFFILTER_MS_SEED3:
  - 意图：用训完的 ms_e43 对训练数据过滤（删掉预测分数与标签差>0.3 的样本）
  - 数据：Math-Shepherd 过滤后重训
  - 观测：是否减少低质量 pair 能改善 ProcessBench 稳定性
```

### 6.2 F2 系列：架构修复

#### F2A: DualHead（双头分离）
```
F2A_DUALHEAD_MS_SEED3:
  - 意图：local_head + terminal_head 分别由不同 pair type 督导
  - 架构：architecture="dual_head"，inference_alpha=0.5（可搜索）
  - 数据：Math-Shepherd，搭配 F1B（35% terminal anchor）
  - 观测：local_head 的 ProcessBench first-bad-step 分项 vs terminal_head 的 all-correct 分项

F2B_DUALHEAD_ALPHA_SWEEP:
  - 意图：搜索最优 alpha（local:terminal 权重）
  - alpha 候选：0.3, 0.5, 0.7
  - 观测：同源 held-out + ProcessBench all-correct + ProcessBench first-bad
```

#### F2C: MeanPooling 消融
```
F2C_MEANPOOL_MS_SEED3:
  - 意图：最廉价的架构改动，看 mean pooling 对 ProcessBench 是否有帮助
  - 架构：pooling="mean_token"（需代码修改）
  - 数据：Math-Shepherd（与 ms_e43 完全可比）
  - 观测：ProcessBench 指标，同源 held-out
```

### 6.3 F3 系列：训练策略修复

#### F3A: DPO 风格 Log-Ratio Loss（PRIME 启发）
```
F3A_DPOLOG_MS_SEED3:
  - 意图：把 ranking 损失替换为 DPO log-ratio 损失，更接近 PRIME 的隐式 PRM
  - 损失：L = -log σ(β * (log(π_θ(chosen) / π_ref(chosen)) - log(π_θ(rejected) / π_ref(rejected))))
  - 注：π 对应的是 value head 输出的 logit，π_ref 是初始化时的 logit
  - 观测：对比 F1B（相同数据，不同损失），ProcessBench 是否受益
```

#### F3B: Curriculum 训练顺序
```
F3B_CURRICULUM_LOCALFIRST_MS_SEED3:
  - 意图：先用纯 first_bad_edge 训，再加 terminal anchor 微调（两阶段课程）
  - 阶段1：E0-E5 epoch 纯 first_bad_edge
  - 阶段2：E6-E10 加入 terminal anchor pairs（35%）继续训
  - 观测：和直接混合的 F1B 对比，看课程学习是否更稳定
```

### 6.4 F4 系列：长期路线（不立即执行，记录方向）

```
F4A: R-PRM 风格 SFT+DPO 路线
  - 用 Qwen2.5-7B-Instruct 对 PRM800K/Math-Shepherd 样本生成推理链分析
  - 先 SFT 学会生成验证链，再 DPO 自我提升
  - 预期：直接对 ProcessBench F1 改善 8-10 点
  - 资源需求：高（需要完整 backbone fine-tuning）

F4B: GenPRM 风格（生成式 + 代码验证）
  - 在 F4A 基础上增加代码生成+执行验证
  - 预期：进一步改善 symbolic math 步骤验证准确率

F4C: MC Relabeling Pipeline
  - 用 Qwen2.5-7B-Instruct 对 Math-Shepherd 每步做 MC rollout（4次）
  - 重新标注步骤质量软标签
  - 用软标签替代 first_bad_edge hard label 训练
  - 预期：提供 sibling-branch 语义，改善 ProcessBench 约 3-5 AUC
```

---

## 7. 推荐执行顺序

```
Priority 1（本周可跑，cost 最低）:
  F1A → F1B → F2C
  （terminal anchor 比例调参 + mean pooling 消融）
  预期耗时：每组约 2-3 小时 × 3 组 = 6-9 小时

Priority 2（本周末，需少量代码改动）:
  F1D → F2A
  （Grid+Terminal 混合 + 双头架构 prototype）
  预期耗时：代码 ~1 天 + 实验 ~6 小时

Priority 3（下周，中等代码工作量）:
  F3A → F3B
  （DPO log-ratio loss + 课程训练）
  预期耗时：代码 ~1 天 + 实验 ~1 天

Priority 4（长期，需大资源）:
  F4A / F4B / F4C
  （生成式路线 / MC 重标注）
```

---

## 8. 提升 ProcessBench 的核心假设链

根据上述分析，以下是最可能 work 的改进路径：

```
假设 1: ProcessBench all-correct gap 主要来自监督缺失
  → 验证方法: 加入 terminal anchor (F1A/F1B) 后看 all-correct 分项
  → 如果 all-correct 分项提升 ≥5 AUC，假设成立

假设 2: 现有 first_bad_edge 已足够捕获 local discrimination
  → 验证方法: F2C (mean pooling) 和 F2A (dual head local part) 与 ms_e43 对比
  → 如果 local discrimination 没有明显下降，说明 local head 本身没问题

假设 3: Local + Terminal 双目标存在 tradeoff，而非可以同时优化
  → 验证方法: F2A dual head 的 alpha sweep (F2B)
  → 如果 alpha=0.5 是 Pareto 优，说明两个目标可以 jointly 优化；如果某个 alpha 明显更好，
     说明存在 tradeoff，需要 curriculum 解法

假设 4: Frozen backbone 是制约泛化的根本原因
  → 验证方法: 比较 F2A/F1B（frozen）和 F3A（DPO，需要 LoRA 解冻）的 ProcessBench 差距
  → 如果差距大，说明 backbone 要动；如果差距小，说明数据修复够用
```

---

## 9. 架构深度研究补充（2026-03-11 第二轮）

### 9.1 关键结论：Frozen Backbone 是 ProcessBench 天花板的根本原因

**来源：** 第二轮文献检索，交叉对比本项目实验记录与已发布 PRM 模型的架构配置。

**核心发现：**

更稳妥的表述应当是：

1. 公开强结果中，多数接近或超过 `70-75%` 的系统都使用了
   backbone adaptation（至少 `LoRA`）或 generative verifier finetuning。
2. 这支持“当前 frozen-backbone 只是基线，不是上限”。
3. 但这不是严格数学定理，也不能被读成“只要上 LoRA 就一定达到同水平”。

支持这一趋势的代表性例子：

| 模型 | 架构 | Backbone 状态 | ProcessBench 成绩 |
|---|---|---|---|
| Qwen2.5-Math-PRM-7B | 线性头 | 全量 fine-tune | 80%+ |
| Skywork-o1-PRM-7B | 线性头 | LoRA 或全量 | 75%+ |
| RLHFlow/Mistral-PRM | 线性头 | LoRA | 70%+ |
| R-PRM (SFT+DPO) | 生成式 | 全量 | F1=70.4 |
| **本项目 ms_e43** | **MLP头** | **冻结** | **AUC≈0.62** |

**结论：**

- 冻结 backbone 下，最后一个 token 的 hidden state 是按"预测下一个 token"优化的，不是按"判断这一步对不对"优化的
- 线性/MLP 头可以在同源分布下达到 91-95%（此时 backbone 的预测质量信号碰巧与 pair 质量对齐）
- 跨域迁移（ProcessBench transfer）需要 backbone representation 本身发生偏移
- **这是 representation gap，不是 capacity gap；换更复杂的头无法修复**

### 9.2 `dual_head` 的实际状态

本仓库的 `src/ours/phase_b/value_head.py` 已实现 `dual_head` 架构：
- `shared`：dropout + Linear(hidden → mlp_hidden) + 激活 + dropout
- `local_proj`：Linear(mlp_hidden → 1)，用于 first_bad 类 pair
- `terminal_proj`：Linear(mlp_hidden → 1)，用于 terminal_completion 类 pair
- 推理时：`logits = alpha * local_logits + (1 - alpha) * terminal_logits`

**关键限制：当前训练循环（`scripts/phase_e_train_value.py` + `src/ours/phase_e/training.py`）没有 pair_semantics 路由逻辑。**

没有路由的 `dual_head` 等价于：
- local_proj 和 terminal_proj 同时接收所有 pair 的梯度
- 等同于参数更多的 `gated_mlp`，但没有额外的归纳偏置
- 预计结果：与 `gated_mlp` 相似（smoke 实验显示 gated_mlp 未帮助）

要让 `dual_head` 生效，需要在训练侧为不同 pair_semantics 路由到对应 head：
- `first_bad_edge` / `first_bad_fanout` / `lastsafe_vs_laterbad` → `local_proj`
- `terminal_completion_anchor` → `terminal_proj`

这是下一个架构实验的前提基础设施，不在 PBR2/PBR3 范围内。

### 9.3 Step-Position Metadata 注入（低成本可行方案）

**方案：** 把 step position 信息作为辅助特征拼接到 pooled vector 后，再送入 head。

具体而言：
- 在训练 pair 中，`positive_step_index` 和 total_steps 已经在 metadata 中
- 构造 2-3 个归一化特征：`step_frac = positive_step_index / total_steps`，`is_terminal = (pair_semantics == "terminal_completion_anchor")`
- 拼接到 hidden vector 后：`[features; step_frac; is_terminal]`，维度从 3584 变为 3586
- head 的输入维度相应调整

**优点：**
- 无需修改 backbone
- 给 gated_mlp / dual_head 的 gate 提供显式的 local-vs-terminal 结构信息
- 理论上直接解决 "gate 只靠特征内容区分，无法感知位置" 的问题

**当前阻碍：** feature cache 缓存的是纯 backbone vector，不包含 step position。

若要实现，需要：
1. 在训练 dataset 侧提取 step_position metadata 并传给 head
2. head 的 `forward(features, aux=None)` 支持可选辅助输入
3. feature cache 的 key 不需要改（metadata 在 head 侧注入，不影响 backbone feature）

**工程优先级：** PBR2/PBR3 完成后，如果 later-bad 仍是 canary，可以做 PBR4 step-position ablation。

### 9.4 LoRA Backbone 路径（中期，最高 impact）

基于文献，LoRA on last 4 transformer layers of Qwen2.5-7B:
- 约 50-100M 可训练参数（rank=4，target modules=qkv+mlp）
- 兼容单 GPU（4090 24GB）
- Feature cache 需要失效重算（LoRA 权重更新后 backbone 特征改变）
- 或者改为 mini-batch on-the-fly encoding（不用 cache）

**与 PBR2/PBR3 的关系：**
- PBR2/PBR3 是 frozen-backbone 路线的充分验证
- 如果 frozen-backbone 的 ProcessBench AUC 在 PBR3 后仍低于 0.65，进入 LoRA 路线
- LoRA 路线需要单独设计 Phase E-LoRA suite（新的 run script 和 train script 分支）

### 9.5 Pooling 策略澄清（mean pooling 不推荐）

基于文献对比，**mean pooling 对 causal LM 不推荐**：
- causal attention 下，last_token 已经 attend 到了完整序列
- mean pooling 会把早期 token（通常是题目描述，没有步骤质量信息）也纳入
- 所有发布的 causal-LM-based PRM 都使用 last_token pooling

**结论：** 不要实现 mean pooling 对照组，这是已知会更差的设计。

---

## 10. 第二轮互联网研究补充（2026-03-11 晚间）

### 10.1 PRIME：隐式 PRM 与 Online 更新

**论文：** Process Reinforcement through IMplicit rEwards（arXiv 2502.01456）

**核心机制：**

- Token-level implicit reward = `log(π_θ(token) / π_ref(token))` — 即 DPO 意义下的 log-ratio
- 无需独立标注 step-level labels，只需 outcome labels（答案对/错）
- PRM 用当前 policy 的 rollout + outcome reward 在线更新（CE loss），LR = 1e-6
- 已开源：[PRIME-RL/PRIME](https://github.com/PRIME-RL/PRIME)
- 结果：Eurus-2-7B-PRIME 在 7 个 benchmarks 上超越 Qwen2.5-Math-7B-Instruct，仅用 10% 训练数据

**对本项目的启发：**

1. **PRIME 不是 discriminative head，是 generative model 自身的 log-ratio** — 不能直接用于当前 frozen-backbone + value head 架构
2. **隐式 PRM 思路可以在 LoRA 路线下间接复用**：一旦解冻 backbone，可以用 DPO-style 目标而非 BCE/ranking，直接优化 log-ratio 使其对齐过程质量
3. **当前阶段（frozen backbone）暂不适用**；作为 PBR6 LoRA 路线完成后的 PBR7 目标留存

### 10.2 OmegaPRM：MCTS Binary Search 数据策划

**论文：** Improve Mathematical Reasoning in Language Models by Automated Process Supervision（arXiv 2406.06592）

**核心机制：**

- Divide-and-Conquer MCTS：用 binary search 快速定位第一个错误步
- 每个节点包含 question + preceding steps，随机采样后续 completion 估计步骤正确率
- 优势：比 Math-Shepherd 的朴素 MC 更精准，正负样本更均衡
- 结果：Gemini Pro 在 MATH500 从 51% 提升至 69.4%，Gemma2 27B 从 42.3% 提升至 58.2%

**对本项目的启发：**

1. Math-Shepherd 本身已经用了类似思路（MC estimation），但噪声较高
2. OmegaPRM 的 binary search 思路可以用于改善 `later-bad` 对构造精度：若某步的 MC 完成率 << 前一步，且 >> 后续步，该步很可能是真正的第一个 later-bad 步
3. **当前实施障碍**：需要访问 golden answers + 多步采样 API，成本较高
4. **近期可行替代**：使用 `pair_type_allowlist=("lastsafe_vs_laterbad", "earlygood_vs_laterbad")` + Math-Shepherd 已有的 MC 标注是次优的近似 — 这就是 `ms_laterbad_v1` 的设计哲学

### 10.3 PathFinder-PRM：误差类型层次化分类

**论文：** Error Typing for Smarter Rewards (arXiv 2505.19706, SOTA on PRMBench 67.7)

**核心机制：**

- 第一步：独立预测 Math error / Consistency error（两路 masked prediction，不自回归）
- 第二步：把误差类型信号聚合成最终步级 reward
- 优势：防止误差类型间的自回归 cascade；3× 数据效率下超越 SOTA
- 训练数据：400K 样本（PRM800K + RLHFlow 轨迹），三维步级标签

**对本项目的启发：**

1. **当前 discriminative scalar head 无法支持层次化分类** — 需要生成式架构才能实现
2. **设计哲学可借鉴**：将"是否有数学错误"与"是否有一致性错误"分开，比一个统一的"好/坏"标签携带更多信息
3. **近期无法直接实现**；但可以考虑在 PRMBench pair 中增加"error_type"字段作为 semantic_weight 的来源（e.g., 明确的数学错误 → 更高 semantic_weight）

### 10.4 Curriculum Learning for PRM

**文献：** The Lessons of Developing PRMs (ACL 2025) + Self-Evolving Curriculum (SEC, arXiv 2505.14970)

**关键发现：**

- Monte Carlo estimation 数据质量 < LLM-as-judge < 人工标注（in terms of generalization）
- 预定义难度课程的局限：对 ModelA 难的，对 ModelB 未必难
- Self-Evolving Curriculum (SEC)：把课程选择建模为 multi-armed bandit，动态优化
- 课程难度对 AIME 有 27% 相对提升

**对本项目的适用路径：**

1. **当前可行（无需新基础设施）**：用 `step_gap` 作为隐式难度指标实现朴素课程：先训 gap1（最易），再训 gap3+（较难）
   - 在 run script 中可以用 `PBR_curriculum` 模式实现
   - 已有 `ms_core_v1 → ms_align_v1 → ms_laterbad_v1` 的渐进式 profile 本身就是一种课程策略
2. **中期**：用 LLM-as-judge 重新标注部分 Math-Shepherd 样本，替换 MC 标注的低质量 pair（有显著的数据效率提升潜力）

### 10.5 对 GRPO/RLOO + PRM Online Update 的关联性

**背景（来自搜索）：** 2025 年 RLVR 训练已完全以 GRPO/RLOO 为主流（critic-free policy gradient），LoRA 已和 GRPO 集成（Unsloth、OpenRLHF 均支持 QLoRA + GRPO）

**对本项目的意义：**

1. 本项目训练的 value head 最终目标是为 BCR/NRAP RL 训练提供 dense reward
2. 如果未来要做 Phase F（RL fine-tuning），需要把 value head 集成进 GRPO 训练循环
3. 最重要的接口：value head 的 score 需要可微（或者 detached scalar reward）作为 per-step reward

### 10.6 ProcessBench 的 "all-correct" 样本处理 — ThinkPRM 视角

**来自 ThinkPRM (arXiv 2504.16828) 的关键补充：**

- ThinkPRM 通过 **让 LLM 推理验证过程**（chain-of-thought verification）解决 all-correct 问题
- 对 all-correct 样本，ThinkPRM 能明确推理"这道题所有步骤都是正确的"
- discriminative scalar head 没有显式机制输出"全部正确"结论 — 只能靠训练分布覆盖 terminal 步

**当前项目的近似方案（无法复现 ThinkPRM 全部能力，但可以减少 gap）：**
- `terminal_completion_anchor` pairs（full correct solution > truncated prefix）训练 terminal 步的正向偏好
- `dual_head` 的 `terminal_proj` 显式分离 terminal 目标
- 这两种都是 discriminative 近似，预计仍会有残差 gap vs. ThinkPRM

### 10.7 完整 PBR 实验路径图（2026-03-11 最新版）

```
当前状态（最好结果）：
  ms_e43: ProcessBench GSM8K AUC=0.6245, Math AUC=0.6341
    状态修正（later audit）:
    - 不应再写成 `provisionally RL-ready`
    - 最新 strict RL-promotion 读法应为:
      - `near_rl_ready_but_terminal_gap`
    - 关键原因:
      - `pb_min_terminal_top1 = 0.0099`
    - 对应审计:
      - `assets/artifacts/phase_e_rl_promotion_diag/rl_promotion_state_review_0311_20260311T070332Z/summary.md`
  pbr2_ms_align_gated: GSM8K AUC=0.4713, Math AUC=0.5055 (ProcessBench redesign, smoke)

可立即运行的实验组（数据 curate + 训练 pipeline 均已就绪）：

  PBR1 [已完成 smoke] → ms_align_v1 + mlp/gated_mlp, 4096 pairs
    pbr1_ms_align_mlp 是当前 ProcessBench 路线的最佳 smoke 结果

  PBR2 [待运行] → ms_align_v1 + mlp + 16384 pairs, 3 seeds
    目标: pb_math_auc >= 0.58, pb_gsm_auc >= 0.52
    意义: 确认 smoke 结论能否在全量数据下复现

  PBR3 [待 PBR2 稳定后运行] → ms_laterbad_v1 + mlp + 16384 pairs, 3 seeds
    目标: later-bad 改善，first_edge regression <= 0.05
    意义: targeted later-bad 是否优于 generic grid (ms_align_v1 的 ms_grid component)

  PBR4 [待 PBR3 后运行] → ms_prm_align_v1 + mlp/gated_mlp + ms_laterbad_v1 + gated_mlp, smoke
    意义: 对比 PRMBench local 监督是否补充 ms 的 first-bad 覆盖

需要基础设施的实验组：

  PBR5 [STATUS UPDATE: routing infra 已实现，但 full rerun 尚未完成] → dual_head + ms_laterbad_v1
    状态修正:
    - `training.py` 的 pair_semantics routing 已在后续代码中实现
    - `scripts/phase_e_train_value.py` 也已支持 `dual_head`
    - 现在的真实阻塞项不再是 routing infra，
      而是：
      - 更强 source 尚未主线化
      - full-scale experiment budget 仍不足

  PBR6 [BLOCKED: 需要 training 脚本 LoRA 模式] → ms_laterbad_v1 + mlp + LoRA backbone
    前置: feature cache 禁用 + on-the-fly encoding + LoRA 适配器支持
    预期: 突破 frozen-backbone ~0.62 AUC 天花板，跨向 0.65-0.70+

战略路线图：
  PBR2 → PBR3 确认数据几何天花板
  若天花板 < 0.60 → 进入 PBR6 (LoRA 路线)，这是最高 impact 架构改动
  若天花板 0.60-0.65 → 先 PBR5 (dual_head routing) 再 PBR6
  若天花板 > 0.65 → ProcessBench 冻结路线已达目标，可进入 RL 训练
```

---

## 11. 参考文献

- [R-PRM: Reasoning-Driven Process Reward Modeling](https://arxiv.org/abs/2503.21295)
- [ThinkPRM: Process Reward Models That Think](https://arxiv.org/abs/2504.16828)
- [GenPRM: Scaling Test-Time Compute of Process Reward Models](https://arxiv.org/abs/2504.00891)
- [OmegaPRM: Automated Process Supervision via Divide-and-Conquer MCTS](https://arxiv.org/abs/2406.06592)
- [PathFinder-PRM: Error-Aware Hierarchical Supervision](https://github.com/declare-lab/PathFinder-PRM)
- [PRIME: Process Reinforcement through Implicit Rewards](https://arxiv.org/abs/2502.01456)
- [VersaPRM: Multi-Domain PRM via Synthetic Reasoning Data](https://arxiv.org/abs/2502.06737)
- [Qwen2.5-Math-PRM: Towards Effective Process Supervision](https://qwenlm.github.io/blog/qwen2.5-math-prm/)
- [ProcessBench: Identifying Process Errors in Mathematical Reasoning](https://arxiv.org/abs/2412.06559)
- [PRMBench: A Fine-grained Benchmark for Process-Level Reward Models](https://arxiv.org/abs/2501.03124)
- [Full-Step-DPO: Self-Supervised Preference Optimization with Step-wise Rewards](https://aclanthology.org/2025.findings-acl.1249.pdf)
- [The Lessons of Developing Process Reward Models in Mathematical Reasoning](https://arxiv.org/abs/2501.07301)
- [BiPRM: Bidirectional Evaluation Paradigm for PRMs](https://arxiv.org/html/2508.01682)
- [ActPRM: Efficient Process Reward Model Training via Active Learning](https://arxiv.org/abs/2504.10559)
- [BiRM: Better Process Supervision with Bi-directional Rewarding Signals](https://arxiv.org/abs/2503.04618)
- [AURORA: Automated Training Framework of Universal PRMs](https://arxiv.org/abs/2502.11520)
- [Curriculum-RLAIF](https://openreview.net/pdf/189f5f32f4df82449edce9c06a6fbd6a0d7ccfea.pdf)

---

## 12. 第三轮检索补充（2026-03-11 深夜）— 文献-实验直接对接

### 12.1 BiRM 架构细节（可立即用于改进当前 training.py）

**来源：** arXiv:2503.04618 HTML 版（ACL Findings 2025）

**BiRM 精确训练配置：**

```
shared_backbone (fine-tuned, 1 epoch from generator checkpoint)
  ├── PRM head (ϕR):   MSE loss on step-correctness labels
  └── VM head (ϕV):    MSE loss on future-success-probability labels

L_BiRM = L_PRM + c · L_VM   (c 是调参超参)

推理时得分融合：
  f(st) = g(st) + β · h(st)     (β 在验证集上搜索最优值)
```

**关键细节：**
- PRM head 擅长早期步骤判断（前 1/3 steps）
- VM head 擅长晚期步骤判断（后 1/3 steps）
- 两者融合 = 全局最优
- **不是门控（gating）机制，而是简单的加权求和**，且权重是 FIXED（不是 learnable）

**直接应用于当前 `training.py`（不需要 backbone 微调）：**

当前问题：terminal anchor pairs 和 local first-bad-edge pairs 用同一个 `contrastive_margin_loss`。
BiRM 洞见：这两类 pair 应该用**不同的损失函数**：
- Local pairs → `contrastive_margin_loss`（当前已有）
- Terminal anchor pairs → `BCE loss`（chosen=1 全程正确，rejected=0 截断前缀）

**实现方案（`compute_pair_objective` 修改）：**

```python
# 新增参数：--terminal-bce-lambda 0.1
# 在 compute_pair_objective() 中:
local_mask = [p for p in pairs if p.metadata.get("pair_semantics") != "terminal_completion_anchor"]
terminal_mask = [p for p in pairs if p.metadata.get("pair_semantics") == "terminal_completion_anchor"]

L_local = contrastive_margin_loss(local_mask)      # 原有 L_PRM
L_terminal = bce_loss(terminal_mask, chosen=1, rejected=0)   # 新增 L_VM 代理
L_total = L_local + lambda_terminal * L_terminal
```

这是目前 **最高优先级的 training.py 改动**，预期直接解决 terminal supervision conflict。

---

### 12.2 ActPRM SOTA 配置（仅供参考，部分可适配）

**SOTA 75% ProcessBench 配置：**
- Backbone: Qwen2.5-Math-7B-Instruct → **全量微调**
- 数据量: 1M+ 轨迹，过滤后保留 60%
- 主动学习: 集成 PRM 估计不确定性，只标注最不确定 20%（QwQ-32B 做 judge）
- 训练: fine-tune on filtered 600K samples

**结论：** 我们在冻结 backbone 下能达到 ~62% AUC，距 SOTA 差 13 个点。
理论上 LoRA 微调（参考 Qwen2.5-Math-7B-Instruct 作为起点）可以弥补 7-10 个点的差距。

---

### 12.3 当前运行实验状态（2026-03-11 T11:45 +0800）

**Experiment A: Terminal Ratio Sweep（正在运行）**
```bash
PID=3347563
日志: assets/artifacts/phase_e_logs/phase_e_ta_ratio_sweep/sweep_stdout.log
TRAIN_BATCH_SIZE=8, EVAL_BATCH_SIZE=8
Ratios: [0.0, 0.05, 0.10, 0.20]
Benchmarks: [processbench_math, processbench_gsm8k]
```

**R-PRM 诊断（已完成）**
```
E48 直接对（legacy direct_pair）：
  - chosen_len p50=1381，94% 超过 max_length=1024
  - first_diff p50=740，11.6% 分歧点被 truncate 隐藏
  - 结论：MODERATE RISK（非主因，但有贡献）

compact_diag 对：
  - chosen_len p50=268，0% 超过 max_length=1024
  - first_diff 0% 被隐藏
  - 结论：compact_verdict 模式完全消除截断风险
```

**待实现实验（本轮）：**
- Exp-B: 课程训练 Phase1→Phase2 (warm-start) → 需要新 shell suite
- Exp-C: MS + PRMBench 多源混合 → 需要新 shell suite
- Exp-D: BiRM-style dual loss (pair_semantics routing) → 需要修改 training.py
