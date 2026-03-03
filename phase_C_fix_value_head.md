# Phase C Value Head 修复备忘（含外援策略）

> 文档目的：把当前价值头问题、可执行修复路线、以及“外援引入”策略沉淀成统一参考，避免后续反复讨论与试错。
>
> 日期：2026-03-03（本地仓库状态）

---

## 1. 问题本质：这是一个“冷启动弱监督”问题

当前任务不是普通分类，而是学习前缀价值函数：

$$
V_\phi(h_t) \approx \Pr(\text{最终答对} \mid h_t)
$$

但 step-level 真值不可见，当前标签来自 rollout Monte Carlo 估计：

$$
\hat q(h_t) = \frac{1}{K}\sum_{k=1}^{K}\mathbf{1}\{\text{rollout}_k \text{最终正确}\}
$$

这会带来两类核心噪声：

1. `Q 标签噪声`：$K$ 小时方差高、离散化严重（例如 $K=8$ 时只取 $\{0,0.125,\dots,1\}$）。
2. `Pair 噪声`：clean/corrupt 之间如果真实质量差很小，contrastive 排序近似随机。

结论：当前“价值头学不动”更像信号质量问题，不是单纯 loss 选错。

---

## 2. 仓库内当前实现（关键点）

### 2.1 C1: Q 标签生成

- 入口：`scripts/phase_b_prepare_value_data.py`
- 聚合位置：`_aggregate_rollout_targets`
- 当前核心字段：`success_rate`, `parseable_rate`, `k_rollouts`
- 现状：`success_rate = n_correct / k_rollouts`（未显式建模不确定度）

### 2.2 C1: corruption 生成与训练对接

- 生成：`src/ours/phase_b/corruptions.py`
- 默认：`max_corruptions_per_prefix = 1`
- 可选类型：`binary_flip`, `operator_flip`, `numeric_perturb`, `step_drop`
- 训练侧 primary 选择：`src/ours/phase_b/value_data.py` 中按 `corruption_id` 字典序选 1 个（与“质量”无关）

### 2.3 C2: 训练过滤逻辑

- 入口：`scripts/phase_b_train_value.py`
- 现有机制：
  - calibration sample weighting（`confidence/parseable/...`）
  - contrastive pair filter（`confidence/parseable/...`）
  - score-gap mining（基于当前模型分数差）
- 问题：过滤依赖当前模型打分，早期容易形成“错误自强化”。

---

## 3. 目前现象的结构性解释

已观测到的典型模式（结合当前产物）：

1. rollout 标签整体偏高（很多 prefix 接近 1.0）。
2. corruption 类型分布不均衡，`numeric_perturb` 占比高。
3. pair 指标（`pair_accuracy`, `auc_clean_vs_corrupt`）经常接近随机。
4. post-hoc calibration 有时能改善 brier，但不能保证排序能力提升。

这说明：

- calibration 后处理能修“概率刻度”，但不能凭空产生“区分能力”；
- 若训练 pair 本身弱，contrastive loss 学到的就是噪声排序。

---

## 4. 修 Q 标签质量：先做这 5 件事

### 4.1 用收缩估计替代裸均值

对每个 prefix，使用 Beta-Binomial 收缩：

$$
\tilde q = \frac{n + \alpha \mu_0}{K + \alpha}
$$

- $n$：正确 rollout 数
- $K$：rollout 总数
- $\mu_0$：全局先验均值（可用 train 全体 success mean）
- $\alpha$：先验强度（建议网格：`2, 4, 8`）

收益：降低小样本极值噪声。

### 4.2 显式记录标签不确定度

建议在 `rollout_targets.jsonl` 增加字段：

- `q_shrunk`
- `q_var`（近似）
- `q_ci_width`

近似可用：

$$
\mathrm{Var}(\tilde q)\approx \frac{\tilde q(1-\tilde q)}{K+\alpha+1}
$$

### 4.3 训练权重改为“方差感知”

当前权重主要看 `|q-0.5|`。应增加：

$$
w_i = \frac{1}{q_{var,i} + \tau}
$$

并做裁剪：

$$
w_i \leftarrow \mathrm{clip}(w_i, w_{\min}, w_{\max})
$$

### 4.4 rollout 预算改为两阶段

1. 全量 prefix 先跑 `K=8`。
2. 对不确定前缀（如 $|q-0.5|<0.2$ 或 CI 宽）补采样到 `K=24/32`。

这样比全量高 K 更省算力，且能显著提纯难例标签。

### 4.5 多采样温度混合

同一 prefix 的 rollout 使用两档温度（如 `0.5 + 0.8`）混合，降低单一采样模式偏差。

---

## 5. 修 Pair 质量：这是当前第一优先级

### 5.1 不要每个 prefix 只留一个 corruption

把 `--max-corruptions-per-prefix` 从 1 提升到 4（或 6），保留多种扰动。

### 5.2 primary corruption 选择改成“最大伤害”

现策略是字典序最小 `corruption_id`，应改为：

$$
c^\* = \arg\max_c \Delta Q_c,\quad \Delta Q_c = \tilde q_{\text{clean}} - \tilde q_{\text{corrupt},c}
$$

### 5.3 引入 pair margin 过滤

只保留高质量 pair：

$$
\Delta Q \ge m
$$

建议 `m` 从 `0.2` 起扫（`0.15/0.2/0.3`）。

### 5.4 控制 corruption 类型分布

训练时按 type 近似均衡采样，避免 `numeric_perturb` 垄断。

### 5.5 从“一对一”升级到“一对多”

每个 clean 对多个 corrupt，contrastive loss 采用平均或 hardest-k：

$$
\mathcal{L}_{ctr} = \frac{1}{|C_i|}\sum_{c\in C_i}\max(0, m - s_i^{clean} + s_{ic}^{corrupt})
$$

---

## 6. 外援方案（cold-start 引导）

### 6.1 外援的定位（关键）

外援应定位为 `cold-start labeler`，不是最终方法主体。

可用组合：

1. MC rollout（你们已有）
2. 强教师 judge/verifier（外部大模型或已有 PRM）
3. 共识过滤（只保留一致样本）

融合标签：

$$
\tilde q_i = \lambda q_i^{MC} + (1-\lambda) q_i^{teacher}
$$

共识权重：

$$
w_i = \mathbf{1}\{|q_i^{MC}-q_i^{teacher}|<\tau\}\cdot (1-\mathrm{Entropy}(\tilde q_i))
$$

### 6.2 可借鉴工作（高相关）

1. Let’s Verify Step by Step (2023): 大规模过程监督可行  
2. Math-Shepherd (ACL 2024): 自动过程标注 pipeline  
3. Tree-PLV (EMNLP 2024): tree search + pairwise verifier  
4. Rewarding Progress / PAV (ICLR 2025): 用 progress 代替粗 outcome  
5. ThinkPRM (2025): 生成式 verifier，低标签预算可用  
6. The Lessons of Developing PRMs (2025): PRM 数据噪声与泛化教训  
7. ProcessBench / PRMBench: 过程奖励模型泛化挑战

---

## 7. 引入外援会不会破坏创新性？

不会，前提是边界清楚。

### 7.1 不会破坏的做法

- 把外援用于 C1 标签提纯与冷启动。
- 论文/报告主贡献放在：
  1. 前缀价值建模设计
  2. pair 质量控制机制
  3. ABR/BCR 路由策略与效率-准确率权衡
  4. 同预算下的净收益

### 7.2 会稀释贡献的做法

- 直接把外部 PRM 当核心算法，不做自家方法增量。
- 缺少消融，导致无法证明收益来源。

### 7.3 必做消融

1. no external aid  
2. external aid only  
3. external aid + ours  
4. equal-budget 对比（同 token、同推理调用次数）

---

## 8. 可执行路线图（建议 2 周）

### 阶段 A（2-3 天）：数据审计

1. 统计每类 corruption 的数量、$\Delta Q$ 分布、通过 margin 比例。
2. 统计 `q_var` 与误差相关性。
3. 输出审计报告（必须先做，再训）。

### 阶段 B（4-5 天）：C1 提纯

1. 上线 `q_shrunk/q_var` 字段。
2. `max_corruptions_per_prefix >= 4`。
3. primary 选择改 `argmax \Delta Q`。
4. 生成“高质量 pair 子集”版本数据。

### 阶段 C（4-5 天）：C2 重训与对照

1. variance-aware calibration weighting。
2. margin-filtered contrastive。
3. 记录 raw 与 posthoc 全指标。
4. 跑固定 budget 的下游 rerank utility（必须有）。

---

## 9. 现有脚本可直接尝试的命令模板

### 9.1 仅通过现有参数先提升 pair 覆盖度（不改代码）

```bash
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_TRICK10_K16_COMBINED \
RUN_PREFIX=phase_c_try_more_corr \
PHASE_C_PREP_EXTRA_ARGS="--max-corruptions-per-prefix 4" \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_value_suite.sh
```

### 9.2 强化 pair 过滤（现有参数）

```bash
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_TRICK9_HARD_NEG_MINING \
RUN_PREFIX=phase_c_try_pair_filter \
PHASE_C_TRAIN_EXTRA_ARGS="--contrastive-pair-filter confidence_parseable --contrastive-confidence-threshold 0.3 --contrastive-parseable-threshold 0.85 --contrastive-score-gap-min 0.0 --contrastive-score-gap-max 0.2" \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_value_suite.sh
```

> 注意：以上仅是“立即可跑”尝试，真正关键仍是 C1 标签与 pair 构造逻辑升级。

---

## 10. 阶段门槛（建议沿用仓库 gate）

在进入更深 BCR/ABR 前，建议至少满足：

1. `brier_score` 稳定优于 trivial baseline  
2. `pearson` 稳定提升并跨 split 保持  
3. `auc_clean_vs_corrupt >= 0.60`  
4. `pair_accuracy >= 0.55`  
5. 下游 utility（rerank/router）有净收益

---

## 11. 参考文献与链接

1. Training Verifiers to Solve Math Word Problems (2021)  
   https://arxiv.org/abs/2110.14168
2. Solving Math Word Problems with Process- and Outcome-Based Feedback (2022)  
   https://arxiv.org/abs/2211.14275
3. Let’s Verify Step by Step (2023)  
   https://arxiv.org/abs/2305.20050
4. Math-Shepherd (ACL 2024)  
   https://aclanthology.org/2024.acl-long.510/
5. Tree-PLV (EMNLP 2024)  
   https://aclanthology.org/2024.emnlp-main.125/
6. Rewarding Progress / PAV (ICLR 2025)  
   https://proceedings.iclr.cc/paper_files/paper/2025/hash/98711dea460bdefe0e651ca23ec98ba2-Abstract-Conference.html
7. ThinkPRM (2025)  
   https://arxiv.org/abs/2504.16828
8. The Lessons of Developing PRMs (2025)  
   https://arxiv.org/abs/2501.07301
9. ProcessBench (2025)  
   https://arxiv.org/abs/2412.06559
10. Do We Need to Verify Step by Step? (ICML 2025)  
    https://proceedings.mlr.press/v267/jia25f.html
11. STaR (2022)  
    https://arxiv.org/abs/2203.14465
12. Constitutional AI (2022)  
    https://arxiv.org/abs/2212.08073
13. Self-Rewarding Language Models (2024)  
    https://arxiv.org/abs/2401.10020

---

## 12. 一句话结论

当前价值头瓶颈不在“再加一个损失”，而在“先把 C1 标签和 pair 信号做干净”。  
外援可用，且应当作为 cold-start 数据引擎；创新点应保持在你们自己的价值建模与路由控制机制上。
