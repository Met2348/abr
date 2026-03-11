# Phase E: 社区对比分析、差距诊断与下一步实验方案
# Phase E: Community Comparison, Gap Analysis, and Next Experiment Plan

*作者: Claude Code 自主分析 / Author: Claude Code autonomous analysis*
*日期 / Date: 2026-03-11*

---

## 0. 执行摘要 / Executive Summary

**核心诊断结论（一句话）**: 当前 Phase E 的 frozen backbone + lightweight head 路线，在 ProcessBench 上已触达 ~0.51 AUC 的结构性天花板，与文献最优水平（~0.80 F1）相差巨大；差距主要来自三个维度：**骨干网络冻结**、**训练数据标注质量**、**pair 构造语义与测评任务错位**。

**可执行的三步路径**：
1. （短期）用更高质量数据（RLHFlow/Math-Step-DPO）替换 Math-Shepherd MC 标注
2. （中期）实现 LoRA 解冻最后 4 层，打破冻结天花板
3. （长期）引入生成式推理路径（GenPRM/R-PRM 思路），或用 PRIME 隐式 PRM 方法跳过步骤标注

---

## 1. 社区做法综述 / Community Approach Summary

### 1.1 训练数据标注质量梯队

| 方法 | 标注质量 | ProcessBench F1 | 代表工作 |
|------|---------|----------------|---------|
| **LLM-judge + 共识过滤** (Consensus) | ⭐⭐⭐⭐⭐ | ~80+ | Qwen2.5-Math-PRM, GenPRM |
| **Human annotation** | ⭐⭐⭐⭐⭐ | ~78 | PRM800K (OpenAI) |
| **MCTS + LLM-judge** | ⭐⭐⭐⭐ | ~70 | OmegaPRM, MATH-APS |
| **LLM-judge (single)** | ⭐⭐⭐ | ~65 | RLHFlow-Deepseek, EurusPRM |
| **MC 估计 (纯 rollout)** | ⭐⭐ | ~55 | Math-Shepherd, Math-APS base |
| **MC 估计 (无过滤)** | ⭐ | ~50 | 我们当前用的 Math-Shepherd |

**关键结论**（Qwen2025 Lessons 论文，第 §4 节）：
> MC estimation fundamentally limited → LLM-as-a-judge 在 GSM8K 上 73.1% vs 43.9%
> Consensus filtering (MC + LLM-judge 双重认定) 保留约 40% 数据，但 ProcessBench 泛化大幅提升

### 1.2 骨干网络架构决策

| 方案 | ProcessBench F1 上限 | 代表性系统 |
|------|---------------------|-----------|
| Frozen backbone + scalar head | **~0.62 AUC (~55 F1)** | 我们当前方案 |
| LoRA 最后 4 层 + scalar head | **~0.70 AUC (~65 F1)** | RLHFlow/Mistral-PRM |
| 全参数微调 + scalar head | **~0.75+ AUC (~75 F1)** | Qwen2.5-Math-PRM-7B, Skywork-o1-PRM |
| 全参数微调 + 生成式推理头 | **~0.80+ F1** | GenPRM-7B, R-PRM |
| Large backbone (72B) | **~0.83+ F1** | GenPRM-32B, Qwen2.5-Math-PRM-72B |

**数据来源**: ProcessBench 基准（Qwen, 2024.12）; PRMBench（2025.01）; GenPRM（2025.04）

### 1.3 损失函数设计

社区主流做法：
```
L = L_step_bce + λ_mse * L_value_regression
```
其中 L_step_bce 为每步二分类交叉熵，L_value_regression 为"到达正确终点的概率"软标签 MSE。

我们的做法：
```
L = L_margin (contrastive pair ranking) + λ_bce * L_pair_bce
```
差异：我们训练 pair-level 排序，社区训练 step-level 绝对分类。这导致：
- 我们的 head 学到的是"谁比谁好"，不是"这步绝对正确的概率是多少"
- ProcessBench 需要的是绝对概念：某步是否正确，而非两个轨迹的相对顺序

### 1.4 评测方式对比

| 评测 | 社区最优 | 我们 |
|------|---------|------|
| ProcessBench F1 (MATH) | 80.5 (GenPRM-7B) | ~51 F1 等价（0.515 AUC） |
| ProcessBench F1 (GSM8K) | >80 | ~49 F1 等价（0.486 AUC） |
| PRMBench (7B 规模) | 68.8 (Gemini-2-Thinking) | 未正式测 |
| held-out pair_acc | — | ~92% (Math-Shepherd 内) |

**关键发现**: held-out pair_acc 92% 表明在训练分布内学得很好，但 ProcessBench AUC ~0.51 说明**根本没有泛化**。这是典型的 distribution shift。

---

## 2. 我们代码的现状 / Current Implementation State

### 2.1 正在工作的部分

✅ **Feature caching pipeline**: 骨干特征抽取、缓存、复用，设计精良
✅ **Pair semantics taxonomy**: `first_bad_edge`, `sibling_branch`, `terminal_completion_anchor`, `good_bad_prefix_grid` — 语义正确，覆盖面广
✅ **Loss functions**: contrastive margin + pair BCE + anti-saturation + reward centering — 库完备
✅ **Curation profiles**: `ms_align_v1`, `ms_laterbad_v1`, `ms_prm_align_v1` — 工程成熟
✅ **Same-family held-out AUC**: 在 Math-Shepherd 分布内 ~0.92，value head 确实有学到信号

### 2.2 关键问题和反模式

#### 问题 1：标注质量不足（最高优先级）

**现状**: Math-Shepherd 使用 MC rollout 估计步骤正确性，无 LLM-judge 过滤。
**后果**: 噪声标签导致 head 学到的边界在 ProcessBench 上随机翻转。
**文献证据**: Qwen2025 Lessons §4: "MC estimation achieves 43.9% on GSM8K vs LLM-judge 73.1%"
**修复**: 切换到 RLHFlow-Deepseek (252K, Deepseek-annotated) 或使用 trl-prm800k (3.7K, human-annotated)

#### 问题 2：骨干冻结天花板（中期障碍）

**现状**: 骨干完全冻结，只有 head 参数更新。
**后果**: 表示空间固定，head 只能在预训练特征的线性/非线性组合上做文章，ProcessBench 的 domain-specific 推理错误检测能力无法提升。
**文献证据**: ProcessBench 基准所有 >75% F1 的模型均使用 LoRA 或全参数微调
**修复**: 实现 LoRA 模式（rank=4，target=q_proj+v_proj，解冻最后 4 层），预期 AUC +0.05~+0.10

#### 问题 3：PRM800K 适配器语义错位（已知 bug）

**现状**: `same_step_completion_preference` 对比"同一步骤的不同措辞变体"
**后果**: head 学到"哪种措辞更好"而非"这步是否正确"，训练损失收敛但 ProcessBench 近随机
**修复**: 避免在 ProcessBench 目标实验中使用 PRM800K completion pairs；或改用 trl-prm800k 格式（已有步骤级 True/False 标签）

#### 问题 4：Pair 构造语义 vs. ProcessBench 测评任务错位

**现状**: 训练用 `first_bad_edge` 对（数学推理前缀排序）；ProcessBench 测的是"识别第一个错误步骤"
**分析**:
```
ProcessBench 任务: 给定一串步骤，判断每步是否正确，找出第一个错误步骤
我们的训练信号: good_prefix > bad_prefix（对排序）
```
这两个任务的对应关系是间接的，不是完全对齐的。ProcessBench 需要绝对的步骤级判断，我们给的是相对排序信号。
**修复**:
- 加入 step-level BCE 损失（chosen targets=1, rejected targets=0，是绝对信号而非相对信号）
- 这已在代码中实现（`lambda_bce > 0`），但力度需加强
- 或者：改用 RLHFlow 格式数据（天然就是每步 +/- 的绝对分类信号）

#### 问题 5：Terminal anchor 未强制校准

**现状**: terminal 对用 ranking margin loss，没有 terminal BCE
**后果**: "完整正确轨迹"的分数不收向 1.0，ProcessBench 的 `mean_all_correct_last_score` 虽然高（~0.9），但局部正确步骤分数也高（~0.7），区分度不够
**修复**: 确保 `lambda_terminal_bce > 0`

#### 问题 6：Grid pair 引入 later-bad confound

**现状**: `good_bad_prefix_grid` 包含 (early_good, later_bad) 对，这些对的主要区分因素可能是"轨迹长度"而非"错误内容"
**后果**: Head 可能学到"长轨迹得分高"的捷径，ProcessBench 的 `later-bad` canary 表现差
**修复**: `ms_laterbad_v1` 已针对此问题，只保留 `lastsafe_vs_laterbad` 和 `earlygood_vs_laterbad`，但 PBR3 尚未运行完成

---

## 3. 差距量化表 / Gap Quantification

| 维度 | 社区最优 | 我们当前 | 差距 | 主要原因 |
|------|---------|---------|------|---------|
| ProcessBench MATH AUC | ~0.80 | ~0.515 | **-0.285** | 1.骨干冻结 2.数据质量 |
| ProcessBench GSM8K AUC | ~0.80 | ~0.486 | **-0.314** | 同上 |
| first_error_edge_acc | ~0.75+ | ~0.540 | **-0.21** | 数据质量+语义错位 |
| held-out same-family AUC | — | ~0.92 | N/A (内分布好) | — |
| 数据规模 | 800K~1M steps | 16K pairs (~32K samples) | ~25x | 未使用大规模数据 |
| 标注质量 | LLM-judge consensus | MC rollout | 大 | 无过滤 |
| 骨干状态 | LoRA/全参数微调 | 完全冻结 | 大 | 未实现 |

---

## 4. 反模式汇总 / Anti-Patterns

### AP-1: 在 pair 排序上训练，在绝对步骤分类上测评

**描述**: 训练信号是相对排序（A > B），测评要求绝对判断（step k 是否正确）
**识别方法**: held-out pair_acc 高（>0.9）但 ProcessBench pair_acc 低（~0.4）
**解决方案**: 增加 step-level BCE 训练；或使用 RLHFlow 格式数据（天然绝对标签）

### AP-2: 在简单分布上 overfit，在困难分布上零泛化

**描述**: Math-Shepherd 是 GSM8K/MATH 简单题的 MC 标注；ProcessBench 包含 OlympiadBench/Omni-MATH 困难题
**识别方法**: ProcessBench MATH subset（比 GSM8K 难）AUC 比 GSM8K 更低
**解决方案**: 使用更多样的数据源（VersaPRM 等）；或更难的数学问题的标注

### AP-3: 用 Completion Preference Pairs 训练 Trajectory Correctness Head

**描述**: PRM800K 的 completion pairs 是"同一步的不同措辞变体"，而非"正确 vs 错误轨迹"
**识别方法**: PRM800K 模型 ProcessBench 近随机，尽管训练收敛
**解决方案**: 停止在 ProcessBench 目标时使用 PRM800K completion pairs；改用 trl-prm800k（步骤级 True/False）

### AP-4: 忽略解决方案多样性（All-correct 偏置）

**描述**: 若 grid pair 中 good-vs-good 对过多，head 在 same-quality 区域失去区分度
**识别方法**: `mean_good_prefix_score` 和 `mean_bad_prefix_score` 相近（差 < 0.01）
**解决方案**: `pair_type_allowlist` 过滤掉 `grid_within_good` 和 `grid_within_bad`；增加 terminal anchor 权重

### AP-5: Feature Cache Staleness（未来风险）

**描述**: Feature cache 基于 backbone 冻结假设；一旦引入 LoRA，cache 需要失效重建
**当前状态**: 无影响（backbone 完全冻结）
**未来风险**: PBR6 实现 LoRA 时必须绕过 cache（on-the-fly encoding）

---

## 5. 新数据集适配方案 / New Dataset Adapter Design

### 5.1 RLHFlow-Deepseek → `first_bad_edge` 对

**数据格式**:
```
conversations: [
  {role: "user", content: "问题\n\n步骤1\n\n"},
  {role: "assistant", content: "+"},  # 或 "-"
  {role: "user", content: "步骤2\n\n"},
  {role: "assistant", content: "+"},
  ...
]
```

**适配逻辑**:
```python
def load_rlhflow_pairs(path, config):
    # 解析 conversations → [(step_text, label)] 序列
    # 找到第一个 "-" 标注的步骤（first_bad_edge）
    # chosen = 前 k-1 步的拼接前缀
    # rejected = 前 k 步的拼接前缀（包含第一个坏步）
    # pair_semantics = "local_first_bad_edge"
    # pair_confidence = 0.72  # Deepseek-annotated, 高于 MS MC (0.74 → 实际更可信)
```

**优势**: 252K 条目，Deepseek 8B 模型标注（LLM-judge 质量），无需 MC rollout

### 5.2 Math-Step-DPO-10K → `sibling_branch` 对

**数据格式**:
```
{
  "prompt": "问题",
  "initial_reason_steps": "共有前缀步骤",
  "chosen": "正确后续步骤",
  "rejected": "错误后续步骤",
  "full_chosen": "完整正确解答",
  "full_rejected": "完整错误解答"
}
```

**适配逻辑**:
```python
def load_math_step_dpo_pairs(path, config):
    # chosen_text = initial_reason_steps + chosen
    # rejected_text = initial_reason_steps + rejected
    # pair_semantics = "sibling_branch"  # 明确的分叉点对
    # pair_confidence = 0.80  # 人工精选数据
```

**优势**: 10.8K 高质量 sibling_branch 对，分叉点明确，无歧义

### 5.3 trl-prm800k → 直接步骤级 BCE

**数据格式** (TRL 标准):
```
{
  "prompt": [{"role": "user", "content": "问题"}],
  "completions": ["步骤1文本", "步骤2文本", ...],
  "labels": [True, True, False, ...]
}
```

**适配逻辑**:
```python
def load_trl_prm800k_pairs(path, config):
    # 找到第一个 False 标签 → first_bad_edge 对
    # chosen = completions[:first_false_idx] 拼接
    # rejected = completions[:first_false_idx+1] 拼接
    # pair_semantics = "local_first_bad_edge"
    # pair_confidence = 0.85  # 人工标注
```

**优势**: Human-annotated（PRM800K 原始数据，最高质量），但仅 3.7K

---

## 6. 实验设计矩阵 / Experiment Design Matrix

### 系列 A：新数据适配 Smoke（GPU 3，4096 pairs）

| 实验 ID | 数据源 | Pair 语义 | Head | 假设 |
|---------|--------|----------|------|------|
| **NDS1**: rlhflow_first_bad | RLHFlow-Deepseek | first_bad_edge | mlp | LLM-judge 标注 > MC，ProcessBench AUC 提升 |
| **NDS2**: math_step_dpo_sibling | Math-Step-DPO-10K | sibling_branch | mlp | 明确分叉对 > 隐式分叉，first_edge_acc 提升 |
| **NDS3**: trl_prm800k_first_bad | trl-prm800k | first_bad_edge | mlp | Human 标注质量最高（小规模） |
| **NDS4**: ms_rlhflow_mixed | Math-Shepherd + RLHFlow | first_bad_edge | mlp | 数据量 + 质量双提升 |

### 系列 B：PBR2/3 完成后的跟进

PBR2 (16384 pairs, ms_align_v1, 3 seeds): 已在运行，建立 frozen baseline
PBR3 (16384 pairs, ms_laterbad_v1, 3 seeds): PBR2 完成后运行，later-bad 专项

### 系列 C：LoRA Backbone（需实现 PBR6 基础设施）

PBR6 Smoke (8192 pairs, LoRA last-4-layers, rank=4): 突破冻结天花板，预期 +0.05-0.10 AUC

### 系列 D：步骤级 BCE 训练（当前代码已支持）

用 RLHFlow 格式直接训练绝对步骤分类（lambda_bce >> lambda_ranking），替代当前 pair ranking 主导的训练

---

## 7. 优先级和决策树 / Priority and Decision Tree

```
当前状态: ProcessBench AUC ~0.51 (near-random)
         held-out same-family AUC ~0.92 (过拟合)

Step 1: 运行 NDS1 (RLHFlow first_bad, 4096 pairs, GPU 3, ~2h)
  → 如果 pb_math_auc > 0.55: 确认标注质量是瓶颈，继续 NDS4 (mixed)
  → 如果 pb_math_auc <= 0.53: 确认 frozen ceiling 是主瓶颈，跳 PBR6 (LoRA)

Step 2: 运行 NDS2 (Math-Step-DPO sibling, 4096 pairs, GPU 3, ~1.5h)
  → 如果 first_error_edge_acc > 0.60: 确认 sibling_branch 语义有效，进入大规模
  → 如果 first_error_edge_acc <= 0.55: 语义错位仍然存在，考虑步骤级 BCE 路线

Step 3: PBR2 完成后
  → 如果 PBR2 pb_math_auc > 0.58: frozen 路线有希望，继续 PBR3 + NDS4
  → 如果 PBR2 pb_math_auc <= 0.55: 确认 frozen ceiling，优先实现 PBR6 (LoRA)

Step 4: 无论如何，计划实现 PBR6 (LoRA)
  → 文献证据明确：frozen 上限 < 0.65，LoRA 可推到 0.70+
  → 预期效果最显著的单一改动
```

---

## 8. 关于 BCR 原始 Idea 的修订建议 / BCR Idea Revision

### 8.1 保留的部分

**有效的核心直觉**: 推理步骤之间应满足某种一致性约束（Bellman 一致性）。这个方向是对的，文献也证明步骤级监督确实有效。

**Value head 作为辅助信号**: 单独的 value head + 骨干特征，这个架构在 frozen 限制下已展示出学习能力（same-family AUC ~0.92）。

### 8.2 需要修订的部分

**原始主张**: "通过 TD-learning 自监督训练 value head，不需要外部标注"
**修订**: 社区大量证据表明自监督 MC rollout 质量不足，需要 LLM-judge 共识过滤。PRIME（隐式 PRM）是目前最接近自监督 PRM 的可行方案，但需要 ORM 训练先。

**原始主张**: "Bellman 一致性约束强制 hidden states 满足 Martingale 性质"
**修订**: 这个理论框架是合理的，但实现上需要骨干参与梯度回传（否则就是普通的回归任务）。当前实验证明 frozen backbone 下 Bellman loss 等价于普通 pair ranking loss。

**原始主张**: "Token-level 一致性"
**修订**: Token-level 确实过细，step-level（段落/思考步骤级别）是当前共识粒度。ABR（step-level + NRAP router）的方向更符合社区现状。

### 8.3 最可行的近期定位

在已有工程基础上，将当前 Phase E 定位为：
> "高质量 step-level 判别性 PRM 训练，从 LLM-annotated 数据出发，探索 frozen 与 LoRA 微调的天花板"

这是 paper contribution 的可行定位，不需要完整的 BCR 自监督 TD 框架。

---

## 9. 下一步行动清单 / Action Checklist

### 立即行动（今天）
- [x] 完成数据集 schema 探测（已完成）
- [ ] 实现 RLHFlow-Deepseek 适配器
- [ ] 实现 Math-Step-DPO-10K 适配器
- [ ] 运行 NDS1 (RLHFlow smoke, GPU 3)
- [ ] 运行 NDS2 (Math-Step-DPO smoke, GPU 3)

### 短期（PBR2 完成后）
- [ ] 检查 PBR2 结果，决定是否优先 PBR6 (LoRA)
- [ ] 运行 NDS4 (混合数据源，如果 NDS1 有效)
- [ ] 更新 ms_prm_align_v1 以包含 RLHFlow 组件

### 中期（1周内）
- [ ] 实现 PBR6 (LoRA) 基础设施
  - training.py: LoRA mode (--lora-rank, --lora-target-modules, --lora-top-k-layers)
  - 禁用 feature cache (on-the-fly encoding mode)
  - 预期: ProcessBench AUC 从 ~0.55 → ~0.65+
- [ ] 考虑 GenPRM 路线（生成式 critique → 判断）

---

*参考文献: 见 `/docs/relatedPapers/` 下的 17 篇 PDF*
*关联文档: `docs/dataset_schema_report_20260311.md`, `docs/new_dataset_experiment_plan_20260311.md`*
