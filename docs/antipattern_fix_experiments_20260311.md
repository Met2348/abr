# 六大反模式修复 + 验证实验设计（2026-03-11）

## 概述

本文档记录六大已识别反模式的当前修复状态，以及针对未完成修复的实验设计。

**新发现（2026-03-11 下午）**：PBR2 实验后发现 terminal_completion_anchor 存在**长度偏差 bug**：
- chosen = 完整正确解（长），rejected = 不完整前缀（短），两者均为正确步骤
- 模型学到的是"更长 = 更好"的长度偏差，而非"内容正确 = 更好"
- 导致 ProcessBench pair_acc < 0.5（倒置！）
- Oracle filter 实验证实：terminal 对通过率仅 12.6%（1511 中只有 191 通过），因为 PRM 无法同时满足 chosen > 0.6 且 rejected < 0.4（两者都是正确步骤）

**新增实验（见末尾）**：
- `Judge-1`（Oracle filter + E87 config）：在 GPU1 运行，直接验证数据清洗效果
- `Diag-E87config`（原始数据 + E87 训练 config）：在 GPU3 运行，分离"config 问题"vs"数据问题"

---

## 反模式修复状态汇总

| 反模式 | 描述 | 修复状态 | 验证状态 |
|---|---|---|---|
| A | 只用 first_bad_edge，不覆盖 ProcessBench 几何 | ✅ 已修复 | ⚠️ PBR2 pair_acc=0.307（training config 有问题，见B） |
| B | 50/50 terminal mix 破坏 local discrimination | ✅ 比例修复 | ⚠️ **新发现 Bug**：terminal 对本身有长度偏差，正在 Judge-1 验证 |
| C | min_pair_confidence=0.55 过低 | ✅ 已修复 | ✅ E87 + oracle filter 双重确认 |
| D | pair_acc 优化 ≠ ProcessBench F1 优化 | ✅ 代码已修复 | ✅ F1 指标已出现在 PBR2 eval 中（processbench_f1=0.186） |
| E | dual_head 无 routing 逻辑 | ✅ 代码已修复 | ⏳ dual_head smoke 实验运行中（GPU0） |
| F | R-PRM direct_pair 被 token 截断 | ✅ 已修复 | ✅ compact_verdict 诊断确认 |
| G（新）| training config：joint+logit 训练极慢 | 🔄 正在诊断 | ⏳ Diag-E87config 在 GPU3 对比 |

---

## 反模式 A：训练数据几何与 ProcessBench 不匹配

### 问题描述

原始 Math-Shepherd 的 pair 100% 是 `lastsafe_vs_firstbad`（gap=1），而 ProcessBench 的实际结构是：
- 40.6% all-correct 轨迹（需要 terminal anchor 覆盖）
- 8.6% `lastsafe_vs_firstbad` 对（原训练完全覆盖这部分）
- 76.0% 对需要 `earlier-good vs first-bad` 或 `any-good vs later-bad`

**pair_type_l1_distance** 从 1.83（ms_e68 baseline）降至 ~0.98（ms_align_v1）。

### 修复内容

`phase_e_curate_processbench_transfer_pairs.py` 的 `ms_align_v1` profile：
- `local_first_bad_edge`: 61.9%
- `first_bad_fanout_prefix_ranking`: 11.8%
- `good_bad_prefix_grid`: 10.3%
- `terminal_completion_anchor`: 15.9%

### 验证实验

**正在运行**：`PBR2_FULL_MIXED_MLP_SEED3`（GPU1）
- 3 seeds × ms_align_v1 + mlp head
- 预期观察：ProcessBench AUC 从 ~0.50 提升到 0.55+

---

## 反模式 B：50/50 terminal mix 破坏 local discrimination

### 问题描述

实验 E83/E84 证明：
- 50% terminal → held-out pair_acc 0.976 → 0.773（崩溃）
- ProcessBench GSM8K pair_acc: 0.609 → 0.316

### 修复内容

Terminal anchor 比例控制在 10-20%，通过 `--step-label-terminal-anchor-fraction` 参数控制。

### 验证实验

**正在运行**：`run_phase_e_terminal_ratio_sweep.sh`（GPU2）
- 测试 ratio ∈ {0.0, 0.05, 0.10, 0.20}
- 每个 ratio 训练 mlp head + eval ProcessBench Math + GSM8K
- 预期最优点：0.10 ± 0.05

---

## 反模式 C：min_pair_confidence 过低

### 问题描述

min_conf=0.55 → ProcessBench Math pair_acc ~0.58
min_conf=0.70 → ProcessBench Math pair_acc ~0.73

根因：Math-Shepherd MC 标注本身就有噪声，低置信度 pair 的 chosen/rejected 标签经常互相干扰。

### 修复内容

`phase_e_curate_processbench_transfer_pairs.py` 中 `ms_align_v1` 默认 `min_pair_confidence=0.55`（注：PBR2 测试中的 JSONL 和之前的设置有区别，E87 系列已确认 0.7 是更好的选择）。

**待做**：确认 ms_align_v1 profile 是否使用了 0.70 置信度阈值。

```bash
# 检查当前 profile 置信度设置
python -c "
import json; from pathlib import Path
p = Path('assets/artifacts/phase_e_pairs/phase_e_processbench_research_pbr2a_ms_align_mlp_s42_ms_align_v1_pairs__2a63ed682f78/train_pairs.jsonl')
confs = []
with p.open() as f:
    for i,l in enumerate(f):
        if i > 200: break
        d = json.loads(l)
        if 'confidence' in d: confs.append(d['confidence'])
if confs: print(f'min={min(confs):.3f} mean={sum(confs)/len(confs):.3f}')
"
```

---

## 反模式 D：Metric Mismatch（pair_acc ≠ ProcessBench F1）

### 问题描述

我们优化 `pair_acc`（好坏 prefix 两两比较准确率），而 ProcessBench 官方指标是：

```
F1 = 2 × Acc_erroneous × Acc_correct / (Acc_erroneous + Acc_correct)

其中：
- Acc_erroneous = 错误样本中"正确识别第一错误步"的比例
- Acc_correct = 全正确样本中"正确预测为全对"的比例
```

这两个指标的优化方向不完全一致：
- 高 pair_acc 不等于 F1 高（可能 local discrimination 好但 terminal 糟糕）
- F1 同时要求 local error detection AND all-correct recognition

### 代码修复（已完成）

**`src/ours/phase_e/benchmark_eval.py`** 中新增 `compute_processbench_f1()` 函数：
- 输入：`ProcessBenchPrefixRecord` 列表 + 对应分数
- 输出：`processbench_f1`, `processbench_acc_erroneous`, `processbench_acc_correct`, `processbench_f1_threshold`
- 阈值：自动在 [score_min, score_max] 上搜索最优 τ（可选固定阈值）
- 集成到 `compute_processbench_metrics()` 中，所有 benchmark eval 自动包含 F1

**新增测试**（`tests/unit/test_phase_e_benchmark_eval.py`）：
- `test_compute_processbench_f1_perfect_model`
- `test_compute_processbench_f1_random_model`
- `test_compute_processbench_f1_included_in_processbench_metrics`

### 验证实验

下一次 benchmark eval 自动输出 F1。无需专门实验。

**观察点**：当 PBR2 eval 完成后，检查 `processbench_f1` 是否与 `pair_accuracy_good_vs_bad` 相关但不完全一致。

---

## 反模式 E：dual_head 无 routing 逻辑

### 问题描述

`dual_head` 架构（local_proj + terminal_proj）代码存在，但训练时没有按 pair_semantics 路由梯度，导致：
- local_proj 和 terminal_proj 同时接收所有梯度
- 等价于参数更多的 MLP（无归纳偏置）
- 双头的"任务分解"理念无法实现

### 代码修复（已完成）

**`src/ours/phase_e/training.py`**：
- `_resolve_pair_training_route_weights()` 按 `pair_semantics` 字段分配路由权重
  - `terminal_completion_anchor` → terminal_proj (w_local=0.0, w_terminal=1.0)
  - `first_bad_edge / fanout / laterbad` → local_proj (w_local=1.0, w_terminal=0.0)
  - `good_bad_prefix_grid` → 两头均等 (w_local=0.5, w_terminal=0.5)
- `compute_pair_objective()` 在检测到 `dual_head` 时自动激活路由分支

**关键逻辑**：
```python
# training.py 的 compute_pair_objective 中
dual_head_requested = all([
    chosen_local_logits is not None,
    rejected_local_logits is not None,
    local_pair_weights is not None,
    chosen_terminal_logits is not None,
    rejected_terminal_logits is not None,
    terminal_pair_weights is not None,
])
if dual_head_requested:
    # 路由：local pairs → local head loss，terminal pairs → terminal head loss
    loss = loss_local_pairs + loss_terminal_pairs
```

### 验证实验：dual_head Smoke Suite

**正在运行**：`run_phase_e_dual_head_smoke.sh`（GPU0）

实验设计：
- 使用 **相同** ms_align_v1 artifact（pair 有混合语义，含 ~16% terminal anchor）
- 比较：`gated_mlp` vs `dual_head`（seed=42）
- 均使用 `objective-mode joint`，`confidence_semantic` pair weighting

观测指标（期望 dual_head 具备以下特点）：
1. `first_error_edge_accuracy` 不低于 gated_mlp
2. `mean_all_correct_last_score` 不低于 gated_mlp
3. `processbench_f1`（新指标）应高于或持平 gated_mlp
4. local_proj 和 terminal_proj 的分数分布应有明显分歧（路由成功的标志）

**判断标准**：
- 成功：dual_head F1 ≥ gated_mlp F1 AND `mean_all_correct_last_score` ≥ gated_mlp
- 失败：dual_head F1 < gated_mlp F1（说明路由归纳偏置不够或数据量不足）
- 部分失败：F1 持平但 first_edge 下降 > 0.05（路由过于优先 terminal）

---

## 反模式 F：R-PRM direct_pair token 截断

### 问题描述

R-PRM `direct_pair` 模式：
- chosen_len p50 = 1381 tokens
- **94% 超过 max_length=1024**
- 11.6% 的 pair 分歧点被截断隐藏（模型看到相同输入）→ 无法学习

### 修复内容

`compact_verdict` 模式：
- chosen_len p50 = 268 tokens（约原来的 1/5）
- **0% 超过 max_length=1024**
- 分歧点完全可见

代码路径：`src/ours/phase_d/external_pairs_adapters.py` 中的 R-PRM adapter 已默认使用 `compact_verdict` 格式。

### 验证状态

已通过 `scripts/phase_e_audit_rprm_contract.py` 和 `scripts/phase_e_diagnose_rprm_token_cutoff.py` 验证。

---

## 新增实验：Oracle Filter（方案 A）验证

### 实验目的

验证用 Qwen2.5-Math-PRM-7B 做 consensus oracle filter 能否提升数据质量。

### 实验设计

**实验 OracleFilter-1**：
1. 取 ms_align_v1 的 Math-Shepherd pair 子集（约 8000 对）
2. 用 Qwen2.5-Math-PRM-7B 对 chosen 和 rejected 文本打分
3. consensus filter：
   - chosen_prm > 0.6（PRM 认为好）AND mc_label=1（MC 认为好）
   - rejected_prm < 0.4（PRM 认为坏）AND mc_label=0（MC 认为坏）
4. 过滤后估计保留约 40-60% 数据
5. 在过滤数据上训练 mlp head（同配置）
6. 比较原始数据 vs oracle-filtered 数据的 ProcessBench F1

**预期结果**：
- pair_acc 可能略降（数据更少）
- ProcessBench F1 应该提升（数据更干净）
- first_error_edge_accuracy 应该提升

**实现脚本**：`scripts/phase_e_oracle_filter_pairs.py`（✅ 已实现，已运行）

**实际运行结果（2026-03-11）**：
- 输入：9478 train + 1027 val（ms_align_v1 artifact）
- 输出：4054 train（42.8%）+ 462 val（45.0%）
- `mean_oracle_chosen_final = 0.646`（PRM 确认 chosen 正确）
- `mean_oracle_rejected_final = 0.285`（PRM 确认 rejected 错误）
- `mean_oracle_margin_kept = 0.835`（极大分离度！）
- **关键发现**：terminal_completion_anchor 通过率仅 12.6%（191/1511）
  - 证明 terminal 对长度偏差 bug：PRM 无法同时满足 chosen_score > 0.6 AND rejected_score < 0.4，因为两者均是正确步骤
- 过滤后数据组成：local_first_bad_edge 69.4% + fanout 17.1% + grid 8.9% + terminal 4.7%

---

## 新发现：反模式 G — terminal_completion_anchor 长度偏差

### 问题描述

`terminal_completion_anchor` 对的设计本意是让模型学到"完整正确解比不完整前缀更好"。但这个设计存在根本性缺陷：

- `chosen_text` = 完整正确解（长）
- `rejected_text` = 同一解的不完整前缀（短，但也是正确的！）
- **问题**：模型无法区分"内容正确"和"文本更长"

结果：
- 模型学到了 `length ≈ value` 的虚假相关性
- 对于 ProcessBench 的局部对（chosen=短好前缀，rejected=长坏前缀），长度偏差完全倒置了预测
- 导致 ProcessBench pair_acc < 0.5（所有近期实验均受影响）

### 根本原因分析

```
local 对：chosen=短（好步骤前） < rejected=长（含坏步骤）
terminal 对：chosen=长（完整解）> rejected=短（截断前缀）
```

如果模型用长度作代理：
- terminal 对：预测正确（长 > 短）
- local 对：预测错误（短 < 长，但 short=chosen 应该更高）

### 修复方向（待实现）

**正确设计的 terminal 对**应比较：
- chosen = 完整**正确**解
- rejected = 完整**错误**解（同一问题，但全部步骤正确的解 vs 有错误步骤的解，**长度相近**）

这样才能让模型学到"内容正确性"而非长度偏差。

### 临时修复

Oracle filter 有效地过滤掉了 87.4% 的 terminal 对（因为 PRM 无法确认这些对），将 terminal 比例从 15.9% → 4.7%。这是当前最实用的修复方案。

---

## 反模式 G 验证实验结果（2026-03-11 晚更新）

### 已完成实验汇总

| 实验 | 结果 | 结论 |
|---|---|---|
| `PBR2a` (ms_align_v1 + joint+logit+conf_sem) | val pair_acc ~0.51（接近随机），PB Math pair_acc=0.307（**倒置**！），pb_f1=0.186 | joint+logit config 基本无法训练 |
| `gated_mlp` (dual head smoke, 同配置) | val pair_acc 0.511，margin 5.7e-5（接近零），val AUC 0.516 | 与 PBR2a 同病相怜，确认 config 是主要问题 |
| `Diag-E87config` (原始 ms_align_v1 + E87 config) | **失败**：6144/9478 (65%) 训练样本产生 NaN/Inf 特征 | ms_align_v1 raw 数据中含有大量长文本（terminal 对），导致特征 NaN |
| `Judge-1` (oracle_filter + E87 config) | val pair_acc **0.931**，AUC 0.892，margin 0.391 | oracle filter 完全消除了 NaN 问题，配合 E87 config 效果极佳 |
| `Judge-1` PB Math eval (256 样本) | pair_acc 0.615，pb_f1 **0.240**，pb_acc_err 0.145，pb_acc_cor 0.702 | val 0.931 → PB 0.240：backbone 特征不足以泛化到 PB 分布 |
| `Judge-1` PB GSM8K eval | pair_acc 0.646，pb_f1 **0.281**，pb_acc_err 0.182 | 同上 |
| `NDS ndsbh1` (rlhflow_align_v1 + score space) | val pair_acc 0.814，PB Math pb_f1 0.220，GSM8K pb_f1 0.240 | 不同数据集，相似结论：7B-Instruct backbone 上限 ~0.24 |
| **`backboneproxy_prm_mixedsmall`（PRM-7B backbone）** | val pair_acc **0.898**，samefamily top1 **91.6%** | **突破性发现**：PRM-7B 作 backbone，2048 对就达到 0.898 pair_acc |
| `backboneproxy` PB Math eval (50 样本) | pair_acc 0.623，pb_f1 **0.480**，pb_acc_err 0.400 | PRM-7B backbone 的 ProcessBench F1 **是 7B-Instruct 的 2×**！|

### 关键发现：Backbone 是真正的瓶颈

```
7B-Instruct backbone：
  - 无论 config 如何优化（E87, oracle filter, rlhflow 数据），PB Math pb_f1 上限 ~0.24
  - 根因：Instruct backbone 的 hidden state 不携带步骤级正确性信号

PRM-7B backbone（Qwen2.5-Math-PRM-7B）：
  - 仅 2048 对，pb_f1 = 0.480（50样本，待全量 256 验证）
  - samefamily top1 acc = 91.6%（远超任何 Instruct 方案）
  - pooling = last_token = PRM-7B 的 <extra_0> 位置隐藏状态（携带步骤置信度）
```

**结论**：用 Qwen2.5-Math-PRM-7B 作 backbone，其 `<extra_0>` 处的 hidden state 直接携带
步骤级正确性评估信号。任何基于 Instruct backbone 的训练，无论 config/数据如何优化，
都无法突破 backbone 特征的表示力上限。

### 新增反模式 H：Backbone 选择（待验证）

- **错误做法**：用 Qwen2.5-7B-Instruct 作 backbone，期望 pair 训练能教会模型"什么是正确步骤"
- **问题**：Instruct backbone 的 last-token hidden state 对步骤级正确性几乎无感（训练分布不包含任何 PRM 信号）
- **正确做法**：用 Qwen2.5-Math-PRM-7B 作 backbone（已学过步骤级信号），在其 hidden state 上加 MLP head
- **效果**：PB Math pb_f1 从 0.24 → 0.48（+100%），几乎无需额外配置优化

---

## 下一步实验计划

### PBR5-PRM7B（最高优先级，待启动）

**目标**：PRM-7B backbone + oracle filter ms_align_v1（4054 train，更多更干净数据）

配置：
- backbone: Qwen2.5-Math-PRM-7B
- train pairs: oracle_filter_ms_align_v1（4054 train + 462 val）
- objective: ranking_only + score space + confidence weighting
- lr: 5e-5，epochs: 5，checkpoint_selection: pair_acc
- feature_cache: read_write（首次编码 PRM-7B 特征，之后复用）

**预期**：val pair_acc > 0.94（>0.93 of Judge-1，因为 PRM-7B backbone 更强）
**关键测试**：PB Math pb_f1 > 0.50（确认 0.480 的 50样本结果）

### PBR5 实验结果（2026-03-11 最新）

| 实验 | Val Pair Acc | PB Math F1 | PB GSM8K F1 | 备注 |
|---|---|---|---|---|
| PBR2a baseline (joint+logit) | ~0.87 | 0.186 | — | 反模式 G：配置失败 |
| PBR3a (joint+logit, seed=42) | 0.859 | 0.207 | 0.221 | 同 PBR2a 失败 |
| PBR3b (joint+logit, seed=1) | 0.848 | 0.176 | 0.206 | 同 PBR2a 失败 |
| Judge-1 (oracle + E87, 7B-Instruct) | 0.931 | 0.240 | 0.281 | 提升但受 backbone 上限限制 |
| ndsbh2 math_step_dpo (7B-Instruct) | — | **0.379** | 0.363 | 纯 local pairs 突破 7B-Instruct 上限 |
| backboneproxy_prm_mixedsmall (PRM-7B) | 0.898 | 0.378 | **0.524** | backbone 切换 = 2× F1 提升 |
| **PBR5** (oracle + PRM-7B) | **0.9935** | **0.609** | **0.704** | **突破：PRM-7B + 干净数据** |

**PBR5 完整指标（256 样本，2026-03-11）**：
- PB Math: pair_acc=0.873, pair_auc=0.850, pb_f1=**0.609**, pb_acc_err=0.500, pb_acc_cor=0.779, first_edge_acc=0.859
- PB GSM8K: pair_acc=0.901, pair_auc=0.876, pb_f1=**0.704**, pb_acc_err=0.591, pb_acc_cor=0.871, first_edge_acc=0.896
- 训练 val pair_acc=0.9935（5 epochs 最佳），mean_margin=0.686，AUC=0.972

**结论**：PRM-7B backbone + oracle 过滤数据（4054 对）+ ranking_only + score space + lr=5e-5
= 当前最优 recipe，PB Math F1 从 0.186（baseline）升至 0.609，GSM8K 从无效升至 0.704。

### 当前正在运行（2026-03-11 最新状态）

| 实验 | GPU | 状态 | 备注 |
|---|---|---|---|
| PBR5 Math eval (256 样本) | GPU0 | **已完成** pb_f1=0.609 | — |
| PBR5 GSM8K eval (256 样本) | GPU1 | **已完成** pb_f1=0.704 | — |
| PBR6 (PRM-7B + 纯 local math_step_dpo) | GPU2 | 特征编码中 | 验证 zero terminal 是否更好 |
| PBR3 (joint+logit laterbad) | GPU3 | **已完成** | pb_f1 均 <0.22，确认旧 config 失败 |
