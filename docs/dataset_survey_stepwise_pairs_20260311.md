# 可用 Step-Level Good/Bad Pair 数据集调研

**日期：** 2026-03-11
**目的：** 系统整理现有公开可下载的数学/通用领域步骤级好坏对数据集，解决当前依赖 MC rollout 自构建训练数据的成本问题。

---

## 0. Executive Summary（先读这里）

**结论：无需从头做 MC rollout 构造 pair，社区已有多个高质量可用数据集。**

| 优先级 | 数据集 | HuggingFace 链接 | 规模 | 标注质量 | 关联价值 |
|---|---|---|---|---|---|
| ★★★ | **EurusPRM-Stage2-Data** | [PRIME-RL/EurusPRM-Stage2-Data](https://huggingface.co/datasets/PRIME-RL/EurusPRM-Stage2-Data) | 未公开 | LLM-judge 注错 | 质量 > MC；最接近 ProcessBench 要求 |
| ★★★ | **MATH-APS (OmegaPRM)** | [openreasoner/MATH-APS](https://huggingface.co/datasets/openreasoner/MATH-APS) | ~1.5M 步骤标注 | MCTS 二分搜索 | 正负均衡；later-bad 识别精准 |
| ★★★ | **PRM800K** | [trl-lib/prm800k](https://huggingface.co/datasets/trl-lib/prm800k) | 800K 步骤标注 | **人工标注** | 最高质量；MIT license；已有 adapter 但当前 near-random |
| ★★ | **GenPRM-MATH-Data** | [GenPRM/GenPRM-MATH-Data](https://huggingface.co/datasets/GenPRM/GenPRM-MATH-Data) | 23K 实例 | MC + LLM-judge 共识过滤 | 小而精；训出来的模型超 Qwen2.5-Math-PRM-72B |
| ★★ | **RLHFlow/Deepseek-PRM-Data** | [RLHFlow/Deepseek-PRM-Data](https://huggingface.co/datasets/RLHFlow/Deepseek-PRM-Data) | 253K | MC（更强 source model）| Math-Shepherd 补充；更强的 source model |
| ★★ | **Math-Step-DPO-10K** | [xinlai/Math-Step-DPO-10K](https://huggingface.co/datasets/xinlai/Math-Step-DPO-10K) | 10K | GPT-4 识别 first error | 最干净的 chosen/rejected 格式；直接可用于 discriminative head |
| ★ | **UltraInteract_pair** | [openbmb/UltraInteract_pair](https://huggingface.co/datasets/openbmb/UltraInteract_pair) | 219K | 轨迹树正确性 | 数学+代码+逻辑推理；MIT license |
| ★ | **MMLU-Pro-CoT-Train-Labeled** | [UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled](https://huggingface.co/datasets/UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled) | ~84K | Llama-70B-judge | 14 个非数学领域；VersaPRM 训练数据 |

---

## 1. 数学领域核心数据集

### 1.1 PRM800K（OpenAI，最高质量基准）

- **HuggingFace（TRL 格式）**：[trl-lib/prm800k](https://huggingface.co/datasets/trl-lib/prm800k)
- **原始 GitHub**：[github.com/openai/prm800k](https://github.com/openai/prm800k)
- **规模**：800K 步骤级标注，覆盖约 75K 个解法（约 12K 道 MATH 竞赛题）
- **标注方式**：**人工标注**，每步打 +1（正确）/ 0（中性，通常按正确处理）/ -1（错误）
- **格式**：TRL 版本有 `prompt`、`completions`（步骤列表）、`labels`（bool 列表），直接兼容 `PRMTrainer`
- **License**：MIT
- **Domain**：MATH 数据集（竞赛难度）
- **本项目现状**：已有 adapter（`PRM800K`），但当前给出"near-random"结果（Phase E 记录）
- **为什么 near-random**：当前 adapter 可能没有正确对齐 PRM800K 的标注格式（+1/0/-1 vs. bool，或对 neutral 步骤的处理方式）；adapter 本身需要 debug，而不是数据本身无效
- **对本项目的价值**：PRM800K 是 ProcessBench >75% 所有已发布 PRM 的通用训练数据；修复 adapter 后可直接提供人工质量的 first_bad 监督

---

### 1.2 MATH-APS / OmegaPRM（OmegaPRM 风格 MCTS 正负均衡）

- **HuggingFace**：[openreasoner/MATH-APS](https://huggingface.co/datasets/openreasoner/MATH-APS)
- **规模**：~1.5M+ 步骤级标注（OmegaPRM 从 MATH 数据集生成）
- **标注方式**：Divide-and-Conquer MCTS + 二分搜索定位第一个错误步；显式均衡正负样本
- **格式**：步骤级 process supervision labels
- **License**：未明确
- **Domain**：MATH 数据集（竞赛难度）
- **对本项目的核心价值**：
  - 当前项目 later-bad 识别弱，根本原因是训练只有 `lastsafe_vs_firstbad` 的单步边界
  - OmegaPRM 的 MCTS 构造的 pair 包含"同一 prefix 的不同 continuation"——这正是 sibling-branch pair，能直接教会 later-bad 识别
  - 无需自己跑 MC rollout，数据已经在 HF 上
- **注意**：格式需要验证是否直接兼容当前 `phase_e_pairs` 的 schema；可能需要新 adapter

---

### 1.3 EurusPRM-Stage2-Data（PRIME 框架，LLM-judge 注错）

- **HuggingFace**：[PRIME-RL/EurusPRM-Stage2-Data](https://huggingface.co/datasets/PRIME-RL/EurusPRM-Stage2-Data)
- **相关**：[PRIME-RL/EurusPRM-Stage1-Data](https://huggingface.co/datasets/PRIME-RL/EurusPRM-Stage1-Data)（response-level ORM 数据）
- **规模**：未公开精确数量
- **标注方式**：Llama-3.1-70B-Instruct + Qwen2.5-72B-Instruct **向正确解法注入步骤级错误**，构造 step-level good/bad pair；额外混入 response-level 数据
- **格式**：步骤级 partial pairs，使用 "Step K:" 格式（需要对应的格式化方式）
- **License**：未明确
- **Domain**：数学推理（UltraInteract 风格问题）
- **对本项目的核心价值**：
  - LLM-judge 注错比 MC 估计**质量更高**（Qwen 2025 Lessons 论文证明）
  - Stage 2 中的步骤错误注入正好对应"first_bad_edge"语义，但质量高于 Math-Shepherd 的 MC 标注
  - 如果当前 ProcessBench 弱主要因为 Math-Shepherd MC 标注噪声，换 EurusPRM-Stage2 可能直接改善

---

### 1.4 GenPRM-MATH-Data（小而精，MC + 共识过滤）

- **HuggingFace**：[GenPRM/GenPRM-MATH-Data](https://huggingface.co/datasets/GenPRM/GenPRM-MATH-Data)
- **GitHub**：[github.com/RyanLiu112/GenPRM](https://github.com/RyanLiu112/GenPRM)
- **规模**：23K 训练实例
- **标注方式**：MC 估计 + LLM-as-judge **共识过滤**（两者一致才保留）；每个实例附带步骤验证 CoT 推理链
- **格式**：每题包含 problem + 步骤分段解法 + 步骤级正确性判断（含 CoT 解释）
- **License**：Apache 2.0（AAAI 2026）
- **Domain**：MATH 数据集
- **对本项目的价值**：
  - 仅 23K 实例，但 GenPRM-7B 在 ProcessBench 上超越 Qwen2.5-Math-PRM-72B
  - 这证明：数据质量（MC+共识过滤）> 数据数量
  - VersaPRM 的 self-filtering 思路（Phase 3 in survey）与此完全吻合

---

### 1.5 Math-Step-DPO-10K（最干净的 pair 格式）

- **HuggingFace**：[xinlai/Math-Step-DPO-10K](https://huggingface.co/datasets/xinlai/Math-Step-DPO-10K)
- **规模**：10K step-level preference pairs
- **标注方式**：GPT-4 定位 first incorrect step，构造 (prefix + correct_step, prefix + incorrect_step) 三元组
- **格式**：标准 DPO 格式（`prompt`, `chosen`, `rejected`），chosen/rejected 在 first error 处分叉
- **License**：Apache 2.0
- **Domain**：GSM8K + MATH
- **对本项目的价值**：
  - 这是最接近当前项目 `first_bad_edge` pair 语义的公开数据集
  - 格式和当前 pair schema 最接近，适配成本最低
  - 数量小（10K），但质量高（GPT-4 标注）
  - 适合作为**数据质量 ablation**：将 Math-Step-DPO-10K 与 Math-Shepherd 同规模对比，验证标注质量的影响

---

### 1.6 RLHFlow PRM Data（额外的 MC 覆盖）

- **Mistral variant**：[RLHFlow/Mistral-PRM-Data](https://huggingface.co/datasets/RLHFlow/Mistral-PRM-Data)（273K）
- **Deepseek variant**：[RLHFlow/Deepseek-PRM-Data](https://huggingface.co/datasets/RLHFlow/Deepseek-PRM-Data)（253K）
- **Collection**：[RLHFlow MATH PRM collection](https://huggingface.co/collections/RLHFlow/rlhflow-math-process-reward-model-6725a42fc8808e12aa1cb144)
- **标注方式**：与 Math-Shepherd 相同的 MC hard estimation（16 次 rollout），但 source model 更强（Mistral-MetaMath / deepseek-math-7b-instruct）
- **格式**：与 Math-Shepherd TRL 格式相同（`prompt`, `completions`, `labels`）
- **Domain**：MATH + GSM8K
- **对本项目的价值**：作为 Math-Shepherd 的补充或替代，source model 更强可能降低 label noise

---

## 2. 非数学领域数据集

### 2.1 UltraInteract_pair（数学 + 代码 + 逻辑推理，最全面）

- **HuggingFace**：[openbmb/UltraInteract_pair](https://huggingface.co/datasets/openbmb/UltraInteract_pair)
- **规模**：219K correct/incorrect action pairs，86K 条 instruction
- **标注方式**：preference tree 结构——每个推理步骤节点，有一个 correct action 和一个 incorrect action 的配对
- **Format**：multi-turn 轨迹，每个树节点包含"当前状态 → 正确继续 vs 错误继续"
- **License**：MIT
- **Domain**：**数学 + 代码 + 逻辑推理**（包含 HotpotQA、FOLIO、StrategyQA、TheoremQA 等）
- **对本项目的特殊价值**：
  - 包含逻辑推理领域的 step-level pair，这在其他数据集中极为罕见
  - 如果未来 BCR/ABR 想 claim 跨域泛化（原始论文的 StrategyQA 目标），UltraInteract_pair 是唯一可用的 step-level 逻辑推理训练数据
  - 树形结构天然包含 terminal completion（根到叶 = 完整正确解法）和 sibling-branch pairs

---

### 2.2 MMLU-Pro-CoT-Train-Labeled（VersaPRM，14 个非数学领域）

- **HuggingFace**：[UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled](https://huggingface.co/datasets/UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled)
- **GitHub**：[github.com/UW-Madison-Lee-Lab/VersaPRM](https://github.com/UW-Madison-Lee-Lab/VersaPRM)
- **规模**：~84K CoT 解法（5,750 道 MMLU-Pro 题 × 多条解法）
- **标注方式**：Llama-3.1-70B-Instruct 对每步标注 correct/incorrect；solution-level 标记 first error 位置
- **Format**：含步骤级标注的解法，标明第一个错误步骤
- **License**：Apache 2.0（ICML 2025）
- **Domain**：Law、Philosophy、Biology、Chemistry、History、Math、CS 等 14 个 MMLU-Pro 领域
- **对本项目的价值**：
  - 目前**唯一**覆盖非数学学科步骤级标注的大规模公开数据集
  - VersaPRM 论文证明：训练在此数据上的 PRM 在 Law 领域超过数学专用 PRM +7.9%
  - 如果项目需要在 BCR 论文中 claim 跨领域泛化，这是数据基础

---

## 3. 与本项目当前状态的对应分析

### 3.1 当前 bottleneck 和对应数据

| 当前 bottleneck | 根本原因 | 最相关数据集 | 预期效果 |
|---|---|---|---|
| ProcessBench AUC 0.62 | 冻结 backbone + MC 标注噪声 | EurusPRM-Stage2-Data | LLM-judge 质量 → AUC ↑ |
| later-bad pair 识别弱 | 只有 single-trajectory first-bad-edge | MATH-APS (OmegaPRM) | sibling-branch 语义 → later-bad ↑ |
| PRM800K adapter near-random | adapter 格式对齐问题 | PRM800K (debug adapter) | 修复后 human-quality labels → 所有指标 ↑ |
| terminal completion 弱 | 无 complete-correct-solution > truncated 监督 | Math-Step-DPO-10K + UltraInteract | terminal 信号补充 |
| 跨域泛化（未来 StrategyQA 目标）| 只训数学数据 | UltraInteract_pair / MMLU-Pro | 逻辑推理 step-label 数据 |

---

### 3.2 推荐的数据 adapter 实施顺序

**Priority 1（无需新数据构造，直接从 HF 下载）：**

1. **Debug PRM800K adapter**
   - 当前 adapter 给 near-random，可能是格式问题（+1/0/-1 vs bool，neutral 步骤处理）
   - 参考 `trl-lib/prm800k` 的格式规范，对齐 `positive_step_index` 提取逻辑
   - 这是最高 ROI 的 fix：人工质量标注，MIT license，已有代码框架

2. **MATH-APS adapter**（解决 later-bad 问题）
   - 下载 `openreasoner/MATH-APS`
   - 在 `external_pairs_adapters.py` 中新增 `MathAPSAdapter`
   - 核心：将 MCTS 生成的步骤质量序列转化为 `first_bad_edge` 和 `sibling_branch` pairs
   - 预期效果：OmegaPRM 的正负均衡设计直接命中 later-bad 问题

**Priority 2（需了解格式细节）：**

3. **EurusPRM-Stage2 adapter**（高质量 first_bad 替换 Math-Shepherd）
   - 格式需要 "Step K:" 分段，可能与当前 Math-Shepherd 格式有差异
   - 如果质量确实高于 MC，可部分替换 Math-Shepherd

4. **Math-Step-DPO-10K adapter**（最干净的 ablation 基准）
   - Apache 2.0，格式最简单（标准 DPO format）
   - 10K 规模，适合快速 ablation：数据质量 vs 数据数量

**Priority 3（跨域泛化，非近期目标）：**

5. **UltraInteract_pair adapter**
   - 逻辑推理 step-level 数据
   - 未来 BCR 跨域 claim 的基础

---

## 4. 标注质量层级（基于 Qwen 2025 Lessons 论文）

从高到低：

```
1. 人工标注（PRM800K）                     → ProcessBench 最强
2. LLM-judge 注错（EurusPRM-Stage2）       → 高于 MC
3. MC + LLM-judge 共识过滤（GenPRM-MATH）  → 中高
4. MCTS 二分搜索（MATH-APS/OmegaPRM）      → 中（正负均衡更好）
5. MC hard estimation（Math-Shepherd、     → 中（噪声较高，量大）
   RLHFlow variants）
6. 自构 MC rollout（当前 Phase E 原始方案）→ 与 Math-Shepherd 等价，但需要自己跑
```

**核心结论**：当前 Phase E 依赖的 Math-Shepherd 已经是社区预构建好的数据集，本身不需要自己跑 MC rollout。问题不是"没有数据"，而是"现有 adapter 没有充分利用高质量数据源"。

---

## 5. 立即可执行的行动项

```bash
# 验证 PRM800K TRL 格式可访问
python -c "from datasets import load_dataset; d = load_dataset('trl-lib/prm800k'); print(d['train'][0])"

# 验证 MATH-APS 可访问
python -c "from datasets import load_dataset; d = load_dataset('openreasoner/MATH-APS'); print(d['train'][0])"

# 验证 EurusPRM-Stage2 可访问
python -c "from datasets import load_dataset; d = load_dataset('PRIME-RL/EurusPRM-Stage2-Data'); print(d['train'][0])"

# 验证 Math-Step-DPO-10K
python -c "from datasets import load_dataset; d = load_dataset('xinlai/Math-Step-DPO-10K'); print(d['train'][0])"

# 验证 UltraInteract_pair
python -c "from datasets import load_dataset; d = load_dataset('openbmb/UltraInteract_pair'); print(d['train'][0])"
```

---

## 6. 参考文献

- [PRM800K: Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) | [GitHub](https://github.com/openai/prm800k)
- [Math-Shepherd: Verify and Reinforce LLMs Step-by-Step](https://arxiv.org/abs/2312.08935)
- [OmegaPRM: Improve Mathematical Reasoning via Automated Process Supervision](https://arxiv.org/abs/2406.06592)
- [GenPRM: Scaling Test-Time Compute of Process Reward Models](https://arxiv.org/abs/2504.00891) | [GitHub](https://github.com/RyanLiu112/GenPRM)
- [Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning](https://arxiv.org/abs/2406.18629)
- [PRIME: Free Process Rewards without Process Labels](https://arxiv.org/abs/2412.01981)
- [VersaPRM: Multi-Domain Process Reward Models via Synthetic Reasoning Data](https://arxiv.org/abs/2502.06737) | [GitHub](https://github.com/UW-Madison-Lee-Lab/VersaPRM)
- [UltraInteract: Eurus: Advancing LLM Reasoning with Reinforcement Learning](https://arxiv.org/abs/2404.02078)
- [The Lessons of Developing PRMs in Mathematical Reasoning](https://arxiv.org/abs/2501.07301)
- [RLHFlow Implementation of Generative PRM](https://huggingface.co/collections/RLHFlow/rlhflow-math-process-reward-model-6725a42fc8808e12aa1cb144)
- [TRL Stepwise Supervision Datasets Collection](https://huggingface.co/collections/trl-lib/stepwise-supervision-datasets-677ea27fd4c5941beed7a96e)
