# LLM-as-Judge 技术选型与使用方案（2026-03-11）

## 1. 背景和动机

### 1.1 当前数据质量瓶颈

我们的 Phase E 主线数据来自 Math-Shepherd（MC 标注）。从社区研究来看，MC 估计是三种方法中质量最差的：

| 标注方法 | ProcessBench 平均 F1 | 说明 |
|---|---|---|
| MC estimation (Math-Shepherd style) | ~40（参见 Qwen2.5-Math-7B-PRM-MC） | 弱最终泛化 |
| LLM-as-judge | ~56（Qwen2.5-Math-7B-PRM-7B 初步版） | 显著改善 |
| Human annotation (PRM800K) | ~57（PRM800K baseline） | 上限参考 |
| **Consensus (MC + judge 一致)** | **73.5（Qwen2.5-Math-PRM-7B SOTA）** | **最优策略** |

**关键论文**：arXiv:2501.07301（Qwen 团队）明确证明：
> "MC estimation-based data construction yields inferior performance and generalization compared to LLM-as-a-judge and human annotation methods."

consensus filtering（MC 与 judge 均一致才保留）是 Qwen2.5-Math-PRM 的核心贡献之一。

### 1.2 为什么 MC 不好

MC 的根本缺陷不是噪声，而是**语义错误**：
- MC 估计的是"从当前步之后能否推导出正确答案"（value model）
- PRM 需要的是"当前步是否正确"（deterministic evaluator）
- 一个错误的步骤可能仍然能 MC-complete 到正确答案（MC 误标为"正确"）
- 一个正确的步骤偶尔 MC-complete 失败（MC 误标为"错误"）

---

## 2. 社区 LLM-as-judge 方案（来自 arXiv:2501.07301 Appendix C）

### 2.1 judge 模型选择

| 方案 | judge 模型 | ProcessBench F1 | 说明 |
|---|---|---|---|
| Qwen 官方 | Qwen2.5-72B-Instruct | 73.5（7B PRM） | 官方方案，质量最优 |
| ThinkPRM | QwQ-32B-Preview | 88（ThinkPRM-7B） | CoT 验证链 |
| 我们可用 | Qwen2.5-Math-7B-Instruct | 估计 ~55-60 | 数学专项，可用 |
| 我们可用 | Qwen2.5-Math-PRM-7B | 用于打分过滤 | 直接作为 oracle |

### 2.2 官方 LLM-as-judge 提示模板（Appendix C 原文）

```
I will provide a math problem along with a solution. They will be formatted as follows:

[Math Problem]

<math_problem>
...(math problem)...
</math_problem>

[Solution]

<paragraph_1>
...(paragraph 1 of solution)...
</paragraph_1>

...

<paragraph_n>
...(paragraph n of solution)...
</paragraph_n>

Your task is to review each paragraph of the solution in sequence, analyzing,
verifying, and critiquing the reasoning in detail. You need to provide the
analyses and the conclusion in the following format:

<analysis_1>
...(analysis of paragraph 1)...
</analysis_1>

...

<analysis_n>
...(analysis of paragraph n)...
</analysis_n>

<conclusion>
Correct/Incorrect
</conclusion>

* When you analyze each paragraph, you should use proper verification,
recalculation, or reflection to indicate whether it is logically and
mathematically valid. Please elaborate on the analysis process carefully.

* If an error is detected in any paragraph, you should describe the nature and
cause of the error in detail, and suggest how to correct the error or the correct
approach. Once a paragraph is found to contain any error, stop further analysis
of subsequent paragraphs (as they may depend on the identified error) and directly
provide the conclusion of "Incorrect."

[Example omitted - see paper]

* Respond with your analyses and conclusion directly.

---------------------------------------------------

The following is the math problem and the solution for you task:

[Math Problem]

{tagged_problem}

[Solution]

{tagged_response}
```

### 2.3 输出解析

- 提取 `<conclusion>Correct/Incorrect</conclusion>` → 步骤是否正确
- 如果找到第 k 步有错误 → 第 k 步是 first_bad_step
- 后续步骤跳过（stop-at-first-error）
- 最终输出：每个 solution 的 `first_bad_step_index`（或 -1 表示全正确）

### 2.4 数据规模

- Qwen 团队用 860k query×response 对进行 judge 标注
- 每条响应平均 8-15 步 → 最多 860k × 15 ≈ 12.9M judge 调用
- 但用 consensus filter 后只保留 MC 和 judge 一致的子集
  - 实践中保留率约 30-50%，最终有效对在 200-400k 级别

---

## 3. 我们的技术选型

### 3.1 服务器现状

| 资源 | 状态 |
|---|---|
| GPU | 4× A100 80GB PCIe |
| 当前空闲 GPU | GPU0（77GB 空闲）、GPU3（75GB 空闲）|
| 已有模型 | Qwen2.5-7B-Instruct、Qwen2.5-Math-7B-Instruct、Qwen2.5-Math-PRM-7B、DeepSeek-R1-Distill-Qwen-14B |
| 磁盘剩余 | 348GB（96% 已用，无法下载 72B 模型） |

### 3.2 技术选型决策

**核心约束**：无法下载 Qwen2.5-72B-Instruct（磁盘不够）

**推荐方案**：两阶段 hybrid

#### 方案 A（近期优先）：Qwen2.5-Math-PRM-7B 作为 oracle 过滤

原理：
- 直接用已有 Qwen2.5-Math-PRM-7B 对 Math-Shepherd steps 打分
- 保留 MC 标签与 PRM 分数一致的样本（consensus filtering）
- 具体：MC=1（正确）且 PRM_score > 0.6 的步骤 = 高质量正确步
- 具体：MC=0（错误）且 PRM_score < 0.4 的步骤 = 高质量错误步
- 不一致的步骤 → 丢弃

优点：
1. **无需额外推理成本**：PRM-7B 已经在服务器上，直接用
2. **与 consensus 原理一致**：两个独立系统（MC 和 PRM）均认为这一步有问题
3. **质量提升有理论保证**：Qwen 论文图2显示 consensus 数据效率是纯 MC 的 2倍以上

实现步骤：
```python
# 用 Qwen2.5-Math-PRM-7B 对每个 step 打分
# PRM 输出：每步的 <extra_0> token 概率 = correctness_score
# 过滤逻辑：
def keep_pair(mc_label, prm_score, threshold_hi=0.6, threshold_lo=0.4):
    if mc_label == 1 and prm_score > threshold_hi:
        return True  # 双确认正确
    if mc_label == 0 and prm_score < threshold_lo:
        return True  # 双确认错误
    return False  # 不一致，丢弃
```

#### 方案 B（中期）：Qwen2.5-Math-7B-Instruct 作为 LLM-judge

原理：
- 用 Appendix C 的完整提示模板
- 每条 response 输入 judge，得到 per-step Correct/Incorrect 判断
- 与 MC 做 consensus filtering

适合场景：
- 新问题 dataset 的标注（没有 Math-Shepherd MC label 的数据）
- 对 PRM-oracle filtered 数据的进一步质量提升

批处理 engineering：
- 使用 vLLM offline inference
- batch_size = 32-64（取决于平均步骤数，avg ~10步/response）
- 每秒 ~2-5 responses（7B 模型在 A100 上）
- 100K responses → 约 6-14 小时

#### 方案 C（长期，等磁盘空间）：Qwen2.5-72B-Instruct 官方方案

等清理磁盘后：
- 72B 模型需要约 140GB（4-bit）或约 288GB（bf16）
- 需要 2× A100 80GB（4-bit）或 4× A100 80GB（bf16）
- 质量最高，与 Qwen2.5-Math-PRM-7B paper 方案完全一致

---

## 4. 实现计划

### 4.1 新脚本：`scripts/phase_e_oracle_filter_pairs.py`

**目的**：用 Qwen2.5-Math-PRM-7B 对 Math-Shepherd 数据重新打分并过滤

输入：
- `--source-pairs-jsonl`：原始 Math-Shepherd pair JSONL（train_pairs.jsonl）
- `--model-path`：Qwen2.5-Math-PRM-7B 路径
- `--prm-score-threshold-hi`：正确步阈值（default 0.60）
- `--prm-score-threshold-lo`：错误步阈值（default 0.40）
- `--output-root`：输出目录
- `--run-name`：实验名称

逻辑：
1. 加载 pair JSONL
2. 对每对，将 chosen（好前缀）和 rejected（坏前缀）的最后一步送入 PRM
3. 提取 `<extra_0>` token 概率作为 PRM 分数
4. 检查：
   - `chosen_prm_score > threshold_hi` AND `mc_label_chosen = 1` → keep
   - `rejected_prm_score < threshold_lo` AND `mc_label_rejected = 0` → keep
5. 只保留两个条件都满足的对
6. 输出过滤后的 JSONL + summary.json（保留率、分布等）

### 4.2 新脚本：`scripts/phase_e_judge_annotate.py`

**目的**：用 Qwen2.5-Math-7B-Instruct 对新 solution 做 step-level LLM-judge 标注

输入：
- `--problems-jsonl`：问题+解题 JSONL（每行 `{question, solution_steps: [...]}`）
- `--model-path`：Qwen2.5-Math-7B-Instruct 路径
- `--batch-size`：vLLM 推理批大小（default 32）
- `--output-root`：输出目录
- `--run-name`：实验名称

逻辑：
1. 用 Appendix C 提示模板构造每个 sample 的 judge 输入
2. vLLM batch inference
3. 解析 `<conclusion>Correct/Incorrect</conclusion>`
4. 从 `<analysis_k>` 提取第一个错误步（stop-at-first-error）
5. 输出：每个 solution 的 `{solution_id, judge_first_bad_step, judge_label, raw_output}`

### 4.3 consensus filtering 的实现位置

在 `phase_e_curate_processbench_transfer_pairs.py` 中增加 `--consensus-filter-mode prm_oracle` 选项：
- `none`（默认，现有行为）
- `prm_oracle`：调用方案 A 的 oracle filter
- `llm_judge`：调用方案 B 的 judge filter

---

## 5. 预期收益与实验设计

### 5.1 预期收益

基于 arXiv:2501.07301 的数据：

| 阶段 | 数据策略 | 预期 ProcessBench F1 |
|---|---|---|
| 当前 | MC only（Math-Shepherd 原始） | ~29（7B PRM）→ ~40（7B PRM MC） |
| 方案 A | MC + PRM oracle consensus | 估计 +5-10 F1 （类比 Qwen 结果） |
| 方案 B | MC + LLM-judge-7B consensus | 估计 +8-15 F1 |
| 最终目标 | MC + judge-72B consensus + backbone LoRA | 目标 60+ F1 |

### 5.2 验证实验设计

**实验 Judge-1（方案 A 验证）**：
- 取 ms_align_v1 artifact 中的 Math-Shepherd pair（~8K pairs）
- 用 PRM oracle 过滤（预期保留 40-60%）
- 在过滤后数据上训练 MLP head（相同配置）
- 比较：原始 vs oracle-filtered 的 ProcessBench F1
- 预期：pair_acc 会略降（数据更少），但 F1 提升

**实验 Judge-2（方案 B 验证）**：
- 对 Math-Shepherd 中的 1000 个问题进行 LLM-judge 标注
- 与 MC labels 对比一致率
- 估计标注时间：~1-2 小时（Qwen2.5-Math-7B on 1× A100）

### 5.3 judge 的正确使用方式：论文与社区更一致支持什么

结合 `JudgeLM`、`Prometheus 2`、`ProcessBench`、`ThinkPRM`、`The Lessons of Developing Process Reward Models in Mathematical Reasoning`，当前更稳妥的做法不是“让 judge 一次性输出冗长完美 JSON”，而是：

1. **优先轻合同，不要先上重合同**
   - 对本地 open-source judge，`first_bad_only` 或简短 final block 往往比逐步 verbose 判定更稳定。
   - 复杂 schema 会显著提高 parse failure 和格式漂移。

2. **优先 comparative / pairwise judging，而不是纯 pointwise scalar**
   - 社区经验和 `JudgeLM` 一致表明：pairwise 选择通常比直接打绝对分更稳。
   - 如果任务允许，应优先让 judge 回答：
     - `A better than B`
     - `B better than A`
     - `tie`
   - 而不是先输出一个 0-10 分。

3. **显式做 position-bias 控制**
   - 最低成本做法是 `A/B` 和 `B/A` 两次判定，然后：
     - 若结论一致，则采信
     - 若结论冲突，则降权或送二级 adjudicator
   - 这一步在 judge 任务里非常重要，否则模型会偏向第一个或第二个候选。

4. **保留 raw output，不要只保留 parse 后结果**
   - judge 的错误模式往往藏在 raw generation 里：
     - 冗长解释
     - 格式近似正确但字段错位
     - 过度乐观地输出 `all correct`
   - 所以 artifact 里必须同时保留：
     - `raw_text`
     - `parsed_payload`
     - `parse_ok`

5. **judge 更适合做高质量过滤和复判，不一定适合直接端到端替代 verifier**
   - 这和我们当前项目最匹配：
     - 用 judge 过滤低质量 pair
     - 给低置信或分歧样本复标
     - 为后续 verifier 训练提供更干净 supervision

### 5.4 本地真实 benchmark 小实验：这两个 judge 实际上表现如何

为了避免只停留在“模型卡看起来不错”，我们已经对两个本地可用 judge 做了真实 `ProcessBench` 小切片 benchmark。

统一对照汇总：
- [judge_model_compare_20260311T074514Z summary](../assets/artifacts/phase_e_judge_bench_compare/judge_model_compare_20260311T074514Z/summary.md)

关键结果：

| 模型 | 合同 | Benchmark | parse_ok | overall_acc | first_bad_exact |
|---|---|---|---:|---:|---:|
| Qwen2.5-Math-7B-Instruct | full_steps | ProcessBench math | 0.6667 | 0.3333 | 0.0000 |
| Qwen2.5-Math-7B-Instruct | full_steps | ProcessBench gsm8k | 1.0000 | 0.5000 | 0.0000 |
| Qwen2.5-Math-7B-Instruct | first_bad_only | ProcessBench math | 0.5000 | 0.3333 | 0.0000 |
| Qwen2.5-Math-7B-Instruct | first_bad_only | ProcessBench gsm8k | 1.0000 | 0.5000 | 0.0000 |
| DeepSeek-R1-Distill-Qwen-14B | full_steps | ProcessBench math | 0.5000 | 0.1667 | 0.0000 |
| DeepSeek-R1-Distill-Qwen-14B | full_steps | ProcessBench gsm8k | 0.5000 | 0.3333 | 0.0000 |
| DeepSeek-R1-Distill-Qwen-14B | first_bad_only | ProcessBench math | 0.5000 | 0.1667 | 0.0000 |
| DeepSeek-R1-Distill-Qwen-14B | first_bad_only | ProcessBench gsm8k | 0.8333 | 0.6667 | 0.3333 |

由此得到的工程结论：

1. **`Qwen2.5-Math-7B-Instruct` 是当前更适合接主线的 bulk judge**
   - 稳定性更好
   - 解析成功率更高
   - 但存在明显的 `OVERALL=correct / FIRST_BAD=none` 乐观偏置

2. **`DeepSeek-R1-Distill-Qwen-14B` 不适合直接做 bulk judge**
   - 在 verbose contract 下不稳定
   - 但在 `first_bad_only` 轻合同下，对 `gsm8k` 出现了真信号
   - 更适合作为二级 adjudicator，而不是一线批量标注器

3. **当前最不该做的事**
   - 直接假设本地 judge 可以稳定输出 strict JSON
   - 直接把它们当成可替代 ProcessBench verifier 的主系统

### 5.5 更新后的 judge 接入方案

基于文献和本地实验，当前最合理的接入方案应改成：

#### 主线 A：bulk relabel / filter

- 模型：`Qwen2.5-Math-7B-Instruct`
- 合同：`first_bad_only`
- 角色：
  - 低置信 pair 复判
  - disagreement mining
  - weak pair 过滤
- 工程要求：
  - tolerant parser
  - 保留 raw output
  - 不要求 strict full-step JSON

#### 主线 B：hard-case adjudication

- 模型：`DeepSeek-R1-Distill-Qwen-14B`
- 合同：`first_bad_only`
- 角色：
  - 只复判 hardest / ambiguous 样本
  - 尤其适合 `gsm8k`-style verbal / arithmetic mixed cases
- 工程要求：
  - 避免复杂 system prompt
  - 输出合同尽量轻
  - 不做 bulk

#### 下一步优先实验

1. `pairwise judge benchmark`
   - 在 `PRMBench_Preview` 或本地 curated pair 上做 `A/B` 与 `B/A` 双向 judge
   - 指标：
     - pair accuracy
     - swap consistency
     - tie rate

2. `judge relabel for low-confidence pairs`
   - 先只改写低置信 / 高分歧样本
   - 不直接重写全量训练集

3. `judge-as-filter, not judge-as-full-replacement`
   - 当前证据不支持直接把 LLM judge 当完整替代 verifier 主系统
   - 更支持把 judge 当作 supervision quality improver

### 5.6 pairwise judge 小实验更新：理论方向对，但当前本地小 judge 还不够强

为了验证第 5.3 节里“pairwise / swap-debias 更合理”的判断，我们又直接在本仓库 pair artifact 上做了小实验：

统一对照：
- [judge_pairwise_compare_20260311T084132Z summary](../assets/artifacts/phase_e_pairwise_judge_compare/judge_pairwise_compare_20260311T084132Z/summary.md)

关键结果：

1. `Qwen2.5-Math-7B-Instruct` on held-out `PRMBench_Preview` (`32` pairs)
   - `both_parse_ok_rate = 0.3125`
   - `pair_acc_majority = 0.3438`
   - `swap_consistency_rate = 0.6000`
   - `label_preserving_keep_rate = 0.1250`

2. `Qwen2.5-Math-7B-Instruct` on held-out `Math-Shepherd local_first_bad_edge` (`32` pairs)
   - `both_parse_ok_rate = 0.2188`
   - `pair_acc_majority = 0.0625`
   - `label_preserving_keep_rate = 0.0312`
   - `judge_contradiction_rate = 0.0625`

3. `Qwen2.5-Math-7B-Instruct` as train-slice filter
   - `PRMBench_Preview train64`: `keep_rate = 0.0469`
   - `Math-Shepherd train64`: `keep_rate = 0.0000`

4. `DeepSeek-R1-Distill-Qwen-14B` on held-out `PRMBench_Preview` (`16` pairs)
   - `both_parse_ok_rate = 0.5625`
   - `pair_acc_majority = 0.0625`
   - `tie_rate = 0.8125`

这组结果说明：

1. pairwise + swap-debias 确实是比 pointwise strict-JSON 更合理的 judge 方向；
2. 但当前本地小 judge 仍然不够强，不能直接升级成 bulk Phase E pair filter；
3. `PRMBench_Preview` 比 `Math-Shepherd local_first_bad_edge` 更接近 judge-friendly semantics；
4. `Math-Shepherd` 上的问题主要不是格式，而是语义错位：
   - 训练标签里更短但仍然正确的安全前缀，
   - 常被 judge 误判得不如更长但已引入错误的前缀。

## 6. 一句话总结给用户

**当前最合理的 judge 主线不是“直接上最强模型做全量严格判题”，也不是“立刻做 bulk pair filter”，而是：用 `Qwen2.5-Math-7B-Instruct` 在轻合同下做 selected relabel / disagreement mining，用 `DeepSeek-R1-Distill-Qwen-14B` 只做少量难例复判，并优先推进 pairwise + swap-debias judge 实验。**

## 7. 联网核验后的实现规范总结

这部分对应我们重新检索论文和社区经验后的更硬结论。当前最相关的外部依据包括：

1. `JudgeLM`
2. `Prometheus / Prometheus 2`
3. `G-Eval`
4. `ProcessBench`
5. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`

这些工作的共同结论，与我们本地实验形成了很强的一致性：

### 7.1 `PRMBench_Preview` 风格 pairwise 数据

外部经验：

1. judge 更适合做 `pairwise comparative judging`，而不是对单条推理直接给绝对分。
2. 为了减轻 `position bias`，应做：
   - `A/B`
   - `B/A`
   - majority / consistency aggregation
3. 输出合同应该尽量轻，不要要求冗长自由发挥。

本地验证：

1. 我们把 judge 改成 pairwise + swap-debias 后，`PRMBench_Preview` 明显比 `Math-Shepherd local_first_bad_edge` 更 judge-friendly。
2. 这说明 `PRMBench_Preview` 这种“显式二选一修改轨迹”更接近 judge 的天然适用面。

工程结论：

1. judge 主线优先接 `PRMBench_Preview` 风格数据。
2. 不要优先接 `Math-Shepherd local_first_bad_edge` 风格数据做 bulk filtering。

### 7.2 selected relabel

外部经验：

1. 社区更常见的用法不是“全量重标”，而是：
   - 只重标低置信样本
   - 只重标 hard cases
   - 或只对 disagreement 样本做复判
2. 原因是 judge 本身也有噪声，全量重写标签很容易把错误系统化。

本地验证：

1. `Qwen2.5-Math-7B-Instruct` 在 `PRMBench_Preview train64` 上的 `label_preserving_keep_rate` 只有 `0.0469`。
2. 在 `Math-Shepherd train64` 上甚至是 `0.0000`。

工程结论：

1. 当前证据只支持 `selected relabel`。
2. 不支持 `bulk relabel`。

### 7.3 disagreement mining

外部经验：

1. judge 更适合作为 `disagreement detector`，而不是绝对真值机。
2. 实践上常见做法是：
   - 先用主模型/现有标签筛出分歧样本
   - 再让 judge 介入
   - 把 judge 当作“二次证据”

本地验证：

1. `Qwen` 在 `PRMBench_Preview` 上虽然整体精度不高，但已经能产生少量 `swap-consistent + label-preserving` 样本。
2. 这类样本很适合进 `disagreement mining` 支路。

工程结论：

1. `disagreement mining` 是当前 judge 最值得保留的用途之一。
2. 它比“直接做主监督源”稳得多。

### 7.4 benchmark-side adjudication

外部经验：

1. judge 在 benchmark-side adjudication 上更安全，因为它不会反向污染训练集。
2. 对 benchmark 样本做二级复判，本质上是：
   - error taxonomy 辅助
   - ambiguity resolution
   - failure slice inspection

本地验证：

1. 当前 judge 对 `ProcessBench` / `Math-Shepherd` 风格数据不够强，不适合直接拿来重写训练标签。
2. 但它仍可用于 benchmark 侧的：
   - pairwise adjudication
   - disagreement audit
   - hard-slice analysis

工程结论：

1. `benchmark-side adjudication` 比 `training-side bulk filtering` 更现实。
2. 这也是当前阶段最安全的 judge integration 方式。
