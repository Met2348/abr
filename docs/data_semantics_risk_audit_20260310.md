# 数据语义风险审计（2026-03-10）

本文档记录截至 `2026-03-10` 对仓库中“监督数据构造语义是否与评测目标一致”的一次专项审计。

结论先写在前面：

1. 当前 `Math-Shepherd -> pair` 的主适配器存在一级风险。
2. 这个风险不是训练超参问题，而是监督样本语义与目标任务语义不对齐。
3. 相同风险还会传播到：
   - `RLHFlow` 适配器；
   - `PRM800K` 的 fallback `+/-` 解析路径。
4. `PRMBench_Preview` 与 `R-PRM` 没有这个同级别问题。
5. `StrategyQA C1` synthetic corruption 不是同一个 bug，但也有次一级监督语义风险。

## 0.1 修复状态（同日后续实现）

本审计提出后，仓库已经完成一轮强制修复。

当前默认行为：
1. `Math-Shepherd` / `RLHFlow` / `PRM800K` fallback 的 step-label 转换，
   默认改为 `first_bad_edge_strict`。
2. 只保留：
   - 首个负步出现前的最后一个干净 prefix，
   - 与包含首个负步的 prefix。
3. 若首个负步之前没有任何正步，则该样本直接丢弃。
4. 新 artifact 会显式记录：
   - `pair_build_mode`
   - `pair_semantics`
5. 新 artifact stage：
   - `phase_e_pairs_v2`
   - `phase_d_external_pairs_v2`

因此：
1. 本文档第 1~4 节描述的风险对旧 artifact 仍然成立；
2. 但新生成 artifact 的默认风险等级已下降；
3. 新结果应解释为：
   - 严格 `first_bad_edge` 监督结果，
   而不是：
   - nearest-negative 深度混合监督结果。

---

## 1. 一级风险：`Math-Shepherd` 当前 pair 不是“同一分叉点的好坏步骤对比”

### 1.1 代码证据

关键代码路径：

1. `src/ours/phase_d/external_pairs_adapters.py:161`
   - `load_math_shepherd_pairs(...)`
2. `src/ours/phase_d/external_pairs_adapters.py:716`
   - `_convert_step_labels_to_pairs(...)`

当前逻辑不是：

1. 在同一个 prefix 历史下，取“当前步的一个好候选”和“当前步的一个坏候选”构成 pair。

当前逻辑实际上是：

1. 找到一个 `positive_step_index = pos_idx`
2. 再找距离它最近的 `negative_step_index = neg_idx`
3. 然后构造：
   - `chosen_text = prefix[:pos_idx]`
   - `rejected_text = prefix[:neg_idx]`

这意味着 pair 往往是：

1. 两个不同长度的前缀；
2. 来自同一条轨迹的两个不同阶段；
3. 而不是“同一个局部分叉点的好坏候选”。

### 1.2 实际 artifact 证据

对 `E15` 当前共享 pair 产物做了直接统计：

文件：
`assets/artifacts/phase_e_pairs/phase_e_ms_robust_0310_1421_e15_math_shepherd_trust_robust_seed3_sharedsplit_s42_pairs__db2dbba4da9a/train_pairs.jsonl`

统计结果：

1. `Math-Shepherd` train pairs 总数：`5358`
2. `chosen/rejected` 行数完全相同的 pair：`0`
3. 不同长度 pair：`5358`
4. 其中 `rejected` 比 `chosen` 更长的 pair：`4971`

这说明当前主适配器大多数样本其实是在比较：

1. “较短、较早的 prefix”
2. 对
3. “较长、较晚、后面开始出错的 prefix”

这更接近：

1. `trajectory stage / progress depth` 排序，

而不是：

1. `local step quality` 排序。

### 1.3 为什么这会直接伤害 `ProcessBench`

`ProcessBench` 的核心语义见：
`src/ours/phase_e/benchmark_eval.py:103` 与 `:138`

它要求模型做的是：

1. 区分同一道题里 `good prefixes` 与 `bad prefixes`
2. 尤其要识别 `first_error_edge`
   - 即从最后一个好前缀到第一个坏前缀的局部下降

但当前 `Math-Shepherd` pair 并没有专门监督这个局部转折点。

所以出现下面这种现象是合理的，而不是偶然：

1. same-source held-out 很好；
2. `ProcessBench GSM8K/Math` 仍接近随机。

这正对应当前结果：

1. `assets/artifacts/phase_e_logs/phase_e_ms_robust_0310_1421/final_summary.md`
   - `heldout_pair_acc = 0.7518`
   - `heldout_auc = 0.7280`
   - `processbench_gsm8k_auc = 0.4834`
   - `processbench_math_auc = 0.4746`

---

## 2. 其他数据源的同类风险扫描

### 2.1 `PRM800K`

代码路径：

1. `src/ours/phase_d/external_pairs_adapters.py:203`
   - `load_prm800k_pairs(...)`
2. `src/ours/phase_d/external_pairs_adapters.py:253`
   - `_extract_prm800k_completion_pairs(...)`
3. `src/ours/phase_d/external_pairs_adapters.py:716`
   - `_convert_step_labels_to_pairs(...)`

结论分两种情况：

1. 官方 completion-ratings 路径：风险较低
   - chosen/rejected 来自同一步的不同 completion；
   - 这更接近真正的局部好坏候选对比。
2. fallback `+/-` 路径：风险与 `Math-Shepherd` 同级
   - 因为也会落到 `_convert_step_labels_to_pairs(...)`。

当前仓库已有 `PRM800K` artifact 统计表明：

1. 当前 `E5` 主体样本几乎都来自 `pair_build_mode = prm800k_completion_ratings`
2. 因此 `PRM800K` 当前主线结果弱，不应简单归咎于“和 Math-Shepherd 一样的 pair 语义错位”
3. 它更像是：
   - 当前 source 本身较弱，
   - 或当前训练 recipe 对它利用得不好

所以：

1. `PRM800K` 当前主线是弱源；
2. 但它不是当前仓库里“最严重的数据语义风险源”。

### 2.2 `RLHFlow Mistral / Deepseek`

代码路径：

1. `src/ours/phase_d/external_pairs_adapters.py:360`
2. `src/ours/phase_d/external_pairs_adapters.py:389`
3. `src/ours/phase_d/external_pairs_adapters.py:424`

结论：

1. `RLHFlow` 两条路径都复用 `_convert_step_labels_to_pairs(...)`
2. 因而与 `Math-Shepherd` 存在同类一级风险
3. 如果未来把 `RLHFlow` 拉回主线训练，必须先重写 pair 构造，不应直接照搬当前转换器

### 2.3 `PRMBench_Preview`

代码路径：

1. `src/ours/phase_d/external_pairs_adapters.py:97`
2. `src/ours/phase_e/benchmark_eval.py:207`

结论：

1. 它直接使用：
   - `original_process`
   - `modified_process`
   - `error_step_index`
2. chosen/rejected 在同一错误步位置上截断
3. 这更接近：
   - 同一问题；
   - 同一局部分叉点；
   - 局部错误注入后的过程对比

因此：

1. `PRMBench_Preview` 是当前仓库里语义最干净的外部 pair 源之一
2. 它更适合作为 benchmark-facing 训练/验证参考

### 2.4 `R-PRM`

代码路径：

1. `src/ours/phase_d/external_pairs_adapters.py:48`

结论：

1. 它本身就是直接的 `instruction/chosen/rejected` pair
2. 没有 `Math-Shepherd` 这种“从单轨迹 step label 反推 pair”带来的结构性歧义

当前风险主要是次一级的：

1. `pair_confidence=0.78` 是固定启发式，不是校准概率
2. `domain_tag="general_math"` 较粗糙

但这不是当前导致实验失败的主矛盾。

---

## 3. 仓库旧数据链路中最接近的次一级风险

### 3.1 `StrategyQA C1` synthetic corruption

代码路径：

1. `src/ours/phase_b/corruptions.py:193`
   - `_build_corruptions_for_prefix(...)`

这条链路与 `Math-Shepherd` 风险不同：

1. 它只改最后一个可见步骤；
2. clean/corrupt 至少在“局部扰动”这个意义上是对齐的。

所以它不是同一个 bug。

但它依然有监督语义风险：

1. corruption 是手工启发式，不是真实人类/teacher 标注的局部坏步骤；
2. 某些扰动可能只改变表面形式，不一定真的改变推理正确性；
3. `step_drop` 等 fallback 如果占比过高，会把 pair 语义退化成“信息缺失 vs 信息更全”。

因此：

1. `StrategyQA C1` 更像是弱监督 / 合成监督
2. 不应把它解读成 PRM-grade 过程监督

### 3.2 Phase E checkpoint 选择与 benchmark 口径错位

代码路径：

1. `scripts/run_phase_e_suite.sh:520`
   - 训练时用 `CHECKPOINT_SELECTION_METRIC`
2. `scripts/run_phase_e_suite.sh:590`
   - benchmark eval 固定 `--checkpoint-name best`

风险点：

1. `best` checkpoint 是按 source held-out 指标选出来的；
2. 但最后又拿它去汇报 `ProcessBench` / `PRMBench`；
3. 这会让 benchmark 结果受到 source-heldout 选模偏好的影响。

这不是当前最主要的问题，但属于评测口径风险。

更严谨的做法应当至少补：

1. `best` 与 `final` 的 benchmark 双报告；
2. 或明确区分：
   - source-selected benchmark result
   - benchmark-aware selected result

---

## 4. 本次审计的最终判断

### 4.1 哪些是一级风险

1. `Math-Shepherd` 当前 pair 适配器
2. `RLHFlow` 当前 pair 适配器
3. `PRM800K` 的 fallback `+/-` 适配路径

它们的问题共同点是：

1. 把单轨迹 `+/-` step labels 直接转成不同深度前缀 pair；
2. 监督的是“轨迹阶段差异 + 局部错误混合信号”；
3. 不是严格意义上的“同一步局部分叉好坏对比”。

### 4.2 哪些不是同级别风险

1. `PRMBench_Preview`
2. `R-PRM`
3. `PRM800K` 官方 completion-ratings 路径

这些源更接近真正的 pair supervision。

### 4.3 哪些是次一级风险

1. `StrategyQA C1` synthetic corruption 的监督语义较弱
2. Phase E benchmark eval 的 checkpoint 选模口径与 benchmark 目标不完全一致

---

## 5. 对当前实验结论的约束

因此，当前仓库不应再做以下叙述：

1. “Math-Shepherd 单源已经学会了局部 first-error detection”
2. “same-source held-out 高，就说明方法已经接近 ProcessBench 要求”
3. “Math-Shepherd 当前 adapter 的成功，可直接当作 step-quality verifier 的成功”

当前更严谨的说法是：

1. 现有 `Math-Shepherd` adapter 证明了：
   - 这个训练栈可以学到一种稳定的同源 prefix ranking 信号
2. 但它尚未证明：
   - 学到的是 `ProcessBench` 风格的局部 first-error discrimination

---

## 6. 建议的后续动作

优先级建议：

1. 把 `Math-Shepherd` / `RLHFlow` 的 pair builder 改成更严格的局部构造：
   - 同一步候选对比；
   - 或 first-bad-edge 对比；
   - 或至少保证 shared-history 明确、局部差异单一
2. Phase E 报告中，把 `same-source held-out` 与 `benchmark-native` 明确拆开叙述
3. 对 benchmark eval 至少补：
   - `best` checkpoint
   - `final` checkpoint
4. 在所有 summary/manifest 中保留 `pair_build_mode`，避免不同语义的 pair 被混为一谈

相关主文档同步位置：

1. `docs/readme.md`
2. `docs/readme_full.md`
3. `docs/phase_E_plan.md`
4. `docs/phase_D_plan.md`
