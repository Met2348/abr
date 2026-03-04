# Phase A 全流水线个人备忘录（自用）

## 1. 一句话总览
Phase A 的本质是：
1. 把原始数据集统一成 `CanonicalSample`。
2. 按模板把样本变成 `PreparedSample`（`prompt_text + target_text`）。
3. 用基座模型生成预测并评测，产出 `predictions/scored/metrics/manifest`。
4. 做可选的不稳定性分析（run 内和 run 间翻转）。

## 2. 文件图谱（按执行顺序）

### 2.1 数据规范层
- `src/ours/data/schema.py`
- `src/ours/data/loaders.py`

### 2.2 Phase A 核心模块
- `src/ours/phase_a/contracts.py`
- `src/ours/phase_a/prompt_builder.py`
- `src/ours/phase_a/splitting.py`
- `src/ours/phase_a/answer_extraction.py`
- `src/ours/phase_a/evaluator.py`
- `src/ours/phase_a/instability.py`
- `src/ours/phase_a/__init__.py`

### 2.3 Phase A 入口脚本
- `scripts/phase_a_prepare.py`
- `scripts/phase_a_generate_and_eval.py`
- `scripts/phase_a_eval_predictions.py`
- `scripts/phase_a_analyze_instability.py`
- 套件入口：`scripts/run_phase_a_benchmark_suite.sh`

## 3. 调用链（高层）

### 3.1 准备数据链
`phase_a_prepare.py` -> `load_dataset_canonical` -> `build_prepared_sample` -> 写 `train/validation/test.jsonl`

### 3.2 推理评测链
`phase_a_generate_and_eval.py` -> 读取 `prepared jsonl` -> 模型生成 -> `evaluate_predictions` -> 写 `scored_predictions.jsonl` + `metrics.json`

### 3.3 离线复评测链
`phase_a_eval_predictions.py` -> 读取历史 `predictions.jsonl` -> `evaluate_predictions`

### 3.4 不稳定性分析链
`phase_a_analyze_instability.py` -> `summarize_strategyqa_instability` + `compute_pairwise_prediction_flip`

## 4. 各文件详解（详细版）

## 4.1 `src/ours/data/schema.py`
作用：定义统一样本结构 `CanonicalSample`。

关键点：
- 必填：`id/dataset/question/answer`
- 可选：`cot/metadata`
- `ensure_canonical_samples(...)` 保证 loader 输出统一格式。

最常改：
- 一般不改结构本身。
- 如要新增字段，必须同步更新所有 loader、Phase A builder、下游读写逻辑。

风险：
- 这里一旦字段变化，属于“全链路破坏级”变更。

## 4.2 `src/ours/data/loaders.py`
作用：把多数据集变成统一 `CanonicalSample` 列表。

关键点：
- `load_dataset_canonical(...)` 是统一入口。
- 数据集私有逻辑在各 `load_*` 函数。
- StrategyQA 会把 `decomposition` 转成 `cot`。
- GSM8K 会拆分原始 answer，得到 `cot + final answer`。

最常改：
- 新增数据集 loader。
- 调整某数据集 answer 解析规则。

风险：
- loader 解析错误会把错标签带入整个后续流程。
- split 兜底（如 test-only 数据集）会影响你对 train/val/test 语义的理解。

## 4.3 `src/ours/phase_a/contracts.py`
作用：Phase A 的中间契约。

核心类：
- `PromptTemplateSpec`
- `PreparedSample`
- `PredictionRecord`
- `ScoredPrediction`

关键点：
- `PreparedSample.target_text` 是 B/C 阶段监督桥梁。
- `ScoredPrediction.parse_error` 表示提取失败或不可信，不等于一定答错。

最常改：
- 仅在确有必要时扩展 `metadata`。

风险：
- 强行改核心字段会破坏脚本读写和历史 artifact 兼容。

## 4.4 `src/ours/phase_a/prompt_builder.py`
作用：模板注册与 prompt/target 构建。

关键点：
- `PROMPT_TEMPLATE_REGISTRY`：模板主库。
- `resolve_template(...)`：模板 id + version 校验入口。
- `build_prepared_sample(...)`：把 canonical sample 变成 prepared sample。
- `_build_target_text(...)`：`answer_only` 或 `cot_then_answer`。

最常改：
- 新增模板（StrategyQA/GSM8K 风格实验）。
- 微调模板文案。

风险：
- 模板文案改了但 version 不变，会让实验不可追溯。
- `target_style` 改变会直接改变监督语义。

## 4.5 `src/ours/phase_a/splitting.py`
作用：确定性哈希切分。

关键点：
- `assign_split(sample_id, config)` 用 `seed + sample_id` 做稳定映射。
- 同样本同 seed 必定落同 split。

最常改：
- 基本不改算法。
- 仅在 `SplitConfig` 比例层面改参数。

风险：
- 改算法会导致历史实验不可比。

## 4.6 `src/ours/phase_a/answer_extraction.py`
作用：任务感知答案提取。

关键点：
- `extract_answer(...)` 按 dataset 路由。
- StrategyQA 提取优先级：
  - one-token yes/no
  - final-answer tag
  - last binary token
  - fallback
- 数学题提取优先级：
  - final-answer tag
  - `####`
  - `\\boxed{}`
  - last number fallback

最常改：
- 新增提取规则或正则。

风险：
- 这是 accuracy 和 parse_error 的关键决定点。
- 小改动就可能改变历史指标口径。

## 4.7 `src/ours/phase_a/evaluator.py`
作用：把 `PredictionRecord` 转为 `ScoredPrediction` 并汇总指标。

关键点：
- `score_prediction(...)`：提取 + 归一化 + 等价判断。
- `evaluate_predictions(...)`：总指标 + per-dataset 指标。
- 输出包含：`accuracy`, `parse_error_rate`, `accuracy_parseable`。

最常改：
- 一般不改。
- 如果改，必须 bump `EVALUATOR_VERSION` 并注明口径变化。

风险：
- 评分逻辑变化会让跨 run delta 混入 evaluator drift。

## 4.8 `src/ours/phase_a/instability.py`
作用：分析 freeform 输出不稳定性。

关键点：
- `extract_final_answer_sequence(...)`：抽取最终标签序列。
- `summarize_strategyqa_instability(...)`：run 内多标签/切换统计。
- `compute_pairwise_prediction_flip(...)`：run 间翻转统计。

最常改：
- final-answer 正则模式。

风险：
- pattern 改动会导致历史 instability 指标不可直接对比。

## 4.9 `scripts/phase_a_prepare.py`
作用：生成 Phase A prepared artifact。

输入：
- 数据集名、split policy、模板、target style。

输出目录：
- `assets/artifacts/phase_a_prepared/<dataset>/<fingerprint>/`
- 文件：`train.jsonl`, `validation.jsonl`, `test.jsonl`, `summary.json`, `manifest.json`

关键点：
- `run_spec -> _stable_fingerprint` 决定目录稳定性。
- `--resume` 会复用同指纹且校验通过的旧产物。
- `--overwrite` 会清理并重建。

常改：
- `--template-id`, `--target-style`, `--split-policy`, `--limit`。

高风险：
- 数据路径/loader 参数配置错，会造成“能跑但数据语义错”。

## 4.10 `scripts/phase_a_generate_and_eval.py`
作用：Phase A 主实验脚本（生成 + 评测 + 对比）。

输入：
- prepared jsonl
- model path（可叠 adapter）
- 生成配置

输出目录：
- `assets/artifacts/phase_a_runs/<run_name>_<timestamp>/`
- 核心文件：`predictions.jsonl`, `scored_predictions.jsonl`, `metrics.json`, `manifest.json`, `console.log`

关键逻辑：
- StrategyQA 可走 `freeform` 或 `binary_choice`。
- `oom_backoff`：批次 OOM 时自动二分重试。
- `truncation_recovery`：命中 token cap 且疑似未完结时做 continuation。
- 可与同名历史 run 自动对比 delta。

最常改：
- `--max-new-tokens`, `--batch-size`, `--strategyqa-decode-mode`。
- `--truncation-recovery-*`。

高风险：
- 比较 run 时如果 evaluator 版本不同，delta 不纯。
- decode mode 变更会显著影响 parse_error，不可与旧模式直接横比。

## 4.11 `scripts/phase_a_eval_predictions.py`
作用：离线复评估，不重新生成。

适用场景：
- 只改了 evaluator/提取规则。
- 要快速重算历史 run 指标。

高风险：
- 输入 JSONL 必须满足 `PredictionRecord` 合约。

## 4.12 `scripts/phase_a_analyze_instability.py`
作用：分析单 run 与跨 run 的不稳定性。

输入方式：
- `--run-dirs ...`
- `--scored-jsonl ...`

输出：
- stdout markdown 表格
- 可选 json/markdown 文件

高风险：
- 输入 run 的 sample_id 若重叠很少，pairwise 统计意义有限。

## 4.13 `scripts/run_phase_a_benchmark_suite.sh`
作用：Phase A 套件编排（A1~A12 参数组）。

关键点：
- 统一准备输入、统一跑多配置、统一汇总。
- `GROUP_RUN_SPECS` 是最常改点。

高风险：
- 改 spec 字段顺序会破坏解析。

## 5. 你最常用的排查路径

### 5.1 结果全是 0
- 先看 `num_inputs` 是否为 0（通常是输入 JSONL 路径错）。
- 再看 `predictions.jsonl` 是否实际有内容。

### 5.2 parse_error 异常高
- 先看 decode mode（freeform 还是 binary_choice）。
- 再看模板是否鼓励长输出。
- 再看 `max_new_tokens` 是否过小导致截断。

### 5.3 速度异常慢
- 看 `batch_size` 是否固定为 1。
- 看是否启用了 multi-GPU 分片但模型规模不需要。
- 看 truncation recovery 是否频繁触发。

### 5.4 复现实验对不上
- 核对 prepared fingerprint（模板与 target style）。
- 核对 evaluator version。
- 核对 decode mode 与 max_new_tokens。

## 6. 修改优先级建议（稳妥）
1. 先改入口脚本参数，不改底层逻辑。
2. 再改模板与 target style（`prompt_builder.py`）。
3. 再改提取器（`answer_extraction.py`），此时要版本化评测口径。
4. 最后才碰 data schema 或 splitting 算法。

## 7. 个人实验记录建议
每次至少记录这 8 项：
- prepared 指纹目录
- template_id/template_version
- target_style
- decode_mode
- max_new_tokens
- batch_size
- evaluator_version
- metrics.json 路径
