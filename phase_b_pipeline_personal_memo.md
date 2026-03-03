# Phase B 全流水线个人备忘录（自用）

## 1. 一句话总览
Phase B（按仓库命名）实际上覆盖两层：
1. `B1/B2`：SFT/PEFT 训练与基准评测（模型能力侧）。
2. `C0/C1/C2`：value head 数据准备、训练与忠实性评估（过程价值侧）。

一句话：
先把 Phase A 的 `prepared jsonl` 训成 adapter（B1/B2），再把同一轨迹切成 prefix + corruption + rollout target，训练并评估 value head（C0/C1/C2）。

## 2. 文件图谱（按执行顺序）

### 2.1 入口与编排（Shell）
- `scripts/run_phase_b_training_suite.sh`
- `scripts/run_phase_b_cross_task_suite.sh`

### 2.2 B1/B2 主脚本
- `scripts/phase_b_train_sft.py`
- `scripts/phase_b_eval.py`
- `scripts/phase_b_compare_eval.py`
- `scripts/phase_b_checkpoint_sweep.py`

### 2.3 C0/C1/C2 主脚本（文件名仍是 phase_b_*）
- `scripts/phase_b_prepare_value_data.py`
- `scripts/phase_b_train_value.py`
- `scripts/phase_b_eval_faithfulness.py`

### 2.4 Phase B 核心模块
- `src/ours/phase_b/contracts.py`
- `src/ours/phase_b/data.py`
- `src/ours/phase_b/supervision.py`
- `src/ours/phase_b/corruptions.py`
- `src/ours/phase_b/value_targets.py`
- `src/ours/phase_b/value_data.py`
- `src/ours/phase_b/value_head.py`
- `src/ours/phase_b/value_losses.py`
- `src/ours/phase_b/faithfulness_eval.py`
- `src/ours/phase_b/posthoc_calibration.py`
- `src/ours/phase_b/pik_data.py`

## 3. 高层调用链

### 3.1 B1/B2（模型训练与对比）
`run_phase_b_training_suite.sh`
-> `phase_b_train_sft.py`
-> （可选）`phase_b_eval.py` pre/post
-> `phase_b_compare_eval.py` 生成 gain 报告
-> （可选）`phase_b_checkpoint_sweep.py` 找最佳 checkpoint

### 3.2 C0/C1（价值监督数据构建）
`phase_b_prepare_value_data.py`
-> `load_phase_b_rows`
-> `build_step_sequence_from_phase_b_row`
-> `build_prefix_artifacts`
-> `build_corruptions_for_prefixes`
-> （可选）rollout + `score_prediction`
-> 产出 `prefixes/corruptions/rollout_targets/pair_quality`

### 3.3 C2（value head 训练与复评）
`phase_b_train_value.py`
-> `load_value_supervision_examples` + `load_corruption_variants`
-> 冻结 backbone + 编码 feature cache
-> value head 训练（calibration + contrastive）
-> `compute_calibration_summary` / `compute_corruption_summary`
-> 输出 `best/final_value_head.pt + eval_metrics`

`phase_b_eval_faithfulness.py`
-> 独立重载 checkpoint 和 eval dir
-> 重新编码 + 打分
-> 输出独立 `metrics.json`（用于复核训练时结论）

## 4. 各文件详解（详细版）

## 4.1 `scripts/run_phase_b_training_suite.sh`
作用：
- 统一定义 B 组实验（配置、意图、预期、评测计划）。
- 支持 pre/post 自动评测并产出 gain report。

关键点：
- `ACTIVE_PHASE_B_GROUP` 决定跑哪组。
- `CONFIG_JSON` 是真实训练超参来源。
- `EVAL_SPECS` 控制评测 split 与 decode mode。

最常改：
- 组定义里的 `CONFIG_JSON`、`EVAL_SPECS`。

风险：
- 评测输入换了 prepared 指纹但没记录，会破坏可比性。

## 4.2 `scripts/run_phase_b_cross_task_suite.sh`
作用：
- 做跨任务干扰验证（source adapter -> target task）。

关键点：
- `SOURCE_RUN_NAME_PREFIX` 指向已完成训练 run。
- pre 用 base 模型，post 用 source adapter。

风险：
- 若 source run 定位错，会把“跨任务结论”建在错误模型上。

## 4.3 `scripts/phase_b_train_sft.py`
作用：
- B1 主训练脚本（PEFT/SFT）。

关键逻辑：
1. 两阶段参数解析：先读 `--config-json` 默认值，再允许 CLI 覆盖。
2. `load_phase_b_rows` 加载并校验训练样本。
3. `build_supervision_plan` 对 target 做变换（如短 CoT）。
4. `_build_features` 构建 causal LM 监督（prompt 侧 label=-100）。
5. 可选 LoRA 注入；失败可 fallback 到 full SFT。
6. `Trainer` 训练并保存 manifest/metrics/final_model。

最常改：
- `target_transform`、`answer_weighting_mode`。
- LoRA 参数：`lora-rank/lora-alpha/target_modules`。
- 训练吞吐：batch、grad accumulation、max_seq_length。

风险：
- `target_transform` 变化会直接改变监督语义。
- `max_seq_length` 太小会截断 supervision，造成“能训但标签失真”。

## 4.4 `scripts/phase_b_eval.py`
作用：
- 把 Phase B 产物桥接到冻结的 Phase A evaluator。

关键点：
- 三选一输入源：`--model-path` / `--phase-b-run-dir` / `--phase-b-checkpoint-dir`。
- PEFT 模式会自动走 base + adapter 组合。

风险：
- 输入源混用会导致评测对象不明确。

## 4.5 `scripts/phase_b_compare_eval.py`
作用：
- 对 pre/post `metrics.json` 做 split 级与 aggregate 级对比。

关键点：
- 强制 before/after `n_total` 一致。
- 输出 `delta_accuracy`、`delta_parse_error_rate`、`delta_n_correct`。

风险：
- 若 before/after evaluator 口径变化（版本漂移），delta 不再纯净。

## 4.6 `scripts/phase_b_checkpoint_sweep.py`
作用：
- 扫 checkpoint，定位最佳 held-out 表现。

关键点：
- baseline（base model）先测一遍。
- 每个 checkpoint + final 都走同一评测链。

风险：
- 只看训练 loss 会错过中途最好点，必须看 benchmark sweep。

## 4.7 `scripts/phase_b_prepare_value_data.py`
作用：
- C0/C1 数据入口：prefix、corruption、rollout target、pair_quality。

关键逻辑：
1. 构建 step config / prefix config / corruption config。
2. 从 Phase B rows 构造 step 序列与 prefix 样本。
3. 可选构造 corruption（legacy 或 cqr_balanced）。
4. 可选 rollout 估计每个 prefix 的经验 Q（含不确定性统计）。
5. 可选 clean-vs-corrupt pair quality（delta_q / z_delta / pair_weight）。

最常改：
- `--corruption-selection-policy` 与 semantic corruption 开关。
- `--rollout-count` / 两阶段 rollout 参数。
- target quality 参数（alpha/beta/ci_z/weight_floor）。

风险：
- rollout 不开时，后续 value head 缺监督目标。
- corruption 过弱会导致对比训练信号弱。

## 4.8 `scripts/phase_b_train_value.py`
作用：
- C2 主训练脚本：冻结 backbone，只训练 value head。

关键逻辑：
1. 检查 train/eval artifact 兼容性。
2. 冻结 backbone，编码 clean/corrupt features 到缓存。
3. calibration loss（mse/bce/bce_mse）+ contrastive loss 组合训练。
4. 支持 pair filter、pair weight、stratified sampling、adaptive balancing。
5. 每轮 eval 输出 calibration/corruption 指标，按指标选 best checkpoint。

最常改：
- `calibration_loss`、`lambda_contrastive`。
- `contrastive_pair_filter` 与阈值。
- `checkpoint_selection_metric` 与 posthoc 开关。

风险：
- pair 质量差时，contrastive 可能接近噪声学习。
- 只看 raw 指标不看 posthoc/ablation，结论容易偏。

## 4.9 `scripts/phase_b_eval_faithfulness.py`
作用：
- 独立重评 C2 checkpoint，验证训练日志结论。

关键点：
- 支持 `best/final` checkpoint 切换。
- 支持 posthoc：`none/temperature/isotonic/from_run`。

风险：
- 若 eval_dir 与训练时契约不一致，结果不可解释。

## 4.10 `src/ours/phase_b/contracts.py`
作用：
- `PhaseBTrainRow` 契约定义。

风险：
- 字段变更会连带破坏 loader、训练、artifact 兼容。

## 4.11 `src/ours/phase_b/data.py`
作用：
- 严格加载 JSONL，检查重复 sample_id。

风险：
- 重复 ID 若不拦截会污染训练分布。

## 4.12 `src/ours/phase_b/supervision.py`
作用：
- supervision transform 与 reasoning/answer 拆分。

关键点：
- `split_reasoning_and_answer` 是 answer-weighting 的前提。

风险：
- 拆分规则改动会显著影响 token-level loss 分布。

## 4.13 `src/ours/phase_b/value_targets.py`
作用：
- prefix / rollout target / pair quality 数据契约和构造。

关键点：
- `build_step_sequence_from_phase_b_row` 和 `build_prefix_artifacts` 是 C0/C1 根入口。

风险：
- prefix 定义一旦漂移，C2 与历史实验不可横比。

## 4.14 `src/ours/phase_b/corruptions.py`
作用：
- 生成可追踪 corruption 变体。

关键点：
- `legacy` vs `cqr_balanced` 两类策略。

风险：
- step_drop 过多会弱化“语义 corruption”信号。

## 4.15 `src/ours/phase_b/value_data.py`
作用：
- 将 prefixes / rollout_targets / corruptions / pair_quality join 成 C2 可用样本。

关键点：
- 若有 pair_quality，优先质量更高 corruption 作为 primary。

风险：
- join 键错配会造成 silent bug（prefix 与 target 不一致）。

## 4.16 `src/ours/phase_b/value_head.py`
作用：
- value head 定义、pooling、保存/加载。

关键点：
- C2 默认只训练这一个小头，backbone 全冻结。

风险：
- hidden_size 推断错误会直接 shape mismatch。

## 4.17 `src/ours/phase_b/value_losses.py`
作用：
- calibration 与 contrastive loss 定义。

关键点：
- `mixed_calibration_loss` 显式混合 BCE 与 MSE。

风险：
- 权重配置失衡会导致一侧信号被淹没。

## 4.18 `src/ours/phase_b/faithfulness_eval.py`
作用：
- 计算 Brier/ECE/Pearson/AUC/pair accuracy 等核心指标。

风险：
- 指标口径改动会影响跨实验可比性。

## 4.19 `src/ours/phase_b/posthoc_calibration.py`
作用：
- 温度缩放与 isotonic 后处理校准。

风险：
- 在很小样本上强行后校准会过拟合。

## 4.20 `src/ours/phase_b/pik_data.py`
作用：
- question-level P(IK) 路径的数据契约与加载（旁路方案）。

风险：
- 与 prefix-level 路径混用时要严格区分任务目标。

## 5. Phase B 常见排查路径

### 5.1 B1 训练跑完但指标不涨
- 先看 `phase_b_compare_eval.py` 的 held-out aggregate，而不是只看 train loss。
- 检查 target transform 是否把有效 supervision 压得太狠。

### 5.2 C2 pair_acc≈0.5 / AUC≈0.5
- 先看 `pair_quality.jsonl` 分布（delta_q / pair_weight 是否塌缩）。
- 再看 corruption 类型分布是否被 step_drop 主导。

### 5.3 posthoc 后指标反而差
- 检查 eval 样本量是否太小。
- 检查 `from_run` 与当前 checkpoint 是否匹配。

### 5.4 结果不可复现
- 核对 train/eval manifest 的 step/prefix 签名。
- 核对 rollout backbone provenance（model/adapter/dtype）。

## 6. 修改优先级建议（稳妥）
1. 先改 suite 参数组和 config JSON，不改底层契约。
2. 再改 supervision transform 与 corruption policy。
3. 再改 C2 的 loss/filter/weighting 组合。
4. 最后才动 contracts / prefix 构造 /指标口径。

## 7. 每次实验至少记录这 12 项
- B 组名与 run_name
- config JSON 路径
- prepared 输入指纹目录
- model_path / adapter_path
- target_transform / answer_weighting_mode
- corruption policy 与参数
- rollout-count / 两阶段参数
- calibration_loss / lambda_contrastive
- pair filter 与阈值
- checkpoint_selection_metric
- posthoc mode
- metrics / manifest / summary 路径
