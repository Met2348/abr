# Phase C 全流水线个人备忘录（自用）

## 1. 一句话总览
Phase C 现在有两条并行主线 + 一个侧车：
1. `C1/C2 value-head 主线`：构建 prefix/corruption/rollout targets，再训练 value head。
2. `PIK 诊断主线`：做 question-level P(IK) 数据、训练和复评。
3. `PRM teacher sidecar`：对已有 C1 样本做外部教师打分，给后续融合做准备。

一句话：
Phase C 的核心目标是先把“价值监督信号”做出来并评估可靠性，再为 Phase D 的 teacher-fusion 铺路。

## 2. 文件图谱（当前 Phase C 文件）

### 2.1 主入口（Suite）
- `scripts/run_phase_c_value_suite.sh`
- `scripts/run_phase_c_pik_suite.sh`

### 2.2 脚本入口
- `scripts/phase_c_prepare_pik_data.py`
- `scripts/phase_c_train_pik.py`
- `scripts/phase_c_eval_pik.py`
- `scripts/phase_c_score_prm_teacher.py`

### 2.3 关键依赖（虽不叫 phase_c，但被 C 套件直接调用）
- `scripts/phase_b_prepare_value_data.py`（C1 value 数据构建）
- `scripts/phase_b_train_value.py`（C2 value head 训练）
- `scripts/phase_b_eval_faithfulness.py`（C2 standalone eval）
- `src/ours/phase_b/pik_data.py`（PIK 数据契约）
- `src/ours/phase_b/value_head.py`（value head + 特征抽取）
- `src/ours/phase_b/faithfulness_eval.py`（Brier/ECE/Pearson/AUC）

## 3. 高层调用链

### 3.1 Value-head 主线（C1/C2）
`run_phase_c_value_suite.sh`
-> `phase_b_prepare_value_data.py`（train/eval 各跑一次，产 C1）
-> `phase_b_train_value.py`（C2 训练）
-> `phase_b_eval_faithfulness.py`（C2 复评）
-> 产 suite summary

### 3.2 PIK 主线（question-level）
`run_phase_c_pik_suite.sh`
-> `phase_c_prepare_pik_data.py`（train/eval 各跑一次，产 PIK C1）
-> `phase_c_train_pik.py`（PIK C2 训练）
-> `phase_c_eval_pik.py`（PIK C2 复评）
-> 产 suite summary

### 3.3 PRM teacher sidecar（D1）
`phase_c_score_prm_teacher.py`
-> 读取 `prefixes.jsonl`（可选 `corruptions.jsonl`）
-> 构造 teacher 输入
-> 外部 PRM 打分
-> 写 `teacher_prefix_scores.jsonl` 等 sidecar 文件

## 4. 各文件详解（详细版）

## 4.1 `scripts/run_phase_c_value_suite.sh`
作用：
- 一键编排 C1->C2 主线，支持大量参数组（trick/quality/CQR）。

关键点：
- `ACTIVE_PHASE_C_GROUP` 决定整套配置。
- C1 实际调用 `phase_b_prepare_value_data.py`。
- C2 实际调用 `phase_b_train_value.py` 和 `phase_b_eval_faithfulness.py`。

最常改：
- group 分组中的 `C1_PREP_EXTRA_ARGS_DEFAULT` 与 `C2_TRAIN_EXTRA_ARGS_DEFAULT`。

风险：
- 组内参数太多，容易“同名不同义”；必须保留 summary 追溯。

## 4.2 `scripts/run_phase_c_pik_suite.sh`
作用：
- 一键跑 PIK 的 C1/C2 + standalone eval。

关键点：
- train/eval 分别生成 PIK C1 数据目录。
- C2 训练与评测参数分离注入（避免错把训练参数给 eval）。

最常改：
- `ROLLLOUT_COUNT/ROLLLOUT_BATCH_SIZE`、`C2_TRAIN_EXTRA_ARGS_DEFAULT`。

风险：
- PIK 数据规模和 rollout K 直接决定信号质量。

## 4.3 `scripts/phase_c_prepare_pik_data.py`
作用：
- 生成 question-level P(IK) 监督数据（questions + rollout predictions + pik targets）。

关键逻辑：
1. 读取 Phase B rows，转成 `PIKQuestionRecord`。
2. 可选 rollout K 次，沿用 Phase A evaluator 打分。
3. 按 sample 聚合成 `PIKTargetRecord`（成功率/可解析率）。
4. 写 manifest/summary。

最常改：
- `--rollout-count`、`--max-new-tokens`、采样参数。

风险：
- `do_sample=False` 会让 P(IK) 退化成近似单点判定。
- 如果 question fallback 比例高，要单独检查这部分样本。

## 4.4 `scripts/phase_c_train_pik.py`
作用：
- 训练 question-level value head（PIK C2）。

关键逻辑：
1. train/eval artifact 兼容性校验。
2. 冻结 backbone，一次性缓存 question features。
3. 用 BCE/MSE/bce_mse 做 calibration 训练。
4. 计算 calibration 指标 + known_auc。
5. 按 raw/posthoc brier 选 best checkpoint。

最常改：
- `calibration_loss`、`known_threshold`、`posthoc_calibration`。

风险：
- known_threshold 是离散化定义，改阈值会明显改变 AUC 解释。

## 4.5 `scripts/phase_c_eval_pik.py`
作用：
- 独立重评 PIK C2 checkpoint，验证训练期指标。

关键点：
- 支持 `best/final` checkpoint。
- 支持 `none/temperature/isotonic/from_run` 后校准路径。

风险：
- standalone 结果若与 train-time 偏差大，先查 eval 样本集和校准模式是否一致。

## 4.6 `scripts/phase_c_score_prm_teacher.py`
作用：
- 给 C1 的 prefix/corruption 样本打 teacher 分，产 sidecar 监督信号。

关键逻辑：
1. 读取 C1 输入文件。
2. 构造 PRM chat 输入（问题 + steps + `<extra_0>`）。
3. 按 step separator 位置抽取 step scores。
4. 汇总为 row-level mean/min 分数，写 sidecar 文件。

最常改：
- `teacher_system_prompt`、`step_separator_token`、batch/max_length。

风险：
- separator token 配置错会导致“能跑但分数无意义”。
- 输入行缺字段时若 `strict=False`，要关注 `teacher_errors.jsonl`。

## 5. Phase C 常见排查路径

### 5.1 Value C2 指标长期不动
- 先确认 C1 rollout/pair_quality 是否真的生成并有分布。
- 再看 corruption 类型是否被弱模式（如 step_drop）主导。

### 5.2 PIK AUC 接近随机
- 先看 `rollout-count` 是否过小。
- 再看 known_threshold 是否不合适（标签塌缩）。

### 5.3 standalone eval 与 train-time 对不上
- 核对 checkpoint（best/final）。
- 核对 posthoc 模式（none/from_run/temperature/isotonic）。

### 5.4 Teacher sidecar 分数异常
- 先检查 `step_separator_token` 与 tokenizer 映射是否单 token。
- 再检查输入文本里 step 构造是否为空或 fallback 过多。

## 6. 修改优先级建议（稳妥）
1. 先改 suite group 参数，不动底层契约。
2. 再改 rollout 配置和 sampling 参数。
3. 再改训练目标（loss/filter/posthoc）。
4. 最后再改 teacher 输入构造和 step 规则。

## 7. 每次实验至少记录这 12 项
- phase group 名称
- run_prefix / run_name
- C1 输入 prepared 指纹路径
- rollout-count / batch / max-new-tokens
- do_sample / temperature / top_p
- C2 calibration loss 设置
- known_threshold（PIK）或 pair filter（value）
- posthoc 模式
- checkpoint selection metric
- teacher 配置（若启用）
- metrics.json 路径
- manifest/summary 路径
