# Phase C CQR 个人备忘录（自用）

## 1. 一句话结论
他们最近实现的 CQR（这里对应 `cqr_balanced` 路径）本质上是：
- 在 C1 端把 corruption 从“弱扰动+step_drop 容易主导”改成“语义扰动优先 + 类型平衡”；
- 再把 clean-vs-corrupt 的 rollout 差值显式做成 `pair_quality` 标签；
- 最后在 C2 端用这些标签做过滤、加权和分层采样，减少噪声 pair 对训练的破坏。

## 2. 他们到底做了什么（按流水线）

## 2.1 C1: corruption 生成从 `legacy` 扩展到 `cqr_balanced`
代码位置：
- `src/ours/phase_b/corruptions.py`
- 入口参数在 `scripts/phase_b_prepare_value_data.py`

新增要点：
1. 语义型 corruption（在 `cqr_balanced` 下默认开启）
- `negation_flip`
- `comparator_flip`
- `condition_reversal`
- `entity_substitution`

2. 保留 legacy 操作作为补充
- `binary_flip`
- `operator_flip`
- `numeric_perturb`
- `step_drop`（fallback）

3. 每个 prefix 内做“类型平衡”
- `min_non_step_drop_per_prefix`
- `max_step_drop_per_prefix`
- `_round_robin_select_by_type(...)` 保证不同类型轮转入选

设计意图：
- 防止 `step_drop` 这类低信息 fallback 吞掉监督配额；
- 提升 corruption 的“可学习差异密度”。

## 2.2 C1: rollout 目标改成“质量感知标签”
代码位置：
- `scripts/phase_b_prepare_value_data.py`
  - `_build_rollout_targets(...)`
  - `_compute_target_quality_stats(...)`

新增要点：
1. 可选两阶段 rollout（`--rollout-two-stage`）
- stage1：全量 prefix 小预算采样
- stage2：只给不确定 prefix 补采样

2. 目标统计里引入不确定性字段
- `q_mean_smoothed`
- `q_std_error`
- `q_ci_width`
- `q_weight`

核心公式（Obsidian 友好）：

$$
q_{\text{smoothed}}=\frac{n_{\text{correct}}+\alpha}{K+\alpha+\beta}
$$

$$
SE(q)\approx \sqrt{\frac{q_{\text{smoothed}}(1-q_{\text{smoothed}})}{K+\alpha+\beta+1}}
$$

$$
CI=[q-z\cdot SE,\ q+z\cdot SE],\quad q\_ci\_width=CI_{high}-CI_{low}
$$

$$
q\_weight \propto 1-\min(q\_ci\_width,1)
$$

设计意图：
- 给后续训练一个“这条标签有多可靠”的可用信号；
- 不再把所有前缀标签一视同仁。

## 2.3 C1: 新增 clean-vs-corrupt 的 `pair_quality.jsonl`
代码位置：
- `scripts/phase_b_prepare_value_data.py`
  - `_build_corruption_rollout_targets_and_pair_quality(...)`

新增要点：
1. 对 corruption 前缀也 rollout 得到 `q_corrupt`
2. 和 clean 前缀 `q_clean` 做差，生成 pair 质量标签

核心公式：

$$
\Delta q = q_{clean} - q_{corrupt}
$$

$$
SE_{\Delta}=\sqrt{SE_{clean}^2+SE_{corrupt}^2}
$$

$$
z_{\Delta}=\frac{\Delta q}{\max(SE_{\Delta},10^{-8})}
$$

门控条件：

$$
pair\_pass = (\Delta q \ge \tau_{\Delta q}) \land (z_{\Delta}\ge \tau_{z})
$$

连续权重（实现里带裁剪和不通过 gate 的降权）：

$$
pair\_weight \sim f(\Delta q, z_{\Delta}) \times \sqrt{q\_weight^{clean}\cdot q\_weight^{corrupt}}
$$

设计意图：
- 把“这个 pair 值不值得学”显式化，而不是靠训练时瞬时分数猜。

## 2.4 C2: 训练端消费 CQR 标签
代码位置：
- `src/ours/phase_b/value_data.py`
- `scripts/phase_b_train_value.py`

新增要点：
1. 先选 `primary_corruption`
- 如果有 `pair_quality`，优先 `pair_pass_gate=true`，再看 `pair_weight/delta_q/z_delta`
- 否则回退到 deterministic 最小 `corruption_id`

2. calibration 分支可按可靠性加权
- `--calibration-sample-weighting q_weight` 或 `q_weight_parseable`

3. contrastive 分支可按 label quality 过滤
- `--contrastive-pair-filter label_quality`
- `--contrastive-pair-filter confidence_parseable_label`
- 可叠加 `--contrastive-require-pair-pass-gate`
- 可叠加 `--contrastive-use-pair-weights`

4. 新增分层采样（CQR-4）
- `--contrastive-stratified-sampling`
- 按 `(corruption_type, step_bucket)` 轮转取样

设计意图：
- 防止坏 pair 进入梯度；
- 防止训练 batch 被单一 corruption 类型主导。

## 3. 他们为什么要这样做
核心背景是之前问题：value head 学不到稳定信号（常见表现：pair 指标接近随机、校准不稳）。

CQR 的目标不是“马上提大分”，而是先把监督信号质量拉起来：
1. corruption 更有信息量；
2. 标签不确定性可见；
3. pair 质量显式可控；
4. C2 训练不再盲吃全量噪声样本。

## 4. 关键产物文件（你要优先看）
在 C1 目录（`assets/artifacts/phase_c_data/...`）：
- `corruptions.jsonl`：看 corruption 类型分布
- `rollout_targets.jsonl`：看 `q_mean_smoothed/q_ci_width/q_weight`
- `corruption_rollout_targets.jsonl`：看 corruption 侧质量
- `pair_quality.jsonl`：看 `delta_q/z_delta/pair_weight/pair_pass_gate`
- `summary.json`：看各摘要统计是否塌缩

在 C2 目录（`assets/artifacts/phase_c_runs/...`）：
- `train_metrics.json`
- `eval_metrics.json`
- `train_curve.jsonl`

## 5. 快速验证命令（CQR smoke）
```bash
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_CQR_SMOKE \
RUN_PREFIX=debug_cqr_smoke \
TRAIN_MAX_SAMPLES=128 \
EVAL_MAX_SAMPLES=64 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_c_value_suite.sh
```

## 6. 改参数时最容易出 bug 的点
1. `--corruption-selection-policy` 没切到 `cqr_balanced`
- 你以为跑 CQR，实际还是 legacy。

2. `--build-pair-quality` 没开
- C2 的 label-quality filter 没有可用标签。

3. `pair` 阈值过严
- `pair_pass_gate` 几乎全 false，contrastive 样本断流。

4. `rollout-count` 太小
- `q_ci_width` 普遍偏大，`q_weight` 偏低，标签噪声高。

5. C2 开了 label 过滤但 C1 没有对应字段
- 训练会极度稀疏或者直接失效。

## 7. 推荐排查顺序（先数据后模型）
1. 先看 `corruptions.jsonl`：语义类是否真的生成了。
2. 再看 `pair_quality.jsonl`：`delta_q` 分布和 `pair_pass_gate` 比例。
3. 再看 C2 启动日志：`pair_filter`、阈值、是否启用 pair weights。
4. 最后看 `eval_metrics.json` 的 calibration + corruption 两类指标。

## 8. 我建议你先盯的 4 个健康度指标
1. `pair_pass_gate` 比例（不要太接近 0）
2. `mean_pair_weight`（不要整体过低）
3. `rollout_targets` 的 `mean_q_ci_width`（过大说明噪声高）
4. C2 训练中 `avg_effective_contrastive_weight`（避免长期接近 0）

## 9. 关键代码修改入口（后续最常改）
1. C1 corruption 生成策略
- `src/ours/phase_b/corruptions.py`

2. C1 pair_quality 构建逻辑
- `scripts/phase_b_prepare_value_data.py`

3. C2 pair 过滤与加权
- `scripts/phase_b_train_value.py`

4. C2 primary corruption 选择策略
- `src/ours/phase_b/value_data.py`

## 10. 个人判断（当前版本）
优点：
- 链路完整，已经从“只做 corruption”升级到“有 label quality 与训练消费闭环”。

风险：
- 语义 corruption 仍偏规则驱动，任务覆盖有限；
- pair 阈值和 rollout 预算耦合很强，稍不当就会“信号变稀”；
- 最终上限仍取决于 teacher/更高质量标签引入（Phase D）。
