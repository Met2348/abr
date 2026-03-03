# Phase C Debug Playbook（实操版）

## 1. 目标
Phase C 调试建议拆成 3 条线：
1. `value_suite`：`C1(prefix/corruption/rollout)` -> `C2(value head)` -> `standalone eval`。
2. `pik_suite`：问题级 `P(IK)` 的 `C1 -> C2 -> eval`（冷启动诊断主线）。
3. `teacher sidecar`：给已有 C1 产物打外部 PRM 分数（D1）。

你要优先回答 4 个问题：
1. C1 标签是否真的有信号（不是全噪声）？
2. C2 训练是否在学（不是配置/数据对不上）？
3. eval 指标是否来自独立复评（不是训练日志幻觉）？
4. teacher 打分是否语义正确（分数不是“看起来有值”而已）？

## 2. 入口与调用关系

### 2.1 Value Suite（Phase C 主线）
- Shell 入口：`scripts/run_phase_c_value_suite.sh`
- 实际调用：
  - `scripts/phase_b_prepare_value_data.py`（C1）
  - `scripts/phase_b_train_value.py`（C2）
  - `scripts/phase_b_eval_faithfulness.py`（standalone）

说明：虽然脚本名是 `phase_b_*`，但它们在这里就是 Phase C value 流程的一部分。

### 2.2 P(IK) Suite（问题级诊断）
- Shell 入口：`scripts/run_phase_c_pik_suite.sh`
- 实际调用：
  - `scripts/phase_c_prepare_pik_data.py`（C1）
  - `scripts/phase_c_train_pik.py`（C2）
  - `scripts/phase_c_eval_pik.py`（standalone）

### 2.3 Teacher Sidecar（D1）
- Python 入口：`scripts/phase_c_score_prm_teacher.py`
- 读取已有 `prefixes/corruptions`，写 teacher 分数 sidecar 文件。

## 3. 最小可复现调试命令

### 3.1 Value Suite（小规模）
```bash
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_SMOKE \
RUN_PREFIX=debug_c_value \
TRAIN_MAX_SAMPLES=32 \
EVAL_MAX_SAMPLES=16 \
ROLLOUT_COUNT=2 \
ROLLOUT_BATCH_SIZE=32 \
C2_EPOCHS=1 \
C2_TRAIN_BATCH_SIZE=32 \
C2_EVAL_BATCH_SIZE=32 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_c_value_suite.sh
```

### 3.2 P(IK) Suite（小规模）
```bash
ACTIVE_PHASE_C_PIK_GROUP=PIK_STRATEGYQA_SMOKE \
RUN_PREFIX=debug_c_pik \
TRAIN_MAX_SAMPLES=64 \
EVAL_MAX_SAMPLES=32 \
ROLLOUT_COUNT=4 \
ROLLOUT_BATCH_SIZE=32 \
C2_EPOCHS=2 \
C2_TRAIN_BATCH_SIZE=64 \
C2_EVAL_BATCH_SIZE=64 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_c_pik_suite.sh
```

### 3.3 Teacher Sidecar（先只打 prefix）
```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_c_score_prm_teacher.py \
  --phase-c-dir assets/artifacts/phase_c_data/strategyqa/<your_c1_dir> \
  --teacher-model-path assets/models/Qwen2.5-Math-PRM-7B \
  --batch-size 32 \
  --max-length 2048 \
  --score-corruptions false \
  --require-cuda
```

## 4. 断点优先级

### 4.1 Shell 编排层（先看是否走对阶段）
- `scripts/run_phase_c_value_suite.sh`
  - `resolve_group`
  - `run_c1_prepare`
  - `CURRENT_STAGE` 更新点（`c1_prepare_*`, `c2_train`, `c2_eval`, `final_summary`）
- `scripts/run_phase_c_pik_suite.sh`
  - 同样看 `resolve_group / run_c1_prepare / CURRENT_STAGE`

重点：若中断，先看 `suite.log` 里最后一个 `CURRENT_STAGE`。

### 4.2 PIK C1：`scripts/phase_c_prepare_pik_data.py`
- `main` 中 `load_phase_b_rows(...)` 之后
- `_build_question_record(...)`
- `_build_rollout_targets(...)`
- `_aggregate_pik_targets(...)`

必看变量：
- `row_summary`
- `question_fallback_used`
- `rollout_count / do_sample / temperature`
- `target_success_rate / target_parseable_rate`

### 4.3 PIK C2：`scripts/phase_c_train_pik.py`
- `assert_phase_c_pik_compatibility(...)`
- `_encode_example_cache(...)`
- `_run_one_train_epoch(...)`
- `_evaluate_pik_head(...)`

必看变量：
- `train_cache["targets"]` 分布
- `known_threshold` 下 `known_labels` 比例
- `best_eval_selection_brier`
- `calibration_posthoc` 是否实际生成

### 4.4 PIK standalone：`scripts/phase_c_eval_pik.py`
- `main` 里 manifest 兼容性检查后
- `_resolve_posthoc_payload(...)`
- `_encode_text_list(...)`

必看变量：
- `checkpoint_path`（best 是否回退 final）
- `posthoc_calibration` 模式
- `metrics["calibration"]` 与 `metrics["calibration_posthoc"]`

### 4.5 Teacher Sidecar：`scripts/phase_c_score_prm_teacher.py`
- `_resolve_input_paths(...)`
- `_build_prefix_teacher_text(...)` / `_build_corruption_teacher_text(...)`
- `_score_batch(...)`
- `_extract_step_scores_from_probs(...)`

必看变量：
- `sep_ids` 长度（必须是 1）
- `teacher_input_text`
- `teacher_step_scores`
- `teacher_score_mean`

## 5. 输出文件要看哪里

### 5.1 PIK C1 输出目录
`assets/artifacts/phase_c_pik_data/<dataset>/<run_name>__<fingerprint>/`

重点文件：
- `questions.jsonl`
- `rollout_predictions.jsonl`
- `pik_targets.jsonl`
- `summary.json`
- `manifest.json`

### 5.2 PIK C2 输出目录
`assets/artifacts/phase_c_pik_runs/<run_name>_<timestamp>/`

重点文件：
- `train_metrics.json`
- `eval_metrics.json`
- `train_curve.jsonl`
- `eval_question_scores.jsonl`
- `best_value_head.pt`
- `final_value_head.pt`

### 5.3 PIK standalone eval 输出目录
`assets/artifacts/phase_c_pik_eval/<run_name>_<timestamp>/`

重点文件：
- `metrics.json`
- `question_scores.jsonl`
- `summary.md`

### 5.4 Teacher sidecar 输出（写回 C1 目录）
- `teacher_prefix_scores.jsonl`
- `teacher_corruption_scores.jsonl`（若开启）
- `teacher_errors.jsonl`
- `teacher_summary.json`
- `teacher_summary.md`

## 6. 常见问题速查

### 6.1 以为改了参数，结果没变
原因：命中 `resume`（尤其 C1 prepare）。

处理：
1. 改 `run-name` 或开 `--overwrite`。
2. 检查目录名里 fingerprint 是否变化。

### 6.2 P(IK) 的 AUC/Brier 看起来异常
先查：
1. `known_threshold` 是否导致标签几乎全 0 或全 1。
2. `rollout_count` 太小导致目标噪声大。
3. `posthoc` 是否被错误应用/未应用。

### 6.3 `--posthoc-calibration from_run` 报错
原因：对应 `best_posthoc_calibration.json` 或 `final_posthoc_calibration.json` 不存在。

### 6.4 Teacher 报 step separator 错
原因：`--step-separator-token` 映射到多个 token，当前脚本要求单 token。

### 6.5 Teacher 分数“有值但不可信”
常见根因：
1. `teacher_input_text` 构建错（问题或步骤抽取退化到 fallback）。
2. `teacher_num_steps` 过小（步骤分割失败）。
3. 只看均值，没看 step-level 分布和错误行。

## 7. 推荐调试顺序（固定模板）
1. 先跑 `PIK C1` 小样本，确认 `pik_targets.jsonl` 有合理分布。
2. 再跑 `PIK C2` 1-2 epoch，确认 loss 和 eval 指标在变化。
3. 再跑 `phase_c_eval_pik.py` 做独立复评，确认训练结论可复现。
4. 在同一个 C1 目录跑 teacher sidecar，检查 `teacher_*` 输出。
5. 最后再上 `run_phase_c_value_suite.sh` 的完整组。

## 8. 每次调试至少记录 12 项
- group id（`ACTIVE_PHASE_C_GROUP` 或 `ACTIVE_PHASE_C_PIK_GROUP`）
- run prefix
- C1 train/eval dir
- C2 run dir
- rollout_count
- rollout_max_new_tokens
- calibration_loss
- posthoc_calibration
- checkpoint_selection_metric
- known_threshold
- final metrics path
- suite.log path 与失败 stage
