# Phase B Debug Playbook（实操版）

## 1. 目标
Phase B 调试要分两层看：
1. `B1/B2`：SFT/PEFT 是否真的学到并提升 benchmark。
2. `C1/C2`：value 监督数据质量与 value head 训练是否可靠。

## 2. 入口与调用关系
- Shell 入口：
  - `scripts/run_phase_b_training_suite.sh`
  - `scripts/run_phase_b_cross_task_suite.sh`
- Python 入口：
  - `scripts/phase_b_train_sft.py`
  - `scripts/phase_b_eval.py`
  - `scripts/phase_b_compare_eval.py`
  - `scripts/phase_b_checkpoint_sweep.py`
  - `scripts/phase_b_prepare_value_data.py`
  - `scripts/phase_b_train_value.py`
  - `scripts/phase_b_eval_faithfulness.py`

原则：
- 先调 Python 入口，确认逻辑无误后再回到 suite。

## 3. B1/B2 调试（SFT/PEFT）

### 3.1 最小命令
```bash
python -u scripts/phase_b_train_sft.py \
  --config-json configs/phase_b/peft_smoke_strategyqa.json \
  --run-name debug_b1 \
  --max-train-samples 64 \
  --max-eval-samples 32
```

### 3.2 关键断点
`phase_b_train_sft.py`：
1. `parse_args(...)` 结束（确认 config 覆盖是否生效）。
2. `load_phase_b_rows(...)` 后（看样本数量与字段）。
3. `_build_features(...)` 内（看 `input_ids/labels/loss_weights`）。
4. `_attach_lora_if_requested(...)`（确认是否真的走 peft）。
5. `trainer.train(...)` 前后。

必看变量：
- `effective_mode`
- `target_transform`
- `answer_weighting_mode`
- `train_features[0]`
- `manifest['training_args']`

### 3.3 常见坑
- 配置文件 key 拼错：会在 unknown-key 检查处报错。
- 以为在 PEFT，实际 fallback 到 SFT（看 `effective_mode`）。
- `max_seq_length` 太短，监督 token 被截断。

## 4. C1 调试（value 数据准备）

### 4.1 最小命令
```bash
python -u scripts/phase_b_prepare_value_data.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl \
  --run-name debug_c1 \
  --max-samples 64 \
  --build-corruptions \
  --build-rollouts \
  --rollout-count 4 \
  --batch-size 16 \
  --require-cuda
```

### 4.2 关键断点
`phase_b_prepare_value_data.py`：
1. `main(...)` 配置构建后（step/prefix/corruption config）。
2. `build_step_sequence_from_phase_b_row(...)`。
3. `build_prefix_artifacts(...)`。
4. `build_corruptions_for_prefixes(...)`。
5. `_build_rollout_targets(...)` 与 `_aggregate_rollout_targets(...)`。

必看变量：
- `num_prefixes`
- `corruption_type` 分布
- `rollout_targets[*].q_mean_smoothed/q_ci_width/q_weight`
- `pair_quality[*].delta_q/z_delta/pair_weight`

### 4.3 常见坑
- `resume` 命中导致你以为“新配置生效”但实际复用了旧产物。
- corruption 太弱（step_drop 主导），contrastive 信号很差。
- rollout K 太小，Q 标签方差大。

## 5. C2 调试（value head 训练）

### 5.1 最小命令
```bash
python -u scripts/phase_b_train_value.py \
  --train-dir assets/artifacts/phase_c_data/strategyqa/<train_dir> \
  --eval-dir assets/artifacts/phase_c_data/strategyqa/<eval_dir> \
  --run-name debug_c2 \
  --require-cuda \
  --per-device-train-batch-size 64 \
  --per-device-eval-batch-size 64 \
  --num-train-epochs 2 \
  --calibration-loss bce \
  --use-contrastive-loss
```

### 5.2 关键断点
`phase_b_train_value.py`：
1. `assert_phase_c_compatibility(...)`（先挡掉数据契约问题）。
2. `_encode_example_cache(...)`（看 features 和 pair 相关 tensor）。
3. `_run_one_train_epoch(...)`（看 calibration/contrastive 两支 loss）。
4. `_evaluate_value_head(...)`（看 raw/posthoc 指标）。

必看变量：
- `train_cache['targets']`
- `train_cache['has_primary_corruption']`
- `loss_cal`, `loss_ctr`, `effective_ctr_weight`
- `eval_metrics['calibration']`
- `eval_metrics['corruption']`

### 5.3 常见坑
- pair filter 太严导致对比样本几乎为 0。
- checkpoint selection metric 与 posthoc 配置不一致。
- 只看一个指标，不看 calibration 与 ranking 的 tradeoff。

## 6. standalone 复评（强烈建议）
```bash
python -u scripts/phase_b_eval_faithfulness.py \
  --value-run-dir assets/artifacts/phase_c_runs/<run_dir> \
  --eval-dir assets/artifacts/phase_c_data/strategyqa/<eval_dir> \
  --checkpoint-name best \
  --posthoc-calibration from_run
```

用途：
- 验证训练期最佳 checkpoint 的结果能否独立复现。

## 7. 推荐调试流程
1. B1 小样本跑通（确认训练链路）。
2. C1 小样本构建（确认 prefix/corruption/rollout target 质量）。
3. C2 小样本训练 + standalone eval。
4. 再回到 suite 跑完整参数组。

## 8. 每次调试记录 10 项
- train/eval artifact dir
- 是否 resume 命中
- rollout-count
- corruption policy
- calibration_loss
- lambda_contrastive
- pair filter 参数
- checkpoint selection metric
- raw/posthoc brier
- pair_accuracy/auc
