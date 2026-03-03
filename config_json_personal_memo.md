# Config JSON 个人备忘录（自用）

## 1. 这个备忘录管什么
- 目标：给自己一份“改 `configs/phase_b/*.json` 不容易翻车”的操作手册。
- 范围：当前仓库里，`configs/` 下真正生效的是 **Phase B** 配置。
- 现状：`configs/phase_c/` 目前是空目录，Phase C 主要靠 suite shell 脚本里的参数组和 extra args 控制，不走这里的 JSON。

## 2. 配置是怎么被加载的（非常关键）
入口脚本：`scripts/phase_b_train_sft.py`

核心机制：
- 先做一次“部分解析”，拿到 `--config-json` 路径。
- 读取 JSON 后，把 JSON 内容作为 parser defaults。
- 再解析完整 CLI。
- **CLI 显式传入的参数优先于 JSON**。
- JSON 里如果有未知 key，会直接报错退出（不会默默忽略）。

结论：
- 不能在 JSON 里随便加备注字段（比如 `_note`），会被当成未知 key 报错。
- JSON 不能写注释（标准 JSON 语法不支持）。
- 如果你想做“临时覆盖”，优先在命令行传参，不要改原始 JSON。

## 3. 当前配置文件全景
目录：`configs/phase_b/`，目前 18 个文件。

按用途分组：
- 基线：
  - `peft_smoke_strategyqa.json`
  - `peft_first_run_strategyqa.json`
  - `peft_full_strategyqa_cot.json`
  - `peft_full_gsm8k_cot.json`
- StrategyQA 诊断：
  - `peft_diag_strategyqa_epoch200.json`
  - `peft_diag_strategyqa_epoch300.json`
  - `peft_diag_strategyqa_lora_r8.json`
  - `peft_diag_strategyqa_lora_r32.json`
- GSM8K 诊断：
  - `peft_diag_gsm8k_cot_lr5e5.json`
  - `peft_diag_gsm8k_cot_lr1e4.json`
  - `peft_diag_gsm8k_cot_epoch025.json`
  - `peft_diag_gsm8k_cot_epoch050.json`
  - `peft_diag_gsm8k_direct_style.json`
  - `peft_diag_gsm8k_equation_style.json`
  - `peft_diag_gsm8k_short_cot.json`
  - `peft_diag_gsm8k_answer_weighted.json`
  - `peft_diag_gsm8k_checkpoint_sweep.json`
- GSM8K 修复组合：
  - `peft_repair_gsm8k_answer_weighted_ckpt.json`

## 4. 字段字典（按“最常改”优先）

### 4.1 数据与输入
- `train_jsonl` / `validation_jsonl`
  - 指向 Phase A prepared 数据。
  - 这是最容易“跑起来但结果不对”的来源（路径指错版本）。
- `max_train_samples` / `max_eval_samples`
  - `null` 表示全量；小整数用于 smoke。

### 4.2 模型与训练形态
- `model_path`
  - 当前都指向 `assets/models/Qwen2.5-7B-Instruct`。
- `training_mode`
  - `peft` 或 `sft`。
- `peft_fallback_to_sft`
  - `true` 时，PEFT 依赖异常会自动退回 SFT（可能让你“以为在跑 LoRA，实际上不是”）。
- `lora_rank` / `lora_alpha` / `lora_dropout`
  - 只有部分诊断配置显式写了（比如 r8/r32）。
  - 没写就用脚本默认值（rank=16, alpha=32, dropout=0.05）。

### 4.3 监督目标（GSM8K 常改）
- `target_transform`
  - 如 `none`、`gsm8k_short_cot_last2`。
- `target_max_reasoning_lines`
  - 配合 short-cot 变换。
- `answer_weighting_mode`
  - `none` / `final_answer_line`。
- `reasoning_loss_weight` / `answer_loss_weight`
  - 当 `final_answer_line` 开启时生效，用于调“推理token vs 最终答案token”损失比重。

### 4.4 优化与资源
- `learning_rate`
- `num_train_epochs`
- `max_steps`
  - `-1` 表示按 epoch；正数表示固定步数（smoke 常用）。
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `max_seq_length`
  - 典型值：512（smoke）或 1024（full）。

### 4.5 训练稳定与日志
- `auto_find_batch_size`
- `gradient_checkpointing`
- `dtype`（基本都是 `bfloat16`）
- `eval_steps` / `save_steps` / `logging_steps`
- `save_total_limit`
- `seed`
- `require_cuda`

## 5. “改哪里”速查（按目的）

### 5.1 我只想快速验证链路
- 基于 `peft_smoke_strategyqa.json`。
- 只动：`max_train_samples`、`max_eval_samples`、`max_steps`。
- 不要先动 loss/transform 类字段。

### 5.2 我想在同数据上做 LR ablation
- 固定 `train_jsonl`、`validation_jsonl`、`target_transform` 不变。
- 只复制并修改 `learning_rate`。
- 建议同时固定 `num_train_epochs` 与 batch 相关字段，保持可比性。

### 5.3 我想做 LoRA 容量 ablation
- 固定数据与优化参数。
- 只改 `lora_rank`，并同步合理改 `lora_alpha`。
- 不要同时改 `max_seq_length` / `learning_rate`，否则归因混淆。

### 5.4 我想验证“短CoT/答案加权”
- `target_transform` 与 `answer_weighting_mode` 二选一先单独试。
- 如果开 `final_answer_line`，务必显式写 `reasoning_loss_weight` 和 `answer_loss_weight`。

## 6. 高风险坑（务必看）
- 不要在 JSON 里加未知字段：会直接报错。
- 不要在同一次实验里同时改太多轴：很难解释提升来自哪里。
- `run_name` 虽然会被 suite 的 `--run-name` 覆盖，但建议 JSON 里也保持语义清晰，便于单独运行时追踪。
- `peft_fallback_to_sft=true` 会掩盖 PEFT 环境问题。做严谨对比时建议关注 manifest 里的 `effective_training_mode`。
- `max_seq_length` 改小可能造成监督截断，影响并不总是显性的。
- `train_jsonl` 指到不同 prepared 版本（哈希目录）会导致实验语义变化，先确认 prompt 样式一致。

## 7. 和 suite 分组的对应关系（方便反查）
来源：`scripts/run_phase_b_training_suite.sh` 的 `resolve_group()`

- `B1_SMOKE` -> `peft_smoke_strategyqa.json`
- `B1_FIRST` -> `peft_first_run_strategyqa.json`
- `B2_STRATEGYQA_FULL` -> `peft_full_strategyqa_cot.json`
- `B2_STRATEGYQA_DIAG_EPOCH_200` -> `peft_diag_strategyqa_epoch200.json`
- `B2_STRATEGYQA_DIAG_EPOCH_300` -> `peft_diag_strategyqa_epoch300.json`
- `B2_STRATEGYQA_DIAG_LORA_R8` -> `peft_diag_strategyqa_lora_r8.json`
- `B2_STRATEGYQA_DIAG_LORA_R32` -> `peft_diag_strategyqa_lora_r32.json`
- `B2_GSM8K_FULL` -> `peft_full_gsm8k_cot.json`
- `B2_GSM8K_DIAG_LR_5E5` -> `peft_diag_gsm8k_cot_lr5e5.json`
- `B2_GSM8K_DIAG_LR_1E4` -> `peft_diag_gsm8k_cot_lr1e4.json`
- `B2_GSM8K_DIAG_EPOCH_025` -> `peft_diag_gsm8k_cot_epoch025.json`
- `B2_GSM8K_DIAG_EPOCH_050` -> `peft_diag_gsm8k_cot_epoch050.json`
- `B2_GSM8K_DIAG_DIRECT_STYLE` -> `peft_diag_gsm8k_direct_style.json`
- `B2_GSM8K_DIAG_EQUATION_STYLE` -> `peft_diag_gsm8k_equation_style.json`
- `B2_GSM8K_DIAG_CHECKPOINT_SWEEP` -> `peft_diag_gsm8k_checkpoint_sweep.json`
- `B2_GSM8K_DIAG_SHORT_COT` -> `peft_diag_gsm8k_short_cot.json`
- `B2_GSM8K_DIAG_ANSWER_WEIGHTED` -> `peft_diag_gsm8k_answer_weighted.json`
- `B2_GSM8K_REPAIR_ANSWER_WEIGHTED_CKPT` -> `peft_repair_gsm8k_answer_weighted_ckpt.json`

## 8. 推荐个人工作流（稳）
- 第一步：复制最接近的 JSON，新文件名体现“唯一变化轴”。
- 第二步：只改 1-2 个字段，其他保持不动。
- 第三步：通过 suite 跑，保证前后 eval 口径一致。
- 第四步：检查输出 run 的 `manifest.json`，确认 `effective_training_mode`、transform、weighting 与预期一致。

## 9. 常用命令模板

### 9.1 直接用某个 JSON 启动训练
```bash
python -u scripts/phase_b_train_sft.py \
  --config-json configs/phase_b/peft_full_strategyqa_cot.json \
  --run-name my_phaseb_trial
```

### 9.2 用 CLI 临时覆盖 JSON（推荐做小改实验）
```bash
python -u scripts/phase_b_train_sft.py \
  --config-json configs/phase_b/peft_full_gsm8k_cot.json \
  --run-name my_lr_ablation \
  --learning-rate 1e-4
```

### 9.3 走 suite（推荐，带 pre/post 对比）
```bash
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_LR_1E4 \
RUN_PREFIX=my_diag_lr \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_b_training_suite.sh
```
