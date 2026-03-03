# Phase A Debug Playbook（实操版）

## 1. 目标
你在 Phase A 调试时，核心要回答 3 个问题：
1. 输入到底有没有样本（`num_inputs` 为什么可能是 0）？
2. 生成逻辑走了哪条分支（`decode_mode`、truncation、oom backoff）？
3. 评测口径怎么落到 `accuracy / parse_error`？

## 2. 入口与调用关系
- Shell 套件入口：`scripts/run_phase_a_benchmark_suite.sh`
- 真正 Python 入口：
  - `scripts/phase_a_prepare.py`
  - `scripts/phase_a_generate_and_eval.py`
  - `scripts/phase_a_eval_predictions.py`
  - `scripts/phase_a_analyze_instability.py`

推荐调试策略：
- 不直接调 `.sh`。
- 先把 `.sh` 展开成单条 Python 命令，再在 Python 入口打断点。

## 3. 最小可复现调试命令

### 3.1 Prepare（小样本）
```bash
python -u scripts/phase_a_prepare.py \
  --datasets strategyqa \
  --source-split train \
  --split-policy hash \
  --limit 20 \
  --template-id qa_direct \
  --template-version 1.0.0 \
  --target-style answer_only \
  --seed 42 \
  --overwrite
```

### 3.2 Generate+Eval（2 样本）
```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --run-name debug_phase_a \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --no-do-sample \
  --max-new-tokens 16 \
  --strategyqa-decode-mode freeform \
  --batch-size 1 \
  --max-samples 2 \
  --no-compare-latest-same-name
```

## 4. 断点优先级（按收益排序）

### 4.1 `scripts/phase_a_generate_and_eval.py`
1. 参数解析完成后（看最终 args）。
2. 输入 JSONL 读取后（看 `num_inputs` 与第一条样本）。
3. `_run_generation(...)` 内 batch 循环。
4. 调 `evaluate_predictions(...)` 前后。

必看变量：
- `args.input_jsonl`
- `records` / `num_inputs`
- `gen_config`（`max_new_tokens/do_sample/temperature`）
- `strategyqa_decode_mode`
- `raw_prediction`
- `extracted_prediction`、`parse_error`

### 4.2 `src/ours/phase_a/answer_extraction.py`
当 parse_error 异常高时，在 `extract_answer(...)` 路由与正则提取路径打断点。

### 4.3 `src/ours/phase_a/evaluator.py`
在 `score_prediction(...)` 打断点，确认：
- `gold_answer` 规范化是否符合预期。
- `is_correct` 判定路径是否走偏。

## 5. launch.json 用法（你现有配置可直接用）
你已有 `.vscode/launch.json` 的 Phase A 配置。
建议顺序：
1. 先点 `Phase A Generate+Eval: ... (2 samples)`。
2. 命中断点后，先看 `num_inputs`。
3. 再单步看第一条样本从 prompt 到 extracted answer 的全链路。

## 6. 常见问题速查

### 6.1 `num_inputs : 0`
优先检查：
1. `--input-jsonl` 是否为空文件或路径变量没展开。
2. 是不是拿错 split 文件（比如不存在的 validation fingerprint）。

### 6.2 速度非常慢
优先检查：
1. `batch-size=1` + `max-new-tokens` 太大。
2. 是否频繁进入 truncation recovery。
3. 是否多 GPU 分片导致额外开销（小模型反而慢）。

### 6.3 `parse_error_rate` 偏高
优先检查：
1. `strategyqa_decode_mode`（freeform vs binary_choice）
2. prompt 模板是否鼓励冗长自由输出。
3. `max-new-tokens` 是否截断关键答案段。

## 7. 推荐调试流程（固定模板）
1. 先用 `--max-samples 2` 跑通并命中断点。
2. 锁定 1 个样本，记录 `raw -> extracted -> is_correct`。
3. 扩到 20 样本，看错误类型分布。
4. 最后再全量跑，避免在大实验中盲调。

## 8. 每次调试要记的 8 项
- input_jsonl
- run_name
- decode_mode
- max_new_tokens
- batch_size
- do_sample
- parse_error_rate
- 对应 run_dir 的 `metrics.json`
