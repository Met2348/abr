# Phase D4A/B/C 外部 Pair 引导教程（新手版）

## 1. 这三个阶段在做什么

- `D4A`：只用高质量的**直接偏好对**（`R-PRM` + `PRMBench_Preview`）给 C2 做 ranking warm start。
- `D4B`：在 D4A 基础上加入**step-label 转 pair**数据（`Math-Shepherd` + `RLHFlow`），扩大 pair 覆盖。
- `D4C`：在 D4B 基础上降低外部权重、加强稳定性（two-stage），让 in-domain（StrategyQA 的 C1）仍然是主锚点。

一句话：
`D4A = 先把信号拉起来`，`D4B = 扩覆盖`，`D4C = 稳住并防 domain shift`。

---

## 2. 你需要先准备好的东西

### 2.1 C1 训练/评估目录（必须）

你需要两个目录（通常来自 D2 或 C1）：

- `PHASE_C_TRAIN_DIR`：包含 `prefixes.jsonl`、`rollout_targets.jsonl`、`manifest.json`
- `PHASE_C_EVAL_DIR`：包含 `prefixes.jsonl`、`rollout_targets.jsonl`、`manifest.json`

### 2.2 外部数据（建议放在默认路径）

脚本默认读取：

- `assets/external_datasets/kevinpro_r_prm`
- `assets/external_datasets/hitsmy_prmbench_preview/prmbench_preview.jsonl`
- `assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl`
- `assets/external_datasets/rlhflow_mistral_prm`
- `assets/external_datasets/rlhflow_deepseek_prm/deepseek_instruct_data.jsonl`

---

## 3. 一键跑 D4A+B+C（推荐先跑 smoke）

```bash
ACTIVE_PHASE_D4_GROUP=D4ABC_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_d4abc_smoke \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<your_train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<your_eval_dir> \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_d_external_pair_suite.sh
```

脚本会自动做三件事（每个 stage 都做一遍）：

1. 调 `scripts/phase_d_prepare_external_pairs.py` 产出外部 pair artifact。
2. 调 `scripts/phase_b_train_value.py` 用 external pair 分支训练 C2。
3. 调 `scripts/phase_b_eval_faithfulness.py` 做独立复评。

---

## 4. 只跑某一个阶段

### 4.1 只跑 D4A

```bash
ACTIVE_PHASE_D4_GROUP=D4A_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_d4a_smoke \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<your_train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<your_eval_dir> \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_d_external_pair_suite.sh
```

### 4.2 只跑 D4B

```bash
ACTIVE_PHASE_D4_GROUP=D4B_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_d4b_smoke \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<your_train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<your_eval_dir> \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_d_external_pair_suite.sh
```

### 4.3 只跑 D4C

```bash
ACTIVE_PHASE_D4_GROUP=D4C_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_d4c_smoke \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<your_train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<your_eval_dir> \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_d_external_pair_suite.sh
```

---

## 5. 结果去哪里看

统一看：

- `assets/artifacts/phase_d_logs/<RUN_PREFIX>/suite.log`
- `assets/artifacts/phase_d_logs/<RUN_PREFIX>/final_summary.md`

每个 stage 会记录三类产物路径：

- external pair 数据：`assets/artifacts/phase_d_external_pairs/<run_name>__<fp>/`
- C2 训练 run：`assets/artifacts/phase_c_runs/<run_name>_<timestamp>/`
- standalone eval：`assets/artifacts/phase_c_eval/<run_name>_<timestamp>/`

重点指标：

- `corr_pair_acc`
- `corr_auc`
- `brier_score`

---

## 6. 常用可调参数（先看这几个）

### 6.1 全局

- `C2_EPOCHS`
- `C2_TRAIN_BATCH_SIZE`
- `C2_EVAL_BATCH_SIZE`
- `C2_LR`

### 6.2 外部 pair 强度

- `D4A_EXTERNAL_PAIR_WEIGHT`
- `D4B_EXTERNAL_PAIR_WEIGHT`
- `D4C_EXTERNAL_PAIR_WEIGHT`

建议：先不要把 weight 调太大，避免外部数据把 in-domain 信号淹没。

### 6.3 每阶段附加参数（高级）

- `D4A_PAIR_PREP_EXTRA_ARGS`
- `D4B_PAIR_PREP_EXTRA_ARGS`
- `D4C_PAIR_PREP_EXTRA_ARGS`
- `D4A_C2_TRAIN_EXTRA_ARGS`
- `D4B_C2_TRAIN_EXTRA_ARGS`
- `D4C_C2_TRAIN_EXTRA_ARGS`

示例：

```bash
D4B_PAIR_PREP_EXTRA_ARGS="--max-pairs-total 6000 --min-pair-confidence 0.62"
```

---

## 7. 常见报错排查

### 7.1 找不到 C1 文件

报错通常是 `prefixes.jsonl` 或 `rollout_targets.jsonl` 缺失。

处理：

1. 检查 `PHASE_C_TRAIN_DIR` / `PHASE_C_EVAL_DIR` 是否指向正确目录。
2. 检查目录是否是 prepare 阶段的输出目录，而不是 run log 目录。

### 7.2 外部数据路径不存在

处理：

1. 用 `ls` 验证默认路径。
2. 若你放在别处，显式设置：
   - `R_PRM_ROOT=...`
   - `PRMBENCH_PREVIEW_PATH=...`
   - `MATH_SHEPHERD_PATH=...`
   - `RLHFLOW_MISTRAL_ROOT=...`
   - `RLHFLOW_DEEPSEEK_PATH=...`

### 7.3 训练很慢

优先改这三个：

1. `C2_TRAIN_BATCH_SIZE`
2. `C2_EVAL_BATCH_SIZE`
3. `D4*_MAX_PAIRS_TOTAL`（减少 external pair 数）

### 7.4 `pyarrow` 报 binary incompatibility

典型报错关键词：`IpcReadOptions size changed`、`binary incompatibility`。

处理：

```bash
python -m pip install -U pyarrow
```

如果你是 conda + pip 混装环境，建议在同一个环境里统一重装 `pyarrow`，避免 ABI 冲突。

---

## 8. 建议的执行顺序（团队协作版）

1. 先跑 `D4A_STRATEGYQA_SMOKE`（验证流程和路径）。
2. 再跑 `D4B_STRATEGYQA_SMOKE`（看扩覆盖是否真的涨指标）。
3. 再跑 `D4C_STRATEGYQA_SMOKE`（看稳定性和泛化风险）。
4. 只有 smoke 结论明确，再上 `D4ABC_STRATEGYQA_FULL`。
