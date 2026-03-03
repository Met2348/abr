# Phase C Value Head 修复备忘（含外援策略）

> 文档目的：把当前价值头问题、可执行修复路线、以及“外援引入”策略沉淀成统一参考，避免后续反复讨论与试错。
>
> 日期：2026-03-03（本地仓库状态）

---

## 1. 问题本质：这是一个“冷启动弱监督”问题

当前任务不是普通分类，而是学习前缀价值函数：

$$
V_\phi(h_t) \approx \Pr(\text{最终答对} \mid h_t)
$$

但 step-level 真值不可见，当前标签来自 rollout Monte Carlo 估计：

$$
\hat q(h_t) = \frac{1}{K}\sum_{k=1}^{K}\mathbf{1}\{\text{rollout}_k \text{最终正确}\}
$$

这会带来两类核心噪声：

1. `Q 标签噪声`：$K$ 小时方差高、离散化严重（例如 $K=8$ 时只取 $\{0,0.125,\dots,1\}$）。
2. `Pair 噪声`：clean/corrupt 之间如果真实质量差很小，contrastive 排序近似随机。

结论：当前“价值头学不动”更像信号质量问题，不是单纯 loss 选错。

---

## 2. 仓库内当前实现（关键点）

### 2.1 C1: Q 标签生成

- 入口：`scripts/phase_b_prepare_value_data.py`
- 聚合位置：`_aggregate_rollout_targets`
- 当前核心字段：`success_rate`, `parseable_rate`, `k_rollouts`
- 现状：`success_rate = n_correct / k_rollouts`（未显式建模不确定度）

### 2.2 C1: corruption 生成与训练对接

- 生成：`src/ours/phase_b/corruptions.py`
- 默认：`max_corruptions_per_prefix = 1`
- 可选类型：`binary_flip`, `operator_flip`, `numeric_perturb`, `step_drop`
- 训练侧 primary 选择：`src/ours/phase_b/value_data.py` 中按 `corruption_id` 字典序选 1 个（与“质量”无关）

### 2.3 C2: 训练过滤逻辑

- 入口：`scripts/phase_b_train_value.py`
- 现有机制：
  - calibration sample weighting（`confidence/parseable/...`）
  - contrastive pair filter（`confidence/parseable/...`）
  - score-gap mining（基于当前模型分数差）
- 问题：过滤依赖当前模型打分，早期容易形成“错误自强化”。

---

## 3. 目前现象的结构性解释

已观测到的典型模式（结合当前产物）：

1. rollout 标签整体偏高（很多 prefix 接近 1.0）。
2. corruption 类型分布不均衡，`numeric_perturb` 占比高。
3. pair 指标（`pair_accuracy`, `auc_clean_vs_corrupt`）经常接近随机。
4. post-hoc calibration 有时能改善 brier，但不能保证排序能力提升。

这说明：

- calibration 后处理能修“概率刻度”，但不能凭空产生“区分能力”；
- 若训练 pair 本身弱，contrastive loss 学到的就是噪声排序。

---

## 4. 修 Q 标签质量：先做这 5 件事

### 4.1 用收缩估计替代裸均值

对每个 prefix，使用 Beta-Binomial 收缩：

$$
\tilde q = \frac{n + \alpha \mu_0}{K + \alpha}
$$

- $n$：正确 rollout 数
- $K$：rollout 总数
- $\mu_0$：全局先验均值（可用 train 全体 success mean）
- $\alpha$：先验强度（建议网格：`2, 4, 8`）

收益：降低小样本极值噪声。

### 4.2 显式记录标签不确定度

建议在 `rollout_targets.jsonl` 增加字段：

- `q_shrunk`
- `q_var`（近似）
- `q_ci_width`

近似可用：

$$
\mathrm{Var}(\tilde q)\approx \frac{\tilde q(1-\tilde q)}{K+\alpha+1}
$$

### 4.3 训练权重改为“方差感知”

当前权重主要看 `|q-0.5|`。应增加：

$$
w_i = \frac{1}{q_{var,i} + \tau}
$$

并做裁剪：

$$
w_i \leftarrow \mathrm{clip}(w_i, w_{\min}, w_{\max})
$$

### 4.4 rollout 预算改为两阶段

1. 全量 prefix 先跑 `K=8`。
2. 对不确定前缀（如 $|q-0.5|<0.2$ 或 CI 宽）补采样到 `K=24/32`。

这样比全量高 K 更省算力，且能显著提纯难例标签。

### 4.5 多采样温度混合

同一 prefix 的 rollout 使用两档温度（如 `0.5 + 0.8`）混合，降低单一采样模式偏差。

---

## 5. 修 Pair 质量：这是当前第一优先级

### 5.1 不要每个 prefix 只留一个 corruption

把 `--max-corruptions-per-prefix` 从 1 提升到 4（或 6），保留多种扰动。

### 5.2 primary corruption 选择改成“最大伤害”

现策略是字典序最小 `corruption_id`，应改为：

$$
c^\* = \arg\max_c \Delta Q_c,\quad \Delta Q_c = \tilde q_{\text{clean}} - \tilde q_{\text{corrupt},c}
$$

### 5.3 引入 pair margin 过滤

只保留高质量 pair：

$$
\Delta Q \ge m
$$

建议 `m` 从 `0.2` 起扫（`0.15/0.2/0.3`）。

### 5.4 控制 corruption 类型分布

训练时按 type 近似均衡采样，避免 `numeric_perturb` 垄断。

### 5.5 从“一对一”升级到“一对多”

每个 clean 对多个 corrupt，contrastive loss 采用平均或 hardest-k：

$$
\mathcal{L}_{ctr} = \frac{1}{|C_i|}\sum_{c\in C_i}\max(0, m - s_i^{clean} + s_{ic}^{corrupt})
$$

---

## 6. 外援方案（cold-start 引导）

### 6.1 外援的定位（关键）

外援应定位为 `cold-start labeler`，不是最终方法主体。

可用组合：

1. MC rollout（你们已有）
2. 强教师 judge/verifier（外部大模型或已有 PRM）
3. 共识过滤（只保留一致样本）

融合标签：

$$
\tilde q_i = \lambda q_i^{MC} + (1-\lambda) q_i^{teacher}
$$

共识权重：

$$
w_i = \mathbf{1}\{|q_i^{MC}-q_i^{teacher}|<\tau\}\cdot (1-\mathrm{Entropy}(\tilde q_i))
$$

### 6.2 可借鉴工作（高相关）

1. Let’s Verify Step by Step (2023): 大规模过程监督可行  
2. Math-Shepherd (ACL 2024): 自动过程标注 pipeline  
3. Tree-PLV (EMNLP 2024): tree search + pairwise verifier  
4. Rewarding Progress / PAV (ICLR 2025): 用 progress 代替粗 outcome  
5. ThinkPRM (2025): 生成式 verifier，低标签预算可用  
6. The Lessons of Developing PRMs (2025): PRM 数据噪声与泛化教训  
7. ProcessBench / PRMBench: 过程奖励模型泛化挑战

---

## 7. 引入外援会不会破坏创新性？

不会，前提是边界清楚。

### 7.1 不会破坏的做法

- 把外援用于 C1 标签提纯与冷启动。
- 论文/报告主贡献放在：
  1. 前缀价值建模设计
  2. pair 质量控制机制
  3. ABR/BCR 路由策略与效率-准确率权衡
  4. 同预算下的净收益

### 7.2 会稀释贡献的做法

- 直接把外部 PRM 当核心算法，不做自家方法增量。
- 缺少消融，导致无法证明收益来源。

### 7.3 必做消融

1. no external aid  
2. external aid only  
3. external aid + ours  
4. equal-budget 对比（同 token、同推理调用次数）

---

## 8. 可执行路线图（建议 2 周）

### 阶段 A（2-3 天）：数据审计

1. 统计每类 corruption 的数量、$\Delta Q$ 分布、通过 margin 比例。
2. 统计 `q_var` 与误差相关性。
3. 输出审计报告（必须先做，再训）。

### 阶段 B（4-5 天）：C1 提纯

1. 上线 `q_shrunk/q_var` 字段。
2. `max_corruptions_per_prefix >= 4`。
3. primary 选择改 `argmax \Delta Q`。
4. 生成“高质量 pair 子集”版本数据。

### 阶段 C（4-5 天）：C2 重训与对照

1. variance-aware calibration weighting。
2. margin-filtered contrastive。
3. 记录 raw 与 posthoc 全指标。
4. 跑固定 budget 的下游 rerank utility（必须有）。

---

## 9. 现有脚本可直接尝试的命令模板

### 9.1 仅通过现有参数先提升 pair 覆盖度（不改代码）

```bash
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_TRICK10_K16_COMBINED \
RUN_PREFIX=phase_c_try_more_corr \
PHASE_C_PREP_EXTRA_ARGS="--max-corruptions-per-prefix 4" \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_value_suite.sh
```

### 9.2 强化 pair 过滤（现有参数）

```bash
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_TRICK9_HARD_NEG_MINING \
RUN_PREFIX=phase_c_try_pair_filter \
PHASE_C_TRAIN_EXTRA_ARGS="--contrastive-pair-filter confidence_parseable --contrastive-confidence-threshold 0.3 --contrastive-parseable-threshold 0.85 --contrastive-score-gap-min 0.0 --contrastive-score-gap-max 0.2" \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_value_suite.sh
```

> 注意：以上仅是“立即可跑”尝试，真正关键仍是 C1 标签与 pair 构造逻辑升级。

---

## 10. 阶段门槛（建议沿用仓库 gate）

在进入更深 BCR/ABR 前，建议至少满足：

1. `brier_score` 稳定优于 trivial baseline  
2. `pearson` 稳定提升并跨 split 保持  
3. `auc_clean_vs_corrupt >= 0.60`  
4. `pair_accuracy >= 0.55`  
5. 下游 utility（rerank/router）有净收益

---

## 11. 参考文献与链接

1. Training Verifiers to Solve Math Word Problems (2021)  
   https://arxiv.org/abs/2110.14168
2. Solving Math Word Problems with Process- and Outcome-Based Feedback (2022)  
   https://arxiv.org/abs/2211.14275
3. Let’s Verify Step by Step (2023)  
   https://arxiv.org/abs/2305.20050
4. Math-Shepherd (ACL 2024)  
   https://aclanthology.org/2024.acl-long.510/
5. Tree-PLV (EMNLP 2024)  
   https://aclanthology.org/2024.emnlp-main.125/
6. Rewarding Progress / PAV (ICLR 2025)  
   https://proceedings.iclr.cc/paper_files/paper/2025/hash/98711dea460bdefe0e651ca23ec98ba2-Abstract-Conference.html
7. ThinkPRM (2025)  
   https://arxiv.org/abs/2504.16828
8. The Lessons of Developing PRMs (2025)  
   https://arxiv.org/abs/2501.07301
9. ProcessBench (2025)  
   https://arxiv.org/abs/2412.06559
10. Do We Need to Verify Step by Step? (ICML 2025)  
    https://proceedings.mlr.press/v267/jia25f.html
11. STaR (2022)  
    https://arxiv.org/abs/2203.14465
12. Constitutional AI (2022)  
    https://arxiv.org/abs/2212.08073
13. Self-Rewarding Language Models (2024)  
    https://arxiv.org/abs/2401.10020

---

## 12. 一句话结论

当前价值头瓶颈不在“再加一个损失”，而在“先把 C1 标签和 pair 信号做干净”。  
外援可用，且应当作为 cold-start 数据引擎；创新点应保持在你们自己的价值建模与路由控制机制上。

---

## 13. 社区现成 PRM Teacher（可直接当外援）

结论先说：有，且已经有多个可直接下载的开源模型；但它们多数在数学域训练，直接迁移到你们当前任务前，需要先做小规模可靠性筛查。

### 13.1 可用清单（2026-03-03 核对）

1. Qwen2.5-Math-PRM-7B（官方 PRM）  
   https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B
2. Qwen2.5-Math-7B-PRM800K（官方 PRM800K 版本）  
   https://huggingface.co/Qwen/Qwen2.5-Math-7B-PRM800K
3. Skywork-o1-Open-PRM-Qwen-2.5-7B（社区开源 PRM）  
   https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B
4. Math-Shepherd-Mistral-7B-PRM（Math-Shepherd 路线）  
   https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm
5. RLHFlow PRM（DeepSeek 数据路线）  
   https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Deepseek-Data
6. OpenR（完整流程+模型）  
   代码：https://github.com/openreasoner/openr  
   模型：https://huggingface.co/openreasoner/Math-psa

### 13.2 如何接入你们现有 C1/C2（最小改动）

1. 把 teacher 当成 `cold-start labeler`，用于 C1 的 `q_teacher` 字段。  
2. 用 `q_mc` 与 `q_teacher` 做一致性过滤，再生成 `q_fused` 与 `pair_quality`。  
3. C2 优先使用 `label_quality` / `q_weight` 路线训练，不把 teacher 当最终评测器。  

推荐融合式（与前文一致）：

$$
\tilde q_i=\lambda q_i^{MC}+(1-\lambda)q_i^{teacher}
$$

并保留一致性 gate：

$$
\mathbf{1}\{|q_i^{MC}-q_i^{teacher}|<\tau\}
$$

---

## 14. ProcessBench 与 PRMBench：能帮什么，不能帮什么

### 14.1 这两者是什么

1. ProcessBench（过程推理评测集）  
   论文：https://arxiv.org/abs/2412.06559  
   代码：https://github.com/QwenLM/ProcessBench  
   数据：https://huggingface.co/datasets/Qwen/ProcessBench
2. PRMBench（PRM 专项评测基准）  
   论文：https://arxiv.org/abs/2501.03124  
   代码：https://github.com/ssmisya/PRMBench  
   网站：https://prmbench.github.io/

### 14.2 它们对你们项目的实际价值

1. 能帮你做 `teacher 选型`：先在外部基准测几个候选 PRM，再决定谁进 C1 标注链路。  
2. 能帮你做 `回归防线`：每次改 C1/C2 后跑同一 benchmark，防止“局部指标涨、泛化下滑”。  
3. 能帮你做 `论文可信度`：证明你们不是只在单一内部 split 上有效。  

### 14.3 它们不能直接解决的问题

1. 它们不是你们数据的自动标注器，不能直接替代 C1 rollout/融合。  
2. 它们不能自动修复 `Q 标签噪声` 和 `低质量 pair`。  
3. 它们不能替你们完成“同预算下收益”的系统工程验证。  

---

## 15. 你们仓库近期 quality-first 动作（本地代码核对）

### 15.1 C1 新增“标签/配对质量”参数

位置：`scripts/phase_b_prepare_value_data.py`

1. `--build-pair-quality`
2. `--pair-rollout-count`
3. `--target-alpha`, `--target-beta`, `--target-ci-z`
4. `--target-weight-floor`, `--target-weight-gamma`
5. `--pair-delta-q-min`, `--pair-z-min`

对应产物侧新增字段（训练可消费）：`q_mean_smoothed`, `q_weight`，以及 pair 质量相关文件。

### 15.2 C2 新增“质量优先”训练入口

位置：`scripts/phase_b_train_value.py`, `src/ours/phase_b/value_data.py`

1. calibration weighting 新模式：`q_weight`, `q_weight_parseable`
2. pair filter 新模式：`label_quality`, `confidence_parseable_label`
3. 新阈值：`--contrastive-label-delta-q-min`, `--contrastive-label-z-min`, `--contrastive-label-pair-weight-min`
4. 可选：`--contrastive-use-pair-weights`, `--contrastive-require-pair-pass-gate`

### 15.3 Suite 层新增组

位置：`scripts/run_phase_c_value_suite.sh`

1. `C2_STRATEGYQA_QUALITY_FIRST`
2. `C2_STRATEGYQA_QUALITY_FIRST_FULL`

默认会打开 `--build-pair-quality` 和 label-quality 过滤链路，意图是先提升监督信号质量，再看校准与排序是否同时变好。

### 15.4 当前状态（本地检查）

1. 质量优先组已有日志：`assets/artifacts/phase_c_logs/phase_c_quality_first/suite.log`。  
2. 目前仍有 `run_phase_c_value_suite.sh` 在跑，暂未看到完整 quality-first 评测落盘。  
3. 单测 `tests/unit/test_phase_c_prepare_value.py` 与 `tests/unit/test_phase_c_value_components.py` 为 `14 passed`，但针对新 pair-quality 分支的专项测试仍应补齐。  

---

## 16. 外援+基准的推荐落地顺序

1. 先用 `ProcessBench/PRMBench` 做 teacher 候选筛查（只做评测，不改主流程）。  
2. 选 1-2 个 teacher 接入 C1，做 `q_mc + q_teacher` 融合与一致性 gate。  
3. 用 `QUALITY_FIRST` 训练组跑 C2，对比 `no external aid` / `external aid only` / `external aid + ours`。  
4. 最终只保留在“同预算条件下”能提升 `brier + auc_clean_vs_corrupt + pair_accuracy + 下游 utility` 的方案。  

---

## 17. 你已选 `Qwen2.5-Math-PRM-7B`：部署与接入结论

你当前的选择是合理的。  
建议策略是：

1. `Qwen2.5-Math-PRM-7B` 当主 teacher。  
2. `Qwen2.5-Math-7B-PRM800K` 作为对照 teacher（做消融）。  
3. `Qwen2.5-7B-Instruct` 继续做生成 backbone，不当 PRM teacher。  

官方链接（部署必看）：

1. Qwen2.5-Math-PRM-7B（模型卡）  
   https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B
2. Qwen2.5-Math-7B-PRM800K（模型卡）  
   https://huggingface.co/Qwen/Qwen2.5-Math-7B-PRM800K
3. Qwen PRM 博客（含训练与用法）  
   https://qwenlm.github.io/blog/qwen2.5-math-prm/

---

## 18. 怎么部署（按你当前仓库环境）

### 18.1 部署前检查

先在你平时跑实验的 conda 环境里执行（你之前是 `bcr`）：

```bash
conda activate bcr
python -V
python - <<'PY'
import importlib
mods = ["torch", "transformers", "accelerate", "huggingface_hub"]
for m in mods:
    mod = importlib.import_module(m)
    print(m, getattr(mod, "__version__", "unknown"))
PY
```

建议环境变量（与仓库 readme 一致）：

```bash
export HF_HOME=$PWD/assets/hf_cache
export HF_DATASETS_CACHE=$PWD/assets/hf_cache/datasets
export PYTHONPATH=$PWD/src
```

### 18.2 下载到仓库约定目录

当前本地 `assets/models/` 下只有 `Qwen2.5-7B-Instruct`，还没有 PRM 目录。  
建议下载到：

- `assets/models/Qwen2.5-Math-PRM-7B`

命令：

```bash
huggingface-cli login
huggingface-cli download Qwen/Qwen2.5-Math-PRM-7B \
  --local-dir assets/models/Qwen2.5-Math-PRM-7B \
  --local-dir-use-symlinks False
```

可选再下对照 teacher：

```bash
huggingface-cli download Qwen/Qwen2.5-Math-7B-PRM800K \
  --local-dir assets/models/Qwen2.5-Math-7B-PRM800K \
  --local-dir-use-symlinks False
```

### 18.3 本地 smoke test（确认模型可推理）

先做最小验证（只看能否得到 step 分数）：

```bash
CUDA_VISIBLE_DEVICES=1 python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "assets/models/Qwen2.5-Math-PRM-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
).eval()

question = "Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year?"
response = "Janet spends 40*3=120 on clarinet lessons per week. <extra_0> Janet spends 28*5=140 on piano lessons per week. <extra_0> Janet spends 140-120=20 more on piano lessons per week than clarinet lessons. <extra_0> She spends 20*52=1040 more on piano lessons than clarinet lessons in a year. <extra_0>"

input_text = f"{question} {response}"
input_id = tokenizer.encode(input_text, return_tensors="pt").to(model.device)[0]
candidate_tokens = tokenizer.encode(" - +")[1:]  # [minus_id, plus_id]
step_sep_id = tokenizer.encode("<extra_0>")[0]

token_ids, labels = [], []
for tid in input_id.tolist():
    token_ids.append(tid)
    labels.append(-100)
    if tid == step_sep_id:
        token_ids.extend(candidate_tokens)
        labels.extend([0, 1])

input_ids = torch.tensor([token_ids], device=model.device)
labels = torch.tensor([labels], device=model.device)
with torch.no_grad():
    logits = model(input_ids=input_ids, labels=labels).logits[:, :, 1]
scores = torch.sigmoid(logits).squeeze(0)
step_mask = input_ids.squeeze(0).eq(candidate_tokens[0])
step_scores = scores[step_mask].tolist()
print("num_steps =", len(step_scores))
print("step_scores =", [round(float(s), 4) for s in step_scores])
PY
```

说明：这段是按官方 PRM 用法改成“本地路径加载”版本；`step_scores` 出来即部署成功。

---

## 19. 当前 codebase 对“PRM 当 teacher”的支持度

### 19.1 已支持（基础很好）

1. C1 已有质量链路：`q_mean_smoothed/q_weight/pair_quality`。  
2. C2 已有质量过滤：`label_quality`, `q_weight_parseable` 等。  
3. 数据结构已有 `metadata` 可扩展，不需要破坏已有 JSONL 合约。  

### 19.2 还不支持（关键缺口）

1. 没有任何 `--teacher-*` 参数。  
2. C1 的 `--model-path` 语义是“生成 rollout 的 CausalLM”，不是 PRM 打分器。  
3. C2 目标固定读 `target_q_mean_smoothed`，没有 `q_fused` / `q_teacher` 目标源切换。  
4. suite 脚本默认把 C1 `--model-path` 固定成 `assets/models/Qwen2.5-7B-Instruct`。  

代码定位：

1. C1 CLI（暂无 teacher 参数）：`scripts/phase_b_prepare_value_data.py`  
2. suite 固定 model-path：`scripts/run_phase_c_value_suite.sh`  
3. C2 数据加载（当前只吃 rollout target）：`src/ours/phase_b/value_data.py`  
4. C2 训练 target 来源：`scripts/phase_b_train_value.py`  

---

## 20. 为了“真正当 teacher”，要补哪些代码

建议按最小侵入分 4 步补齐。

### 20.1 M0：新增 teacher 打分脚本（独立于 C1）

新增：

- `scripts/phase_c_score_prm_teacher.py`

输入：

1. `prefixes.jsonl`（clean）  
2. `corruptions.jsonl`（corrupt，可选）  
3. `--teacher-model-path assets/models/Qwen2.5-Math-PRM-7B`

输出：

1. `teacher_prefix_scores.jsonl`
2. `teacher_corruption_scores.jsonl`

字段建议：

1. `prefix_id` / `corruption_id`
2. `teacher_score_mean`
3. `teacher_score_min`
4. `teacher_num_steps`
5. `teacher_model_id`
6. `teacher_infer_meta`

### 20.2 M1：C1 合并 teacher 与 MC 标签

改：

- `scripts/phase_b_prepare_value_data.py`

新增参数：

1. `--teacher-prefix-scores-path`
2. `--teacher-corruption-scores-path`
3. `--teacher-fusion-lambda`（$\lambda$）
4. `--teacher-disagree-threshold`（$\tau$）

新增产物字段（写入 rollout/corruption target）：

1. `q_teacher`
2. `q_fused`
3. `teacher_disagree`
4. `teacher_available`

pair_quality 计算建议优先用 `q_fused`：

$$
\Delta Q = q^{fused}_{clean} - q^{fused}_{corrupt}
$$

### 20.3 M2：C2 支持目标源切换

改：

- `src/ours/phase_b/value_data.py`
- `scripts/phase_b_train_value.py`

新增参数：

1. `--target-source {q_mean_smoothed,q_fused,q_teacher}`
2. `--require-teacher-coverage`

并在 `train_cache` 里显式记录：

1. `target_source_name`
2. `teacher_coverage`

### 20.4 M3：suite 与测试补齐

改：

1. `scripts/run_phase_c_value_suite.sh`：增加 teacher 相关环境变量透传。  
2. `tests/unit/test_phase_c_prepare_value.py`：覆盖 teacher 融合字段。  
3. `tests/unit/test_phase_c_value_components.py`：覆盖 `target-source` 切换。  

最小验收标准：

1. 无 teacher 路径时，旧流程完全不变（回归兼容）。  
2. 有 teacher 路径时，`summary.json` 里出现 teacher 覆盖率。  
3. C2 训练日志明确打印当前 target source。  

---

## 21. 先跑什么（你今天可执行）

1. 先部署并 smoke-test `Qwen2.5-Math-PRM-7B`（18 节）。  
2. 再补 `M0` 脚本，先只打分不训练。  
3. 用一个 smoke split 验证 `q_teacher` 与 `q_mc` 的相关性和分歧率。  
4. 分歧率可控后，再做 `M1/M2` 合并进 C2。  
