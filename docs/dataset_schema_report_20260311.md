# 外部数据集 Schema 详细报告 / External Dataset Schema Detailed Report

*生成日期 / Generated: 2026-03-11*

---

## 0. 执行摘要 / Executive Summary

共 11 个数据集全部通过完整性校验。核心发现：

- **步骤级对监督** (直接可用于 PRM 训练): RLHFlow-Deepseek/Mistral, Math-Step-DPO-10K, trl-prm800k, Math-Shepherd, PRM800K
- **解决方案级二元标签** (需要 terminal-anchor 适配): MATH-APS, EurusPRM-Stage2
- **步骤级 DPO 对** (需要文本解析): UltraInteract_pair (Math_CoT 子集)
- **步骤级 critique 格式** (需要复杂解析): GenPRM-MATH-Data

关键颠覆性发现: **MATH-APS 不是步骤级数据集**——每行是完整解答+terminal 标签，且同一问题有多个不同解答尝试（~100 rows/question）。

---

## 1. 数据集逐一 Schema

### 1.1 MATH-APS (openreasoner/math_aps)

```
文件: math_aps.json, 196MB
行数: 152,293
唯一问题数: ~1,400 (10K行内92个唯一问题 → 约108次尝试/问题)
```

**字段结构:**
```python
{
  "question": str,          # 数学问题
  "process":  str,          # 完整解答过程 (多段落, 无Step N:标记)
  "label":    list[str]     # 始终是单元素列表: ['+'] 或 ['-']
}
```

**标签分布** (前1000行采样):
- `'+'`: 27.5%（正确解答）
- `'-'`: 72.5%（错误解答）

**重要注意**: 这是**解答级**二元分类数据，不是步骤级。`label` 始终只有1个元素。每道题有大量独立解答尝试（MCTS 采样所得）。适合用于 terminal-anchor 对：同一问题的 `+` 解答 vs. `-` 解答。

---

### 1.2 RLHFlow-Deepseek (rlhflow_deepseek_prm)

```
文件: deepseek_instruct_data.jsonl, ~350MB
行数: 252,950
```

**字段结构:**
```python
{
  "conversations": [
    {"role": "user",      "content": "问题\n\n步骤1文本\n\n"},
    {"role": "assistant", "content": "+"},   # 或 "-"
    {"role": "user",      "content": "步骤2文本\n\n"},
    {"role": "assistant", "content": "+"},
    ...
  ]
}
```

**特征:**
- 每行 = 一个完整多步解答，每个步骤都有 `+`/`-` 标签
- user 首轮包含问题 + 第1步；后续 user 轮仅含该步骤文本
- assistant 轮只有 `+` 或 `-`
- 这是最干净的步骤级标注格式

---

### 1.3 RLHFlow-Mistral (rlhflow_mistral_prm)

```
文件: mistral_data.jsonl, ~102MB
行数: 273,226
格式: 与 RLHFlow-Deepseek 完全相同
```

---

### 1.4 EurusPRM-Stage2 (prime_rl_eurus_prm_stage2)

```
文件: train.parquet, 15MB
行数: 30,102
```

**字段结构:**
```python
{
  "id":       str,
  "dataset":  str,       # "MATH" | "mathqa" | "gsm8k" | "numglue"
  "response": np.ndarray(shape=(2,), dtype=object),
              # [{"role": "user",      "content": "问题"},
              #  {"role": "assistant", "content": "Step 1: ...\nStep 2: ..."}]
  "label":    bool       # True = 正确解答, False = 错误解答
}
```

**数据集分布:**
- MATH: 25,034 (83%)
- mathqa: 3,024 (10%)
- gsm8k: 1,534 (5%)
- numglue: 510 (2%)

**注意**: 这是解答级二元标签。assistant 内容包含 "Step N:" 格式，但标签是整体正确性。背后用 LLM-judge 将正确解答中注入错误产生的数据。

---

### 1.5 Math-Step-DPO-10K (xinlai/math_step_dpo)

```
文件: train.parquet, 12MB
行数: 10,795
```

**字段结构:**
```python
{
  "prompt":               str,   # 数学问题
  "initial_reason_steps": str,   # 共有前缀步骤
  "chosen":               str,   # 分叉点之后的正确后续步骤
  "rejected":             str,   # 分叉点之后的错误后续步骤
  "full_chosen":          str,   # 完整正确解答
  "full_rejected":        str,   # 完整错误解答
  "dataset":              str,   # "MATH_Rephrased" 等
  "answer":               str    # 标准答案
}
```

**这是最适合的 first_bad_edge / sibling_branch 格式**: `chosen` 和 `rejected` 是从相同前缀（`initial_reason_steps`）分叉的正确和错误后续，完全对应实验设计中的 `sibling_branch` pair 语义。

---

### 1.6 UltraInteract_pair (openbmb/UltraInteract_pair)

```
文件: train.parquet, 248MB
行数: 219,522
```

**字段结构:**
```python
{
  "task":       str,   # "Coding" | "Math_CoT" | "Math_PoT" | "Logic"
  "dataset":    str,   # "MATH" | "codecontest" | "mathqa" | "gsm8k" | ...
  "chosen":     str,   # 完整正确轨迹，"Step 1: ...\nStep 2: ..." 格式
  "rejected":   str,   # 完整错误轨迹，同格式
  "trajectory": ...,   # 待查
  "id":         str,
  "parent_id":  str    # 同一 (question, wrong_step) 的同组
}
```

**子集大小:**
- Coding: 96,740
- Math_CoT: 57,231
- Math_PoT: 55,843
- Logic: 9,708

**Math 子集** (MATH+mathqa+gsm8k 合计 ~103K 行): chosen/rejected 是步骤级分叉对，但对比分叉发生在哪步需要通过 `parent_id` 或文本对比来确定。

---

### 1.7 GenPRM-MATH-Data

```
文件: train.parquet, 75MB
行数: 22,557
```

**字段结构:**
```python
{
  "conversations": np.ndarray of dicts,
  # 格式: system + (user-step, assistant-<analyze>) 交替轮
  # user 轮 = 一个解答段落
  # assistant 轮 = <analyze>...</analyze> + <conclusion>Correct/Wrong</conclusion>
}
```

**Conversation 模式** (以13轮为例):
```
Turn 0 [system]:    "You are a math teacher..."
Turn 1 [user]:      "Question: ...\n\nParagraph 1: ..."
Turn 2 [assistant]: "<analyze>...<conclusion>Correct</conclusion>"
Turn 3 [user]:      "Paragraph 2: ..."
Turn 4 [assistant]: "<analyze>...<conclusion>Correct</conclusion>"
...
Turn 11 [user]:     "Final paragraph: ..."
Turn 12 [assistant]: "<analyze>...<conclusion>Wrong</conclusion>"
```

**适配注意**: 需要解析 `<conclusion>` 标签来提取 Correct/Wrong 判断，作为步骤级标签。

---

### 1.8 trl-lib/prm800k (trl_prm800k_formatted)

```
文件: train.parquet + test.parquet, 2.2MB 合计
行数: 3,695 (train) + ? (test)
```

**字段结构** (TRL 标准格式):
```python
{
  "prompt":      list[dict],   # [{"role": "user", "content": "问题"}]
  "completions": list[str],    # 每个步骤的文本
  "labels":      list[bool]    # 每个步骤对应 True/False
}
```

**这是最易于适配的格式** — TRL PRM 训练直接使用此格式，与现有 value_head 训练 pipeline 对接最简单。

---

### 1.9 PRM800K (openai_prm800k)

```
文件:
  phase1_train.jsonl: 949 行, 7.9MB
  phase2_train.jsonl: 97,782 行, 456MB
原始格式说明见 OpenAI 论文
```

**字段结构:**
```python
{
  "question": {"problem": str, ...},
  "label": {
    "steps": [
      {"completions": [{"text": str, "rating": 1/0/-1, ...}], ...},
      ...
    ]
  }
}
```

- `rating`: `+1` = 正确, `0` = 中性/可能正确, `-1` = 错误
- 现有适配器 (`_collect_prm800k_files`) 效果近随机，需要修复

---

### 1.10 Math-Shepherd (peiyi9979)

```
文件: math-shepherd.jsonl, 793MB
行数: ~444,655
```

**字段结构:**
```
"问题 Step 1: 内容 ки\nStep 2: 内容 ки\n..."
标注：ки 后接 + 或 -
```

---

## 2. 适配优先级矩阵

| 数据集 | 步骤级? | 对格式 | 适配难度 | 推荐用途 |
|--------|---------|--------|----------|----------|
| Math-Step-DPO-10K | ✅ 分叉对 | `chosen`/`rejected` 分叉步 | **低** | `sibling_branch` |
| RLHFlow-Deepseek | ✅ 逐步 | conversations +/- | **低** | `first_bad_edge` |
| trl-prm800k | ✅ 逐步 | TRL 标准格式 | **极低** | 直接对接 |
| UltraInteract Math | ✅ 完整轨迹对 | chosen/rejected 文本 | **中** | `good_bad_prefix_grid` |
| EurusPRM-Stage2 | ❌ 解答级 | 全局 True/False | **中** | `terminal_completion_anchor` |
| MATH-APS | ❌ 解答级 | ['+'] / ['-'] | **中** | `terminal_completion_anchor` |
| GenPRM | ✅ 逐步 critique | `<analyze>` 轮次 | **高** | 步骤级标签提取 |
| PRM800K | ✅ 逐步 | +1/0/-1 | **低**(需修bug) | `first_bad_edge` |
| Math-Shepherd | ✅ 逐步 | ки 分隔符 | **低** | `first_bad_edge` |

---

## 3. 修订后的实验优先级

基于 schema 分析，调整 `new_dataset_experiment_plan_20260311.md` 中的执行顺序:

### 最高优先级 (直接可用)
1. **DS2C-v2**: Math-Step-DPO-10K → `sibling_branch` 对 (chosen/rejected 天然对齐)
2. **DS1B-fix**: PRM800K adapter bug 修复 (97K 高质量 human-annotated 步骤)
3. **trl-baseline**: trl-prm800k 直接接入现有 TRL pipeline (3.7K 行快速 smoke test)

### 中优先级 (需解析)
4. **DS2A-terminal**: MATH-APS `terminal_completion_anchor` 对 (同一问题 + vs - 解答)
5. **DS2B-terminal**: EurusPRM-Stage2 terminal 对 (LLM-judge 注入错误)
6. **DS5A-math**: UltraInteract Math_CoT 子集 (103K 对, 最大规模)

### 低优先级 (复杂解析)
7. **GenPRM**: `<analyze>` critique 解析 → 步骤标签

---

## 4. 关键设计决策

### MATH-APS 适配方案
```python
# 按 question 分组，取 + 和 - 各一个，构成 terminal_completion_anchor 对
groups = defaultdict(lambda: {"pos": [], "neg": []})
for row in data:
    q = row["question"]
    label = row["label"][0]
    groups[q]["pos" if label == "+" else "neg"].append(row["process"])

pairs = [(g["pos"][0], g["neg"][0]) for q, g in groups.items()
         if g["pos"] and g["neg"]]
# → 约 1,400 pairs (每个唯一问题一对)
```

### EurusPRM-Stage2 适配方案
```python
# response[1]["content"] 包含 "Step 1: ...\nStep 2: ..." 格式
# label 表示整体正确性; 同 id 的 True/False 对构成 terminal 对
# 但数据集本身未提供配对 id → 需要按 question 文本匹配
assistant_text = row["response"][1]["content"]
steps = re.split(r"(?=Step \d+:)", assistant_text)
is_correct = row["label"]  # True/False
```

### Math-Step-DPO-10K 适配方案 (最简单)
```python
# 直接使用 initial_reason_steps + chosen vs. rejected
prefix = row["initial_reason_steps"]
good_step = row["chosen"]    # 分叉后的正确步骤
bad_step  = row["rejected"]  # 分叉后的错误步骤
# → 直接构成 sibling_branch 对
```

---

*本文档由 Claude Code 自动生成，基于对所有数据集文件的 schema 探测*
