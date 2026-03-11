# 新数据集利用实验方案（2026-03-11）

**目的：** 定义如何利用新下载的外部数据集改善 Phase E value head 的 ProcessBench 迁移能力，每个方案都与当前已知 bottleneck 直接对应。

---

## 0. 现有数据集盘点（更新后）

| 数据集 | 路径 | 规模 | 已有 adapter | 当前使用状态 |
|---|---|---|---|---|
| Math-Shepherd | `peiyi_math_shepherd/math-shepherd.jsonl` | 757MB | ✅ | 主训练源 |
| PRM800K (OpenAI) | `openai_prm800k/prm800k/data/` | 943MB | ✅ 有但 near-random | 需 debug |
| RLHFlow Deepseek | `rlhflow_deepseek_prm/` | 350MB | ✅ | 已在 adapter |
| RLHFlow Mistral | `rlhflow_mistral_prm/` | 102MB | ✅ | 已在 adapter |
| PRMBench Preview | `hitsmy_prmbench_preview/` | 20MB | ✅ | 训练+评测 |
| ProcessBench | `qwen_processbench/` | 7.9MB | ✅ | 评测专用 |
| R-PRM | `kevinpro_r_prm/` | 676MB | ✅ | 已在 adapter |
| **MATH-APS** | `openreasoner_math_aps/` | 下载中 | ❌ 需新建 | **高优先级** |
| **Math-Step-DPO-10K** | `xinlai_math_step_dpo/` | 下载中 | ❌ 需新建 | **高优先级** |
| **EurusPRM-Stage2** | `prime_rl_eurus_prm_stage2/` | 下载中 | ❌ 需新建 | **高优先级** |
| **UltraInteract_pair** | `openbmb_ultrainteract_pair/` | 下载中 | ❌ 需新建 | 中期（跨域） |
| **GenPRM-MATH-Data** | `genprm_math_data/` | 下载中 | ❌ 需新建 | 中期（质量 ablation）|
| **trl-lib/prm800k** | `trl_prm800k_formatted/` | 下载中 | 参考 debug | PRM800K adapter debug 辅助 |

---

## 1. 当前 Bottleneck → 对应数据集映射

```
Bottleneck 1: ProcessBench later-bad 识别弱
  根本原因: Math-Shepherd 只有 single-trajectory first-bad-edge
            缺少 sibling-branch pairs（同一 prefix，不同 continuation）
  对应数据集: MATH-APS (OmegaPRM)
  → MCTS 二分搜索生成的 pair 天然包含 sibling-branch 语义

Bottleneck 2: PRM800K adapter near-random
  根本原因: phase1/phase2 JSONL 的 +1/0/-1 label 到 bool 的转换可能错误
            neutral (0) 步骤的处理方式影响 pair 构造
  对应数据集: PRM800K (已有) + trl-lib/prm800k（参考格式）
  → debug adapter，用 TRL 格式对照 openai 格式，修复转换逻辑

Bottleneck 3: terminal completion 仍弱（gated_mlp 后 terminal_top1 仍 < 0.5）
  根本原因: terminal anchor 比例调参不够，且 contrastive loss 和 BCE 混用不当
  对应数据集: EurusPRM-Stage2 (有完整正确解法 vs 截断前缀的 pair)
              Math-Step-DPO-10K (clean chosen/rejected format)
  → 替换或补充 terminal anchor 数据源，改用 BCE loss for terminal pairs

Bottleneck 4: 数据质量噪声（MC estimation 标注错误率 ~15-20%）
  根本原因: Math-Shepherd 用 MC estimation，存在 label noise
  对应数据集: GenPRM-MATH-Data (MC + LLM-judge 共识过滤，23K)
  → 小规模高质量 ablation：同等数量 GenPRM vs Math-Shepherd，看质量 vs 数量

Bottleneck 5: 跨域泛化（未来 StrategyQA / 通用推理）
  对应数据集: UltraInteract_pair (math + code + logic reasoning)
  → 引入逻辑推理 step-level pair，验证 BCR 跨域 claim
```

---

## 2. 实验方案矩阵

实验命名：`DS<tier><idx>_<描述>`
- `DS1` = 数据集 adapter debug
- `DS2` = 新数据源引入
- `DS3` = 混合配方优化
- `DS4` = 质量 vs 数量 ablation
- `DS5` = 跨域泛化

---

### 2.1 DS1 系列：PRM800K Adapter Debug（最高 ROI）

**目标：** 找出 PRM800K adapter 给 near-random 的根本原因并修复

#### DS1A: schema 对比诊断
```
步骤：
1. 加载 openai_prm800k/prm800k/data/phase1_train.jsonl 前 100 条
2. 对比 trl-lib/prm800k parquet 的前 100 条
3. 打印两者的 step label 分布（+1:0:-1 比例）
4. 检查 _extract_prm800k_completion_pairs() 的 label 处理逻辑

诊断命令：
python3 -c "
import json
from pathlib import Path

fp = Path('assets/external_datasets/openai_prm800k/prm800k/data/phase1_train.jsonl')
pos, neg, neu = 0, 0, 0
for i, line in enumerate(open(fp)):
    if i >= 200: break
    row = json.loads(line)
    steps = row.get('label', {}).get('steps', [])
    for step in steps:
        for comp in step.get('completions', []):
            r = comp.get('rating')
            if r == 1: pos += 1
            elif r == -1: neg += 1
            else: neu += 1
print(f'+1:{pos}, 0:{neu}, -1:{neg}')
"

诊断点：
- label.steps[].completions[].rating 是否为 +1/-1/null
- 是否有 finish_reason 字段影响最后一步的处理
- neutral (rating=null) 步骤：当前 adapter 如何处理？
  建议：rating=1 → positive, rating=-1 → negative, rating=null → skip（不构造 pair）
```

#### DS1B: adapter 修复运行
```
修复后：以 1000 pairs smoke 验证 PRM800K adapter 准确率
参数组：DS1B_PRM800K_ADAPTER_FIX_SMOKE
  source_type: prm800k
  max_pairs: 1000
  expected: same-source pair_acc >= 0.80（人工标注质量应高于 MC）
```

---

### 2.2 DS2 系列：新数据源引入

#### DS2A: MATH-APS adapter（直接解决 later-bad）

**预计效果：** sibling-branch pairs → later-bad pair_acc ↑ 10-15 点

```
数据集: openreasoner/MATH-APS
格式（待验证，需先运行 verify_external_datasets.py）:
  每条记录应包含:
    - problem / question
    - step annotations (MCTS estimated)
    - balanced positive/negative step labels

新建 adapter: src/ours/phase_d/external_pairs_adapters.py
  def load_math_aps_pairs(path, config) -> list[PairCandidate]:
    - 读取 MATH-APS 文件（parquet 或 jsonl）
    - 将 MCTS 步骤质量标注转为 first_bad_edge + sibling_branch pairs
    - 支持 step_label_pair_mode: "first_bad_edge" | "sibling_branch" | "both"

实验对比组：
  DS2A_MATH_APS_ONLY:
    source: math_aps, pairs=4096, mode=both, mlp head
    对比基准: ms_e43（Math-Shepherd same-source winner）
    观测：ProcessBench later-bad 指标，同源 pair_acc

  DS2A_MS_PLUS_MATH_APS:
    source: math_shepherd(60%) + math_aps(40%), pairs=8192
    观测：是否比纯 Math-Shepherd 的 later-bad 更好
```

#### DS2B: EurusPRM-Stage2 adapter（LLM-judge 质量替换 MC）

**预计效果：** 更干净的 first_bad_edge 信号 → ProcessBench AUC ↑ 3-5 点

```
数据集: PRIME-RL/EurusPRM-Stage2-Data
格式（待验证）:
  - "Step K:" 格式的步骤标注
  - LLM-judge 注入错误，chosen=正确解法，rejected=注入错误的解法

新建 adapter: load_eurus_prm_stage2_pairs()
  - 解析 "Step K:" 分段格式
  - 将 chosen/rejected 对转为 first_bad_edge pairs
  - 注入错误的步骤 = rejected 的 bad step

实验对比组：
  DS2B_EURUS_VS_MATHSHEPHERD:
    对比: EurusPRM-Stage2 (2048 pairs) vs Math-Shepherd (2048 pairs)
    head: mlp（完全可比）
    观测：同等数量下哪个在 ProcessBench 更好？
    假设：LLM-judge 质量 > MC，预期 EurusPRM 更好
```

#### DS2C: Math-Step-DPO-10K adapter（最干净的 pair 格式）

```
数据集: xinlai/Math-Step-DPO-10K
格式：
  {
    "question": str,
    "process": str,          ← chosen (correct full process)
    "answer": str,
    "wrong_step": int,       ← step index where error occurs
    "neg_process": str       ← process with error at wrong_step
  }

新建 adapter: load_math_step_dpo_pairs()
  - 从 wrong_step 提取 first_bad_edge pair
  - chosen_text = process[:wrong_step 结尾]（正确到 wrong_step-1）
  - rejected_text = neg_process[:wrong_step 结尾]（包含 wrong_step 的错误版本）

实验：
  DS2C_STEP_DPO_QUALITY_ABLATION:
    比较: Math-Step-DPO-10K (10K, GPT-4 标注) vs Math-Shepherd (10K 子集)
    核心问题: 在相同数量下，GPT-4 标注的质量优势能否弥补数量少的劣势？
```

---

### 2.3 DS3 系列：混合配方优化

基于 DS2 的 adapter 实现后，设计最优的多源混合配方：

#### DS3A: 最优同源数学混合
```
配方: MS-core(50%) + MATH-APS(30%) + EurusPRM-Stage2(20%)
意图: 兼顾 first-bad、later-bad、高质量标注三个维度
pairs: 16384, mlp head, 3 seeds
对比基准: PBR2 (纯 ms_align_v1)
观测: ProcessBench AUC, later-bad, terminal_top1 三项指标
```

#### DS3B: 引入 terminal anchor 层
```
配方: DS3A_base(80%) + terminal_anchor_pairs(20%)
  terminal anchor 来源: EurusPRM-Stage2 完整正确解法
  损失: BiRM 风格双损失 (local: contrastive, terminal: BCE)
pairs: 16384, mlp head, 3 seeds
观测: DS3A vs DS3B 的 terminal_top1 对比
```

---

### 2.4 DS4 系列：数据质量 vs 数量 Ablation

验证 Qwen 2025 "Lessons" 论文的核心结论：LLM-judge > MC estimation

```
DS4A_QUALITY_VS_QUANTITY:
  对比组:
    A: GenPRM-MATH-Data, 23K (MC+LLM-judge consensus)
    B: Math-Shepherd 随机子集 23K (pure MC)
    C: PRM800K phase1 23K 子集 (human annotation)
  head: mlp, seed=42（固定），其他配置完全相同
  观测:
    - 同源 pair_acc（各数据集 held-out）
    - ProcessBench AUC transfer
    - 重点: A>B 则证明 LLM-judge 过滤有效; C>A 则证明人工标注仍然更好

这组实验结果将直接指导后续数据策略：
  - 如果 A≈C: GenPRM 风格的 self-filtering 是最优性价比方案
  - 如果 C>>A>>B: 要么追求人工标注数据，要么用 LLM-judge 重新标注 Math-Shepherd
```

---

### 2.5 DS5 系列：跨域泛化（中期）

**前提：** DS2/DS3 的数学域实验稳定后再做

#### DS5A: UltraInteract 逻辑推理 step-level 引入
```
数据集: openbmb/UltraInteract_pair
子集: 只取 task_type in [logic, reasoning]（排除 math/code）
pairs: 4096 (smoke), mlp head

目标: 验证 BCR 论文中 StrategyQA 泛化能力的实验基础
  - 训练: UltraInteract 逻辑推理 pairs
  - 评测: ProcessBench（Math）as sanity check + 自定义逻辑推理 eval

注意: UltraInteract 的 step-level 质量取决于 tool/environment feedback
      对于纯推理链（非代码执行）的质量需要先做 schema 验证
```

---

## 3. 实施优先级和依赖关系

```
Phase 1（本周，需先完成下载 + verify）:
  DS1A → DS1B: PRM800K adapter debug
    前置: verify_external_datasets.py 通过
    预计时间: 0.5 天

  DS2C: Math-Step-DPO-10K adapter（最简单，格式最清晰）
    前置: 数据下载完成
    预计时间: 1 天（adapter + smoke run）

Phase 2（下周，基于 Phase 1 结论）:
  DS2A: MATH-APS adapter（需要先理解 MATH-APS 的实际格式）
    前置: DS2C 验证 adapter 框架，MATH-APS 下载 + schema 验证
    预计时间: 1-2 天

  DS2B: EurusPRM-Stage2 adapter
    前置: 数据格式验证
    预计时间: 1 天

  DS4A: 质量 vs 数量 ablation（结论重要，指导后续数据策略）
    前置: DS1B (PRM800K fix) + DS2A/B adapters
    预计时间: 0.5 天设计 + 运行

Phase 3（两周后）:
  DS3A/3B: 最优混合配方
    前置: DS2A + DS2B + DS4A 结论
    预计时间: 2-3 天

  DS5A: 跨域泛化（可选，根据论文需求决定）
```

---

## 4. Adapter 设计规范（统一接口）

所有新 adapter 应遵循 `external_pairs_adapters.py` 中现有的接口规范：

```python
def load_<dataset>_pairs(
    path: Path,
    config: PairBuildConfig,
    max_pairs: int | None = None,
) -> list[PairCandidate]:
    """
    加载 <数据集名称> 并转换为 Phase E pair 候选列表。
    Load <dataset> and convert to Phase E PairCandidate list.

    Args:
        path: 数据集本地路径 / Local dataset path
        config: pair 构造配置 / Pair build configuration
        max_pairs: 可选的最大 pair 数量上限 / Optional cap

    Returns:
        PairCandidate 列表，每个都有完整的 metadata 字段
        List of PairCandidate with complete metadata fields

    Key metadata fields to populate:
        source_tag: str           <- 数据集标识符
        pair_semantics: str       <- first_bad_edge / sibling_branch / terminal_completion_anchor
        positive_step_index: int  <- 正例步骤位置
        confidence: float         <- pair 质量置信度 (0-1)
    """
    ...
```

新 adapter 在 `dispatch_load_pairs()` 中注册：
```python
elif spec.source_type == "math_aps":
    rows = load_math_aps_pairs(spec.path, config, max_pairs)
elif spec.source_type == "eurus_prm_stage2":
    rows = load_eurus_prm_stage2_pairs(spec.path, config, max_pairs)
elif spec.source_type == "math_step_dpo":
    rows = load_math_step_dpo_pairs(spec.path, config, max_pairs)
```

---

## 5. 预期结论和决策树

```
实验结论 → 下一步决策：

DS1B (PRM800K fix):
  pair_acc >= 0.85 → PRM800K 数据质量验证，进入 DS4A
  pair_acc < 0.70  → 人工标注数据本身对当前 backbone/head 不友好，
                      可能需要 LoRA 解冻才能利用

DS4A (质量 vs 数量):
  GenPRM ≈ Math-Shepherd → 质量过滤对 frozen backbone 无法体现，说明 representation gap 更根本
  GenPRM > Math-Shepherd → 先做 VersaPRM 风格 self-filtering（低成本）再决定是否继续扩规模
  PRM800K >> GenPRM     → 人工标注价值极高，尝试找更多人工标注数据（SCDPO？）

DS2A (MATH-APS later-bad):
  later-bad ↑ >= 5 points → MATH-APS 有价值，加入主训练配方
  later-bad 无改善         → sibling-branch pairs 对 frozen backbone 同样无效，
                              确认 LoRA 解冻是必要路径

DS3A/3B (最优混合):
  ProcessBench AUC >= 0.65 → 冻结 backbone 达标，可进入 Phase F
  ProcessBench AUC < 0.65  → 进入 PBR6 (LoRA) 路线，数据问题已排除
```

---

## 6. 快速验证命令

下载完成后，运行验证脚本：
```bash
# 验证所有数据集完整性 / Verify all datasets
CUDA_VISIBLE_DEVICES="" python3 scripts/verify_external_datasets.py

# 快速 schema 预览 MATH-APS / Quick schema preview
python3 -c "
from datasets import load_dataset
d = load_dataset('openreasoner/MATH-APS', split='train', streaming=True)
print(next(iter(d)))
" 2>/dev/null

# 快速 schema 预览 Math-Step-DPO-10K
python3 -c "
from datasets import load_dataset
d = load_dataset('xinlai/Math-Step-DPO-10K', split='train', streaming=True)
r = next(iter(d)); print({k: str(v)[:80] for k, v in r.items()})
" 2>/dev/null
```

---

## 7. 与现有 PBR 实验组的整合

DS 系列实验**不替代** PBR2/PBR3，而是**并行运行**后在 DS3 中融合：

```
PBR2 (现有): ms_align_v1 + mlp + 16384 pairs, 3 seeds
  目的: 确认 frozen backbone 冻结上限

DS2A (新): MATH-APS adapter smoke (4096 pairs)
  目的: 验证 sibling-branch 是否有帮助

合并 DS3A: MS-core + MATH-APS + EurusPRM 混合 (16384 pairs)
  目的: 如果两者都有效，合并配方是否更优
```

若 PBR2 冻结上限确认 < 0.60，DS 系列实验仍有价值：
- 找出数据层面的最优配置，然后在 LoRA 路线（PBR6）上复用这个配置
- 避免在 LoRA 阶段同时面对数据问题 + 架构问题的 confounding

---

## 8. 参考论文（本实验方案依据）

- Math-Shepherd: [arXiv:2312.08935](https://arxiv.org/abs/2312.08935)
- OmegaPRM / MATH-APS: [arXiv:2406.06592](https://arxiv.org/abs/2406.06592)
- EurusPRM / PRIME: [arXiv:2412.01981](https://arxiv.org/abs/2412.01981)
- GenPRM: [arXiv:2504.00891](https://arxiv.org/abs/2504.00891)
- Step-DPO: [arXiv:2406.18629](https://arxiv.org/abs/2406.18629)
- BiRM (dual loss): [arXiv:2503.04618](https://arxiv.org/abs/2503.04618)
- VersaPRM (self-filtering): [arXiv:2502.06737](https://arxiv.org/abs/2502.06737)
- UltraInteract: [arXiv:2404.02078](https://arxiv.org/abs/2404.02078)
- The Lessons of PRMs: [arXiv:2501.07301](https://arxiv.org/abs/2501.07301)
