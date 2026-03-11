# BCR Phase E: 仓库现状、论文对照与下一步实验结论

**日期**: 2026-03-11  
**目的**: 对队友新增代码、文档、实验结果和本地论文 PDF 做一次严格复核，形成当前可信结论、社区对照、差距、反模式和下一步实验建议。  
**本次只采信四类证据**: 代码、可运行测试、实际 artifact、PDF 原文。已有 Markdown 调研文档只作为线索，不直接视为结论。

---

## 补充：Judge LLM 本地选型与可用性

在本次综合复核之后，又单独补做了一轮 `LLM-as-a-judge` 本地落地检查。结论单独记录在：

1. `docs/phase_e_judge_llm_selection_20260311.md`

当前最重要的工程判断是：

1. `Qwen2.5-Math-7B-Instruct` 已经是可接入的 bulk local judge 候选；
2. `DeepSeek-R1-Distill-Qwen-14B` 已安装，但在当前本地 `transformers + strict JSON judge prompt` 链路下仍不稳定；
3. 因此 judge 主线应先围绕 `Qwen2.5-Math-7B-Instruct` 展开，而不是直接把更强 reasoning model 拉进主训练环。

---

## 0. 审计范围与证据边界

### 0.1 本次实际核对了什么

1. **代码**
   - `src/ours/phase_e/training.py`
   - `src/ours/phase_e/processbench_alignment.py`
   - `src/ours/phase_b/value_head.py`
   - `scripts/phase_e_curate_processbench_transfer_pairs.py`
   - `scripts/run_phase_e_processbench_research_suite.sh`
   - `scripts/run_phase_e_curated_rlready_suite.sh`
   - `scripts/phase_e_eval_benchmark.py`
   - `scripts/verify_external_datasets.py`

2. **实验产物**
   - `ms_e43`
   - `prm_e46`
   - `e87_repair`
   - `c3_curated_gated`
   - `c4_dual`
   - `pbr2_align_gated`
   - 以及对应的 `samefamily_eval / transfer_compare / rl_promotion_diag / scored_rows.jsonl`

3. **本地 PDF 原文**
   - `PRM800K`
   - `Math-Shepherd`
   - `OmegaPRM`
   - `Tree-PLV`
   - `Rewarding Progress`
   - `ProcessBench`
   - `PRMBench`
   - `The Lessons of Developing PRMs`
   - `PRIME`
   - `VersaPRM`
   - `BiRM`
   - `R-PRM`
   - `GenPRM`
   - `ActPRM`
   - `Stop Summation / PURE`
   - `ThinkPRM`
   - 以及与新下载数据集直接相关的 `Step-DPO / SCDPO / Full-Step-DPO / UltraInteract-Eurus`

4. **可运行性验证**
   - `python -m py_compile` 覆盖 Phase E 新增主脚本和核心模块
   - `pytest` 覆盖新增 curate / alignment / training / benchmark eval / value head 测试
   - `bash -n` 覆盖新的 suite shell 入口

### 0.2 这次真正确认的结果

- 新增 Phase E 代码当前**可运行**，不是纯文档先行。
- 但多份现有文档里存在**超前解释**、**指标口径混用**和**代码状态过期描述**。
- 队友说“修复了 ProcessBench 一些表现”这句话只能算**部分成立**：某些 slice 改善了，但没有形成新的稳定主线，也没有接近社区强基线。

---

## 1. 当前仓库真实状态

## 1.1 最新主线不是一个单一 recipe，而是一组 tradeoff probe

当前 Phase E 新代码不是围绕一个已经稳定的主方案展开，而是围绕以下几类问题做局部探针：

1. `ProcessBench` 几何对齐
2. terminal anchor 是否必要、比例多大
3. `mlp / gated_mlp / dual_head` 哪种头更适合
4. `same-family trust` 与 benchmark transfer 是否同时成立
5. `RL promotion gate` 什么时候才算过线

这意味着当前仓库更像：

- **高质量诊断平台**
- 还不是**接近社区上限的 verifier 系统**

## 1.2 代码上已经真正落地的关键能力

### 已落地

1. **保守 curate profile**
   - `ms_core_v1`
   - `ms_align_v1`
   - `ms_prm_align_v1`
   - `ms_laterbad_v1`

2. **语义权重 / 分组平衡**
   - `semantic`
   - `confidence_semantic`
   - `group_balance`
   - `confidence_group_balance`

3. **多种 head**
   - `linear`
   - `mlp`
   - `gated_mlp`
   - `dual_head`

4. **pair 语义路由**
   - `training.py` 已实现 `compute_pair_route_weights()` 和 `_resolve_pair_training_route_weights()`
   - `terminal_completion_anchor` 会路由到 terminal branch
   - `first_bad/fanout/local_modified_process_error_step` 会路由到 local branch

### 还没有真正落地

1. **LoRA / backbone adaptation**
   - 当前仍是 frozen-backbone regime
   - 没有 on-the-fly LoRA 训练路径

2. **MATH-APS / Eurus / GenPRM / UltraInteract 正式 adapter**
   - `verify_external_datasets.py` 能看到这些目录
   - 但 `external_pairs_adapters.py` 里没有对应训练 adapter
   - 所以这些数据大多仍停留在“已下载”，不是“已接入主线”

3. **高质量数据过滤闭环**
   - 还没有 LLM-judge / consensus filtering
   - 还没有真正的 self-filtering retrain loop

## 1.3 一处明显的代码-文档漂移

`scripts/run_phase_e_processbench_research_suite.sh` 里仍写着：

- `PBR5_DUAL_HEAD_ROUTING_SEED3` blocked
- 需要先实现 pair routing

但当前代码里 routing 已经存在，且 `c4_dual` 也已经实际跑过。  
这说明：

- 文档和 suite 注释里有一部分状态说明已经过期
- 后续阅读时不能只看 plan 文本，必须反查代码和 artifact

---

## 2. 当前实验结果的严格结论

## 2.1 用现有 artifact 回算官方 `ProcessBench F1` 后，最好的也只有约 `0.35`

这是本次最重要的新发现。  
仓库大多数历史汇总文档用的是：

- `pair_acc`
- `AUC`
- `first_edge`
- `terminal_top1`

但论文和社区比较主要看的是 `ProcessBench F1`。我用当前代码的 `compute_processbench_f1()` 从已有 `scored_rows.jsonl` 直接回算了关键 run：

| case | gsm_f1 | gsm_acc_err | gsm_acc_correct | math_f1 | math_acc_err | math_acc_correct |
|---|---:|---:|---:|---:|---:|---:|
| `ms_e43` | 0.2294 | 0.1836 | 0.3057 | 0.1853 | 0.1330 | 0.3054 |
| `prm_e46` | **0.3493** | 0.2560 | 0.5492 | **0.2970** | 0.2239 | 0.4409 |
| `e87_repair` | 0.1959 | 0.1400 | 0.3261 | 0.2074 | 0.1228 | 0.6667 |
| `c3_curated_gated` | 0.2485 | 0.2273 | 0.2742 | 0.2436 | 0.1711 | 0.4231 |
| `c4_dual` | 0.3185 | 0.2273 | 0.5323 | 0.2432 | 0.2368 | 0.2500 |
| `pbr2_align_gated` | 0.2041 | 0.1212 | 0.6452 | 0.2300 | 0.1579 | 0.4231 |

**直接结论**：

1. 目前仓库里最接近 community PRM baseline 的不是最新 smoke repair，而是 `prm_e46`。
2. 很多“terminal 修好了”的方案，本质上只是把 `acc_correct` 拉高了，但 `acc_erroneous` 仍然很弱。
3. 现有文档中把 `AUC≈0.62` 粗暴类比成“接近 55 F1+”的说法，不成立。

## 2.2 队友所说“修复了 ProcessBench 的一些表现”如何准确表述

**准确说法应该是**：

1. `terminal_top1` 的确被若干新配方显著拉起来了。
2. `same-family trust` 在 `c3 / pbr2` 这类配置上也明显强于 `e87`。
3. 但截至当前，没有任何一条新路线同时守住：
   - official-style `F1`
   - `AUC`
   - `first_edge`
   - `good_vs_laterbad`
   - `same-family utility`
   - RL promotion gate

所以这轮新增工作更准确的定位是：

- **把 failure mode 诊断清楚了**
- **还没有把主问题解决掉**

## 2.3 现在最应该怎么读这些实验

### `ms_e43`

优点：
- `AUC` 仍是当前较强基线
- `good_vs_laterbad` 也比很多 repair 路线更强

缺点：
- terminal 几乎完全不会
- 官方 `F1` 很低

定位：
- **local / later-bad 基线**
- 不是最终主线

### `prm_e46`

优点：
- 当前回算后是这批候选里 `ProcessBench F1` 最好的
- 对 `acc_erroneous / acc_correct` 的平衡也相对最好

缺点：
- `good_vs_laterbad` 仍不够
- 不是同源主训练集上的最强 learnability 结果

定位：
- **当前最好的 benchmark-facing 参考点**

### `e87_repair`

优点：
- 说明低比例 terminal anchor 方向是有效信号

缺点：
- benchmark ranking 大幅塌缩
- `same-family` 也明显变差

定位：
- **证明“要加 terminal，但不能这么加”**

### `c3_curated_gated`

优点：
- `same-family top1` 很强
- `Math AUC` 比很多 smoke 好

缺点：
- `F1` 仍弱
- 本质上还是没有解决 `good_vs_laterbad`

定位：
- **更好的 curated+gated probe**
- 不是已通过的 candidate

### `c4_dual`

优点：
- 证明 dual-head 不是纯 paper design，代码里真的能跑
- GSM 上 `acc_correct` 很强

缺点：
- Math 上 `acc_correct` 掉到 0.25
- 说明 dual-head 目前仍未形成稳定归纳偏置

定位：
- **说明“显式分头值得继续，但现在还不够”**

### `pbr2_align_gated`

优点：
- 当前 smoke 中 `same-family trust + terminal` 平衡较好
- `Math AUC` 在新 redesign 线中较优

缺点：
- 官方 `F1` 仍然低
- `acc_erroneous` 仍不够

定位：
- **当前 redesign 线里最值得继续放大的候选**
- 不是主线终点

---

## 3. 社区论文真正告诉了我们什么

## 3.1 社区强方法不是只靠“换一个更复杂的 head”

从本地 PDF 直接看，最近强结果几乎都同时具备以下几项：

1. **更高质量的数据监督**
   - `PRM800K`: 人工 step label
   - `Qwen Lessons`: MC < LLM-judge < human
   - `GenPRM / ActPRM`: 强过滤、主动学习、不确定性选择

2. **更接近 benchmark 的监督几何**
   - `OmegaPRM`: sibling-branch / binary-search first error
   - `Tree-PLV`: tree-based step preference
   - `Rewarding Progress`: 监督的是 progress / advantage，不只是静态好坏

3. **更强的 verifier 形态**
   - `BiRM`: PRM + VM 双向监督
   - `R-PRM`: reasoning-driven evaluator + DPO + inference-time scaling
   - `GenPRM / ThinkPRM`: 生成式 verifier，不只是标量打分

4. **backbone adaptation**
   - 社区强 `ProcessBench` 结果基本都不是 frozen scalar head
   - 至少 LoRA，很多是 full finetune

## 3.2 跟我们最相关的论文结论

### `ProcessBench`

给出的约束是：

1. benchmark 测的是**最早错误定位 + all-correct 判定**
2. critic-style prompted models 往往能打败不少 PRM
3. 许多现有 PRM 在更难数学上泛化不佳

对我们意味着：

- 只学 `lastsafe > firstbad` 天然不够
- 同源 held-out 很高，不代表 benchmark 已对齐

### `PRMBench`

给出的约束是：

1. PRM 需要识别多种 error type
2. 冗余、soundness、consistency、sensitivity 这些细粒度维度都重要

对我们意味着：

- 单一 scalar head 混合所有错误类型，本来就容易 blur
- PRMBench 高 pair_acc 不能直接推出 ProcessBench 强

### `The Lessons of Developing PRMs`

最重要的结论：

1. MC supervision 的 generalization 最弱
2. 只看 `Best-of-N` 会高估 PRM
3. 要同时看 response-level 和 step-level 评测
4. 共识过滤很关键

对我们意味着：

- 纯 `Math-Shepherd + confidence` 仍然是弱监督
- 我们当前大量依赖 `AUC/pair_acc` 的历史比较，本身就容易过乐观

### `Rewarding Progress`

最重要的结论：

1. 好的 process reward 应该衡量“这一步使未来成功率提升了多少”
2. 需要 distinct prover，不是 base policy 自己
3. “first pit” 一类数据收集是有理论动机的

对我们意味着：

- `first_bad_edge` 有价值，但它只是 progress signal 的粗糙近似
- 如果未来要做更强数据构造，应该向 progress / branch signal 靠拢

### `BiRM`

最重要的结论：

1. 早期 error detection 和晚期 terminal preference 最好分开建模
2. 不只是“数据混在一起”，而是训练目标本身分解

对我们意味着：

- 我们做 `dual_head` 方向是合理的
- 但现在还不能宣称“已经复现了 BiRM 思路”
- 因为我们的数据、head、目标仍然远比论文简单

### `R-PRM / GenPRM / ThinkPRM`

共同结论：

1. 生成式 verifier 已经是社区上限路线
2. reasoning-based critique 比单纯 scalar reward 更强
3. inference-time scaling 是重要组成部分

对我们意味着：

- 当前仓库路线的正确定位不是“与社区 SOTA 同层竞争”
- 而是“在更便宜、更可诊断的判别式 regime 里摸清失败结构”

### `ActPRM`

最重要的结论：

1. 主动学习 + 高质量 teacher 标注可以显著节省成本
2. 在 7B 规模上也能把 `ProcessBench` 做到 75.0

对我们意味着：

- 数据质量和采样策略，不是可有可无的细节
- 继续盲目堆 low-quality MC pair 的收益很有限

### `PURE / Stop Summation`

最重要的结论：

1. 即使 offline verifier 做好了，RL 也仍然可能 reward hack
2. sum-form credit assignment 风险很大
3. min-form + 少量 verifiable reward 更稳

对我们意味着：

- 现在还远不能说“RL-ready”
- Phase F 必须建立在更强 verifier 之上，而且 credit assignment 设计要保守

---

## 4. 我们当前想法与社区最佳实践的差距

## 4.1 对齐的部分

1. 已经明确放弃 `StrategyQA` 主线，转回 benchmark-native math/process setting
2. 已经开始显式分析 `first_bad / later_bad / terminal` 三种 slice
3. 已经开始做 `same-family trust + benchmark + rl-promotion` 多层诊断
4. 已经认识到 terminal anchor 是必要的，但不能占主导
5. 已经开始尝试分头结构而不是单一路 scalar

## 4.2 仍然明显落后的部分

1. **数据质量**
   - 仍以 `Math-Shepherd` MC supervision 为主
   - 没有 LLM-judge / consensus filtering

2. **监督几何**
   - 真 sibling-branch 数据仍缺
   - `ms_laterbad_v1` 只是更接近 benchmark，不是真 branch pair

3. **模型形态**
   - 仍是 frozen feature scorer
   - 不是 adapted verifier，更不是 generative critic

4. **指标体系**
   - 现在代码里能算 official `F1`
   - 但大部分历史 artifact、summary、选择逻辑还主要围绕 `AUC/pair_acc`

5. **数据接入成熟度**
   - `Math-Step-DPO` loader 已写
   - 但 parquet 环境坏了，数据还不能稳定用
   - `MATH-APS / Eurus / GenPRM / UltraInteract` 只完成了 verify，不算接入

---

## 5. 目前已经能确认的反模式

### 反模式 1: 把 surrogate 指标当成正式 benchmark 结论

典型表现：

- `pair_acc / AUC` 好看
- official-style `F1` 很低

后果：

- 容易误判“已经接近论文水平”
- 容易错误选择 checkpoint 和主线

### 反模式 2: 把 downloaded dataset 写成 ready-to-train dataset

当前真实情况：

- `Math-Step-DPO`: loader 有，但 parquet 环境坏
- `MATH-APS / Eurus / GenPRM / UltraInteract`: 目录存在，verify 能看到，但没有训练 adapter 主线化

后果：

- 文档看上去“数据源丰富”
- 实际训练还在围着同一两类旧源打转

### 反模式 3: 把 frozen-backbone ceiling 当作已证定理

当前真实情况：

- 外部文献强烈提示 LoRA/full FT 很重要
- 但仓库内部还没有做过关键的 `frozen vs LoRA` 对照

所以更严谨的表述应该是：

- **现有证据强烈暗示 frozen regime 上限较低**
- 但**仓库内部尚未完成决定性的对照实验**

### 反模式 4: 继续扩充 smoke suites，却不先统一 benchmark 口径

当前问题：

- suite 越来越多
- 但很多 summary 仍不是按 official-style `F1 + slice` 主导

后果：

- 结果越来越多
- 但真正可决策的信息密度没有同步提升

### 反模式 5: 把“terminal 修复”和“benchmark 迁移修复”混为一谈

这轮结果已经很清楚：

- terminal 可以单独修起来
- 但 `acc_erroneous / later-bad / first-edge` 可能同时变差

因此以后不能再用单个 `terminal_top1` 上升来宣称 transfer 被修复。

---

## 6. 下一步应该怎么做

## 6.1 立即要做的，不需要再发明新理论

### A. 把 official `ProcessBench F1` 正式纳入所有 Phase E 选择逻辑

这是下一步最高优先级。  
原因：

1. 代码里已经能算
2. 当前主要对照仍在用 `AUC / pair_acc`
3. 这会直接导致 checkpoint 选错、主线判断错

最小要求：

- `phase_e_eval_benchmark.py` 新产物必须持久化 `processbench_f1`
- `compare / select_candidate / rl_promotion` 必须同时读取 `F1`
- 后续文档不再把 `AUC` 与论文的 `F1` 混写

### B. 不再继续泛化地造新 mix，先做两条最有信息量的对照

#### 对照 1: `PBR3_LATER_BAD_BRANCH_SEED3`

目标：
- 验证 `ms_laterbad_v1` 是否真能提高 `acc_erroneous / good_vs_laterbad`

为什么值得做：
- 当前最主要 failure 已经不是 terminal，而是 later-bad generalization
- 这个 profile 和当前诊断是直接对齐的

#### 对照 2: `PBR4_PRM_AND_LATERBAD_FOLLOWUP_SMOKE`

目标：
- 比较 `ms_prm_align_v1` 和 `ms_laterbad_v1`
- 同时比较 `mlp` 与 `gated_mlp`

为什么值得做：
- 这能回答“问题主要在数据几何还是 head 结构”
- 比继续随机添加新 heuristic 更高 ROI

### C. 把 `c4_dual` 从“已跑过一次”升级成“真正被审计过的方案”

现在 `dual_head` 最大问题不是代码不存在，而是：

- 没有做系统 alpha sweep
- 没有做 seed 稳定性验证
- 没有把 `local branch` 和 `terminal branch` 的独立行为单独报出来

所以 `dual_head` 现在还不能下结论“有效”或“无效”。

---

## 6.2 接下来最应该接入哪个新数据源

如果只按当前 ROI 排序：

### 第一优先: `Math-Step-DPO`

原因：

1. loader 已有
2. 数据语义干净，是显式 fork-point pair
3. 与我们当前最缺的 `sibling-branch` 最接近

当前 blocker：

- `pyarrow` 环境损坏，需要先修 parquet 读取

### 第二优先: `RLHFlow-Deepseek`

原因：

1. 已有 adapter
2. 是更强 judge-style supervision，不是纯 MC
3. 比继续堆 Math-Shepherd 更可能带来真实增益

### 第三优先: `MATH-APS`

原因：

1. 从论文角度最符合 `branch / progress / first-error` 方向
2. 但当前还没有 adapter
3. 工程成本高于前两个

### 暂不优先: `UltraInteract`

原因：

- 当前主问题还没在 math 内解决
- 现在做跨域只会同时引入更多变量

---

## 6.3 什么时间点才应该进入 LoRA

建议触发条件：

1. `PBR3 / PBR4 / dual_head follow-up` 都跑完
2. official `F1` 仍停在 `~0.35` 左右量级
3. 结果稳定指向“数据几何修补已给不出根本提升”

到那时再做 LoRA，论证链才完整：

- 不是因为 impatience 解冻 backbone
- 而是因为 frozen regime 的主要 repair path 已被系统跑过

---

## 7. 当前最可靠的总判断

### 社区是怎么做的

社区强方法普遍是：

1. 更高质量的 step supervision
2. 更接近 benchmark 的 branch/progress 数据几何
3. 至少 LoRA，很多是 full FT
4. 越来越多地采用 reasoning-style / generative verifier
5. RL 时显式防 reward hacking

### 我们代码是怎么做的

我们当前最成熟的部分是：

1. 诊断体系
2. curate profile
3. benchmark slice 分析
4. same-family trust / rl-promotion 这些 meta-eval

我们当前最弱的部分是：

1. 数据质量
2. benchmark 正式指标闭环
3. 高质量新数据源接入
4. backbone adaptation

### 差距有哪些

最大的差距不是某一个超参数，而是三件事同时缺：

1. 高质量 supervision
2. adapted backbone
3. 更强 verifier 形态

### 当前有没有反模式

有，而且已经比较明确：

1. surrogate 指标替代 official benchmark
2. downloaded dataset 冒充 ready dataset
3. frozen ceiling 被写成定理
4. suite 数量增长快于决策质量增长
5. terminal improvement 被误写成 transfer improvement

### 下一步怎么做

最合理的顺序是：

1. 先统一 `ProcessBench F1` 口径
2. 再跑 `PBR3 later-bad` 和 `PBR4 follow-up`
3. 再把 `dual_head` 做成真正可判定的对照
4. 同时修好 parquet 环境，优先接 `Math-Step-DPO` 和 `RLHFlow-Deepseek`
5. 如果这些仍然把 `F1` 卡在低位，再进入 LoRA

**当前结论一句话**：  
这套仓库已经把“为什么失败”诊断得比之前清楚得多，但离“已经修好 ProcessBench”还差得很远；离社区强 PRM 更远；离 RL-ready 仍然明显不够。
