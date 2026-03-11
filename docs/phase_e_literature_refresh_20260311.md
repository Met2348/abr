# Phase E Literature Refresh (2025-03 to 2026-03): Directional Risks And ABR-Preserving Redesign

**Date:** 2026-03-11  
**Purpose:** 在保留 ABR 核心思想不变的前提下，补做一轮截至 **2026-03-11** 的文献与社区刷新，判断当前研究方向里哪些假设已经过时，哪些需要根本性重设计。

---

## 0. Why This Refresh Exists

到 `2025-03` 为止的参考文献已经不足以覆盖当前 verifier / PRM / RLVR 的主流变化。

在 `2025-03 -> 2026-03` 这一年里，社区的变化不是“小修小补”，而是方向级别的：

1. **从单标量 PRM 转向会“思考”的 verifier / reward model**
2. **从只看 pair / AUC 转向显式评估 verifier soundness / completeness / hard-case robustness**
3. **从 outcome-only reward 转向 composite / dense / structured verifiable reward**
4. **从 ad-hoc regex answer check 转向独立训练的 answer verifier**
5. **从“一个 verifier 打天下”转向 generator-verifier co-adaptation / uncertainty gating / inference-time verification compute**

这意味着：  
如果我们继续把“冻结 backbone + last-token scalar head + same-source high acc”当主线，方向性风险已经明显升高。

---

## 1. New External Evidence That Materially Changes The Picture

下面只记录对本项目最相关、且会改变设计判断的外部证据。

### 1.1 Verification itself is now a first-class research object

1. `CompassVerifier`（2025-08）
   - paper: https://arxiv.org/abs/2508.03686
   - repo: https://github.com/open-compass/CompassVerifier
   - model card: https://huggingface.co/opencompass/CompassVerifier-32B
   - 关键信号：
     - 社区已经不再把 answer verification 当成几条 regex 规则；
     - verifier 被当成可复用、可部署、可直接进 RL 的独立模型；
     - 官方模型卡还直接报告了“model-based verifier as reward model in RL”的结果。

2. `VerifyBench / VerifyBench-Hard`（ICLR 2026 Poster，发布于 2026-01）
   - source: https://openreview.net/forum?id=JfsjGmuFxz
   - 关键信号：
     - 社区已经开始专门 benchmark “reference-based reward systems”；
     - 重点不是 preference pair，而是 verifier 对 ground-truth reference 的判断能力；
     - 所有系统在 hard cases 上都还有明显缺口。

3. `Hard2Verify`（2025-10）
   - source: https://arxiv.org/abs/2510.13744
   - 关键信号：
     - frontier math 上的 step-level verification 比 `ProcessBench` 更难；
     - open-source verifiers 在这类 open-ended proof setting 下明显落后 closed models。

**对本项目的含义：**

1. 只围绕 `ProcessBench/PRMBench` 打磨，已经不够覆盖 verifier 的真实困难面。
2. 我们当前缺少一个专门的 **answer-verifier / equivalence / false-negative** 评测层。
3. “PRM 很强”与“verifier 真的可用于 RL”之间，已经被社区明确分成两件事。

### 1.2 Reward models are increasingly expected to reason, not just score

1. `Reward Reasoning Model`（2025-05）
   - source: https://arxiv.org/abs/2505.14674
   - 关键信号：
     - reward model 被设计成先进行 deliberate reasoning，再输出 reward；
     - test-time compute 可以显著改善 reward accuracy。

2. `Libra: Assessing and Improving Reward Model by Learning to Think`（2025-07）
   - source: https://arxiv.org/abs/2507.21645
   - 关键信号：
     - 专门构建 reasoning-oriented reward benchmark；
     - generative reward model + learning-to-think 已被视为主线能力。

3. `Think Twice: Branch-and-Rethink Reasoning Reward Model`（ICLR 2026 提交，2025-09 首版，2026-02 修改）
   - source: https://openreview.net/forum?id=iH2yiN1xsn
   - 关键信号：
     - 单次 scalar judgment 会出现 `judgment diffusion`；
     - reward model 可以通过 “先 branch 再 rethink” 的两段式评审改善细粒度错误识别。

4. `Solve-Detect-Verify / FlexiVe`（2025-05）
   - source: https://arxiv.org/abs/2505.11966
   - 关键信号：
     - generative verifier 的 test-time compute allocation 已成为显式设计对象；
     - 文中直接报告对 `ProcessBench` 的 error pinpointing 改进。

**对本项目的含义：**

1. 我们当前把 verifier 近似成“冻结特征 + 小头打分”，已经落后于主流上界路线。
2. 这不意味着要立刻放弃 ABR，而是意味着：
   - **ABR 应该降格为一个 cheap student / auxiliary regularizer / online scorer，**
   - 而不是继续承担全部 verifier 能力。

### 1.3 The field is moving toward structured / factorized supervision

1. `Error Typing for Smarter Rewards / PathFinder-PRM`（2025-05）
   - source: https://arxiv.org/abs/2505.19706
   - 关键信号：
     - 先区分 math error / consistency error，再估计 step correctness；
     - decoupled error detection + reward estimation 提升数据效率与 end-to-end reasoning。

2. `BiPRM`（2025-08）
   - source: https://arxiv.org/abs/2508.01682
   - 关键信号：
     - 用右到左评估流帮助更早步骤的判断；
     - later context 对 earlier-step verification 有直接价值。

3. `ActPRM`（2025-04）
   - source: https://arxiv.org/abs/2504.10559
   - 关键信号：
     - 主动学习 / 不确定性采样可以大幅降低标注成本；
     - 把最不确定数据优先打标，比 uniform 扩数更划算。

4. `PACR`（ICLR 2026 提交，2025-09 首版）
   - source: https://openreview.net/forum?id=jKAqtb63Bl
   - 关键信号：
     - dense process reward 可以来自“模型对正确答案的信心应总体上升”这种内生进度信号；
     - 不是所有有用 process reward 都必须来自人工 step label。

**对本项目的含义：**

1. 当前“一个 scalar 头同时学 local error、later-bad、terminal completion”本身就有结构性压力。
2. terminal undervaluation 不一定只靠“多加 terminal anchor”能修好；
3. 更合理的是把监督拆成：
   - `local validity`
   - `progress`
   - `completion correctness`
   - `error type`
   - `uncertainty / abstention`

### 1.4 The RL literature now makes verifier failure modes more concrete

1. `LongRLVR`（ICLR 2026 Poster，2026-01）
   - source: https://openreview.net/forum?id=omVhYvyTPJ
   - 关键信号：
     - outcome-only reward 太 sparse，会让某些关键子过程 learning intractable；
     - 加入 dense, verifiable auxiliary reward 后显著改善。

2. `TinyV`（TMLR under review，2026-01）
   - source: https://openreview.net/forum?id=HMGsqApBM3
   - 关键信号：
     - false negatives 是 widespread 问题；
     - 其对 RL 的损害是实打实的，不是纯评测噪声；
     - 文中在 `Big-Math-RL-Verified` 上报告超过 `38%` 的生成响应遭遇 false negative。

3. `Online Learnability of Chain-of-Thought Verifiers`（2026-03）
   - source: https://arxiv.org/abs/2603.03538
   - 关键信号：
     - verifier 的 `soundness` 与 `completeness` 代价并不对称；
     - distribution shift 是 verifier 学习里的核心难点，而不是次要细节。

4. `V1: Unifying Generation and Self-Verification for Parallel Reasoners`（2026-03）
   - source: https://openreview.net/forum?id=ZUFJQrZuRp
   - 关键信号：
     - generator 与 self-verifier 共同训练，避免 verifier 与 generator 分布脱节；
     - inference 也会动态分配 verification compute。

**对本项目的含义：**

1. 我们目前把 terminal gap 主要理解为“completion 信号不够”还不完整。
2. 它也可能是：
   - `completeness failure`
   - `answer equivalence failure`
   - `false-negative-heavy verifier behavior`
   - `student/verifier 与未来 policy 分布脱节`

### 1.5 Widespread 2025H2-2026 verifier patterns that the current repo still underweights

下面这些证据和前四类不同。它们说明：社区已经把 verifier 当成一个独立系统，而不是“附在 policy 后的一个分数头”。

1. `CompassVerifier`（ACL 2025）
   - source: https://aclanthology.org/2025.acl-long.1102/
   - 关键信号：
     - open-source 社区已经在训练独立 verifier；
     - 数据里显式包含：
       - invalid / malformed responses
       - formula augmentation
       - multi-expert annotation
       - multi-prompt voting
   - 对本项目的含义：
     - `final answer equivalence / invalidity handling` 不应继续被埋在单一 scalar head 里。

2. `VerifyBench`（2025-07）
   - source: https://arxiv.org/abs/2507.09884
   - 关键信号：
     - verifier 的强弱不仅取决于打分能力，还取决于：
       - reference usage
       - prompt contract
       - hard-case robustness
   - 对本项目的含义：
     - 只在单一 `ProcessBench` prompt contract 上追分，存在明显风险。

3. `Verifying the Verifiers`（2025-06）
   - source: https://arxiv.org/abs/2506.13342
   - 关键信号：
     - verifier 排名对输入结构、候选形式、提示方式都可能敏感。
   - 对本项目的含义：
     - 需要把 `format / prompt / answer-style invariance audit` 升级成正式 gate，
     - 而不是继续默认“一个 benchmark 输入模板就够了”。

4. `Weaver / Shrinking the Generation-Verification Gap with Weak Verifiers`（2025-10）
   - sources:
     - paper: https://arxiv.org/abs/2510.18084
     - community note: https://hazyresearch.stanford.edu/blog/2025-10-05-weaver
   - 关键信号：
     - 弱 verifier 不是没用，而是可以通过弱监督组合与蒸馏，合成更强训练信号。
   - 对本项目的含义：
     - 当前最不该做的是让一个不稳定的单一 judge 去 bulk relabel 全部数据；
     - 更合理的是：
       - 多弱源 disagreement mining
       - hard-slice 复判
       - teacher-to-student distillation

5. `When to Trust the Cheap Check`（2026-03）
   - source: https://arxiv.org/abs/2603.05390
   - 关键信号：
     - 社区已经开始明确研究“便宜 verifier 什么时候该信、什么时候必须升级到强 verifier”。
   - 对本项目的含义：
     - 这与“保留 ABR 核心，但把它降格为 cheap student / router”完全一致。

6. `MathQ-Verify`（2026-03）
   - source: https://arxiv.org/abs/2603.03307
   - 关键信号：
     - 连“题目是否本身可稳定评判”都开始被单独 benchmark。
   - 对本项目的含义：
     - 后续 curate 不能只想“哪里有 step label”，还要想“这些样本是否真的可稳定验证”。

---

## 2. Updated Diagnosis: Which Existing Assumptions Are Now Directionally Risky

### 2.1 “Frozen scalar head can be the mainline verifier”

这个假设在 2025 年早期还算合理的低成本诊断路线，但到 2026 年已经更像：

1. 一个有用的 **baseline / student / ablation regime**
2. 不是最值得继续大规模投资的终局主线

原因不是它没用，而是：

1. 生成式 verifier / reasoning RM 的外部证据已经太强；
2. answer verification 已被单独建模；
3. structured / factorized supervision 已经成为明显趋势；
4. RL-ready verifier 关注的 failure mode 比 scalar AUC 复杂得多。

### 2.2 “ProcessBench + PRMBench 基本足够定义 RL-ready”

这个假设也在变弱。

现在至少还缺：

1. **answer equivalence / abnormal response** 能力
2. **false negative audit**
3. **soundness / completeness split**
4. **更难 open-ended step verification**（如 `Hard2Verify` 代表的能力面）

因此当前仓库的 RL-ready gate 需要升级，不应只由：

1. held-out
2. same-family
3. `ProcessBench/PRMBench`

这三层组成。

### 2.3 “terminal undervaluation 主要靠多加 terminal anchor 解决”

现在看，这个假设过于窄了。

更可能的真实成因是四类因素叠加：

1. completion 监督不足
2. answer equivalence / completeness failure
3. scalar head 把 local 与 terminal 混成同一个通道
4. frozen unidirectional representation 无法高效利用 later context

### 2.4 “如果未来 RL 分数涨了，就说明 verifier 路线成立”

这个假设现在风险更大。

原因：

1. 最新 RLVR 文献已经显示：
   - sparse reward 本身可能诱导奇怪的训练动态；
   - false negatives 会严重误导更新；
   - 某些设置里即便 reward 很差，policy 仍可能涨分。
2. 因此未来 ABR / RL 线必须避免：
   - 只看 final benchmark 提升，
   - 不审 verifier 质量本身。

### 2.5 “单一 verdict + 不允许 abstain” 仍然够用

这个假设也在过时。

更晚近的 verifier 系统越来越常见三种输出：

1. `correct / incorrect / invalid`
2. `verdict + confidence`
3. `verdict + abstain / escalate`

如果我们继续要求一个 head 在所有样本上强行输出单一分数：

1. hard slice 会被错误地压成 confident false negative；
2. RL 场景下会把“本该升级验证”的样本误当成可直接给 reward 的样本；
3. terminal / equivalence / malformed cases 会长期混在一起，形成难以解释的噪声。

---

## 3. What Should Remain Stable: The ABR Core Still Worth Keeping

ABR 的核心思想并没有被这些新文献推翻。  
真正应该保留的是：

1. **过程级别建模**
   - 不是只看 final answer；
2. **相邻 prefix / step 间的局部一致性约束**
   - 这仍然是 process faithfulness 的重要 inductive bias；
3. **Bellman-style / temporal-consistency flavor**
   - “越接近正确完成，价值/置信应满足结构约束”这一点仍然成立；
4. **cheap student scorer**
   - 在 RL 或大规模筛样里，低成本 verifier 仍然必要。

更准确的定位应该变成：

1. **ABR 不是整个 verifier 系统的全部**
2. **ABR 是 student / regularizer / shaping prior**
3. **更强 teacher / answer verifier / uncertainty gating 应包在外层**

---

## 4. Proposed Redesign: Keep ABR Core, Change Everything Around It

## 4.1 New system decomposition

建议把当前单一路线改成三层 verifier stack：

### Layer A: answer verifier

职责：

1. 判断 final answer equivalence
2. 检测 abnormal / malformed / unsupported outputs
3. 输出 `correct / incorrect / invalid / abstain` 风格的更诚实终态判断
4. 降低 false negatives

来源启发：

1. `CompassVerifier`
2. `VerifyBench`
3. `TinyV`

### Layer B: strong teacher verifier

职责：

1. 对 hard / uncertain slices 给出更强 supervision
2. 生成 step-level critique / first-error / support-needed labels
3. 只在困难样本或评测时使用较多 compute

来源启发：

1. `R-PRM`
2. `Reward Reasoning Model`
3. `Think Twice`
4. `Solve-Detect-Verify`

### Layer C: ABR student

职责：

1. 低成本大规模打分
2. 作为 online / RL 中的轻量 scorer
3. 承担 temporal consistency / local process structure prior

这里 ABR 的 Bellman-style core 被完整保留，但不再单独承担所有 verifier 功能。

## 4.2 Replace one scalar head with factorized outputs

建议 student 头从“一个分数”改为以下最小因子化输出：

1. `v_local`
   - 当前步 / 当前前缀是否局部有效
2. `v_progress`
   - 是否朝最终正确解前进
3. `v_complete`
   - 完整正确 completion 的绝对偏好
4. `u_uncertainty`
   - 不确定时触发 abstain / second-pass verify
5. `e_type`（可选）
   - arithmetic / consistency / unsupported leap / malformed
6. `a_abstain`
   - 何时应升级到更强 verifier / answer checker / deterministic tool

ABR 约束不删，而是作用在：

1. `v_local` 的相邻前缀平滑 / first-bad consistency
2. `v_progress` 的正确轨迹单调性
3. `v_complete` 与 answer verifier 的 terminal agreement
4. `a_abstain` 与 hard-slice disagreement / invalid 样本的一致性

## 4.3 Data curation must become multi-view, not single-pair-only

当前 pair 体系过窄。  
建议至少固定五类训练单元：

1. `branch pairs`
   - 同一 prefix，不同 continuation
   - 主要来自 DPO / rollouts / sibling branches

2. `all-correct ladders`
   - 同一正确解上的多个前缀，两两构成 completion/progress pair
   - 直接针对 terminal undervaluation

3. `answer-equivalence hard set`
   - 非标准格式但正确的答案
   - 用于 answer verifier 与 student terminal head 的 completeness 校准

4. `uncertain hard slices`
   - 由当前 student / answer verifier disagreement 选出
   - 采用 `ActPRM` 风格优先打标

5. `reverse-view / bidirectional views`
   - 同一轨迹加 reverse prompt 或 reverse encoding
   - 引入 later context 帮助 earlier-step verification
6. `format-stress / contract-variation slices`
   - 同一语义样本使用：
     - 不同答案格式
     - 不同 prefix delimiter
     - 不同 prompt contract
   - 用于检查 verifier 是否只是学了输入模板
7. `process-outcome mismatch slices`
   - final correct but derivation flawed
   - final wrong but locally plausible for a long prefix
   - 直接对应 `PRIME` 风格 process-outcome alignment

## 4.4 Training pipeline redesign

### Stage 0: verifier calibration first

先单独把 answer verifier 校准好，不要一开始就把“终态真值”混在 scalar head 里。

补充：

1. 同时建立一小份 `invalid / malformed / abstain-needed` 校准集；
2. 把 answer equivalence / completeness failure 从一开始就显式拆出来。

### Stage 1: teacher-assisted annotation on uncertainty slices

不是全量 expensive relabel，而是：

1. disagreement mining
2. false-negative mining
3. terminal-hard mining
4. unsupported-leap mining
5. invalid / malformed mining
6. format-fragility mining

### Stage 2: factorized ABR student pretraining

loss 不再只是 pair ranking，至少应包含：

1. local ranking / boundary loss
2. progress monotonicity loss
3. terminal BCE / completion ranking
4. uncertainty calibration loss
5. abstention / escalation loss
6. ABR temporal consistency regularization

### Stage 3: optional DPO / implicit-reward continuation

来源启发：

1. `PRIME`
2. `V1`
3. `R-PRM`

如果 student 需要继续适应 policy 分布，优先考虑：

1. DPO-style branch continuation
2. partial LoRA / limited unfreeze
3. 不要直接跳到 full RL

### Stage 4: conservative RL only after verifier gate upgrade

真正进入 RL 前，要先通过：

1. answer verifier completeness gate
2. false-negative gate
3. same-family local gate
4. benchmark gate
5. uncertainty / abstention gate
6. format-robustness gate
7. process-outcome alignment gate

而且 reward aggregation 应维持保守：

1. `min-form`
2. 或 constrained blend
3. 不推荐 naive sum-form

---

## 5. Concrete Repository-Level Recommendations

### 5.1 Stop optimizing the wrong bottleneck

短期内不应该继续把主要资源砸在：

1. 单一 scalar head architecture churn
2. 只在 `first_bad_edge` 上继续扫超参
3. 只追 `AUC / pair_acc`

### 5.2 New mainline proposal

建议新的主线改成：

1. `ABR-S`:
   - factorized student head
   - 保留 ABR consistency
2. `AV`:
   - 独立 answer verifier 层
3. `TV`:
   - expensive teacher verifier 只服务 hard slices

即：

`AV + TV -> annotate / gate -> ABR-S`

而不是：

`single scalar head -> solve everything`

### 5.3 New evaluation sheet

新的 promotion sheet 至少应包含：

1. same-source held-out
2. same-family top1 / first-bad
3. `ProcessBench / PRMBench`
4. answer equivalence accuracy
5. false-negative rate on correct answers
6. soundness / completeness split
7. all-correct completion top1
8. abstention coverage-accuracy
9. hard-slice / frontier-slice verifier score
10. format-contract invariance
11. invalid / malformed detection accuracy
12. process-outcome alignment slice score

### 5.4 Highest-value next experiments

1. `ABR_AVCal_v1`
   - build answer-equivalence / invalid / false-negative calibration set
2. `ABR_WeaverLite_v1`
   - combine:
     - ABR score
     - answer verifier
     - deterministic math equivalence
     - one stronger teacher judge on disagreement slices
   - learn a conservative weak-ensemble target before retraining the student
3. `ABR_MH_v1`
   - multi-head student: `local + progress + completion + uncertainty + abstain`
4. `ABR_FormatInv_v1`
   - add format / prompt / delimiter perturbation audit and training canaries
5. `ABR_POA_v1`
   - construct a small `process-outcome alignment` slice set from existing data
6. `ABR_Active_v1`
   - ActPRM-style uncertainty relabel only on disagreement / hard slices
7. `ABR_TeacherDistill_v1`
   - distill teacher verifier outputs into student factor heads
8. `ABR_Restart_v1`
   - use ABR uncertainty + completion heads to trigger:
     - critique
     - restart
     - or escalation
   - rather than only offline reranking

优先级上，建议：

1. 先做 `AVCal`
2. 再做 `ABR_WeaverLite_v1`
3. 然后是 `ABR_MH_v1`
4. 再补 `ABR_POA_v1`
5. 最后才考虑 RL / restart-style intervention

---

## 6. Bottom Line

截至 **2026-03-11**，最重要的新判断是：

1. **ABR 核心思想没有过时。**
2. **过时的是把 ABR 近似成“一个冻结 scalar head 的全部设计”。**
3. **当前项目最需要修整的不是 Bellman-style core，而是 verifier system design。**

更具体地说：

1. 保留 ABR 的 temporal-consistency / process-faithfulness inductive bias；
2. 但把它放进：
   - factorized student head，
   - answer verifier，
   - teacher verifier，
   - uncertainty gating，
   - active hard-slice curation
   这个更完整的系统里。

如果不做这一步，项目很可能继续在：

1. same-source 很强，
2. benchmark 有进步，
3. 但总是离真正 RL-ready 差一口气

的状态里反复打转。
