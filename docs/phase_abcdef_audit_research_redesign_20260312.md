# A-F 全链路审计、联网研究与重设计结论（2026-03-12）

## 0. 本文只采信什么

这份报告只采信四类证据：

1. 当前仓库代码、测试与可执行脚本。
2. 当前仓库已有 artifact 与我本轮新跑出的 artifact。
3. `assets/` 里的原始构想材料：
   - `assets/BCR draft_20251211.pdf`
   - `assets/BCR on step level.pptx`
   - `assets/BCR_discussion_20251219.pptx`
4. 我本轮联网核对过的主来源：
   - ProcessBench: https://aclanthology.org/2025.acl-long.50/
   - PRMBench: https://arxiv.org/abs/2501.03124
   - PRM800K: https://arxiv.org/abs/2305.20050
   - Tree-PLV: https://arxiv.org/abs/2407.00390
   - Rewarding Progress: https://arxiv.org/abs/2410.08146
   - Lessons of Developing PRMs: https://arxiv.org/abs/2501.07301
   - R-PRM: https://github.com/NJUNLP/R-PRM
   - GenPRM: https://arxiv.org/abs/2504.00891
   - ThinkPRM: https://arxiv.org/abs/2504.16828
   - PRIME: https://arxiv.org/abs/2602.11570
   - VPRM: https://arxiv.org/abs/2601.17223
   - TRL PRMTrainer: https://huggingface.co/docs/trl/en/prm_trainer
   - TRL GRPOTrainer: https://huggingface.co/docs/trl/en/grpo_trainer

补充说明：

1. 仓库里不少文档已经非常有价值，但有些强结论建立在旧代码、旧 artifact 或未核对的晚近论文之上。
2. 因此下面所有“高危问题”“该不该继续投资源”的判断，优先以本轮 fresh audit 为准，而不是以历史文档字面表述为准。

---

## 1. 今天我实际核对了什么

### 1.1 仓库规则与脏工作区

我已显式读取：

1. `docs/chat_system_prompts.md`
2. 根 `README.md`
3. 当前脏工作区状态

注意到当前工作区本身就有用户未提交变更：

1. `docs/phase_F_plan.md`
2. `docs/readme.md`
3. `docs/readme_full.md`
4. `docs/result_records.md`
5. `scripts/run_phase_e_overnight_frontier_suite.sh`
6. 若干新的 `Phase F` 脚本 / 计划文档

所以本轮我没有回退任何已有改动，只在必要处增量修复。

### 1.2 我跑过的代码级验证

1. `PYTHONPATH=src pytest -q tests/unit`
   - `231 passed, 2 warnings`
2. `PYTHONPATH=src pytest -q tests/integration`
   - `2 passed, 2 skipped`
3. `python -m py_compile $(find src scripts -type f -name '*.py')`
   - 通过
4. `bash -n` 覆盖所有 `scripts/*.sh`
   - 通过

结论：

1. 当前仓库不是“到处都坏”的状态。
2. 当前最危险的点已经不是语法级崩坏，而是：
   - 研究解释偏差，
   - 评测口径偏乐观，
   - 数据几何与 benchmark 目标不对齐，
   - 新加 `Phase F` 路线缺少充分测试。

### 1.3 本轮新确认并修复的真实高危问题

#### `Phase F robust_lambda` 之前实际上没有梯度

文件：

1. `scripts/phase_f_train_trainable_controller.py`

旧行为：

1. `robust_lambda` 只是把 `mean_reward - worst_gen_reward` 算成一个常数张量加到 loss。
2. 这个项不依赖 policy 的 `log_prob`，所以对参数更新没有任何作用。
3. 表面上像是在做 worst-generator-aware RL，实际上梯度和 non-robust 几乎一样。

为什么这是高危问题：

1. 它会直接污染 `Phase F` 的研究结论。
2. 历史 artifact 里凡是声称“robust objective 改善了 controller”的结论，都必须重新解读。

本轮修复：

1. 新增可微的 worst-generator policy-gradient 项。
2. `robust_lambda` 现在会真实地对当前最差 generator slice 施加额外梯度压力。
3. 训练曲线会显式记录 epoch 内的 worst-generator 信息，方便后续 audit。

回归：

1. 新增 `tests/unit/test_phase_f_trainable_controller.py`
2. 定向测试通过：
   - `PYTHONPATH=src pytest -q tests/unit/test_phase_f_trainable_controller.py`

---

## 2. 原始 BCR / ABR 想法与当前代码现实之间的关系

我直接抽取了 `assets/` 里的原始材料文本后，判断如下。

### 2.1 原始 BCR draft 的核心想法

`assets/BCR draft_20251211.pdf` 的主张是：

1. 用 Bellman consistency 正则化 CoT 轨迹的 value trajectory。
2. 希望 value 的“平滑 / 一致”对应 faithful reasoning。
3. 希望避免昂贵过程标注。

### 2.2 原始 ABR/step-level 讨论的真正价值

`assets/BCR on step level.pptx` 更进一步提出：

1. 不要每个 token 都做 Bellman 检查，而要 step-level。
2. 用一个轻量 router 决定何时 `generate / verify / finish`。
3. 用 `Target-Step Selection` 从历史步骤里选“验证锚点”。
4. 把 AnyMAC 的 `Next-Agent Prediction / Next-Context Selection` 思路搬到 reasoning controller 上。

### 2.3 我对这些原始想法的结论

我认为这里面有两层必须分开：

1. **不该继续当 Phase E 主监督假设的部分**
   - “value smoothness/Bellman consistency 本身就足够近似 step correctness”
   - “插值 reward 或 token-level Bellman regularization 可以替代高质量 process supervision”
2. **仍然值得保留成系统层设计的部分**
   - “何时验证”本身是个独立决策问题
   - “轻量 router / controller” 很可能是有价值的
   - “全历史锚点选择”更接近 controller / critique / routing 层，而不是主 reward label 构造层

一句话总结：

1. 原始 ABR 思路**不适合继续当 Phase E 主训练目标**。
2. 但它非常适合被重解释成 **Phase F controller / router / abstain layer**。

这和本轮代码与实验结果是吻合的：

1. `Phase E` 真正有效的提升来自更好的数据几何、benchmark-aligned 监督与 backbone adaptation。
2. `Phase F` 才是在回答“何时停、何时继续、何时保守处理”的问题。

---

## 3. A-F 当前代码状态判断

## 3.1 Phase A

当前判断：

1. 实现稳定。
2. 本轮 fresh audit 没有再发现新的高危 bug。
3. 主要风险仍然是历史上已经修过的 split/provenance 类问题，而不是当前主代码不可用。

仍要注意：

1. Phase A 的价值主要是 baseline 与 instability diagnosis。
2. 不应继续把它当 process-verifier 主线证据。

## 3.2 Phase B

当前判断：

1. 训练和评测链已稳定。
2. 最大风险仍然是 checkpoint / calibrator provenance 的解释错误，而不是训练 loop 本身坏掉。

残余风险：

1. 仍有 legacy 路径会显式允许 `best -> final` fallback。
2. 这是兼容性保留，不是推荐研究口径。

## 3.3 Phase C

当前判断：

1. 代码稳定，测试覆盖足够。
2. 当前瓶颈主要是 supervision quality，不是实现 correctness。

研究层结论：

1. 不应继续把“多调一点 value loss / calibration 技巧”当主要突破路径。

## 3.4 Phase D

当前判断：

1. 外部 source 接入与 teacher scoring 主链可用。
2. 外部高质量 triplet 的确能学到 ranking。
3. 但迁回 StrategyQA 仍接近随机，说明外部 learnability 不等于目标迁移。

最重要的解释：

1. 这里暴露的不是“训练器坏了”。
2. 而是 source-target gap 真存在。

## 3.5 Phase E

当前判断：

1. 这是当前仓库最有产出的主线。
2. `same-source -> same-family -> benchmark` 的分层结构是对的。
3. 但目前 strongest scalar verifier 仍没有过 strict RL-ready gate。

高危暗坑不在“跑不起来”，而在：

1. `ProcessBench F1` 默认 oracle threshold 仍然偏乐观。
2. 历史 wrapper / 历史文档仍习惯以 AUC / pair_acc 叙事。
3. high same-family acc 不能替代 benchmark-facing trust。
4. terminal completion 仍是系统性薄弱环节。

## 3.6 Phase F

当前判断：

1. 这是当前最容易被“假结论”污染的阶段。
2. 因为很多结论来自离线 controller/reward 分析，而不是 live deployment。
3. 本轮新修复的 `robust_lambda` bug 证明：Phase F 之前确实存在一条会直接污染研究判断的实现暗坑。

修复后的新结论：

1. heuristic / BC controller 仍然是更稳的主线。
2. from-scratch RL-like controller 不是完全没希望，但比历史 artifact 显示得更两极分化。
3. robust objective 现在终于是真优化目标了，但它依然不是“自动超越 heuristic”的捷径。

---

## 4. 联网研究后，我认为现在真正成立的社区共识

## 4.1 最强共识：数据质量与监督几何比“多调 head”更重要

来自：

1. PRM800K
2. ProcessBench
3. PRMBench
4. Lessons of Developing PRMs
5. Tree-PLV
6. Rewarding Progress

结论：

1. step correctness 不是 outcome probability。
2. MC-estimation 风格标签很容易退化成 outcome scoring。
3. shared-prefix / sibling-branch / first-error localization 对泛化更关键。
4. benchmark-facing verifier 质量不能用 same-source held-out 代替。

## 4.2 第二层共识：scalar head 仍有用，但角色应收缩

来自：

1. ProcessBench
2. ThinkPRM
3. GenPRM
4. R-PRM
5. PRIME
6. VPRM

结论：

1. 社区 frontier 正在从“小标量头判分”向：
   - critique-style verifier
   - generative verifier
   - teacher-assisted filtering
   - deterministic checks
   迁移。
2. 这不意味着 scalar head 没价值。
3. 更合理的定位是：
   - cheap student
   - router score
   - conservative reranker
   - one weak verifier inside a larger verifier stack

## 4.3 第三层共识：RL 前必须显式防 reward hacking

来自：

1. Rewarding Progress
2. PRIME
3. VPRM
4. TRL 的 PRM / GRPO 工程路径

结论：

1. “能排对 pair”不等于“可直接当 RL reward”。
2. 真正进入 RL 之前，需要：
   - process-outcome alignment gate
   - terminal completeness gate
   - format / generator shift gate
   - reward hacking probe
   - deterministic or verifiable reward hooks where possible

---

## 5. 我建议的全新数据 curate pipeline

这里不是对现有 `phase_e_prepare_pairs.py` 做小修，而是重新定义主线。

## 5.1 C0: source registry 冻结

为每个 source 记录并冻结：

1. `label_semantics`
   - first_bad_step
   - step_correctness
   - preference_pair
   - terminal_quality
2. `index_base`
   - 0-based / 1-based / mixed / unknown
3. `pair_geometry`
   - shared-prefix sibling
   - local edge
   - later-bad
   - terminal-anchor
   - fanout/grid
4. `confidence_origin`
   - human / LLM judge / MC / heuristic
5. `programmatic_verifiability`
   - yes / partial / no

为什么必须先做这个：

1. 现在很多失败不是模型不会学，而是 source contract 不统一。
2. 这会让后续所有“混源训练”带着隐形语义冲突。

## 5.2 C1: 训练 source 按三层分级

### Tier 1: 高可信主监督

优先纳入：

1. PRM800K 风格 human / curated step labels
2. PRMBench preview 这种显式 pair benchmark
3. shared-prefix DPO / sibling-branch 风格数据

### Tier 2: 可用但要过滤

优先纳入：

1. Math-Shepherd
2. RLHFlow
3. judge-filtered / oracle-filtered source

要求：

1. 必须经过 consensus 或 disagreement-mining 过滤。
2. 不能直接全量裸混。

### Tier 3: 只做 auxiliary

只做弱辅助：

1. terminal-anchor source
2. format-heavy / style-heavy source
3. 明显 length-biased 的 fanout/grid

规则：

1. 只能 capped weight 混入。
2. 不能主导 main loss。

## 5.3 C2: pair family 不再只保留一种

我建议主线明确保留四类 pair / supervision family：

1. `local_first_bad_edge`
   - 保局部错误定位
2. `shared_prefix_preference`
   - 保真正 fork-point 几何
3. `later_bad_support`
   - 保长程退化与后发错误
4. `all_correct_terminal_support`
   - 保 complete-correct 终局排序

注意：

1. fanout/grid 不再当主力 family。
2. 只能当 coverage support。

## 5.4 C3: 过滤策略从“阈值裁样”升级为“证据组合”

建议三类过滤同时存在：

1. `consensus_keep`
   - MC / judge / oracle 一致才保留
2. `disagreement_mine`
   - disagreement 不直接进主训练，而是进 hard slice / relabel pool
3. `programmatic_keep`
   - 能做规则验证的样本优先保留

## 5.5 C4: 新评测切分必须显式包含三桶

每个 benchmark 至少拆三桶：

1. all-correct
2. locally wrong / derivation wrong
3. final-correct but process flawed

否则你永远不知道模型是在学 process，还是在学 final answer flavor。

---

## 6. 我建议的全新训练 pipeline

## 6.1 Track S: Scalar student，继续保留，但明确降级定位

目标：

1. 继续做 cheap verifier / reranker / controller score

配置建议：

1. frozen 或 LoRA backbone
2. dual-head:
   - local head
   - terminal head
3. 显式加入 contrastive hidden objective
4. 用 fixed-threshold + terminal_top1 + worst-generator robustness 选模型

这条线适合：

1. 工程快
2. 诊断强
3. 资源省

但不该继续被包装成最终 verifier 形态。

## 6.2 Track H: Hybrid verifier，应该成为新的主线

推荐形态：

1. local scalar head
2. terminal completeness head
3. optional critique / binary error-type auxiliary

为什么：

1. 当前最稳定的问题就是 local 与 terminal 目标互相挤压。
2. 单头很难同时把：
   - local edge
   - later-bad
   - terminal completeness
   全守住。

## 6.3 Track G: Generative verifier / critique model，必须开正式支线

来自 R-PRM / ThinkPRM / GenPRM 的共识是：

1. frontier 方向已经明显偏生成式 verifier。

我建议本仓库至少做一个轻量版本：

1. 输入问题与 prefix
2. 输出：
   - `correct / incorrect`
   - `first_bad_step` 或短 critique

作用：

1. 先不替代 main verifier
2. 先做：
   - hard-slice evidence generator
   - relabel assistant
   - disagreement adjudicator

## 6.4 Track C: Controller / ABR router，把原始 ABR 想法放回正确位置

这条线对应 Phase F，而不是主 label 构造。

推荐职责：

1. decide continue / stop / abstain / escalate
2. combine weak verifier signals
3. route hard cases to stronger critique / deterministic checks

也就是说：

1. 原始 ABR 该做 controller。
2. 不该做主 reward model 训练信号。

---

## 7. 我建议优先落地的架构尝试

按优先级排序：

1. `LoRA dual-head + contrastive`
   - 最低风险
   - 直接对当前 terminal/local 冲突下手
2. `shared-prefix pair + local/terminal mixture rebalance`
   - 直接修数据几何
3. `scalar + critique auxiliary`
   - 连接到生成式 verifier 支线
4. `implicit PRM / policy-ratio reward`
   - 等 LoRA 稳定后再进
5. `deterministic-check hybrid verifier`
   - 在可验证任务上必须做
6. `ABR-style controller / router`
   - 作为 Phase F 系统层推进

---

## 8. 本轮我实际新跑的实验

## 8.1 实验 A：修复 `robust_lambda` 后重跑 from-scratch robust controller

新 artifact：

1. `assets/artifacts/phase_f_rl_like/phase_f_rl_like_robust_fixed_0312_20260311T201229Z`

旧 artifact：

1. `assets/artifacts/phase_f_rl_like/phase_f_rl_like_robust_fromscratch_0312_20260311T200645Z`

结果对比：

| case | 旧 broken-robust eval | 新 fixed-robust eval | 旧 worst-gen | 新 worst-gen |
|---|---:|---:|---:|---:|
| `pbr31_math` | `0.3493` | `0.6623` | `0.1404` | `0.5803` |
| `pbr31_gsm` | `0.8289` | `0.8935` | `0.5714` | `0.6441` |

解释：

1. 这不是小抖动，而是研究结论级别的变化。
2. 历史 broken-robust artifact 低估了 robust objective 的真实上限。
3. 但即便修复后，from-scratch robust 仍没有自动超过 heuristic / BC。

## 8.2 实验 B：修复后补跑 non-robust mean baseline

新 artifact：

1. `assets/artifacts/phase_f_rl_like/phase_f_rl_like_mean_fixed_0312_20260311T201515Z`

结果对比：

| case | mean eval | robust eval | mean worst-gen | robust worst-gen |
|---|---:|---:|---:|---:|
| `pbr31_math` | `0.3493` | `0.6623` | `0.1404` | `0.5803` |
| `pbr31_gsm` | `0.7680` | `0.8935` | `0.3978` | `0.6441` |

这是本轮最关键的新证据：

1. 旧 broken-robust artifact 实际更接近 mean-policy 行为。
2. 修复后的 robust 目标确实显著改善了 from-scratch RL-like controller。

## 8.3 实验 C：修复后重跑 `BC -> robust RL`

新 artifact：

1. `assets/artifacts/phase_f_bc/phase_f_bc_then_rl_robust_fixed_0312_20260311T201229Z`

对比：

| case | bc_only | 旧 broken BC->RL | 新 fixed BC->RL |
|---|---:|---:|---:|
| `pbr31_math` | `0.8552` | `0.8415` | `0.8351` |
| `pbr31_gsm` | `0.9045` | `0.9001` | `0.9012` |

解释：

1. 对 BC warm-start 路线，修复后 robust 梯度不是“巨大增益器”。
2. 它更像是让结论更诚实：
   - GSM 近似持平
   - Math 仍轻微伤害 teacher policy
3. 所以当前最可信的 Phase F 结论是：
   - from-scratch RL-like 可以被 robust objective 明显救起来；
   - 但 BC teacher 仍然更强；
   - RL 微调还没有稳定超 teacher。

---

## 9. 现在最合理的主线排序

我建议把资源优先级改成：

1. **Phase E 主线**
   - 数据 contract 统一
   - shared-prefix / local / later-bad / terminal 四族监督重构
   - LoRA dual-head + contrastive
2. **Phase G / critique verifier 支线**
   - 先做 hard-slice evidence generator
3. **Phase F controller**
   - 先 heuristic / ensemble / BC
   - 后 robust RL-like
4. **真正 RL**
   - 必须等 process-outcome alignment、reward hacking、deterministic hooks 都齐了再谈

---

## 10. 我建议你接下来优先跑的实验

### 10.1 E-next-1：LoRA dual-head + four-family mix

目标：

1. 同时守 local / later-bad / terminal

必须包含：

1. `local_first_bad_edge`
2. `shared_prefix_preference`
3. `later_bad_support`
4. `all_correct_terminal_support`

### 10.2 E-next-2：critique auxiliary

目标：

1. 测试 `score-only` 与 `score + critique` 是否能提升 benchmark transfer 与 hard-slice diagnostics

### 10.3 F-next-1：BC controller live trial

目标：

1. 不再只做 offline trace replay
2. 先把 `guarded_drop / delayed_drop / threshold_only` 上 live

### 10.4 F-next-2：robust RL-like 只做增量微调，不再 from scratch 主线化

原因：

1. 本轮修复后它证明了自己“不是无效目标”
2. 但依然没有强到足以取代 BC / heuristic

### 10.5 RL-next：只在 hybrid reward 条件满足后进入

至少先满足：

1. fixed-threshold deployment 不是 oracle 幻觉
2. terminal completeness 过线
3. reward-hacking probe 可接受
4. 至少一类 deterministic / verifiable reward 可接入

---

## 11. 最后的仓库级结论

一句话版本：

1. 当前仓库最强的不是“已经接近社区终局的 verifier”，而是“一个已经相当成熟的 process-verifier 诊断平台”。

两句版本：

1. `Phase E` 该继续，但必须以数据几何重构 + hybrid verifier 为核心，而不是继续把单一 scalar head 当万金油。
2. 原始 `ABR` 想法应该保留，但要降落到 `Phase F controller / router`，而不是继续作为主监督假设硬推。

三句版本：

1. 我本轮没有发现 A-E 全面崩坏。
2. 我本轮确认并修掉了一个会直接污染 `Phase F` 研究结论的高危 bug。
3. 修复后，新实验明确说明：
   - robust RL-like controller 不是假的了，
   - 但 heuristic / BC 依然是当前更可信的主线。
