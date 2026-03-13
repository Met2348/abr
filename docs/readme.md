## 2026-03-12 22:10 Update

1. `PBR32` same-family is now complete and stronger than we previously knew:
   - `top1=0.8985`
   - `local_first_bad=0.9081`
2. `PBR40C frozen` formal suite completed:
   - same-family is slightly better than `PBR26`
   - but Math/GSM benchmark `F1` still do not beat `PBR26/PBR31/PBR32`
   - RL diag still blocks on `terminal completion`
3. `implicit PRM v2` has been repaired and now has a real 200-sample Math result:
   - `pair_auc=0.6885`
   - `F1_oracle=0.3537`
   - far below explicit value-head performance, so this line is demoted to diagnostic-only.
4. `PBR40C LoRA` full formal suite is still running in the background.

## 2026-03-12 21:40 Update

1. 新增了 `PBR44/PBR45` 的自动 follow-up 链：训练一结束，就会自动接 `ProcessBench GSM/Math` 和 `Phase F preflight`，对应 watcher 在 `phase_e_pbr44_pbr45_followup_wait_0312_1750`。
2. 这轮进一步确认 `offline implicit PRM` 该降级：最新 `pbr38f` 在 Math fixed-threshold 上是 `F1=0.0000`，更像分数动态范围塌掉，不像单纯阈值问题。
3. 目前最值得读的主线不再是“再加一种数据 recipe”，而是把 `PBR44` 和 `PBR45` 跑完整条验证链后再决定 promotion。

## 2026-03-12 17:33 Update

1. `docs/relatedPapers/` 已再次补齐；当前文档里检测到的 `113` 个 paper-like URL 已全部有本地镜像，缺失数为 `0`。
2. 论文写作时要区分两个 `PRIME`：
   - `2502.01456` = implicit-reward RL method
   - `2602.11570` = verifier benchmark
3. `scripts/phase_f_grpo_lite.py` 已新增 optional replay-buffer GRPO 入口，对应新的等待队列：
   - `phase_f_replay_grpo_canary_wait_0312_1732`
4. 当前 `Phase F` 的最新判断没有变：RL 仍是 selective infra exploration，主线仍是 hybrid verifier + controller / BC。

## 2026-03-12 17:21 Update

1. 新的长报告：
   - `docs/phased_e_reports_9_20260312T172129+0800.md`
2. 当前最新结论：
   - `PBR32` 仍是最强 Math LoRA 候选
   - `PBR31` 仍是最平衡 same-family / GSM LoRA 候选
   - `PBR26` 仍是最强 frozen 参考
3. `PBR38B` 的失败已经被本轮审计查清：
   - 不是“共识过滤这个方向整体失败”
   - 而是该数据池把 `local_first_bad_edge` 全部滤掉，并把 terminal 比例抬到了 `22.5%`
4. 本轮新增了 CPU-only 诊断与数据修复入口：
   - `scripts/run_phase_e_current_frontier_audit_suite.sh`
   - `scripts/phase_e_curate_semantic_consensus_pairs.py`
5. 新的 `PBR40` pair pool 是下一轮训练的正确起点：
   - `7366 -> 4837`
   - local 语义保留
   - terminal 只占 `5.0%`

## 2026-03-12 17:20 Update

1. `Phase F` 新完成的 hybrid usability suite 进一步支持“verifier system / ensemble”路线，而不是单一标量 head：
   - `math ph1mlp + ph1gated = 0.8926`
   - `gsm ph2mlp + ph2gated = 0.8815`
2. `BC->RL` 仍然只能算 selective follow-up：部分切片有增益，但 from-scratch RL-like 依然不稳定，当前默认顺序仍是 `heuristic -> BC -> selective BC->RL`。
3. `scripts/phase_f_grpo_lite.py` 已升级为 modern TRL / `Dr.GRPO` 入口，并显式检查解释器环境；`scripts/run_phase_f_overnight_suite.sh` 也已安全化，避免旧 artifact 解析和 silent failure 风险。
4. 目前还挂着两个不抢占 GPU 的自动队列：
   - `phase_f_hybrid_preflight_wait_0312_1712`
   - `phase_f_modern_grpo_canary_wait_0312_1712`

## 2026-03-12 Evening Update

1. New long-form D/E/F sourcebook:
   - `docs/phased_e_reports_7_20260312T170216+0800.md`
2. New overnight queue:
   - `scripts/run_phase_e_phase_f_autonomy_overnight_suite.sh`
3. Current queued research direction:
   - `PRMBench selected-relabel + Phase F preflight`
   - `PBR6 LoRA backbone smoke`

## 2026-03-12 Late Afternoon Hybrid Refresh

最新确认的结论：

1. `PH1/PH2` 的 old summary 里 `held_pair=0.0` 只是汇总 bug，不是训练失败；代码和 artifact 都已经修正。
2. `Math-PRM-7B + PRMBench-local hybrid` 现在是 `Phase E` 最强主线。
3. `PH2-mlp` 在 full `ProcessBench` fixed@0.5 上更像可部署候选：
   - GSM `F1@0.5 = 0.7063`
   - reward-hacking probe 全部 `low`
   - threshold penalty 很小
4. `PH1-gated` 更像 Math 特化候选：
   - Math best-F1 更高（`0.6181@0.400`）
   - 但 fixed-threshold 稳定性不如 `PH2-mlp`
5. `PH3_PRM_LOCAL_TA15_MSGRID5_ARCH_SWEEP_SMOKE` 已启动，正在检验能否桥接 `PH1` 与 `PH2` 的优势。
6. 新的 focused `Phase F` 复核已经把一个容易误判的点澄清了：
   - `PH2-MLP-Math` 上先前那次 `BC->RL` 小幅正信号，在 teacher-aligned + 3-seed 审计下不再稳健；
   - 当前更稳的默认顺序仍然是：
     - heuristic
     - BC
     - selective BC->RL
   - 不能把那一次单点结果当成“RL controller 已经稳定有效”的证据。

# BCR/ABR Research Pipeline (Condensed)

## 2026-03-12 Midday E/F Refresh

这轮新增的已验证结论很直接：

1. `docs/relatedPapers/` 现在已经补齐到 `106/106` 个文档中出现的论文 URL。
2. `Phase E` 的 cheap -> strong verifier gate 只在当前 `GSM` 风格切片上成立：
   - `p19 -> p31` / `p19 -> p33` 都能小幅提高 AUC，且只需要 `9%-16%` 的 strong usage；
   - 但 `Math` 上 `p26/p31 -> p32` 当前没有收益。
3. `Phase F` 的 corrected mixed-pool GRPO proxy 显示 reward signal 并不缺，真正的问题更像是：
   - `GSM8K` 太饱和，
   - from-scratch RL 仍然极不稳定。
4. 聚焦 controller 实验里：
   - `bc_then_rl` 比 `bc_only` 有小幅提升，
   - 但 from-scratch RL-like 直接塌到 `0.0000`，所以主线仍应保持 `heuristic / BC first`。
5. 新的 `Phase E L2` gated LoRA frontier 已经在 GPU2 正式启动。

详细研究与实验汇总见：

1. `docs/phase_e_phase_f_web_research_refresh_20260312.md`
2. `docs/result_records.md` 顶部最新条目

本仓库用于推进“推理可靠性/一致性/Faithfulness”研究，当前主线已经切换为：
- 先用 Phase A/B 建立稳定基线与现象诊断，
- 在 Phase C/D 验证 ranking-style value supervision 是否可学，
- 但不再把 StrategyQA 当作主监督训练基准，
- 后续先在具备高质量步骤监督的数据集上验证方法，再把结果迁移回 StrategyQA。

## 2026-03-12 Phase E/F 审计补充：LoRA 默认安全化 + Controller test-split 修正

这轮又修了两个会误导结论的实现口径问题：

1. `scripts/phase_e_train_value_lora.py`
   - 现在默认改成仓库安全配方：
     - `--ranking-target-space score`
     - `--pair-weight-mode none`
     - `--checkpoint-selection-metric pair_acc`
   - 含义：
     - 直接调用 LoRA trainer 时，不会再因为 legacy 默认值而静默落入已知危险 recipe。

2. `scripts/phase_f_train_trainable_controller.py`
   - `scripts/phase_f_behavior_clone_controller.py`
   - 现在主结果固定看 `test_eval`，而不是把 train/dev 混回 full traces 再报 headline。
   - `full_eval` 仍然保留，但只作为 in-benchmark 上界参考，不能当外部泛化证据。

这次修复后，Phase F controller 的 summary 解释标准更严格：

1. `best_dev_eval`
   - 只用于选模型
2. `test_eval`
   - 才是主结果
3. `full_eval`
   - 只能当“同 benchmark 家族内的上界参考”，不能拿去替代真正 held-out / external 结论

## 2026-03-12 Phase E / F Overnight Packaging

这一轮又补了两条更贴近当前瓶颈的一键入口：

1. `scripts/run_phase_e_lora_overnight_suite.sh`
   - 直接围绕 `PBR26` 数据池做 stronger LoRA frontier
   - 包含：
     - all-layer LoRA + mild contrastive + centering
     - all-layer LoRA + stronger contrastive + gated head
     - stronger-setting dual-head controlled retry
2. `scripts/run_phase_f_usability_overnight_suite.sh`
   - 直接围绕 stronger verifier slices 做 controller usability chain
   - 串行覆盖：
     - controller sweep
     - generator robustness
     - weak-verifier ensemble
     - BC / BC->RL
     - robust-from-scratch RL-like probe

同时修掉了一个会污染结论的 LoRA launcher 风险：

1. 手写 `PBR33/34/35` launchers 以前是 `set -e` 但没开 `pipefail`
2. `python ... | tee log` 中途挂掉时，shell 仍可能继续做 eval
3. 现在这些脚本只会接受已经写出 `manifest.json` 的完整 run dir

## 2026-03-12 Phase E/F 安全化与 verifier 系统重设计补充

1. `docs/relatedPapers/` 现已补齐文档里提到的论文 PDF 镜像，可离线查阅。
2. `Phase E` 活跃入口进一步收紧：
   - candidate selector 默认严格要求 `best_value_head.pt`
   - 不再默认静默回退到 `final_value_head.pt`
   - `run_phase_e_dual_head_smoke.sh` 也不再默认跑已知危险 recipe
3. 新增一个离线 `cheap -> strong verifier` 门控实验，直接回答：
   - 便宜 verifier 什么时候该自己判，
   - 什么时候必须升级到更强 verifier。

当前最重要的新结论：

1. `cheap -> strong` 门控方向是合理的，但当前 weak verifier 还不够强。
2. 以 `prm_e46` 或 `ms_e43` 为 weak、`pbr26` 为 strong 时：
   - benchmark AUC 的确会随着强 verifier 覆盖率上升而改善；
   - 但要接近强 verifier 本体，通常需要把 `~87%-97%` 的样本都交给 strong。
3. 这说明当前 repo 还不能依赖“cheap verifier 只审大部分样本，少量升级”来解决 RL-ready 问题。
4. 更合理的新主线是：
   - local/process verifier
   - terminal/answer verifier
   - abstain / escalate gate
   三者分开建模，而不是继续押注单一 scalar head。

## 2026-03-12 当前活跃 Overnight 运行

已重新修复并启动这轮更贴近当前瓶颈的夜间任务：

1. `phase_e_phase_f_overnight_0312_1621`
   - 修复后的单卡顺序包：`PH2` + `selected relabel` + `modern preflight`
2. `phase_e_selrel_wide_0312_1615`
   - 更宽 low-margin slice 的 `selected relabel`
3. `phase_e_selrel_strict_0312_1615`
   - 更严格共识过滤版 `selected relabel`
4. `phase_f_usability_0312_1615`
   - Phase F controller usable 性 overnight

当前最重要的新增实现修复：

1. `run_phase_e_processbench_hybrid_suite.sh` 的目录解析不再被 `pipefail + head` 误伤
2. `gated_mlp` 不再错误复用 `mlp` warm-start checkpoint

## 2026-03-13 单卡 Overnight 新包

为了避免拥挤服务器上的多任务互相抢 VRAM，新增了一套更保守、也更适合真实 overnight 的顺序包：

1. 文档：
   - `docs/phase_e_phase_f_overnight_bestpractice_20260313.md`
2. launcher：
   - `scripts/run_phase_e_phase_f_single_gpu_overnight.sh`
3. preflight：
   - `scripts/run_phase_f_modern_preflight_suite.sh`

设计原则：

1. `Phase E` 继续沿 benchmark-oriented + selective judge relabel 路线推进；
2. `Phase F` 先审 `PBR26 / PBR31` 的 threshold-shift 与 reward-hacking 面；
3. 暂不把 verifier 直接提升为 RL 主奖励。

推荐启动方式：

```bash
RUN_PREFIX=phase_e_phase_f_overnight_$(date +%m%d_%H%M) \
GPU_ID=3 \
bash scripts/run_phase_e_phase_f_single_gpu_overnight.sh
```

## 2026-03-13 Overnight Frontier 最新结论

1. `F2_DUAL_HEAD_PBR10` 是负结果：
   - dual-head 没修好 terminal blind spot，反而把 same-family 与 benchmark 几何都拉坏了
2. `PH2_PRM_LOCAL_TA10_MSGRID10_ARCH_SWEEP_SMOKE` 说明：
   - held-out 很高不等于 benchmark-facing trustworthiness
   - 该混合配方达到 `held-out 0.9318 / 0.9040`，但 256-sample ProcessBench AUC 仍只有：
     - GSM `0.5224`
     - Math `0.5321`
3. `CR1_CURATED_CENTER_GATE_SMOKE` 被 recipe guard 正确拦截
4. `F3_LORA_PBR10` 仍在跑，当前还不能下最终结论


## 2026-03-13 A-F 审计补充

这一轮补充审计没有发现新的 A-F 主链路高危实现 bug。

新确认并修掉的风险只有一类：

1. 夜间 frontier launcher 的链式依赖以前只看 `final_summary.md` 是否出现；
2. 这会让下游实验在上游 `status: failed` 时也被错误启动；
3. 现在已经改成必须等待 `- status: ok`。

对应文件：

1. `scripts/wait_for_summary_status.py`
2. `scripts/run_phase_e_overnight_frontier_suite.sh`

这说明当前仓库的主要瓶颈已经更偏向：

1. 监督几何与 benchmark 不对齐
2. controller / reward 设计问题
3. 不是新的实现层面 silent corruption

## 2026-03-13 Overnight Frontier Package

最新夜间实验协调说明：

1. `docs/phase_e_overnight_frontier_plan_20260313.md`

一键 launcher：

```bash
RUN_PREFIX=phase_e_overnight_$(date +%m%d_%H%M) \
bash scripts/run_phase_e_overnight_frontier_suite.sh
```

这一包实验只做五件高价值的事：

1. `dual-head` 修 terminal blind spot
2. 最小 `LoRA` probe 测 frozen-backbone ceiling
3. benchmark-oriented hybrid supervision
4. 一个 curated + centering 后续修复
5. 一个 terminal-anchor ratio sweep

## 2026-03-13 最新突破复核

最新核实文档：

1. `docs/phase_e_latest_breakthrough_verification_20260312.md`

当前最准确的状态：

1. `PBR26`
   - 是当前已验证最强的 frozen benchmark candidate
   - 不是 RL-ready
2. `PBR31`
   - 是当前已验证 LoRA 路线里最好的 balance 提升
   - 不是 `Math AUC` 新 SOTA
   - 也不是 RL-ready
3. 当前最顽固的残差仍然是：
   - `terminal completion ordering`
   - 以及 `first_bad` slice 的 reward-hacking surface
4. 所以现在不能把 repo 结论表述成：
   - "`PBR26` 可以直接做 RL verifier"
   - 或 "`PBR31` 已经干净超越 frozen"

## 2026-03-13 Phase F robust / ensemble 更新

在 controller family sweep 之外，又补做了两类更接近部署的问题：

1. worst-generator robust policy 搜索
2. weak-verifier score ensemble

关键结果：

1. `Math` 上，`guarded_drop` 是当前最稳的 robust family
2. `GSM` 上，`delayed_drop / drop_needs_low` 更合适
3. `pbr26+pbr31` 或 `pbr19+pbr31` 的弱集成还能继续提升 offline controller 指标
   - `Math` 可到 `balanced_f1≈0.876`
   - `GSM` 可到 `balanced_f1≈0.913-0.914`

这进一步支持：

1. 当前 verifier 已经足够支撑 heuristic controller 试运行；
2. 下一步优先级应是 live heuristic controller，而不是立刻上 RL；
3. RL 仍需等 live validation、reward-hacking、threshold stability 全部复核后再推进。

## 2026-03-12 Phase F RL-like 学习更新

这轮又进一步验证了一个关键判断：

1. trainable controller 不是做不到；
2. 但“从随机初始化直接做 RL”目前很不稳；
3. `behavior cloning` warm-start 明显更靠谱。

关键结果：

1. `pbr26_math` 纯 REINFORCE 两种 reward 都塌到 `balanced_f1 = 0.0000`
2. `pbr31_math` robust RL from scratch 只有 `0.3493`
3. `bc_only` 却能直接做到：
   - `pbr26_math = 0.8502`
   - `pbr31_math = 0.8552`
   - `pbr31_gsm = 0.9045`
4. `BC -> robust RL` 目前没有再带来提升，反而略降

这说明当前最合理的 Phase F 下一步仍然是：

1. heuristic controller live 验证
2. 或 BC-distilled controller live 验证
3. 而不是把 RL 作为默认第一步

## 2026-03-13 Phase F controller 新结论

补做多策略 offline controller sweep 后，结论进一步更新：

1. 主要问题不在 verifier 太弱，而在旧 `ABR-lite` 控制规则设计错误。
2. 更简单、约束更明确的 controller family 明显更好：
   - `Math` 侧：`threshold_only`
   - `GSM` 侧：`delayed_drop / guarded_drop`
3. 多个强候选上，`balanced_f1` 可提升到约 `0.86-0.91`。
4. 这说明下一步不该直接上 RL，而应先做：
   - heuristic controller 重构
   - live generation 验证

## 2026-03-13 Eval Provenance Audit Update

当前 A-F 主线又收紧了一层默认安全语义：

1. `Phase B/C/D/E` 的 standalone eval 路径现在默认都按 `fail-safe` 处理
   - 请求 `best` 但缺失时，会直接报错
   - 不再默认静默回退到 `final`
2. `posthoc_calibration=from_run` 也同样如此
   - 缺失 `best` calibrator 时，默认直接失败
   - 只有显式传 `fallback_final` 才允许 legacy 兼容
3. 一旦真的发生 fallback，现在必须显式写进：
   - `metrics.json`
   - `manifest.json`
   - `summary.md`
   - 控制台输出

这次额外 grep 复核还确认了一点：

1. 当前主 `Phase E source bundle` 训练入口没有直接把 `ProcessBench`/`PRMBench` benchmark 测试集吞进训练；
2. benchmark 相关脚本仍然主要属于：
   - eval
   - alignment audit
   - research-only curated utilities

## 2026-03-12 Phase F 审计更正

`Phase F` 目前还不能算“已完全审计通过”。

已确认有 artifact 支撑的部分：

1. `threshold / generator-shift` 检查
2. `reward-hacking` probe

需要降级的部分：

1. 之前文档里对 `F2 ABR-lite` 的描述过于乐观
2. 原始仿真结果显示：
   - `balanced_f1=0.3388`
   - `acc_erroneous=0.9882`
   - `acc_correct=0.2044`
3. 解释：
   - 控制器会非常激进地拦截错误，
   - 但也会错误拦下大量本来正确的推理，
   - 所以还不能作为 `Phase F` 的 live RL / controller promotion 依据

## 2026-03-11 Frontier 三方向（最新设计）

在当前最强共享 artifact `PBR10` 上，我们新开了三条更贴近 2026 verifier 范式的方向：

1. `judge-filter`
   - 用更强数学 judge 做保守离线去噪
2. `dual-head`
   - 把 local 与 terminal 目标显式拆开
3. `minimal LoRA`
   - 用最小骨干适配测试 frozen-head ceiling

目前已经确认的一条结论是：

1. 当前 `judge-filter` 契约在 `PBR10` 上是负方向：
   - 所有 `1783` 条可审计 `local_first_bad_edge` pair 全部被 `parse_failed`
   - 过滤后训练集只剩：
     - `sibling_branch`
     - `terminal_completion_anchor`
   - 说明它不是在“清洗 local supervision”，而是在破坏监督几何

更详细解释见：

1. `docs/phase_e_frontier_paradigms_and_experiments_20260311.md`
2. `docs/phase_e_paradigm_refresh_experiments_20260311.md`
   - 用通俗术语解释更新后的 verifier / PRM 范式
   - 记录 `PBR10 -> PRX1 / PRX2` 的定向修复实验
   - 当前最重要结论：
     - `PRX1` 明显改善 terminal completeness
     - `PRX2 dual_head` 虽然修了 terminal，但 same-family verifier 本体明显变差

## 2026-03-11 最新主线更新

当前最重要的新结论：

1. A-E 全链路代码已做过一轮实现审计，最高危的 cache lock / checkpoint fallback / dtype drift 问题已经修掉。
2. 新的 benchmark-facing 主线不再是 `Math-Shepherd` 纯 local 配方，而是：
   - `math_step_dpo_v1` 数据几何，
   - `Qwen2.5-Math-PRM-7B` backbone，
   - `mlp` head，
   - 并且同时报告 oracle 与 fixed-threshold `ProcessBench F1`。
3. 当前最强新候选：
   - `PBR10 mlp`
   - `ProcessBench Math`: `AUC=0.863`, `F1_oracle=0.659`, `F1_fixed@0.5=0.654`
   - `ProcessBench GSM8K`: `AUC=0.873`, `F1_oracle=0.724`, `F1_fixed@0.5=0.693`
4. `gated_mlp` 虽然能继续抬高 held-out / edge metrics，但 fixed-threshold F1 更差，因此暂不升主线。
5. `Phase E` 当前所有活跃 wrapper 已显式绑定 `recipe-risk-policy`，旧的危险 recipe 不再能通过 suite 默默回流。
6. 2026 新文献更新后，当前主线不再把“单一 scalar head”视为最终形态，而是把它降级为：
   - bounded-support verifier baseline，
   - 后续需要加上 process-outcome alignment 与 critique / self-verification 分支。
7. 最新一轮更广的 `2025H2 -> 2026-03` verifier refresh 进一步收紧了这个判断：
   - `ABR` 核心应保留，
   - 但未来主线应变成：
     - 独立 answer verifier
     - `invalid / abstain / escalate`
     - weak-ensemble / teacher-assisted hard-slice 标注
     - factorized ABR student
     - `process-outcome alignment + format robustness` gate
8. 这轮新的 transfer follow-up 进一步说明：
   - `Math-Shepherd strict-only` 不行；
   - `RLHFlow strict-only` 只带来有限增益；
   - `Math-Step-DPO sibling_branch` 是当前最有效的新信号。
9. 在 `Qwen2.5-7B-Instruct` frozen-head 路线上，新的最强 transfer 线已变成：
   - `NDS7 ms_dpo_calibrated + gated_mlp`
   - 3-seed aggregate:
     - `ProcessBench GSM8K AUC = 0.6596 ± 0.0158`
     - `ProcessBench Math AUC = 0.7460 ± 0.0145`
10. 更新后的真实瓶颈不再是“benchmark 全面迁移失败”，而是：
   - local / global prefix ranking 已明显起来，
   - `all-correct terminal completion ordering` 仍然偏弱。
11. 围绕这个残差又做了一轮受控三分支修复：
   - `数据修复`：`ms_dpo_terminalboost_v1`
   - `损失修复`：`NDS7 + terminal BCE`
   - `架构修复`：`NDS7 + dual_head + terminal BCE`
12. 结果显示：
   - `terminalboost` 能修 `GSM8K` 的 terminal，但会伤 `Math` 的 local/global；
   - `terminal BCE` 单独不够；
   - `dual_head` 会把模型推成更强的局部错误探测器，但 terminal completion 明显更差。
13. 所以下一步不该再做“更重的 terminal 混合监督”，而应把 terminal completion 当成单独 residual 子任务处理。
14. 队友新提交的“Math-PRM-7B 冻结 backbone 突破”经过本地补评测后，结论应表述为：
   - backbone 突破是真的；
   - 但 `PRX1` 还不能直接宣称是全局最强。
15. 当前更准确的候选格局是：
   - `PBR12`：`Math AUC` 最强；
   - `PBR21`：`Math/GSM oracle F1` 最强；
   - `PRX1`：当前已验证 `GSM fixed@0.5 F1` 最强。

详见：

1. `docs/phase_abcde_impl_audit_and_redesign_20260311.md`
2. `docs/result_records.md`
3. `docs/progress_detailed.md`
4. `docs/phase_e_pipeline_redesign_20260311.md`
5. `docs/phase_e_updated_literature_redesign_20260311.md`
6. `docs/phase_e_literature_refresh_20260311.md`
7. `docs/phase_e_rl_ready_research_redesign_20260311.md`

## 2026-03-11 Postfix 复核：当前最强候选已很强，但仍未过 strict RL gate

这轮在最新 `PBR10/PBR13` 路线上又补了一次实现与评测后验复核，结论有两层：

1. `Phase E` 选择器默认已切到 `heldout_only`，benchmark 只保留为 promotion gate / canary，不再默认直接参与选模打分。
2. `Qwen2.5-Math-PRM-7B` 的 same-family trust 评测现在会对长 batch 的异步 CUDA 容量错误做显式同步与 backoff，不再把长 batch crash 误报成泛化错误。
3. 最新 `pbr10_dpo8k` 是当前最强 frozen-head offline 候选：
   - samefamily `top1=0.9098`
   - samefamily `local_first_bad=0.8981`
   - `ProcessBench GSM F1=0.7334`
   - `ProcessBench Math F1=0.6313`
4. 但 strict 诊断仍然判它 `not_rl_ready_terminal_completion_risk`：
   - `pb_min_auc=0.8651`
   - `pb_min_terminal_top1=0.1626`
5. `pbr13_terminalbce` 把 benchmark headline 再抬高了一些，但仍未解开 terminal completion ordering，而且 samefamily 还更弱。

这意味着当前真实瓶颈已经收缩成一个更具体的问题：

1. 不是 generic benchmark transfer 全面失败；
2. 不是 frozen-head 完全不可用；
3. 而是 `all-correct final completion` 仍然被系统性低估。

详见：

1. `docs/result_records.md`
2. `docs/progress_detailed.md`
3. `docs/phase_e_impl_audit_20260311.md`

## 2026-03-11 Judge 正式实验结论

这轮把 judge 从 pilot 推进到了两条正式实验线：

1. `PRMBench_Preview selected relabel`
2. `ProcessBench hard-slice benchmark-side adjudication`

结果要点：

1. `selected relabel` 的 formal run 确实把 held-out `PRMBench_Preview` 拉高了：
   - baseline `E78`: `pair_acc=0.9521`, `auc=0.9071`
   - selected relabel: `pair_acc=0.9555`, `auc=0.9207`
2. 但这个结果不能被解释成“judge 已经会重标 hardest slice”
   - judge 在最不确定的 `64` 条边界样本上只保住了 `1` 条
   - 更合理的解释是：
     - 我们删掉了最不确定、且更容易截断塌缩的边界样本，
     - 因而得到了一个更干净的 same-source 训练集
3. `ProcessBench hard-slice adjudication` 则给了更强的负结论：
   - `math`: `pair_acc_majority=0.1034`
   - `gsm8k`: `pair_acc_majority=0.0000`
   - 说明当前本地 judge 连 benchmark-side hard-slice 裁决都还不能依赖

因此当前最合理的 judge 定位应进一步收紧为：

1. `PRMBench_Preview` 风格 pair 数据上的极保守 selected pruning / disagreement audit
2. 不做 bulk relabel
3. 不做 `ProcessBench` benchmark-side adjudicator

## 2026-03-11 Phase E 基础设施加固

`Phase E` 现在默认加入两层防护：

1. **recipe guard**
   - 训练前拦截仓库里已知会灾难性失败的超参组合；
2. **collapse diagnosis**
   - 训练后自动产出结构化健康诊断，而不是只靠人工读日志。

关键脚本：
1. `scripts/phase_e_train_value.py`
2. `scripts/phase_e_diagnose_training_collapse.py`

这一步的作用不是提高 benchmark 分数，而是防止 `Phase E` 后续的 source 质量比较继续被“坏配方塌陷”污染。

## 2026-03-11 Judge LLM 本地落地结论

这轮把 `LLM-as-a-judge` 方案真正落到了本机，结论很明确：

1. 当前最适合先接入 Phase E 的本地 judge：
   - `assets/models/Qwen2.5-Math-7B-Instruct`
2. 已安装但暂不建议直接主线化的更强 judge：
   - `assets/models/DeepSeek-R1-Distill-Qwen-14B`
3. 已新增本地 smoke 工具：
   - `scripts/phase_e_smoke_judge_llm.py`
4. 更完整的选型 / 下载 / 调试记录：
   - `docs/phase_e_judge_llm_selection_20260311.md`

现阶段工程建议：

1. 先用 `Qwen2.5-Math-7B-Instruct` 做 bulk relabel / disagreement mining。
2. 不要立刻把 `DeepSeek-R1-Distill-Qwen-14B` 当作主线 strict-JSON judge。

补充：在真实 `ProcessBench` 小切片上做完对照后，结论进一步收紧为：

1. `Qwen2.5-Math-7B-Instruct`
   - 更稳，但明显偏向把长 reasoning 判成“全对”
2. `DeepSeek-R1-Distill-Qwen-14B`
   - 在轻量 `first_bad_only` 契约下，对 `gsm8k` 出现了一些首错步定位信号
   - 但整体仍不适合直接升主线
3. 因此当前最合理的 judge 部署仍然是：
   - `Qwen2.5-Math-7B-Instruct` 做 bulk
   - `DeepSeek-R1-Distill-Qwen-14B` 只做二级复判

详细对照：

1. `docs/phase_e_judge_llm_selection_20260311.md`

进一步补做 `pairwise + swap-debias` judge benchmark 之后，结论又更具体了一步：

1. 这条 judge 用法确实比之前的 `strict-JSON pointwise prefix audit` 更合理；
2. 但当前本地 judge 仍然不够强，不能直接当 bulk Phase E pair filter；
3. `PRMBench_Preview` 上还有一些可用信号；
4. `Math-Shepherd local_first_bad_edge` 上几乎不可用，因为更短但仍然正确的前缀常被 judge 误判得不如更长但已出错的前缀。

新增材料：

1. `docs/llm_judge_design_20260311.md`
2. `assets/artifacts/phase_e_pairwise_judge_compare/judge_pairwise_compare_20260311T084132Z/summary.md`

## 2026-03-11 Backbone Relaxation + Judge Pilot

这轮把两个很直觉的修复方向真正跑成了实验：

1. 放开 frozen backbone，做最小 LoRA 训练；
2. 把本地 judge 接成 Phase E pair 的 hard filter / prefix audit。

详细记录：

1. `docs/phase_e_backbone_judge_audit_20260311.md`

核心结论：

1. `small-data LoRA` 不是当前解：
   - 在同一 `96/127` raw subset 上，
   - held-out、same-family 都明显劣于 raw frozen，
   - `ProcessBench` 上还出现 benchmark decision rule 不对齐和数值稳定性问题。
2. `strict-JSON local judge hard filter` 也不是当前解：
   - 96 条训练 pair 只剩 16 条，
   - 全部可审计 pair 都因为 parse failure 被丢掉。
3. 因此 judge 当前更适合：
   - disagreement mining
   - selective relabel
   - offline audit
   而不是直接做主训练前置过滤。
4. 当前 benchmark-facing 参考仍应保留：
   - `PBR2`

## 2026-03-11 ProcessBench State Audit + Community Gap Review

这轮不是再做一个新 head，而是把“队友新代码 + 新文档 + 新 PDF + 新下载数据”
全部放到一起审一遍，看仓库现在到底到了哪一步。

新的综合诊断文档：

1. `docs/processbench_state_and_community_gap_20260311.md`

新的横向审计 artifact：

1. `ProcessBench Math`:
   - `assets/artifacts/phase_e_transfer_compare/processbench_state_review_math_0311_20260311T070248Z/summary.md`
2. `ProcessBench GSM8K`:
   - `assets/artifacts/phase_e_transfer_compare/processbench_state_review_gsm_0311_20260311T070308Z/summary.md`
3. strict RL promotion audit:
   - `assets/artifacts/phase_e_rl_promotion_diag/rl_promotion_state_review_0311_20260311T070332Z/summary.md`

核心结论：

1. 队友说“修复了部分 `ProcessBench` 表现”是部分正确的：
   - 新 repair 线确实修了某些 slice，
   - 尤其是 `terminal_top1` 和部分 `first_edge`
2. 但它们还没有取代当前最强旧基线：
   - `ms_e43`
   - `prm_e46`
3. fresh RL audit 的最好读法是：
   - `ms_e43`:
     - `near_rl_ready_but_terminal_gap`
   - `prm_e46`:
     - `terminal_and_local_tradeoff_unresolved`
   - `c3/c4/pbr2`:
     - `not_rl_ready_laterbad_generalization_weak`

这意味着当前仓库的真实瓶颈已经变成：

1. local / later-bad / terminal 三种 verifier 子能力还没有被同一训练栈一起学稳，
2. 而不是简单的“head 容量不够”。

下一步因此也改了：

1. 暂停一轮 head-only churn，
2. 优先把已下载到本地的强数据源真正拉进主线矩阵：
   - `PRM800K`
   - `MATH-APS`
   - `EurusPRM-Stage2`
   - `Math-Step-DPO-10K`
3. 做 equal-budget source-only compare，
4. 然后再做 frozen-vs-LoRA 的最小正面对照。

## 2026-03-11 Verified Internet Reading + Dual-Head Smoke

这轮把外部论文/社区经验又收紧了一步，并做了一个更贴近这些经验的结构实验。

外部结论重新核对后，最 relevant 的信号是：

1. `ProcessBench` 和 `PRMBench` 这类 benchmark 不只是测单个 local edge。
2. 文献更支持“分解验证子任务”而不是盲目给单头加容量。
3. 真正重要的 slice 不是一个总 AUC，而是：
   - `first_edge`
   - `good_vs_laterbad`
   - `terminal_top1`

因此我在现有 frozen-feature Phase E 里实现了一个最小版 `dual_head`：

1. `local_head` 主要吃 local / fanout pair
2. `terminal_head` 主要吃 terminal anchor
3. 推理时用 `inference_alpha` 混成一个分数

代码入口：

1. `src/ours/phase_b/value_head.py`
2. `src/ours/phase_e/training.py`
3. `scripts/phase_e_train_value.py`

在与 `C3_CURATED_GATED_CENTER` 同一 curated artifact 的对照里，新 `C4 dual_head` 的结果是：

1. held-out / samefamily 明显变差：
   - held-out `pair_acc=0.8608`, `auc=0.7390`
   - samefamily `top1=0.8787`
2. `ProcessBench` 上出现了“对了一半”的现象：
   - `first_edge` 提升
   - `terminal_top1` 提升
   - 但整体 `auc` 和 `good_vs_laterbad` 下降

具体地：

1. `ProcessBench Math`
   - `C3 gated`:
     - `auc=0.5152`
     - `first_edge=0.5397`
     - `terminal_top1=0.8654`
     - `good_vs_laterbad=0.4738`
   - `C4 dual`:
     - `auc=0.4789`
     - `first_edge=0.5714`
     - `terminal_top1=0.9038`
     - `good_vs_laterbad=0.3672`
2. `ProcessBench GSM8K`
   - `C3 gated`:
     - `auc=0.4861`
     - `first_edge=0.4906`
     - `terminal_top1=0.7097`
     - `good_vs_laterbad=0.4267`
   - `C4 dual`:
     - `auc=0.4730`
     - `first_edge=0.5660`
     - `terminal_top1=0.8548`
     - `good_vs_laterbad=0.2756`

当前判断：

1. `dual_head` 不是纯负结果，它证明“拆 local / terminal 子任务”方向有信号。
2. 但当前这版硬路由太粗，导致模型过度专门化，伤到了 broader ranking transfer。
3. 所以它不该取代 `C3 gated_mlp` 成为当前主线。
4. 如果继续走这条路，下一步更值得做：
   - `alpha` sweep
   - softer route weights
   - staged curriculum，而不是继续硬拆头。

## 2026-03-11 Semantic Curation + Reward Centering Smoke

这一轮把互联网调研直接落成了一套新的 Phase E 小规模修复：

1. 新增 `docs/phase_e_internet_research_20260311.md`
   - 把 `PRM800K / Math-Shepherd / PRMBench / ThinkPRM / GenRM / Dyve`
     和 `TRL RewardTrainer` 的关键信号整理成仓库内设计原则。
2. 新增语义配额 curate 层：
   - `scripts/phase_e_curate_semantic_pairs.py`
   - 可直接从已有 artifact 按 `pair_semantics + source_tag + quota`
     重组一个新训练池。
3. 在 Phase E trainer 里加入 `reward centering`：
   - `scripts/phase_e_train_value.py`
   - `src/ours/phase_e/training.py`
   - `src/ours/phase_b/value_losses.py`
4. 新增一键 smoke suite：
   - `scripts/run_phase_e_curated_rlready_suite.sh`
   - 对比：
     - `C1 curated mlp`
     - `C2 curated mlp + centering`
     - `C3 curated gated_mlp + centering`

当前结论：

1. `samefamily` 仍然很强，但三组都没有达到 strict RL-ready。
2. `reward centering` 在这条路径上几乎没有改善 benchmark transfer。
3. `gated_mlp` 是这轮唯一能把 `ProcessBench Math` 拉高一点的配置：
   - `math_auc 0.4553/0.4545 -> 0.5152`
   - `math_first_edge 0.5079 -> 0.5397`
4. 但它同时拖累了 `ProcessBench GSM8K`，所以仍不能升格为 RL-ready 候选。

最重要的新判断不是“已经找到修复”，而是：

1. 在当前这条 curated regime 下，
2. 主要瓶颈已经更像 `local benchmark transfer weak`,
3. 而不是 `terminal completion still missing`.

## 2026-03-11 RL Promotion 基础设施更新

这一轮最重要的新增结论不是“找到了 RL-ready 候选”，而是：

1. 仓库现在终于能把 `RL promotion` 失败拆成三类不同问题：
   - local / later-bad benchmark 迁移不足
   - terminal completion 不足
   - 基础设施层面的数值污染
2. 新增了 `scripts/phase_e_diagnose_rl_promotion.py`，把
   - same-family top1 / pressure
   - `ProcessBench` first-edge
   - `good_vs_laterbad`
   - `terminal_top1 / terminal_gap`
   固定成一套更严格的 promotion gate。
3. 新增了 `PRMBench terminal-anchor ratio` 调节杆，不再只能在
   “完全没有 terminal anchors” 和 “terminal anchors 很重” 之间二选一。
4. 更关键的是，修掉了一个真实高危问题：
   - 某些 Phase E cached pooled features 本身含有 `NaN`
   - 这会让 warm-start continuation 在第一批就出现 `loss=nan`
   - 现在 trainer 默认会对非有限 loss 直接报错，
   - 并支持 `--nonfinite-feature-policy drop` 把坏行显式滤掉。

这轮实际最有价值的实验结果是：

1. `E80 fanout`、`E84 heavy terminal`、`E46 PRM local` 都没有通过新的 strict RL-promotion gate。
2. `Math fanout + light terminal` 在修掉非有限特征污染后，终于可以稳定训练和评测：
   - 但它的 same-family trust 明显下降，
   - `ProcessBench Math` 仍弱，
   - 所以这说明“基础设施修好”不等于“候选已经 RL-ready”。
3. `PRM` 轻量 terminal continuation 这一轮没有跑完：
   - shared cache contention 已经通过 `feature-cache-mode off` 绕开，
   - 但共享服务器吞吐仍然太慢，
   - 所以当前它是“未完成”，不是“已证伪”。

## 1. 当前仓库状态

### Phase A（已稳定）
- 数据准备、生成评测、批量推理、truncation safeguard 已落地。
- 已有全量与多组模板/token sweep 的可复现实验框架。
- 产物：`assets/artifacts/phase_a_*`。

### Phase B（已完成核心诊断）
- SFT/PEFT 训练-评测链路已稳定。
- 关键结论：
  - StrategyQA 上可见增益；
  - GSM8K 出现“训练后反降”现象，已做多维排查（LR/epoch/rank/checkpoint/风格等）。
- 汇总：`docs/phase_B_report.md`。

### Phase C（已完成多轮尝试，结论明确）
- C0/C1/C2 与 P(IK) 支线均可运行。
- 已实现多种 loss/校准/对比/过滤技巧，但整体信号质量仍偏弱。
- 当前判断：核心瓶颈在 supervision 噪声与 pair 质量，不是单纯参数没调好。

### Phase D（关键前置阶段，已完成方向纠偏）
- 已接入外部 PRM（Qwen2.5-Math-PRM-7B）做 teacher scoring（D1）。
- 已实现 teacher+MC 融合标签（D2）与目标源切换评估（D3）。
- 已上线 D6（ranking-first）工程版本：
  - C2 支持按 `corr_pair_acc/corr_auc/ranking_score` 选 best checkpoint，
  - 新增 D6 参数组（MC 控制 vs PRM pair gate）做直接对照。
- 已新增 D6-T 分支（triplet validation）：
  - `DT1..DT6` 一键套件已实现，
  - 支持 Math-Shepherd / PRM800K triplet 构造、ranking-only 训练、外部 held-out pair 直接评测。
- 最新关键进展（2026-03-07）：
  - `DT2_MATH_SHEPHERD_SEED3_STABLE` 已稳定过门槛：
    - `mean_pair_acc=0.8154`
    - `mean_auc=0.7857`
    - `std_pair_acc=0.0075`
    - `std_auc=0.0086`
  - `DT4_MIXED_MS_PRM800K_SEED3_STABLE` 仅属“边缘通过”：
    - 总体 `mean_pair_acc=0.6612`
    - 总体 `mean_auc=0.6616`
    - 但分源看，`Math-Shepherd` 明显有效，`PRM800K` 近随机。
  - `DT2_MATH_SHEPHERD_SEED3_STABLE_C1_TRANSFER` 已完成有效重跑：
    - 外部 held-out 仍强，
    - 但回到 StrategyQA 自有 C1 指标后近随机，
    - 说明“外部 ranking 可学”不等于“目标任务自动迁移”。
  - `DT3_PRM800K_SEED3_STABLE` 已确认是稳定弱源：
    - `mean_pair_acc=0.5471`
    - `mean_auc=0.5504`
    - 当前 adapter 下不适合作为主线来源。
  - 因此主线已切到 bridge：
    1. 先用 `Math-Shepherd` 做 external ranking 预训练；
    2. 再 warm-start 到 StrategyQA in-domain CQR/C1；
    3. 看能否把外部 ranking 能力桥接回本任务。
- 当前重点已切换为：
  1. 跑 bridge 两阶段实验，验证外部 ranking 是否能经 in-domain continue training 转成真实增益。
  2. 保留 `PRM800K` 仅作对照/消融来源，不再直接作为主线扩张依据。
  3. 只有 bridge 出现正迁移，才继续推进更大的 mixed-source 或 Phase D promotion。
- 战略调整（2026-03-10，现行执行基线）：
  1. `DT2 stable -> transfer_fix -> DB3/DB4` 这条证据链说明：
     - ranking supervision 本身可学，
     - 但 StrategyQA 上缺少高质量原生步骤标签，
     - 因而不适合作为主监督 value-head benchmark。
  2. 研究联网调研结论：
     - 公开 StrategyQA 资源提供 question / yes-no answer / decomposition / evidence，
     - 但不提供 PRM 级的 step-quality 标注或高质量 chosen/rejected process pairs。
  3. 新主线改为：
     - 主训练/监督来源：`Math-Shepherd`、`PRM800K`
     - 主评测基准：`ProcessBench`、`PRMBench`
     - 后备逻辑线：`ProofWriter`、`EntailmentBank`、`FOLIO`、`FoVer`
  4. StrategyQA 的新角色：
     - bridge continue-training 目标，
     - downstream transfer benchmark，
     - OOD / stress test，
     - 不再承担“方法是否成立”的第一性验证任务。

### Phase E（当前主线）
- 定义：
  - 在高质量 pair / process-supervision benchmark 上先验证同一 benchmark 家族内的 value/ranking 可学习性。
- 2026-03-10 数据语义审计后的修复约束：
  - `Math-Shepherd` / `RLHFlow` / `PRM800K` fallback 不再默认使用 nearest-negative 启发式 pair；
  - 现在默认只构造严格 `first_bad_edge` pair；
  - 新 artifact 会显式记录 `pair_build_mode / pair_semantics`；
  - 旧的 nearest-negative 结果应作为 `legacy` 处理，不再与新结果混读；
  - 详见：`docs/data_semantics_risk_audit_20260310.md`
 - 现在的实现状态（2026-03-10, E0-E3 已落地）：
  1. benchmark-native 新入口已经实现：
     - `scripts/phase_e_prepare_pairs.py`
     - `scripts/phase_e_train_value.py`
     - `scripts/phase_e_eval_benchmark.py`
     - `scripts/run_phase_e_suite.sh`
  2. 新中间层模块已经实现：
     - `src/ours/phase_e/contracts.py`
     - `src/ours/phase_e/pairs.py`
     - `src/ours/phase_e/runtime.py`
     - `src/ours/phase_e/training.py`
     - `src/ours/phase_e/benchmark_eval.py`
  3. 已经验证的能力：
     - 不依赖 `Phase C` artifact 的 external-pair-only 训练
     - `ProcessBench` benchmark-native 评测
     - `PRMBench_Preview` benchmark-native 评测
     - `run_phase_e_suite.sh` 的真实 smoke 跑通
- 当前执行顺序因此更新为：
  1. 直接运行 `Phase E` smoke/full learnability suites，
  2. 先看 source-family held-out learnability，
  3. 再看 benchmark-native 指标，
  4. 最后才回到 StrategyQA 做 transfer。
 - 2026-03-11 运行时可靠性修复：
  1. shared feature cache 已升级到更严格的 provenance 追踪：
     - 不再只记录 `config/index`，还会绑定 index 引用的 shard 文件；
  2. `Phase B` 与 `Phase C P(IK)` 训练器现在都遵循：
     - cache 常驻 CPU，
     - 训练/评测只搬当前 mini-batch 到 head device；
  3. 这次修复的目的不是提速，而是避免两类静默风险：
 - stale feature cache 误复用，
 - cache hit 后整包特征常驻 GPU 导致显存被慢慢吃满。
 - 2026-03-11 `ProcessBench` 重设计更新：
  1. 新增 `scripts/phase_e_curate_processbench_transfer_pairs.py`，把 `Math-Shepherd` 的 `strict/fanout/grid/terminal` 监督按保守 profile 重新 curate；
  2. 新增 `semantic / confidence_semantic` 训练权重；
  3. 新增 `scripts/run_phase_e_processbench_research_suite.sh` 做 `curate -> train -> same-family -> ProcessBench -> RL promotion` 一键研究套件；
  4. 当前最佳新候选是 `pbr2_ms_align_gated`：比 `E87` 更强，但仍未 RL-ready，下一步应重点补 `later-bad` 泛化，而不是继续加重 terminal supervision。
- 2026-03-11 RL-readiness 审计更新：
  1. 两组统一审计已完成：
     - `assets/artifacts/phase_e_logs/phase_e_rltops_0311_1124/final_summary.md`
     - `assets/artifacts/phase_e_logs/phase_e_rlrepairs_0311_1124/final_summary.md`
  2. 当前 bounded-support RL 候选：
     - `ms_e43`
     - `prm_e46`
  3. 当前最值得继续放大的 repair 方向：
     - `ms_grid_micro`
  4. 当前最重要警示：
     - `prm_e78` 说明 same-source 更高分数不等于更安全的 RL prior。
 - 2026-03-11 新增 `intradataset ACC90` 分支：
  1. 只关心单一数据集自己的 held-out pair 判别准确率，
  2. 不把跨数据集迁移和 benchmark 泛化混进这一分支，
  3. 新增 `MLP value head` 选项用于同源高精度拟合，
  4. 新入口：`scripts/run_phase_e_intradataset_suite.sh`
 - 2026-03-11 source-specific 结论更新：
  1. `Math-Shepherd` 已经是强 same-source source：
     - 当前同源 held-out 已稳定超过 `95%`。
  2. `PRMBench_Preview` 也已被当前 trainer 学到：
     - 最佳 push 组已过 `95% pair_acc`。
  3. `R-PRM` 需要单独看：
     - 旧的长文本 / truncation 问题已经不再是主矛盾；
     - 新的 polarity-repair 试验表明：
       - 可以把 `chosen=yes/no` 偏置明显压平，
       - 但总体准确率仍远低于 `ACC90`；
     - 因此当前主问题不是再加长度或简单 class-balance，
       而是 `compact R-PRM` 与当前 frozen feature head 的监督契约不匹配。
 - 2026-03-11 `R-PRM compact` 新增更强证据：
  1. 在 repaired compact 合同上，当前 `MLP + joint + 2048` 已能把
     train-distribution eval 拉到：
     - `pair_acc=0.9090`
     - `auc=0.9131`
  2. 但把完全相同 recipe 放到真 held-out validation 后，会掉到：
     - `pair_acc=0.6280`
     - `auc=0.6508`
  3. 这意味着：
     - `R-PRM compact` 不是“根本学不会”，
     - 也不是“只要继续加容量或 epoch 就够了”，
     - 当前主问题已经收敛为：
       - held-out generalization / supervision-contract mismatch。
 - 2026-03-10 新增 `R-PRM` root-cause 诊断结论：
  1. `direct_pair_legacy` 与 `compact_verdict` 已确认不是同一风险等级：
     - legacy 第一个完全干净的 cutoff 是 `2048`
     - compact 第一个完全干净的 cutoff 是 `1536`
  2. `legacy@1024` 现在会在训练入口的 truncation gate 直接失败，
     不再先加载 backbone 再浪费 GPU。
  3. 在修复后的 compact 契约上，`R-PRM` 已经不是随机可学性：
     - `mlp + joint @2048`
       - `pair_acc=0.6694`
       - `auc=0.6611`
       - `samefamily_top1=0.6829`
  4. 这说明：
     - 旧问题确实部分来自 contract / truncation，
     - 但修好之后它仍不是 `ACC90` 级强源，
     - 当前应把它视为“中等强度、recipe-sensitive”的 source，而不是主锚点。
 - 2026-03-11 `ProcessBench` 对齐审计更新：
  1. 新增诊断脚本：
     - `scripts/phase_e_analyze_processbench_failures.py`
  2. 审计结论：
     - `ProcessBench` 不只是 local first-bad benchmark，
     - 其中 `all-correct` 比例很高，
     - 因而纯 local supervision 会系统性低估 terminal completion。
  3. 定向 micro repair 结果：
     - `terminal anchors` 能明显抬高 `all-correct terminal` 分数，
       但会削弱 local/good-vs-bad discrimination；
     - `all_good_vs_all_bad grid` 能改善 broader prefix-ranking，
       但几乎不修 terminal gap。
  4. 因此下一步主线不再是泛化调参，而是：
     - `local + terminal + optional grid` 的 mixed / staged curriculum repair。
 - 2026-03-11 `MCTS` 文献判断更新：
  1. `MCTS` 不是当前 `Phase E / ProcessBench` 问题的第一修复手段；
  2. 当前主矛盾仍是 supervision semantics mismatch，而不是 search budget 不足；
  3. 若后续引入 `MCTS`，更合理的用途是：
     - 离线 tree-harvested pair / tree 数据构造；
     - 或 test-time judge/search baseline；
  4. 因此当前主线仍应先完成：
     - `local + terminal + optional grid` 的监督契约修复。

## 2. 近期关键实验结论

1. `DT2_MATH_SHEPHERD_SEED3_STABLE` 已证明高质量外部 triplet 上 ranking 分支可以稳定学习。
2. `DT2 ... C1_TRANSFER` 与 `DT3_PRM800K_SEED3_STABLE` 共同说明：
   - “外部 ranking 可学”不等于“自动迁移到 StrategyQA”，
   - `PRM800K` 在当前 adapter 下也不是强源。
3. `DB3` 是当前最重要的正结果：
   - ranking-only bridge 真正提升了 StrategyQA in-domain `corr_pair_acc/corr_auc`。
4. `DB4` 是当前最重要的反例：
   - better Brier 并不代表 better ranking，
   - joint 标量目标会伤害排序能力。
5. 因此项目的科学判断已更新：
   - 当前问题更像 `process ranking`，不是 `scalar value regression`。
6. 因为 StrategyQA 没有公开高质量步骤监督，仓库主验证 benchmark 已转向 PRM-grade 数据集；StrategyQA 仅保留为迁移检验集。
7. 最新 `Phase E` 结果与社区结论一致：
   - `Math-Shepherd smoke` 在 source 内可学，但 `ProcessBench` 仍弱；
   - `PRM800K seed3` 在 source 内和 `ProcessBench` 上都偏弱；
   - 因此当前不应把“跨数据集迁移”设为第一性成功标准。
8. `Math-Shepherd` 正式 seed3 又补了一条更关键的证据：
   - 两个 seed 在 same-source held-out 上很强，
   - 一个 seed 明显塌陷，
   - 因此下一步重点是运行 `Math-Shepherd trust matrix`，
     选出真正可托付的 `best_value_head.pt`，而不是再看单次峰值。
9. 最新 `E15_MATH_SHEPHERD_TRUST_ROBUST_SEED3` 进一步说明：
   - `Math-Shepherd` 单源已经足够证明可学习性，
   - 但对 `ProcessBench` 仍然几乎不泛化，
   - 因此下一步主线应切到“同家族多源数学数据混训”，而不是继续期待单源直接迁移。
10. 还必须保留一条实现层面的谨慎结论：
   - 修复后 `Math-Shepherd` adapter 已对齐到严格 `first_bad_edge`，
   - 但它仍不是“同一步 sibling branch”监督，
   - 所以它更像“local first-bad-edge learnability”证据，
   - 不是“benchmark-grade process verifier 已成立”的证据。
11. 最新 `intradataset ACC90` 结果又补了一条关键结构诊断：
   - `Math-Shepherd` 上，线性头已经能把同源 held-out `ACC` 做到 `0.917+`，
     所以不能把旧弱结果简单解释成“head 太简单”；
   - `PRMBench_Preview` 上，线性头只有 `0.738` 左右，而 `MLP` 可到 `0.93+`，
     这里的确存在明显的结构瓶颈；
   - `E12` 的低学习率长训练仍弱，说明很多坏结果也不只是“欠训练”。
12. 最新 `R-PRM` 专门重诊断又纠正了一条关键认知：
   - 旧 `I4` 的弱结果主要对应历史 `direct_pair_legacy` 长 verifier-essay 合同，
     不能直接代表当前修复后的 `R-PRM`；
   - 在当前 `compact_verdict` 合同下：
     - `1024` 仍略有截断风险，
     - `1536` 是第一个完全干净的长度，
     - `2048` 不再是“为了解决截断而必须”的长度；
   - 同时，修复后 same-source held-out 的确提升了：
     - `1536`: `pair_acc=0.6129`, `auc=0.6031`
     - `2048`: `pair_acc=0.7016`, `auc=0.6735`
   - 这说明：
     - 截断曾经是真问题，
     - 但现在已经不是唯一主问题，
     - `R-PRM compact` 剩下的瓶颈更像 supervision semantics / head 契约匹配问题。
13. 最新 `Math-Shepherd ACC95 push` 又补了一条很直接的工程结论：
   - `Math-Shepherd` 的 same-source 目标其实早已超过 `95% ACC`，
     当前代码栈并不缺这个源上的可学习性；
   - 新增的 `I6_MS_ACC95_PUSH_MATRIX` 说明：
     - verify 控制组 `E67` 仍有 `0.9633`，
     - `joint + logit-space ranking` 的 `E68` 可到 `0.97245`，
     - 更激进的 overfit 推进 `E69` 反而不如 `E68`；
   - 因而当前这个源上的推荐 recipe 是：
     - `MLP + joint + ranking_target_space=logit`，
   - 而不是继续盲目加大过拟合压力。
14. 最新补上的 `strict transfer diagnosis` 又把这个结论进一步收紧了：
   - 之前偏宽松的 `RL-readiness heuristic` 确实能把 `PRMBench E46` 标成 `provisionally_rl_ready`；
   - 但更严格的诊断显示：
     - `Math-Shepherd E68` 的 same-family 很强，
       可是对 `all-correct` 最终前缀存在严重低估，
       `ProcessBench Math` 上还有明显长度漂移；
     - `PRMBench E46` 的 benchmark local discrimination 是当前最强基线，
       但 `all-correct final-prefix top1` 仍只有
       `0.2332 / 0.1970`（`GSM8K / Math`）；
     - `PRMBench E78` 和新的 `terminal-anchor smoke` 虽然把 terminal completion 明显抬高，
       却又把 local good-vs-bad ranking 压坏了；
   - 因此仓库当前**还没有**真正达到 RL 标准的 value head；
   - 更准确的说法是：
     - 已证明 same-family learnability，
     - 也已经定位出一个真实缺口：terminal completion undervaluation，
     - 但还没有找到能同时保住
       `local process discrimination + terminal completion safety`
       的 recipe。

## 2.1 社区/学界结论（影响当前 Phase E 规划）

1. `ProcessBench` 与 `PRMBench` 的出发点就是：现有 PRMs 在更严格的过程错误识别上仍然不足。
2. `VersaPRM`、`ThinkPRM`、`R-PRM` 这类后续工作进一步说明：
   - 单一数据集直接训一个小型判别式 value head，并不天然带来强跨域泛化；
   - 若真要追求跨 benchmark 泛化，通常需要：
     - 多域训练，
     - 更成熟的 preference/process 构造，
     - 或生成式 verifier。
3. 因此仓库的现行策略改为：
   - 先证明 same-benchmark learnability，
   - 再把跨 benchmark/StrategyQA transfer 作为次级目标。
4. 最新补充判断：
   - 多源混训是合理下一步，但必须是**同家族、source-balanced、benchmark-clean** 的混训，
   - 不能把 benchmark 测试集直接掺入训练，
   - 也不能让弱源（如当前 recipe 下的 `PRM800K`）按体量主导训练。
5. 与此同时，Phase E 现在又额外拆出一个更窄的问题：
   - 单一高质量 pair 数据集本身，能不能把 held-out ACC 拉到 `>90%`？
   - 这个问题现在由 `intradataset ACC90` 分支单独回答。
6. 对 ACC90 分支的当前解释规则也已经更新：
   - same-source 高分只能证明“该 source 上可学”；
   - 不能自动证明 benchmark transfer、跨数据集 trust 或 RL 可托付性。
7. 对“RL 里要不要信这个 value utility”这个更高层问题，当前文献与本仓库证据的交集结论是：
   - same-source `ACC/AUC` 很高只是第一道门槛；
   - 还需要证明它在同一数据集家族内能真实改善 reranking / rejection / conservative search，
     并且在更强选择压力下不只是利用 source-specific shortcut；
   - 所以我们当前最多能说“已有 same-source learnability”，还不能说“已经达到 RL 可托付程度”。

## 3. 运行入口

### Phase A
```bash
bash scripts/run_phase_a_benchmark_suite.sh
```

### Phase B
```bash
bash scripts/run_phase_b_training_suite.sh
```

### Phase C（value lifecycle）
```bash
bash scripts/run_phase_c_value_suite.sh
```

### Phase C（P(IK) lifecycle）
```bash
bash scripts/run_phase_c_pik_suite.sh
```

### Phase D（teacher suite）
```bash
bash scripts/run_phase_d_teacher_suite.sh
```

### Phase D6-T（triplet validation suite）
```bash
bash scripts/run_phase_d_triplet_validation_suite.sh
```

### Phase D Bridge（external pretrain -> in-domain continue）
```bash
bash scripts/run_phase_d_bridge_suite.sh
```

### Phase E（high-quality benchmark learnability）
```bash
# 轻量烟测
ACTIVE_PHASE_E_GROUP=E1_MATH_SHEPHERD_PAIR_LEARN_SMOKE \
RUN_PREFIX=phase_e_smoke_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_suite.sh

# 主线 3-seed
ACTIVE_PHASE_E_GROUP=E2_MATH_SHEPHERD_PAIR_LEARN_SEED3 \
RUN_PREFIX=phase_e_math_seed3_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_suite.sh

# Math-Shepherd trust candidate search
ACTIVE_PHASE_E_MS_GROUP=MS2_MATH_SHEPHERD_TRUST_SEED3_MATRIX \
RUN_PREFIX=phase_e_ms_trust_seed3_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_mathshepherd_trust_suite.sh

# 多源数学混训 Stage A-E（推荐新主线）
ACTIVE_PHASE_E_MM_GROUP=MM2_MULTISOURCE_MATH_STAGE_ABCD_SEED3 \
RUN_PREFIX=phase_e_multisource_abcd_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_multisource_math_suite.sh

# 单数据集 held-out ACC90 专项套件
ACTIVE_PHASE_E_INTRADATASET_GROUP=I5_ALL_ACC90_MATRIX \
RUN_PREFIX=phase_e_all_acc90_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_intradataset_suite.sh

# Math-Shepherd 单源 ACC95 验证 / 推高矩阵
ACTIVE_PHASE_E_INTRADATASET_GROUP=I6_MS_ACC95_PUSH_MATRIX \
RUN_PREFIX=phase_e_ms_acc95_push_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_e_intradataset_suite.sh

# 当前 top candidate 的 RL-readiness 审计
ACTIVE_PHASE_E_RL_GROUP=RR4_COMPARE_CURRENT_TOPS \
RUN_PREFIX=phase_e_rl_readiness_$(date +%m%d_%H%M) \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_e_rl_readiness_suite.sh

# 更严格的 transfer / RL 风险诊断
python -u scripts/phase_e_diagnose_transfer.py \
  --run-name phase_e_transfer_diag_manual \
  --audit-spec 'ms_e68|math_shepherd|<samefamily_dir>|<pb_gsm_dir>|<pb_math_dir>' \
  --audit-spec 'prm_e46|prmbench_preview|<samefamily_dir>|<pb_gsm_dir>|<pb_math_dir>'

# PRMBench terminal-anchor pair artifact
python -u scripts/phase_e_prepare_prmbench_terminal_anchor_pairs.py \
  --run-name phase_e_prmbench_terminal_anchor_$(date +%m%d_%H%M) \
  --seed 42
```

### Phase C/D（one-click control + diagnosis）
```bash
bash scripts/run_phase_cd_control_suite.sh
```

## 4. 文档导航

- 总体路线与状态：`docs/readme.md`
- 完整技术手册：`docs/readme_full.md`
- Phase E 正式计划：`docs/phase_E_plan.md`
- Phase E 多源数学混训计划：`docs/phase_E_multisource_math_plan.md`
- Phase E 单数据集 ACC90 计划：`docs/phase_E_intradataset_acc90_plan.md`
- Phase B 结果总表：`docs/phase_B_report.md`
- Phase C 方案与诊断：`docs/phase_C_plan.md`, `docs/phase_C_fix_value_head.md`
- Phase D 正式计划：`docs/phase_D_plan.md`
- C/D 统一对照诊断：`scripts/phase_cd_compare_report.py`
- 调试手册：`docs/phase_a_debug_playbook.md`, `docs/phase_b_debug_playbook.md`, `docs/phase_c_debug_playbook.md`

## 5. 输出位置

- A/B/C/D 运行日志与摘要：`assets/artifacts/phase_*_logs/`
- 运行产物：`assets/artifacts/phase_*_runs/`
- C1/C2 相关数据：`assets/artifacts/phase_c_data/`, `assets/artifacts/phase_c_eval/`
- 命令级复现实验日志：`assets/artifacts/command_logs/`
  - 通过 `bash scripts/run_with_exp_log.sh <your_command...>` 自动记录
  - 自动维护：`docs/commands_to_run.md` 与 `docs/command_result.md`
- 默认状态下仓库不含 `assets\external_datasets`, `assets\model`中各个`safetensors`文件(Model Shard)，以及部分缓存结果`assets\artifact`中的文件。 需要手动补齐这部分文件，代码才能够运行
---

如需“从 0 到结果”的完整流程，请直接看：`docs/readme_full.md`。

## Update: ProcessBench-Oriented Hybrid Pilot (2026-03-11)

最新一轮 `Phase E` 增加了一个更贴近 benchmark 的混合训练试验：

1. 以 `PRMBench` 的 local error-step pairs 作为主锚点。
2. 只加入很小比例的 terminal-completion 辅助。
3. 比较 `mlp` 和 `gated_mlp` 两种头，判断数据契约问题还是结构问题更主导。

关键结果：

1. same-source held-out 很强：
   - `mlp`: `pair_acc=0.9196`, `auc=0.8925`
   - `gated_mlp`: `pair_acc=0.9129`, `auc=0.8711`
2. 但 `ProcessBench` 迁移没有改善，反而出现明显 tradeoff：
   - terminal-completion slice 大幅变好
   - `pair_auc_good_vs_bad` 和 `first_error_edge_accuracy` 明显变差
3. 更关键的是：
   - 训练集中真正的 `terminal_completion_anchor` 只占 `3.68%`
   - 即便这么小的比例，也已经足以“过修” terminal 行为

当前结论：

1. 主问题仍然是 supervision geometry mismatch，不是 head capacity 不够。
2. `gated_mlp` 只能轻微改变 tradeoff，不能从根本上修复 `ProcessBench` 迁移。
3. 下一步应优先考虑：
   - 更小的 terminal 比例
   - benchmark-aware checkpoint selection
   - staged / two-objective 训练，而不是继续做 naive mixture

相关 artifact：

1. hybrid artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_pbhybrid_ph1_0311_1230_pairs__d6fb5a3ec28c`
2. transfer compare:
   - `assets/artifacts/phase_e_transfer_compare/phase_e_pbhybrid_ph1_0311_1230_processbench_gsm8k_compare_20260311T045320Z/summary.md`
   - `assets/artifacts/phase_e_transfer_compare/phase_e_pbhybrid_ph1_0311_1230_processbench_math_compare_20260311T045332Z/summary.md`

## Update: Tiny-Terminal Follow-up (2026-03-11)

在第一轮 hybrid pilot 里，我们已经看到：

1. 很小比例的 terminal support 就能大幅拉高 `ProcessBench` 的 terminal-completion slice。
2. 但它同时显著伤害 `first_error_edge_accuracy` 和整体 `pair_auc`。

为了确认问题到底是“terminal 比例太大”还是“terminal supervision 本身不应直接混入同一 ranking 池”，又补了一条 tiny-terminal pilot：

1. artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_pbhybrid_tinyterm_0311__dd5acae29427`
2. 实际 terminal 语义只占：
   - `17 / 3136 = 0.54%`

结果：

1. same-source held-out 仍然很强：
   - `pair_acc=0.9184`
   - `auc=0.8903`
2. benchmark 上相比上一版 `ta15` 确实更温和：
   - terminal `top1` 没有那么夸张
   - `first_edge` 有小幅恢复
3. 但仍明显不如 `E46`：
   - 说明问题不只是比例，而是 objective / supervision contract 本身

当前更稳妥的下一步：

1. staged training
2. two-objective training
3. benchmark-aware checkpoint selection

而不是继续做简单的 `local + terminal` naive mixture。
