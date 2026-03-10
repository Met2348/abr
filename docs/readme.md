# BCR/ABR Research Pipeline (Condensed)

本仓库用于推进“推理可靠性/一致性/Faithfulness”研究，当前主线已经切换为：
- 先用 Phase A/B 建立稳定基线与现象诊断，
- 在 Phase C/D 验证 ranking-style value supervision 是否可学，
- 但不再把 StrategyQA 当作主监督训练基准，
- 后续先在具备高质量步骤监督的数据集上验证方法，再把结果迁移回 StrategyQA。

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
 - 2026-03-11 新增 `intradataset ACC90` 分支：
  1. 只关心单一数据集自己的 held-out pair 判别准确率，
  2. 不把跨数据集迁移和 benchmark 泛化混进这一分支，
  3. 新增 `MLP value head` 选项用于同源高精度拟合，
  4. 新入口：`scripts/run_phase_e_intradataset_suite.sh`

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
