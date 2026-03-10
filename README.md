# BCR/ABR Research Pipeline (Condensed)

本仓库用于推进“推理可靠性/一致性/Faithfulness”研究，当前主线已经更新为：
- 先用 Phase A/B 建立稳定基线与现象诊断，
- 再在 Phase C/D 验证 ranking-style value supervision 是否可学，
- 但从 2026-03-10 起，不再把 StrategyQA 视作 value head 的主监督 benchmark，
- 后续主验证将转向具备高质量步骤级监督的数据集，再回到 StrategyQA 做迁移/OOD 检验。

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
- 最近关键结果：
  - `DT2_MATH_SHEPHERD_SEED3_STABLE` 证明高质量外部 triplet 上 ranking 可稳定学到：
    - `mean_pair_acc=0.8154`
    - `mean_auc=0.7857`
  - `DT2 ... C1_TRANSFER` 说明直接迁移回 StrategyQA 近随机，外部可学不等于目标任务自动迁移。
  - `DT3_PRM800K_SEED3_STABLE` 仅有 `mean_pair_acc=0.5471`、`mean_auc=0.5504`，当前 adapter 下是弱源。
  - `DB3_STRATEGYQA_BRIDGE_FULL_RANK_SEED3` 为当前最强正结果：
    - `mean_stage2_corr_pair_acc=0.5874`
    - `mean_stage2_corr_auc=0.5461`
    - 相比强 C2 基线提升 `+0.0499 / +0.0303`
  - `DB4_STRATEGYQA_BRIDGE_FULL_JOINT_SEED3` 虽改善 Brier，但 ranking 退化到近随机：
    - `mean_stage2_corr_pair_acc=0.4957`
    - `mean_stage2_corr_auc=0.4948`
- 战略调整（2026-03-10）：
  - StrategyQA 没有公开的 PRM 级逐步推理质量标注，不再作为主监督 value-head benchmark。
  - 新主线将转向：
    - 训练/监督：`Math-Shepherd`、`PRM800K`
    - 评测：`ProcessBench`、`PRMBench`
    - 可选非数学逻辑线：`ProofWriter`、`EntailmentBank`、`FOLIO`、`FoVer`
  - StrategyQA 之后只承担：
    - bridge continue-training 目标，
    - downstream transfer，
    - OOD / stress test。

### Phase E（当前主线）
- 目标：在高质量 pair / process-supervision benchmark 上先验证同一 benchmark 家族内的 value/ranking 可学习性。
- 现状（E0-E3 已实现并经 smoke 验证）：
  - 新 benchmark-native 代码入口：
    - `scripts/phase_e_prepare_pairs.py`
    - `scripts/phase_e_train_value.py`
    - `scripts/phase_e_eval_benchmark.py`
    - `scripts/run_phase_e_suite.sh`
  - 新核心模块：
    - `src/ours/phase_e/contracts.py`
    - `src/ours/phase_e/pairs.py`
    - `src/ours/phase_e/runtime.py`
    - `src/ours/phase_e/training.py`
    - `src/ours/phase_e/benchmark_eval.py`
  - 已完成的真实烟测：
    - `prepare_pairs -> train_value -> ProcessBench eval`
    - `run_phase_e_suite.sh` 的 `E1_MATH_SHEPHERD_PAIR_LEARN_SMOKE`
- 这意味着 `Phase E` 已经从“规划阶段”进入“可直接起跑的实现阶段”。
- 当前成功标准也已更新：
  - 先看 source-family held-out learnability，
  - 再看 benchmark-native 指标，
  - 最后才看 StrategyQA transfer。
- 当前新的执行重点：
  - 不再只跑单个 `Math-Shepherd` baseline，
  - 而是运行 `Math-Shepherd trust matrix`，
  - 用固定脚本从多组 recipe 中选出一个更可托付的 `best_value_head.pt`
    供后续 RL/价值判断阶段复用。
  - 最新 `E15` 结果又进一步说明：
    - 单源 `Math-Shepherd` 已足够证明 learnability，
    - 但 benchmark-native 指标仍弱，
    - 因此下一步主线应切到同家族多源数学数据混训。
  - 2026-03-10 数据语义修复后：
    - `Math-Shepherd` / `RLHFlow` / `PRM800K` fallback 默认只构造严格 `first_bad_edge` pair，
    - 新 artifact 会记录 `pair_build_mode / pair_semantics`，
    - 修复前的 nearest-negative 结果应视为 `legacy`。
  - 2026-03-11 cache/显存可靠性修复后：
    - shared feature cache 已升级为更严格的 provenance 追踪，
      会绑定 index 引用的实际权重 shard；
    - `Phase B` / `Phase C P(IK)` 训练已改为 CPU-cache + mini-batch 搬运；
    - cache hit 不再默认意味着“整包特征常驻 GPU”。
  - 2026-03-11 新增 `intradataset ACC90` 分支：
    - 只看单一数据集自己的 held-out pair ACC/AUC，
    - 不再把 benchmark transfer 混进这一分支，
    - 新入口：`scripts/run_phase_e_intradataset_suite.sh`

## 2. 近期关键实验结论

1. Phase A/B 的基线价值已足够：流程可复现，指标可稳定产出。
2. `DB3` 与 `DB4` 的对照说明：ranking-first 目标可以在 StrategyQA bridge 上产生真实增益，但 joint scalar-calibration 目标会系统性伤害 ranking。
3. 因此当前研究对象更像“process ranking”而不是“scalar value regression”。
4. StrategyQA 缺少公开高质量步骤监督，这会导致明显的 `garbage in, garbage out` 风险。
5. 仓库主线已正式改为：
   - 用有明确步骤监督的数据集先验证 BCR/value-head 思路，
   - 再把已验证的方法迁移回 StrategyQA。
6. 社区/学界的最新结论也支持这条顺序：
   - 跨数据集 PRM 泛化本身困难，
   - 通常需要多域训练、更成熟的 preference/process 构造，或生成式 verifier，
   - 因而不应把“单一数据集训练后立即迁移”作为第一性成功标准。
7. `Math-Shepherd` seed3 的最新结果进一步说明：
   - same-source held-out 可学性是存在的，
   - 但 seed fragility 仍然明显，
   - 所以下一步必须优先解决“checkpoint trustworthiness”，而不是只看最高分。

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

### Phase E（benchmark-native learnability suites）
```bash
bash scripts/run_phase_e_suite.sh

# Math-Shepherd trust matrix（推荐）
bash scripts/run_phase_e_mathshepherd_trust_suite.sh

# 多源数学混训主线（Stage A-E）
bash scripts/run_phase_e_multisource_math_suite.sh

# 单数据集 ACC90 专项套件
bash scripts/run_phase_e_intradataset_suite.sh
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
- 调试手册：`docs/phase_a_debug_playbook.md`, `docs/phase_b_debug_playbook.md`, `docs/phase_c_debug_playbook.md`

## 5. 输出位置

- A/B/C/D 运行日志与摘要：`assets/artifacts/phase_*_logs/`
- 运行产物：`assets/artifacts/phase_*_runs/`
- C1/C2 相关数据：`assets/artifacts/phase_c_data/`, `assets/artifacts/phase_c_eval/`
- 默认状态下仓库不含 `assets\external_datasets`, `assets\model`中各个`safetensors`文件(Model Shard)，以及部分缓存结果`assets\artifact`中的文件。 需要手动补齐这部分文件，代码才能够运行
---

如需“从 0 到结果”的完整流程，请直接看：`docs/readme_full.md`。其中包含部分实际用于shell中执行的命令， 用于复现实验的结果
