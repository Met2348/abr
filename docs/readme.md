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

### Phase D（当前主线）
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

### Phase C/D（one-click control + diagnosis）
```bash
bash scripts/run_phase_cd_control_suite.sh
```

## 4. 文档导航

- 总体路线与状态：`docs/readme.md`
- 完整技术手册：`docs/readme_full.md`
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
