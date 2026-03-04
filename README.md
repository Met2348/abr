# BCR/ABR Research Repo

当前已完成 A/B/C 基线，正在推进 D（value supervision + external PRM）。

## 当前进展
- Phase A：稳定，可复现的评测与数据契约。
- Phase B：SFT/PEFT 诊断完成，StrategyQA 有增益，GSM8K 出现退化并已定位为方法层问题而非单纯参数问题。
- Phase C：value head 全链路可跑，但已验证监督噪声与 pair 质量是核心瓶颈。
- Phase D：已接入外部 PRM teacher 并完成融合训练框架，当前主攻“高质量监督构造”。

## 给老师的推荐阅读顺序
1. `docs/readme.md`（总览）
2. `docs/phase_B_report.md`（已完成实验结论）
3. `docs/phase_C_fix_value_head.md`（C 阶段价值头训练情况）
4. `docs/phase_D_plan.md`（当前正式推进方案）
5. `docs/readme_full.md`（完整技术细节）

## 常用入口
- `scripts/run_phase_a_benchmark_suite.sh`
- `scripts/run_phase_b_training_suite.sh`
- `scripts/run_phase_c_value_suite.sh`
- `scripts/run_phase_c_pik_suite.sh`
- `scripts/run_phase_d_teacher_suite.sh`

## 主要代码目录
- `assets/artifacts/phase_a_*`
- `assets/artifacts/phase_b_*`
- `assets/artifacts/phase_c_*`
- `assets/artifacts/phase_d_*`
