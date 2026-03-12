# Phase E / F Overnight Best-Practice Plan (2026-03-13)

## Why this package exists

当前仓库已经走到一个比较明确的分叉口：

1. `Phase E` 还需要继续提升 benchmark-facing verifier / value head；
2. `Phase F` 已经出现了很强的 offline controller 信号；
3. 但还不能把当前 verifier 直接当 RL 主奖励来用。

因此，这一轮 overnight 设计遵循一个保守原则：

1. 先继续做 **benchmark-oriented verifier improvement**；
2. 再做 **Phase F preflight / usability audit**；
3. 暂不做大规模 live RL 或 LM-level RL 更新。

## Literature refresh used in this plan

### 1. VerifyBench
Link: https://arxiv.org/abs/2507.09884

关键信息：

1. verifier 对输入结构和任务表述非常敏感；
2. 不能把 same-source 高分直接等价成 benchmark transfer；
3. 最佳实践是：训练数据要尽量朝 benchmark 几何靠拢。

对当前仓库的直接含义：

1. 优先跑 `PH2_PRM_LOCAL_TA10_MSGRID10_ARCH_SWEEP_SMOKE` 这类 benchmark-oriented hybrid；
2. 不继续把“naive source concat”当主线。

### 2. AbstentionBench
Link: https://arxiv.org/abs/2506.09038

关键信息：

1. reasoning model 的 abstain / reject 行为本身就是独立能力；
2. 一个高分 verifier，如果不能支持更好的 abstention / rejection 决策，也不算真正 deployment-ready。

对当前仓库的直接含义：

1. `Phase F` 不能只看 verifier AUC；
2. 必须继续做 threshold-shift、generator-shift、reward-hacking 预审。

### 3. PURE / Stop Summation
Link: https://arxiv.org/abs/2504.15275

关键信息：

1. naive PRM reward shaping 很容易被 exploit；
2. 直接把过程分数相加做 RL reward 风险很高；
3. 在 reward surface 没有被重新审计前，不应该把 verifier 直接当 RL 主奖励。

对当前仓库的直接含义：

1. `Phase F` 暂时优先 heuristic / preflight，而不是直接进 RL；
2. 先审 `PBR26 / PBR31` 的 fixed-threshold 与 reward-hacking 面。

### 4. Consensus filtering style PRM practice
Representative context: Qwen-style PRM filtering / judge agreement pipelines

关键信息：

1. 最值得补 judge 的不是全量样本，而是低 margin、最模糊的 slice；
2. selective relabel 常常比“全量 judge 重打标”更便宜也更干净。

对当前仓库的直接含义：

1. 优先跑 `run_phase_e_prmbench_selected_relabel_suite.sh`；
2. 用 judge 只处理低 margin 的 `PRMBench_Preview` 训练对子集。

## Selected overnight package

### Phase E improvement jobs

1. `PH2_PRM_LOCAL_TA10_MSGRID10_ARCH_SWEEP_SMOKE`
   - 目标：继续验证 benchmark-oriented hybrid supervision 是否比单源更强。
2. `PRMBench selected relabel`
   - 目标：把 judge / consensus filtering 落到最模糊的低 margin slice 上。
3. `CR1_CURATED_CENTER_GATE_SMOKE`
   - 目标：验证 curated semantic buckets + reward centering + gated head 是否继续提升 transfer。

### Phase F usability job

1. `modern preflight on PBR26 / PBR31`
   - 目标：在进入 live controller 或 RL 前，再看一轮：
     - fixed-threshold stability
     - generator-shift
     - reward-hacking surface

## Operational choice

考虑到本机 GPU VRAM 使用天然波动，本轮选择：

1. 单 GPU 顺序执行；
2. 保守 batch size；
3. 显式 `MAX_GPU_MEMORY_GIB`；
4. 把 `Phase F` 审计放到 E 改良任务之后。

对应 launcher：

- [run_phase_e_phase_f_single_gpu_overnight.sh](/home/zling/y/bcr/ref/scripts/run_phase_e_phase_f_single_gpu_overnight.sh)

## Expected morning readout

第二天早上最值得看的不是单一 headline，而是四类信息：

1. `PH2` 有没有在 `ProcessBench` 上超过旧 benchmark-facing baseline；
2. `selected relabel` 是否比原始 `PRMBench` baseline 更稳；
3. `CR1` 的 reward centering / gated head 是否继续有增益；
4. `PBR26 / PBR31` 在 `modern preflight` 下，是否有一个明显更适合 controller / RL pre-stage。

## Current recommendation

如果 overnight 后没有出现明显反例，下一步优先级仍然是：

1. `Phase E`: 继续沿 benchmark-oriented + judge-filtered 路线推进；
2. `Phase F`: 先做 live heuristic / preflight-backed controller；
3. `RL`: 继续延后，直到 reward-hacking 和 threshold stability 证据更强。
