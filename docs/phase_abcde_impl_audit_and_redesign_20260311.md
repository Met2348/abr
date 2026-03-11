# A-E 实现审计、社区对照与重设计结论（2026-03-11）

## 2026-03-13 追加更新：B/C/D/E 评测 provenance 收口

这轮又补了一类会直接污染实验结论的实现风险：

1. 多条 standalone eval 路径在请求 `best` checkpoint 时，历史上允许静默回退到 `final`；
2. 部分 `posthoc_calibration=from_run` 路径也会在缺失 `best` calibrator 时静默读 `final` calibrator；
3. 这会导致研究者以为自己在评估：
   - `best checkpoint`
   - `best calibrator`
   但实际拿到的是另一套对象。

本轮已完成的修复：

1. `scripts/phase_b_eval_faithfulness.py`
2. `scripts/phase_c_eval_pik.py`
3. `scripts/phase_d_eval_external_pairs.py`
4. `src/ours/phase_e/runtime.py`

新的统一语义：

1. 默认 `fail`
   - `best` 缺失时直接报错；
2. 只有显式传：
   - `--checkpoint-missing-policy fallback_final`
   - `--posthoc-missing-policy fallback_final`
   才允许 legacy 回退；
3. 一旦发生 fallback，必须同时写入：
   - `metrics.json`
   - `manifest.json`
   - `summary.md`
   - 控制台输出

验证：

1. `18` 个定向单测通过：
   - `tests/unit/test_phase_e_runtime.py`
   - `tests/unit/test_phase_b_eval_faithfulness_script.py`
   - `tests/unit/test_phase_c_eval_pik_script.py`
   - `tests/unit/test_phase_d_eval_external_pairs_script.py`
2. `python -m py_compile` 已覆盖相关 B/C/D/E eval 文件。

补充结论：

1. 这一轮 grep-based 复查没有发现主 Phase E source-bundle 训练链直接吞入 `ProcessBench`/`PRMBench` benchmark 测试集；
2. benchmark 相关脚本主要仍属于：
   - eval
   - alignment audit
   - curated research utilities
3. 因此当前更现实的高危点，已经不再是“benchmark 直接泄漏进主训练”，而是：
   - provenance 解释错误
   - recipe 失真
   - 数据语义与 benchmark 几何不对齐

## 2026-03-11 21:40 补充更新

本文件之后又完成了一轮 `Phase E` 基础设施收口：

1. 所有活跃 `run_phase_e*.sh` wrapper 现已显式传 `--recipe-risk-policy`；
2. 直接 `scripts/phase_e_train_value.py` 的默认 best-checkpoint 指标已改为：
   - `pair_acc`
3. 静态审计证据：
   - `assets/artifacts/phase_e_audits/phase_e_suite_recipe_audit_0311_1931/summary.md`
   - `assets/artifacts/phase_e_audits/phase_e_suite_recipe_audit_postfix_0311_1933/summary.md`

因此，本文件后文关于“旧 wrapper 仍可能把危险 recipe 带回主线”的风险描述，现在应理解为：
1. 历史证据的解释风险仍然存在；
2. 但活跃主线入口已经补齐了安全闸门。

## 0. 证据边界

本报告只采信四类证据：

1. 当前仓库代码与可运行测试。
2. 实际 artifact 与评测输出。
3. `docs/relatedPapers/` 下 PDF 原文与本地抽取文本。
4. 互联网主来源：
   - ProcessBench: https://aclanthology.org/2025.acl-long.50/
   - PRM800K: https://arxiv.org/abs/2305.20050
   - OmegaPRM: https://arxiv.org/abs/2406.06592
   - Tree-PLV: https://arxiv.org/abs/2407.00390
   - Rewarding Progress: https://arxiv.org/abs/2410.08146
   - Lessons of Developing PRMs: https://arxiv.org/abs/2501.07301
   - PRMBench: https://arxiv.org/abs/2501.03124
   - BiRM: https://arxiv.org/abs/2503.04618
   - GenPRM: https://arxiv.org/abs/2504.00891
   - ThinkPRM: https://arxiv.org/abs/2504.16828
   - PURE / Stop Summation: https://arxiv.org/abs/2504.15275
   - R-PRM repo: https://github.com/NJUNLP/R-PRM

补充：

1. 以上列表覆盖了第一轮 A-E 审计需要的主证据；
2. 对 `2025-03 -> 2026-03` 晚近 verifier / reward-model / RLVR 方向变化的
   更系统补充，已单独整理到：
   - `docs/phase_e_literature_refresh_20260311.md`

---

## 1. 当前代码审计结论

### 1.1 这轮新确认并修复的高危问题

1. `feature cache` 锁会错误抢占活着的写进程。
   - 文件：`src/ours/phase_b/feature_cache.py`
   - 旧逻辑只看锁文件 `mtime`，写得慢的活进程会被误判成 stale。
   - 后果：两个进程可能同时进入 cache critical section，直接污染 Phase B/C/D/E 的特征缓存。
   - 修复：只有在 lock owner pid 缺失或已死时才允许清理 stale lock；活进程超时现在会显式 `TimeoutError`。
   - 回归：`tests/unit/test_feature_cache.py`

2. `Phase B` 评估请求 `best` checkpoint 时会静默回退到 `final`。
   - 文件：`scripts/phase_b_eval_faithfulness.py`
   - 后果：研究者以为在评估最佳 checkpoint，实际却在评估 final checkpoint，直接污染 Phase C 结论。
   - 修复：新增显式 `checkpoint_resolution` 记录，控制台警告，写入 `metrics.json` / `manifest.json` / `summary.md` 元数据。
   - 回归：`tests/unit/test_phase_b_eval_faithfulness_script.py`

3. A/B/C 阶段 dtype 别名支持不一致。
   - 文件：
     - `scripts/phase_a_generate_and_eval.py`
     - `scripts/phase_b_prepare_value_data.py`
     - `scripts/phase_b_train_value.py`
     - `scripts/phase_b_eval_faithfulness.py`
     - `scripts/phase_c_prepare_pik_data.py`
     - `scripts/phase_c_train_pik.py`
     - `scripts/phase_c_eval_pik.py`
   - 后果：`bf16/fp16/fp32` 在某些入口能跑、某些入口直接报错，很容易被误判成环境问题或训练问题。
   - 修复：统一支持 `bf16/fp16/fp32` 别名。

### 1.2 本轮复核后仍然重要的旧风险

1. `Phase E` runtime 仍保留 `best -> final` 回退，只是现在会显式 warning。
   - 文件：`src/ours/phase_e/runtime.py`
   - 这是兼容旧 artifact 的折中，不应当被当成“完全无风险”。

2. `ProcessBench F1` 默认仍可能走 oracle threshold sweep。
   - 文件：`scripts/phase_e_eval_benchmark.py`
   - 如果只汇报 oracle F1，会系统性高估 RL-ready 程度。
   - 这轮已经用固定阈值 `0.5` 做了额外诊断，见第 4 节。

3. `Phase B` 的 `posthoc_calibration=from_run` 仍可能从 `best` 回落到 `final` calibrator。
   - 目前还不是主风险，但如果后面把 posthoc 校准主线化，需要和 checkpoint 一样显式记录 provenance。

### 1.3 审计后对 A-E 代码的整体判断

1. 当前仓库的实现质量已经高于“AI agent 随机拼接脚本”的水平。
2. 真正危险的地方，不再是大量语法级 bug，而是：
   - 回退路径是否显式记录，
   - 数据语义是否和 benchmark 对齐，
   - 评测阈值是否诚实，
   - 长度/截断/缓存等基础设施是否污染结论。
3. 现在最应该警惕的是“研究解释错误”，不是“代码完全跑不起来”。

### 1.4 本轮再复核：当前不是“普遍实现崩坏”

这轮又做了一次 fresh audit：

1. `PYTHONPATH=src pytest -q tests/unit`
   - `217 passed, 2 warnings`
2. 活跃 `run_phase_*.sh` 入口：
   - shell 语法通过

因此当前仓库不适合再把“到处都有实现 bug”当成默认解释。

这轮唯一新确认并修掉的高危实现问题是：

1. `Phase A official split` 泄漏
   - 文件：`scripts/phase_a_prepare.py`
   - 旧逻辑：
     - official split 输出时只看请求的 `--source-split`
   - 风险：
     - loader 实际解析出的 `source_split` 可能不同，
     - 从而把 test rows 静默写进 `validation.jsonl`
   - 新逻辑：
     - 以 loader 已记录的有效 `metadata["source_split"]` 为准，
     - 并同时记录 `requested_source_split`
   - 回归：
     - `tests/unit/test_phase_a_prepare_script.py`

---

## 2. 社区现在怎么做

### 2.1 数据侧

社区强方法的共同点，不是“单纯更多数据”，而是更强的监督几何和更强的数据过滤：

1. `PRM800K`
   - 人工步骤标注。
   - `ProcessBench` 原文明确：`1/0` 视为正类，`-1` 视为负类。

2. `Lessons of Developing PRMs`
   - 明确指出：纯 `MC estimation` 泛化最差。
   - `LLM-as-a-judge` 与 `consensus filtering` 明显更强。
   - 还指出很多 PRM 会从 process drift 到 outcome。

3. `OmegaPRM`
   - 用 binary search + MCTS 高效定位 first error。
   - 核心价值是更高质量的 first-error supervision，而不只是多 rollout。

4. `Step-DPO / Math-Step-DPO`
   - 直接给出 sibling-branch 分叉对。
   - 对 `ProcessBench` 这种 first-error / good-vs-bad prefix 任务天然更对齐。

5. `GenPRM / ThinkPRM`
   - 已经明显从“标量判别头”转向“生成式 verifier / critic”。
   - 说明社区 frontier 不再认为 frozen scalar head 是最终形态。

### 2.2 模型侧

1. 强 `ProcessBench` 结果通常需要：
   - LoRA 或全参数微调；
   - 或直接生成式 verifier / critic。
2. `PURE / Stop Summation` 还额外说明：
   - 就算 PRM 已经强了，直接用 sum-form credit assignment 进 RL 也会 reward hack。
   - 真正的 RL-ready 还需要 min-form 或混合 verifiable reward 设计。

### 2.3 评测侧

1. 社区越来越强调：
   - step-level metric，
   - first-error localization，
   - fixed-threshold/realistic deployment metric，
   - 而不是只看 oracle-sweep F1 或内部 pair AUC。
2. 这和仓库最近的经验完全一致：held-out pair_acc 很高，不代表 ProcessBench 真好。

---

## 3. 我们代码和社区的差距

### 3.1 现在已经做对了什么

1. 数据/训练/评测拆分明确，且 artifact 记录比较完整。
2. 截断诊断、recipe 风险检查、nonfinite feature 检查、benchmark/native eval 都已经比较成熟。
3. 仓库已经从“盲目调 head”转向“先诊断监督几何和数据语义”，这点方向是对的。

### 3.2 还差什么

1. 训练目标仍以 pair ranking 为主。
   - 社区强方法更常见的是 step-level BCE / token-end classification / critique generation。

2. 主干仍是 frozen-backbone regime。
   - 这适合做诊断，不适合追社区上限。

3. 仍缺少真正强力的数据过滤闭环。
   - judge / oracle filter 目前更多是局部诊断工具，不是稳定主线。

4. 还没有 honest RL gate。
   - 如果只看 oracle F1，很容易过度乐观。

### 3.3 当前最关键的反模式

1. 用 same-family 高 acc 代替 benchmark transfer。
2. 用 oracle-sweep F1 代替 fixed-threshold deployment metric。
3. 在 mixed semantics 数据上继续堆 head 复杂度，而不先修数据几何。
4. 继续把 fanout/grid 这类长度偏置 pair 当默认主力监督。

---

## 4. 这轮新增实验与结论

### 4.1 已有强基线复核

旧 frozen-head 强基线（`Qwen2.5-7B-Instruct` backbone）仍成立：

| config | data | head | PB Math AUC | PB Math F1 |
|---|---|---:|---:|---:|
| `FIX_C` | `dpo_scale_v1` | `mlp` | 0.721 | 0.399 |
| `FIX_C gated` | `dpo_scale_v1` | `gated_mlp` | 0.749 | 0.393 |

解释：

1. `gated_mlp` 确实能提高 ranking / AUC。
2. 但它并没有同步提高 F1。
3. 这已经在提示：head 不是主瓶颈，数据与 calibration 更关键。

### 4.2 本轮新实验：`PBR10`（PRM-7B backbone + DPO-aligned curation）

训练数据：

1. profile = `math_step_dpo_v1`
2. 目标配比：
   - `dpo_fork = 5200`
   - `ms_strict = 2000`
   - `ms_terminal = 800`
3. 实际 train pair：
   - `sibling_branch = 4679`
   - `local_first_bad_edge = 1783`
   - `terminal_completion_anchor = 485`
4. `min_pair_confidence = 0.7`

这条 recipe 的意义非常明确：

1. 用 DPO sibling-branch 作为主信号；
2. 用 MS strict 补 first-error；
3. 用低比例 terminal 补 all-correct；
4. 明确不用 fanout/grid 当主监督。

#### `PBR10 mlp`

训练：

1. run: `assets/artifacts/phase_e_runs/phase_e_pbr10_prm7b_dpo8k_s42_value_20260311T110527Z`
2. held-out:
   - `pair_acc = 0.906`
   - `auc = 0.859`

Benchmark:

| metric | Math | GSM8K |
|---|---:|---:|
| pair_auc_good_vs_bad | 0.863 | 0.873 |
| first_error_edge_accuracy | 0.844 | 0.915 |
| processbench_f1 (oracle) | 0.659 | 0.724 |
| processbench_f1 (fixed 0.5) | 0.654 | 0.693 |

关键解释：

1. 这已经明显高出旧 frozen-head `0.39~0.40 F1` 档。
2. 更重要的是 fixed-threshold 下仍然高，不是只靠 oracle threshold 扫出来的假强。

#### `PBR10 gated_mlp`

训练：

1. run: `assets/artifacts/phase_e_runs/phase_e_pbr10_prm7b_dpo8k_gated_s42_value_20260311T112849Z`
2. held-out:
   - `pair_acc = 0.914`
   - `auc = 0.873`

Benchmark:

| metric | Math | GSM8K |
|---|---:|---:|
| pair_auc_good_vs_bad | 0.869 | 0.873 |
| first_error_edge_accuracy | 0.852 | 0.925 |
| processbench_f1 (oracle) | 0.647 | 0.708 |
| processbench_f1 (fixed 0.5) | 0.567 | 0.664 |

关键解释：

1. `gated_mlp` 在 ranking/AUC/first-edge 上继续提高。
2. 但 fixed-threshold F1 明显比 `mlp` 更差，尤其是 `Math`。
3. 这说明它学到的更像“排序几何”，不是“部署时可直接用的 calibrated decision boundary”。

### 4.3 这轮实验的核心判断

1. 让仓库真正跨过旧上限的不是复杂 head，而是：
   - DPO sibling-branch 主导的数据几何；
   - 更强的 `Qwen2.5-Math-PRM-7B` backbone。

2. `gated_mlp` 现在应被视为：
   - 一个有价值的 ranking/first-edge probe，
   - 不是当前主线部署头。

3. 当前最好的 benchmark-facing 主线应暂时收敛到：
   - `PBR10 mlp`
   - 并且汇报 fixed-threshold F1，而不是只报 oracle F1。

### 4.4 新一轮 benchmark transfer 复核：`NDS5 / NDS6 / NDS7`

这轮新增实验不是再堆更复杂 head，而是专门问：

1. 是不是 `Math-Shepherd` 的 fanout/grid 长度偏误在害我们？
2. 更好的步骤标签（`RLHFlow`）能不能单独解决问题？
3. `sibling_branch` 是否才是当前最缺的监督几何？

#### `NDS5_MS_STRICT_ONLY_SMOKE`

结果：

1. `PB GSM AUC = 0.4210`
2. `PB Math AUC = 0.4321`

结论：

1. 不是简单把 fanout/grid 去掉就能好；
2. 纯 strict-local 监督本身也不够。

#### `NDS6_RLHFLOW_STRICT_ONLY_SMOKE`

结果：

1. `PB GSM AUC = 0.5022`
2. `PB Math AUC = 0.4520`

结论：

1. 更好的步骤标签有帮助，
2. 但 strict-only 几何仍然不够，尤其对 `ProcessBench Math`。

#### `NDS7_MS_DPO_CALIBRATED_SMOKE`

`mlp` 结果：

1. `PB GSM AUC = 0.5575`
2. `PB Math AUC = 0.5594`
3. `PB Math first_edge = 0.7234`

结论：

1. `Math-Step-DPO sibling_branch` 是这轮最明确有效的新信号。

### 4.5 新候选：`NDS7 + gated_mlp`

在同一份 `NDS7` curated data 上把 head 换成 `gated_mlp` 后，出现了明显档位切换。

3-seed 结果：

1. `PB GSM`
   - `pair_acc = 0.6757 ± 0.0297`
   - `auc = 0.6596 ± 0.0158`
   - `first_edge = 0.7642 ± 0.0415`
2. `PB Math`
   - `pair_acc = 0.8099 ± 0.0112`
   - `auc = 0.7460 ± 0.0145`
   - `first_edge = 0.7447 ± 0.0347`

相比旧的 `PBR5B ms_prm_align_v1 + gated_mlp`：

1. `PB GSM AUC` 从约 `0.589` 提到 `0.660`
2. `PB Math AUC` 从约 `0.613` 提到 `0.746`

这说明：

1. 真正起作用的是：
   - `sibling_branch` 主导的数据几何，
   - 再加一个更适合吸收混合 ranking 信号的 head
2. 当前 frozen-head 主线已经不是“整体迁移不会”，而是进入了一个更窄的问题区间。

---

## 5. 重新设计后的主线方案

### 5.1 数据 curate 主线

默认主线不再是 `ms_align_v1` 一类 mixed geometry 大锅炖，而是：

1. 主体：`sibling_branch` / `fork-point` 对
   - 来源：`Math-Step-DPO` 类数据。
2. 辅助：`local_first_bad_edge`
   - 来源：`Math-Shepherd strict` 或 judge-quality strict data。
3. 少量：`terminal_completion_anchor`
   - 只做低比例辅助，不做 20% 以上大占比默认值。
4. 默认禁用：
   - `first_bad_fanout_prefix_ranking`
   - `good_bad_prefix_grid`
   除非某个专项诊断明确需要。

### 5.1 这轮更新后的更精确主线判断

当前真实问题已经收缩成：

1. `DPO sibling_branch` 已经把 local / global prefix ranking 明显拉起来；
2. 当前主残差不再是“generic ProcessBench transfer collapse”，而是：
   - `all-correct terminal completion ordering`

所以后续 pipeline 不应继续泛泛地找“更好的 source”，而应当：

1. 保留 `NDS7` 这种 `sibling_branch + strict + small terminal` 主骨架；
2. 单独设计 terminal completion 修复支路：
   - terminal-specific BCE / pair loss
   - 或 completion-vs-safe-prefix 对比支路；
3. 暂时不再把纯 `strict-only` 或纯 `RLHFlow strict-only` 当主线回滚。

### 5.2 训练 pipeline 主线

1. backbone：
   - frozen 阶段优先 `Qwen2.5-Math-PRM-7B`，不再默认普通 instruct backbone。
2. objective：
   - 继续以 `ranking_only + score-space` 为安全基线；
   - 但 benchmark 侧必须补 fixed-threshold F1。
3. checkpoint 选择：
   - `pair_acc`
   - 不再把 `ranking_score` 当通用默认。
4. benchmark 报告：
   - 必报 `oracle F1`
   - 必报 `fixed-threshold F1`
   - 必报 `first_error_edge_accuracy`

### 5.3 架构主线

1. `mlp`
   - 当前主线。
   - 理由：fixed-threshold F1 更稳。

2. `gated_mlp`
   - 保留为 secondary probe。
   - 适合回答“排序几何能否更强”，不适合作为当前默认部署头。

3. `dual_head`
   - 当前不升主线。
   - 先前证据显示它还没有稳定解决 local/terminal tradeoff，且固定阈值行为更不清楚。

### 5.4 新一轮 terminal residual 受控修复

围绕当前最强平衡线 `NDS7 + gated_mlp`，又补了一轮只改一个变量的三分支修复：

1. `数据修复`
   - `ms_dpo_terminalboost_v1`
2. `损失修复`
   - `NDS7 + terminal BCE`
3. `架构修复`
   - `NDS7 + dual_head + terminal BCE`

目的不是“再碰运气跑一个新配方”，而是精确归因：

1. terminal 残差到底更像数据问题，
2. 还是 loss 问题，
3. 还是 head / routing 问题。

#### 基线：`NDS7 + gated_mlp`

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.6565` | `0.6456` | `0.8049` | `0.2174` | `-0.1640` |
| `PB Math`  | `0.7992` | `0.7519` | `0.7447` | `0.1538` | `-0.1461` |

#### A. 数据修复：`ms_dpo_terminalboost_v1`

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.6752` | `0.6816` | `0.7118` | `0.2798` | `-0.1038` |
| `PB Math`  | `0.7316` | `0.7046` | `0.6848` | `0.1823` | `-0.1670` |

解释：

1. terminal 增强确实能修 `GSM`；
2. 但会带来 `Math` 的 local/global tradeoff；
3. 所以“继续加 terminal 数据”不能当默认主线。

#### B. 损失修复：`NDS7 + terminal BCE`

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.6053` | `0.6612` | `0.6353` | `0.2073` | `-0.1383` |
| `PB Math`  | `0.7229` | `0.7049` | `0.6555` | `0.1650` | `-0.1842` |

解释：

1. 单独加 `terminal BCE` 没有修掉残差；
2. 主要是拖累 local/global，而 terminal 只得到极小收益。

#### C. 架构修复：`NDS7 + dual_head + terminal BCE`

| benchmark | pair_acc | auc | first_edge | terminal_top1 | terminal_gap |
|---|---:|---:|---:|---:|---:|
| `PB GSM8K` | `0.8150` | `0.7489` | `0.7882` | `0.0829` | `-0.2312` |
| `PB Math`  | `0.7820` | `0.7105` | `0.6806` | `0.0739` | `-0.2535` |

解释：

1. 它把模型推成了更强的局部错误探测器；
2. 但 terminal completion 明显更差；
3. 这说明 local / terminal 的确是可分解子任务，
   但当前 `dual_head` 的 inference mixing 还不对。

#### 这一轮受控修复后的新判断

1. 当前主问题已经不是“generic ProcessBench transfer collapse”；
2. 也不是“只要再多一点 terminal anchor 就行”；
3. 更准确地说，是：
   - 如何在不牺牲 local discrimination 的前提下，修正 terminal completion ordering。

---

## 6. 下一步实验建议

### 6.1 最高优先级

1. `PBR10 mlp` 做 3 seeds。
   - 目标：确认这不是单 seed 偶然。
   - 评测必须同时保留 oracle 与 fixed-threshold F1。

2. `PBR10 mlp` 做最小 LoRA 对照。
   - 只解冻最后几层 / rank 小。
   - 目的是判断 frozen ceiling 是否还能继续破。

3. 固定阈值的 threshold provenance 正式化。
   - 后面不要再只用 benchmark 内 oracle sweep 说“RL-ready”。

### 6.2 中优先级

1. judge / oracle filtering 只用于局部高风险子集：
   - terminal anchors；
   - 低 margin strict local pairs；
   - 不再做 bulk filter 主线。

2. 引入 step-level absolute calibration。
   - 即使不改成完整 generative verifier，也要补更强的绝对正确性信号。

### 6.3 暂缓

1. 再继续在 `fanout/grid` mixed profiles 上堆 head。
2. 把 `gated_mlp` 或 `dual_head` 直接升为主线。
3. 在 oracle-threshold 指标上直接宣称 RL-ready。

---

## 7. 当前最终判断

1. A-E 阶段的代码现在已经具备“可信诊断平台”水准，但不是“随便跑都科学”的状态。
2. 当前主问题已经从“会不会训练塌”转成“什么监督几何和什么 backbone 才能对齐 ProcessBench”。
3. 本轮最关键的新结论是：
   - `PBR10 mlp` 已经把仓库推进到一个新的强基线；
   - `gated_mlp` 证明 head 还能改善 ranking，但 calibration 没跟上；
   - 所以下一步最值钱的是 `LoRA + PBR10 mlp`，不是再做一轮 head churn。
