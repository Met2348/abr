# Phase E Judge LLM 选型与本地部署记录（2026-03-11）

## 1. 目标

本轮工作的目标不是立刻把 `LLM-as-a-judge` 接进主训练，而是先解决三个更基础的问题：

1. 社区和论文里，当前更推荐什么样的 judge 技术路线。
2. 在当前服务器资源下，哪些本地 judge 模型最值得优先落地。
3. 下载后，这些模型在本仓库的本地推理链里是否真的可用。

## 2. 外部证据与社区结论

这轮选型结合了：

1. 互联网公开资料 / model cards
2. 本地已下载 PDF 论文
3. 仓库内已有综合调研文档

重点证据：

1. `ProcessBench`
   - 论文与基准都显示：prompted critic / judge models 往往明显强于传统 PRM。
   - 链接：`https://arxiv.org/abs/2412.06559`
2. `ThinkPRM`
   - 指出直接 `LLM-as-a-judge` 不是终点，但强 judge 生成高质量 verification traces 很有效。
   - 链接：`https://arxiv.org/abs/2504.16828`
3. `The Lessons of Developing Process Reward Models in Mathematical Reasoning`
   - 明确支持：`MC-estimation < LLM-judge < human annotation`。
   - 链接：`https://arxiv.org/abs/2501.07301`
4. `DeepSeek-R1` model card
   - 对本地调用方式有明确推荐：
     - 避免 system prompt
     - 推荐 `temperature=0.6`
     - reasoning 模式建议从 `<think>\n` 起手
   - 链接：`https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
5. `Qwen2.5-Math-7B-Instruct` model card
   - 适合数学推理和结构化任务，是便宜、可本地化的 bulk math judge 候选。
   - 链接：`https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct`
6. `QwQ-32B`
   - 从 `ProcessBench` 相关讨论看，它是强 open-source critic 候选，但本地日常成本明显更高。
   - 链接：`https://huggingface.co/Qwen/QwQ-32B`

与本仓库已有文档一致的部分：

1. `docs/bcr_synthesis_report_20260311.md`
   - 已明确记录：`MC < LLM-judge < human`
2. `docs/phase_E_multisource_math_plan.md`
   - 已明确记录：当前主线瓶颈是 supervision quality / geometry，而更强 judge supervision 是合理下一步

## 3. 当前服务器约束

本轮检查时的资源状态：

1. GPU 空闲度
   - GPU0: 约 `3.3G / 80G` 已用
   - GPU3: 约 `5.7G / 80G` 已用
   - 这两个卡适合做 7B / 14B smoke
2. 磁盘
   - `/home/zling` 尚余约 `348G`
3. 已有模型
   - `assets/models/Qwen2.5-7B-Instruct`
   - `assets/models/Qwen2.5-Math-PRM-7B`

因此这轮不适合直接上 `QwQ-32B` 做日常主 judge；更合理的是先落地：

1. 一个便宜、可批量运行的 7B judge
2. 一个更强但更贵的 14B adjudicator

## 4. 技术选型

### 4.1 最终选择

主张采用两级 judge 栈：

1. `Qwen2.5-Math-7B-Instruct`
   - 角色：bulk math judge / triage judge
   - 作用：
     - 先筛低质量样本
     - 先对 step 做粗粒度判定
     - 为 active learning / disagreement mining 提供便宜初判
2. `DeepSeek-R1-Distill-Qwen-14B`
   - 角色：strong adjudicator / disagreement resolver
   - 作用：
     - 只处理低置信 / 高争议 / 复判样本
     - 不作为当前本地主线 bulk judge
3. 保留已有 `Qwen2.5-7B-Instruct`
   - 角色：最便宜的通用 baseline judge
   - 作用：只做 sanity baseline，不建议作为主 judge

### 4.2 为什么没有先下载 `QwQ-32B`

不是因为它不强，而是因为它当前不符合“最适合当下服务器”的标准：

1. `ProcessBench` 方向上它是强 open-source critic，但成本高。
2. 当前 Phase E 还处在 judge contract / pipeline 对接期，没必要先上 32B。
3. 14B 已足够做本地强 judge feasibility check。

## 5. 下载与落盘情况

本轮已下载：

1. `assets/models/Qwen2.5-Math-7B-Instruct`
2. `assets/models/DeepSeek-R1-Distill-Qwen-14B`

并验证目录存在：

1. `config.json`
2. tokenizer 文件
3. safetensors shards

## 6. 新增本地 smoke 工具

为了让 judge 接线变成可重复工程，而不是临时 prompt 实验，本轮新增：

1. `scripts/phase_e_smoke_judge_llm.py`

这个脚本支持：

1. 加载一个本地 instruct model
2. 发送严格 JSON 的 step-judge prompt
3. 跑小型 sanity examples
4. 输出 artifact 到：
   - `assets/artifacts/phase_e_judge_smoke/`

当前 smoke 的评价维度：

1. `json_ok`
2. `overall_match`
3. `first_incorrect_match`
4. 原始输出是否出现异常退化（如无意义重复）

## 7. 本地可用性调试结果

### 7.1 `Qwen2.5-7B-Instruct`

artifact：

1. `assets/artifacts/phase_e_judge_smoke/judge_smoke_qwen25_7b_20260311T071409Z`

结果：

1. `n_json_ok = 1 / 2`
2. `n_overall_match = 1 / 2`
3. `n_first_incorrect_match = 1 / 2`

诊断：

1. 对 `math_good` 能输出正确 JSON。
2. 对 `math_bad` 出现 `!!!!!!!!...` 异常退化。
3. 因此它可以作为极便宜 baseline，但不适合直接升为主 judge。

### 7.2 `Qwen2.5-Math-7B-Instruct`

artifacts：

1. `assets/artifacts/phase_e_judge_smoke/judge_smoke_qwen25_math_7b_20260311T071409Z`
2. `assets/artifacts/phase_e_judge_smoke/judge_smoke_qwen25_math_7b_v2_20260311T071746Z`

`v2` 结果：

1. `n_json_ok = 1 / 2`
2. `n_overall_match = 1 / 2`
3. `n_first_incorrect_match = 1 / 2`

更细的观察：

1. 它在语义上能正确指出 `math_bad` 的第一处错误是 step 2。
2. 但它经常先写解释，再给伪 JSON 或被 `\boxed{...}` 风格污染。
3. 对 `math_good`，经过更鲁棒的 parser，已经能稳定恢复出正确 JSON。

工程结论：

1. 它已经可以作为 bulk math judge 候选。
2. 但正式接入时不应直接假设“总能 strict JSON”。
3. 更稳妥的做法是：
   - 允许 tolerant parser
   - 或先让它输出判定字段，再做后处理标准化

### 7.3 `DeepSeek-R1-Distill-Qwen-14B`

artifacts：

1. `assets/artifacts/phase_e_judge_smoke/judge_smoke_deepseek_r1_qwen14b_20260311T071746Z`
2. `assets/artifacts/phase_e_judge_smoke/judge_smoke_deepseek_r1_qwen14b_v2_20260311T072126Z`

测试情况：

1. 第一轮：沿用通用 system-prompt judge contract
   - 两个样本都退化成 `!!!!!!!!...`
2. 第二轮：按 model card 调整
   - 去掉 system prompt
   - user-only contract
   - assistant prefix 为 `<think>\n`
   - 仍然退化成重复输出
3. 第三轮：再按 model card 增加 sampling
   - `temperature=0.6`
   - `top_p=0.95`
   - 直接触发 `probability tensor contains inf/nan/<0` 的 generation 错误

工程结论：

1. 14B 模型本身已经成功落盘并可加载。
2. 但在当前本地 `transformers + direct generate + strict JSON judge prompt` 链路下，表现不稳定。
3. 所以它现在不适合直接作为 Phase E 的主线 local judge。
4. 若后续仍要用它，建议改成：
   - `vLLM` / `SGLang` 服务化调用
   - 或只用于少量人工复判 / exploratory analysis

## 8. 现阶段推荐的 judge 方案

### 8.1 立即可用方案

1. 主 bulk judge：`Qwen2.5-Math-7B-Instruct`
2. 解析策略：
   - tolerant JSON extraction
   - 重点保存：
     - `overall_verdict`
     - `first_incorrect_step`
     - `step_labels`
3. 使用场景：
   - 低成本重标注
   - low-confidence pair 复审
   - active learning 候选排序

### 8.2 暂不主推方案

1. `DeepSeek-R1-Distill-Qwen-14B`
   - 原因不是能力弱，而是当前本地推理协议不稳
2. `Qwen2.5-7B-Instruct`
   - 只建议保留为 baseline judge

## 9. 对 Phase E 的直接影响

这轮选型后的现实落点是：

1. 我们已经有一个可以立即接进 pipeline 的本地 bulk judge 候选：
   - `Qwen2.5-Math-7B-Instruct`
2. 我们也验证了：
   - 更强 reasoning 模型并不会自动变成稳定的 structured local judge
3. 因此 Phase E 下一步不该直接把 `14B reasoning model` 硬接进主线，而应该先：
   - 用 `Qwen2.5-Math-7B-Instruct` 做一版 judge relabel / disagreement-mining pilot
   - 把 output parsing、artifact 留存、judge confidence 统计链补齐

## 10. 后续建议

1. 新增 `Phase E judge relabel` 支路：
   - 输入：低置信 / 高分歧 pair
   - judge：`Qwen2.5-Math-7B-Instruct`
   - 输出：machine-readable relabel artifact
2. 把 `DeepSeek-R1-Distill-Qwen-14B` 留作：
   - 少量 hard-case adjudication
   - 或未来服务化部署后再复评
3. 如果 judge 路线有效，再考虑是否值得上 `QwQ-32B`

## 11. 这轮最重要的一句话

当前服务器条件下，最合理的本地 judge 技术选型不是“直接上最强 reasoning 模型”，而是：

1. 先用 `Qwen2.5-Math-7B-Instruct` 建立稳定、可批量、可留存 artifact 的 judge 流水线；
2. 再把更强的 14B/32B judge 当作有限预算下的二级 adjudicator。

## 12. 两个 judge LLM 的真实 benchmark 小实验

为了避免只凭玩具样例下结论，这轮又补了一个真实 benchmark 小实验：

1. 新脚本：
   - `scripts/phase_e_benchmark_judge_llm.py`
2. 数据：
   - `ProcessBench math`
   - `ProcessBench gsm8k`
3. 每个 benchmark 各取 6 条确定性子样本
4. 评测两种输出契约：
   - `full_steps`
   - `first_bad_only`

### 12.1 实际运行命令

`Qwen2.5-Math-7B-Instruct`，完整 step 契约：

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_e_benchmark_judge_llm.py \
  --model-path assets/models/Qwen2.5-Math-7B-Instruct \
  --run-name judge_bench_qwen25_math_7b \
  --benchmark processbench_math \
  --benchmark processbench_gsm8k \
  --max-samples-per-benchmark 6 \
  --max-new-tokens 256
```

`DeepSeek-R1-Distill-Qwen-14B`，完整 step 契约：

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_e_benchmark_judge_llm.py \
  --model-path assets/models/DeepSeek-R1-Distill-Qwen-14B \
  --run-name judge_bench_deepseek_r1_14b \
  --benchmark processbench_math \
  --benchmark processbench_gsm8k \
  --max-samples-per-benchmark 6 \
  --max-new-tokens 256 \
  --no-use-system-prompt \
  --assistant-prefix '<think>\n'
```

`Qwen2.5-Math-7B-Instruct`，轻量 `FIRST_BAD_ONLY` 契约：

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_e_benchmark_judge_llm.py \
  --model-path assets/models/Qwen2.5-Math-7B-Instruct \
  --run-name judge_bench_qwen25_math_7b_fbonly \
  --benchmark processbench_math \
  --benchmark processbench_gsm8k \
  --max-samples-per-benchmark 6 \
  --max-new-tokens 160 \
  --contract-mode first_bad_only
```

`DeepSeek-R1-Distill-Qwen-14B`，轻量 `FIRST_BAD_ONLY` 契约：

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_e_benchmark_judge_llm.py \
  --model-path assets/models/DeepSeek-R1-Distill-Qwen-14B \
  --run-name judge_bench_deepseek_r1_14b_fbonly \
  --benchmark processbench_math \
  --benchmark processbench_gsm8k \
  --max-samples-per-benchmark 6 \
  --max-new-tokens 160 \
  --contract-mode first_bad_only \
  --no-use-system-prompt \
  --assistant-prefix '<think>\n'
```

### 12.2 结果 artifact

1. 原始 run：
   - `assets/artifacts/phase_e_judge_bench/judge_bench_qwen25_math_7b_20260311T073507Z`
   - `assets/artifacts/phase_e_judge_bench/judge_bench_deepseek_r1_14b_20260311T073507Z`
   - `assets/artifacts/phase_e_judge_bench/judge_bench_qwen25_math_7b_fbonly_20260311T074138Z`
   - `assets/artifacts/phase_e_judge_bench/judge_bench_deepseek_r1_14b_fbonly_20260311T074138Z`
2. 统一 compare：
   - `assets/artifacts/phase_e_judge_bench_compare/judge_model_compare_20260311T074514Z/summary.md`

### 12.3 统一结果表

| run | benchmark | parse_ok | overall_acc | first_bad_exact | first_bad_within1 | mean_step_acc |
|---|---|---:|---:|---:|---:|---:|
| qwen_math_7b_full_steps | processbench_math | 0.6667 | 0.3333 | 0.0000 | 0.0000 | 0.4167 |
| qwen_math_7b_full_steps | processbench_gsm8k | 1.0000 | 0.5000 | 0.0000 | 0.0000 | 0.5893 |
| deepseek_r1_14b_full_steps | processbench_math | 0.5000 | 0.1667 | 0.0000 | 0.0000 | 0.2500 |
| deepseek_r1_14b_full_steps | processbench_gsm8k | 0.5000 | 0.3333 | 0.0000 | 0.3333 | 0.3611 |
| qwen_math_7b_first_bad_only | processbench_math | 0.5000 | 0.3333 | 0.0000 | 0.0000 | 0.4167 |
| qwen_math_7b_first_bad_only | processbench_gsm8k | 1.0000 | 0.5000 | 0.0000 | 0.0000 | 0.5893 |
| deepseek_r1_14b_first_bad_only | processbench_math | 0.5000 | 0.1667 | 0.0000 | 0.0000 | 0.2500 |
| deepseek_r1_14b_first_bad_only | processbench_gsm8k | 0.8333 | 0.6667 | 0.3333 | 0.3333 | 0.6667 |

### 12.4 关键解读

1. `Qwen2.5-Math-7B-Instruct`
   - 在真实样本上并不是不会 judge。
   - 它的更大问题是明显偏向：
     - `OVERALL=correct`
     - `FIRST_BAD=none`
   - 即使把契约简化成 `first_bad_only`，它也仍然几乎不报首错步。
   - 这说明它的主问题不是“输出太长”，而是“在真实长样本上过度乐观”。

2. `DeepSeek-R1-Distill-Qwen-14B`
   - 在 `full_steps` 契约下依然不够稳。
   - 但在 `first_bad_only` 契约下，`gsm8k` 上终于出现了实质性 signal：
     - `overall_acc = 0.6667`
     - `first_bad_exact = 0.3333`
   - 这说明它并非完全不适合 judge，而是：
     - 更适合 lighter contract
     - 更适合 wordy / narrative `gsm8k` 风格
   - 但在 `math` 上仍然明显不够好。

3. 两个模型都还不能直接当成“ProcessBench-ready 主 judge”。

### 12.5 当前最务实的部署结论

1. 数学主线：
   - 先继续把 `Qwen2.5-Math-7B-Instruct` 当 bulk triage judge
   - 但它更适合：
     - 粗筛
     - 低置信样本重标
     - disagreement mining
   - 不适合直接当“精确首错步标注器”

2. `DeepSeek-R1-Distill-Qwen-14B`
   - 不应作为本地主 judge 批量跑
   - 但它在轻量 `first_bad_only` 契约下，已经显示出对 `gsm8k` 的潜力
   - 更适合后续当：
     - 二级 adjudicator
     - 或只在 hard cases 上复判

3. 如果下一步要把 judge 真的接进 Phase E：
   - 不要直接要求 `full_steps + strict JSON`
   - 先从：
     - `first_bad_only`
     - tolerant parser
     - low-confidence pair relabel
     这条便宜而稳的链开始。

## 13. Pairwise + Swap-Debias judge benchmark

第 12 节解决的是 pointwise judge 的问题。  
这一步继续验证文献和社区更常推荐的 judge 用法：

1. pairwise comparison
2. swap-debias (`A/B` 与 `B/A`)
3. 轻量 anchored output contract

### 13.1 新脚本

1. `scripts/phase_e_pairwise_judge_benchmark.py`

它直接对 canonical pair JSONL 做：
1. `A/B` 判定
2. `B/A` 判定
3. majority 聚合
4. `swap_consistency_rate`
5. `label_preserving_keep_rate`

### 13.2 统一 compare artifact

1. `assets/artifacts/phase_e_pairwise_judge_compare/judge_pairwise_compare_20260311T084132Z/summary.md`

### 13.3 关键结果

`Qwen2.5-Math-7B-Instruct` on held-out `PRMBench_Preview` (`32` pairs):
1. `both_parse_ok_rate = 0.3125`
2. `pair_acc_majority = 0.3438`
3. `swap_consistency_rate = 0.6000`
4. `label_preserving_keep_rate = 0.1250`

`Qwen2.5-Math-7B-Instruct` on held-out `Math-Shepherd local_first_bad_edge` (`32` pairs):
1. `both_parse_ok_rate = 0.2188`
2. `pair_acc_majority = 0.0625`
3. `label_preserving_keep_rate = 0.0312`
4. `judge_contradiction_rate = 0.0625`

`Qwen2.5-Math-7B-Instruct` as train-slice filter:
1. `PRMBench_Preview train64`: `keep_rate = 0.0469`
2. `Math-Shepherd train64`: `keep_rate = 0.0000`

`DeepSeek-R1-Distill-Qwen-14B` on held-out `PRMBench_Preview` (`16` pairs):
1. `both_parse_ok_rate = 0.5625`
2. `pair_acc_majority = 0.0625`
3. `tie_rate = 0.8125`

### 13.4 这组实验真正说明了什么

1. pairwise + swap-debias 的 judge 用法，方向上确实比 pointwise strict-JSON 更对；
2. 但当前本地小 judge 仍然不够强，不能被提升成 bulk Phase E pair filter；
3. `PRMBench_Preview` 比 `Math-Shepherd local_first_bad_edge` 更接近 judge-friendly pair semantics；
4. `Math-Shepherd` 上的主要问题不是 parser，而是语义错位：
   - 更短但仍然正确的安全前缀，
   - 常被 judge 误判得不如更长但已经出错的前缀。

### 13.5 当前最合理的 judge 工程定位

1. judge 继续保留为：
   - disagreement mining
   - selected relabel
   - benchmark-side adjudication
2. 不要把当前本地 judge 直接接成 bulk Phase E pair filter
3. 如果继续推进 judge 主线，优先做：
   - `PRMBench_Preview` 风格 pairwise benchmark
   - selected hard-case relabel
   - 而不是 `Math-Shepherd local_first_bad_edge` 的大规模自动过滤
