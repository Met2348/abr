# Result Records (Phase A)

Last updated: 2026-02-27 01:16:23 +0800

## 0. Latest Diagnosis Update (2026-02-26, sweeper2 + A5)

### 0.1 What Problem We Tried To Fix

The active issue was poor output compliance in StrategyQA yes/no evaluation:
- CoT runs produced long explanations without extractable final yes/no.
- Parse errors dominated total accuracy at mid token budgets.
- We introduced strict yes/no prompting (`A5`) to force parseable outputs.

### 0.2 New Results Added

#### 0.2.1 CoT Sweeper2 (`qa_cot_then_final`, n=193)

| Run Dir | Max New Tokens | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---:|---:|---:|---:|---:|
| `strat_cot_sweep_t128_20260226T103204Z` | 128 | 0.1036 | 0.8497 | 29 | 0.6897 |
| `strat_cot_sweep_t192_20260226T104650Z` | 192 | 0.2591 | 0.6684 | 64 | 0.7812 |
| `strat_cot_sweep_t256_20260226T110535Z` | 256 | 0.4974 | 0.3212 | 131 | 0.7328 |
| `strat_cot_sweep_t320_20260226T112825Z` | 320 | 0.6684 | 0.1140 | 171 | 0.7544 |
| `strat_cot_sweep_t384_20260226T115617Z` | 384 | 0.6839 | 0.0518 | 183 | 0.7213 |

#### 0.2.2 A5 Strict Compliance Fix (`qa_binary_strict`, n=193)

| Run Dir | Max New Tokens | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---:|---:|---:|---:|---:|
| `my_run_baseline_direct_t16_20260226T125946Z` | 16 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `my_run_strict_t4_20260226T130128Z` | 4 | 0.5078 | 0.2383 | 147 | 0.6667 |
| `my_run_strict_t8_20260226T130206Z` | 8 | 0.5078 | 0.2383 | 147 | 0.6667 |
| `my_run_strict_t16_20260226T130259Z` | 16 | 0.5285 | 0.0363 | 186 | 0.5484 |
| `my_run_strict_t16_20260226T130431Z` | 16 | 0.5285 | 0.0363 | 186 | 0.5484 |

### 0.3 Updated Diagnosis

1. CoT quality was suppressed by parse failures, not only by reasoning quality.
   - Evidence: parseable-only accuracy is already high at low token budgets (`0.6897` at `t128`) despite very low total accuracy.
2. CoT with enough token budget becomes the strongest quality setup.
   - `t320/t384` clearly outperform direct baseline total accuracy.
3. Strict prompt is an effective compliance control, but not a full quality fix.
   - `strict_t16` sharply reduces parse errors (`0.0363`) but lowers parseable-only correctness (`0.5484`).
4. Extremely short strict budgets (`t4/t8`) over-truncate.
   - They do not improve either total accuracy or parse errors vs direct baseline.
5. Reproducibility is confirmed for strict runs.
   - `strict_t16` repeated with identical metrics and zero changed samples.

### 0.4 Fix Verdict By Problem

| Problem | Status | Notes |
|---|---|---|
| Parse/compliance failures | Partially fixed | Strongly improved by strict template or larger CoT budget. |
| Overall answer quality | Mixed | Best with large CoT budget; strict short format alone hurts quality. |
| Inference efficiency | Unfixed for high-quality mode | CoT `t320/t384` is accurate but expensive. |
| Deterministic reproducibility | Fixed | Repeat runs match exactly under no-sample seed-fixed settings. |

### 0.5 Next Plan (Actionable)

1. Keep two official baselines:
   - Quality baseline: CoT `t320` and `t384`.
   - Efficiency baseline: direct `t16` or `t32`.
2. Tighten strict-mode decoding boundaries:
   - stop sequences for `Human:`, `User:`, `[SYSTEM]`, `[END]`.
   - keep strict runs at small token limits (`16-32`) after stop fix.
3. Add extractor robustness checks:
   - test explicit variants like `yes.`, `no.` and leakage strings (`noHuman:`).
4. Run a targeted strict follow-up sweep after stop/extractor fix:
   - tokens: `16, 24, 32`.
   - compare against `baseline_direct_t16` and `strat_cot_sweep_t320`.
5. Freeze one benchmark profile for Phase A reporting:
   - deterministic, no-sampling, fixed artifact fingerprint, fixed evaluator version.

### 0.6 Parser/Decode Remedy Update (2026-02-26 late)

We implemented a targeted remedy for parse/compliance failures:
1. stronger StrategyQA extraction rules,
2. chat-leak truncation (`[USER]`, `Human:` etc.),
3. optional binary-choice decode mode (`yes` vs `no` scoring) to remove free-form formatting variance.

Quick re-evaluation on existing predictions (same generation outputs, new parser):

| Run Dir | Old Accuracy | Old Parse Error | New Accuracy | New Parse Error | New Accuracy on Parseable |
|---|---:|---:|---:|---:|---:|
| `strat_cot_sweep_t128_20260226T103204Z` | 0.1036 | 0.8497 | 0.1036 | 0.8446 | 0.6667 |
| `strat_cot_sweep_t320_20260226T112825Z` | 0.6684 | 0.1140 | 0.6684 | 0.1088 | 0.7500 |
| `strat_cot_sweep_t384_20260226T115617Z` | 0.6839 | 0.0518 | 0.6995 | 0.0518 | 0.7377 |
| `my_run_baseline_direct_t16_20260226T125946Z` | 0.5492 | 0.1762 | 0.5492 | 0.1762 | 0.6667 |
| `my_run_strict_t16_20260226T130259Z` | 0.5285 | 0.0363 | 0.6580 | 0.0000 | 0.6580 |

Interpretation:
- A non-trivial share of previous parse errors came from evaluation/code-side extraction brittleness (especially strict leakage forms like `noHuman:`).
- Remaining CoT parse errors at low token budgets are still largely model/protocol behavior (model does not emit an explicit binary decision often enough).
- To isolate true model decision quality from formatting noise, use `strategyqa_decode_mode=binary_choice` in A6 runs.

### 0.7 A6 Confirmation + Freeform-vs-Binary Diagnosis (2026-02-26)

New A6 run (`strategyqa_decode_mode=binary_choice`) result:

| label | n | accuracy | parse_error_rate | acc_parseable |
|---|---:|---:|---:|---:|
| `direct_binchoice` | 193 | 0.6788 | 0.0000 | 0.6788 |
| `cot_binchoice` | 193 | 0.5803 | 0.0000 | 0.5803 |
| `direct_binchoice_repro` | 193 | 0.6788 | 0.0000 | 0.6788 |

CoT A2 under binary-choice decode (token sweep):
- `t128/t192/t256/t320/t384` all reported exactly `acc=0.5803`, `parse_err=0.0000`.

Root-cause conclusion:
1. **Parse-error problem is mostly protocol/extraction/generation-format**, not pure model incapability.
   - Binary-choice decode eliminates parse errors to `0.0000`.
2. **Model-quality problem is still real after removing formatting noise**.
   - Direct binary-choice reaches `0.6788`, while CoT-prompt binary-choice is lower (`0.5803`).
3. **“Accuracy on parseable subset” can be misleading alone**.
   - Freeform CoT `t128` has `acc_parseable=0.6667` but only `n_parseable=30/193`.
   - Binary-choice gives full coverage (`193/193`) and is the fairer decision-quality view.

Important interpretation note:
- Under current implementation, StrategyQA `binary_choice` mode scores yes/no directly and does not depend on `max_new_tokens`.
- Therefore identical A2 token-sweep accuracies in binary mode are expected behavior, not a runtime bug.

Plan update:
1. Use two official StrategyQA metrics tracks:
   - **Decision quality track**: `binary_choice` (coverage fixed at 100%).
   - **End-to-end format track**: `freeform` (includes compliance burden).
2. Keep direct prompt as binary-choice baseline (`~0.6788`) for Phase A.
3. For CoT prompt, improve question framing/system instruction before attributing gap to model limits.

### 0.8 Phase A Closeout (Concise)

Conclusion:
- Phase A goals are met for benchmark infrastructure and diagnosis quality.
- We now have a reliable split between:
  - decision-quality evaluation (`binary_choice`), and
  - end-to-end formatting/compliance evaluation (`freeform`).

Achievements:
1. Deterministic one-click benchmark suite with param groups (`A1..A6`) and persisted logs.
2. Stable prepared-data pipeline and evaluation pipeline for StrategyQA.
3. Parse-error root cause identified and partially fixed in extraction/decoding.
4. Reproducibility validated (repeat runs produce identical results under fixed settings).

Problems probed and fixed:
1. Prompt-leak parse failures (`noHuman: ...`) fixed via extraction hardening and marker truncation.
2. Misleading evaluation from parseable-only subset mitigated by always reporting:
   - `n_parseable`,
   - `acc_parseable`,
   - and total-coverage metrics.
3. Free-form format variance isolated using `binary_choice` mode.

Open (unsolved) problems:
1. CoT prompt underperforms direct prompt on decision quality (`0.5803` vs `0.6788`).
2. Free-form CoT still has severe formatting burden at small budgets.
3. Phase A is inference-only; value learning and faithfulness-training components are not implemented yet.

To-do next (immediate):
1. Freeze Phase A benchmark protocol (artifact fingerprint + decode mode + reporting schema).
2. Start Phase B in separate scripts/modules:
   - model wrapper + value head,
   - BCR-lite losses,
   - first training smoke runs.
3. Keep Phase A scripts unchanged as regression baselines.

Re-planned route to final ABR/BCR goal:
1. Stage B: BCR-lite implementation and training stability.
2. Stage C: faithfulness calibration/corruption metrics and robust ablations.
3. Stage D: ABR heuristic router (`gen/ver/fin`) with fixed budget.
4. Stage E: ABR learned router (RL) only after Stage D is stable and reproducible.

### 0.9 GSM8K Smoke Criticality Scan and Fix (2026-02-26)

Observed critical symptom (direct baseline smoke, 20 samples):
- run: `gsm8k_direct_smoke_20260226T150012Z`
- `accuracy=0.0500`, `parse_error_rate=0.0000`
- extraction methods: 20/20 were `last_number`.

Diagnosis:
1. This was not a parser crash, but a weak extraction regime.
2. Model outputs were partial reasoning text under token cap; evaluator fell back to `last_number`.
3. Therefore parse-error stayed zero while answer quality collapsed.

Fixes implemented:
1. Added new prompt template `qa_math_direct_final@1.0.0`:
   - forces one-line `Final answer: <number>` format.
2. Extended math extraction:
   - supports both `Final answer: ...` and `Final answer is ...`.
3. Added math diagnostics in `phase_a_generate_and_eval.py`:
   - `last_number_rate`,
   - `hit_token_limit_rate`,
   - `final_answer_tag_rate`,
   - warning text when extraction looks unreliable.

Post-fix smoke validation:
1. run: `gsm8k_math_direct_smoke_20260226T150913Z` (`max_new_tokens=64`)
   - `accuracy=0.6000`, `parse_error_rate=0.0000`,
   - `last_number_rate=0.0000`,
   - `final_answer_tag_rate=1.0000`.
2. run: `gsm8k_math_direct_smoke_t16_20260226T151029Z` (`max_new_tokens=16`)
   - `accuracy=0.6000`, `parse_error_rate=0.0000`,
   - same smoke accuracy with much faster runtime.

Practical guidance:
1. For GSM8K direct baseline in Phase A, use `qa_math_direct_final`.
2. Start with `max_new_tokens=16` for fast baseline iteration.
3. Treat heavy `last_number` reliance as a quality warning, not as a healthy parse signal.

### 0.10 GSM8K Sweep Re-Diagnosis (2026-02-26 late)

New finding from your full GSM8K runs:
1. Direct sweep (`t16/t32/t64/t96`) stayed exactly at `accuracy=0.3663` with:
   - `parse_error_rate=0.0000`,
   - `hit_cap_rate=1.0000`,
   - `final_tag_rate=1.0000`.
2. CoT run (`t256`) looked worse at `accuracy=0.2733`, but diagnostics showed:
   - `final_answer_tag=119`,
   - `last_number=53`,
   - many tagged answers contained units/currency text (for example: `10 meters.`, `$140.`),
   - strict numeric comparison undercounted true correctness.

Code-side remedy:
1. Math evaluator now supports relaxed numeric equivalence when gold is numeric:
   - strict parse first,
   - fallback to first numeric token from predicted text (for example `75% ... -> 75`).
2. Math final-tag extraction now normalizes numeric answers from tagged text directly.

Re-evaluation (same prediction files, new evaluator):
1. `gsm8k_cot_t256_20260226T151603Z`:
   - old `accuracy=0.2733` -> new `accuracy=0.7035`.
2. `gsm8k_math_direct_t16_20260226T152107Z`:
   - old `accuracy=0.3663` -> new `accuracy=0.3721`.

Interpretation update:
1. CoT was strongly underestimated due to extraction strictness, not only model failure.
2. Direct baseline remains relatively stable around ~0.37 in this setup.
3. CoT now appears much stronger than direct on GSM8K under corrected numeric matching.
4. `hit_cap_rate=1.0` still indicates throughput waste; speed optimization remains required.

### 0.11 Foundation Reliability Hardening Gate (2026-02-27)

Reason:
1. We observed that low-level evaluation and parsing details can drastically shift conclusions.
2. To avoid Phase B/C being misled, we ran a base-component reliability hardening pass.

Completed hardening:
1. Binary-choice generation path now has branch-safe metadata handling.
2. Empty model outputs are treated as parse-error cases (no hard evaluator crash).
3. Loader split normalization now fails fast on split typos.
4. Metrics now carry evaluator version and comparison warns on version mismatch.
5. Prepared-input loader now rejects duplicate `sample_id`.

Artifacts:
1. Audit document: `foundation_reliability_audit.md`.
2. Execution checklist: `TODO_ours.md` section `18`.

### 0.12 Batched Inference Fix Validation (2026-02-27)

Problem:
1. Early batched freeform runs produced severe quality regression and repeated warnings:
   - decoder-only model warned about right-padding,
   - parse errors spiked and accuracy dropped vs `batch_size=1`.

Fix:
1. In batched freeform tokenization path, force `padding_side='left'` and restore tokenizer state after call.
2. Keep deterministic ordering and existing OOM backoff safety net.

Validation runs (StrategyQA, direct prompt, `max_new_tokens=32`, no-sample, seed=42, n=193):

| Run | Batch Size | Accuracy | Parse Error Rate | n_parseable | acc_parseable | gen_sample_rate | gen_elapsed_sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| `strat_batch1_20260226T164044Z` | 1 | 0.5492 | 0.1762 | 159 | 0.6667 | 1.180 | 163.53 |
| `strat_batch4_fix_20260226T165830Z` | 4 | 0.5596 | 0.1658 | 161 | 0.6708 | 4.189 | 46.08 |

Inference:
1. Throughput gain is substantial (`~3.55x` sample/s, `~72%` less wall time).
2. Quality is not degraded after fix; metrics are slightly improved.
3. Current batching implementation is trustworthy for Phase A benchmarking at `batch_size=4` on this setup.

Next checks:
1. Run one reproducibility pair for `strat_batch4_fix` (same run-name, same seed) to verify `changed_samples=0`.
2. Repeat the same pair on GSM8K direct path to confirm cross-dataset batching parity.

### 0.13 Phase A Retrospective Lessons (Full-Period Consolidation)

This section consolidates what we learned across the full Phase A cycle, beyond single-run snapshots.

#### 0.13.1 Evaluation semantics can dominate conclusions

1. Parser/extractor design can move apparent accuracy dramatically without any model-weight change.
2. Never compare old/new metrics across evaluator logic changes unless predictions are rescored under the same evaluator version.
3. Keep `evaluator_version` in metrics and treat version mismatch as a hard caution.

#### 0.13.2 Always separate coverage from decision quality

1. `accuracy` alone is insufficient when parse failures are possible.
2. Always report this set together:
   - `accuracy`,
   - `parse_error_rate`,
   - `n_parseable`,
   - `acc_parseable`.
3. For StrategyQA, maintain two official tracks:
   - binary-choice (decision quality),
   - freeform (end-to-end format compliance).

#### 0.13.3 Prompt/protocol choices can outweigh token-budget tweaks

1. StrategyQA freeform CoT was highly sensitive to protocol compliance.
2. Binary-choice mode showed that much of the previous gap was format/channel noise.
3. Token budget sweeps are meaningful only after decode/eval protocol is stable.

#### 0.13.4 Throughput optimizations require explicit correctness guards

1. Batching initially improved speed but introduced a decoder-only padding bug that degraded quality.
2. Batch rollout must include:
   - padding-side correctness checks,
   - ordering parity checks,
   - deterministic rerun checks.
3. Throughput gains are valid only when metric parity is preserved.

#### 0.13.5 Truncation pressure is a real research confound

1. High hit-cap rates can hide weak extraction behavior and understate model quality.
2. Math runs need extraction diagnostics (`last_number_rate`, `final_answer_tag_rate`, cap rate) before interpretation.
3. Short-token baselines are useful for speed, but should not be treated as quality ceilings.

#### 0.13.6 Phase A final operational rule set (for Phase B carry-over)

1. Freeze baseline protocol before Phase B sweeps:
   - dataset fingerprint,
   - template/version,
   - decode mode,
   - evaluator version,
   - seed and no-sample setting.
2. Any infra change (parser/prompt/decode/batching) must be followed by:
   - replay on at least one stored prediction file,
   - one reproducibility pair run.
3. Treat infra regressions as P0, because they can invalidate all higher-level claims.

## 1. Purpose

This file records meaningful experiment outcomes so new teammates can quickly understand:
- what was run,
- what settings were used,
- what happened,
- and what conclusions we can trust.

Scope: Phase A inference/evaluation experiments on StrategyQA with local `Qwen2.5-7B-Instruct`.

## 2. Prepared Artifact Sets (Inputs)

These artifact folders were generated by `scripts/phase_a_prepare.py` and reused by inference runs.

| Fingerprint | Created (UTC) | Template | Target Style | Source Split | Split Policy | Limit | Validation File |
|---|---|---|---|---|---|---:|---|
| `21095d3c688a` | 2026-02-25T19:09:43.283792+00:00 | `qa_direct` | `answer_only` | `train` | `hash` | 200 | `assets/artifacts/phase_a_prepared/strategyqa/21095d3c688a/validation.jsonl` |
| `b0f373610f96` | 2026-02-26T07:13:22.068668+00:00 | `qa_direct` | `answer_only` | `train` | `hash` | 2000 | `assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl` |
| `f3e476b514c3` | 2026-02-26T07:18:52.297402+00:00 | `qa_cot_then_final` | `cot_then_answer` | `train` | `hash` | 2000 | `assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl` |

## 3. Early Smoke Runs (Small Validation Set)

These were used to verify pipeline wiring and deterministic rerun behavior.

| Run Dir | Created (UTC) | n | Max New Tokens | Accuracy | Parse Error Rate | Notes |
|---|---|---:|---:|---:|---:|---|
| `qwen_strategyqa_val_20260226T065337Z` | 2026-02-26T06:55:19.075627+00:00 | 2 | 32 | 1.0000 | 0.0000 | smoke |
| `qwen_strategyqa_val_20260226T065605Z` | 2026-02-26T06:56:41.906460+00:00 | 1 | 16 | 1.0000 | 0.0000 | smoke |
| `qwen_strategyqa_val_20260226T065859Z` | 2026-02-26T06:59:51.576498+00:00 | 17 | 64 | 0.6471 | 0.1176 | first stable mini run |
| `qwen_strategyqa_val_20260226T070029Z` | 2026-02-26T07:01:14.792043+00:00 | 17 | 64 | 0.6471 | 0.1176 | deterministic repeat |
| `qwen_strategyqa_val_20260226T070404Z` | 2026-02-26T07:04:50.085621+00:00 | 17 | 64 | 0.6471 | 0.1176 | deterministic repeat |

## 4. Main StrategyQA Runs (n=193)

All runs below are deterministic (`--no-do-sample`, seed=42).

| Run Dir | Created (UTC) | Input Prepared Set | Max New Tokens | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---|---|---:|---:|---:|---:|---:|
| `strat_vA_direct_20260226T084052Z` | 2026-02-26T09:05:30.421698+00:00 | `f3e476b514c3` (CoT prompt/target) | 128 | 0.1036 | 0.8497 | 29 | 0.6897 |
| `strat_vA_direct_20260226T090829Z` | 2026-02-26T09:28:41.357635+00:00 | `f3e476b514c3` (CoT prompt/target) | 128 | 0.1036 | 0.8497 | 29 | 0.6897 |
| `strat_direct_t32_20260226T093714Z` | 2026-02-26T09:40:12.219698+00:00 | `b0f373610f96` (Direct prompt/target) | 32 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_direct_t32_20260226T095322Z` | 2026-02-26T09:56:16.570779+00:00 | `b0f373610f96` (Direct prompt/target) | 32 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_direct_t32_20260226T100949Z` | 2026-02-26T10:12:38.609192+00:00 | `b0f373610f96` (Direct prompt/target) | 32 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_cot_t128_20260226T095113Z` | 2026-02-26T10:02:08.907577+00:00 | `f3e476b514c3` (CoT prompt/target) | 128 | 0.1036 | 0.8497 | 29 | 0.6897 |
| `strat_cot_t256_20260226T095906Z` | 2026-02-26T10:21:17.394167+00:00 | `f3e476b514c3` (CoT prompt/target) | 256 | 0.4974 | 0.3212 | 131 | 0.7328 |

## 5. Operational Incidents (Important Context)

### 5.1 GPU unavailable period (resolved later)
- During one period, CUDA/NVML was unavailable in runtime; jobs fell back to CPU and ran very slowly.
- Later checks showed GPU access restored (`torch.cuda.is_available() == True`, `nvidia-smi -L` normal).

### 5.2 CPU offload caused severe slowdown
- Run directory: `assets/artifacts/phase_a_runs/strat_cot_t128_20260226T094126Z`
- Symptom: `device_map=auto` offloaded tail layers to CPU (`Some parameters are on the meta device...`), causing very slow decode speed.
- Status: interrupted/incomplete run; only partial `predictions.jsonl` exists (no `metrics.json`).

## 6. Key Findings We Can Trust

1. Determinism is stable for fixed config.
   - Repeated runs with same settings produced identical metrics (and in key cases identical predictions).

2. Main bottleneck in CoT setting is format/completion, not core correctness.
   - At CoT `t128`, parse error is very high (`0.8497`) while accuracy on parseable subset remains much higher (`0.6897`).

3. Increasing CoT budget helps a lot but does not fully solve formatting issues.
   - `t128 -> t256` improved:
     - accuracy: `0.1036 -> 0.4974`
     - parse error rate: `0.8497 -> 0.3212`

4. Direct answer setup is currently the strongest robust baseline.
   - `strat_direct_t32` is fast and stable:
     - accuracy `0.5492`
     - parse error `0.1762`

## 7. Suggested Next Experiments (Queued)

To finish diagnosing truncation/compliance tradeoffs, run:
- CoT token sweep: `128, 192, 256, 320, 384`
- Direct token sweep: `16, 24, 32, 48, 64`
- Determinism repeat at one key CoT point (`t256`) to ensure run-to-run stability.

Keep fixed for fair comparisons:
- `--no-do-sample`
- `--seed 42`
- same input artifact file per variant
- no CPU offload (ensure model fully fits on selected GPU)

## 8. Direct Token Sweep Result (Completed)

Run family: `strat_direct_sweep_t*` on prepared input
`assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl`

| Run Dir | Max New Tokens | Elapsed (from console) | Throughput (sample/s) | Accuracy | Parse Error Rate | Parseable n | Accuracy on Parseable |
|---|---:|---:|---:|---:|---:|---:|---:|
| `strat_direct_sweep_t16_20260226T103213Z` | 16 | 00:01:52 | 1.709 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_direct_sweep_t24_20260226T103417Z` | 24 | 00:02:47 | 1.155 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_direct_sweep_t32_20260226T103717Z` | 32 | 00:03:42 | 0.866 | 0.5492 | 0.1762 | 159 | 0.6667 |
| `strat_direct_sweep_t48_20260226T104125Z` | 48 | 00:05:25 | 0.592 | 0.5492 | 0.1710 | 160 | 0.6625 |
| `strat_direct_sweep_t64_20260226T104715Z` | 64 | 00:07:21 | 0.437 | 0.5544 | 0.1658 | 161 | 0.6646 |

### 8.1 Direct Sweep Analysis

1. Accuracy is mostly flat from `t16` to `t48`; only a small increase appears at `t64`.
2. Parse error improves only slightly with larger budgets (`0.1762 -> 0.1658`).
3. Runtime cost increases strongly as token budget grows; quality gain is small.
4. Outputs are still almost always near token cap even in direct mode, suggesting generation tends to keep talking until cutoff.
5. Prediction bias remains strong toward `no` on parseable outputs (roughly 85% `no`, 15% `yes`), so class-bias is still a key issue.

Operational inference:
- For current direct template, `t16`/`t24` are much more efficient and nearly as accurate as larger budgets.
- Purely increasing token budget is not enough to fix format/bias issues.

## 9. New Param Block Added to Address Current Problems

New one-click group in `scripts/run_phase_a_benchmark_suite.sh`:
- `ACTIVE_PARAM_GROUP=A5`

Goal of `A5`:
- improve answer-format compliance,
- reduce token waste,
- check whether stricter binary prompting helps yes/no behavior.

`A5` run plan:
1. baseline direct reference: `baseline_direct_t16`
2. strict binary prompt runs: `strict_t4`, `strict_t8`, `strict_t16`
3. strict determinism check: `strict_t16` repeated (run1 vs run2 with comparison enabled)

Template used by strict runs:
- `qa_binary_strict@1.0.0` (added in prompt builder)
- target style: `answer_only`

Recommended execution command:

```bash
ACTIVE_PARAM_GROUP=A5 \
CUDA_VISIBLE_DEVICES=3 \
RUN_PREFIX=strat_strict_fix_20260226 \
bash scripts/run_phase_a_benchmark_suite.sh
```

What to focus on for A5 results:
1. parse error rate vs `baseline_direct_t16`
2. total accuracy change
3. parseable-only accuracy
4. determinism (`delta_accuracy=0`, `changed_samples=0` on strict repeat)
