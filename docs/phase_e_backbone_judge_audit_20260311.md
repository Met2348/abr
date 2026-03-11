# Phase E Backbone And Judge Pilot Audit (2026-03-11)

## Scope

This note records one focused Phase E pilot around two questions:

1. does relaxing the frozen-backbone assumption help on the current Math-Shepherd-aligned pair pipeline?
2. can the newly installed local judge LLMs already reduce data-noise enough to improve value-head learning?

This document is intentionally narrow. It is not another general Phase E plan.

## Community Reading Relevant To This Pilot

The papers already reviewed under `docs/relatedPapers/` point in a fairly consistent direction:

1. stronger process verifiers usually do **not** come from a tiny frozen scalar head alone;
2. stronger results often involve either:
   - higher-quality supervision (`LLM-as-judge`, consensus filtering, human labels), or
   - backbone adaptation / verifier-style modeling,
3. but judge models are usually used as **offline relabel / filter / adjudication tools**, not as a fragile always-on preprocessing step with a strict formatting contract.

Concretely:

1. `2501.07301_lessons_developing_prm.pdf`
   - `MC < LLM-judge < human` in supervision quality,
   - but the value comes from better labels, not from brittle prompt plumbing.
2. `2504.10559_actprm.pdf` and `2504.00891_genprm.pdf`
   - stronger verifier quality is tied to better data geometry and stronger verifier modeling, not just "same pipeline + a little more capacity".
3. `2503.21295_r_prm.pdf` and `2503.04618_birm.pdf`
   - backbone-adapted or richer verifier forms are plausible,
   - but they assume enough supervision volume and a training objective aligned with the downstream evaluation slices.

That led to the following conservative pilot design:

1. keep the pair contract fixed,
2. compare raw frozen vs raw LoRA on the **same** curated slice,
3. test judge use in the most conservative way first,
   - hard filtering of auditable pairs,
   - direct prefix-audit on actual Phase E pairs,
4. compare everything back to the repository reference `PBR2`.

## Artifacts Used

### Reference

1. `PBR2` value run:
   - `assets/artifacts/phase_e_runs/phase_e_processbench_research_v2_pbr2_ms_align_gated_value_20260311T043818Z`
2. `PBR2` same-family eval:
   - `assets/artifacts/phase_e_samefamily_eval/phase_e_processbench_research_v2_pbr2_ms_align_gated_samefamily_20260311T043905Z`
3. `PBR2` benchmark evals:
   - `assets/artifacts/phase_e_eval/phase_e_processbench_research_v2_pbr2_ms_align_gated_processbench_gsm8k_20260311T043919Z`
   - `assets/artifacts/phase_e_eval/phase_e_processbench_research_v2_pbr2_ms_align_gated_processbench_math_20260311T043935Z`

### New pilot subset

1. subset pair artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_backbone_judge_pilot_small_0311_subset__9b43b747715c`
2. judge-filtered pair artifact:
   - `assets/artifacts/phase_e_pairs/phase_e_backbone_judge_pilot_small_0311_judgefilter__99566654869c`

### New raw-slice training runs

1. frozen raw subset:
   - `assets/artifacts/phase_e_runs/phase_e_backbone_judge_manual_base_frozen_nocache_g0_20260311T080638Z`
2. LoRA raw subset:
   - `assets/artifacts/phase_e_runs/phase_e_backbone_judge_manual_base_lora_20260311T080936Z`

### New eval artifacts

1. frozen same-family:
   - `assets/artifacts/phase_e_samefamily_eval/phase_e_backbone_judge_manual_base_frozen_samefamily_20260311T081030Z`
2. frozen `ProcessBench`:
   - `assets/artifacts/phase_e_eval/phase_e_backbone_judge_manual_base_frozen_pb_gsm8k_20260311T081024Z`
   - `assets/artifacts/phase_e_eval/phase_e_backbone_judge_manual_base_frozen_pb_math_20260311T081022Z`
3. LoRA same-family:
   - `assets/artifacts/phase_e_samefamily_eval/phase_e_backbone_judge_manual_base_lora_samefamily_20260311T081259Z`
4. LoRA `ProcessBench`:
   - `assets/artifacts/phase_e_eval/phase_e_backbone_judge_manual_base_lora_pb_gsm8k_20260311T081253Z`
   - `assets/artifacts/phase_e_eval/phase_e_backbone_judge_manual_base_lora_pb_math_1024_20260311T081526Z`

### Judge-only diagnostics

1. hard filter summary:
   - `assets/artifacts/phase_e_pairs/phase_e_backbone_judge_pilot_small_0311_judgefilter__99566654869c/summary.json`
2. prefix audit summary:
   - `assets/artifacts/phase_e_judge_audit/phase_e_judge_prefix_audit_rawsubset_20260311T081549Z/summary.json`
3. local judge benchmark compare:
   - `assets/artifacts/phase_e_judge_bench_compare/judge_model_compare_20260311T074514Z/summary.md`

## Experiment Design

### A. Frozen baseline on raw subset

Train on the 96-train / 127-val curated subset with the existing Phase E frozen-feature pipeline.

Important detail:

1. this had to be rerun with `feature-cache-mode=off`;
2. two reused eval feature caches were corrupted and produced non-finite pooled features.

### B. LoRA baseline on the exact same raw subset

Train a minimal LoRA adapter plus value head on the same subset:

1. same pair slice,
2. same loss family,
3. only the backbone freezing assumption changes.

### C. Judge hard filter

Use `Qwen2.5-Math-7B-Instruct` as a local prefix-correctness judge to filter auditable pair semantics before training.

### D. Judge prefix audit

Directly ask whether the local judge can even produce stable structured outputs on actual Phase E prefix pairs.

## Headline Comparison

| case | held-out pair acc | held-out auc | same-family top1 | same-family first-bad | gsm auc | gsm first-edge | gsm f1 | math auc | math first-edge | math f1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `raw_frozen` | 0.7874 | 0.7343 | 0.6842 | 0.7253 | 0.4481 | 0.3200 | 0.2133 | 0.4479 | 0.2500 | 0.2087 |
| `raw_lora` | 0.6693 | 0.6859 | 0.4947 | 0.0000 | 0.5390 | 0.5200 | 0.0745 | 0.7903 | 0.0357 | 0.0513 |
| `pbr2_ref` | 0.8819 | 0.8723 | 0.8947 | 0.9231 | 0.4713 | 0.5600 | n/a | 0.5055 | 0.5357 | n/a |

## How To Read The LoRA Result

The `raw_lora` line is **not** a win.

Reasons:

1. held-out dropped materially versus raw frozen;
2. same-family collapsed badly:
   - `top1=0.4947`
   - `local_first_bad=0.0000`
3. `ProcessBench GSM8K` shows superficially better AUC / first-edge than raw frozen,
   but `processbench_f1=0.0745` is far worse;
4. `ProcessBench Math` at `max_length=1024` shows a numerically unstable pattern:
   - `auc=0.7903`
   - `first_edge=0.0357`
   - `f1=0.0513`

This means the model learned a score geometry that is not aligned with the benchmark decision rule.

In practical terms:

1. this pilot does **not** support promoting small-data LoRA as the next mainline fix;
2. it supports the opposite conclusion:
   - loosening the frozen-backbone assumption without fixing data scale / training geometry makes the system less trustworthy.

## Judge Result

### Hard filter outcome

The hard-filter attempt failed as a usable training preprocessor.

Summary:

1. input train pairs: `96`
2. output train pairs: `16`
3. keep rate: `0.1667`
4. all `80` auditable pairs were dropped because of `parse_failed`
5. only the `16` bypassed `terminal_completion_anchor` pairs survived

This means the filtered training set stopped being a meaningful process-learning dataset.

### Prefix audit outcome

The direct audit on 32 real Phase E validation pairs also failed operationally:

1. `pair_json_ok_rate = 0.0`
2. `pair_agreement_rate = 0.0`
3. runtime was still `230.5s` for only 32 pairs

This is a strong engineering conclusion:

1. the current strict-JSON local judge contract is not stable enough for direct Phase E pair auditing,
2. and even this failed audit is too slow to be treated as cheap preprocessing.

### What the earlier judge benchmark already said

The smaller `ProcessBench` judge benchmark had already shown:

1. `Qwen2.5-Math-7B-Instruct` is the more stable local bulk judge candidate,
2. `DeepSeek-R1-Distill-Qwen-14B` only shows limited value on light `gsm8k` adjudication,
3. neither model is ready to be plugged in as a strict structured local judge on the main Phase E data path.

The new hard-filter and prefix-audit results are consistent with that earlier benchmark.

## Engineering Findings That Matter

### 1. Frozen feature cache can silently poison small pilots

A reused eval cache produced non-finite pooled features for the pilot subset.

Observed pattern:

1. cached train tensors were finite,
2. cached eval tensors had large contiguous bad tails,
3. the training script then failed with `Non-finite pooled features detected`.

Interpretation:

1. this is an infrastructure issue, not a research conclusion,
2. frozen pilot reruns should default to `feature-cache-mode=off` unless the cache lineage is trusted.

### 2. Single-GPU backbone loading is resource-sensitive

At run time, only `GPU0` had enough free memory to run the no-cache frozen pilot safely.

So the manual frozen rerun used `CUDA_VISIBLE_DEVICES=0` plus `--max-gpu-memory-gib 45`.

This is not a preference change. It was forced by momentary cluster load.

### 3. LoRA eval needs benchmark-aligned max length

`ProcessBench Math` with LoRA at `max_length=768` failed the truncation guard.

Therefore:

1. LoRA training may use a shorter context for memory,
2. but benchmark-facing evaluation must remain on the longer budget unless truncation diagnostics are explicitly clean.

## Diagnosis

The combined reading is:

1. current transfer difficulty is **not** solved by simply unfreezing part of the backbone on a tiny curated slice;
2. current data difficulty is **not** solved by plugging in a local judge with a brittle strict-output contract;
3. the stronger repository reference `PBR2` is still ahead because it preserves the current best available tradeoff across:
   - held-out,
   - same-family,
   - `ProcessBench`.

## What Judge Should Be Used For Instead

For the current repository state, the judge line should be demoted from `direct hard filter` to `offline auxiliary tooling`.

Recommended uses:

1. disagreement mining:
   - only send pairs where current PRM/value head and MC labels disagree,
2. selective adjudication:
   - focus on `local_first_bad_edge` and `good_bad_prefix_grid`,
   - exclude `terminal_completion_anchor` from direct prefix judging,
3. tolerant output contracts:
   - use relaxed final-block parsing rather than strict raw JSON,
4. audit-first workflow:
   - judge should first produce artifacts for inspection,
   - only later become a data-retention decision rule.

## Immediate Next Steps

1. keep `PBR2` as the benchmark-facing reference until a stronger line actually beats it;
2. if backbone adaptation is revisited, do it only under:
   - larger supervision volume,
   - benchmark-aligned checkpoint selection,
   - explicit stability diagnostics,
3. move judge work toward:
   - tolerant contract parsing,
   - disagreement-mining,
   - consensus relabeling on selected local-first-bad slices,
4. treat current raw-slice LoRA and strict-JSON judge hard filtering as negative results.

## Bottom Line

This pilot ruled out two intuitive next steps:

1. `small-data LoRA` is not the current fix,
2. `strict-JSON local judge hard filtering` is not the current fix.

The current best practical direction is still:

1. stronger data geometry,
2. stronger but selective judge-assisted relabeling,
3. only then revisit backbone adaptation under a larger, cleaner supervision budget.
