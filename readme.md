# OURS Reasoning Research Pipeline

A research codebase for building and evaluating reasoning-faithfulness workflows (Phase A/B baselines completed, Phase D external-PRM-supported track active).

## Repository Logic (High-Level)

This repository currently follows a staged pipeline:

1. **Phase A (Data + Benchmark Contracts)**
   - Load canonical samples from datasets.
   - Build prepared artifacts with `prompt_text` and `target_text`.
   - Run frozen benchmark-style generation/evaluation for reproducible baselines.

2. **Phase B (SFT/PEFT Baselines)**
   - Train instruction-tuned baselines from Phase A prepared JSONL.
   - Evaluate answer quality and transfer behavior to establish strong references.

3. **Phase C (Value-Head Bootstrapping on Prefixes)**
   - Split `target_text` into reasoning trajectory and final answer signal.
   - Build step prefixes (`question-only` + reasoning-step boundaries).
   - Build minimally corrupted prefixes for clean-vs-corrupt comparisons.
   - Run Monte Carlo rollouts per prefix to estimate prefix value targets
     (`q_mean_smoothed`, uncertainty, reliability weights).
   - Train/evaluate a frozen-backbone value head on those C1/C2 artifacts.

4. **Phase D (External PRM Teacher Integration, Active)**
   - Score existing C1 prefixes/corruptions with an external PRM teacher
     (sidecar outputs first).
   - Fuse teacher/MC supervision (`q_teacher`, `q_fused`) in C1.
   - Re-run C2 target-source ablations (`mc`, `teacher`, `fused`) and promote
     only if calibration + ranking + utility gates pass.

Planned continuation (only after Phase D promotion):
- restart BCR-lite, then ABR-lite routing.

## What This Repo Includes

- Canonical dataset schema and loaders (`src/ours/data/`)
- Step preprocessing pipeline (`src/ours/data/step_builder.py`, `scripts/preprocess_steps.py`)
- Phase A benchmark stack:
  - prompt templates,
  - deterministic splitting,
  - answer extraction + evaluation,
  - one-click benchmark suites (`scripts/run_phase_a_benchmark_suite.sh`)
- Phase B training skeleton:
  - SFT/PEFT training entrypoint (`scripts/phase_b_train_sft.py`)
  - Phase B suite runner (`scripts/run_phase_b_training_suite.sh`)
  - Live Phase B report (`phase_B_report.md`)

## Current Status

- Phase A infrastructure is stable and reproducible.
- Batched inference path is enabled and validated.
- Phase B SFT/PEFT baselines and diagnostics are complete enough to serve as references.
- Phase C value-head infrastructure is implemented end-to-end (`C0/C1/C2` + `P(IK)` branch).
- Phase D is now the official active development track: external PRM-supported value supervision.
- Authoritative Phase D plan: `phase_D_plan.md`.

## Quick Start

### 1) Prepare environment

Use your preferred Python environment and install required runtime deps (`torch`, `transformers`, `datasets`, `pytest`, etc.).

Recommended env vars:

```bash
export HF_HOME=$PWD/assets/hf_cache
export HF_DATASETS_CACHE=$PWD/assets/hf_cache/datasets
export PYTHONPATH=$PWD/src
```

### 2) Download datasets

```bash
bash download_datasets.sh
```

### 3) Smoke-check canonical loaders

```bash
python scripts/check_data.py --datasets gsm8k strategyqa --split train --limit 5
```

### 4) Run core tests

```bash
python -m pytest -q
```

## Phase A Public Entry Points

Prepare artifacts:

```bash
python scripts/phase_a_prepare.py \
  --datasets strategyqa \
  --source-split train \
  --split-policy hash \
  --target-style answer_only \
  --template-id qa_direct \
  --template-version 1.0.0 \
  --limit 200
```

Generate + evaluate:

```bash
python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl <prepared_validation_jsonl> \
  --run-name baseline_eval \
  --require-cuda \
  --no-do-sample \
  --max-new-tokens 64
```

Run output and `metrics.json` now include sampled VRAM usage stats (`vram_mean_gib`, `vram_max_gib`) for reporting.
Phase A JSONL readers are robust to Unicode line separators (for example `U+2028`) in prompt/prediction text.

Truncation recovery is enabled by default for math datasets (`gsm8k`, `hendrycks_math`).
You can tune it with:
- `--truncation-recovery-rounds`
- `--truncation-recovery-extra-tokens`
- `--truncation-recovery-datasets`
- `--no-truncation-recovery`

Run one-click benchmark suite:

```bash
bash scripts/run_phase_a_benchmark_suite.sh
```

Suite final summaries now include artifact-only instability diagnostics under
`RESULT TABLE` (multi-tag rate, first/last tag disagreement, tag-switch rate, pairwise flip rate).

Standalone artifact analysis (no re-inference):

```bash
python scripts/phase_a_analyze_instability.py \
  --run-dirs assets/artifacts/phase_a_runs/<run_dir_a> assets/artifacts/phase_a_runs/<run_dir_b>
```

Prompt-style sweeps:

```bash
ACTIVE_PARAM_GROUP=A7 RUN_PREFIX=strategyqa_style_sweep bash scripts/run_phase_a_benchmark_suite.sh
ACTIVE_PARAM_GROUP=A8 RUN_PREFIX=gsm8k_style_sweep bash scripts/run_phase_a_benchmark_suite.sh
# Whole-corpus StrategyQA review (train+validation+test aggregate)
ACTIVE_PARAM_GROUP=A11 RUN_PREFIX=strategyqa_whole_2290 bash scripts/run_phase_a_benchmark_suite.sh
# StrategyQA token-stress variant (example)
ACTIVE_PARAM_GROUP=A11_256 RUN_PREFIX=strategyqa_whole_t256 bash scripts/run_phase_a_benchmark_suite.sh
# Whole-corpus GSM8K review (train+validation+test aggregate)
ACTIVE_PARAM_GROUP=A12 RUN_PREFIX=gsm8k_whole_corpus bash scripts/run_phase_a_benchmark_suite.sh
```

For whole-corpus groups (`A11`/`A12`), env runtime overrides such as `BATCH_SIZE=...` are honored.

## Phase B Public Entry Points

Current consolidated Phase B findings:
- `phase_B_report.md`

Official Phase B kickoff (recommended):

```bash
ACTIVE_PHASE_B_GROUP=B1_SMOKE RUN_PREFIX=phase_b_kickoff bash scripts/run_phase_b_training_suite.sh
```

Heavy-run note:
- do not launch multiple full-dataset Phase B suites on one GPU at the same time,
- if a suite exits early, `final_summary.md` now records `status: failed` and `failed_stage`.

Full-dataset gain runs:

```bash
ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_FULL RUN_PREFIX=phase_b_strategyqa_full bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_FULL RUN_PREFIX=phase_b_gsm8k_full bash scripts/run_phase_b_training_suite.sh
```

StrategyQA scaling diagnostics:

```bash
ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_DIAG_EPOCH_200 RUN_PREFIX=strategyqa_diag_e200 bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_DIAG_EPOCH_300 RUN_PREFIX=strategyqa_diag_e300 bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_DIAG_LORA_R8 RUN_PREFIX=strategyqa_diag_r8 bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_STRATEGYQA_DIAG_LORA_R32 RUN_PREFIX=strategyqa_diag_r32 bash scripts/run_phase_b_training_suite.sh
```

GSM8K diagnostic runs:

```bash
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_LR_5E5 RUN_PREFIX=gsm8k_diag_lr5e5 bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_LR_1E4 RUN_PREFIX=gsm8k_diag_lr1e4 bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_EPOCH_025 RUN_PREFIX=gsm8k_diag_e025 bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_EPOCH_050 RUN_PREFIX=gsm8k_diag_e050 bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_DIRECT_STYLE RUN_PREFIX=gsm8k_diag_direct bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_EQUATION_STYLE RUN_PREFIX=gsm8k_diag_equation bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_CHECKPOINT_SWEEP RUN_PREFIX=gsm8k_diag_ckpt_sweep bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_SHORT_COT RUN_PREFIX=gsm8k_diag_short_cot bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_DIAG_ANSWER_WEIGHTED RUN_PREFIX=gsm8k_diag_answer_weighted bash scripts/run_phase_b_training_suite.sh
ACTIVE_PHASE_B_GROUP=B2_GSM8K_REPAIR_ANSWER_WEIGHTED_CKPT RUN_PREFIX=gsm8k_repair_aw_ckpt bash scripts/run_phase_b_training_suite.sh
```

New GSM8K diagnostics now cover three additional hypotheses:
- best checkpoint may occur before the final saved adapter,
- shorter CoT supervision may preserve arithmetic quality better than long CoT,
- final-answer tokens may need more loss weight than rationale tokens.

The current combined GSM8K repair attempt is:
- answer-weighted long-CoT supervision,
- plus dense checkpoint saving and held-out checkpoint sweep,
- so the best checkpoint is selected instead of the final adapter.

Cross-task interference runs:

```bash
ACTIVE_CROSS_TASK_GROUP=B3_XTASK_STRAT_R32_TO_GSM8K RUN_PREFIX=xtask_strat_r32_to_gsm8k bash scripts/run_phase_b_cross_task_suite.sh
ACTIVE_CROSS_TASK_GROUP=B3_XTASK_GSM8K_FULL_TO_STRAT RUN_PREFIX=xtask_gsm8k_full_to_strat bash scripts/run_phase_b_cross_task_suite.sh
ACTIVE_CROSS_TASK_GROUP=B3_XTASK_GSM8K_DIRECT_TO_STRAT RUN_PREFIX=xtask_gsm8k_direct_to_strat bash scripts/run_phase_b_cross_task_suite.sh
ACTIVE_CROSS_TASK_GROUP=B3_XTASK_GSM8K_EQUATION_TO_STRAT RUN_PREFIX=xtask_gsm8k_equation_to_strat bash scripts/run_phase_b_cross_task_suite.sh
```

These groups measure whether a task-specific adapter transfers or interferes when
evaluated on the other task, which is directly relevant to later BCR/ABR work.

## Phase C Public Entry Points

Phase C guidance:
- `phase_C_plan.md`

Current implemented scope:
- `C0`: freeze the Phase C contracts and execution order
- `C1`: build deterministic prefix, corruption, and optional rollout-target artifacts
- `C2`: train and evaluate a frozen-backbone value head on C1 artifacts

Full lifecycle suite entrypoint:
- `scripts/run_phase_c_value_suite.sh`
- question-level P(IK) lifecycle: `scripts/run_phase_c_pik_suite.sh`

### PRM Teacher Setup (Qwen2.5-Math-PRM-7B)

For external teacher bootstrapping in Phase D, we currently recommend:
- `Qwen2.5-Math-PRM-7B` as the primary teacher model.

Download to the repo-local model directory:

```bash
huggingface-cli login
huggingface-cli download Qwen/Qwen2.5-Math-PRM-7B \
  --local-dir assets/models/Qwen2.5-Math-PRM-7B \
  --local-dir-use-symlinks False
```

Quick local smoke test (step-level scores):

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

model_name = "assets/models/Qwen2.5-Math-PRM-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
).eval()

messages = [
    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
    {"role": "user", "content": "If Tom has 3 apples and buys 2 more, how many apples does he have?"},
    {"role": "assistant", "content": "Tom has 3 apples. <extra_0> He buys 2 apples. <extra_0> So he has 5 apples. <extra_0>"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

# Keep use_cache=False for compatibility with some transformers + remote-code combinations.
with torch.no_grad():
    logits = model(input_ids=input_ids, use_cache=False)[0]

step_sep_id = tokenizer.encode("<extra_0>")[0]
token_masks = (input_ids == step_sep_id)
probs = F.softmax(logits, dim=-1)
step_scores = probs[token_masks][:, 1].detach().float().cpu().tolist()

print("num_steps =", len(step_scores))
print("step_scores =", [round(x, 4) for x in step_scores])
PY
```

How to read the smoke-test output:
- `num_steps` is the number of `<extra_0>` step separators found in the assistant response.
- `step_scores` are step-level positive probabilities in `[0, 1]` (higher means the teacher considers that step more reliable).

Warning notes:
- `torch_dtype is deprecated` is harmless; use `dtype=` (as shown above).
- `lm_head.weight not used` is expected for process-reward model loading.
- If you hit cache-API errors (for example around `DynamicCache`), keep `use_cache=False` in teacher-scoring scripts.

Main entrypoint:

```bash
python -u scripts/phase_b_prepare_value_data.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl \
  --run-name strategyqa_value_smoke \
  --max-samples 128 \
  --build-corruptions \
  --no-build-rollouts
```

Phase C C0/C1 output directory:
- `assets/artifacts/phase_c_data/<dataset>/<run_name>__<fingerprint>/`

Key output files:
- `step_sequences.jsonl`
- `prefixes.jsonl`
- `corruptions.jsonl` if enabled
- `rollout_predictions.jsonl` and `rollout_targets.jsonl` if enabled
- `corruption_rollout_targets.jsonl` and `pair_quality.jsonl` when `--build-pair-quality` is enabled
- `manifest.json`
- `summary.json`
- `summary.md`

C2 training entrypoint:

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/phase_b_train_value.py \
  --train-dir assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts__<train_fingerprint> \
  --eval-dir assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts_val__<eval_fingerprint> \
  --run-name strategyqa_value_c2_smoke \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --per-device-train-batch-size 192 \
  --per-device-eval-batch-size 192 \
  --learning-rate 1e-3 \
  --num-train-epochs 5 \
  --use-contrastive-loss \
  --lambda-contrastive 1.0 \
  --contrastive-margin 0.1
```

New C2 options (can be toggled independently):
- calibration objective: `--calibration-loss {mse,bce,bce_mse}`
- post-hoc calibration: `--posthoc-calibration {none,temperature,isotonic}` and `--checkpoint-selection-metric {raw_brier,posthoc_brier}`
- adaptive cal/contrastive balancing: `--adaptive-loss-balancing {none,uncertainty}`
- confidence-aware calibration weighting: `--calibration-sample-weighting` (includes `q_weight`, `q_weight_parseable`), `--calibration-weight-floor`, `--calibration-weight-gamma`
- contrastive pair filtering: `--contrastive-pair-filter` (includes `label_quality`, `confidence_parseable_label`), `--contrastive-confidence-threshold`, `--contrastive-parseable-threshold`, `--contrastive-label-delta-q-min`, `--contrastive-label-z-min`, `--contrastive-label-pair-weight-min`, `--contrastive-require-pair-pass-gate`, `--contrastive-use-pair-weights`
- calibration target smoothing: `--calibration-target-smoothing`
- contrastive score-gap mining: `--contrastive-score-gap-min`, `--contrastive-score-gap-max`
- CQR C1 corruption policy: `--corruption-selection-policy`, `--min-non-step-drop-per-prefix`, `--max-step-drop-per-prefix`, semantic toggles (`--enable-negation-flip`, `--enable-comparator-flip`, `--enable-condition-reversal`, `--enable-entity-substitution`)
- CQR C2 stratified sampling: `--contrastive-stratified-sampling`, `--contrastive-stratify-step-bucket-size`, `--contrastive-stratify-include-no-corruption`

Persistent feature cache (safe default, recommended):
- `--feature-cache-root assets/artifacts/phase_c_feature_cache`
- `--feature-cache-mode {off,read,write,read_write}` (default: `read_write`)
- `--feature-cache-lock-timeout-sec 600`

Safety guarantees for persistent cache reuse:
1. cache key includes frozen-backbone provenance and data signature hashes,
2. payload shape contract is validated before reuse,
3. writes are lock-protected and atomically committed.

One-click healthcheck for cache infra:

```bash
# Fast: static + module selftest
RUN_PREFIX=cache_hc_fast CHECK_PROFILE=fast bash scripts/run_feature_cache_healthcheck.sh

# Full: include runtime smoke when prerequisites are available
RUN_PREFIX=cache_hc_full CHECK_PROFILE=full CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_feature_cache_healthcheck.sh
```

Healthcheck outputs:
- `assets/artifacts/healthcheck_logs/<RUN_PREFIX>/suite.log`
- `assets/artifacts/healthcheck_logs/<RUN_PREFIX>/final_summary.md`

C2 standalone evaluation entrypoint:

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_b_eval_faithfulness.py \
  --value-run-dir assets/artifacts/phase_c_runs/strategyqa_value_c2_smoke_<timestamp> \
  --eval-dir assets/artifacts/phase_c_data/strategyqa/strategyqa_value_rollouts_val__<eval_fingerprint> \
  --checkpoint-name best \
  --posthoc-calibration from_run \
  --run-name strategyqa_value_c2_eval
```

One-command lifecycle (C1 train + C1 eval + C2 train + C2 eval):

- Default Phase C/P(IK) batch sizing is now `192` (`ROLLOUT_BATCH_SIZE`, `C2_TRAIN_BATCH_SIZE`, `C2_EVAL_BATCH_SIZE`) to safely fit 3 concurrent jobs per 80GB GPU.

```bash
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_c_strategyqa_smoke \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_value_suite.sh
```

Quality-first lifecycle examples:

```bash
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_QUALITY_FIRST \
RUN_PREFIX=phase_c_quality_first \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_c_value_suite.sh

ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_QUALITY_FIRST_FULL \
RUN_PREFIX=phase_c_quality_first_full \
CUDA_VISIBLE_DEVICES=3 \
bash scripts/run_phase_c_value_suite.sh

ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_CQR_SMOKE \
RUN_PREFIX=phase_c_cqr_smoke \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_value_suite.sh
```

Question-level P(IK) lifecycle (new):

```bash
ACTIVE_PHASE_C_PIK_GROUP=PIK_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_c_pik_smoke \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_c_pik_suite.sh
```

Why add P(IK):
- Prefix-level value runs are harder and noisier.
- P(IK) is a simpler diagnostic: one question -> `K` sampled answers -> empirical success-rate target.
- It checks whether the value head can learn any confidence signal before returning to prefix-level objectives.

Main P(IK) scripts:
- `scripts/phase_c_prepare_pik_data.py`
- `scripts/phase_c_train_pik.py`
- `scripts/phase_c_eval_pik.py`

PIK train/eval also supports the same persistent feature-cache flags:
- `--feature-cache-root`
- `--feature-cache-mode`
- `--feature-cache-lock-timeout-sec`

Supported lifecycle groups:
1. `C2_STRATEGYQA_SMOKE`
2. `C2_STRATEGYQA_FULL`
3. `C2_STRATEGYQA_TRICK1_BCE`
4. `C2_STRATEGYQA_TRICK2_POSTHOC_TEMP`
5. `C2_STRATEGYQA_TRICK3_ADAPTIVE_BALANCE`
6. `C2_STRATEGYQA_TRICK4_ISOTONIC`
7. `C2_STRATEGYQA_TRICK5_WEIGHTED_CAL`
8. `C2_STRATEGYQA_TRICK6_PAIR_FILTER`
9. `C2_STRATEGYQA_TRICK7_COMBINED`
10. `C2_STRATEGYQA_TRICK8_LABEL_SMOOTH`
11. `C2_STRATEGYQA_TRICK9_HARD_NEG_MINING`
12. `C2_STRATEGYQA_TRICK10_K16_COMBINED`
13. `C2_STRATEGYQA_QUALITY_FIRST`
14. `C2_STRATEGYQA_QUALITY_FIRST_FULL`
15. `C2_STRATEGYQA_CQR_SMOKE`
16. `C2_STRATEGYQA_CQR_FULL`
17. `C2_STRATEGYQA_CQR_RERUN_TRICK10`
18. `C2_STRATEGYQA_CQR_RERUN_QUALITY_FIRST`

## Phase D Public Direction

Phase D focus:
1. use external PRM teacher signals to improve value-label quality,
2. fuse teacher and Monte Carlo supervision in C1,
3. re-train C2 with target-source ablations (`mc`, `teacher`, `fused`),
4. promote to ABR/BCR continuation only if calibration and corruption-ordering gates pass.

Primary planning docs:
- `phase_D_plan.md` (official active plan)
- `phase_C_fix_value_head.md` (detailed diagnosis and external-help rationale)

D1 teacher-sidecar entrypoint (implemented):

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_c_score_prm_teacher.py \
  --phase-c-dir <phase_c_artifact_dir> \
  --teacher-model-path assets/models/Qwen2.5-Math-PRM-7B \
  --batch-size 192 \
  --max-length 2048 \
  --require-cuda
```

If a historical artifact directory is missing `manifest.json`, add:
- `--allow-missing-manifest`

D2 C1 teacher/MC fusion (implemented):
- `scripts/phase_b_prepare_value_data.py` now supports teacher join + fusion,
- writes `q_teacher`, `q_fused`, `teacher_available`, `teacher_disagree` into `rollout_targets.jsonl`,
- writes join diagnostics under `teacher_fusion_summary` in `summary.json`,
- supports pair-quality teacher consensus (`--teacher-corruption-scores-jsonl`, `--pair-consensus-*`).

```bash
python -u scripts/phase_b_prepare_value_data.py \
  --input-jsonl <phase_a_prepared_jsonl> \
  --run-name <phase_d_c1_fused_name> \
  --build-corruptions \
  --build-rollouts \
  --model-path assets/models/Qwen2.5-7B-Instruct \
  --batch-size 192 \
  --rollout-count 16 \
  --max-new-tokens 128 \
  --dtype bfloat16 \
  --device-map auto \
  --require-cuda \
  --teacher-prefix-scores-jsonl <path_to_teacher_prefix_scores.jsonl> \
  --teacher-corruption-scores-jsonl <path_to_teacher_corruption_scores.jsonl> \
  --teacher-fuse-mode fixed \
  --teacher-fusion-lambda 0.5 \
  --teacher-min-coverage 0.98 \
  --build-pair-quality \
  --pair-consensus-enable
```

D3 C2 target-source switch (implemented):
- `scripts/phase_b_train_value.py`:
  - `--target-source {q_mean_smoothed,q_teacher,q_fused}`
  - `--target-source-missing-policy {fail,fallback_mc}`
  - `--contrastive-max-corruptions-per-prefix`
  - `--train-mode {joint,ranking_only,calibration_only,two_stage}`
  - `--two-stage-ranking-ratio`
- `scripts/phase_b_eval_faithfulness.py`:
  - `--target-source from_run` to reuse training selection.

```bash
python -u scripts/phase_b_train_value.py \
  --train-dir <phase_d_c1_train_dir_with_teacher_fields> \
  --eval-dir <phase_d_c1_eval_dir_with_teacher_fields> \
  --run-name phase_d_c2_fused \
  --target-source q_fused \
  --target-source-missing-policy fail \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto
```

```bash
python -u scripts/phase_b_eval_faithfulness.py \
  --value-run-dir <phase_d_c2_run_dir> \
  --eval-dir <phase_d_c1_eval_dir_with_teacher_fields> \
  --checkpoint-name best \
  --target-source from_run \
  --target-source-missing-policy from_run \
  --run-name phase_d_c2_eval
```

Phase D bundled suite (recommended):
- One command runs D2 prep (train+eval) and D3 three-way ablation (`mc/teacher/fused`).
- Consolidates all metrics in one summary table.
- New HQ group enables pair-consensus gating + C2 two-stage (`ranking -> calibration`) + top-k corruption candidates.

Smoke (fast, recommended first):

```bash
ACTIVE_PHASE_D_GROUP=D4_STRATEGYQA_SMOKE_3WAY_HQ \
RUN_PREFIX=phase_d_bundle_smoke_hq \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/run_phase_d_teacher_suite.sh
```

Full (promotion-oriented):

```bash
ACTIVE_PHASE_D_GROUP=D4_STRATEGYQA_FULL_3WAY_HQ \
RUN_PREFIX=phase_d_bundle_full_hq \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/run_phase_d_teacher_suite.sh
```

Optional overrides (common):

```bash
TEACHER_TRAIN_SCORES=<.../teacher_prefix_scores.jsonl> \
TEACHER_TRAIN_CORR_SCORES=<.../teacher_corruption_scores.jsonl> \
TEACHER_EVAL_SCORES=<.../teacher_prefix_scores.jsonl> \
TEACHER_EVAL_CORR_SCORES=<.../teacher_corruption_scores.jsonl> \
ROLLOUT_BATCH_SIZE=192 \
C2_TRAIN_BATCH_SIZE=192 \
C2_EVAL_BATCH_SIZE=192
```

Main outputs:
1. `assets/artifacts/phase_d_logs/<RUN_PREFIX>/suite.log`
2. `assets/artifacts/phase_d_logs/<RUN_PREFIX>/final_summary.md`
3. auto-linked D2 dirs in `assets/artifacts/phase_c_data/...`
4. auto-linked D3 runs in `assets/artifacts/phase_c_runs/...` and `assets/artifacts/phase_c_eval/...`

Phase D4 external-pair suite (new):
- One command runs D4A -> D4B -> D4C with external pair preparation + C2 train/eval.
- D4A: direct pair warm start (`R-PRM` + `PRMBench_Preview`).
- D4B: add step-converted pairs (`Math-Shepherd` + `RLHFlow`).
- D4C: stabilize with conservative external weight + two-stage C2.

Smoke (recommended first):

```bash
ACTIVE_PHASE_D4_GROUP=D4ABC_STRATEGYQA_SMOKE \
RUN_PREFIX=phase_d4abc_smoke \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<your_train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<your_eval_dir> \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_d_external_pair_suite.sh
```

Single-stage examples:

```bash
ACTIVE_PHASE_D4_GROUP=D4A_STRATEGYQA_SMOKE bash scripts/run_phase_d_external_pair_suite.sh
ACTIVE_PHASE_D4_GROUP=D4B_STRATEGYQA_SMOKE bash scripts/run_phase_d_external_pair_suite.sh
ACTIVE_PHASE_D4_GROUP=D4C_STRATEGYQA_SMOKE bash scripts/run_phase_d_external_pair_suite.sh
```

Full-scale group:

```bash
ACTIVE_PHASE_D4_GROUP=D4ABC_STRATEGYQA_FULL \
RUN_PREFIX=phase_d4abc_full \
PHASE_C_TRAIN_DIR=assets/artifacts/phase_c_data/strategyqa/<your_train_dir> \
PHASE_C_EVAL_DIR=assets/artifacts/phase_c_data/strategyqa/<your_eval_dir> \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_phase_d_external_pair_suite.sh
```

D4 suite outputs:
1. `assets/artifacts/phase_d_logs/<RUN_PREFIX>/suite.log`
2. `assets/artifacts/phase_d_logs/<RUN_PREFIX>/final_summary.md`
3. stage-wise external pair artifacts in `assets/artifacts/phase_d_external_pairs/...`
4. stage-wise C2 runs/evals in `assets/artifacts/phase_c_runs/...` and `assets/artifacts/phase_c_eval/...`

Suite usage for C2-only ablation (example):

```bash
ACTIVE_PHASE_C_GROUP=C2_STRATEGYQA_CQR_SMOKE \
RUN_PREFIX=phase_d_fused_smoke \
PHASE_C_TRAIN_EXTRA_ARGS="--target-source q_fused --target-source-missing-policy fail" \
bash scripts/run_phase_c_value_suite.sh
```

Smoke training run:

```bash
python -u scripts/phase_b_train_sft.py \
  --config-json configs/phase_b/peft_smoke_strategyqa.json
```

If PEFT/transformers env errors appear, repair in your env:

```bash
python -m pip install -U "transformers>=4.44,<5" "huggingface-hub>=0.23.2,<1.0" "accelerate>=1.1,<2" "peft>=0.11,<0.14"
```

Training suite:

```bash
bash scripts/run_phase_b_training_suite.sh
```

Full-dataset Phase B groups now run:
1. held-out baseline eval with the frozen base model,
2. whole-train-split PEFT training,
3. held-out post-train eval with the saved adapter/model,
4. one gain report that states the before/after delta.

Post-train evaluation against frozen Phase A metrics:

```bash
python -u scripts/phase_b_eval.py \
  --phase-b-run-dir assets/artifacts/phase_b_runs/<phase_b_run_dir> \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/<prepared_dir>/validation.jsonl \
  --run-name phase_b_eval_after_train \
  --strategyqa-decode-mode binary_choice \
  --max-new-tokens 16
```

This evaluates the trained Phase B artifact with the same Phase A benchmark logic,
so you can compare post-train task accuracy against pre-train baselines.

For the new `B2_*` groups, this comparison is automated and written to:
- `assets/artifacts/phase_b_logs/<RUN_PREFIX>/peft_gain_summary.json`
- `assets/artifacts/phase_b_logs/<RUN_PREFIX>/peft_gain_summary.md`

## Key Documentation

- `phase_A_report.md`: Phase A conclusions and outcomes
- `phase_B_plan.md`: Phase B lifecycle and goals
- `phase_D_plan.md`: Phase D lifecycle and external-PRM integration plan
- `phase_D4ABC_external_pairs_tutorial.md`: D4A/B/C newcomer tutorial and troubleshooting
- `result_records.md`: experiment records and diagnosis history
- `foundation_reliability_audit.md`: reliability risks and hardening notes

## External Dataset Variants (Community Snapshots)

This repo now supports loading common community mirrors directly from:
- `assets/external_datasets/openai_gsm8k`
- `assets/external_datasets/voidful_strategyqa`

You can use them without manual folder renaming by setting `--dataset-root`.

Quick check:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bcr
python scripts/check_data.py --datasets gsm8k strategyqa --dataset-root assets/external_datasets --split train --limit 2
```

Prepare Phase A artifacts from external snapshots:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bcr
python scripts/phase_a_prepare.py \
  --datasets gsm8k strategyqa \
  --dataset-root assets/external_datasets \
  --source-split train \
  --split-policy hash \
  --target-style cot_then_answer \
  --template-id qa_strategyqa_cot_compact \
  --output-dir assets/artifacts/phase_a_prepared_external
```

## Note for Maintainers

Detailed/private operational notes and hardcoded local workflow details are maintained in `readme_full.md`.

Documentation convention for maintained runtime files:
- every `py`/`sh` file should begin with a short abstract,
- every function/class should carry a beginner-friendly docstring or nearby comment,
- examples should be included when they materially improve readability.
