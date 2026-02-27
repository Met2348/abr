(bcr) zling@zling-U2003:~/y/bcr/ref$ for TOK in 16 24 32 48 64; do
  CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_a_generate_and_eval.py \
    --input-jsonl "$PREP_DIRECT" \
    --model-path "$MODEL" \
    --run-name "strat_direct_sweep_t${TOK}" \
    --require-cuda \
    --dtype bfloat16 \
    --device-map auto \
    --no-do-sample \
    --seed 42 \
    --max-new-tokens "$TOK" \
    --log-every 5 \
    --no-compare-latest-same-name
done
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/strat_direct_sweep_t16_20260226T103213Z
seed        : 42
gen_config  : {'max_new_tokens': 16, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
log_every   : 5
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 193
Loading weights: 100%|█| 339/339 [00:03
first_param : cuda:0
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:00:03 | rate=1.413 sample/s | eta=00:02:13
generation  : 193/193 (100.0%) | elapsed=00:01:52 | rate=1.709 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.5492
parse_error_rate : 0.1762
metrics_path     : assets/artifacts/phase_a_runs/strat_direct_sweep_t16_20260226T103213Z/metrics.json
========================================================================================
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/strat_direct_sweep_t24_20260226T103417Z
seed        : 42
gen_config  : {'max_new_tokens': 24, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
log_every   : 5
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 193
Loading weights: 100%|█| 339/339 [00:03
first_param : cuda:0
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:00:04 | rate=1.008 sample/s | eta=00:03:06
generation  : 193/193 (100.0%) | elapsed=00:02:47 | rate=1.155 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.5492
parse_error_rate : 0.1762
metrics_path     : assets/artifacts/phase_a_runs/strat_direct_sweep_t24_20260226T103417Z/metrics.json
========================================================================================
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/strat_direct_sweep_t32_20260226T103717Z
seed        : 42
gen_config  : {'max_new_tokens': 32, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
log_every   : 5
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 193
Loading weights: 100%|█| 339/339 [00:11
first_param : cuda:0
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:00:06 | rate=0.783 sample/s | eta=00:04:00
generation  : 193/193 (100.0%) | elapsed=00:03:42 | rate=0.866 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.5492
parse_error_rate : 0.1762
metrics_path     : assets/artifacts/phase_a_runs/strat_direct_sweep_t32_20260226T103717Z/metrics.json
========================================================================================
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/strat_direct_sweep_t48_20260226T104125Z
seed        : 42
gen_config  : {'max_new_tokens': 48, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
log_every   : 5
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 193
Loading weights: 100%|█| 339/339 [00
first_param : cuda:0
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:00:09 | rate=0.501 sample/s | eta=00:06:15
generation  : 193/193 (100.0%) | elapsed=00:05:25 | rate=0.592 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.5492
parse_error_rate : 0.1710
metrics_path     : assets/artifacts/phase_a_runs/strat_direct_sweep_t48_20260226T104125Z/metrics.json
========================================================================================
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/strat_direct_sweep_t64_20260226T104715Z
seed        : 42
gen_config  : {'max_new_tokens': 64, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
log_every   : 5
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 193
Loading weights: 100%|█| 339/339 [00
first_param : cuda:0
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:00:12 | rate=0.416 sample/s | eta=00:07:32
generation  : 193/193 (100.0%) | elapsed=00:07:21 | rate=0.437 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.5544
parse_error_rate : 0.1658
metrics_path     : assets/artifacts/phase_a_runs/strat_direct_sweep_t64_20260226T104715Z/metrics.json
========================================================================================







sweeper 2
(bcr) zling@zling-U2003:~/y/bcr/ref$ for TOK in 128 192 256 320 384; do
  CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_a_generate_and_eval.py \
    --input-jsonl "$PREP_COT" \
    --model-path "$MODEL" \
    --run-name "strat_cot_sweep_t${TOK}" \
    --require-cuda \
    --dtype bfloat16 \
    --device-map auto \
    --no-do-sample \
    --seed 42 \
    --max-new-tokens "$TOK" \
    --log-every 5 \
    --no-compare-latest-same-name
done
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/strat_cot_sweep_t128_20260226T103204Z
seed        : 42
gen_config  : {'max_new_tokens': 128, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
log_every   : 5
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 193
Loading weights: 100%|█| 339/339 [00:12
first_param : cuda:0
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:00:23 | rate=0.211 sample/s | eta=00:14:50
generation  : 193/193 (100.0%) | elapsed=00:14:24 | rate=0.223 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.1036
parse_error_rate : 0.8497
metrics_path     : assets/artifacts/phase_a_runs/strat_cot_sweep_t128_20260226T103204Z/metrics.json
========================================================================================
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/strat_cot_sweep_t192_20260226T104650Z
seed        : 42
gen_config  : {'max_new_tokens': 192, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
log_every   : 5
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 193
Loading weights: 100%|█| 339/339 [00
first_param : cuda:0
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:00:31 | rate=0.159 sample/s | eta=00:19:44
generation  : 193/193 (100.0%) | elapsed=00:18:22 | rate=0.175 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.2591
parse_error_rate : 0.6684
metrics_path     : assets/artifacts/phase_a_runs/strat_cot_sweep_t192_20260226T104650Z/metrics.json
========================================================================================
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/strat_cot_sweep_t256_20260226T110535Z
seed        : 42
gen_config  : {'max_new_tokens': 256, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
log_every   : 5
console_log : assets/artifacts/phase_a_runs/strat_cot_sweep_t256_20260226T110535Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 193
Loading weights: 100%|█| 339/339 [00
first_param : cuda:0
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:00:36 | rate=0.138 sample/s | eta=00:22:38
generation  : 193/193 (100.0%) | elapsed=00:22:25 | rate=0.143 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.4974
parse_error_rate : 0.3212
metrics_path     : assets/artifacts/phase_a_runs/strat_cot_sweep_t256_20260226T110535Z/metrics.json
========================================================================================
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/strat_cot_sweep_t320_20260226T112825Z
seed        : 42
gen_config  : {'max_new_tokens': 320, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
log_every   : 5
console_log : assets/artifacts/phase_a_runs/strat_cot_sweep_t320_20260226T112825Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 193
Loading weights: 100%|█| 339/339 [00
first_param : cuda:0
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:00:43 | rate=0.114 sample/s | eta=00:27:31
generation  : 193/193 (100.0%) | elapsed=00:27:28 | rate=0.117 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.6684
parse_error_rate : 0.1140
metrics_path     : assets/artifacts/phase_a_runs/strat_cot_sweep_t320_20260226T112825Z/metrics.json
========================================================================================
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/strat_cot_sweep_t384_20260226T115617Z
seed        : 42
gen_config  : {'max_new_tokens': 384, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
log_every   : 5
console_log : assets/artifacts/phase_a_runs/strat_cot_sweep_t384_20260226T115617Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 193
Loading weights: 100%|█| 339/339 [00
first_param : cuda:0
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:00:52 | rate=0.095 sample/s | eta=00:33:00
generation  : 193/193 (100.0%) | elapsed=00:33:49 | rate=0.095 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.6839
parse_error_rate : 0.0518
metrics_path     : assets/artifacts/phase_a_runs/strat_cot_sweep_t384_20260226T115617Z/metrics.json
========================================================================================




A5
FINAL EXPERIMENT SUMMARY
==========================================================================================================================================
generated_at      : 2026-02-26T21:06:07.878995+08:00
group_id          : A5
group_title       : Strict Yes/No Compliance Fix
run_prefix        : my_run
intention         : Use strict binary prompt to improve format compliance and efficiency.
observe           : Watch parse_error_rate first; then total accuracy and reproducibility deltas.
expectation       : Strict runs should be shorter and cleaner; determinism should hold on repeat.
------------------------------------------------------------------------------------------------------------------------------------------
SETTINGS
dataset           : strategyqa
source_split      : train
split_policy      : hash
limit             : 2000
seed              : 42
dtype             : bfloat16
log_every         : 5
cuda_devices      : 3
model_path        : assets/models/Qwen2.5-7B-Instruct
direct_input      : assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl
strict_input      : assets/artifacts/phase_a_prepared/strategyqa/8ef3593759f0/validation.jsonl
suite_log_file    : assets/artifacts/phase_a_logs/my_run/suite.log
summary_file      : assets/artifacts/phase_a_logs/my_run/final_summary.md
------------------------------------------------------------------------------------------------------------------------------------------
PLANNED RUN SPECS
- label=baseline_direct_t16 | input=direct | tok=16 | compare=no | run_name=my_run_baseline_direct_t16
- label=strict_t4 | input=strict | tok=4 | compare=no | run_name=my_run_strict_t4
- label=strict_t8 | input=strict | tok=8 | compare=no | run_name=my_run_strict_t8
- label=strict_t16_r1 | input=strict | tok=16 | compare=no | run_name=my_run_strict_t16
- label=strict_t16_r2 | input=strict | tok=16 | compare=yes | run_name=my_run_strict_t16
------------------------------------------------------------------------------------------------------------------------------------------
RESULT TABLE
label                   tok     n      acc  parse_err parseable_n acc_parseable  delta_acc  changed
------------------------------------------------------------------------------------------------------------------------------------------
baseline_direct_t16      16   193   0.5492     0.1762         159        0.6667        n/a      n/a
strict_t4                 4   193   0.5078     0.2383         147        0.6667        n/a      n/a
strict_t8                 8   193   0.5078     0.2383         147        0.6667        n/a      n/a
strict_t16_r1            16   193   0.5285     0.0363         186        0.5484        n/a      n/a
strict_t16_r2            16   193   0.5285     0.0363         186        0.5484    +0.0000        0
------------------------------------------------------------------------------------------------------------------------------------------
best_accuracy     : baseline_direct_t16 (acc=0.5492, parse_err=0.1762)
lowest_parse_err  : strict_t16_r1 (parse_err=0.0363, acc=0.5285)
==========================================================================================================================================
[2026-02-26 21:06:07 +0800] Group run complete






**Re-check with binary-choice mode (best for separating model quality from formatting):< A6 >**
CTIVE_PARAM_GROUP=A6 \
CUDA_VISIBLE_DEVICES=3 \
RUN_PREFIX=strat_binchoice_diag \
bash scripts/run_phase_a_benchmark_suite.sh

FINAL EXPERIMENT SUMMARY
==========================================================================================================================================
generated_at      : 2026-02-26T21:45:48.140362+08:00
group_id          : A6
group_title       : Binary-Choice Decode Validation
run_prefix        : strat_binchoice_diag
intention         : Diagnose model quality after removing free-form format failures.
observe           : Check parse_error first (should be near zero), then compare direct vs CoT accuracy.
expectation       : Coverage improves sharply; remaining errors mostly reflect model reasoning/calibration.
------------------------------------------------------------------------------------------------------------------------------------------
SETTINGS
dataset           : strategyqa
source_split      : train
split_policy      : hash
limit             : 2000
seed              : 42
dtype             : bfloat16
log_every         : 5
max_progress_lines: 5
strategyqa_decode : binary_choice
truncate_markers  : 1
cuda_devices      : 3
model_path        : assets/models/Qwen2.5-7B-Instruct
direct_input      : assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl
cot_input         : assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl
suite_log_file    : assets/artifacts/phase_a_logs/strat_binchoice_diag/suite.log
summary_file      : assets/artifacts/phase_a_logs/strat_binchoice_diag/final_summary.md
------------------------------------------------------------------------------------------------------------------------------------------
PLANNED RUN SPECS
- label=direct_binchoice | input=direct | tok=16 | compare=no | run_name=strat_binchoice_diag_direct_binchoice
- label=cot_binchoice | input=cot | tok=16 | compare=no | run_name=strat_binchoice_diag_cot_binchoice
- label=direct_binchoice_repro | input=direct | tok=16 | compare=yes | run_name=strat_binchoice_diag_direct_binchoice
------------------------------------------------------------------------------------------------------------------------------------------
RESULT TABLE
label                   tok     n      acc  parse_err parseable_n acc_parseable  delta_acc  changed
------------------------------------------------------------------------------------------------------------------------------------------
direct_binchoice         16   193   0.6788     0.0000         193        0.6788        n/a      n/a
cot_binchoice            16   193   0.5803     0.0000         193        0.5803        n/a      n/a
direct_binchoice_repro   16   193   0.6788     0.0000         193        0.6788    +0.0000        0
------------------------------------------------------------------------------------------------------------------------------------------
best_accuracy     : direct_binchoice (acc=0.6788, parse_err=0.0000)
lowest_parse_err  : direct_binchoice (parse_err=0.0000, acc=0.6788)
==========================================================================================================================================




**Compare freeform vs binary-choice on same group:**
**binary**
command = 
ACTIVE_PARAM_GROUP=A2 STRATEGYQA_DECODE_MODE=binary_choice **RUN_PREFIX=cot_binchoice** bash scripts/run_phase_a_benchmark_suite.sh

result =
=======================
generated_at      : 2026-02-26T21:53:11.132053+08:00
group_id          : A2
group_title       : CoT Token Sweep
run_prefix        : cot_binchoice
intention         : Measure how CoT token budget affects compliance and accuracy.
observe           : Look for monotonic or near-monotonic parse_error reductions with larger token budgets.
expectation       : Accuracy should rise with token budget until a plateau; runtime rises sharply.
------------------------------------------------------------------------------------------------------------------------------------------
SETTINGS
dataset           : strategyqa
source_split      : train
split_policy      : hash
limit             : 2000
seed              : 42
dtype             : bfloat16
log_every         : 5
max_progress_lines: 5
strategyqa_decode : binary_choice
truncate_markers  : 1
cuda_devices      : 0
model_path        : assets/models/Qwen2.5-7B-Instruct
cot_input         : assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl
suite_log_file    : assets/artifacts/phase_a_logs/cot_binchoice/suite.log
summary_file      : assets/artifacts/phase_a_logs/cot_binchoice/final_summary.md
------------------------------------------------------------------------------------------------------------------------------------------
PLANNED RUN SPECS
- label=cot_t128 | input=cot | tok=128 | compare=no | run_name=cot_binchoice_cot_t128
- label=cot_t192 | input=cot | tok=192 | compare=no | run_name=cot_binchoice_cot_t192
- label=cot_t256 | input=cot | tok=256 | compare=no | run_name=cot_binchoice_cot_t256
- label=cot_t320 | input=cot | tok=320 | compare=no | run_name=cot_binchoice_cot_t320
- label=cot_t384 | input=cot | tok=384 | compare=no | run_name=cot_binchoice_cot_t384
------------------------------------------------------------------------------------------------------------------------------------------
RESULT TABLE
label                   tok     n      acc  parse_err parseable_n acc_parseable  delta_acc  changed
------------------------------------------------------------------------------------------------------------------------------------------
cot_t128                128   193   0.5803     0.0000         193        0.5803        n/a      n/a
cot_t192                192   193   0.5803     0.0000         193        0.5803        n/a      n/a
cot_t256                256   193   0.5803     0.0000         193        0.5803        n/a      n/a
cot_t320                320   193   0.5803     0.0000         193        0.5803        n/a      n/a
cot_t384                384   193   0.5803     0.0000         193        0.5803        n/a      n/a
------------------------------------------------------------------------------------------------------------------------------------------
best_accuracy     : cot_t128 (acc=0.5803, parse_err=0.0000)
lowest_parse_err  : cot_t128 (parse_err=0.0000, acc=0.5803)
============================================================


**freeform**
command = 
ACTIVE_PARAM_GROUP=A2 STRATEGYQA_DECODE_MODE=freeform CUDA_VISIBLE_DEVICES=1 RUN_PREFIX=cot_binchoice bash scripts/run_phase_a_benchmark_suite.sh

result =
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:00:17 | rate=0.293 sample/s | eta=00:10:41
generation  : 70/193 (36.3%) | elapsed=00:03:50 | rate=0.304 sample/s | eta=00:06:45
generation  : 130/193 (67.4%) | elapsed=00:07:08 | rate=0.303 sample/s | eta=00:03:27
generation  : 193/193 (100.0%) | elapsed=00:10:37 | rate=0.303 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.1036
parse_error_rate : 0.8446
n_parseable      : 30
acc_parseable    : 0.6667
metrics_path     : assets/artifacts/phase_a_runs/cot_freeform_cot_t128_20260226T135701Z/metrics.json
========================================================================================


PHASE A gsm8k

(bcr) zling@zling-U2003:~/y/bcr/ref$ python scripts/phase_a_prepare.py \
  --datasets gsm8k \
  --source-split train \
  --split-policy hash \
  --target-style answer_only \
  --template-id qa_direct \
  --template-version 1.0.0 \
  --limit 2000 \
  --seed 42
========================================================================================
Phase A: Prepare Artifacts
========================================================================================
datasets       : ['gsm8k']
source_split   : train
split_policy   : hash
target_style   : answer_only
template       : qa_direct@1.0.0
limit          : 2000

----------------------------------------------------------------------------------------
[Dataset] gsm8k
----------------------------------------------------------------------------------------
[OK] gsm8k: wrote 2000 records to assets/artifacts/phase_a_prepared/gsm8k/a366bce0e09a
     split_counts={'train': 1628, 'validation': 172, 'test': 200}

========================================================================================
Result: SUCCESS
========================================================================================
(bcr) zling@zling-U2003:~/y/bcr/ref$ python scripts/phase_a_prepare.py \
  --datasets gsm8k \
  --source-split train \
  --split-policy hash \
  --target-style cot_then_answer \
  --template-id qa_cot_then_final \
  --template-version 1.0.0 \
  --limit 2000 \
  --seed 42
========================================================================================
Phase A: Prepare Artifacts
========================================================================================
datasets       : ['gsm8k']
source_split   : train
split_policy   : hash
target_style   : cot_then_answer
template       : qa_cot_then_final@1.0.0
limit          : 2000

----------------------------------------------------------------------------------------
[Dataset] gsm8k
----------------------------------------------------------------------------------------
[OK] gsm8k: wrote 2000 records to assets/artifacts/phase_a_prepared/gsm8k/4625c597943b
     split_counts={'train': 1628, 'validation': 172, 'test': 200}

========================================================================================
Result: SUCCESS
========================================================================================


**PHASE A GSM8K Optional smoke run (fast sanity, 20 samples)**
(bcr) zling@zling-U2003:~/y/bcr/ref$ CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl "$GSM8K_DIRECT_VAL" \
  --model-path "$MODEL" \
  --run-name gsm8k_direct_smoke \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --no-do-sample \
  --seed 42 \
  --max-new-tokens 64 \
  --max-samples 20 \
  --log-every 5 \
  --max-progress-lines 5 \
  --no-compare-latest-same-name
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/gsm8k/a366bce0e09a/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/gsm8k_direct_smoke_20260226T150012Z
seed        : 42
gen_config  : {'max_new_tokens': 64, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
decode_mode : strategyqa=freeform
trim_markers: True
log_every   : 5
max_prog_ln : 5
console_log : assets/artifacts/phase_a_runs/gsm8k_direct_smoke_20260226T150012Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 20
model_load  : start
model_load  : done in 00:00:13
first_param : cuda:0
generation  : starting 20 samples
generation  : 5/20 (25.0%) | elapsed=00:00:08 | rate=0.562 sample/s | eta=00:00:26
generation  : 10/20 (50.0%) | elapsed=00:00:17 | rate=0.579 sample/s | eta=00:00:17
generation  : 15/20 (75.0%) | elapsed=00:00:25 | rate=0.589 sample/s | eta=00:00:08
generation  : 20/20 (100.0%) | elapsed=00:00:33 | rate=0.594 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.0500
parse_error_rate : 0.0000
n_parseable      : 20
acc_parseable    : 0.0500
metrics_path     : assets/artifacts/phase_a_runs/gsm8k_direct_smoke_20260226T150012Z/metrics.json
========================================================================================
ef$ CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl "$GSM8K_DIRECT_VAL" \
  --model-path "$MODEL" \
  --run-name gsm8k_direct_t64 \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --no-do-sample \
  --seed 42 \
  --max-new-tokens 64 \
  --log-every 5 \
  --max-progress-lines 5 \
  --no-compare-latest-same-name
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/gsm8k/a366bce0e09a/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/gsm8k_direct_t64_20260226T150347Z
seed        : 42
gen_config  : {'max_new_tokens': 64, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
decode_mode : strategyqa=freeform
trim_markers: True
log_every   : 5
max_prog_ln : 5
console_log : assets/artifacts/phase_a_runs/gsm8k_direct_t64_20260226T150347Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 172
model_load  : start
model_load  : done in 00:00:08
first_param : cuda:0
generation  : starting 172 samples
generation  : 5/172 (2.9%) | elapsed=00:00:09 | rate=0.517 sample/s | eta=00:05:23
generation  : 60/172 (34.9%) | elapsed=00:01:40 | rate=0.597 sample/s | eta=00:03:07
generation  : 120/172 (69.8%) | elapsed=00:03:22 | rate=0.591 sample/s | eta=00:01:27
generation  : 172/172 (100.0%) | elapsed=00:04:50 | rate=0.592 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.0174
parse_error_rate : 0.0058
n_parseable      : 171
acc_parseable    : 0.0175
metrics_path     : assets/artifacts/phase_a_runs/gsm8k_direct_t64_20260226T150347Z/metrics.json
========================================================================================

 CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl "$GSM8K_COT_VAL" \
  --model-path "$MODEL" \
  --run-name gsm8k_cot_t256 \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --no-do-sample \
  --seed 42 \
  --max-new-tokens 256 \
  --log-every 5 \
  --max-progress-lines 5 \
  --no-compare-latest-same-name
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/gsm8k/4625c597943b/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/gsm8k_cot_t256_20260226T150434Z
seed        : 42
gen_config  : {'max_new_tokens': 256, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
decode_mode : strategyqa=freeform
trim_markers: True
log_every   : 5
max_prog_ln : 5
console_log : assets/artifacts/phase_a_runs/gsm8k_cot_t256_20260226T150434Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 172
model_load  : start
model_load  : done in 00:00:07
first_param : cuda:0
generation  : starting 172 samples
generation  : 5/172 (2.9%) | elapsed=00:00:43 | rate=0.115 sample/s | eta=00:24:09
generation  : 60/172 (34.9%) | elapsed=00:08:36 | rate=0.116 sample/s | eta=00:16:03
generation  : 120/172 (69.8%) | elapsed=00:17:12 | rate=0.116 sample/s | eta=00:07:27
generation  : 172/172 (100.0%) | elapsed=00:24:34 | rate=0.117 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.2733
parse_error_rate : 0.0000
n_parseable      : 172
acc_parseable    : 0.2733
metrics_path     : assets/artifacts/phase_a_runs/gsm8k_cot_t256_20260226T150434Z/metrics.json

**the res above is bugged, fixed res below**

Full direct baseline (recommended)
ref$ CUDA_VISIBLE_DEVICES=3 python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl "$GSM8K_DIRECT_VAL" \
  --model-path "$MODEL" \
  --run-name gsm8k_math_direct_t16 \
  --require-cuda \
  --dtype bfloat16 \
  --device-map auto \
  --no-do-sample \
  --seed 42 \
  --max-new-tokens 16 \
  --log-every 5 \
  --max-progress-lines 5 \
  --no-compare-latest-same-name
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/gsm8k/d2e0c6f17b96/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/gsm8k_math_direct_t16_20260226T151438Z
seed        : 42
gen_config  : {'max_new_tokens': 16, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
decode_mode : strategyqa=freeform
trim_markers: True
log_every   : 5
max_prog_ln : 5
console_log : assets/artifacts/phase_a_runs/gsm8k_math_direct_t16_20260226T151438Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 172
model_load  : start
model_load  : done in 00:00:06
first_param : cuda:0
generation  : starting 172 samples
generation  : 5/172 (2.9%) | elapsed=00:00:02 | rate=1.792 sample/s | eta=00:01:33
generation  : 60/172 (34.9%) | elapsed=00:00:26 | rate=2.251 sample/s | eta=00:00:49
generation  : 120/172 (69.8%) | elapsed=00:00:52 | rate=2.279 sample/s | eta=00:00:22
generation  : 172/172 (100.0%) | elapsed=00:01:14 | rate=2.303 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
math_diag       : extraction reliability check
math_diag       : n=172 acc=0.3663 last_number_rate=0.0000 hit_cap_rate=1.0000 final_tag_rate=1.0000
----------------------------------------------------------------------------------------
accuracy         : 0.3663
parse_error_rate : 0.0000
n_parseable      : 172
acc_parseable    : 0.3663
metrics_path     : assets/artifacts/phase_a_runs/gsm8k_math_direct_t16_20260226T151438Z/metrics.json
========================================================================================



(bcr) zling@zling-U2003:~/y/bcr/ref$ for TOK in 16 32 64 96; do
  CUDA_VISIBLE_DEVICES=2 python -u scripts/phase_a_generate_and_eval.py \
    --input-jsonl "$GSM8K_DIRECT_VAL" \
    --model-path "$MODEL" \
    --run-name "gsm8k_math_direct_t${TOK}" \
    --require-cuda \
    --dtype bfloat16 \
    --device-map auto \
    --no-do-sample \
    --seed 42 \
    --max-new-tokens "$TOK" \
    --log-every 5 \
    --max-progress-lines 5 \
    --no-compare-latest-same-name
done
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/gsm8k/d2e0c6f17b96/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/gsm8k_math_direct_t16_20260226T152107Z
seed        : 42
gen_config  : {'max_new_tokens': 16, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
decode_mode : strategyqa=freeform
trim_markers: True
log_every   : 5
max_prog_ln : 5
console_log : assets/artifacts/phase_a_runs/gsm8k_math_direct_t16_20260226T152107Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 172
model_load  : start
model_load  : done in 00:00:12
first_param : cuda:0
generation  : starting 172 samples
generation  : 5/172 (2.9%) | elapsed=00:00:03 | rate=1.380 sample/s | eta=00:02:01
generation  : 60/172 (34.9%) | elapsed=00:00:27 | rate=2.182 sample/s | eta=00:00:51
generation  : 120/172 (69.8%) | elapsed=00:00:53 | rate=2.242 sample/s | eta=00:00:23
generation  : 172/172 (100.0%) | elapsed=00:01:16 | rate=2.257 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
math_diag       : extraction reliability check
math_diag       : n=172 acc=0.3663 last_number_rate=0.0000 hit_cap_rate=1.0000 final_tag_rate=1.0000
----------------------------------------------------------------------------------------
accuracy         : 0.3663
parse_error_rate : 0.0000
n_parseable      : 172
acc_parseable    : 0.3663
metrics_path     : assets/artifacts/phase_a_runs/gsm8k_math_direct_t16_20260226T152107Z/metrics.json
========================================================================================
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/gsm8k/d2e0c6f17b96/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/gsm8k_math_direct_t32_20260226T152242Z
seed        : 42
gen_config  : {'max_new_tokens': 32, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
decode_mode : strategyqa=freeform
trim_markers: True
log_every   : 5
max_prog_ln : 5
console_log : assets/artifacts/phase_a_runs/gsm8k_math_direct_t32_20260226T152242Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 172
model_load  : start
model_load  : done in 00:00:05
first_param : cuda:0
generation  : starting 172 samples
generation  : 5/172 (2.9%) | elapsed=00:00:05 | rate=0.984 sample/s | eta=00:02:49
generation  : 60/172 (34.9%) | elapsed=00:00:55 | rate=1.089 sample/s | eta=00:01:42
generation  : 120/172 (69.8%) | elapsed=00:01:48 | rate=1.110 sample/s | eta=00:00:46
generation  : 172/172 (100.0%) | elapsed=00:02:34 | rate=1.114 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
math_diag       : extraction reliability check
math_diag       : n=172 acc=0.3663 last_number_rate=0.0000 hit_cap_rate=1.0000 final_tag_rate=1.0000
----------------------------------------------------------------------------------------
accuracy         : 0.3663
parse_error_rate : 0.0000
n_parseable      : 172
acc_parseable    : 0.3663
metrics_path     : assets/artifacts/phase_a_runs/gsm8k_math_direct_t32_20260226T152242Z/metrics.json
========================================================================================
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/gsm8k/d2e0c6f17b96/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/gsm8k_math_direct_t64_20260226T152530Z
seed        : 42
gen_config  : {'max_new_tokens': 64, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
decode_mode : strategyqa=freeform
trim_markers: True
log_every   : 5
max_prog_ln : 5
console_log : assets/artifacts/phase_a_runs/gsm8k_math_direct_t64_20260226T152530Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 172
model_load  : start
model_load  : done in 00:00:05
first_param : cuda:0
generation  : starting 172 samples
generation  : 5/172 (2.9%) | elapsed=00:00:09 | rate=0.501 sample/s | eta=00:05:33
generation  : 60/172 (34.9%) | elapsed=00:01:46 | rate=0.565 sample/s | eta=00:03:18
generation  : 120/172 (69.8%) | elapsed=00:03:32 | rate=0.564 sample/s | eta=00:01:32
generation  : 172/172 (100.0%) | elapsed=00:05:05 | rate=0.563 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
math_diag       : extraction reliability check
math_diag       : n=172 acc=0.3663 last_number_rate=0.0000 hit_cap_rate=1.0000 final_tag_rate=1.0000
----------------------------------------------------------------------------------------
accuracy         : 0.3663
parse_error_rate : 0.0000
n_parseable      : 172
acc_parseable    : 0.3663
metrics_path     : assets/artifacts/phase_a_runs/gsm8k_math_direct_t64_20260226T152530Z/metrics.json
========================================================================================
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/gsm8k/d2e0c6f17b96/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/gsm8k_math_direct_t96_20260226T153047Z
seed        : 42
gen_config  : {'max_new_tokens': 96, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
decode_mode : strategyqa=freeform
trim_markers: True
log_every   : 5
max_prog_ln : 5
console_log : assets/artifacts/phase_a_runs/gsm8k_math_direct_t96_20260226T153047Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 172
model_load  : start
model_load  : done in 00:00:16
first_param : cuda:0
generation  : starting 172 samples
generation  : 5/172 (2.9%) | elapsed=00:00:13 | rate=0.361 sample/s | eta=00:07:42
generation  : 60/172 (34.9%) | elapsed=00:02:39 | rate=0.376 sample/s | eta=00:04:58
generation  : 120/172 (69.8%) | elapsed=00:05:19 | rate=0.375 sample/s | eta=00:02:18
generation  : 172/172 (100.0%) | elapsed=00:07:36 | rate=0.377 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
math_diag       : extraction reliability check
math_diag       : n=172 acc=0.3663 last_number_rate=0.0000 hit_cap_rate=1.0000 final_tag_rate=1.0000
----------------------------------------------------------------------------------------
accuracy         : 0.3663
parse_error_rate : 0.0000
n_parseable      : 172
acc_parseable    : 0.3663
metrics_path     : assets/artifacts/phase_a_runs/gsm8k_math_direct_t96_20260226T153047Z/metrics.json
========================================================================================

the cot is 0.2733 acc, the direct and its sweeping all 0.3663

thorough scan



(bcr) zling@zling-U2003:~/y/bcr/ref$ python scripts/phase_a_eval_predictions.py \
  --predictions assets/artifacts/phase_a_runs/gsm8k_math_direct_t16_20260226T152107Z/predictions.jsonl \
  --run-name gsm8k_direct_reval_latest
========================================================================================
Phase A: Evaluation Result
========================================================================================
input_file       : assets/artifacts/phase_a_runs/gsm8k_math_direct_t16_20260226T152107Z/predictions.jsonl
n_total          : 172
accuracy         : 0.3721
parse_error_rate : 0.0000
n_parseable      : 172
acc_parseable    : 0.3721
output_dir       : assets/artifacts/phase_a_eval/gsm8k_direct_reval_latest_20260226T160435Z
========================================================================================

(bcr) zling@zling-U2003:~/y/bcr/ref$ python scripts/phase_a_eval_predictions.py \
  --predictions assets/artifacts/phase_a_runs/gsm8k_cot_t256_20260226T151603Z/predictions.jsonl \
  --run-name gsm8k_cot_reval_latest
========================================================================================
Phase A: Evaluation Result
========================================================================================
input_file       : assets/artifacts/phase_a_runs/gsm8k_cot_t256_20260226T151603Z/predictions.jsonl
n_total          : 172
accuracy         : 0.7035
parse_error_rate : 0.0000
n_parseable      : 172
acc_parseable    : 0.7035
output_dir       : assets/artifacts/phase_a_eval/gsm8k_cot_reval_latest_20260226T160444Z
========================================================================================

(bcr) zling@zling-U2003:~/y/bcr/ref$ CUDA_VISIBLE_DEVICES=0 python -u scripts/phase_a_generate_and_eval.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl \
  --run-name sanity_after_hardening \
  --require-cuda --dtype bfloat16 --device-map auto \
  --no-do-sample --seed 42 --max-new-tokens 16 --max-samples 20
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/sanity_after_hardening_20260226T160459Z
seed        : 42
gen_config  : {'max_new_tokens': 16, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
decode_mode : strategyqa=freeform
trim_markers: True
log_every   : 10
max_prog_ln : 5
console_log : assets/artifacts/phase_a_runs/sanity_after_hardening_20260226T160459Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 20
model_load  : start
model_load  : done in 00:00:16
first_param : cuda:0
generation  : starting 20 samples
generation  : 10/20 (50.0%) | elapsed=00:00:05 | rate=1.876 sample/s | eta=00:00:05
generation  : 20/20 (100.0%) | elapsed=00:00:09 | rate=2.127 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.6000
parse_error_rate : 0.1000
n_parseable      : 18
acc_parseable    : 0.6667
metrics_path     : assets/artifacts/phase_a_runs/sanity_after_hardening_20260226T160459Z/metrics.json
========================================================================================


(bcr) zling@zling-U2003:~/y/bcr/ref$ ACTIVE_PARAM_GROUP=A2 STRATEGYQA_DECODE_MODE=freeform  RUN_PREFIX=cot_freeform CUDA_VISIBLE_DEVICES=1 bash scripts/run_phase_a_benchmark_suite.sh
[2026-02-26 22:02:36 +0800] Repo root      : /home/zling/y/bcr/ref
[2026-02-26 22:02:36 +0800] Python         : python
[2026-02-26 22:02:36 +0800] CUDA devices   : 1
[2026-02-26 22:02:36 +0800] Model path     : assets/models/Qwen2.5-7B-Instruct
[2026-02-26 22:02:36 +0800] Run prefix     : cot_freeform
[2026-02-26 22:02:36 +0800] Log settings   : log_every=5, max_progress_lines=5
[2026-02-26 22:02:36 +0800] Decode mode    : strategyqa_decode_mode=freeform, truncate_chat_markers=1
[2026-02-26 22:02:36 +0800] Suite log file : assets/artifacts/phase_a_logs/cot_freeform/suite.log
[2026-02-26 22:02:36 +0800] Summary file   : assets/artifacts/phase_a_logs/cot_freeform/final_summary.md
[2026-02-26 22:02:36 +0800] Dataset config : dataset=strategyqa, source_split=train, split_policy=hash, limit=2000
[2026-02-26 22:02:36 +0800] WARNING: Detected other running phase_a_generate_and_eval.py processes.
[2026-02-26 22:02:36 +0800] WARNING: This suite will continue, but concurrent runs can distort speed/comparison fairness.
[2026-02-26 22:02:36 +0800] Param group    : A2 (CoT Token Sweep)
[2026-02-26 22:02:36 +0800] Intention      : Measure how CoT token budget affects compliance and accuracy.
[2026-02-26 22:02:36 +0800] Observe        : Look for monotonic or near-monotonic parse_error reductions with larger token budgets.
[2026-02-26 22:02:36 +0800] Expectation    : Accuracy should rise with token budget until a plateau; runtime rises sharply.
[2026-02-26 22:02:36 +0800] RUN: python scripts/phase_a_prepare.py --datasets strategyqa --source-split train --split-policy hash --limit 2000 --template-id qa_cot_then_final --template-version 1.0.0 --target-style cot_then_answer --seed 42 --resume
========================================================================================
Phase A: Prepare Artifacts
========================================================================================
datasets       : ['strategyqa']
source_split   : train
split_policy   : hash
target_style   : cot_then_answer
template       : qa_cot_then_final@1.0.0
limit          : 2000

----------------------------------------------------------------------------------------
[Dataset] strategyqa
----------------------------------------------------------------------------------------
[SKIP] Matching artifacts already exist.

========================================================================================
Result: SUCCESS
========================================================================================
[2026-02-26 22:02:36 +0800] Prepared CoT validation file   : assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl
[2026-02-26 22:02:36 +0800] RUN: python -u scripts/phase_a_generate_and_eval.py --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl --model-path assets/models/Qwen2.5-7B-Instruct --run-name cot_freeform_cot_t128 --require-cuda --dtype bfloat16 --device-map auto --no-do-sample --seed 42 --max-new-tokens 128 --strategyqa-decode-mode freeform --truncate-chat-markers --log-every 5 --max-progress-lines 5 --no-compare-latest-same-name
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/cot_freeform_cot_t128_20260226T140241Z
seed        : 42
gen_config  : {'max_new_tokens': 128, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
decode_mode : strategyqa=freeform
trim_markers: True
log_every   : 5
max_prog_ln : 5
console_log : assets/artifacts/phase_a_runs/cot_freeform_cot_t128_20260226T140241Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 193
model_load  : start
model_load  : done in 00:00:06
first_param : cuda:0
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:00:17 | rate=0.284 sample/s | eta=00:11:02
generation  : 70/193 (36.3%) | elapsed=00:03:54 | rate=0.299 sample/s | eta=00:06:52
generation  : 130/193 (67.4%) | elapsed=00:07:14 | rate=0.299 sample/s | eta=00:03:30
generation  : 193/193 (100.0%) | elapsed=00:10:42 | rate=0.301 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.1036
parse_error_rate : 0.8446
n_parseable      : 30
acc_parseable    : 0.6667
metrics_path     : assets/artifacts/phase_a_runs/cot_freeform_cot_t128_20260226T140241Z/metrics.json
========================================================================================
[2026-02-26 22:13:31 +0800] RUN: python -u scripts/phase_a_generate_and_eval.py --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl --model-path assets/models/Qwen2.5-7B-Instruct --run-name cot_freeform_cot_t192 --require-cuda --dtype bfloat16 --device-map auto --no-do-sample --seed 42 --max-new-tokens 192 --strategyqa-decode-mode freeform --truncate-chat-markers --log-every 5 --max-progress-lines 5 --no-compare-latest-same-name
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/cot_freeform_cot_t192_20260226T141336Z
seed        : 42
gen_config  : {'max_new_tokens': 192, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
decode_mode : strategyqa=freeform
trim_markers: True
log_every   : 5
max_prog_ln : 5
console_log : assets/artifacts/phase_a_runs/cot_freeform_cot_t192_20260226T141336Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 193
model_load  : start
model_load  : done in 00:00:14
first_param : cuda:0
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:00:26 | rate=0.190 sample/s | eta=00:16:31
generation  : 70/193 (36.3%) | elapsed=00:05:53 | rate=0.198 sample/s | eta=00:10:21
generation  : 130/193 (67.4%) | elapsed=00:10:59 | rate=0.197 sample/s | eta=00:05:19
generation  : 193/193 (100.0%) | elapsed=00:16:21 | rate=0.197 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.2591
parse_error_rate : 0.6684
n_parseable      : 64
acc_parseable    : 0.7812
metrics_path     : assets/artifacts/phase_a_runs/cot_freeform_cot_t192_20260226T141336Z/metrics.json
========================================================================================
[2026-02-26 22:30:13 +0800] RUN: python -u scripts/phase_a_generate_and_eval.py --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl --model-path assets/models/Qwen2.5-7B-Instruct --run-name cot_freeform_cot_t256 --require-cuda --dtype bfloat16 --device-map auto --no-do-sample --seed 42 --max-new-tokens 256 --strategyqa-decode-mode freeform --truncate-chat-markers --log-every 5 --max-progress-lines 5 --no-compare-latest-same-name
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/cot_freeform_cot_t256_20260226T143019Z
seed        : 42
gen_config  : {'max_new_tokens': 256, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
decode_mode : strategyqa=freeform
trim_markers: True
log_every   : 5
max_prog_ln : 5
console_log : assets/artifacts/phase_a_runs/cot_freeform_cot_t256_20260226T143019Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 193
model_load  : start
model_load  : done in 00:00:05
first_param : cuda:0
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:00:35 | rate=0.142 sample/s | eta=00:22:07
generation  : 70/193 (36.3%) | elapsed=00:07:52 | rate=0.148 sample/s | eta=00:13:50
generation  : 130/193 (67.4%) | elapsed=00:14:33 | rate=0.149 sample/s | eta=00:07:03
generation  : 193/193 (100.0%) | elapsed=00:21:38 | rate=0.149 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.4974
parse_error_rate : 0.3161
n_parseable      : 132
acc_parseable    : 0.7273
metrics_path     : assets/artifacts/phase_a_runs/cot_freeform_cot_t256_20260226T143019Z/metrics.json
========================================================================================
[2026-02-26 22:52:05 +0800] RUN: python -u scripts/phase_a_generate_and_eval.py --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl --model-path assets/models/Qwen2.5-7B-Instruct --run-name cot_freeform_cot_t320 --require-cuda --dtype bfloat16 --device-map auto --no-do-sample --seed 42 --max-new-tokens 320 --strategyqa-decode-mode freeform --truncate-chat-markers --log-every 5 --max-progress-lines 5 --no-compare-latest-same-name
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/cot_freeform_cot_t320_20260226T145211Z
seed        : 42
gen_config  : {'max_new_tokens': 320, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
decode_mode : strategyqa=freeform
trim_markers: True
log_every   : 5
max_prog_ln : 5
console_log : assets/artifacts/phase_a_runs/cot_freeform_cot_t320_20260226T145211Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 193
model_load  : start
model_load  : done in 00:00:14
first_param : cuda:0
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:00:42 | rate=0.118 sample/s | eta=00:26:34
generation  : 70/193 (36.3%) | elapsed=00:09:46 | rate=0.119 sample/s | eta=00:17:11
generation  : 130/193 (67.4%) | elapsed=00:19:47 | rate=0.110 sample/s | eta=00:09:35
generation  : 193/193 (100.0%) | elapsed=00:31:03 | rate=0.104 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.6684
parse_error_rate : 0.1088
n_parseable      : 172
acc_parseable    : 0.7500
metrics_path     : assets/artifacts/phase_a_runs/cot_freeform_cot_t320_20260226T145211Z/metrics.json
========================================================================================
[2026-02-26 23:23:31 +0800] RUN: python -u scripts/phase_a_generate_and_eval.py --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl --model-path assets/models/Qwen2.5-7B-Instruct --run-name cot_freeform_cot_t384 --require-cuda --dtype bfloat16 --device-map auto --no-do-sample --seed 42 --max-new-tokens 384 --strategyqa-decode-mode freeform --truncate-chat-markers --log-every 5 --max-progress-lines 5 --no-compare-latest-same-name
========================================================================================
Phase A: Generate + Evaluate
========================================================================================
input_jsonl : assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl
model_path  : assets/models/Qwen2.5-7B-Instruct
run_dir     : assets/artifacts/phase_a_runs/cot_freeform_cot_t384_20260226T152335Z
seed        : 42
gen_config  : {'max_new_tokens': 384, 'do_sample': False, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 50}
decode_mode : strategyqa=freeform
trim_markers: True
log_every   : 5
max_prog_ln : 5
console_log : assets/artifacts/phase_a_runs/cot_freeform_cot_t384_20260226T152335Z/console.log
torch       : 2.10.0+cu128 (build CUDA=12.8)
cuda_avail  : True
cuda_count  : 1
cuda_names  : ['NVIDIA A100 80GB PCIe']
num_inputs  : 193
model_load  : start
model_load  : done in 00:00:10
first_param : cuda:0
generation  : starting 193 samples
generation  : 5/193 (2.6%) | elapsed=00:01:04 | rate=0.077 sample/s | eta=00:40:40
generation  : 70/193 (36.3%) | elapsed=00:13:00 | rate=0.090 sample/s | eta=00:22:52
generation  : 130/193 (67.4%) | elapsed=00:23:04 | rate=0.094 sample/s | eta=00:11:10
generation  : 193/193 (100.0%) | elapsed=00:33:28 | rate=0.096 sample/s | eta=00:00:00
----------------------------------------------------------------------------------------
accuracy         : 0.6995
parse_error_rate : 0.0518
n_parseable      : 183
acc_parseable    : 0.7377
metrics_path     : assets/artifacts/phase_a_runs/cot_freeform_cot_t384_20260226T152335Z/metrics.json
========================================================================================
==========================================================================================================================================
FINAL EXPERIMENT SUMMARY
==========================================================================================================================================
generated_at      : 2026-02-26T23:57:16.453232+08:00
group_id          : A2
group_title       : CoT Token Sweep
run_prefix        : cot_freeform
intention         : Measure how CoT token budget affects compliance and accuracy.
observe           : Look for monotonic or near-monotonic parse_error reductions with larger token budgets.
expectation       : Accuracy should rise with token budget until a plateau; runtime rises sharply.
------------------------------------------------------------------------------------------------------------------------------------------
SETTINGS
dataset           : strategyqa
source_split      : train
split_policy      : hash
limit             : 2000
seed              : 42
dtype             : bfloat16
log_every         : 5
max_progress_lines: 5
strategyqa_decode : freeform
truncate_markers  : 1
cuda_devices      : 1
model_path        : assets/models/Qwen2.5-7B-Instruct
cot_input         : assets/artifacts/phase_a_prepared/strategyqa/f3e476b514c3/validation.jsonl
suite_log_file    : assets/artifacts/phase_a_logs/cot_freeform/suite.log
summary_file      : assets/artifacts/phase_a_logs/cot_freeform/final_summary.md
------------------------------------------------------------------------------------------------------------------------------------------
PLANNED RUN SPECS
- label=cot_t128 | input=cot | tok=128 | compare=no | run_name=cot_freeform_cot_t128
- label=cot_t192 | input=cot | tok=192 | compare=no | run_name=cot_freeform_cot_t192
- label=cot_t256 | input=cot | tok=256 | compare=no | run_name=cot_freeform_cot_t256
- label=cot_t320 | input=cot | tok=320 | compare=no | run_name=cot_freeform_cot_t320
- label=cot_t384 | input=cot | tok=384 | compare=no | run_name=cot_freeform_cot_t384
------------------------------------------------------------------------------------------------------------------------------------------
RESULT TABLE
label                   tok     n      acc  parse_err parseable_n acc_parseable  delta_acc  changed
------------------------------------------------------------------------------------------------------------------------------------------
cot_t128                128   193   0.1036     0.8446          30        0.6667        n/a      n/a
cot_t192                192   193   0.2591     0.6684          64        0.7812        n/a      n/a
cot_t256                256   193   0.4974     0.3161         132        0.7273        n/a      n/a
cot_t320                320   193   0.6684     0.1088         172        0.7500        n/a      n/a
cot_t384                384   193   0.6995     0.0518         183        0.7377        n/a      n/a
------------------------------------------------------------------------------------------------------------------------------------------
best_accuracy     : cot_t384 (acc=0.6995, parse_err=0.0518)
lowest_parse_err  : cot_t384 (parse_err=0.0518, acc=0.6995)
==========================================================================================================================================



333333333\\\\


A7

FINAL EXPERIMENT SUMMARY
==========================================================================================================================================
generated_at      : 2026-02-27T20:24:46.715315+08:00
group_id          : A7
group_title       : StrategyQA Prompt Style Sweep
run_prefix        : strategyqa_style_sweep
intention         : Compare three StrategyQA prompt styles under one reproducible setup.
observe           : Check style ranking on accuracy, parse_error_rate, and generation speed.
expectation       : Minimal-binary style should be cleanest; CoT style may help only if token budget is sufficient.
------------------------------------------------------------------------------------------------------------------------------------------
SETTINGS
dataset           : strategyqa
source_split      : train
split_policy      : hash
limit             : 2000
seed              : 42
dtype             : bfloat16
log_every         : 5
max_progress_lines: 5
batch_size        : 1
oom_backoff       : 1
strategyqa_decode : freeform
truncate_markers  : 1
cuda_devices      : 1
model_path        : assets/models/Qwen2.5-7B-Instruct
direct_template  : qa_strategyqa_minimal_binary (answer_only)
cot_template     : qa_strategyqa_cot_compact (cot_then_answer)
strict_template  : qa_strategyqa_evidence_verdict (answer_only)
direct_input      : assets/artifacts/phase_a_prepared/strategyqa/b98514da0ff4/validation.jsonl
cot_input         : assets/artifacts/phase_a_prepared/strategyqa/ef2ae6864f9c/validation.jsonl
strict_input      : assets/artifacts/phase_a_prepared/strategyqa/74916b892b6b/validation.jsonl
suite_log_file    : assets/artifacts/phase_a_logs/strategyqa_style_sweep/suite.log
summary_file      : assets/artifacts/phase_a_logs/strategyqa_style_sweep/final_summary.md
------------------------------------------------------------------------------------------------------------------------------------------
PLANNED RUN SPECS
- label=style_minimal_t16 | input=direct | tok=16 | compare=no | run_name=strategyqa_style_sweep_style_minimal_t16
- label=style_cot_compact_t96 | input=cot | tok=96 | compare=no | run_name=strategyqa_style_sweep_style_cot_compact_t96
- label=style_evidence_verdict_t32 | input=strict | tok=32 | compare=no | run_name=strategyqa_style_sweep_style_evidence_verdict_t32
------------------------------------------------------------------------------------------------------------------------------------------
RESULT TABLE
label                   tok     n      acc  parse_err parseable_n acc_parseable  delta_acc  changed
------------------------------------------------------------------------------------------------------------------------------------------
style_minimal_t16        16   193   0.6632     0.0000         193        0.6632        n/a      n/a
style_cot_compact_t96    96   193   0.6943     0.0311         187        0.7166        n/a      n/a
style_evidence_verdict_t32   32   193   0.3782     0.4663         103        0.7087        n/a      n/a
------------------------------------------------------------------------------------------------------------------------------------------
best_accuracy     : style_cot_compact_t96 (acc=0.6943, parse_err=0.0311)
lowest_parse_err  : style_minimal_t16 (parse_err=0.0000, acc=0.6632)
==========================================================================================================================================



A8
FINAL EXPERIMENT SUMMARY
==========================================================================================================================================
generated_at      : 2026-02-27T20:35:59.929405+08:00
group_id          : A8
group_title       : GSM8K Prompt Style Sweep
run_prefix        : gsm8k_style_sweep
intention         : Compare three GSM8K prompt styles with deterministic decode settings.
observe           : Check accuracy first, then runtime and extraction diagnostics.
expectation       : CoT style may win on quality; direct style should win on speed.
------------------------------------------------------------------------------------------------------------------------------------------
SETTINGS
dataset           : gsm8k
source_split      : train
split_policy      : hash
limit             : 2000
seed              : 42
dtype             : bfloat16
log_every         : 5
max_progress_lines: 5
batch_size        : 1
oom_backoff       : 1
strategyqa_decode : freeform
truncate_markers  : 1
cuda_devices      : 2
model_path        : assets/models/Qwen2.5-7B-Instruct
direct_template  : qa_gsm8k_direct_final_only (answer_only)
cot_template     : qa_gsm8k_cot_compact_final (cot_then_answer)
strict_template  : qa_gsm8k_equation_then_final (answer_only)
direct_input      : assets/artifacts/phase_a_prepared/gsm8k/18ffeb7b40f2/validation.jsonl
cot_input         : assets/artifacts/phase_a_prepared/gsm8k/09d73d23f451/validation.jsonl
strict_input      : assets/artifacts/phase_a_prepared/gsm8k/bdcce4830551/validation.jsonl
suite_log_file    : assets/artifacts/phase_a_logs/gsm8k_style_sweep/suite.log
summary_file      : assets/artifacts/phase_a_logs/gsm8k_style_sweep/final_summary.md
------------------------------------------------------------------------------------------------------------------------------------------
PLANNED RUN SPECS
- label=style_direct_final_t32 | input=direct | tok=32 | compare=no | run_name=gsm8k_style_sweep_style_direct_final_t32
- label=style_cot_compact_t192 | input=cot | tok=192 | compare=no | run_name=gsm8k_style_sweep_style_cot_compact_t192
- label=style_equation_t64 | input=strict | tok=64 | compare=no | run_name=gsm8k_style_sweep_style_equation_t64
------------------------------------------------------------------------------------------------------------------------------------------
RESULT TABLE
label                   tok     n      acc  parse_err parseable_n acc_parseable  delta_acc  changed
------------------------------------------------------------------------------------------------------------------------------------------
style_direct_final_t32   32   172   0.3895     0.0000         172        0.3895        n/a      n/a
style_cot_compact_t192  192   172   0.7616     0.0000         172        0.7616        n/a      n/a
style_equation_t64       64   172   0.3895     0.0000         172        0.3895        n/a      n/a
------------------------------------------------------------------------------------------------------------------------------------------
best_accuracy     : style_cot_compact_t192 (acc=0.7616, parse_err=0.0000)
lowest_parse_err  : style_direct_final_t32 (parse_err=0.0000, acc=0.3895)
==========================================================================================================================================


A8 

ACTIVE_PARAM_GROUP=A8 \
CUDA_VISIBLE_DEVICES=2 \
RUN_PREFIX=gsm8k_style_sweep \
bash scripts/run_phase_a_benchmark_suite.sh

FINAL EXPERIMENT SUMMARY
==========================================================================================================================================
generated_at      : 2026-02-27T22:29:34.083072+08:00
group_id          : A8
group_title       : GSM8K Prompt Style Sweep
run_prefix        : gsm8k_style_sweep
intention         : Compare three GSM8K prompt styles with deterministic decode settings.
observe           : Check accuracy first, then runtime and extraction diagnostics.
expectation       : CoT style may win on quality; direct style should win on speed.
------------------------------------------------------------------------------------------------------------------------------------------
SETTINGS
dataset           : gsm8k
source_split      : train
split_policy      : hash
limit             : 2000
seed              : 42
dtype             : bfloat16
log_every         : 5
max_progress_lines: 5
batch_size        : 1
oom_backoff       : 1
trunc_recovery    : 1
trunc_recov_rounds: 2
trunc_recov_extra : 96
trunc_recov_data  : gsm8k,hendrycks_math
trunc_recov_reqfa : 1
strategyqa_decode : freeform
truncate_markers  : 1
cuda_devices      : 2
model_path        : assets/models/Qwen2.5-7B-Instruct
direct_template  : qa_gsm8k_direct_final_only (answer_only)
cot_template     : qa_gsm8k_cot_compact_final (cot_then_answer)
strict_template  : qa_gsm8k_equation_then_final (answer_only)
direct_input      : assets/artifacts/phase_a_prepared/gsm8k/18ffeb7b40f2/validation.jsonl
cot_input         : assets/artifacts/phase_a_prepared/gsm8k/09d73d23f451/validation.jsonl
strict_input      : assets/artifacts/phase_a_prepared/gsm8k/bdcce4830551/validation.jsonl
suite_log_file    : assets/artifacts/phase_a_logs/gsm8k_style_sweep/suite.log
summary_file      : assets/artifacts/phase_a_logs/gsm8k_style_sweep/final_summary.md
------------------------------------------------------------------------------------------------------------------------------------------
PLANNED RUN SPECS
- label=style_direct_final_t32 | input=direct | tok=32 | compare=no | run_name=gsm8k_style_sweep_style_direct_final_t32
- label=style_cot_compact_t192 | input=cot | tok=192 | compare=no | run_name=gsm8k_style_sweep_style_cot_compact_t192
- label=style_equation_t64 | input=strict | tok=64 | compare=no | run_name=gsm8k_style_sweep_style_equation_t64
------------------------------------------------------------------------------------------------------------------------------------------
RESULT TABLE
label                   tok     n      acc  parse_err parseable_n acc_parseable  delta_acc  changed
------------------------------------------------------------------------------------------------------------------------------------------
style_direct_final_t32   32   172   0.3895     0.0000         172        0.3895        n/a      n/a
style_cot_compact_t192  192   172   0.7849     0.0000         172        0.7849        n/a      n/a
style_equation_t64       64   172   0.3895     0.0000         172        0.3895        n/a      n/a
------------------------------------------------------------------------------------------------------------------------------------------
best_accuracy     : style_cot_compact_t192 (acc=0.7849, parse_err=0.0000)
lowest_parse_err  : style_direct_final_t32 (parse_err=0.0000, acc=0.3895)
==========================================================================================================================================





A9
FINAL EXPERIMENT SUMMARY
==========================================================================================================================================
generated_at      : 2026-02-27T23:58:25.737626+08:00
group_id          : A9
group_title       : StrategyQA Full-Data Best Setting
run_prefix        : strategyqa_full_best_b4
intention         : Validate full-data performance of the best StrategyQA prompt configuration.
observe           : Track full-data accuracy, parse_error_rate, and deterministic rerun deltas.
expectation       : CoT compact t96 should remain top quality with low parse error on full-data validation.
------------------------------------------------------------------------------------------------------------------------------------------
SETTINGS
dataset           : strategyqa
source_split      : train
split_policy      : hash
limit             : None
seed              : 42
dtype             : bfloat16
log_every         : 5
max_progress_lines: 5
batch_size        : 4
oom_backoff       : 1
trunc_recovery    : 1
trunc_recov_rounds: 2
trunc_recov_extra : 96
trunc_recov_data  : gsm8k,hendrycks_math
trunc_recov_reqfa : 1
strategyqa_decode : freeform
truncate_markers  : 1
cuda_devices      : 0
model_path        : assets/models/Qwen2.5-7B-Instruct
direct_template  : qa_direct (answer_only)
cot_template     : qa_strategyqa_cot_compact (cot_then_answer)
strict_template  : qa_binary_strict (answer_only)
cot_input         : assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl
suite_log_file    : assets/artifacts/phase_a_logs/strategyqa_full_best_b4/suite.log
summary_file      : assets/artifacts/phase_a_logs/strategyqa_full_best_b4/final_summary.md
------------------------------------------------------------------------------------------------------------------------------------------
PLANNED RUN SPECS
- label=full_best_cot_t96_r1 | input=cot | tok=96 | compare=no | run_name=strategyqa_full_best_b4_full_best_cot_t96
- label=full_best_cot_t96_r2 | input=cot | tok=96 | compare=yes | run_name=strategyqa_full_best_b4_full_best_cot_t96
------------------------------------------------------------------------------------------------------------------------------------------
RESULT TABLE
label                   tok     n      acc  parse_err parseable_n acc_parseable  delta_acc  changed
------------------------------------------------------------------------------------------------------------------------------------------
full_best_cot_t96_r1     96   215   0.7116     0.0233         210        0.7286        n/a      n/a
full_best_cot_t96_r2     96   215   0.7116     0.0233         210        0.7286    +0.0000        0
------------------------------------------------------------------------------------------------------------------------------------------
best_accuracy     : full_best_cot_t96_r1 (acc=0.7116, parse_err=0.0233)
lowest_parse_err  : full_best_cot_t96_r1 (parse_err=0.0233, acc=0.7116)
==================================================================================



A11



==================================================================      generated_at      : 2026-02-28T00:27:19.039247+08:00              
group_id          : A11                                                 group_title       : StrategyQA Whole-Corpus Review (2290)     
run_prefix        : strategyqa_whole_2290_b128                          intention         : Produce report-ready whole-corpus metrics using the 
current best StrategyQA prompt setting.                                 observe           : Check split-wise metrics and the weighted aggregate 
over train+validation+test.                                             expectation       : Aggregate should be stable and reproducible; truncat
ion-safe settings should keep parse errors low.                         ------------------------------------------------------------------------
------------------------------------------------------------------
SETTINGS                                                                
dataset           : strategyqa                                          
source_split      : train                                               
split_policy      : hash                                                limit             : None                                                
seed              : 42                                                  
dtype             : bfloat16                                            
log_every         : 5                                                   max_progress_lines: 5                                                   
batch_size        : 128                                                 
oom_backoff       : 1                                                   trunc_recovery    : 1                                                   
trunc_recov_rounds: 2                                                   
trunc_recov_extra : 96                                                  
trunc_recov_data  : gsm8k,hendrycks_math,strategyqa               
trunc_recov_reqfa : 1
strategyqa_decode : freeform                                            
       
truncate_markers  : 1                                  
cuda_devices      : 1                                                   model_path        : assets/models/Qwen2.5-7B-Instruct             
direct_template  : qa_direct (answer_only)                              cot_template     : qa_strategyqa_cot_compact (cot_then_answer)
strict_template  : qa_binary_strict (answer_only)                       cot_train_input   : assets/artifacts/phase_a_prepared/strategyqa/16f7dd6
39f3e/train.jsonl                                                       cot_val_input     : assets/artifacts/phase_a_prepared/strategyqa/16f7dd6
39f3e/validation.jsonl                                                  cot_test_input    : assets/artifacts/phase_a_prepared/strategyqa/16f7dd6
39f3e/test.jsonl                                                        suite_log_file    : assets/artifacts/phase_a_logs/strategyqa_whole_2290_
b128/suite.log                      
summary_file      : assets/artifacts/phase_a_logs/strategyqa_whole_2290_
b128/final_summary.md                                                   
------------------------------------------------------------------------
------------------------------------------------------------------      PLANNED RUN SPECS                                                       
- label=full_train_t96 | input=cot_train | tok=96 | compare=no | run_nam
e=strategyqa_whole_2290_b128_full_train_t96                      
- label=full_validation_t96 | input=cot_validation | tok=96 | compare=no | run_name=strategyqa_whole_2290_b128_full_validation_t96        
- label=full_test_t96 | input=cot_test | tok=96 | compare=no | run_name=
strategyqa_whole_2290_b128_full_test_t96                                - label=full_train_t96_repro | input=cot_train | tok=96 | compare=yes | 
run_name=strategyqa_whole_2290_b128_full_train_t96
------------------------------------------------------------------------
------------------------------------------------------------------
RESULT TABLE
label                   tok     n      acc  parse_err parseable_n acc_pa
rseable  delta_acc  changed
------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------
full_train_t96           96  1834   0.7007     0.0000        1834        0.7007        n/a      n/a
full_validation_t96      96   215   0.6977     0.0000         215        0.6977        n/a      n/a
full_test_t96            96   241   0.6929     0.0000         241        0.6929        n/a      n/a
full_train_t96_repro     96  1834   0.7007     0.0000        1834        0.7007    +0.0000        0
------------------------------------------------------------------------------------------------------------------------------------------
WHOLE-CORPUS AGGREGATE
included_runs=3 (compare=no only, reproducibility reruns excluded)
n_total=2290 | n_correct=1602 | n_parse_error=0 | n_parseable=2290
accuracy=0.6996 | parse_error_rate=0.0000 | acc_parseable=0.6996
------------------------------------------------------------------------------------------------------------------------------------------
best_accuracy     : full_train_t96 (acc=0.7007, parse_err=0.0000)
lowest_parse_err  : full_train_t96 (parse_err=0.0000, acc=0.7007)
==========================================================================================================================================
[2026-02-28 00:27:19 +0800] Group run complete.


A11-128
FINAL EXPERIMENT SUMMARY
==========================================================================================================================================
generated_at      : 2026-02-28T00:44:57.318134+08:00
group_id          : A11_128
group_title       : StrategyQA Whole-Corpus Token Stress t128
run_prefix        : strategyqa_whole_t128
intention         : Stress-test suspected token-limit effects on whole-corpus StrategyQA with larger CoT budgets.
observe           : Track split-wise and aggregate accuracy while monitoring truncation-recovery activity and throughput.
expectation       : Larger token budgets should reduce cap-related failures but increase runtime; gains should eventually plateau.
------------------------------------------------------------------------------------------------------------------------------------------
SETTINGS
dataset           : strategyqa
source_split      : train
split_policy      : hash
limit             : None
seed              : 42
dtype             : bfloat16
log_every         : 5
max_progress_lines: 50
batch_size        : 128
oom_backoff       : 1
trunc_recovery    : 1
trunc_recov_rounds: 2
trunc_recov_extra : 96
trunc_recov_data  : gsm8k,hendrycks_math,strategyqa
trunc_recov_reqfa : 1
strategyqa_decode : freeform
truncate_markers  : 1
cuda_devices      : 1
model_path        : assets/models/Qwen2.5-7B-Instruct
direct_template  : qa_direct (answer_only)
cot_template     : qa_strategyqa_cot_compact (cot_then_answer)
strict_template  : qa_binary_strict (answer_only)
cot_train_input   : assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/train.jsonl
cot_val_input     : assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/validation.jsonl
cot_test_input    : assets/artifacts/phase_a_prepared/strategyqa/16f7dd639f3e/test.jsonl
suite_log_file    : assets/artifacts/phase_a_logs/strategyqa_whole_t128/suite.log
summary_file      : assets/artifacts/phase_a_logs/strategyqa_whole_t128/final_summary.md
------------------------------------------------------------------------------------------------------------------------------------------
PLANNED RUN SPECS
- label=full_train_t128 | input=cot_train | tok=128 | compare=no | run_name=strategyqa_whole_t128_full_train_t128
- label=full_validation_t128 | input=cot_validation | tok=128 | compare=no | run_name=strategyqa_whole_t128_full_validation_t128
- label=full_test_t128 | input=cot_test | tok=128 | compare=no | run_name=strategyqa_whole_t128_full_test_t128
------------------------------------------------------------------------------------------------------------------------------------------
RESULT TABLE
label                   tok     n      acc  parse_err parseable_n acc_parseable  delta_acc  changed
------------------------------------------------------------------------------------------------------------------------------------------
full_train_t128         128  1834   0.7023     0.0000        1834        0.7023        n/a      n/a
full_validation_t128    128   215   0.6977     0.0000         215        0.6977        n/a      n/a
full_test_t128          128   241   0.6888     0.0000         241        0.6888        n/a      n/a
------------------------------------------------------------------------------------------------------------------------------------------
WHOLE-CORPUS AGGREGATE
included_runs=3 (compare=no only, reproducibility reruns excluded)
n_total=2290 | n_correct=1604 | n_parse_error=0 | n_parseable=2290
accuracy=0.7004 | parse_error_rate=0.0000 | acc_parseable=0.7004
------------------------------------------------------------------------------------------------------------------------------------------
best_accuracy     : full_train_t128 (acc=0.7023, parse_err=0.0000)
lowest_parse_err  : full_train_t128 (parse_err=0.0000, acc=0.7023)
==========================================================================================================================================