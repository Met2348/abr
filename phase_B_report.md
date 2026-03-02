# Phase B Report

This report is the current single-source summary of Phase B.

Why this file exists:
- capture the Phase B experiment state in one place,
- separate completed findings from unfinished or invalid runs,
- preserve the current diagnosis for later newcomers and future replications,
- provide one stable document that can be updated after each new diagnostic finishes.

What this file contains:
- the Phase B objective and evaluation method,
- a table of completed runs,
- a diagnosis of StrategyQA and GSM8K behavior under PEFT,
- operational lessons learned from running large suites,
- the current recommended next actions.

Update rule:
- when a new Phase B diagnostic or full run finishes and has a valid summary artifact,
  update this file,
- do not treat runs as valid unless they have both:
  - a completed Phase B run directory under `assets/artifacts/phase_b_runs/`,
  - a completed summary under `assets/artifacts/phase_b_logs/<run_prefix>/`.

## 1. Current Status

- Phase B is active.
- The PEFT training and held-out re-evaluation pipeline is working end-to-end.
- StrategyQA currently shows consistent PEFT gains.
- GSM8K currently shows task-dependent behavior:
  - full CoT-style PEFT regresses,
  - shorter or style-matched GSM8K targets can improve.
- The main scientific question is now causal:
  - why does the same PEFT framework help StrategyQA but hurt the strongest GSM8K setup?

## 2. Phase B Goal

Phase B is the stage that leads from frozen Phase A benchmarks to the first trustworthy SFT/PEFT training results.

The working goal is:
- train PEFT adapters on Phase A-prepared supervision data,
- re-evaluate the trained artifact with the frozen Phase A benchmark protocol,
- measure real held-out gain before and after PEFT,
- diagnose task-specific failure modes before moving toward later BCR/ABR milestones.

## 3. Evaluation Method

All full Phase B gain runs follow the same pattern:

1. Run a pre-train held-out benchmark evaluation on the frozen base model.
2. Train a PEFT adapter on the full train split.
3. Run a post-train held-out benchmark evaluation on the same held-out splits.
4. Summarize:
   - `before_accuracy`
   - `after_accuracy`
   - `delta_accuracy`
   - `delta_n_correct`
   - parse-error changes

Important interpretation rules:
- Held-out `validation + test` is the real comparison target.
- Trainer `eval_loss` is useful, but not sufficient to judge benchmark quality.
- Runs that stop before the train stage are infrastructure failures, not experimental evidence.

## 4. Completed Runs

### 4.1 Infrastructure / Early B1 Runs

| Run | Purpose | Status | Main result |
|---|---|---|---|
| `B1_SMOKE` | Validate training/checkpoint/eval path | complete | PEFT smoke passed in about 10s on a 256-row subset |
| `B1_FIRST` | First full StrategyQA PEFT candidate | complete | training succeeded; post-train freeform eval reached `0.7617` on 193-row validation |
| `phase_b_first_eval_binchoice` | early binary-choice probe on first adapter | complete but misleading | `0.5440`; later diagnosed as evaluation-mismatch-sensitive |
| `phase_b_first_eval_freeform` | early freeform probe on first adapter | complete | `0.7617`, parse error `0.0000` |

Evidence:
- `assets/artifacts/phase_b_runs/phase_b_kickoff_b1_smoke_20260227T184939Z/summary.json`
- `assets/artifacts/phase_b_runs/phase_b_first_b1_first_20260301T125445Z/summary.json`
- `assets/artifacts/phase_a_runs/phase_b_first_eval_binchoice_20260302T092648Z/metrics.json`
- `assets/artifacts/phase_a_runs/phase_b_first_eval_freeform_20260302T101700Z/metrics.json`

### 4.2 Completed Held-Out Gain Runs

| Group | Dataset | Variant | Before acc | After acc | Delta acc | Delta correct | Status |
|---|---|---|---:|---:|---:|---:|---|
| `B2_STRATEGYQA_FULL` | StrategyQA | CoT compact, LoRA rank 16, 1.0 epoch | 0.6908 | 0.7632 | +0.0724 | +33 | complete |
| `B2_STRATEGYQA_DIAG_EPOCH_200` | StrategyQA | CoT compact, LoRA rank 16, 2.0 epochs | 0.6908 | 0.7588 | +0.0680 | +31 | complete |
| `B2_STRATEGYQA_DIAG_EPOCH_300` | StrategyQA | CoT compact, LoRA rank 16, 3.0 epochs | 0.6908 | 0.7632 | +0.0724 | +33 | complete |
| `B2_STRATEGYQA_DIAG_LORA_R8` | StrategyQA | CoT compact, LoRA rank 8, 1.0 epoch | 0.6908 | 0.7588 | +0.0680 | +31 | complete |
| `B2_STRATEGYQA_DIAG_LORA_R32` | StrategyQA | CoT compact, LoRA rank 32, 1.0 epoch | 0.6908 | 0.7675 | +0.0768 | +35 | complete |
| `B2_STRATEGYQA_DIAG_LORA_R32` (repro) | StrategyQA | CoT compact, LoRA rank 32, 1.0 epoch | 0.6864 | 0.7675 | +0.0811 | +37 | complete |
| `B2_GSM8K_FULL` | GSM8K | CoT compact, full setting | 0.8415 | 0.8118 | -0.0297 | -45 | complete |
| `B2_GSM8K_DIAG_LR_5E5` | GSM8K | CoT compact, LR 5e-5 | 0.8441 | 0.8032 | -0.0410 | -62 | complete |
| `B2_GSM8K_DIAG_LR_1E4` | GSM8K | CoT compact, LR 1e-4 | 0.8388 | 0.8045 | -0.0343 | -52 | complete |
| `B2_GSM8K_DIAG_EPOCH_025` | GSM8K | CoT compact, 0.25 epoch | 0.8415 | 0.8144 | -0.0271 | -41 | complete |
| `B2_GSM8K_DIAG_EPOCH_050` | GSM8K | CoT compact, 0.50 epoch | 0.8441 | 0.8210 | -0.0231 | -35 | complete |
| `B2_GSM8K_DIAG_SHORT_COT` | GSM8K | naive last-2-line short-CoT transform | 0.8415 | 0.5634 | -0.2781 | -421 | complete |
| `B2_GSM8K_DIAG_ANSWER_WEIGHTED` | GSM8K | rationale `0.5`, final-answer `3.0` | 0.8415 | 0.8197 | -0.0218 | -33 | complete |
| `B2_GSM8K_DIAG_CHECKPOINT_SWEEP` | GSM8K | best held-out checkpoint (`ckpt200`) | 0.8415 | 0.8369 | -0.0046 | -7 | complete |
| `B2_GSM8K_DIAG_DIRECT_STYLE` | GSM8K | direct-final style | 0.4220 | 0.5403 | +0.1183 | +44 | complete |
| `B2_GSM8K_DIAG_EQUATION_STYLE` | GSM8K | equation-then-final style | 0.4140 | 0.5349 | +0.1210 | +45 | complete |
| `B3_XTASK_STRAT_R32_TO_GSM8K` | cross-task | StrategyQA rank-32 adapter -> GSM8K | 0.8415 | 0.2041 | -0.6374 | -965 | complete |
| `B3_XTASK_GSM8K_FULL_TO_STRAT` | cross-task | GSM8K full-CoT adapter -> StrategyQA | 0.6864 | 0.6864 | +0.0000 | +0 | complete |
| `B3_XTASK_GSM8K_DIRECT_TO_STRAT` | cross-task | GSM8K direct-style adapter -> StrategyQA | 0.6864 | 0.5482 | -0.1382 | -63 | complete |
| `B3_XTASK_GSM8K_EQUATION_TO_STRAT` | cross-task | GSM8K equation-style adapter -> StrategyQA | 0.6864 | 0.6338 | -0.0526 | -24 | complete |

Primary summary files:
- `assets/artifacts/phase_b_logs/phase_b_strategyqa_full/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/strategyqa_diag_e200/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/strategyqa_diag_e300/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/strategyqa_diag_r8/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/strategyqa_diag_r32/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/strategyqa_diag_r32_repro/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/phase_b_gsm8k_full/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/gsm8k_diag_lr5e5/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/gsm8k_diag_lr1e4/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/gsm8k_diag_e025/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/gsm8k_diag_e050/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/gsm8k_diag_short_cot/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/gsm8k_diag_answer_weighted/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/gsm8k_diag_ckpt_sweep/checkpoint_sweep_summary.md`
- `assets/artifacts/phase_b_logs/gsm8k_diag_direct/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/gsm8k_diag_equation/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/xtask_strat_r32_to_gsm8k/cross_task_gain_summary.md`
- `assets/artifacts/phase_b_logs/xtask_gsm8k_full_to_strat/cross_task_gain_summary.md`
- `assets/artifacts/phase_b_logs/xtask_gsm8k_direct_to_strat/cross_task_gain_summary.md`
- `assets/artifacts/phase_b_logs/xtask_gsm8k_equation_to_strat/cross_task_gain_summary.md`

## 5. StrategyQA Diagnosis

### 5.1 Stable conclusion

StrategyQA is the clear Phase B success case so far.

Across completed full held-out runs:
- PEFT consistently improves StrategyQA by about `+0.068` to `+0.0768`.
- Parse error rate drops to zero in the successful rank-16, rank-32, and epoch-3 runs.
- The gain is not only formatting cleanup:
  - the total correct-answer gain is much larger than the number of removed parse errors.

### 5.2 Rank comparison

Relevant runs:
- rank 8: `assets/artifacts/phase_b_logs/strategyqa_diag_r8/peft_gain_summary.md`
- rank 16 baseline: `assets/artifacts/phase_b_logs/phase_b_strategyqa_full/peft_gain_summary.md`
- rank 32: `assets/artifacts/phase_b_logs/strategyqa_diag_r32/peft_gain_summary.md`

Aggregate comparison:

| Rank | After acc | Delta acc | Delta correct | Parse after |
|---|---:|---:|---:|---:|
| 8 | 0.7588 | +0.0680 | +31 | 0.0022 |
| 16 | 0.7632 | +0.0724 | +33 | 0.0000 |
| 32 | 0.7675 | +0.0768 | +35 | 0.0000 |

Diagnosis:
- `rank 32` is the current best completed StrategyQA setting.
- `rank 8` still works, so capacity is not a hard bottleneck.
- But `rank 8` is measurably weaker and leaves a small residual parse issue.
- Therefore:
  - LoRA capacity matters,
  - but only modestly,
  - and the gain from increasing rank beyond 16 is real but small.

Practical interpretation:
- rank 16 is already strong and efficient,
- rank 32 is currently the best-quality StrategyQA PEFT setting,
- rank 8 is probably too compressed if peak held-out performance is the goal.

### 5.3 Epoch comparison

Relevant runs:
- 1 epoch baseline: `assets/artifacts/phase_b_logs/phase_b_strategyqa_full/peft_gain_summary.md`
- 2 epoch diagnostic: `assets/artifacts/phase_b_logs/strategyqa_diag_e200/peft_gain_summary.md`
- 3 epoch diagnostic: `assets/artifacts/phase_b_logs/strategyqa_diag_e300/peft_gain_summary.md`

Aggregate:
- 1 epoch after accuracy: `0.7632`
- 2 epochs after accuracy: `0.7588`
- 3 epochs after accuracy: `0.7632`

Diagnosis:
- 2 epochs is slightly worse than the 1-epoch baseline.
- 3 epochs did not produce a meaningful aggregate gain beyond 1 epoch.
- Longer training changed the answer distribution much more than the aggregate score.
- As epochs increased, the model became progressively more `no`-heavy on held-out data:
  - validation predictions:
    - 1 epoch: `yes=111`, `no=104`
    - 2 epochs: `yes=93`, `no=122`
    - 3 epochs: `yes=84`, `no=131`
  - test predictions:
    - 1 epoch: `yes=118`, `no=123`
    - 2 epochs: `yes=92`, `no=149`
    - 3 epochs: `yes=89`, `no=152`
- This looks like a calibration shift toward `no`, not a genuine new quality level.

Trainer-loss note:
- training loss kept dropping with more epochs,
- but held-out benchmark accuracy did not improve accordingly,
- so StrategyQA checkpoint selection should not be based on trainer loss alone.

Current judgment:
- StrategyQA appears close to saturation around the 1-epoch regime.
- More epochs are not the most promising next lever.
- Rank scaling is more informative than epoch scaling at this point.

### 5.4 Current best StrategyQA setting

Current best completed StrategyQA run:
- `B2_STRATEGYQA_DIAG_LORA_R32`
- artifact: `assets/artifacts/phase_b_logs/strategyqa_diag_r32/peft_gain_summary.md`

Best current held-out result:
- after accuracy: `0.7675`
- delta accuracy: `+0.0768`
- delta correct: `+35`

Reproducibility note:
- the `rank 32` reproduction run reached the same after-accuracy `0.7675`,
- and because its frozen baseline happened to be slightly lower, its measured delta is `+0.0811`,
- this strengthens the conclusion that the StrategyQA `rank 32` result is stable rather than accidental.

## 6. GSM8K Diagnosis

### 6.1 Stable conclusion

GSM8K is not a simple “PEFT fails” story.

What is true:
- the strongest full GSM8K CoT-style setting regresses after PEFT,
- but shorter or style-matched GSM8K targets can improve substantially.

Therefore:
- PEFT itself is not the problem,
- the problem is the interaction between:
  - task,
  - target style,
  - optimization regime,
  - and exposure length.

### 6.2 Full CoT-style regression

Relevant run:
- `assets/artifacts/phase_b_logs/phase_b_gsm8k_full/peft_gain_summary.md`

Result:
- held-out accuracy: `0.8415 -> 0.8118`
- delta: `-0.0297`
- delta correct: `-45`
- parse error stayed `0.0000`

Diagnosis:
- this is not a parse/extraction problem,
- it is primarily a reasoning/calculation quality regression.

### 6.3 Learning-rate diagnostics

Relevant runs:
- `assets/artifacts/phase_b_logs/gsm8k_diag_lr5e5/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/gsm8k_diag_lr1e4/peft_gain_summary.md`

Results:

| LR | Before acc | After acc | Delta acc | Delta correct |
|---|---:|---:|---:|---:|
| `2e-4` (full baseline) | 0.8415 | 0.8118 | -0.0297 | -45 |
| `1e-4` | 0.8388 | 0.8045 | -0.0343 | -52 |
| `5e-5` | 0.8441 | 0.8032 | -0.0410 | -62 |

Diagnosis:
- lowering the learning rate did not recover GSM8K,
- in fact, both smaller LR runs were worse than the original `2e-4` full CoT run,
- so optimizer overshoot is not the main explanation for the regression.

Interpretation:
- the failure mechanism is more likely tied to supervision target design or
  training objective mismatch than to simple step-size instability.

### 6.4 Exposure reduction tests: 0.25 epoch and 0.50 epoch

Relevant runs:
- `assets/artifacts/phase_b_logs/gsm8k_diag_e025/peft_gain_summary.md`
- `assets/artifacts/phase_b_logs/gsm8k_diag_e050/peft_gain_summary.md`

Results:

| Exposure | Before acc | After acc | Delta acc | Delta correct |
|---|---:|---:|---:|---:|
| `1.0` epoch (full baseline) | 0.8415 | 0.8118 | -0.0297 | -45 |
| `0.50` epoch | 0.8441 | 0.8210 | -0.0231 | -35 |
| `0.25` epoch | 0.8415 | 0.8144 | -0.0271 | -41 |

Diagnosis:
- shorter exposure reduces the damage slightly,
- `0.50` epoch is the best of the long-CoT exposure variants completed so far,
- but both shorter-exposure runs still remain clearly negative.

Interpretation:
- over-exposure is probably part of the problem,
- but it is not the full explanation.
- The deeper problem is likely tied to the long CoT supervision style or to the
  loss objective itself.

### 6.5 Style diagnostics

Relevant runs:
- direct style: `assets/artifacts/phase_b_logs/gsm8k_diag_direct/peft_gain_summary.md`
- equation style: `assets/artifacts/phase_b_logs/gsm8k_diag_equation/peft_gain_summary.md`

Results:

| Style | Before acc | After acc | Delta acc | Delta correct |
|---|---:|---:|---:|---:|
| Direct-final | 0.4220 | 0.5403 | +0.1183 | +44 |
| Equation-then-final | 0.4140 | 0.5349 | +0.1210 | +45 |

Interpretation:
- these styles improve under PEFT,
- but they start from much weaker baselines than the strong full GSM8K CoT setup,
- so they do not replace the full CoT benchmark in absolute quality.

Most important implication:
- the Phase B GSM8K problem is highly likely to be specific to the strong long-CoT training target and its optimization dynamics,
- not to PEFT in general.

### 6.6 Short-CoT transform diagnostic

Relevant run:
- `assets/artifacts/phase_b_logs/gsm8k_diag_short_cot/peft_gain_summary.md`

Result:
- held-out accuracy: `0.8415 -> 0.5634`
- delta: `-0.2781`
- delta correct: `-421`

This is much worse than every other completed GSM8K long-CoT run.

Important interpretation:
- this does **not** support the statement "short CoT is inherently bad",
- it supports the narrower statement:
  - **naively truncating full GSM8K rationales to the last two reasoning lines is a bad supervision transform**.

Why this likely happened:
- many transformed targets keep downstream equations but delete the lines that
  introduced the intermediate values they depend on.
- example pattern:
  - original:
    - `In the beginning, Betty has only 100 / 2 = 50`
    - `Betty's grandparents gave her 15 * 2 = 30`
    - `This means, Betty needs 100 - 50 - 30 - 15 = 5 more`
  - transformed:
    - `Betty's grandparents gave her 15 * 2 = 30`
    - `This means, Betty needs 100 - 50 - 30 - 15 = 5 more`
- the shortened target is grammatically plausible but logically incomplete.

Additional evidence:
- training loss became much worse than the full GSM8K CoT baseline:
  - full CoT train loss: about `0.1675`
  - short-CoT transform train loss: about `1.4988`
- this strongly suggests the transformed targets are low-quality supervision,
  not simply a different but valid rationale style.

Conclusion from this run:
- naive rationale truncation is not a valid GSM8K repair,
- if we revisit short-CoT later, it must use:
  - genuinely rewritten short rationales,
  - or distilled short-CoT targets from a stronger model,
  - not simple last-N-line clipping.

### 6.7 Answer-weighted supervision diagnostic

Relevant run:
- `assets/artifacts/phase_b_logs/gsm8k_diag_answer_weighted/peft_gain_summary.md`

Result:
- held-out accuracy: `0.8415 -> 0.8197`
- delta: `-0.0218`
- delta correct: `-33`

Interpretation:
- answer-weighting is the first new objective-level change that actually improves over the original full CoT baseline:
  - full CoT baseline: `-0.0297`
  - answer-weighted: `-0.0218`
- but it is still negative overall,
- so rationale-dominated loss is part of the GSM8K problem, but not the whole problem.

More detailed split view:
- validation is essentially preserved:
  - `0.8366 -> 0.8352` (`-1`)
- the remaining damage is almost entirely concentrated on test:
  - `0.8458 -> 0.8060` (`-32`)

Conclusion from this run:
- the objective matters,
- emphasizing the final answer reduces damage,
- but the current long-CoT supervision recipe is still not safe enough for GSM8K.

### 6.8 Checkpoint sweep diagnostic

Relevant run:
- `assets/artifacts/phase_b_logs/gsm8k_diag_ckpt_sweep/checkpoint_sweep_summary.md`

Result:
- best checkpoint: `200`
- best held-out accuracy: `0.8369`
- delta from frozen baseline: `-0.0046`
- delta correct: `-7`

Checkpoint table:

| Checkpoint | Held-out acc | Delta acc | Delta correct |
|---|---:|---:|---:|
| 100 | 0.8111 | -0.0304 | -46 |
| 200 | 0.8369 | -0.0046 | -7 |
| 300 | 0.8190 | -0.0225 | -34 |
| 400 | 0.8210 | -0.0205 | -31 |
| 500 | 0.8137 | -0.0277 | -42 |
| 600 | 0.8104 | -0.0310 | -47 |
| 700 | 0.8078 | -0.0337 | -51 |
| 745 | 0.8058 | -0.0357 | -54 |
| final | 0.8058 | -0.0357 | -54 |

Interpretation:
- checkpoint timing is now the strongest single factor discovered for GSM8K,
- the model is much better around step `200` than at the final adapter,
- most of the eventual GSM8K damage is accumulated after that point.

Practical implication:
- selecting the final checkpoint by training completion or trainer loss is actively harmful for GSM8K,
- held-out benchmark selection or explicit early stopping is necessary for this task under the current recipe.

Important caveat:
- even the best checkpoint is still slightly negative,
- so checkpoint timing is not the whole story,
- but it is strong enough to change the diagnosis from “recipe just fails” to
  “recipe deteriorates badly if allowed to drift late in training.”

### 6.9 Current GSM8K diagnosis

Most likely current explanation:
- full GSM8K CoT PEFT teaches style and structured solution format,
- but distorts or overwrites part of the base model’s arithmetic faithfulness,
- producing cleaner but less trustworthy reasoning traces.

This is consistent with the sample-level errors already observed in the artifacts:
- cleaner equation formatting,
- earlier local arithmetic or unit mistake,
- then consistent downstream propagation of the wrong intermediate result.

Additional important observation:
- the frozen pre-train GSM8K baseline itself shifts slightly across suites
  (about `0.8388` to `0.8441` held-out accuracy),
- so the evaluation path is not perfectly numerically stable across all reruns,
- but every completed long-CoT GSM8K variant is still negative,
- therefore the negative GSM8K PEFT conclusion is robust to that small baseline drift.

More precise current diagnosis after all completed GSM8K diagnostics:
- the dominant factor is now:
  - checkpoint drift / late-run degradation,
- and the secondary factor is:
  - residual supervision-target damage even after better answer weighting,
- while naive clipped short-CoT is clearly invalid supervision.

Current best mechanistic reading:
- early in training, PEFT can preserve most GSM8K quality,
- later training steps push the adapter toward a worse solution basin,
- answer-weighting helps, which means the loss objective matters,
- but answer-weighting alone cannot fully prevent the late-run damage,
- therefore the GSM8K problem is best described as:
  - **late-run drift under a long-CoT objective, with rationale-heavy supervision amplifying the drift.**

### 6.10 Current best GSM8K conclusion

Current strongest statement that is defensible:
- full CoT GSM8K PEFT is currently harmful under the tested recipe,
- lowering learning rate does not fix it,
- shortening exposure alone does not fix it,
- answer-weighting reduces but does not remove the damage,
- checkpoint selection changes the conclusion dramatically:
  - the best retained checkpoint is only `-7` correct from the frozen base model,
  - while the final checkpoint in the sweep run is `-54`,
- style choice clearly matters,
- naive short-CoT truncation is catastrophically bad and should not be used as a repair,
- so the next GSM8K work should prioritize:
  - checkpoint selection by held-out benchmark or explicit early stopping,
  - confirming whether answer-weighted supervision plus early checkpointing is enough,
  - and, if short rationale is revisited later, rewritten/distilled short-CoT data rather than clipped long-CoT traces,
  - and only after that, more complex mixed-data strategies.

### 6.11 New GSM8K diagnostics now added to the codebase

The next GSM8K diagnostic layer is now implemented, even if some runs are still
pending:

1. `B2_GSM8K_DIAG_CHECKPOINT_SWEEP`
   - purpose:
     - test whether benchmark quality peaks before the final saved adapter,
     - separate late-run drift from supervision-target damage.
   - implementation:
     - dense checkpoint saving (`save_steps=100`, `save_total_limit=12`),
     - automatic held-out checkpoint sweep summary after training.

2. `B2_GSM8K_DIAG_SHORT_COT`
   - purpose:
     - test whether long-CoT supervision is the real cause of the GSM8K drop.
   - implementation:
     - same full GSM8K CoT corpus,
     - but supervision keeps only the last two reasoning lines plus the final answer.

3. `B2_GSM8K_DIAG_ANSWER_WEIGHTED`
   - purpose:
     - test whether the current token loss overweights rationale style and
       underweights final-answer correctness.
   - implementation:
     - same full GSM8K CoT corpus and optimizer regime,
     - rationale tokens weighted `0.5`,
     - final-answer tokens weighted `3.0`.

4. `B2_GSM8K_REPAIR_ANSWER_WEIGHTED_CKPT`
   - purpose:
     - combine the two strongest completed GSM8K repair signals:
       - answer-weighted supervision,
       - and early/best checkpoint selection.
   - implementation:
     - answer-weighted long-CoT training,
     - dense checkpoint saving (`save_steps=100`, `save_total_limit=12`),
     - automatic held-out checkpoint sweep after training.

## 7. Running Status

- No previously-tracked critical Phase B diagnostic remains pending in the current repo snapshot.
- All major GSM8K and StrategyQA diagnostics discussed so far now have completed summary artifacts.

Operational lesson:
- heavy full-dataset Phase B suites should still be run one-at-a-time per GPU,
- but the current evidence base is now sufficient for a consolidated Phase B diagnosis.

## 8. Main Lessons Learned So Far

1. Phase B infrastructure is now trustworthy enough to produce real held-out PEFT evidence.
2. StrategyQA is a genuine PEFT success case.
3. GSM8K is a task-sensitive failure-analysis case, not a universal PEFT success.
4. Trainer loss alone is not enough to select the best checkpoint or recipe.
5. Evaluation-protocol alignment matters:
   - early StrategyQA binary-choice probing was misleading for the first adapter,
   - matched freeform evaluation gave the more meaningful signal.
6. LoRA capacity matters somewhat on StrategyQA, but more training epochs do not currently buy much.
7. GSM8K’s current failure mode looks more like supervision-style or objective damage than a formatting bug.
8. Lower LR and shorter exposure reduce or worsen the damage only marginally; they do not solve it.
9. Naively clipped short-CoT supervision is not a valid substitute for true short-CoT data.

## 9. Recommended Next Actions

### StrategyQA

1. Keep `rank 16` as the stable default if compute efficiency matters.
2. Treat `rank 32` as the best-quality completed StrategyQA setting.
3. Do not spend more effort on longer-epoch StrategyQA runs unless a later result contradicts the current saturation signal.

### GSM8K

1. Promote held-out checkpoint selection or explicit early stopping to the default GSM8K model-selection rule.
2. Run the combined repair group:
   - `B2_GSM8K_REPAIR_ANSWER_WEIGHTED_CKPT`
3. Treat that combined run as the decisive next GSM8K repair test:
   - if it matches or exceeds the frozen base model, the main Phase B GSM8K issue is late-run drift plus loss imbalance,
   - if it still remains clearly negative, move on from hyperparameter-level tuning.
4. Do not use the current naive `short_cot_last2` transform again for GSM8K repair experiments.
5. If the combined repair run still does not recover quality, escalate to more complex designs:
   - mixed-data training,
   - dataset subsampling,
   - staged math-then-general alignment,
   - or a rewritten/distilled short-CoT dataset rather than clipped CoT traces.
6. Continue collecting sample-level failure cases from the post-PEFT full CoT GSM8K run.

## 10. Cross-Task Interference Suite

This suite is now implemented in:
- `scripts/run_phase_b_cross_task_suite.sh`

Purpose:
- move Phase B from single-task gain measurement to the cross-task conflict
  question that is directly relevant for later BCR/ABR work.

Implemented groups:
1. `B3_XTASK_STRAT_R32_TO_GSM8K`
2. `B3_XTASK_GSM8K_FULL_TO_STRAT`
3. `B3_XTASK_GSM8K_DIRECT_TO_STRAT`
4. `B3_XTASK_GSM8K_EQUATION_TO_STRAT`

What these groups should tell us:
- whether the best StrategyQA adapter interferes with arithmetic reasoning,
- whether the harmful full-CoT GSM8K adapter also causes cross-task damage on StrategyQA,
- whether shorter GSM8K target styles reduce that cross-task damage.

Current completed cross-task results:

| Group | Transfer | Before acc | After acc | Delta acc | Delta correct | Parse change |
|---|---|---:|---:|---:|---:|---:|
| `B3_XTASK_STRAT_R32_TO_GSM8K` | StrategyQA rank-32 -> GSM8K | 0.8415 | 0.2041 | -0.6374 | -965 | +0.0000 |
| `B3_XTASK_GSM8K_FULL_TO_STRAT` | GSM8K full-CoT -> StrategyQA | 0.6864 | 0.6864 | +0.0000 | +0 | +0.0175 |
| `B3_XTASK_GSM8K_DIRECT_TO_STRAT` | GSM8K direct -> StrategyQA | 0.6864 | 0.5482 | -0.1382 | -63 | +0.2390 |
| `B3_XTASK_GSM8K_EQUATION_TO_STRAT` | GSM8K equation -> StrategyQA | 0.6864 | 0.6338 | -0.0526 | -24 | -0.0132 |

Current interpretation:
- the best StrategyQA adapter transfers catastrophically to GSM8K even with zero parse errors,
  which is the strongest completed evidence of genuine cross-task capability interference in Phase B so far.
- GSM8K full-CoT training does not improve StrategyQA, but it also does not obviously destroy it in aggregate; instead it shifts errors and worsens parse compliance.
- GSM8K direct-style training transfers badly to StrategyQA, mostly by causing a large parse-error spike.
- GSM8K equation-style training transfers less destructively than direct-style, but still reduces StrategyQA answer quality.
- This suggests the cross-task interference story is both:
  - asymmetric across source/target directions,
  - and style-dependent within GSM8K itself.

## 10. Maintenance Note

After each completed Phase B diagnostic:
- update the completed-runs table,
- move newly finished runs out of the incomplete section,
- revise the StrategyQA or GSM8K diagnosis if the new run changes the current causal story.
