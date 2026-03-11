# Phase E Backbone-And-Judge Experiments (2026-03-11)

## Purpose

This note answers three related questions:

1. does backbone adaptation matter more than the current frozen-head ablations suggest,
2. can an existing local PRM oracle clean the current pair artifacts enough to improve transfer without touching the backbone,
3. is the newly installed `Qwen2.5-Math-7B-Instruct` judge already good enough to act as an automatic prefix-pair relabeler.

This round did **not** implement the full in-repo `LoRA / PEFT` Phase E path.
That path is still blocked by:

1. no online-backprop backbone path in `phase_e_train_value.py`,
2. no cache-invalidation design for trainable backbone runs,
3. `peft` not installed in the active environment.

So the experiments below are the shortest bounded probes that still answer the
scientific question cleanly.

---

## Experiment A: adapted-backbone proxy with `Qwen2.5-Math-PRM-7B`

### Why this is informative

If the main bottleneck is really backbone representation rather than small-head
capacity, then holding the local head pipeline fixed and swapping in an already
PRM-tuned community backbone should help benchmark transfer substantially.

To make this possible, `src/ours/phase_e/runtime.py` was extended so Phase E can
load reward-model style backbones (for example `Qwen2ForProcessRewardModel`)
through `AutoModel`, not only `AutoModelForCausalLM`.

### Training setup

Run:

1. `assets/artifacts/phase_e_runs/phase_e_backboneproxy_prm_mixedsmall_20260311T074134Z`

Data:

1. `phase_e_pb_repair_0311_mixed_small_fanout_terminal__99976bcc33a8`

Backbone:

1. `assets/models/Qwen2.5-Math-PRM-7B`

Head / objective:

1. `mlp`
2. `joint`
3. `pair_weight_mode=group_balance`
4. `source_balance=uniform`
5. no head warm-start

### Held-out pair result

1. `pair_acc = 0.8984`
2. `auc = 0.8946`

Reference frozen-base control on the same artifact family:

1. `mixed MLP` run:
   - `assets/artifacts/phase_e_runs/phase_e_pb_repair_0311_e75_mixed_small_mlp_20260311T042122Z`
2. held-out:
   - `pair_acc = 0.9141`
   - `auc = 0.8784`

Interpretation:

1. adapted backbone did **not** improve in-domain held-out pair accuracy,
2. but it slightly improved held-out AUC.

### Same-family trust result

Run:

1. `assets/artifacts/phase_e_samefamily_eval/phase_e_backboneproxy_prm_mixedsmall_samefamily_20260311T080555Z`

Metrics:

1. `prompt_pool_top1_accuracy = 0.9160`
2. `prompt_pool_mean_regret = 0.0840`
3. `local_last_safe_top1_accuracy = 0.9415`
4. `local_first_bad_edge_accuracy = 0.9487`

Interpretation:

1. same-family utility is **not** dominant here;
2. compared with the repository's stronger same-family winners (`E82`, `ms_e43`),
   this is weaker.

### ProcessBench Math 50 result

Run:

1. `assets/artifacts/phase_e_eval/phase_e_backboneproxy_prm_mixedsmall_math50_20260311T080653Z`

Metrics:

1. `pair_accuracy_good_vs_bad = 0.6234`
2. `pair_auc_good_vs_bad = 0.6810`
3. `first_error_edge_accuracy = 0.6923`
4. `mean_all_correct_last_score = 0.5749`
5. `processbench_f1 = 0.4800`
6. `processbench_acc_erroneous = 0.4000`
7. `processbench_acc_correct = 0.6000`

Frozen-base control on the same `Math 50` slice:

1. `assets/artifacts/phase_e_eval/phase_e_pb_repair_0311_e75_mixed_small_mlp_math50_20260311T042808Z`
   - `pair_acc = 0.5000`
   - `auc = 0.5400`
   - `first_edge = 0.5000`
   - `mean_all_correct_last_score = 0.4822`
2. `assets/artifacts/phase_e_eval/phase_e_pb_repair_0311_e46_math50_20260310T170837Z`
   - `pair_acc = 0.4778`
   - `auc = 0.5335`
   - `first_edge = 0.5000`
   - `mean_all_correct_last_score = 0.2572`

### Conclusion of Experiment A

This is the strongest result of the round:

1. swapping in an already adapted PRM backbone gives a **large** ProcessBench lift,
2. without changing the head architecture or data curation recipe,
3. but it does **not** preserve the strongest same-family ordering behavior.

Practical interpretation:

1. the repo's ProcessBench ceiling is indeed heavily representation-limited,
2. backbone adaptation is a more important lever than another scalar-head tweak,
3. but naive adaptation can trade away same-family trust geometry.

---

## Experiment B: PRM-oracle pair filtering

### Why this is informative

This isolates the data-quality question:

1. keep the backbone fixed,
2. keep the Phase E trainer fixed,
3. use an independent PRM oracle to keep only pairs where:
   - chosen final-step score is high,
   - rejected final-step score is low.

### New filtering script

1. `scripts/phase_e_oracle_filter_pairs.py`

Artifact:

1. `assets/artifacts/phase_e_pairs/phase_e_prm_oracle_filter_mixedsmall_20260311T081105Z`

Filter config:

1. `chosen_threshold = 0.60`
2. `rejected_threshold = 0.40`
3. `min_margin = 0.10`

### Filter outcome

Train keep rate:

1. `878 / 2048 = 0.4287`

Eval keep rate:

1. `121 / 256 = 0.4727`

By semantics on train:

1. `first_bad_fanout_prefix_ranking`: kept `770 / 1536`
2. `local_modified_process_error_step`: kept `107 / 380`
3. `terminal_completion_anchor`: kept `1 / 132`

Interpretation:

1. the oracle keeps a large local/fanout subset,
2. but it almost completely removes terminal anchors,
3. so this is **not** a neutral denoiser; it is a geometry-changing filter.

### Training on filtered data

Run:

1. `assets/artifacts/phase_e_runs/phase_e_oraclefilter_base_mixedsmall_20260311T081526Z`

Backbone:

1. `assets/models/Qwen2.5-7B-Instruct`

Head / objective:

1. same frozen-base `mixed MLP` recipe as before
2. warm-started from the same `E46` head used by the earlier mixed-small run

Held-out result:

1. `pair_acc = 0.8678`
2. `auc = 0.8427`

This is worse than the unfiltered mixed-base control:

1. control `mixed MLP`:
   - `pair_acc = 0.9141`
   - `auc = 0.8784`

### ProcessBench Math 50 result

Run:

1. `assets/artifacts/phase_e_eval/phase_e_oraclefilter_base_mixedsmall_math50_20260311T082011Z`

Metrics:

1. `pair_accuracy_good_vs_bad = 0.5190`
2. `pair_auc_good_vs_bad = 0.5535`
3. `first_error_edge_accuracy = 0.5000`
4. `mean_all_correct_last_score = 0.1543`
5. `processbench_f1 = 0.2933`

Compared with the unfiltered mixed-base control:

1. control `mixed MLP`:
   - `pair_acc = 0.5000`
   - `auc = 0.5400`
   - `mean_all_correct_last_score = 0.4822`

### Conclusion of Experiment B

This filtering policy is a **negative** result:

1. it does not materially improve local benchmark discrimination,
2. it badly damages terminal behavior,
3. and the root cause is visible in the kept-set composition:
   - terminal anchors were almost entirely deleted.

Practical interpretation:

1. a PRM oracle can still be useful,
2. but not as a single global keep/drop gate over a mixed artifact,
3. because it over-selects the local/fanout regime and destroys the multi-objective mix.

---

## Experiment C: direct prefix judge audit with `Qwen2.5-Math-7B-Instruct`

### Why this is informative

The repo already installed a local judge LLM. Before using it to relabel data,
we need to know whether it can satisfy the actual Phase E prefix contract:

1. judge whether the **displayed steps so far** are valid,
2. not punish incomplete prefixes,
3. return machine-readable outputs.

### New audit script

1. `scripts/phase_e_audit_judge_prefix_pairs.py`

Run:

1. `assets/artifacts/phase_e_judge_audit/phase_e_judge_prefix_audit_mixedsmall_20260311T082237Z`

Judge:

1. `assets/models/Qwen2.5-Math-7B-Instruct`

Data:

1. 24 validation pairs, stratified by `pair_semantics`

### Result

1. `pair_json_ok_rate = 0.0000`
2. `pair_agreement_rate = 0.0000`
3. elapsed time:
   - `369.3 sec` for only 24 pairs

Observed failure mode:

1. the model usually produced free-form analysis,
2. not the required JSON object,
3. even though the prompt explicitly told it:
   - incomplete-but-correct prefixes should still count as `correct`.

### Conclusion of Experiment C

At least in the current local `transformers + direct generate` stack,
`Qwen2.5-Math-7B-Instruct` is **not** ready to be the main automatic prefix-pair
relabeler.

This does **not** mean the judge is useless. It means its immediate role should
be narrower:

1. disagreement audit,
2. terminal / full-solution relabel,
3. low-volume human-in-the-loop review,
4. or a future service-based pipeline with stronger constrained decoding /
   output parsing.

---

## Overall Takeaways

1. **Backbone adaptation matters a lot.**
   - The adapted PRM backbone produced the clearest benchmark gain of the round.
2. **Data filtering by a second verifier is not automatically good.**
   - A naive global PRM-oracle filter changed the supervision geometry too much
     and hurt terminal behavior badly.
3. **The new judge LLM is not yet a drop-in automatic prefix cleaner.**
   - It failed the structured-output contract on the current prefix-pair task.

## What this changes in the roadmap

1. the next serious architecture investment should be:
   - real in-repo backbone adaptation (`LoRA` / online-backprop path),
   - not another small frozen-head variant;
2. the next judge integration should be:
   - disagreement mining or terminal/full-solution relabel,
   - not direct global prefix-pair filtering;
3. any future oracle filtering must be semantics-aware:
   - different rules for local/fanout vs terminal anchors,
   - otherwise the filter silently destroys the mixed objective.
