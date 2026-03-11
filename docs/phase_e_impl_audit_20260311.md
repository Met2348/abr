# Phase E Implementation Audit (2026-03-11)

## Scope

This audit focused on the current Phase E implementation paths that can silently
distort training or evaluation outcomes:

1. runtime / backbone loading
2. pair training objectives
3. same-family trust evaluation
4. benchmark-native evaluation
5. candidate checkpoint promotion

## High-Risk Findings Fixed

### 1. `dual_head` silently dropped terminal BCE

- File: `src/ours/phase_e/training.py`
- Risk:
  - `lambda_terminal_bce` was threaded through the public trainer config,
    but the `dual_head` branch never forwarded it into the terminal-head loss.
  - Result: experiments that claimed to use BiRM-style terminal supervision on
    `dual_head` were in fact missing that term.
- Fix:
  - The terminal routed branch now forwards both
    `terminal_pair_weights_for_bce` and `lambda_terminal_bce`.

### 2. same-family trust pools could merge different sources with the same prompt text

- File: `src/ours/phase_e/samefamily_trust.py`
- Risk:
  - prompt pools and candidate ids were keyed only by `prompt_text` and
    `candidate_text`.
  - in mixed-source artifacts, two different sources with the same prompt text
    could be silently merged into one pool.
  - this contaminates `prompt_pool_top1`, rejection curves, and local-first-bad
    diagnostics.
- Fix:
  - prompt pools and candidate ids are now source-scoped:
    `source_tag + prompt_text [+ candidate_text]`.

### 3. synthesized pad token could outgrow model embeddings

- File: `src/ours/phase_e/runtime.py`
- Risk:
  - when a tokenizer snapshot had neither `pad_token` nor `eos_token`, the
    runtime added a synthetic pad token.
  - the model embeddings were not resized afterwards.
  - batched scoring could then feed an out-of-range pad id into the backbone.
- Fix:
  - the runtime now resizes embeddings when tokenizer growth introduces new ids.

### 4. PRMBench preview index normalization was a silent one-based heuristic

- File: `src/ours/phase_e/benchmark_eval.py`
- Risk:
  - the loader always shifted positive `error_steps` left by one.
  - if a future snapshot used 0-based indices, the benchmark would silently
    mislabel every positive pair.
- Fix:
  - loader now infers index base from dataset evidence:
    - `idx == 0` => zero-based
    - `idx >= len(process)` => one-based
  - ambiguous snapshots now raise instead of silently guessing.
  - `phase_e_eval_benchmark.py` also exposes
    `--prmbench-error-step-index-base` for explicit override.

### 5. checkpoint resolution and ProcessBench F1 calibration were under-explained

- Files:
  - `src/ours/phase_e/runtime.py`
  - `scripts/phase_e_eval_benchmark.py`
  - `scripts/phase_e_eval_samefamily_trust.py`
  - `scripts/phase_e_select_candidate.py`
- Risk:
  - requesting `checkpoint_name=best` could silently fall back to
    `final_value_head.pt`.
  - ProcessBench F1 defaulted to an oracle threshold sweep on the evaluated
    benchmark rows, but reports did not make that explicit.
- Fix:
  - checkpoint fallback now emits an explicit warning and is recorded in eval
    manifests.
  - candidate selection resolves published checkpoint paths explicitly instead of
    blindly pointing at `<run_dir>/best_value_head.pt`.
  - benchmark eval now records whether ProcessBench F1 used:
    - a fixed threshold, or
    - an oracle sweep (`processbench_f1_is_oracle=true`)
  - benchmark eval also exposes `--processbench-f1-threshold` for fixed-threshold
    runs.

## Residual Risks Not Fully Eliminated

### 1. Historical suite wrappers still mostly summarize ProcessBench by pair/AUC

- Several shell suites still aggregate benchmark outputs around `pair_acc/auc`.
- The benchmark evaluator now exposes ProcessBench F1 provenance more cleanly,
  but not every historical suite report automatically uses it yet.
- This is now easier to diagnose, but not fully normalized across all wrappers.

### 2. `best -> final` fallback still exists for legacy artifact compatibility

- This was kept because older runs rely on it.
- The danger is now surfaced in logs/manifests/candidate reports, but the
  fallback behavior itself remains by design.

## Verification

- `python -m py_compile` on the touched Phase E modules and scripts
- `pytest -q` on targeted regression tests
- broader Phase E core regression set: `46 passed`

## Late Addendum: Postfix Evaluation Hygiene Closure

After the first Phase E implementation audit, one more postfix pass found three
evaluation-side gaps that still affected the trust story.

### 1. benchmark-informed candidate ranking leakage by default

- File: `scripts/phase_e_select_candidate.py`
- Risk:
  - benchmark metrics still influenced seed/group ranking directly
  - even when the benchmark was intended to act only as a promotion canary
- Fix:
  - default `selection_policy=heldout_only`
  - legacy `hybrid` ranking remains available only as an explicit option
- Regression:
  - `tests/unit/test_phase_e_select_candidate.py`

### 2. intradataset candidate reports could publish a bogus best-checkpoint path

- File: `scripts/phase_e_select_intradataset_candidate.py`
- Risk:
  - older runs that only retained `final_value_head.pt` still published
    `<run_dir>/best_value_head.pt` in candidate reports
- Fix:
  - manifest-aware best-checkpoint resolution with explicit fallback notes
- Regression:
  - `tests/unit/test_phase_e_select_intradataset_candidate.py`

### 3. PRM-7B samefamily trust could crash on long batches with async CUDA failures

- File: `src/ours/phase_e/runtime.py`
- Risk:
  - long candidate-node batches sometimes first surfaced as a later generic
    `device-side assert triggered`, which made evaluation failure look like a
    semantic model issue instead of a batch-capacity problem
- Fix:
  - explicit `cuda.synchronize()` inside the encoding retry block
  - relaxed retry rule for allocator / async capacity failure strings
- Regression:
  - `tests/unit/test_phase_e_runtime.py`

### Postfix interpretation

These fixes did not by themselves make the best checkpoints RL-ready.

What they did change:

1. candidate promotion is now benchmark-cleaner by default,
2. intradataset reports no longer publish misleading checkpoint paths,
3. PRM-7B samefamily trust audits now complete robustly instead of aborting on
   long-batch capacity spikes.
