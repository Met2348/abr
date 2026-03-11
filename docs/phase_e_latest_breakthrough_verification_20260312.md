# Phase E Latest Breakthrough Verification (2026-03-12)

This note verifies the newest strong claims that appeared in the repo after the
previous Phase E paradigm-refresh pass.

The main question here is simple:

1. is there a real new breakthrough in the latest artifacts?
2. if yes, what kind of breakthrough is it exactly?
3. does it survive same-family trust and RL-facing diagnostics?

## 1. Claims That Needed Verification

The latest docs / artifacts were making two kinds of strong claims:

1. `PBR26` is the new best frozen candidate and may be the new RL-facing mainline.
2. `PBR31` LoRA may be the first meaningful post-frozen breakthrough on top of `Math-PRM-7B`.

These claims were not equally well verified when this note started:

- `PBR26` already had benchmark evals, but same-family trust / strict transfer diagnosis
  were missing.
- `PBR31` had training summaries and empty auto-eval directories, so the real benchmark
  result still needed to be run explicitly.

## 2. Verification Runs Added In This Pass

### 2.1 `PBR26` same-family + strict diagnosis

- same-family eval:
  - `assets/artifacts/phase_e_samefamily_eval/pbr26_samefamily_verify_0312_20260311T165945Z`
- strict transfer diagnosis:
  - `assets/artifacts/phase_e_transfer_diag/phase_e_pbr26_verify_diag_0312_00`
- math failure analysis:
  - `assets/artifacts/phase_e_failures/pbr26_math_failure_verify_0312_20260311T170622Z`

### 2.2 `PBR31` full benchmark + same-family + strict diagnosis

- same-family eval:
  - `assets/artifacts/phase_e_samefamily_eval/pbr31_samefamily_verify_0312_20260311T170244Z`
- ProcessBench GSM:
  - `assets/artifacts/phase_e_eval/pbr31_verify_gsm_0312_20260311T170309Z`
- ProcessBench Math:
  - `assets/artifacts/phase_e_eval/pbr31_verify_math_0312_20260311T170630Z`
- strict transfer diagnosis:
  - `assets/artifacts/phase_e_transfer_diag/phase_e_pbr26_pbr31_verify_diag_0312_00`
- math failure analysis:
  - `assets/artifacts/phase_e_failures/pbr31_math_failure_verify_0312_20260311T171825Z`
- reward-hacking probe:
  - `assets/artifacts/phase_f_reward_hacking/phase_f_pbr31_probe_verify_0312_20260311T172022Z`

## 3. Verified Results

### 3.1 Compact table

| candidate | heldout pair_acc | sf top1 | sf local | gsm auc | math auc | gsm f1 | math f1 | gsm terminal_top1 | math terminal_top1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `PBR26` | 0.8532 | 0.8606 | 0.8510 | 0.9148 | 0.8882 | 0.7793 | 0.6700 | 0.1503 | 0.1355 |
| `PBR31_LORA` | 0.8917 | 0.8916 | 0.8799 | 0.9006 | 0.8823 | 0.7882 | 0.6762 | 0.2332 | 0.2192 |

### 3.2 What is actually broken in `PBR26`

`PBR26` really is a strong frozen benchmark model.

But it is not RL-ready.

Why:

1. same-family trust is weaker than the recent docs implied:
   - `sf_top1 = 0.8606`
   - `sf_local_first_bad = 0.8510`
2. strict diagnosis still says:
   - `not_rl_ready_terminal_completion_risk`
3. the main residual is still very specific:
   - `gsm terminal_top1 = 0.1503`
   - `math terminal_top1 = 0.1355`
4. the math failure buckets are explicit:
   - `all_correct terminal_gap = -0.1216`
   - complete correct traces are still ranked too low

So the correct wording is:

- `PBR26` is the strongest verified frozen benchmark candidate in the newest batch,
- but it is not a verified RL-deployable verifier.

### 3.3 What is actually new in `PBR31`

`PBR31` is a real improvement, but not the exact improvement the repo was
starting to imply.

What it **does** improve:

1. same-family trust:
   - `sf_top1: 0.8606 -> 0.8916`
   - `sf_local_first_bad: 0.8510 -> 0.8799`
2. `ProcessBench` F1:
   - `gsm f1: 0.7793 -> 0.7882`
   - `math f1: 0.6700 -> 0.6762`
3. terminal completeness is still bad, but less bad:
   - `gsm terminal_top1: 0.1503 -> 0.2332`
   - `math terminal_top1: 0.1355 -> 0.2192`

What it **does not** improve:

1. it does **not** beat `PBR26` on `Math AUC`
   - `0.8823 < 0.8882`
2. it does **not** pass strict RL-ready
3. it still has the same core diagnosis:
   - `terminal_completion_undervalued`

So the correct wording is:

- `PBR31` is the first verified LoRA line that improves same-family trust and
  oracle-F1 balance over `PBR26`,
- but it is not a new `Math AUC` SOTA and not a verified RL-ready breakthrough.

## 4. RL-Facing Safety Check

### 4.1 `PBR26`

Existing reward-hacking probe:

- `assets/artifacts/phase_f_reward_hacking/phase_f_pbr26_probe_0312_20260311T140931Z`

Interpretation:

- better than older `PBR12` on some GSM attacks,
- but still not "safe enough to stop worrying":
  - `processbench_math first_bad filler_tail` was still `high`
  - `processbench_math first_bad confidence_tail` was still `high`

### 4.2 `PBR31`

New reward-hacking probe:

- `assets/artifacts/phase_f_reward_hacking/phase_f_pbr31_probe_verify_0312_20260311T172022Z`

Interpretation:

- `PBR31` improved same-family and F1,
- but its attack surface on GSM first-bad prefixes is worse than `PBR26`:
  - `confidence_tail flip@0.5 = 0.1875` vs `0.0417`
  - `confidence_tail outrank_safe = 0.1458` vs `0.0833`

So even where `PBR31` is better on static metrics, it is not automatically a
better RL reward model.

## 5. Final Verification Verdict

### 5.1 Verified breakthroughs

Two statements are now verified:

1. `PBR26` is a genuine frozen-backbone benchmark breakthrough inside the current repo.
2. `PBR31` is a genuine LoRA-direction improvement on same-family trust and oracle-F1 balance.

### 5.2 Statements that should be rejected or downgraded

These stronger statements are **not** verified:

1. "`PBR26` is RL-ready"
2. "`PBR31` is a clean post-frozen SOTA breakthrough"
3. "the terminal completion problem is basically solved"

All three are false under the current strict gate.

## 6. What The Repo Should Say Now

The most accurate current summary is:

1. `PBR26`:
   - best verified frozen benchmark candidate
   - not RL-ready
2. `PBR31`:
   - best verified LoRA balance so far
   - not a new Math-AUC SOTA
   - not RL-ready
3. unresolved core problem:
   - terminal completion ordering
   - plus reward-hacking exposure on first-bad prefixes

## 7. Immediate Next Experiments

The next highest-value experiments should be:

1. `PBR31 + calibrated terminal completion repair`
   - do not chase AUC only
   - explicitly measure whether terminal_top1 can move without harming attack surface
2. `PBR26/PBR31 fixed-threshold deployment audit`
   - compare `f1@0.5`, threshold width, and generator shift stability directly
3. `two-stage verifier`
   - frozen / LoRA local scorer
   - separate outcome verifier
   - abstain / route instead of forcing one scalar to do everything
