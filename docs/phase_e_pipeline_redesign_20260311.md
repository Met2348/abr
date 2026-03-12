# Phase E Pipeline Redesign (2026-03-11)

## 2026-03-12 更新：加入 cheap->strong gate 结果后的主线修正

本轮新增一个重要结论：

1. `cheap -> strong verifier` 门控在方向上成立；
2. 但现有 weak verifier 还不够强，无法显著减少 strong verifier 覆盖率。

直接证据：

1. `prm_e46 -> pbr26`
   - 要接近 strong verifier，本仓库需要 `95%-97%` 的 strong usage
2. `ms_e43 -> pbr26`
   - 略好，但仍需 `86%-91%` 的 strong usage

这会直接改变 pipeline 设计：

1. 不应把“cheap verifier + 少量强 verifier 升级”当作当前默认部署路线；
2. 更合理的是先把 verifier 功能拆开：
   - `local/process verifier`
   - `terminal/answer verifier`
   - `abstain / escalate gate`
3. gate 仍然保留，但应当服务于“已拆开的 verifier system”，而不是试图挽救一个单一 scalar verifier。

## Purpose

This note consolidates:
1. recent local experiment evidence,
2. infrastructure audit findings,
3. community / paper-level lessons,
into one practical redesign for the active Phase E pipeline.

## What Is Already Clear

### Good breakthroughs

1. same-source learnability is established on part of the math stack:
   - `Math-Shepherd` and `PRMBench_Preview` can exceed `90%` held-out pair accuracy under the right recipe.
2. catastrophic recipe combinations are now infrastructure-level guarded.
3. the codebase can now separate:
   - source weakness,
   - from wrapper / recipe-induced collapse.

### Remaining bad parts

1. benchmark-facing robustness is still much weaker than same-source held-out scores.
2. mixed-semantics artifacts remain fragile under the wrong objective geometry.
3. some sources are learnable-but-not-yet-trustworthy:
   - high same-source accuracy,
   - but weak or unstable `ProcessBench` behavior.

## Community / Literature Guidance

The current papers point in one consistent direction:

1. `ProcessBench`
   - many PRMs fail on explicit process-error detection.
2. `PRMBench`
   - fine-grained process verification is harder than same-source pair fit.
3. `The Lessons of Developing Process Reward Models`
   - data construction and training objective quality dominate naive scaling.
4. `VersaPRM`
   - broader supervision is a direct response to weak single-source generalization.
5. `R-PRM`
   - stronger pair construction materially changes downstream quality.
6. `ThinkPRM`
   - architecture changes may be needed once small discriminative heads saturate.

## Redesign Principles

### 1. Separate three questions

Do not collapse these into one number:
1. same-source learnability
2. same-family robustness
3. benchmark-facing trustworthiness

### 2. Treat recipe safety as infrastructure

The following are not "bad experiments"; they are infrastructure hazards:
1. unguarded dangerous recipe combinations
2. stale wrappers that silently bypass safety
3. ambiguous pair semantics in step-label derived sources

### 3. Prefer staged diagnosis over blind scaling

Recommended order:
1. source contract audit
2. safe same-source learnability
3. same-family utility tests
4. benchmark utility tests
5. only then larger mixtures or stronger architectures

## Recommended Data-Curation And Training Pipeline

### Stage A: source contract freeze

Per source, record:
1. pair semantics
2. first-bad vs terminal ratio
3. length asymmetry
4. confidence distribution
5. source-family held-out split behavior

### Stage B: safe single-source anchor

For each source:
1. train one safe baseline
2. train one capacity-upgraded baseline
3. select the stronger source-local anchor

### Stage C: same-family utility

Require:
1. pool top-1 improvement
2. local first-bad ordering
3. rejection gain under fixed coverage

### Stage D: controlled mixtures

Only mix sources after Stage B and C are understood.

Priority:
1. DPO-like / fork-point aligned source
2. benchmark-adjacent source
3. weaker auxiliary sources at capped weight

### Stage E: architecture escalation

Only escalate architecture once the source contract is stable.

Order:
1. linear
2. mlp
3. gated_mlp
4. dual-head / generative verifier only if smaller heads saturate

## Immediate Practical Commands

### Static suite audit

```bash
python -u scripts/phase_e_audit_suite_recipes.py \
  --run-name phase_e_suite_recipe_audit_$(date +%m%d_%H%M)
```

### Direct trainer sanity

```bash
python -u scripts/phase_e_train_value.py --help | rg "checkpoint-selection-metric|recipe-risk-policy|train-oom-backoff"
```

## Bottom Line

The next stable Phase E mainline is:
1. explicit safe recipe defaults,
2. explicit pair semantics,
3. same-source anchors,
4. same-family trust,
5. only then carefully weighted mixtures.

That is a stronger foundation than continuing to compare sources under drifting or partially unsafe wrappers.
