"""Recipe-risk and collapse diagnostics for Phase E value-head training.

English
-------
This module hardens Phase E in two places:

1. **Preflight recipe risk**
   Detect known-dangerous combinations before the expensive backbone load starts.
   The goal is not to ban all unusual recipes, but to stop combinations that
   already produced repeated catastrophic failures in this repo.

2. **Post-run collapse diagnosis**
   Convert flat-loss / near-zero-margin / inversion symptoms into a structured
   artifact that later suites and humans can read consistently.

中文
----
这个模块专门负责 Phase E 的两类“基础设施级防呆”：

1. **训练前 recipe 风险检查**
   在大模型加载之前，就把仓库里已经被反复证伪的危险超参组合拦下来。
   目标不是禁止所有奇怪配方，而是阻止那些已经确认会灾难性失败的组合。

2. **训练后 collapse 诊断**
   把“loss 完全不动 / margin 接近 0 / 分数方向反了”这类现象变成结构化产物，
   避免以后继续依赖人工肉眼读日志。
"""

from __future__ import annotations

import statistics
from typing import Any


def assess_phase_e_recipe_risk(
    *,
    train_pair_summary: dict[str, Any],
    train_config: dict[str, Any],
) -> dict[str, Any]:
    """Assess whether one Phase E recipe matches known dangerous patterns.

    English
    -------
    The current risk model is intentionally conservative and repo-specific.
    It only escalates combinations that already failed repeatedly in local
    experiments, especially:

    - mixed semantics,
    - visible terminal-anchor presence,
    - logit-space ranking,
    - semantic/confidence_semantic weighting,
    - ranking_score checkpoint selection.

    中文
    ----
    当前这套规则是“仓库本地经验规则”，不是通用理论定理。
    它只会升级那些已经在本仓库里被多次证伪的组合，特别是：

    - 语义混杂的数据，
    - 明显带 terminal anchor 的 pair 集，
    - logit-space 排序目标，
    - semantic / confidence_semantic 权重，
    - ranking_score 做 checkpoint 选择。
    """

    by_semantics = dict(train_pair_summary.get("by_pair_semantics", {}) or {})
    num_pairs = max(int(train_pair_summary.get("num_pairs", 0)), 1)
    semantics_present = {key: int(value) for key, value in by_semantics.items() if int(value) > 0}
    num_semantics = int(len(semantics_present))
    frac_by_semantics = {
        key: float(int(value) / num_pairs)
        for key, value in sorted(semantics_present.items())
    }
    terminal_fraction = float(frac_by_semantics.get("terminal_completion_anchor", 0.0))
    grid_fraction = float(frac_by_semantics.get("good_bad_prefix_grid", 0.0))
    fanout_fraction = float(frac_by_semantics.get("first_bad_fanout_prefix_ranking", 0.0))
    local_fraction = float(frac_by_semantics.get("local_first_bad_edge", 0.0))
    sibling_fraction = float(frac_by_semantics.get("sibling_branch", 0.0))

    ranking_target_space = str(train_config.get("ranking_target_space", "score"))
    pair_weight_mode = str(train_config.get("pair_weight_mode", "none"))
    checkpoint_selection_metric = str(train_config.get("checkpoint_selection_metric", "ranking_score"))
    objective_mode = str(train_config.get("objective_mode", "ranking_only"))

    semantic_weight_modes = {
        "semantic",
        "confidence_semantic",
        "group_balance",
        "confidence_group_balance",
    }
    uses_semantic_weighting = pair_weight_mode in semantic_weight_modes
    uses_logit_space = ranking_target_space == "logit"
    uses_ranking_score_selection = checkpoint_selection_metric == "ranking_score"
    has_terminal_anchors = terminal_fraction > 0.0
    mixed_semantics = num_semantics > 1
    local_signal_dominant = local_fraction + sibling_fraction >= 0.80

    findings: list[dict[str, Any]] = []
    recommended_fixes: list[str] = []

    def add_finding(*, severity: str, code: str, title: str, detail: str, fix: str | None = None) -> None:
        findings.append(
            {
                "severity": str(severity),
                "code": str(code),
                "title": str(title),
                "detail": str(detail),
            }
        )
        if fix is not None:
            recommended_fixes.append(str(fix))

    if uses_logit_space and uses_semantic_weighting and uses_ranking_score_selection and mixed_semantics:
        add_finding(
            severity="critical",
            code="ANTI_PATTERN_G_FULL",
            title="Known catastrophic recipe combination",
            detail=(
                "logit-space ranking + semantic-style pair weighting + ranking_score selection "
                "on mixed-semantics data previously caused flat loss / near-zero margin collapse."
            ),
            fix="Switch to score-space ranking, pair_weight_mode=none or confidence, and checkpoint_selection_metric=pair_acc.",
        )

    if uses_logit_space and mixed_semantics and has_terminal_anchors:
        add_finding(
            severity="high",
            code="LOGIT_MIXED_TERMINAL",
            title="Logit ranking on mixed terminal/local semantics",
            detail=(
                "This dataset mixes terminal anchors with local process pairs. "
                "Raw-logit ranking tends to overfit magnitude and compress useful local geometry."
            ),
            fix="Prefer score-space ranking for mixed local+terminal artifacts unless there is a strong source-specific reason not to.",
        )

    if uses_ranking_score_selection and mixed_semantics:
        add_finding(
            severity="high",
            code="RANKING_SCORE_MIXED",
            title="ranking_score selection on mixed semantics",
            detail=(
                "ranking_score can overvalue source-internal fit even when held-out pair accuracy or benchmark behavior degrades."
            ),
            fix="Prefer checkpoint_selection_metric=pair_acc or auc for safety-critical comparisons.",
        )

    if uses_semantic_weighting and mixed_semantics and terminal_fraction >= 0.05:
        add_finding(
            severity="high",
            code="SEMANTIC_WEIGHT_MIXED_TERMINAL",
            title="Semantic weighting on mixed local/terminal artifact",
            detail=(
                "semantic/confidence_semantic weighting can amplify the terminal-anchor subset and distort gradients away from local error pairs."
            ),
            fix="Use pair_weight_mode=none first, then re-introduce weighting only after the base recipe is stable.",
        )

    if mixed_semantics and (grid_fraction + fanout_fraction) >= 0.25 and uses_logit_space:
        add_finding(
            severity="medium",
            code="LOGIT_WIDE_GEOMETRY",
            title="Logit ranking on wide-geometry artifact",
            detail=(
                "This artifact contains a large non-local subset (grid/fanout), so a raw-logit objective may prefer shortcut magnitudes over local ordering."
            ),
            fix="Re-test in score space before drawing data-quality conclusions.",
        )

    if local_signal_dominant and not has_terminal_anchors and not uses_semantic_weighting:
        add_finding(
            severity="info",
            code="LOCAL_SIGNAL_CLEAN",
            title="Recipe is aligned with mostly local pair semantics",
            detail=(
                "The artifact is dominated by local/sibling semantics and does not exhibit the usual mixed-terminal risk signature."
            ),
            fix=None,
        )

    severity_rank = {"info": 0, "medium": 1, "high": 2, "critical": 3}
    max_severity = "info"
    if findings:
        max_severity = max(
            findings,
            key=lambda item: severity_rank.get(str(item["severity"]), 0),
        )["severity"]

    return {
        "num_pairs": int(num_pairs),
        "num_semantics": int(num_semantics),
        "by_pair_semantics": semantics_present,
        "frac_by_pair_semantics": frac_by_semantics,
        "objective_mode": str(objective_mode),
        "ranking_target_space": str(ranking_target_space),
        "pair_weight_mode": str(pair_weight_mode),
        "checkpoint_selection_metric": str(checkpoint_selection_metric),
        "mixed_semantics": bool(mixed_semantics),
        "has_terminal_anchors": bool(has_terminal_anchors),
        "terminal_fraction": float(terminal_fraction),
        "uses_semantic_weighting": bool(uses_semantic_weighting),
        "uses_logit_space": bool(uses_logit_space),
        "uses_ranking_score_selection": bool(uses_ranking_score_selection),
        "findings": findings,
        "max_severity": str(max_severity),
        "recommended_fixes": list(dict.fromkeys(recommended_fixes)),
    }


def enforce_phase_e_recipe_risk(
    *,
    recipe_risk_report: dict[str, Any],
    policy: str,
) -> None:
    """Apply one explicit risk policy to a recipe report.

    中文
    ----
    这个函数只负责“是否拦截”：
    - `off`  : 完全忽略
    - `warn` : 只打印，不中断
    - `error`: 只要出现 high/critical 风险就直接失败
    """

    normalized_policy = str(policy).strip().lower()
    if normalized_policy not in {"off", "warn", "error"}:
        raise ValueError("recipe risk policy must be one of: off, warn, error")
    if normalized_policy == "off":
        return
    max_severity = str(recipe_risk_report.get("max_severity", "info"))
    dangerous = max_severity in {"high", "critical"}
    if normalized_policy == "warn":
        return
    if dangerous:
        raise ValueError(_render_recipe_risk_message(recipe_risk_report))


def render_phase_e_recipe_risk_console_report(recipe_risk_report: dict[str, Any]) -> list[str]:
    """Render one compact console report for recipe risk.

    中文
    ----
    这里只返回字符串列表，方便训练脚本决定逐行打印还是写日志文件。
    """

    lines = [
        "recipe_risk       : "
        f"severity={recipe_risk_report.get('max_severity', 'info')} "
        f"mixed={bool(recipe_risk_report.get('mixed_semantics', False))} "
        f"terminal_frac={float(recipe_risk_report.get('terminal_fraction', 0.0)):.4f} "
        f"target={recipe_risk_report.get('ranking_target_space', 'score')} "
        f"weights={recipe_risk_report.get('pair_weight_mode', 'none')} "
        f"selection={recipe_risk_report.get('checkpoint_selection_metric', 'ranking_score')}"
    ]
    for finding in list(recipe_risk_report.get("findings", [])):
        if str(finding.get("severity", "info")) == "info":
            continue
        lines.append(
            "recipe_risk_note  : "
            f"[{finding.get('severity', 'info')}] "
            f"{finding.get('code', 'UNKNOWN')} "
            f"{finding.get('title', '')} | {finding.get('detail', '')}"
        )
    return lines


def diagnose_phase_e_training_health(
    *,
    train_curve: list[dict[str, Any]],
    best_eval_metrics: dict[str, Any],
    chosen_scores: list[float],
    rejected_scores: list[float],
    recipe_risk_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Diagnose whether a Phase E run looks healthy or collapsed.

    English
    -------
    This is intentionally heuristic.  The goal is not to prove convergence in
    a strict optimization sense, but to detect recurring failure modes:

    - flat loss across all epochs,
    - mean margin essentially zero,
    - chosen/rejected score inversion,
    - random-looking held-out pair metrics.

    中文
    ----
    这是启发式诊断，不是严格优化理论证明。
    它主要针对仓库里已经重复出现的失败模式：

    - 多个 epoch 的 loss 基本不动，
    - margin 近乎为 0，
    - chosen/rejected 分数方向反了，
    - held-out pair 指标接近随机。
    """

    epoch_losses = [float(item["train"]["avg_loss"]) for item in train_curve if "train" in item]
    epoch_pair_acc = [float(item["eval"]["pair_accuracy"]) for item in train_curve if "eval" in item]
    epoch_auc = [float(item["eval"]["auc"]) for item in train_curve if "eval" in item]
    mean_chosen = float(statistics.mean(chosen_scores)) if chosen_scores else 0.0
    mean_rejected = float(statistics.mean(rejected_scores)) if rejected_scores else 0.0
    margins = [float(chosen - rejected) for chosen, rejected in zip(chosen_scores, rejected_scores, strict=True)]
    positive_margin_rate = (
        float(sum(1 for margin in margins if margin > 0.0) / len(margins)) if margins else 0.0
    )
    mean_margin = float(best_eval_metrics.get("mean_margin", 0.0))
    pair_accuracy = float(best_eval_metrics.get("pair_accuracy", 0.0))
    auc = float(best_eval_metrics.get("auc", 0.0))

    loss_range = (
        float(max(epoch_losses) - min(epoch_losses)) if len(epoch_losses) >= 2 else 0.0
    )
    loss_delta = (
        float(epoch_losses[-1] - epoch_losses[0]) if len(epoch_losses) >= 2 else 0.0
    )
    loss_std = float(statistics.pstdev(epoch_losses)) if len(epoch_losses) >= 2 else 0.0
    pair_acc_delta = (
        float(epoch_pair_acc[-1] - epoch_pair_acc[0]) if len(epoch_pair_acc) >= 2 else 0.0
    )
    auc_delta = float(epoch_auc[-1] - epoch_auc[0]) if len(epoch_auc) >= 2 else 0.0

    flags = {
        "flat_loss": bool(len(epoch_losses) >= 2 and abs(loss_delta) <= 0.03 and loss_range <= 0.05),
        "tiny_margin": bool(abs(mean_margin) <= 0.01),
        "score_inversion": bool(mean_chosen <= mean_rejected),
        "near_random_pair_metrics": bool(abs(pair_accuracy - 0.5) <= 0.03 and abs(auc - 0.5) <= 0.03),
        "pair_metric_inversion": bool(pair_accuracy < 0.47 or auc < 0.47),
        "no_eval_improvement": bool(pair_acc_delta <= 0.01 and auc_delta <= 0.01),
    }
    known_collapse_signature = bool(
        (flags["flat_loss"] and flags["tiny_margin"])
        or (flags["score_inversion"] and flags["pair_metric_inversion"])
        or (flags["flat_loss"] and flags["near_random_pair_metrics"])
    )

    likely_causes: list[str] = []
    if recipe_risk_report is not None and str(recipe_risk_report.get("max_severity", "info")) in {"high", "critical"}:
        likely_causes.append("known_risky_recipe_combination")
    if flags["flat_loss"]:
        likely_causes.append("loss_flatlined")
    if flags["tiny_margin"]:
        likely_causes.append("margin_geometry_collapsed")
    if flags["score_inversion"]:
        likely_causes.append("chosen_rejected_scores_inverted")
    if flags["near_random_pair_metrics"]:
        likely_causes.append("heldout_pair_metrics_near_random")
    if flags["pair_metric_inversion"]:
        likely_causes.append("heldout_pair_metrics_inverted")
    if flags["no_eval_improvement"]:
        likely_causes.append("training_failed_to_improve_eval")

    diagnosis = "healthy_or_undetermined"
    if known_collapse_signature:
        diagnosis = "collapse_detected"
    elif flags["pair_metric_inversion"] or flags["score_inversion"]:
        diagnosis = "unstable_or_inverted"
    elif flags["near_random_pair_metrics"]:
        diagnosis = "weak_signal"

    return {
        "diagnosis": str(diagnosis),
        "known_collapse_signature": bool(known_collapse_signature),
        "epoch_loss": {
            "values": epoch_losses,
            "delta_last_minus_first": float(loss_delta),
            "range": float(loss_range),
            "std": float(loss_std),
        },
        "eval_progress": {
            "pair_acc_values": epoch_pair_acc,
            "auc_values": epoch_auc,
            "pair_acc_delta_last_minus_first": float(pair_acc_delta),
            "auc_delta_last_minus_first": float(auc_delta),
        },
        "score_distribution": {
            "mean_chosen": float(mean_chosen),
            "mean_rejected": float(mean_rejected),
            "mean_margin": float(mean_margin),
            "positive_margin_rate": float(positive_margin_rate),
        },
        "heldout_metrics": {
            "pair_accuracy": float(pair_accuracy),
            "auc": float(auc),
        },
        "flags": flags,
        "likely_causes": likely_causes,
        "recipe_risk_max_severity": (
            str(recipe_risk_report.get("max_severity", "info")) if recipe_risk_report is not None else None
        ),
    }


def render_phase_e_training_health_markdown(diagnostics: dict[str, Any]) -> str:
    """Render one short Markdown report for training health diagnostics."""
    flags = diagnostics.get("flags", {})
    score_dist = diagnostics.get("score_distribution", {})
    heldout = diagnostics.get("heldout_metrics", {})
    lines = [
        "# Phase E Training Health",
        "",
        f"- diagnosis: `{diagnostics.get('diagnosis', 'unknown')}`",
        f"- known_collapse_signature: `{bool(diagnostics.get('known_collapse_signature', False))}`",
        f"- likely_causes: `{', '.join(diagnostics.get('likely_causes', [])) or 'none'}`",
        "",
        "## Held-Out Snapshot",
        "",
        f"- pair_accuracy: `{float(heldout.get('pair_accuracy', 0.0)):.6f}`",
        f"- auc: `{float(heldout.get('auc', 0.0)):.6f}`",
        f"- mean_chosen: `{float(score_dist.get('mean_chosen', 0.0)):.6f}`",
        f"- mean_rejected: `{float(score_dist.get('mean_rejected', 0.0)):.6f}`",
        f"- mean_margin: `{float(score_dist.get('mean_margin', 0.0)):.6f}`",
        f"- positive_margin_rate: `{float(score_dist.get('positive_margin_rate', 0.0)):.6f}`",
        "",
        "## Flags",
        "",
    ]
    for key, value in flags.items():
        lines.append(f"- {key}: `{bool(value)}`")
    lines.append("")
    return "\n".join(lines)


def _render_recipe_risk_message(recipe_risk_report: dict[str, Any]) -> str:
    """Render one concise exception message for dangerous recipes."""
    findings = list(recipe_risk_report.get("findings", []))
    dangerous = [
        item for item in findings if str(item.get("severity", "info")) in {"high", "critical"}
    ]
    detail = "; ".join(
        f"{item.get('code', 'UNKNOWN')}: {item.get('title', '')}"
        for item in dangerous
    )
    fixes = ", ".join(recipe_risk_report.get("recommended_fixes", [])[:3])
    return (
        "Phase E recipe risk rejected by policy. "
        f"max_severity={recipe_risk_report.get('max_severity', 'info')} | "
        f"details={detail or 'none'} | "
        f"recommended_fixes={fixes or 'none'}"
    )
