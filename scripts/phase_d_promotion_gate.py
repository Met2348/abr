#!/usr/bin/env python3
"""Evaluate Phase D promotion readiness from existing suite artifacts.

Why this file exists
--------------------
Phase D decisions currently depend on multiple logs:
1. Phase C reference controls (`phase_c_logs/*/final_summary.md`)
2. D6 control-vs-PRM-gate teacher suites (`phase_d_logs/*/final_summary.md`)
3. D6-T seed-stability suites (`phase_d6t_logs/*/seed_results.jsonl`)

Reading these files manually is slow and inconsistent. This script provides a
single promotion-gate judgment with explicit pass/fail reasons.

What this file does
-------------------
1. Resolve one comparable C2 reference row.
2. Resolve latest D6 control and D6 gated runs, then compare uplift.
3. Resolve required D6-T groups and compute seed mean/std checks.
4. Emit machine-readable and markdown reports with decision and interpretation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalSnapshot:
    """Compact metrics view used by gate checks."""

    brier: float | None
    posthoc_brier: float | None
    pair_acc: float | None
    auc: float | None
    pearson: float | None
    brier_improvement_vs_baseline: float | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate Phase D promotion gates and write a concise decision report."
    )
    parser.add_argument(
        "--phase-c-logs-root",
        type=Path,
        default=Path("assets/artifacts/phase_c_logs"),
        help="Root path of Phase C suite logs.",
    )
    parser.add_argument(
        "--phase-d-logs-root",
        type=Path,
        default=Path("assets/artifacts/phase_d_logs"),
        help="Root path of Phase D teacher/external suite logs.",
    )
    parser.add_argument(
        "--phase-d6t-logs-root",
        type=Path,
        default=Path("assets/artifacts/phase_d6t_logs"),
        help="Root path of D6-T suite logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write gate_report.json and summary.md.",
    )
    parser.add_argument(
        "--dataset",
        default="strategyqa",
        help="Dataset name for C reference filtering. Default: strategyqa.",
    )
    parser.add_argument(
        "--d6-control-group-id",
        default="D6_STRATEGYQA_SMOKE_RANKING_CTRL",
        help="Group ID for D6 control arm.",
    )
    parser.add_argument(
        "--d6-gated-group-id",
        default="D6_STRATEGYQA_SMOKE_RANKING_PRM_GATE",
        help="Group ID for D6 PRM-gated arm.",
    )
    parser.add_argument(
        "--required-d6t-groups",
        default="DT2_MATH_SHEPHERD_SEED3,DT4_MIXED_MS_PRM800K_SEED3",
        help=(
            "Comma-separated required D6-T groups. "
            "Promotion requires every listed group to pass."
        ),
    )
    parser.add_argument(
        "--min-corr-pair-acc",
        type=float,
        default=0.55,
        help="Minimum required corr_pair_acc for D6 gated run.",
    )
    parser.add_argument(
        "--min-corr-auc",
        type=float,
        default=0.60,
        help="Minimum required corr_auc for D6 gated run.",
    )
    parser.add_argument(
        "--min-d6t-seeds",
        type=int,
        default=3,
        help="Minimum seed count required for each D6-T required group.",
    )
    parser.add_argument(
        "--d6t-mean-pair-acc-min",
        type=float,
        default=0.65,
        help="Minimum required mean external pair_acc for each D6-T required group.",
    )
    parser.add_argument(
        "--d6t-mean-auc-min",
        type=float,
        default=0.65,
        help="Minimum required mean external auc for each D6-T required group.",
    )
    parser.add_argument(
        "--d6t-std-max",
        type=float,
        default=0.05,
        help="Maximum allowed std for pair_acc/auc in each D6-T required group.",
    )
    parser.add_argument(
        "--trivial-brier-baseline",
        type=float,
        default=0.1394,
        help=(
            "Fallback trivial baseline for Brier when brier_improvement_vs_baseline "
            "is unavailable."
        ),
    )
    parser.add_argument(
        "--require-positive-brier-improvement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prefer using calibration.brier_improvement_vs_baseline > 0 as calibration "
            "gate when available."
        ),
    )
    parser.add_argument(
        "--max-brier-regression-vs-control",
        type=float,
        default=0.03,
        help=(
            "Maximum allowed raw Brier regression (gated - control). "
            "Set negative to disable this check."
        ),
    )
    parser.add_argument(
        "--require-c-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require a resolved C reference, otherwise mark decision as blocked.",
    )
    return parser.parse_args(argv)


def _safe_float(value: Any) -> float | None:
    """Convert one value to float safely."""

    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _clean_md_value(value: str) -> str:
    """Normalize markdown bullet/table values."""

    out = value.strip()
    if out.startswith("`") and out.endswith("`") and len(out) >= 2:
        out = out[1:-1]
    return out.strip()


def _extract_bullets(md_text: str) -> dict[str, str]:
    """Extract `- key: value` pairs from markdown text."""

    out: dict[str, str] = {}
    pattern = re.compile(r"^- ([^:]+):\s*(.+)$", flags=re.MULTILINE)
    for match in pattern.finditer(md_text):
        key = match.group(1).strip()
        val = _clean_md_value(match.group(2))
        out[key] = val
    return out


def _is_md_separator_row(line: str) -> bool:
    """Return True if one markdown table row is a separator row."""

    s = line.strip().strip("|").strip()
    if not s:
        return False
    cells = [cell.strip() for cell in s.split("|")]
    if not cells:
        return False
    return all(cell and set(cell) <= {"-", ":"} for cell in cells)


def _parse_markdown_table(md_text: str, header_prefix: str) -> list[dict[str, str]]:
    """Parse one markdown table identified by header prefix."""

    lines = md_text.splitlines()
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line.startswith("|"):
            continue
        if not line.lower().startswith(header_prefix.lower()):
            continue
        header = [cell.strip() for cell in line.strip("|").split("|")]
        rows: list[dict[str, str]] = []
        for body in lines[idx + 1 :]:
            text = body.strip()
            if not text.startswith("|"):
                break
            if _is_md_separator_row(text):
                continue
            cells = [cell.strip() for cell in text.strip("|").split("|")]
            if len(cells) != len(header):
                continue
            rows.append({header[i]: _clean_md_value(cells[i]) for i in range(len(header))})
        return rows
    return []


def _parse_iso_ts(value: str | None) -> datetime:
    """Parse one ISO timestamp safely."""

    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    text = value.strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _load_eval_snapshot(metrics_path: Path) -> EvalSnapshot:
    """Load eval metrics into one compact snapshot."""

    if not metrics_path.exists():
        return EvalSnapshot(
            brier=None,
            posthoc_brier=None,
            pair_acc=None,
            auc=None,
            pearson=None,
            brier_improvement_vs_baseline=None,
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    cal = payload.get("calibration") or {}
    post = payload.get("calibration_posthoc") or {}
    corr = payload.get("corruption") or {}
    return EvalSnapshot(
        brier=_safe_float(cal.get("brier_score")),
        posthoc_brier=_safe_float(post.get("brier_score")),
        pair_acc=_safe_float(corr.get("pair_accuracy")),
        auc=_safe_float(corr.get("auc_clean_vs_corrupt")),
        pearson=_safe_float(cal.get("pearson")),
        brier_improvement_vs_baseline=_safe_float(cal.get("brier_improvement_vs_baseline")),
    )


def _iter_summary_dirs(root: Path) -> list[tuple[Path, str, dict[str, str], datetime]]:
    """List summary markdown contexts under one log root."""

    if not root.exists():
        return []
    rows: list[tuple[Path, str, dict[str, str], datetime]] = []
    for log_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        summary = log_dir / "final_summary.md"
        if not summary.exists():
            continue
        md_text = summary.read_text(encoding="utf-8")
        bullets = _extract_bullets(md_text)
        ts = _parse_iso_ts(bullets.get("generated_at"))
        rows.append((log_dir, md_text, bullets, ts))
    return rows


def _pick_latest(entries: list[tuple[Path, str, dict[str, str], datetime]]) -> tuple[Path, str, dict[str, str], datetime] | None:
    """Pick the latest summary context by generated timestamp."""

    if not entries:
        return None
    return sorted(entries, key=lambda item: (item[3].timestamp(), item[0].name))[-1]


def _resolve_c_reference(phase_c_logs_root: Path, dataset: str) -> dict[str, Any] | None:
    """Resolve one comparable C2 reference row."""

    preferred = [
        "C2_STRATEGYQA_CQR_FULL",
        "C2_STRATEGYQA_QUALITY_FIRST_FULL",
        "C2_STRATEGYQA_CQR_SMOKE",
    ]
    candidates: list[dict[str, Any]] = []
    for log_dir, _, bullets, ts in _iter_summary_dirs(phase_c_logs_root):
        group_id = str(bullets.get("group_id", ""))
        if not group_id.startswith("C2_"):
            continue
        ds = str(bullets.get("dataset", "")).strip().lower()
        if ds and ds != dataset.strip().lower():
            continue
        c2_eval_dir = bullets.get("c2_eval_dir")
        if not c2_eval_dir:
            continue
        metrics_path = Path(c2_eval_dir) / "metrics.json"
        snap = _load_eval_snapshot(metrics_path)
        candidates.append(
            {
                "group_id": group_id,
                "run_prefix": bullets.get("run_prefix", log_dir.name),
                "log_dir": str(log_dir),
                "generated_at": ts.isoformat(),
                "c2_eval_dir": c2_eval_dir,
                "metrics_path": str(metrics_path),
                "metrics": {
                    "brier": snap.brier,
                    "posthoc_brier": snap.posthoc_brier,
                    "pair_acc": snap.pair_acc,
                    "auc": snap.auc,
                    "pearson": snap.pearson,
                    "brier_improvement_vs_baseline": snap.brier_improvement_vs_baseline,
                },
            }
        )
    if not candidates:
        return None

    def _priority(item: dict[str, Any]) -> int:
        gid = str(item.get("group_id", ""))
        if gid in preferred:
            return preferred.index(gid)
        return len(preferred)

    # 中文：这里不是简单取“最新 C2 结果”，而是优先取方法学上更可比的 reference。
    # 否则 promotion gate 可能会把 smoke / 旧 ablation 当 baseline，导致判断失真。
    best_priority = min(_priority(item) for item in candidates)
    best = [item for item in candidates if _priority(item) == best_priority]
    return sorted(
        best,
        key=lambda item: _parse_iso_ts(str(item.get("generated_at", ""))).timestamp(),
    )[-1]


def _resolve_d6_group(phase_d_logs_root: Path, group_id: str) -> dict[str, Any] | None:
    """Resolve one latest D6 group row from teacher-suite summaries."""

    matched: list[tuple[Path, str, dict[str, str], datetime]] = []
    for ctx in _iter_summary_dirs(phase_d_logs_root):
        _, _, bullets, _ = ctx
        if str(bullets.get("group_id", "")) == group_id:
            matched.append(ctx)
    latest = _pick_latest(matched)
    if latest is None:
        return None
    log_dir, md_text, bullets, ts = latest
    rows = _parse_markdown_table(md_text, header_prefix="| label | target_source |")
    if not rows:
        return {
            "group_id": group_id,
            "run_prefix": bullets.get("run_prefix", log_dir.name),
            "log_dir": str(log_dir),
            "generated_at": ts.isoformat(),
            "status": "missing_table",
        }
    picked = rows[0]
    for row in rows:
        label = str(row.get("label", "")).strip().lower()
        target_source = str(row.get("target_source", "")).strip().lower()
        if label == "mc" or target_source == "q_mean_smoothed":
            # 中文：优先选 MC / q_mean_smoothed 这一行，是为了跟 C 阶段原始监督更可比。
            # 若直接拿 fused/teacher 行去和 control 比，往往会把“监督源变化”与
            # “模型能力变化”混在一起。
            picked = row
            break
    c2_eval_dir = picked.get("c2_eval_dir")
    if not c2_eval_dir:
        return {
            "group_id": group_id,
            "run_prefix": bullets.get("run_prefix", log_dir.name),
            "log_dir": str(log_dir),
            "generated_at": ts.isoformat(),
            "status": "missing_c2_eval_dir",
        }
    metrics_path = Path(c2_eval_dir) / "metrics.json"
    snap = _load_eval_snapshot(metrics_path)
    return {
        "group_id": group_id,
        "run_prefix": bullets.get("run_prefix", log_dir.name),
        "log_dir": str(log_dir),
        "generated_at": ts.isoformat(),
        "label": picked.get("label"),
        "target_source": picked.get("target_source"),
        "c2_eval_dir": c2_eval_dir,
        "metrics_path": str(metrics_path),
        "metrics": {
            "brier": snap.brier,
            "posthoc_brier": snap.posthoc_brier,
            "pair_acc": snap.pair_acc,
            "auc": snap.auc,
            "pearson": snap.pearson,
            "brier_improvement_vs_baseline": snap.brier_improvement_vs_baseline,
        },
        "status": "ok",
    }


def _load_seed_rows(seed_results_path: Path) -> list[dict[str, Any]]:
    """Load D6-T seed rows from seed_results.jsonl."""

    if not seed_results_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in seed_results_path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _resolve_d6t_group(phase_d6t_logs_root: Path, group_id: str) -> dict[str, Any] | None:
    """Resolve one latest D6-T group summary and aggregated seed statistics."""

    matched: list[tuple[Path, str, dict[str, str], datetime]] = []
    for ctx in _iter_summary_dirs(phase_d6t_logs_root):
        _, _, bullets, _ = ctx
        if str(bullets.get("group_id", "")) == group_id:
            matched.append(ctx)
    latest = _pick_latest(matched)
    if latest is None:
        return None
    log_dir, _, bullets, ts = latest
    seed_rows = _load_seed_rows(log_dir / "seed_results.jsonl")
    pair_values = [_safe_float(row.get("ext_pair_acc")) for row in seed_rows]
    auc_values = [_safe_float(row.get("ext_auc")) for row in seed_rows]
    pair_values = [v for v in pair_values if v is not None]
    auc_values = [v for v in auc_values if v is not None]
    # 中文：这里用跨 seed 的均值和总体标准差做门控，而不是只看单次最好结果。
    # D6-T 的核心目标就是检验“稳定可复现”，所以最怕 cherry-pick 单个高分 seed。
    pair_mean = statistics.mean(pair_values) if pair_values else None
    auc_mean = statistics.mean(auc_values) if auc_values else None
    pair_std = statistics.pstdev(pair_values) if len(pair_values) > 1 else 0.0 if pair_values else None
    auc_std = statistics.pstdev(auc_values) if len(auc_values) > 1 else 0.0 if auc_values else None
    gate_pass_raw = str(bullets.get("gate_pass", "")).strip().lower()
    gate_pass = gate_pass_raw == "true"
    return {
        "group_id": group_id,
        "run_prefix": bullets.get("run_prefix", log_dir.name),
        "log_dir": str(log_dir),
        "generated_at": ts.isoformat(),
        "seed_results_path": str(log_dir / "seed_results.jsonl"),
        "seed_count": len(seed_rows),
        "mean_pair_acc": pair_mean,
        "mean_auc": auc_mean,
        "std_pair_acc": pair_std,
        "std_auc": auc_std,
        "suite_gate_pass": gate_pass if gate_pass_raw in {"true", "false"} else None,
        "status": "ok",
    }


def _make_check(name: str, passed: bool, detail: str, reason_code: str | None = None) -> dict[str, Any]:
    """Build one normalized check row."""

    return {
        "name": name,
        "pass": bool(passed),
        "detail": str(detail),
        "reason_code": reason_code,
    }


def _evaluate_d6(
    *,
    control: dict[str, Any] | None,
    gated: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate D6 control-vs-gated checks."""

    checks: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []

    if control is None:
        checks.append(
            _make_check(
                name="d6_control_exists",
                passed=False,
                detail=f"Missing D6 control group: {args.d6_control_group_id}",
                reason_code="missing_d6_control_group",
            )
        )
        blocking_reasons.append("missing_d6_control_group")
    else:
        checks.append(
            _make_check(
                name="d6_control_exists",
                passed=True,
                detail=f"Resolved D6 control run_prefix={control.get('run_prefix')}",
            )
        )
    if gated is None:
        checks.append(
            _make_check(
                name="d6_gated_exists",
                passed=False,
                detail=f"Missing D6 gated group: {args.d6_gated_group_id}",
                reason_code="missing_d6_gated_group",
            )
        )
        blocking_reasons.append("missing_d6_gated_group")
    else:
        checks.append(
            _make_check(
                name="d6_gated_exists",
                passed=True,
                detail=f"Resolved D6 gated run_prefix={gated.get('run_prefix')}",
            )
        )

    if control is None or gated is None:
        return {
            "pass": False,
            "checks": checks,
            "blocking_reasons": blocking_reasons,
        }

    cm = control.get("metrics", {}) if isinstance(control, dict) else {}
    gm = gated.get("metrics", {}) if isinstance(gated, dict) else {}
    c_pair = _safe_float(cm.get("pair_acc"))
    g_pair = _safe_float(gm.get("pair_acc"))
    c_auc = _safe_float(cm.get("auc"))
    g_auc = _safe_float(gm.get("auc"))
    c_brier = _safe_float(cm.get("brier"))
    g_brier = _safe_float(gm.get("brier"))
    g_brier_imp = _safe_float(gm.get("brier_improvement_vs_baseline"))

    pair_available = c_pair is not None and g_pair is not None
    auc_available = c_auc is not None and g_auc is not None
    checks.append(
        _make_check(
            name="d6_pair_metrics_available",
            passed=pair_available and auc_available,
            detail=(
                f"control_pair={c_pair}, gated_pair={g_pair}, "
                f"control_auc={c_auc}, gated_auc={g_auc}"
            ),
            reason_code=("missing_d6_pair_metrics" if not (pair_available and auc_available) else None),
        )
    )
    if not (pair_available and auc_available):
        blocking_reasons.append("missing_d6_pair_metrics")
        return {
            "pass": False,
            "checks": checks,
            "blocking_reasons": blocking_reasons,
        }

    pair_uplift = g_pair > c_pair
    auc_uplift = g_auc > c_auc
    checks.append(
        _make_check(
            name="d6_uplift_pair_acc",
            passed=pair_uplift,
            detail=f"gated_pair_acc - control_pair_acc = {g_pair - c_pair:.6f}",
            reason_code=("d6_no_pair_uplift" if not pair_uplift else None),
        )
    )
    checks.append(
        _make_check(
            name="d6_uplift_auc",
            passed=auc_uplift,
            detail=f"gated_auc - control_auc = {g_auc - c_auc:.6f}",
            reason_code=("d6_no_auc_uplift" if not auc_uplift else None),
        )
    )
    if not pair_uplift:
        blocking_reasons.append("d6_no_pair_uplift")
    if not auc_uplift:
        blocking_reasons.append("d6_no_auc_uplift")

    pair_abs_ok = g_pair >= float(args.min_corr_pair_acc)
    auc_abs_ok = g_auc >= float(args.min_corr_auc)
    checks.append(
        _make_check(
            name="d6_abs_pair_acc",
            passed=pair_abs_ok,
            detail=f"gated_pair_acc={g_pair:.6f}, threshold={float(args.min_corr_pair_acc):.6f}",
            reason_code=("d6_pair_acc_below_threshold" if not pair_abs_ok else None),
        )
    )
    checks.append(
        _make_check(
            name="d6_abs_auc",
            passed=auc_abs_ok,
            detail=f"gated_auc={g_auc:.6f}, threshold={float(args.min_corr_auc):.6f}",
            reason_code=("d6_auc_below_threshold" if not auc_abs_ok else None),
        )
    )
    if not pair_abs_ok:
        blocking_reasons.append("d6_pair_acc_below_threshold")
    if not auc_abs_ok:
        blocking_reasons.append("d6_auc_below_threshold")

    if bool(args.require_positive_brier_improvement) and g_brier_imp is not None:
        cal_ok = g_brier_imp > 0.0
        detail = f"gated_brier_improvement_vs_baseline={g_brier_imp:.6f} (>0 required)"
        reason = "d6_calibration_not_beating_baseline" if not cal_ok else None
    else:
        cal_ok = (g_brier is not None) and (g_brier <= float(args.trivial_brier_baseline))
        detail = (
            f"gated_brier={g_brier}, "
            f"trivial_baseline={float(args.trivial_brier_baseline):.6f}"
        )
        reason = "d6_calibration_not_beating_trivial_brier" if not cal_ok else None
    checks.append(
        _make_check(
            name="d6_calibration_gate",
            passed=cal_ok,
            detail=detail,
            reason_code=reason,
        )
    )
    if not cal_ok:
        blocking_reasons.append(reason if reason is not None else "d6_calibration_gate_failed")

    max_reg = float(args.max_brier_regression_vs_control)
    if max_reg >= 0.0 and c_brier is not None and g_brier is not None:
        brier_delta = g_brier - c_brier
        brier_robust = brier_delta <= max_reg
        checks.append(
            _make_check(
                name="d6_calibration_robustness_vs_control",
                passed=brier_robust,
                detail=f"gated_brier - control_brier = {brier_delta:.6f}, max={max_reg:.6f}",
                reason_code=("d6_brier_regression_too_large" if not brier_robust else None),
            )
        )
        if not brier_robust:
            blocking_reasons.append("d6_brier_regression_too_large")

    return {
        "pass": len(blocking_reasons) == 0,
        "checks": checks,
        "blocking_reasons": blocking_reasons,
    }


def _evaluate_d6t(
    *,
    group_results: dict[str, dict[str, Any] | None],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate D6-T seed-stability checks for required groups."""

    checks: list[dict[str, Any]] = []
    blocking_reasons: list[str] = []
    per_group: dict[str, Any] = {}

    for group_id, result in group_results.items():
        group_checks: list[dict[str, Any]] = []
        group_blocking: list[str] = []
        if result is None:
            reason = f"missing_d6t_group:{group_id}"
            group_checks.append(
                _make_check(
                    name=f"{group_id}.exists",
                    passed=False,
                    detail=f"Missing required D6-T group summary: {group_id}",
                    reason_code=reason,
                )
            )
            group_blocking.append(reason)
            per_group[group_id] = {
                "pass": False,
                "checks": group_checks,
                "blocking_reasons": group_blocking,
            }
            checks.extend(group_checks)
            blocking_reasons.extend(group_blocking)
            continue

        seed_count = int(result.get("seed_count") or 0)
        mean_pair = _safe_float(result.get("mean_pair_acc"))
        mean_auc = _safe_float(result.get("mean_auc"))
        std_pair = _safe_float(result.get("std_pair_acc"))
        std_auc = _safe_float(result.get("std_auc"))

        enough_seeds = seed_count >= int(args.min_d6t_seeds)
        group_checks.append(
            _make_check(
                name=f"{group_id}.seed_count",
                passed=enough_seeds,
                detail=f"seed_count={seed_count}, min_required={int(args.min_d6t_seeds)}",
                reason_code=(f"d6t_seed_count_low:{group_id}" if not enough_seeds else None),
            )
        )
        if not enough_seeds:
            group_blocking.append(f"d6t_seed_count_low:{group_id}")

        pair_ok = mean_pair is not None and mean_pair >= float(args.d6t_mean_pair_acc_min)
        group_checks.append(
            _make_check(
                name=f"{group_id}.mean_pair_acc",
                passed=pair_ok,
                detail=f"mean_pair_acc={mean_pair}, threshold={float(args.d6t_mean_pair_acc_min):.6f}",
                reason_code=(f"d6t_mean_pair_fail:{group_id}" if not pair_ok else None),
            )
        )
        if not pair_ok:
            group_blocking.append(f"d6t_mean_pair_fail:{group_id}")

        auc_ok = mean_auc is not None and mean_auc >= float(args.d6t_mean_auc_min)
        group_checks.append(
            _make_check(
                name=f"{group_id}.mean_auc",
                passed=auc_ok,
                detail=f"mean_auc={mean_auc}, threshold={float(args.d6t_mean_auc_min):.6f}",
                reason_code=(f"d6t_mean_auc_fail:{group_id}" if not auc_ok else None),
            )
        )
        if not auc_ok:
            group_blocking.append(f"d6t_mean_auc_fail:{group_id}")

        std_pair_ok = std_pair is not None and std_pair <= float(args.d6t_std_max)
        std_auc_ok = std_auc is not None and std_auc <= float(args.d6t_std_max)
        group_checks.append(
            _make_check(
                name=f"{group_id}.std_pair_acc",
                passed=std_pair_ok,
                detail=f"std_pair_acc={std_pair}, threshold={float(args.d6t_std_max):.6f}",
                reason_code=(f"d6t_std_pair_fail:{group_id}" if not std_pair_ok else None),
            )
        )
        group_checks.append(
            _make_check(
                name=f"{group_id}.std_auc",
                passed=std_auc_ok,
                detail=f"std_auc={std_auc}, threshold={float(args.d6t_std_max):.6f}",
                reason_code=(f"d6t_std_auc_fail:{group_id}" if not std_auc_ok else None),
            )
        )
        if not std_pair_ok:
            group_blocking.append(f"d6t_std_pair_fail:{group_id}")
        if not std_auc_ok:
            group_blocking.append(f"d6t_std_auc_fail:{group_id}")

        group_pass = len(group_blocking) == 0
        per_group[group_id] = {
            "pass": group_pass,
            "checks": group_checks,
            "blocking_reasons": group_blocking,
            "summary": result,
        }
        checks.extend(group_checks)
        blocking_reasons.extend(group_blocking)

    return {
        "pass": len(blocking_reasons) == 0,
        "checks": checks,
        "blocking_reasons": blocking_reasons,
        "groups": per_group,
    }


def _interpretation_lines(decision: dict[str, Any]) -> list[str]:
    """Generate short interpretation lines for common failure patterns."""

    lines: list[str] = []
    if bool(decision.get("promotion_ready")):
        lines.append("All configured gates passed; promotion to the next stage is allowed.")
        lines.append(
            "Keep utility-gate verification under equal budget before final external claims."
        )
        return lines

    reasons = [str(item) for item in decision.get("blocking_reasons", [])]
    if any(reason.startswith("missing_") for reason in reasons):
        lines.append(
            "Blocking artifacts are missing. Re-run missing suites first; current state is not decision-complete."
        )
    if "d6_no_pair_uplift" in reasons or "d6_no_auc_uplift" in reasons:
        lines.append(
            "D6 PRM-gated arm did not beat control on ranking uplift; pair gating policy is not yet effective."
        )
    if "d6_pair_acc_below_threshold" in reasons or "d6_auc_below_threshold" in reasons:
        lines.append(
            "D6 gated run remains below absolute ranking thresholds; current supervision is still insufficient."
        )
    if any(reason.startswith("d6t_std_") for reason in reasons):
        lines.append(
            "D6-T shows seed instability (seed-collapse risk). Prioritize variance triage before architecture changes."
        )
    if any(reason.startswith("d6t_mean_") for reason in reasons):
        lines.append(
            "D6-T mean ranking quality is below gate; pair construction quality should be tightened first."
        )
    if any(reason.startswith("d6_calibration_") or reason == "d6_brier_regression_too_large" for reason in reasons):
        lines.append(
            "Calibration guardrail failed. Avoid promotion claims based only on ranking-side gains."
        )
    if not lines:
        lines.append("Decision is blocked; inspect check details for the first failing condition.")
    return lines


def _render_markdown(report: dict[str, Any]) -> str:
    """Render one human-readable markdown summary."""

    lines: list[str] = []
    lines.append("# Phase D Promotion Gate Report")
    lines.append("")
    lines.append(f"- generated_at: {report.get('generated_at')}")
    lines.append(f"- promotion_ready: {report.get('decision', {}).get('promotion_ready')}")
    lines.append("")

    cfg = report.get("config", {})
    lines.append("## Config")
    lines.append("")
    for key in sorted(cfg.keys()):
        lines.append(f"- {key}: `{cfg[key]}`")
    lines.append("")

    lines.append("## C Reference")
    lines.append("")
    c_ref = report.get("c_reference")
    if c_ref is None:
        lines.append("- not found")
    else:
        lines.append(f"- group_id: `{c_ref.get('group_id')}`")
        lines.append(f"- run_prefix: `{c_ref.get('run_prefix')}`")
        lines.append(f"- c2_eval_dir: `{c_ref.get('c2_eval_dir')}`")
        metrics = c_ref.get("metrics", {})
        lines.append(f"- brier: `{metrics.get('brier')}`")
        lines.append(f"- pair_acc: `{metrics.get('pair_acc')}`")
        lines.append(f"- auc: `{metrics.get('auc')}`")
    lines.append("")

    lines.append("## D6 Checks")
    lines.append("")
    lines.append("| check | pass | detail | reason_code |")
    lines.append("|---|---|---|---|")
    for check in report.get("d6", {}).get("checks", []):
        lines.append(
            f"| {check.get('name')} | {check.get('pass')} | {check.get('detail')} | {check.get('reason_code') or ''} |"
        )
    lines.append("")

    lines.append("## D6-T Checks")
    lines.append("")
    lines.append("| check | pass | detail | reason_code |")
    lines.append("|---|---|---|---|")
    for check in report.get("d6t", {}).get("checks", []):
        lines.append(
            f"| {check.get('name')} | {check.get('pass')} | {check.get('detail')} | {check.get('reason_code') or ''} |"
        )
    lines.append("")

    lines.append("## Decision")
    lines.append("")
    decision = report.get("decision", {})
    lines.append(f"- promotion_ready: `{decision.get('promotion_ready')}`")
    lines.append(f"- blocking_reasons: `{decision.get('blocking_reasons')}`")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    for line in decision.get("interpretation", []):
        lines.append(f"- {line}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run gate evaluation and write reports."""

    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required_d6t_groups = [
        item.strip()
        for item in str(args.required_d6t_groups).split(",")
        if item.strip()
    ]

    c_ref = _resolve_c_reference(args.phase_c_logs_root, dataset=str(args.dataset))
    d6_control = _resolve_d6_group(args.phase_d_logs_root, group_id=str(args.d6_control_group_id))
    d6_gated = _resolve_d6_group(args.phase_d_logs_root, group_id=str(args.d6_gated_group_id))
    d6 = _evaluate_d6(control=d6_control, gated=d6_gated, args=args)

    d6t_groups: dict[str, dict[str, Any] | None] = {}
    for group_id in required_d6t_groups:
        d6t_groups[group_id] = _resolve_d6t_group(args.phase_d6t_logs_root, group_id=group_id)
    d6t = _evaluate_d6t(group_results=d6t_groups, args=args)

    blocking_reasons: list[str] = []
    if bool(args.require_c_reference) and c_ref is None:
        blocking_reasons.append("missing_c_reference")
    blocking_reasons.extend(d6.get("blocking_reasons", []))
    blocking_reasons.extend(d6t.get("blocking_reasons", []))

    decision = {
        "promotion_ready": len(blocking_reasons) == 0,
        "blocking_reasons": blocking_reasons,
    }
    decision["interpretation"] = _interpretation_lines(decision)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "phase_c_logs_root": str(args.phase_c_logs_root),
            "phase_d_logs_root": str(args.phase_d_logs_root),
            "phase_d6t_logs_root": str(args.phase_d6t_logs_root),
            "dataset": str(args.dataset),
            "d6_control_group_id": str(args.d6_control_group_id),
            "d6_gated_group_id": str(args.d6_gated_group_id),
            "required_d6t_groups": required_d6t_groups,
            "min_corr_pair_acc": float(args.min_corr_pair_acc),
            "min_corr_auc": float(args.min_corr_auc),
            "min_d6t_seeds": int(args.min_d6t_seeds),
            "d6t_mean_pair_acc_min": float(args.d6t_mean_pair_acc_min),
            "d6t_mean_auc_min": float(args.d6t_mean_auc_min),
            "d6t_std_max": float(args.d6t_std_max),
            "trivial_brier_baseline": float(args.trivial_brier_baseline),
            "require_positive_brier_improvement": bool(args.require_positive_brier_improvement),
            "max_brier_regression_vs_control": float(args.max_brier_regression_vs_control),
            "require_c_reference": bool(args.require_c_reference),
        },
        "c_reference": c_ref,
        "d6_control": d6_control,
        "d6_gated": d6_gated,
        "d6": d6,
        "d6t": d6t,
        "decision": decision,
    }

    json_path = args.output_dir / "gate_report.json"
    md_path = args.output_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report) + "\n", encoding="utf-8")

    print("=" * 88)
    print("Phase D Promotion Gate")
    print("=" * 88)
    print(f"promotion_ready : {decision['promotion_ready']}")
    print(f"blocking_reasons: {decision['blocking_reasons']}")
    print(f"report_json     : {json_path}")
    print(f"report_md       : {md_path}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
