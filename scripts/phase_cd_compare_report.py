#!/usr/bin/env python
"""Build a cross-stage diagnostic report for Phase C vs Phase D experiments.

Why this file exists:
- We currently have multiple Phase D pipelines (teacher 3-way bundle, D4ABC
  external-pair chain) plus many historical C2 control runs.
- Manually reading dozens of `final_summary.md` files is slow and error-prone.
- This script centralizes result loading and emits a single report with:
  1) C baseline candidates,
  2) all D-method results,
  3) D-vs-C comparable deltas,
  4) failure diagnostics.

What this script does:
1. Scan `assets/artifacts/phase_c_logs/*` and `assets/artifacts/phase_d_logs/*`.
2. Parse suite summaries and locate referenced `metrics.json`.
3. Build normalized records for:
   - C2 suite runs,
   - D teacher 3-way rows (`mc`, `teacher`, `fused`),
   - D4 external-pair stages (`D4A`, `D4B`, `D4C`).
4. Select a comparable C reference baseline (priority: `C2_STRATEGYQA_CQR_FULL`).
5. Write machine-readable + human-readable artifacts:
   - `records.jsonl`
   - `leaderboard.csv`
   - `failures.jsonl`
   - `summary.md`

Example:
    python -u scripts/phase_cd_compare_report.py \
      --phase-c-logs-root assets/artifacts/phase_c_logs \
      --phase-d-logs-root assets/artifacts/phase_d_logs \
      --output-dir assets/artifacts/phase_cd_reports/manual_diag_20260305
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricsSnapshot:
    """Compact metric view used by cross-stage diagnostics."""

    brier: float | None
    pearson: float | None
    posthoc_brier: float | None
    posthoc_pearson: float | None
    corr_pair_acc: float | None
    corr_auc: float | None
    target_source: str | None
    target_cov: float | None
    teacher_dis: float | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase C/D unified reporting."""

    parser = argparse.ArgumentParser(description="Summarize and compare Phase C vs D results.")
    parser.add_argument(
        "--phase-c-logs-root",
        type=Path,
        default=Path("assets/artifacts/phase_c_logs"),
        help="Root directory containing Phase C suite logs.",
    )
    parser.add_argument(
        "--phase-d-logs-root",
        type=Path,
        default=Path("assets/artifacts/phase_d_logs"),
        help="Root directory containing Phase D suite logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write report outputs (summary.md, csv/jsonl).",
    )
    parser.add_argument(
        "--dataset",
        default="strategyqa",
        help="Dataset filter used for comparable C baseline selection. Default: strategyqa.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=80,
        help="Max rows per table in summary markdown to keep report readable.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    """Convert a value to float safely, returning None for invalid inputs."""

    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _clean_bullet_value(value: str) -> str:
    """Normalize bullet values parsed from markdown summaries."""

    v = value.strip()
    if v.startswith("`") and v.endswith("`") and len(v) >= 2:
        v = v[1:-1]
    return v.strip()


def _parse_iso_ts(value: str | None) -> datetime:
    """Parse timestamp from summary text; fallback to epoch for robustness."""

    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    text = value.strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _extract_bullets(md: str) -> dict[str, str]:
    """Extract `- key: value` bullets from markdown into a dictionary."""

    out: dict[str, str] = {}
    pattern = re.compile(r"^- ([^:]+):\s*(.+)$", flags=re.MULTILINE)
    for match in pattern.finditer(md):
        key = match.group(1).strip()
        value = _clean_bullet_value(match.group(2))
        out[key] = value
    return out


def _is_md_separator_row(line: str) -> bool:
    """Return True if line is a markdown table separator row."""

    s = line.strip().strip("|").strip()
    if not s:
        return False
    parts = [p.strip() for p in s.split("|")]
    if not parts:
        return False
    return all(part and set(part) <= {"-", ":"} for part in parts)


def _parse_markdown_table(md: str, header_prefix: str) -> list[dict[str, str]]:
    """Parse a markdown table whose header row starts with `header_prefix`."""

    lines = md.splitlines()
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line.startswith("|"):
            continue
        if not line.lower().startswith(header_prefix.lower()):
            continue
        header_cells = [c.strip() for c in line.strip("|").split("|")]
        rows: list[dict[str, str]] = []
        for body in lines[idx + 1 :]:
            b = body.strip()
            if not b.startswith("|"):
                break
            if _is_md_separator_row(b):
                continue
            cells = [c.strip() for c in b.strip("|").split("|")]
            if len(cells) != len(header_cells):
                continue
            rows.append({header_cells[i]: _clean_bullet_value(cells[i]) for i in range(len(header_cells))})
        return rows
    return []


def _load_metrics(metrics_path: Path) -> MetricsSnapshot:
    """Load a faithfulness metrics.json file into a compact snapshot."""

    if not metrics_path.exists():
        return MetricsSnapshot(
            brier=None,
            pearson=None,
            posthoc_brier=None,
            posthoc_pearson=None,
            corr_pair_acc=None,
            corr_auc=None,
            target_source=None,
            target_cov=None,
            teacher_dis=None,
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    cal = payload.get("calibration") or {}
    post = payload.get("calibration_posthoc") or {}
    cor = payload.get("corruption") or {}
    target_stats = payload.get("target_source_stats") or {}
    return MetricsSnapshot(
        brier=_safe_float(cal.get("brier_score")),
        pearson=_safe_float(cal.get("pearson")),
        posthoc_brier=_safe_float(post.get("brier_score")),
        posthoc_pearson=_safe_float(post.get("pearson")),
        corr_pair_acc=_safe_float(cor.get("pair_accuracy")),
        corr_auc=_safe_float(cor.get("auc_clean_vs_corrupt")),
        target_source=(payload.get("target_source") if isinstance(payload.get("target_source"), str) else None),
        target_cov=_safe_float(target_stats.get("coverage_ratio")),
        teacher_dis=_safe_float(target_stats.get("teacher_disagree_ratio")),
    )


def _record_from_metrics(
    *,
    kind: str,
    group_id: str,
    run_prefix: str,
    log_dir: Path,
    generated_at: datetime,
    label: str,
    dataset: str | None,
    c2_eval_dir: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one normalized record row from an eval metrics directory."""

    metrics_path = Path(c2_eval_dir) / "metrics.json" if c2_eval_dir else Path("")
    snap = _load_metrics(metrics_path)
    row: dict[str, Any] = {
        "kind": kind,
        "group_id": group_id,
        "run_prefix": run_prefix,
        "label": label,
        "dataset": dataset,
        "generated_at": generated_at.isoformat(),
        "log_dir": str(log_dir),
        "c2_eval_dir": c2_eval_dir,
        "metrics_path": (str(metrics_path) if c2_eval_dir else None),
        "brier": snap.brier,
        "pearson": snap.pearson,
        "posthoc_brier": snap.posthoc_brier,
        "posthoc_pearson": snap.posthoc_pearson,
        "corr_pair_acc": snap.corr_pair_acc,
        "corr_auc": snap.corr_auc,
        "target_source": snap.target_source,
        "target_cov": snap.target_cov,
        "teacher_dis": snap.teacher_dis,
    }
    if extra:
        row.update(extra)
    return row


def _load_phase_c_records(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load Phase C suite records and failures from `phase_c_logs`."""

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not root.exists():
        return records, failures
    for log_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        summary = log_dir / "final_summary.md"
        if not summary.exists():
            continue
        md = summary.read_text(encoding="utf-8")
        bullets = _extract_bullets(md)
        group_id = bullets.get("group_id", "")
        if not group_id.startswith("C2_") and not group_id.startswith("PIK_"):
            continue
        run_prefix = bullets.get("run_prefix", log_dir.name)
        status = bullets.get("status", "ok").strip().lower()
        generated_at = _parse_iso_ts(bullets.get("generated_at"))
        if status == "failed":
            failures.append(
                {
                    "kind": "C",
                    "group_id": group_id,
                    "run_prefix": run_prefix,
                    "log_dir": str(log_dir),
                    "failed_stage": bullets.get("failed_stage"),
                    "exit_code": bullets.get("exit_code"),
                }
            )
            continue
        c2_eval_dir = bullets.get("c2_eval_dir")
        dataset = bullets.get("dataset")
        records.append(
            _record_from_metrics(
                kind=("C_PIK" if group_id.startswith("PIK_") else "C2"),
                group_id=group_id,
                run_prefix=run_prefix,
                log_dir=log_dir,
                generated_at=generated_at,
                label="main",
                dataset=dataset,
                c2_eval_dir=c2_eval_dir,
                extra={"status": status},
            )
        )
    return records, failures


def _load_phase_d_teacher_records(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load D teacher-suite records (`D4_STRATEGYQA_*`) and failures."""

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not root.exists():
        return records, failures
    for log_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        summary = log_dir / "final_summary.md"
        if not summary.exists():
            continue
        md = summary.read_text(encoding="utf-8")
        bullets = _extract_bullets(md)
        group_id = bullets.get("group_id", "")
        if not (
            group_id.startswith("D4_STRATEGYQA_")
            or group_id.startswith("D5_STRATEGYQA_")
            or group_id.startswith("D6_STRATEGYQA_")
        ):
            continue
        run_prefix = bullets.get("run_prefix", log_dir.name)
        status = bullets.get("status", "ok").strip().lower()
        generated_at = _parse_iso_ts(bullets.get("generated_at"))
        if status == "failed":
            failures.append(
                {
                    "kind": "D_teacher",
                    "group_id": group_id,
                    "run_prefix": run_prefix,
                    "log_dir": str(log_dir),
                    "failed_stage": bullets.get("failed_stage"),
                    "exit_code": bullets.get("exit_code"),
                }
            )
            continue
        table = _parse_markdown_table(md, header_prefix="| label | target_source |")
        for row in table:
            label = row.get("label", "")
            c2_eval_dir = row.get("c2_eval_dir")
            records.append(
                _record_from_metrics(
                    kind="D_teacher",
                    group_id=group_id,
                    run_prefix=run_prefix,
                    log_dir=log_dir,
                    generated_at=generated_at,
                    label=label,
                    dataset=bullets.get("dataset"),
                    c2_eval_dir=c2_eval_dir,
                    extra={"status": status, "target_source_summary": row.get("target_source")},
                )
            )
    return records, failures


def _load_phase_d_external_records(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load D4 external-pair suite records from `stage_results.jsonl`."""

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not root.exists():
        return records, failures
    for log_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        summary = log_dir / "final_summary.md"
        if not summary.exists():
            continue
        md = summary.read_text(encoding="utf-8")
        bullets = _extract_bullets(md)
        group_id = bullets.get("group_id", "")
        if not group_id.startswith("D4A") and not group_id.startswith("D4B") and not group_id.startswith("D4C") and not group_id.startswith("D4ABC"):
            continue
        run_prefix = bullets.get("run_prefix", log_dir.name)
        status = bullets.get("status", "ok").strip().lower()
        stage_results = log_dir / "stage_results.jsonl"
        generated_at = _parse_iso_ts(bullets.get("generated_at"))

        if stage_results.exists():
            for raw in stage_results.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                payload = json.loads(line)
                eval_dir = payload.get("eval_run_dir")
                rec = _record_from_metrics(
                    kind="D_external",
                    group_id=group_id,
                    run_prefix=run_prefix,
                    log_dir=log_dir,
                    generated_at=generated_at,
                    label=str(payload.get("stage", "stage")),
                    dataset="strategyqa",
                    c2_eval_dir=(str(eval_dir) if eval_dir else None),
                    extra={
                        "status": status,
                        "pair_dir": payload.get("pair_dir"),
                        "c2_run_dir": payload.get("c2_run_dir"),
                        "num_train_pairs": payload.get("num_train_pairs"),
                        "num_val_pairs": payload.get("num_val_pairs"),
                        "pair_mean_conf": payload.get("pair_mean_conf"),
                        "pair_sources": payload.get("pair_sources"),
                    },
                )
                # Stage metrics recorded by suite are kept as fallbacks when
                # eval metrics are unavailable.
                if rec.get("brier") is None:
                    rec["brier"] = _safe_float(payload.get("brier"))
                if rec.get("corr_pair_acc") is None:
                    rec["corr_pair_acc"] = _safe_float(payload.get("corr_pair_acc"))
                if rec.get("corr_auc") is None:
                    rec["corr_auc"] = _safe_float(payload.get("corr_auc"))
                records.append(rec)

        if status == "failed":
            failures.append(
                {
                    "kind": "D_external",
                    "group_id": group_id,
                    "run_prefix": run_prefix,
                    "log_dir": str(log_dir),
                    "failed_stage": bullets.get("failed_stage"),
                    "exit_code": bullets.get("exit_code"),
                }
            )
    return records, failures


def _pick_c_reference(rows: list[dict[str, Any]], dataset: str) -> dict[str, Any] | None:
    """Pick one comparable C baseline row for D-vs-C deltas."""

    candidates = [r for r in rows if r.get("kind") == "C2" and (r.get("dataset") or "") == dataset]
    if not candidates:
        return None

    def priority(row: dict[str, Any]) -> int:
        gid = str(row.get("group_id") or "")
        if gid == "C2_STRATEGYQA_CQR_FULL":
            return 0
        if gid == "C2_STRATEGYQA_QUALITY_FIRST_FULL":
            return 1
        if gid == "C2_STRATEGYQA_CQR_SMOKE":
            return 2
        return 3

    best_priority = min(priority(row) for row in candidates)
    bucket = [row for row in candidates if priority(row) == best_priority]
    # Within the same priority bucket, pick the latest run.
    return sorted(bucket, key=lambda r: _parse_iso_ts(str(r.get("generated_at") or "")))[-1]


def _with_deltas(rows: list[dict[str, Any]], ref: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Attach D-vs-reference deltas for comparable metrics."""

    if ref is None:
        return rows
    ref_post = ref.get("posthoc_brier")
    ref_acc = ref.get("corr_pair_acc")
    ref_auc = ref.get("corr_auc")
    out: list[dict[str, Any]] = []
    for row in rows:
        cloned = dict(row)
        if row.get("kind", "").startswith("D_") or row.get("kind") == "D_teacher":
            post = row.get("posthoc_brier")
            if post is None:
                post = row.get("brier")
            cloned["delta_posthoc_brier_vs_c_ref"] = (
                (float(post) - float(ref_post))
                if (post is not None and ref_post is not None)
                else None
            )
            cloned["delta_pair_acc_vs_c_ref"] = (
                (float(row["corr_pair_acc"]) - float(ref_acc))
                if (row.get("corr_pair_acc") is not None and ref_acc is not None)
                else None
            )
            cloned["delta_auc_vs_c_ref"] = (
                (float(row["corr_auc"]) - float(ref_auc))
                if (row.get("corr_auc") is not None and ref_auc is not None)
                else None
            )
        out.append(cloned)
    return out


def _fmt_float(value: Any, digits: int = 6) -> str:
    """Format numeric values for markdown tables."""

    v = _safe_float(value)
    return f"{v:.{digits}f}" if v is not None else "n/a"


def _to_markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int) -> str:
    """Render simple markdown table from row dictionaries."""

    if not rows:
        return "_no rows_\n"
    subset = rows[:max_rows]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in subset:
        cells: list[str] = []
        for c in columns:
            val = row.get(c)
            if isinstance(val, float):
                cells.append(_fmt_float(val))
            elif isinstance(val, dict):
                cells.append(json.dumps(val, ensure_ascii=False))
            else:
                cells.append(str(val) if val is not None else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n_showing first {max_rows}/{len(rows)} rows_\n")
    return "\n".join(lines) + "\n"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSONL."""

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as CSV with dynamic field union."""

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for k, v in list(flat.items()):
                if isinstance(v, (dict, list)):
                    flat[k] = json.dumps(v, ensure_ascii=False)
            writer.writerow(flat)


def _diagnose_text(rows: list[dict[str, Any]], ref: dict[str, Any] | None) -> list[str]:
    """Generate concise diagnostic bullets from computed rows."""

    notes: list[str] = []
    d_rows = [r for r in rows if str(r.get("kind", "")).startswith("D_") or r.get("kind") == "D_teacher"]
    d_teacher = [r for r in d_rows if r.get("kind") == "D_teacher"]
    d_external = [r for r in d_rows if r.get("kind") == "D_external"]

    if ref is not None:
        notes.append(
            "Comparable C reference: "
            f"{ref.get('group_id')} / {ref.get('run_prefix')} "
            f"(posthoc_brier={_fmt_float(ref.get('posthoc_brier'))}, "
            f"pair_acc={_fmt_float(ref.get('corr_pair_acc'))}, auc={_fmt_float(ref.get('corr_auc'))})."
        )

    if d_teacher:
        better_pair = [
            r
            for r in d_teacher
            if r.get("delta_pair_acc_vs_c_ref") is not None and float(r["delta_pair_acc_vs_c_ref"]) > 0.0
        ]
        better_auc = [
            r for r in d_teacher if r.get("delta_auc_vs_c_ref") is not None and float(r["delta_auc_vs_c_ref"]) > 0.0
        ]
        notes.append(
            "D teacher rows: "
            f"{len(d_teacher)} total, "
            f"{len(better_pair)} beat C ref on pair_acc, "
            f"{len(better_auc)} beat C ref on auc."
        )

    if d_external:
        mean_brier = _safe_float(sum((r.get("brier") or 0.0) for r in d_external) / max(len(d_external), 1))
        mean_acc = _safe_float(sum((r.get("corr_pair_acc") or 0.0) for r in d_external) / max(len(d_external), 1))
        notes.append(
            "D4 external stages: "
            f"{len(d_external)} rows, mean_brier={_fmt_float(mean_brier)}, mean_pair_acc={_fmt_float(mean_acc)}."
        )

    # A lightweight consistency warning for suspiciously low brier.
    suspicious = [r for r in d_rows if r.get("brier") is not None and float(r["brier"]) < 0.08]
    if suspicious:
        notes.append(
            "Detected very-low brier rows (<0.08). Re-check target distribution and leakage risk for those runs."
        )

    if not notes:
        notes.append("No comparable rows found yet. Run at least one successful C2 and one successful D suite.")
    return notes


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    c_rows, c_fail = _load_phase_c_records(args.phase_c_logs_root)
    d_teacher_rows, d_teacher_fail = _load_phase_d_teacher_records(args.phase_d_logs_root)
    d_external_rows, d_external_fail = _load_phase_d_external_records(args.phase_d_logs_root)

    all_rows = c_rows + d_teacher_rows + d_external_rows
    all_failures = c_fail + d_teacher_fail + d_external_fail

    # Sort by timestamp descending so newest rows show first.
    all_rows.sort(key=lambda r: _parse_iso_ts(str(r.get("generated_at") or "")).timestamp(), reverse=True)
    ref = _pick_c_reference(all_rows, dataset=args.dataset)
    all_rows = _with_deltas(all_rows, ref)

    records_jsonl = args.output_dir / "records.jsonl"
    leaderboard_csv = args.output_dir / "leaderboard.csv"
    failures_jsonl = args.output_dir / "failures.jsonl"
    summary_md = args.output_dir / "summary.md"

    _write_jsonl(records_jsonl, all_rows)
    _write_csv(leaderboard_csv, all_rows)
    _write_jsonl(failures_jsonl, all_failures)

    c_table_rows = [r for r in all_rows if r.get("kind") in {"C2", "C_PIK"}]
    d_teacher_table_rows = [r for r in all_rows if r.get("kind") == "D_teacher"]
    d_external_table_rows = [r for r in all_rows if r.get("kind") == "D_external"]
    diagnostics = _diagnose_text(all_rows, ref)

    lines: list[str] = []
    lines.append("# Phase C/D Unified Diagnostic Report")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- phase_c_logs_root: `{args.phase_c_logs_root}`")
    lines.append(f"- phase_d_logs_root: `{args.phase_d_logs_root}`")
    lines.append(f"- dataset_filter: `{args.dataset}`")
    lines.append(f"- total_rows: {len(all_rows)}")
    lines.append(f"- total_failures: {len(all_failures)}")
    lines.append("")
    lines.append("## Reference Baseline")
    lines.append("")
    if ref is None:
        lines.append("- not found")
    else:
        lines.append(f"- group_id: `{ref.get('group_id')}`")
        lines.append(f"- run_prefix: `{ref.get('run_prefix')}`")
        lines.append(f"- c2_eval_dir: `{ref.get('c2_eval_dir')}`")
        lines.append(f"- posthoc_brier: {_fmt_float(ref.get('posthoc_brier'))}")
        lines.append(f"- corr_pair_acc: {_fmt_float(ref.get('corr_pair_acc'))}")
        lines.append(f"- corr_auc: {_fmt_float(ref.get('corr_auc'))}")
    lines.append("")
    lines.append("## Key Diagnostics")
    lines.append("")
    for note in diagnostics:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## C Rows (Comparable Controls)")
    lines.append("")
    lines.append(
        _to_markdown_table(
            c_table_rows,
            [
                "kind",
                "group_id",
                "run_prefix",
                "posthoc_brier",
                "corr_pair_acc",
                "corr_auc",
                "c2_eval_dir",
            ],
            max_rows=args.max_rows,
        )
    )
    lines.append("## D Teacher Rows (D1/D2/D3 Path)")
    lines.append("")
    lines.append(
        _to_markdown_table(
            d_teacher_table_rows,
            [
                "group_id",
                "run_prefix",
                "label",
                "target_source",
                "posthoc_brier",
                "corr_pair_acc",
                "corr_auc",
                "delta_posthoc_brier_vs_c_ref",
                "delta_pair_acc_vs_c_ref",
                "delta_auc_vs_c_ref",
                "c2_eval_dir",
            ],
            max_rows=args.max_rows,
        )
    )
    lines.append("## D External Rows (D4A/B/C Path)")
    lines.append("")
    lines.append(
        _to_markdown_table(
            d_external_table_rows,
            [
                "group_id",
                "run_prefix",
                "label",
                "num_train_pairs",
                "pair_mean_conf",
                "brier",
                "corr_pair_acc",
                "corr_auc",
                "delta_posthoc_brier_vs_c_ref",
                "delta_pair_acc_vs_c_ref",
                "delta_auc_vs_c_ref",
                "c2_eval_dir",
            ],
            max_rows=args.max_rows,
        )
    )
    lines.append("## Failure Rows")
    lines.append("")
    lines.append(
        _to_markdown_table(
            all_failures,
            ["kind", "group_id", "run_prefix", "failed_stage", "exit_code", "log_dir"],
            max_rows=args.max_rows,
        )
    )
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- records_jsonl: `{records_jsonl}`")
    lines.append(f"- leaderboard_csv: `{leaderboard_csv}`")
    lines.append(f"- failures_jsonl: `{failures_jsonl}`")
    lines.append(f"- summary_md: `{summary_md}`")
    lines.append("")
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 88)
    print("Phase C/D Unified Diagnostic")
    print("=" * 88)
    print(f"rows          : {len(all_rows)}")
    print(f"failures      : {len(all_failures)}")
    if ref is not None:
        print(
            "c_ref         : "
            f"{ref.get('group_id')} / {ref.get('run_prefix')} "
            f"(posthoc_brier={_fmt_float(ref.get('posthoc_brier'))}, "
            f"pair_acc={_fmt_float(ref.get('corr_pair_acc'))}, auc={_fmt_float(ref.get('corr_auc'))})"
        )
    else:
        print("c_ref         : <none>")
    print(f"summary_path  : {summary_md}")
    print(f"records_path  : {records_jsonl}")
    print(f"csv_path      : {leaderboard_csv}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
