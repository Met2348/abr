#!/usr/bin/env python3
"""Build one comparable frontier matrix for current Phase E verifier candidates.

English
-------
This script joins four artifact families:
1. the training run itself,
2. held-out pair metrics,
3. same-family trust metrics,
4. ProcessBench GSM/MATH evaluation.

The result is a single matrix that makes it easier to compare data choices,
architecture choices, and benchmark-facing trust without hand-reading many
artifact directories.

中文
----
这个脚本把四类 artifact 合并成一张对照矩阵：
1. 训练 run 本身，
2. held-out pair 指标，
3. same-family trust 指标，
4. ProcessBench GSM/MATH 评测。

这样就不需要手工翻很多目录，也能直接比较数据选择、结构选择和
benchmark-facing trust 的差异。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FrontierRow:
    label: str
    run_dir: str
    pair_pool: str | None
    train_pairs: int | None
    objective_mode: str | None
    ranking_target_space: str | None
    pair_weight_mode: str | None
    head_architecture: str | None
    heldout_pair_acc: float | None
    heldout_auc: float | None
    samefamily_top1: float | None
    samefamily_first_bad: float | None
    samefamily_safe_bad_pair: float | None
    gsm_auc: float | None
    gsm_f1: float | None
    gsm_first_error: float | None
    math_auc: float | None
    math_f1: float | None
    math_first_error: float | None
    findings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze the current Phase E frontier matrix.")
    parser.add_argument(
        "--entry",
        nargs=5,
        action="append",
        metavar=("LABEL", "RUN_DIR", "SAMEFAMILY_DIR", "GSM_EVAL_DIR", "MATH_EVAL_DIR"),
        required=True,
        help=(
            "One frontier row as LABEL RUN_DIR SAMEFAMILY_DIR GSM_EVAL_DIR MATH_EVAL_DIR. "
            "Use `-` for any missing evaluation directory."
        ),
    )
    parser.add_argument("--run-name", default="phase_e_frontier_matrix")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_analysis"),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    normalized: list[tuple[str, Path, Path | None, Path | None, Path | None]] = []
    seen_labels: set[str] = set()
    for label, run_dir_text, samefamily_text, gsm_text, math_text in args.entry:
        if label in seen_labels:
            raise ValueError(f"Duplicate --entry label: {label}")
        seen_labels.add(label)
        run_dir = Path(run_dir_text)
        samefamily_dir = None if samefamily_text == "-" else Path(samefamily_text)
        gsm_dir = None if gsm_text == "-" else Path(gsm_text)
        math_dir = None if math_text == "-" else Path(math_text)
        required_paths = []
        if samefamily_dir is not None:
            required_paths.append((samefamily_dir / "metrics.json", "metrics.json"))
        if gsm_dir is not None:
            required_paths.append((gsm_dir / "summary.json", "summary.json"))
        if math_dir is not None:
            required_paths.append((math_dir / "summary.json", "summary.json"))
        for path, required in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing {required}: {path}")
        normalized.append((label, run_dir, samefamily_dir, gsm_dir, math_dir))
    args.entry = normalized
    return args


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_frontier_row(
    *,
    label: str,
    run_dir: Path,
    samefamily_dir: Path | None,
    gsm_dir: Path | None,
    math_dir: Path | None,
) -> FrontierRow:
    manifest = _load_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else {}
    recipe_risk = _load_json(run_dir / "recipe_risk.json") if (run_dir / "recipe_risk.json").exists() else {}
    value_head_cfg = _load_json(run_dir / "value_head_config.json") if (run_dir / "value_head_config.json").exists() else {}
    samefamily = _load_json(samefamily_dir / "metrics.json") if samefamily_dir is not None else {}
    gsm_summary = _load_json(gsm_dir / "summary.json") if gsm_dir is not None else {}
    math_summary = _load_json(math_dir / "summary.json") if math_dir is not None else {}
    gsm_metrics = dict(gsm_summary.get("metrics") or {})
    math_metrics = dict(math_summary.get("metrics") or {})
    train_config = dict(manifest.get("train_config") or {})
    input_files = dict(manifest.get("input_files") or {})
    pair_pool = input_files.get("train_pairs_jsonl")
    eval_metrics, heldout_source = _load_heldout_metrics(run_dir)

    findings: list[str] = []
    math_f1 = _safe_float(math_metrics.get("processbench_f1"))
    if math_f1 is not None and math_f1 < 0.67:
        findings.append(f"math_f1_low:{math_f1:.3f}")
    samefamily_first_bad = _safe_float(samefamily.get("local_first_bad_edge_accuracy"))
    if samefamily_dir is None:
        findings.append("samefamily_missing")
    elif samefamily_first_bad is not None and samefamily_first_bad < 0.85:
        findings.append(f"samefamily_first_bad_low:{samefamily_first_bad:.3f}")
    if gsm_dir is None:
        findings.append("gsm_benchmark_missing")
    if math_dir is None:
        findings.append("math_benchmark_missing")
    if recipe_risk:
        semantics = dict(recipe_risk.get("by_pair_semantics") or {})
        if "local_first_bad_edge" not in semantics:
            findings.append("no_local_first_bad_training_semantics")
    if heldout_source != "eval_metrics":
        findings.append(f"heldout_source:{heldout_source}")

    return FrontierRow(
        label=label,
        run_dir=str(run_dir),
        pair_pool=str(pair_pool) if pair_pool else None,
        train_pairs=_safe_int(recipe_risk.get("num_pairs")),
        objective_mode=_safe_str(train_config.get("objective_mode")),
        ranking_target_space=_safe_str(train_config.get("ranking_target_space")),
        pair_weight_mode=_safe_str(train_config.get("pair_weight_mode")),
        head_architecture=_safe_str(value_head_cfg.get("architecture")),
        heldout_pair_acc=_safe_float((eval_metrics.get("eval_pairs") or {}).get("pair_accuracy")),
        heldout_auc=_safe_float((eval_metrics.get("eval_pairs") or {}).get("auc")),
        samefamily_top1=_safe_float(samefamily.get("prompt_pool_top1_accuracy")),
        samefamily_first_bad=samefamily_first_bad,
        samefamily_safe_bad_pair=_safe_float(samefamily.get("local_safe_vs_bad_pair_accuracy")),
        gsm_auc=_safe_float(gsm_metrics.get("pair_auc_good_vs_bad")),
        gsm_f1=_safe_float(gsm_metrics.get("processbench_f1")),
        gsm_first_error=_safe_float(gsm_metrics.get("first_error_edge_accuracy")),
        math_auc=_safe_float(math_metrics.get("pair_auc_good_vs_bad")),
        math_f1=math_f1,
        math_first_error=_safe_float(math_metrics.get("first_error_edge_accuracy")),
        findings=findings,
    )


def _load_heldout_metrics(run_dir: Path) -> tuple[dict[str, Any], str]:
    eval_metrics_path = run_dir / "eval_metrics.json"
    if eval_metrics_path.exists():
        return _load_json(eval_metrics_path), "eval_metrics"

    train_curve_path = run_dir / "train_curve.jsonl"
    if train_curve_path.exists():
        best_eval: dict[str, Any] | None = None
        best_value = float("-inf")
        for raw in train_curve_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            eval_block = dict(record.get("eval") or {})
            selection_value = float(record.get("selection_value", float("-inf")))
            if selection_value > best_value and eval_block:
                best_value = selection_value
                best_eval = eval_block
        if best_eval is not None:
            return {"eval_pairs": best_eval}, "train_curve_best_eval"

    return {"eval_pairs": {}}, "missing"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _render_summary_markdown(rows: list[FrontierRow]) -> str:
    lines = [
        "# Phase E Frontier Matrix",
        "",
        "| label | train_pairs | heldout_pair | heldout_auc | same_top1 | first_bad | gsm_f1 | math_f1 | gsm_auc | math_auc | findings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {train_pairs} | {heldout_pair:.3f} | {heldout_auc:.3f} | {same_top1:.3f} | "
            "{first_bad:.3f} | {gsm_f1:.3f} | {math_f1:.3f} | {gsm_auc:.3f} | {math_auc:.3f} | {findings} |".format(
                label=row.label,
                train_pairs=row.train_pairs or 0,
                heldout_pair=row.heldout_pair_acc or 0.0,
                heldout_auc=row.heldout_auc or 0.0,
                same_top1=row.samefamily_top1 or 0.0,
                first_bad=row.samefamily_first_bad or 0.0,
                gsm_f1=row.gsm_f1 or 0.0,
                math_f1=row.math_f1 or 0.0,
                gsm_auc=row.gsm_auc or 0.0,
                math_auc=row.math_auc or 0.0,
                findings=", ".join(row.findings) if row.findings else "ok",
            )
        )
    lines.extend(["", "## Detailed Rows", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row.label}",
                "",
                f"- run_dir: `{row.run_dir}`",
                f"- pair_pool: `{row.pair_pool}`",
                f"- objective_mode: `{row.objective_mode}`",
                f"- ranking_target_space: `{row.ranking_target_space}`",
                f"- pair_weight_mode: `{row.pair_weight_mode}`",
                f"- head_architecture: `{row.head_architecture}`",
                f"- findings: `{row.findings or ['ok']}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.output_root) / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _extract_frontier_row(
            label=label,
            run_dir=run_dir_path,
            samefamily_dir=samefamily_dir,
            gsm_dir=gsm_dir,
            math_dir=math_dir,
        )
        for label, run_dir_path, samefamily_dir, gsm_dir, math_dir in args.entry
    ]
    rows.sort(key=lambda item: ((item.math_f1 or 0.0), (item.math_auc or 0.0), (item.samefamily_top1 or 0.0)), reverse=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "rows": [row.to_dict() for row in rows],
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(_render_summary_markdown(rows), encoding="utf-8")

    print("=" * 88)
    print("Phase E Frontier Matrix")
    print("=" * 88)
    print(f"run_dir: {run_dir}")
    for row in rows:
        print(
            f"- {row.label}: math_f1={row.math_f1} gsm_f1={row.gsm_f1} "
            f"same_top1={row.samefamily_top1} findings={row.findings or ['ok']}"
        )
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
