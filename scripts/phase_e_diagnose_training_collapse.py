#!/usr/bin/env python3
"""Diagnose one Phase E training run for collapse / inversion symptoms.

English
-------
This script is intentionally lightweight: it reads an existing Phase E run
directory and re-computes the training-health diagnosis from persisted
artifacts.  It is useful when an older run predates the automatic
`training_health.json` artifact, or when operators want one uniform re-audit.

中文
----
这个脚本专门做“事后审计”：
它读取一个已经完成的 Phase E run 目录，然后根据现有产物重新诊断：
1. 是否出现了 flat loss，
2. 是否 margin 接近 0，
3. 是否 chosen/rejected 分数方向反了，
4. 是否符合已知 collapse 签名。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_e.recipe_safety import (  # noqa: E402
    assess_phase_e_recipe_risk,
    diagnose_phase_e_training_health,
    render_phase_e_training_health_markdown,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose one finished Phase E run for collapse signatures.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    train_metrics_path = run_dir / "train_metrics.json"
    eval_metrics_path = run_dir / "eval_metrics.json"
    eval_pair_scores_path = run_dir / "eval_pair_scores.jsonl"
    output_json_path = run_dir / "training_health_rediagnosed.json"
    output_md_path = run_dir / "training_health_rediagnosed.md"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json: {manifest_path}")
    if not train_metrics_path.exists():
        raise FileNotFoundError(f"Missing train_metrics.json: {train_metrics_path}")
    if not eval_metrics_path.exists():
        raise FileNotFoundError(f"Missing eval_metrics.json: {eval_metrics_path}")
    if not eval_pair_scores_path.exists():
        raise FileNotFoundError(f"Missing eval_pair_scores.jsonl: {eval_pair_scores_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))
    eval_metrics = json.loads(eval_metrics_path.read_text(encoding="utf-8"))

    chosen_scores: list[float] = []
    rejected_scores: list[float] = []
    with eval_pair_scores_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            chosen_scores.append(float(row["chosen_score"]))
            rejected_scores.append(float(row["rejected_score"]))

    recipe_risk_report = assess_phase_e_recipe_risk(
        train_pair_summary=dict(manifest.get("train_pair_summary", {}) or {}),
        train_config=dict(manifest.get("train_config", {}) or {}),
    )
    diagnostics = diagnose_phase_e_training_health(
        train_curve=list(train_metrics.get("train_curve", []) or []),
        best_eval_metrics=dict(eval_metrics.get("eval_pairs", {}) or {}),
        chosen_scores=chosen_scores,
        rejected_scores=rejected_scores,
        recipe_risk_report=recipe_risk_report,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "recipe_risk_report": recipe_risk_report,
        "training_health": diagnostics,
    }
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md_path.write_text(render_phase_e_training_health_markdown(diagnostics), encoding="utf-8")

    print("=" * 88)
    print("Phase E: Diagnose Training Collapse")
    print("=" * 88)
    print(f"run_dir           : {run_dir}")
    print(f"diagnosis         : {diagnostics.get('diagnosis', 'unknown')}")
    print(f"known_collapse    : {bool(diagnostics.get('known_collapse_signature', False))}")
    print(f"pair_accuracy     : {float(diagnostics['heldout_metrics']['pair_accuracy']):.6f}")
    print(f"auc               : {float(diagnostics['heldout_metrics']['auc']):.6f}")
    print(f"mean_margin       : {float(diagnostics['score_distribution']['mean_margin']):.6f}")
    print(f"recipe_severity   : {recipe_risk_report.get('max_severity', 'info')}")
    print(f"json_path         : {output_json_path}")
    print(f"markdown_path     : {output_md_path}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
