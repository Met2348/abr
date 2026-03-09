"""Unit tests for scripts/phase_d_promotion_gate.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_gate_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_d_promotion_gate.py"
    spec = importlib.util.spec_from_file_location("phase_d_promotion_gate", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_eval_metrics(
    eval_dir: Path,
    *,
    brier: float,
    pair_acc: float,
    auc: float,
    brier_improvement_vs_baseline: float = 0.01,
) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "calibration": {
            "brier_score": brier,
            "pearson": 0.2,
            "brier_improvement_vs_baseline": brier_improvement_vs_baseline,
        },
        "calibration_posthoc": {"brier_score": max(brier - 0.01, 0.0)},
        "corruption": {
            "pair_accuracy": pair_acc,
            "auc_clean_vs_corrupt": auc,
        },
    }
    (eval_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_phase_c_summary(
    log_dir: Path,
    *,
    generated_at: str,
    group_id: str,
    run_prefix: str,
    dataset: str,
    c2_eval_dir: Path,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    text = (
        "# Phase C Summary\n\n"
        f"- generated_at: {generated_at}\n"
        f"- group_id: {group_id}\n"
        f"- run_prefix: {run_prefix}\n"
        f"- dataset: {dataset}\n"
        f"- c2_eval_dir: {c2_eval_dir}\n"
    )
    (log_dir / "final_summary.md").write_text(text, encoding="utf-8")


def _write_d6_summary(
    log_dir: Path,
    *,
    generated_at: str,
    group_id: str,
    run_prefix: str,
    dataset: str,
    c2_eval_dir: Path,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    text = (
        "# Phase D Teacher Suite Summary\n\n"
        f"- generated_at: {generated_at}\n"
        f"- group_id: {group_id}\n"
        f"- run_prefix: {run_prefix}\n"
        f"- dataset: {dataset}\n\n"
        "## D3 Result Table\n\n"
        "| label | target_source | brier | pearson | posthoc_brier | pair_acc | auc | target_cov | teacher_dis | c2_train_dir | c2_eval_dir |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
        f"| mc | q_mean_smoothed | 0.1 | 0.2 | 0.09 | 0.6 | 0.6 | 1.0 | 0.2 | /tmp/train | {c2_eval_dir} |\n"
    )
    (log_dir / "final_summary.md").write_text(text, encoding="utf-8")


def _write_d6t_summary(
    log_dir: Path,
    *,
    generated_at: str,
    group_id: str,
    run_prefix: str,
    seed_rows: list[dict],
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_text = (
        "# Phase D6-T Triplet Validation Suite Summary\n\n"
        f"- generated_at: {generated_at}\n"
        f"- group_id: {group_id}\n"
        f"- run_prefix: {run_prefix}\n"
        "- gate_pass: `True`\n"
    )
    (log_dir / "final_summary.md").write_text(summary_text, encoding="utf-8")
    (log_dir / "seed_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in seed_rows),
        encoding="utf-8",
    )


def _seed_rows(values: list[tuple[float, float]]) -> list[dict]:
    rows: list[dict] = []
    for idx, (pair_acc, auc) in enumerate(values, start=42):
        rows.append(
            {
                "seed": idx,
                "ext_pair_acc": pair_acc,
                "ext_auc": auc,
            }
        )
    return rows


def test_phase_d_promotion_gate_passes_when_all_gates_pass(tmp_path: Path) -> None:
    module = _load_gate_module()
    phase_c_logs = tmp_path / "phase_c_logs"
    phase_d_logs = tmp_path / "phase_d_logs"
    phase_d6t_logs = tmp_path / "phase_d6t_logs"
    out_dir = tmp_path / "gate_out"

    c_eval = tmp_path / "evals" / "c_ref"
    ctrl_eval = tmp_path / "evals" / "d6_ctrl"
    gate_eval = tmp_path / "evals" / "d6_gate"
    _write_eval_metrics(c_eval, brier=0.14, pair_acc=0.54, auc=0.58, brier_improvement_vs_baseline=0.001)
    _write_eval_metrics(ctrl_eval, brier=0.145, pair_acc=0.53, auc=0.57, brier_improvement_vs_baseline=0.0001)
    _write_eval_metrics(gate_eval, brier=0.138, pair_acc=0.58, auc=0.63, brier_improvement_vs_baseline=0.01)

    _write_phase_c_summary(
        phase_c_logs / "c_ref_log",
        generated_at="2026-03-05T01:00:00+00:00",
        group_id="C2_STRATEGYQA_CQR_FULL",
        run_prefix="c_ref",
        dataset="strategyqa",
        c2_eval_dir=c_eval,
    )
    _write_d6_summary(
        phase_d_logs / "d6_ctrl_log",
        generated_at="2026-03-05T02:00:00+00:00",
        group_id="D6_STRATEGYQA_SMOKE_RANKING_CTRL",
        run_prefix="d6_ctrl",
        dataset="strategyqa",
        c2_eval_dir=ctrl_eval,
    )
    _write_d6_summary(
        phase_d_logs / "d6_gate_log",
        generated_at="2026-03-05T03:00:00+00:00",
        group_id="D6_STRATEGYQA_SMOKE_RANKING_PRM_GATE",
        run_prefix="d6_gate",
        dataset="strategyqa",
        c2_eval_dir=gate_eval,
    )
    _write_d6t_summary(
        phase_d6t_logs / "dt2_log",
        generated_at="2026-03-05T04:00:00+00:00",
        group_id="DT2_MATH_SHEPHERD_SEED3",
        run_prefix="dt2_ok",
        seed_rows=_seed_rows([(0.68, 0.70), (0.69, 0.68), (0.67, 0.69)]),
    )
    _write_d6t_summary(
        phase_d6t_logs / "dt4_log",
        generated_at="2026-03-05T05:00:00+00:00",
        group_id="DT4_MIXED_MS_PRM800K_SEED3",
        run_prefix="dt4_ok",
        seed_rows=_seed_rows([(0.70, 0.71), (0.69, 0.70), (0.68, 0.69)]),
    )

    rc = module.main(
        [
            "--phase-c-logs-root",
            str(phase_c_logs),
            "--phase-d-logs-root",
            str(phase_d_logs),
            "--phase-d6t-logs-root",
            str(phase_d6t_logs),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    report = json.loads((out_dir / "gate_report.json").read_text(encoding="utf-8"))
    assert report["decision"]["promotion_ready"] is True
    assert report["decision"]["blocking_reasons"] == []


def test_phase_d_promotion_gate_blocks_on_d6t_high_variance(tmp_path: Path) -> None:
    module = _load_gate_module()
    phase_c_logs = tmp_path / "phase_c_logs"
    phase_d_logs = tmp_path / "phase_d_logs"
    phase_d6t_logs = tmp_path / "phase_d6t_logs"
    out_dir = tmp_path / "gate_out"

    c_eval = tmp_path / "evals" / "c_ref"
    ctrl_eval = tmp_path / "evals" / "d6_ctrl"
    gate_eval = tmp_path / "evals" / "d6_gate"
    _write_eval_metrics(c_eval, brier=0.14, pair_acc=0.54, auc=0.58)
    _write_eval_metrics(ctrl_eval, brier=0.145, pair_acc=0.53, auc=0.57)
    _write_eval_metrics(gate_eval, brier=0.138, pair_acc=0.58, auc=0.63)

    _write_phase_c_summary(
        phase_c_logs / "c_ref_log",
        generated_at="2026-03-05T01:00:00+00:00",
        group_id="C2_STRATEGYQA_CQR_FULL",
        run_prefix="c_ref",
        dataset="strategyqa",
        c2_eval_dir=c_eval,
    )
    _write_d6_summary(
        phase_d_logs / "d6_ctrl_log",
        generated_at="2026-03-05T02:00:00+00:00",
        group_id="D6_STRATEGYQA_SMOKE_RANKING_CTRL",
        run_prefix="d6_ctrl",
        dataset="strategyqa",
        c2_eval_dir=ctrl_eval,
    )
    _write_d6_summary(
        phase_d_logs / "d6_gate_log",
        generated_at="2026-03-05T03:00:00+00:00",
        group_id="D6_STRATEGYQA_SMOKE_RANKING_PRM_GATE",
        run_prefix="d6_gate",
        dataset="strategyqa",
        c2_eval_dir=gate_eval,
    )
    _write_d6t_summary(
        phase_d6t_logs / "dt2_log",
        generated_at="2026-03-05T04:00:00+00:00",
        group_id="DT2_MATH_SHEPHERD_SEED3",
        run_prefix="dt2_unstable",
        seed_rows=_seed_rows([(0.90, 0.90), (0.30, 0.30), (0.90, 0.90)]),
    )
    _write_d6t_summary(
        phase_d6t_logs / "dt4_log",
        generated_at="2026-03-05T05:00:00+00:00",
        group_id="DT4_MIXED_MS_PRM800K_SEED3",
        run_prefix="dt4_ok",
        seed_rows=_seed_rows([(0.70, 0.71), (0.69, 0.70), (0.68, 0.69)]),
    )

    rc = module.main(
        [
            "--phase-c-logs-root",
            str(phase_c_logs),
            "--phase-d-logs-root",
            str(phase_d_logs),
            "--phase-d6t-logs-root",
            str(phase_d6t_logs),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    report = json.loads((out_dir / "gate_report.json").read_text(encoding="utf-8"))
    assert report["decision"]["promotion_ready"] is False
    reasons = report["decision"]["blocking_reasons"]
    assert any(item.startswith("d6t_std_pair_fail:DT2_MATH_SHEPHERD_SEED3") for item in reasons)
    assert any(item.startswith("d6t_std_auc_fail:DT2_MATH_SHEPHERD_SEED3") for item in reasons)


def test_phase_d_promotion_gate_blocks_when_d6_gated_group_missing(tmp_path: Path) -> None:
    module = _load_gate_module()
    phase_c_logs = tmp_path / "phase_c_logs"
    phase_d_logs = tmp_path / "phase_d_logs"
    phase_d6t_logs = tmp_path / "phase_d6t_logs"
    out_dir = tmp_path / "gate_out"

    c_eval = tmp_path / "evals" / "c_ref"
    ctrl_eval = tmp_path / "evals" / "d6_ctrl"
    _write_eval_metrics(c_eval, brier=0.14, pair_acc=0.54, auc=0.58)
    _write_eval_metrics(ctrl_eval, brier=0.145, pair_acc=0.53, auc=0.57)

    _write_phase_c_summary(
        phase_c_logs / "c_ref_log",
        generated_at="2026-03-05T01:00:00+00:00",
        group_id="C2_STRATEGYQA_CQR_FULL",
        run_prefix="c_ref",
        dataset="strategyqa",
        c2_eval_dir=c_eval,
    )
    _write_d6_summary(
        phase_d_logs / "d6_ctrl_log",
        generated_at="2026-03-05T02:00:00+00:00",
        group_id="D6_STRATEGYQA_SMOKE_RANKING_CTRL",
        run_prefix="d6_ctrl",
        dataset="strategyqa",
        c2_eval_dir=ctrl_eval,
    )
    _write_d6t_summary(
        phase_d6t_logs / "dt2_log",
        generated_at="2026-03-05T04:00:00+00:00",
        group_id="DT2_MATH_SHEPHERD_SEED3",
        run_prefix="dt2_ok",
        seed_rows=_seed_rows([(0.68, 0.70), (0.69, 0.68), (0.67, 0.69)]),
    )
    _write_d6t_summary(
        phase_d6t_logs / "dt4_log",
        generated_at="2026-03-05T05:00:00+00:00",
        group_id="DT4_MIXED_MS_PRM800K_SEED3",
        run_prefix="dt4_ok",
        seed_rows=_seed_rows([(0.70, 0.71), (0.69, 0.70), (0.68, 0.69)]),
    )

    rc = module.main(
        [
            "--phase-c-logs-root",
            str(phase_c_logs),
            "--phase-d-logs-root",
            str(phase_d_logs),
            "--phase-d6t-logs-root",
            str(phase_d6t_logs),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    report = json.loads((out_dir / "gate_report.json").read_text(encoding="utf-8"))
    assert report["decision"]["promotion_ready"] is False
    assert "missing_d6_gated_group" in report["decision"]["blocking_reasons"]
