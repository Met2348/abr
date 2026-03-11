"""Unit tests for scripts/phase_e_diagnose_rl_promotion.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_e_diagnose_rl_promotion.py"
    spec = importlib.util.spec_from_file_location("phase_e_diagnose_rl_promotion", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_processbench_example(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "ex_bad",
                    "generator": "stub",
                    "problem": "Problem bad",
                    "steps": ["good0", "good1", "bad2", "bad3"],
                    "label": 2,
                    "final_answer_correct": False,
                },
                {
                    "id": "ex_good",
                    "generator": "stub",
                    "problem": "Problem good",
                    "steps": ["good0", "good1", "good2"],
                    "label": -1,
                    "final_answer_correct": True,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_samefamily_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps({"value_run_dir": "assets/artifacts/phase_e_runs/example"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "prompt_pool_top1_accuracy": 0.95,
                "local_first_bad_edge_accuracy": 0.93,
                "pressure_curve": [
                    {"subset_size": 8, "top1_accuracy": 0.94},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "prompt_rows.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt_id": "p0",
                        "num_candidates": 4,
                        "gold_top_candidate_ids": ["c0"],
                    },
                    ensure_ascii=False,
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_processbench_eval_dir(path: Path, processbench_path: Path, *, weak_terminal: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "benchmark_path": str(processbench_path),
                "metrics": {
                    "pair_auc_good_vs_bad": 0.71,
                    "pair_accuracy_good_vs_bad": 0.70,
                    "first_error_edge_accuracy": 0.66,
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    terminal_scores = [0.1, 0.2, 0.15] if weak_terminal else [0.1, 0.2, 0.3]
    (path / "scored_rows.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "row_id": "ex_bad:prefix:0",
                        "example_id": "ex_bad",
                        "prefix_step_index": 0,
                        "label": 2,
                        "is_good_prefix": True,
                        "is_first_bad_prefix": False,
                        "score": 0.90,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "row_id": "ex_bad:prefix:1",
                        "example_id": "ex_bad",
                        "prefix_step_index": 1,
                        "label": 2,
                        "is_good_prefix": True,
                        "is_first_bad_prefix": False,
                        "score": 0.80,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "row_id": "ex_bad:prefix:2",
                        "example_id": "ex_bad",
                        "prefix_step_index": 2,
                        "label": 2,
                        "is_good_prefix": False,
                        "is_first_bad_prefix": True,
                        "score": 0.30,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "row_id": "ex_bad:prefix:3",
                        "example_id": "ex_bad",
                        "prefix_step_index": 3,
                        "label": 2,
                        "is_good_prefix": False,
                        "is_first_bad_prefix": False,
                        "score": 0.10,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "row_id": "ex_good:prefix:0",
                        "example_id": "ex_good",
                        "prefix_step_index": 0,
                        "label": -1,
                        "is_good_prefix": True,
                        "is_first_bad_prefix": False,
                        "score": terminal_scores[0],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "row_id": "ex_good:prefix:1",
                        "example_id": "ex_good",
                        "prefix_step_index": 1,
                        "label": -1,
                        "is_good_prefix": True,
                        "is_first_bad_prefix": False,
                        "score": terminal_scores[1],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "row_id": "ex_good:prefix:2",
                        "example_id": "ex_good",
                        "prefix_step_index": 2,
                        "label": -1,
                        "is_good_prefix": True,
                        "is_first_bad_prefix": False,
                        "score": terminal_scores[2],
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_phase_e_rl_promotion_marks_terminal_weak_candidate(tmp_path: Path) -> None:
    module = _load_script_module()
    samefamily_dir = tmp_path / "samefamily"
    pb_gsm_dir = tmp_path / "pb_gsm"
    pb_math_dir = tmp_path / "pb_math"
    processbench_path = tmp_path / "processbench.json"

    _write_samefamily_dir(samefamily_dir)
    _write_processbench_example(processbench_path)
    _write_processbench_eval_dir(pb_gsm_dir, processbench_path, weak_terminal=True)
    _write_processbench_eval_dir(pb_math_dir, processbench_path, weak_terminal=True)

    audit_row, benchmark_rows = module._run_one_audit(
        spec_text=f"cand|math_shepherd|{samefamily_dir}|{pb_gsm_dir}|{pb_math_dir}",
        thresholds={
            "samefamily_top1_min": 0.90,
            "samefamily_local_min": 0.90,
            "pressure_top1_min": 0.90,
            "benchmark_auc_min": 0.60,
            "first_edge_min": 0.60,
            "anygood_firstbad_min": 0.60,
            "good_laterbad_min": 0.60,
            "terminal_top1_min": 0.50,
            "terminal_gap_min": -0.05,
        },
    )

    assert audit_row["strict_rl_promotion_ready"] is False
    assert audit_row["assessment"] == "near_rl_ready_but_terminal_gap"
    assert "terminal_completion_weak" in audit_row["failure_tags"]
    assert len(benchmark_rows) == 2


def test_phase_e_rl_promotion_can_mark_clean_candidate(tmp_path: Path) -> None:
    module = _load_script_module()
    samefamily_dir = tmp_path / "samefamily"
    pb_gsm_dir = tmp_path / "pb_gsm"
    pb_math_dir = tmp_path / "pb_math"
    processbench_path = tmp_path / "processbench.json"

    _write_samefamily_dir(samefamily_dir)
    _write_processbench_example(processbench_path)
    _write_processbench_eval_dir(pb_gsm_dir, processbench_path, weak_terminal=False)
    _write_processbench_eval_dir(pb_math_dir, processbench_path, weak_terminal=False)

    audit_row, _ = module._run_one_audit(
        spec_text=f"cand|math_shepherd|{samefamily_dir}|{pb_gsm_dir}|{pb_math_dir}",
        thresholds={
            "samefamily_top1_min": 0.90,
            "samefamily_local_min": 0.90,
            "pressure_top1_min": 0.90,
            "benchmark_auc_min": 0.60,
            "first_edge_min": 0.60,
            "anygood_firstbad_min": 0.60,
            "good_laterbad_min": 0.60,
            "terminal_top1_min": 0.50,
            "terminal_gap_min": -0.05,
        },
    )

    assert audit_row["strict_rl_promotion_ready"] is True
    assert audit_row["assessment"] == "strict_rl_promotion_ready"
