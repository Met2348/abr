from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module(script_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_frontier_row_flags_low_math_and_missing_local(tmp_path: Path) -> None:
    module = _load_module("phase_e_analyze_frontier_matrix.py")
    run_dir = tmp_path / "run"
    same_dir = tmp_path / "same"
    gsm_dir = tmp_path / "gsm"
    math_dir = tmp_path / "math"
    for path in (run_dir, same_dir, gsm_dir, math_dir):
        path.mkdir()

    (run_dir / "eval_metrics.json").write_text(
        json.dumps({"eval_pairs": {"pair_accuracy": 0.6, "auc": 0.58}}),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "input_files": {"train_pairs_jsonl": "pairs/train_pairs.jsonl"},
                "train_config": {
                    "objective_mode": "joint",
                    "ranking_target_space": "score",
                    "pair_weight_mode": "none",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "recipe_risk.json").write_text(
        json.dumps({"num_pairs": 100, "by_pair_semantics": {"sibling_branch": 100}}),
        encoding="utf-8",
    )
    (run_dir / "value_head_config.json").write_text(
        json.dumps({"architecture": "mlp"}),
        encoding="utf-8",
    )
    (same_dir / "metrics.json").write_text(
        json.dumps(
            {
                "prompt_pool_top1_accuracy": 0.8,
                "local_first_bad_edge_accuracy": 0.82,
                "local_safe_vs_bad_pair_accuracy": 0.81,
            }
        ),
        encoding="utf-8",
    )
    metric_payload = {
        "metrics": {
            "pair_auc_good_vs_bad": 0.55,
            "processbench_f1": 0.61,
            "first_error_edge_accuracy": 0.57,
        }
    }
    (gsm_dir / "summary.json").write_text(json.dumps(metric_payload), encoding="utf-8")
    (math_dir / "summary.json").write_text(
        json.dumps({"metrics": {"pair_auc_good_vs_bad": 0.56, "processbench_f1": 0.64, "first_error_edge_accuracy": 0.58}}),
        encoding="utf-8",
    )

    row = module._extract_frontier_row(
        label="bad_frontier",
        run_dir=run_dir,
        samefamily_dir=same_dir,
        gsm_dir=gsm_dir,
        math_dir=math_dir,
    )

    assert "math_f1_low:0.640" in row.findings
    assert "samefamily_first_bad_low:0.820" in row.findings
    assert "no_local_first_bad_training_semantics" in row.findings
