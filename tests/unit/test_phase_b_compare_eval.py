"""Unit tests for the Phase B pre/post benchmark comparison script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_compare_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_b_compare_eval.py"
    spec = importlib.util.spec_from_file_location("phase_b_compare_eval", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_metrics(
    path: Path,
    *,
    n_total: int,
    n_parseable: int,
    accuracy: float,
    parse_error_rate: float,
    accuracy_parseable: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "n_total": n_total,
                "n_parseable": n_parseable,
                "accuracy": accuracy,
                "parse_error_rate": parse_error_rate,
                "accuracy_parseable": accuracy_parseable,
                "generation_stats": {
                    "sample_per_second": 10.0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_compare_eval_builds_positive_gain_report(tmp_path: Path) -> None:
    module = _load_compare_module()
    before_val = tmp_path / "before_val.json"
    after_val = tmp_path / "after_val.json"
    before_test = tmp_path / "before_test.json"
    after_test = tmp_path / "after_test.json"

    _write_metrics(
        before_val,
        n_total=100,
        n_parseable=100,
        accuracy=0.60,
        parse_error_rate=0.00,
        accuracy_parseable=0.60,
    )
    _write_metrics(
        after_val,
        n_total=100,
        n_parseable=100,
        accuracy=0.65,
        parse_error_rate=0.00,
        accuracy_parseable=0.65,
    )
    _write_metrics(
        before_test,
        n_total=50,
        n_parseable=50,
        accuracy=0.40,
        parse_error_rate=0.00,
        accuracy_parseable=0.40,
    )
    _write_metrics(
        after_test,
        n_total=50,
        n_parseable=50,
        accuracy=0.44,
        parse_error_rate=0.00,
        accuracy_parseable=0.44,
    )

    args = module.parse_args(
        [
            "--dataset",
            "strategyqa",
            "--compare",
            "validation",
            str(before_val),
            str(after_val),
            "--compare",
            "test",
            str(before_test),
            str(after_test),
        ]
    )
    pairs = module._load_pairs(args)
    payload = module._build_report_payload(args, pairs)

    assert payload["headline"]["direction"] == "improved"
    assert payload["aggregate"]["n_total"] == 150
    assert payload["aggregate"]["before_n_correct"] == 80
    assert payload["aggregate"]["after_n_correct"] == 87
    assert abs(payload["aggregate"]["delta_accuracy"] - (7 / 150)) < 1e-9


def test_compare_eval_rejects_mismatched_totals(tmp_path: Path) -> None:
    module = _load_compare_module()
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    _write_metrics(
        before_path,
        n_total=10,
        n_parseable=10,
        accuracy=0.5,
        parse_error_rate=0.0,
        accuracy_parseable=0.5,
    )
    _write_metrics(
        after_path,
        n_total=11,
        n_parseable=11,
        accuracy=0.5,
        parse_error_rate=0.0,
        accuracy_parseable=0.5,
    )

    args = module.parse_args(
        [
            "--dataset",
            "gsm8k",
            "--compare",
            "validation",
            str(before_path),
            str(after_path),
        ]
    )

    try:
        module._load_pairs(args)
    except ValueError as exc:
        assert "mismatched totals" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched totals")
