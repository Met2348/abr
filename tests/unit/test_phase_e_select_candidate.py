"""验证 benchmark-aware candidate selector 的重复目录解析与选择逻辑。

这个测试覆盖两类风险：
1. wrapper 重复传入多个 `--suite-log-dirs` 时，selector 不能只吃到最后一组。
2. 候选选择必须基于 `seed_results.jsonl` 中的结构化 benchmark 指标，而不是偶然的目录顺序。

This test covers two risks:
1. when wrappers repeat `--suite-log-dirs`, the selector must not keep only
   the last occurrence;
2. candidate selection must follow the structured benchmark metrics in
   `seed_results.jsonl`, not accidental directory ordering.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_selector_module():
    """动态加载 benchmark-aware selector 脚本模块。

    Dynamically load the benchmark-aware selector script module.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_e_select_candidate.py"
    spec = importlib.util.spec_from_file_location("phase_e_select_candidate", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_suite_log_dir(
    *,
    root: Path,
    group_id: str,
    group_title: str,
    heldout_pair_acc: float,
    heldout_auc: float,
    heldout_ranking_score: float,
    processbench_gsm8k_pair_acc: float,
    processbench_gsm8k_auc: float,
    processbench_math_pair_acc: float,
    processbench_math_auc: float,
) -> Path:
    """构造 selector 所需的最小 suite log 目录。

    Create the smallest suite-log directory required by the selector.
    """
    suite_dir = root / group_id.lower()
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "final_summary.md").write_text(
        "\n".join(
            [
                "# Phase E Suite Summary",
                "",
                f"- group_id: {group_id}",
                f"- group_title: {group_title}",
                "- status: ok",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    seed_row = {
        "seed": 42,
        "value_run_dir": str(suite_dir / "fake_run"),
        "heldout_pair_acc": heldout_pair_acc,
        "heldout_auc": heldout_auc,
        "heldout_ranking_score": heldout_ranking_score,
        "benchmarks": {
            "processbench_gsm8k": {
                "pair_acc": processbench_gsm8k_pair_acc,
                "auc": processbench_gsm8k_auc,
            },
            "processbench_math": {
                "pair_acc": processbench_math_pair_acc,
                "auc": processbench_math_auc,
            },
        },
    }
    (suite_dir / "seed_results.jsonl").write_text(
        json.dumps(seed_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return suite_dir


def _attach_fake_run_manifest(
    *,
    suite_dir: Path,
    keep_best: bool,
    keep_final: bool,
) -> None:
    fake_run = suite_dir / "fake_run"
    fake_run.mkdir(parents=True, exist_ok=True)
    best_path = fake_run / "best_value_head.pt"
    final_path = fake_run / "final_value_head.pt"
    if keep_best:
        best_path.write_text("best\n", encoding="utf-8")
    if keep_final:
        final_path.write_text("final\n", encoding="utf-8")
    (fake_run / "manifest.json").write_text(
        json.dumps(
            {
                "output_files": {
                    "best_value_head": str(best_path),
                    "final_value_head": str(final_path),
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_parse_args_flattens_repeated_suite_log_dirs(tmp_path: Path) -> None:
    module = _load_selector_module()
    first = _write_suite_log_dir(
        root=tmp_path,
        group_id="E20_STAGEA_MS_ANCHOR_SEED3",
        group_title="E20",
        heldout_pair_acc=0.82,
        heldout_auc=0.79,
        heldout_ranking_score=0.80,
        processbench_gsm8k_pair_acc=0.52,
        processbench_gsm8k_auc=0.55,
        processbench_math_pair_acc=0.51,
        processbench_math_auc=0.54,
    )
    second = _write_suite_log_dir(
        root=tmp_path,
        group_id="E21_STAGEA_RPRM_ANCHOR_SEED3",
        group_title="E21",
        heldout_pair_acc=0.61,
        heldout_auc=0.59,
        heldout_ranking_score=0.60,
        processbench_gsm8k_pair_acc=0.50,
        processbench_gsm8k_auc=0.51,
        processbench_math_pair_acc=0.49,
        processbench_math_auc=0.50,
    )

    args = module.parse_args(
        [
            "--suite-log-dirs",
            str(first),
            "--suite-log-dirs",
            str(second),
            str(first),
        ]
    )

    assert args.suite_log_dirs == [first, second]


def test_main_selects_best_group_across_repeated_suite_log_dirs(tmp_path: Path) -> None:
    module = _load_selector_module()
    first = _write_suite_log_dir(
        root=tmp_path,
        group_id="E20_STAGEA_MS_ANCHOR_SEED3",
        group_title="E20",
        heldout_pair_acc=0.82,
        heldout_auc=0.79,
        heldout_ranking_score=0.80,
        processbench_gsm8k_pair_acc=0.56,
        processbench_gsm8k_auc=0.58,
        processbench_math_pair_acc=0.55,
        processbench_math_auc=0.57,
    )
    second = _write_suite_log_dir(
        root=tmp_path,
        group_id="E21_STAGEA_RPRM_ANCHOR_SEED3",
        group_title="E21",
        heldout_pair_acc=0.61,
        heldout_auc=0.59,
        heldout_ranking_score=0.60,
        processbench_gsm8k_pair_acc=0.50,
        processbench_gsm8k_auc=0.51,
        processbench_math_pair_acc=0.49,
        processbench_math_auc=0.50,
    )
    _attach_fake_run_manifest(suite_dir=first, keep_best=True, keep_final=True)
    _attach_fake_run_manifest(suite_dir=second, keep_best=True, keep_final=True)
    output_root = tmp_path / "candidates"

    exit_code = module.main(
        [
            "--run-name",
            "candidate_fixcheck",
            "--output-root",
            str(output_root),
            "--suite-log-dirs",
            str(second),
            "--suite-log-dirs",
            str(first),
        ]
    )

    assert exit_code == 0
    report_path = output_root / "candidate_fixcheck" / "candidate_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["selected_group"]["group_id"] == "E20_STAGEA_MS_ANCHOR_SEED3"
    assert len(payload["groups"]) == 2


def test_main_fails_when_best_checkpoint_missing_under_strict_policy(tmp_path: Path) -> None:
    module = _load_selector_module()
    first = _write_suite_log_dir(
        root=tmp_path,
        group_id="E20_STAGEA_MS_ANCHOR_SEED3",
        group_title="E20",
        heldout_pair_acc=0.82,
        heldout_auc=0.79,
        heldout_ranking_score=0.80,
        processbench_gsm8k_pair_acc=0.56,
        processbench_gsm8k_auc=0.58,
        processbench_math_pair_acc=0.55,
        processbench_math_auc=0.57,
    )
    _attach_fake_run_manifest(suite_dir=first, keep_best=False, keep_final=True)
    output_root = tmp_path / "candidates"

    try:
        module.main(
            [
                "--run-name",
                "candidate_checkpoint_resolution",
                "--output-root",
                str(output_root),
                "--suite-log-dirs",
                str(first),
            ]
        )
    except FileNotFoundError as exc:
        assert "Missing best_value_head.pt" in str(exc)
    else:
        raise AssertionError("strict candidate selection should fail when best checkpoint is missing")


def test_main_can_fallback_to_final_when_explicitly_requested(tmp_path: Path) -> None:
    module = _load_selector_module()
    first = _write_suite_log_dir(
        root=tmp_path,
        group_id="E20_STAGEA_MS_ANCHOR_SEED3",
        group_title="E20",
        heldout_pair_acc=0.82,
        heldout_auc=0.79,
        heldout_ranking_score=0.80,
        processbench_gsm8k_pair_acc=0.56,
        processbench_gsm8k_auc=0.58,
        processbench_math_pair_acc=0.55,
        processbench_math_auc=0.57,
    )
    _attach_fake_run_manifest(suite_dir=first, keep_best=False, keep_final=True)
    output_root = tmp_path / "candidates"

    exit_code = module.main(
        [
            "--run-name",
            "candidate_checkpoint_resolution",
            "--output-root",
            str(output_root),
            "--checkpoint-missing-policy",
            "fallback_final",
            "--suite-log-dirs",
            str(first),
        ]
    )

    assert exit_code == 0
    report_path = output_root / "candidate_checkpoint_resolution" / "candidate_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["selected_group"]["best_checkpoint_path"].endswith("final_value_head.pt")


def test_main_defaults_to_heldout_only_selection_policy(tmp_path: Path) -> None:
    module = _load_selector_module()
    heldout_better = _write_suite_log_dir(
        root=tmp_path,
        group_id="E30_HELDOUT_BETTER",
        group_title="E30",
        heldout_pair_acc=0.91,
        heldout_auc=0.90,
        heldout_ranking_score=0.89,
        processbench_gsm8k_pair_acc=0.50,
        processbench_gsm8k_auc=0.50,
        processbench_math_pair_acc=0.50,
        processbench_math_auc=0.50,
    )
    benchmark_better = _write_suite_log_dir(
        root=tmp_path,
        group_id="E31_BENCHMARK_BETTER",
        group_title="E31",
        heldout_pair_acc=0.85,
        heldout_auc=0.84,
        heldout_ranking_score=0.83,
        processbench_gsm8k_pair_acc=0.95,
        processbench_gsm8k_auc=0.95,
        processbench_math_pair_acc=0.95,
        processbench_math_auc=0.95,
    )
    _attach_fake_run_manifest(suite_dir=heldout_better, keep_best=True, keep_final=True)
    _attach_fake_run_manifest(suite_dir=benchmark_better, keep_best=True, keep_final=True)
    _attach_fake_run_manifest(suite_dir=heldout_better, keep_best=True, keep_final=True)
    _attach_fake_run_manifest(suite_dir=benchmark_better, keep_best=True, keep_final=True)
    output_root = tmp_path / "candidates"

    exit_code = module.main(
        [
            "--run-name",
            "candidate_heldout_only_default",
            "--output-root",
            str(output_root),
            "--suite-log-dirs",
            str(heldout_better),
            str(benchmark_better),
        ]
    )

    assert exit_code == 0
    report_path = output_root / "candidate_heldout_only_default" / "candidate_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["selection_policy"] == "heldout_only"
    assert payload["selected_group"]["group_id"] == "E30_HELDOUT_BETTER"
