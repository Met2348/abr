"""验证 intradataset candidate selector 的重复目录解析与选择逻辑。

这个测试文件专门覆盖一次真实暴露过的 bug：
wrapper 重复传入多个 `--suite-log-dirs` 时，selector 过去只看到了最后一个。

This test file covers a real bug that was observed in practice:
when the wrapper repeated multiple `--suite-log-dirs` flags, the selector used
to see only the last one.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_selector_module():
    """动态加载 selector 脚本模块，避免把 `scripts/` 当常规包导入。

    这里沿用仓库里其他脚本单测的加载方式，确保测试环境和真实 CLI 路径一致。

    Dynamically load the selector script module instead of importing from
    `scripts/` as a normal package.  This mirrors other script tests in the
    repository and keeps the test path close to the real CLI path.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_e_select_intradataset_candidate.py"
    spec = importlib.util.spec_from_file_location("phase_e_select_intradataset_candidate", script_path)
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
    pair_acc: float,
    auc: float,
    ranking_score: float,
) -> Path:
    """构造一个最小可用的 suite log 目录。

    目录里只写 selector 真正依赖的两个文件：
    - `final_summary.md`
    - `seed_results.jsonl`

    Create the smallest valid suite-log directory the selector needs:
    - `final_summary.md`
    - `seed_results.jsonl`
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
        "heldout_pair_acc": pair_acc,
        "heldout_auc": auc,
        "heldout_ranking_score": ranking_score,
    }
    (suite_dir / "seed_results.jsonl").write_text(
        json.dumps(seed_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return suite_dir


def test_parse_args_flattens_repeated_suite_log_dirs(tmp_path: Path) -> None:
    module = _load_selector_module()
    first = _write_suite_log_dir(
        root=tmp_path,
        group_id="E41_MS_ACC90_MLP_RANK_SEED3",
        group_title="E41",
        pair_acc=0.96,
        auc=0.89,
        ranking_score=0.93,
    )
    second = _write_suite_log_dir(
        root=tmp_path,
        group_id="E45_PRMBENCH_ACC90_MLP_RANK_SEED3",
        group_title="E45",
        pair_acc=0.94,
        auc=0.83,
        ranking_score=0.89,
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
        group_id="E41_MS_ACC90_MLP_RANK_SEED3",
        group_title="E41",
        pair_acc=0.96,
        auc=0.89,
        ranking_score=0.93,
    )
    second = _write_suite_log_dir(
        root=tmp_path,
        group_id="E45_PRMBENCH_ACC90_MLP_RANK_SEED3",
        group_title="E45",
        pair_acc=0.94,
        auc=0.83,
        ranking_score=0.89,
    )
    third = _write_suite_log_dir(
        root=tmp_path,
        group_id="E48_RPRM_ACC90_MLP_RANK_SEED3",
        group_title="E48",
        pair_acc=0.43,
        auc=0.46,
        ranking_score=0.45,
    )
    output_root = tmp_path / "candidates"

    exit_code = module.main(
        [
            "--run-name",
            "candidate_fixcheck",
            "--output-root",
            str(output_root),
            "--suite-log-dirs",
            str(first),
            "--suite-log-dirs",
            str(second),
            "--suite-log-dirs",
            str(third),
        ]
    )

    assert exit_code == 0
    report_path = output_root / "candidate_fixcheck" / "candidate_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["selected_group"]["group_id"] == "E41_MS_ACC90_MLP_RANK_SEED3"
    assert len(payload["group_summaries"]) == 3


def test_main_resolves_best_checkpoint_path_to_final_when_best_missing(tmp_path: Path) -> None:
    module = _load_selector_module()
    first = _write_suite_log_dir(
        root=tmp_path,
        group_id="E41_MS_ACC90_MLP_RANK_SEED3",
        group_title="E41",
        pair_acc=0.96,
        auc=0.89,
        ranking_score=0.93,
    )
    fake_run = first / "fake_run"
    fake_run.mkdir(parents=True, exist_ok=True)
    final_path = fake_run / "final_value_head.pt"
    final_path.write_text("final\n", encoding="utf-8")
    (fake_run / "manifest.json").write_text(
        json.dumps(
            {
                "output_files": {
                    "best_value_head": str(fake_run / "best_value_head.pt"),
                    "final_value_head": str(final_path),
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "candidates"

    exit_code = module.main(
        [
            "--run-name",
            "candidate_checkpoint_resolution",
            "--output-root",
            str(output_root),
            "--suite-log-dirs",
            str(first),
        ]
    )

    assert exit_code == 0
    report_path = output_root / "candidate_checkpoint_resolution" / "candidate_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["selected_group"]["best_checkpoint_path"].endswith("final_value_head.pt")
