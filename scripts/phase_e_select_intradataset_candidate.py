#!/usr/bin/env python3
"""Select the best same-source Phase E candidate under an ACC90 objective.

这个脚本只回答一个更窄的问题：在单个数据集自己的 held-out pair 上，
当前 value head 能不能把 chosen / rejected 分得足够开。
This script only answers a narrower question: on one dataset's own held-out
pairs, can the current value head separate chosen from rejected strongly enough.

为什么这个文件存在：
1. 旧的 `phase_e_select_candidate.py` 主要面向 benchmark-native trust。
2. 对当前 `ACC90` 分支来说，这个目标太宽了。
Why this file exists:
1. The older `phase_e_select_candidate.py` mainly targets benchmark-native trust.
2. That target is too broad for the current `ACC90` branch.

这个脚本只问三件事：
1. 单数据集 held-out discriminability 是否足够高？
2. 多个 recipe 里哪个 same-source candidate 最强？
3. 有没有组真正通过了显式 `ACC90` gate？
This script only asks three things:
1. Is single-dataset held-out discriminability high enough?
2. Which same-source candidate is strongest across recipe sweeps?
3. Did any group actually clear the explicit `ACC90` gate?
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SeedRow:
    seed: int
    value_run_dir: str
    heldout_pair_acc: float
    heldout_auc: float
    heldout_ranking_score: float


@dataclass(slots=True)
class GroupSummary:
    group_id: str
    group_title: str
    status: str
    suite_log_dir: str
    summary_path: str
    num_seeds: int
    mean_pair_acc: float
    mean_auc: float
    mean_ranking_score: float
    std_pair_acc: float | None
    std_auc: float | None
    worst_pair_acc: float
    worst_auc: float
    gate_pass: bool
    trust_score: float
    best_seed: int
    best_checkpoint_path: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select same-source Phase E candidates under a held-out ACC90 objective."
    )
    # 修复点：wrapper 会重复传入多次 `--suite-log-dirs DIR`。
    # 这里用 `append + nargs='+'` 接住所有出现，再在 parse 阶段展开。
    # Bug fix: the wrapper repeats `--suite-log-dirs DIR` multiple times.
    # We accept all occurrences via `append + nargs='+'` and flatten later.
    parser.add_argument(
        "--suite-log-dirs",
        type=Path,
        nargs="+",
        action="append",
        required=True,
    )
    parser.add_argument("--run-name", default="phase_e_intradataset_candidate")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_candidates"),
    )
    parser.add_argument("--min-mean-pair-acc", type=float, default=0.90)
    parser.add_argument("--min-mean-auc", type=float, default=0.90)
    parser.add_argument("--min-worst-pair-acc", type=float, default=0.85)
    parser.add_argument("--min-worst-auc", type=float, default=0.85)
    parser.add_argument("--max-pair-std", type=float, default=0.05)
    parser.add_argument("--max-auc-std", type=float, default=0.05)
    parser.add_argument(
        "--checkpoint-missing-policy",
        choices=["fail", "fallback_final"],
        default="fail",
        help=(
            "How to handle a missing best_value_head.pt when promoting an intradataset candidate. "
            "`fail` is the safe default; `fallback_final` is only for legacy diagnostics."
        ),
    )
    return parser


def _flatten_suite_log_dirs(grouped_dirs: list[list[Path]]) -> list[Path]:
    """展开并去重重复传入的 suite 目录列表。

    关键输入：
    - `grouped_dirs`: argparse 在 `append + nargs='+'` 下返回的嵌套列表。

    关键输出：
    - 按首次出现顺序展开后的 `Path` 列表。

    边界行为：
    - 重复目录会被去重，避免同一 group 被重复统计。

    简短示例：
    - `[[Path("a")], [Path("b"), Path("c")], [Path("a")]]`
      -> `[Path("a"), Path("b"), Path("c")]`

    Flatten and deduplicate repeated suite-log-dir groups.

    Key inputs:
    - `grouped_dirs`: nested lists returned by argparse under
      `append + nargs='+'`.

    Key outputs:
    - a flattened `Path` list that preserves first-seen order.

    Edge behavior:
    - duplicate directories are removed so one group is not counted twice.

    Short example:
    - `[[Path("a")], [Path("b"), Path("c")], [Path("a")]]`
      -> `[Path("a"), Path("b"), Path("c")]`
    """
    flattened: list[Path] = []
    seen: set[str] = set()
    for dir_group in grouped_dirs:
        for suite_dir in dir_group:
            suite_dir_key = str(suite_dir)
            if suite_dir_key in seen:
                continue
            seen.add(suite_dir_key)
            flattened.append(suite_dir)
    return flattened


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    args.suite_log_dirs = _flatten_suite_log_dirs(args.suite_log_dirs)
    for suite_dir in args.suite_log_dirs:
        if not suite_dir.exists():
            raise FileNotFoundError(f"Missing suite log dir: {suite_dir}")
    return args


def _parse_summary_header(summary_path: Path) -> dict[str, str]:
    text = summary_path.read_text(encoding="utf-8")

    def grab(key: str) -> str:
        match = re.search(rf"^- {re.escape(key)}: (.+)$", text, flags=re.MULTILINE)
        return match.group(1).strip() if match else ""

    return {
        "group_id": grab("group_id"),
        "group_title": grab("group_title"),
        "status": grab("status"),
    }


def _load_seed_rows(path: Path) -> list[SeedRow]:
    rows: list[SeedRow] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        record = json.loads(raw)
        rows.append(
            SeedRow(
                seed=int(record["seed"]),
                value_run_dir=str(record["value_run_dir"]),
                heldout_pair_acc=float(record["heldout_pair_acc"]),
                heldout_auc=float(record["heldout_auc"]),
                heldout_ranking_score=float(record.get("heldout_ranking_score", 0.0)),
            )
        )
    if not rows:
        raise ValueError(f"No seed rows found in {path}")
    return rows


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def _std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return None
    return float(statistics.pstdev(values))


def _resolve_best_checkpoint_path(value_run_dir: Path, *, missing_policy: str) -> tuple[str, list[str]]:
    """Resolve the checkpoint path this report should publish.

    English
    -------
    Older Phase E runs are not uniform:
    1. some keep `best_value_head.pt`,
    2. some only retain `final_value_head.pt`,
    3. and some publish those paths through `manifest.json`.

    中文
    ----
    老的 Phase E intradataset 运行并不统一：
    1. 有些保留 `best_value_head.pt`，
    2. 有些只保留 `final_value_head.pt`，
    3. 还有些需要从 `manifest.json` 里解析真实路径。

    关键策略变化是：默认不再把缺失的 `best` 静默回退到 `final`。
    只有显式传 `fallback_final` 才允许做 legacy 兼容。
    """
    notes: list[str] = []
    if missing_policy not in {"fail", "fallback_final"}:
        raise ValueError(f"Unsupported --checkpoint-missing-policy: {missing_policy!r}")
    manifest_path = value_run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_files = dict(manifest.get("output_files", {}))
        best_path_text = str(output_files.get("best_value_head") or "").strip()
        if best_path_text:
            best_path = Path(best_path_text)
            if best_path.exists():
                return str(best_path), notes
        final_path_text = str(output_files.get("final_value_head") or "").strip()
        if final_path_text and missing_policy == "fallback_final":
            final_path = Path(final_path_text)
            if final_path.exists():
                notes.append("best checkpoint missing; resolved best_checkpoint_path to final_value_head")
                return str(final_path), notes
    best_path = value_run_dir / "best_value_head.pt"
    if best_path.exists():
        return str(best_path), notes
    final_path = value_run_dir / "final_value_head.pt"
    if final_path.exists() and missing_policy == "fallback_final":
        notes.append("best checkpoint missing; resolved best_checkpoint_path to final_value_head")
        return str(final_path), notes
    raise FileNotFoundError(
        "Missing best_value_head.pt for intradataset promotion under strict policy: "
        f"{value_run_dir}. Rerun with --checkpoint-missing-policy fallback_final "
        "only for legacy diagnostics."
    )


def _build_group_summary(suite_dir: Path, args: argparse.Namespace) -> GroupSummary:
    summary_path = suite_dir / "final_summary.md"
    rows_path = suite_dir / "seed_results.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing final_summary.md: {summary_path}")
    if not rows_path.exists():
        raise FileNotFoundError(f"Missing seed_results.jsonl: {rows_path}")

    header = _parse_summary_header(summary_path)
    seed_rows = _load_seed_rows(rows_path)
    pair_values = [row.heldout_pair_acc for row in seed_rows]
    auc_values = [row.heldout_auc for row in seed_rows]
    ranking_values = [row.heldout_ranking_score for row in seed_rows]
    std_pair = _std(pair_values)
    std_auc = _std(auc_values)
    worst_pair = min(pair_values)
    worst_auc = min(auc_values)
    notes: list[str] = []

    gate_pass = True
    if _mean(pair_values) < float(args.min_mean_pair_acc):
        gate_pass = False
        notes.append("mean held-out pair_acc below ACC90 target")
    if _mean(auc_values) < float(args.min_mean_auc):
        gate_pass = False
        notes.append("mean held-out auc below threshold")
    if worst_pair < float(args.min_worst_pair_acc):
        gate_pass = False
        notes.append("worst-seed held-out pair_acc below threshold")
    if worst_auc < float(args.min_worst_auc):
        gate_pass = False
        notes.append("worst-seed held-out auc below threshold")
    if std_pair is not None and std_pair > float(args.max_pair_std):
        gate_pass = False
        notes.append("held-out pair_acc std above threshold")
    if std_auc is not None and std_auc > float(args.max_auc_std):
        gate_pass = False
        notes.append("held-out auc std above threshold")

    best_row = max(
        seed_rows,
        key=lambda row: (
            float(row.heldout_pair_acc),
            float(row.heldout_auc),
            float(row.heldout_ranking_score),
        ),
    )
    best_checkpoint_path, checkpoint_notes = _resolve_best_checkpoint_path(
        Path(best_row.value_run_dir),
        missing_policy=str(args.checkpoint_missing_policy),
    )
    notes.extend(f"best_seed={int(best_row.seed)}: {note}" for note in checkpoint_notes)
    trust_score = float(
        0.50 * _mean(pair_values)
        + 0.30 * _mean(auc_values)
        + 0.10 * (1.0 - min(float(std_pair or 0.0), 1.0))
        + 0.10 * (1.0 - min(float(std_auc or 0.0), 1.0))
    )
    return GroupSummary(
        group_id=str(header.get("group_id") or suite_dir.name),
        group_title=str(header.get("group_title") or ""),
        status=str(header.get("status") or "unknown"),
        suite_log_dir=str(suite_dir),
        summary_path=str(summary_path),
        num_seeds=len(seed_rows),
        mean_pair_acc=_mean(pair_values),
        mean_auc=_mean(auc_values),
        mean_ranking_score=_mean(ranking_values),
        std_pair_acc=std_pair,
        std_auc=std_auc,
        worst_pair_acc=worst_pair,
        worst_auc=worst_auc,
        gate_pass=gate_pass,
        trust_score=trust_score,
        best_seed=int(best_row.seed),
        best_checkpoint_path=best_checkpoint_path,
        notes=notes,
    )


def _render_markdown(
    *,
    run_name: str,
    output_json: Path,
    group_summaries: list[GroupSummary],
    selected_group: GroupSummary,
) -> str:
    lines = [
        "# Phase E Intradataset Candidate Report",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- run_name: {run_name}",
        f"- report_json: `{output_json}`",
        "",
        "## Selected Candidate",
        "",
        f"- group_id: `{selected_group.group_id}`",
        f"- best_seed: `{selected_group.best_seed}`",
        f"- best_checkpoint_path: `{selected_group.best_checkpoint_path}`",
        f"- gate_pass: `{int(selected_group.gate_pass)}`",
        f"- trust_score: `{selected_group.trust_score:.6f}`",
        "",
        "## Group Comparison",
        "",
        "| group_id | gate_pass | mean_pair_acc | mean_auc | worst_pair_acc | worst_auc | std_pair_acc | std_auc | trust_score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in sorted(group_summaries, key=lambda item: item.trust_score, reverse=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.group_id,
                    str(int(summary.gate_pass)),
                    f"{summary.mean_pair_acc:.4f}",
                    f"{summary.mean_auc:.4f}",
                    f"{summary.worst_pair_acc:.4f}",
                    f"{summary.worst_auc:.4f}",
                    f"{float(summary.std_pair_acc or 0.0):.4f}",
                    f"{float(summary.std_auc or 0.0):.4f}",
                    f"{summary.trust_score:.4f}",
                ]
            )
            + " |"
        )
        lines.append(f"Path: `{summary.summary_path}`")
        if summary.notes:
            lines.append(f"Notes: {', '.join(summary.notes)}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_root) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [_build_group_summary(suite_dir, args) for suite_dir in args.suite_log_dirs]
    selected = max(summaries, key=lambda item: item.trust_score)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "thresholds": {
            "min_mean_pair_acc": float(args.min_mean_pair_acc),
            "min_mean_auc": float(args.min_mean_auc),
            "min_worst_pair_acc": float(args.min_worst_pair_acc),
            "min_worst_auc": float(args.min_worst_auc),
            "max_pair_std": float(args.max_pair_std),
            "max_auc_std": float(args.max_auc_std),
        },
        "selected_group": selected.to_dict(),
        "group_summaries": [summary.to_dict() for summary in summaries],
    }
    json_path = output_dir / "candidate_report.json"
    md_path = output_dir / "candidate_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(
        _render_markdown(
            run_name=str(args.run_name),
            output_json=json_path,
            group_summaries=summaries,
            selected_group=selected,
        )
        + "\n",
        encoding="utf-8",
    )
    print("=" * 88)
    print("Phase E: Select Intradataset Candidate")
    print("=" * 88)
    print(f"run_name          : {args.run_name}")
    print(f"selected_group    : {selected.group_id}")
    print(f"best_seed         : {selected.best_seed}")
    print(f"best_checkpoint   : {selected.best_checkpoint_path}")
    print(f"gate_pass         : {int(selected.gate_pass)}")
    print(f"trust_score       : {selected.trust_score:.6f}")
    print(f"report_json       : {json_path}")
    print(f"report_md         : {md_path}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
