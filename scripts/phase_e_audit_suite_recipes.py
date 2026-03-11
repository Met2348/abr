#!/usr/bin/env python3
"""Audit Phase E suite scripts for risky or ambiguous trainer recipes.

Why this script exists
----------------------
Phase E accumulated many historical suite scripts. Some are safe, some are
legacy reproductions, and some silently rely on trainer defaults that are now
known to be dangerous on mixed-semantics data.

This audit keeps that risk visible by scanning shell suites for:
1. hard-coded `logit + confidence_semantic + ranking_score`,
2. dangerous defaults via `${VAR:-...}`,
3. missing `--recipe-risk-policy`,
4. missing explicit `step_label_pair_mode` where step-label sources are used.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass
class AuditFinding:
    path: str
    severity: str
    code: str
    message: str


def _iter_suite_paths(root: Path) -> Iterable[Path]:
    for path in sorted((root / "scripts").glob("run_phase_e_*suite.sh")):
        if path.is_file():
            yield path


def _add(finding_list: list[AuditFinding], path: Path, severity: str, code: str, message: str) -> None:
    finding_list.append(
        AuditFinding(
            path=str(path),
            severity=str(severity),
            code=str(code),
            message=str(message),
        )
    )


def audit_suite(path: Path) -> list[AuditFinding]:
    text = path.read_text(encoding="utf-8")
    findings: list[AuditFinding] = []
    if "phase_e_train_value.py" not in text:
        return findings

    if "--recipe-risk-policy" not in text:
        _add(
            findings,
            path,
            "high",
            "MISSING_RECIPE_RISK_POLICY",
            "suite invokes phase_e_train_value.py but does not pass --recipe-risk-policy explicitly",
        )

    hardcoded_bad = (
        "--ranking-target-space logit" in text
        and "--pair-weight-mode confidence_semantic" in text
        and "--checkpoint-selection-metric ranking_score" in text
    )
    if hardcoded_bad:
        _add(
            findings,
            path,
            "critical",
            "HARDCODED_ANTI_PATTERN_G",
            "suite hard-codes logit + confidence_semantic + ranking_score",
        )

    if re.search(r'ranking-target-space\s+"\$\{[^}]+:-logit\}"', text):
        _add(
            findings,
            path,
            "high",
            "DANGEROUS_DEFAULT_LOGIT",
            "suite defaults ranking-target-space to logit",
        )
    if re.search(r'pair-weight-mode\s+"\$\{[^}]+:-confidence_semantic\}"', text):
        _add(
            findings,
            path,
            "high",
            "DANGEROUS_DEFAULT_CONFIDENCE_SEMANTIC",
            "suite defaults pair-weight-mode to confidence_semantic",
        )
    if re.search(r'checkpoint-selection-metric\s+"\$\{[^}]+:-ranking_score\}"', text):
        _add(
            findings,
            path,
            "medium",
            "DANGEROUS_DEFAULT_RANKING_SCORE",
            "suite defaults checkpoint-selection-metric to ranking_score",
        )

    if "phase_e_prepare_pairs.py" in text or "phase_e_prepare_mathshepherd_terminal_anchor_pairs.py" in text:
        if "step-label-pair-mode" not in text and "STEP_LABEL_PAIR_MODE" not in text and "MS_STEP_LABEL_PAIR_MODE" not in text:
            _add(
                findings,
                path,
                "medium",
                "MISSING_STEP_LABEL_PAIR_MODE",
                "step-label pair prep is used but step_label_pair_mode is not explicit",
            )
    return findings


def render_markdown(findings: list[AuditFinding]) -> str:
    lines = [
        "# Phase E Suite Recipe Audit",
        "",
        f"- total_findings: {len(findings)}",
        "",
        "| severity | code | path | message |",
        "|---|---|---|---|",
    ]
    for row in findings:
        lines.append(
            f"| `{row.severity}` | `{row.code}` | `{row.path}` | {row.message} |"
        )
    if not findings:
        lines.append("| `info` | `NO_FINDINGS` | n/a | No risky Phase E suite recipe patterns found. |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Phase E suite scripts for risky trainer recipes.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_audits"),
    )
    parser.add_argument(
        "--run-name",
        default="phase_e_suite_recipe_audit",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_root).resolve() / str(args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    findings: list[AuditFinding] = []
    for suite_path in _iter_suite_paths(repo_root):
        findings.extend(audit_suite(suite_path))

    findings_json = [asdict(item) for item in findings]
    (output_dir / "findings.json").write_text(
        json.dumps(findings_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        render_markdown(findings),
        encoding="utf-8",
    )
    print(f"findings_json: {output_dir / 'findings.json'}")
    print(f"summary_md   : {output_dir / 'summary.md'}")
    print(f"num_findings : {len(findings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
