#!/usr/bin/env python
"""Run experiment commands with reproducible logging and compact result tracking.

Why this file exists
--------------------
The project runs many long shell commands across Phase A/B/C/D. Manually copying
commands and outcomes into Markdown is error-prone and inconsistent.

This helper provides one stable workflow:
1. Execute one command.
2. Persist full raw terminal output into artifacts.
3. Extract only high-signal fields (accuracy, brier, auc, paths, etc.).
4. Auto-update two docs:
   - `docs/commands_to_run.md` (command-family catalog)
   - `docs/command_result.md` (per-run compact outcomes)

Design choices
--------------
- Logging is opt-in by wrapper usage, so users keep control.
- Raw logs are always kept for traceability.
- Markdown keeps only compact summaries (no progress-bar spam).
- Repeated runs of the same command-family are explicitly indexed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HIGH_SIGNAL_KEY_ORDER = [
    "accuracy",
    "parse_error_rate",
    "acc_parseable",
    "n_parseable",
    "brier_score",
    "ece",
    "pearson",
    "corr_pair_acc",
    "corr_auc",
    "teacher_cov_train",
    "teacher_cov_eval",
    "external_pairs",
    "external_pair_w",
    "gen_sample_rate",
    "gen_elapsed_sec",
    "global_step",
    "train_elapsed_sec",
]

JSON_CANDIDATE_KEYS = {
    "metrics_path",
    "train_metrics",
    "eval_metrics",
    "summary",
    "manifest",
}

NOISE_LINE_PATTERNS = [
    re.compile(r"\b\d+/\d+\b"),
    re.compile(r"\belapsed=\d+"),
    re.compile(r"\brate=\d+"),
    re.compile(r"\beta=\d+"),
]

KV_LINE_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_./-]*)\s*:\s*(.*?)\s*$")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for command execution and log/doc behavior."""
    parser = argparse.ArgumentParser(
        description=(
            "Run one command, persist full logs, and auto-update reproducibility "
            "docs with compact key results."
        )
    )
    parser.add_argument(
        "--records-root",
        type=Path,
        default=Path("assets/artifacts/command_logs"),
        help="Directory for JSONL run records and per-run raw logs.",
    )
    parser.add_argument(
        "--commands-md",
        type=Path,
        default=Path("docs/commands_to_run.md"),
        help="Command catalog markdown output path.",
    )
    parser.add_argument(
        "--results-md",
        type=Path,
        default=Path("docs/command_result.md"),
        help="Per-run result markdown output path.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Optional run tag (repeatable). Example: --tag phase_c --tag smoke",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional short note attached to this run record.",
    )
    parser.add_argument(
        "--no-update-docs",
        action="store_true",
        help="Record JSON only; do not regenerate markdown outputs.",
    )
    parser.add_argument(
        "--max-families",
        type=int,
        default=120,
        help="Maximum command families rendered in commands markdown.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=240,
        help="Maximum run entries rendered in results markdown.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not mirror command stdout/stderr to this terminal.",
    )
    parser.add_argument(
        "--rebuild-docs-only",
        action="store_true",
        help=(
            "Do not execute a command; rebuild commands/results markdown from "
            "existing JSONL records only."
        ),
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to execute. Use '--' before command args.",
    )
    args = parser.parse_args()
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd and not args.rebuild_docs_only:
        parser.error("Missing command. Example: -- python -u scripts/phase_a_generate_and_eval.py ...")
    args.cmd = cmd
    return args


def _now_utc() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _iso_utc(ts: datetime) -> str:
    """Format UTC timestamp as ISO 8601 string."""
    return ts.isoformat()


def _sha12(text: str) -> str:
    """Stable 12-char SHA1 digest for compact IDs."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _normalize_command_for_family(command_raw: str) -> str:
    """Normalize volatile command fields to build command-family identity.

    Notes
    -----
    This keeps semantic command structure while masking fields that are
    typically run-specific (for example run name, run prefix, device id).
    """
    normalized = command_raw
    replacements = [
        (r"\bRUN_PREFIX=\S+", "RUN_PREFIX=<RUN_PREFIX>"),
        (r"\bCUDA_VISIBLE_DEVICES=\S+", "CUDA_VISIBLE_DEVICES=<GPU_SET>"),
        (r"(--run-name\s+)\S+", r"\1<RUN_NAME>"),
        (r"(--seed\s+)\S+", r"\1<SEED>"),
        (r"\b\d{8}T\d{6}Z\b", "<TS>"),
    ]
    for pattern, repl in replacements:
        normalized = re.sub(pattern, repl, normalized)
    return " ".join(normalized.split())


def _run_command(
    cmd_argv: list[str],
    log_path: Path,
    quiet: bool,
) -> tuple[int, list[str], float]:
    """Execute command, stream to log, and return code/lines/duration.

    Returns
    -------
    tuple[int, list[str], float]
        `(return_code, output_lines, duration_seconds)`.
    """
    start = time.time()
    output_lines: list[str] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        cmd_argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as fout:
        for line in process.stdout:
            fout.write(line)
            output_lines.append(line.rstrip("\n"))
            if not quiet:
                print(line, end="")

    return_code = process.wait()
    duration_sec = time.time() - start
    return return_code, output_lines, duration_sec


def _is_noise_line(line: str) -> bool:
    """Heuristic filter for progress-like lines."""
    if "\r" in line:
        return True
    lowered = line.lower()
    if "loading checkpoint shards" in lowered:
        return True
    return any(pattern.search(lowered) for pattern in NOISE_LINE_PATTERNS)


def _parse_compact_kv(output_lines: list[str]) -> dict[str, str]:
    """Extract compact key/value lines from command output.

    Expected format examples:
    - `accuracy         : 0.6000`
    - `metrics_path     : assets/artifacts/.../metrics.json`
    """
    parsed: dict[str, str] = {}
    for line in output_lines:
        if _is_noise_line(line):
            continue
        match = KV_LINE_PATTERN.match(line)
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        parsed[key] = value
    return parsed


def _resolve_json_candidate_paths(
    parsed_kv: dict[str, str],
    cwd: Path,
) -> dict[str, Path]:
    """Resolve JSON-like output paths collected from parsed key/value lines."""
    results: dict[str, Path] = {}
    for key, value in parsed_kv.items():
        if key not in JSON_CANDIDATE_KEYS and not value.endswith(".json"):
            continue
        raw = value.strip().strip("'").strip('"')
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (cwd / candidate).resolve()
        if candidate.exists() and candidate.is_file():
            results[key] = candidate
    return results


def _to_float(value: Any) -> float | None:
    """Convert value to float when possible, otherwise return None."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _summarize_metrics_blob(blob: dict[str, Any]) -> dict[str, float]:
    """Extract cross-phase high-signal metrics from one metrics-like JSON blob."""
    out: dict[str, float] = {}

    # Phase A style
    for key in ("accuracy", "parse_error_rate", "acc_parseable", "n_parseable"):
        if key in blob:
            value = _to_float(blob.get(key))
            if value is not None:
                out[key] = value

    # Phase C/D style calibration metrics
    calibration = blob.get("calibration")
    if isinstance(calibration, dict):
        mapping = {
            "brier_score": "brier_score",
            "expected_calibration_error": "ece",
            "pearson_corr": "pearson",
            "pearson": "pearson",
        }
        for src_key, dst_key in mapping.items():
            value = _to_float(calibration.get(src_key))
            if value is not None:
                out[dst_key] = value

    # Corruption ranking metrics
    corruption = blob.get("corruption")
    if isinstance(corruption, dict):
        mapping = {
            "pair_accuracy": "corr_pair_acc",
            "auc": "corr_auc",
            "teacher_coverage_train": "teacher_cov_train",
            "teacher_coverage_eval": "teacher_cov_eval",
        }
        for src_key, dst_key in mapping.items():
            value = _to_float(corruption.get(src_key))
            if value is not None:
                out[dst_key] = value

    # Some scripts write selected metrics at root level.
    for key in ("selected_brier", "selected_pearson", "corr_pair_acc", "corr_auc"):
        value = _to_float(blob.get(key))
        if value is not None:
            out[key] = value

    return out


def _load_json_summaries(candidate_paths: dict[str, Path]) -> dict[str, dict[str, float]]:
    """Load and summarize all discovered metrics-like JSON files."""
    summaries: dict[str, dict[str, float]] = {}
    for key, path in candidate_paths.items():
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(blob, dict):
            continue
        summary = _summarize_metrics_blob(blob)
        if summary:
            summaries[key] = summary
    return summaries


def _merge_high_signal(
    parsed_kv: dict[str, str],
    json_summaries: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Merge compact output fields and JSON-derived metrics into one summary map."""
    merged: dict[str, Any] = {}

    # Prefer strongly-typed values parsed from structured metrics JSON.
    for summary in json_summaries.values():
        for key, value in summary.items():
            merged[key] = value

    # Add known top-level fields from plain console output.
    for key in HIGH_SIGNAL_KEY_ORDER + ["metrics_path", "train_metrics", "eval_metrics", "run_dir"]:
        if key in parsed_kv and key not in merged:
            merged[key] = parsed_kv[key]

    # Include D4-specific fields frequently printed in summaries.
    for key in ("external_pairs", "external_pair_w", "teacher_cov_train", "teacher_cov_eval"):
        if key in parsed_kv and key not in merged:
            value = _to_float(parsed_kv[key])
            merged[key] = value if value is not None else parsed_kv[key]

    return merged


def _read_records(records_path: Path) -> list[dict[str, Any]]:
    """Load historical run records from JSONL."""
    if not records_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with records_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _append_record(records_path: Path, record: dict[str, Any]) -> None:
    """Append one run record to JSONL storage."""
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def _format_metric_value(value: Any) -> str:
    """Format metric-like value in compact markdown-friendly style."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _build_family_index(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate records by command-family for command catalog rendering."""
    families: dict[str, dict[str, Any]] = {}
    for rec in records:
        family_id = str(rec.get("command_family_id", "unknown"))
        item = families.setdefault(
            family_id,
            {
                "family_id": family_id,
                "runs": 0,
                "successes": 0,
                "failures": 0,
                "first_seen": rec.get("started_at_utc", ""),
                "last_seen": rec.get("started_at_utc", ""),
                "command_template": rec.get("command_family_template", ""),
                "tags": set(),
                "latest_key_results": {},
            },
        )
        item["runs"] += 1
        if rec.get("status") == "success":
            item["successes"] += 1
        else:
            item["failures"] += 1

        started = str(rec.get("started_at_utc", ""))
        if started and (not item["first_seen"] or started < item["first_seen"]):
            item["first_seen"] = started
        if started and (not item["last_seen"] or started > item["last_seen"]):
            item["last_seen"] = started
            item["latest_key_results"] = rec.get("key_results", {})
            if rec.get("command_family_template"):
                item["command_template"] = rec["command_family_template"]

        for tag in rec.get("tags", []):
            item["tags"].add(str(tag))

    rendered = list(families.values())
    for row in rendered:
        row["tags"] = sorted(row["tags"])
    rendered.sort(key=lambda x: x.get("last_seen", ""), reverse=True)
    return rendered


def _render_commands_markdown(
    families: list[dict[str, Any]],
    max_families: int,
) -> str:
    """Render `docs/commands_to_run.md` from command-family aggregates."""
    lines: list[str] = []
    lines.append("# Commands To Run")
    lines.append("")
    lines.append("Auto-generated by `scripts/experiment_command_logger.py`.")
    lines.append(
        "This file tracks command families (normalized signatures), not every raw run."
    )
    lines.append("")
    lines.append("| Family ID | Runs | Success | Failure | Last Seen (UTC) | Tags |")
    lines.append("| --- | ---: | ---: | ---: | --- | --- |")

    for fam in families[:max_families]:
        lines.append(
            "| `{family_id}` | {runs} | {successes} | {failures} | `{last_seen}` | {tags} |".format(
                family_id=fam["family_id"],
                runs=fam["runs"],
                successes=fam["successes"],
                failures=fam["failures"],
                last_seen=fam.get("last_seen", ""),
                tags=", ".join(fam.get("tags", [])) or "-",
            )
        )

    lines.append("")
    lines.append("## Family Details")
    lines.append("")
    for fam in families[:max_families]:
        lines.append(f"### `{fam['family_id']}`")
        lines.append(f"- first_seen_utc: `{fam.get('first_seen', '')}`")
        lines.append(f"- last_seen_utc: `{fam.get('last_seen', '')}`")
        lines.append(
            f"- status_count: success={fam.get('successes', 0)}, fail={fam.get('failures', 0)}"
        )
        lines.append("- command_template:")
        lines.append("```bash")
        lines.append(str(fam.get("command_template", "")).strip() or "<empty>")
        lines.append("```")
        key_results = fam.get("latest_key_results", {})
        if key_results:
            lines.append("- latest_key_results:")
            for key in HIGH_SIGNAL_KEY_ORDER:
                if key in key_results:
                    lines.append(f"  - `{key}`: `{_format_metric_value(key_results[key])}`")
            for key in ("metrics_path", "train_metrics", "eval_metrics", "run_dir"):
                if key in key_results:
                    lines.append(f"  - `{key}`: `{key_results[key]}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_results_markdown(
    records: list[dict[str, Any]],
    max_results: int,
) -> str:
    """Render `docs/command_result.md` with compact per-run summaries."""
    ordered = sorted(records, key=lambda x: str(x.get("started_at_utc", "")), reverse=True)
    lines: list[str] = []
    lines.append("# Command Results")
    lines.append("")
    lines.append("Auto-generated by `scripts/experiment_command_logger.py`.")
    lines.append("Only high-signal fields are shown; raw logs remain in artifacts.")
    lines.append("")
    lines.append("| Run ID | Time (UTC) | Status | Family | Attempt | Duration (s) | Key Results | Log |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | --- | --- |")

    for rec in ordered[:max_results]:
        key_results = rec.get("key_results", {})
        metric_parts: list[str] = []
        for key in HIGH_SIGNAL_KEY_ORDER:
            if key in key_results:
                metric_parts.append(f"{key}={_format_metric_value(key_results[key])}")
        if not metric_parts:
            metric_parts.append("-")
        lines.append(
            "| `{run_id}` | `{time}` | `{status}` | `{family}` | {attempt} | {dur:.2f} | {metrics} | `{log}` |".format(
                run_id=rec.get("run_id", ""),
                time=rec.get("started_at_utc", ""),
                status=rec.get("status", ""),
                family=rec.get("command_family_id", ""),
                attempt=int(rec.get("family_run_index", 0)),
                dur=float(rec.get("duration_sec", 0.0)),
                metrics="; ".join(metric_parts),
                log=rec.get("log_path", ""),
            )
        )

    lines.append("")
    lines.append("## Latest Runs (Detail)")
    lines.append("")
    for rec in ordered[:max_results]:
        lines.append(f"### `{rec.get('run_id', '')}`")
        lines.append(f"- time_utc: `{rec.get('started_at_utc', '')}`")
        lines.append(f"- status: `{rec.get('status', '')}` (code={rec.get('return_code', '')})")
        lines.append(f"- family_id: `{rec.get('command_family_id', '')}`")
        lines.append(f"- family_attempt: `{rec.get('family_run_index', '')}`")
        lines.append(f"- duration_sec: `{float(rec.get('duration_sec', 0.0)):.3f}`")
        lines.append(f"- cwd: `{rec.get('cwd', '')}`")
        lines.append(f"- log: `{rec.get('log_path', '')}`")
        if rec.get("tags"):
            lines.append(f"- tags: `{', '.join(rec.get('tags', []))}`")
        if rec.get("note"):
            lines.append(f"- note: {rec.get('note')}")
        lines.append("- command:")
        lines.append("```bash")
        lines.append(str(rec.get("command_raw", "")).strip())
        lines.append("```")
        key_results = rec.get("key_results", {})
        if key_results:
            lines.append("- key_results:")
            for key in HIGH_SIGNAL_KEY_ORDER:
                if key in key_results:
                    lines.append(f"  - `{key}`: `{_format_metric_value(key_results[key])}`")
            for key in ("metrics_path", "train_metrics", "eval_metrics", "run_dir"):
                if key in key_results:
                    lines.append(f"  - `{key}`: `{key_results[key]}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _safe_git_value(args: list[str], fallback: str = "unknown") -> str:
    """Read compact git metadata without failing the run logger."""
    try:
        out = subprocess.run(
            ["git", *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return fallback
    value = out.stdout.strip()
    return value or fallback


def main() -> int:
    """Program entry point."""
    args = _parse_args()
    records_root = args.records_root.resolve()
    records_path = records_root / "run_records.jsonl"
    raw_log_dir = records_root / "raw_logs"
    cwd = Path.cwd().resolve()

    if args.rebuild_docs_only:
        records = _read_records(records_path)
        families = _build_family_index(records)
        commands_md = _render_commands_markdown(families, max_families=int(args.max_families))
        results_md = _render_results_markdown(records, max_results=int(args.max_results))
        args.commands_md.parent.mkdir(parents=True, exist_ok=True)
        args.results_md.parent.mkdir(parents=True, exist_ok=True)
        args.commands_md.write_text(commands_md, encoding="utf-8")
        args.results_md.write_text(results_md, encoding="utf-8")
        print(
            "[experiment-log] rebuild-only | records={} | commands_md={} | results_md={}".format(
                len(records),
                args.commands_md,
                args.results_md,
            )
        )
        return 0

    command_argv: list[str] = [str(part) for part in args.cmd]
    command_raw = shlex.join(command_argv)
    command_family_template = _normalize_command_for_family(command_raw)
    command_id = _sha12(command_raw)
    command_family_id = _sha12(command_family_template)

    existing_records = _read_records(records_path)
    family_run_index = (
        sum(1 for rec in existing_records if rec.get("command_family_id") == command_family_id)
        + 1
    )

    started = _now_utc()
    run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}_{command_id}"
    log_path = raw_log_dir / f"{run_id}.log"

    return_code, output_lines, duration_sec = _run_command(
        cmd_argv=command_argv,
        log_path=log_path,
        quiet=bool(args.quiet),
    )
    ended = _now_utc()

    parsed_kv = _parse_compact_kv(output_lines)
    candidate_paths = _resolve_json_candidate_paths(parsed_kv, cwd=cwd)
    json_summaries = _load_json_summaries(candidate_paths)
    key_results = _merge_high_signal(parsed_kv, json_summaries)

    record = {
        "run_id": run_id,
        "started_at_utc": _iso_utc(started),
        "ended_at_utc": _iso_utc(ended),
        "duration_sec": round(duration_sec, 6),
        "status": "success" if return_code == 0 else "failed",
        "return_code": int(return_code),
        "cwd": str(cwd),
        "command_argv": command_argv,
        "command_raw": command_raw,
        "command_id": command_id,
        "command_family_id": command_family_id,
        "command_family_template": command_family_template,
        "family_run_index": int(family_run_index),
        "tags": [str(tag) for tag in args.tag],
        "note": str(args.note).strip(),
        "log_path": str(log_path),
        "parsed_fields": parsed_kv,
        "json_summaries": json_summaries,
        "key_results": key_results,
        "git_commit": _safe_git_value(["rev-parse", "--short", "HEAD"]),
        "git_branch": _safe_git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
    }

    _append_record(records_path, record)
    all_records = existing_records + [record]

    if not args.no_update_docs:
        families = _build_family_index(all_records)
        commands_md = _render_commands_markdown(families, max_families=int(args.max_families))
        results_md = _render_results_markdown(all_records, max_results=int(args.max_results))
        args.commands_md.parent.mkdir(parents=True, exist_ok=True)
        args.results_md.parent.mkdir(parents=True, exist_ok=True)
        args.commands_md.write_text(commands_md, encoding="utf-8")
        args.results_md.write_text(results_md, encoding="utf-8")

    # Final one-line summary for quick terminal visibility.
    summary_parts = [
        f"run_id={run_id}",
        f"status={'ok' if return_code == 0 else 'fail'}",
        f"family={command_family_id}",
        f"attempt={family_run_index}",
        f"log={log_path}",
    ]
    if "accuracy" in key_results:
        summary_parts.append(f"accuracy={_format_metric_value(key_results['accuracy'])}")
    if "brier_score" in key_results:
        summary_parts.append(f"brier={_format_metric_value(key_results['brier_score'])}")
    if "corr_pair_acc" in key_results:
        summary_parts.append(f"corr_pair_acc={_format_metric_value(key_results['corr_pair_acc'])}")
    print("[experiment-log] " + " | ".join(summary_parts))

    return int(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
