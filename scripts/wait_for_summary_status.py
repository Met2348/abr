#!/usr/bin/env python3
"""Wait for a markdown summary file to report a terminal status.

This helper is used by overnight launchers to chain jobs safely.
It avoids the silent failure mode where a downstream job starts merely
because `final_summary.md` exists, even if the upstream run finished with
`status: failed`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_status(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- status:"):
            return line.split(":", 1)[1].strip().strip('`')
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wait until a summary markdown file reports an expected status.")
    parser.add_argument("summary_path", help="Path to final_summary.md")
    parser.add_argument("--expect-status", default="ok")
    parser.add_argument("--timeout-sec", type=int, default=12 * 3600)
    parser.add_argument("--poll-sec", type=int, default=60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.summary_path)
    deadline = time.time() + max(args.timeout_sec, 1)
    while time.time() < deadline:
        if path.exists():
            status = parse_status(path.read_text(encoding="utf-8", errors="ignore"))
            if status == args.expect_status:
                print(f"summary_status_ok: {path} -> {status}")
                return 0
            if status is not None and status != args.expect_status:
                print(
                    f"summary_status_mismatch: {path} -> {status} (expected {args.expect_status})",
                    file=sys.stderr,
                )
                return 2
        time.sleep(max(args.poll_sec, 1))
    print(f"summary_status_timeout: {path}", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
