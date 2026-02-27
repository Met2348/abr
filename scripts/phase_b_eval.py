#!/usr/bin/env python3
"""Phase B evaluation bridge.

This script evaluates a Phase B checkpoint/model by delegating inference+eval
to the stable Phase A evaluator pipeline. Keeping Phase A eval frozen is a
design requirement for fair comparisons.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase B model evaluation using the frozen Phase A generation+eval script."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-name", default="phase_b_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--strategyqa-decode-mode",
        choices=["freeform", "binary_choice"],
        default="freeform",
    )
    parser.add_argument(
        "--truncate-chat-markers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Optional extra flags forwarded to scripts/phase_a_generate_and_eval.py",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")

    repo_root = Path(__file__).resolve().parents[1]
    target_script = repo_root / "scripts" / "phase_a_generate_and_eval.py"
    if not target_script.exists():
        raise FileNotFoundError(f"Expected script not found: {target_script}")

    cmd = [
        sys.executable,
        "-u",
        str(target_script),
        "--input-jsonl",
        str(args.input_jsonl),
        "--model-path",
        args.model_path,
        "--run-name",
        args.run_name,
        "--seed",
        str(args.seed),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--batch-size",
        str(args.batch_size),
        "--strategyqa-decode-mode",
        args.strategyqa_decode_mode,
        "--no-do-sample",
    ]
    if args.require_cuda:
        cmd.append("--require-cuda")
    else:
        cmd.append("--no-require-cuda")

    if args.truncate_chat_markers:
        cmd.append("--truncate-chat-markers")
    else:
        cmd.append("--no-truncate-chat-markers")

    cmd.extend(args.extra_args)

    print("=" * 88)
    print("Phase B: Eval Bridge -> Phase A Generate+Eval")
    print("=" * 88)
    print("command:")
    print(" ".join(cmd))
    print("-" * 88)

    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
