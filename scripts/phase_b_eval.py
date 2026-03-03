#!/usr/bin/env python3
"""Evaluate a Phase B model or adapter through the frozen Phase A evaluator path.

Why this file exists
--------------------
Phase B changes the model weights, but it should not quietly change the evaluation
logic. This bridge keeps evaluation anchored to the already-tested Phase A
generation/evaluation script.

What this file does
-------------------
1. Parse evaluation options.
2. Resolve whether evaluation should use:
   - a direct model path, or
   - a finished Phase B run directory.
   - or one saved checkpoint directory from a Phase B run.
3. Rebuild a Phase A `generate_and_eval` command line.
4. Forward execution to that script and return its exit code.

Interaction with other files
----------------------------
- `scripts/phase_a_generate_and_eval.py`: the actual frozen evaluator.
- `scripts/phase_b_train_sft.py`: produces the model or adapter being evaluated.

Example
-------
```bash
python -u scripts/phase_b_eval.py \
  --input-jsonl assets/artifacts/phase_a_prepared/strategyqa/b0f373610f96/validation.jsonl \
  --phase-b-run-dir assets/artifacts/phase_b_runs/phase_b_first_b1_first_20260301T125445Z \
  --run-name phase_b_eval_smoke
```
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the Phase B evaluation bridge.

    Returns
    -------
    argparse.Namespace
        Validated arguments controlling which prepared set and model path are passed
        to the frozen Phase A evaluator.

    Example
    -------
    ```python
    args = parse_args()
    ```
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase B model evaluation using the frozen Phase A generation+eval script."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Base or fully fine-tuned model path. "
            "Use this for full-SFT eval, or provide --phase-b-run-dir instead."
        ),
    )
    parser.add_argument(
        "--phase-b-run-dir",
        type=Path,
        default=None,
        help=(
            "Optional finished Phase B run directory. When provided, the script reads "
            "its manifest and final_model directory to resolve evaluation inputs."
        ),
    )
    parser.add_argument(
        "--phase-b-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint directory inside one Phase B run, for example "
            "`.../checkpoints/checkpoint-400`. This is primarily used by "
            "checkpoint-sweep diagnostics."
        ),
    )
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
    return parser.parse_args(argv)


def _resolve_eval_model_paths(args: argparse.Namespace) -> tuple[str, Path | None]:
    """Resolve base-model path and optional adapter path for evaluation.

    Returns
    -------
    tuple[str, Path | None]
        `(model_path, adapter_path)` where `adapter_path` is only populated for PEFT
        runs that saved adapter-only artifacts.

    Example
    -------
    ```python
    model_path, adapter_path = _resolve_eval_model_paths(args)
    ```
    """
    # 三选一约束：评测来源只能有一个，避免“到底在评谁”不清楚。
    provided = [
        args.model_path is not None,
        args.phase_b_run_dir is not None,
        args.phase_b_checkpoint_dir is not None,
    ]
    if sum(bool(x) for x in provided) != 1:
        raise ValueError(
            "Use exactly one of --model-path, --phase-b-run-dir, or "
            "--phase-b-checkpoint-dir."
        )

    if args.model_path is not None:
        return str(args.model_path), None

    # run_dir 路径既可由 `--phase-b-run-dir` 直接提供，也可从 checkpoint 回推。
    run_dir = args.phase_b_run_dir
    model_dir = None
    if args.phase_b_checkpoint_dir is not None:
        model_dir = args.phase_b_checkpoint_dir
        if not model_dir.exists():
            raise FileNotFoundError(f"Phase B checkpoint directory not found: {model_dir}")
        run_dir = model_dir.parents[1]

    assert run_dir is not None
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Phase B manifest not found: {manifest_path}")
    if model_dir is None:
        model_dir = run_dir / "final_model"
        if not model_dir.exists():
            raise FileNotFoundError(f"Phase B final_model directory not found: {model_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    effective_mode = str(manifest.get("effective_training_mode", "")).strip().lower()
    base_model_path = str(manifest.get("model_path", "")).strip()
    if base_model_path == "":
        raise ValueError(f"Missing `model_path` in Phase B manifest: {manifest_path}")

    adapter_config = model_dir / "adapter_config.json"
    if effective_mode == "peft":
        # PEFT: phase_a_generate_and_eval 需要 base model + adapter-path 组合。
        if not adapter_config.exists():
            raise FileNotFoundError(
                f"Expected adapter_config.json for PEFT run: {adapter_config}"
            )
        return base_model_path, model_dir

    # Full SFT: 直接把训练产物目录当作 model-path 即可。
    return str(model_dir), None


def main() -> int:
    """Delegate Phase B evaluation to `scripts/phase_a_generate_and_eval.py`.

    Returns
    -------
    int
        Exit code returned by the delegated subprocess.

    Example
    -------
    ```bash
    python scripts/phase_b_eval.py --input-jsonl validation.jsonl --model-path my_model
    ```
    """
    args = parse_args()
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")
    model_path, adapter_path = _resolve_eval_model_paths(args)

    repo_root = Path(__file__).resolve().parents[1]
    target_script = repo_root / "scripts" / "phase_a_generate_and_eval.py"
    if not target_script.exists():
        raise FileNotFoundError(f"Expected script not found: {target_script}")

    # 固定走 Phase A evaluator，确保口径和历史基线一致。
    cmd = [
        sys.executable,
        "-u",
        str(target_script),
        "--input-jsonl",
        str(args.input_jsonl),
        "--model-path",
        model_path,
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
    if adapter_path is not None:
        cmd.extend(["--adapter-path", str(adapter_path)])
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
    print(f"resolved_model_path : {model_path}")
    print(f"resolved_adapter    : {adapter_path if adapter_path is not None else '<none>'}")
    print("command:")
    print(" ".join(cmd))
    print("-" * 88)

    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
