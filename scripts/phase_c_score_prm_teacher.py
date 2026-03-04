#!/usr/bin/env python3
"""Score Phase C prefixes/corruptions with an external PRM teacher model.

Why this file exists
--------------------
Phase D introduces external PRM support to improve value-label quality.
Before changing C1/C2 training targets, we need one deterministic sidecar tool
that can score already-built C1 artifacts:
- `prefixes.jsonl`
- `corruptions.jsonl` (optional)

This script is that sidecar tool (`D1` in `docs/phase_D_plan.md`).

What this file does
-------------------
1. Load one Phase C artifact directory.
2. Build PRM-compatible chat input from each prefix/corruption row.
3. Run batched teacher scoring with OOM backoff.
4. Persist:
   - `teacher_prefix_scores.jsonl`
   - `teacher_corruption_scores.jsonl` (optional)
   - `teacher_errors.jsonl`
   - `teacher_summary.json`
   - `teacher_summary.md`

Interaction with other files
----------------------------
- Reads: `prefixes.jsonl`, `corruptions.jsonl`, `manifest.json` in one C1 dir.
- Writes teacher sidecar outputs in the same dir for later D2 fusion.
- D2/C2 can then consume teacher outputs to produce/use `q_teacher`, `q_fused`.

Example
-------
```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/phase_c_score_prm_teacher.py \
  --phase-c-dir assets/artifacts/phase_c_data/strategyqa/phase_c_quality_first_full_c2_strategyqa_quality_first_full_c1_train__90dcbacfbae1 \
  --teacher-model-path assets/models/Qwen2.5-Math-PRM-7B \
  --batch-size 192 \
  --max-length 2048 \
  --require-cuda
```
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _bootstrap_src_path() -> None:
    """Allow running this script from repo root without package installation."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()


@dataclass(slots=True)
class TeacherScoringConfig:
    """Teacher model inference settings written into summary artifacts."""

    teacher_model_path: str
    teacher_model_id: str
    teacher_system_prompt: str
    step_separator_token: str
    batch_size: int
    max_length: int
    dtype: str
    device_map: str
    require_cuda: bool
    oom_backoff: bool
    log_every: int
    score_corruptions: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config payload."""
        return asdict(self)


@dataclass(slots=True)
class TextBuildResult:
    """One preprocessed teacher input with metadata for diagnostics."""

    input_text: str
    question: str
    steps: list[str]
    used_question_fallback: bool
    used_step_fallback: bool


def _build_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for D1 teacher scoring."""
    parser = argparse.ArgumentParser(
        description=(
            "Score Phase C prefixes/corruptions with an external PRM teacher "
            "and write sidecar JSONL outputs."
        )
    )
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument(
        "--phase-c-dir",
        type=Path,
        required=False,
        default=None,
        help="Phase C artifact directory containing prefixes.jsonl and manifest.json.",
    )
    parser.add_argument(
        "--prefixes-jsonl",
        type=Path,
        default=None,
        help="Optional explicit prefix file path. Defaults to <phase-c-dir>/prefixes.jsonl.",
    )
    parser.add_argument(
        "--corruptions-jsonl",
        type=Path,
        default=None,
        help="Optional explicit corruption file path. Defaults to <phase-c-dir>/corruptions.jsonl.",
    )
    parser.add_argument(
        "--teacher-model-path",
        default="assets/models/Qwen2.5-Math-PRM-7B",
        help="Local PRM teacher path.",
    )
    parser.add_argument(
        "--teacher-model-id",
        default=None,
        help="Optional explicit teacher model ID for output metadata.",
    )
    parser.add_argument(
        "--teacher-system-prompt",
        default="Please reason step by step, and put your final answer within \\boxed{}.",
        help="System prompt used when building teacher chat inputs.",
    )
    parser.add_argument(
        "--step-separator-token",
        default="<extra_0>",
        help="Separator token used by Qwen PRM to mark reasoning steps.",
    )
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail fast when CUDA is unavailable.",
    )
    parser.add_argument(
        "--oom-backoff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively split batches on CUDA OOM.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=512,
        help="Progress logging interval for scored records.",
    )
    parser.add_argument(
        "--score-corruptions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score corruptions.jsonl in addition to prefixes.jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing teacher output files.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail on the first malformed row instead of recording to teacher_errors.jsonl.",
    )
    parser.add_argument(
        "--allow-missing-manifest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow scoring when <phase-c-dir>/manifest.json is missing. "
            "Use this only for legacy/incomplete artifact directories."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args with optional config-JSON defaults."""
    parser = _build_parser()
    # 两段解析：先吸收 config 默认值，再由 CLI 覆盖。
    initial = parser.parse_known_args(argv)[0]
    if initial.config_json is not None:
        defaults = _load_config_defaults(initial.config_json)
        valid_keys = {action.dest for action in parser._actions}  # noqa: SLF001
        unknown = sorted(set(defaults.keys()) - valid_keys)
        if unknown:
            raise KeyError(f"Unknown keys in config JSON {initial.config_json}: {unknown}")
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def _load_config_defaults(path: Path) -> dict[str, Any]:
    """Load one JSON object used as CLI defaults."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Config JSON {path} must contain an object")
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run D1 teacher scoring for one Phase C artifact directory."""
    args = parse_args(argv)
    started = time.time()

    # Stage 1: 解析输入与输出路径，保证 sidecar 输出不会误覆盖。
    phase_c_dir, prefixes_path, corruptions_path, manifest_path = _resolve_input_paths(
        args=args,
        allow_missing_manifest=bool(args.allow_missing_manifest),
    )
    teacher_model_id = args.teacher_model_id or os.path.basename(str(args.teacher_model_path.rstrip("/")))

    outputs = _resolve_output_paths(
        phase_c_dir=phase_c_dir,
        score_corruptions=bool(args.score_corruptions),
    )
    _assert_output_writable(outputs=outputs, overwrite=bool(args.overwrite))

    prefixes_raw = _read_jsonl(prefixes_path)
    corruptions_raw = _read_jsonl(corruptions_path) if outputs.score_corruptions else []
    manifest = _read_manifest(
        path=manifest_path,
        allow_missing=bool(args.allow_missing_manifest),
    )

    print("=" * 88)
    print("Phase D: Score PRM Teacher (D1)")
    print("=" * 88)
    print(f"phase_c_dir      : {phase_c_dir}")
    print(f"prefixes_jsonl   : {prefixes_path}")
    print(f"corruptions_jsonl: {corruptions_path if outputs.score_corruptions else '<disabled>'}")
    print(f"teacher_model    : {args.teacher_model_path}")
    print(f"teacher_model_id : {teacher_model_id}")
    print(f"batch_size       : {args.batch_size}")
    print(f"max_length       : {args.max_length}")
    print(f"dtype            : {args.dtype}")
    print(f"device_map       : {args.device_map}")
    print(f"score_corruptions: {outputs.score_corruptions}")
    print(f"manifest_path    : {manifest_path if manifest_path is not None else '<missing>'}")
    if manifest_path is None:
        print("warning          : source manifest missing; lineage fields will be partial")
    print("=" * 88)

    config = TeacherScoringConfig(
        teacher_model_path=str(args.teacher_model_path),
        teacher_model_id=teacher_model_id,
        teacher_system_prompt=str(args.teacher_system_prompt),
        step_separator_token=str(args.step_separator_token),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        dtype=str(args.dtype),
        device_map=str(args.device_map),
        require_cuda=bool(args.require_cuda),
        oom_backoff=bool(args.oom_backoff),
        log_every=int(args.log_every),
        score_corruptions=bool(outputs.score_corruptions),
    )
    _validate_scoring_config(config)

    # Keep these imports local so unit tests for pure helper functions do not
    # require GPU/runtime setup until main() actually runs.
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    if config.require_cuda and not torch.cuda.is_available():
        raise EnvironmentError("--require-cuda was set but CUDA is unavailable")

    # Stage 2: 加载 teacher，并校验 step separator token 映射。
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_path, trust_remote_code=True)
    sep_ids = tokenizer.encode(config.step_separator_token, add_special_tokens=False)
    if len(sep_ids) != 1:
        raise ValueError(
            "Expected step separator token to map to one token ID, got "
            f"{len(sep_ids)} for token={config.step_separator_token!r}"
        )
    sep_token_id = int(sep_ids[0])
    dtype = _resolve_torch_dtype(config.dtype)
    model = AutoModel.from_pretrained(
        config.teacher_model_path,
        dtype=dtype,
        device_map=config.device_map,
        trust_remote_code=True,
    ).eval()
    setattr(model, "_teacher_tokenizer", tokenizer)

    device = _infer_model_device(model)

    # Stage 3: 先构建 rows，再统一批量打分，方便错误行旁路记录。
    prefix_rows, prefix_errors = _build_prefix_scoring_rows(
        prefixes_raw=prefixes_raw,
        tokenizer=tokenizer,
        config=config,
        strict=bool(args.strict),
    )
    prefix_scores = _score_rows_with_teacher(
        rows=prefix_rows,
        model=model,
        device=device,
        sep_token_id=sep_token_id,
        batch_size=config.batch_size,
        max_length=config.max_length,
        oom_backoff=config.oom_backoff,
        log_every=config.log_every,
        progress_label="prefixes",
        softmax_fn=F.softmax,
        torch_mod=torch,
    )

    corruption_scores: list[dict[str, Any]] = []
    corruption_errors: list[dict[str, Any]] = []
    if outputs.score_corruptions:
        prefix_lookup = {str(row.get("prefix_id", "")): row for row in prefixes_raw}
        corruption_rows, corruption_errors = _build_corruption_scoring_rows(
            corruptions_raw=corruptions_raw,
            prefix_lookup=prefix_lookup,
            tokenizer=tokenizer,
            config=config,
            strict=bool(args.strict),
        )
        corruption_scores = _score_rows_with_teacher(
            rows=corruption_rows,
            model=model,
            device=device,
            sep_token_id=sep_token_id,
            batch_size=config.batch_size,
            max_length=config.max_length,
            oom_backoff=config.oom_backoff,
            log_every=config.log_every,
            progress_label="corruptions",
            softmax_fn=F.softmax,
            torch_mod=torch,
        )

    error_rows = [*prefix_errors, *corruption_errors]
    _write_jsonl(outputs.prefix_scores_path, prefix_scores)
    if outputs.score_corruptions:
        _write_jsonl(outputs.corruption_scores_path, corruption_scores)
    _write_jsonl(outputs.errors_path, error_rows)

    elapsed = time.time() - started
    summary = _build_summary(
        phase_c_dir=phase_c_dir,
        manifest=manifest,
        config=config,
        prefixes_path=prefixes_path,
        corruptions_path=corruptions_path if outputs.score_corruptions else None,
        prefix_scores=prefix_scores,
        corruption_scores=corruption_scores,
        error_rows=error_rows,
        elapsed_sec=elapsed,
    )
    outputs.summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    outputs.summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")

    print("-" * 88)
    print(f"num_prefix_scores    : {summary['num_prefix_scores']}")
    print(f"num_corruption_scores: {summary['num_corruption_scores']}")
    print(f"num_errors           : {summary['num_errors']}")
    print(f"mean_prefix_score    : {summary['prefix_score_summary']['mean_score']:.6f}")
    if outputs.score_corruptions:
        print(
            f"mean_corrupt_score   : {summary['corruption_score_summary']['mean_score']:.6f}"
        )
    print(f"elapsed_sec          : {summary['elapsed_sec']:.2f}")
    print(f"prefix_scores_path   : {outputs.prefix_scores_path}")
    if outputs.score_corruptions:
        print(f"corrupt_scores_path  : {outputs.corruption_scores_path}")
    print(f"errors_path          : {outputs.errors_path}")
    print(f"summary_path         : {outputs.summary_path}")
    print("=" * 88)
    return 0


@dataclass(slots=True)
class OutputPaths:
    """Resolved output file set for one D1 run."""

    prefix_scores_path: Path
    corruption_scores_path: Path
    errors_path: Path
    summary_path: Path
    summary_md_path: Path
    score_corruptions: bool


def _resolve_input_paths(
    *,
    args: argparse.Namespace,
    allow_missing_manifest: bool,
) -> tuple[Path, Path, Path, Path | None]:
    """Resolve and validate input file paths from CLI args."""
    phase_c_dir = Path(args.phase_c_dir) if args.phase_c_dir is not None else None
    if phase_c_dir is None and args.prefixes_jsonl is None:
        raise ValueError("Provide --phase-c-dir or --prefixes-jsonl")
    if phase_c_dir is None:
        phase_c_dir = Path(args.prefixes_jsonl).resolve().parent
    phase_c_dir = phase_c_dir.resolve()

    prefixes_path = (
        Path(args.prefixes_jsonl).resolve()
        if args.prefixes_jsonl is not None
        else (phase_c_dir / "prefixes.jsonl")
    )
    corruptions_path = (
        Path(args.corruptions_jsonl).resolve()
        if args.corruptions_jsonl is not None
        else (phase_c_dir / "corruptions.jsonl")
    )
    if not prefixes_path.exists():
        raise FileNotFoundError(f"Missing prefixes.jsonl: {prefixes_path}")
    manifest_path = phase_c_dir / "manifest.json"
    if not manifest_path.exists() and not allow_missing_manifest:
        raise FileNotFoundError(f"Missing manifest.json in phase_c_dir: {phase_c_dir}")
    return (
        phase_c_dir,
        prefixes_path,
        corruptions_path,
        (manifest_path if manifest_path.exists() else None),
    )


def _resolve_output_paths(*, phase_c_dir: Path, score_corruptions: bool) -> OutputPaths:
    """Return default D1 output paths located in the Phase C artifact directory."""
    return OutputPaths(
        prefix_scores_path=phase_c_dir / "teacher_prefix_scores.jsonl",
        corruption_scores_path=phase_c_dir / "teacher_corruption_scores.jsonl",
        errors_path=phase_c_dir / "teacher_errors.jsonl",
        summary_path=phase_c_dir / "teacher_summary.json",
        summary_md_path=phase_c_dir / "teacher_summary.md",
        score_corruptions=score_corruptions,
    )


def _assert_output_writable(*, outputs: OutputPaths, overwrite: bool) -> None:
    """Prevent accidental overwrite unless explicitly requested."""
    candidates = [
        outputs.prefix_scores_path,
        outputs.errors_path,
        outputs.summary_path,
        outputs.summary_md_path,
    ]
    if outputs.score_corruptions:
        candidates.append(outputs.corruption_scores_path)
    existing = [path for path in candidates if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Output files already exist. Re-run with --overwrite to replace them: "
            f"{joined}"
        )


def _read_manifest(*, path: Path | None, allow_missing: bool) -> dict[str, Any]:
    """Load source Phase C manifest for lineage logging.

    When `allow_missing` is true and `path` is missing, return a synthetic
    minimal manifest so historical incomplete artifact dirs can still be scored.
    """
    if path is None:
        if not allow_missing:
            raise FileNotFoundError("Missing manifest.json and allow_missing=False")
        return {
            "artifact_stage": "phase_c_c0_c1_missing_manifest",
            "run_name": "<missing-manifest>",
            "metadata": {"manifest_missing": True},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Manifest must contain a JSON object: {path}")
    return payload


def _validate_scoring_config(config: TeacherScoringConfig) -> None:
    """Validate numeric/string controls before model loading."""
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if config.max_length <= 0:
        raise ValueError("--max-length must be > 0")
    if config.log_every <= 0:
        raise ValueError("--log-every must be > 0")
    if not config.step_separator_token:
        raise ValueError("--step-separator-token must be non-empty")


def _resolve_torch_dtype(dtype_name: str):
    """Map CLI dtype string to torch dtype object or `None` (auto)."""
    import torch

    mapping = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def _infer_model_device(model) -> Any:
    """Infer one device for input tensor placement.

    With `device_map=auto`, most local runs still place this PRM on one GPU.
    We use the first parameter device for input placement.
    """

    for param in model.parameters():
        return param.device
    raise RuntimeError("Failed to infer model device from parameters")


def _build_prefix_scoring_rows(
    *,
    prefixes_raw: list[dict[str, Any]],
    tokenizer,
    config: TeacherScoringConfig,
    strict: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build PRM input rows for prefixes and collect recoverable row errors."""
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for idx, row in enumerate(prefixes_raw):
        try:
            prefix_id = _require_str(row, "prefix_id")
            question = _extract_question(row)
            text_build = _build_prefix_teacher_text(
                row=row,
                tokenizer=tokenizer,
                config=config,
                question=question,
            )
            rows.append(
                {
                    "row_type": "prefix",
                    "id": prefix_id,
                    "prefix_id": prefix_id,
                    "sample_id": str(row.get("sample_id", "")),
                    "dataset": str(row.get("dataset", "")),
                    "split": str(row.get("split", "")),
                    "question": text_build.question,
                    "steps": text_build.steps,
                    "teacher_input_text": text_build.input_text,
                    "used_question_fallback": text_build.used_question_fallback,
                    "used_step_fallback": text_build.used_step_fallback,
                    "teacher_model_id": config.teacher_model_id,
                }
            )
        except Exception as exc:  # noqa: BLE001
            err = {
                "row_type": "prefix",
                "row_index": idx,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if strict:
                raise RuntimeError(err["error"]) from exc
            errors.append(err)
    return rows, errors


def _build_corruption_scoring_rows(
    *,
    corruptions_raw: list[dict[str, Any]],
    prefix_lookup: dict[str, dict[str, Any]],
    tokenizer,
    config: TeacherScoringConfig,
    strict: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build PRM input rows for corruptions and collect recoverable row errors."""
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for idx, row in enumerate(corruptions_raw):
        try:
            corruption_id = _require_str(row, "corruption_id")
            clean_prefix_id = _require_str(row, "clean_prefix_id")
            clean_prefix = prefix_lookup.get(clean_prefix_id)
            if clean_prefix is None:
                raise KeyError(
                    f"Missing clean prefix reference for corruption_id={corruption_id}: "
                    f"clean_prefix_id={clean_prefix_id}"
                )
            question = _extract_question(clean_prefix)
            text_build = _build_corruption_teacher_text(
                row=row,
                tokenizer=tokenizer,
                config=config,
                question=question,
            )
            rows.append(
                {
                    "row_type": "corruption",
                    "id": corruption_id,
                    "corruption_id": corruption_id,
                    "clean_prefix_id": clean_prefix_id,
                    "sample_id": str(row.get("sample_id", "")),
                    "dataset": str(row.get("dataset", "")),
                    "split": str(row.get("split", "")),
                    "corruption_type": str(row.get("corruption_type", "")),
                    "question": text_build.question,
                    "steps": text_build.steps,
                    "teacher_input_text": text_build.input_text,
                    "used_question_fallback": text_build.used_question_fallback,
                    "used_step_fallback": text_build.used_step_fallback,
                    "teacher_model_id": config.teacher_model_id,
                }
            )
        except Exception as exc:  # noqa: BLE001
            err = {
                "row_type": "corruption",
                "row_index": idx,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if strict:
                raise RuntimeError(err["error"]) from exc
            errors.append(err)
    return rows, errors


def _extract_question(row: dict[str, Any]) -> str:
    """Extract question text with fallback to prompt when needed."""
    question = str(row.get("question", "")).strip()
    if question:
        return question
    prompt_text = str(row.get("prompt_text", "")).strip()
    if prompt_text:
        return prompt_text
    return "<missing-question>"


def _build_prefix_teacher_text(
    *,
    row: dict[str, Any],
    tokenizer,
    config: TeacherScoringConfig,
    question: str,
) -> TextBuildResult:
    """Build one PRM-compatible input text for a prefix row."""
    prefix_text = str(row.get("prefix_target_text", "")).strip()
    steps = _split_reasoning_steps(prefix_text)
    used_step_fallback = False
    if not steps:
        fallback = str(row.get("current_step_text", "")).strip()
        if fallback:
            steps = [fallback]
            used_step_fallback = True
    if not steps:
        # 最终兜底：至少给 teacher 一个可打分的 step 文本。
        steps = [question]
        used_step_fallback = True
    used_question_fallback = str(row.get("question", "")).strip() == ""
    input_text = _build_teacher_chat_text(
        tokenizer=tokenizer,
        system_prompt=config.teacher_system_prompt,
        question=question,
        steps=steps,
        step_separator_token=config.step_separator_token,
    )
    return TextBuildResult(
        input_text=input_text,
        question=question,
        steps=steps,
        used_question_fallback=used_question_fallback,
        used_step_fallback=used_step_fallback,
    )


def _build_corruption_teacher_text(
    *,
    row: dict[str, Any],
    tokenizer,
    config: TeacherScoringConfig,
    question: str,
) -> TextBuildResult:
    """Build one PRM-compatible input text for a corruption row."""
    corrupt_text = str(row.get("corrupted_prefix_text", "")).strip()
    steps = _split_reasoning_steps(corrupt_text)
    used_step_fallback = False
    if not steps:
        fallback = str(row.get("corrupted_step_text", "")).strip()
        if fallback:
            steps = [fallback]
            used_step_fallback = True
    if not steps:
        fallback_original = str(row.get("original_step_text", "")).strip()
        if fallback_original:
            steps = [fallback_original]
            used_step_fallback = True
    if not steps:
        steps = [question]
        used_step_fallback = True
    input_text = _build_teacher_chat_text(
        tokenizer=tokenizer,
        system_prompt=config.teacher_system_prompt,
        question=question,
        steps=steps,
        step_separator_token=config.step_separator_token,
    )
    return TextBuildResult(
        input_text=input_text,
        question=question,
        steps=steps,
        used_question_fallback=False,
        used_step_fallback=used_step_fallback,
    )


def _split_reasoning_steps(text: str) -> list[str]:
    """Split multi-line prefix text into clean step strings.

    The C1 artifacts store prefix text with newline-separated reasoning fragments.
    This helper normalizes and filters blanks. It also strips list markers that
    frequently appear in target text.
    """
    steps: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        if line.startswith("* "):
            line = line[2:].strip()
        if line:
            steps.append(line)
    return steps


def _build_teacher_chat_text(
    *,
    tokenizer,
    system_prompt: str,
    question: str,
    steps: list[str],
    step_separator_token: str,
) -> str:
    """Render one chat-formatted teacher input string from question + steps."""
    assistant_steps = _render_assistant_step_text(
        steps=steps,
        step_separator_token=step_separator_token,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_steps},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:  # noqa: BLE001
        # Fallback keeps D1 usable on tokenizers without a chat template.
        return (
            f"[SYSTEM]\n{system_prompt}\n\n"
            f"[USER]\n{question}\n\n"
            f"[ASSISTANT]\n{assistant_steps}"
        )


def _render_assistant_step_text(*, steps: list[str], step_separator_token: str) -> str:
    """Render assistant text with explicit PRM step separators after each step."""
    cleaned_steps = [
        step.replace(step_separator_token, "").strip()
        for step in steps
        if step.strip()
    ]
    return "\n".join(f"{step} {step_separator_token}" for step in cleaned_steps).strip()


def _score_rows_with_teacher(
    *,
    rows: list[dict[str, Any]],
    model,
    device,
    sep_token_id: int,
    batch_size: int,
    max_length: int,
    oom_backoff: bool,
    log_every: int,
    progress_label: str,
    softmax_fn,
    torch_mod,
) -> list[dict[str, Any]]:
    """Score prebuilt text rows and return JSONL-ready records."""
    if not rows:
        return []
    total = len(rows)
    progress_every = _resolve_progress_interval(
        total_items=total,
        requested_every=int(log_every),
        max_updates=8,
    )
    next_progress = progress_every
    scored: list[dict[str, Any]] = []
    started = time.time()
    total_batches = (total + batch_size - 1) // batch_size
    print(
        f"{progress_label:14s}: start {total} rows in {total_batches} batches "
        f"(bs={batch_size}, progress_every~{progress_every})"
    )
    for offset in range(0, total, batch_size):
        batch = rows[offset : offset + batch_size]
        batch_scores = _score_batch_with_optional_backoff(
            batch=batch,
            model=model,
            device=device,
            sep_token_id=sep_token_id,
            max_length=max_length,
            oom_backoff=oom_backoff,
            softmax_fn=softmax_fn,
            torch_mod=torch_mod,
        )
        scored.extend(batch_scores)
        done = len(scored)
        if done >= next_progress or done == total:
            elapsed = max(time.time() - started, 1e-6)
            rate = done / elapsed
            print(
                f"{progress_label:14s}: {done}/{total} "
                f"({(done / total) * 100.0:.1f}%) | elapsed={elapsed:.1f}s "
                f"| rate={rate:.3f} row/s"
            )
            while next_progress <= done:
                next_progress += progress_every
    return scored


def _resolve_progress_interval(
    *,
    total_items: int,
    requested_every: int,
    max_updates: int = 8,
) -> int:
    """Return a bounded update interval to avoid long silent teacher-scoring phases."""
    total_items = max(int(total_items), 0)
    max_updates = max(int(max_updates), 1)
    if total_items <= 0:
        return 1
    _ = int(requested_every)  # kept for API stability/documentation symmetry.
    return max(1, math.ceil(total_items / max_updates))


def _score_batch_with_optional_backoff(
    *,
    batch: list[dict[str, Any]],
    model,
    device,
    sep_token_id: int,
    max_length: int,
    oom_backoff: bool,
    softmax_fn,
    torch_mod,
) -> list[dict[str, Any]]:
    """Score one batch and optionally split recursively on CUDA OOM."""
    try:
        return _score_batch(
            batch=batch,
            model=model,
            device=device,
            sep_token_id=sep_token_id,
            max_length=max_length,
            softmax_fn=softmax_fn,
            torch_mod=torch_mod,
        )
    except RuntimeError as exc:  # noqa: PERF203
        if "out of memory" not in str(exc).lower() or not oom_backoff or len(batch) <= 1:
            raise
        mid = len(batch) // 2
        left = _score_batch_with_optional_backoff(
            batch=batch[:mid],
            model=model,
            device=device,
            sep_token_id=sep_token_id,
            max_length=max_length,
            oom_backoff=oom_backoff,
            softmax_fn=softmax_fn,
            torch_mod=torch_mod,
        )
        right = _score_batch_with_optional_backoff(
            batch=batch[mid:],
            model=model,
            device=device,
            sep_token_id=sep_token_id,
            max_length=max_length,
            oom_backoff=oom_backoff,
            softmax_fn=softmax_fn,
            torch_mod=torch_mod,
        )
        return [*left, *right]


def _score_batch(
    *,
    batch: list[dict[str, Any]],
    model,
    device,
    sep_token_id: int,
    max_length: int,
    softmax_fn,
    torch_mod,
) -> list[dict[str, Any]]:
    """Run one forward pass and convert logits to teacher step statistics."""
    # The tokenizer is attached once in `main()` to keep function signatures
    # short while allowing unit tests to inject a lightweight fake model.
    if not hasattr(model, "_teacher_tokenizer"):
        raise AttributeError("Model is missing `_teacher_tokenizer` helper")
    tok = model._teacher_tokenizer  # noqa: SLF001

    texts = [str(row["teacher_input_text"]) for row in batch]
    encoded = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch_mod.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    logits = _extract_logits(outputs)
    if logits.shape[-1] < 2:
        raise RuntimeError(
            "Expected teacher logits with class dimension >= 2, got shape "
            f"{tuple(logits.shape)}"
        )
    # 二分类 teacher 的“正类概率”作为 step 级分数来源。
    probs = softmax_fn(logits, dim=-1)[..., 1]
    scores_by_row = _extract_step_scores_from_probs(
        probs=probs,
        input_ids=input_ids,
        sep_token_id=sep_token_id,
    )

    scored_rows: list[dict[str, Any]] = []
    for row, step_scores, row_mask in zip(batch, scores_by_row, attention_mask, strict=True):
        row_mean = float(sum(step_scores) / len(step_scores)) if step_scores else 0.0
        row_min = float(min(step_scores)) if step_scores else 0.0
        scored_rows.append(
            {
                **{k: v for k, v in row.items() if k != "teacher_input_text"},
                "teacher_model_id": str(row.get("teacher_model_id", "")),
                "teacher_score_mean": row_mean,
                "teacher_score_min": row_min,
                "teacher_num_steps": len(step_scores),
                "teacher_step_scores": [float(x) for x in step_scores],
                "teacher_input_num_tokens": int(row_mask.sum().item()),
                "teacher_infer_meta": {
                    "scoring_version": "phase_d_d1_v1",
                    "step_separator_token_id": int(sep_token_id),
                },
            }
        )
    return scored_rows


def _extract_logits(outputs) -> Any:
    """Extract logits tensor from common HF output layouts."""
    if isinstance(outputs, tuple):
        return outputs[0]
    if hasattr(outputs, "logits"):
        return outputs.logits
    raise TypeError(f"Unsupported model output type: {type(outputs).__name__}")


def _extract_step_scores_from_probs(*, probs, input_ids, sep_token_id: int) -> list[list[float]]:
    """Extract per-row step scores at separator-token positions."""
    # 只在 `<extra_0>` 对应位置取分，得到 step-level 序列分数。
    rows: list[list[float]] = []
    for row_probs, row_ids in zip(probs, input_ids, strict=True):
        mask = row_ids.eq(sep_token_id)
        selected = row_probs[mask].detach().float().cpu().tolist()
        rows.append([float(x) for x in selected])
    return rows


def _build_summary(
    *,
    phase_c_dir: Path,
    manifest: dict[str, Any],
    config: TeacherScoringConfig,
    prefixes_path: Path,
    corruptions_path: Path | None,
    prefix_scores: list[dict[str, Any]],
    corruption_scores: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    elapsed_sec: float,
) -> dict[str, Any]:
    """Build JSON summary with compact score statistics."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_stage": "phase_d_d1_teacher_scoring",
        "phase_c_dir": str(phase_c_dir),
        "source_manifest_run_name": str(manifest.get("run_name", "")),
        "source_manifest_missing": bool(manifest.get("metadata", {}).get("manifest_missing", False)),
        "prefixes_jsonl": str(prefixes_path),
        "corruptions_jsonl": str(corruptions_path) if corruptions_path is not None else None,
        "teacher_config": config.to_dict(),
        "num_prefix_scores": len(prefix_scores),
        "num_corruption_scores": len(corruption_scores),
        "num_errors": len(error_rows),
        "prefix_score_summary": _summarize_score_rows(prefix_scores),
        "corruption_score_summary": _summarize_score_rows(corruption_scores),
        "elapsed_sec": float(elapsed_sec),
    }


def _summarize_score_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize teacher score records for reporting."""
    if not rows:
        return {
            "mean_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "mean_num_steps": 0.0,
        }
    means = [float(row.get("teacher_score_mean", 0.0)) for row in rows]
    steps = [int(row.get("teacher_num_steps", 0)) for row in rows]
    return {
        "mean_score": float(sum(means) / len(means)),
        "min_score": float(min(means)),
        "max_score": float(max(means)),
        "mean_num_steps": float(sum(steps) / len(steps)),
    }


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    """Render a small Markdown report next to `teacher_summary.json`."""
    lines = [
        "# Phase D D1 Teacher Scoring Summary",
        "",
        "## Inputs",
        f"- phase_c_dir: `{summary['phase_c_dir']}`",
        f"- source_manifest_run_name: `{summary['source_manifest_run_name']}`",
        f"- source_manifest_missing: `{summary['source_manifest_missing']}`",
        f"- prefixes_jsonl: `{summary['prefixes_jsonl']}`",
        f"- corruptions_jsonl: `{summary['corruptions_jsonl']}`",
        "",
        "## Counts",
        f"- num_prefix_scores: `{summary['num_prefix_scores']}`",
        f"- num_corruption_scores: `{summary['num_corruption_scores']}`",
        f"- num_errors: `{summary['num_errors']}`",
        "",
        "## Prefix Score Summary",
        f"- mean_score: `{summary['prefix_score_summary']['mean_score']:.6f}`",
        f"- min_score: `{summary['prefix_score_summary']['min_score']:.6f}`",
        f"- max_score: `{summary['prefix_score_summary']['max_score']:.6f}`",
        f"- mean_num_steps: `{summary['prefix_score_summary']['mean_num_steps']:.2f}`",
        "",
        "## Corruption Score Summary",
        f"- mean_score: `{summary['corruption_score_summary']['mean_score']:.6f}`",
        f"- min_score: `{summary['corruption_score_summary']['min_score']:.6f}`",
        f"- max_score: `{summary['corruption_score_summary']['max_score']:.6f}`",
        f"- mean_num_steps: `{summary['corruption_score_summary']['mean_num_steps']:.2f}`",
        "",
        f"- elapsed_sec: `{summary['elapsed_sec']:.2f}`",
    ]
    return "\n".join(lines).strip() + "\n"


def _require_str(row: dict[str, Any], key: str) -> str:
    """Read one required non-empty string field."""
    value = str(row.get(key, "")).strip()
    if not value:
        raise KeyError(f"Missing required field `{key}`")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL as object rows with line-level validation."""
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"{path}:{idx + 1} must be a JSON object")
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSONL with UTF-8 encoding."""
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
