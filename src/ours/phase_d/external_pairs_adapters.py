"""Source adapters for building canonical Phase D external pair artifacts.

Why this file exists
--------------------
External process-supervision datasets use heterogeneous schemas. This module
adapts those raw formats into one canonical `(prompt, chosen, rejected)` pair
contract so Phase D can train/evaluate with deterministic behavior.

What this file contains
-----------------------
1. Direct pair loaders (for example R-PRM, PRMBench preview).
2. Step-label converters (for example Math-Shepherd, RLHFlow, PRM800K).
3. Shared quality filters to prevent degenerate pair artifacts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .external_pairs import ExternalPairRecord


@dataclass(slots=True)
class PairBuildConfig:
    """Quality-control config shared by external-pair source adapters."""

    min_chars: int = 12
    max_length_ratio: float = 4.0
    max_token_overlap: float = 0.995
    max_pairs_per_sample: int = 2

    def validate(self) -> None:
        if self.min_chars <= 0:
            raise ValueError("`min_chars` must be > 0")
        if self.max_length_ratio <= 1.0:
            raise ValueError("`max_length_ratio` must be > 1")
        if not (0.0 <= self.max_token_overlap <= 1.0):
            raise ValueError("`max_token_overlap` must be in [0, 1]")
        if self.max_pairs_per_sample <= 0:
            raise ValueError("`max_pairs_per_sample` must be > 0")


def load_r_prm_dpo_pairs(
    *,
    root: Path,
    split: str,
    config: PairBuildConfig,
    max_pairs: int | None = None,
) -> list[ExternalPairRecord]:
    """Load direct chosen/rejected pairs from R-PRM DPO split."""
    config.validate()
    split = str(split).strip()
    if split not in {"train", "validation"}:
        raise ValueError("R-PRM split must be one of {train, validation}")
    source_root = root / "dpo" / split
    files = sorted(source_root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards found in {source_root}")

    rows: list[ExternalPairRecord] = []
    for shard_idx, payload in enumerate(
        _iter_parquet_rows(
            files=files,
            columns=("instruction", "chosen", "rejected"),
        )
    ):
        prompt_text = str(payload.get("instruction", ""))
        chosen_text = str(payload.get("chosen", ""))
        rejected_text = str(payload.get("rejected", ""))
        record = _build_record(
            source_tag="r_prm",
            domain_tag="general_math",
            prompt_text=prompt_text,
            chosen_text=chosen_text,
            rejected_text=rejected_text,
            pair_confidence=0.78,
            metadata={
                "source_split": split,
                "source_row_index": int(shard_idx),
                "source_root": str(root),
            },
            config=config,
        )
        if record is None:
            continue
        rows.append(record)
        if max_pairs is not None and len(rows) >= int(max_pairs):
            break
    return rows


def load_prmbench_preview_pairs(
    *,
    path: Path,
    config: PairBuildConfig,
    max_pairs: int | None = None,
) -> list[ExternalPairRecord]:
    """Load pair candidates from PRMBench preview JSONL."""
    config.validate()
    if not path.exists():
        raise FileNotFoundError(f"PRMBench preview JSONL not found: {path}")

    rows: list[ExternalPairRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            payload = json.loads(text)
            question = str(payload.get("question") or payload.get("modified_question") or "")
            original_process = payload.get("original_process")
            modified_process = payload.get("modified_process")
            error_steps = payload.get("error_steps")
            if not isinstance(original_process, list) or not isinstance(modified_process, list):
                continue
            if not isinstance(error_steps, list):
                continue
            for error_step in error_steps:
                try:
                    # PRMBench stores 1-based step indices in most rows.
                    idx = int(error_step)
                except Exception:  # noqa: BLE001
                    continue
                idx = idx - 1 if idx > 0 else idx
                if idx < 0:
                    continue
                if idx >= len(original_process) or idx >= len(modified_process):
                    continue
                chosen_text = _join_steps_as_prefix(original_process, idx)
                rejected_text = _join_steps_as_prefix(modified_process, idx)
                confidence = 0.86 if len(error_steps) > 0 else 0.8
                record = _build_record(
                    source_tag="prmbench_preview",
                    domain_tag="general_math",
                    prompt_text=f"{question}\n\n",
                    chosen_text=chosen_text,
                    rejected_text=rejected_text,
                    pair_confidence=confidence,
                    metadata={
                        "source_row_line": int(line_no),
                        "source_idx": payload.get("idx"),
                        "classification": payload.get("classification"),
                        "error_step_index": int(idx),
                        "num_error_steps": int(len(error_steps)),
                    },
                    config=config,
                )
                if record is None:
                    continue
                rows.append(record)
                if max_pairs is not None and len(rows) >= int(max_pairs):
                    return rows
    return rows


def load_math_shepherd_pairs(
    *,
    path: Path,
    config: PairBuildConfig,
    max_pairs: int | None = None,
) -> list[ExternalPairRecord]:
    """Convert Math-Shepherd step labels into pair candidates."""
    config.validate()
    if not path.exists():
        raise FileNotFoundError(f"Math-Shepherd JSONL not found: {path}")

    rows: list[ExternalPairRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            payload = json.loads(text)
            label_text = str(payload.get("label", ""))
            task = str(payload.get("task", "")).strip().lower()
            if not label_text:
                continue
            prompt = _extract_problem_prefix(label_text)
            steps = _extract_step_labels_from_math_shepherd(label_text)
            domain_tag = "gsm8k_math" if task == "gsm8k" else "general_math"
            converted = _convert_step_labels_to_pairs(
                prompt_text=f"{prompt}\n\n",
                step_labels=steps,
                source_tag="math_shepherd",
                domain_tag=domain_tag,
                config=config,
                base_metadata={
                    "source_line": int(line_no),
                    "task": payload.get("task"),
                },
            )
            rows.extend(converted)
            if max_pairs is not None and len(rows) >= int(max_pairs):
                return rows[: int(max_pairs)]
    return rows


def load_prm800k_pairs(
    *,
    path: Path,
    config: PairBuildConfig,
    max_pairs: int | None = None,
) -> list[ExternalPairRecord]:
    """Convert PRM800K records into canonical pair candidates.

    Notes
    -----
    PRM800K releases have multiple community mirrors and schema variants.
    This loader intentionally supports several common shapes and gracefully
    skips rows that cannot be parsed into explicit `+/-` step labels.
    """
    config.validate()
    files = _collect_prm800k_files(path)
    rows: list[ExternalPairRecord] = []
    for file_path in files:
        for row_idx, payload in enumerate(_iter_json_records(file_path), start=1):
            prompt = _extract_prm800k_prompt(payload)
            direct_pairs = _extract_prm800k_completion_pairs(
                payload=payload,
                prompt_text=prompt,
                source_file=file_path,
                source_row_index=int(row_idx),
                config=config,
            )
            if direct_pairs:
                rows.extend(direct_pairs)
                if max_pairs is not None and len(rows) >= int(max_pairs):
                    return rows[: int(max_pairs)]
                continue
            step_labels = _extract_step_labels_from_prm800k_payload(payload)
            converted = _convert_step_labels_to_pairs(
                prompt_text=f"{prompt}\n\n",
                step_labels=step_labels,
                source_tag="prm800k",
                domain_tag="general_math",
                config=config,
                base_metadata={
                    "source_file": str(file_path),
                    "source_row_index": int(row_idx),
                },
            )
            rows.extend(converted)
            if max_pairs is not None and len(rows) >= int(max_pairs):
                return rows[: int(max_pairs)]
    return rows


def _extract_prm800k_completion_pairs(
    *,
    payload: dict[str, Any],
    prompt_text: str,
    source_file: Path,
    source_row_index: int,
    config: PairBuildConfig,
) -> list[ExternalPairRecord]:
    """Build pairs from official PRM800K `label.steps[].completions[]` schema.

    Strategy:
    - For each step, select one positive completion (`rating > 0`) and one
      non-positive completion (`rating <= 0`).
    - Compose pair texts as prefixes over chosen history + current step variant.
    """
    label_obj = payload.get("label")
    if not isinstance(label_obj, dict):
        return []
    steps = label_obj.get("steps")
    if not isinstance(steps, list) or not steps:
        return []

    rows: list[ExternalPairRecord] = []
    history_steps: list[str] = []
    for step_idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        completions = step.get("completions")
        if not isinstance(completions, list):
            continue

        positive: tuple[str, float] | None = None
        non_positive: tuple[str, float] | None = None
        for item in completions:
            if not isinstance(item, dict):
                continue
            text_raw = item.get("text")
            if not isinstance(text_raw, str):
                continue
            text = text_raw.strip()
            if text == "":
                continue
            rating = _safe_rating_value(item.get("rating"))
            if rating is None:
                continue
            if rating > 0.0:
                if positive is None or rating > positive[1]:
                    positive = (text, rating)
            else:
                if non_positive is None or rating < non_positive[1]:
                    non_positive = (text, rating)

        if positive is not None and non_positive is not None:
            chosen_prefix = _join_steps_as_prefix(
                [*history_steps, positive[0]],
                len(history_steps),
            )
            rejected_prefix = _join_steps_as_prefix(
                [*history_steps, non_positive[0]],
                len(history_steps),
            )
            rating_gap = positive[1] - non_positive[1]
            if rating_gap >= 2.0:
                confidence = 0.78
            elif rating_gap >= 1.0:
                confidence = 0.70
            else:
                confidence = 0.62
            record = _build_record(
                source_tag="prm800k",
                domain_tag="general_math",
                prompt_text=f"{prompt_text}\n\n",
                chosen_text=chosen_prefix,
                rejected_text=rejected_prefix,
                pair_confidence=confidence,
                metadata={
                    "source_file": str(source_file),
                    "source_row_index": int(source_row_index),
                    "positive_step_index": int(step_idx),
                    "negative_step_index": int(step_idx),
                    "rating_positive": float(positive[1]),
                    "rating_negative": float(non_positive[1]),
                    "pair_build_mode": "prm800k_completion_ratings",
                },
                config=config,
            )
            if record is not None:
                rows.append(record)
                if len(rows) >= int(config.max_pairs_per_sample):
                    break

        # Keep prefix progression close to annotator-selected chain when possible.
        chosen_idx = step.get("chosen_completion")
        next_history = None
        if isinstance(chosen_idx, int) and 0 <= chosen_idx < len(completions):
            chosen_item = completions[chosen_idx]
            if isinstance(chosen_item, dict):
                text_raw = chosen_item.get("text")
                if isinstance(text_raw, str) and text_raw.strip():
                    next_history = text_raw.strip()
        if next_history is None and positive is not None:
            next_history = positive[0]
        if next_history is not None:
            history_steps.append(next_history)
    return rows


def load_rlhflow_pairs(
    *,
    mistral_root: Path | None,
    deepseek_path: Path | None,
    config: PairBuildConfig,
    max_pairs_per_source: int | None = None,
) -> list[ExternalPairRecord]:
    """Convert RLHFlow step-label conversations into pair candidates."""
    config.validate()
    rows: list[ExternalPairRecord] = []
    if mistral_root is not None:
        rows.extend(
            _load_rlhflow_mistral_pairs(
                root=mistral_root,
                config=config,
                max_pairs=max_pairs_per_source,
            )
        )
    if deepseek_path is not None:
        rows.extend(
            _load_rlhflow_deepseek_pairs(
                path=deepseek_path,
                config=config,
                max_pairs=max_pairs_per_source,
            )
        )
    return rows


def _load_rlhflow_mistral_pairs(
    *,
    root: Path,
    config: PairBuildConfig,
    max_pairs: int | None = None,
) -> list[ExternalPairRecord]:
    source_root = root / "data"
    files = sorted(source_root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No RLHFlow mistral shards found in {source_root}")

    rows: list[ExternalPairRecord] = []
    for row_idx, payload in enumerate(_iter_parquet_rows(files=files, columns=("conversations",))):
        conv = payload.get("conversations")
        if not isinstance(conv, list):
            continue
        step_labels = _extract_step_labels_from_conversations(conv)
        prompt = _extract_problem_prefix(str(conv[0].get("content", ""))) if conv else ""
        converted = _convert_step_labels_to_pairs(
            prompt_text=f"{prompt}\n\n",
            step_labels=step_labels,
            source_tag="rlhflow_mistral",
            domain_tag="general_math",
            config=config,
            base_metadata={
                "source_row_index": int(row_idx),
                "source_root": str(root),
            },
        )
        rows.extend(converted)
        if max_pairs is not None and len(rows) >= int(max_pairs):
            return rows[: int(max_pairs)]
    return rows


def _load_rlhflow_deepseek_pairs(
    *,
    path: Path,
    config: PairBuildConfig,
    max_pairs: int | None = None,
) -> list[ExternalPairRecord]:
    if not path.exists():
        raise FileNotFoundError(f"RLHFlow deepseek JSONL not found: {path}")
    rows: list[ExternalPairRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            payload = json.loads(text)
            conv = payload.get("conversations")
            if not isinstance(conv, list):
                continue
            step_labels = _extract_step_labels_from_conversations(conv)
            prompt = _extract_problem_prefix(str(conv[0].get("content", ""))) if conv else ""
            converted = _convert_step_labels_to_pairs(
                prompt_text=f"{prompt}\n\n",
                step_labels=step_labels,
                source_tag="rlhflow_deepseek",
                domain_tag="general_math",
                config=config,
                base_metadata={
                    "source_line": int(line_no),
                    "source_path": str(path),
                },
            )
            rows.extend(converted)
            if max_pairs is not None and len(rows) >= int(max_pairs):
                return rows[: int(max_pairs)]
    return rows


def _iter_parquet_rows(
    *,
    files: list[Path],
    columns: Iterable[str],
) -> Iterable[dict[str, Any]]:
    """Yield parquet rows as plain dictionaries."""
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import `pyarrow.parquet` while reading external parquet shards. "
            "Please fix your pyarrow runtime first, for example: "
            "`python -m pip install -U pyarrow` "
            "and make sure no mixed conda/pip binary conflict remains."
        ) from exc

    for file_path in files:
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(batch_size=1024, columns=list(columns)):
            names = list(batch.schema.names)
            cols = [batch.column(i) for i in range(len(names))]
            for row_idx in range(batch.num_rows):
                payload: dict[str, Any] = {}
                for col_idx, name in enumerate(names):
                    payload[name] = cols[col_idx][row_idx].as_py()
                yield payload


def _collect_prm800k_files(path: Path) -> list[Path]:
    """Resolve PRM800K input path into concrete JSON/JSONL files."""
    if not path.exists():
        raise FileNotFoundError(f"PRM800K path not found: {path}")
    if path.is_file():
        return [path]

    files = sorted(path.rglob("*.jsonl")) + sorted(path.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON/JSONL files found under PRM800K path: {path}")
    return files


def _iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON object rows from JSONL or JSON files.

    Supported JSON structures:
    1. JSONL: one dict per line.
    2. JSON list: `[ {...}, {...} ]`
    3. JSON dict with list payload under common keys such as `data`, `records`,
       or `examples`.
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                text = raw.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    yield payload
                else:
                    raise TypeError(f"{path}:{line_no} must be a JSON object")
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
        return
    if isinstance(payload, dict):
        for key in ("data", "records", "examples", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        yield row
                return
        # Some mirrors store one sample per JSON file.
        yield payload
        return
    raise TypeError(f"Unsupported JSON payload type in {path}: {type(payload)!r}")


def _extract_problem_prefix(text: str) -> str:
    """Extract question/problem prefix before step markers."""
    normalized = " ".join(str(text).replace("\r", "\n").split())
    match = re.split(r"(?i)\bstep\s*1\s*:", normalized, maxsplit=1)
    if not match:
        return normalized[:512]
    return match[0].strip()[:512]


def _extract_prm800k_prompt(payload: dict[str, Any]) -> str:
    """Extract a question/prompt string from one PRM800K row."""
    for key in ("question", "problem", "prompt", "input", "query"):
        value = payload.get(key)
        text = _coerce_question_text(value)
        if text:
            return text[:1024]
    # Fallback to empty prompt to keep deterministic behavior.
    return ""


def _coerce_question_text(value: Any) -> str:
    """Coerce mixed question payloads into one plain string."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("problem", "question", "prompt", "text", "input"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""


def _extract_step_labels_from_math_shepherd(label_text: str) -> list[tuple[str, str]]:
    """Extract `(step_text, label)` from Math-Shepherd `label` field."""
    rows: list[tuple[str, str]] = []
    lines = [line.strip() for line in str(label_text).replace("\r", "\n").split("\n")]
    for line in lines:
        if not line or not line.lower().startswith("step "):
            continue
        match = re.match(r"^(Step\s+\d+\s*:\s*.*?)(?:\s+([+-]))\s*$", line)
        if not match:
            continue
        step_text = str(match.group(1)).strip()
        label = str(match.group(2)).strip()
        if label not in {"+", "-"}:
            continue
        rows.append((step_text, label))
    return rows


def _extract_step_labels_from_prm800k_payload(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract `(step_text, +/-)` rows from one PRM800K-like payload."""
    # Path A: textual `label` with "Step ... +/-" format.
    raw_label = payload.get("label")
    if isinstance(raw_label, str):
        parsed = _extract_step_labels_from_suffix_text(raw_label)
        if parsed:
            return parsed

    # Path B: explicit step container under common fields.
    step_candidates: list[Any] = []
    if isinstance(payload.get("steps"), list):
        step_candidates.extend(list(payload["steps"]))
    label_obj = payload.get("label")
    if isinstance(label_obj, dict):
        if isinstance(label_obj.get("steps"), list):
            step_candidates.extend(list(label_obj.get("steps")))
        if isinstance(label_obj.get("process"), list):
            step_candidates.extend(list(label_obj.get("process")))
    if isinstance(payload.get("process"), list):
        step_candidates.extend(list(payload["process"]))

    rows: list[tuple[str, str]] = []
    for step in step_candidates:
        parsed = _extract_one_step_label(step)
        if parsed is None:
            continue
        rows.append(parsed)
    return rows


def _extract_step_labels_from_suffix_text(text: str) -> list[tuple[str, str]]:
    """Parse lines like `Step 2: ... +` or `Step 3: ... -`."""
    rows: list[tuple[str, str]] = []
    lines = [line.strip() for line in str(text).replace("\r", "\n").split("\n")]
    for line in lines:
        if not line:
            continue
        match = re.match(r"^(Step\s+\d+\s*:\s*.*?)(?:\s+([+-]))\s*$", line)
        if not match:
            continue
        step_text = str(match.group(1)).strip()
        sign = str(match.group(2)).strip()
        if sign in {"+", "-"}:
            rows.append((step_text, sign))
    return rows


def _extract_one_step_label(step: Any) -> tuple[str, str] | None:
    """Extract one `(step_text, +/-)` tuple from mixed step payload formats."""
    if isinstance(step, str):
        # String-only step cannot infer correctness label.
        return None
    if not isinstance(step, dict):
        return None

    text = ""
    for key in ("text", "content", "step", "completion", "output"):
        value = step.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            break
    if text == "":
        return None

    sign = _parse_sign_value(
        step.get("label", step.get("rating", step.get("is_correct", step.get("correct", step.get("value")))))
    )
    if sign is None:
        return None
    return text, sign


def _parse_sign_value(value: Any) -> str | None:
    """Map mixed label/rating payloads to `+` or `-`."""
    if isinstance(value, bool):
        return "+" if value else "-"
    if isinstance(value, (int, float)):
        return "+" if float(value) > 0 else "-"
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"+", "correct", "true", "yes", "good", "positive", "1"}:
            return "+"
        if norm in {"-", "incorrect", "false", "no", "bad", "negative", "0"}:
            return "-"
    return None


def _safe_rating_value(value: Any) -> float | None:
    """Convert one rating field to float, ignoring invalid values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return None
    return out


def _extract_step_labels_from_conversations(conversations: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Extract `(step_text, label)` from RLHFlow conversation turns."""
    rows: list[tuple[str, str]] = []
    for idx in range(len(conversations) - 1):
        left = conversations[idx]
        right = conversations[idx + 1]
        if str(left.get("role", "")).strip().lower() != "user":
            continue
        if str(right.get("role", "")).strip().lower() != "assistant":
            continue
        label = str(right.get("content", "")).strip()
        if label not in {"+", "-"}:
            continue
        step_text = str(left.get("content", "")).strip()
        if step_text == "":
            continue
        rows.append((step_text, label))
    return rows


def _convert_step_labels_to_pairs(
    *,
    prompt_text: str,
    step_labels: list[tuple[str, str]],
    source_tag: str,
    domain_tag: str,
    config: PairBuildConfig,
    base_metadata: dict[str, Any],
) -> list[ExternalPairRecord]:
    """Convert `+/-` step labels into prefix-level chosen/rejected pairs."""
    positives = [idx for idx, (_, label) in enumerate(step_labels) if label == "+"]
    negatives = [idx for idx, (_, label) in enumerate(step_labels) if label == "-"]
    if not positives or not negatives:
        return []

    rows: list[ExternalPairRecord] = []
    used_pairs: set[tuple[int, int]] = set()
    for pos_idx in positives:
        if len(rows) >= int(config.max_pairs_per_sample):
            break
        # 中文：这里不是拿“任意一个负样本”，而是优先选距离最近的错误步骤。
        # 目的不是构造最难/最强 pair，而是让 chosen/rejected 共享尽量多上下文，
        # 这样 ranking 信号更像“这一步好坏”的差异，而不是整条长推理完全不同。
        neg_idx = min(negatives, key=lambda n: abs(n - pos_idx))
        key = (pos_idx, neg_idx)
        if key in used_pairs:
            continue
        used_pairs.add(key)
        chosen_text = _join_steps_as_prefix([step for step, _ in step_labels], pos_idx)
        rejected_text = _join_steps_as_prefix([step for step, _ in step_labels], neg_idx)
        distance = abs(pos_idx - neg_idx)
        # 中文：当前 confidence 是经验型启发式，不是假设它等于“真实概率”。
        # 距离近说明两个 prefix 只在局部步骤附近分叉，通常更适合作为高质量排序监督。
        confidence = 0.62 if distance <= 2 else 0.55
        record = _build_record(
            source_tag=source_tag,
            domain_tag=domain_tag,
            prompt_text=prompt_text,
            chosen_text=chosen_text,
            rejected_text=rejected_text,
            pair_confidence=confidence,
            metadata={
                **base_metadata,
                "positive_step_index": int(pos_idx),
                "negative_step_index": int(neg_idx),
                "num_step_labels": int(len(step_labels)),
            },
            config=config,
        )
        if record is not None:
            rows.append(record)
    return rows


def _join_steps_as_prefix(steps: list[Any], step_index: int) -> str:
    """Join step list as one prefix ending at `step_index` (inclusive)."""
    if not steps:
        return ""
    end = min(max(int(step_index), 0), len(steps) - 1)
    fragments = [str(step).strip() for step in steps[: end + 1] if str(step).strip()]
    return "\n".join(fragments) + ("\n" if fragments else "")


def _build_record(
    *,
    source_tag: str,
    domain_tag: str,
    prompt_text: str,
    chosen_text: str,
    rejected_text: str,
    pair_confidence: float,
    metadata: dict[str, Any],
    config: PairBuildConfig,
) -> ExternalPairRecord | None:
    """Build one validated canonical pair record, or return None if filtered."""
    flags = _compute_quality_flags(
        prompt_text=prompt_text,
        chosen_text=chosen_text,
        rejected_text=rejected_text,
    )
    if not _passes_quality_filter(flags=flags, config=config):
        # 中文：质量过滤在 adapter 层提前做，而不是等到训练再发现坏样本。
        # 这样 summary.json 能直接反映“外部 pair 源本身质量如何”，便于调数据。
        return None
    pair_id = _stable_hash(
        source_tag,
        domain_tag,
        _normalize_space(prompt_text),
        _normalize_space(chosen_text),
        _normalize_space(rejected_text),
    )[:20]
    # 中文：pair_id 由内容稳定哈希生成，而不是依赖行号。
    # 好处是不同批次重建/合并多个来源时，去重和复现都更可靠。
    record = ExternalPairRecord(
        pair_id=pair_id,
        source_tag=str(source_tag).strip(),
        domain_tag=str(domain_tag).strip(),
        prompt_text=str(prompt_text),
        chosen_text=str(chosen_text),
        rejected_text=str(rejected_text),
        pair_confidence=float(min(max(pair_confidence, 0.0), 1.0)),
        quality_flags=flags,
        metadata=dict(metadata),
    )
    record.validate()
    return record


def _compute_quality_flags(
    *,
    prompt_text: str,
    chosen_text: str,
    rejected_text: str,
) -> dict[str, Any]:
    chosen_norm = _normalize_space(chosen_text)
    rejected_norm = _normalize_space(rejected_text)
    chosen_chars = len(chosen_norm)
    rejected_chars = len(rejected_norm)
    min_len = min(chosen_chars, rejected_chars) if chosen_chars and rejected_chars else 0
    max_len = max(chosen_chars, rejected_chars) if chosen_chars or rejected_chars else 0
    ratio = float(max_len / max(min_len, 1))
    chosen_tokens = set(chosen_norm.split())
    rejected_tokens = set(rejected_norm.split())
    union = chosen_tokens | rejected_tokens
    overlap = (len(chosen_tokens & rejected_tokens) / len(union)) if union else 1.0
    # 中文：这里保留的是“可解释的质量特征”，而不是直接在此处做 hard filter。
    # 后续 config 可以基于这些 flag 改阈值，避免把过滤逻辑散落在多处。
    return {
        "chosen_chars": int(chosen_chars),
        "rejected_chars": int(rejected_chars),
        "length_ratio": float(ratio),
        "token_overlap_ratio": float(overlap),
        "non_empty": bool(chosen_chars > 0 and rejected_chars > 0),
        "distinct_pair": bool(chosen_norm != rejected_norm),
        "prompt_chars": int(len(_normalize_space(prompt_text))),
    }


def _passes_quality_filter(*, flags: dict[str, Any], config: PairBuildConfig) -> bool:
    if not bool(flags.get("non_empty", False)):
        return False
    if not bool(flags.get("distinct_pair", False)):
        return False
    if int(flags.get("chosen_chars", 0)) < int(config.min_chars):
        return False
    if int(flags.get("rejected_chars", 0)) < int(config.min_chars):
        return False
    if float(flags.get("length_ratio", 999.0)) > float(config.max_length_ratio):
        return False
    if float(flags.get("token_overlap_ratio", 1.0)) > float(config.max_token_overlap):
        return False
    return True


def _normalize_space(text: str) -> str:
    return " ".join(str(text).replace("\r", "\n").split())


def _stable_hash(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()
