"""Source adapters for building canonical external pair artifacts.

English
-------
Public process-supervision datasets come in very different raw formats:

1. some already provide direct `chosen/rejected` pairs,
2. some provide one trajectory with step-level `+/-` labels,
3. some provide multiple completions for the same step.

The rest of the repo does not want to understand every source-specific schema.
It only wants one canonical contract:

1. `prompt_text`
2. `chosen_text`
3. `rejected_text`
4. metadata explaining how the pair was built

This file performs that normalization.

中文
----
公开的 process supervision 数据源原始格式差别很大：

1. 有些直接给 `chosen/rejected`；
2. 有些只给一条轨迹和逐步 `+/-` 标签；
3. 有些给的是同一步的多个 completion。

但仓库后面的训练/评测代码不想理解每一种原始格式，它们只想消费统一 pair：

1. `prompt_text`
2. `chosen_text`
3. `rejected_text`
4. 以及说明 pair 如何构造出来的 metadata

本文件就是把各种外部格式标准化成统一 pair contract 的地方。

Important note / 关键提醒
-------------------------
For step-labeled sources, "convert to pair" is not a neutral ETL step.  It
encodes a research assumption about what the supervision actually means.

对于 step-labeled 数据源，把它“转换成 pair”并不是中性的 ETL 操作，而是在编码一种
研究假设：这个监督信号到底代表什么。
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
    """Quality-control config shared by external-pair source adapters.

    English
    -------
    This config controls both:

    1. ordinary quality filtering
       - remove degenerate or low-information pairs
    2. pair semantics for step-labeled sources
       - choose how `+/-` trajectories become canonical pairs

    中文
    ----
    这组配置不只是在做普通清洗，还同时决定两件事：

    1. 质量过滤
       - 去掉明显退化、信息量很低的 pair
    2. step-labeled 数据的 pair 语义
       - 决定 `+/-` 轨迹如何变成 canonical pair
    """

    min_chars: int = 12
    max_length_ratio: float = 4.0
    max_token_overlap: float = 0.995
    max_pairs_per_sample: int = 2
    step_label_pair_mode: str = "first_bad_edge_strict"

    def validate(self) -> None:
        if self.min_chars <= 0:
            raise ValueError("`min_chars` must be > 0")
        if self.max_length_ratio <= 1.0:
            raise ValueError("`max_length_ratio` must be > 1")
        if not (0.0 <= self.max_token_overlap <= 1.0):
            raise ValueError("`max_token_overlap` must be in [0, 1]")
        if self.max_pairs_per_sample <= 0:
            raise ValueError("`max_pairs_per_sample` must be > 0")
        if self.step_label_pair_mode not in {"first_bad_edge_strict", "legacy_nearest"}:
            raise ValueError(
                "`step_label_pair_mode` must be one of "
                "{'first_bad_edge_strict', 'legacy_nearest'}"
            )


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
                "pair_build_mode": "r_prm_direct_pair",
                "pair_semantics": "direct_preference_pair",
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
                        "pair_build_mode": "prmbench_explicit_error_step",
                        "pair_semantics": "local_modified_process_error_step",
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
    """Convert Math-Shepherd step labels into local first-bad-edge pair candidates.

    Important
    ---------
    Math-Shepherd only provides one labeled trajectory, not same-step sibling
    alternatives. The safe default is therefore `first_bad_edge_strict`:

    1. find the first negative step,
    2. require at least one positive step before it,
    3. compare the last clean prefix before that step against the prefix that
       includes the first bad step.

    This should be narrated as first-bad-edge supervision, not as exact
    same-step branch preference supervision.

    中文补充
    --------
    这里最容易让新手误解的一点是：Math-Shepherd 并没有给你严格的
    “同一个 prefix 下，当前步的好候选 vs 坏候选”。

    它给的是：
    1. 一条过程轨迹
    2. 每步好坏标签

    所以当前最保守、最诚实的转换，是“最后一个好 prefix”对“第一个坏 prefix”。
    这代表 local first-bad-edge，不代表严格 sibling-branch preference。
    """
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
                    "pair_semantics": "same_step_completion_preference",
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
    """Convert RLHFlow step-label conversations into local first-bad-edge pairs."""
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
    column_names = list(columns)
    try:
        import pyarrow.parquet as pq
        reader_mode = "pyarrow.parquet"
    except Exception as exc:  # noqa: BLE001
        # Some of our lab environments end up with a mixed pyarrow install where
        # `pyarrow.parquet` imports the broken fs/azure bindings, but the low-level
        # parquet reader still works. 这里优先降级到 `_parquet`，避免整个 Phase E
        # 因为 parquet runtime 的 ABI 冲突而完全不可用。
        try:
            import pyarrow._parquet as pq  # type: ignore[no-redef]
            reader_mode = "pyarrow._parquet"
        except Exception as fallback_exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to import a usable parquet reader while reading external parquet shards. "
                "Tried `pyarrow.parquet` and fallback `pyarrow._parquet`. "
                "Please fix your pyarrow runtime first, for example: "
                "`python -m pip install -U pyarrow` "
                "and make sure no mixed conda/pip binary conflict remains."
            ) from fallback_exc

    for file_path in files:
        if reader_mode == "pyarrow.parquet":
            pf = pq.ParquetFile(file_path)
            batches = pf.iter_batches(batch_size=1024, columns=column_names)
        else:
            reader = pq.ParquetReader()
            reader.open(
                str(file_path),
                use_memory_map=False,
                read_dictionary=[],
                buffer_size=0,
                pre_buffer=False,
            )
            schema_names = list(reader.schema_arrow.names)
            col_indices = [schema_names.index(name) for name in column_names]
            row_groups = list(range(reader.num_row_groups))
            batches = reader.iter_batches(
                batch_size=1024,
                row_groups=row_groups,
                column_indices=col_indices,
                use_threads=True,
            )
        for batch in batches:
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
    """Convert `+/-` step labels into local first-bad-edge or legacy pairs.

    Important
    ---------
    Step-labeled sources such as Math-Shepherd / RLHFlow only provide one
    trajectory with per-step `+/-` tags. They do *not* natively provide
    same-step sibling alternatives.

    Safe default:
    1. `first_bad_edge_strict`
       - find the first negative step,
       - require at least one positive step before it,
       - compare the last clean prefix against the prefix that includes the
         first bad step.

    Legacy mode remains available only for historical forensics:
    1. `legacy_nearest`
       - compare a positive-index prefix with the nearest negative-index prefix
         from the same trajectory.
       - This mixes depth/progress signal with local error signal and should
         not be used for new mainline experiments.

    中文
    ----
    这个分支点本身就有研究含义：

    1. `first_bad_edge_strict`
       - 语义更干净，更接近“局部第一次出错”
       - 代价是会丢掉一部分难以严格构造的样本
    2. `legacy_nearest`
       - 样本可能更多
       - 但会把“推理深度/进度”与“局部错误”混在一起

    所以后面的实验结果怎么解释，很大程度取决于这里选的是哪条路径。
    """
    if config.step_label_pair_mode == "first_bad_edge_strict":
        return _convert_step_labels_to_first_bad_edge_pairs(
            prompt_text=prompt_text,
            step_labels=step_labels,
            source_tag=source_tag,
            domain_tag=domain_tag,
            config=config,
            base_metadata=base_metadata,
        )
    if config.step_label_pair_mode == "legacy_nearest":
        return _convert_step_labels_to_legacy_pairs(
            prompt_text=prompt_text,
            step_labels=step_labels,
            source_tag=source_tag,
            domain_tag=domain_tag,
            config=config,
            base_metadata=base_metadata,
        )
    raise ValueError(f"Unsupported step_label_pair_mode: {config.step_label_pair_mode!r}")


def _convert_step_labels_to_first_bad_edge_pairs(
    *,
    prompt_text: str,
    step_labels: list[tuple[str, str]],
    source_tag: str,
    domain_tag: str,
    config: PairBuildConfig,
    base_metadata: dict[str, Any],
) -> list[ExternalPairRecord]:
    """Build one strict local first-bad-edge pair from a labeled trajectory.

    English
    -------
    Example:
    1. step 0 = good
    2. step 1 = good
    3. step 2 = bad

    Then we emit exactly one pair:
    1. chosen = prefix through step 1
    2. rejected = prefix through step 2

    中文
    ----
    举例：
    1. 第 0 步是好
    2. 第 1 步是好
    3. 第 2 步是坏

    那么这里只会构造一个 pair：
    1. chosen = 截到第 1 步的 prefix
    2. rejected = 截到第 2 步的 prefix
    """
    first_negative_idx = next((idx for idx, (_, label) in enumerate(step_labels) if label == "-"), None)
    if first_negative_idx is None:
        return []
    if int(first_negative_idx) <= 0:
        # If the first negative label appears immediately, there is no reliable
        # "last good prefix" to compare against.
        # 如果一开始就出现负标签，就不存在可信的“最后一个好 prefix”。
        #
        # Keeping such rows would force us to invent a pseudo-positive sample,
        # which would quietly change the semantics of the supervision.
        # 这类样本如果强行保留，相当于硬造一个正样本，会悄悄改变监督语义。
        return []

    chosen_idx = int(first_negative_idx) - 1
    rejected_idx = int(first_negative_idx)
    chosen_text = _join_steps_as_prefix([step for step, _ in step_labels], chosen_idx)
    rejected_text = _join_steps_as_prefix([step for step, _ in step_labels], rejected_idx)
    record = _build_record(
        source_tag=source_tag,
        domain_tag=domain_tag,
        prompt_text=prompt_text,
        chosen_text=chosen_text,
        rejected_text=rejected_text,
        pair_confidence=0.74,
        metadata={
            **base_metadata,
            "positive_step_index": int(chosen_idx),
            "negative_step_index": int(rejected_idx),
            "first_negative_index": int(first_negative_idx),
            "num_step_labels": int(len(step_labels)),
            "pair_build_mode": "step_label_first_bad_edge_strict",
            "pair_semantics": "local_first_bad_edge",
        },
        config=config,
    )
    return [record] if record is not None else []


def _convert_step_labels_to_legacy_pairs(
    *,
    prompt_text: str,
    step_labels: list[tuple[str, str]],
    source_tag: str,
    domain_tag: str,
    config: PairBuildConfig,
    base_metadata: dict[str, Any],
) -> list[ExternalPairRecord]:
    """Legacy nearest-boundary converter retained for forensic reproducibility.

    English
    -------
    This mode is kept so old experiments remain reproducible.  It should not be
    the default interpretation for new mainline science.

    The main problem is that chosen/rejected often terminate at different
    depths in the same trajectory, so the model may learn:
    1. progress depth,
    2. position in trajectory,
    3. or text length artifacts,
    rather than purely local error discrimination.

    中文
    ----
    这个模式保留主要是为了让旧实验还能复现，不该再作为新主线默认语义。

    它的核心问题是 chosen/rejected 往往结束在同一条轨迹的不同深度，所以模型
    可能学到的是：
    1. 推理走了多远，
    2. 当前在轨迹哪个位置，
    3. 文本长度差异，
    而不是真正的局部错误识别。
    """
    positives = [idx for idx, (_, label) in enumerate(step_labels) if label == "+"]
    negatives = [idx for idx, (_, label) in enumerate(step_labels) if label == "-"]
    if not positives or not negatives:
        return []

    rows: list[ExternalPairRecord] = []
    used_pairs: set[tuple[int, int]] = set()
    for pos_idx in positives:
        if len(rows) >= int(config.max_pairs_per_sample):
            break
        # Pick the nearest negative step rather than an arbitrary one so the
        # pair shares as much context as possible.
        # 这里优先选距离最近的错误步骤，而不是随便拿一个负样本，尽量让 pair 共享更多上下文。
        #
        # Caution: the two prefixes still end at different step indices, so
        # this is only a heuristic same-trajectory comparison, not a true
        # same-step local branch pair.
        # 注意：这里比较的仍然是两个不同步数终止的 prefix，所以它只是“同轨迹近邻对比”，
        # 不是严格意义上的“同一步分叉的局部好坏对比”。
        neg_idx = min(negatives, key=lambda n: abs(n - pos_idx))
        key = (pos_idx, neg_idx)
        if key in used_pairs:
            continue
        used_pairs.add(key)
        chosen_text = _join_steps_as_prefix([step for step, _ in step_labels], pos_idx)
        rejected_text = _join_steps_as_prefix([step for step, _ in step_labels], neg_idx)
        distance = abs(pos_idx - neg_idx)
        # This confidence is heuristic metadata, not a calibrated probability.
        # 当前 confidence 只是经验型启发式元数据，不是假设它等于“真实概率”。
        # Shorter step distance usually means the two prefixes diverge only locally and make a cleaner ranking pair.
        # 距离越近通常说明两个 prefix 只在局部步骤附近分叉，更适合作为高质量排序监督。
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
                "pair_build_mode": "step_label_legacy_nearest_boundary",
                "pair_semantics": "same_trajectory_depth_mixed",
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
    """Build one validated canonical pair record, or return None if filtered.

    English
    -------
    This is the final source-agnostic quality gate before a pair enters the
    repo-wide canonical contract.

    中文
    ----
    这是 source-specific pair 进入“全仓统一 canonical pair”之前的最后一道门。
    如果这里不过滤，后面的训练和评测都会把它当成正式监督样本。
    """
    flags = _compute_quality_flags(
        prompt_text=prompt_text,
        chosen_text=chosen_text,
        rejected_text=rejected_text,
    )
    if not _passes_quality_filter(flags=flags, config=config):
        # Reject low-quality pairs here so the adapter summary already reflects source quality.
        # 质量过滤要在 adapter 层提前做，这样 summary.json 才能直接反映外部 pair 源本身质量。
        return None
    pair_id = _stable_hash(
        source_tag,
        domain_tag,
        _normalize_space(prompt_text),
        _normalize_space(chosen_text),
        _normalize_space(rejected_text),
    )[:20]
    # Content-hash ids make deduplication and reruns stable even when file order changes.
    # 基于内容哈希的 pair_id 能让去重与复现不受文件顺序变化影响。
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
    # Keep interpretable quality features here and centralize the actual thresholding elsewhere.
    # 这里先保留可解释的质量特征，再由统一配置决定阈值，避免过滤逻辑散落多处。
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
