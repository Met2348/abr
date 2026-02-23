"""Dataset loaders that normalize multiple sources into one canonical schema.

Beginner note
-------------
The most important design decision here is:
    every loader returns ``list[CanonicalSample]`` with the same fields.

This allows model/training/evaluation code to stay generic and reusable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from .schema import CanonicalSample, ensure_canonical_samples

# Keep cache under project folder by default to avoid permission issues in
# restricted/sandboxed environments.
DEFAULT_CACHE_DIR = Path("assets/hf_cache/datasets")


def load_dataset_canonical(
    dataset_name: str,
    dataset_root: str | Path = "assets/datasets",
    split: str = "train",
    limit: int | None = None,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    **kwargs: Any,
) -> list[CanonicalSample]:
    """Public entry point for loading any supported dataset.

    Parameters
    ----------
    dataset_name:
        One of keys from ``DATASET_LOADERS``.
    dataset_root:
        Root path containing downloaded dataset folders.
    split:
        Requested split. We normalize common aliases where needed.
    limit:
        Optional cap for quick smoke runs.
    cache_dir:
        HuggingFace datasets cache path for parquet/script loading.
    kwargs:
        Dataset-specific options (e.g. ``config`` for gsm8k).
    """
    name = dataset_name.strip().lower()
    if name not in DATASET_LOADERS:
        supported = ", ".join(sorted(DATASET_LOADERS.keys()))
        raise KeyError(f"Unknown dataset={dataset_name!r}. Supported: {supported}")

    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}. "
            "Did you run download_datasets.sh?"
        )

    loader = DATASET_LOADERS[name]
    samples = loader(
        dataset_root=root,
        split=split,
        limit=limit,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        **kwargs,
    )
    return ensure_canonical_samples(samples, source_name=name)


def load_gsm8k(
    dataset_root: Path,
    split: str = "train",
    limit: int | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    config: str = "main",
) -> list[CanonicalSample]:
    """Load GSM8K from local parquet files.

    ``config`` can be ``main`` or ``socratic``.
    """
    requested_split = split
    split = _normalize_split(split, available={"train", "test"}, fallback="test")
    if split != requested_split:
        _note(
            f"GSM8K does not provide split={requested_split!r}; "
            f"falling back to split={split!r}."
        )

    files = _glob_required(
        dataset_root / "gsm8k" / config, f"{split}-*.parquet", dataset="gsm8k"
    )
    rows = _load_parquet_rows(files, cache_dir=cache_dir, limit=limit)

    samples: list[CanonicalSample] = []
    for idx, row in enumerate(rows):
        question = _as_clean_str(row.get("question", ""))
        raw_answer = _as_clean_str(row.get("answer", ""))
        cot, final_answer = _split_gsm8k_answer(raw_answer)
        samples.append(
            CanonicalSample(
                id=f"gsm8k:{config}:{split}:{idx}",
                dataset="gsm8k",
                question=question,
                answer=final_answer,
                cot=cot,
                metadata={
                    "source_split": split,
                    "requested_split": requested_split,
                    "config": config,
                },
            )
        )
    return samples


def load_strategyqa(
    dataset_root: Path,
    split: str = "train",
    limit: int | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> list[CanonicalSample]:
    """Load StrategyQA from local JSON.

    Current snapshot in this repo includes only ``strategyQA_train.json``.
    """
    del cache_dir  # not used; kept in signature for a uniform loader interface

    filename_map = {
        "train": "strategyQA_train.json",
        "validation": "strategyQA_validation.json",
        "test": "strategyQA_test.json",
    }
    if split not in filename_map:
        raise ValueError("StrategyQA split must be one of: train/validation/test")

    path = dataset_root / "strategyqa" / filename_map[split]
    if not path.exists():
        if split != "train":
            raise FileNotFoundError(
                f"Requested split={split!r} but file is missing: {path}. "
                "Your local snapshot currently contains only train split."
            )
        raise FileNotFoundError(f"Missing StrategyQA file: {path}")

    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {path}, got {type(rows)!r}")
    if limit is not None:
        rows = rows[:limit]

    samples: list[CanonicalSample] = []
    for idx, row in enumerate(rows):
        question = _as_clean_str(row.get("question", ""))
        answer = _normalize_strategyqa_answer(row.get("answer"))
        decomposition = row.get("decomposition")
        cot: str | None = None
        if isinstance(decomposition, list) and decomposition:
            cot = "\n".join(f"- {str(step).strip()}" for step in decomposition)

        sample_id = _as_clean_str(str(row.get("qid", f"row-{idx}")))
        samples.append(
            CanonicalSample(
                id=f"strategyqa:{sample_id}",
                dataset="strategyqa",
                question=question,
                answer=answer,
                cot=cot,
                metadata={
                    "term": row.get("term"),
                    "description": row.get("description"),
                    "facts": row.get("facts"),
                },
            )
        )
    return samples


def load_drop(
    dataset_root: Path,
    split: str = "train",
    limit: int | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> list[CanonicalSample]:
    """Load DROP from local parquet files."""
    requested_split = split
    split = _normalize_split(split, available={"train", "validation"}, fallback="validation")
    if split != requested_split:
        _note(
            f"DROP does not provide split={requested_split!r}; "
            f"falling back to split={split!r}."
        )

    files = _glob_required(
        dataset_root / "drop" / "data", f"{split}-*.parquet", dataset="drop"
    )
    rows = _load_parquet_rows(files, cache_dir=cache_dir, limit=limit)

    samples: list[CanonicalSample] = []
    for idx, row in enumerate(rows):
        answer = _extract_drop_answer(row.get("answers_spans"))
        samples.append(
            CanonicalSample(
                id=f"drop:{split}:{idx}",
                dataset="drop",
                question=_as_clean_str(row.get("question", "")),
                answer=answer,
                cot=None,
                metadata={
                    "section_id": row.get("section_id"),
                    "query_id": row.get("query_id"),
                    "passage": row.get("passage"),
                    "source_split": split,
                    "requested_split": requested_split,
                },
            )
        )
    return samples


def load_proofwriter(
    dataset_root: Path,
    split: str = "train",
    limit: int | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> list[CanonicalSample]:
    """Load ProofWriter from local parquet files."""
    split = _normalize_split(split, available={"train", "validation", "test"}, fallback="validation")
    files = _glob_required(
        dataset_root / "proofwriter" / "data", f"{split}-*.parquet", dataset="proofwriter"
    )
    rows = _load_parquet_rows(files, cache_dir=cache_dir, limit=limit)

    samples: list[CanonicalSample] = []
    for idx, row in enumerate(rows):
        samples.append(
            CanonicalSample(
                id=f"proofwriter:{split}:{_as_clean_str(row.get('id', str(idx)))}",
                dataset="proofwriter",
                question=_as_clean_str(row.get("question", "")),
                answer=_as_clean_str(row.get("answer", "")),
                cot=_as_optional_clean_str(row.get("allProofs")),
                metadata={
                    "theory": row.get("theory"),
                    "maxD": row.get("maxD"),
                    "NFact": row.get("NFact"),
                    "NRule": row.get("NRule"),
                    "config": row.get("config"),
                    "source_split": split,
                },
            )
        )
    return samples


def load_bbh(
    dataset_root: Path,
    split: str = "test",
    limit: int | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    task: str = "boolean_expressions",
) -> list[CanonicalSample]:
    """Load one BBH task from local parquet files.

    Note: current local snapshot is task-based and usually test-only.
    """
    requested_split = split
    split = _normalize_split(split, available={"test"}, fallback="test")
    if split != requested_split:
        _note(
            f"BBH local snapshot is test-only; falling back to split={split!r}."
        )

    files = _glob_required(
        dataset_root / "bigbench_hard" / task, f"{split}-*.parquet", dataset=f"bbh/{task}"
    )
    rows = _load_parquet_rows(files, cache_dir=cache_dir, limit=limit)

    samples: list[CanonicalSample] = []
    for idx, row in enumerate(rows):
        samples.append(
            CanonicalSample(
                id=f"bbh:{task}:{split}:{idx}",
                dataset="bigbench_hard",
                question=_as_clean_str(row.get("input", "")),
                answer=_as_clean_str(row.get("target", "")),
                cot=None,
                metadata={
                    "task": task,
                    "source_split": split,
                    "requested_split": requested_split,
                },
            )
        )
    return samples


def load_hendrycks_math(
    dataset_root: Path,
    split: str = "train",
    limit: int | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    subset: str = "algebra",
) -> list[CanonicalSample]:
    """Load Hendrycks MATH from local parquet files.

    Each subset (algebra, geometry, ...) is a separate folder.
    """
    requested_split = split
    split = _normalize_split(split, available={"train", "test"}, fallback="test")
    if split != requested_split:
        _note(
            f"Hendrycks MATH does not provide split={requested_split!r}; "
            f"falling back to split={split!r}."
        )

    files = _glob_required(
        dataset_root / "hendrycks_math" / subset, f"{split}-*.parquet", dataset=f"hendrycks_math/{subset}"
    )
    rows = _load_parquet_rows(files, cache_dir=cache_dir, limit=limit)

    samples: list[CanonicalSample] = []
    for idx, row in enumerate(rows):
        solution = _as_clean_str(row.get("solution", ""))
        samples.append(
            CanonicalSample(
                id=f"hendrycks_math:{subset}:{split}:{idx}",
                dataset="hendrycks_math",
                question=_as_clean_str(row.get("problem", "")),
                # Dataset has rich worked solutions but no easy final-answer-only field.
                # For now we keep full solution in answer and store details in metadata.
                answer=solution,
                cot=None,
                metadata={
                    "subset": subset,
                    "level": row.get("level"),
                    "type": row.get("type"),
                    "source_split": split,
                    "requested_split": requested_split,
                },
            )
        )
    return samples


def load_logiqa(
    dataset_root: Path,
    split: str = "train",
    limit: int | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> list[CanonicalSample]:
    """Load LogiQA from local files.

    Preferred path:
    - local parquet files (if present)

    Fallback path:
    - raw text files Train.txt/Eval.txt/Test.txt
    """
    root = dataset_root / "logiqa"
    if not root.exists():
        raise FileNotFoundError(f"Missing logiqa directory: {root}")

    # Option A: parquet snapshot
    parquet_files = list(sorted(root.glob(f"{split}-*.parquet")))
    if parquet_files:
        rows = _load_parquet_rows(parquet_files, cache_dir=cache_dir, limit=limit)
        return _rows_to_logiqa_samples(rows=rows, split=split)

    # Option B: raw txt files
    txt_map = {
        "train": root / "Train.txt",
        "validation": root / "Eval.txt",
        "test": root / "Test.txt",
    }
    if split not in txt_map:
        raise ValueError("LogiQA split must be one of: train/validation/test")

    txt_path = txt_map[split]
    if txt_path.exists():
        rows = _parse_logiqa_raw_text(txt_path)
        if limit is not None:
            rows = rows[:limit]
        return _rows_to_logiqa_samples(rows=rows, split=split)

    raise FileNotFoundError(
        "LogiQA data files were not found in parquet or raw txt form. "
        "Try re-downloading with: "
        "`hf download lucasmccabe/logiqa --repo-type dataset --local-dir assets/datasets/logiqa`"
    )


DATASET_LOADERS: dict[str, Callable[..., list[CanonicalSample]]] = {
    "gsm8k": load_gsm8k,
    "strategyqa": load_strategyqa,
    "drop": load_drop,
    "proofwriter": load_proofwriter,
    "bigbench_hard": load_bbh,
    "bbh": load_bbh,
    "hendrycks_math": load_hendrycks_math,
    "logiqa": load_logiqa,
}


def _load_parquet_rows(
    data_files: list[Path],
    cache_dir: Path | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Read parquet files through Hugging Face datasets.

    Why datasets library?
    - common abstraction used later in training pipeline
    - handles parquet shards and schema safely
    """
    load_dataset = _import_hf_load_dataset()
    split = "train" if limit is None else f"train[:{limit}]"
    ds = load_dataset(
        "parquet",
        data_files=[str(p) for p in data_files],
        split=split,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return [dict(row) for row in ds]


def _import_hf_load_dataset():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # noqa: BLE001 - include pyarrow mismatch failures
        raise RuntimeError(
            "Failed to import `datasets` runtime. "
            "Please activate your clean environment first (for example `conda activate bcr`) "
            "and ensure pyarrow is compatible."
        ) from exc
    return load_dataset


def _glob_required(root: Path, pattern: str, dataset: str) -> list[Path]:
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found for dataset={dataset!r}, path={root}, pattern={pattern!r}."
        )
    return files


def _normalize_split(split: str, available: set[str], fallback: str) -> str:
    split = split.strip().lower()
    if split in available:
        return split
    if split in {"val", "valid"} and "validation" in available:
        return "validation"
    if split in {"dev"} and "validation" in available:
        return "validation"
    return fallback


def _split_gsm8k_answer(answer: str) -> tuple[str | None, str]:
    text = answer.strip()
    if "####" in text:
        # GSM8K uses "#### final_answer" marker at end.
        cot_text, final = text.rsplit("####", 1)
        cot_text = cot_text.strip()
        final = final.strip()
        return (cot_text if cot_text else None), (final if final else text)
    return None, text


def _normalize_strategyqa_answer(answer_value: Any) -> str:
    # StrategyQA commonly stores bool labels.
    if isinstance(answer_value, bool):
        return "yes" if answer_value else "no"
    text = str(answer_value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "yes"
    if text in {"false", "0", "no"}:
        return "no"
    return str(answer_value).strip()


def _extract_drop_answer(answers_spans: Any) -> str:
    if isinstance(answers_spans, dict):
        spans = answers_spans.get("spans")
        if isinstance(spans, list) and spans:
            return _as_clean_str(spans[0])
    return ""


def _rows_to_logiqa_samples(rows: list[dict[str, Any]], split: str) -> list[CanonicalSample]:
    samples: list[CanonicalSample] = []
    for idx, row in enumerate(rows):
        context = _as_clean_str(row.get("context", ""))
        query = _as_clean_str(row.get("query", ""))
        options = row.get("options", [])
        if not isinstance(options, list):
            options = []
        option_lines = []
        letters = ["A", "B", "C", "D"]
        for i, opt in enumerate(options):
            letter = letters[i] if i < len(letters) else str(i + 1)
            option_lines.append(f"{letter}. {str(opt).strip()}")
        options_block = "\n".join(option_lines)
        question = f"Context: {context}\nQuestion: {query}\nOptions:\n{options_block}".strip()

        correct_option = row.get("correct_option")
        answer = ""
        if isinstance(correct_option, int) and 0 <= correct_option < len(options):
            answer = str(options[correct_option]).strip()
        else:
            answer = str(correct_option).strip()

        samples.append(
            CanonicalSample(
                id=f"logiqa:{split}:{idx}",
                dataset="logiqa",
                question=question,
                answer=answer,
                cot=None,
                metadata={
                    "context": context,
                    "query": query,
                    "options": options,
                    "correct_option": correct_option,
                    "source_split": split,
                },
            )
        )
    return samples


def _parse_logiqa_raw_text(path: Path) -> list[dict[str, Any]]:
    """Parse raw LogiQA text format (8 lines per question block)."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    # Remove fully empty lines to stabilize block indexing.
    lines = [ln for ln in lines if ln != ""]
    if len(lines) < 8:
        raise ValueError(f"LogiQA raw file too short: {path}")

    rows: list[dict[str, Any]] = []
    n_blocks = len(lines) // 8
    for block_idx in range(n_blocks):
        base = block_idx * 8
        correct = lines[base + 1].replace(".", "").strip().lower()
        context = _process_logiqa_sentence(lines[base + 2])
        query = _process_logiqa_sentence(lines[base + 3])
        answers = [_process_logiqa_answer(lines[base + i]) for i in range(4, 8)]
        correct_idx = "abcd".find(correct) if correct in "abcd" else -1
        rows.append(
            {
                "context": context,
                "query": query,
                "options": answers,
                "correct_option": correct_idx,
            }
        )
    return rows


def _process_logiqa_answer(answer: str) -> str:
    text = answer.strip()
    if any(text.startswith(prefix) for prefix in ("A. ", "B. ", "C. ", "D. ")):
        return text[3:].strip()
    return text


def _process_logiqa_sentence(text: str) -> str:
    text = text.replace("\\'", "'").replace("\n", " ").strip()
    # Keep punctuation formatting close to the original dataset script.
    text = re.sub(r"\s+", " ", text)
    if text and text[-1] not in ".!?":
        text = text + "."
    text = text.replace("?.", "?").replace("!.", "!").replace("..", ".")
    return text


def _as_clean_str(value: Any) -> str:
    return str(value).strip()


def _as_optional_clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _note(message: str) -> None:
    # Explicit note helper keeps warning style consistent across loaders.
    print(f"[data-loader note] {message}")

