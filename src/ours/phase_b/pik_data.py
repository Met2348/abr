"""Load question-level P(IK) supervision artifacts for Phase C.

Why this file exists
--------------------
The original Phase C path is prefix-level and corruption-oriented. The P(IK)
path is intentionally simpler and question-level:
- one question prompt,
- many sampled answers,
- one empirical success-rate target.

This module defines strict contracts for those artifacts and provides loaders that
fail loudly when files or manifest provenance do not match.

Interaction with other files
----------------------------
- `scripts/phase_c_prepare_pik_data.py`: writes question-level rollout artifacts.
- `scripts/phase_c_train_pik.py`: loads train/eval P(IK) examples.
- `scripts/phase_c_eval_pik.py`: standalone re-evaluation for the trained head.

Example
-------
```python
from pathlib import Path
from ours.phase_b.pik_data import load_pik_supervision_examples

examples, manifest = load_pik_supervision_examples(
    Path("assets/artifacts/phase_c_pik_data/strategyqa/my_run__abcdef123456")
)
print(len(examples), manifest["run_name"])
```
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PIKQuestionRecord:
    """One question-level input row used for rollout generation.

    This record mirrors one prepared Phase-B row but strips away token-level
    supervision details and keeps only fields required for question-level
    confidence estimation.
    """

    sample_id: str
    dataset: str
    split: str
    question: str
    prompt_text: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate one question record before persistence."""
        for name in ("sample_id", "dataset", "split", "question", "prompt_text", "answer"):
            _validate_non_empty_str(getattr(self, name), name)
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        """Return a validated JSON-serializable dictionary."""
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class PIKRolloutPredictionRecord:
    """One sampled answer used to estimate question-level P(IK)."""

    sample_id: str
    dataset: str
    split: str
    question: str
    prompt_text: str
    answer: str
    rollout_index: int
    raw_continuation: str
    full_prediction: str
    extracted_prediction: str
    extraction_method: str
    is_correct: bool
    parse_error: bool
    generated_char_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate one rollout prediction record."""
        for name in ("sample_id", "dataset", "split", "question", "prompt_text", "answer"):
            _validate_non_empty_str(getattr(self, name), name)
        if not isinstance(self.rollout_index, int) or self.rollout_index < 0:
            raise ValueError("`rollout_index` must be a non-negative int")
        _validate_str(self.raw_continuation, "raw_continuation")
        _validate_str(self.full_prediction, "full_prediction")
        _validate_str(self.extracted_prediction, "extracted_prediction")
        _validate_non_empty_str(self.extraction_method, "extraction_method")
        if not isinstance(self.is_correct, bool):
            raise TypeError("`is_correct` must be bool")
        if not isinstance(self.parse_error, bool):
            raise TypeError("`parse_error` must be bool")
        if not isinstance(self.generated_char_count, int) or self.generated_char_count < 0:
            raise ValueError("`generated_char_count` must be a non-negative int")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        """Return a validated JSON-serializable dictionary."""
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class PIKTargetRecord:
    """Empirical question-level P(IK) target from K sampled answers."""

    sample_id: str
    dataset: str
    split: str
    question: str
    prompt_text: str
    answer: str
    k_rollouts: int
    n_correct: int
    n_parse_error: int
    success_rate: float
    parseable_rate: float
    mean_generated_char_count: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate one aggregated target record."""
        for name in ("sample_id", "dataset", "split", "question", "prompt_text", "answer"):
            _validate_non_empty_str(getattr(self, name), name)
        if not isinstance(self.k_rollouts, int) or self.k_rollouts <= 0:
            raise ValueError("`k_rollouts` must be a positive int")
        if not isinstance(self.n_correct, int) or self.n_correct < 0:
            raise ValueError("`n_correct` must be a non-negative int")
        if not isinstance(self.n_parse_error, int) or self.n_parse_error < 0:
            raise ValueError("`n_parse_error` must be a non-negative int")
        if self.n_correct > self.k_rollouts:
            raise ValueError("`n_correct` cannot exceed `k_rollouts`")
        if self.n_parse_error > self.k_rollouts:
            raise ValueError("`n_parse_error` cannot exceed `k_rollouts`")
        for name in ("success_rate", "parseable_rate"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"`{name}` must be a float in [0, 1]")
        if not isinstance(self.mean_generated_char_count, (int, float)) or float(self.mean_generated_char_count) < 0.0:
            raise ValueError("`mean_generated_char_count` must be non-negative")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        """Return a validated JSON-serializable dictionary."""
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class PIKSupervisionExample:
    """One question-level training/eval example for P(IK) head fitting."""

    sample_id: str
    dataset: str
    split: str
    question: str
    prompt_text: str
    answer: str
    target_success_rate: float
    target_parseable_rate: float
    target_k_rollouts: int
    mean_generated_char_count: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate one supervision example."""
        for name in ("sample_id", "dataset", "split", "question", "prompt_text", "answer"):
            _validate_non_empty_str(getattr(self, name), name)
        for name in ("target_success_rate", "target_parseable_rate"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"`{name}` must be a float in [0, 1]")
        if not isinstance(self.target_k_rollouts, int) or self.target_k_rollouts <= 0:
            raise ValueError("`target_k_rollouts` must be a positive int")
        if not isinstance(self.mean_generated_char_count, (int, float)) or float(self.mean_generated_char_count) < 0.0:
            raise ValueError("`mean_generated_char_count` must be non-negative")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        """Return a validated JSON-serializable dictionary."""
        self.validate()
        return asdict(self)

    def model_input_text(self) -> str:
        """Return the exact text used as encoder input for question-level P(IK)."""
        return self.prompt_text


def load_phase_c_pik_manifest(run_dir: Path) -> dict[str, Any]:
    """Load and validate one Phase C P(IK) manifest."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Phase C P(IK) manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Manifest {manifest_path} must contain a JSON object")
    if str(payload.get("artifact_stage", "")).strip() != "phase_c_pik_c1":
        raise ValueError(
            f"Run directory {run_dir} is not a Phase C P(IK) C1 artifact dir: "
            f"artifact_stage={payload.get('artifact_stage')!r}"
        )
    return payload


def assert_phase_c_pik_compatibility(
    train_manifest: dict[str, Any],
    eval_manifest: dict[str, Any],
) -> None:
    """Fail when train/eval P(IK) artifacts are not provenance-compatible."""
    train_rollout = train_manifest.get("rollout_config")
    eval_rollout = eval_manifest.get("rollout_config")
    if not isinstance(train_rollout, dict) or not isinstance(eval_rollout, dict):
        raise ValueError("Both train/eval P(IK) artifacts must contain rollout_config")
    for key in ("model_path", "adapter_path", "dtype"):
        if train_rollout.get(key) != eval_rollout.get(key):
            raise ValueError(
                "Phase C P(IK) train/eval artifacts must share rollout backbone provenance: "
                f"mismatch on {key}: {train_rollout.get(key)!r} vs {eval_rollout.get(key)!r}"
            )


def load_pik_supervision_examples(
    run_dir: Path,
    *,
    max_samples: int | None = None,
) -> tuple[list[PIKSupervisionExample], dict[str, Any]]:
    """Load question-level P(IK) examples joined from `pik_targets.jsonl`."""
    manifest = load_phase_c_pik_manifest(run_dir)
    target_path = run_dir / "pik_targets.jsonl"
    if not target_path.exists():
        raise FileNotFoundError(
            f"Missing pik_targets.jsonl: {target_path}. P(IK) training requires rollout targets."
        )
    # question-level 路径只依赖 pik_targets，不走 prefix/corruption join。
    rows = _read_jsonl(target_path)
    examples: list[PIKSupervisionExample] = []
    for row in sorted(rows, key=lambda item: str(item["sample_id"])):
        example = PIKSupervisionExample(
            sample_id=str(row["sample_id"]),
            dataset=str(row["dataset"]),
            split=str(row["split"]),
            question=str(row["question"]),
            prompt_text=str(row["prompt_text"]),
            answer=str(row["answer"]),
            target_success_rate=float(row["success_rate"]),
            target_parseable_rate=float(row["parseable_rate"]),
            target_k_rollouts=int(row["k_rollouts"]),
            mean_generated_char_count=float(row["mean_generated_char_count"]),
            metadata=dict(row.get("metadata", {})),
        )
        example.validate()
        examples.append(example)
        if max_samples is not None and len(examples) >= max_samples:
            break
    if not examples:
        raise ValueError(f"No usable P(IK) supervision examples loaded from {run_dir}")
    return examples, manifest


def summarize_pik_targets(targets: list[PIKTargetRecord]) -> dict[str, Any]:
    """Return compact aggregate stats for question-level P(IK) targets."""
    if not targets:
        return {
            "num_questions": 0,
            "mean_success_rate": 0.0,
            "mean_parseable_rate": 0.0,
            "mean_generated_char_count": 0.0,
        }
    total_success = 0.0
    total_parseable = 0.0
    total_chars = 0.0
    for target in targets:
        target.validate()
        total_success += float(target.success_rate)
        total_parseable += float(target.parseable_rate)
        total_chars += float(target.mean_generated_char_count)
    n = float(len(targets))
    return {
        "num_questions": len(targets),
        "mean_success_rate": total_success / n,
        "mean_parseable_rate": total_parseable / n,
        "mean_generated_char_count": total_chars / n,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one UTF-8 JSONL file into a list of dictionaries."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected JSON object in {path} at line {line_no}")
            rows.append(payload)
    return rows


def _validate_non_empty_str(value: Any, field_name: str) -> None:
    """Validate that one field is a non-empty string."""
    if not isinstance(value, str):
        raise TypeError(f"`{field_name}` must be str, got {type(value)!r}")
    if value.strip() == "":
        raise ValueError(f"`{field_name}` must be a non-empty string")


def _validate_str(value: Any, field_name: str) -> None:
    """Validate that one field is a string, allowing empty content."""
    if not isinstance(value, str):
        raise TypeError(f"`{field_name}` must be str, got {type(value)!r}")
