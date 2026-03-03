"""Load joined Phase C value-supervision examples from artifact directories.

Why this file exists
--------------------
Phase C C1 writes several JSONL artifact files:
- clean prefixes,
- rollout targets,
- corrupted prefixes.

C2 training and evaluation need those files joined into one strict in-memory
contract so later code does not silently mismatch `prefix_id`, `sample_id`, or
backbone provenance.

What this file contains
-----------------------
- dataclasses describing clean value examples and corruption variants
- manifest/compatibility checks for Phase C artifact directories
- loaders that join prefixes, rollout targets, and corruptions deterministically
  (and, when present, pair-quality artifacts from corruption rollouts)

Interaction with other files
----------------------------
- `src/ours/phase_b/value_targets.py`: defines the clean prefix/rollout records
- `src/ours/phase_b/corruptions.py`: defines corruption artifact records
- `scripts/phase_b_train_value.py`: consumes the joined training/eval examples
- `scripts/phase_b_eval_faithfulness.py`: consumes the joined eval examples

Example
-------
```python
from pathlib import Path
from ours.phase_b.value_data import load_value_supervision_examples

examples, manifest = load_value_supervision_examples(
    Path("assets/artifacts/phase_c_data/strategyqa/my_run__abcdef123456")
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
class ValueSupervisionExample:
    """One clean prefix plus its rollout target and optional primary corruption.

    This is the main C2 training/eval unit. Each example describes one clean
    reasoning state `h_t` together with:
    - empirical success target from rollouts,
    - parseability metadata,
    - one deterministic corruption variant when available.
    """

    prefix_id: str
    sample_id: str
    dataset: str
    split: str
    question: str
    prompt_text: str
    prefix_target_text: str
    current_step_role: str
    current_step_id: str
    prefix_step_index: int
    num_reasoning_steps_seen: int
    num_reasoning_steps_total: int
    target_success_rate: float
    target_q_mean_smoothed: float
    target_q_std_error: float
    target_q_ci_width: float
    target_q_weight: float
    target_parseable_rate: float
    target_k_rollouts: int
    mean_generated_char_count: float
    metadata: dict[str, Any] = field(default_factory=dict)
    primary_corruption_text: str | None = None
    primary_corruption_type: str | None = None
    primary_corruption_step_index: int | None = None
    primary_pair_delta_q: float | None = None
    primary_pair_z_delta: float | None = None
    primary_pair_weight: float | None = None
    primary_pair_pass_gate: bool | None = None

    def validate(self) -> None:
        """Validate field types and numeric ranges.

        Example
        -------
        ```python
        example.validate()
        ```
        """
        for name in (
            "prefix_id",
            "sample_id",
            "dataset",
            "split",
            "question",
            "prompt_text",
            "current_step_role",
            "current_step_id",
        ):
            _validate_non_empty_str(getattr(self, name), name)
        if not isinstance(self.prefix_target_text, str):
            raise TypeError("`prefix_target_text` must be str")
        if not isinstance(self.prefix_step_index, int) or self.prefix_step_index < 0:
            raise ValueError("`prefix_step_index` must be a non-negative int")
        if not isinstance(self.num_reasoning_steps_seen, int) or self.num_reasoning_steps_seen < 0:
            raise ValueError("`num_reasoning_steps_seen` must be a non-negative int")
        if not isinstance(self.num_reasoning_steps_total, int) or self.num_reasoning_steps_total < 0:
            raise ValueError("`num_reasoning_steps_total` must be a non-negative int")
        if self.num_reasoning_steps_seen > self.num_reasoning_steps_total:
            raise ValueError("`num_reasoning_steps_seen` cannot exceed total")
        for name in ("target_success_rate", "target_parseable_rate"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"`{name}` must be a float in [0, 1]")
        for name in ("target_q_mean_smoothed", "target_q_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"`{name}` must be a float in [0, 1]")
        for name in ("target_q_std_error", "target_q_ci_width"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or float(value) < 0.0:
                raise ValueError(f"`{name}` must be a non-negative float")
        if not isinstance(self.target_k_rollouts, int) or self.target_k_rollouts <= 0:
            raise ValueError("`target_k_rollouts` must be a positive int")
        if not isinstance(self.mean_generated_char_count, (int, float)) or float(self.mean_generated_char_count) < 0.0:
            raise ValueError("`mean_generated_char_count` must be non-negative")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")
        if self.primary_corruption_text is not None and not isinstance(self.primary_corruption_text, str):
            raise TypeError("`primary_corruption_text` must be str or None")
        if self.primary_corruption_type is not None and not isinstance(self.primary_corruption_type, str):
            raise TypeError("`primary_corruption_type` must be str or None")
        if self.primary_corruption_step_index is not None:
            if not isinstance(self.primary_corruption_step_index, int) or self.primary_corruption_step_index < 0:
                raise ValueError("`primary_corruption_step_index` must be a non-negative int or None")
        if self.primary_pair_delta_q is not None and not isinstance(
            self.primary_pair_delta_q, (int, float)
        ):
            raise TypeError("`primary_pair_delta_q` must be float or None")
        if self.primary_pair_z_delta is not None and not isinstance(
            self.primary_pair_z_delta, (int, float)
        ):
            raise TypeError("`primary_pair_z_delta` must be float or None")
        if self.primary_pair_weight is not None:
            if not isinstance(self.primary_pair_weight, (int, float)):
                raise TypeError("`primary_pair_weight` must be float or None")
            if not (0.0 <= float(self.primary_pair_weight) <= 1.0):
                raise ValueError("`primary_pair_weight` must be in [0, 1]")
        if self.primary_pair_pass_gate is not None and not isinstance(
            self.primary_pair_pass_gate, bool
        ):
            raise TypeError("`primary_pair_pass_gate` must be bool or None")

    def to_dict(self) -> dict[str, Any]:
        """Return a validated JSON-serializable dictionary."""
        self.validate()
        return asdict(self)

    def clean_input_text(self) -> str:
        """Return the exact clean prefix text presented to the backbone."""
        return f"{self.prompt_text}{self.prefix_target_text}"

    def has_primary_corruption(self) -> bool:
        """Return whether this example carries one deterministic corruption."""
        return self.primary_corruption_text is not None

    def primary_corruption_input_text(self) -> str:
        """Return full prompt text for the stored primary corruption.

        Raises
        ------
        ValueError
            If the example does not carry a primary corruption.
        """
        if self.primary_corruption_text is None:
            raise ValueError(f"Example {self.prefix_id!r} does not have a primary corruption")
        return f"{self.prompt_text}{self.primary_corruption_text}"


@dataclass(slots=True)
class CorruptionVariant:
    """One scored corruption candidate paired with its clean prefix metadata."""

    corruption_id: str
    clean_prefix_id: str
    sample_id: str
    dataset: str
    split: str
    prompt_text: str
    question: str
    clean_prefix_target_text: str
    corrupted_prefix_text: str
    corruption_type: str
    corruption_step_index: int
    current_step_role: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate corruption-variant fields."""
        for name in (
            "corruption_id",
            "clean_prefix_id",
            "sample_id",
            "dataset",
            "split",
            "prompt_text",
            "question",
            "corrupted_prefix_text",
            "corruption_type",
            "current_step_role",
        ):
            _validate_non_empty_str(getattr(self, name), name)
        if not isinstance(self.clean_prefix_target_text, str):
            raise TypeError("`clean_prefix_target_text` must be str")
        if not isinstance(self.corruption_step_index, int) or self.corruption_step_index < 0:
            raise ValueError("`corruption_step_index` must be a non-negative int")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        """Return a validated JSON-serializable dictionary."""
        self.validate()
        return asdict(self)

    def clean_input_text(self) -> str:
        """Return the clean prefix text for this corruption pair."""
        return f"{self.prompt_text}{self.clean_prefix_target_text}"

    def corrupted_input_text(self) -> str:
        """Return the corrupted prefix text for this corruption pair."""
        return f"{self.prompt_text}{self.corrupted_prefix_text}"


def load_phase_c_manifest(run_dir: Path) -> dict[str, Any]:
    """Load and validate one Phase C manifest.

    Example
    -------
    ```python
    manifest = load_phase_c_manifest(run_dir)
    ```
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Phase C manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"Manifest {manifest_path} must contain a JSON object")
    if str(manifest.get("artifact_stage", "")).strip() != "phase_c_c0_c1":
        raise ValueError(
            f"Run directory {run_dir} is not a Phase C C0/C1 artifact dir: "
            f"artifact_stage={manifest.get('artifact_stage')!r}"
        )
    return manifest


def assert_phase_c_compatibility(
    train_manifest: dict[str, Any],
    eval_manifest: dict[str, Any],
) -> None:
    """Fail if train/eval artifact directories are incompatible.

    Current strict checks intentionally block mixed contracts because those bugs
    are expensive later and hard to notice from losses alone.
    """
    checks = [
        ("step_config_signature", train_manifest.get("step_config_signature"), eval_manifest.get("step_config_signature")),
        ("prefix_config_signature", train_manifest.get("prefix_config_signature"), eval_manifest.get("prefix_config_signature")),
    ]
    for name, left, right in checks:
        if left != right:
            raise ValueError(
                f"Phase C train/eval artifacts disagree on {name}: {left!r} vs {right!r}"
            )

    train_rollout = train_manifest.get("rollout_config")
    eval_rollout = eval_manifest.get("rollout_config")
    if not isinstance(train_rollout, dict) or not isinstance(eval_rollout, dict):
        raise ValueError("Both train and eval Phase C artifacts must contain rollout_config")
    for key in ("model_path", "adapter_path", "dtype"):
        if train_rollout.get(key) != eval_rollout.get(key):
            raise ValueError(
                "Phase C train/eval artifacts must be built from the same rollout backbone: "
                f"mismatch on {key}: {train_rollout.get(key)!r} vs {eval_rollout.get(key)!r}"
            )


def load_value_supervision_examples(
    run_dir: Path,
    *,
    max_samples: int | None = None,
    require_corruptions: bool = False,
) -> tuple[list[ValueSupervisionExample], dict[str, Any]]:
    """Load clean prefix examples joined with rollout targets.

    Parameters
    ----------
    run_dir:
        One Phase C artifact directory containing `prefixes.jsonl` and
        `rollout_targets.jsonl`.
    max_samples:
        Optional hard cap after deterministic sorting by `prefix_id`.
    require_corruptions:
        If true, fail when the run directory does not contain any corruptions.

    Returns
    -------
    tuple[list[ValueSupervisionExample], dict[str, Any]]
        Joined training/eval examples and the loaded manifest.
    """
    manifest = load_phase_c_manifest(run_dir)
    prefix_path = run_dir / "prefixes.jsonl"
    rollout_target_path = run_dir / "rollout_targets.jsonl"
    corruption_path = run_dir / "corruptions.jsonl"
    pair_quality_path = run_dir / "pair_quality.jsonl"

    if not prefix_path.exists():
        raise FileNotFoundError(f"Missing prefixes.jsonl: {prefix_path}")
    if not rollout_target_path.exists():
        raise FileNotFoundError(
            f"Missing rollout_targets.jsonl: {rollout_target_path}. "
            "C2 training requires rollout targets."
        )
    if require_corruptions and not corruption_path.exists():
        raise FileNotFoundError(f"Missing corruptions.jsonl: {corruption_path}")

    prefixes = {row["prefix_id"]: row for row in _read_jsonl(prefix_path)}
    rollout_targets = {row["prefix_id"]: row for row in _read_jsonl(rollout_target_path)}
    pair_quality_by_corruption, pair_quality_by_prefix = (
        _build_pair_quality_maps(_read_jsonl(pair_quality_path))
        if pair_quality_path.exists()
        else ({}, {})
    )
    primary_corruptions = (
        _build_primary_corruption_map(
            _read_jsonl(corruption_path),
            pair_quality_by_corruption=pair_quality_by_corruption,
        )
        if corruption_path.exists()
        else {}
    )

    examples: list[ValueSupervisionExample] = []
    for prefix_id in sorted(rollout_targets):
        prefix = prefixes.get(prefix_id)
        target = rollout_targets[prefix_id]
        if prefix is None:
            raise KeyError(f"Missing prefix record for rollout target prefix_id={prefix_id!r}")
        corruption = primary_corruptions.get(prefix_id)
        pair_quality = None
        if corruption is not None:
            pair_quality = pair_quality_by_corruption.get(str(corruption["corruption_id"]))
        if pair_quality is None:
            pair_quality = pair_quality_by_prefix.get(str(prefix_id))

        target_success_rate = float(target["success_rate"])
        target_q_mean_smoothed = float(target.get("q_mean_smoothed", target_success_rate))
        target_q_std_error = float(target.get("q_std_error", 0.0))
        target_q_ci_width = float(target.get("q_ci_width", 0.0))
        target_q_weight = float(target.get("q_weight", 1.0))
        example = ValueSupervisionExample(
            prefix_id=str(prefix_id),
            sample_id=str(prefix["sample_id"]),
            dataset=str(prefix["dataset"]),
            split=str(prefix["split"]),
            question=str(prefix["question"]),
            prompt_text=str(prefix["prompt_text"]),
            prefix_target_text=str(prefix["prefix_target_text"]),
            current_step_role=str(prefix["current_step_role"]),
            current_step_id=str(prefix["current_step_id"]),
            prefix_step_index=int(prefix["prefix_step_index"]),
            num_reasoning_steps_seen=int(prefix["num_reasoning_steps_seen"]),
            num_reasoning_steps_total=int(prefix["num_reasoning_steps_total"]),
            target_success_rate=target_success_rate,
            target_q_mean_smoothed=target_q_mean_smoothed,
            target_q_std_error=target_q_std_error,
            target_q_ci_width=target_q_ci_width,
            target_q_weight=target_q_weight,
            target_parseable_rate=float(target["parseable_rate"]),
            target_k_rollouts=int(target["k_rollouts"]),
            mean_generated_char_count=float(target["mean_generated_char_count"]),
            metadata={
                "prefix_metadata": dict(prefix.get("metadata", {})),
                "target_metadata": dict(target.get("metadata", {})),
                "has_pair_quality": bool(pair_quality is not None),
            },
            primary_corruption_text=(str(corruption["corrupted_prefix_text"]) if corruption is not None else None),
            primary_corruption_type=(str(corruption["corruption_type"]) if corruption is not None else None),
            primary_corruption_step_index=(int(corruption["corruption_step_index"]) if corruption is not None else None),
            primary_pair_delta_q=(
                float(pair_quality["delta_q"]) if pair_quality is not None else None
            ),
            primary_pair_z_delta=(
                float(pair_quality["z_delta"]) if pair_quality is not None else None
            ),
            primary_pair_weight=(
                float(pair_quality["pair_weight"]) if pair_quality is not None else None
            ),
            primary_pair_pass_gate=(
                bool(pair_quality.get("metadata", {}).get("pair_pass_gate"))
                if pair_quality is not None
                else None
            ),
        )
        example.validate()
        examples.append(example)
        if max_samples is not None and len(examples) >= max_samples:
            break

    if not examples:
        raise ValueError(f"No usable value-supervision examples loaded from {run_dir}")
    if require_corruptions and not any(example.has_primary_corruption() for example in examples):
        raise ValueError(
            f"Corruptions were required but none of the loaded prefixes in {run_dir} have one"
        )
    return examples, manifest


def load_corruption_variants(
    run_dir: Path,
    *,
    max_variants: int | None = None,
) -> tuple[list[CorruptionVariant], dict[str, Any]]:
    """Load all corruption variants joined with their clean prefix metadata."""
    manifest = load_phase_c_manifest(run_dir)
    prefix_path = run_dir / "prefixes.jsonl"
    corruption_path = run_dir / "corruptions.jsonl"
    if not prefix_path.exists():
        raise FileNotFoundError(f"Missing prefixes.jsonl: {prefix_path}")
    if not corruption_path.exists():
        raise FileNotFoundError(f"Missing corruptions.jsonl: {corruption_path}")

    prefixes = {row["prefix_id"]: row for row in _read_jsonl(prefix_path)}
    variants: list[CorruptionVariant] = []
    for corruption in sorted(_read_jsonl(corruption_path), key=lambda row: str(row["corruption_id"])):
        clean_prefix = prefixes.get(str(corruption["clean_prefix_id"]))
        if clean_prefix is None:
            raise KeyError(
                f"Missing clean prefix for corruption clean_prefix_id={corruption['clean_prefix_id']!r}"
            )
        variant = CorruptionVariant(
            corruption_id=str(corruption["corruption_id"]),
            clean_prefix_id=str(corruption["clean_prefix_id"]),
            sample_id=str(corruption["sample_id"]),
            dataset=str(corruption["dataset"]),
            split=str(corruption["split"]),
            prompt_text=str(clean_prefix["prompt_text"]),
            question=str(clean_prefix["question"]),
            clean_prefix_target_text=str(clean_prefix["prefix_target_text"]),
            corrupted_prefix_text=str(corruption["corrupted_prefix_text"]),
            corruption_type=str(corruption["corruption_type"]),
            corruption_step_index=int(corruption["corruption_step_index"]),
            current_step_role=str(clean_prefix["current_step_role"]),
            metadata={
                "clean_prefix_metadata": dict(clean_prefix.get("metadata", {})),
                "corruption_metadata": dict(corruption.get("metadata", {})),
            },
        )
        variant.validate()
        variants.append(variant)
        if max_variants is not None and len(variants) >= max_variants:
            break

    return variants, manifest


def _build_primary_corruption_map(
    rows: list[dict[str, Any]],
    *,
    pair_quality_by_corruption: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Pick one deterministic corruption per clean prefix.

    Policy:
    1. If pair-quality artifacts are available, pick the best-quality variant
       per prefix (pass-gate first, then highest pair_weight, then largest
       delta_q/z_delta).
    2. If no pair-quality info exists for that prefix, fall back to the
       lexicographically smallest `corruption_id` for deterministic behavior.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["clean_prefix_id"]), []).append(row)

    pair_quality_by_corruption = pair_quality_by_corruption or {}

    resolved: dict[str, dict[str, Any]] = {}
    for prefix_id, variants in grouped.items():
        if pair_quality_by_corruption:
            ranked = sorted(
                variants,
                key=lambda item: _primary_corruption_sort_key(
                    item=item,
                    pair_quality_by_corruption=pair_quality_by_corruption,
                ),
            )
            resolved[prefix_id] = ranked[0]
        else:
            resolved[prefix_id] = sorted(
                variants, key=lambda item: str(item["corruption_id"])
            )[0]
    return resolved


def _primary_corruption_sort_key(
    *,
    item: dict[str, Any],
    pair_quality_by_corruption: dict[str, dict[str, Any]],
) -> tuple[int, float, float, float, str]:
    """Build a deterministic quality-aware sort key for primary corruption selection.

    Sorting order (ascending key, so negated for larger-is-better terms):
    1. prefer pair_pass_gate=True
    2. prefer larger pair_weight
    3. prefer larger delta_q
    4. prefer larger z_delta
    5. deterministic fallback on corruption_id
    """
    corruption_id = str(item.get("corruption_id", ""))
    pair = pair_quality_by_corruption.get(corruption_id)
    if pair is None:
        # Unknown quality -> lowest priority, keep deterministic by id.
        return (1, 0.0, 0.0, 0.0, corruption_id)
    pair_pass_gate = bool(pair.get("metadata", {}).get("pair_pass_gate", False))
    pair_weight = float(pair.get("pair_weight", 0.0))
    delta_q = float(pair.get("delta_q", 0.0))
    z_delta = float(pair.get("z_delta", 0.0))
    return (
        0 if pair_pass_gate else 1,
        -pair_weight,
        -delta_q,
        -z_delta,
        corruption_id,
    )


def _build_pair_quality_maps(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Index pair-quality rows by corruption and by clean prefix.

    The clean-prefix map keeps a deterministic "best" record for fallback use
    when the primary corruption chosen for C2 training does not have a direct
    pair-quality record.
    """
    by_corruption: dict[str, dict[str, Any]] = {}
    grouped_by_prefix: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        corruption_id = str(row.get("corruption_id", ""))
        clean_prefix_id = str(row.get("clean_prefix_id", ""))
        if corruption_id:
            by_corruption[corruption_id] = row
        if clean_prefix_id:
            grouped_by_prefix.setdefault(clean_prefix_id, []).append(row)

    by_prefix: dict[str, dict[str, Any]] = {}
    for prefix_id, variants in grouped_by_prefix.items():
        ranked = sorted(
            variants,
            key=lambda item: (
                0 if bool(item.get("metadata", {}).get("pair_pass_gate")) else 1,
                -float(item.get("pair_weight", 0.0)),
                -float(item.get("delta_q", 0.0)),
                -float(item.get("z_delta", 0.0)),
                str(item.get("pair_id", "")),
            ),
        )
        by_prefix[prefix_id] = ranked[0]
    return by_corruption, by_prefix


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one UTF-8 JSONL file into a list of dictionaries."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
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
