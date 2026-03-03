"""Build deterministic corrupted-prefix artifacts for Phase C faithfulness tests.

Why this file exists
--------------------
The first unique-method milestone after Phase B is not just answer accuracy. We
also need to test whether a value head can distinguish:
- a clean reasoning prefix,
- from a minimally corrupted one.

This module generates those corrupted prefixes in a strict, traceable way. The
goal is not to create every possible adversarial attack. The goal is to create a
small, deterministic, auditable corruption set that is good enough for early
calibration and localization experiments.

Interaction with other files
----------------------------
- `src/ours/phase_b/value_targets.py`: defines the clean `PrefixArtifact`
- `scripts/phase_b_prepare_value_data.py`: writes corruption artifacts to disk
- later `phase_b_eval_faithfulness.py`: will consume these artifacts for AUC and
  localization metrics
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .value_targets import PrefixArtifact


@dataclass(slots=True)
class CorruptionBuildConfig:
    """Configuration controlling how corrupted prefixes are generated."""

    max_corruptions_per_prefix: int = 1
    enable_binary_flip: bool = True
    enable_operator_flip: bool = True
    enable_numeric_perturb: bool = True
    enable_step_drop_fallback: bool = True

    def validate(self) -> None:
        """Validate configuration values before artifact construction."""
        if not isinstance(self.max_corruptions_per_prefix, int) or self.max_corruptions_per_prefix < 1:
            raise ValueError("`max_corruptions_per_prefix` must be an int >= 1")
        for name in (
            "enable_binary_flip",
            "enable_operator_flip",
            "enable_numeric_perturb",
            "enable_step_drop_fallback",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be bool")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config payload."""
        self.validate()
        return asdict(self)

    def stable_signature(self) -> str:
        """Return a short stable signature used in manifests/IDs."""
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(slots=True)
class CorruptionArtifact:
    """One minimally corrupted variant of one clean prefix."""

    corruption_id: str
    clean_prefix_id: str
    sample_id: str
    dataset: str
    split: str
    corruption_type: str
    corrupted_prefix_text: str
    original_step_text: str
    corrupted_step_text: str
    corruption_step_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate corruption artifact fields before persistence."""
        _validate_non_empty_str(self.corruption_id, "corruption_id")
        _validate_non_empty_str(self.clean_prefix_id, "clean_prefix_id")
        _validate_non_empty_str(self.sample_id, "sample_id")
        _validate_non_empty_str(self.dataset, "dataset")
        _validate_non_empty_str(self.split, "split")
        _validate_non_empty_str(self.corruption_type, "corruption_type")
        _validate_non_empty_str(self.corrupted_prefix_text, "corrupted_prefix_text")
        _validate_non_empty_str(self.original_step_text, "original_step_text")
        if not isinstance(self.corrupted_step_text, str):
            raise TypeError("`corrupted_step_text` must be str")
        if not isinstance(self.corruption_step_index, int) or self.corruption_step_index < 0:
            raise ValueError("`corruption_step_index` must be a non-negative int")
        if not isinstance(self.metadata, dict):
            raise TypeError("`metadata` must be dict[str, Any]")

    def to_dict(self) -> dict[str, Any]:
        """Convert the record into a validated plain dictionary."""
        self.validate()
        return asdict(self)


def build_corruptions_for_prefixes(
    prefixes: list[PrefixArtifact],
    *,
    config: CorruptionBuildConfig | None = None,
) -> list[CorruptionArtifact]:
    """Build deterministic corruption artifacts for a list of clean prefixes.

    中文要点
    --------
    - 输入是 clean prefix 列表，输出是可追踪的 corruption 变体。
    - 目标是构造“最小但可复现”的扰动集，用于 clean/corrupt 对比评估。
    """
    cfg = config or CorruptionBuildConfig()
    cfg.validate()

    artifacts: list[CorruptionArtifact] = []
    seen_ids: set[str] = set()
    for prefix in prefixes:
        prefix.validate()
        artifacts.extend(_build_corruptions_for_prefix(prefix=prefix, config=cfg))

    for idx, artifact in enumerate(artifacts):
        artifact.validate()
        if artifact.corruption_id in seen_ids:
            raise ValueError(
                f"Duplicate corruption_id detected at index={idx}: "
                f"{artifact.corruption_id!r}"
            )
        seen_ids.add(artifact.corruption_id)
    return artifacts


def summarize_corruptions(artifacts: list[CorruptionArtifact]) -> dict[str, Any]:
    """Return compact summary stats for a corruption artifact list."""
    type_counts: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}
    for artifact in artifacts:
        artifact.validate()
        type_counts[artifact.corruption_type] = (
            type_counts.get(artifact.corruption_type, 0) + 1
        )
        dataset_counts[artifact.dataset] = dataset_counts.get(artifact.dataset, 0) + 1
    return {
        "num_corruptions": len(artifacts),
        "type_counts": type_counts,
        "dataset_counts": dataset_counts,
    }


def _build_corruptions_for_prefix(
    *,
    prefix: PrefixArtifact,
    config: CorruptionBuildConfig,
) -> list[CorruptionArtifact]:
    """Build up to `max_corruptions_per_prefix` variants for one prefix.

    中文要点
    --------
    - 仅对当前前缀最后一个推理步施加局部扰动。
    - 优先保持上下文不变，便于把分数差异归因到局部步骤变化。
    """
    if prefix.num_reasoning_steps_seen <= 0:
        return []

    lines = [line for line in prefix.prefix_target_text.splitlines() if line.strip()]
    if not lines:
        return []

    original_last = lines[-1].strip()
    candidates: list[tuple[str, str, str]] = []

    # 当前只改“最后一个推理步”，保持其余上下文不变，
    # 这样 clean/corrupt 的对比更聚焦于局部推理扰动。
    if config.enable_binary_flip:
        flipped = _flip_binary_token(original_last)
        if flipped is not None:
            candidates.append(("binary_flip", flipped, original_last))

    if config.enable_operator_flip:
        flipped = _flip_operator(original_last)
        if flipped is not None:
            candidates.append(("operator_flip", flipped, original_last))

    if config.enable_numeric_perturb:
        perturbed = _perturb_first_number(original_last)
        if perturbed is not None:
            candidates.append(("numeric_perturb", perturbed, original_last))

    if config.enable_step_drop_fallback and len(lines) > 1:
        candidates.append(("step_drop", "", original_last))

    artifacts: list[CorruptionArtifact] = []
    seen_payloads: set[tuple[str, str]] = set()
    for corruption_type, corrupted_last, before_text in candidates:
        # 每个 clean prefix 最多保留 max_corruptions_per_prefix 个变体。
        if len(artifacts) >= config.max_corruptions_per_prefix:
            break

        if corruption_type == "step_drop":
            corrupted_lines = lines[:-1]
            corrupted_text = "\n".join(corrupted_lines).strip()
        else:
            corrupted_lines = list(lines)
            corrupted_lines[-1] = corrupted_last
            corrupted_text = "\n".join(corrupted_lines).strip()

        if corrupted_text == "" or corrupted_text == prefix.prefix_target_text.strip():
            continue

        dedupe_key = (corruption_type, corrupted_text)
        if dedupe_key in seen_payloads:
            continue
        seen_payloads.add(dedupe_key)

        artifact = CorruptionArtifact(
            corruption_id=_stable_hash(
                "phase_c_corruption",
                prefix.prefix_id,
                corruption_type,
                corrupted_text,
                config.stable_signature(),
            )[:24],
            clean_prefix_id=prefix.prefix_id,
            sample_id=prefix.sample_id,
            dataset=prefix.dataset,
            split=prefix.split,
            corruption_type=corruption_type,
            corrupted_prefix_text=corrupted_text,
            original_step_text=before_text,
            corrupted_step_text=corrupted_last,
            corruption_step_index=int(prefix.prefix_step_index),
            metadata={
                "num_reasoning_steps_seen": int(prefix.num_reasoning_steps_seen),
                "num_reasoning_steps_total": int(prefix.num_reasoning_steps_total),
                "current_step_id": prefix.current_step_id,
                "current_step_role": prefix.current_step_role,
            },
        )
        artifact.validate()
        artifacts.append(artifact)
    return artifacts


def _flip_binary_token(text: str) -> str | None:
    """Flip one yes/no style token in a line if present."""
    replacements = {
        "yes": "no",
        "no": "yes",
        "true": "false",
        "false": "true",
    }
    for src, dst in replacements.items():
        pattern = re.compile(rf"\b{re.escape(src)}\b", flags=re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return text[: match.start()] + _match_case(match.group(0), dst) + text[match.end() :]
    return None


def _flip_operator(text: str) -> str | None:
    """Flip the first arithmetic operator between digits if present."""
    match = re.search(r"(?<=\d)\s*([+\-*/])\s*(?=\d)", text)
    if not match:
        return None
    original = match.group(1)
    replacement_map = {
        "+": "-",
        "-": "+",
        "*": "/",
        "/": "*",
    }
    replacement = replacement_map.get(original)
    if replacement is None:
        return None
    return text[: match.start(1)] + replacement + text[match.end(1) :]


def _perturb_first_number(text: str) -> str | None:
    """Perturb the first numeric literal by +1 or -1, preserving simple style."""
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    raw = match.group(0)
    if "." in raw:
        value = float(raw)
        replacement = f"{value + 1.0:.1f}"
    else:
        value = int(raw)
        replacement = str(value + 1)
    if replacement == raw:
        return None
    return text[: match.start()] + replacement + text[match.end() :]


def _match_case(source: str, replacement: str) -> str:
    """Apply a simple case pattern from the matched token to its replacement."""
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def _stable_hash(*parts: str) -> str:
    """Return a deterministic SHA256 hex digest from string parts."""
    payload = "||".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_non_empty_str(value: Any, field_name: str) -> None:
    """Validate that a field is a non-empty string after trimming."""
    if not isinstance(value, str):
        raise TypeError(f"`{field_name}` must be str, got {type(value)!r}")
    if value.strip() == "":
        raise ValueError(f"`{field_name}` must be a non-empty string")
