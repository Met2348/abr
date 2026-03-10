"""Canonical registry for Phase E sources and benchmarks.

Why this file exists
--------------------
Phase E needs one explicit contract describing:

1. which datasets are training sources,
2. which datasets are evaluation benchmarks,
3. which local paths are treated as the default snapshots.

Without this file, shell suites and Python scripts tend to drift apart and
quietly encode different assumptions about what "Math-Shepherd" or
"ProcessBench" means in the repository.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PhaseEPairSourceSpec:
    """Describe one pair/process supervision source used in Phase E."""

    source_id: str
    source_type: str
    description: str
    default_path: str
    default_split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""
        return asdict(self)

    def default_path_obj(self) -> Path:
        """Return the source default path as `Path`."""
        return Path(self.default_path)


@dataclass(frozen=True, slots=True)
class PhaseEEvalBenchmarkSpec:
    """Describe one benchmark used to judge Phase E learnability."""

    benchmark_id: str
    benchmark_type: str
    description: str
    default_path: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""
        return asdict(self)

    def default_path_obj(self) -> Path:
        """Return the benchmark default path as `Path`."""
        return Path(self.default_path)


def get_phase_e_pair_source_registry() -> dict[str, PhaseEPairSourceSpec]:
    """Return the canonical Phase E training-source registry."""
    return {
        "math_shepherd": PhaseEPairSourceSpec(
            source_id="math_shepherd",
            source_type="math_shepherd",
            description="Step-labeled Math-Shepherd conversion pairs.",
            default_path="assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl",
        ),
        "prm800k": PhaseEPairSourceSpec(
            source_id="prm800k",
            source_type="prm800k",
            description="Canonical PRM800K public mirror or local snapshot.",
            default_path="assets/external_datasets/openai_prm800k",
        ),
        "r_prm_train": PhaseEPairSourceSpec(
            source_id="r_prm_train",
            source_type="r_prm",
            description="R-PRM direct chosen/rejected pairs (train split).",
            default_path="assets/external_datasets/kevinpro_r_prm",
            default_split="train",
        ),
        "r_prm_validation": PhaseEPairSourceSpec(
            source_id="r_prm_validation",
            source_type="r_prm",
            description="R-PRM direct chosen/rejected pairs (validation split).",
            default_path="assets/external_datasets/kevinpro_r_prm",
            default_split="validation",
        ),
        "prmbench_preview": PhaseEPairSourceSpec(
            source_id="prmbench_preview",
            source_type="prmbench_preview",
            description="PRMBench preview converted process pairs.",
            default_path="assets/external_datasets/hitsmy_prmbench_preview/prmbench_preview.jsonl",
        ),
    }


def get_phase_e_pair_bundle_registry() -> dict[str, tuple[str, ...]]:
    """Return named source bundles used by Phase E experiment suites."""
    return {
        "math_shepherd": ("math_shepherd",),
        "prm800k": ("prm800k",),
        "r_prm_train": ("r_prm_train",),
        "prmbench_preview": ("prmbench_preview",),
        "r_prm_prmbench_preview": ("r_prm_train", "prmbench_preview"),
        "math_shepherd_prm800k": ("math_shepherd", "prm800k"),
        # Stage A/B/C/D multi-source math bundles.
        # 这些 bundle 对应 Phase E 多源数学混训矩阵，目的是比较单源锚点、双源混训、
        # 三源主混训，以及弱源(PRM800K)低权重消融。
        "math_shepherd_r_prm": ("math_shepherd", "r_prm_train"),
        "math_shepherd_prmbench_preview": ("math_shepherd", "prmbench_preview"),
        "math_shepherd_r_prm_prmbench_preview": (
            "math_shepherd",
            "r_prm_train",
            "prmbench_preview",
        ),
        "math_shepherd_r_prm_prmbench_preview_prm800k": (
            "math_shepherd",
            "r_prm_train",
            "prmbench_preview",
            "prm800k",
        ),
    }


def get_phase_e_eval_benchmark_registry() -> dict[str, PhaseEEvalBenchmarkSpec]:
    """Return the canonical Phase E benchmark registry."""
    return {
        "processbench_gsm8k": PhaseEEvalBenchmarkSpec(
            benchmark_id="processbench_gsm8k",
            benchmark_type="processbench",
            description="ProcessBench GSM8K split.",
            default_path="assets/external_datasets/qwen_processbench/gsm8k.json",
        ),
        "processbench_math": PhaseEEvalBenchmarkSpec(
            benchmark_id="processbench_math",
            benchmark_type="processbench",
            description="ProcessBench MATH split.",
            default_path="assets/external_datasets/qwen_processbench/math.json",
        ),
        "prmbench_preview": PhaseEEvalBenchmarkSpec(
            benchmark_id="prmbench_preview",
            benchmark_type="prmbench_preview",
            description="PRMBench preview pair benchmark.",
            default_path="assets/external_datasets/hitsmy_prmbench_preview/prmbench_preview.jsonl",
        ),
    }


def resolve_phase_e_pair_bundle(bundle_id: str) -> list[PhaseEPairSourceSpec]:
    """Resolve one bundle id into concrete pair-source specs."""
    bundles = get_phase_e_pair_bundle_registry()
    registry = get_phase_e_pair_source_registry()
    normalized = str(bundle_id).strip()
    if normalized not in bundles:
        raise KeyError(
            f"Unknown Phase E pair bundle: {bundle_id!r}. "
            f"Expected one of {sorted(bundles)}"
        )
    return [registry[source_id] for source_id in bundles[normalized]]


def resolve_phase_e_benchmark_specs(benchmark_ids: list[str]) -> list[PhaseEEvalBenchmarkSpec]:
    """Resolve benchmark ids into canonical evaluation specs."""
    registry = get_phase_e_eval_benchmark_registry()
    resolved: list[PhaseEEvalBenchmarkSpec] = []
    for benchmark_id in benchmark_ids:
        normalized = str(benchmark_id).strip()
        if normalized == "":
            continue
        if normalized not in registry:
            raise KeyError(
                f"Unknown Phase E benchmark id: {benchmark_id!r}. "
                f"Expected one of {sorted(registry)}"
            )
        resolved.append(registry[normalized])
    if not resolved:
        raise ValueError("At least one benchmark id must be provided")
    return resolved


def render_phase_e_contract_markdown() -> str:
    """Render the Phase E benchmark contract as compact Markdown."""
    source_registry = get_phase_e_pair_source_registry()
    benchmark_registry = get_phase_e_eval_benchmark_registry()
    bundle_registry = get_phase_e_pair_bundle_registry()

    lines = [
        "# Phase E Contract",
        "",
        "## Pair Sources",
        "",
        "| source_id | source_type | default_path | default_split |",
        "|---|---|---|---|",
    ]
    for spec in source_registry.values():
        lines.append(
            f"| {spec.source_id} | {spec.source_type} | `{spec.default_path}` | "
            f"{spec.default_split or ''} |"
        )
    lines.extend(
        [
            "",
            "## Pair Bundles",
            "",
            "| bundle_id | members |",
            "|---|---|",
        ]
    )
    for bundle_id, members in bundle_registry.items():
        lines.append(f"| {bundle_id} | `{', '.join(members)}` |")
    lines.extend(
        [
            "",
            "## Eval Benchmarks",
            "",
            "| benchmark_id | benchmark_type | default_path |",
            "|---|---|---|",
        ]
    )
    for spec in benchmark_registry.values():
        lines.append(
            f"| {spec.benchmark_id} | {spec.benchmark_type} | `{spec.default_path}` |"
        )
    lines.append("")
    return "\n".join(lines)
