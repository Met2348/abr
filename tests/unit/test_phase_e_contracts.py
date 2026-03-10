"""Unit tests for Phase E benchmark/source contracts."""

from __future__ import annotations

from ours.phase_e.contracts import (
    get_phase_e_eval_benchmark_registry,
    get_phase_e_pair_bundle_registry,
    get_phase_e_pair_source_registry,
    render_phase_e_contract_markdown,
    resolve_phase_e_benchmark_specs,
    resolve_phase_e_pair_bundle,
)


def test_phase_e_pair_bundle_resolution_returns_expected_sources() -> None:
    bundles = get_phase_e_pair_bundle_registry()
    assert "math_shepherd" in bundles
    specs = resolve_phase_e_pair_bundle("math_shepherd_prm800k")
    assert [spec.source_id for spec in specs] == ["math_shepherd", "prm800k"]
    tri_mix = resolve_phase_e_pair_bundle("math_shepherd_r_prm_prmbench_preview")
    assert [spec.source_id for spec in tri_mix] == [
        "math_shepherd",
        "r_prm_train",
        "prmbench_preview",
    ]


def test_phase_e_benchmark_registry_contains_processbench_and_prmbench() -> None:
    registry = get_phase_e_eval_benchmark_registry()
    assert "processbench_gsm8k" in registry
    assert "processbench_math" in registry
    assert "prmbench_preview" in registry
    resolved = resolve_phase_e_benchmark_specs(["processbench_gsm8k", "prmbench_preview"])
    assert [spec.benchmark_type for spec in resolved] == ["processbench", "prmbench_preview"]


def test_phase_e_contract_markdown_mentions_core_benchmarks() -> None:
    registry = get_phase_e_pair_source_registry()
    assert "r_prm_train" in registry
    text = render_phase_e_contract_markdown()
    assert "Math-Shepherd" not in text  # table uses ids/paths, not prose descriptions
    assert "processbench_gsm8k" in text
    assert "r_prm_train" in text
