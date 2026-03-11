"""Unit tests for scripts/phase_e_curate_processbench_transfer_pairs.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ours.phase_d.external_pairs import ExternalPairRecord


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_e_curate_processbench_transfer_pairs.py"
    spec = importlib.util.spec_from_file_location(
        "phase_e_curate_processbench_transfer_pairs",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_row(pair_id: str) -> ExternalPairRecord:
    return ExternalPairRecord(
        pair_id=pair_id,
        source_tag="math_shepherd",
        domain_tag="general_math",
        prompt_text="Q\n\n",
        chosen_text=f"good-{pair_id}",
        rejected_text=f"bad-{pair_id}",
        pair_confidence=0.8,
        metadata={},
    )


def test_allocate_component_quotas_sums_to_target() -> None:
    module = _load_module()
    components = tuple(module.PROFILE_REGISTRY["ms_align_v1"]["components"])

    quotas = module._allocate_component_quotas(total_pairs=17, components=components)

    assert sum(quotas.values()) == 17
    assert set(quotas) == {component.component_id for component in components}


def test_stable_select_rows_is_deterministic() -> None:
    module = _load_module()
    rows = [_make_row(f"p{i}") for i in range(10)]

    left = module._stable_select_rows(rows=rows, quota=4, namespace="demo")
    right = module._stable_select_rows(rows=rows, quota=4, namespace="demo")

    assert [row.pair_id for row in left] == [row.pair_id for row in right]


def test_tag_component_rows_adds_semantic_weight_and_mix_label() -> None:
    module = _load_module()
    component = module.PROFILE_REGISTRY["ms_align_v1"]["components"][0]
    rows = [_make_row("p0")]

    tagged = module._tag_component_rows(
        rows=rows,
        component=component,
        profile_name="ms_align_v1",
    )

    assert len(tagged) == 1
    metadata = tagged[0].metadata
    assert metadata["curation_profile"] == "ms_align_v1"
    assert metadata["curation_component"] == component.component_id
    assert metadata["artifact_mix_source_label"] == component.component_id
    assert metadata["semantic_weight"] == component.semantic_weight
