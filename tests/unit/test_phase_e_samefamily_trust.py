"""Unit tests for Phase E same-family trust helpers."""

from __future__ import annotations

from ours.phase_d.external_pairs import ExternalPairRecord
from ours.phase_e.samefamily_trust import build_prompt_pools_from_pairs, build_unique_candidate_rows


def test_build_prompt_pools_from_pairs_keeps_sources_separate_for_same_prompt_text() -> None:
    pairs = [
        ExternalPairRecord(
            pair_id="ms_pair",
            source_tag="math_shepherd",
            domain_tag="general_math",
            prompt_text="What is 1+1?\n\n",
            chosen_text="Step 1: 2",
            rejected_text="Step 1: 3",
            pair_confidence=0.9,
        ),
        ExternalPairRecord(
            pair_id="pb_pair",
            source_tag="prmbench_preview",
            domain_tag="general_math",
            prompt_text="What is 1+1?\n\n",
            chosen_text="Prefix A",
            rejected_text="Prefix B",
            pair_confidence=0.8,
        ),
    ]

    pools = build_prompt_pools_from_pairs(pairs)

    assert len(pools) == 2
    assert sorted(pool.source_tag for pool in pools) == ["math_shepherd", "prmbench_preview"]


def test_build_unique_candidate_rows_keeps_candidate_ids_source_scoped() -> None:
    pairs = [
        ExternalPairRecord(
            pair_id="ms_pair",
            source_tag="math_shepherd",
            domain_tag="general_math",
            prompt_text="Shared prompt\n\n",
            chosen_text="Shared candidate",
            rejected_text="Math bad",
            pair_confidence=0.9,
        ),
        ExternalPairRecord(
            pair_id="pb_pair",
            source_tag="prmbench_preview",
            domain_tag="general_math",
            prompt_text="Shared prompt\n\n",
            chosen_text="Shared candidate",
            rejected_text="PRM bad",
            pair_confidence=0.8,
        ),
    ]

    nodes, pools = build_unique_candidate_rows(pairs)

    assert len(pools) == 2
    shared_nodes = [node for node in nodes if node.candidate_text == "Shared candidate"]
    assert len(shared_nodes) == 2
    assert len({node.candidate_id for node in shared_nodes}) == 2
