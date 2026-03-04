"""Phase D utilities for external pair-data bootstrapping."""

from .external_pairs import (
    ExternalPairRecord,
    load_external_pair_jsonl,
    summarize_external_pairs,
)

__all__ = [
    "ExternalPairRecord",
    "load_external_pair_jsonl",
    "summarize_external_pairs",
]

