"""Unit tests for shared feature-cache safety helpers."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from ours.phase_b.feature_cache import (
    FEATURE_CACHE_SCHEMA_VERSION,
    build_cache_key,
    collect_path_signature,
    save_feature_cache,
    try_load_feature_cache,
)


def test_collect_path_signature_tracks_index_referenced_shards(tmp_path: Path) -> None:
    """Model provenance must include actual weight shard files referenced by the index."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    index_payload = {
        "weight_map": {
            "layer0": "model-00001-of-00002.safetensors",
            "layer1": "model-00002-of-00002.safetensors",
        }
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(index_payload),
        encoding="utf-8",
    )
    (model_dir / "model-00001-of-00002.safetensors").write_text("a", encoding="utf-8")
    (model_dir / "model-00002-of-00002.safetensors").write_text("b", encoding="utf-8")

    signature = collect_path_signature(
        model_dir,
        tracked_files=[
            "config.json",
            "model.safetensors.index.json",
        ],
    )

    tracked = signature["tracked"]
    assert "model-00001-of-00002.safetensors" in tracked
    assert "model-00002-of-00002.safetensors" in tracked
    assert tracked["model-00001-of-00002.safetensors"]["exists"] is True


def test_save_feature_cache_rewrites_corrupt_existing_entry(tmp_path: Path) -> None:
    """Broken cache payloads should be replaced instead of causing sticky misses forever."""
    cache_root = tmp_path / "cache"
    cache_key, signature_hash = build_cache_key("unit_cache", {"k": 1})
    cache_dir = cache_root / cache_key[:2] / cache_key
    cache_dir.mkdir(parents=True)
    (cache_dir / "payload.pt").write_text("not a torch payload", encoding="utf-8")
    (cache_dir / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
                "cache_key": cache_key,
                "signature_hash": signature_hash,
            }
        ),
        encoding="utf-8",
    )

    payload = {"features": torch.arange(6, dtype=torch.float32).view(3, 2)}
    save_feature_cache(
        cache_root=cache_root,
        cache_key=cache_key,
        signature_hash=signature_hash,
        payload=payload,
        torch_module=torch,
        producer="tests",
        lock_timeout_sec=1.0,
    )

    loaded, _, _ = try_load_feature_cache(
        cache_root=cache_root,
        cache_key=cache_key,
        expected_signature_hash=signature_hash,
        torch_module=torch,
    )
    assert loaded is not None
    assert torch.equal(loaded["features"], payload["features"])


def test_save_feature_cache_recovers_from_stale_lock(tmp_path: Path) -> None:
    """Stale lock files should not block all future cache writes."""
    cache_root = tmp_path / "cache"
    cache_key, signature_hash = build_cache_key("unit_cache", {"k": 2})
    cache_dir = cache_root / cache_key[:2] / cache_key
    cache_dir.mkdir(parents=True)
    lock_path = cache_dir / ".write.lock"
    lock_path.write_text("pid=dead\n", encoding="utf-8")
    stale_time = time.time() - 120.0
    os.utime(lock_path, (stale_time, stale_time))

    payload = {"features": torch.ones((2, 2), dtype=torch.float32)}
    save_feature_cache(
        cache_root=cache_root,
        cache_key=cache_key,
        signature_hash=signature_hash,
        payload=payload,
        torch_module=torch,
        producer="tests",
        lock_timeout_sec=1.0,
    )

    assert not lock_path.exists()
    loaded, _, _ = try_load_feature_cache(
        cache_root=cache_root,
        cache_key=cache_key,
        expected_signature_hash=signature_hash,
        torch_module=torch,
    )
    assert loaded is not None
    assert torch.equal(loaded["features"], payload["features"])
