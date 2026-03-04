"""Persistent feature-cache helpers for Phase C/D frozen-backbone encoding.

Why this module exists
----------------------
Phase C/D repeatedly encodes the same texts with a frozen backbone across
reruns (train/eval and ablations). This module provides a conservative on-disk
cache with strong cache keys, schema/version checks, and atomic writes.

Safety model
------------
1. Cache key must include full provenance signature (model/tokenizer/data/config).
2. Cache payloads are saved with metadata and validated before load.
3. Writes are lock-protected and file-atomic to avoid partial/corrupt cache files.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

FEATURE_CACHE_SCHEMA_VERSION = "phase_feature_cache_v1"
_VALID_CACHE_MODES = {"off", "read", "write", "read_write"}


def validate_feature_cache_mode(mode: str) -> str:
    """Validate feature-cache mode and return normalized lowercase value."""
    normalized = str(mode).strip().lower()
    if normalized not in _VALID_CACHE_MODES:
        raise ValueError(
            f"Unsupported feature-cache mode: {mode!r}. "
            f"Expected one of {sorted(_VALID_CACHE_MODES)}"
        )
    return normalized


def feature_cache_can_read(mode: str) -> bool:
    """Return whether this mode allows cache reads."""
    normalized = validate_feature_cache_mode(mode)
    return normalized in {"read", "read_write"}


def feature_cache_can_write(mode: str) -> bool:
    """Return whether this mode allows cache writes."""
    normalized = validate_feature_cache_mode(mode)
    return normalized in {"write", "read_write"}


def hash_jsonable(payload: Any) -> str:
    """Hash one JSON-serializable payload with stable canonical formatting."""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_cache_key(namespace: str, signature_payload: dict[str, Any]) -> tuple[str, str]:
    """Build `(cache_key, signature_hash)` from namespace + signature payload."""
    signature_hash = hash_jsonable(signature_payload)
    key = f"{namespace}_{signature_hash[:24]}"
    return key, signature_hash


def hash_text_list(texts: list[str]) -> dict[str, Any]:
    """Return stable digest metadata for an ordered text list."""
    digest = hashlib.sha256()
    total_chars = 0
    for text in texts:
        normalized = str(text)
        data = normalized.encode("utf-8")
        digest.update(len(data).to_bytes(8, byteorder="little", signed=False))
        digest.update(data)
        total_chars += len(normalized)
    return {
        "count": int(len(texts)),
        "total_chars": int(total_chars),
        "sha256": digest.hexdigest(),
    }


def hash_float_list(values: list[float], *, precision: int = 12) -> dict[str, Any]:
    """Return stable digest metadata for an ordered float list."""
    digest = hashlib.sha256()
    fmt = f"{{:.{int(precision)}g}}"
    for value in values:
        token = fmt.format(float(value)).encode("utf-8")
        digest.update(len(token).to_bytes(4, byteorder="little", signed=False))
        digest.update(token)
    return {
        "count": int(len(values)),
        "sha256": digest.hexdigest(),
    }


def hash_int_list(values: list[int]) -> dict[str, Any]:
    """Return stable digest metadata for an ordered int list."""
    digest = hashlib.sha256()
    for value in values:
        token = str(int(value)).encode("utf-8")
        digest.update(len(token).to_bytes(4, byteorder="little", signed=False))
        digest.update(token)
    return {
        "count": int(len(values)),
        "sha256": digest.hexdigest(),
    }


def collect_path_signature(path: str | Path | None, *, tracked_files: list[str] | None = None) -> dict[str, Any]:
    """Collect lightweight path/file stat signature for cache provenance.

    This intentionally avoids hashing giant model shards. We include stable path,
    existence, and selected metadata-file stats to detect most practical drift.
    """
    if path is None:
        return {"path": None, "exists": False, "kind": None, "tracked": {}}

    p = Path(path)
    signature: dict[str, Any] = {
        "path": str(p.resolve()) if p.exists() else str(p),
        "exists": bool(p.exists()),
        "kind": None,
        "tracked": {},
    }
    if not p.exists():
        return signature
    if p.is_file():
        st = p.stat()
        signature["kind"] = "file"
        signature["size"] = int(st.st_size)
        signature["mtime_ns"] = int(st.st_mtime_ns)
        return signature

    signature["kind"] = "dir"
    tracked = tracked_files or []
    tracked_payload: dict[str, Any] = {}
    for rel in tracked:
        file_path = p / rel
        if not file_path.exists() or not file_path.is_file():
            tracked_payload[rel] = {"exists": False}
            continue
        st = file_path.stat()
        tracked_payload[rel] = {
            "exists": True,
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }
    signature["tracked"] = tracked_payload
    return signature


def build_backbone_signature(
    *,
    model_path: str,
    adapter_path: str | None,
    tokenizer_path: str,
    dtype: str,
    max_length: int,
) -> dict[str, Any]:
    """Build conservative provenance signature for frozen-backbone encoding."""
    return {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "model": collect_path_signature(
            model_path,
            tracked_files=[
                "config.json",
                "model.safetensors.index.json",
                "generation_config.json",
            ],
        ),
        "adapter": collect_path_signature(
            adapter_path,
            tracked_files=[
                "adapter_config.json",
                "adapter_model.safetensors",
            ],
        ) if adapter_path is not None else None,
        "tokenizer": collect_path_signature(
            tokenizer_path,
            tracked_files=[
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ],
        ),
        "dtype": str(dtype),
        "max_length": int(max_length),
    }


def get_cache_dir(cache_root: Path, cache_key: str) -> Path:
    """Return canonical cache directory for one key."""
    shard = cache_key[:2] if len(cache_key) >= 2 else "zz"
    return cache_root / shard / cache_key


def try_load_feature_cache(
    *,
    cache_root: Path,
    cache_key: str,
    expected_signature_hash: str,
    torch_module: Any,
) -> tuple[Any | None, dict[str, Any] | None, Path]:
    """Try loading one cached payload; return `(payload, metadata, cache_dir)`.

    Returns `(None, None, cache_dir)` when missing or signature mismatch.
    """
    cache_dir = get_cache_dir(cache_root, cache_key)
    payload_path = cache_dir / "payload.pt"
    meta_path = cache_dir / "meta.json"
    if not payload_path.exists() or not meta_path.exists():
        return None, None, cache_dir

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None, None, cache_dir
    if not isinstance(metadata, dict):
        return None, None, cache_dir
    if str(metadata.get("schema_version")) != FEATURE_CACHE_SCHEMA_VERSION:
        return None, None, cache_dir
    if str(metadata.get("cache_key")) != str(cache_key):
        return None, None, cache_dir
    if str(metadata.get("signature_hash")) != str(expected_signature_hash):
        return None, None, cache_dir

    try:
        load_kwargs: dict[str, Any] = {"map_location": "cpu"}
        load_sig = inspect.signature(torch_module.load)
        if "weights_only" in load_sig.parameters:
            # Cache payload only stores tensors/basic containers.
            # Prefer strict weights-only loading for safer deserialization.
            load_kwargs["weights_only"] = True
        payload = torch_module.load(str(payload_path), **load_kwargs)
    except Exception:  # noqa: BLE001
        return None, None, cache_dir
    return payload, metadata, cache_dir


def save_feature_cache(
    *,
    cache_root: Path,
    cache_key: str,
    signature_hash: str,
    payload: Any,
    torch_module: Any,
    producer: str,
    lock_timeout_sec: float,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist one cache payload with lock + atomic file replacement."""
    cache_dir = get_cache_dir(cache_root, cache_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload_path = cache_dir / "payload.pt"
    meta_path = cache_dir / "meta.json"
    lock_path = cache_dir / ".write.lock"

    with _exclusive_lock(lock_path=lock_path, timeout_sec=float(lock_timeout_sec)):
        # Another process may have written it while we were waiting.
        if payload_path.exists() and meta_path.exists():
            return cache_dir

        tmp_payload = cache_dir / f"payload.pt.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        tmp_meta = cache_dir / f"meta.json.tmp.{os.getpid()}.{uuid.uuid4().hex}"

        payload_cpu = move_tensors_to_device(payload, torch_module.device("cpu"), torch_module)
        torch_module.save(payload_cpu, str(tmp_payload))

        metadata = {
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cache_key": str(cache_key),
            "signature_hash": str(signature_hash),
            "producer": str(producer),
            "extra": dict(extra_metadata or {}),
        }
        tmp_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        os.replace(tmp_payload, payload_path)
        os.replace(tmp_meta, meta_path)
    return cache_dir


@contextmanager
def _exclusive_lock(*, lock_path: Path, timeout_sec: float, poll_interval_sec: float = 0.2) -> Iterator[None]:
    """Acquire a coarse file lock using `O_EXCL` create semantics."""
    started = time.time()
    fd: int | None = None
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
            break
        except FileExistsError:
            if (time.time() - started) >= float(timeout_sec):
                raise TimeoutError(f"Timed out waiting feature-cache lock: {lock_path}")
            time.sleep(float(poll_interval_sec))
    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:  # noqa: BLE001
                pass
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


def move_tensors_to_device(payload: Any, device: Any, torch_module: Any) -> Any:
    """Recursively move tensor leaves in one nested payload to `device`."""
    if torch_module.is_tensor(payload):
        return payload.to(device)
    if isinstance(payload, dict):
        return {key: move_tensors_to_device(value, device, torch_module) for key, value in payload.items()}
    if isinstance(payload, list):
        return [move_tensors_to_device(value, device, torch_module) for value in payload]
    if isinstance(payload, tuple):
        return tuple(move_tensors_to_device(value, device, torch_module) for value in payload)
    return payload
