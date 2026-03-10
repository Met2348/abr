"""Shared runtime helpers for Phase E training and benchmark evaluation.

Why this file exists
--------------------
Phase E needs the same frozen-backbone workflow in several places:

1. load the backbone and tokenizer safely,
2. freeze and cache encoded features,
3. score batches with a small value head,
4. keep deterministic behavior explicit.

Keeping these helpers in one place reduces the chance that training and
benchmark evaluation drift apart silently.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from ours.phase_b.feature_cache import (
    build_backbone_signature,
    build_cache_key,
    feature_cache_can_read,
    feature_cache_can_write,
    hash_jsonable,
    hash_text_list,
    save_feature_cache,
    try_load_feature_cache,
    validate_feature_cache_mode,
)
from ours.phase_b.value_head import encode_text_features


def import_runtime_deps() -> tuple[Any, Any, Any]:
    """Import heavy runtime dependencies lazily."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(name: str, torch_module: Any):
    """Map one user-facing dtype string to a torch dtype."""
    normalized = str(name).strip().lower()
    if normalized == "auto":
        if torch_module.cuda.is_available():
            return torch_module.bfloat16
        return torch_module.float32
    mapping = {
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {name!r}")
    return mapping[normalized]


def set_seed(
    seed: int,
    torch_module: Any,
    *,
    strict_determinism: bool = False,
) -> None:
    """Seed RNGs and optionally enable stricter deterministic backend behavior."""
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    if bool(strict_determinism):
        # This is best-effort determinism: reduce seed noise aggressively without forbidding unsupported ops.
        # 这里追求的是“尽力而为”的严格确定性，在不过度限制算子的前提下尽量压低 seed 波动。
        torch_module.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch_module.backends, "cudnn"):
            torch_module.backends.cudnn.deterministic = True
            torch_module.backends.cudnn.benchmark = False
            torch_module.backends.cudnn.allow_tf32 = False
        if hasattr(torch_module.backends, "cuda") and hasattr(torch_module.backends.cuda, "matmul"):
            torch_module.backends.cuda.matmul.allow_tf32 = False


def resolve_tokenizer_load_path(model_path: str, adapter_path: Path | None) -> str:
    """Resolve tokenizer source, preferring adapter-local tokenizer files."""
    if adapter_path is None:
        return str(model_path)
    if (adapter_path / "tokenizer_config.json").exists():
        return str(adapter_path)
    return str(model_path)


def attach_peft_adapter_for_inference(model: Any, adapter_path: Path):
    """Attach one PEFT adapter to a loaded base model."""
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    try:
        from peft import PeftModel  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to import `peft` while attaching adapter") from exc
    return PeftModel.from_pretrained(model, str(adapter_path))


def load_backbone_and_tokenizer(
    *,
    model_path: str,
    adapter_path: Path | None,
    dtype_name: str,
    device_map: str,
    torch_module: Any,
    AutoModelForCausalLM: Any,
    AutoTokenizer: Any,
) -> tuple[Any, Any, Any, str]:
    """Load tokenizer and frozen backbone for Phase E."""
    tokenizer_path = resolve_tokenizer_load_path(model_path, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        # Some released tokenizer snapshots omit a pad token even though batched scoring needs one.
        # 一些公开 tokenizer 快照没有 pad token，但批量打分时必须补齐这一语义。
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    resolved_dtype = resolve_dtype(dtype_name, torch_module)
    load_kwargs: dict[str, Any] = {
        "device_map": str(device_map),
        "trust_remote_code": True,
    }
    from_pretrained_sig = __import__("inspect").signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in from_pretrained_sig.parameters:
        load_kwargs["dtype"] = resolved_dtype
    else:
        load_kwargs["torch_dtype"] = resolved_dtype
    backbone = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    if adapter_path is not None:
        backbone = attach_peft_adapter_for_inference(backbone, adapter_path)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone, tokenizer, resolved_dtype, tokenizer_path


def resolve_model_input_device(model: Any, torch_module: Any):
    """Resolve which device should receive tokenized model inputs."""
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch_module.device("cpu")


def resolve_value_device(backbone: Any, torch_module: Any):
    """Resolve where cached features and the value head should live."""
    return resolve_model_input_device(backbone, torch_module)


def load_or_encode_text_features(
    *,
    cache_namespace: str,
    cache_kind: str,
    texts: list[str],
    backbone: Any,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    feature_cache_root: Path,
    feature_cache_mode: str,
    lock_timeout_sec: float,
    backbone_signature: dict[str, Any],
    extra_signature: dict[str, Any],
    torch_module: Any,
    feature_cache_stats: dict[str, Any],
):
    """Load cached features or encode them with OOM-safe backoff."""
    if not texts:
        raise ValueError("Feature encoding expects at least one text")
    feature_cache_mode = validate_feature_cache_mode(feature_cache_mode)
    # Bind cache reuse to both text content and backbone provenance.
    # cache 是否可复用必须同时绑定文本内容和 backbone 来源，二者缺一不可。
    signature_payload = {
        "cache_namespace": str(cache_namespace),
        "cache_kind": str(cache_kind),
        "texts_digest": hash_text_list(texts),
        "backbone_signature": backbone_signature,
        "extra_signature_hash": hash_jsonable(extra_signature),
        "max_length": int(max_length),
    }
    cache_key, signature_hash = build_cache_key(cache_namespace, signature_payload)
    stat_key = f"{cache_kind}_features"

    if feature_cache_can_read(feature_cache_mode):
        cached_payload, _, _ = try_load_feature_cache(
            cache_root=feature_cache_root,
            cache_key=cache_key,
            expected_signature_hash=signature_hash,
            torch_module=torch_module,
        )
        if isinstance(cached_payload, dict):
            try:
                features = cached_payload["features"]
                if not torch_module.is_tensor(features) or features.ndim != 2:
                    raise TypeError("cached features payload must be tensor[batch, hidden]")
                if int(features.shape[0]) != len(texts):
                    raise ValueError(
                        f"cached feature count mismatch: got {features.shape[0]}, expected {len(texts)}"
                    )
                if int(features.shape[1]) <= 0:
                    raise ValueError("cached feature hidden size must be > 0")
                feature_cache_stats["hits"] = int(feature_cache_stats.get("hits", 0)) + 1
                feature_cache_stats.setdefault("entries", {})[stat_key] = {
                    "hit": True,
                    "cache_key": cache_key,
                }
                print(f"feature_cache    : {stat_key} hit ({cache_key})", flush=True)
                return features
            except Exception as exc:  # noqa: BLE001
                print(
                    f"feature_cache    : {stat_key} invalid payload, fallback to re-encode ({exc})",
                    flush=True,
                )

    feature_cache_stats["misses"] = int(feature_cache_stats.get("misses", 0)) + 1
    feature_cache_stats.setdefault("entries", {})[stat_key] = {
        "hit": False,
        "cache_key": cache_key,
    }
    print(f"feature_cache    : {stat_key} miss ({cache_key})", flush=True)
    encoded = encode_texts_batched_oom_safe(
        texts=texts,
        backbone=backbone,
        tokenizer=tokenizer,
        max_length=int(max_length),
        batch_size=int(batch_size),
        torch_module=torch_module,
        log_prefix=f"cache_{cache_kind}",
    )
    if feature_cache_can_write(feature_cache_mode):
        save_feature_cache(
            cache_root=feature_cache_root,
            cache_key=cache_key,
            signature_hash=signature_hash,
            payload={"features": encoded},
            torch_module=torch_module,
            producer="src/ours/phase_e/runtime.py",
            lock_timeout_sec=float(lock_timeout_sec),
            extra_metadata={"num_texts": int(len(texts)), "cache_kind": str(cache_kind)},
        )
        feature_cache_stats["writes"] = int(feature_cache_stats.get("writes", 0)) + 1
        feature_cache_stats["entries"][stat_key]["write"] = True
    return encoded


def encode_texts_batched_oom_safe(
    *,
    texts: list[str],
    backbone: Any,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    torch_module: Any,
    log_prefix: str,
):
    """Encode texts in batches with OOM backoff."""
    outputs: list[Any] = []
    total = len(texts)
    index = 0
    current_batch_size = max(1, int(batch_size))
    progress_every = max(1, total // 8)
    start = time.perf_counter()
    print(
        f"{log_prefix:16}: start {total} texts in {((total - 1) // current_batch_size) + 1} batches (bs={current_batch_size})",
        flush=True,
    )
    while index < total:
        effective_bs = min(current_batch_size, total - index)
        batch_texts = texts[index : index + effective_bs]
        try:
            batch_features = encode_text_features(
                backbone=backbone,
                tokenizer=tokenizer,
                texts=batch_texts,
                max_length=int(max_length),
                torch_module=torch_module,
            )
        except RuntimeError as exc:
            # Retry the same slice with a smaller batch on OOM instead of dropping data silently.
            # 遇到 OOM 就缩小 batch 重试同一段文本，绝不通过静默丢样本来“绕过”错误。
            if "out of memory" not in str(exc).lower() or effective_bs <= 1:
                raise
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
            current_batch_size = max(1, effective_bs // 2)
            print(
                f"{log_prefix:16}: OOM backoff -> batch_size={current_batch_size}",
                flush=True,
            )
            continue
        # Keep the long-lived feature cache on CPU.
        # 关键点：Phase E 的缓存特征必须尽快落到 CPU，不能把所有 batch 的结果一直堆在 GPU 上。
        # 否则随着文本批次累积，显存会线性上涨，最后即使单 batch 不大也会把整张卡慢慢吃满。
        outputs.append(batch_features.detach().cpu())
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        index += effective_bs
        if index % progress_every == 0 or index >= total:
            elapsed = max(time.perf_counter() - start, 1e-9)
            rate = index / elapsed
            print(
                f"{log_prefix:16}: {index}/{total} ({100.0 * index / total:.1f}%) | elapsed={elapsed:.1f}s | rate={rate:.3f} text/s",
                flush=True,
            )
    print(f"{log_prefix:16}: done", flush=True)
    return torch_module.cat(outputs, dim=0)


def score_feature_tensor(
    *,
    value_head: Any,
    features: Any,
    batch_size: int,
    torch_module: Any,
) -> list[float]:
    """Score one feature tensor in batches with OOM backoff."""
    total = int(features.shape[0])
    head_device = next(value_head.parameters()).device
    head_dtype = next(value_head.parameters()).dtype
    scores: list[float] = []
    index = 0
    current_batch_size = max(1, int(batch_size))
    with torch_module.no_grad():
        while index < total:
            effective_bs = min(current_batch_size, total - index)
            batch = features[index : index + effective_bs]
            if batch.device != head_device or batch.dtype != head_dtype:
                batch = batch.to(device=head_device, dtype=head_dtype)
            try:
                outputs = value_head(batch)["scores"]
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or effective_bs <= 1:
                    raise
                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
                current_batch_size = max(1, effective_bs // 2)
                continue
            scores.extend(float(v) for v in outputs.detach().cpu().tolist())
            index += effective_bs
    return scores


def score_pair_features(
    *,
    value_head: Any,
    chosen_features: Any,
    rejected_features: Any,
    batch_size: int,
    torch_module: Any,
) -> tuple[list[float], list[float]]:
    """Score chosen/rejected feature tensors in aligned batches."""
    if chosen_features.shape != rejected_features.shape:
        raise ValueError("chosen_features and rejected_features must have equal shapes")
    total = int(chosen_features.shape[0])
    head_device = next(value_head.parameters()).device
    head_dtype = next(value_head.parameters()).dtype
    chosen_scores: list[float] = []
    rejected_scores: list[float] = []
    index = 0
    current_batch_size = max(1, int(batch_size))
    with torch_module.no_grad():
        while index < total:
            effective_bs = min(current_batch_size, total - index)
            chosen_batch = chosen_features[index : index + effective_bs]
            rejected_batch = rejected_features[index : index + effective_bs]
            if chosen_batch.device != head_device or chosen_batch.dtype != head_dtype:
                chosen_batch = chosen_batch.to(device=head_device, dtype=head_dtype)
            if rejected_batch.device != head_device or rejected_batch.dtype != head_dtype:
                rejected_batch = rejected_batch.to(device=head_device, dtype=head_dtype)
            try:
                chosen_out = value_head(chosen_batch)["scores"]
                rejected_out = value_head(rejected_batch)["scores"]
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or effective_bs <= 1:
                    raise
                if torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
                current_batch_size = max(1, effective_bs // 2)
                continue
            chosen_scores.extend(float(v) for v in chosen_out.detach().cpu().tolist())
            rejected_scores.extend(float(v) for v in rejected_out.detach().cpu().tolist())
            index += effective_bs
    return chosen_scores, rejected_scores


def build_phase_e_backbone_signature(
    *,
    model_path: str,
    adapter_path: str | None,
    tokenizer_path: str,
    dtype: str,
    max_length: int,
) -> dict[str, Any]:
    """Build cache provenance signature for Phase E feature encoding."""
    return build_backbone_signature(
        model_path=str(model_path),
        adapter_path=(str(adapter_path) if adapter_path is not None else None),
        tokenizer_path=str(tokenizer_path),
        dtype=str(dtype),
        max_length=int(max_length),
    )


def stable_hash_order(values: list[int], *, ids: list[str]) -> list[int]:
    """Return deterministic order by item id hash."""
    if len(values) != len(ids):
        raise ValueError("stable_hash_order expects aligned `values` and `ids`")
    indexed_ids = {idx: item_id for idx, item_id in zip(values, ids, strict=True)}
    return sorted(
        values,
        key=lambda idx: hashlib.sha256(str(indexed_ids[idx]).encode("utf-8")).hexdigest(),
    )


def resolve_checkpoint_path(
    *,
    value_run_dir: Path,
    run_manifest: dict[str, Any],
    checkpoint_name: str,
) -> Path:
    """Resolve checkpoint path from one Phase E run manifest."""
    files = dict(run_manifest.get("output_files", {}))
    if checkpoint_name == "best":
        best_path = files.get("best_value_head")
        if isinstance(best_path, str) and best_path.strip():
            path = Path(best_path)
            if path.exists():
                return path
        final_path = files.get("final_value_head")
        if isinstance(final_path, str) and final_path.strip():
            path = Path(final_path)
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Neither best nor final value-head checkpoint found under manifest: {value_run_dir}"
        )
    final_path = files.get("final_value_head")
    if isinstance(final_path, str) and final_path.strip():
        path = Path(final_path)
        if path.exists():
            return path
    raise FileNotFoundError(f"Requested checkpoint=final but file missing in {value_run_dir}")
