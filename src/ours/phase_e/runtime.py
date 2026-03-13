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
from ours.phase_b.value_head import (
    encode_text_features,
    ensure_tokenizer_has_pad_token,
    maybe_resize_embeddings_for_tokenizer as _maybe_resize_embeddings_for_tokenizer,
)


def import_runtime_deps() -> tuple[Any, Any, Any]:
    """Import heavy runtime dependencies lazily."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


def resolve_backbone_loader_family(
    *,
    model_path: str,
    trust_remote_code: bool = True,
) -> str:
    """Resolve which Hugging Face auto-model family should load one backbone.

    English
    -------
    Most Phase E runs use a causal LM backbone, but some community PRM releases
    package the tuned transformer under a custom reward-model head rather than a
    causal-LM head.  For frozen feature extraction we only need the transformer
    hidden states, so this helper allows the runtime to load either family
    transparently.

    中文
    ----
    Phase E 大多数实验默认把 backbone 当作 causal LM 加载，但社区里已有一些
    PRM checkpoint 是挂在自定义 reward-model 头下面的。对当前冻结特征流程来说，
    我们真正需要的只是 hidden states，因此这里先解析模型家族，再决定用
    `AutoModelForCausalLM` 还是 `AutoModel`。
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            str(model_path),
            trust_remote_code=bool(trust_remote_code),
        )
    except Exception:  # noqa: BLE001
        # Fail open to the legacy path so ordinary causal-LM runs never depend
        # on config introspection succeeding.
        return "causal_lm"

    architectures = getattr(config, "architectures", None)
    if not isinstance(architectures, list):
        return "causal_lm"
    normalized = [str(item).strip().lower() for item in architectures if str(item).strip()]
    if any("processrewardmodel" in item or "rewardmodel" in item for item in normalized):
        return "process_reward_model"
    return "causal_lm"


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
    """Attach one PEFT adapter to a loaded base model.

    Works for both CausalLM and non-CausalLM models (e.g. Qwen2ForProcessRewardModel).
    PeftModel.from_pretrained() fails on PRM models when task_type=CAUSAL_LM is saved
    in the adapter config because it tries to wrap the model as PeftModelForCausalLM,
    which requires prepare_inputs_for_generation.  This function avoids that by using
    get_peft_model() + manual weight loading instead.
    """
    import dataclasses
    import json

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    try:
        from peft import LoraConfig, get_peft_model  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to import `peft` while attaching adapter") from exc

    config_path = adapter_path / "adapter_config.json"
    config_dict = json.loads(config_path.read_text())
    # Strip task_type so get_peft_model does not create a task-specific wrapper
    # (PeftModelForCausalLM) that requires methods absent on PRM models.
    config_dict.pop("task_type", None)
    valid_fields = {f.name for f in dataclasses.fields(LoraConfig)}
    lora_config = LoraConfig(**{k: v for k, v in config_dict.items() if k in valid_fields})
    model = get_peft_model(model, lora_config)

    # Load saved weights
    import torch

    sf_path = adapter_path / "adapter_model.safetensors"
    bin_path = adapter_path / "adapter_model.bin"
    if sf_path.exists():
        from safetensors.torch import load_file  # type: ignore

        weights = load_file(str(sf_path))
    elif bin_path.exists():
        weights = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter_path}")
    # Remap legacy key format: older PEFT saves used `lora_A.weight` (no adapter name),
    # but current PEFT (≥0.7) expects `lora_A.default.weight` (`.default.` inserted).
    # If zero keys matched and the first saved key looks like `lora_A.weight`, remap.
    model_keys = set(model.state_dict().keys())
    needs_remap = any((".lora_A.weight" in k or ".lora_B.weight" in k) for k in weights)
    has_default = any(".lora_A.default.weight" in k for k in model_keys)
    if needs_remap and has_default:
        import re
        remapped = {}
        for k, v in weights.items():
            # `lora_A.weight` → `lora_A.default.weight`, same for lora_B
            k2 = re.sub(r"\.(lora_[AB])\.weight$", r".\1.default.weight", k)
            remapped[k2] = v
        weights = remapped
    model.load_state_dict(weights, strict=False)
    return model


def apply_lora_to_backbone(
    *,
    backbone: Any,
    lora_rank: int,
    lora_alpha: float,
    target_modules: list[str] | None = None,
    num_top_layers: int | None = None,
    lora_dropout: float = 0.05,
) -> Any:
    """Apply LoRA adapters to a Phase E backbone using PEFT.

    English
    -------
    This function wraps the backbone with trainable LoRA layers.  After this
    call the backbone is **no longer frozen**: the LoRA `A` and `B` matrices
    are enabled for gradient computation while all original backbone weights
    remain frozen.

    The typical use-case is to call this function immediately after
    `load_backbone_and_tokenizer`, then rebuild the optimizer to include both
    value-head parameters and backbone LoRA parameters.

    中文
    ----
    调用此函数后，backbone 不再完全冻结：LoRA A/B 矩阵可以接收梯度，
    而原始 backbone 权重仍然被锁定。
    通常在 `load_backbone_and_tokenizer` 之后立即调用，然后重建优化器，
    同时包含 value head 参数和 backbone LoRA 参数。

    Parameters
    ----------
    backbone:
        A loaded HuggingFace model (output of `load_backbone_and_tokenizer`).
    lora_rank:
        LoRA rank `r`.  Larger rank = more capacity but more memory.
        Typical: 8–32 for small GPUs, 64–128 for full fine-tuning.
    lora_alpha:
        LoRA scaling factor.  Effective scale = lora_alpha / lora_rank.
        Commonly set to lora_rank (scale=1.0) or 2*lora_rank (scale=2.0).
    target_modules:
        List of module name patterns to apply LoRA to.  If None, defaults
        to ["q_proj", "v_proj"] which is safe and memory-efficient for
        Qwen2.5 / LLaMA-style architectures.
    num_top_layers:
        If set, only the top-N transformer layers receive LoRA adapters.
        Useful when GPU memory is tight: attaching LoRA to all 28 layers
        of Qwen2.5-7B requires ~1.5 GB extra; attaching to top 8 layers
        requires ~430 MB.
    lora_dropout:
        Dropout applied inside LoRA adapters during training.  Default 0.05.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "peft is required for LoRA training.  Install with: pip install peft>=0.10"
        ) from exc

    if lora_rank <= 0:
        raise ValueError(f"lora_rank must be > 0, got {lora_rank}")
    if lora_alpha <= 0.0:
        raise ValueError(f"lora_alpha must be > 0, got {lora_alpha}")

    # Default target modules for Qwen2.5 / LLaMA-style architectures.
    # q_proj + v_proj is the minimum effective set (Hu et al. 2021 LoRA paper).
    resolved_target_modules = target_modules if target_modules is not None else ["q_proj", "v_proj"]

    # Build layer-selection filter if num_top_layers is specified.
    # Qwen2.5-7B has 28 layers (model.layers[0..27]).  "Top" means closest to
    # the output (highest indices), since those layers produce the most
    # task-relevant representations for a value head.
    layers_to_transform: list[int] | None = None
    if num_top_layers is not None and num_top_layers > 0:
        # Try to infer total layer count from the backbone config.
        num_layers = getattr(getattr(backbone, "config", None), "num_hidden_layers", None)
        if num_layers is None:
            # Fall back: count children named "layers"
            try:
                layers_obj = backbone.model.layers  # type: ignore
                num_layers = len(layers_obj)
            except AttributeError:
                num_layers = 32  # conservative default

        start = max(0, int(num_layers) - int(num_top_layers))
        layers_to_transform = list(range(start, int(num_layers)))

    lora_config = LoraConfig(
        task_type=None,  # None avoids PeftModelForCausalLM wrapper on PRM backbones
        r=int(lora_rank),
        lora_alpha=float(lora_alpha),
        target_modules=resolved_target_modules,
        lora_dropout=float(lora_dropout),
        bias="none",
        layers_to_transform=layers_to_transform,
    )

    backbone = get_peft_model(backbone, lora_config)

    # After get_peft_model the model is in train mode.  Only LoRA layers should
    # receive gradients; everything else stays frozen.
    for name, param in backbone.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    # Enable gradient checkpointing to avoid storing all intermediate activations.
    # Without this, LoRA backprop through a 7B model at batch=4 seq=1024 easily
    # exceeds 60 GB VRAM.  Gradient checkpointing trades ~30% extra compute for
    # ~4-5× activation memory reduction, making LoRA feasible on A100-80GB.
    #
    # 开启梯度检查点，避免在反向传播时存储全部中间激活值。
    # 不开的话，batch=4 seq=1024 下对 7B 模型做 LoRA 反传，激活内存超过 60 GB。
    # 开启后以约 30% 额外计算换取 ~4-5 倍激活内存节省，让 A100-80G 能够运行 LoRA。
    try:
        # `enable_input_require_grads` ensures gradients flow into the first module.
        # Required by PEFT's gradient checkpointing integration.
        backbone.enable_input_require_grads()
    except AttributeError:
        # Some PEFT model wrappers may not have this; safe to skip.
        pass
    try:
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except (AttributeError, TypeError):
        # Older transformers or models that do not support kwargs form.
        try:
            backbone.gradient_checkpointing_enable()
        except AttributeError:
            print("[LoRA][WARNING] Could not enable gradient checkpointing — may OOM on large batches")

    # Print trainable parameter summary.
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    print(
        f"[LoRA] Applied LoRA (rank={lora_rank}, alpha={lora_alpha}, "
        f"modules={resolved_target_modules}, top_layers={num_top_layers}). "
        f"Trainable backbone params: {trainable:,} / {total:,} "
        f"({100.0 * trainable / total:.3f}%) | gradient_checkpointing=enabled"
    )

    return backbone


def load_backbone_and_tokenizer(
    *,
    model_path: str,
    adapter_path: Path | None,
    dtype_name: str,
    device_map: str,
    max_gpu_memory_gib: int | None,
    max_cpu_memory_gib: int | None,
    torch_module: Any,
    AutoModelForCausalLM: Any,
    AutoTokenizer: Any,
) -> tuple[Any, Any, Any, str]:
    """Load tokenizer and frozen backbone for Phase E."""
    tokenizer_path = resolve_tokenizer_load_path(model_path, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    synthesized_pad_token = ensure_tokenizer_has_pad_token(tokenizer)

    resolved_dtype = resolve_dtype(dtype_name, torch_module)
    load_kwargs: dict[str, Any] = {
        "device_map": str(device_map),
        "trust_remote_code": True,
        # Stream weights through the meta-device path instead of materializing
        # a full temporary CPU copy first.
        # 这里显式打开 low_cpu_mem_usage，避免先在 CPU 上临时摊开完整权重副本，
        # 对单机研究环境更稳。
        "low_cpu_mem_usage": True,
    }
    from_pretrained_sig = __import__("inspect").signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in from_pretrained_sig.parameters:
        load_kwargs["dtype"] = resolved_dtype
    else:
        load_kwargs["torch_dtype"] = resolved_dtype
    max_memory = build_max_memory_map(
        torch_module=torch_module,
        max_gpu_memory_gib=max_gpu_memory_gib,
        max_cpu_memory_gib=max_cpu_memory_gib,
    )
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    loader_family = resolve_backbone_loader_family(
        model_path=str(model_path),
        trust_remote_code=True,
    )
    if loader_family == "process_reward_model":
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(str(model_path), **load_kwargs)
    else:
        backbone = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    if synthesized_pad_token:
        _maybe_resize_embeddings_for_tokenizer(backbone=backbone, tokenizer=tokenizer)
    if adapter_path is not None:
        backbone = attach_peft_adapter_for_inference(backbone, adapter_path)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone, tokenizer, resolved_dtype, tokenizer_path


def build_max_memory_map(
    *,
    torch_module: Any,
    max_gpu_memory_gib: int | None,
    max_cpu_memory_gib: int | None,
) -> dict[Any, str] | None:
    """Build one Hugging Face `max_memory` map for controlled CPU/GPU offload.

    English
    -------
    This is a safety valve for busy shared machines:
    1. cap visible GPU memory per device,
    2. optionally allow the rest to spill into CPU RAM,
    3. keep the setting explicit so manifests record that the run was
       memory-constrained by design.

    中文
    ----
    这是给共享机器准备的安全阀：
    1. 对每张可见 GPU 设显存上限，
    2. 剩余部分按需 spill 到 CPU RAM，
    3. 同时把这个设定显式记录下来，避免后续忘记这次运行其实是 memory-constrained 的。
    """
    payload: dict[Any, str] = {}
    if max_gpu_memory_gib is not None:
        if int(max_gpu_memory_gib) <= 0:
            raise ValueError("`max_gpu_memory_gib` must be > 0 when provided")
        for device_idx in range(int(torch_module.cuda.device_count())):
            payload[int(device_idx)] = f"{int(max_gpu_memory_gib)}GiB"
    if max_cpu_memory_gib is not None:
        if int(max_cpu_memory_gib) <= 0:
            raise ValueError("`max_cpu_memory_gib` must be > 0 when provided")
        payload["cpu"] = f"{int(max_cpu_memory_gib)}GiB"
    return payload or None


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
            # Force async CUDA kernel failures to surface inside this try block.
            # 让异步 CUDA 错误在当前 try 块里暴露，避免后面的 `.cpu()` 才把真正的
            # 容量问题或 kernel failure 报成更难读的泛化错误。
            if torch_module.cuda.is_available():
                torch_module.cuda.synchronize()
        except RuntimeError as exc:
            # Retry the same slice with a smaller batch on OOM instead of dropping data silently.
            # 遇到 OOM 就缩小 batch 重试同一段文本，绝不通过静默丢样本来“绕过”错误。
            if not _is_retryable_cuda_capacity_error(exc) or effective_bs <= 1:
                raise
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
            current_batch_size = max(1, effective_bs // 2)
            print(
                f"{log_prefix:16}: capacity backoff ({type(exc).__name__}) -> batch_size={current_batch_size}",
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


def _is_retryable_cuda_capacity_error(exc: RuntimeError) -> bool:
    """Return whether one encode-time CUDA failure should trigger batch backoff.

    English
    -------
    In practice, long batched forwards can surface memory pressure in more than
    one textual form:
    1. the normal `CUDA out of memory`,
    2. allocator-style failures such as `CUBLAS_STATUS_ALLOC_FAILED`,
    3. or a later `device-side assert triggered` when an async kernel failure is
       first synchronized on a subsequent CUDA call.

    We only use this relaxed retry rule inside the frozen-feature encoding path.
    The follow-up retry still encodes the exact same texts with a smaller batch,
    so it does not silently drop data or alter semantics.

    中文
    ----
    实际上，长批次前向的显存压力不一定总是以同一种报错字符串出现：
    1. 常见的 `CUDA out of memory`，
    2. 分配器类失败，如 `CUBLAS_STATUS_ALLOC_FAILED`，
    3. 以及异步 kernel 在后续同步点才暴露成 `device-side assert triggered`。

    这里放宽重试规则，只用于冻结特征编码路径。
    后续 retry 仍然编码同一批文本，只是缩小 batch，不会静默丢样本或改变监督语义。
    """
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "out of memory",
            "cublas_status_alloc_failed",
            "device-side assert triggered",
        )
    )


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


def resolve_checkpoint_resolution(
    *,
    value_run_dir: Path,
    run_manifest: dict[str, Any],
    checkpoint_name: str,
    checkpoint_missing_policy: str = "fail",
) -> dict[str, Any]:
    """Resolve checkpoint path from one Phase E run manifest with explicit policy.

    中文
    ----
    这里不再把 `best -> final` 当成理所当然的隐式行为。
    默认策略应该是 `fail`，因为研究者请求 `best` 却评到了 `final`
    会直接污染实验结论。只有显式允许时，才做回退。
    """
    files = dict(run_manifest.get("output_files", {}))
    requested = str(checkpoint_name).strip().lower()
    if requested not in {"best", "final"}:
        raise ValueError(f"Unsupported checkpoint_name: {checkpoint_name!r}")
    requested_path_text = str(files.get("best_value_head" if requested == "best" else "final_value_head") or "").strip()
    final_path_text = str(files.get("final_value_head") or "").strip()
    best_path = Path(requested_path_text) if requested == "best" and requested_path_text else None
    final_path = Path(final_path_text) if final_path_text else None
    resolved_checkpoint_name = requested
    fallback_to_final = False

    checkpoint_missing_policy = str(checkpoint_missing_policy).strip().lower()
    if checkpoint_missing_policy not in {"fail", "fallback_final"}:
        checkpoint_missing_policy = "fail"

    if requested == "best":
        if best_path is not None and best_path.exists():
            resolved_path = best_path
        elif checkpoint_missing_policy == "fallback_final" and final_path is not None and final_path.exists():
            print(
                "checkpoint_resolve: requested=best but best_value_head missing; "
                f"falling back to final checkpoint at {final_path}",
                flush=True,
            )
            resolved_path = final_path
            resolved_checkpoint_name = "final"
            fallback_to_final = True
        else:
            raise FileNotFoundError(
                f"Requested checkpoint=best but best_value_head missing under {value_run_dir}; "
                "rerun with explicit checkpoint-missing-policy=fallback_final only for legacy diagnostics."
            )
    else:
        if final_path is None or not final_path.exists():
            raise FileNotFoundError(f"Requested checkpoint=final but file missing in {value_run_dir}")
        resolved_path = final_path

    return {
        "requested_checkpoint_name": requested,
        "requested_checkpoint_path": requested_path_text,
        "resolved_checkpoint_name": resolved_checkpoint_name,
        "resolved_checkpoint_path": str(resolved_path),
        "fallback_to_final": bool(fallback_to_final),
        "checkpoint_missing_policy": checkpoint_missing_policy,
    }


def resolve_checkpoint_path(
    *,
    value_run_dir: Path,
    run_manifest: dict[str, Any],
    checkpoint_name: str,
    checkpoint_missing_policy: str = "fail",
) -> Path:
    """Backward-compatible wrapper that returns only the resolved checkpoint path."""
    resolution = resolve_checkpoint_resolution(
        value_run_dir=value_run_dir,
        run_manifest=run_manifest,
        checkpoint_name=checkpoint_name,
        checkpoint_missing_policy=checkpoint_missing_policy,
    )
    return Path(str(resolution["resolved_checkpoint_path"]))
