"""Define the Phase C value head and frozen-backbone feature extraction helpers.

Why this file exists
--------------------
C2 is the first actual Phase C training stage. It trains a scalar value head on
prefix-level supervision while keeping the backbone frozen. This module keeps the
head definition, checkpoint helpers, and pooled-feature extraction in one place
so training/evaluation scripts do not silently disagree.

What this file contains
-----------------------
- `ValueHeadConfig`: serializable head configuration
- `SigmoidValueHead`: simple scalar head with output in `[0, 1]`
- pooled-feature extraction helpers for frozen causal-LM backbones
- save/load helpers for value-head checkpoints

Interaction with other files
----------------------------
- `scripts/phase_b_train_value.py`: trains the head
- `scripts/phase_b_eval_faithfulness.py`: reloads and evaluates the head
- `src/ours/phase_b/faithfulness_eval.py`: scores the head outputs

Example
-------
```python
config = ValueHeadConfig(hidden_size=3584)
head = SigmoidValueHead(config)
```
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ValueHeadConfig:
    """Configuration for the scalar Phase C value head."""

    hidden_size: int
    dropout_prob: float = 0.0
    init_std: float = 0.02
    pooling: str = "last_token"

    def validate(self) -> None:
        """Validate configuration values before constructing the head."""
        if not isinstance(self.hidden_size, int) or self.hidden_size <= 0:
            raise ValueError("`hidden_size` must be a positive int")
        if not isinstance(self.dropout_prob, (int, float)) or not (0.0 <= float(self.dropout_prob) < 1.0):
            raise ValueError("`dropout_prob` must be in [0, 1)")
        if not isinstance(self.init_std, (int, float)) or float(self.init_std) <= 0.0:
            raise ValueError("`init_std` must be > 0")
        if self.pooling != "last_token":
            raise ValueError("Only `last_token` pooling is currently supported")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration payload."""
        self.validate()
        return asdict(self)


class SigmoidValueHead:  # runtime subclass resolved lazily to avoid top-level torch import in light paths
    """One-layer scalar value head with sigmoid output.

    The actual `torch.nn.Module` implementation is constructed in `__new__` so
    importing this module does not require `torch` until the caller truly needs
    it.
    """

    def __new__(cls, config: ValueHeadConfig):
        config.validate()
        import torch
        import torch.nn as nn

        class _Impl(nn.Module):
            def __init__(self, cfg: ValueHeadConfig) -> None:
                super().__init__()
                self.config = cfg
                self.dropout = nn.Dropout(float(cfg.dropout_prob))
                self.proj = nn.Linear(int(cfg.hidden_size), 1)
                nn.init.normal_(self.proj.weight, mean=0.0, std=float(cfg.init_std))
                nn.init.zeros_(self.proj.bias)

            def forward(self, features):
                if features.ndim != 2:
                    raise ValueError(
                        f"Expected feature tensor of shape [batch, hidden], got {tuple(features.shape)!r}"
                    )
                logits = self.proj(self.dropout(features)).squeeze(-1)
                scores = torch.sigmoid(logits)
                return {"logits": logits, "scores": scores}

        return _Impl(config)


def freeze_backbone(backbone: Any) -> None:
    """Freeze all backbone parameters in place and switch to eval mode.

    Example
    -------
    ```python
    freeze_backbone(model)
    ```
    """
    # C2 默认冻结 backbone；只训练 value head，降低不稳定性和显存压力。
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()


def infer_backbone_hidden_size(backbone: Any) -> int:
    """Infer hidden size from a Hugging Face causal LM backbone."""
    hidden_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
    if isinstance(hidden_size, int) and hidden_size > 0:
        return hidden_size
    hidden_size = getattr(getattr(backbone, "config", None), "n_embd", None)
    if isinstance(hidden_size, int) and hidden_size > 0:
        return hidden_size
    raise ValueError("Failed to infer hidden size from backbone config")


def encode_text_features(
    *,
    backbone: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    torch_module: Any,
) -> Any:
    """Encode texts once with a frozen backbone and return pooled features.

    This function performs the expensive LM forward exactly once per input text.
    Since C2 freezes the backbone, callers can cache these features and train the
    small value head on top without repeatedly re-running the transformer.
    """
    if not texts:
        raise ValueError("`texts` must contain at least one item")
    # 统一通过 attention_mask 做池化定位，兼容左/右 padding。
    model_device = resolve_model_input_device(backbone)
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_length),
    ).to(model_device)

    with torch_module.no_grad():
        outputs = backbone(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        pooled = pool_last_token(last_hidden, inputs["attention_mask"], torch_module=torch_module)
    return pooled.detach()


def pool_last_token(hidden_states: Any, attention_mask: Any, *, torch_module: Any):
    """Pool the hidden state at the last attended token position.

    Right-padding and left-padding are both supported because the location is
    determined from `attention_mask`, not from the raw sequence length.
    """
    if hidden_states.ndim != 3:
        raise ValueError("`hidden_states` must have shape [batch, seq, hidden]")
    if attention_mask.ndim != 2:
        raise ValueError("`attention_mask` must have shape [batch, seq]")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("Hidden states and attention mask batch/seq dims must match")

    lengths = attention_mask.sum(dim=1) - 1
    lengths = torch_module.clamp(lengths, min=0)
    batch_indices = torch_module.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_indices, lengths]


def resolve_model_input_device(model: Any):
    """Resolve which device should receive tokenized model inputs."""
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def save_value_head_checkpoint(
    path: Path,
    *,
    value_head: Any,
    config: ValueHeadConfig,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Persist one value-head checkpoint as a torch `.pt` payload."""
    import torch

    config.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.to_dict(),
        "state_dict": value_head.state_dict(),
        "extra_state": dict(extra_state or {}),
    }
    torch.save(payload, path)


def load_value_head_checkpoint(
    path: Path,
    *,
    map_location: str | None = None,
) -> tuple[Any, ValueHeadConfig, dict[str, Any]]:
    """Load one saved value head and its config.

    Returns
    -------
    tuple[Any, ValueHeadConfig, dict[str, Any]]
        `(value_head_module, config, extra_state)`
    """
    import torch

    if not path.exists():
        raise FileNotFoundError(f"Value head checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Value head checkpoint {path} must contain a dict payload")
    config = ValueHeadConfig(**dict(payload.get("config", {})))
    config.validate()
    value_head = SigmoidValueHead(config)
    value_head.load_state_dict(dict(payload.get("state_dict", {})))
    extra_state = dict(payload.get("extra_state", {}))
    return value_head, config, extra_state


def write_value_head_config_json(path: Path, config: ValueHeadConfig) -> None:
    """Write the head config as a standalone JSON file for inspection."""
    config.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
