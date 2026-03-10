"""Define the lightweight Phase C/Phase E value head and frozen-feature helpers.

English
-------
This file sits at the boundary between the large language-model backbone and the
small scalar head used in Phase C / Phase E experiments.

The intended workflow is:
1. run the backbone once to produce hidden states,
2. pool one vector per text/prefix,
3. feed that vector into a tiny scalar head,
4. train/evaluate the head without repeatedly re-running the full transformer.

This deliberately conservative design makes debugging easier:
1. if results are bad, we can separate "feature problem" from "head/loss problem",
2. we avoid joint-training confounds too early,
3. checkpoint save/load stays consistent across scripts.

中文
----
这个文件位于“大模型 backbone”和“小型标量 value head”之间的边界位置。

它对应的工作流是：
1. 先让 backbone 跑一次，得到 hidden states；
2. 再把每段文本或 prefix 池化成一个向量；
3. 把这个向量送进一个很小的标量头；
4. 之后训练/评测只围绕这个 head 展开，不再反复重跑整个 transformer。

这里故意采用保守设计，目的是让实验更容易排查：
1. 结果不好时，更容易区分是“特征不行”还是“head/损失不行”；
2. 避免一开始就把 joint training 的复杂耦合混进来；
3. 不同脚本对 checkpoint 的保存/重载保持一致。

What this file contains / 本文件包含
-----------------------------------
1. `ValueHeadConfig`
   - serializable config for the scalar head
   - 标量头的可序列化配置
2. `SigmoidValueHead`
   - one-layer head returning both logits and sigmoid scores
   - 单层 value head，同时返回 logits 与 sigmoid 后分数
3. feature helpers for frozen backbones
   - helpers that run the LM once and pool one vector per sample
   - 冻结 backbone 后提取 pooled 特征的辅助函数
4. checkpoint helpers
   - save/load utilities shared by training and evaluation
   - 训练与评测共用的 checkpoint 读写工具
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ValueHeadConfig:
    """Configuration for the scalar value head.

    English
    -------
    The head is intentionally simple: dropout + one linear projection.
    This config object exists so we can rebuild the exact same module structure
    when loading checkpoints later.

    中文
    ----
    这个 head 故意设计得很简单：dropout + 一个线性投影。
    单独保留配置对象，是为了后续重载 checkpoint 时能准确重建模块结构，
    而不是靠调用方自己猜。
    """

    hidden_size: int
    dropout_prob: float = 0.0
    init_std: float = 0.02
    pooling: str = "last_token"
    architecture: str = "linear"
    mlp_hidden_size: int = 1024
    activation: str = "gelu"

    def validate(self) -> None:
        """Validate configuration values before constructing the head.

        中文
        ----
        把校验逻辑放在配置类内部，而不是散落在训练/评测脚本里，可以保证所有
        调用方都遵循同一套约束。
        """
        if not isinstance(self.hidden_size, int) or self.hidden_size <= 0:
            raise ValueError("`hidden_size` must be a positive int")
        if not isinstance(self.dropout_prob, (int, float)) or not (0.0 <= float(self.dropout_prob) < 1.0):
            raise ValueError("`dropout_prob` must be in [0, 1)")
        if not isinstance(self.init_std, (int, float)) or float(self.init_std) <= 0.0:
            raise ValueError("`init_std` must be > 0")
        if self.pooling != "last_token":
            raise ValueError("Only `last_token` pooling is currently supported")
        if self.architecture not in {"linear", "mlp"}:
            raise ValueError("`architecture` must be one of {'linear', 'mlp'}")
        if not isinstance(self.mlp_hidden_size, int) or self.mlp_hidden_size <= 0:
            raise ValueError("`mlp_hidden_size` must be a positive int")
        if self.activation not in {"gelu", "relu", "tanh"}:
            raise ValueError("`activation` must be one of {'gelu', 'relu', 'tanh'}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration payload."""
        self.validate()
        return asdict(self)


class SigmoidValueHead:  # runtime subclass resolved lazily to avoid top-level torch import in light paths
    """Scalar value head with sigmoid output.

    English
    -------
    The module returns two related views of the same prediction:
    1. `logits`
       - raw unbounded scalar before sigmoid
       - useful for BCE losses and anti-saturation penalties
    2. `scores`
       - `sigmoid(logits)` in `[0, 1]`
       - useful when we want a probability-like score

    `torch` is imported lazily in `__new__` so lightweight code paths do not
    need a full torch import just to inspect configs.

    中文
    ----
    这个模块会返回同一个预测的两种视角：
    1. `logits`
       - sigmoid 之前的原始标量
       - 适合 BCE loss、logit 饱和惩罚等
    2. `scores`
       - `sigmoid(logits)` 后得到的 `[0, 1]` 分数
       - 适合做“概率式分数”解释

    `torch` 被延迟到 `__new__` 中导入，是为了让一些轻量路径
    （例如只看配置）不用一加载文件就触发完整 torch 依赖。

    结构说明
    --------
    1. `linear`
       - 原始保守版本：dropout + 线性层
       - 适合作为最小可解释基线
    2. `mlp`
       - 两层 MLP：dropout + Linear + 激活 + dropout + Linear
       - 当我们只关心“同一数据集内部能不能判定准确”时，允许它提供更强的非线性判别能力
       - 这不会改变 backbone 特征提取方式，只是增强 head 的表达能力
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
                if cfg.activation == "gelu":
                    activation_layer = nn.GELU()
                elif cfg.activation == "relu":
                    activation_layer = nn.ReLU()
                else:
                    activation_layer = nn.Tanh()

                if cfg.architecture == "linear":
                    self.net = nn.Sequential(
                        self.dropout,
                        nn.Linear(int(cfg.hidden_size), 1),
                    )
                else:
                    # 当同源 learnability 的目标提高到 90%+ 时，单线性头可能不够。
                    # 这里提供一个受控的两层 MLP：容量更强，但仍远小于 backbone。
                    self.net = nn.Sequential(
                        self.dropout,
                        nn.Linear(int(cfg.hidden_size), int(cfg.mlp_hidden_size)),
                        activation_layer,
                        nn.Dropout(float(cfg.dropout_prob)),
                        nn.Linear(int(cfg.mlp_hidden_size), 1),
                    )
                self._init_linear_layers(std=float(cfg.init_std))

            def _init_linear_layers(self, *, std: float) -> None:
                for module in self.net.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, mean=0.0, std=float(std))
                        nn.init.zeros_(module.bias)

            def forward(self, features):
                # The upstream code must already pool each sample to one vector.
                # 上游必须已经把每个样本压成一个向量；head 本身不负责处理时序维。
                if features.ndim != 2:
                    raise ValueError(
                        f"Expected feature tensor of shape [batch, hidden], got {tuple(features.shape)!r}"
                    )
                logits = self.net(features).squeeze(-1)
                scores = torch.sigmoid(logits)
                return {"logits": logits, "scores": scores}

        return _Impl(config)


def freeze_backbone(backbone: Any) -> None:
    """Freeze all backbone parameters in place and switch to eval mode.

    English
    -------
    Two things happen here:
    1. `requires_grad=False` so the optimizer does not update the backbone
    2. `eval()` so dropout and similar modules switch to inference behavior

    Both are necessary.  Doing only one of them is not enough.

    中文
    ----
    这里同时做两件事：
    1. `requires_grad=False`，保证优化器不会更新 backbone
    2. `eval()`，保证 dropout 等模块切换到推理态

    两步都重要，只做其中一步都不完整。
    """
    # C2 默认冻结 backbone；只训练 value head，降低不稳定性和显存压力。
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()


def infer_backbone_hidden_size(backbone: Any) -> int:
    """Infer hidden size from a Hugging Face causal LM backbone.

    中文
    ----
    不同模型的配置字段名不一定一致，例如可能叫 `hidden_size`，也可能叫
    `n_embd`。这里统一封装，避免脚本里反复写兼容分支。
    """
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

    English
    -------
    This is the expensive part of the frozen-feature workflow: the transformer
    forward pass.  We do it once here, detach the pooled features, and let the
    small value head train on those tensors later.

    中文
    ----
    这是“冻结特征训练”流程里真正昂贵的部分：跑 transformer 前向。
    这里执行一次后，把 pooled 特征 `detach` 出来，后续训练只在这些张量上
    更新小 head，不再反复跑大模型。
    """
    if not texts:
        raise ValueError("`texts` must contain at least one item")
    # We locate the last valid token through `attention_mask`, which works for
    # both left-padding and right-padding.
    # 通过 `attention_mask` 定位最后一个有效 token，才能同时兼容左/右 padding。
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

    English
    -------
    The last useful token is not always `seq_len - 1` because batches may use
    left-padding or right-padding.  We therefore compute the last real token
    index from `attention_mask`.

    中文
    ----
    一个 batch 中的样本可能使用左 padding，也可能使用右 padding，所以“最后一个
    有效 token”不能简单写成 `seq_len - 1`。这里通过 `attention_mask` 计算每行
    的最后真实 token 位置。
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
    """Resolve which device should receive tokenized model inputs.

    中文
    ----
    有些模型对象直接暴露 `.device`，有些则需要从参数上推断。这里统一做兼容。
    """
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
    """Persist one value-head checkpoint as a torch `.pt` payload.

    中文
    ----
    这里不仅保存参数，还会一起保存 config 和额外元信息。这样后面重载时，
    就不会只拿到一堆权重却不知道对应的结构和训练状态。
    """
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

    中文
    ----
    返回三部分而不是一个对象，目的是把“模块本体”“结构配置”“附带训练元信息”
    明确拆开，方便评测和继续训练时分别处理。
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
    """Write the head config as a standalone JSON file for inspection.

    中文
    ----
    单独落一个 JSON 配置文件，对新手排查“这个 checkpoint 当初到底怎么建出来的”
    特别有帮助。
    """
    config.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
