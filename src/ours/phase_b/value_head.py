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
    inference_alpha: float = 0.5

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
        if self.architecture not in {"linear", "mlp", "gated_mlp", "dual_head"}:
            raise ValueError("`architecture` must be one of {'linear', 'mlp', 'gated_mlp', 'dual_head'}")
        if not isinstance(self.mlp_hidden_size, int) or self.mlp_hidden_size <= 0:
            raise ValueError("`mlp_hidden_size` must be a positive int")
        if self.activation not in {"gelu", "relu", "tanh"}:
            raise ValueError("`activation` must be one of {'gelu', 'relu', 'tanh'}")
        if not isinstance(self.inference_alpha, (int, float)) or not (0.0 <= float(self.inference_alpha) <= 1.0):
            raise ValueError("`inference_alpha` must be in [0, 1]")

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
    3. `dual_head`
       - 一个 shared trunk，后接 `local_head` 和 `terminal_head`
       - 训练时可以让 local 类 pair 主要更新 `local_head`，terminal 类 pair
         主要更新 `terminal_head`
       - 推理时再用 `inference_alpha` 把两头的 logit 混成一个最终分数
       - 这是对文献里“分解验证子任务”的一个低成本近似
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

                self._proj_alias_name = ""
                if cfg.architecture == "linear":
                    self.net = nn.Sequential(
                        self.dropout,
                        nn.Linear(int(cfg.hidden_size), 1),
                    )
                    self._proj_alias_name = "net.1"
                elif cfg.architecture == "mlp":
                    # 当同源 learnability 的目标提高到 90%+ 时，单线性头可能不够。
                    # 这里提供一个受控的两层 MLP：容量更强，但仍远小于 backbone。
                    self.net = nn.Sequential(
                        self.dropout,
                        nn.Linear(int(cfg.hidden_size), int(cfg.mlp_hidden_size)),
                        activation_layer,
                        nn.Dropout(float(cfg.dropout_prob)),
                        nn.Linear(int(cfg.mlp_hidden_size), 1),
                    )
                    self._proj_alias_name = "net.4"
                elif cfg.architecture == "gated_mlp":
                    # `gated_mlp` tries to capture two regimes inside the same
                    # frozen-feature scorer:
                    # 1. one expert can stay closer to local margin ranking
                    # 2. another expert can specialize more on harder/global
                    #    calibration patterns such as terminal completion
                    #
                    # The gate is still feature-only, so this remains usable at
                    # inference time without benchmark-only metadata.
                    self.shared = nn.Sequential(
                        self.dropout,
                        nn.Linear(int(cfg.hidden_size), int(cfg.mlp_hidden_size)),
                        activation_layer,
                        nn.Dropout(float(cfg.dropout_prob)),
                    )
                    self.local_expert = nn.Linear(int(cfg.mlp_hidden_size), int(cfg.mlp_hidden_size))
                    self.global_expert = nn.Linear(int(cfg.mlp_hidden_size), int(cfg.mlp_hidden_size))
                    self.gate = nn.Sequential(
                        nn.Linear(int(cfg.mlp_hidden_size), int(cfg.mlp_hidden_size)),
                        activation_layer,
                        nn.Linear(int(cfg.mlp_hidden_size), 1),
                    )
                    self.final_proj = nn.Linear(int(cfg.mlp_hidden_size), 1)
                    self._proj_alias_name = "final_proj"
                else:
                    # `dual_head` is different from `gated_mlp`:
                    # it keeps two explicit scalar heads instead of blending two
                    # hidden experts into one scalar.  The training loop can then
                    # route local-vs-terminal supervision to different heads.
                    #
                    # 中文：
                    # `gated_mlp` 仍然只产出一个“隐式混合”的标量；`dual_head`
                    # 则明确保留两个头，便于训练侧按 pair 语义分别督导。
                    self.shared = nn.Sequential(
                        self.dropout,
                        nn.Linear(int(cfg.hidden_size), int(cfg.mlp_hidden_size)),
                        activation_layer,
                        nn.Dropout(float(cfg.dropout_prob)),
                    )
                    self.local_proj = nn.Linear(int(cfg.mlp_hidden_size), 1)
                    self.terminal_proj = nn.Linear(int(cfg.mlp_hidden_size), 1)
                    self._proj_alias_name = "local_proj"
                self._init_linear_layers(std=float(cfg.init_std))

            @property
            def proj(self):
                """Backward-compatible alias for the final linear projection.

                English
                -------
                Older Phase B/C utilities, notebooks, and tests accessed
                `value_head.proj` directly when the head was a single linear
                layer.  The newer implementation stores layers under `self.net`,
                but keeping a read-only alias here avoids silent API breakage in
                warm-start/debug paths.

                中文
                ----
                较早的 Phase B/C 工具、notebook 和测试会直接访问
                `value_head.proj`。现在实现改成了 `self.net`，但这里保留一个只读
                alias，避免这类调用链在重构后悄悄断掉。
                """
                # RISK WARNING:
                # Removing this compatibility alias broke the existing Phase C
                # checkpoint-initialization test and can also break downstream
                # inspection code that expects `.proj.weight/.bias`.
                if self._proj_alias_name == "final_proj":
                    return self.final_proj
                if self._proj_alias_name == "local_proj":
                    return self.local_proj
                if self._proj_alias_name == "net.1":
                    return self.net[1]
                if self._proj_alias_name == "net.4":
                    return self.net[4]
                raise AttributeError("Projection alias is not configured")

            def _init_linear_layers(self, *, std: float) -> None:
                for module in self.modules():
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
                if self.config.architecture in {"linear", "mlp"}:
                    logits = self.net(features).squeeze(-1)
                    scores = torch.sigmoid(logits)
                    return {"logits": logits, "scores": scores}
                if self.config.architecture == "gated_mlp":
                    hidden = self.shared(features)
                    local_hidden = self.local_expert(hidden)
                    global_hidden = self.global_expert(hidden)
                    gate_value = torch.sigmoid(self.gate(hidden))
                    mixed_hidden = gate_value * local_hidden + (1.0 - gate_value) * global_hidden
                    logits = self.final_proj(mixed_hidden).squeeze(-1)
                    scores = torch.sigmoid(logits)
                    return {"logits": logits, "scores": scores}
                hidden = self.shared(features)
                local_logits = self.local_proj(hidden).squeeze(-1)
                terminal_logits = self.terminal_proj(hidden).squeeze(-1)
                alpha = float(self.config.inference_alpha)
                logits = alpha * local_logits + (1.0 - alpha) * terminal_logits
                return {
                    "logits": logits,
                    "scores": torch.sigmoid(logits),
                    "local_logits": local_logits,
                    "local_scores": torch.sigmoid(local_logits),
                    "terminal_logits": terminal_logits,
                    "terminal_scores": torch.sigmoid(terminal_logits),
                }

        return _Impl(config)


def ensure_tokenizer_has_pad_token(tokenizer: Any) -> bool:
    """Ensure one tokenizer exposes a usable pad token for batched scoring.

    Returns
    -------
    bool
        `True` when this helper had to synthesize a brand-new pad token and
        therefore the caller must consider resizing model embeddings.

    English
    -------
    Many research checkpoints omit `pad_token`, which is fine for single-sample
    generation but unsafe for batched encoding/scoring.  If `eos_token` exists,
    reusing it as padding is the least invasive option.  Only when both are
    absent do we grow the tokenizer vocabulary.

    中文
    ----
    很多研究 checkpoint 没有显式 `pad_token`，单样本生成通常没事，但批量编码/
    打分就不安全了。若存在 `eos_token`，优先直接复用；只有两者都不存在时，
    才真正向 tokenizer 词表里新增一个 pad token。
    """
    if tokenizer.pad_token_id is not None:
        return False
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return False
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return True


def maybe_resize_embeddings_for_tokenizer(*, backbone: Any, tokenizer: Any) -> bool:
    """Resize model embeddings when tokenizer growth introduced new token ids.

    English
    -------
    This is a small but important safety valve.  If we synthesized a new pad
    token, the tokenizer length grows by one.  Without resizing the backbone
    embedding table, later batched forward passes may feed an out-of-range token
    id and fail only at runtime.

    中文
    ----
    这是一个很小但很关键的安全阀。如果我们真的给 tokenizer 新增了 pad token，
    它的词表长度就会增长。若不同步扩展 backbone 的 embedding table，后续批量
    前向时就可能喂入越界 token id，只在运行时才爆炸。
    """
    get_input_embeddings = getattr(backbone, "get_input_embeddings", None)
    if not callable(get_input_embeddings):
        return False
    embeddings = get_input_embeddings()
    if embeddings is None or not hasattr(embeddings, "num_embeddings"):
        return False
    current_vocab_size = int(getattr(embeddings, "num_embeddings"))
    target_vocab_size = int(len(tokenizer))
    if target_vocab_size <= current_vocab_size:
        return False
    resize_token_embeddings = getattr(backbone, "resize_token_embeddings", None)
    if not callable(resize_token_embeddings):
        raise RuntimeError(
            "Tokenizer vocabulary grew beyond the model embedding table, "
            "but the backbone does not expose `resize_token_embeddings`."
        )
    resize_token_embeddings(target_vocab_size)
    return True


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
    # 通过 `attention_mask` 定位最后一个有效 token，必须恢复“最后一个被 attend
    # 的位置”，而不能把它偷换成 `sum(mask) - 1`，否则 left padding 会出错。
    # We locate the last valid token through `attention_mask`, which works for
    # both left-padding and right-padding as long as we recover the last
    # attended position instead of assuming `sum(mask) - 1`.
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
        # device_map="auto" 跨多卡时，最后一层 hidden states 可能在 cuda:1，
        # 而 attention_mask 仍在 cuda:0（输入设备）。
        # pool_last_token 内部的 torch.where 跨设备操作会静默产生全零结果。
        # 这里显式对齐，确保两者在同一设备上。
        # When device_map="auto" splits across GPUs, outputs.hidden_states[-1] may
        # land on cuda:1 while attention_mask is still on cuda:0 (the input device).
        # A cross-device torch.where silently produces all-zeros.  Align explicitly.
        attn_mask = inputs["attention_mask"].to(last_hidden.device)
        pooled = pool_last_token(last_hidden, attn_mask, torch_module=torch_module)
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

    # `sum(mask) - 1` 只对 right padding 成立；这里改成找“mask > 0 的最大位置”，
    # 才能同时兼容 left/right padding。若整行都被 mask，则回退到 index 0。
    # `sum(mask) - 1` only works for right padding.  Here we compute the maximum
    # position whose mask value is positive, which is correct for both left and
    # right padding.  Fully-masked rows fall back to index 0.
    seq_positions = torch_module.arange(
        attention_mask.shape[1],
        device=hidden_states.device,
    ).unsqueeze(0)
    masked_positions = torch_module.where(
        attention_mask > 0,
        seq_positions,
        torch_module.zeros_like(seq_positions),
    )
    lengths = masked_positions.max(dim=1).values
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
