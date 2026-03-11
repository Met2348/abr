#!/usr/bin/env python3
# Changelog (prepend-only, newest first):
# 2026-03-12 11:10: [FIX] Support Math-PRM-7B backbone (Qwen2RMConfig).
#   AutoModelForCausalLM.from_pretrained fails for process_reward_model backbones.
#   Now uses resolve_backbone_loader_family() to detect PRM family and falls back
#   to AutoModel.from_pretrained, which successfully loads Qwen2ForProcessRewardModel.
#   Math-PRM-7B hidden_states[-1] extraction is identical to CausalLM — no other changes.
#   Math-PRM-7B 用 Qwen2RMConfig，不能用 AutoModelForCausalLM 加载，改用 AutoModel。
"""Train a Phase E value head while adapting the backbone with LoRA.

English
-------
This is the minimal online-encoding counterpart to the frozen-feature Phase E
trainer.

Why a separate script exists:
1. the existing `phase_e_train_value.py` is deliberately built around frozen
   feature caching,
2. once LoRA updates the backbone, cached features become stale immediately,
3. so the cleanest comparison is a parallel trainer that keeps the same pair
   contract, loss contract, and run manifest style, while replacing only the
   encoding/training path.

中文
----
这是 Phase E 冻结特征 trainer 的一个最小在线编码版本，用于 LoRA 解冻
backbone。

之所以单独做一个脚本，而不是强行塞进原 trainer，是因为：
1. 现有 `phase_e_train_value.py` 的核心就是冻结特征缓存，
2. 只要 LoRA 开始更新 backbone，缓存特征就立刻过期，
3. 因此最干净的对照方式，是保留同样的 pair 合同、loss 合同和 manifest
   结构，只替换编码/训练路径。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_b.value_head import (  # noqa: E402
    SigmoidValueHead,
    ValueHeadConfig,
    ensure_tokenizer_has_pad_token,
    infer_backbone_hidden_size,
    load_value_head_checkpoint,
    maybe_resize_embeddings_for_tokenizer,
    pool_last_token,
    save_value_head_checkpoint,
    write_value_head_config_json,
)
from ours.phase_d.external_pairs import ExternalPairRecord, load_external_pair_jsonl  # noqa: E402
from ours.phase_e.recipe_safety import (  # noqa: E402
    assess_phase_e_recipe_risk,
    enforce_phase_e_recipe_risk,
    render_phase_e_recipe_risk_console_report,
)
from ours.phase_e.runtime import import_runtime_deps, resolve_dtype, set_seed, stable_hash_order  # noqa: E402
from ours.phase_e.training import (  # noqa: E402
    compute_external_pair_metrics,
    compute_pair_objective,
    compute_pair_route_weights,
    compute_pair_truncation_diagnostics,
    compute_pair_weights,
    select_metric_value,
    validate_pair_truncation_diagnostics,
)


@dataclass(slots=True)
class PhaseELoraTrainConfig:
    objective_mode: str
    learning_rate: float
    weight_decay: float
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    max_grad_norm: float
    max_length: int
    lambda_ranking: float
    lambda_bce: float
    lambda_terminal_bce: float
    ranking_margin: float
    ranking_target_space: str
    pair_weight_mode: str
    source_balance: str
    permutation_mode: str
    anti_saturation_weight: float
    anti_saturation_logit_threshold: float
    reward_centering_weight: float
    checkpoint_selection_metric: str
    logging_steps: int
    seed: int
    dtype: str
    require_cuda: bool
    strict_determinism: bool
    init_value_head_path: str | None
    head_architecture: str
    head_dropout_prob: float
    head_init_std: float
    head_mlp_hidden_size: int
    head_activation: str
    head_inference_alpha: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    lora_top_k_layers: int | None
    gradient_checkpointing: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PairTrainingRows:
    pairs: list[ExternalPairRecord]
    pair_weights: list[float]
    local_route_weights: list[float]
    terminal_route_weights: list[float]


class _PairTextBatchCollator:
    """Tokenize chosen/rejected texts together in one forward-friendly batch.

    English
    -------
    We concatenate chosen and rejected texts into one tokenizer call so one
    transformer forward can produce both pooled feature sets.

    中文
    ----
    chosen 和 rejected 合并到一次 tokenizer / backbone 前向里，目的是让一轮
    transformer forward 同时产出两边的 pooled features，减少重复开销。
    """

    def __init__(self, *, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, pairs: list[ExternalPairRecord]) -> dict[str, Any]:
        texts = [pair.chosen_input_text() for pair in pairs] + [pair.rejected_input_text() for pair in pairs]
        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.max_length),
        )
        return {"tokenized": tokenized, "num_pairs": len(pairs)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one Phase E value head with LoRA-adapted backbone on external pairs."
    )
    parser.add_argument("--train-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--eval-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-name", default="phase_e_value_lora")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_runs"),
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--objective-mode", choices=["ranking_only", "pair_bce_only", "joint"], default="joint")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=int, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lambda-ranking", type=float, default=1.0)
    parser.add_argument("--lambda-bce", type=float, default=1.0)
    parser.add_argument("--terminal-bce-lambda", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=0.02)
    parser.add_argument("--ranking-target-space", choices=["score", "logit"], default="logit")
    parser.add_argument(
        "--pair-weight-mode",
        choices=[
            "none",
            "confidence",
            "semantic",
            "confidence_semantic",
            "verdict_balance",
            "confidence_verdict_balance",
            "group_balance",
            "confidence_group_balance",
        ],
        default="confidence_semantic",
    )
    parser.add_argument("--source-balance", choices=["none", "uniform"], default="none")
    parser.add_argument("--permutation-mode", choices=["random", "stable_hash"], default="stable_hash")
    parser.add_argument("--anti-saturation-weight", type=float, default=5e-4)
    parser.add_argument("--anti-saturation-logit-threshold", type=float, default=3.5)
    parser.add_argument("--reward-centering-weight", type=float, default=0.0)
    parser.add_argument("--checkpoint-selection-metric", choices=["pair_acc", "auc", "ranking_score", "pair_loss"], default="pair_acc")
    parser.add_argument(
        "--recipe-risk-policy",
        choices=["off", "warn", "error"],
        default="error",
        help=(
            "Preflight recipe risk policy. `error` is recommended for production-like "
            "Phase E LoRA runs; use `warn` only for controlled diagnostic reproductions."
        ),
    )
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--strict-determinism",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--init-value-head-path", type=Path, default=None)
    parser.add_argument("--head-architecture", choices=["linear", "mlp", "gated_mlp", "dual_head"], default="mlp")
    parser.add_argument("--head-dropout-prob", type=float, default=0.05)
    parser.add_argument("--head-init-std", type=float, default=0.02)
    parser.add_argument("--head-mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--head-activation", choices=["gelu", "relu", "tanh"], default="gelu")
    parser.add_argument("--head-inference-alpha", type=float, default=0.5)
    parser.add_argument("--truncation-diagnostics-batch-size", type=int, default=64)
    parser.add_argument("--max-truncation-over-limit-fraction", type=float, default=0.10)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,v_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument(
        "--lora-top-k-layers",
        type=int,
        default=4,
        help="Apply LoRA only to the last K transformer layers. Use 0 to target all layers.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.train_pairs_jsonl.exists():
        raise FileNotFoundError(f"--train-pairs-jsonl not found: {args.train_pairs_jsonl}")
    if not args.eval_pairs_jsonl.exists():
        raise FileNotFoundError(f"--eval-pairs-jsonl not found: {args.eval_pairs_jsonl}")
    if args.init_value_head_path is not None and not Path(args.init_value_head_path).exists():
        raise FileNotFoundError(f"--init-value-head-path not found: {args.init_value_head_path}")
    if args.max_train_samples is not None and int(args.max_train_samples) <= 0:
        raise ValueError("--max-train-samples must be > 0")
    if args.max_eval_samples is not None and int(args.max_eval_samples) <= 0:
        raise ValueError("--max-eval-samples must be > 0")
    if float(args.learning_rate) <= 0.0:
        raise ValueError("--learning-rate must be > 0")
    if float(args.weight_decay) < 0.0:
        raise ValueError("--weight-decay must be >= 0")
    if int(args.num_train_epochs) <= 0:
        raise ValueError("--num-train-epochs must be > 0")
    if int(args.per_device_train_batch_size) <= 0 or int(args.per_device_eval_batch_size) <= 0:
        raise ValueError("Per-device batch sizes must be > 0")
    if int(args.gradient_accumulation_steps) <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0")
    if not (0.0 <= float(args.warmup_ratio) < 1.0):
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if float(args.max_grad_norm) <= 0.0:
        raise ValueError("--max-grad-norm must be > 0")
    if int(args.max_length) <= 8:
        raise ValueError("--max-length must be > 8")
    if float(args.lambda_ranking) < 0.0 or float(args.lambda_bce) < 0.0:
        raise ValueError("--lambda-ranking and --lambda-bce must be >= 0")
    if float(args.ranking_margin) < 0.0:
        raise ValueError("--ranking-margin must be >= 0")
    if float(args.anti_saturation_weight) < 0.0:
        raise ValueError("--anti-saturation-weight must be >= 0")
    if float(args.anti_saturation_logit_threshold) <= 0.0:
        raise ValueError("--anti-saturation-logit-threshold must be > 0")
    if float(args.reward_centering_weight) < 0.0:
        raise ValueError("--reward-centering-weight must be >= 0")
    if not (0.0 <= float(args.head_dropout_prob) < 1.0):
        raise ValueError("--head-dropout-prob must be in [0, 1)")
    if float(args.head_init_std) <= 0.0:
        raise ValueError("--head-init-std must be > 0")
    if int(args.head_mlp_hidden_size) <= 0:
        raise ValueError("--head-mlp-hidden-size must be > 0")
    if int(args.logging_steps) <= 0:
        raise ValueError("--logging-steps must be > 0")
    if int(args.truncation_diagnostics_batch_size) <= 0:
        raise ValueError("--truncation-diagnostics-batch-size must be > 0")
    if not (0.0 <= float(args.max_truncation_over_limit_fraction) <= 1.0):
        raise ValueError("--max-truncation-over-limit-fraction must be in [0, 1]")
    if int(args.lora_rank) <= 0:
        raise ValueError("--lora-rank must be > 0")
    if int(args.lora_alpha) <= 0:
        raise ValueError("--lora-alpha must be > 0")
    if not (0.0 <= float(args.lora_dropout) < 1.0):
        raise ValueError("--lora-dropout must be in [0, 1)")
    if int(args.lora_top_k_layers) < 0:
        raise ValueError("--lora-top-k-layers must be >= 0")
    return args


def _resolve_total_optimizer_steps(*, num_pairs: int, batch_size: int, grad_accum_steps: int, num_epochs: int) -> int:
    steps_per_epoch = max(1, math.ceil(int(num_pairs) / int(batch_size)))
    return max(1, math.ceil((steps_per_epoch * int(num_epochs)) / int(grad_accum_steps)))


def _linear_warmup_decay(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-8, float(step + 1) / float(warmup_steps))
    remaining_steps = max(1, total_steps - warmup_steps)
    progress = float(step - warmup_steps) / float(remaining_steps)
    return max(0.0, 1.0 - progress)


def _write_eval_pair_scores(
    *,
    path: Path,
    eval_pairs: list[ExternalPairRecord],
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for pair, chosen_score, rejected_score in zip(eval_pairs, chosen_scores, rejected_scores, strict=True):
            handle.write(
                json.dumps(
                    {
                        "pair_id": pair.pair_id,
                        "source_tag": pair.source_tag,
                        "domain_tag": pair.domain_tag,
                        "pair_semantics": str((pair.metadata or {}).get("pair_semantics", "unspecified")),
                        "chosen_score": float(chosen_score),
                        "rejected_score": float(rejected_score),
                        "margin": float(chosen_score - rejected_score),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _initialize_value_head_from_checkpoint(
    *,
    value_head: Any,
    current_config: ValueHeadConfig,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return {"mode": "fresh_init", "checkpoint_path": None}
    loaded_head, loaded_config, extra_state = load_value_head_checkpoint(checkpoint_path)
    if loaded_config.to_dict() != current_config.to_dict():
        raise ValueError(
            "Init value-head config mismatch: "
            f"loaded={loaded_config.to_dict()} current={current_config.to_dict()}"
        )
    value_head.load_state_dict(loaded_head.state_dict())
    return {
        "mode": "loaded_checkpoint",
        "checkpoint_path": str(checkpoint_path),
        "extra_state": extra_state,
    }


def _build_pair_training_rows(*, pairs: list[ExternalPairRecord], pair_weight_mode: str) -> PairTrainingRows:
    verdict_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    if pair_weight_mode in {"verdict_balance", "confidence_verdict_balance"}:
        for pair in pairs:
            verdict = str((pair.metadata or {}).get("chosen_verdict", "")).strip().lower()
            if verdict:
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    if pair_weight_mode in {"group_balance", "confidence_group_balance"}:
        for pair in pairs:
            metadata = dict(pair.metadata or {})
            mixed_label = str(metadata.get("artifact_mix_source_label", "")).strip()
            if mixed_label:
                group_label = mixed_label
            else:
                group_label = str(metadata.get("pair_semantics", "")).strip() or str(pair.source_tag)
            group_counts[group_label] = group_counts.get(group_label, 0) + 1
    pair_weights = compute_pair_weights(
        pairs=pairs,
        pair_weight_mode=pair_weight_mode,
        verdict_counts=verdict_counts,
        group_counts=group_counts,
    )
    local_route_weights, terminal_route_weights = compute_pair_route_weights(pairs=pairs)
    return PairTrainingRows(
        pairs=list(pairs),
        pair_weights=list(pair_weights),
        local_route_weights=list(local_route_weights),
        terminal_route_weights=list(terminal_route_weights),
    )


def _build_pair_permutation(
    *,
    pair_rows: PairTrainingRows,
    permutation_mode: str,
    source_balance: str,
    torch_module: Any,
) -> list[int]:
    num_pairs = len(pair_rows.pairs)
    if num_pairs <= 1:
        return list(range(num_pairs))
    if permutation_mode not in {"random", "stable_hash"}:
        raise ValueError(f"Unsupported permutation_mode: {permutation_mode!r}")
    if source_balance not in {"none", "uniform"}:
        raise ValueError(f"Unsupported source_balance: {source_balance!r}")
    pair_ids = [pair.pair_id for pair in pair_rows.pairs]
    base_indices = list(range(num_pairs))

    def _order(values: list[int]) -> list[int]:
        if permutation_mode == "random":
            order = torch_module.randperm(len(values)).tolist()
            return [values[pos] for pos in order]
        ordered_ids = [pair_ids[idx] for idx in values]
        return stable_hash_order(values, ids=ordered_ids)

    if source_balance == "none":
        return _order(base_indices)
    buckets: dict[str, list[int]] = {}
    for idx, pair in enumerate(pair_rows.pairs):
        buckets.setdefault(str(pair.source_tag), []).append(int(idx))
    for key in list(buckets.keys()):
        buckets[key] = _order(buckets[key])
    ordered: list[int] = []
    pointers = {key: 0 for key in buckets}
    key_order = sorted(buckets)
    while True:
        active = [key for key in key_order if pointers[key] < len(buckets[key])]
        if not active:
            break
        if permutation_mode == "random" and len(active) > 1:
            shuffle_order = torch_module.randperm(len(active)).tolist()
            active = [active[pos] for pos in shuffle_order]
        for key in active:
            ordered.append(buckets[key][pointers[key]])
            pointers[key] += 1
    return ordered


def _forward_value_pair_batch(
    *,
    model: Any,
    value_head: Any,
    tokenized_batch: dict[str, Any],
    num_pairs: int,
    torch_module: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_device = next(model.parameters()).device
    head_device = next(value_head.parameters()).device
    head_dtype = next(value_head.parameters()).dtype
    inputs = {key: value.to(model_device) for key, value in tokenized_batch.items()}
    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    last_hidden = outputs.hidden_states[-1]
    pooled = pool_last_token(last_hidden, inputs["attention_mask"], torch_module=torch_module)
    pooled = pooled.to(device=head_device, dtype=head_dtype)
    chosen_features = pooled[: int(num_pairs)]
    rejected_features = pooled[int(num_pairs) :]
    chosen_out = value_head(chosen_features)
    rejected_out = value_head(rejected_features)
    return chosen_out, rejected_out


def _evaluate_pairs(
    *,
    model: Any,
    value_head: Any,
    tokenizer: Any,
    pair_rows: PairTrainingRows,
    batch_size: int,
    objective_mode: str,
    ranking_target_space: str,
    ranking_margin: float,
    lambda_ranking: float,
    lambda_bce: float,
    lambda_terminal_bce: float,
    anti_saturation_weight: float,
    anti_saturation_logit_threshold: float,
    reward_centering_weight: float,
    max_length: int,
    torch_module: Any,
) -> tuple[dict[str, Any], list[float], list[float]]:
    collator = _PairTextBatchCollator(tokenizer=tokenizer, max_length=int(max_length))
    chosen_scores: list[float] = []
    rejected_scores: list[float] = []
    losses: list[float] = []
    model.eval()
    value_head.eval()
    with torch_module.no_grad():
        for start in range(0, len(pair_rows.pairs), int(batch_size)):
            batch_pairs = pair_rows.pairs[start : start + int(batch_size)]
            batch_payload = collator(batch_pairs)
            chosen_out, rejected_out = _forward_value_pair_batch(
                model=model,
                value_head=value_head,
                tokenized_batch=batch_payload["tokenized"],
                num_pairs=batch_payload["num_pairs"],
                torch_module=torch_module,
            )
            chosen_scores.extend(float(v) for v in chosen_out["scores"].detach().cpu().tolist())
            rejected_scores.extend(float(v) for v in rejected_out["scores"].detach().cpu().tolist())
            pair_weights = torch_module.tensor(
                pair_rows.pair_weights[start : start + len(batch_pairs)],
                device=next(value_head.parameters()).device,
                dtype=next(value_head.parameters()).dtype,
            )
            local_route_weights = torch_module.tensor(
                pair_rows.local_route_weights[start : start + len(batch_pairs)],
                device=pair_weights.device,
                dtype=pair_weights.dtype,
            )
            terminal_route_weights = torch_module.tensor(
                pair_rows.terminal_route_weights[start : start + len(batch_pairs)],
                device=pair_weights.device,
                dtype=pair_weights.dtype,
            )
            batch_loss = compute_pair_objective(
                chosen_logits=chosen_out["logits"],
                rejected_logits=rejected_out["logits"],
                chosen_scores=chosen_out["scores"],
                rejected_scores=rejected_out["scores"],
                pair_weights=pair_weights,
                objective_mode=objective_mode,
                ranking_target_space=ranking_target_space,
                ranking_margin=ranking_margin,
                lambda_ranking=lambda_ranking,
                lambda_bce=lambda_bce,
                anti_saturation_weight=anti_saturation_weight,
                anti_saturation_logit_threshold=anti_saturation_logit_threshold,
                reward_centering_weight=reward_centering_weight,
                chosen_local_logits=chosen_out.get("local_logits"),
                rejected_local_logits=rejected_out.get("local_logits"),
                chosen_local_scores=chosen_out.get("local_scores"),
                rejected_local_scores=rejected_out.get("local_scores"),
                local_pair_weights=pair_weights * local_route_weights,
                chosen_terminal_logits=chosen_out.get("terminal_logits"),
                rejected_terminal_logits=rejected_out.get("terminal_logits"),
                chosen_terminal_scores=chosen_out.get("terminal_scores"),
                rejected_terminal_scores=rejected_out.get("terminal_scores"),
                terminal_pair_weights=pair_weights * terminal_route_weights,
                lambda_terminal_bce=lambda_terminal_bce,
                torch_module=torch_module,
            )
            losses.append(float(batch_loss.detach().cpu().item()))
    metrics = compute_external_pair_metrics(
        chosen_scores=chosen_scores,
        rejected_scores=rejected_scores,
        source_tags=[pair.source_tag for pair in pair_rows.pairs],
    )
    metrics["pair_loss"] = float(sum(losses) / len(losses)) if losses else 0.0
    metrics["ranking_score"] = float((metrics["pair_accuracy"] + metrics["auc"]) / 2.0)
    return metrics, chosen_scores, rejected_scores


def _attach_lora(
    *,
    model: Any,
    num_hidden_layers: int,
    target_modules: list[str],
    rank: int,
    alpha: int,
    dropout: float,
    top_k_layers: int | None,
) -> tuple[Any, dict[str, Any]]:
    from peft import LoraConfig, TaskType, get_peft_model

    if top_k_layers is not None and int(top_k_layers) > 0:
        start_idx = max(0, int(num_hidden_layers) - int(top_k_layers))
        layers_to_transform = list(range(start_idx, int(num_hidden_layers)))
    else:
        layers_to_transform = None
    # Qwen2ForProcessRewardModel lacks prepare_inputs_for_generation, so TaskType.CAUSAL_LM
    # wrapping fails on older peft versions. FEATURE_EXTRACTION creates a plain PeftModel
    # without any generation-specific attributes — safe for all backbone types.
    # Qwen2ForProcessRewardModel 缺少 prepare_inputs_for_generation，旧版 peft 用
    # CAUSAL_LM 会报错；FEATURE_EXTRACTION 不需要该方法，对所有 backbone 类型均安全。
    _resolved_task_type = TaskType.FEATURE_EXTRACTION
    # layers_pattern is only valid when layers_to_transform is also specified (peft constraint).
    _lora_cfg_kwargs: dict = dict(
        task_type=_resolved_task_type,
        inference_mode=False,
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        bias="none",
    )
    if layers_to_transform is not None:
        _lora_cfg_kwargs["layers_to_transform"] = layers_to_transform
        _lora_cfg_kwargs["layers_pattern"] = "layers"
    peft_cfg = LoraConfig(**_lora_cfg_kwargs)
    peft_model = get_peft_model(model, peft_cfg)
    return peft_model, {
        "rank": int(rank),
        "alpha": int(alpha),
        "dropout": float(dropout),
        "target_modules": list(target_modules),
        "layers_to_transform": layers_to_transform,
        "layers_pattern": ("layers" if layers_to_transform is not None else None),
    }


def _trainable_parameter_summary(model: Any) -> dict[str, Any]:
    total = 0
    trainable = 0
    for param in model.parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_fraction": float(trainable / total) if total > 0 else 0.0,
    }


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    eval_pairs = dict(summary["eval_pairs"])
    lines = [
        "# Phase E LoRA Value Run",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- train_pairs: `{summary['train_pairs']}`",
        f"- eval_pairs_n: `{summary['eval_pairs_n']}`",
        f"- best_epoch: `{summary['best_epoch']}`",
        f"- selection_metric: `{summary['selection_metric']}`",
        f"- selection_value: `{summary['selection_value']:.6f}`",
        f"- head_architecture: `{summary['head_architecture']}`",
        f"- lora_top_k_layers: `{summary['lora_top_k_layers']}`",
        f"- lora_target_modules: `{summary['lora_target_modules']}`",
        "",
        "## Eval Pairs",
        f"- pair_accuracy: `{eval_pairs['pair_accuracy']:.6f}`",
        f"- auc: `{eval_pairs['auc']:.6f}`",
        f"- mean_margin: `{eval_pairs['mean_margin']:.6f}`",
        f"- pair_loss: `{eval_pairs['pair_loss']:.6f}`",
        f"- ranking_score: `{eval_pairs['ranking_score']:.6f}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_pairs, train_pair_stats = load_external_pair_jsonl(
        Path(args.train_pairs_jsonl),
        max_samples=args.max_train_samples,
    )
    eval_pairs, eval_pair_stats = load_external_pair_jsonl(
        Path(args.eval_pairs_jsonl),
        max_samples=args.max_eval_samples,
    )
    if not train_pairs:
        raise RuntimeError("No training pairs were loaded")
    if not eval_pairs:
        raise RuntimeError("No eval pairs were loaded")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if bool(args.strict_determinism):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch, AutoModelForCausalLM, AutoTokenizer = import_runtime_deps()
    if bool(args.require_cuda) and not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA is required by --require-cuda but no GPU is visible")
    set_seed(int(args.seed), torch, strict_determinism=bool(args.strict_determinism))

    train_config = PhaseELoraTrainConfig(
        objective_mode=str(args.objective_mode),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        num_train_epochs=int(args.num_train_epochs),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        warmup_ratio=float(args.warmup_ratio),
        max_grad_norm=float(args.max_grad_norm),
        max_length=int(args.max_length),
        lambda_ranking=float(args.lambda_ranking),
        lambda_bce=float(args.lambda_bce),
        lambda_terminal_bce=float(args.terminal_bce_lambda),
        ranking_margin=float(args.ranking_margin),
        ranking_target_space=str(args.ranking_target_space),
        pair_weight_mode=str(args.pair_weight_mode),
        source_balance=str(args.source_balance),
        permutation_mode=str(args.permutation_mode),
        anti_saturation_weight=float(args.anti_saturation_weight),
        anti_saturation_logit_threshold=float(args.anti_saturation_logit_threshold),
        reward_centering_weight=float(args.reward_centering_weight),
        checkpoint_selection_metric=str(args.checkpoint_selection_metric),
        logging_steps=int(args.logging_steps),
        seed=int(args.seed),
        dtype=str(args.dtype),
        require_cuda=bool(args.require_cuda),
        strict_determinism=bool(args.strict_determinism),
        init_value_head_path=(str(args.init_value_head_path) if args.init_value_head_path is not None else None),
        head_architecture=str(args.head_architecture),
        head_dropout_prob=float(args.head_dropout_prob),
        head_init_std=float(args.head_init_std),
        head_mlp_hidden_size=int(args.head_mlp_hidden_size),
        head_activation=str(args.head_activation),
        head_inference_alpha=float(args.head_inference_alpha),
        lora_rank=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        lora_target_modules=[m.strip() for m in str(args.lora_target_modules).split(",") if m.strip()],
        lora_top_k_layers=(int(args.lora_top_k_layers) if int(args.lora_top_k_layers) > 0 else None),
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )

    run_dir = Path(args.output_root) / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    train_metrics_path = run_dir / "train_metrics.json"
    eval_metrics_path = run_dir / "eval_metrics.json"
    summary_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    best_ckpt_path = run_dir / "best_value_head.pt"
    final_ckpt_path = run_dir / "final_value_head.pt"
    best_adapter_dir = run_dir / "best_adapter"
    final_adapter_dir = run_dir / "final_adapter"
    config_json_path = run_dir / "value_head_config.json"
    eval_pair_scores_path = run_dir / "eval_pair_scores.jsonl"
    train_curve_path = run_dir / "train_curve.jsonl"
    recipe_risk_path = run_dir / "recipe_risk.json"

    print("=" * 88)
    print("Phase E: Train Value Head with LoRA")
    print("=" * 88)
    print(f"train_pairs_jsonl  : {args.train_pairs_jsonl}")
    print(f"eval_pairs_jsonl   : {args.eval_pairs_jsonl}")
    print(f"run_dir            : {run_dir}")
    print(f"model_path         : {args.model_path}")
    print(f"objective_mode     : {args.objective_mode}")
    print(f"train_pairs        : {len(train_pairs)}")
    print(f"eval_pairs         : {len(eval_pairs)}")
    print(f"batch_train        : {args.per_device_train_batch_size}")
    print(f"batch_eval         : {args.per_device_eval_batch_size}")
    print(f"grad_accum         : {args.gradient_accumulation_steps}")
    print(f"max_length         : {args.max_length}")
    print(f"pair_weight_mode   : {args.pair_weight_mode}")
    print(f"selection_metric   : {args.checkpoint_selection_metric}")
    print(f"recipe_risk_policy : {args.recipe_risk_policy}")
    print(f"head_architecture  : {args.head_architecture}")
    print(f"lora_rank          : {args.lora_rank}")
    print(f"lora_target_modules: {train_config.lora_target_modules}")
    print(f"lora_top_k_layers  : {train_config.lora_top_k_layers}")
    print("=" * 88)

    recipe_risk_report = assess_phase_e_recipe_risk(
        train_pair_summary=train_pair_stats,
        train_config=train_config.to_dict(),
    )
    for line in render_phase_e_recipe_risk_console_report(recipe_risk_report):
        print(line, flush=True)
    enforce_phase_e_recipe_risk(
        recipe_risk_report=recipe_risk_report,
        policy=str(args.recipe_risk_policy),
    )
    recipe_risk_path.write_text(
        json.dumps(recipe_risk_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    synthesized_pad_token = ensure_tokenizer_has_pad_token(tokenizer)
    train_truncation_diagnostics = compute_pair_truncation_diagnostics(
        pairs=train_pairs,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        batch_size=int(args.truncation_diagnostics_batch_size),
    )
    eval_truncation_diagnostics = compute_pair_truncation_diagnostics(
        pairs=eval_pairs,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        batch_size=int(args.truncation_diagnostics_batch_size),
    )
    validate_pair_truncation_diagnostics(
        diagnostics=train_truncation_diagnostics,
        context_label="Phase E LoRA train split",
        max_allowed_over_limit_fraction=float(args.max_truncation_over_limit_fraction),
    )
    validate_pair_truncation_diagnostics(
        diagnostics=eval_truncation_diagnostics,
        context_label="Phase E LoRA eval split",
        max_allowed_over_limit_fraction=float(args.max_truncation_over_limit_fraction),
    )

    load_start = time.perf_counter()
    resolved_dtype = resolve_dtype(str(args.dtype), torch)
    _load_kwargs: dict = {
        "torch_dtype": resolved_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    from ours.phase_e.runtime import resolve_backbone_loader_family
    _backbone_family = resolve_backbone_loader_family(model_path=str(args.model_path), trust_remote_code=True)
    if _backbone_family == "process_reward_model":
        # Math-PRM-7B uses Qwen2RMConfig / Qwen2ForProcessRewardModel — not a CausalLM.
        # AutoModel loads it correctly via trust_remote_code; LoRA then attaches to the
        # underlying Qwen2 transformer layers the same way as for standard CausalLM.
        # Math-PRM-7B 使用自定义 Qwen2RMConfig，不能用 AutoModelForCausalLM 加载。
        # 用 AutoModel + trust_remote_code 加载，LoRA 方式与普通 CausalLM 相同。
        from transformers import AutoModel
        model = AutoModel.from_pretrained(str(args.model_path), **_load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(args.model_path), **_load_kwargs)
    if synthesized_pad_token:
        maybe_resize_embeddings_for_tokenizer(backbone=model, tokenizer=tokenizer)
    if bool(args.gradient_checkpointing) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False
    num_hidden_layers = int(getattr(getattr(model, "config", None), "num_hidden_layers", 0))
    if num_hidden_layers <= 0:
        raise ValueError("Failed to infer num_hidden_layers for LoRA routing")
    model, lora_spec = _attach_lora(
        model=model,
        num_hidden_layers=num_hidden_layers,
        target_modules=train_config.lora_target_modules,
        rank=int(args.lora_rank),
        alpha=int(args.lora_alpha),
        dropout=float(args.lora_dropout),
        top_k_layers=train_config.lora_top_k_layers,
    )
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(model_device)
    model_load_elapsed = time.perf_counter() - load_start

    hidden_size = infer_backbone_hidden_size(model)
    value_head_config = ValueHeadConfig(
        hidden_size=int(hidden_size),
        dropout_prob=float(args.head_dropout_prob),
        init_std=float(args.head_init_std),
        architecture=str(args.head_architecture),
        mlp_hidden_size=int(args.head_mlp_hidden_size),
        activation=str(args.head_activation),
        inference_alpha=float(args.head_inference_alpha),
    )
    value_head = SigmoidValueHead(value_head_config)
    value_head.to(model_device)
    init_info = _initialize_value_head_from_checkpoint(
        value_head=value_head,
        current_config=value_head_config,
        checkpoint_path=(Path(args.init_value_head_path) if args.init_value_head_path is not None else None),
    )

    print(
        "trainable_params    : "
        f"{_trainable_parameter_summary(model)['trainable_parameters']} / {_trainable_parameter_summary(model)['total_parameters']}"
    )

    write_value_head_config_json(config_json_path, value_head_config)
    train_rows = _build_pair_training_rows(pairs=train_pairs, pair_weight_mode=str(args.pair_weight_mode))
    eval_rows = _build_pair_training_rows(pairs=eval_pairs, pair_weight_mode=str(args.pair_weight_mode))

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad] + list(value_head.parameters()),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    total_optimizer_steps = _resolve_total_optimizer_steps(
        num_pairs=len(train_rows.pairs),
        batch_size=int(args.per_device_train_batch_size),
        grad_accum_steps=int(args.gradient_accumulation_steps),
        num_epochs=int(args.num_train_epochs),
    )
    warmup_steps = int(total_optimizer_steps * float(args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _linear_warmup_decay(step, warmup_steps=warmup_steps, total_steps=total_optimizer_steps),
    )

    collator = _PairTextBatchCollator(tokenizer=tokenizer, max_length=int(args.max_length))
    selection_higher_is_better = str(args.checkpoint_selection_metric) != "pair_loss"
    best_selection_value = float("-inf") if selection_higher_is_better else float("inf")
    best_eval_metrics: dict[str, Any] | None = None
    best_eval_scores: tuple[list[float], list[float]] | None = None
    best_epoch = -1
    global_step = 0
    train_curve: list[dict[str, Any]] = []

    train_start = time.perf_counter()
    for epoch_idx in range(int(args.num_train_epochs)):
        model.train()
        value_head.train()
        permutation = _build_pair_permutation(
            pair_rows=train_rows,
            permutation_mode=str(args.permutation_mode),
            source_balance=str(args.source_balance),
            torch_module=torch,
        )
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batches = 0
        optimizer_steps = 0
        for batch_idx, start in enumerate(range(0, len(permutation), int(args.per_device_train_batch_size)), start=1):
            batch_indices = permutation[start : start + int(args.per_device_train_batch_size)]
            batch_pairs = [train_rows.pairs[idx] for idx in batch_indices]
            batch_payload = collator(batch_pairs)
            chosen_out, rejected_out = _forward_value_pair_batch(
                model=model,
                value_head=value_head,
                tokenized_batch=batch_payload["tokenized"],
                num_pairs=batch_payload["num_pairs"],
                torch_module=torch,
            )
            head_device = next(value_head.parameters()).device
            head_dtype = next(value_head.parameters()).dtype
            pair_weights = torch.tensor(
                [train_rows.pair_weights[idx] for idx in batch_indices],
                device=head_device,
                dtype=head_dtype,
            )
            local_route_weights = torch.tensor(
                [train_rows.local_route_weights[idx] for idx in batch_indices],
                device=head_device,
                dtype=head_dtype,
            )
            terminal_route_weights = torch.tensor(
                [train_rows.terminal_route_weights[idx] for idx in batch_indices],
                device=head_device,
                dtype=head_dtype,
            )
            loss = compute_pair_objective(
                chosen_logits=chosen_out["logits"],
                rejected_logits=rejected_out["logits"],
                chosen_scores=chosen_out["scores"],
                rejected_scores=rejected_out["scores"],
                pair_weights=pair_weights,
                objective_mode=str(args.objective_mode),
                ranking_target_space=str(args.ranking_target_space),
                ranking_margin=float(args.ranking_margin),
                lambda_ranking=float(args.lambda_ranking),
                lambda_bce=float(args.lambda_bce),
                anti_saturation_weight=float(args.anti_saturation_weight),
                anti_saturation_logit_threshold=float(args.anti_saturation_logit_threshold),
                reward_centering_weight=float(args.reward_centering_weight),
                chosen_local_logits=chosen_out.get("local_logits"),
                rejected_local_logits=rejected_out.get("local_logits"),
                chosen_local_scores=chosen_out.get("local_scores"),
                rejected_local_scores=rejected_out.get("local_scores"),
                local_pair_weights=pair_weights * local_route_weights,
                chosen_terminal_logits=chosen_out.get("terminal_logits"),
                rejected_terminal_logits=rejected_out.get("terminal_logits"),
                chosen_terminal_scores=chosen_out.get("terminal_scores"),
                rejected_terminal_scores=rejected_out.get("terminal_scores"),
                terminal_pair_weights=pair_weights * terminal_route_weights,
                lambda_terminal_bce=float(args.terminal_bce_lambda),
                torch_module=torch,
            )
            loss = loss / int(args.gradient_accumulation_steps)
            loss.backward()
            running_loss += float(loss.detach().cpu().item())
            batches += 1
            should_step = (batch_idx % int(args.gradient_accumulation_steps) == 0) or (
                start + int(args.per_device_train_batch_size) >= len(permutation)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad] + list(value_head.parameters()),
                    max_norm=float(args.max_grad_norm),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                global_step += 1
                if global_step % int(args.logging_steps) == 0:
                    print(
                        "train_step          : "
                        f"epoch={epoch_idx} global_step={global_step} "
                        f"avg_loss={(running_loss / max(1, batches)):.6f} "
                        f"lr={scheduler.get_last_lr()[0]:.8f}",
                        flush=True,
                    )
        epoch_stats = {
            "epoch": int(epoch_idx),
            "avg_loss": float(running_loss / max(1, batches)),
            "num_batches": int(batches),
            "optimizer_steps": int(optimizer_steps),
            "global_step_end": int(global_step),
            "last_lr": float(scheduler.get_last_lr()[0]),
        }
        eval_metrics, chosen_scores, rejected_scores = _evaluate_pairs(
            model=model,
            value_head=value_head,
            tokenizer=tokenizer,
            pair_rows=eval_rows,
            batch_size=int(args.per_device_eval_batch_size),
            objective_mode=str(args.objective_mode),
            ranking_target_space=str(args.ranking_target_space),
            ranking_margin=float(args.ranking_margin),
            lambda_ranking=float(args.lambda_ranking),
            lambda_bce=float(args.lambda_bce),
            lambda_terminal_bce=float(args.terminal_bce_lambda),
            anti_saturation_weight=float(args.anti_saturation_weight),
            anti_saturation_logit_threshold=float(args.anti_saturation_logit_threshold),
            reward_centering_weight=float(args.reward_centering_weight),
            max_length=int(args.max_length),
            torch_module=torch,
        )
        selection_value, _ = select_metric_value(
            eval_metrics,
            selection_metric=str(args.checkpoint_selection_metric),
        )
        improved = (
            selection_value > best_selection_value if selection_higher_is_better else selection_value < best_selection_value
        )
        if improved:
            best_selection_value = float(selection_value)
            best_eval_metrics = dict(eval_metrics)
            best_eval_scores = (list(chosen_scores), list(rejected_scores))
            best_epoch = int(epoch_idx)
            save_value_head_checkpoint(
                best_ckpt_path,
                value_head=value_head,
                config=value_head_config,
                extra_state={
                    "epoch": int(epoch_idx),
                    "selection_metric": str(args.checkpoint_selection_metric),
                    "selection_value": float(selection_value),
                },
            )
            model.save_pretrained(str(best_adapter_dir))
        curve_row = {
            "epoch": int(epoch_idx),
            "train": epoch_stats,
            "eval": eval_metrics,
            "selection_metric": str(args.checkpoint_selection_metric),
            "selection_value": float(selection_value),
            "best_so_far": bool(improved),
        }
        train_curve.append(curve_row)
        with train_curve_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(curve_row, ensure_ascii=False) + "\n")
        print(
            "epoch_summary      : "
            f"epoch={epoch_idx} "
            f"train_loss={epoch_stats['avg_loss']:.6f} "
            f"eval_pair_acc={eval_metrics['pair_accuracy']:.6f} "
            f"eval_auc={eval_metrics['auc']:.6f} "
            f"selection={float(selection_value):.6f} "
            f"best={bool(improved)}",
            flush=True,
        )

    train_elapsed = time.perf_counter() - train_start
    save_value_head_checkpoint(
        final_ckpt_path,
        value_head=value_head,
        config=value_head_config,
        extra_state={"epoch": int(args.num_train_epochs - 1)},
    )
    model.save_pretrained(str(final_adapter_dir))

    if best_eval_metrics is None or best_eval_scores is None:
        raise RuntimeError("No best checkpoint was selected")
    _write_eval_pair_scores(
        path=eval_pair_scores_path,
        eval_pairs=eval_rows.pairs,
        chosen_scores=best_eval_scores[0],
        rejected_scores=best_eval_scores[1],
    )

    parameter_summary = _trainable_parameter_summary(model)
    train_metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "train_curve": train_curve,
        "model_load_elapsed_sec": float(model_load_elapsed),
        "train_elapsed_sec": float(train_elapsed),
        "global_step": int(global_step),
        "parameter_summary": parameter_summary,
    }
    eval_metrics_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "best_epoch": int(best_epoch),
        "selection_metric": str(args.checkpoint_selection_metric),
        "selection_value": float(best_selection_value),
        "eval_pairs": best_eval_metrics,
    }
    manifest = {
        "artifact_stage": "phase_e_value_lora_run_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": str(args.run_name),
        "run_dir": str(run_dir),
        "resolved_backbone": {
            "model_path": str(args.model_path),
            "adapter_path": str(best_adapter_dir),
            "dtype": str(args.dtype),
            # Reuse a valid Hugging Face device_map token so existing Phase E
            # eval scripts can reload the adapter-backed backbone without any
            # special-case branch for LoRA runs.
            # 这里必须写成 Hugging Face 认识的 device_map 值，否则现有评测脚本
            # 会把 manifest 里的字符串直接传回 `from_pretrained`，从而导致重载失败。
            "device_map": "auto",
            "max_gpu_memory_gib": None,
            "max_cpu_memory_gib": None,
            "tokenizer_path": str(args.model_path),
        },
        "input_files": {
            "train_pairs_jsonl": str(args.train_pairs_jsonl),
            "eval_pairs_jsonl": str(args.eval_pairs_jsonl),
        },
        "train_config": train_config.to_dict(),
        "train_pair_summary": train_pair_stats,
        "eval_pair_summary": eval_pair_stats,
        "recipe_risk": recipe_risk_report,
        "train_truncation_diagnostics": train_truncation_diagnostics,
        "eval_truncation_diagnostics": eval_truncation_diagnostics,
        "init_value_head": init_info,
        "lora_spec": lora_spec,
        "parameter_summary": parameter_summary,
        "output_files": {
            "manifest": str(manifest_path),
            "train_metrics": str(train_metrics_path),
            "eval_metrics": str(eval_metrics_path),
            "summary": str(summary_path),
            "summary_md": str(summary_md_path),
            "best_value_head": str(best_ckpt_path),
            "final_value_head": str(final_ckpt_path),
            "best_adapter": str(best_adapter_dir),
            "final_adapter": str(final_adapter_dir),
            "value_head_config": str(config_json_path),
            "eval_pair_scores": str(eval_pair_scores_path),
            "train_curve": str(train_curve_path),
            "recipe_risk": str(recipe_risk_path),
        },
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "best_epoch": int(best_epoch),
        "selection_metric": str(args.checkpoint_selection_metric),
        "selection_value": float(best_selection_value),
        "head_architecture": str(args.head_architecture),
        "recipe_risk_max_severity": str(recipe_risk_report.get("max_severity", "info")),
        "lora_top_k_layers": train_config.lora_top_k_layers,
        "lora_target_modules": train_config.lora_target_modules,
        "eval_pairs": best_eval_metrics,
        "train_pairs": int(len(train_pairs)),
        "eval_pairs_n": int(len(eval_pairs)),
        "train_elapsed_sec": float(train_elapsed),
        "model_load_elapsed_sec": float(model_load_elapsed),
        "parameter_summary": parameter_summary,
    }

    train_metrics_path.write_text(json.dumps(train_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    eval_metrics_path.write_text(json.dumps(eval_metrics_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")

    print("-" * 88)
    print(f"global_step        : {global_step}")
    print(f"train_elapsed_sec  : {train_elapsed:.2f}")
    print(f"selected_metric    : {args.checkpoint_selection_metric}")
    print(f"selected_value     : {float(best_selection_value):.6f}")
    print(f"pair_accuracy      : {float(best_eval_metrics['pair_accuracy']):.6f}")
    print(f"auc                : {float(best_eval_metrics['auc']):.6f}")
    print(f"mean_margin        : {float(best_eval_metrics['mean_margin']):.6f}")
    print(f"best_adapter_dir   : {best_adapter_dir}")
    print(f"train_metrics      : {train_metrics_path}")
    print(f"eval_metrics       : {eval_metrics_path}")
    print(f"manifest           : {manifest_path}")
    print(f"summary            : {summary_path}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
