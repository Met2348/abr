#!/usr/bin/env python3
"""Train a value head on external pair artifacts only.

English
-------
This is the Phase E trainer entrypoint.  It intentionally depends on only:

1. a base model path,
2. one training pair artifact,
3. one evaluation pair artifact.

It does **not** depend on Phase C StrategyQA corruption manifests.  That
decoupling is one of the main methodological changes in Phase E.

中文
----
这是 Phase E 的训练入口脚本。它刻意只依赖：

1. 一个 base model 路径，
2. 一份 train pair artifact，
3. 一份 eval pair artifact。

它**不再**依赖 Phase C 的 StrategyQA corruption 产物。这种解耦是 Phase E
 最重要的方法学变化之一。
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
    infer_backbone_hidden_size,
    load_value_head_checkpoint,
    save_value_head_checkpoint,
    write_value_head_config_json,
)
from ours.phase_d.external_pairs import load_external_pair_jsonl  # noqa: E402
from ours.phase_e.runtime import (  # noqa: E402
    build_phase_e_backbone_signature,
    import_runtime_deps,
    load_backbone_and_tokenizer,
    resolve_value_device,
    set_seed,
)
from ours.phase_e.training import (  # noqa: E402
    build_pair_feature_cache,
    build_pair_permutation,
    compute_pair_objective,
    evaluate_pair_cache,
    select_metric_value,
)


@dataclass(slots=True)
class PhaseETrainConfig:
    """Compact train config snapshot persisted into the run manifest."""

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
    ranking_margin: float
    pair_weight_mode: str
    source_balance: str
    permutation_mode: str
    anti_saturation_weight: float
    anti_saturation_logit_threshold: float
    checkpoint_selection_metric: str
    logging_steps: int
    seed: int
    dtype: str
    device_map: str
    require_cuda: bool
    strict_determinism: bool
    init_value_head_path: str | None
    feature_cache_root: str
    feature_cache_mode: str
    feature_cache_lock_timeout_sec: float
    head_architecture: str
    head_dropout_prob: float
    head_init_std: float
    head_mlp_hidden_size: int
    head_activation: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable config snapshot."""
        return asdict(self)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a benchmark-native Phase E value head on external pair artifacts only."
    )
    parser.add_argument("--train-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--eval-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--run-name", default="phase_e_value")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_runs"),
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--objective-mode", choices=["ranking_only", "pair_bce_only", "joint"], default="ranking_only")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=32)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lambda-ranking", type=float, default=1.0)
    parser.add_argument("--lambda-bce", type=float, default=0.25)
    parser.add_argument("--ranking-margin", type=float, default=0.05)
    parser.add_argument("--pair-weight-mode", choices=["none", "confidence"], default="confidence")
    parser.add_argument("--source-balance", choices=["none", "uniform"], default="none")
    parser.add_argument("--permutation-mode", choices=["random", "stable_hash"], default="stable_hash")
    parser.add_argument("--anti-saturation-weight", type=float, default=0.0)
    parser.add_argument("--anti-saturation-logit-threshold", type=float, default=4.0)
    parser.add_argument("--checkpoint-selection-metric", choices=["pair_acc", "auc", "ranking_score", "pair_loss"], default="ranking_score")
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict-determinism", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--init-value-head-path", type=Path, default=None)
    parser.add_argument("--head-architecture", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--head-dropout-prob", type=float, default=0.0)
    parser.add_argument("--head-init-std", type=float, default=0.02)
    parser.add_argument("--head-mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--head-activation", choices=["gelu", "relu", "tanh"], default="gelu")
    parser.add_argument(
        "--feature-cache-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_feature_cache"),
    )
    parser.add_argument(
        "--feature-cache-mode",
        choices=["off", "read", "write", "read_write"],
        default="read_write",
    )
    parser.add_argument("--feature-cache-lock-timeout-sec", type=float, default=600.0)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and fail fast on obviously invalid settings.

    中文
    ----
    这里做的是“配置级别”的 fail-fast，例如：
    1. 路径是否存在，
    2. batch size 是否为正，
    3. 某个 objective 的必要权重是否给了。

    真正的数据内容与训练行为检查，会在主流程中继续完成。
    """
    args = _build_parser().parse_args(argv)
    if not args.train_pairs_jsonl.exists():
        raise FileNotFoundError(f"--train-pairs-jsonl not found: {args.train_pairs_jsonl}")
    if not args.eval_pairs_jsonl.exists():
        raise FileNotFoundError(f"--eval-pairs-jsonl not found: {args.eval_pairs_jsonl}")
    if args.adapter_path is not None and not Path(args.adapter_path).exists():
        raise FileNotFoundError(f"--adapter-path not found: {args.adapter_path}")
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
    if not (0.0 <= float(args.head_dropout_prob) < 1.0):
        raise ValueError("--head-dropout-prob must be in [0, 1)")
    if float(args.head_init_std) <= 0.0:
        raise ValueError("--head-init-std must be > 0")
    if int(args.head_mlp_hidden_size) <= 0:
        raise ValueError("--head-mlp-hidden-size must be > 0")
    if int(args.logging_steps) <= 0:
        raise ValueError("--logging-steps must be > 0")
    if float(args.feature_cache_lock_timeout_sec) <= 0.0:
        raise ValueError("--feature-cache-lock-timeout-sec must be > 0")
    if args.objective_mode == "ranking_only" and float(args.lambda_ranking) == 0.0:
        raise ValueError("ranking_only requires --lambda-ranking > 0")
    if args.objective_mode == "pair_bce_only" and float(args.lambda_bce) == 0.0:
        raise ValueError("pair_bce_only requires --lambda-bce > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Load canonical pair artifacts first.  At this point we are still operating
    # on JSONL records, not tensors.
    # 先读取 canonical pair artifact。此时我们处理的还只是 JSONL 记录，不是张量。
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

    # Set allocator policy before importing torch/CUDA so memory behavior is
    # more predictable during large-batch feature encoding.
    # 在导入 torch/CUDA 之前先设置分配器策略，减少大 batch 特征编码时的显存碎片问题。
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # On some GPUs, deterministic CuBLAS behavior also needs an explicit
    # workspace setting.
    # 某些 GPU 上，要想让 CuBLAS 行为更可复现，还需要额外设置 workspace。
    if bool(args.strict_determinism):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch, AutoModelForCausalLM, AutoTokenizer = import_runtime_deps()
    if bool(args.require_cuda) and not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA is required by --require-cuda but no GPU is visible")
    set_seed(
        int(args.seed),
        torch,
        strict_determinism=bool(args.strict_determinism),
    )

    # Persist the fully resolved knobs so later suites compare runs by the exact
    # effective settings, not by incomplete memory of what was passed on CLI.
    # 把最终生效的训练参数完整落盘，避免后续对比时只记得“大概传了什么”。
    train_config = PhaseETrainConfig(
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
        ranking_margin=float(args.ranking_margin),
        pair_weight_mode=str(args.pair_weight_mode),
        source_balance=str(args.source_balance),
        permutation_mode=str(args.permutation_mode),
        anti_saturation_weight=float(args.anti_saturation_weight),
        anti_saturation_logit_threshold=float(args.anti_saturation_logit_threshold),
        checkpoint_selection_metric=str(args.checkpoint_selection_metric),
        logging_steps=int(args.logging_steps),
        seed=int(args.seed),
        dtype=str(args.dtype),
        device_map=str(args.device_map),
        require_cuda=bool(args.require_cuda),
        strict_determinism=bool(args.strict_determinism),
        init_value_head_path=(str(args.init_value_head_path) if args.init_value_head_path is not None else None),
        feature_cache_root=str(args.feature_cache_root),
        feature_cache_mode=str(args.feature_cache_mode),
        feature_cache_lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
        head_architecture=str(args.head_architecture),
        head_dropout_prob=float(args.head_dropout_prob),
        head_init_std=float(args.head_init_std),
        head_mlp_hidden_size=int(args.head_mlp_hidden_size),
        head_activation=str(args.head_activation),
    )

    # Every run gets a timestamped directory so repeated executions with the
    # same logical name do not overwrite each other.
    # 每次运行都用时间戳隔离目录，避免同名实验互相覆盖。
    run_dir = Path(args.output_root) / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    train_metrics_path = run_dir / "train_metrics.json"
    eval_metrics_path = run_dir / "eval_metrics.json"
    summary_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    best_ckpt_path = run_dir / "best_value_head.pt"
    final_ckpt_path = run_dir / "final_value_head.pt"
    config_json_path = run_dir / "value_head_config.json"
    eval_pair_scores_path = run_dir / "eval_pair_scores.jsonl"
    train_curve_path = run_dir / "train_curve.jsonl"

    print("=" * 88)
    print("Phase E: Train Value Head")
    print("=" * 88)
    print(f"train_pairs_jsonl  : {args.train_pairs_jsonl}")
    print(f"eval_pairs_jsonl   : {args.eval_pairs_jsonl}")
    print(f"run_dir            : {run_dir}")
    print(f"model_path         : {args.model_path}")
    print(f"adapter_path       : {args.adapter_path if args.adapter_path is not None else '<none>'}")
    print(f"objective_mode     : {args.objective_mode}")
    print(f"train_pairs        : {len(train_pairs)}")
    print(f"eval_pairs         : {len(eval_pairs)}")
    print(f"batch_train        : {args.per_device_train_batch_size}")
    print(f"batch_eval         : {args.per_device_eval_batch_size}")
    print(f"grad_accum         : {args.gradient_accumulation_steps}")
    print(f"max_length         : {args.max_length}")
    print(f"pair_weight_mode   : {args.pair_weight_mode}")
    print(f"source_balance     : {args.source_balance}")
    print(f"permutation_mode   : {args.permutation_mode}")
    print(f"selection_metric   : {args.checkpoint_selection_metric}")
    print(f"strict_determinism : {bool(args.strict_determinism)}")
    print("=" * 88)

    load_start = time.perf_counter()
    backbone, tokenizer, _, tokenizer_path = load_backbone_and_tokenizer(
        model_path=str(args.model_path),
        adapter_path=(Path(args.adapter_path) if args.adapter_path is not None else None),
        dtype_name=str(args.dtype),
        device_map=str(args.device_map),
        torch_module=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
    )
    hidden_size = infer_backbone_hidden_size(backbone)
    value_head_config = ValueHeadConfig(
        hidden_size=int(hidden_size),
        dropout_prob=float(args.head_dropout_prob),
        init_std=float(args.head_init_std),
        architecture=str(args.head_architecture),
        mlp_hidden_size=int(args.head_mlp_hidden_size),
        activation=str(args.head_activation),
    )
    value_head = SigmoidValueHead(value_head_config)
    value_device = resolve_value_device(backbone, torch)
    value_head.to(value_device)
    init_info = _initialize_value_head_from_checkpoint(
        value_head=value_head,
        current_config=value_head_config,
        checkpoint_path=(Path(args.init_value_head_path) if args.init_value_head_path is not None else None),
    )
    model_load_elapsed = time.perf_counter() - load_start

    # This object is later written to metrics so operators can see whether a run
    # truly reused cached features or recomputed them.
    # 这个统计对象会写进 metrics，方便事后确认本次运行到底命中了多少缓存。
    feature_cache_stats: dict[str, Any] = {
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "entries": {},
    }
    # Core Phase E idea:
    # 1. expensive backbone forward pass happens once here,
    # 2. later training only updates the small value head on cached tensors.
    #
    # Phase E 的核心工程思想就是：
    # 1. 昂贵的 backbone 前向只在这里跑一次，
    # 2. 后续训练只更新小型 value head。
    feature_start = time.perf_counter()
    backbone_signature = build_phase_e_backbone_signature(
        model_path=str(args.model_path),
        adapter_path=(str(args.adapter_path) if args.adapter_path is not None else None),
        tokenizer_path=str(tokenizer_path),
        dtype=str(args.dtype),
        max_length=int(args.max_length),
    )
    use_confidence_weights = bool(args.pair_weight_mode == "confidence")
    train_cache = build_pair_feature_cache(
        pairs=train_pairs,
        split_label="train",
        backbone=backbone,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        batch_size=int(args.per_device_eval_batch_size),
        feature_cache_root=Path(args.feature_cache_root),
        feature_cache_mode=str(args.feature_cache_mode),
        lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
        backbone_signature=backbone_signature,
        use_confidence_weights=use_confidence_weights,
        torch_module=torch,
        feature_cache_stats=feature_cache_stats,
    )
    eval_cache = build_pair_feature_cache(
        pairs=eval_pairs,
        split_label="eval",
        backbone=backbone,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        batch_size=int(args.per_device_eval_batch_size),
        feature_cache_root=Path(args.feature_cache_root),
        feature_cache_mode=str(args.feature_cache_mode),
        lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
        backbone_signature=backbone_signature,
        use_confidence_weights=use_confidence_weights,
        torch_module=torch,
        feature_cache_stats=feature_cache_stats,
    )
    feature_cache_elapsed = time.perf_counter() - feature_start

    # Keep the full cached tensors on CPU and move only mini-batches to GPU.
    # Otherwise the backbone plus full feature cache would occupy GPU memory at
    # the same time and easily cause OOM.
    #
    # 这里故意让完整缓存留在 CPU，只把 mini-batch 搬到 GPU。
    # 否则 backbone 和整套特征缓存会同时占显存，非常容易 OOM。
    #
    # Once features are cached, the backbone is no longer needed for training.
    # 特征缓存完成后，训练阶段就不再需要 backbone，本体应尽快释放。
    del backbone
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    write_value_head_config_json(config_json_path, value_head_config)
    optimizer = torch.optim.AdamW(
        value_head.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    total_optimizer_steps = _resolve_total_optimizer_steps(
        num_pairs=int(train_cache["num_pairs"]),
        batch_size=int(args.per_device_train_batch_size),
        grad_accum_steps=int(args.gradient_accumulation_steps),
        num_epochs=int(args.num_train_epochs),
    )
    warmup_steps = int(total_optimizer_steps * float(args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _linear_warmup_decay(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_optimizer_steps,
        ),
    )

    # All selection metrics except `pair_loss` are maximization targets.
    # 也就是说除了 `pair_loss` 外，其余 checkpoint 选择指标都遵循“越高越好”。
    selection_higher_is_better = str(args.checkpoint_selection_metric) != "pair_loss"
    best_selection_value = float("-inf") if selection_higher_is_better else float("inf")
    best_eval_metrics: dict[str, Any] | None = None
    best_eval_scores: tuple[list[float], list[float]] | None = None
    best_epoch = -1
    global_step = 0
    train_curve: list[dict[str, Any]] = []

    train_start = time.perf_counter()
    # The outer loop is intentionally simple: one training epoch, then one
    # held-out evaluation, then optional checkpoint promotion.
    # 外层循环刻意保持简单：先训一个 epoch，再跑一次 held-out eval，再决定是否晋升 checkpoint。
    for epoch_idx in range(int(args.num_train_epochs)):
        epoch_stats = _run_one_epoch(
            value_head=value_head,
            pair_cache=train_cache,
            torch_module=torch,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=int(args.per_device_train_batch_size),
            grad_accum_steps=int(args.gradient_accumulation_steps),
            max_grad_norm=float(args.max_grad_norm),
            objective_mode=str(args.objective_mode),
            ranking_margin=float(args.ranking_margin),
            lambda_ranking=float(args.lambda_ranking),
            lambda_bce=float(args.lambda_bce),
            anti_saturation_weight=float(args.anti_saturation_weight),
            anti_saturation_logit_threshold=float(args.anti_saturation_logit_threshold),
            source_balance=str(args.source_balance),
            permutation_mode=str(args.permutation_mode),
            logging_steps=int(args.logging_steps),
            global_step_start=int(global_step),
        )
        global_step = int(epoch_stats["global_step_end"])
        eval_metrics, chosen_scores, rejected_scores = evaluate_pair_cache(
            value_head=value_head,
            pair_cache=eval_cache,
            batch_size=int(args.per_device_eval_batch_size),
            objective_mode=str(args.objective_mode),
            ranking_margin=float(args.ranking_margin),
            lambda_ranking=float(args.lambda_ranking),
            lambda_bce=float(args.lambda_bce),
            anti_saturation_weight=float(args.anti_saturation_weight),
            anti_saturation_logit_threshold=float(args.anti_saturation_logit_threshold),
            torch_module=torch,
        )
        selection_value, _ = select_metric_value(
            eval_metrics,
            selection_metric=str(args.checkpoint_selection_metric),
        )
        improved = (
            selection_value > best_selection_value
            if selection_higher_is_better
            else selection_value < best_selection_value
        )
        if improved:
            # "Best checkpoint" always means "best under the selected metric",
            # not necessarily the final epoch.
            # 这里的“best checkpoint”永远指“在所选指标下最好”，不等于最后一个 epoch。
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
            f"train_loss={float(epoch_stats['avg_loss']):.6f} "
            f"eval_pair_acc={float(eval_metrics['pair_accuracy']):.6f} "
            f"eval_auc={float(eval_metrics['auc']):.6f} "
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

    if best_eval_metrics is None or best_eval_scores is None:
        raise RuntimeError("No best checkpoint was selected")
    _write_eval_pair_scores(
        path=eval_pair_scores_path,
        eval_pairs=eval_pairs,
        chosen_scores=best_eval_scores[0],
        rejected_scores=best_eval_scores[1],
    )

    train_metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "train_curve": train_curve,
        "feature_cache_stats": feature_cache_stats,
        "feature_cache_elapsed_sec": float(feature_cache_elapsed),
        "model_load_elapsed_sec": float(model_load_elapsed),
        "train_elapsed_sec": float(train_elapsed),
        "global_step": int(global_step),
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
        "artifact_stage": "phase_e_value_run_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": str(args.run_name),
        "run_dir": str(run_dir),
        "resolved_backbone": {
            "model_path": str(args.model_path),
            "adapter_path": (str(args.adapter_path) if args.adapter_path is not None else None),
            "dtype": str(args.dtype),
            "device_map": str(args.device_map),
            "tokenizer_path": str(tokenizer_path),
        },
        "input_files": {
            "train_pairs_jsonl": str(args.train_pairs_jsonl),
            "eval_pairs_jsonl": str(args.eval_pairs_jsonl),
        },
        "train_config": train_config.to_dict(),
        "train_pair_summary": train_pair_stats,
        "eval_pair_summary": eval_pair_stats,
        "init_value_head": init_info,
        "output_files": {
            "manifest": str(manifest_path),
            "train_metrics": str(train_metrics_path),
            "eval_metrics": str(eval_metrics_path),
            "summary": str(summary_path),
            "summary_md": str(summary_md_path),
            "best_value_head": str(best_ckpt_path),
            "final_value_head": str(final_ckpt_path),
            "value_head_config": str(config_json_path),
            "eval_pair_scores": str(eval_pair_scores_path),
            "train_curve": str(train_curve_path),
        },
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "best_epoch": int(best_epoch),
        "selection_metric": str(args.checkpoint_selection_metric),
        "selection_value": float(best_selection_value),
        "eval_pairs": best_eval_metrics,
        "train_pairs": int(len(train_pairs)),
        "eval_pairs_n": int(len(eval_pairs)),
        "train_elapsed_sec": float(train_elapsed),
        "feature_cache_elapsed_sec": float(feature_cache_elapsed),
        "model_load_elapsed_sec": float(model_load_elapsed),
    }

    train_metrics_path.write_text(json.dumps(train_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    eval_metrics_path.write_text(json.dumps(eval_metrics_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")

    print("-" * 88)
    print(f"global_step        : {global_step}")
    print(f"train_elapsed_sec  : {train_elapsed:.2f}")
    print(f"feature_cache_sec  : {feature_cache_elapsed:.2f}")
    print(f"selected_metric    : {args.checkpoint_selection_metric}")
    print(f"selected_value     : {float(best_selection_value):.6f}")
    print(f"pair_accuracy      : {float(best_eval_metrics['pair_accuracy']):.6f}")
    print(f"auc                : {float(best_eval_metrics['auc']):.6f}")
    print(f"mean_margin        : {float(best_eval_metrics['mean_margin']):.6f}")
    print(f"train_metrics      : {train_metrics_path}")
    print(f"eval_metrics       : {eval_metrics_path}")
    print(f"manifest           : {manifest_path}")
    print(f"summary            : {summary_path}")
    print("=" * 88)
    return 0


def _run_one_epoch(
    *,
    value_head: Any,
    pair_cache: dict[str, Any],
    torch_module: Any,
    optimizer: Any,
    scheduler: Any,
    batch_size: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    objective_mode: str,
    ranking_margin: float,
    lambda_ranking: float,
    lambda_bce: float,
    anti_saturation_weight: float,
    anti_saturation_logit_threshold: float,
    source_balance: str,
    permutation_mode: str,
    logging_steps: int,
    global_step_start: int,
) -> dict[str, Any]:
    """Train one epoch on cached pair features.

    English
    -------
    Important implementation detail:
    `pair_cache` already contains frozen features, so the only trainable module
    touched here is `value_head`.

    中文
    ----
    一个关键点是：`pair_cache` 里已经是冻结特征了，所以这个函数里真正被训练的
    模块只有 `value_head`，不会再回到 backbone。
    """
    permutation = build_pair_permutation(
        pair_cache=pair_cache,
        torch_module=torch_module,
        permutation_mode=str(permutation_mode),
        source_balance=str(source_balance),
    )
    chosen_features = pair_cache["chosen_features"]
    rejected_features = pair_cache["rejected_features"]
    pair_weights = pair_cache["pair_weights"]
    head_device = next(value_head.parameters()).device
    head_dtype = next(value_head.parameters()).dtype

    # `set_to_none=True` is a small performance/memory optimization in PyTorch.
    # `set_to_none=True` 是 PyTorch 里一个常见的小优化，能减少一些不必要的内存写入。
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    batches = 0
    optimizer_steps = 0
    global_step = int(global_step_start)

    for batch_idx, start in enumerate(range(0, int(pair_cache["num_pairs"]), int(batch_size)), start=1):
        indices = permutation[start : start + int(batch_size)]
        chosen_batch = chosen_features[indices]
        rejected_batch = rejected_features[indices]
        weight_batch = pair_weights[indices]
        if chosen_batch.device != head_device or chosen_batch.dtype != head_dtype:
            chosen_batch = chosen_batch.to(device=head_device, dtype=head_dtype)
        if rejected_batch.device != head_device or rejected_batch.dtype != head_dtype:
            rejected_batch = rejected_batch.to(device=head_device, dtype=head_dtype)
        if weight_batch.device != head_device or weight_batch.dtype != head_dtype:
            weight_batch = weight_batch.to(device=head_device, dtype=head_dtype)

        chosen_out = value_head(chosen_batch)
        rejected_out = value_head(rejected_batch)
        loss = compute_pair_objective(
            chosen_logits=chosen_out["logits"],
            rejected_logits=rejected_out["logits"],
            chosen_scores=chosen_out["scores"],
            rejected_scores=rejected_out["scores"],
            pair_weights=weight_batch,
            objective_mode=str(objective_mode),
            ranking_margin=float(ranking_margin),
            lambda_ranking=float(lambda_ranking),
            lambda_bce=float(lambda_bce),
            anti_saturation_weight=float(anti_saturation_weight),
            anti_saturation_logit_threshold=float(anti_saturation_logit_threshold),
            torch_module=torch_module,
        )
        (loss / float(grad_accum_steps)).backward()
        running_loss += float(loss.detach().cpu().item())
        batches += 1

        # We step the optimizer either:
        # 1. every `grad_accum_steps` mini-batches, or
        # 2. at the very last batch of the epoch.
        #
        # 优化器更新发生在两种情况下：
        # 1. 累积满 `grad_accum_steps` 个 mini-batch；
        # 2. 或者已经到了 epoch 最后一个 batch。
        should_step = (batch_idx % int(grad_accum_steps) == 0) or (
            start + int(batch_size) >= int(pair_cache["num_pairs"])
        )
        if should_step:
            if float(max_grad_norm) > 0.0:
                torch_module.nn.utils.clip_grad_norm_(value_head.parameters(), float(max_grad_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            global_step += 1
            if global_step % int(logging_steps) == 0:
                print(
                    "train            : "
                    f"step={global_step} "
                    f"loss={running_loss / max(batches, 1):.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.6g}",
                    flush=True,
                )

    return {
        "avg_loss": float(running_loss / max(batches, 1)),
        "num_batches": int(batches),
        "optimizer_steps": int(optimizer_steps),
        "global_step_end": int(global_step),
    }


def _initialize_value_head_from_checkpoint(
    *,
    value_head: Any,
    current_config: ValueHeadConfig,
    checkpoint_path: Path | None,
) -> dict[str, Any] | None:
    """Optionally warm-start the value head from a previous checkpoint.

    中文
    ----
    这主要给 staged curriculum 或 bridge-style 训练使用：先在前一阶段训练出一个
    head，再把它作为当前阶段的初始化，而不是从随机参数重新开始。
    """
    if checkpoint_path is None:
        return None
    loaded_head, loaded_config, extra_state = load_value_head_checkpoint(checkpoint_path)
    loaded_signature = loaded_config.to_dict()
    current_signature = current_config.to_dict()
    if loaded_signature != current_signature:
        raise ValueError(
            "Warm-start value head config mismatch: "
            f"loaded={loaded_signature} current={current_signature}"
        )
    value_head.load_state_dict(loaded_head.state_dict())
    return {
        "path": str(checkpoint_path),
        "loaded_config": loaded_config.to_dict(),
        "extra_state": extra_state,
    }


def _resolve_total_optimizer_steps(
    *,
    num_pairs: int,
    batch_size: int,
    grad_accum_steps: int,
    num_epochs: int,
) -> int:
    """Return the total optimizer-step budget for the run.

    中文
    ----
    这里算的是“优化器真正 step 多少次”，不是“看到了多少个 mini-batch”。
    因为 warmup/decay scheduler 需要的正是 optimizer step 数。
    """
    num_batches = max(1, math.ceil(float(num_pairs) / float(max(1, batch_size))))
    steps_per_epoch = max(1, math.ceil(float(num_batches) / float(max(1, grad_accum_steps))))
    return int(max(1, steps_per_epoch * int(num_epochs)))


def _linear_warmup_decay(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup + linear decay schedule multiplier.

    中文
    ----
    返回的是一个乘子而不是绝对学习率。外层 optimizer 的基础学习率会再乘上这里的值。
    """
    if total_steps <= 1:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    remaining = max(total_steps - step - 1, 0)
    decay_steps = max(total_steps - warmup_steps, 1)
    return max(float(remaining) / float(decay_steps), 0.0)


def _write_eval_pair_scores(
    *,
    path: Path,
    eval_pairs,
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> None:
    """Persist per-pair scored rows for the selected checkpoint.

    中文
    ----
    这个文件非常适合做误差分析，因为它保留了每个 pair 的：
    1. chosen/rejected 分数，
    2. margin，
    3. source/domain 标签。
    """
    with path.open("w", encoding="utf-8") as handle:
        for pair, chosen, rejected in zip(eval_pairs, chosen_scores, rejected_scores, strict=True):
            handle.write(
                json.dumps(
                    {
                        "pair_id": pair.pair_id,
                        "source_tag": pair.source_tag,
                        "domain_tag": pair.domain_tag,
                        "pair_confidence": float(pair.pair_confidence),
                        "chosen_score": float(chosen),
                        "rejected_score": float(rejected),
                        "margin": float(chosen - rejected),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    """Render a short Markdown summary for one Phase E value run.

    中文
    ----
    Markdown summary 只保留最关键的可读指标；更完整的结构化信息仍然在 JSON 里。
    """
    eval_metrics = summary["eval_pairs"]
    lines = [
        "# Phase E Value Run Summary",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- run_dir: {summary['run_dir']}",
        f"- best_epoch: {summary['best_epoch']}",
        f"- selection_metric: {summary['selection_metric']}",
        f"- selection_value: {summary['selection_value']:.6f}",
        f"- train_pairs: {summary['train_pairs']}",
        f"- eval_pairs: {summary['eval_pairs_n']}",
        "",
        "## Held-Out Pair Metrics",
        "",
        f"- pair_accuracy: {eval_metrics['pair_accuracy']:.6f}",
        f"- auc: {eval_metrics['auc']:.6f}",
        f"- ranking_score: {eval_metrics['ranking_score']:.6f}",
        f"- mean_margin: {eval_metrics['mean_margin']:.6f}",
        f"- pair_loss: {eval_metrics['pair_loss']:.6f}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
