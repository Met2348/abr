#!/usr/bin/env python3
"""Evaluate one Phase E value-head run on benchmark-native datasets.

English
-------
Training-time held-out pair metrics answer only one question:
"does the head fit the source-family supervision it trained on?"

This script answers the next question:
"what happens when we score benchmark-native examples that were not used in the
same training loop?"

Supported benchmark families:
1. ProcessBench prefix-quality structure
2. PRMBench preview prefix pairs

中文
----
训练时的 held-out pair 指标只能回答一个问题：
“模型有没有拟合它训练时看到的 source-family 监督？”

本脚本要回答的是下一层问题：
“把这个 head 拿去打 benchmark-native 样本时，会发生什么？”

当前支持两类 benchmark：
1. ProcessBench 的 prefix 质量结构
2. PRMBench preview 的显式 pair
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_b.value_head import load_value_head_checkpoint  # noqa: E402
from ours.phase_e.benchmark_eval import (  # noqa: E402
    build_processbench_prefix_records,
    compute_pair_ranking_metrics,
    compute_processbench_metrics,
    load_prmbench_preview_pairs,
    load_processbench_examples,
    render_phase_e_benchmark_summary_markdown,
)
from ours.phase_e.contracts import (  # noqa: E402
    get_phase_e_eval_benchmark_registry,
    resolve_phase_e_benchmark_specs,
)
from ours.phase_e.runtime import (  # noqa: E402
    build_phase_e_backbone_signature,
    import_runtime_deps,
    load_backbone_and_tokenizer,
    load_or_encode_text_features,
    resolve_checkpoint_path,
    resolve_value_device,
    score_feature_tensor,
    score_pair_features,
)
from ours.phase_e.training import (  # noqa: E402
    compute_text_truncation_diagnostics,
    validate_text_truncation_diagnostics,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one Phase E value-head checkpoint on benchmark-native datasets."
    )
    parser.add_argument("--value-run-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark-id",
        required=True,
        choices=sorted(get_phase_e_eval_benchmark_registry()),
    )
    parser.add_argument("--benchmark-path", type=Path, default=None)
    parser.add_argument("--run-name", default="phase_e_eval_benchmark")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_eval"),
    )
    parser.add_argument("--checkpoint-name", choices=["best", "final"], default="best")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--processbench-f1-threshold",
        type=float,
        default=None,
        help=(
            "Optional fixed threshold for ProcessBench F1. "
            "If unset, the benchmark metric reports the optimistic oracle sweep threshold."
        ),
    )
    parser.add_argument(
        "--processbench-f1-threshold-candidates",
        type=int,
        default=50,
        help="Number of threshold candidates to sweep when --processbench-f1-threshold is unset.",
    )
    parser.add_argument(
        "--prmbench-error-step-index-base",
        choices=["auto", "zero_based", "one_based"],
        default="auto",
        help="How to interpret PRMBench preview error_steps indices.",
    )
    parser.add_argument(
        "--max-truncation-over-limit-fraction",
        type=float,
        default=0.10,
        help=(
            "Fail fast if more than this fraction of benchmark texts exceed max_length. "
            "This keeps benchmark evals from silently scoring heavily truncated inputs."
        ),
    )
    parser.add_argument("--dtype", default="")
    parser.add_argument("--device-map", default="")
    parser.add_argument(
        "--max-gpu-memory-gib",
        type=int,
        default=None,
        help=(
            "Optional per-visible-GPU memory cap forwarded to Hugging Face `max_memory`. "
            "Useful when evaluation must coexist with other long-running jobs."
        ),
    )
    parser.add_argument(
        "--max-cpu-memory-gib",
        type=int,
        default=None,
        help="Optional CPU RAM budget paired with --max-gpu-memory-gib for controlled offload.",
    )
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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
    """Parse CLI arguments and validate obvious operator mistakes.

    中文
    ----
    这里主要检查路径、batch size、缓存锁超时等明显配置错误，
    让命令在真正加载模型前尽早失败。
    """
    args = _build_parser().parse_args(argv)
    if not args.value_run_dir.exists():
        raise FileNotFoundError(f"--value-run-dir not found: {args.value_run_dir}")
    if args.benchmark_path is not None and not Path(args.benchmark_path).exists():
        raise FileNotFoundError(f"--benchmark-path not found: {args.benchmark_path}")
    if args.max_samples is not None and int(args.max_samples) <= 0:
        raise ValueError("--max-samples must be > 0")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_length is not None and int(args.max_length) <= 8:
        raise ValueError("--max-length must be > 8")
    if int(args.processbench_f1_threshold_candidates) <= 0:
        raise ValueError("--processbench-f1-threshold-candidates must be > 0")
    if args.max_gpu_memory_gib is not None and int(args.max_gpu_memory_gib) <= 0:
        raise ValueError("--max-gpu-memory-gib must be > 0")
    if args.max_cpu_memory_gib is not None and int(args.max_cpu_memory_gib) <= 0:
        raise ValueError("--max-cpu-memory-gib must be > 0")
    if not (0.0 <= float(args.max_truncation_over_limit_fraction) <= 1.0):
        raise ValueError("--max-truncation-over-limit-fraction must be in [0, 1]")
    if float(args.feature_cache_lock_timeout_sec) <= 0.0:
        raise ValueError("--feature-cache-lock-timeout-sec must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Set allocator policy before torch import so large benchmark-side feature
    # encoding is less likely to hit fragmentation-related OOMs.
    # 在 torch 导入前设置分配器策略，降低 benchmark 侧大批量编码时因显存碎片导致 OOM 的概率。
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    benchmark_spec = resolve_phase_e_benchmark_specs([str(args.benchmark_id)])[0]
    benchmark_path = Path(args.benchmark_path) if args.benchmark_path is not None else benchmark_spec.default_path_obj()

    manifest_path = Path(args.value_run_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Phase E manifest: {manifest_path}")
    # Reuse the exact resolved backbone/adapter choices recorded during training.
    # This avoids a very common failure mode: evaluating with a slightly
    # different model stack than the one that produced the checkpoint.
    # 直接复用训练时记录下来的 backbone/adapter 解析结果，
    # 避免评测时不小心换了一个“看起来差不多、其实不完全一样”的模型栈。
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = resolve_checkpoint_path(
        value_run_dir=Path(args.value_run_dir),
        run_manifest=run_manifest,
        checkpoint_name=str(args.checkpoint_name),
    )
    requested_checkpoint_path = str(
        dict(run_manifest.get("output_files", {})).get(
            "best_value_head" if str(args.checkpoint_name) == "best" else "final_value_head",
            "",
        )
    ).strip()
    checkpoint_resolution = {
        "requested_checkpoint_name": str(args.checkpoint_name),
        "requested_checkpoint_path": requested_checkpoint_path,
        "resolved_checkpoint_path": str(checkpoint_path),
        "fallback_to_final": bool(
            str(args.checkpoint_name) == "best"
            and requested_checkpoint_path != ""
            and Path(requested_checkpoint_path) != checkpoint_path
        ),
    }
    value_head, _, _ = load_value_head_checkpoint(checkpoint_path)

    resolved_backbone = dict(run_manifest.get("resolved_backbone", {}))
    model_path = str(resolved_backbone.get("model_path", "")).strip()
    adapter_path_text = str(resolved_backbone.get("adapter_path") or "").strip()
    if model_path == "":
        raise ValueError("Phase E manifest missing resolved_backbone.model_path")

    torch, AutoModelForCausalLM, AutoTokenizer = import_runtime_deps()
    if bool(args.require_cuda) and not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA is required by --require-cuda but no GPU is visible")

    dtype_name = str(args.dtype).strip() or str(resolved_backbone.get("dtype", "bfloat16"))
    device_map = str(args.device_map).strip() or str(resolved_backbone.get("device_map", "auto"))
    max_gpu_memory_gib = (
        int(args.max_gpu_memory_gib)
        if args.max_gpu_memory_gib is not None
        else (
            int(resolved_backbone["max_gpu_memory_gib"])
            if resolved_backbone.get("max_gpu_memory_gib") is not None
            else None
        )
    )
    max_cpu_memory_gib = (
        int(args.max_cpu_memory_gib)
        if args.max_cpu_memory_gib is not None
        else (
            int(resolved_backbone["max_cpu_memory_gib"])
            if resolved_backbone.get("max_cpu_memory_gib") is not None
            else None
        )
    )
    max_length = int(args.max_length) if args.max_length is not None else int(
        run_manifest.get("train_config", {}).get("max_length", 1024)
    )
    adapter_path = Path(adapter_path_text) if adapter_path_text else None

    print("=" * 88)
    print("Phase E: Eval Benchmark")
    print("=" * 88)
    print(f"value_run_dir     : {args.value_run_dir}")
    print(f"benchmark_id      : {benchmark_spec.benchmark_id}")
    print(f"benchmark_type    : {benchmark_spec.benchmark_type}")
    print(f"benchmark_path    : {benchmark_path}")
    print(f"checkpoint_path   : {checkpoint_path}")
    print(f"checkpoint_fallback: {checkpoint_resolution['fallback_to_final']}")
    print(f"model_path        : {model_path}")
    print(f"adapter_path      : {adapter_path if adapter_path is not None else '<none>'}")
    print(f"batch_size        : {args.batch_size}")
    print(f"max_length        : {max_length}")
    print(f"trunc_overlimit_max: {float(args.max_truncation_over_limit_fraction):.4f}")
    print(f"dtype             : {dtype_name}")
    print(f"device_map        : {device_map}")
    print(f"max_gpu_memory_gib: {max_gpu_memory_gib if max_gpu_memory_gib is not None else '<none>'}")
    print(f"max_cpu_memory_gib: {max_cpu_memory_gib if max_cpu_memory_gib is not None else '<none>'}")
    print("=" * 88)

    load_start = time.perf_counter()
    backbone, tokenizer, _, tokenizer_path = load_backbone_and_tokenizer(
        model_path=model_path,
        adapter_path=adapter_path,
        dtype_name=dtype_name,
        device_map=device_map,
        max_gpu_memory_gib=max_gpu_memory_gib,
        max_cpu_memory_gib=max_cpu_memory_gib,
        torch_module=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
    )
    load_elapsed = time.perf_counter() - load_start
    value_device = resolve_value_device(backbone, torch)
    value_head.to(value_device)
    value_head.eval()

    feature_cache_stats: dict[str, Any] = {
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "entries": {},
    }
    backbone_signature = build_phase_e_backbone_signature(
        model_path=model_path,
        adapter_path=(str(adapter_path) if adapter_path is not None else None),
        tokenizer_path=tokenizer_path,
        dtype=dtype_name,
        max_length=int(max_length),
    )

    run_dir = Path(args.output_root) / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.json"
    manifest_out_path = run_dir / "manifest.json"
    summary_md_path = run_dir / "summary.md"
    scored_rows_path = run_dir / "scored_rows.jsonl"

    # Different benchmarks need different row-construction logic:
    # 1. ProcessBench -> expand each example into many prefixes
    # 2. PRMBench_Preview -> score explicit chosen/rejected pairs
    #
    # 不同 benchmark 的数据展开逻辑不同：
    # 1. ProcessBench 需要先把一题展开成多个 prefix
    # 2. PRMBench_Preview 则直接对 chosen/rejected pair 打分
    if benchmark_spec.benchmark_type == "processbench":
        metrics, scored_rows, truncation_diagnostics = _eval_processbench(
            benchmark_path=benchmark_path,
            max_samples=args.max_samples,
            backbone=backbone,
            tokenizer=tokenizer,
            value_head=value_head,
            batch_size=int(args.batch_size),
            max_length=int(max_length),
            processbench_f1_threshold=args.processbench_f1_threshold,
            processbench_f1_threshold_candidates=int(args.processbench_f1_threshold_candidates),
            max_truncation_over_limit_fraction=float(args.max_truncation_over_limit_fraction),
            feature_cache_root=Path(args.feature_cache_root),
            feature_cache_mode=str(args.feature_cache_mode),
            lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
            backbone_signature=backbone_signature,
            torch_module=torch,
            feature_cache_stats=feature_cache_stats,
        )
    elif benchmark_spec.benchmark_type == "prmbench_preview":
        metrics, scored_rows, truncation_diagnostics = _eval_prmbench_preview(
            benchmark_path=benchmark_path,
            max_samples=args.max_samples,
            backbone=backbone,
            tokenizer=tokenizer,
            value_head=value_head,
            batch_size=int(args.batch_size),
            max_length=int(max_length),
            prmbench_error_step_index_base=str(args.prmbench_error_step_index_base),
            max_truncation_over_limit_fraction=float(args.max_truncation_over_limit_fraction),
            feature_cache_root=Path(args.feature_cache_root),
            feature_cache_mode=str(args.feature_cache_mode),
            lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
            backbone_signature=backbone_signature,
            torch_module=torch,
            feature_cache_stats=feature_cache_stats,
        )
    else:
        raise ValueError(f"Unsupported benchmark_type: {benchmark_spec.benchmark_type!r}")

    # Persist both:
    # 1. machine-readable JSON outputs for automation
    # 2. compact Markdown for human inspection
    #
    # 同时落两类产物：
    # 1. 机器可读 JSON，方便自动汇总
    # 2. 紧凑 Markdown，方便人工检查
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "benchmark_id": benchmark_spec.benchmark_id,
        "benchmark_type": benchmark_spec.benchmark_type,
        "benchmark_path": str(benchmark_path),
        "value_run_dir": str(args.value_run_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_resolution": checkpoint_resolution,
        "metrics": metrics,
        "truncation_diagnostics": truncation_diagnostics,
        "feature_cache_stats": feature_cache_stats,
        "model_load_elapsed_sec": float(load_elapsed),
    }
    manifest = {
        "artifact_stage": "phase_e_benchmark_eval_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "value_run_dir": str(args.value_run_dir),
        "benchmark_spec": benchmark_spec.to_dict(),
        "benchmark_path": str(benchmark_path),
        "checkpoint_resolution": checkpoint_resolution,
        "processbench_f1_threshold": (
            None if args.processbench_f1_threshold is None else float(args.processbench_f1_threshold)
        ),
        "processbench_f1_threshold_candidates": int(args.processbench_f1_threshold_candidates),
        "prmbench_error_step_index_base": str(args.prmbench_error_step_index_base),
        "max_truncation_over_limit_fraction": float(args.max_truncation_over_limit_fraction),
        "truncation_diagnostics": truncation_diagnostics,
        "output_files": {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "summary_md": str(summary_md_path),
            "scored_rows": str(scored_rows_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md = render_phase_e_benchmark_summary_markdown(
            title=f"Phase E Benchmark Eval: {benchmark_spec.benchmark_id}",
            metadata={
                "value_run_dir": args.value_run_dir,
                "checkpoint_path": checkpoint_path,
                "benchmark_path": benchmark_path,
            },
            metrics=metrics,
    )
    summary_md += _render_text_truncation_markdown(truncation_diagnostics)
    summary_md_path.write_text(summary_md, encoding="utf-8")
    with scored_rows_path.open("w", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("-" * 88)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:18}: {value:.6f}")
        elif not isinstance(value, (dict, list)):
            print(f"{key:18}: {value}")
    print(f"metrics_path      : {metrics_path}")
    print("=" * 88)
    return 0


def _eval_processbench(
    *,
    benchmark_path: Path,
    max_samples: int | None,
    backbone: Any,
    tokenizer: Any,
    value_head: Any,
    batch_size: int,
    max_length: int,
    processbench_f1_threshold: float | None,
    processbench_f1_threshold_candidates: int,
    max_truncation_over_limit_fraction: float,
    feature_cache_root: Path,
    feature_cache_mode: str,
    lock_timeout_sec: float,
    backbone_signature: dict[str, Any],
    torch_module: Any,
    feature_cache_stats: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one checkpoint on ProcessBench-style prefix data.

    中文
    ----
    这里的执行顺序可以理解成：
    1. 先加载 benchmark 原始样本，
    2. 再展开成 prefix 级记录，
    3. 把每个 prefix 编码成特征并打分，
    4. 最后汇总成 ProcessBench 指标。
    """
    examples = load_processbench_examples(benchmark_path, max_samples=max_samples)
    rows = build_processbench_prefix_records(examples)
    texts = [row.input_text() for row in rows]
    truncation_diagnostics = compute_text_truncation_diagnostics(
        texts=texts,
        tokenizer=tokenizer,
        max_length=int(max_length),
        batch_size=int(batch_size),
        group_labels=[
            "good_prefix"
            if bool(row.is_good_prefix)
            else ("first_bad_prefix" if bool(row.is_first_bad_prefix) else "bad_prefix")
            for row in rows
        ],
    )
    validate_text_truncation_diagnostics(
        diagnostics=truncation_diagnostics,
        context_label="Phase E benchmark eval (ProcessBench)",
        max_allowed_over_limit_fraction=float(max_truncation_over_limit_fraction),
    )
    features = load_or_encode_text_features(
        cache_namespace="phase_e_processbench_eval",
        cache_kind="processbench_prefixes",
        texts=texts,
        backbone=backbone,
        tokenizer=tokenizer,
        max_length=int(max_length),
        batch_size=int(batch_size),
        feature_cache_root=feature_cache_root,
        feature_cache_mode=feature_cache_mode,
        lock_timeout_sec=float(lock_timeout_sec),
        backbone_signature=backbone_signature,
        extra_signature={
            "benchmark_path": str(benchmark_path),
            "max_samples": max_samples,
        },
        torch_module=torch_module,
        feature_cache_stats=feature_cache_stats,
    )
    scores = score_feature_tensor(
        value_head=value_head,
        features=features,
        batch_size=int(batch_size),
        torch_module=torch_module,
    )
    metrics = compute_processbench_metrics(
        rows,
        scores,
        processbench_f1_threshold=processbench_f1_threshold,
        processbench_f1_threshold_candidates=int(processbench_f1_threshold_candidates),
    )
    scored_rows = [
        {
            "row_id": row.row_id,
            "example_id": row.example_id,
            "prefix_step_index": row.prefix_step_index,
            "label": row.label,
            "is_good_prefix": row.is_good_prefix,
            "is_first_bad_prefix": row.is_first_bad_prefix,
            "score": float(score),
        }
        for row, score in zip(rows, scores, strict=True)
    ]
    return metrics, scored_rows, truncation_diagnostics


def _eval_prmbench_preview(
    *,
    benchmark_path: Path,
    max_samples: int | None,
    backbone: Any,
    tokenizer: Any,
    value_head: Any,
    batch_size: int,
    max_length: int,
    prmbench_error_step_index_base: str,
    max_truncation_over_limit_fraction: float,
    feature_cache_root: Path,
    feature_cache_mode: str,
    lock_timeout_sec: float,
    backbone_signature: dict[str, Any],
    torch_module: Any,
    feature_cache_stats: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one checkpoint on PRMBench_Preview explicit pairs.

    中文
    ----
    与 ProcessBench 路径不同，这里不需要先展开 prefix 序列，因为输入已经是
    显式 chosen/rejected pair。
    """
    rows = load_prmbench_preview_pairs(
        benchmark_path,
        max_samples=max_samples,
        error_step_index_base=prmbench_error_step_index_base,
    )
    chosen_texts = [row.chosen_input_text() for row in rows]
    rejected_texts = [row.rejected_input_text() for row in rows]
    truncation_diagnostics = compute_text_truncation_diagnostics(
        texts=[*chosen_texts, *rejected_texts],
        tokenizer=tokenizer,
        max_length=int(max_length),
        batch_size=int(batch_size),
        group_labels=(["chosen"] * len(chosen_texts)) + (["rejected"] * len(rejected_texts)),
    )
    validate_text_truncation_diagnostics(
        diagnostics=truncation_diagnostics,
        context_label="Phase E benchmark eval (PRMBench_Preview)",
        max_allowed_over_limit_fraction=float(max_truncation_over_limit_fraction),
    )
    chosen_features = load_or_encode_text_features(
        cache_namespace="phase_e_prmbench_preview_eval",
        cache_kind="prmbench_chosen",
        texts=chosen_texts,
        backbone=backbone,
        tokenizer=tokenizer,
        max_length=int(max_length),
        batch_size=int(batch_size),
        feature_cache_root=feature_cache_root,
        feature_cache_mode=feature_cache_mode,
        lock_timeout_sec=float(lock_timeout_sec),
        backbone_signature=backbone_signature,
        extra_signature={"benchmark_path": str(benchmark_path), "max_samples": max_samples},
        torch_module=torch_module,
        feature_cache_stats=feature_cache_stats,
    )
    rejected_features = load_or_encode_text_features(
        cache_namespace="phase_e_prmbench_preview_eval",
        cache_kind="prmbench_rejected",
        texts=rejected_texts,
        backbone=backbone,
        tokenizer=tokenizer,
        max_length=int(max_length),
        batch_size=int(batch_size),
        feature_cache_root=feature_cache_root,
        feature_cache_mode=feature_cache_mode,
        lock_timeout_sec=float(lock_timeout_sec),
        backbone_signature=backbone_signature,
        extra_signature={"benchmark_path": str(benchmark_path), "max_samples": max_samples},
        torch_module=torch_module,
        feature_cache_stats=feature_cache_stats,
    )
    chosen_scores, rejected_scores = score_pair_features(
        value_head=value_head,
        chosen_features=chosen_features,
        rejected_features=rejected_features,
        batch_size=int(batch_size),
        torch_module=torch_module,
    )
    metrics = compute_pair_ranking_metrics(
        pair_ids=[row.pair_id for row in rows],
        group_keys=[row.classification for row in rows],
        chosen_scores=chosen_scores,
        rejected_scores=rejected_scores,
    )
    scored_rows = [
        {
            "pair_id": row.pair_id,
            "classification": row.classification,
            "error_step_index": row.error_step_index,
            "chosen_score": float(chosen),
            "rejected_score": float(rejected),
            "margin": float(chosen - rejected),
        }
        for row, chosen, rejected in zip(rows, chosen_scores, rejected_scores, strict=True)
    ]
    return metrics, scored_rows, truncation_diagnostics


def _render_text_truncation_markdown(diagnostics: dict[str, Any]) -> str:
    """Render one compact Markdown appendix for text-level truncation diagnostics."""
    overall = dict(diagnostics["overall"])
    lines = [
        "",
        "## Truncation Diagnostics",
        "",
        f"- frac_texts_over_limit: `{float(overall['frac_texts_over_limit']):.6f}`",
        f"- text_length_p95: `{int(overall['text_length']['p95'])}`",
    ]
    by_group = dict(diagnostics.get("by_group", {}))
    if by_group:
        lines.extend(
            [
                "",
                "| group | frac_over_limit | text_length_p95 |",
                "|---|---:|---:|",
            ]
        )
        for group_label, payload in by_group.items():
            lines.append(
                f"| {group_label} | {float(payload['frac_texts_over_limit']):.4f} | "
                f"{int(payload['text_length']['p95'])} |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
