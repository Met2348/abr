#!/usr/bin/env python3
"""Select a low-margin training slice from one Phase E pair artifact.

English
-------
This script prepares the first stage of a selected-relabel experiment:
1. score all training pairs with one existing Phase E value run,
2. rank pairs by low absolute chosen-vs-rejected margin,
3. export a selected low-margin slice plus the untouched remainder.

The judge can then be run only on the selected slice instead of the whole
training artifact.

中文
----
这个脚本负责 selected-relabel 实验的第一步：
1. 用一个已有的 Phase E value run 给训练 pair 全量打分，
2. 按 chosen-vs-rejected 的低绝对 margin 排序，
3. 导出待 judge 的低 margin 子集，以及未触碰的剩余训练集。

这样 judge 只需要处理选中的困难样本，而不是整份训练集。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
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
from ours.phase_d.external_pairs import load_external_pair_jsonl  # noqa: E402
from ours.phase_e.runtime import (  # noqa: E402
    build_phase_e_backbone_signature,
    load_backbone_and_tokenizer,
    load_or_encode_text_features,
    resolve_checkpoint_path,
    score_feature_tensor,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select low-margin train pairs from one Phase E artifact.")
    parser.add_argument("--train-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--eval-pairs-jsonl", type=Path, required=True)
    parser.add_argument("--value-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", choices=["best", "final"], default="best")
    parser.add_argument("--selection-size", type=int, default=64)
    parser.add_argument(
        "--selection-mode",
        choices=["lowest_abs_margin", "lowest_margin"],
        default="lowest_abs_margin",
        help="`lowest_abs_margin` captures uncertain pairs; `lowest_margin` focuses on outright contradictions first.",
    )
    parser.add_argument("--run-name", default="phase_e_selected_relabel_slice")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_pairs"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-gpu-memory-gib", type=int, default=None)
    parser.add_argument("--max-cpu-memory-gib", type=int, default=None)
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
    args = _build_parser().parse_args(argv)
    if not args.train_pairs_jsonl.exists():
        raise FileNotFoundError(f"--train-pairs-jsonl not found: {args.train_pairs_jsonl}")
    if not args.eval_pairs_jsonl.exists():
        raise FileNotFoundError(f"--eval-pairs-jsonl not found: {args.eval_pairs_jsonl}")
    if not args.value_run_dir.exists():
        raise FileNotFoundError(f"--value-run-dir not found: {args.value_run_dir}")
    if int(args.selection_size) <= 0:
        raise ValueError("--selection-size must be > 0")
    if int(args.max_length) <= 0:
        raise ValueError("--max-length must be > 0")
    return args


def _selection_key(*, margin: float, mode: str) -> tuple[float, float]:
    """Return sort key for one selected-relabel difficulty policy.

    English
    -------
    `lowest_abs_margin` targets uncertain pairs regardless of sign.
    `lowest_margin` first pulls pairs the current model explicitly gets wrong.

    中文
    ----
    `lowest_abs_margin` 关注接近打平的不确定 pair。
    `lowest_margin` 会优先挑出当前模型已经判反的 pair。
    """
    if mode == "lowest_margin":
        return (float(margin), abs(float(margin)))
    return (abs(float(margin)), float(margin))


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * q))))
    return float(sorted_values[idx])


def _annotate_pair_dict(pair_dict: dict[str, Any], *, margin: float, rank: int, selected: bool) -> dict[str, Any]:
    payload = json.loads(json.dumps(pair_dict, ensure_ascii=False))
    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "selected_relabel_baseline_margin": float(margin),
            "selected_relabel_baseline_abs_margin": abs(float(margin)),
            "selected_relabel_margin_rank": int(rank),
            "selected_relabel_selected": bool(selected),
        }
    )
    payload["metadata"] = metadata
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    train_pairs, train_summary = load_external_pair_jsonl(Path(args.train_pairs_jsonl))
    run_manifest = json.loads((Path(args.value_run_dir) / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = resolve_checkpoint_path(
        value_run_dir=Path(args.value_run_dir),
        run_manifest=run_manifest,
        checkpoint_name=str(args.checkpoint_name),
    )
    value_head, _, _ = load_value_head_checkpoint(checkpoint_path)
    value_head.eval()
    model_meta = dict(run_manifest.get("resolved_backbone") or {})
    model_path = str(model_meta.get("model_path") or "").strip()
    if model_path == "":
        raise ValueError(f"Resolved model_path missing in manifest: {args.value_run_dir}")
    adapter_path_text = str(model_meta.get("adapter_path") or "").strip()
    adapter_path = Path(adapter_path_text) if adapter_path_text else None

    backbone, tokenizer, _, tokenizer_path = load_backbone_and_tokenizer(
        model_path=model_path,
        adapter_path=adapter_path,
        dtype_name=str(args.dtype),
        device_map=str(args.device_map),
        max_gpu_memory_gib=(int(args.max_gpu_memory_gib) if args.max_gpu_memory_gib is not None else None),
        max_cpu_memory_gib=(int(args.max_cpu_memory_gib) if args.max_cpu_memory_gib is not None else None),
        torch_module=torch,
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
    )
    value_head.to(next(backbone.parameters()).device)

    backbone_signature = build_phase_e_backbone_signature(
        model_path=str(model_path),
        adapter_path=(str(adapter_path) if adapter_path is not None else None),
        tokenizer_path=str(tokenizer_path),
        dtype=str(args.dtype),
        max_length=int(args.max_length),
    )
    feature_cache_stats: dict[str, Any] = {"hits": 0, "misses": 0, "writes": 0, "entries": {}}

    chosen_features = load_or_encode_text_features(
        cache_namespace="phase_e_selected_relabel",
        cache_kind="train_chosen",
        texts=[pair.chosen_input_text() for pair in train_pairs],
        backbone=backbone,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        feature_cache_root=Path(args.feature_cache_root),
        feature_cache_mode=str(args.feature_cache_mode),
        lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
        backbone_signature=backbone_signature,
        extra_signature={
            "train_pairs_jsonl": str(args.train_pairs_jsonl),
            "value_run_dir": str(args.value_run_dir),
            "checkpoint_path": str(checkpoint_path),
            "selection_mode": str(args.selection_mode),
        },
        torch_module=torch,
        feature_cache_stats=feature_cache_stats,
    )
    rejected_features = load_or_encode_text_features(
        cache_namespace="phase_e_selected_relabel",
        cache_kind="train_rejected",
        texts=[pair.rejected_input_text() for pair in train_pairs],
        backbone=backbone,
        tokenizer=tokenizer,
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        feature_cache_root=Path(args.feature_cache_root),
        feature_cache_mode=str(args.feature_cache_mode),
        lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
        backbone_signature=backbone_signature,
        extra_signature={
            "train_pairs_jsonl": str(args.train_pairs_jsonl),
            "value_run_dir": str(args.value_run_dir),
            "checkpoint_path": str(checkpoint_path),
            "selection_mode": str(args.selection_mode),
        },
        torch_module=torch,
        feature_cache_stats=feature_cache_stats,
    )

    chosen_scores = score_feature_tensor(
        value_head=value_head,
        features=chosen_features,
        batch_size=int(args.batch_size),
        torch_module=torch,
    )
    rejected_scores = score_feature_tensor(
        value_head=value_head,
        features=rejected_features,
        batch_size=int(args.batch_size),
        torch_module=torch,
    )

    pair_payloads = []
    for idx, (pair, chosen_score, rejected_score) in enumerate(zip(train_pairs, chosen_scores, rejected_scores, strict=True)):
        margin = float(chosen_score - rejected_score)
        pair_payloads.append(
            {
                "index": int(idx),
                "pair": pair,
                "pair_dict": pair.to_dict(),
                "chosen_score": float(chosen_score),
                "rejected_score": float(rejected_score),
                "margin": float(margin),
                "abs_margin": abs(float(margin)),
                "selection_key": _selection_key(margin=float(margin), mode=str(args.selection_mode)),
            }
        )

    pair_payloads.sort(key=lambda item: item["selection_key"])
    selected_size = min(int(args.selection_size), len(pair_payloads))
    selected_indices = {int(item["index"]) for item in pair_payloads[:selected_size]}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root) / f"{args.run_name}__{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    selected_path = run_dir / "selected_train_pairs.jsonl"
    retained_path = run_dir / "retained_train_pairs.jsonl"
    validation_path = run_dir / "validation_pairs.jsonl"
    ranked_rows_path = run_dir / "ranked_rows.jsonl"

    with selected_path.open("w", encoding="utf-8") as selected_handle, retained_path.open("w", encoding="utf-8") as retained_handle:
        for rank, item in enumerate(pair_payloads):
            record = _annotate_pair_dict(
                item["pair_dict"],
                margin=float(item["margin"]),
                rank=int(rank),
                selected=bool(item["index"] in selected_indices),
            )
            target = selected_handle if item["index"] in selected_indices else retained_handle
            target.write(json.dumps(record, ensure_ascii=False) + "\n")

    validation_path.write_text(Path(args.eval_pairs_jsonl).read_text(encoding="utf-8"), encoding="utf-8")
    with ranked_rows_path.open("w", encoding="utf-8") as handle:
        for rank, item in enumerate(pair_payloads):
            handle.write(
                json.dumps(
                    {
                        "rank": int(rank),
                        "pair_id": item["pair"].pair_id,
                        "source_tag": item["pair"].source_tag,
                        "pair_semantics": str((item["pair"].metadata or {}).get("pair_semantics", "unspecified")),
                        "chosen_score": float(item["chosen_score"]),
                        "rejected_score": float(item["rejected_score"]),
                        "margin": float(item["margin"]),
                        "abs_margin": float(item["abs_margin"]),
                        "selected": bool(item["index"] in selected_indices),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    sorted_margins = sorted(float(item["margin"]) for item in pair_payloads)
    sorted_abs_margins = sorted(float(item["abs_margin"]) for item in pair_payloads)
    selected_payloads = pair_payloads[:selected_size]
    selected_margin_values = [float(item["margin"]) for item in selected_payloads]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "train_pairs_jsonl": str(args.train_pairs_jsonl),
        "eval_pairs_jsonl": str(args.eval_pairs_jsonl),
        "value_run_dir": str(args.value_run_dir),
        "checkpoint_path": str(checkpoint_path),
        "selection_mode": str(args.selection_mode),
        "selection_size": int(selected_size),
        "num_train_pairs": int(len(pair_payloads)),
        "feature_cache_stats": feature_cache_stats,
        "train_pair_summary": train_summary,
        "margin_summary": {
            "mean_margin": float(statistics.mean(sorted_margins)) if sorted_margins else 0.0,
            "median_margin": float(statistics.median(sorted_margins)) if sorted_margins else 0.0,
            "p05_margin": _quantile(sorted_margins, 0.05),
            "p50_margin": _quantile(sorted_margins, 0.50),
            "p95_margin": _quantile(sorted_margins, 0.95),
            "p05_abs_margin": _quantile(sorted_abs_margins, 0.05),
            "p50_abs_margin": _quantile(sorted_abs_margins, 0.50),
            "p95_abs_margin": _quantile(sorted_abs_margins, 0.95),
            "num_negative_margin_pairs": int(sum(float(item["margin"]) <= 0.0 for item in pair_payloads)),
        },
        "selected_margin_summary": {
            "mean_margin": float(statistics.mean(selected_margin_values)) if selected_margin_values else 0.0,
            "median_margin": float(statistics.median(selected_margin_values)) if selected_margin_values else 0.0,
            "min_margin": float(min(selected_margin_values)) if selected_margin_values else 0.0,
            "max_margin": float(max(selected_margin_values)) if selected_margin_values else 0.0,
            "num_negative_margin_pairs": int(sum(value <= 0.0 for value in selected_margin_values)),
        },
        "output_files": {
            "selected_train_pairs": str(selected_path),
            "retained_train_pairs": str(retained_path),
            "validation_pairs": str(validation_path),
            "ranked_rows": str(ranked_rows_path),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_stage": "phase_e_selected_relabel_slice_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "train_pairs_jsonl": str(args.train_pairs_jsonl),
                "eval_pairs_jsonl": str(args.eval_pairs_jsonl),
                "value_run_dir": str(args.value_run_dir),
                "checkpoint_path": str(checkpoint_path),
                "selection_mode": str(args.selection_mode),
                "selection_size": int(selected_size),
                "train_pair_summary": train_summary,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_lines = [
        "# Phase E Selected Relabel Slice",
        "",
        f"- train_pairs_jsonl: `{args.train_pairs_jsonl}`",
        f"- value_run_dir: `{args.value_run_dir}`",
        f"- checkpoint_path: `{checkpoint_path}`",
        f"- selection_mode: `{args.selection_mode}`",
        f"- selection_size: `{selected_size}`",
        f"- num_train_pairs: `{len(pair_payloads)}`",
        f"- negative_margin_pairs: `{summary['margin_summary']['num_negative_margin_pairs']}`",
        f"- selected_negative_margin_pairs: `{summary['selected_margin_summary']['num_negative_margin_pairs']}`",
        f"- p05_abs_margin: `{summary['margin_summary']['p05_abs_margin']:.6f}`",
        f"- selected_median_margin: `{summary['selected_margin_summary']['median_margin']:.6f}`",
        "",
        "## Outputs",
        "",
        f"- selected_train_pairs: `{selected_path}`",
        f"- retained_train_pairs: `{retained_path}`",
        f"- validation_pairs: `{validation_path}`",
        f"- ranked_rows: `{ranked_rows_path}`",
    ]
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("=" * 88)
    print("Phase E: Selected Relabel Margin Slice")
    print("=" * 88)
    print(f"train_pairs_jsonl     : {args.train_pairs_jsonl}")
    print(f"value_run_dir         : {args.value_run_dir}")
    print(f"checkpoint_path       : {checkpoint_path}")
    print(f"selection_mode        : {args.selection_mode}")
    print(f"selection_size        : {selected_size}")
    print(f"num_train_pairs       : {len(pair_payloads)}")
    print(f"negative_margin_pairs : {summary['margin_summary']['num_negative_margin_pairs']}")
    print(f"selected_negative     : {summary['selected_margin_summary']['num_negative_margin_pairs']}")
    print(f"selected_path         : {selected_path}")
    print(f"retained_path         : {retained_path}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
