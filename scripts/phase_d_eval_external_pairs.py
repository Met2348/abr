#!/usr/bin/env python3
"""Evaluate value-head ranking on external held-out pair artifacts.

Why this file exists
--------------------
Phase D6-T needs a direct ranking gate on high-confidence external pairs.
`phase_b_eval_faithfulness.py` focuses on C1 corruption artifacts, while this
script scores canonical external pairs (`chosen` vs `rejected`) directly.

What this file does
-------------------
1. Load one trained C2 value-head run (`manifest.json` + checkpoint).
2. Load external validation pairs (`validation_pairs.jsonl`).
3. Encode chosen/rejected texts with frozen backbone (batched + cache + OOM backoff).
4. Score both sides with value head and compute ranking metrics.
5. Persist machine-readable metrics and per-pair scores for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    """Add repo-local `src/` to `sys.path` for script execution."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_b.faithfulness_eval import compute_binary_auc  # noqa: E402
from ours.phase_b.feature_cache import (  # noqa: E402
    build_backbone_signature,
    build_cache_key,
    feature_cache_can_read,
    feature_cache_can_write,
    hash_jsonable,
    hash_text_list,
    move_tensors_to_device,
    save_feature_cache,
    try_load_feature_cache,
    validate_feature_cache_mode,
)
from ours.phase_b.value_head import (  # noqa: E402
    encode_text_features,
    ensure_tokenizer_has_pad_token,
    load_value_head_checkpoint,
    maybe_resize_embeddings_for_tokenizer,
)
from ours.phase_d.external_pairs import ExternalPairRecord, load_external_pair_jsonl  # noqa: E402
from ours.phase_e.training import (  # noqa: E402
    compute_pair_truncation_diagnostics,
    validate_pair_truncation_diagnostics,
)


@dataclass(slots=True)
class EvalRuntimeConfig:
    """Compact runtime snapshot persisted into eval manifest."""

    value_run_dir: str
    external_pair_jsonl: str
    checkpoint_name: str
    max_samples: int | None
    min_confidence: float
    allowed_sources: str | None
    allowed_domains: str | None
    batch_size: int
    max_length: int
    dtype: str
    device_map: str
    require_cuda: bool
    feature_cache_root: str
    feature_cache_mode: str
    feature_cache_lock_timeout_sec: float
    max_truncation_over_limit_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_run_dir": self.value_run_dir,
            "external_pair_jsonl": self.external_pair_jsonl,
            "checkpoint_name": self.checkpoint_name,
            "max_samples": self.max_samples,
            "min_confidence": self.min_confidence,
            "allowed_sources": self.allowed_sources,
            "allowed_domains": self.allowed_domains,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "dtype": self.dtype,
            "device_map": self.device_map,
            "require_cuda": self.require_cuda,
            "feature_cache_root": self.feature_cache_root,
            "feature_cache_mode": self.feature_cache_mode,
            "feature_cache_lock_timeout_sec": self.feature_cache_lock_timeout_sec,
            "max_truncation_over_limit_fraction": self.max_truncation_over_limit_fraction,
        }


def _build_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for external-pair ranking eval."""
    parser = argparse.ArgumentParser(
        description="Evaluate one value-head run on external held-out pair artifacts."
    )
    parser.add_argument("--value-run-dir", type=Path, required=True)
    parser.add_argument("--external-pair-jsonl", type=Path, required=True)
    parser.add_argument("--run-name", default="phase_d_triplet_eval")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_d_triplet_eval"),
    )
    parser.add_argument("--checkpoint-name", choices=["best", "final"], default="best")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--allowed-sources", default="")
    parser.add_argument("--allowed-domains", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional override; default uses C2 train config max_length.",
    )
    parser.add_argument(
        "--dtype",
        default="",
        help="Optional override; empty string uses C2 manifest dtype.",
    )
    parser.add_argument(
        "--device-map",
        default="",
        help="Optional override; empty string uses C2 manifest device_map.",
    )
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--feature-cache-root",
        type=Path,
        default=Path("assets/artifacts/phase_c_feature_cache"),
        help="Persistent cache root for encoded chosen/rejected external texts.",
    )
    parser.add_argument(
        "--feature-cache-mode",
        choices=["off", "read", "write", "read_write"],
        default="read_write",
    )
    parser.add_argument("--feature-cache-lock-timeout-sec", type=float, default=600.0)
    parser.add_argument(
        "--max-truncation-over-limit-fraction",
        type=float,
        default=0.10,
        help=(
            "Fail fast if more than this fraction of external pairs exceed max_length. "
            "This prevents heavily truncated evals from looking like genuine model failures."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate CLI arguments."""
    args = _build_parser().parse_args(argv)
    if not args.value_run_dir.exists():
        raise FileNotFoundError(f"--value-run-dir not found: {args.value_run_dir}")
    if not args.external_pair_jsonl.exists():
        raise FileNotFoundError(f"--external-pair-jsonl not found: {args.external_pair_jsonl}")
    if args.max_samples is not None and int(args.max_samples) <= 0:
        raise ValueError("--max-samples must be > 0")
    if not (0.0 <= float(args.min_confidence) <= 1.0):
        raise ValueError("--min-confidence must be in [0, 1]")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_length is not None and int(args.max_length) <= 8:
        raise ValueError("--max-length must be > 8")
    if not (0.0 <= float(args.max_truncation_over_limit_fraction) <= 1.0):
        raise ValueError("--max-truncation-over-limit-fraction must be in [0, 1]")
    if float(args.feature_cache_lock_timeout_sec) <= 0.0:
        raise ValueError("--feature-cache-lock-timeout-sec must be > 0")
    args.feature_cache_mode = validate_feature_cache_mode(str(args.feature_cache_mode))
    return args


def main(argv: list[str] | None = None) -> int:
    """Main entrypoint for external pair ranking eval."""
    args = parse_args(argv)
    manifest_path = args.value_run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing C2 manifest: {manifest_path}")
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    checkpoint_path = _resolve_checkpoint_path(
        value_run_dir=args.value_run_dir,
        run_manifest=run_manifest,
        checkpoint_name=str(args.checkpoint_name),
    )
    value_head, _, _ = load_value_head_checkpoint(checkpoint_path)

    allowed_sources = _parse_csv_allow_list(str(args.allowed_sources))
    allowed_domains = _parse_csv_allow_list(str(args.allowed_domains))
    external_pairs, external_stats = load_external_pair_jsonl(
        args.external_pair_jsonl,
        max_samples=args.max_samples,
        min_confidence=float(args.min_confidence),
        allowed_sources=allowed_sources,
        allowed_domains=allowed_domains,
    )
    if not external_pairs:
        raise RuntimeError("No external pairs survived filtering")

    torch, AutoModelForCausalLM, AutoTokenizer = _import_runtime_deps()
    if bool(args.require_cuda) and not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA is required by --require-cuda but no GPU is visible")

    resolved_backbone = dict(run_manifest.get("resolved_backbone", {}))
    model_path = str(resolved_backbone.get("model_path", "")).strip()
    adapter_path = resolved_backbone.get("adapter_path")
    if model_path == "":
        raise ValueError("C2 manifest missing resolved_backbone.model_path")

    dtype_name = str(args.dtype).strip() or str(resolved_backbone.get("dtype", "bfloat16"))
    device_map = str(args.device_map).strip() or str(resolved_backbone.get("device_map", "auto"))
    max_length = int(args.max_length) if args.max_length is not None else int(
        run_manifest.get("train_config", {}).get("max_length", 1024)
    )

    tokenizer_path = _resolve_tokenizer_load_path(
        model_path=model_path,
        adapter_path=(Path(str(adapter_path)) if adapter_path else None),
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    synthesized_pad_token = ensure_tokenizer_has_pad_token(tokenizer)

    model_load_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": True,
    }
    resolved_dtype = _resolve_dtype(dtype_name, torch)
    from_pretrained_sig = __import__("inspect").signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in from_pretrained_sig.parameters:
        model_load_kwargs["dtype"] = resolved_dtype
    else:
        model_load_kwargs["torch_dtype"] = resolved_dtype

    print("=" * 88)
    print("Phase D: External Pair Ranking Eval")
    print("=" * 88)
    print(f"value_run_dir     : {args.value_run_dir}")
    print(f"checkpoint        : {checkpoint_path}")
    print(f"external_pairs    : {len(external_pairs)}")
    print(f"external_sources  : {external_stats.get('by_source', {})}")
    print(f"model_path        : {model_path}")
    print(f"adapter_path      : {adapter_path if adapter_path else '<none>'}")
    print(f"dtype             : {dtype_name}")
    print(f"device_map        : {device_map}")
    print(f"batch_size        : {args.batch_size}")
    print(f"max_length        : {max_length}")
    print(f"trunc_overlimit_max: {float(args.max_truncation_over_limit_fraction):.4f}")
    print(f"feature_cache_mode: {args.feature_cache_mode}")
    print(f"feature_cache_root: {args.feature_cache_root}")
    print("=" * 88)

    model_load_start = time.perf_counter()
    backbone = AutoModelForCausalLM.from_pretrained(str(model_path), **model_load_kwargs)
    if synthesized_pad_token:
        maybe_resize_embeddings_for_tokenizer(backbone=backbone, tokenizer=tokenizer)
    if adapter_path:
        backbone = _attach_peft_adapter_for_inference(backbone, Path(str(adapter_path)))
    backbone.eval()
    model_load_elapsed = time.perf_counter() - model_load_start

    # Value head is tiny; scoring on one device is enough.
    score_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value_head.to(score_device)
    value_head.eval()

    chosen_texts = [pair.chosen_input_text() for pair in external_pairs]
    rejected_texts = [pair.rejected_input_text() for pair in external_pairs]
    truncation_diagnostics = compute_pair_truncation_diagnostics(
        pairs=external_pairs,
        tokenizer=tokenizer,
        max_length=int(max_length),
        batch_size=int(args.batch_size),
    )
    validate_pair_truncation_diagnostics(
        diagnostics=truncation_diagnostics,
        context_label="Phase D external pair eval",
        max_allowed_over_limit_fraction=float(args.max_truncation_over_limit_fraction),
    )

    feature_cache_root = Path(args.feature_cache_root)
    backbone_signature = build_backbone_signature(
        model_path=model_path,
        adapter_path=(str(adapter_path) if adapter_path else None),
        tokenizer_path=tokenizer_path,
        dtype=dtype_name,
        max_length=int(max_length),
    )
    feature_cache_stats = {
        "mode": str(args.feature_cache_mode),
        "root": str(feature_cache_root),
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "entries": {},
    }

    encode_start = time.perf_counter()
    chosen_features = _load_or_encode_external_text_features(
        cache_kind="chosen",
        texts=chosen_texts,
        backbone=backbone,
        tokenizer=tokenizer,
        max_length=int(max_length),
        batch_size=int(args.batch_size),
        feature_cache_root=feature_cache_root,
        feature_cache_mode=str(args.feature_cache_mode),
        lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
        backbone_signature=backbone_signature,
        extra_signature={
            "external_pair_jsonl": str(args.external_pair_jsonl),
            "max_samples": args.max_samples,
            "min_confidence": float(args.min_confidence),
            "allowed_sources": sorted(allowed_sources) if allowed_sources else None,
            "allowed_domains": sorted(allowed_domains) if allowed_domains else None,
        },
        torch_module=torch,
        feature_cache_stats=feature_cache_stats,
    )
    rejected_features = _load_or_encode_external_text_features(
        cache_kind="rejected",
        texts=rejected_texts,
        backbone=backbone,
        tokenizer=tokenizer,
        max_length=int(max_length),
        batch_size=int(args.batch_size),
        feature_cache_root=feature_cache_root,
        feature_cache_mode=str(args.feature_cache_mode),
        lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
        backbone_signature=backbone_signature,
        extra_signature={
            "external_pair_jsonl": str(args.external_pair_jsonl),
            "max_samples": args.max_samples,
            "min_confidence": float(args.min_confidence),
            "allowed_sources": sorted(allowed_sources) if allowed_sources else None,
            "allowed_domains": sorted(allowed_domains) if allowed_domains else None,
        },
        torch_module=torch,
        feature_cache_stats=feature_cache_stats,
    )
    encode_elapsed = time.perf_counter() - encode_start

    chosen_scores, rejected_scores = _score_pair_features(
        value_head=value_head,
        chosen_features=chosen_features,
        rejected_features=rejected_features,
        batch_size=int(args.batch_size),
        torch_module=torch,
    )
    margins = [float(c - r) for c, r in zip(chosen_scores, rejected_scores, strict=True)]
    pair_acc = float(sum(1 for margin in margins if margin > 0.0) / len(margins))
    pair_acc_ties = float(sum(1 for margin in margins if margin >= 0.0) / len(margins))
    auc = float(
        compute_binary_auc(
            scores=[*chosen_scores, *rejected_scores],
            labels=[1] * len(chosen_scores) + [0] * len(rejected_scores),
        )
    )

    pair_rows = _build_pair_rows(
        pairs=external_pairs,
        chosen_scores=chosen_scores,
        rejected_scores=rejected_scores,
    )
    by_source = _group_margin_metrics(pair_rows, key="source_tag")
    by_domain = _group_margin_metrics(pair_rows, key="domain_tag")

    run_dir = args.output_root / f"{args.run_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pair_scores_path = run_dir / "pair_scores.jsonl"
    metrics_path = run_dir / "metrics.json"
    manifest_out_path = run_dir / "manifest.json"
    summary_md_path = run_dir / "summary.md"

    _write_jsonl(pair_scores_path, pair_rows)
    metrics = {
        "n_pairs": int(len(pair_rows)),
        "pair_accuracy": float(pair_acc),
        "pair_accuracy_with_ties": float(pair_acc_ties),
        "auc_chosen_vs_rejected": float(auc),
        "mean_margin": float(statistics.mean(margins)),
        "median_margin": float(statistics.median(margins)),
        "by_source": by_source,
        "by_domain": by_domain,
        "truncation_diagnostics": truncation_diagnostics,
        "model_load_seconds": float(model_load_elapsed),
        "encode_seconds": float(encode_elapsed),
        "feature_cache": feature_cache_stats,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    runtime_cfg = EvalRuntimeConfig(
        value_run_dir=str(args.value_run_dir),
        external_pair_jsonl=str(args.external_pair_jsonl),
        checkpoint_name=str(args.checkpoint_name),
        max_samples=(int(args.max_samples) if args.max_samples is not None else None),
        min_confidence=float(args.min_confidence),
        allowed_sources=(",".join(sorted(allowed_sources)) if allowed_sources else None),
        allowed_domains=(",".join(sorted(allowed_domains)) if allowed_domains else None),
        batch_size=int(args.batch_size),
        max_length=int(max_length),
        dtype=str(dtype_name),
        device_map=str(device_map),
        require_cuda=bool(args.require_cuda),
        feature_cache_root=str(feature_cache_root),
        feature_cache_mode=str(args.feature_cache_mode),
        feature_cache_lock_timeout_sec=float(args.feature_cache_lock_timeout_sec),
        max_truncation_over_limit_fraction=float(args.max_truncation_over_limit_fraction),
    )
    manifest_out = {
        "artifact_stage": "phase_d_external_pair_eval",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/phase_d_eval_external_pairs.py",
        "runtime_config": runtime_cfg.to_dict(),
        "resolved_backbone": {
            "model_path": model_path,
            "adapter_path": adapter_path,
            "tokenizer_path": tokenizer_path,
        },
        "truncation_diagnostics": truncation_diagnostics,
        "value_run_manifest": str(manifest_path),
        "value_head_checkpoint": str(checkpoint_path),
        "output_files": {
            "pair_scores": str(pair_scores_path),
            "metrics": str(metrics_path),
            "summary_md": str(summary_md_path),
        },
    }
    manifest_out_path.write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_lines = [
        "# Phase D External Pair Eval Summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- value_run_dir: `{args.value_run_dir}`",
        f"- checkpoint: `{checkpoint_path}`",
        f"- external_pair_jsonl: `{args.external_pair_jsonl}`",
        f"- n_pairs: `{metrics['n_pairs']}`",
        f"- pair_accuracy: `{metrics['pair_accuracy']:.6f}`",
        f"- pair_accuracy_with_ties: `{metrics['pair_accuracy_with_ties']:.6f}`",
        f"- auc_chosen_vs_rejected: `{metrics['auc_chosen_vs_rejected']:.6f}`",
        f"- mean_margin: `{metrics['mean_margin']:.6f}`",
        f"- median_margin: `{metrics['median_margin']:.6f}`",
        "",
        "## Truncation Diagnostics",
        "",
        f"- frac_pairs_over_limit: `{float(truncation_diagnostics['overall']['frac_pairs_over_limit']):.6f}`",
        f"- frac_pairs_identical_after_truncation: `{float(truncation_diagnostics['overall']['frac_pairs_identical_after_truncation']):.6f}`",
        f"- frac_pairs_first_diff_after_cutoff: `{float(truncation_diagnostics['overall']['frac_pairs_first_diff_after_cutoff']):.6f}`",
        "",
        "| source | frac_over_limit | frac_collapse_after_cut | frac_hidden_diff_after_cut |",
        "|---|---:|---:|---:|",
    ]
    for source, payload in sorted(truncation_diagnostics.get("by_source", {}).items()):
        summary_lines.append(
            f"| {source} | {float(payload['frac_pairs_over_limit']):.4f} | "
            f"{float(payload['frac_pairs_identical_after_truncation']):.4f} | "
            f"{float(payload['frac_pairs_first_diff_after_cutoff']):.4f} |"
        )
    summary_lines.extend(
        [
            "",
        "## By Source",
        "",
        "| source | n | pair_acc | auc | mean_margin |",
        "|---|---:|---:|---:|---:|",
        ]
    )
    for source, payload in sorted(by_source.items()):
        summary_lines.append(
            f"| {source} | {payload['n']} | {payload['pair_accuracy']:.4f} | "
            f"{payload['auc']:.4f} | {payload['mean_margin']:.4f} |"
        )
    summary_lines.append("")
    summary_md_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("-" * 88)
    print(f"pair_accuracy     : {metrics['pair_accuracy']:.6f}")
    print(f"pair_acc_with_tie : {metrics['pair_accuracy_with_ties']:.6f}")
    print(f"auc               : {metrics['auc_chosen_vs_rejected']:.6f}")
    print(f"mean_margin       : {metrics['mean_margin']:.6f}")
    print(f"median_margin     : {metrics['median_margin']:.6f}")
    print(f"metrics_path      : {metrics_path}")
    print(f"summary_md        : {summary_md_path}")
    print("=" * 88)
    return 0


def _resolve_checkpoint_path(*, value_run_dir: Path, run_manifest: dict[str, Any], checkpoint_name: str) -> Path:
    """Resolve checkpoint path from run manifest with safe fallback logic."""
    files = dict(run_manifest.get("output_files", {}))
    if checkpoint_name == "best":
        best_path = files.get("best_value_head")
        if isinstance(best_path, str) and best_path.strip():
            path = Path(best_path)
            if path.exists():
                return path
        # fallback keeps this script robust for runs without best checkpoint saved.
        # RISK WARNING:
        # When this fallback triggers, the operator asked for "best" but the
        # script evaluates the final checkpoint instead.  That behavior is easy
        # to miss in retrospective experiment analysis.
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


def _parse_csv_allow_list(text: str) -> set[str] | None:
    """Parse comma-separated allow list string into normalized set."""
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    return set(values) if values else None


def _resolve_dtype(dtype_name: str, torch_module: Any):
    """Resolve dtype name into a torch dtype object."""
    normalized = str(dtype_name).strip().lower()
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name!r}")
    return mapping[normalized]


def _resolve_tokenizer_load_path(model_path: str, adapter_path: Path | None) -> str:
    """Resolve tokenizer path with adapter override fallback."""
    if adapter_path is None:
        return model_path
    if (adapter_path / "tokenizer_config.json").exists():
        return str(adapter_path)
    return model_path


def _attach_peft_adapter_for_inference(model: Any, adapter_path: Path):
    """Attach one PEFT adapter to a loaded base model."""
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    try:
        from peft import PeftModel  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Failed to import `peft` while attaching adapter for eval") from exc
    return PeftModel.from_pretrained(model, str(adapter_path))


def _load_or_encode_external_text_features(
    *,
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
    """Load cached encoded features, or encode with batched OOM-safe fallback."""
    signature_payload = {
        "cache_kind": "phase_d_external_pair_eval_features",
        "cache_side": str(cache_kind),
        "texts_digest": hash_text_list(texts),
        "backbone_signature": backbone_signature,
        "extra_signature_hash": hash_jsonable(extra_signature),
        "max_length": int(max_length),
    }
    cache_key, signature_hash = build_cache_key("phase_d_external_pair_eval", signature_payload)

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
                if int(features.shape[0]) != len(texts):
                    raise ValueError(
                        f"cached feature count mismatch: got {features.shape[0]}, expected {len(texts)}"
                    )
                feature_cache_stats["hits"] += 1
                feature_cache_stats["entries"][f"{cache_kind}_features"] = {
                    "hit": True,
                    "cache_key": cache_key,
                }
                print(f"feature_cache    : {cache_kind}_features hit ({cache_key})", flush=True)
                return features
            except Exception as exc:  # noqa: BLE001
                print(
                    f"feature_cache    : {cache_kind}_features invalid payload, fallback to re-encode ({exc})",
                    flush=True,
                )

    feature_cache_stats["misses"] += 1
    feature_cache_stats["entries"][f"{cache_kind}_features"] = {
        "hit": False,
        "cache_key": cache_key,
    }
    print(f"feature_cache    : {cache_kind}_features miss ({cache_key})", flush=True)
    encoded = _encode_texts_batched_oom_safe(
        texts=texts,
        backbone=backbone,
        tokenizer=tokenizer,
        max_length=int(max_length),
        batch_size=int(batch_size),
        torch_module=torch_module,
        log_prefix=f"cache_{cache_kind}_texts",
    )
    if feature_cache_can_write(feature_cache_mode):
        save_feature_cache(
            cache_root=feature_cache_root,
            cache_key=cache_key,
            signature_hash=signature_hash,
            payload={"features": encoded},
            torch_module=torch_module,
            producer="scripts/phase_d_eval_external_pairs.py",
            lock_timeout_sec=float(lock_timeout_sec),
            extra_metadata={"num_texts": int(len(texts)), "cache_side": str(cache_kind)},
        )
        feature_cache_stats["writes"] += 1
        feature_cache_stats["entries"][f"{cache_kind}_features"]["write"] = True
    return encoded


def _encode_texts_batched_oom_safe(
    *,
    texts: list[str],
    backbone: Any,
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    torch_module: Any,
    log_prefix: str,
):
    """Encode texts in batches with OOM backoff for robustness."""
    if not texts:
        raise ValueError("encode_texts requires non-empty texts")

    outputs: list[Any] = []
    total = len(texts)
    index = 0
    current_batch_size = int(max(batch_size, 1))
    progress_every = max(1, total // 6)
    start = time.perf_counter()
    print(
        f"{log_prefix:16}: start {total} texts (bs={current_batch_size}, max_length={int(max_length)})",
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
        outputs.append(batch_features.detach())
        index += effective_bs
        if index % progress_every == 0 or index >= total:
            elapsed = max(time.perf_counter() - start, 1e-9)
            rate = index / elapsed
            print(
                f"{log_prefix:16}: {index}/{total} ({100.0 * index / total:.1f}%) | "
                f"elapsed={elapsed:.1f}s | rate={rate:.3f} text/s",
                flush=True,
            )
    return torch_module.cat(outputs, dim=0)


def _score_pair_features(
    *,
    value_head: Any,
    chosen_features: Any,
    rejected_features: Any,
    batch_size: int,
    torch_module: Any,
) -> tuple[list[float], list[float]]:
    """Score chosen/rejected features in batches with OOM backoff."""
    if chosen_features.shape != rejected_features.shape:
        raise ValueError("chosen_features and rejected_features must have same shape")
    total = int(chosen_features.shape[0])
    head_device = next(value_head.parameters()).device
    head_dtype = next(value_head.parameters()).dtype
    chosen_scores: list[float] = []
    rejected_scores: list[float] = []
    index = 0
    current_batch_size = int(max(batch_size, 1))
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


def _build_pair_rows(
    *,
    pairs: list[ExternalPairRecord],
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> list[dict[str, Any]]:
    """Build per-pair scored rows for JSONL output."""
    rows: list[dict[str, Any]] = []
    for pair, chosen, rejected in zip(pairs, chosen_scores, rejected_scores, strict=True):
        rows.append(
            {
                "pair_id": pair.pair_id,
                "source_tag": pair.source_tag,
                "domain_tag": pair.domain_tag,
                "pair_confidence": float(pair.pair_confidence),
                "chosen_score": float(chosen),
                "rejected_score": float(rejected),
                "margin": float(chosen - rejected),
            }
        )
    return rows


def _group_margin_metrics(rows: list[dict[str, Any]], *, key: str) -> dict[str, dict[str, Any]]:
    """Aggregate pair metrics by one categorical key."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_key = str(row.get(key, ""))
        groups.setdefault(group_key, []).append(row)

    result: dict[str, dict[str, Any]] = {}
    for group_key, group_rows in groups.items():
        chosen_scores = [float(item["chosen_score"]) for item in group_rows]
        rejected_scores = [float(item["rejected_score"]) for item in group_rows]
        margins = [float(item["margin"]) for item in group_rows]
        auc = compute_binary_auc(
            scores=[*chosen_scores, *rejected_scores],
            labels=[1] * len(chosen_scores) + [0] * len(rejected_scores),
        )
        result[group_key] = {
            "n": int(len(group_rows)),
            "pair_accuracy": float(sum(1 for margin in margins if margin > 0.0) / len(margins)),
            "auc": float(auc),
            "mean_margin": float(statistics.mean(margins)),
            "median_margin": float(statistics.median(margins)),
        }
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one list of dict rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _import_runtime_deps() -> tuple[Any, Any, Any]:
    """Import torch + transformers runtime dependencies lazily."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    raise SystemExit(main())
