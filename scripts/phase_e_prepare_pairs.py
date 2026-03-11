#!/usr/bin/env python3
"""Prepare benchmark-native external pair artifacts for Phase E.

English
-------
Phase E still uses the canonical chosen/rejected pair artifact, but that
artifact is no longer a side product.  It is now the primary supervised input
to the value-head trainer.

This script is therefore responsible for turning:
1. a named source bundle,
2. a pair-construction policy,
3. and a few quality thresholds
into one deterministic artifact directory that later stages can trust.

中文
----
Phase E 仍然沿用 canonical chosen/rejected pair artifact，但它已经不再是
“顺手产生的中间件”，而是 value head 训练的主监督输入。

因此这个脚本的职责是：把
1. 一个 source bundle，
2. 一套 pair 构造语义，
3. 若干质量阈值
转换成一个可复用、可追踪、可复现实验目录。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_d.external_pairs_adapters import PairBuildConfig  # noqa: E402
from ours.phase_e.contracts import (  # noqa: E402
    get_phase_e_pair_bundle_registry,
    resolve_phase_e_pair_bundle,
)
from ours.phase_e.pairs import prepare_phase_e_pair_artifact  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare canonical Phase E pair artifacts from registered source bundles."
    )
    parser.add_argument(
        "--source-bundle",
        required=True,
        choices=sorted(get_phase_e_pair_bundle_registry()),
        help="Named Phase E source bundle.",
    )
    parser.add_argument("--run-name", default="phase_e_pairs")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_pairs"),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse an existing matching artifact if all outputs already exist.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regenerate outputs even if a matching artifact already exists.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument(
        "--split-granularity",
        choices=["pair_id", "source_sample"],
        default="pair_id",
        help=(
            "How train/validation splitting should be performed. "
            "`pair_id` reproduces the legacy behavior; `source_sample` keeps all pairs from one raw sample together."
        ),
    )
    parser.add_argument(
        "--global-cap-mode",
        choices=["pair_id_head", "balanced_support_bucket"],
        default="pair_id_head",
        help=(
            "How the final `max_pairs_total` cap should be applied after confidence filtering and dedup. "
            "`pair_id_head` preserves the legacy deterministic head-of-list behavior; "
            "`balanced_support_bucket` round-robins over `(source_tag, pair_semantics)` buckets so rare repair semantics survive smoke-time caps."
        ),
    )
    parser.add_argument("--max-pairs-total", type=int, default=None)
    parser.add_argument("--max-pairs-per-source", type=int, default=None)
    parser.add_argument("--min-pair-confidence", type=float, default=0.0)
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-length-ratio", type=float, default=4.0)
    parser.add_argument("--max-token-overlap", type=float, default=0.995)
    parser.add_argument("--max-pairs-per-sample", type=int, default=2)
    parser.add_argument(
        "--step-label-pair-mode",
        choices=[
            "first_bad_edge_strict",
            "first_bad_fanout",
            "all_good_vs_all_bad",
            "legacy_nearest",
        ],
        default="first_bad_edge_strict",
        help=(
            "How to convert single-trajectory +/- step labels into pairs. "
            "Default is the safe local first-bad-edge mode. "
            "`first_bad_fanout` and `all_good_vs_all_bad` are explicit benchmark-alignment experiments, not neutral ETL choices."
        ),
    )
    parser.add_argument(
        "--step-label-terminal-anchor-mode",
        choices=["none", "all_positive_fanout"],
        default="none",
        help=(
            "Optional terminal-anchor augmentation for all-positive step-labeled trajectories. "
            "`all_positive_fanout` adds `full correct solution > earlier safe prefix` pairs so ProcessBench all-correct slices are no longer entirely unseen during training."
        ),
    )
    parser.add_argument(
        "--step-label-terminal-anchor-fraction",
        type=float,
        default=0.5,
        help=(
            "When terminal anchors are enabled and a source cap is active, reserve this fraction of the Math-Shepherd source budget for terminal anchors. "
            "Use values below 0.5 to keep terminal supervision auxiliary rather than co-equal."
        ),
    )
    parser.add_argument(
        "--r-prm-pair-mode",
        choices=["direct_pair_legacy", "compact_verdict", "compact_correctness"],
        default="compact_verdict",
        help=(
            "How R-PRM DPO rows should be converted into canonical pairs. "
            "`compact_verdict` is the Phase E default because it removes the long verifier-essay contract. "
            "`compact_correctness` keeps the same compact prompt but answers in Correct/Incorrect space."
        ),
    )

    # Keep dataset-path overrides explicit on the CLI so operators can swap
    # local dataset mirrors without mutating the registry baked into code.
    # 数据路径 override 必须显式走 CLI，这样切换本地镜像路径时不会污染代码里的默认 registry。
    parser.add_argument("--math-shepherd-path", type=Path, default=None)
    parser.add_argument("--prm800k-path", type=Path, default=None)
    parser.add_argument("--r-prm-root", type=Path, default=None)
    parser.add_argument("--prmbench-preview-path", type=Path, default=None)
    parser.add_argument(
        "--source-weight-overrides-json",
        default="",
        help=(
            "Optional JSON object mapping source_id -> positive float weight, "
            'for example: \'{"prm800k": 0.25}\'.'
        ),
    )
    parser.add_argument(
        "--r-prm-split",
        choices=["train", "validation"],
        default=None,
        help="Override split for R-PRM based bundles.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate CLI arguments.

    中文
    ----
    这里主要检查的是“参数取值是否合法”，而不是“数据本身是否存在问题”。
    数据内容层面的检查会在真正构造 artifact 时完成。
    """
    args = _build_parser().parse_args(argv)
    if not (0.0 < float(args.validation_ratio) < 0.5):
        raise ValueError("--validation-ratio must be in (0, 0.5)")
    if args.max_pairs_total is not None and int(args.max_pairs_total) <= 0:
        raise ValueError("--max-pairs-total must be > 0")
    if args.max_pairs_per_source is not None and int(args.max_pairs_per_source) <= 0:
        raise ValueError("--max-pairs-per-source must be > 0")
    if not (0.0 <= float(args.min_pair_confidence) <= 1.0):
        raise ValueError("--min-pair-confidence must be in [0, 1]")
    if int(args.min_chars) <= 0:
        raise ValueError("--min-chars must be > 0")
    if float(args.max_length_ratio) <= 1.0:
        raise ValueError("--max-length-ratio must be > 1")
    if not (0.0 <= float(args.max_token_overlap) <= 1.0):
        raise ValueError("--max-token-overlap must be in [0, 1]")
    if int(args.max_pairs_per_sample) <= 0:
        raise ValueError("--max-pairs-per-sample must be > 0")
    if not (0.0 < float(args.step_label_terminal_anchor_fraction) < 1.0):
        raise ValueError("--step-label-terminal-anchor-fraction must be in (0, 1)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Step 1: resolve the named bundle to concrete source specs.
    # 第一步：把 source bundle 名称解析成具体 source 规格。
    source_specs = resolve_phase_e_pair_bundle(str(args.source_bundle))
    # Step 2: apply path/split overrides from the current machine or current run.
    # 第二步：把当前机器、当前实验临时需要的路径或 split 覆盖项应用上去。
    source_specs = _apply_source_overrides(
        source_specs,
        math_shepherd_path=args.math_shepherd_path,
        prm800k_path=args.prm800k_path,
        r_prm_root=args.r_prm_root,
        prmbench_preview_path=args.prmbench_preview_path,
        r_prm_split=args.r_prm_split,
    )
    source_weight_overrides = _parse_source_weight_overrides(args.source_weight_overrides_json)
    # A good mental model:
    # 1. `source bundle` decides "where rows come from"
    # 2. `PairBuildConfig` decides "how those rows become canonical pairs"
    #
    # 一个很重要的理解方式：
    # 1. `source bundle` 决定“样本来自哪里”
    # 2. `PairBuildConfig` 决定“这些样本如何变成 canonical pair”
    build_config = PairBuildConfig(
        min_chars=int(args.min_chars),
        max_length_ratio=float(args.max_length_ratio),
        max_token_overlap=float(args.max_token_overlap),
        max_pairs_per_sample=int(args.max_pairs_per_sample),
        step_label_pair_mode=str(args.step_label_pair_mode),
        step_label_terminal_anchor_mode=str(args.step_label_terminal_anchor_mode),
        step_label_terminal_anchor_fraction=float(args.step_label_terminal_anchor_fraction),
        r_prm_pair_mode=str(args.r_prm_pair_mode),
    )

    # Step 3: build or reuse the deterministic artifact directory.
    # 第三步：真正构建或复用确定性的 artifact 目录。
    artifact = prepare_phase_e_pair_artifact(
        run_name=str(args.run_name),
        output_root=Path(args.output_root),
        source_specs=source_specs,
        build_config=build_config,
        seed=int(args.seed),
        validation_ratio=float(args.validation_ratio),
        max_pairs_total=(
            int(args.max_pairs_total) if args.max_pairs_total is not None else None
        ),
        max_pairs_per_source=(
            int(args.max_pairs_per_source) if args.max_pairs_per_source is not None else None
        ),
        min_pair_confidence=float(args.min_pair_confidence),
        split_granularity=str(args.split_granularity),
        global_cap_mode=str(args.global_cap_mode),
        source_weight_overrides=source_weight_overrides,
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
    )

    # The stdout summary is intentionally short and operator-facing.
    # 真正详细的信息写入 summary/manifest；终端这里只打印操作员最需要的摘要。
    print("=" * 88)
    print("Phase E: Prepare Pair Artifact")
    print("=" * 88)
    print(f"source_bundle     : {args.source_bundle}")
    print(f"run_dir           : {artifact.run_dir}")
    print(f"num_total_pairs   : {artifact.summary['num_rows_after_dedup']}")
    print(f"num_train_pairs   : {artifact.summary['num_train_rows']}")
    print(f"num_val_pairs     : {artifact.summary['num_validation_rows']}")
    print(f"split_granularity : {artifact.summary['build_config'].get('split_granularity')}")
    print(f"global_cap_mode   : {artifact.summary['build_config'].get('global_cap_mode')}")
    print(f"source_weights    : {artifact.summary['build_config'].get('source_weight_overrides', {})}")
    print(f"step_label_mode   : {artifact.summary['build_config'].get('step_label_pair_mode')}")
    print(
        "step_label_terminal_anchor_mode : "
        f"{artifact.summary['build_config'].get('step_label_terminal_anchor_mode')}"
    )
    print(
        "step_label_terminal_anchor_fraction : "
        f"{artifact.summary['build_config'].get('step_label_terminal_anchor_fraction')}"
    )
    print(f"r_prm_pair_mode   : {artifact.summary['build_config'].get('r_prm_pair_mode')}")
    print(f"train_pairs_path  : {artifact.train_pairs_path}")
    print(f"val_pairs_path    : {artifact.validation_pairs_path}")
    print(f"summary_path      : {artifact.summary_path}")
    print(f"manifest_path     : {artifact.manifest_path}")
    print("=" * 88)
    return 0


def _apply_source_overrides(
    source_specs,
    *,
    math_shepherd_path: Path | None,
    prm800k_path: Path | None,
    r_prm_root: Path | None,
    prmbench_preview_path: Path | None,
    r_prm_split: str | None,
):
    """Apply CLI path overrides without mutating the registry objects.

    中文
    ----
    这里特意不原地修改 registry 中的对象，而是先复制再覆盖，防止一次临时运行
    把默认 contract 悄悄改坏。
    """
    updated = []
    for spec in source_specs:
        # Copy first, override second.
        # 先复制，再覆盖，保证默认 registry 仍然是不可变的基线。
        payload = spec.to_dict()
        if spec.source_type == "math_shepherd" and math_shepherd_path is not None:
            payload["default_path"] = str(math_shepherd_path)
        if spec.source_type == "prm800k" and prm800k_path is not None:
            payload["default_path"] = str(prm800k_path)
        if spec.source_type == "r_prm" and r_prm_root is not None:
            payload["default_path"] = str(r_prm_root)
        if spec.source_type == "r_prm" and r_prm_split is not None:
            payload["default_split"] = str(r_prm_split)
        if spec.source_type == "prmbench_preview" and prmbench_preview_path is not None:
            payload["default_path"] = str(prmbench_preview_path)
        updated.append(type(spec)(**payload))
    return updated


def _parse_source_weight_overrides(raw: str) -> dict[str, float] | None:
    """Parse optional source-weight overrides from one compact JSON object.

    中文
    ----
    这里要求用户显式传 JSON，而不是发明一套自定义字符串格式，是为了减少 shell
    解析歧义，也方便日志里原样记录。
    """
    text = str(raw or "").strip()
    if text == "":
        return None
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("--source-weight-overrides-json must decode to a JSON object")
    parsed: dict[str, float] = {}
    for key, value in payload.items():
        weight = float(value)
        if weight <= 0.0:
            raise ValueError(
                f"--source-weight-overrides-json requires positive weights, got {key!r}={value!r}"
            )
        parsed[str(key)] = weight
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
