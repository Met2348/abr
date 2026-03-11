#!/usr/bin/env python3
"""Curate ProcessBench-aligned Phase E pair artifacts from existing training sources.

English
-------
This script operationalizes one literature-backed lesson for the current repo:
benchmark transfer usually fails not because the model never learns anything,
but because the training support does not cover the evaluation geometry.

The curation here is intentionally conservative:
1. keep a strong local first-error core,
2. add fanout/grid support for broader good-vs-bad relations,
3. add only low-mass terminal anchors,
4. optionally add PRMBench local explicit-error pairs as aligned auxiliary data.

中文
----
这个脚本把当前仓库的一条关键方法学判断正式落成代码：
benchmark 迁移失败，很多时候不是“模型什么都学不会”，而是训练支持面没有覆盖评测几何。

这里的 curate 是故意保守的：
1. 保留强 local first-error 主干，
2. 再补 fanout / grid 这类更宽的 good-vs-bad 关系，
3. terminal anchor 只做低占比辅助，
4. 可选再加 PRMBench 的显式局部错误 pair 作为对齐辅助源。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_bootstrap_src_path()

from ours.phase_d.external_pairs import ExternalPairRecord, summarize_external_pairs  # noqa: E402
from ours.phase_d.external_pairs_adapters import (  # noqa: E402
    PairBuildConfig,
    load_math_shepherd_pairs,
    load_math_step_dpo_pairs,
    load_prmbench_preview_pairs,
    load_rlhflow_pairs,
)
from ours.phase_e.pairs import (  # noqa: E402
    _count_split_units,
    _deduplicate_pairs,
    _split_train_validation,
    _stable_fingerprint,
    _write_jsonl,
)
from ours.phase_e.processbench_alignment import classify_step_label_pair_type  # noqa: E402


@dataclass(frozen=True, slots=True)
class CurateComponent:
    """One component inside a curated transfer profile.

    English
    -------
    `fraction` controls how much of the final pool this component should occupy.
    `semantic_weight` is not a sampling ratio; it is later consumed by the
    trainer when `pair_weight_mode=confidence_semantic`.

    中文
    ----
    这里的 `fraction` 只控制抽样占比。
    `semantic_weight` 不是抽样比例，而是后面训练时在
    `pair_weight_mode=confidence_semantic` 下真正进入损失的语义权重。
    """

    component_id: str
    source_kind: str
    fraction: float
    semantic_weight: float
    step_label_pair_mode: str = "first_bad_edge_strict"
    terminal_anchor_mode: str = "none"
    terminal_anchor_fraction: float = 0.10
    pair_type_allowlist: tuple[str, ...] = ()
    description: str = ""


PROFILE_REGISTRY: dict[str, dict[str, Any]] = {
    "ms_core_v1": {
        "title": "Math-Shepherd Core Local+Fanout Warm Start",
        "intent": (
            "Stage-1 conservative warm start that keeps the main supervision on "
            "clean local first-error detection while adding some any-good-vs-first-bad support."
        ),
        "components": (
            CurateComponent(
                component_id="ms_strict",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                fraction=0.65,
                semantic_weight=1.00,
                description="Strict last-safe vs first-bad core.",
            ),
            CurateComponent(
                component_id="ms_fanout",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_fanout",
                fraction=0.35,
                semantic_weight=0.90,
                description="Broader any-good vs first-bad support without later-bad mixing.",
            ),
        ),
    },
    "ms_align_v1": {
        "title": "Math-Shepherd Conservative ProcessBench Alignment",
        "intent": (
            "One-shot conservative profile: local core remains dominant, while grid and terminal support are added as bounded auxiliaries."
        ),
        "components": (
            CurateComponent(
                component_id="ms_strict",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                fraction=0.40,
                semantic_weight=1.00,
                description="Primary local first-error anchor.",
            ),
            CurateComponent(
                component_id="ms_fanout",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_fanout",
                fraction=0.25,
                semantic_weight=0.90,
                description="Adds broader good-vs-first-bad coverage.",
            ),
            CurateComponent(
                component_id="ms_grid",
                source_kind="math_shepherd",
                step_label_pair_mode="all_good_vs_all_bad",
                fraction=0.25,
                semantic_weight=0.75,
                description="Adds later-bad support, but with lower trust due to depth/length confounds.",
            ),
            CurateComponent(
                component_id="ms_terminal",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                terminal_anchor_mode="all_positive_fanout",
                terminal_anchor_fraction=0.95,
                fraction=0.10,
                semantic_weight=0.45,
                description="Low-mass all-correct terminal anchor auxiliary.",
            ),
        ),
    },
    "ms_prm_align_v1": {
        "title": "Math-Shepherd + PRMBench Conservative Alignment",
        "intent": (
            "Keep the conservative Math-Shepherd profile, then add PRMBench local explicit-error pairs as an aligned auxiliary source."
        ),
        "components": (
            CurateComponent(
                component_id="ms_strict",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                fraction=0.32,
                semantic_weight=1.00,
                description="Primary local first-error anchor.",
            ),
            CurateComponent(
                component_id="ms_fanout",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_fanout",
                fraction=0.22,
                semantic_weight=0.90,
                description="Adds broader good-vs-first-bad coverage.",
            ),
            CurateComponent(
                component_id="ms_grid",
                source_kind="math_shepherd",
                step_label_pair_mode="all_good_vs_all_bad",
                fraction=0.16,
                semantic_weight=0.75,
                description="Adds later-bad support with bounded trust.",
            ),
            CurateComponent(
                component_id="ms_terminal",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                terminal_anchor_mode="all_positive_fanout",
                terminal_anchor_fraction=0.95,
                fraction=0.10,
                semantic_weight=0.45,
                description="Low-mass all-correct terminal anchor auxiliary.",
            ),
            CurateComponent(
                component_id="prm_local",
                source_kind="prmbench_preview",
                fraction=0.20,
                semantic_weight=0.95,
                description="Explicit local modified-step supervision from PRMBench.",
            ),
        ),
    },
    # 2026-03-11: RLHFlow-Deepseek first-bad-edge profile.
    # RLHFlow-Deepseek 提供 LLM-judge (Deepseek-8B) 标注的步骤级 +/- 标签，
    # 标注质量高于 Math-Shepherd MC 估计，直接构造 first_bad_edge pair。
    "rlhflow_align_v1": {
        "title": "RLHFlow-Deepseek First-Bad-Edge Alignment",
        "intent": (
            "Replace MC-estimated Math-Shepherd labels with LLM-judge-annotated (Deepseek-8B) "
            "step labels from RLHFlow. Tests if annotation quality uplift transfers to ProcessBench."
        ),
        "components": (
            CurateComponent(
                component_id="rlh_strict",
                source_kind="rlhflow_deepseek",
                step_label_pair_mode="first_bad_edge_strict",
                fraction=0.45,
                semantic_weight=1.00,
                description="Primary LLM-judge first-error anchor from RLHFlow-Deepseek.",
            ),
            CurateComponent(
                component_id="rlh_fanout",
                source_kind="rlhflow_deepseek",
                step_label_pair_mode="first_bad_fanout",
                fraction=0.30,
                semantic_weight=0.90,
                description="Broader any-good vs first-bad from RLHFlow-Deepseek.",
            ),
            CurateComponent(
                component_id="rlh_terminal",
                source_kind="rlhflow_deepseek",
                step_label_pair_mode="first_bad_edge_strict",
                terminal_anchor_mode="all_positive_fanout",
                terminal_anchor_fraction=0.95,
                fraction=0.12,
                semantic_weight=0.45,
                description="Low-mass all-correct terminal anchor from RLHFlow-Deepseek.",
            ),
            CurateComponent(
                component_id="rlh_grid",
                source_kind="rlhflow_deepseek",
                step_label_pair_mode="all_good_vs_all_bad",
                fraction=0.13,
                semantic_weight=0.70,
                description="Later-bad grid support from RLHFlow-Deepseek.",
            ),
        ),
    },
    # 2026-03-11: Math-Step-DPO-10K sibling-branch profile.
    # Math-Step-DPO-10K 提供显式分叉点 pair（initial_reason_steps + chosen vs rejected），
    # 是最干净的 sibling_branch 语义，直接测试分叉点监督对 first_error 检测的效果。
    "math_step_dpo_v1": {
        "title": "Math-Step-DPO-10K Sibling-Branch Profile",
        "intent": (
            "Use explicit fork-point pairs from Math-Step-DPO-10K where chosen/rejected diverge at "
            "a known step boundary. Tests whether clean sibling-branch supervision improves "
            "first_error_edge_accuracy on ProcessBench."
        ),
        "components": (
            CurateComponent(
                component_id="dpo_fork",
                source_kind="math_step_dpo",
                fraction=0.65,
                semantic_weight=0.80,
                description="Explicit fork-point pairs from Math-Step-DPO-10K.",
            ),
            CurateComponent(
                component_id="ms_strict",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                fraction=0.25,
                semantic_weight=1.00,
                description="Math-Shepherd strict first-error anchor as backbone.",
            ),
            CurateComponent(
                component_id="ms_terminal",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                terminal_anchor_mode="all_positive_fanout",
                terminal_anchor_fraction=0.95,
                fraction=0.10,
                semantic_weight=0.45,
                description="Low-mass all-correct terminal anchor.",
            ),
        ),
    },
    # 2026-03-11: Mixed Math-Shepherd + RLHFlow-Deepseek profile.
    # 同时保留 Math-Shepherd 的覆盖宽度和 RLHFlow 的标注质量。
    "ms_rlhflow_mixed_v1": {
        "title": "Mixed Math-Shepherd + RLHFlow-Deepseek Alignment",
        "intent": (
            "Combine Math-Shepherd breadth with RLHFlow LLM-judge quality. "
            "Tests if mixing MC-annotated (large) + LLM-annotated (diverse) data helps."
        ),
        "components": (
            CurateComponent(
                component_id="ms_strict",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                fraction=0.25,
                semantic_weight=1.00,
                description="Math-Shepherd strict first-error anchor.",
            ),
            CurateComponent(
                component_id="ms_fanout",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_fanout",
                fraction=0.15,
                semantic_weight=0.90,
                description="Math-Shepherd fanout coverage.",
            ),
            CurateComponent(
                component_id="rlh_strict",
                source_kind="rlhflow_deepseek",
                step_label_pair_mode="first_bad_edge_strict",
                fraction=0.25,
                semantic_weight=1.00,
                description="RLHFlow-Deepseek LLM-judge first-error anchor.",
            ),
            CurateComponent(
                component_id="rlh_fanout",
                source_kind="rlhflow_deepseek",
                step_label_pair_mode="first_bad_fanout",
                fraction=0.15,
                semantic_weight=0.90,
                description="RLHFlow-Deepseek fanout coverage.",
            ),
            CurateComponent(
                component_id="dpo_fork",
                source_kind="math_step_dpo",
                fraction=0.10,
                semantic_weight=0.80,
                description="Math-Step-DPO fork-point pairs.",
            ),
            CurateComponent(
                component_id="ms_terminal",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                terminal_anchor_mode="all_positive_fanout",
                terminal_anchor_fraction=0.95,
                fraction=0.10,
                semantic_weight=0.45,
                description="Low-mass terminal anchor.",
            ),
        ),
    },
    # 2026-03-11 FIX: ms_strict_only_v1 — eliminates length-biased fanout/grid pairs from MS.
    # Diagnosis showed NDS3 failure: 20.1% of ms_align_v1 pairs (fanout rej-cho=+194, grid rej-cho=+203)
    # teach "shorter=better" LENGTH BIAS. On ProcessBench bad_prefix is LONGER → inverted scores.
    # This profile keeps ONLY local_first_bad_edge (rej-cho=+99) + terminal (rej-cho=-141),
    # both of which have moderate/reverse length deltas that do NOT cause systematic inversion.
    #
    # NDS3 长度偏差诊断修复：ms_align_v1 中有 20.1% 的 fanout/grid pair 教会模型"更短=更好"，
    # 而 ProcessBench 的 bad_prefix 比 good_prefix 更长，导致评分倒置。
    # 这个 profile 只保留不含系统性长度偏差的 strict 和 terminal pair 类型。
    "ms_strict_only_v1": {
        "title": "Math-Shepherd Strict-Only (No Length-Bias Pairs)",
        "intent": (
            "Ablation that removes ALL fanout/grid pair types from MS, keeping only "
            "local_first_bad_edge (strict) and terminal_completion_anchor. "
            "Tests whether the NDS3 failure was entirely due to length-biased pair types."
        ),
        "components": (
            CurateComponent(
                component_id="ms_strict",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                fraction=0.80,
                semantic_weight=1.00,
                description="Strict local first-error anchor (rej-cho≈+99, no systematic inversion).",
            ),
            CurateComponent(
                component_id="ms_terminal",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                terminal_anchor_mode="all_positive_fanout",
                terminal_anchor_fraction=0.95,
                fraction=0.20,
                semantic_weight=0.45,
                description="All-correct terminal anchor (rej-cho≈-141, reverse direction, not problematic).",
            ),
        ),
    },
    # 2026-03-11 FIX: ms_dpo_calibrated_v1 — DPO sibling_branch pairs as length-bias debiaser.
    # Math-Step-DPO sibling_branch pairs have rej-cho≈0 (same prefix length, different next step),
    # which force the model to learn CONTENT quality rather than length shortcuts.
    # Using DPO as 50% anchor + MS strict as 40% step-detection teacher = best of both worlds.
    #
    # 用 DPO 分叉点 pair 作为长度偏差矫正锚点：DPO pair 的 rej-cho≈0，
    # 迫使模型学习内容质量而非长度捷径；MS strict 提供 first-error 检测信号。
    "ms_dpo_calibrated_v1": {
        "title": "MS + DPO Calibrated (Length-Bias Corrected)",
        "intent": (
            "Use Math-Step-DPO sibling_branch pairs (rej-cho≈0) as length-bias debiasers, "
            "combined with MS strict first-error supervision. DPO 50% anchor prevents the model "
            "from learning the shorter=better shortcut that collapses NDS3."
        ),
        "components": (
            CurateComponent(
                component_id="dpo_fork",
                source_kind="math_step_dpo",
                fraction=0.50,
                semantic_weight=0.80,
                description="DPO fork-point pairs (rej-cho≈0): content-quality anchor, no length bias.",
            ),
            CurateComponent(
                component_id="ms_strict",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                fraction=0.40,
                semantic_weight=1.00,
                description="MS strict local first-error anchor for step-detection supervision.",
            ),
            CurateComponent(
                component_id="ms_terminal",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                terminal_anchor_mode="all_positive_fanout",
                terminal_anchor_fraction=0.95,
                fraction=0.10,
                semantic_weight=0.45,
                description="Low-mass terminal anchor for all-correct trajectory calibration.",
            ),
        ),
    },
    # 2026-03-11: terminal-boosted variant of the NDS7 recipe.
    # 核心假设：
    # 当前 NDS7 + gated 已经把 local / global ranking 拉起来，剩余主残差集中在
    # all-correct final completion ordering。这个 profile 不改变主干几何，只把 terminal
    # anchor 从 10% 提到 20%，测试“是否只是 terminal 覆盖不够”。
    "ms_dpo_terminalboost_v1": {
        "title": "MS + DPO Terminal-Boosted Alignment",
        "intent": (
            "Keep the successful NDS7 geometry (DPO sibling_branch + MS strict), "
            "but increase terminal anchor mass from 10% to 20%. "
            "Tests whether the remaining ProcessBench gap is primarily caused by "
            "under-coverage of all-correct terminal completion ordering."
        ),
        "components": (
            CurateComponent(
                component_id="dpo_fork",
                source_kind="math_step_dpo",
                fraction=0.45,
                semantic_weight=0.80,
                description="Primary sibling-branch supervision remains dominant.",
            ),
            CurateComponent(
                component_id="ms_strict",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                fraction=0.35,
                semantic_weight=1.00,
                description="Strict first-error anchor is preserved but slightly reduced.",
            ),
            CurateComponent(
                component_id="ms_terminal",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                terminal_anchor_mode="all_positive_fanout",
                terminal_anchor_fraction=0.95,
                fraction=0.20,
                semantic_weight=0.55,
                description="Terminal anchor is explicitly boosted to test completion-ordering coverage.",
            ),
        ),
    },
    # 2026-03-11 FIX: dpo_scale_v1 — Pure DPO at scale (8192 pairs, ~75% of full dataset).
    # NDS2 (3705 pairs) achieves MATH AUC=0.712 (new frozen-head SOTA). This tests whether
    # scaling up to 8192 DPO pairs breaks the frozen backbone ceiling.
    # Math-Step-DPO has 10.8K rows total; 8192 pairs uses ~75% of available data.
    #
    # 纯 DPO 扩展实验：NDS2 用 3705 pair 已达 MATH AUC=0.712 (冻结头最优)。
    # 这里扩展到 8192 pair，测试 DPO 数据规模能否进一步突破冻结骨干上限。
    "dpo_scale_v1": {
        "title": "Math-Step-DPO Pure Scale (8K pairs)",
        "intent": (
            "Scale NDS2's winning pure-DPO configuration from 3705 to 8192 pairs. "
            "Tests whether DPO sibling_branch data at larger scale can push frozen-head "
            "MATH AUC beyond the current 0.712 ceiling."
        ),
        "components": (
            CurateComponent(
                component_id="dpo_fork",
                source_kind="math_step_dpo",
                fraction=1.00,
                semantic_weight=0.80,
                description="All DPO fork-point pairs at 8K scale — pure sibling_branch.",
            ),
        ),
    },
    # 2026-03-11 FIX: rlh_strict_only_v1 — RLHFlow without length-biased fanout/grid.
    # NDS1 (rlhflow_align_v1) had 43% fanout+grid pairs → even worse length bias than NDS3.
    # RLHFlow labels are LLM-judge quality (Deepseek-8B), much better than MC estimation.
    # Removing fanout/grid should reveal the true signal quality of LLM-judge annotations.
    #
    # RLHFlow 去除长度偏差版本：NDS1 包含 43% fanout/grid pair（比 NDS3 的 20.1% 更高），
    # 导致 MATH AUC=0.552 严重倒置。仅保留 strict 标注，测试 LLM-judge 标签质量是否足够好。
    "rlh_strict_only_v1": {
        "title": "RLHFlow-Deepseek Strict-Only (No Length-Bias Pairs)",
        "intent": (
            "Remove ALL fanout/grid pair types from RLHFlow-Deepseek, keeping only "
            "first_bad_edge_strict. NDS1 had 43% length-biased pairs (fanout+grid), "
            "causing MATH AUC=0.552. Tests if LLM-judge annotation quality, freed from "
            "length bias, can rival NDS2's fork-point performance."
        ),
        "components": (
            CurateComponent(
                component_id="rlh_strict",
                source_kind="rlhflow_deepseek",
                step_label_pair_mode="first_bad_edge_strict",
                fraction=0.90,
                semantic_weight=1.00,
                description="RLHFlow LLM-judge strict first-error (no fanout/grid contamination).",
            ),
            CurateComponent(
                component_id="rlh_terminal",
                source_kind="rlhflow_deepseek",
                step_label_pair_mode="first_bad_edge_strict",
                terminal_anchor_mode="all_positive_fanout",
                terminal_anchor_fraction=0.95,
                fraction=0.10,
                semantic_weight=0.45,
                description="Low-mass all-correct terminal anchor from RLHFlow-Deepseek.",
            ),
        ),
    },
    # 2026-03-11: targeted later-bad profile; uses pair_type_allowlist to restrict to
    #             only lastsafe_vs_laterbad + earlygood_vs_laterbad pairs (not all grid).
    "ms_laterbad_v1": {
        "title": "Math-Shepherd Later-Bad Targeted Alignment",
        "intent": (
            "Keep the local first-error core, but explicitly oversample later-bad pairs instead of generic grid coverage."
        ),
        "components": (
            CurateComponent(
                component_id="ms_strict",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                fraction=0.40,
                semantic_weight=1.00,
                description="Primary local first-error anchor.",
            ),
            CurateComponent(
                component_id="ms_fanout",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_fanout",
                fraction=0.20,
                semantic_weight=0.90,
                description="Broader good-vs-first-bad coverage.",
            ),
            CurateComponent(
                component_id="ms_laterbad",
                source_kind="math_shepherd",
                step_label_pair_mode="all_good_vs_all_bad",
                pair_type_allowlist=("lastsafe_vs_laterbad", "earlygood_vs_laterbad"),
                fraction=0.30,
                semantic_weight=0.85,
                description="Targeted later-bad support rather than generic grid mixing.",
            ),
            CurateComponent(
                component_id="ms_terminal",
                source_kind="math_shepherd",
                step_label_pair_mode="first_bad_edge_strict",
                terminal_anchor_mode="all_positive_fanout",
                terminal_anchor_fraction=0.95,
                fraction=0.10,
                semantic_weight=0.40,
                description="Low-mass all-correct terminal anchor auxiliary.",
            ),
        ),
    },
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Curate conservative ProcessBench-aligned Phase E pair artifacts."
    )
    parser.add_argument("--profile", choices=sorted(PROFILE_REGISTRY), default="ms_align_v1")
    parser.add_argument("--run-name", default="phase_e_processbench_curated_pairs")
    parser.add_argument("--output-root", type=Path, default=Path("assets/artifacts/phase_e_pairs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--split-granularity", choices=["pair_id", "source_sample"], default="source_sample")
    parser.add_argument(
        "--target-total-pairs",
        type=int,
        default=8192,
        help="Total pair budget before train/validation split.",
    )
    parser.add_argument("--min-pair-confidence", type=float, default=0.55)
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-length-ratio", type=float, default=4.0)
    parser.add_argument("--max-token-overlap", type=float, default=0.995)
    parser.add_argument("--max-pairs-per-sample", type=int, default=4)
    parser.add_argument(
        "--math-shepherd-path",
        type=Path,
        default=Path("assets/external_datasets/peiyi_math_shepherd/math-shepherd.jsonl"),
    )
    parser.add_argument(
        "--prmbench-preview-path",
        type=Path,
        default=Path("assets/external_datasets/hitsmy_prmbench_preview/prmbench_preview.jsonl"),
    )
    parser.add_argument(
        "--rlhflow-deepseek-path",
        type=Path,
        default=Path("assets/external_datasets/rlhflow_deepseek_prm/deepseek_instruct_data.jsonl"),
    )
    parser.add_argument(
        "--math-step-dpo-path",
        type=Path,
        default=Path("assets/external_datasets/xinlai_math_step_dpo"),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not (0.0 < float(args.validation_ratio) < 0.5):
        raise ValueError("--validation-ratio must be in (0, 0.5)")
    if int(args.target_total_pairs) <= 0:
        raise ValueError("--target-total-pairs must be > 0")
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
    # 只有在 profile 中包含对应 source_kind 的 component 时才严格校验路径
    # Only validate paths that are actually needed by the selected profile's components
    profile_data = PROFILE_REGISTRY.get(str(args.profile), {})
    used_kinds = {c.source_kind for c in profile_data.get("components", [])}
    if "math_shepherd" in used_kinds and not args.math_shepherd_path.exists():
        raise FileNotFoundError(f"--math-shepherd-path not found: {args.math_shepherd_path}")
    if "prmbench_preview" in used_kinds and not args.prmbench_preview_path.exists():
        raise FileNotFoundError(f"--prmbench-preview-path not found: {args.prmbench_preview_path}")
    if "rlhflow_deepseek" in used_kinds and not args.rlhflow_deepseek_path.exists():
        raise FileNotFoundError(f"--rlhflow-deepseek-path not found: {args.rlhflow_deepseek_path}")
    if "math_step_dpo" in used_kinds and not args.math_step_dpo_path.exists():
        raise FileNotFoundError(f"--math-step-dpo-path not found: {args.math_step_dpo_path}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = PROFILE_REGISTRY[str(args.profile)]
    components: tuple[CurateComponent, ...] = tuple(profile["components"])
    quotas = _allocate_component_quotas(
        total_pairs=int(args.target_total_pairs),
        components=components,
    )

    selected_rows: list[ExternalPairRecord] = []
    component_summaries: list[dict[str, Any]] = []
    for component in components:
        quota = int(quotas[component.component_id])
        pool = _load_component_rows(
            component=component,
            quota=quota,
            math_shepherd_path=Path(args.math_shepherd_path),
            prmbench_preview_path=Path(args.prmbench_preview_path),
            rlhflow_deepseek_path=Path(args.rlhflow_deepseek_path),
            math_step_dpo_path=Path(args.math_step_dpo_path),
            min_chars=int(args.min_chars),
            max_length_ratio=float(args.max_length_ratio),
            max_token_overlap=float(args.max_token_overlap),
            max_pairs_per_sample=int(args.max_pairs_per_sample),
        )
        tagged_rows = _tag_component_rows(
            rows=pool,
            component=component,
            profile_name=str(args.profile),
        )
        selected_rows.extend(tagged_rows)
        component_summaries.append(
            {
                "component_id": component.component_id,
                "source_kind": component.source_kind,
                "target_pairs": quota,
                "selected_pairs": int(len(tagged_rows)),
                "semantic_weight": float(component.semantic_weight),
                "description": component.description,
                "pair_summary": summarize_external_pairs(tagged_rows),
            }
        )

    filtered_rows = [
        row for row in selected_rows if float(row.pair_confidence) >= float(args.min_pair_confidence)
    ]
    dedup_rows = _deduplicate_pairs(filtered_rows)
    train_rows, validation_rows = _split_train_validation(
        rows=dedup_rows,
        seed=int(args.seed),
        validation_ratio=float(args.validation_ratio),
        split_granularity=str(args.split_granularity),
    )

    fingerprint = _stable_fingerprint(
        {
            "profile": str(args.profile),
            "seed": int(args.seed),
            "validation_ratio": float(args.validation_ratio),
            "split_granularity": str(args.split_granularity),
            "target_total_pairs": int(args.target_total_pairs),
            "min_pair_confidence": float(args.min_pair_confidence),
            "math_shepherd_path": str(args.math_shepherd_path),
            "prmbench_preview_path": str(args.prmbench_preview_path),
            "rlhflow_deepseek_path": str(args.rlhflow_deepseek_path),
            "math_step_dpo_path": str(args.math_step_dpo_path),
            "component_targets": quotas,
            "profile_components": [
                {
                    "component_id": component.component_id,
                    "source_kind": component.source_kind,
                    "fraction": float(component.fraction),
                    "semantic_weight": float(component.semantic_weight),
                    "step_label_pair_mode": component.step_label_pair_mode,
                    "terminal_anchor_mode": component.terminal_anchor_mode,
                    "terminal_anchor_fraction": float(component.terminal_anchor_fraction),
                    "pair_type_allowlist": list(component.pair_type_allowlist),
                }
                for component in components
            ],
        }
    )
    run_dir = Path(args.output_root) / f"{args.run_name}__{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_path = run_dir / "train_pairs.jsonl"
    validation_path = run_dir / "validation_pairs.jsonl"
    summary_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    manifest_path = run_dir / "manifest.json"

    _write_jsonl(train_path, [row.to_dict() for row in train_rows])
    _write_jsonl(validation_path, [row.to_dict() for row in validation_rows])

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "profile": str(args.profile),
        "profile_title": str(profile["title"]),
        "profile_intent": str(profile["intent"]),
        "num_rows_before_confidence_filter": int(len(selected_rows)),
        "num_rows_after_confidence_filter": int(len(filtered_rows)),
        "num_rows_after_dedup": int(len(dedup_rows)),
        "num_train_rows": int(len(train_rows)),
        "num_validation_rows": int(len(validation_rows)),
        "num_split_units_after_dedup": int(
            _count_split_units(rows=dedup_rows, split_granularity=str(args.split_granularity))
        ),
        "num_train_split_units": int(
            _count_split_units(rows=train_rows, split_granularity=str(args.split_granularity))
        ),
        "num_validation_split_units": int(
            _count_split_units(rows=validation_rows, split_granularity=str(args.split_granularity))
        ),
        "component_targets": quotas,
        "component_summaries": component_summaries,
        "overall_summary": summarize_external_pairs(dedup_rows),
        "train_summary": summarize_external_pairs(train_rows),
        "validation_summary": summarize_external_pairs(validation_rows),
        "build_config": {
            "seed": int(args.seed),
            "validation_ratio": float(args.validation_ratio),
            "split_granularity": str(args.split_granularity),
            "target_total_pairs": int(args.target_total_pairs),
            "min_pair_confidence": float(args.min_pair_confidence),
            "math_shepherd_path": str(args.math_shepherd_path),
            "prmbench_preview_path": str(args.prmbench_preview_path),
            "rlhflow_deepseek_path": str(args.rlhflow_deepseek_path),
            "math_step_dpo_path": str(args.math_step_dpo_path),
            "min_chars": int(args.min_chars),
            "max_length_ratio": float(args.max_length_ratio),
            "max_token_overlap": float(args.max_token_overlap),
            "max_pairs_per_sample": int(args.max_pairs_per_sample),
        },
    }
    manifest = {
        "artifact_stage": "phase_e_processbench_curated_pairs_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": str(args.run_name),
        "run_dir": str(run_dir),
        "profile": str(args.profile),
        "fingerprint": str(fingerprint),
        "output_files": {
            "train_pairs": str(train_path),
            "validation_pairs": str(validation_path),
            "summary": str(summary_path),
            "summary_md": str(summary_md_path),
        },
        "summary_snapshot": summary,
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")

    print("=" * 88)
    print("Phase E: Curate ProcessBench Transfer Pairs")
    print("=" * 88)
    print(f"profile             : {args.profile}")
    print(f"run_dir             : {run_dir}")
    print(f"num_train_rows      : {len(train_rows)}")
    print(f"num_validation_rows : {len(validation_rows)}")
    for component_row in component_summaries:
        print(
            f"{component_row['component_id']:>18}: target={component_row['target_pairs']} "
            f"selected={component_row['selected_pairs']} weight={component_row['semantic_weight']:.2f}"
        )
    print("=" * 88)
    return 0


def _allocate_component_quotas(
    *,
    total_pairs: int,
    components: tuple[CurateComponent, ...],
) -> dict[str, int]:
    """Allocate deterministic integer pair budgets per component.

    English
    -------
    We keep the profile fractions explicit, then use largest-remainder rounding
    so the final integer budgets still sum exactly to `total_pairs`.

    中文
    ----
    这里先保留 profile 的显式比例，再用 largest-remainder rounding，把它稳定落成整数配额，
    同时保证总数严格等于 `total_pairs`。
    """
    raw_targets = {
        component.component_id: float(total_pairs) * float(component.fraction)
        for component in components
    }
    base_targets = {key: int(math.floor(value)) for key, value in raw_targets.items()}
    remainder = int(total_pairs) - int(sum(base_targets.values()))
    if remainder > 0:
        ranked = sorted(
            ((key, raw_targets[key] - base_targets[key]) for key in raw_targets),
            key=lambda item: (-float(item[1]), item[0]),
        )
        for key, _ in ranked[:remainder]:
            base_targets[key] += 1
    return base_targets


def _load_component_rows(
    *,
    component: CurateComponent,
    quota: int,
    math_shepherd_path: Path,
    prmbench_preview_path: Path,
    rlhflow_deepseek_path: Path,
    math_step_dpo_path: Path,
    min_chars: int,
    max_length_ratio: float,
    max_token_overlap: float,
    max_pairs_per_sample: int,
) -> list[ExternalPairRecord]:
    if quota <= 0:
        return []
    config = PairBuildConfig(
        min_chars=int(min_chars),
        max_length_ratio=float(max_length_ratio),
        max_token_overlap=float(max_token_overlap),
        max_pairs_per_sample=int(max_pairs_per_sample),
        step_label_pair_mode=str(component.step_label_pair_mode),
        step_label_terminal_anchor_mode=str(component.terminal_anchor_mode),
        step_label_terminal_anchor_fraction=float(component.terminal_anchor_fraction),
        r_prm_pair_mode="compact_verdict",
    )
    if component.source_kind == "math_shepherd":
        requested_max_pairs = int(quota)
        if component.component_id == "ms_terminal":
            # Terminal anchors are sparse and appear only on all-positive traces.
            # 这里对 terminal 组件额外 oversample，是因为它本来就稀疏，只出现在 all-positive 轨迹上。
            requested_max_pairs = max(int(math.ceil(float(quota) / 0.90)), int(quota) + 64)
        rows = load_math_shepherd_pairs(
            path=math_shepherd_path,
            config=config,
            max_pairs=requested_max_pairs,
        )
        if component.component_id == "ms_terminal":
            rows = [
                row
                for row in rows
                if str((row.metadata or {}).get("pair_semantics", "")) == "terminal_completion_anchor"
            ]
        elif component.pair_type_allowlist:
            rows = _filter_rows_by_pair_type(
                rows=rows,
                pair_type_allowlist=component.pair_type_allowlist,
            )
    elif component.source_kind == "prmbench_preview":
        rows = load_prmbench_preview_pairs(
            path=prmbench_preview_path,
            config=config,
            max_pairs=int(quota),
        )
    elif component.source_kind == "rlhflow_deepseek":
        # RLHFlow-Deepseek: LLM-judge 标注的逐步 +/- labels → first_bad_edge pairs
        # RLHFlow-Deepseek: per-step LLM-judge +/- labels → first_bad_edge pairs
        requested_max_pairs = int(quota)
        if "terminal" in component.component_id:
            requested_max_pairs = max(int(math.ceil(float(quota) / 0.90)), int(quota) + 64)
        rows = load_rlhflow_pairs(
            mistral_root=None,
            deepseek_path=rlhflow_deepseek_path,
            config=config,
            max_pairs_per_source=requested_max_pairs,
        )
        if "terminal" in component.component_id:
            rows = [
                row
                for row in rows
                if str((row.metadata or {}).get("pair_semantics", "")) == "terminal_completion_anchor"
            ]
        elif component.pair_type_allowlist:
            rows = _filter_rows_by_pair_type(
                rows=rows,
                pair_type_allowlist=component.pair_type_allowlist,
            )
    elif component.source_kind == "math_step_dpo":
        # Math-Step-DPO-10K: 显式分叉点 sibling_branch pairs，高质量小规模
        # Math-Step-DPO-10K: explicit fork-point sibling_branch pairs, high quality small scale
        rows = load_math_step_dpo_pairs(
            path=math_step_dpo_path,
            config=config,
            max_pairs=int(quota),
        )
    else:
        raise ValueError(f"Unsupported component source_kind: {component.source_kind!r}")
    return _stable_select_rows(rows=rows, quota=int(quota), namespace=component.component_id)


def _stable_select_rows(
    *,
    rows: list[ExternalPairRecord],
    quota: int,
    namespace: str,
) -> list[ExternalPairRecord]:
    if quota <= 0 or not rows:
        return []
    if len(rows) <= quota:
        return list(rows)
    ranked = sorted(
        rows,
        key=lambda row: _stable_hash(f"{namespace}|{row.pair_id}"),
    )
    return ranked[: int(quota)]


def _filter_rows_by_pair_type(
    *,
    rows: list[ExternalPairRecord],
    pair_type_allowlist: tuple[str, ...],
) -> list[ExternalPairRecord]:
    """Keep only rows whose geometry maps to the requested ProcessBench slice types.

    English
    -------
    `all_good_vs_all_bad` is still too broad for the current bottleneck.
    This filter lets one curation profile ask a more direct question:
    can we improve `later-bad` generalization specifically, instead of adding
    every non-local relation at once?

    中文
    ----
    对当前问题来说，`all_good_vs_all_bad` 还是太宽。
    这个过滤器的作用是把问题问得更尖锐：
    我们到底能不能只补 `later-bad` 这类关系，而不是一口气把所有非局部关系都塞进来？
    """
    allowed = {str(item) for item in pair_type_allowlist}
    kept: list[ExternalPairRecord] = []
    for row in rows:
        metadata = dict(row.metadata or {})
        pair_type = classify_step_label_pair_type(metadata)
        if pair_type in allowed:
            kept.append(row)
    return kept


def _tag_component_rows(
    *,
    rows: list[ExternalPairRecord],
    component: CurateComponent,
    profile_name: str,
) -> list[ExternalPairRecord]:
    tagged: list[ExternalPairRecord] = []
    for row in rows:
        metadata = deepcopy(dict(row.metadata or {}))
        metadata["curation_profile"] = str(profile_name)
        metadata["curation_component"] = str(component.component_id)
        metadata["semantic_weight"] = float(component.semantic_weight)
        metadata["artifact_mix_source_label"] = str(component.component_id)
        tagged.append(
            ExternalPairRecord(
                pair_id=str(row.pair_id),
                source_tag=str(row.source_tag),
                domain_tag=str(row.domain_tag),
                prompt_text=str(row.prompt_text),
                chosen_text=str(row.chosen_text),
                rejected_text=str(row.rejected_text),
                pair_confidence=float(row.pair_confidence),
                quality_flags=deepcopy(dict(row.quality_flags or {})),
                metadata=metadata,
            )
        )
    return tagged


def _stable_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase E ProcessBench Curated Pair Artifact",
        "",
        f"- profile: `{summary['profile']}`",
        f"- profile_title: `{summary['profile_title']}`",
        f"- profile_intent: {summary['profile_intent']}",
        f"- run_dir: `{summary['run_dir']}`",
        f"- num_rows_after_dedup: `{summary['num_rows_after_dedup']}`",
        f"- num_train_rows: `{summary['num_train_rows']}`",
        f"- num_validation_rows: `{summary['num_validation_rows']}`",
        "",
        "## Components",
        "",
        "| component | source | target_pairs | selected_pairs | semantic_weight |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary["component_summaries"]:
        lines.append(
            f"| {row['component_id']} | {row['source_kind']} | {row['target_pairs']} | "
            f"{row['selected_pairs']} | {row['semantic_weight']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Overall Summary",
            "",
            f"- overall_summary: `{json.dumps(summary['overall_summary'], ensure_ascii=False, sort_keys=True)}`",
            f"- train_summary: `{json.dumps(summary['train_summary'], ensure_ascii=False, sort_keys=True)}`",
            f"- validation_summary: `{json.dumps(summary['validation_summary'], ensure_ascii=False, sort_keys=True)}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
