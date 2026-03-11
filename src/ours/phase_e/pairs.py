"""Phase E pair-artifact preparation helpers.

English
-------
Phase D established that a canonical external-pair artifact is the right
intermediate format between raw datasets and value-head training.

Phase E keeps that idea, but repositions it:
1. the artifact is now benchmark-native,
2. it is no longer entangled with StrategyQA,
3. it is the primary supervised input to training.

This module is therefore responsible for turning a source bundle plus a
pair-building policy into one deterministic artifact directory.

中文
----
Phase D 已经证明，canonical external-pair artifact 是连接“原始外部数据”和
“value head 训练”的合适中间格式。

Phase E 延续了这个设计，但重新定义了它的角色：
1. artifact 现在是 benchmark-native 的，
2. 不再和 StrategyQA 绑定，
3. 它本身就是训练主监督输入。

因此本模块负责把“source bundle + pair 构造策略”落成一个确定性的 artifact 目录。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ours.phase_d.external_pairs import ExternalPairRecord, summarize_external_pairs
from ours.phase_d.external_pairs_adapters import (
    PairBuildConfig,
    load_math_shepherd_pairs,
    load_prm800k_pairs,
    load_prmbench_preview_pairs,
    load_r_prm_dpo_pairs,
)

from .contracts import PhaseEPairSourceSpec


@dataclass(slots=True)
class PhaseEPairArtifact:
    """Bundle together the main Phase E pair artifact paths.

    中文
    ----
    这个 dataclass 的作用是把一次构造 pair artifact 后最常用的路径和摘要统一打包，
    这样调用方不需要自己手动拼路径。
    """

    run_dir: Path
    train_pairs_path: Path
    validation_pairs_path: Path
    summary_path: Path
    manifest_path: Path
    summary_md_path: Path
    summary: dict[str, Any]
    manifest: dict[str, Any]


def prepare_phase_e_pair_artifact(
    *,
    run_name: str,
    output_root: Path,
    source_specs: list[PhaseEPairSourceSpec],
    build_config: PairBuildConfig,
    seed: int,
    validation_ratio: float,
    max_pairs_total: int | None,
    max_pairs_per_source: int | None,
    min_pair_confidence: float,
    split_granularity: str = "pair_id",
    global_cap_mode: str = "pair_id_head",
    source_weight_overrides: dict[str, float] | None = None,
    resume: bool,
    overwrite: bool,
) -> PhaseEPairArtifact:
    """Build one deterministic Phase E pair artifact on disk.

    English
    -------
    "Deterministic" here means:
    1. the same logical config fingerprints to the same run directory name,
    2. resume mode can safely reuse a complete existing artifact,
    3. summaries/manifests record the exact build contract.

    中文
    ----
    这里说的“确定性”主要有三层意思：
    1. 相同实验契约会映射到相同的 fingerprint 和目录名；
    2. `resume=True` 时可以安全复用一份完整现有 artifact；
    3. summary/manifest 会完整记录这次构造所依据的契约。
    """
    build_config.validate()
    if not source_specs:
        raise ValueError("At least one Phase E source spec is required")
    if not (0.0 < float(validation_ratio) < 0.5):
        raise ValueError("`validation_ratio` must be in (0, 0.5)")
    if not (0.0 <= float(min_pair_confidence) <= 1.0):
        raise ValueError("`min_pair_confidence` must be in [0, 1]")
    if str(split_granularity) not in {"pair_id", "source_sample"}:
        raise ValueError("`split_granularity` must be one of {'pair_id', 'source_sample'}")
    if str(global_cap_mode) not in {"pair_id_head", "balanced_support_bucket"}:
        raise ValueError(
            "`global_cap_mode` must be one of {'pair_id_head', 'balanced_support_bucket'}"
        )

    # We fingerprint the *full* artifact contract, not just the source bundle
    # name.  Any meaningful change in split seed, filtering, pair mode, or
    # source weighting should produce a new artifact id.
    # 这里对“完整实验契约”做 fingerprint，而不只是对 source bundle 名称做摘要。
    # 只要切分 seed、过滤阈值、pair mode、source weight 等有实质变化，
    # 就应当得到新的 artifact id。
    fingerprint = _stable_fingerprint(
        {
            "run_name": str(run_name),
            "seed": int(seed),
            "validation_ratio": float(validation_ratio),
            "max_pairs_total": max_pairs_total,
            "max_pairs_per_source": max_pairs_per_source,
            "min_pair_confidence": float(min_pair_confidence),
            "split_granularity": str(split_granularity),
            "global_cap_mode": str(global_cap_mode),
            "source_weight_overrides": dict(sorted((source_weight_overrides or {}).items())),
            "build_config": asdict(build_config),
            "source_specs": [spec.to_dict() for spec in source_specs],
        }
    )
    run_dir = output_root / f"{run_name}__{fingerprint}"
    train_path = run_dir / "train_pairs.jsonl"
    validation_path = run_dir / "validation_pairs.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    summary_md_path = run_dir / "summary.md"

    # Resume is allowed only when *all* canonical outputs exist.
    # Otherwise a half-written directory could be mistaken for a valid artifact.
    # 只有所有标准产物都存在时才允许 resume，
    # 否则半成品目录可能会被误当成有效 artifact。
    if run_dir.exists() and not bool(overwrite) and bool(resume):
        if all(path.exists() for path in (train_path, validation_path, summary_path, manifest_path, summary_md_path)):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return PhaseEPairArtifact(
                run_dir=run_dir,
                train_pairs_path=train_path,
                validation_pairs_path=validation_path,
                summary_path=summary_path,
                manifest_path=manifest_path,
                summary_md_path=summary_md_path,
                summary=summary,
                manifest=manifest,
            )

    run_dir.mkdir(parents=True, exist_ok=True)

    # First load rows source by source, then run one shared filtering pipeline.
    # This keeps the summaries comparable across heterogeneous sources.
    # 先按 source 分别加载，再统一走一套过滤流程，才能让不同 source 的 summary 可比。
    all_rows: list[ExternalPairRecord] = []
    source_rows_before_filter: dict[str, int] = {}
    resolved_inputs: dict[str, dict[str, Any]] = {}
    for spec in source_specs:
        rows = _load_rows_for_source(
            spec=spec,
            config=build_config,
            max_pairs=max_pairs_per_source,
            source_weight_overrides=source_weight_overrides,
        )
        source_rows_before_filter[spec.source_id] = int(len(rows))
        resolved_inputs[spec.source_id] = {
            "source_type": spec.source_type,
            "path": str(spec.default_path_obj()),
            "split": spec.default_split,
            "source_weight": float((source_weight_overrides or {}).get(spec.source_id, 1.0)),
        }
        all_rows.extend(rows)

    # Filtering order matters:
    # 1. confidence filter first,
    # 2. dedup second,
    # 3. train/validation split last.
    #
    # This way downstream counts and metrics reflect the supervision pool that
    # actually enters training.
    # 过滤顺序很重要：
    # 1. 先按 confidence 过滤，
    # 2. 再去重，
    # 3. 最后切 train/validation。
    #
    # 这样后续统计的数量和指标，才能真实反映“最终进入训练监督池”的样本。
    filtered_rows = [
        row for row in all_rows if float(row.pair_confidence) >= float(min_pair_confidence)
    ]
    dedup_rows = _deduplicate_pairs(filtered_rows)
    dedup_rows.sort(key=lambda item: item.pair_id)
    dedup_rows_before_global_cap = list(dedup_rows)
    dedup_rows, global_cap_summary = _apply_global_cap(
        rows=dedup_rows,
        max_pairs_total=max_pairs_total,
        global_cap_mode=str(global_cap_mode),
    )

    train_rows, validation_rows = _split_train_validation(
        rows=dedup_rows,
        seed=int(seed),
        validation_ratio=float(validation_ratio),
        split_granularity=str(split_granularity),
    )
    _write_jsonl(train_path, [row.to_dict() for row in train_rows])
    _write_jsonl(validation_path, [row.to_dict() for row in validation_rows])

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "num_rows_before_filter": int(len(all_rows)),
        "num_rows_after_confidence_filter": int(len(filtered_rows)),
        "num_rows_after_dedup_before_global_cap": int(len(dedup_rows_before_global_cap)),
        "num_rows_after_dedup": int(len(dedup_rows)),
        "num_train_rows": int(len(train_rows)),
        "num_validation_rows": int(len(validation_rows)),
        "num_split_units_after_dedup": int(_count_split_units(rows=dedup_rows, split_granularity=str(split_granularity))),
        "num_train_split_units": int(_count_split_units(rows=train_rows, split_granularity=str(split_granularity))),
        "num_validation_split_units": int(
            _count_split_units(rows=validation_rows, split_granularity=str(split_granularity))
        ),
        "source_rows_before_filter": dict(sorted(source_rows_before_filter.items())),
        "overall_summary_before_global_cap": summarize_external_pairs(dedup_rows_before_global_cap),
        "overall_summary": summarize_external_pairs(dedup_rows),
        "global_cap_summary": global_cap_summary,
        "train_summary": summarize_external_pairs(train_rows),
        "validation_summary": summarize_external_pairs(validation_rows),
        "build_config": {
            "seed": int(seed),
            "validation_ratio": float(validation_ratio),
            "max_pairs_total": max_pairs_total,
            "max_pairs_per_source": max_pairs_per_source,
            "min_pair_confidence": float(min_pair_confidence),
            "split_granularity": str(split_granularity),
            "global_cap_mode": str(global_cap_mode),
            "source_weight_overrides": dict(sorted((source_weight_overrides or {}).items())),
            **asdict(build_config),
        },
        "source_specs": [spec.to_dict() for spec in source_specs],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "artifact_stage": "phase_e_pairs_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": str(run_name),
        "fingerprint": str(fingerprint),
        "run_dir": str(run_dir),
        "source_inputs": resolved_inputs,
        "output_files": {
            "train_pairs": str(train_path),
            "validation_pairs": str(validation_path),
            "summary": str(summary_path),
            "summary_md": str(summary_md_path),
        },
        "summary_snapshot": summary,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")
    return PhaseEPairArtifact(
        run_dir=run_dir,
        train_pairs_path=train_path,
        validation_pairs_path=validation_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        summary_md_path=summary_md_path,
        summary=summary,
        manifest=manifest,
    )


def _load_rows_for_source(
    *,
    spec: PhaseEPairSourceSpec,
    config: PairBuildConfig,
    max_pairs: int | None,
    source_weight_overrides: dict[str, float] | None,
) -> list[ExternalPairRecord]:
    """Load canonical external pairs for one Phase E source.

    中文
    ----
    这里的职责是：
    1. 根据 source type 分发到对应 adapter，
    2. 把结果统一成 `ExternalPairRecord`，
    3. 再补上 source weight 等训练时需要的元信息。
    """
    if spec.source_type == "math_shepherd":
        rows = load_math_shepherd_pairs(
            path=spec.default_path_obj(),
            config=config,
            max_pairs=max_pairs,
        )
    elif spec.source_type == "prm800k":
        rows = load_prm800k_pairs(
            path=spec.default_path_obj(),
            config=config,
            max_pairs=max_pairs,
        )
    elif spec.source_type == "prmbench_preview":
        rows = load_prmbench_preview_pairs(
            path=spec.default_path_obj(),
            config=config,
            max_pairs=max_pairs,
        )
    elif spec.source_type == "r_prm":
        rows = load_r_prm_dpo_pairs(
            root=spec.default_path_obj(),
            split=str(spec.default_split or "train"),
            config=config,
            max_pairs=max_pairs,
        )
    else:
        raise ValueError(f"Unsupported Phase E source_type: {spec.source_type!r}")
    source_weight = float((source_weight_overrides or {}).get(spec.source_id, 1.0))
    if source_weight <= 0.0:
        raise ValueError(
            f"source_weight_overrides[{spec.source_id!r}] must be > 0, got {source_weight}"
        )
    # Keep source weighting explicit in metadata so later stages can distinguish:
    # 1. the source was kept but down-weighted,
    # 2. the source was removed entirely.
    # 把 source weight 明确写进 metadata，后续才能区分：
    # 1. 这个 source 被保留但低权重使用，
    # 2. 还是它被彻底删掉了。
    for row in rows:
        row.metadata["source_weight"] = float(source_weight)
    return rows


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    """Return a short deterministic fingerprint for one pair-artifact config.

    中文
    ----
    这个 fingerprint 的目的不是安全，而是给实验目录命名：
    让“同一份构造契约”稳定映射到同一个简短 id。
    """
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _deduplicate_pairs(rows: list[ExternalPairRecord]) -> list[ExternalPairRecord]:
    """Drop duplicate pairs by canonical `pair_id`."""
    dedup: dict[str, ExternalPairRecord] = {}
    for row in rows:
        # Keep the first occurrence so source-order precedence stays deterministic.
        # 保留首次出现的 pair，保证跨来源合并时的优先级是确定性的。
        if row.pair_id not in dedup:
            dedup[row.pair_id] = row
    return list(dedup.values())


def _apply_global_cap(
    *,
    rows: list[ExternalPairRecord],
    max_pairs_total: int | None,
    global_cap_mode: str,
) -> tuple[list[ExternalPairRecord], dict[str, Any]]:
    """Apply one deterministic global cap policy to the deduplicated pair pool.

    English
    -------
    Phase E historically sorted by `pair_id` and took the head of the list.
    That keeps exact reproducibility, but when a new repair semantic is rare
    and appended later in source order, a small smoke-time `max_pairs_total`
    can silently erase the repair from the artifact.

    `balanced_support_bucket` keeps the pool deterministic but performs a
    round-robin over `(source_tag, pair_semantics)` buckets before the final
    cap. This is intentionally diagnostic: it preserves minority supervision
    types so small smoke suites still test the repaired geometry.

    中文
    ----
    Phase E 过去的做法是按 `pair_id` 排序后直接截头。这保证了可复现性，但当某种
    修复语义本来就稀疏、而且在源数据里出现得更晚时，小规模 smoke 的
    `max_pairs_total` 会把它静默截没。

    `balanced_support_bucket` 仍然是确定性的，但会先按 `(source_tag, pair_semantics)`
    做 round-robin，再施加最终 cap。这个模式的定位是诊断友好：
    让便宜 smoke 也能真正测到稀疏修复监督是否有帮助。
    """
    sorted_rows = sorted(rows, key=lambda item: item.pair_id)
    summary_before = _summarize_cap_buckets(sorted_rows)
    if max_pairs_total is None or len(sorted_rows) <= int(max_pairs_total):
        return sorted_rows, {
            "global_cap_mode": str(global_cap_mode),
            "max_pairs_total": (int(max_pairs_total) if max_pairs_total is not None else None),
            "num_rows_before_global_cap": int(len(sorted_rows)),
            "num_rows_after_global_cap": int(len(sorted_rows)),
            "bucket_summary_before": summary_before,
            "bucket_summary_after": summary_before,
            "cap_applied": False,
        }
    cap = int(max_pairs_total)
    if str(global_cap_mode) == "pair_id_head":
        capped_rows = sorted_rows[:cap]
    elif str(global_cap_mode) == "balanced_support_bucket":
        capped_rows = _select_round_robin_by_support_bucket(sorted_rows, cap=cap)
    else:
        raise ValueError(f"Unsupported global_cap_mode: {global_cap_mode!r}")
    return capped_rows, {
        "global_cap_mode": str(global_cap_mode),
        "max_pairs_total": int(max_pairs_total),
        "num_rows_before_global_cap": int(len(sorted_rows)),
        "num_rows_after_global_cap": int(len(capped_rows)),
        "bucket_summary_before": summary_before,
        "bucket_summary_after": _summarize_cap_buckets(capped_rows),
        "cap_applied": True,
    }


def _select_round_robin_by_support_bucket(
    rows: list[ExternalPairRecord],
    *,
    cap: int,
) -> list[ExternalPairRecord]:
    """Deterministically keep a capped set while preserving rare support buckets."""
    buckets: dict[str, list[ExternalPairRecord]] = {}
    for row in rows:
        bucket_key = _support_bucket_key(row)
        buckets.setdefault(bucket_key, []).append(row)
    for bucket_rows in buckets.values():
        bucket_rows.sort(key=lambda item: item.pair_id)
    ordered_bucket_keys = sorted(buckets)
    next_index = {key: 0 for key in ordered_bucket_keys}
    selected: list[ExternalPairRecord] = []
    while len(selected) < int(cap):
        progressed = False
        for bucket_key in ordered_bucket_keys:
            bucket_rows = buckets[bucket_key]
            idx = next_index[bucket_key]
            if idx >= len(bucket_rows):
                continue
            selected.append(bucket_rows[idx])
            next_index[bucket_key] = idx + 1
            progressed = True
            if len(selected) >= int(cap):
                break
        if not progressed:
            break
    selected.sort(key=lambda item: item.pair_id)
    return selected


def _support_bucket_key(row: ExternalPairRecord) -> str:
    """Return the coarse support bucket used by semantic-aware global caps."""
    pair_semantics = str((row.metadata or {}).get("pair_semantics", "unspecified"))
    return f"{row.source_tag}|{pair_semantics}"


def _summarize_cap_buckets(rows: list[ExternalPairRecord]) -> dict[str, int]:
    """Count rows per support bucket for before/after-cap diagnostics."""
    counts: dict[str, int] = {}
    for row in rows:
        bucket_key = _support_bucket_key(row)
        counts[bucket_key] = counts.get(bucket_key, 0) + 1
    return dict(sorted(counts.items()))


def _split_train_validation(
    *,
    rows: list[ExternalPairRecord],
    seed: int,
    validation_ratio: float,
    split_granularity: str,
) -> tuple[list[ExternalPairRecord], list[ExternalPairRecord]]:
    """Split rows deterministically using pair ids or source-sample groups.

    English
    -------
    `pair_id` mode reproduces the historical behavior exactly: every canonical
    pair is split independently.

    `source_sample` mode keeps all pairs derived from the same raw source sample
    on the same side of the split. This is stricter and avoids near-duplicate
    supervision leaking across train/validation when one raw sample emits
    multiple pairs.

    中文
    ----
    `pair_id` 模式完全复现历史行为：每个 canonical pair 独立切分。

    `source_sample` 模式会把同一个原始样本衍生出的所有 pair 固定放在同一侧，
    这样当一个样本能产生多个相关 pair 时，就不会把近重复监督泄漏到 train/validation 两边。
    """
    unit_groups = _group_rows_by_split_unit(rows=rows, split_granularity=str(split_granularity))
    train_unit_keys: list[str] = []
    validation_unit_keys: list[str] = []
    # Split by a deterministic unit key instead of list position so source
    # reordering does not reshuffle train/val.
    # 按稳定 unit key 而不是列表位置切分，这样来源顺序变化不会让 train/val 漂移。
    for unit_key in sorted(unit_groups):
        if _is_validation_pair(pair_id=unit_key, seed=seed, ratio=validation_ratio):
            validation_unit_keys.append(unit_key)
        else:
            train_unit_keys.append(unit_key)
    # Never allow an empty split, because all downstream suites assume both
    # files exist and are non-empty. When grouping is enabled we move an entire
    # split unit instead of breaking that grouping contract.
    # 不允许出现空 split，因为后续 suite 默认 train/val 都存在且非空。
    # 如果启用了分组切分，这里要整体移动一个 split unit，而不是打破分组契约。
    if not train_unit_keys and validation_unit_keys:
        train_unit_keys.append(validation_unit_keys.pop())
    if not validation_unit_keys and train_unit_keys:
        validation_unit_keys.append(train_unit_keys.pop())
    train_rows = [row for unit_key in train_unit_keys for row in unit_groups[unit_key]]
    validation_rows = [row for unit_key in validation_unit_keys for row in unit_groups[unit_key]]
    return train_rows, validation_rows


def _group_rows_by_split_unit(
    *,
    rows: list[ExternalPairRecord],
    split_granularity: str,
) -> dict[str, list[ExternalPairRecord]]:
    """Group rows by the deterministic split unit requested by the caller."""
    if split_granularity not in {"pair_id", "source_sample"}:
        raise ValueError(f"Unsupported split_granularity: {split_granularity!r}")
    groups: dict[str, list[ExternalPairRecord]] = {}
    for row in rows:
        if split_granularity == "pair_id":
            unit_key = str(row.pair_id)
        else:
            unit_key = _resolve_split_group_id(row)
        groups.setdefault(unit_key, []).append(row)
    return groups


def _count_split_units(*, rows: list[ExternalPairRecord], split_granularity: str) -> int:
    """Return how many effective split units a row set contains."""
    return int(len(_group_rows_by_split_unit(rows=rows, split_granularity=str(split_granularity))))


def _resolve_split_group_id(row: ExternalPairRecord) -> str:
    """Return a stable source-sample split key for one canonical pair row.

    English
    -------
    Adapters now try to populate `metadata.split_group_id` explicitly. This
    helper still keeps a conservative fallback path so older artifacts or ad-hoc
    rows can be grouped without crashing.

    中文
    ----
    adapter 现在会尽量显式写入 `metadata.split_group_id`。但这里仍然保留保守回退，
    这样旧产物或临时构造的行也能正常按组切分，而不是直接崩掉。
    """
    metadata = dict(row.metadata or {})
    explicit = str(metadata.get("split_group_id", "")).strip()
    if explicit:
        return f"{row.source_tag}|{explicit}"
    for key in ("source_idx", "source_row_index", "source_row_line", "source_line"):
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return f"{row.source_tag}|{key}={text}"
    prompt_fingerprint = hashlib.sha256(
        f"{row.source_tag}\n{row.prompt_text}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{row.source_tag}|prompt={prompt_fingerprint}"


def _is_validation_pair(*, pair_id: str, seed: int, ratio: float) -> bool:
    """Return whether one pair id belongs to the validation split."""
    digest = hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).hexdigest()
    value = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    return value < float(ratio)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dict rows into one JSONL file."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    """Render one compact human-readable Phase E pair summary."""
    lines = [
        "# Phase E Pair Summary",
        "",
        f"- generated_at: {summary.get('generated_at')}",
        f"- run_dir: {summary.get('run_dir')}",
        f"- rows_before_filter: {summary.get('num_rows_before_filter')}",
        f"- rows_after_conf_filter: {summary.get('num_rows_after_confidence_filter')}",
        f"- rows_after_dedup_before_global_cap: {summary.get('num_rows_after_dedup_before_global_cap')}",
        f"- rows_after_dedup: {summary.get('num_rows_after_dedup')}",
        f"- train_rows: {summary.get('num_train_rows')}",
        f"- validation_rows: {summary.get('num_validation_rows')}",
        f"- split_granularity: {summary.get('build_config', {}).get('split_granularity')}",
        f"- global_cap_mode: {summary.get('build_config', {}).get('global_cap_mode')}",
        f"- split_units_after_dedup: {summary.get('num_split_units_after_dedup')}",
        f"- train_split_units: {summary.get('num_train_split_units')}",
        f"- validation_split_units: {summary.get('num_validation_split_units')}",
        "",
        "## By Source (Before Filter)",
        "",
    ]
    for key, value in sorted((summary.get("source_rows_before_filter") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Train Summary",
            "",
            f"- global_cap_summary: {summary.get('global_cap_summary')}",
            f"- num_pairs: {summary.get('train_summary', {}).get('num_pairs')}",
            f"- mean_pair_confidence: {summary.get('train_summary', {}).get('mean_pair_confidence')}",
            f"- by_pair_build_mode: {summary.get('train_summary', {}).get('by_pair_build_mode')}",
            f"- by_pair_semantics: {summary.get('train_summary', {}).get('by_pair_semantics')}",
            "",
            "## Validation Summary",
            "",
            f"- num_pairs: {summary.get('validation_summary', {}).get('num_pairs')}",
            f"- mean_pair_confidence: {summary.get('validation_summary', {}).get('mean_pair_confidence')}",
            f"- by_pair_build_mode: {summary.get('validation_summary', {}).get('by_pair_build_mode')}",
            f"- by_pair_semantics: {summary.get('validation_summary', {}).get('by_pair_semantics')}",
            "",
        ]
    )
    return "\n".join(lines)
