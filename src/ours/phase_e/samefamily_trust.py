"""Same-family trust evaluation helpers for Phase E value heads.

English
-------
Phase E has already shown that same-source held-out pair discrimination can be
very high.  That is necessary, but still not enough for RL-style use.

The next tighter question is:
1. if we stay *inside the same dataset family*,
2. and we build prompt-level candidate pools from held-out supervision pairs,
3. can the value head support useful downstream choices such as reranking,
   rejection, and higher-pressure best-of-N selection?

This module computes exactly those offline utility views.

中文
----
Phase E 已经证明，同源 held-out pair 上的判别可以很高；
但这还不足以支撑 RL 场景下的“可信使用”。

接下来更严格、也更贴近下游的问题是：
1. 如果我们严格留在“同一个数据集家族”内部，
2. 并把 held-out pair 还原成按题目分组的候选池，
3. 当前 value head 能不能支持真正有用的下游决策，
   例如 rerank、拒答/保守接收、以及更强选择压力下的 best-of-N？

本模块就是为这些离线 utility 视角服务的。
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ours.phase_d.external_pairs import ExternalPairRecord
from ours.phase_e.runtime import stable_hash_order


@dataclass(slots=True)
class CandidateNode:
    """One unique candidate text inside one prompt-level pool.

    中文
    ----
    一个候选不是“整条 pair 记录”，而是“同一道题下某个唯一的 prefix / process 文本”。
    同一个候选可能在不同 pair 中反复出现，因此这里要把它单独抽出来建图。
    """

    prompt_key: str
    prompt_text: str
    source_tag: str
    candidate_id: str
    candidate_text: str
    safe_step_indices: list[int] = field(default_factory=list)
    bad_step_indices: list[int] = field(default_factory=list)

    @property
    def max_safe_step_index(self) -> int | None:
        return max(self.safe_step_indices) if self.safe_step_indices else None

    @property
    def min_bad_step_index(self) -> int | None:
        return min(self.bad_step_indices) if self.bad_step_indices else None

    def input_text(self) -> str:
        """Return the exact text scored by the frozen backbone.

        中文
        ----
        这里保留 prompt + candidate 的完整拼接文本，确保 same-family trust 评测
        与训练时的打分对象保持一致。
        """
        return f"{self.prompt_text}{self.candidate_text}"


@dataclass(slots=True)
class PreferenceEdge:
    """One directed preference edge inside one prompt pool.

    中文
    ----
    这里的边表示：在同一道题里，`chosen_id` 应当优于 `rejected_id`。
    后续所有 prompt-level utility 指标，都会以这些边为经验监督图。
    """

    prompt_key: str
    pair_id: str
    chosen_id: str
    rejected_id: str
    confidence: float
    metadata: dict[str, Any]


@dataclass(slots=True)
class PromptPool:
    """One prompt-level candidate pool reconstructed from held-out pairs.

    中文
    ----
    这个对象是 same-family utility 评测的核心单位：
    一道题 -> 若干候选 -> 若干 preference edges。
    """

    prompt_key: str
    prompt_text: str
    source_tag: str
    candidate_ids: list[str]
    edges: list[PreferenceEdge]


@dataclass(slots=True)
class RejectionPoint:
    """One operating point on the rejection / abstention curve."""

    target_coverage: float
    actual_coverage: float
    min_gap_threshold: float
    accepted_prompts: int
    top1_accuracy: float
    mean_regret: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_coverage": float(self.target_coverage),
            "actual_coverage": float(self.actual_coverage),
            "min_gap_threshold": float(self.min_gap_threshold),
            "accepted_prompts": int(self.accepted_prompts),
            "top1_accuracy": float(self.top1_accuracy),
            "mean_regret": float(self.mean_regret),
        }


@dataclass(slots=True)
class PressurePoint:
    """One best-of-N stress point."""

    subset_size: int
    num_prompt_subsets: int
    top1_accuracy: float
    mean_regret: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "subset_size": int(self.subset_size),
            "num_prompt_subsets": int(self.num_prompt_subsets),
            "top1_accuracy": float(self.top1_accuracy),
            "mean_regret": float(self.mean_regret),
        }


@dataclass(slots=True)
class PromptTrustRow:
    """One prompt-level row for inspection/debugging output.

    中文
    ----
    这个结构用于把“按题聚合后”的关键决策量写回 JSONL，方便人工 spot-check：
    例如为什么某一道题会被 rerank 选错，或者为何 gap 很大但其实是错选。
    """

    prompt_key: str
    source_tag: str
    num_candidates: int
    num_edges: int
    model_top_candidate_id: str
    gold_top_candidate_ids: list[str]
    model_top_score: float
    model_second_score: float | None
    score_gap: float | None
    model_top_hit: bool
    selected_utility: float
    gold_utility: float
    regret: float
    first_bad_index: int | None
    last_safe_candidate_ids: list[str]
    local_last_safe_hit: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_key": self.prompt_key,
            "source_tag": self.source_tag,
            "num_candidates": int(self.num_candidates),
            "num_edges": int(self.num_edges),
            "model_top_candidate_id": self.model_top_candidate_id,
            "gold_top_candidate_ids": list(self.gold_top_candidate_ids),
            "model_top_score": float(self.model_top_score),
            "model_second_score": (
                None if self.model_second_score is None else float(self.model_second_score)
            ),
            "score_gap": None if self.score_gap is None else float(self.score_gap),
            "model_top_hit": bool(self.model_top_hit),
            "selected_utility": float(self.selected_utility),
            "gold_utility": float(self.gold_utility),
            "regret": float(self.regret),
            "first_bad_index": (
                None if self.first_bad_index is None else int(self.first_bad_index)
            ),
            "last_safe_candidate_ids": list(self.last_safe_candidate_ids),
            "local_last_safe_hit": (
                None if self.local_last_safe_hit is None else bool(self.local_last_safe_hit)
            ),
        }


@dataclass(slots=True)
class SameFamilyTrustResult:
    """All same-family trust outputs computed from one held-out pair set."""

    metrics: dict[str, Any]
    prompt_rows: list[PromptTrustRow]


@dataclass(slots=True)
class _PoolRuntime:
    prompt_key: str
    prompt_text: str
    source_tag: str
    candidates: dict[str, CandidateNode]
    edges: list[PreferenceEdge]


def build_prompt_pools_from_pairs(pairs: list[ExternalPairRecord]) -> list[PromptPool]:
    """Convert held-out pairs into prompt-level candidate pools.

    中文
    ----
    训练时我们看到的是 chosen/rejected pair；
    但同源 utility 评测真正关心的是：同一道题下有多少候选、这些候选之间有哪些偏好边。
    因此这里先把 pair 还原成按 prompt 聚合的 pool。
    """
    prompt_buckets: dict[str, _PoolRuntime] = {}
    for pair in pairs:
        prompt_text = str(pair.prompt_text)
        source_tag = str(pair.source_tag)
        prompt_key = _pool_key(source_tag=source_tag, prompt_text=prompt_text)
        bucket = prompt_buckets.get(prompt_key)
        if bucket is None:
            bucket = _PoolRuntime(
                prompt_key=prompt_key,
                prompt_text=prompt_text,
                source_tag=source_tag,
                candidates={},
                edges=[],
            )
            prompt_buckets[prompt_key] = bucket

        chosen_id = _candidate_id(
            source_tag=source_tag,
            prompt_text=prompt_text,
            candidate_text=str(pair.chosen_text),
        )
        rejected_id = _candidate_id(
            source_tag=source_tag,
            prompt_text=prompt_text,
            candidate_text=str(pair.rejected_text),
        )

        chosen_node = bucket.candidates.get(chosen_id)
        if chosen_node is None:
            chosen_node = CandidateNode(
                prompt_key=prompt_key,
                prompt_text=prompt_text,
                source_tag=str(pair.source_tag),
                candidate_id=chosen_id,
                candidate_text=str(pair.chosen_text),
            )
            bucket.candidates[chosen_id] = chosen_node
        rejected_node = bucket.candidates.get(rejected_id)
        if rejected_node is None:
            rejected_node = CandidateNode(
                prompt_key=prompt_key,
                prompt_text=prompt_text,
                source_tag=str(pair.source_tag),
                candidate_id=rejected_id,
                candidate_text=str(pair.rejected_text),
            )
            bucket.candidates[rejected_id] = rejected_node

        positive_idx = _maybe_int((pair.metadata or {}).get("positive_step_index"))
        negative_idx = _maybe_int((pair.metadata or {}).get("negative_step_index"))
        if positive_idx is not None:
            chosen_node.safe_step_indices.append(int(positive_idx))
        if negative_idx is not None:
            rejected_node.bad_step_indices.append(int(negative_idx))

        bucket.edges.append(
            PreferenceEdge(
                prompt_key=prompt_key,
                pair_id=str(pair.pair_id),
                chosen_id=chosen_id,
                rejected_id=rejected_id,
                confidence=float(pair.pair_confidence),
                metadata=dict(pair.metadata or {}),
            )
        )

    pools: list[PromptPool] = []
    for bucket in prompt_buckets.values():
        candidate_ids = stable_hash_order(
            list(range(len(bucket.candidates))),
            ids=list(bucket.candidates.keys()),
        )
        ordered_candidate_ids = [list(bucket.candidates.keys())[idx] for idx in candidate_ids]
        pools.append(
            PromptPool(
                prompt_key=bucket.prompt_key,
                prompt_text=bucket.prompt_text,
                source_tag=bucket.source_tag,
                candidate_ids=ordered_candidate_ids,
                edges=list(bucket.edges),
            )
        )
    pools.sort(key=lambda item: item.prompt_key)
    return pools


def build_unique_candidate_rows(pairs: list[ExternalPairRecord]) -> tuple[list[CandidateNode], list[PromptPool]]:
    """Return unique candidates plus the prompt pools that reference them."""
    pools = build_prompt_pools_from_pairs(pairs)
    node_index: dict[str, CandidateNode] = {}
    prompt_text_by_key: dict[str, str] = {pool.prompt_key: pool.prompt_text for pool in pools}
    for pair in pairs:
        source_tag = str(pair.source_tag)
        prompt_text = str(pair.prompt_text)
        prompt_key = _pool_key(source_tag=source_tag, prompt_text=prompt_text)
        chosen_id = _candidate_id(
            source_tag=source_tag,
            prompt_text=prompt_text,
            candidate_text=str(pair.chosen_text),
        )
        rejected_id = _candidate_id(
            source_tag=source_tag,
            prompt_text=prompt_text,
            candidate_text=str(pair.rejected_text),
        )
        for candidate_id, candidate_text, safe_idx, bad_idx in (
            (chosen_id, str(pair.chosen_text), _maybe_int((pair.metadata or {}).get("positive_step_index")), None),
            (rejected_id, str(pair.rejected_text), None, _maybe_int((pair.metadata or {}).get("negative_step_index"))),
        ):
            node = node_index.get(candidate_id)
            if node is None:
                node = CandidateNode(
                    prompt_key=prompt_key,
                    prompt_text=prompt_text_by_key[prompt_key],
                    source_tag=str(pair.source_tag),
                    candidate_id=candidate_id,
                    candidate_text=candidate_text,
                )
                node_index[candidate_id] = node
            if safe_idx is not None:
                node.safe_step_indices.append(int(safe_idx))
            if bad_idx is not None:
                node.bad_step_indices.append(int(bad_idx))
    nodes = sorted(node_index.values(), key=lambda item: item.candidate_id)
    return nodes, pools


def compute_samefamily_trust_metrics(
    *,
    pools: list[PromptPool],
    candidate_nodes: list[CandidateNode],
    candidate_scores: dict[str, float],
    edge_weight_mode: str = "unit",
    rejection_coverages: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2),
    pressure_sizes: tuple[int, ...] = (2, 3, 4, 6, 8),
    pressure_repeats: int = 4,
) -> SameFamilyTrustResult:
    """Compute same-family rerank / rejection / pressure / local-step metrics.

    中文
    ----
    这是本模块的主入口。它把“候选分数 + prompt-level 偏好图”转换成四类离线信号：
    1. prompt-pool rerank utility
    2. rejection / abstention 曲线
    3. best-of-N 选择压力曲线
    4. 如果 source 带 step 位置信息，再补 local first-bad 诊断
    """
    if edge_weight_mode not in {"unit", "confidence"}:
        raise ValueError(f"Unsupported edge_weight_mode: {edge_weight_mode!r}")
    if pressure_repeats <= 0:
        raise ValueError("pressure_repeats must be > 0")

    node_map = {node.candidate_id: node for node in candidate_nodes}
    prompt_rows: list[PromptTrustRow] = []
    top1_hits: list[float] = []
    regrets: list[float] = []
    gaps: list[float] = []
    local_last_safe_hits: list[float] = []
    local_first_bad_hits: list[float] = []
    local_first_bad_total = 0
    local_safe_bad_hits: list[float] = []
    local_safe_bad_total = 0

    pool_payloads: list[dict[str, Any]] = []

    for pool in pools:
        if len(pool.candidate_ids) < 2 or len(pool.edges) == 0:
            continue
        utilities = _compute_candidate_utilities(pool=pool, edge_weight_mode=edge_weight_mode)
        ranked_candidates = sorted(
            pool.candidate_ids,
            key=lambda cid: (float(candidate_scores[cid]), cid),
            reverse=True,
        )
        top_cid = ranked_candidates[0]
        second_score = float(candidate_scores[ranked_candidates[1]]) if len(ranked_candidates) >= 2 else None
        top_score = float(candidate_scores[top_cid])
        score_gap = (top_score - second_score) if second_score is not None else None
        gold_utility = max(float(utilities[cid]) for cid in pool.candidate_ids)
        gold_top_ids = sorted([cid for cid in pool.candidate_ids if math.isclose(float(utilities[cid]), gold_utility, abs_tol=1e-9)])
        selected_utility = float(utilities[top_cid])
        hit = 1.0 if top_cid in gold_top_ids else 0.0
        regret = float(gold_utility - selected_utility)
        top1_hits.append(hit)
        regrets.append(regret)
        if score_gap is not None:
            gaps.append(float(score_gap))

        first_bad_index, last_safe_ids = _resolve_local_first_bad(pool=pool, node_map=node_map)
        local_last_safe_hit: bool | None = None
        if first_bad_index is not None and last_safe_ids:
            local_last_safe_hit = bool(top_cid in last_safe_ids)
            local_last_safe_hits.append(1.0 if local_last_safe_hit else 0.0)
            first_bad_ids = _first_bad_candidate_ids(pool=pool, node_map=node_map, first_bad_index=int(first_bad_index))
            for safe_id in last_safe_ids:
                for bad_id in first_bad_ids:
                    local_first_bad_total += 1
                    local_first_bad_hits.append(1.0 if float(candidate_scores[safe_id]) > float(candidate_scores[bad_id]) else 0.0)
            safe_ids = _all_safe_candidate_ids(pool=pool, node_map=node_map, first_bad_index=int(first_bad_index))
            bad_ids = _all_bad_candidate_ids(pool=pool, node_map=node_map, first_bad_index=int(first_bad_index))
            for safe_id in safe_ids:
                for bad_id in bad_ids:
                    local_safe_bad_total += 1
                    local_safe_bad_hits.append(1.0 if float(candidate_scores[safe_id]) > float(candidate_scores[bad_id]) else 0.0)

        prompt_rows.append(
            PromptTrustRow(
                prompt_key=pool.prompt_key,
                source_tag=pool.source_tag,
                num_candidates=len(pool.candidate_ids),
                num_edges=len(pool.edges),
                model_top_candidate_id=top_cid,
                gold_top_candidate_ids=gold_top_ids,
                model_top_score=float(top_score),
                model_second_score=(None if second_score is None else float(second_score)),
                score_gap=(None if score_gap is None else float(score_gap)),
                model_top_hit=bool(hit > 0.0),
                selected_utility=float(selected_utility),
                gold_utility=float(gold_utility),
                regret=float(regret),
                first_bad_index=first_bad_index,
                last_safe_candidate_ids=list(last_safe_ids),
                local_last_safe_hit=local_last_safe_hit,
            )
        )
        pool_payloads.append(
            {
                "pool": pool,
                "utilities": utilities,
                "ranked_candidates": ranked_candidates,
                "score_gap": score_gap,
            }
        )

    rejection_points = _compute_rejection_curve(
        prompt_rows=prompt_rows,
        target_coverages=tuple(float(v) for v in rejection_coverages),
    )
    pressure_points = _compute_pressure_curve(
        pool_payloads=pool_payloads,
        candidate_scores=candidate_scores,
        subset_sizes=tuple(int(v) for v in pressure_sizes),
        repeats=int(pressure_repeats),
    )

    metrics = {
        "num_prompt_pools": int(len(prompt_rows)),
        "num_candidate_nodes": int(len(candidate_nodes)),
        "prompt_pool_top1_accuracy": _safe_mean(top1_hits),
        "prompt_pool_mean_regret": _safe_mean(regrets),
        "prompt_pool_mean_score_gap": _safe_mean(gaps),
        "local_last_safe_top1_accuracy": _safe_mean(local_last_safe_hits) if local_last_safe_hits else None,
        "local_first_bad_edge_accuracy": _safe_mean(local_first_bad_hits) if local_first_bad_hits else None,
        "local_safe_vs_bad_pair_accuracy": _safe_mean(local_safe_bad_hits) if local_safe_bad_hits else None,
        "num_local_first_bad_edges": int(local_first_bad_total),
        "num_local_safe_bad_pairs": int(local_safe_bad_total),
        "rejection_curve": [point.to_dict() for point in rejection_points],
        "pressure_curve": [point.to_dict() for point in pressure_points],
    }
    return SameFamilyTrustResult(metrics=metrics, prompt_rows=prompt_rows)


def render_samefamily_summary_markdown(
    *,
    run_name: str,
    value_run_dir: Path,
    eval_pairs_jsonl: Path,
    metrics: dict[str, Any],
) -> str:
    """Render a small human-readable markdown summary.

    中文
    ----
    终端和 JSON 更适合程序读取；summary.md 的职责是让人快速看清：
    1. prompt-level rerank 好不好，
    2. rejection 有没有收益，
    3. 强选择压力下会不会迅速崩掉。
    """
    lines = [
        "# Phase E Same-Family Trust Summary",
        "",
        f"- run_name: `{run_name}`",
        f"- value_run_dir: `{value_run_dir}`",
        f"- eval_pairs_jsonl: `{eval_pairs_jsonl}`",
        f"- num_prompt_pools: `{metrics['num_prompt_pools']}`",
        f"- num_candidate_nodes: `{metrics['num_candidate_nodes']}`",
        f"- prompt_pool_top1_accuracy: `{metrics['prompt_pool_top1_accuracy']:.6f}`",
        f"- prompt_pool_mean_regret: `{metrics['prompt_pool_mean_regret']:.6f}`",
        f"- prompt_pool_mean_score_gap: `{metrics['prompt_pool_mean_score_gap']:.6f}`",
    ]
    if metrics.get("local_last_safe_top1_accuracy") is not None:
        lines.extend(
            [
                f"- local_last_safe_top1_accuracy: `{float(metrics['local_last_safe_top1_accuracy']):.6f}`",
                f"- local_first_bad_edge_accuracy: `{float(metrics['local_first_bad_edge_accuracy']):.6f}`",
                f"- local_safe_vs_bad_pair_accuracy: `{float(metrics['local_safe_vs_bad_pair_accuracy']):.6f}`",
            ]
        )
    lines.extend(["", "## Rejection Curve", "", "| target_cov | actual_cov | gap_thres | accepted | top1_acc | mean_regret |", "|---:|---:|---:|---:|---:|---:|"])
    for point in metrics.get("rejection_curve", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(point['target_coverage']):.2f}",
                    f"{float(point['actual_coverage']):.4f}",
                    f"{float(point['min_gap_threshold']):.6f}",
                    f"{int(point['accepted_prompts'])}",
                    f"{float(point['top1_accuracy']):.4f}",
                    f"{float(point['mean_regret']):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Pressure Curve", "", "| subset_size | num_prompt_subsets | top1_acc | mean_regret |", "|---:|---:|---:|---:|"])
    for point in metrics.get("pressure_curve", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{int(point['subset_size'])}",
                    f"{int(point['num_prompt_subsets'])}",
                    f"{float(point['top1_accuracy']):.4f}",
                    f"{float(point['mean_regret']):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_prompt_rows_jsonl(path: Path, rows: list[PromptTrustRow]) -> None:
    """Persist prompt-level trust rows as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def _compute_candidate_utilities(pool: PromptPool, *, edge_weight_mode: str) -> dict[str, float]:
    wins = {cid: 0.0 for cid in pool.candidate_ids}
    totals = {cid: 0.0 for cid in pool.candidate_ids}
    for edge in pool.edges:
        weight = float(edge.confidence) if edge_weight_mode == "confidence" else 1.0
        wins[edge.chosen_id] = wins.get(edge.chosen_id, 0.0) + weight
        totals[edge.chosen_id] = totals.get(edge.chosen_id, 0.0) + weight
        totals[edge.rejected_id] = totals.get(edge.rejected_id, 0.0) + weight
    return {
        cid: (float(wins[cid] / totals[cid]) if totals[cid] > 0.0 else 0.5)
        for cid in pool.candidate_ids
    }


def _compute_rejection_curve(
    *,
    prompt_rows: list[PromptTrustRow],
    target_coverages: tuple[float, ...],
) -> list[RejectionPoint]:
    if not prompt_rows:
        return []
    scored_rows = [row for row in prompt_rows if row.score_gap is not None]
    if not scored_rows:
        return []
    ranked = sorted(scored_rows, key=lambda row: (float(row.score_gap or 0.0), row.prompt_key), reverse=True)
    total = len(ranked)
    points: list[RejectionPoint] = []
    for target_cov in target_coverages:
        keep = max(1, int(round(float(target_cov) * total)))
        accepted = ranked[:keep]
        threshold = float(accepted[-1].score_gap or 0.0)
        points.append(
            RejectionPoint(
                target_coverage=float(target_cov),
                actual_coverage=float(len(accepted) / total),
                min_gap_threshold=float(threshold),
                accepted_prompts=int(len(accepted)),
                top1_accuracy=_safe_mean([1.0 if row.model_top_hit else 0.0 for row in accepted]),
                mean_regret=_safe_mean([float(row.regret) for row in accepted]),
            )
        )
    return points


def _compute_pressure_curve(
    *,
    pool_payloads: list[dict[str, Any]],
    candidate_scores: dict[str, float],
    subset_sizes: tuple[int, ...],
    repeats: int,
) -> list[PressurePoint]:
    points: list[PressurePoint] = []
    for subset_size in subset_sizes:
        hits: list[float] = []
        regrets: list[float] = []
        count = 0
        for payload in pool_payloads:
            pool: PromptPool = payload["pool"]
            utilities: dict[str, float] = payload["utilities"]
            if len(pool.candidate_ids) < int(subset_size):
                continue
            for repeat_idx in range(int(repeats)):
                subset_ids = _deterministic_subset(pool.candidate_ids, subset_size=int(subset_size), repeat_idx=repeat_idx)
                if len(subset_ids) < int(subset_size):
                    continue
                ranked_subset = sorted(
                    subset_ids,
                    key=lambda cid: (float(candidate_scores[cid]), cid),
                    reverse=True,
                )
                top_cid = ranked_subset[0]
                gold_utility = max(float(utilities[cid]) for cid in subset_ids)
                selected_utility = float(utilities[top_cid])
                gold_top_ids = [cid for cid in subset_ids if math.isclose(float(utilities[cid]), gold_utility, abs_tol=1e-9)]
                hits.append(1.0 if top_cid in gold_top_ids else 0.0)
                regrets.append(float(gold_utility - selected_utility))
                count += 1
        if count == 0:
            continue
        points.append(
            PressurePoint(
                subset_size=int(subset_size),
                num_prompt_subsets=int(count),
                top1_accuracy=_safe_mean(hits),
                mean_regret=_safe_mean(regrets),
            )
        )
    return points


def _resolve_local_first_bad(pool: PromptPool, *, node_map: dict[str, CandidateNode]) -> tuple[int | None, list[str]]:
    bad_indices = [node_map[cid].min_bad_step_index for cid in pool.candidate_ids if node_map[cid].min_bad_step_index is not None]
    if not bad_indices:
        return None, []
    first_bad_index = int(min(bad_indices))
    safe_candidates = [
        cid
        for cid in pool.candidate_ids
        if node_map[cid].max_safe_step_index is not None and int(node_map[cid].max_safe_step_index) < first_bad_index
    ]
    if not safe_candidates:
        return first_bad_index, []
    max_safe_idx = max(int(node_map[cid].max_safe_step_index or -1) for cid in safe_candidates)
    last_safe_ids = sorted(
        [cid for cid in safe_candidates if int(node_map[cid].max_safe_step_index or -1) == max_safe_idx]
    )
    return first_bad_index, last_safe_ids


def _first_bad_candidate_ids(pool: PromptPool, *, node_map: dict[str, CandidateNode], first_bad_index: int) -> list[str]:
    return sorted(
        [
            cid
            for cid in pool.candidate_ids
            if node_map[cid].min_bad_step_index is not None and int(node_map[cid].min_bad_step_index) == int(first_bad_index)
        ]
    )


def _all_safe_candidate_ids(pool: PromptPool, *, node_map: dict[str, CandidateNode], first_bad_index: int) -> list[str]:
    return sorted(
        [
            cid
            for cid in pool.candidate_ids
            if node_map[cid].max_safe_step_index is not None and int(node_map[cid].max_safe_step_index) < int(first_bad_index)
        ]
    )


def _all_bad_candidate_ids(pool: PromptPool, *, node_map: dict[str, CandidateNode], first_bad_index: int) -> list[str]:
    return sorted(
        [
            cid
            for cid in pool.candidate_ids
            if node_map[cid].min_bad_step_index is not None and int(node_map[cid].min_bad_step_index) >= int(first_bad_index)
        ]
    )


def _deterministic_subset(candidate_ids: list[str], *, subset_size: int, repeat_idx: int) -> list[str]:
    values = list(range(len(candidate_ids)))
    perm = sorted(
        values,
        key=lambda idx: hashlib.sha256(f"{repeat_idx}:{candidate_ids[idx]}".encode("utf-8")).hexdigest(),
    )
    return [candidate_ids[idx] for idx in perm[: int(subset_size)]]


def _safe_mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _stable_text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _pool_key(*, source_tag: str, prompt_text: str) -> str:
    return _stable_text_id(f"{source_tag}\n<SEP>\n{prompt_text}")


def _candidate_id(*, source_tag: str, prompt_text: str, candidate_text: str) -> str:
    return hashlib.sha256(
        f"{source_tag}\n<SEP>\n{prompt_text}\n<SEP>\n{candidate_text}".encode("utf-8")
    ).hexdigest()[:20]


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return None
