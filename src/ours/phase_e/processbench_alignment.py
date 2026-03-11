"""诊断训练 pair 几何与 ProcessBench 评测几何是否对齐。 Diagnose whether training-pair geometry is aligned with ProcessBench evaluation geometry.

这个模块存在的原因不是为了再算一遍主指标，而是为了回答一个更具体的问题：
1. 训练时模型到底看到了哪些 `good/bad prefix` 关系？
2. `ProcessBench` 实际又在评测哪些关系？
3. 当迁移失败时，失败主要集中在哪些 slice？

The goal of this module is not to recompute the main benchmark score again.
It exists to answer a more diagnostic question:
1. what `good/bad prefix` relations does training actually expose?
2. what relations does ProcessBench really evaluate?
3. when transfer fails, which slices fail first?
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ours.phase_d.external_pairs import ExternalPairRecord, load_external_pair_jsonl
from ours.phase_e.benchmark_eval import load_processbench_examples


def summarize_pair_jsonl_alignment(pair_jsonl_path: Path) -> dict[str, Any]:
    """汇总一份训练 pair JSONL 的几何结构。 Summarize the geometry of one training pair JSONL.

    中文
    ----
    这里关注的不是 loss 或 AUC，而是训练监督的“结构支持面”：
    1. 有多少 pair 其实是 prefix-extension 关系，
    2. step gap 分布是什么样，
    3. 是否覆盖了 `last-safe vs first-bad` 以外的 pair 类型。

    English
    -------
    This function is about the support of the supervision geometry, not about
    optimization metrics.
    """
    rows, base_summary = load_external_pair_jsonl(pair_jsonl_path)
    prefix_relation_flags: list[int] = []
    length_ratios: list[float] = []
    step_gaps: list[int] = []
    typed_pair_counts: dict[str, int] = {}
    gap_bucket_counts: dict[str, int] = {}
    num_pairs_with_step_metadata = 0

    for row in rows:
        chosen_norm = str(row.chosen_text).strip()
        rejected_norm = str(row.rejected_text).strip()
        prefix_relation_flags.append(int(bool(chosen_norm) and rejected_norm.startswith(chosen_norm)))
        shorter = max(min(len(chosen_norm), len(rejected_norm)), 1)
        longer = max(len(chosen_norm), len(rejected_norm))
        length_ratios.append(float(longer / shorter))

        pair_type = classify_step_label_pair_type(row.metadata)
        if pair_type is not None:
            num_pairs_with_step_metadata += 1
            typed_pair_counts[pair_type] = typed_pair_counts.get(pair_type, 0) + 1
            step_gap = int(row.metadata.get("step_gap", int(row.metadata["negative_step_index"]) - int(row.metadata["positive_step_index"])))
            step_gaps.append(int(step_gap))
            bucket = gap_bucket_name(int(step_gap))
            gap_bucket_counts[bucket] = gap_bucket_counts.get(bucket, 0) + 1

    pair_type_distribution = _normalize_counter(
        typed_pair_counts,
        keys=(
            "lastsafe_vs_firstbad",
            "earlygood_vs_firstbad",
            "lastsafe_vs_laterbad",
            "earlygood_vs_laterbad",
        ),
    )
    gap_bucket_distribution = _normalize_counter(
        gap_bucket_counts,
        keys=("gap1", "gap2", "gap3_4", "gap5p"),
    )
    return {
        "pair_jsonl_path": str(pair_jsonl_path),
        "num_pairs": int(len(rows)),
        "base_summary": base_summary,
        "prefix_relation_rate": _safe_fraction(sum(prefix_relation_flags), len(prefix_relation_flags)),
        "length_ratio": summarize_numeric_series(length_ratios),
        "num_pairs_with_step_metadata": int(num_pairs_with_step_metadata),
        "step_gap": summarize_numeric_series(step_gaps),
        "pair_type_counts": dict(sorted(typed_pair_counts.items())),
        "pair_type_distribution": pair_type_distribution,
        "gap_bucket_counts": dict(sorted(gap_bucket_counts.items())),
        "gap_bucket_distribution": gap_bucket_distribution,
    }


def summarize_processbench_topology(processbench_path: Path) -> dict[str, Any]:
    """汇总 ProcessBench 原始样本真正评测的 pair 结构。 Summarize the actual pair topology evaluated by ProcessBench.

    中文
    ----
    这里故意直接按 benchmark 合同去展开，而不是按训练 pair 的合同去猜。
    因为我们想知道的是：
    “benchmark 自己到底要求模型区分什么关系？”
    """
    examples = load_processbench_examples(processbench_path)
    step_counts = [int(len(example.steps)) for example in examples]
    first_bad_indices: list[int] = []
    gap_bucket_counts: dict[str, int] = {}
    pair_type_counts: dict[str, int] = {}
    total_pairs = 0
    for example in examples:
        label = int(example.label)
        if label < 0:
            continue
        first_bad_indices.append(int(label))
        for good_idx in range(int(label)):
            for bad_idx in range(int(label), len(example.steps)):
                total_pairs += 1
                pair_type = classify_processbench_pair_type(
                    good_idx=int(good_idx),
                    bad_idx=int(bad_idx),
                    first_bad_idx=int(label),
                )
                pair_type_counts[pair_type] = pair_type_counts.get(pair_type, 0) + 1
                bucket = gap_bucket_name(int(bad_idx) - int(good_idx))
                gap_bucket_counts[bucket] = gap_bucket_counts.get(bucket, 0) + 1
    return {
        "processbench_path": str(processbench_path),
        "num_examples": int(len(examples)),
        "num_error_examples": int(sum(1 for example in examples if int(example.label) >= 0)),
        "num_all_correct_examples": int(sum(1 for example in examples if int(example.label) < 0)),
        "step_count": summarize_numeric_series(step_counts),
        "first_bad_index": summarize_numeric_series(first_bad_indices),
        "num_good_bad_pairs": int(total_pairs),
        "pair_type_counts": dict(sorted(pair_type_counts.items())),
        "pair_type_distribution": _normalize_counter(
            pair_type_counts,
            keys=(
                "lastsafe_vs_firstbad",
                "earlygood_vs_firstbad",
                "lastsafe_vs_laterbad",
                "earlygood_vs_laterbad",
            ),
        ),
        "gap_bucket_counts": dict(sorted(gap_bucket_counts.items())),
        "gap_bucket_distribution": _normalize_counter(
            gap_bucket_counts,
            keys=("gap1", "gap2", "gap3_4", "gap5p"),
        ),
    }


def summarize_scored_rows_alignment(
    *,
    scored_rows_path: Path,
    processbench_path: Path,
) -> dict[str, Any]:
    """按几何 slice 统计一份 ProcessBench `scored_rows.jsonl` 的表现。 Slice one ProcessBench scored-rows artifact by geometry.

    English
    -------
    The main benchmark summary collapses everything into one AUC / pair-acc
    number.  This helper answers the more useful debugging question:
    which geometric relations does the model rank correctly, and which ones
    still break?
    """
    if not scored_rows_path.exists():
        raise FileNotFoundError(f"scored_rows.jsonl not found: {scored_rows_path}")
    scored_rows = [json.loads(line) for line in scored_rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not scored_rows:
        raise RuntimeError(f"No scored rows found in {scored_rows_path}")

    example_ids = {str(row["example_id"]) for row in scored_rows}
    examples = [
        example
        for example in load_processbench_examples(processbench_path)
        if str(example.example_id) in example_ids
    ]
    if not examples:
        raise RuntimeError("No overlapping ProcessBench examples found for scored rows")

    grouped_scores: dict[str, dict[int, float]] = {}
    for row in scored_rows:
        example_id = str(row["example_id"])
        grouped_scores.setdefault(example_id, {})[int(row["prefix_step_index"])] = float(row["score"])

    gap_bucket_outcomes: dict[str, list[int]] = {}
    pair_type_outcomes: dict[str, list[int]] = {}
    first_bad_edge_outcomes: list[int] = []
    total_good_bad_pairs = 0
    total_positive_pairs = 0

    for example in examples:
        label = int(example.label)
        score_map = grouped_scores.get(str(example.example_id), {})
        if label < 0:
            continue
        for step_idx in range(len(example.steps)):
            if int(step_idx) not in score_map:
                raise KeyError(
                    f"Missing scored prefix for example={example.example_id!r}, step={step_idx}"
                )
        if 0 < label < len(example.steps):
            first_bad_edge_outcomes.append(int(score_map[int(label) - 1] > score_map[int(label)]))
        for good_idx in range(int(label)):
            for bad_idx in range(int(label), len(example.steps)):
                total_good_bad_pairs += 1
                positive = int(score_map[int(good_idx)] > score_map[int(bad_idx)])
                total_positive_pairs += positive
                gap_bucket_outcomes.setdefault(gap_bucket_name(int(bad_idx) - int(good_idx)), []).append(positive)
                pair_type_outcomes.setdefault(
                    classify_processbench_pair_type(
                        good_idx=int(good_idx),
                        bad_idx=int(bad_idx),
                        first_bad_idx=int(label),
                    ),
                    [],
                ).append(positive)

    pair_type_metrics = {
        key: build_binary_outcome_summary(values)
        for key, values in sorted(pair_type_outcomes.items())
    }
    gap_bucket_metrics = {
        key: build_binary_outcome_summary(values)
        for key, values in sorted(gap_bucket_outcomes.items())
    }
    anygood_firstbad_values = (
        pair_type_outcomes.get("lastsafe_vs_firstbad", [])
        + pair_type_outcomes.get("earlygood_vs_firstbad", [])
    )
    good_laterbad_values = (
        pair_type_outcomes.get("lastsafe_vs_laterbad", [])
        + pair_type_outcomes.get("earlygood_vs_laterbad", [])
    )
    return {
        "scored_rows_path": str(scored_rows_path),
        "num_examples": int(len(examples)),
        "num_scored_rows": int(len(scored_rows)),
        "pair_accuracy_good_vs_bad": _safe_fraction(total_positive_pairs, total_good_bad_pairs),
        "first_error_edge_accuracy": _safe_fraction(sum(first_bad_edge_outcomes), len(first_bad_edge_outcomes)),
        "gap_bucket_metrics": gap_bucket_metrics,
        "pair_type_metrics": pair_type_metrics,
        "aggregate_metrics": {
            "anygood_vs_firstbad": build_binary_outcome_summary(anygood_firstbad_values),
            "good_vs_laterbad": build_binary_outcome_summary(good_laterbad_values),
        },
    }


def classify_step_label_pair_type(metadata: dict[str, Any]) -> str | None:
    """把 step-label 训练 pair 归到与 ProcessBench slice 对齐的类型。 Map one step-label training pair to a ProcessBench-aligned slice type."""
    if "positive_step_index" not in metadata or "negative_step_index" not in metadata or "first_negative_index" not in metadata:
        return None
    chosen_idx = int(metadata["positive_step_index"])
    rejected_idx = int(metadata["negative_step_index"])
    first_negative_idx = int(metadata["first_negative_index"])
    return classify_processbench_pair_type(
        good_idx=int(chosen_idx),
        bad_idx=int(rejected_idx),
        first_bad_idx=int(first_negative_idx),
    )


def classify_processbench_pair_type(
    *,
    good_idx: int,
    bad_idx: int,
    first_bad_idx: int,
) -> str:
    """按 `first bad` 与 `later bad` 的局部/远距关系给 pair 分型。 Classify one good-vs-bad pair by its local-vs-later structure."""
    if int(bad_idx) == int(first_bad_idx):
        if int(good_idx) == int(first_bad_idx) - 1:
            return "lastsafe_vs_firstbad"
        return "earlygood_vs_firstbad"
    if int(good_idx) == int(first_bad_idx) - 1:
        return "lastsafe_vs_laterbad"
    return "earlygood_vs_laterbad"


def gap_bucket_name(step_gap: int) -> str:
    """把 step gap 离散成稳定 bucket，方便横向比较。 Bucketize one step gap into a stable coarse slice."""
    gap = int(step_gap)
    if gap <= 1:
        return "gap1"
    if gap == 2:
        return "gap2"
    if 3 <= gap <= 4:
        return "gap3_4"
    return "gap5p"


def build_binary_outcome_summary(values: list[int]) -> dict[str, Any]:
    """把 0/1 结果列表压成计数与准确率摘要。 Summarize one list of binary outcomes into counts plus accuracy."""
    return {
        "count": int(len(values)),
        "positive": int(sum(int(value) for value in values)),
        "accuracy": _safe_fraction(sum(int(value) for value in values), len(values)),
    }


def compute_alignment_distances(
    *,
    pair_summary: dict[str, Any],
    benchmark_summary: dict[str, Any],
) -> dict[str, Any]:
    """计算训练监督分布与 benchmark 分布之间的粗粒度距离。 Compute coarse-grained distances between training and benchmark distributions."""
    pair_type_distance = l1_distance_between_distributions(
        pair_summary.get("pair_type_distribution", {}),
        benchmark_summary.get("pair_type_distribution", {}),
        keys=(
            "lastsafe_vs_firstbad",
            "earlygood_vs_firstbad",
            "lastsafe_vs_laterbad",
            "earlygood_vs_laterbad",
        ),
    )
    gap_distance = l1_distance_between_distributions(
        pair_summary.get("gap_bucket_distribution", {}),
        benchmark_summary.get("gap_bucket_distribution", {}),
        keys=("gap1", "gap2", "gap3_4", "gap5p"),
    )
    return {
        "pair_type_l1_distance": float(pair_type_distance),
        "gap_bucket_l1_distance": float(gap_distance),
    }


def l1_distance_between_distributions(
    left: dict[str, float],
    right: dict[str, float],
    *,
    keys: tuple[str, ...],
) -> float:
    """计算两个稀疏分布在固定 key 集上的 L1 距离。 Compute the L1 distance between two sparse distributions on a fixed key set."""
    return float(sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys))


def summarize_numeric_series(values: list[int] | list[float]) -> dict[str, Any]:
    """把一列数字压成少量分位点摘要。 Compress one numeric series into a few quantile-style statistics."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": int(len(ordered)),
        "mean": float(sum(ordered) / len(ordered)),
        "min": float(ordered[0]),
        "p50": _pick_quantile(ordered, 0.50),
        "p90": _pick_quantile(ordered, 0.90),
        "p95": _pick_quantile(ordered, 0.95),
        "max": float(ordered[-1]),
    }


def render_processbench_alignment_markdown(
    *,
    title: str,
    pair_summary: dict[str, Any],
    benchmark_summary: dict[str, Any],
    alignment_distance: dict[str, Any],
    scored_run_summaries: dict[str, dict[str, Any]],
) -> str:
    """渲染一份便于人读的 alignment 审计摘要。 Render one human-readable alignment audit summary."""
    lines = [
        f"# {title}",
        "",
        "## Training Pair Geometry",
        "",
        f"- pair_jsonl_path: `{pair_summary['pair_jsonl_path']}`",
        f"- num_pairs: `{pair_summary['num_pairs']}`",
        f"- prefix_relation_rate: `{float(pair_summary['prefix_relation_rate']):.4f}`",
        f"- num_pairs_with_step_metadata: `{pair_summary['num_pairs_with_step_metadata']}`",
        f"- pair_type_distribution: `{json.dumps(pair_summary['pair_type_distribution'], ensure_ascii=False, sort_keys=True)}`",
        f"- gap_bucket_distribution: `{json.dumps(pair_summary['gap_bucket_distribution'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## ProcessBench Topology",
        "",
        f"- processbench_path: `{benchmark_summary['processbench_path']}`",
        f"- num_examples: `{benchmark_summary['num_examples']}`",
        f"- num_error_examples: `{benchmark_summary['num_error_examples']}`",
        f"- num_good_bad_pairs: `{benchmark_summary['num_good_bad_pairs']}`",
        f"- pair_type_distribution: `{json.dumps(benchmark_summary['pair_type_distribution'], ensure_ascii=False, sort_keys=True)}`",
        f"- gap_bucket_distribution: `{json.dumps(benchmark_summary['gap_bucket_distribution'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Alignment Distance",
        "",
        f"- pair_type_l1_distance: `{float(alignment_distance['pair_type_l1_distance']):.4f}`",
        f"- gap_bucket_l1_distance: `{float(alignment_distance['gap_bucket_l1_distance']):.4f}`",
        "",
    ]
    if scored_run_summaries:
        lines.extend(
            [
                "## Scored Runs",
                "",
                "| run | pair_acc | first_edge | gap1 | gap2 | gap3_4 | gap5p | anygood_vs_firstbad | good_vs_laterbad |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for run_name, payload in sorted(scored_run_summaries.items()):
            gap_metrics = payload.get("gap_bucket_metrics", {})
            aggregate = payload.get("aggregate_metrics", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        run_name,
                        f"{float(payload.get('pair_accuracy_good_vs_bad', 0.0)):.4f}",
                        f"{float(payload.get('first_error_edge_accuracy', 0.0)):.4f}",
                        _format_metric_accuracy(gap_metrics.get("gap1")),
                        _format_metric_accuracy(gap_metrics.get("gap2")),
                        _format_metric_accuracy(gap_metrics.get("gap3_4")),
                        _format_metric_accuracy(gap_metrics.get("gap5p")),
                        _format_metric_accuracy(aggregate.get("anygood_vs_firstbad")),
                        _format_metric_accuracy(aggregate.get("good_vs_laterbad")),
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def _format_metric_accuracy(payload: dict[str, Any] | None) -> str:
    """把切片指标格式化成 markdown 表格里的单格文本。 Format one sliced metric for markdown tables."""
    if not payload:
        return "N/A"
    return f"{float(payload.get('accuracy', 0.0)):.4f}"


def _normalize_counter(counter: dict[str, int], *, keys: tuple[str, ...]) -> dict[str, float]:
    """把计数字典归一化成稳定的概率分布。 Normalize one counter into a stable probability distribution."""
    total = sum(int(counter.get(key, 0)) for key in keys)
    return {
        key: _safe_fraction(int(counter.get(key, 0)), int(total))
        for key in keys
    }


def _pick_quantile(ordered: list[float], quantile: float) -> float:
    """从已排序序列中取一个近似分位点。 Pick one approximate quantile from a sorted list."""
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * float(quantile))))
    return float(ordered[int(index)])


def _safe_fraction(numerator: int | float, denominator: int | float) -> float:
    """安全地计算分数，避免空集合时抛异常。 Compute one fraction safely and return 0.0 on empty denominator."""
    if float(denominator) <= 0.0:
        return 0.0
    return float(float(numerator) / float(denominator))
