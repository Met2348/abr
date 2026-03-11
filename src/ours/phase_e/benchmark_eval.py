"""Benchmark-native evaluation helpers for Phase E.

English
-------
Phase E is trying to answer a benchmark-family question before a transfer
question:

1. can the value head learn meaningful process ranking at all?
2. can it distinguish good and bad prefixes on benchmark-native examples?

To answer that, we need benchmark-specific conversion logic from raw benchmark
rows into the exact text inputs the value head scores.

中文
----
Phase E 先问的是 benchmark 家族内部的问题，而不是迁移问题：

1. 这个 value head 到底能不能学到有意义的过程排序？
2. 它能不能在 benchmark-native 的例子上区分好 prefix 和坏 prefix？

因此这里需要专门的 benchmark 适配逻辑，把原始 benchmark 样本转成
value head 真正能打分的文本输入。
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ours.phase_b.faithfulness_eval import compute_binary_auc


@dataclass(slots=True)
class ProcessBenchExample:
    """One ProcessBench example with step-level error position label.

    中文
    ----
    这里的 `label` 不是“整题对不对”，而是“第一处错误出现在第几步”。
    这决定了后面必须按 prefix 展开样本，而不能只把整条过程当作一个样本。
    """

    example_id: str
    generator: str
    problem: str
    steps: list[str]
    label: int
    final_answer_correct: bool


@dataclass(slots=True)
class ProcessBenchPrefixRecord:
    """One scoreable ProcessBench prefix.

    English
    -------
    One ProcessBench row expands into multiple prefix records because the value
    head scores prefixes, not only full reasoning traces.

    中文
    ----
    一条 ProcessBench 原始样本会展开成多个 prefix record，因为 value head
    的打分对象是 prefix，而不是只看完整过程。
    """

    row_id: str
    example_id: str
    prompt_text: str
    prefix_text: str
    prefix_step_index: int
    is_good_prefix: bool
    is_first_bad_prefix: bool
    label: int

    def input_text(self) -> str:
        """Return the full text presented to the frozen backbone."""
        return f"{self.prompt_text}{self.prefix_text}"


@dataclass(slots=True)
class PRMBenchPreviewPairRecord:
    """One converted PRMBench preview prefix pair.

    中文
    ----
    与 ProcessBench 不同，PRMBench_Preview 这条路径本身已经是 pair 粒度，
    所以后面的评测会更直接一些。
    """

    pair_id: str
    classification: str
    question_text: str
    chosen_prefix_text: str
    rejected_prefix_text: str
    error_step_index: int

    def chosen_input_text(self) -> str:
        """Return the scoreable chosen full text."""
        return f"{self.question_text}{self.chosen_prefix_text}"

    def rejected_input_text(self) -> str:
        """Return the scoreable rejected full text."""
        return f"{self.question_text}{self.rejected_prefix_text}"


def load_processbench_examples(path: Path, *, max_samples: int | None = None) -> list[ProcessBenchExample]:
    """Load ProcessBench JSON examples.

    中文
    ----
    这里对无关字段相对宽容，但对评测必需字段会更严格：
    1. `steps` 必须存在且非空；
    2. 每条保留的记录都必须能展开成 prefix；
    3. 文件最后至少要得到一条有效样本。
    """
    if not path.exists():
        raise FileNotFoundError(f"ProcessBench file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"ProcessBench file must contain a list: {path}")
    rows: list[ProcessBenchExample] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        steps = item.get("steps")
        if not isinstance(steps, list) or not steps:
            continue
        rows.append(
            ProcessBenchExample(
                example_id=str(item.get("id", "")).strip() or f"processbench_{len(rows)}",
                generator=str(item.get("generator", "")).strip(),
                problem=str(item.get("problem", "")),
                steps=[str(step) for step in steps],
                label=int(item.get("label", -1)),
                final_answer_correct=bool(item.get("final_answer_correct", False)),
            )
        )
    if not rows:
        raise RuntimeError(f"No valid ProcessBench rows loaded from {path}")
    if max_samples is not None and len(rows) > int(max_samples):
        rows = _subsample_processbench_examples(
            rows=rows,
            max_samples=int(max_samples),
        )
    return rows


def _subsample_processbench_examples(
    *,
    rows: list[ProcessBenchExample],
    max_samples: int,
) -> list[ProcessBenchExample]:
    """Deterministically subsample ProcessBench while preserving coarse label mix.

    English
    -------
    Smoke runs are often forced to use `max_samples`, but taking the first `N`
    rows can silently destroy the benchmark geometry if the public file is
    block-ordered by error type.  We therefore keep two coarse strata:
    1. all-correct examples (`label < 0`)
    2. error examples (`label >= 0`)

    Then we allocate the subsample proportionally and spread picks across each
    stratum's original order.

    中文
    ----
    smoke 评测经常会开 `max_samples`，但如果直接取前 N 条，而公开文件又刚好按错误类型
    分块排列，就会把 benchmark 几何静默采坏。

    所以这里保留两个粗粒度 strata：
    1. all-correct (`label < 0`)
    2. error (`label >= 0`)

    然后按原始占比分配样本数，并在每个 stratum 内按原顺序均匀抽点。
    """
    if int(max_samples) <= 0:
        raise ValueError("`max_samples` must be > 0")
    if len(rows) <= int(max_samples):
        return list(rows)
    all_correct = [row for row in rows if int(row.label) < 0]
    errors = [row for row in rows if int(row.label) >= 0]
    if not all_correct or not errors:
        return _spread_select(rows, max_samples=int(max_samples))

    total = len(rows)
    target_all_correct = round(int(max_samples) * len(all_correct) / total)
    target_all_correct = min(len(all_correct), max(1, int(target_all_correct)))
    target_errors = int(max_samples) - int(target_all_correct)
    target_errors = min(len(errors), max(1, int(target_errors)))

    # If one bucket hits its cardinality cap, donate the remainder to the other.
    # 如果一个 bucket 的可用样本数不够，就把剩余额度让给另一个 bucket。
    assigned = int(target_all_correct + target_errors)
    if assigned < int(max_samples):
        remaining = int(max_samples) - assigned
        if len(errors) - int(target_errors) >= len(all_correct) - int(target_all_correct):
            target_errors = min(len(errors), int(target_errors) + remaining)
        else:
            target_all_correct = min(len(all_correct), int(target_all_correct) + remaining)

    selected_ids = {
        str(row.example_id)
        for row in (
            _spread_select(all_correct, max_samples=int(target_all_correct))
            + _spread_select(errors, max_samples=int(target_errors))
        )
    }
    selected = [row for row in rows if str(row.example_id) in selected_ids]
    return selected[: int(max_samples)]


def _spread_select(rows: list[ProcessBenchExample], *, max_samples: int) -> list[ProcessBenchExample]:
    """Select evenly spaced rows from one ordered bucket.

    中文
    ----
    这里不用随机采样，是为了让 smoke 结果在相同输入下完全可复现。
    """
    if len(rows) <= int(max_samples):
        return list(rows)
    if int(max_samples) <= 1:
        return [rows[0]]
    indices: list[int] = []
    for slot in range(int(max_samples)):
        raw_index = round(slot * (len(rows) - 1) / max(int(max_samples) - 1, 1))
        normalized = min(max(int(raw_index), 0), len(rows) - 1)
        if normalized not in indices:
            indices.append(normalized)
    if len(indices) < int(max_samples):
        for idx in range(len(rows)):
            if idx in indices:
                continue
            indices.append(idx)
            if len(indices) >= int(max_samples):
                break
    return [rows[idx] for idx in sorted(indices[: int(max_samples)])]


def build_processbench_prefix_records(
    examples: list[ProcessBenchExample],
) -> list[ProcessBenchPrefixRecord]:
    """Convert ProcessBench examples into scoreable prefix rows.

    Label semantics
    ---------------
    Following the public dataset contract, `label=-1` means all steps remain
    correct. Otherwise `label=k` means the first erroneous step is index `k`.

    Therefore:
    - prefixes with `step_index < k` are treated as good,
    - prefixes with `step_index >= k` are treated as bad.
    中文补充
    --------
    这个展开步骤之所以必要，是因为我们真正想测的是：
    1. 好 prefix 是否高于坏 prefix，
    2. 在第一次出错的局部边界上，分数是否及时下降。
    """
    rows: list[ProcessBenchPrefixRecord] = []
    for example in examples:
        prompt_text = f"{example.problem}\n\n"
        for step_idx in range(len(example.steps)):
            prefix_text = _join_steps_as_prefix(example.steps, step_idx)
            is_bad = example.label >= 0 and step_idx >= int(example.label)
            rows.append(
                ProcessBenchPrefixRecord(
                    row_id=f"{example.example_id}:prefix:{step_idx}",
                    example_id=example.example_id,
                    prompt_text=prompt_text,
                    prefix_text=prefix_text,
                    prefix_step_index=int(step_idx),
                    is_good_prefix=not is_bad,
                    is_first_bad_prefix=(example.label >= 0 and step_idx == int(example.label)),
                    label=int(example.label),
                )
            )
    return rows


def compute_processbench_metrics(
    rows: list[ProcessBenchPrefixRecord],
    scores: list[float],
    *,
    processbench_f1_threshold: float | None = None,
    processbench_f1_threshold_candidates: int = 50,
) -> dict[str, Any]:
    """Compute ranking-oriented metrics for ProcessBench prefix scoring.

    English
    -------
    This function reports three complementary views:
    1. good-vs-bad pair accuracy within each example,
    2. global AUC over all good and bad prefixes,
    3. first-error-edge accuracy for local failure localization.

    中文
    ----
    这里会同时给出三种互补视角：
    1. 题内 good-vs-bad pair accuracy，
    2. 全局 good/bad prefix AUC，
    3. first-error-edge accuracy，用来衡量局部错误定位。
    """
    if len(rows) != len(scores):
        raise ValueError("ProcessBench rows and scores must have equal length")
    grouped: dict[str, list[tuple[ProcessBenchPrefixRecord, float]]] = {}
    for row, score in zip(rows, scores, strict=True):
        grouped.setdefault(row.example_id, []).append((row, float(score)))

    global_labels: list[int] = []
    global_scores: list[float] = []
    pair_total = 0
    pair_positive = 0
    first_edge_total = 0
    first_edge_positive = 0
    all_correct_last_scores: list[float] = []
    good_scores: list[float] = []
    bad_scores: list[float] = []

    for example_rows in grouped.values():
        example_rows.sort(key=lambda item: item[0].prefix_step_index)
        label = int(example_rows[0][0].label)
        if label < 0:
            all_correct_last_scores.append(float(example_rows[-1][1]))
        good = [float(score) for row, score in example_rows if row.is_good_prefix]
        bad = [float(score) for row, score in example_rows if not row.is_good_prefix]
        good_scores.extend(good)
        bad_scores.extend(bad)
        for score in good:
            global_scores.append(float(score))
            global_labels.append(1)
        for score in bad:
            global_scores.append(float(score))
            global_labels.append(0)
        # Compare only within the same example so the metric focuses on process
        # quality rather than cross-question difficulty differences.
        # 只在同一道题内部做 good/bad 比较，避免题目难度差异污染过程质量指标。
        for score_good in good:
            for score_bad in bad:
                pair_total += 1
                if float(score_good) > float(score_bad):
                    pair_positive += 1
        # The first-error edge isolates the local transition where reasoning first becomes wrong.
        # first-error edge 只看局部转折点：推理第一次出错时分数是否立刻下降。
        if label > 0 and label < len(example_rows):
            prev_score = float(example_rows[label - 1][1])
            bad_score = float(example_rows[label][1])
            first_edge_total += 1
            if prev_score > bad_score:
                first_edge_positive += 1

    auc = compute_binary_auc(global_scores, global_labels) if len(set(global_labels)) == 2 else 0.5
    f1_metrics = compute_processbench_f1(
        rows=rows,
        scores=scores,
        threshold=processbench_f1_threshold,
        threshold_candidates=int(processbench_f1_threshold_candidates),
    )
    return {
        "num_examples": int(len(grouped)),
        "num_prefixes": int(len(rows)),
        "num_error_examples": int(sum(1 for example_rows in grouped.values() if int(example_rows[0][0].label) >= 0)),
        "num_all_correct_examples": int(sum(1 for example_rows in grouped.values() if int(example_rows[0][0].label) < 0)),
        "pair_accuracy_good_vs_bad": float(pair_positive / pair_total) if pair_total > 0 else 0.5,
        "pair_auc_good_vs_bad": float(auc),
        "num_good_bad_pairs": int(pair_total),
        "first_error_edge_accuracy": float(first_edge_positive / first_edge_total) if first_edge_total > 0 else 0.5,
        "num_first_error_edges": int(first_edge_total),
        "mean_good_prefix_score": float(statistics.mean(good_scores)) if good_scores else 0.0,
        "mean_bad_prefix_score": float(statistics.mean(bad_scores)) if bad_scores else 0.0,
        "mean_all_correct_last_score": float(statistics.mean(all_correct_last_scores)) if all_correct_last_scores else 0.0,
        **f1_metrics,
    }


def compute_processbench_f1(
    rows: list[ProcessBenchPrefixRecord],
    scores: list[float],
    *,
    threshold: float | None = None,
    threshold_candidates: int = 50,
) -> dict[str, Any]:
    """Compute the official ProcessBench F1 metric over prefix scores.

    English
    -------
    ProcessBench F1 is the harmonic mean of two per-class accuracies:
    1. Acc_erroneous: fraction of error examples where the model correctly
       identifies the first wrong step.
    2. Acc_correct: fraction of all-correct examples where the model correctly
       predicts "all steps are correct".

    F1 = 2 * Acc_erroneous * Acc_correct / (Acc_erroneous + Acc_correct)

    Decision rule (same as ProcessBench paper):
    - Apply a scalar threshold τ to all step scores.
    - For each example, find the *first* step with score < τ.
    - If such a step exists, predict that step as the first error.
    - If no step falls below τ, predict "all correct" (label = -1).

    When `threshold` is None, this function sweeps `threshold_candidates`
    evenly-spaced values over [0, 1] and picks the one that maximises F1.
    The chosen threshold is returned in the output dict so callers can
    inspect it or fix it for cross-dataset comparisons.

    中文
    ----
    ProcessBench 官方 F1 = 2 × Acc_error × Acc_correct / (Acc_error + Acc_correct)

    其中：
    - Acc_error = 错误样本中"正确预测第一错误步"的比例
    - Acc_correct = 全正确样本中"正确预测为全对"的比例

    决策规则：扫阈值 τ → 找第一个分数 < τ 的步骤 → 若无则预测"全对"。

    当 `threshold=None` 时，自动在 [0,1] 上搜索使 F1 最大的阈值并返回。
    """
    if len(rows) != len(scores):
        raise ValueError("rows and scores must have equal length")

    # Group prefix records by example, sort by step index within each group.
    grouped: dict[str, list[tuple[ProcessBenchPrefixRecord, float]]] = {}
    for row, score in zip(rows, scores, strict=True):
        grouped.setdefault(row.example_id, []).append((row, float(score)))
    for key in grouped:
        grouped[key].sort(key=lambda item: item[0].prefix_step_index)

    def _f1_at_threshold(tau: float) -> tuple[float, float, float]:
        error_total = 0
        error_correct = 0
        correct_total = 0
        correct_correct = 0
        for example_rows in grouped.values():
            gt_label = int(example_rows[0][0].label)
            # Find first step where score < tau.
            predicted_error_step: int | None = None
            for step_idx, (row, score) in enumerate(example_rows):
                if float(score) < float(tau):
                    predicted_error_step = int(row.prefix_step_index)
                    break
            if gt_label < 0:
                # All-correct ground truth.
                correct_total += 1
                if predicted_error_step is None:
                    correct_correct += 1
            else:
                # Error example: ground truth first error at prefix_step_index = gt_label.
                error_total += 1
                if predicted_error_step is not None and predicted_error_step == gt_label:
                    error_correct += 1
        acc_error = float(error_correct / error_total) if error_total > 0 else 0.0
        acc_correct = float(correct_correct / correct_total) if correct_total > 0 else 0.0
        denom = acc_error + acc_correct
        f1 = float(2.0 * acc_error * acc_correct / denom) if denom > 0.0 else 0.0
        return f1, acc_error, acc_correct

    if threshold is not None:
        best_tau = float(threshold)
        best_f1, best_acc_error, best_acc_correct = _f1_at_threshold(best_tau)
        threshold_selection = "fixed"
    else:
        # Sweep threshold over observed score range for efficiency.
        all_scores_flat = [float(s) for s in scores]
        lo = min(all_scores_flat) if all_scores_flat else 0.0
        hi = max(all_scores_flat) if all_scores_flat else 1.0
        if lo >= hi:
            lo, hi = 0.0, 1.0
        step_size = (hi - lo) / max(1, int(threshold_candidates) - 1)
        best_f1 = -1.0
        best_tau = 0.5
        best_acc_error = 0.0
        best_acc_correct = 0.0
        for i in range(int(threshold_candidates)):
            tau = lo + i * step_size
            f1, acc_error, acc_correct = _f1_at_threshold(tau)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau
                best_acc_error = acc_error
                best_acc_correct = acc_correct
        threshold_selection = "oracle_sweep"

    return {
        "processbench_f1": float(best_f1),
        "processbench_acc_erroneous": float(best_acc_error),
        "processbench_acc_correct": float(best_acc_correct),
        "processbench_f1_threshold": float(best_tau),
        "processbench_f1_threshold_selection": str(threshold_selection),
        "processbench_f1_is_oracle": bool(threshold is None),
    }


def load_prmbench_preview_pairs(
    path: Path,
    *,
    max_samples: int | None = None,
    error_step_index_base: str | int = "auto",
) -> list[PRMBenchPreviewPairRecord]:
    """Load PRMBench preview into explicit prefix pairs.

    中文
    ----
    这条路径语义相对干净，因为数据里本身就提供了：
    1. original process，
    2. modified process，
    3. error step index。
    """
    if not path.exists():
        raise FileNotFoundError(f"PRMBench preview file not found: {path}")
    raw_records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if text == "":
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                raw_records.append((line_no, payload))
    resolved_index_base = _resolve_prmbench_error_step_index_base(
        raw_records=raw_records,
        error_step_index_base=error_step_index_base,
    )
    rows: list[PRMBenchPreviewPairRecord] = []
    for line_no, payload in raw_records:
        original_process = payload.get("original_process")
        modified_process = payload.get("modified_process")
        error_steps = payload.get("error_steps")
        if not isinstance(original_process, list) or not isinstance(modified_process, list):
            continue
        if not isinstance(error_steps, list) or not error_steps:
            continue
        question_text = f"{str(payload.get('question') or payload.get('modified_question') or payload.get('original_question') or '')}\n\n"
        classification = str(payload.get("classification", "unknown"))
        for raw_error_step in error_steps:
            try:
                idx = int(raw_error_step)
            except Exception:  # noqa: BLE001
                continue
            if resolved_index_base == 1:
                idx = idx - 1
            if idx < 0:
                continue
            if idx >= len(original_process) or idx >= len(modified_process):
                continue
            rows.append(
                PRMBenchPreviewPairRecord(
                    pair_id=f"prmbench_preview:{line_no}:{idx}",
                    classification=classification,
                    question_text=question_text,
                    chosen_prefix_text=_join_steps_as_prefix(original_process, idx),
                    rejected_prefix_text=_join_steps_as_prefix(modified_process, idx),
                    error_step_index=int(idx),
                )
            )
            if max_samples is not None and len(rows) >= int(max_samples):
                return rows
    if not rows:
        raise RuntimeError(f"No valid PRMBench preview pairs loaded from {path}")
    return rows


def _resolve_prmbench_error_step_index_base(
    *,
    raw_records: list[tuple[int, dict[str, Any]]],
    error_step_index_base: str | int,
) -> int:
    if isinstance(error_step_index_base, int):
        if int(error_step_index_base) not in {0, 1}:
            raise ValueError("`error_step_index_base` must be 0, 1, or 'auto'")
        return int(error_step_index_base)
    normalized = str(error_step_index_base).strip().lower()
    if normalized in {"0", "zero_based", "zero", "0_based"}:
        return 0
    if normalized in {"1", "one_based", "one", "1_based"}:
        return 1
    if normalized != "auto":
        raise ValueError(
            "`error_step_index_base` must be one of {'auto', 'zero_based', 'one_based', 0, 1}"
        )

    saw_zero_based_signal = False
    saw_one_based_signal = False
    for _, payload in raw_records:
        original_process = payload.get("original_process")
        modified_process = payload.get("modified_process")
        error_steps = payload.get("error_steps")
        if not isinstance(original_process, list) or not isinstance(modified_process, list):
            continue
        if not isinstance(error_steps, list):
            continue
        for raw_error_step in error_steps:
            try:
                idx = int(raw_error_step)
            except Exception:  # noqa: BLE001
                continue
            if idx == 0:
                saw_zero_based_signal = True
            if idx >= len(original_process) or idx >= len(modified_process):
                saw_one_based_signal = True
    if saw_zero_based_signal and saw_one_based_signal:
        raise RuntimeError(
            "PRMBench preview error-step indices look mixed between 0-based and 1-based."
        )
    if saw_zero_based_signal:
        return 0
    if saw_one_based_signal:
        return 1
    raise RuntimeError(
        "PRMBench preview error-step index base is ambiguous. "
        "Pass `error_step_index_base='zero_based'` or `'one_based'` explicitly."
    )


def compute_pair_ranking_metrics(
    *,
    pair_ids: list[str],
    group_keys: list[str],
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> dict[str, Any]:
    """Compute generic chosen-vs-rejected ranking metrics.

    中文
    ----
    这套指标比 ProcessBench 那条路径更直接，因为输入已经是显式 pair：
    1. chosen 是否高于 rejected，
    2. 全局 AUC 如何，
    3. 不同 group 的 margin 分布是否一致。
    """
    if not (
        len(pair_ids) == len(group_keys) == len(chosen_scores) == len(rejected_scores)
    ):
        raise ValueError("Pair metric inputs must have equal length")
    margins = [float(c - r) for c, r in zip(chosen_scores, rejected_scores, strict=True)]
    pair_acc = float(sum(1 for margin in margins if margin > 0.0) / len(margins))
    pair_acc_ties = float(sum(1 for margin in margins if margin >= 0.0) / len(margins))
    auc = float(
        compute_binary_auc(
            scores=[*chosen_scores, *rejected_scores],
            labels=[1] * len(chosen_scores) + [0] * len(rejected_scores),
        )
    )
    by_group: dict[str, list[float]] = {}
    for key, margin in zip(group_keys, margins, strict=True):
        by_group.setdefault(str(key), []).append(float(margin))
    return {
        "num_pairs": int(len(margins)),
        "pair_accuracy": float(pair_acc),
        "pair_accuracy_with_ties": float(pair_acc_ties),
        "auc": float(auc),
        "mean_margin": float(statistics.mean(margins)),
        "median_margin": float(statistics.median(margins)),
        "by_group": {
            key: {
                "n": int(len(values)),
                "mean_margin": float(statistics.mean(values)),
                "positive_margin_rate": float(sum(1 for v in values if v > 0.0) / len(values)),
            }
            for key, values in sorted(by_group.items())
        },
    }


def render_phase_e_benchmark_summary_markdown(
    *,
    title: str,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    """Render a compact Markdown summary for one Phase E benchmark eval.

    中文
    ----
    Markdown 里只放便于人类快速浏览的标量字段；复杂结构保留在 JSON 中供程序分析。
    """
    lines = [f"# {title}", ""]
    for key, value in metadata.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Metrics")
    for key, value in metrics.items():
        if isinstance(value, (dict, list)):
            continue
        if isinstance(value, float):
            lines.append(f"- {key}: `{value:.6f}`")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def _join_steps_as_prefix(steps: list[Any], max_step_index: int) -> str:
    """Join reasoning steps up to one inclusive index as one prefix.

    中文
    ----
    这里的 `max_step_index` 是“包含式”索引，也就是该步本身也会被保留。
    这点和 Python 常见的右开区间切片容易混淆，所以单独说明。
    """
    subset = [str(step).strip() for step in steps[: max_step_index + 1]]
    subset = [step for step in subset if step != ""]
    return "\n".join(subset)
