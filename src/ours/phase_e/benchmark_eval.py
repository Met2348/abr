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
        if max_samples is not None and len(rows) >= int(max_samples):
            break
    if not rows:
        raise RuntimeError(f"No valid ProcessBench rows loaded from {path}")
    return rows


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
    }


def load_prmbench_preview_pairs(
    path: Path,
    *,
    max_samples: int | None = None,
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
    rows: list[PRMBenchPreviewPairRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if text == "":
                continue
            payload = json.loads(text)
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
                # Some public snapshots store 1-based indices while our code uses
                # 0-based indices.  Normalize once at load time so later code
                # stays simple.
                # 有些公开快照用 1-based，而仓库内部统一用 0-based。
                # 在加载时统一归一化，后面逻辑就能保持简单。
                idx = idx - 1 if idx > 0 else idx
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
