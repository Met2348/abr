"""Source adapters for building canonical external pair artifacts.

English
-------
Public process-supervision datasets come in very different raw formats:

1. some already provide direct `chosen/rejected` pairs,
2. some provide one trajectory with step-level `+/-` labels,
3. some provide multiple completions for the same step.

The rest of the repo does not want to understand every source-specific schema.
It only wants one canonical contract:

1. `prompt_text`
2. `chosen_text`
3. `rejected_text`
4. metadata explaining how the pair was built

This file performs that normalization.

中文
----
公开的 process supervision 数据源原始格式差别很大：

1. 有些直接给 `chosen/rejected`；
2. 有些只给一条轨迹和逐步 `+/-` 标签；
3. 有些给的是同一步的多个 completion。

但仓库后面的训练/评测代码不想理解每一种原始格式，它们只想消费统一 pair：

1. `prompt_text`
2. `chosen_text`
3. `rejected_text`
4. 以及说明 pair 如何构造出来的 metadata

本文件就是把各种外部格式标准化成统一 pair contract 的地方。

Important note / 关键提醒
-------------------------
For step-labeled sources, "convert to pair" is not a neutral ETL step.  It
encodes a research assumption about what the supervision actually means.

对于 step-labeled 数据源，把它“转换成 pair”并不是中性的 ETL 操作，而是在编码一种
研究假设：这个监督信号到底代表什么。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .external_pairs import ExternalPairRecord


@dataclass(slots=True)
class PairBuildConfig:
    """Quality-control config shared by external-pair source adapters.

    English
    -------
    This config controls both:

    1. ordinary quality filtering
       - remove degenerate or low-information pairs
    2. pair semantics for step-labeled sources
       - choose how `+/-` trajectories become canonical pairs

    中文
    ----
    这组配置不只是在做普通清洗，还同时决定两件事：

    1. 质量过滤
       - 去掉明显退化、信息量很低的 pair
    2. step-labeled 数据的 pair 语义
       - 决定 `+/-` 轨迹如何变成 canonical pair
    """

    min_chars: int = 12
    max_length_ratio: float = 4.0
    max_token_overlap: float = 0.995
    max_pairs_per_sample: int = 2
    step_label_pair_mode: str = "first_bad_edge_strict"
    step_label_terminal_anchor_mode: str = "none"
    step_label_terminal_anchor_fraction: float = 0.5
    r_prm_pair_mode: str = "direct_pair_legacy"
    prmbench_error_step_index_base: str = "auto"

    def validate(self) -> None:
        if self.min_chars <= 0:
            raise ValueError("`min_chars` must be > 0")
        if self.max_length_ratio <= 1.0:
            raise ValueError("`max_length_ratio` must be > 1")
        if not (0.0 <= self.max_token_overlap <= 1.0):
            raise ValueError("`max_token_overlap` must be in [0, 1]")
        if self.max_pairs_per_sample <= 0:
            raise ValueError("`max_pairs_per_sample` must be > 0")
        if self.step_label_pair_mode not in {
            "first_bad_edge_strict",
            "first_bad_fanout",
            "all_good_vs_all_bad",
            "legacy_nearest",
        }:
            raise ValueError(
                "`step_label_pair_mode` must be one of "
                "{'first_bad_edge_strict', 'first_bad_fanout', 'all_good_vs_all_bad', 'legacy_nearest'}"
            )
        if self.step_label_terminal_anchor_mode not in {
            "none",
            "all_positive_fanout",
        }:
            raise ValueError(
                "`step_label_terminal_anchor_mode` must be one of "
                "{'none', 'all_positive_fanout'}"
            )
        if not (0.0 < float(self.step_label_terminal_anchor_fraction) < 1.0):
            raise ValueError("`step_label_terminal_anchor_fraction` must be in (0, 1)")
        if self.r_prm_pair_mode not in {"direct_pair_legacy", "compact_verdict", "compact_correctness"}:
            raise ValueError(
                "`r_prm_pair_mode` must be one of "
                "{'direct_pair_legacy', 'compact_verdict', 'compact_correctness'}"
            )
        normalized_prmbench_base = str(self.prmbench_error_step_index_base).strip().lower()
        if normalized_prmbench_base not in {"auto", "0", "1", "zero_based", "one_based"}:
            raise ValueError(
                "`prmbench_error_step_index_base` must be one of "
                "{'auto', '0', '1', 'zero_based', 'one_based'}"
            )


def load_r_prm_dpo_pairs(
    *,
    root: Path,
    split: str,
    config: PairBuildConfig,
    max_pairs: int | None = None,
) -> list[ExternalPairRecord]:
    """Load canonical pairs from the R-PRM DPO split.

    English
    -------
    Two modes are supported intentionally:

    1. `direct_pair_legacy`
       - preserve the old behavior exactly
       - use the full chosen/rejected verifier analyses as the pair texts
    2. `compact_verdict`
       - rewrite the sample into a much shorter same-prompt verdict pair
       - keep `Question / Previous Steps / Now Step` in the prompt
       - reduce chosen/rejected to short opposite verdict texts
    3. `compact_correctness`
       - same compact prompt as `compact_verdict`
       - but render answers directly in `Correct/Incorrect` space
       - useful when `Yes/No` phrasing itself introduces polarity bias

    The compact mode exists because the current Phase E value head is a frozen
    feature scorer, not a generative verifier. Feeding multi-thousand-token
    verifier essays into that head was destroying R-PRM's usable signal before
    optimization even started.

    中文
    ----
    这里故意保留两种模式：

    1. `direct_pair_legacy`
       - 完整保留旧行为
       - 直接把长篇 chosen/rejected verifier analysis 当作 pair 文本
    2. `compact_verdict`
       - 把样本改写成更短的“同 prompt 下的相反 verdict pair”
       - prompt 保留 `Question / Previous Steps / Now Step`
       - chosen/rejected 只保留简短 verdict 文本
    3. `compact_correctness`
       - 与 `compact_verdict` 使用同一个紧凑 prompt
       - 但答案直接写成 `Correct/Incorrect`
       - 当 `Yes/No` 这个表达本身带来 polarity bias 时更有用

    `compact_verdict` 的存在原因很明确：当前 Phase E 的 value head 是
    frozen-feature scorer，不是 generative verifier。把几千 token 的分析长文
    直接喂进去，会在优化开始前就把 R-PRM 的有效监督信号破坏掉。
    """
    config.validate()
    split = str(split).strip()
    if split not in {"train", "validation"}:
        raise ValueError("R-PRM split must be one of {train, validation}")
    source_root = root / "dpo" / split
    files = sorted(source_root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards found in {source_root}")

    rows: list[ExternalPairRecord] = []
    for shard_idx, payload in enumerate(
        _iter_parquet_rows(
            files=files,
            columns=("instruction", "chosen", "rejected"),
        )
    ):
        raw_instruction = str(payload.get("instruction", ""))
        raw_chosen = str(payload.get("chosen", ""))
        raw_rejected = str(payload.get("rejected", ""))
        pair_build_mode = "r_prm_direct_pair"
        pair_semantics = "direct_preference_pair"
        pair_confidence = 0.78
        prompt_text = raw_instruction
        chosen_text = raw_chosen
        rejected_text = raw_rejected
        metadata: dict[str, Any] = {
            "source_split": split,
            "source_row_index": int(shard_idx),
            "source_root": str(root),
            "r_prm_pair_mode": str(config.r_prm_pair_mode),
            "raw_instruction_chars": int(len(raw_instruction)),
            "raw_chosen_chars": int(len(raw_chosen)),
            "raw_rejected_chars": int(len(raw_rejected)),
        }
        if config.r_prm_pair_mode in {"compact_verdict", "compact_correctness"}:
            compact_prompt = _extract_r_prm_compact_prompt(raw_instruction)
            chosen_verdict = _extract_r_prm_verdict(raw_chosen)
            rejected_verdict = _extract_r_prm_verdict(raw_rejected)
            # Skip rows whose final supervision signal cannot be reconstructed
            # cleanly into the shorter same-prompt verdict contract.
            # 如果一条样本的最终监督信号无法被干净重建成更短的 same-prompt verdict
            # 形式，就直接跳过，不再退回有问题的长文本契约。
            if compact_prompt is None or chosen_verdict is None or rejected_verdict is None:
                continue
            if chosen_verdict == rejected_verdict:
                continue
            prompt_text = compact_prompt
            if config.r_prm_pair_mode == "compact_verdict":
                chosen_text = _render_r_prm_verdict_text(chosen_verdict)
                rejected_text = _render_r_prm_verdict_text(rejected_verdict)
                pair_build_mode = "r_prm_compact_verdict_pair"
                pair_semantics = "same_prompt_binary_verdict"
            else:
                chosen_text = _render_r_prm_correctness_text(chosen_verdict)
                rejected_text = _render_r_prm_correctness_text(rejected_verdict)
                pair_build_mode = "r_prm_compact_correctness_pair"
                pair_semantics = "same_prompt_binary_correctness"
            pair_confidence = 0.86
            metadata.update(
                {
                    "chosen_verdict": chosen_verdict,
                    "rejected_verdict": rejected_verdict,
                    "compact_prompt_chars": int(len(prompt_text)),
                }
            )
        record = _build_record(
            source_tag="r_prm",
            domain_tag="general_math",
            prompt_text=prompt_text,
            chosen_text=chosen_text,
            rejected_text=rejected_text,
            pair_confidence=pair_confidence,
            metadata={
                **metadata,
                "pair_build_mode": pair_build_mode,
                "pair_semantics": pair_semantics,
            },
            config=config,
        )
        if record is None:
            continue
        rows.append(record)
        if max_pairs is not None and len(rows) >= int(max_pairs):
            break
    return rows


_R_PRM_VERDICT_RE = re.compile(
    r"Verification:\s*Is the step correct \(Yes/No\)\?\s*(Yes|No)",
    flags=re.IGNORECASE,
)
_R_PRM_CASE_RE = re.compile(
    r"Question:\s*(.*?)\nPrevious Steps:\s*(.*?)\nNow Step:\s*(.*?)\n"
    r"Please carefully analyze the correctness of the Now Step\.\s*Reply:\s*$",
    flags=re.DOTALL,
)


def _extract_r_prm_verdict(text: str) -> str | None:
    """Extract the final Yes/No verifier verdict from one R-PRM analysis."""
    matches = list(_R_PRM_VERDICT_RE.finditer(str(text)))
    if not matches:
        return None
    verdict = matches[-1].group(1).strip().lower()
    if verdict not in {"yes", "no"}:
        return None
    return verdict


def _extract_r_prm_compact_prompt(instruction: str) -> str | None:
    """Rewrite one verbose R-PRM instruction into a compact verifier prompt.

    English
    -------
    The original instruction contains a long rubric explaining how to analyze
    the step.  That rubric is useful for generation, but it is mostly noise for
    the current frozen-feature scalar scorer.  We keep only the supervised case
    itself:

    1. question
    2. previous steps
    3. now step
    4. one short verification task line

    中文
    ----
    原始 instruction 里有一大段分析 rubric，对生成 verifier 很有用，但对当前
    frozen-feature scalar scorer 来说大多是噪声。这里保留真正受监督的 case：

    1. question
    2. previous steps
    3. now step
    4. 一行简短的 verification task
    """
    normalized = str(instruction).replace("\r", "\n").strip()
    match = _R_PRM_CASE_RE.search(normalized)
    if match is None:
        return None
    question = match.group(1).strip()
    previous_steps = match.group(2).strip()
    now_step = match.group(3).strip()
    if not question or not previous_steps or not now_step:
        return None
    return (
        f"Question: {question}\n\n"
        f"Previous Steps:\n{previous_steps}\n\n"
        f"Now Step:\n{now_step}\n\n"
        "Task: Decide whether the Now Step is correct.\n"
        "Verification: "
    )


def _render_r_prm_verdict_text(verdict: str) -> str:
    """Render a short verdict string used by compact R-PRM pairs."""
    normalized = str(verdict).strip().lower()
    if normalized == "yes":
        return "The Now Step is correct. Final answer: Yes.\n"
    if normalized == "no":
        return "The Now Step is incorrect. Final answer: No.\n"
    raise ValueError(f"Unsupported R-PRM verdict: {verdict!r}")


def _render_r_prm_correctness_text(verdict: str) -> str:
    """Render a shorter correctness-space answer for compact R-PRM pairs.

    English
    -------
    The repaired compact R-PRM path already strips away the long verifier essay.
    This variant also strips away the explicit `Yes/No` token pair, because deep
    diagnostics showed that the frozen value head can collapse into a global
    polarity preference ("prefer No") even after length issues are fixed.

    中文
    ----
    修复后的 compact R-PRM 已经去掉了长 verifier essay。这个变体进一步去掉
    显式的 `Yes/No` token 对，因为深度诊断显示：即使长度问题已经缓解，当前
    frozen value head 仍可能塌成一个全局 polarity 偏好（例如“总偏爱 No”）。
    """
    normalized = str(verdict).strip().lower()
    if normalized == "yes":
        return "The Now Step is correct.\n"
    if normalized == "no":
        return "The Now Step is incorrect.\n"
    raise ValueError(f"Unsupported R-PRM verdict: {verdict!r}")


def load_prmbench_preview_pairs(
    *,
    path: Path,
    config: PairBuildConfig,
    max_pairs: int | None = None,
) -> list[ExternalPairRecord]:
    """Load step-local process pairs from the PRMBench preview JSONL.

    这份 source 的关键语义不是“完整答案 preference”，而是：
    在某个显式 error step 位置，原始正确 prefix 应当优于被修改后首次出错的 prefix。

    The important contract here is therefore local and step-indexed:
    the clean original prefix at one error step should beat the modified prefix
    that first turns wrong at that same step.

    That local coordinate should be persisted into metadata as
    `positive_step_index` / `negative_step_index`, otherwise the same-family
    trust audit cannot tell whether a checkpoint still preserves first-bad
    discrimination inside PRMBench itself.
    """
    config.validate()
    if not path.exists():
        raise FileNotFoundError(f"PRMBench preview JSONL not found: {path}")

    raw_records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                raw_records.append((line_no, payload))

    resolved_index_base = _resolve_prmbench_preview_error_step_index_base(
        raw_records=raw_records,
        error_step_index_base=str(config.prmbench_error_step_index_base),
    )

    rows: list[ExternalPairRecord] = []
    for line_no, payload in raw_records:
        question = str(
            payload.get("question")
            or payload.get("modified_question")
            or payload.get("original_question")
            or ""
        )
        original_process = payload.get("original_process")
        modified_process = payload.get("modified_process")
        error_steps = payload.get("error_steps")
        if not isinstance(original_process, list) or not isinstance(modified_process, list):
            continue
        if not isinstance(error_steps, list):
            continue
        for error_step in error_steps:
            try:
                idx = int(error_step)
            except Exception:  # noqa: BLE001
                continue
            if resolved_index_base == 1:
                idx = idx - 1
            if idx < 0:
                continue
            if idx >= len(original_process) or idx >= len(modified_process):
                continue
            chosen_text = _join_steps_as_prefix(original_process, idx)
            rejected_text = _join_steps_as_prefix(modified_process, idx)
            record = _build_record(
                source_tag="prmbench_preview",
                domain_tag="general_math",
                prompt_text=f"{question}\n\n",
                chosen_text=chosen_text,
                rejected_text=rejected_text,
                pair_confidence=0.86,
                metadata={
                    "source_row_line": int(line_no),
                    "source_idx": payload.get("idx"),
                    "classification": payload.get("classification"),
                    "error_step_index": int(idx),
                    "positive_step_index": int(idx),
                    "negative_step_index": int(idx),
                    "error_step_index_base": int(resolved_index_base),
                    "num_error_steps": int(len(error_steps)),
                    "pair_build_mode": "prmbench_explicit_error_step",
                    "pair_semantics": "local_modified_process_error_step",
                },
                config=config,
            )
            if record is None:
                continue
            rows.append(record)
            if max_pairs is not None and len(rows) >= int(max_pairs):
                return rows
    return rows


def _resolve_prmbench_preview_error_step_index_base(
    *,
    raw_records: list[tuple[int, dict[str, Any]]],
    error_step_index_base: str,
) -> int:
    """Infer or validate the PRMBench preview step-index convention.

    中文
    ----
    训练 pair 构造和 benchmark 评测必须使用同一个下标约定。这里复用与
    `phase_e.benchmark_eval` 同样的推断逻辑：
    1. 看见 `0` 说明至少有一行是 0-based；
    2. 看见 `idx >= len(process)` 说明至少有一行只能解释成 1-based；
    3. 两种信号同时出现就直接报错，而不是静默猜测。
    """
    normalized = str(error_step_index_base).strip().lower()
    if normalized in {"0", "zero_based"}:
        return 0
    if normalized in {"1", "one_based"}:
        return 1
    if normalized != "auto":
        raise ValueError(
            "`error_step_index_base` must be one of {'auto', '0', '1', 'zero_based', 'one_based'}"
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
        "Set `PairBuildConfig(prmbench_error_step_index_base=...)` explicitly."
    )


def load_math_shepherd_pairs(
    *,
    path: Path,
    config: PairBuildConfig,
    max_pairs: int | None = None,
) -> list[ExternalPairRecord]:
    """Convert Math-Shepherd step labels into local first-bad-edge pair candidates.

    Important
    ---------
    Math-Shepherd only provides one labeled trajectory, not same-step sibling
    alternatives. The safe default is therefore `first_bad_edge_strict`:

    1. find the first negative step,
    2. require at least one positive step before it,
    3. compare the last clean prefix before that step against the prefix that
       includes the first bad step.

    This should be narrated as first-bad-edge supervision, not as exact
    same-step branch preference supervision.

    中文补充
    --------
    这里最容易让新手误解的一点是：Math-Shepherd 并没有给你严格的
    “同一个 prefix 下，当前步的好候选 vs 坏候选”。

    它给的是：
    1. 一条过程轨迹
    2. 每步好坏标签

    所以当前最保守、最诚实的转换，是“最后一个好 prefix”对“第一个坏 prefix”。
    这代表 local first-bad-edge，不代表严格 sibling-branch preference。
    """
    config.validate()
    if not path.exists():
        raise FileNotFoundError(f"Math-Shepherd JSONL not found: {path}")

    rows: list[ExternalPairRecord] = []
    semantic_quotas = _resolve_math_shepherd_semantic_quotas(
        config=config,
        max_pairs=max_pairs,
    )
    semantic_counts = {key: 0 for key in semantic_quotas} if semantic_quotas is not None else {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            payload = json.loads(text)
            label_text = str(payload.get("label", ""))
            task = str(payload.get("task", "")).strip().lower()
            if not label_text:
                continue
            prompt = _extract_problem_prefix(label_text)
            steps = _extract_step_labels_from_math_shepherd(label_text)
            domain_tag = "gsm8k_math" if task == "gsm8k" else "general_math"
            converted = _convert_step_labels_to_pairs(
                prompt_text=f"{prompt}\n\n",
                step_labels=steps,
                source_tag="math_shepherd",
                domain_tag=domain_tag,
                config=config,
                base_metadata={
                    "source_line": int(line_no),
                    "task": payload.get("task"),
                },
            )
            if semantic_quotas is None:
                rows.extend(converted)
                if max_pairs is not None and len(rows) >= int(max_pairs):
                    return rows[: int(max_pairs)]
                continue
            for row in converted:
                pair_semantics = str((row.metadata or {}).get("pair_semantics", "unspecified"))
                quota = semantic_quotas.get(pair_semantics)
                if quota is None:
                    continue
                if semantic_counts[pair_semantics] >= int(quota):
                    continue
                rows.append(row)
                semantic_counts[pair_semantics] += 1
            if all(int(semantic_counts[key]) >= int(semantic_quotas[key]) for key in semantic_quotas):
                return rows[: int(max_pairs)]
    return rows


def _resolve_math_shepherd_semantic_quotas(
    *,
    config: PairBuildConfig,
    max_pairs: int | None,
) -> dict[str, int] | None:
    """Resolve per-semantics source quotas when Math-Shepherd repairs are sparse.

    English
    -------
    Math-Shepherd is stored in a file order where all-positive trajectories
    arrive much later than ordinary first-bad examples. If we apply
    `max_pairs_per_source` as a naive stream head, terminal-anchor repairs can
    disappear before they are ever seen.

    When terminal anchors are enabled, we therefore reinterpret the source cap
    as balanced quotas over the active support semantics. This keeps the source
    budget bounded while ensuring the repair actually enters the artifact.

    中文
    ----
    Math-Shepherd 的文件顺序里，全正轨迹会比普通 first-bad 样本晚很多出现。
    如果直接把 `max_pairs_per_source` 当成流式截头，terminal-anchor 修复样本会在
    还没被看到之前就被截没。

    因此，当 terminal anchor 打开时，这里会把 source cap 重新解释成“按当前激活的
    监督语义做平衡配额”。这样既能控制 source 预算，又能保证修复监督真的进入 artifact。
    """
    if max_pairs is None:
        return None
    if str(config.step_label_terminal_anchor_mode) == "none":
        return None
    terminal_fraction = float(config.step_label_terminal_anchor_fraction)
    terminal_budget = max(1, int(round(int(max_pairs) * terminal_fraction)))
    terminal_budget = min(terminal_budget, int(max_pairs) - 1)
    primary_budget = int(max_pairs) - int(terminal_budget)
    return {
        _math_shepherd_primary_pair_semantics(config): int(primary_budget),
        "terminal_completion_anchor": int(terminal_budget),
    }


def _math_shepherd_primary_pair_semantics(config: PairBuildConfig) -> str:
    """Return the primary canonical pair semantics emitted by the current mode."""
    mode = str(config.step_label_pair_mode)
    if mode == "first_bad_edge_strict":
        return "local_first_bad_edge"
    if mode == "first_bad_fanout":
        return "first_bad_fanout_prefix_ranking"
    if mode == "all_good_vs_all_bad":
        return "good_bad_prefix_grid"
    if mode == "legacy_nearest":
        return "same_trajectory_depth_mixed"
    raise ValueError(f"Unsupported step_label_pair_mode: {mode!r}")


def _distribute_integer_budget_evenly(
    *,
    keys: list[str],
    total_budget: int,
) -> dict[str, int]:
    """Split one integer budget deterministically across ordered keys."""
    if total_budget <= 0:
        raise ValueError("`total_budget` must be > 0")
    unique_keys = sorted({str(key) for key in keys})
    if not unique_keys:
        raise ValueError("At least one key is required to distribute a budget")
    base = int(total_budget) // int(len(unique_keys))
    remainder = int(total_budget) % int(len(unique_keys))
    quotas: dict[str, int] = {}
    for idx, key in enumerate(unique_keys):
        quotas[key] = int(base + (1 if idx < remainder else 0))
    return quotas


def load_prm800k_pairs(
    *,
    path: Path,
    config: PairBuildConfig,
    max_pairs: int | None = None,
) -> list[ExternalPairRecord]:
    """Convert PRM800K records into canonical pair candidates.

    Notes
    -----
    PRM800K releases have multiple community mirrors and schema variants.
    This loader intentionally supports several common shapes and gracefully
    skips rows that cannot be parsed into explicit `+/-` step labels.
    """
    config.validate()
    files = _collect_prm800k_files(path)
    rows: list[ExternalPairRecord] = []
    for file_path in files:
        for row_idx, payload in enumerate(_iter_json_records(file_path), start=1):
            prompt = _extract_prm800k_prompt(payload)
            direct_pairs = _extract_prm800k_completion_pairs(
                payload=payload,
                prompt_text=prompt,
                source_file=file_path,
                source_row_index=int(row_idx),
                config=config,
            )
            if direct_pairs:
                rows.extend(direct_pairs)
                if max_pairs is not None and len(rows) >= int(max_pairs):
                    return rows[: int(max_pairs)]
                continue
            step_labels = _extract_step_labels_from_prm800k_payload(payload)
            converted = _convert_step_labels_to_pairs(
                prompt_text=f"{prompt}\n\n",
                step_labels=step_labels,
                source_tag="prm800k",
                domain_tag="general_math",
                config=config,
                base_metadata={
                    "source_file": str(file_path),
                    "source_row_index": int(row_idx),
                },
            )
            rows.extend(converted)
            if max_pairs is not None and len(rows) >= int(max_pairs):
                return rows[: int(max_pairs)]
    return rows


def _extract_prm800k_completion_pairs(
    *,
    payload: dict[str, Any],
    prompt_text: str,
    source_file: Path,
    source_row_index: int,
    config: PairBuildConfig,
) -> list[ExternalPairRecord]:
    """Build pairs from official PRM800K `label.steps[].completions[]` schema.

    Strategy:
    - For each step, select one acceptable completion (`rating >= 0`) and one
      clearly wrong completion (`rating < 0`).
    - Compose pair texts as prefixes over chosen history + current step variant.

    RISK WARNING
    ------------
    PRM800K community mirrors frequently encode step ratings as `1 / 0 / -1`.
    Treating `0` as a negative class silently flips "neutral/acceptable" steps
    into hard negatives and can make the whole source look near-random.  We
    therefore follow the common downstream convention used by newer benchmark
    baselines: `1` and `0` are non-negative / acceptable, `-1` is negative.
    """
    label_obj = payload.get("label")
    if not isinstance(label_obj, dict):
        return []
    steps = label_obj.get("steps")
    if not isinstance(steps, list) or not steps:
        return []

    rows: list[ExternalPairRecord] = []
    history_steps: list[str] = []
    for step_idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        completions = step.get("completions")
        if not isinstance(completions, list):
            continue

        positive: tuple[str, float] | None = None
        negative: tuple[str, float] | None = None
        for item in completions:
            if not isinstance(item, dict):
                continue
            text_raw = item.get("text")
            if not isinstance(text_raw, str):
                continue
            text = text_raw.strip()
            if text == "":
                continue
            rating = _safe_rating_value(item.get("rating"))
            if rating is None:
                continue
            if rating >= 0.0:
                if positive is None or rating > positive[1]:
                    positive = (text, rating)
            else:
                if negative is None or rating < negative[1]:
                    negative = (text, rating)

        if positive is not None and negative is not None:
            chosen_prefix = _join_steps_as_prefix(
                [*history_steps, positive[0]],
                len(history_steps),
            )
            rejected_prefix = _join_steps_as_prefix(
                [*history_steps, negative[0]],
                len(history_steps),
            )
            rating_gap = positive[1] - negative[1]
            if rating_gap >= 2.0:
                confidence = 0.78
            elif rating_gap >= 1.0:
                confidence = 0.70
            else:
                confidence = 0.62
            record = _build_record(
                source_tag="prm800k",
                domain_tag="general_math",
                prompt_text=f"{prompt_text}\n\n",
                chosen_text=chosen_prefix,
                rejected_text=rejected_prefix,
                pair_confidence=confidence,
                metadata={
                    "source_file": str(source_file),
                    "source_row_index": int(source_row_index),
                    "positive_step_index": int(step_idx),
                    "negative_step_index": int(step_idx),
                    "rating_positive": float(positive[1]),
                    "rating_negative": float(negative[1]),
                    "rating_policy": "non_negative_positive",
                    "pair_build_mode": "prm800k_completion_ratings",
                    "pair_semantics": "same_step_completion_preference",
                },
                config=config,
            )
            if record is not None:
                rows.append(record)
                if len(rows) >= int(config.max_pairs_per_sample):
                    break

        # Keep prefix progression close to annotator-selected chain when possible.
        chosen_idx = step.get("chosen_completion")
        next_history = None
        if isinstance(chosen_idx, int) and 0 <= chosen_idx < len(completions):
            chosen_item = completions[chosen_idx]
            if isinstance(chosen_item, dict):
                text_raw = chosen_item.get("text")
                if isinstance(text_raw, str) and text_raw.strip():
                    next_history = text_raw.strip()
        if next_history is None and positive is not None:
            next_history = positive[0]
        if next_history is not None:
            history_steps.append(next_history)
    return rows


def load_rlhflow_pairs(
    *,
    mistral_root: Path | None,
    deepseek_path: Path | None,
    config: PairBuildConfig,
    max_pairs_per_source: int | None = None,
) -> list[ExternalPairRecord]:
    """Convert RLHFlow step-label conversations into local first-bad-edge pairs."""
    config.validate()
    rows: list[ExternalPairRecord] = []
    if mistral_root is not None:
        rows.extend(
            _load_rlhflow_mistral_pairs(
                root=mistral_root,
                config=config,
                max_pairs=max_pairs_per_source,
            )
        )
    if deepseek_path is not None:
        rows.extend(
            _load_rlhflow_deepseek_pairs(
                path=deepseek_path,
                config=config,
                max_pairs=max_pairs_per_source,
            )
        )
    return rows


def _load_rlhflow_mistral_pairs(
    *,
    root: Path,
    config: PairBuildConfig,
    max_pairs: int | None = None,
) -> list[ExternalPairRecord]:
    source_root = root / "data"
    files = sorted(source_root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No RLHFlow mistral shards found in {source_root}")

    rows: list[ExternalPairRecord] = []
    for row_idx, payload in enumerate(_iter_parquet_rows(files=files, columns=("conversations",))):
        conv = payload.get("conversations")
        if not isinstance(conv, list):
            continue
        step_labels = _extract_step_labels_from_conversations(conv)
        prompt = _extract_problem_prefix(str(conv[0].get("content", ""))) if conv else ""
        converted = _convert_step_labels_to_pairs(
            prompt_text=f"{prompt}\n\n",
            step_labels=step_labels,
            source_tag="rlhflow_mistral",
            domain_tag="general_math",
            config=config,
            base_metadata={
                "source_row_index": int(row_idx),
                "source_root": str(root),
            },
        )
        rows.extend(converted)
        if max_pairs is not None and len(rows) >= int(max_pairs):
            return rows[: int(max_pairs)]
    return rows


def _load_rlhflow_deepseek_pairs(
    *,
    path: Path,
    config: PairBuildConfig,
    max_pairs: int | None = None,
) -> list[ExternalPairRecord]:
    if not path.exists():
        raise FileNotFoundError(f"RLHFlow deepseek JSONL not found: {path}")
    rows: list[ExternalPairRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            payload = json.loads(text)
            conv = payload.get("conversations")
            if not isinstance(conv, list):
                continue
            step_labels = _extract_step_labels_from_conversations(conv)
            prompt = _extract_problem_prefix(str(conv[0].get("content", ""))) if conv else ""
            converted = _convert_step_labels_to_pairs(
                prompt_text=f"{prompt}\n\n",
                step_labels=step_labels,
                source_tag="rlhflow_deepseek",
                domain_tag="general_math",
                config=config,
                base_metadata={
                    "source_line": int(line_no),
                    "source_path": str(path),
                },
            )
            rows.extend(converted)
            if max_pairs is not None and len(rows) >= int(max_pairs):
                return rows[: int(max_pairs)]
    return rows


def _iter_parquet_rows(
    *,
    files: list[Path],
    columns: Iterable[str],
) -> Iterable[dict[str, Any]]:
    """Yield parquet rows as plain dictionaries."""
    column_names = list(columns)
    try:
        import pyarrow.parquet as pq
        reader_mode = "pyarrow.parquet"
    except Exception as exc:  # noqa: BLE001
        # Some of our lab environments end up with a mixed pyarrow install where
        # `pyarrow.parquet` imports the broken fs/azure bindings, but the low-level
        # parquet reader still works. 这里优先降级到 `_parquet`，避免整个 Phase E
        # 因为 parquet runtime 的 ABI 冲突而完全不可用。
        try:
            import pyarrow._parquet as pq  # type: ignore[no-redef]
            reader_mode = "pyarrow._parquet"
        except Exception as fallback_exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to import a usable parquet reader while reading external parquet shards. "
                "Tried `pyarrow.parquet` and fallback `pyarrow._parquet`. "
                "Please fix your pyarrow runtime first, for example: "
                "`python -m pip install -U pyarrow` "
                "and make sure no mixed conda/pip binary conflict remains."
            ) from fallback_exc

    for file_path in files:
        if reader_mode == "pyarrow.parquet":
            pf = pq.ParquetFile(file_path)
            batches = pf.iter_batches(batch_size=1024, columns=column_names)
        else:
            reader = pq.ParquetReader()
            reader.open(
                str(file_path),
                use_memory_map=False,
                read_dictionary=[],
                buffer_size=0,
                pre_buffer=False,
            )
            schema_names = list(reader.schema_arrow.names)
            col_indices = [schema_names.index(name) for name in column_names]
            row_groups = list(range(reader.num_row_groups))
            batches = reader.iter_batches(
                batch_size=1024,
                row_groups=row_groups,
                column_indices=col_indices,
                use_threads=True,
            )
        for batch in batches:
            names = list(batch.schema.names)
            cols = [batch.column(i) for i in range(len(names))]
            for row_idx in range(batch.num_rows):
                payload: dict[str, Any] = {}
                for col_idx, name in enumerate(names):
                    payload[name] = cols[col_idx][row_idx].as_py()
                yield payload


def load_math_step_dpo_pairs(
    *,
    path: Path,
    config: PairBuildConfig,
    max_pairs: int | None = None,
) -> list[ExternalPairRecord]:
    """Convert Math-Step-DPO-10K parquet rows into sibling-branch pairs.

    English
    -------
    Math-Step-DPO-10K (xinlai/Math-Step-DPO-10K) provides explicit fork-point pairs:
    - `initial_reason_steps`: the shared correct prefix ending at the branch step marker
    - `chosen`: the correct step continuation (good branch)
    - `rejected`: the wrong step continuation (bad branch)
    These combine into `sibling_branch` pairs at a known divergence point.
    Confidence is set high (0.80) since pairs are human-curated from the paper.

    中文
    ----
    Math-Step-DPO-10K 提供显式分叉点 pair：
    - `initial_reason_steps`：共享前缀（截止分叉点处）
    - `chosen`：正确后续（好分支）
    - `rejected`：错误后续（坏分支）
    三者拼接构成干净的 `sibling_branch` pair。置信度设为 0.80（论文人工精选）。
    """
    config.validate()
    files = sorted(Path(path).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Math-Step-DPO parquet files found under: {path}")

    rows: list[ExternalPairRecord] = []
    row_idx = 0
    columns = ("prompt", "initial_reason_steps", "chosen", "rejected", "dataset")
    for payload in _iter_parquet_rows(files=files, columns=columns):
        row_idx += 1
        prompt = str(payload.get("prompt") or "")
        prefix = str(payload.get("initial_reason_steps") or "")
        chosen_step = str(payload.get("chosen") or "")
        rejected_step = str(payload.get("rejected") or "")
        dataset_tag = str(payload.get("dataset") or "math_step_dpo")

        if not prefix.strip() or not chosen_step.strip() or not rejected_step.strip():
            continue

        # 拼接完整文本：shared prefix + branch step / Concatenate: shared prefix + branch step
        chosen_text = prefix + chosen_step
        rejected_text = prefix + rejected_step

        flags = _compute_quality_flags(
            prompt_text=prompt,
            chosen_text=chosen_text,
            rejected_text=rejected_text,
        )
        if not _passes_quality_filter(flags=flags, config=config):
            continue

        pair_id = _stable_hash("math_step_dpo", str(row_idx), prompt, chosen_step, rejected_step)
        rows.append(
            ExternalPairRecord(
                pair_id=pair_id,
                source_tag="math_step_dpo",
                domain_tag="math",
                prompt_text=prompt + "\n\n" if prompt else "",
                chosen_text=chosen_text,
                rejected_text=rejected_text,
                pair_confidence=0.80,
                quality_flags=flags,
                metadata={
                    "source_row_index": int(row_idx),
                    "source_path": str(path),
                    "source_dataset_tag": dataset_tag,
                    "pair_semantics": "sibling_branch",
                    "pair_build_mode": "math_step_dpo_fork_pair",
                    "split_group_id": _stable_hash("math_step_dpo", prompt, prefix),
                },
            )
        )
        if max_pairs is not None and len(rows) >= int(max_pairs):
            break
    return rows


def _collect_prm800k_files(path: Path) -> list[Path]:
    """Resolve PRM800K input path into concrete JSON/JSONL files."""
    if not path.exists():
        raise FileNotFoundError(f"PRM800K path not found: {path}")
    if path.is_file():
        return [path]

    files = sorted(path.rglob("*.jsonl")) + sorted(path.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON/JSONL files found under PRM800K path: {path}")
    return files


def _iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON object rows from JSONL or JSON files.

    Supported JSON structures:
    1. JSONL: one dict per line.
    2. JSON list: `[ {...}, {...} ]`
    3. JSON dict with list payload under common keys such as `data`, `records`,
       or `examples`.
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                text = raw.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if isinstance(payload, dict):
                    yield payload
                else:
                    raise TypeError(f"{path}:{line_no} must be a JSON object")
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
        return
    if isinstance(payload, dict):
        for key in ("data", "records", "examples", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        yield row
                return
        # Some mirrors store one sample per JSON file.
        yield payload
        return
    raise TypeError(f"Unsupported JSON payload type in {path}: {type(payload)!r}")


def _extract_problem_prefix(text: str) -> str:
    """Extract question/problem prefix before step markers."""
    normalized = " ".join(str(text).replace("\r", "\n").split())
    match = re.split(r"(?i)\bstep\s*1\s*:", normalized, maxsplit=1)
    if not match:
        return normalized[:512]
    return match[0].strip()[:512]


def _extract_prm800k_prompt(payload: dict[str, Any]) -> str:
    """Extract a question/prompt string from one PRM800K row."""
    for key in ("question", "problem", "prompt", "input", "query"):
        value = payload.get(key)
        text = _coerce_question_text(value)
        if text:
            return text[:1024]
    # Fallback to empty prompt to keep deterministic behavior.
    return ""


def _coerce_question_text(value: Any) -> str:
    """Coerce mixed question payloads into one plain string."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("problem", "question", "prompt", "text", "input"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return ""


def _extract_step_labels_from_math_shepherd(label_text: str) -> list[tuple[str, str]]:
    """Extract `(step_text, label)` from Math-Shepherd `label` field."""
    rows: list[tuple[str, str]] = []
    lines = [line.strip() for line in str(label_text).replace("\r", "\n").split("\n")]
    for line in lines:
        if not line or not line.lower().startswith("step "):
            continue
        match = re.match(r"^(Step\s+\d+\s*:\s*.*?)(?:\s+([+-]))\s*$", line)
        if not match:
            continue
        step_text = str(match.group(1)).strip()
        label = str(match.group(2)).strip()
        if label not in {"+", "-"}:
            continue
        rows.append((step_text, label))
    return rows


def _extract_step_labels_from_prm800k_payload(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract `(step_text, +/-)` rows from one PRM800K-like payload."""
    # Path A: textual `label` with "Step ... +/-" format.
    raw_label = payload.get("label")
    if isinstance(raw_label, str):
        parsed = _extract_step_labels_from_suffix_text(raw_label)
        if parsed:
            return parsed

    # Path B: explicit step container under common fields.
    step_candidates: list[Any] = []
    if isinstance(payload.get("steps"), list):
        step_candidates.extend(list(payload["steps"]))
    label_obj = payload.get("label")
    if isinstance(label_obj, dict):
        if isinstance(label_obj.get("steps"), list):
            step_candidates.extend(list(label_obj.get("steps")))
        if isinstance(label_obj.get("process"), list):
            step_candidates.extend(list(label_obj.get("process")))
    if isinstance(payload.get("process"), list):
        step_candidates.extend(list(payload["process"]))

    rows: list[tuple[str, str]] = []
    for step in step_candidates:
        parsed = _extract_one_step_label(step)
        if parsed is None:
            continue
        rows.append(parsed)
    return rows


def _extract_step_labels_from_suffix_text(text: str) -> list[tuple[str, str]]:
    """Parse lines like `Step 2: ... +` or `Step 3: ... -`."""
    rows: list[tuple[str, str]] = []
    lines = [line.strip() for line in str(text).replace("\r", "\n").split("\n")]
    for line in lines:
        if not line:
            continue
        match = re.match(r"^(Step\s+\d+\s*:\s*.*?)(?:\s+([+-]))\s*$", line)
        if not match:
            continue
        step_text = str(match.group(1)).strip()
        sign = str(match.group(2)).strip()
        if sign in {"+", "-"}:
            rows.append((step_text, sign))
    return rows


def _extract_one_step_label(step: Any) -> tuple[str, str] | None:
    """Extract one `(step_text, +/-)` tuple from mixed step payload formats."""
    if isinstance(step, str):
        # String-only step cannot infer correctness label.
        return None
    if not isinstance(step, dict):
        return None

    text = ""
    for key in ("text", "content", "step", "completion", "output"):
        value = step.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            break
    if text == "":
        return None

    sign = _parse_sign_value(
        step.get("label", step.get("rating", step.get("is_correct", step.get("correct", step.get("value")))))
    )
    if sign is None:
        return None
    return text, sign


def _parse_sign_value(value: Any) -> str | None:
    """Map mixed label/rating payloads to `+` or `-`."""
    if isinstance(value, bool):
        return "+" if value else "-"
    if isinstance(value, (int, float)):
        return "+" if float(value) > 0 else "-"
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"+", "correct", "true", "yes", "good", "positive", "1"}:
            return "+"
        if norm in {"-", "incorrect", "false", "no", "bad", "negative", "0"}:
            return "-"
    return None


def _safe_rating_value(value: Any) -> float | None:
    """Convert one rating field to float, ignoring invalid values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return None
    return out


def _extract_step_labels_from_conversations(conversations: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Extract `(step_text, label)` from RLHFlow conversation turns."""
    rows: list[tuple[str, str]] = []
    for idx in range(len(conversations) - 1):
        left = conversations[idx]
        right = conversations[idx + 1]
        if str(left.get("role", "")).strip().lower() != "user":
            continue
        if str(right.get("role", "")).strip().lower() != "assistant":
            continue
        label = str(right.get("content", "")).strip()
        if label not in {"+", "-"}:
            continue
        step_text = str(left.get("content", "")).strip()
        if step_text == "":
            continue
        rows.append((step_text, label))
    return rows


def _convert_step_labels_to_pairs(
    *,
    prompt_text: str,
    step_labels: list[tuple[str, str]],
    source_tag: str,
    domain_tag: str,
    config: PairBuildConfig,
    base_metadata: dict[str, Any],
) -> list[ExternalPairRecord]:
    """Convert `+/-` step labels into local first-bad-edge or legacy pairs.

    Important
    ---------
    Step-labeled sources such as Math-Shepherd / RLHFlow only provide one
    trajectory with per-step `+/-` tags. They do *not* natively provide
    same-step sibling alternatives.

    Safe default:
    1. `first_bad_edge_strict`
       - find the first negative step,
       - require at least one positive step before it,
       - compare the last clean prefix against the prefix that includes the
         first bad step.

    Benchmark-alignment variants:
    2. `first_bad_fanout`
       - keep the same first-bad negative prefix,
       - but compare it against multiple earlier good prefixes from the same
         trajectory.
       - This is explicitly trying to better match ProcessBench's
         `any-good vs first-bad` evaluation slice.
    3. `all_good_vs_all_bad`
       - compare multiple good prefixes before the first bad step against
         multiple bad prefixes from the same trajectory.
       - This is the closest simple approximation to ProcessBench's
         good-vs-bad pair geometry, but it also mixes local error signal with
         later-bad severity / trajectory depth.

    Legacy mode remains available only for historical forensics:
    4. `legacy_nearest`
       - compare a positive-index prefix with the nearest negative-index prefix
         from the same trajectory.
       - This mixes depth/progress signal with local error signal and should
         not be used for new mainline experiments.

    中文
    ----
    这个分支点本身就有研究含义：

    1. `first_bad_edge_strict`
       - 语义最干净，更接近“局部第一次出错”
       - 代价是监督支持面最窄
    2. `first_bad_fanout`
       - 仍然只围绕 first bad 做负样本
       - 但会把更早的好 prefix 也拉进来，主动靠近 ProcessBench 的
         `any-good vs first-bad` 评测关系
    3. `all_good_vs_all_bad`
       - 最接近 ProcessBench 的全量 good-vs-bad prefix 几何
       - 但研究语义也最“重”，因为它会把局部错误、坏前缀严重程度、轨迹深度一起编码进 pair
    4. `legacy_nearest`
       - 样本可能更多
       - 但会把“推理深度/进度”与“局部错误”混在一起

    所以后面的实验结果怎么解释，很大程度取决于这里选的是哪条路径。
    """
    if config.step_label_pair_mode == "first_bad_edge_strict":
        base_rows = _convert_step_labels_to_first_bad_edge_pairs(
            prompt_text=prompt_text,
            step_labels=step_labels,
            source_tag=source_tag,
            domain_tag=domain_tag,
            config=config,
            base_metadata=base_metadata,
        )
    elif config.step_label_pair_mode == "first_bad_fanout":
        base_rows = _convert_step_labels_to_first_bad_fanout_pairs(
            prompt_text=prompt_text,
            step_labels=step_labels,
            source_tag=source_tag,
            domain_tag=domain_tag,
            config=config,
            base_metadata=base_metadata,
        )
    elif config.step_label_pair_mode == "all_good_vs_all_bad":
        base_rows = _convert_step_labels_to_all_good_vs_all_bad_pairs(
            prompt_text=prompt_text,
            step_labels=step_labels,
            source_tag=source_tag,
            domain_tag=domain_tag,
            config=config,
            base_metadata=base_metadata,
        )
    elif config.step_label_pair_mode == "legacy_nearest":
        base_rows = _convert_step_labels_to_legacy_pairs(
            prompt_text=prompt_text,
            step_labels=step_labels,
            source_tag=source_tag,
            domain_tag=domain_tag,
            config=config,
            base_metadata=base_metadata,
        )
    else:
        raise ValueError(f"Unsupported step_label_pair_mode: {config.step_label_pair_mode!r}")

    # Terminal-anchor supervision is intentionally orthogonal to the main
    # good-vs-bad conversion mode.
    # terminal-anchor 监督故意设计成与主 good-vs-bad 转换模式正交，
    # 这样我们才能单独回答“缺 all-correct 终点锚点到底伤了多少迁移”。
    anchor_rows = _convert_step_labels_to_terminal_anchor_pairs(
        prompt_text=prompt_text,
        step_labels=step_labels,
        source_tag=source_tag,
        domain_tag=domain_tag,
        config=config,
        base_metadata=base_metadata,
    )
    return [*base_rows, *anchor_rows]


def _convert_step_labels_to_terminal_anchor_pairs(
    *,
    prompt_text: str,
    step_labels: list[tuple[str, str]],
    source_tag: str,
    domain_tag: str,
    config: PairBuildConfig,
    base_metadata: dict[str, Any],
) -> list[ExternalPairRecord]:
    """Build synthetic terminal anchors from all-positive trajectories.

    English
    -------
    ProcessBench contains a large `all-correct` slice where the model must rank
    the *final correct prefix* above earlier safe prefixes.

    Pure first-bad-edge conversion never exposes that relation, because
    all-positive trajectories yield zero pairs.  This helper closes that gap by
    turning an all-positive trajectory into:
    1. chosen = full correct solution,
    2. rejected = one or more earlier safe prefixes.

    中文
    ----
    ProcessBench 有相当大一块 `all-correct` 评测：模型需要把“完整正确终点”
    排在更早的安全前缀前面。

    纯 first-bad-edge 转换永远看不到这种关系，因为全正轨迹以前会直接产生 0 个 pair。
    这里就是把全正轨迹补成：
    1. chosen = 完整正确解答，
    2. rejected = 若干更早的安全前缀，
    从而显式补上 terminal-anchor 支持面。
    """
    if config.step_label_terminal_anchor_mode == "none":
        return []
    if config.step_label_terminal_anchor_mode != "all_positive_fanout":
        raise ValueError(
            f"Unsupported step_label_terminal_anchor_mode: {config.step_label_terminal_anchor_mode!r}"
        )
    if len(step_labels) < 2:
        return []
    if any(label != "+" for _, label in step_labels):
        return []

    terminal_idx = len(step_labels) - 1
    candidate_rejected_indices = list(range(int(terminal_idx) - 1, -1, -1))
    selected_rejected_indices = _select_diverse_candidates(
        candidates=candidate_rejected_indices,
        max_items=int(config.max_pairs_per_sample),
    )
    rows: list[ExternalPairRecord] = []
    full_prefix = _join_steps_as_prefix([step for step, _ in step_labels], int(terminal_idx))
    for rejected_idx in selected_rejected_indices:
        rejected_prefix = _join_steps_as_prefix([step for step, _ in step_labels], int(rejected_idx))
        step_gap = int(terminal_idx) - int(rejected_idx)
        # Keep this confidence slightly below clean local first-bad-edge pairs:
        # the anchor is useful, but it can still leak length/progress bias.
        # 这里故意把 confidence 设得略低于局部 first-bad-edge：
        # 它是有价值的补监督，但仍然可能混入长度/进度偏差。
        confidence = 0.70 if step_gap <= 1 else 0.66
        record = _build_record(
            source_tag=source_tag,
            domain_tag=domain_tag,
            prompt_text=prompt_text,
            chosen_text=full_prefix,
            rejected_text=rejected_prefix,
            pair_confidence=confidence,
            metadata={
                **base_metadata,
                "positive_step_index": int(terminal_idx),
                "negative_step_index": int(rejected_idx),
                "num_step_labels": int(len(step_labels)),
                "step_gap": int(step_gap),
                "terminal_step_index": int(terminal_idx),
                "pair_build_mode": "step_label_all_positive_terminal_anchor",
                "pair_semantics": "terminal_completion_anchor",
            },
            config=config,
        )
        if record is not None:
            rows.append(record)
    return rows


def _convert_step_labels_to_first_bad_edge_pairs(
    *,
    prompt_text: str,
    step_labels: list[tuple[str, str]],
    source_tag: str,
    domain_tag: str,
    config: PairBuildConfig,
    base_metadata: dict[str, Any],
) -> list[ExternalPairRecord]:
    """Build one strict local first-bad-edge pair from a labeled trajectory.

    English
    -------
    Example:
    1. step 0 = good
    2. step 1 = good
    3. step 2 = bad

    Then we emit exactly one pair:
    1. chosen = prefix through step 1
    2. rejected = prefix through step 2

    中文
    ----
    举例：
    1. 第 0 步是好
    2. 第 1 步是好
    3. 第 2 步是坏

    那么这里只会构造一个 pair：
    1. chosen = 截到第 1 步的 prefix
    2. rejected = 截到第 2 步的 prefix
    """
    first_negative_idx = next((idx for idx, (_, label) in enumerate(step_labels) if label == "-"), None)
    if first_negative_idx is None:
        return []
    if int(first_negative_idx) <= 0:
        # If the first negative label appears immediately, there is no reliable
        # "last good prefix" to compare against.
        # 如果一开始就出现负标签，就不存在可信的“最后一个好 prefix”。
        #
        # Keeping such rows would force us to invent a pseudo-positive sample,
        # which would quietly change the semantics of the supervision.
        # 这类样本如果强行保留，相当于硬造一个正样本，会悄悄改变监督语义。
        return []

    chosen_idx = int(first_negative_idx) - 1
    rejected_idx = int(first_negative_idx)
    chosen_text = _join_steps_as_prefix([step for step, _ in step_labels], chosen_idx)
    rejected_text = _join_steps_as_prefix([step for step, _ in step_labels], rejected_idx)
    record = _build_record(
        source_tag=source_tag,
        domain_tag=domain_tag,
        prompt_text=prompt_text,
        chosen_text=chosen_text,
        rejected_text=rejected_text,
        pair_confidence=0.74,
        metadata={
            **base_metadata,
            "positive_step_index": int(chosen_idx),
            "negative_step_index": int(rejected_idx),
            "first_negative_index": int(first_negative_idx),
            "num_step_labels": int(len(step_labels)),
            "pair_build_mode": "step_label_first_bad_edge_strict",
            "pair_semantics": "local_first_bad_edge",
        },
        config=config,
    )
    return [record] if record is not None else []


def _convert_step_labels_to_first_bad_fanout_pairs(
    *,
    prompt_text: str,
    step_labels: list[tuple[str, str]],
    source_tag: str,
    domain_tag: str,
    config: PairBuildConfig,
    base_metadata: dict[str, Any],
) -> list[ExternalPairRecord]:
    """Build a capped fanout of `good prefix -> first bad prefix` pairs.

    English
    -------
    This mode keeps the negative side fixed at the first bad prefix, but lets
    multiple earlier good prefixes compete against it.  It exists because
    ProcessBench does not only ask:
    "is the last safe prefix above the first bad prefix?"
    It also implicitly asks:
    "are *other* earlier good prefixes above that first bad prefix too?"

    中文
    ----
    这个模式把负样本固定为 first bad prefix，但允许多个更早的好 prefix 去和它比较。
    它存在的原因是：ProcessBench 并不只测
    “最后一个安全 prefix 是否高于 first bad prefix”，
    它还会隐含地测
    “更早的那些好 prefix，是否也整体高于这个 first bad prefix”。
    """
    first_negative_idx = next((idx for idx, (_, label) in enumerate(step_labels) if label == "-"), None)
    if first_negative_idx is None or int(first_negative_idx) <= 0:
        return []
    candidate_positive_indices = list(range(int(first_negative_idx) - 1, -1, -1))
    selected_positive_indices = _select_diverse_candidates(
        candidates=candidate_positive_indices,
        max_items=int(config.max_pairs_per_sample),
    )
    rows: list[ExternalPairRecord] = []
    for chosen_idx in selected_positive_indices:
        record = _build_step_label_pair_record(
            prompt_text=prompt_text,
            step_labels=step_labels,
            source_tag=source_tag,
            domain_tag=domain_tag,
            chosen_idx=int(chosen_idx),
            rejected_idx=int(first_negative_idx),
            first_negative_idx=int(first_negative_idx),
            config=config,
            base_metadata=base_metadata,
            pair_build_mode="step_label_first_bad_fanout",
            pair_semantics="first_bad_fanout_prefix_ranking",
        )
        if record is not None:
            rows.append(record)
    return rows


def _convert_step_labels_to_all_good_vs_all_bad_pairs(
    *,
    prompt_text: str,
    step_labels: list[tuple[str, str]],
    source_tag: str,
    domain_tag: str,
    config: PairBuildConfig,
    base_metadata: dict[str, Any],
) -> list[ExternalPairRecord]:
    """Build a capped grid of `good prefix -> bad prefix` pairs from one trajectory.

    English
    -------
    This mode is intentionally more benchmark-aligned and less semantically
    pure than `first_bad_edge_strict`.

    RISK WARNING:
    The resulting pairs no longer isolate only the first local failure edge.
    They also expose later-bad prefixes, which means the model can partially
    learn:
    1. cumulative damage after the first error,
    2. trajectory depth,
    3. and text-length growth.

    We keep this mode because ProcessBench itself evaluates many such
    good-vs-bad prefix relations, especially on the math split.

    中文
    ----
    这个模式是故意“更贴近 benchmark、但语义更重”的选择。

    RISK WARNING:
    这里构造出的 pair 已经不再只隔离 first bad 的局部边界。
    它还把 later bad prefix 暴露给模型，因此模型可能部分学到：
    1. 第一次出错之后的累积损伤，
    2. 轨迹走到了多深，
    3. 以及文本长度增长。

    我们仍然保留它，是因为 ProcessBench 本身就大量评测这种
    good-vs-bad prefix 关系，尤其在 `math` split 上更明显。
    """
    first_negative_idx = next((idx for idx, (_, label) in enumerate(step_labels) if label == "-"), None)
    if first_negative_idx is None or int(first_negative_idx) <= 0:
        return []
    candidate_pairs: list[tuple[int, int]] = []
    # Priority order:
    # 1. local edge first,
    # 2. other good-vs-first-bad comparisons,
    # 3. last-safe-vs-later-bad comparisons,
    # 4. remaining grid pairs.
    # 这样在需要裁剪到 `max_pairs_per_sample` 时，最先保留下来的总是研究语义更核心的那些 pair。
    candidate_pairs.append((int(first_negative_idx) - 1, int(first_negative_idx)))
    for chosen_idx in range(int(first_negative_idx) - 2, -1, -1):
        candidate_pairs.append((int(chosen_idx), int(first_negative_idx)))
    for rejected_idx in range(int(first_negative_idx) + 1, len(step_labels)):
        candidate_pairs.append((int(first_negative_idx) - 1, int(rejected_idx)))
    for rejected_idx in range(int(first_negative_idx) + 1, len(step_labels)):
        for chosen_idx in range(int(first_negative_idx) - 2, -1, -1):
            candidate_pairs.append((int(chosen_idx), int(rejected_idx)))

    selected_pairs = _select_diverse_candidates(
        candidates=candidate_pairs,
        max_items=int(config.max_pairs_per_sample),
    )
    rows: list[ExternalPairRecord] = []
    for chosen_idx, rejected_idx in selected_pairs:
        record = _build_step_label_pair_record(
            prompt_text=prompt_text,
            step_labels=step_labels,
            source_tag=source_tag,
            domain_tag=domain_tag,
            chosen_idx=int(chosen_idx),
            rejected_idx=int(rejected_idx),
            first_negative_idx=int(first_negative_idx),
            config=config,
            base_metadata=base_metadata,
            pair_build_mode="step_label_all_good_vs_all_bad",
            pair_semantics="good_bad_prefix_grid",
        )
        if record is not None:
            rows.append(record)
    return rows


def _select_diverse_candidates(
    *,
    candidates: list[Any],
    max_items: int,
) -> list[Any]:
    """Keep a small, deterministic, coverage-preserving subset of candidates.

    English
    -------
    When a trajectory can generate many benchmark-aligned pairs, we still need
    a cap so pair preparation remains tractable on shared research machines.
    The strategy here is deliberately simple:

    1. always keep the first candidate,
    2. then spread the remaining picks across the full candidate list.

    This preserves a strong local anchor while still exposing farther geometric
    variants.

    中文
    ----
    一条轨迹可能生成很多更贴近 benchmark 的 pair，但我们仍然需要上限，避免
    共享服务器上的 pair 准备阶段膨胀失控。

    这里的策略故意保持简单：
    1. 永远保留第一个候选，
    2. 其余名额尽量在整条候选列表上均匀铺开。

    这样既保住最核心的局部边界 pair，又能覆盖更远的几何变体。
    """
    if int(max_items) <= 0:
        raise ValueError("`max_items` must be > 0")
    if len(candidates) <= int(max_items):
        return list(candidates)
    if int(max_items) == 1:
        return [candidates[0]]
    selected = [candidates[0]]
    remaining = candidates[1:]
    if not remaining:
        return selected
    if int(max_items) == 2:
        selected.append(remaining[-1])
        return selected
    used_indices: set[int] = set()
    denominator = int(max_items) - 2
    for slot in range(int(max_items) - 1):
        if denominator <= 0:
            index = len(remaining) - 1
        else:
            index = round(slot * (len(remaining) - 1) / denominator)
        index = min(max(int(index), 0), len(remaining) - 1)
        if index in used_indices:
            continue
        selected.append(remaining[index])
        used_indices.add(index)
        if len(selected) >= int(max_items):
            return selected
    for index, item in enumerate(remaining):
        if index in used_indices:
            continue
        selected.append(item)
        if len(selected) >= int(max_items):
            break
    return selected


def _build_step_label_pair_record(
    *,
    prompt_text: str,
    step_labels: list[tuple[str, str]],
    source_tag: str,
    domain_tag: str,
    chosen_idx: int,
    rejected_idx: int,
    first_negative_idx: int,
    config: PairBuildConfig,
    base_metadata: dict[str, Any],
    pair_build_mode: str,
    pair_semantics: str,
) -> ExternalPairRecord | None:
    """Build one step-label-derived pair with shared metadata and confidence logic.

    中文
    ----
    多种 step-label pair mode 最终都落成同一个 canonical pair contract。
    这个 helper 把公共部分收敛起来，避免不同 mode 因为复制粘贴而慢慢漂移。
    """
    chosen_text = _join_steps_as_prefix([step for step, _ in step_labels], int(chosen_idx))
    rejected_text = _join_steps_as_prefix([step for step, _ in step_labels], int(rejected_idx))
    record = _build_record(
        source_tag=source_tag,
        domain_tag=domain_tag,
        prompt_text=prompt_text,
        chosen_text=chosen_text,
        rejected_text=rejected_text,
        pair_confidence=_step_label_pair_confidence(
            chosen_idx=int(chosen_idx),
            rejected_idx=int(rejected_idx),
            first_negative_idx=int(first_negative_idx),
        ),
        metadata={
            **base_metadata,
            "positive_step_index": int(chosen_idx),
            "negative_step_index": int(rejected_idx),
            "first_negative_index": int(first_negative_idx),
            "num_step_labels": int(len(step_labels)),
            "step_gap": int(rejected_idx) - int(chosen_idx),
            "chosen_is_last_good": bool(int(chosen_idx) == int(first_negative_idx) - 1),
            "negative_is_first_bad": bool(int(rejected_idx) == int(first_negative_idx)),
            "pair_build_mode": str(pair_build_mode),
            "pair_semantics": str(pair_semantics),
        },
        config=config,
    )
    return record


def _step_label_pair_confidence(
    *,
    chosen_idx: int,
    rejected_idx: int,
    first_negative_idx: int,
) -> float:
    """Heuristically lower confidence as one pair moves away from the local error edge.

    English
    -------
    This is not a calibrated probability.  It is only a weak prior encoding:
    "the farther we move away from the first-bad boundary, the more extra
    factors besides local error can leak into the pair."

    中文
    ----
    这不是校准后的概率，只是一个很弱的先验：
    “pair 离 first-bad 局部边界越远，越可能混入更多非局部错误因素。”
    """
    distance_from_last_good = max((int(first_negative_idx) - 1) - int(chosen_idx), 0)
    distance_from_first_bad = max(int(rejected_idx) - int(first_negative_idx), 0)
    confidence = 0.74
    confidence -= 0.04 * float(min(distance_from_last_good, 4))
    confidence -= 0.03 * float(min(distance_from_first_bad, 4))
    return float(max(confidence, 0.55))


def _convert_step_labels_to_legacy_pairs(
    *,
    prompt_text: str,
    step_labels: list[tuple[str, str]],
    source_tag: str,
    domain_tag: str,
    config: PairBuildConfig,
    base_metadata: dict[str, Any],
) -> list[ExternalPairRecord]:
    """Legacy nearest-boundary converter retained for forensic reproducibility.

    English
    -------
    This mode is kept so old experiments remain reproducible.  It should not be
    the default interpretation for new mainline science.

    The main problem is that chosen/rejected often terminate at different
    depths in the same trajectory, so the model may learn:
    1. progress depth,
    2. position in trajectory,
    3. or text length artifacts,
    rather than purely local error discrimination.

    中文
    ----
    这个模式保留主要是为了让旧实验还能复现，不该再作为新主线默认语义。

    它的核心问题是 chosen/rejected 往往结束在同一条轨迹的不同深度，所以模型
    可能学到的是：
    1. 推理走了多远，
    2. 当前在轨迹哪个位置，
    3. 文本长度差异，
    而不是真正的局部错误识别。
    """
    positives = [idx for idx, (_, label) in enumerate(step_labels) if label == "+"]
    negatives = [idx for idx, (_, label) in enumerate(step_labels) if label == "-"]
    if not positives or not negatives:
        return []

    rows: list[ExternalPairRecord] = []
    used_pairs: set[tuple[int, int]] = set()
    for pos_idx in positives:
        if len(rows) >= int(config.max_pairs_per_sample):
            break
        # Pick the nearest negative step rather than an arbitrary one so the
        # pair shares as much context as possible.
        # 这里优先选距离最近的错误步骤，而不是随便拿一个负样本，尽量让 pair 共享更多上下文。
        #
        # Caution: the two prefixes still end at different step indices, so
        # this is only a heuristic same-trajectory comparison, not a true
        # same-step local branch pair.
        # 注意：这里比较的仍然是两个不同步数终止的 prefix，所以它只是“同轨迹近邻对比”，
        # 不是严格意义上的“同一步分叉的局部好坏对比”。
        neg_idx = min(negatives, key=lambda n: abs(n - pos_idx))
        key = (pos_idx, neg_idx)
        if key in used_pairs:
            continue
        used_pairs.add(key)
        chosen_text = _join_steps_as_prefix([step for step, _ in step_labels], pos_idx)
        rejected_text = _join_steps_as_prefix([step for step, _ in step_labels], neg_idx)
        distance = abs(pos_idx - neg_idx)
        # This confidence is heuristic metadata, not a calibrated probability.
        # 当前 confidence 只是经验型启发式元数据，不是假设它等于“真实概率”。
        # Shorter step distance usually means the two prefixes diverge only locally and make a cleaner ranking pair.
        # 距离越近通常说明两个 prefix 只在局部步骤附近分叉，更适合作为高质量排序监督。
        confidence = 0.62 if distance <= 2 else 0.55
        record = _build_record(
            source_tag=source_tag,
            domain_tag=domain_tag,
            prompt_text=prompt_text,
            chosen_text=chosen_text,
            rejected_text=rejected_text,
            pair_confidence=confidence,
            metadata={
                **base_metadata,
                "positive_step_index": int(pos_idx),
                "negative_step_index": int(neg_idx),
                "num_step_labels": int(len(step_labels)),
                "pair_build_mode": "step_label_legacy_nearest_boundary",
                "pair_semantics": "same_trajectory_depth_mixed",
            },
            config=config,
        )
        if record is not None:
            rows.append(record)
    return rows


def _join_steps_as_prefix(steps: list[Any], step_index: int) -> str:
    """Join step list as one prefix ending at `step_index` (inclusive)."""
    if not steps:
        return ""
    end = min(max(int(step_index), 0), len(steps) - 1)
    fragments = [str(step).strip() for step in steps[: end + 1] if str(step).strip()]
    return "\n".join(fragments) + ("\n" if fragments else "")


def _build_record(
    *,
    source_tag: str,
    domain_tag: str,
    prompt_text: str,
    chosen_text: str,
    rejected_text: str,
    pair_confidence: float,
    metadata: dict[str, Any],
    config: PairBuildConfig,
) -> ExternalPairRecord | None:
    """Build one validated canonical pair record, or return None if filtered.

    English
    -------
    This is the final source-agnostic quality gate before a pair enters the
    repo-wide canonical contract.

    中文
    ----
    这是 source-specific pair 进入“全仓统一 canonical pair”之前的最后一道门。
    如果这里不过滤，后面的训练和评测都会把它当成正式监督样本。
    """
    flags = _compute_quality_flags(
        prompt_text=prompt_text,
        chosen_text=chosen_text,
        rejected_text=rejected_text,
    )
    if not _passes_quality_filter(flags=flags, config=config):
        # Reject low-quality pairs here so the adapter summary already reflects source quality.
        # 质量过滤要在 adapter 层提前做，这样 summary.json 才能直接反映外部 pair 源本身质量。
        return None
    pair_id = _stable_hash(
        source_tag,
        domain_tag,
        _normalize_space(prompt_text),
        _normalize_space(chosen_text),
        _normalize_space(rejected_text),
    )[:20]
    # Content-hash ids make deduplication and reruns stable even when file order changes.
    # 基于内容哈希的 pair_id 能让去重与复现不受文件顺序变化影响。
    metadata_payload = dict(metadata)
    metadata_payload.setdefault(
        "split_group_id",
        _derive_split_group_id(
            source_tag=str(source_tag).strip(),
            prompt_text=str(prompt_text),
            metadata=metadata_payload,
        ),
    )
    record = ExternalPairRecord(
        pair_id=pair_id,
        source_tag=str(source_tag).strip(),
        domain_tag=str(domain_tag).strip(),
        prompt_text=str(prompt_text),
        chosen_text=str(chosen_text),
        rejected_text=str(rejected_text),
        pair_confidence=float(min(max(pair_confidence, 0.0), 1.0)),
        quality_flags=flags,
        metadata=metadata_payload,
    )
    record.validate()
    return record


def _derive_split_group_id(
    *,
    source_tag: str,
    prompt_text: str,
    metadata: dict[str, Any],
) -> str:
    """Derive a stable source-sample grouping key for split hygiene.

    English
    -------
    Multiple canonical pairs can originate from one raw source sample.  When we
    want stricter train/validation hygiene, those derived pairs should move
    together. This helper normalizes that grouping key once at adapter time.

    中文
    ----
    一条原始样本可能衍生出多个 canonical pair。若后面启用更严格的切分卫生，
    这些 pair 应该整体移动。这里在 adapter 阶段先把分组键标准化出来。
    """
    explicit = str(metadata.get("split_group_id", "")).strip()
    if explicit:
        return explicit
    prefix_parts = [str(source_tag).strip()]
    for key in ("source_split", "task", "source_root", "source_file", "source_path"):
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            prefix_parts.append(f"{key}={text}")
    for key in ("source_idx", "source_row_index", "source_row_line", "source_line"):
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            prefix_parts.append(f"{key}={text}")
            return "|".join(prefix_parts)
    prompt_fingerprint = _stable_hash(str(source_tag).strip(), _normalize_space(prompt_text))[:16]
    prefix_parts.append(f"prompt={prompt_fingerprint}")
    return "|".join(prefix_parts)


def _compute_quality_flags(
    *,
    prompt_text: str,
    chosen_text: str,
    rejected_text: str,
) -> dict[str, Any]:
    chosen_norm = _normalize_space(chosen_text)
    rejected_norm = _normalize_space(rejected_text)
    chosen_chars = len(chosen_norm)
    rejected_chars = len(rejected_norm)
    min_len = min(chosen_chars, rejected_chars) if chosen_chars and rejected_chars else 0
    max_len = max(chosen_chars, rejected_chars) if chosen_chars or rejected_chars else 0
    ratio = float(max_len / max(min_len, 1))
    chosen_tokens = set(chosen_norm.split())
    rejected_tokens = set(rejected_norm.split())
    union = chosen_tokens | rejected_tokens
    overlap = (len(chosen_tokens & rejected_tokens) / len(union)) if union else 1.0
    # Keep interpretable quality features here and centralize the actual thresholding elsewhere.
    # 这里先保留可解释的质量特征，再由统一配置决定阈值，避免过滤逻辑散落多处。
    return {
        "chosen_chars": int(chosen_chars),
        "rejected_chars": int(rejected_chars),
        "length_ratio": float(ratio),
        "token_overlap_ratio": float(overlap),
        "non_empty": bool(chosen_chars > 0 and rejected_chars > 0),
        "distinct_pair": bool(chosen_norm != rejected_norm),
        "prompt_chars": int(len(_normalize_space(prompt_text))),
    }


def _passes_quality_filter(*, flags: dict[str, Any], config: PairBuildConfig) -> bool:
    if not bool(flags.get("non_empty", False)):
        return False
    if not bool(flags.get("distinct_pair", False)):
        return False
    if int(flags.get("chosen_chars", 0)) < int(config.min_chars):
        return False
    if int(flags.get("rejected_chars", 0)) < int(config.min_chars):
        return False
    if float(flags.get("length_ratio", 999.0)) > float(config.max_length_ratio):
        return False
    if float(flags.get("token_overlap_ratio", 1.0)) > float(config.max_token_overlap):
        return False
    return True


def _normalize_space(text: str) -> str:
    return " ".join(str(text).replace("\r", "\n").split())


def _stable_hash(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()
