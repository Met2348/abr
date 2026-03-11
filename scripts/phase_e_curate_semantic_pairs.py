#!/usr/bin/env python3
"""按语义桶重组已有 pair artifact，生成新的可复用训练集。 Curate a new reusable pair artifact from existing artifacts by semantic buckets.

这个脚本存在的原因是：
1. 现在仓库里已经积累了很多高质量 pair artifact，
2. 但新的 repair 往往不是“再回 raw dataset 重建一遍”，而是“按语义桶重新配平”，
3. 因此需要一个稳定、可复现、可配额控制的 curate 层。

This script exists because:
1. the repository already contains many useful pair artifacts,
2. but new repairs often need semantic rebalancing rather than raw-data rebuilding,
3. so we need one deterministic curation layer with explicit quotas.

控制流：
1. 读取多个已有 artifact 的 train/validation rows，
2. 按 `pair_semantics` 与 `source_tag` 过滤，
3. 依配额稳定采样，
4. 输出一个新的 canonical artifact 目录。

Control flow:
1. read train/validation rows from existing artifacts,
2. filter by `pair_semantics` and `source_tag`,
3. sample deterministically under explicit quotas,
4. write one new canonical artifact directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CurateSliceSpec:
    """一条语义切片合同。 One semantic curation slice contract.

    中文
    ----
    一条切片定义：
    1. 从哪一个 artifact 读，
    2. 保留哪些 pair semantics，
    3. 保留哪些 source tags，
    4. train/validation 各取多少。

    English
    -------
    One slice defines:
    1. which artifact to read from,
    2. which pair semantics to keep,
    3. which source tags to keep,
    4. how many rows to sample for train/validation.
    """

    label: str
    artifact_dir: Path
    semantic_filters: tuple[str, ...] | None
    source_filters: tuple[str, ...] | None
    train_cap: int | None
    val_cap: int | None

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 友好的切片摘要。 Return a JSON-friendly slice summary."""
        return {
            "label": str(self.label),
            "artifact_dir": str(self.artifact_dir),
            "semantic_filters": list(self.semantic_filters) if self.semantic_filters is not None else None,
            "source_filters": list(self.source_filters) if self.source_filters is not None else None,
            "train_cap": self.train_cap,
            "val_cap": self.val_cap,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Curate one new Phase E pair artifact from existing artifacts by semantic buckets."
    )
    parser.add_argument(
        "--slice",
        action="append",
        required=True,
        metavar="LABEL=ARTIFACT_DIR|SEMANTICS|SOURCES|TRAIN_CAP|VAL_CAP",
        help=(
            "Repeatable curation slice. Use `*` for all semantics or all sources. "
            "Example: ms_local=path/to/artifact|first_bad_fanout_prefix_ranking|math_shepherd|1600|160"
        ),
    )
    parser.add_argument("--run-name", default="phase_e_curated_semantic_pairs")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_pairs"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 CLI 并尽早拒绝明显无效的切片合同。 Parse CLI and fail fast on invalid slice contracts."""
    args = _build_parser().parse_args(argv)
    if not args.slice:
        raise ValueError("At least one --slice is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    slice_specs = [_parse_slice_spec(text) for text in args.slice]

    fingerprint = _stable_fingerprint(
        {
            "run_name": str(args.run_name),
            "seed": int(args.seed),
            "slices": [spec.to_dict() for spec in slice_specs],
        }
    )
    run_dir = Path(args.output_root) / f"{args.run_name}__{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []

    for spec in slice_specs:
        artifact_summary = _load_json(spec.artifact_dir / "summary.json")
        raw_train_rows = _load_jsonl(spec.artifact_dir / "train_pairs.jsonl")
        raw_val_rows = _load_jsonl(spec.artifact_dir / "validation_pairs.jsonl")
        filtered_train = _filter_rows(
            rows=raw_train_rows,
            semantic_filters=spec.semantic_filters,
            source_filters=spec.source_filters,
        )
        filtered_val = _filter_rows(
            rows=raw_val_rows,
            semantic_filters=spec.semantic_filters,
            source_filters=spec.source_filters,
        )
        selected_train = _sample_rows(
            rows=filtered_train,
            cap=spec.train_cap,
            seed=int(args.seed),
            namespace=f"{spec.label}|train",
        )
        selected_val = _sample_rows(
            rows=filtered_val,
            cap=spec.val_cap,
            seed=int(args.seed),
            namespace=f"{spec.label}|val",
        )
        tagged_train = _tag_rows(rows=selected_train, spec=spec)
        tagged_val = _tag_rows(rows=selected_val, spec=spec)
        train_rows.extend(tagged_train)
        val_rows.extend(tagged_val)
        slice_rows.append(
            {
                **spec.to_dict(),
                "artifact_train_rows": int(len(raw_train_rows)),
                "artifact_val_rows": int(len(raw_val_rows)),
                "filtered_train_rows": int(len(filtered_train)),
                "filtered_val_rows": int(len(filtered_val)),
                "selected_train_rows": int(len(tagged_train)),
                "selected_val_rows": int(len(tagged_val)),
                "artifact_summary_path": str(spec.artifact_dir / "summary.json"),
                "artifact_train_pair_semantics": dict(
                    ((artifact_summary.get("train_summary") or {}).get("by_pair_semantics") or {})
                ),
            }
        )

    train_path = run_dir / "train_pairs.jsonl"
    val_path = run_dir / "validation_pairs.jsonl"
    summary_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    manifest_path = run_dir / "manifest.json"

    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "seed": int(args.seed),
        "num_train_rows": int(len(train_rows)),
        "num_validation_rows": int(len(val_rows)),
        "train_pair_semantics": _count_key(
            rows=train_rows,
            getter=lambda row: str((row.get("metadata") or {}).get("pair_semantics", "unspecified")),
        ),
        "validation_pair_semantics": _count_key(
            rows=val_rows,
            getter=lambda row: str((row.get("metadata") or {}).get("pair_semantics", "unspecified")),
        ),
        "train_source_tags": _count_key(rows=train_rows, getter=lambda row: str(row.get("source_tag", ""))),
        "validation_source_tags": _count_key(rows=val_rows, getter=lambda row: str(row.get("source_tag", ""))),
        "slice_rows": slice_rows,
    }
    manifest = {
        "artifact_stage": "phase_e_curated_semantic_pairs_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": str(args.run_name),
        "seed": int(args.seed),
        "run_dir": str(run_dir),
        "slices": [spec.to_dict() for spec in slice_specs],
        "output_files": {
            "train_pairs": str(train_path),
            "validation_pairs": str(val_path),
            "summary": str(summary_path),
            "summary_md": str(summary_md_path),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_path.write_text(_render_summary_markdown(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 88)
    print("Phase E: Curate Semantic Pairs")
    print("=" * 88)
    print(f"run_dir            : {run_dir}")
    print(f"num_train_rows     : {len(train_rows)}")
    print(f"num_validation_rows: {len(val_rows)}")
    for row in slice_rows:
        print(
            f"{row['label']:>18}: "
            f"train={row['selected_train_rows']}/{row['filtered_train_rows']} "
            f"val={row['selected_val_rows']}/{row['filtered_val_rows']}"
        )
    print("=" * 88)
    return 0


def _parse_slice_spec(text: str) -> CurateSliceSpec:
    """解析一条切片字符串。 Parse one slice string.

    中文
    ----
    规范格式：
    `LABEL=ARTIFACT_DIR|SEMANTICS|SOURCES|TRAIN_CAP|VAL_CAP`

    其中：
    1. `SEMANTICS` 可用 `*` 表示全部，或用逗号分隔多值，
    2. `SOURCES` 可用 `*` 表示全部，或用逗号分隔多值，
    3. `TRAIN_CAP/VAL_CAP` 允许留空，表示不过滤配额。

    English
    -------
    Canonical format:
    `LABEL=ARTIFACT_DIR|SEMANTICS|SOURCES|TRAIN_CAP|VAL_CAP`
    """
    raw = str(text).strip()
    if "=" not in raw:
        raise ValueError(
            "--slice must look like LABEL=ARTIFACT_DIR|SEMANTICS|SOURCES|TRAIN_CAP|VAL_CAP"
        )
    label, rhs = raw.split("=", 1)
    parts = rhs.split("|")
    if len(parts) != 5:
        raise ValueError(
            "--slice must have exactly five pipe-separated fields after LABEL=: "
            "ARTIFACT_DIR|SEMANTICS|SOURCES|TRAIN_CAP|VAL_CAP"
        )
    artifact_dir = Path(parts[0]).expanduser()
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact dir not found: {artifact_dir}")
    return CurateSliceSpec(
        label=str(label).strip(),
        artifact_dir=artifact_dir,
        semantic_filters=_parse_filter_field(parts[1]),
        source_filters=_parse_filter_field(parts[2]),
        train_cap=_parse_optional_positive_int(parts[3], field_name="TRAIN_CAP"),
        val_cap=_parse_optional_positive_int(parts[4], field_name="VAL_CAP"),
    )


def _parse_filter_field(raw: str) -> tuple[str, ...] | None:
    """把 `*` 或逗号列表解析成过滤器。 Parse `*` or a CSV filter list."""
    text = str(raw).strip()
    if text in {"", "*"}:
        return None
    values = tuple(sorted({item.strip() for item in text.split(",") if item.strip()}))
    if not values:
        return None
    return values


def _parse_optional_positive_int(raw: str, *, field_name: str) -> int | None:
    """解析可选正整数。 Parse one optional positive integer."""
    text = str(raw).strip()
    if text == "":
        return None
    value = int(text)
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0 when provided")
    return value


def _filter_rows(
    *,
    rows: list[dict[str, Any]],
    semantic_filters: tuple[str, ...] | None,
    source_filters: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    """按语义与 source 过滤 rows。 Filter rows by semantics and source tags."""
    kept: list[dict[str, Any]] = []
    semantic_allow = set(semantic_filters or [])
    source_allow = set(source_filters or [])
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        pair_semantics = str(metadata.get("pair_semantics", "unspecified")).strip() or "unspecified"
        source_tag = str(row.get("source_tag", "")).strip()
        if semantic_filters is not None and pair_semantics not in semantic_allow:
            continue
        if source_filters is not None and source_tag not in source_allow:
            continue
        kept.append(row)
    return kept


def _sample_rows(
    *,
    rows: list[dict[str, Any]],
    cap: int | None,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    """稳定采样一组 rows。 Deterministically sample one row list."""
    if cap is None or len(rows) <= int(cap):
        return list(rows)
    rng = random.Random(f"{seed}|{namespace}")
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = sorted(indices[: int(cap)])
    return [rows[idx] for idx in selected]


def _tag_rows(*, rows: list[dict[str, Any]], spec: CurateSliceSpec) -> list[dict[str, Any]]:
    """为输出 rows 写入 curate 元信息。 Attach curation metadata to output rows."""
    tagged_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        payload = json.loads(json.dumps(row, ensure_ascii=False))
        payload["pair_id"] = f"{spec.label}::{payload.get('pair_id', idx)}"
        metadata = dict(payload.get("metadata") or {})
        metadata["artifact_mix_source_label"] = str(spec.label)
        metadata["artifact_mix_source_dir"] = str(spec.artifact_dir)
        metadata["curated_semantic_filters"] = (
            list(spec.semantic_filters) if spec.semantic_filters is not None else ["*"]
        )
        metadata["curated_source_filters"] = (
            list(spec.source_filters) if spec.source_filters is not None else ["*"]
        )
        payload["metadata"] = metadata
        tagged_rows.append(payload)
    return tagged_rows


def _count_key(*, rows: list[dict[str, Any]], getter: Any) -> dict[str, int]:
    """统计一个离散键的出现次数。 Count one discrete key over rows."""
    counts: dict[str, int] = {}
    for row in rows:
        key = str(getter(row)).strip() or "unspecified"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    """计算稳定短指纹。 Compute a stable short fingerprint."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _load_json(path: Path) -> dict[str, Any]:
    """读取一份 JSON 文件。 Load one JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取一份 JSONL 文件。 Load one JSONL file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出一份 JSONL 文件。 Write one JSONL file."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    """渲染一份简洁 Markdown 摘要。 Render one concise Markdown summary."""
    lines = [
        "# Phase E Curated Semantic Pair Artifact",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- seed: `{summary['seed']}`",
        f"- num_train_rows: `{summary['num_train_rows']}`",
        f"- num_validation_rows: `{summary['num_validation_rows']}`",
        "",
        "## Train Pair Semantics",
        "",
    ]
    for key, value in sorted((summary.get("train_pair_semantics") or {}).items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Slices",
            "",
            "| label | semantics | sources | filtered_train | selected_train | filtered_val | selected_val |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("slice_rows") or []:
        semantics = ",".join(row["semantic_filters"] or ["*"])
        sources = ",".join(row["source_filters"] or ["*"])
        lines.append(
            f"| {row['label']} | {semantics} | {sources} | "
            f"{row['filtered_train_rows']} | {row['selected_train_rows']} | "
            f"{row['filtered_val_rows']} | {row['selected_val_rows']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
