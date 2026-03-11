#!/usr/bin/env python3
"""Mix multiple existing Phase E pair artifacts into one new canonical artifact.

English
-------
The repository can already:
1. build canonical pair artifacts from raw datasets,
2. run targeted repairs such as fanout / grid / terminal-anchor.

What was still awkward is the next obvious experiment step:
combine already-built artifacts in controlled ratios without rewriting the raw
dataset adapters again.  This script fills that gap.

中文
----
仓库已经能：
1. 从原始数据构造 canonical pair artifact，
2. 跑 fanout / grid / terminal-anchor 这类定向修复。

但下一步更自然的实验其实是：
把这些已经构好的 artifact 按受控比例混起来，而不是再回到原始数据层重写一遍
adapter。本脚本就是做这个的。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mix multiple Phase E pair artifacts into one deterministic combined artifact."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=ARTIFACT_DIR[:TRAIN_CAP[:VAL_CAP[:WEIGHT]]]",
        help=(
            "One input artifact specification. Repeat this flag for multiple artifacts. "
            "Example: fanout=path/to/artifact:3072:384:0.75"
        ),
    )
    parser.add_argument("--run-name", default="phase_e_mixed_pairs")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/artifacts/phase_e_pairs"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = _build_parser().parse_args(argv)
    if not args.input:
        raise ValueError("At least one --input is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = [_parse_input_spec(text) for text in args.input]
    combined_train: list[dict[str, Any]] = []
    combined_val: list[dict[str, Any]] = []
    source_rows = []
    source_summaries = {}

    for spec in specs:
        train_path = spec["artifact_dir"] / "train_pairs.jsonl"
        val_path = spec["artifact_dir"] / "validation_pairs.jsonl"
        summary_path = spec["artifact_dir"] / "summary.json"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(f"Missing train/validation pairs in {spec['artifact_dir']}")
        train_rows = _load_jsonl(train_path)
        val_rows = _load_jsonl(val_path)
        source_summary = _load_json(summary_path) if summary_path.exists() else {}
        sampled_train = _sample_rows(
            rows=train_rows,
            cap=spec["train_cap"],
            seed=int(args.seed),
            namespace=f"{spec['label']}|train",
        )
        sampled_val = _sample_rows(
            rows=val_rows,
            cap=spec["val_cap"],
            seed=int(args.seed),
            namespace=f"{spec['label']}|val",
        )
        combined_train.extend(_tag_rows(sampled_train, spec))
        combined_val.extend(_tag_rows(sampled_val, spec))
        source_rows.append(
            {
                "label": spec["label"],
                "artifact_dir": str(spec["artifact_dir"]),
                "original_train_rows": int(len(train_rows)),
                "original_val_rows": int(len(val_rows)),
                "selected_train_rows": int(len(sampled_train)),
                "selected_val_rows": int(len(sampled_val)),
                "train_cap": spec["train_cap"],
                "val_cap": spec["val_cap"],
                "mix_weight": float(spec["mix_weight"]),
            }
        )
        source_summaries[spec["label"]] = source_summary

    fingerprint = _stable_fingerprint(
        {
            "run_name": str(args.run_name),
            "seed": int(args.seed),
            "inputs": [
                {
                    "label": row["label"],
                    "artifact_dir": row["artifact_dir"],
                    "selected_train_rows": row["selected_train_rows"],
                    "selected_val_rows": row["selected_val_rows"],
                    "mix_weight": row["mix_weight"],
                }
                for row in source_rows
            ],
        }
    )
    run_dir = Path(args.output_root) / f"{args.run_name}__{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_out = run_dir / "train_pairs.jsonl"
    val_out = run_dir / "validation_pairs.jsonl"
    summary_out = run_dir / "summary.json"
    summary_md_out = run_dir / "summary.md"
    manifest_out = run_dir / "manifest.json"

    _write_jsonl(train_out, combined_train)
    _write_jsonl(val_out, combined_val)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "num_train_rows": int(len(combined_train)),
        "num_validation_rows": int(len(combined_val)),
        "source_rows": source_rows,
        "source_summaries": source_summaries,
    }
    manifest = {
        "artifact_stage": "phase_e_mixed_pair_artifact_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_name": str(args.run_name),
        "seed": int(args.seed),
        "run_dir": str(run_dir),
        "inputs": source_rows,
        "output_files": {
            "train_pairs": str(train_out),
            "validation_pairs": str(val_out),
            "summary": str(summary_out),
            "summary_md": str(summary_md_out),
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md_out.write_text(_render_summary_markdown(summary), encoding="utf-8")

    print("=" * 88)
    print("Phase E: Mix Pair Artifacts")
    print("=" * 88)
    print(f"run_dir            : {run_dir}")
    print(f"num_train_rows     : {len(combined_train)}")
    print(f"num_validation_rows: {len(combined_val)}")
    for row in source_rows:
        print(
            f"{row['label']:>18}: w={float(row.get('mix_weight', 1.0)):.3f} "
            f"train={row['selected_train_rows']}/{row['original_train_rows']} "
            f"val={row['selected_val_rows']}/{row['original_val_rows']}"
        )
    print("=" * 88)
    return 0


def _parse_input_spec(text: str) -> dict[str, Any]:
    raw = str(text).strip()
    if "=" not in raw:
        raise ValueError(
            f"--input must look like LABEL=ARTIFACT_DIR[:TRAIN_CAP[:VAL_CAP[:WEIGHT]]], got {text!r}"
        )
    label, rhs = raw.split("=", 1)
    parts = rhs.split(":")
    artifact_dir = Path(parts[0]).expanduser()
    train_cap = int(parts[1]) if len(parts) >= 2 and parts[1].strip() else None
    val_cap = int(parts[2]) if len(parts) >= 3 and parts[2].strip() else None
    mix_weight = float(parts[3]) if len(parts) >= 4 and parts[3].strip() else 1.0
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact dir not found: {artifact_dir}")
    if float(mix_weight) <= 0.0:
        raise ValueError(f"Mix weight must be > 0, got {mix_weight!r}")
    return {
        "label": str(label).strip(),
        "artifact_dir": artifact_dir,
        "train_cap": train_cap,
        "val_cap": val_cap,
        "mix_weight": float(mix_weight),
    }


def _sample_rows(
    *,
    rows: list[dict[str, Any]],
    cap: int | None,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    if cap is None or len(rows) <= int(cap):
        return list(rows)
    rng = random.Random(f"{seed}|{namespace}")
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = sorted(indices[: int(cap)])
    return [rows[idx] for idx in selected]


def _tag_rows(rows: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    tagged = []
    for idx, row in enumerate(rows):
        payload = json.loads(json.dumps(row, ensure_ascii=False))
        payload["pair_id"] = f"{spec['label']}::{payload.get('pair_id', idx)}"
        metadata = dict(payload.get("metadata", {}))
        metadata["artifact_mix_source_label"] = spec["label"]
        metadata["artifact_mix_source_dir"] = str(spec["artifact_dir"])
        # 把混合权重写回最终 metadata，训练侧就能直接消费这一层 curate 决策。
        # Persist the curate-time mix weight into metadata so training can honor
        # the final supervision contract without extra source-specific logic.
        metadata["artifact_mix_weight"] = float(spec["mix_weight"])
        metadata["source_weight"] = float(metadata.get("source_weight", 1.0)) * float(spec["mix_weight"])
        payload["metadata"] = metadata
        tagged.append(payload)
    return tagged


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase E Mixed Pair Artifact",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- num_train_rows: `{summary['num_train_rows']}`",
        f"- num_validation_rows: `{summary['num_validation_rows']}`",
        "",
        "| label | mix_weight | original_train | selected_train | original_val | selected_val |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["source_rows"]:
        lines.append(
            f"| {row['label']} | {float(row.get('mix_weight', 1.0)):.3f} | "
            f"{row['original_train_rows']} | {row['selected_train_rows']} | "
            f"{row['original_val_rows']} | {row['selected_val_rows']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
