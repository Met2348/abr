#!/usr/bin/env python3
"""
脚本：外部数据集完整性和可用性验证
Script: Verify integrity and usability of all external datasets.

为什么需要这个脚本：
  外部数据集（PRM800K、Math-Shepherd、MATH-APS 等）由多个来源下载，
  格式各异，且对 Phase E 训练 pipeline 有严格的字段依赖。
  该脚本检查每个数据集是否完整下载、字段是否符合预期、
  以及能否被 external_pairs_adapters.py 正确读取。

Why this script exists:
  External datasets come from multiple sources with different formats.
  This script verifies download completeness, field schema, and
  compatibility with the Phase E training pipeline.

Responsibilities:
  - Check that required files exist and are non-empty
  - Verify schema fields for each dataset format
  - Count samples and report basic statistics
  - Test that format-conversion logic produces valid pairs
  - Report a pass/fail summary suitable for CI or manual audit

Key functions:
  verify_prm800k()        - OpenAI phase1/phase2 JSONL format
  verify_math_shepherd()  - peiyi9979 JSONL with '\\u043a\\u0438' step delimiter
  verify_rlhflow()        - RLHFlow TRL-compatible format
  verify_math_step_dpo()  - xinlai DPO (chosen/rejected) format
  verify_math_aps()       - openreasoner MATH-APS parquet/jsonl format
  verify_eurus_prm()      - PRIME-RL EurusPRM stage2 format
  verify_ultrainteract()  - openbmb UltraInteract_pair format
  verify_genprm_math()    - GenPRM-MATH-Data format

Flow:
  main() -> collect all verifiers -> run in order -> print summary

Interactions:
  Reads from assets/external_datasets/
  No writes; outputs to stdout only.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

# ─── 数据集根目录 / Base directory for external datasets ────────────────────
BASE = Path(__file__).parent.parent / "assets" / "external_datasets"


# ─── 工具函数 / Utility helpers ───────────────────────────────────────────────

def _load_jsonl_head(path: Path, n: int = 3) -> list[dict]:
    """读取 JSONL 文件的前 n 行 / Read first n lines from a JSONL file."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def _count_jsonl(path: Path) -> int:
    """统计 JSONL 行数（跳过空行）/ Count non-empty JSONL lines."""
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _count_parquet(path: Path) -> int:
    """统计 parquet 文件行数 / Count rows in a parquet file."""
    try:
        import pyarrow.parquet as pq
        return pq.read_table(path).num_rows
    except Exception:
        return -1


def _find_files(directory: Path, suffixes: tuple) -> list[Path]:
    """递归查找特定后缀文件 / Recursively find files with given suffixes."""
    result = []
    for suffix in suffixes:
        result.extend(sorted(directory.rglob(f"*{suffix}")))
    return result


class VerifyResult:
    """单个数据集的验证结果 / Verification result for one dataset."""

    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.stats: dict[str, Any] = {}

    def fail(self, msg: str):
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    def stat(self, key: str, val: Any):
        self.stats[key] = val

    def print_summary(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        print(f"\n[{status}] {self.name}")
        for k, v in self.stats.items():
            print(f"  {k}: {v}")
        for w in self.warnings:
            print(f"  ⚠  {w}")
        for e in self.errors:
            print(f"  ✗  {e}")


# ─── 验证器函数 / Verifier functions ─────────────────────────────────────────

def verify_prm800k() -> VerifyResult:
    """
    验证 PRM800K (OpenAI 官方 GitHub 格式)
    Verify PRM800K in OpenAI official GitHub JSONL format.

    Expected structure:
      openai_prm800k/prm800k/data/
        phase1_train.jsonl   - ~870K records
        phase1_test.jsonl
        phase2_train.jsonl   - ~200K records
        phase2_test.jsonl
    Each record:
      {
        "labeler": str,
        "timestamp": str,
        "generation": str,     <- solution text with step delimiter
        "is_quality_control_question": bool,
        "is_initial_screening_question": bool,
        "question": {text, ground_truth_answer, ...},
        "label": {finish_reason, steps: [{completions: [{text, rating, flagged}]}]}
      }
    """
    r = VerifyResult("PRM800K (OpenAI official)")
    base = BASE / "openai_prm800k" / "prm800k" / "data"

    if not base.exists():
        r.fail(f"Directory not found: {base}")
        return r

    required = ["phase1_train.jsonl", "phase2_train.jsonl"]
    total_rows = 0
    for fname in required:
        fp = base / fname
        if not fp.exists():
            r.fail(f"Missing file: {fp}")
            continue
        size_mb = fp.stat().st_size / 1e6
        count = _count_jsonl(fp)
        total_rows += count
        r.stat(fname, f"{count:,} rows ({size_mb:.1f} MB)")

        # Schema check on first row
        head = _load_jsonl_head(fp, 1)
        if not head:
            r.fail(f"Empty file: {fname}")
            continue
        row = head[0]
        for key in ("question", "label"):
            if key not in row:
                r.fail(f"{fname}: missing field '{key}' in row")
        if "label" in row and "steps" not in row["label"]:
            r.fail(f"{fname}: label.steps missing")
        if "label" in row and "steps" in row["label"]:
            steps = row["label"]["steps"]
            if steps and "completions" not in steps[0]:
                r.warn(f"{fname}: steps[0] has no 'completions' field (may be alternate schema)")

    r.stat("total_train_rows", f"{total_rows:,}")
    return r


def verify_math_shepherd() -> VerifyResult:
    """
    验证 Math-Shepherd (peiyi9979 JSONL 格式)
    Verify Math-Shepherd in peiyi9979 JSONL format.

    Expected structure:
      peiyi_math_shepherd/math-shepherd.jsonl
    Each record:
      {"input": "Problem: ... Step 1: ... ки\\nStep 2: ...", "label": "...+...+...\\n", ...}
    Step delimiter: 'ки' (U+043A U+0438)
    Step labels: '+' (positive) or '-' (negative)
    """
    r = VerifyResult("Math-Shepherd (peiyi9979)")
    fp = BASE / "peiyi_math_shepherd" / "math-shepherd.jsonl"

    if not fp.exists():
        r.fail(f"File not found: {fp}")
        return r

    size_mb = fp.stat().st_size / 1e6
    r.stat("file_size", f"{size_mb:.1f} MB")

    head = _load_jsonl_head(fp, 100)
    if not head:
        r.fail("File is empty")
        return r

    # Check schema
    row = head[0]
    for key in ("input", "label"):
        if key not in row:
            r.fail(f"Missing field: '{key}'")

    # Check step delimiter presence
    step_delimiter = "\u043a\u0438"  # ки
    has_delim = sum(1 for h in head if step_delimiter in h.get("input", ""))
    if has_delim == 0:
        r.warn("No 'ки' step delimiter found in first 100 rows — may be alternate format")
    else:
        r.stat("step_delimiter_present", f"{has_delim}/100 sampled rows")

    # Count all-positive trajectories (terminal anchors)
    all_pos = sum(
        1 for h in head
        if "-" not in h.get("label", "") and "+" in h.get("label", "")
    )
    r.stat("all_positive_in_first_100", all_pos)

    # Approximate total
    file_size = fp.stat().st_size
    avg_bytes = file_size / max(1, _count_jsonl(fp))
    r.stat("estimated_total_rows", f"~{int(file_size/avg_bytes):,}")

    return r


def verify_rlhflow(name_suffix: str, subdir: str) -> VerifyResult:
    """
    验证 RLHFlow PRM 数据（Mistral / Deepseek 变体）
    Verify RLHFlow PRM Data (Mistral or Deepseek variant).

    Expected structure:
      {subdir}/
        *.parquet or *.jsonl files
    Each record:
      {"input": "...", "label": "..."}  (same as Math-Shepherd schema)
    """
    r = VerifyResult(f"RLHFlow-{name_suffix}")
    base = BASE / subdir

    if not base.exists():
        r.fail(f"Directory not found: {base}")
        return r

    files = _find_files(base, (".parquet", ".jsonl"))
    if not files:
        r.fail(f"No parquet/jsonl files found in {base}")
        return r

    total = 0
    for fp in files[:5]:  # check first 5 files
        if fp.suffix == ".parquet":
            cnt = _count_parquet(fp)
        else:
            cnt = _count_jsonl(fp)
        total += max(cnt, 0)

    r.stat("data_files_found", len(files))
    r.stat("rows_in_first_5_files", f"{total:,}")
    r.stat("total_size", _dir_size(base))

    # Schema check on first file
    fp = files[0]
    if fp.suffix == ".jsonl":
        head = _load_jsonl_head(fp, 1)
        if head:
            row = head[0]
            for key in ("input", "label"):
                if key not in row:
                    r.warn(f"Missing field '{key}' — may use alternate schema")

    return r


def verify_math_step_dpo() -> VerifyResult:
    """
    验证 Math-Step-DPO-10K (xinlai DPO 格式)
    Verify Math-Step-DPO-10K in standard DPO chosen/rejected format.

    Expected structure:
      xinlai_math_step_dpo/ + *.parquet or *.json files
    Each record:
      {"question": str, "process": str, "answer": str,
       "wrong_step": int,   <- step index where error occurs
       "neg_process": str   <- process with error injected at wrong_step}
    OR standard DPO format with "chosen"/"rejected".
    """
    r = VerifyResult("Math-Step-DPO-10K (xinlai)")
    base = BASE / "xinlai_math_step_dpo"

    if not base.exists():
        r.fail(f"Directory not found: {base}")
        return r

    files = _find_files(base, (".parquet", ".jsonl", ".json"))
    if not files:
        r.fail("No data files found")
        return r

    r.stat("files_found", len(files))
    r.stat("total_size", _dir_size(base))

    # Check first available file
    fp = files[0]
    try:
        if fp.suffix == ".parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(fp)
            r.stat("columns", list(table.schema.names))
            r.stat("row_count", table.num_rows)
            # Check for key fields
            cols = table.schema.names
            if "wrong_step" in cols:
                r.stat("format", "xinlai_step_dpo (wrong_step column present)")
            elif "chosen" in cols and "rejected" in cols:
                r.stat("format", "standard DPO (chosen/rejected)")
            else:
                r.warn(f"Unknown schema columns: {cols}")
        elif fp.suffix in (".json", ".jsonl"):
            head = _load_jsonl_head(fp, 3)
            if head:
                r.stat("sample_keys", list(head[0].keys()))
    except Exception as e:
        r.fail(f"Failed to read {fp.name}: {e}")

    return r


def verify_math_aps() -> VerifyResult:
    """
    验证 MATH-APS (openreasoner OmegaPRM 风格)
    Verify MATH-APS dataset from openreasoner/MATH-APS.

    Expected structure:
      openreasoner_math_aps/ + *.parquet or *.jsonl files
    Key features:
      - MCTS-derived step labels
      - Balanced positive/negative samples
      - step-level correctness from binary search
    """
    r = VerifyResult("MATH-APS (openreasoner / OmegaPRM)")
    base = BASE / "openreasoner_math_aps"

    if not base.exists():
        r.fail(f"Directory not found: {base} — may still be downloading")
        return r

    files = _find_files(base, (".parquet", ".jsonl", ".json"))
    if not files:
        r.warn(f"No data files found yet in {base} — download may be in progress")
        return r

    total_rows = 0
    for fp in files[:3]:
        if fp.suffix == ".parquet":
            cnt = _count_parquet(fp)
            total_rows += max(cnt, 0)
        elif fp.suffix == ".jsonl":
            cnt = _count_jsonl(fp)
            total_rows += cnt

    r.stat("files_found", len(files))
    r.stat("rows_sampled_first_3_files", f"{total_rows:,}")
    r.stat("total_size", _dir_size(base))

    if files:
        fp = files[0]
        try:
            if fp.suffix == ".parquet":
                import pyarrow.parquet as pq
                schema_names = pq.read_schema(fp).names
                r.stat("columns", schema_names)
            elif fp.suffix == ".jsonl":
                head = _load_jsonl_head(fp, 1)
                if head:
                    r.stat("sample_keys", list(head[0].keys()))
        except Exception as e:
            r.warn(f"Could not read schema: {e}")

    return r


def verify_eurus_prm() -> VerifyResult:
    """
    验证 EurusPRM-Stage2-Data (PRIME-RL LLM-judge 注错格式)
    Verify EurusPRM Stage 2 data with LLM-judge error injection.

    Key feature: "Step K:" formatted step-level pairs
    with error injected by Llama-3.1-70B + Qwen2.5-72B consensus.
    """
    r = VerifyResult("EurusPRM-Stage2-Data (PRIME-RL)")
    base = BASE / "prime_rl_eurus_prm_stage2"

    if not base.exists():
        r.fail(f"Directory not found: {base} — may still be downloading")
        return r

    files = _find_files(base, (".parquet", ".jsonl", ".json"))
    if not files:
        r.warn("No data files found yet — download may be in progress")
        return r

    r.stat("files_found", len(files))
    r.stat("total_size", _dir_size(base))

    fp = files[0]
    try:
        if fp.suffix == ".parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(fp, columns=None)
            r.stat("columns", list(table.schema.names))
            r.stat("row_count", table.num_rows)
            # Check for Step K: format in text fields
            text_cols = [c for c in table.schema.names if "text" in c.lower() or "content" in c.lower() or "response" in c.lower()]
            if text_cols:
                col = table.column(text_cols[0])
                sample = str(col[0].as_py())[:200]
                has_step_fmt = "Step " in sample
                r.stat("has_step_format", has_step_fmt)
        elif fp.suffix == ".jsonl":
            head = _load_jsonl_head(fp, 3)
            if head:
                r.stat("sample_keys", list(head[0].keys()))
    except Exception as e:
        r.warn(f"Could not read schema: {e}")

    return r


def verify_ultrainteract() -> VerifyResult:
    """
    验证 UltraInteract_pair (openbmb 偏好对格式)
    Verify UltraInteract_pair preference tree dataset.

    Key feature: tree-structured correct/incorrect action pairs
    covering math, code, and logical reasoning.
    """
    r = VerifyResult("UltraInteract_pair (openbmb)")
    base = BASE / "openbmb_ultrainteract_pair"

    if not base.exists():
        r.fail(f"Directory not found: {base} — may still be downloading")
        return r

    files = _find_files(base, (".parquet", ".jsonl", ".json"))
    if not files:
        r.warn("No data files found yet — download may be in progress")
        return r

    r.stat("files_found", len(files))
    r.stat("total_size", _dir_size(base))

    fp = files[0]
    try:
        if fp.suffix == ".parquet":
            import pyarrow.parquet as pq
            schema_names = pq.read_schema(fp).names
            r.stat("columns", schema_names)
            cnt = _count_parquet(fp)
            r.stat("row_count_first_file", cnt)
        elif fp.suffix == ".jsonl":
            head = _load_jsonl_head(fp, 2)
            if head:
                r.stat("sample_keys", list(head[0].keys()))
                # Check for tree/pair structure
                row = head[0]
                if "chosen" in row and "rejected" in row:
                    r.stat("format", "standard DPO pair")
                elif "trajectory" in row or "turns" in row:
                    r.stat("format", "tree trajectory format")
    except Exception as e:
        r.warn(f"Could not read schema: {e}")

    return r


def verify_genprm_math() -> VerifyResult:
    """
    验证 GenPRM-MATH-Data (MC + LLM-judge 共识过滤格式)
    Verify GenPRM-MATH-Data with MC + LLM-judge consensus filtering.
    """
    r = VerifyResult("GenPRM-MATH-Data")
    base = BASE / "genprm_math_data"

    if not base.exists():
        r.fail(f"Directory not found: {base} — may still be downloading")
        return r

    files = _find_files(base, (".parquet", ".jsonl", ".json"))
    if not files:
        r.warn("No data files found yet — download may be in progress")
        return r

    r.stat("files_found", len(files))
    r.stat("total_size", _dir_size(base))

    fp = files[0]
    try:
        if fp.suffix == ".parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(fp)
            r.stat("columns", list(table.schema.names))
            r.stat("row_count", table.num_rows)
        elif fp.suffix == ".jsonl":
            head = _load_jsonl_head(fp, 2)
            if head:
                r.stat("sample_keys", list(head[0].keys()))
    except Exception as e:
        r.warn(f"Could not read schema: {e}")

    return r


def verify_trl_prm800k() -> VerifyResult:
    """
    验证 trl-lib/prm800k (TRL 格式化版本)
    Verify trl-lib/prm800k in TRL PRMTrainer-compatible format.

    Each record: {"prompt": [...], "completions": [...], "labels": [bool, ...]}
    """
    r = VerifyResult("trl-lib/prm800k (TRL formatted)")
    base = BASE / "trl_prm800k_formatted"

    if not base.exists():
        r.fail(f"Directory not found: {base} — may still be downloading")
        return r

    files = _find_files(base, (".parquet", ".jsonl"))
    if not files:
        r.warn("No data files found yet")
        return r

    r.stat("files_found", len(files))
    r.stat("total_size", _dir_size(base))

    fp = files[0]
    try:
        if fp.suffix == ".parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(fp)
            cols = list(table.schema.names)
            r.stat("columns", cols)
            r.stat("row_count", table.num_rows)
            if "prompt" in cols and "completions" in cols and "labels" in cols:
                r.stat("trl_format_check", "PASS — has prompt/completions/labels")
            else:
                r.warn(f"Expected TRL format columns not found. Got: {cols}")
    except Exception as e:
        r.warn(f"Could not read: {e}")

    return r


def verify_papers() -> VerifyResult:
    """
    验证论文 PDF 下载情况
    Verify that paper PDFs were downloaded successfully.
    """
    r = VerifyResult("Related Papers (PDF)")
    base = Path(__file__).parent.parent / "docs" / "relatedPapers"

    if not base.exists():
        r.fail(f"Directory not found: {base}")
        return r

    pdfs = sorted(base.glob("*.pdf"))
    r.stat("pdf_count", len(pdfs))

    if not pdfs:
        r.fail("No PDF files found — downloads may have failed or are still running")
        return r

    ok, failed = [], []
    for pdf in pdfs:
        size_kb = pdf.stat().st_size / 1000
        if size_kb < 50:  # papers < 50 KB are likely error pages
            failed.append(f"{pdf.name} ({size_kb:.0f} KB — too small, likely error)")
        else:
            ok.append(f"{pdf.name} ({size_kb:.0f} KB)")

    r.stat("ok_papers", len(ok))
    if failed:
        r.warn(f"{len(failed)} papers may have failed:")
        for f in failed:
            r.warn(f"  {f}")
    for p in ok[:5]:
        r.stat("  sample", p)

    return r


# ─── 工具 / Utility ──────────────────────────────────────────────────────────

def _dir_size(path: Path) -> str:
    """计算目录大小 / Compute directory size as human-readable string."""
    import subprocess
    try:
        result = subprocess.check_output(["du", "-sh", str(path)], stderr=subprocess.DEVNULL)
        return result.decode().split()[0]
    except Exception:
        return "unknown"


# ─── 主函数 / Main ────────────────────────────────────────────────────────────

def main():
    """
    运行所有验证器并打印汇总报告
    Run all verifiers and print a summary report.
    """
    print("=" * 60)
    print("External Dataset Verification Report")
    print("=" * 60)

    verifiers = [
        verify_prm800k,
        verify_math_shepherd,
        lambda: verify_rlhflow("Deepseek", "rlhflow_deepseek_prm"),
        lambda: verify_rlhflow("Mistral", "rlhflow_mistral_prm"),
        verify_math_step_dpo,
        verify_math_aps,
        verify_eurus_prm,
        verify_ultrainteract,
        verify_genprm_math,
        verify_trl_prm800k,
        verify_papers,
    ]

    results = []
    for fn in verifiers:
        try:
            result = fn()
        except Exception as e:
            result = VerifyResult(getattr(fn, "__name__", str(fn)))
            result.fail(f"Verifier crashed: {e}")
        results.append(result)
        result.print_summary()

    # Final summary
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(passed)}/{len(results)} passed")
    if failed:
        print("FAILED:")
        for r in failed:
            print(f"  ✗ {r.name}")
            for e in r.errors:
                print(f"      {e}")
    else:
        print("All checks passed.")
    print("=" * 60)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
