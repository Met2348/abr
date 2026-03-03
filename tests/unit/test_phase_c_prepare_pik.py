"""Unit tests for `scripts/phase_c_prepare_pik_data.py`."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_prepare_pik_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_c_prepare_pik_data.py"
    spec = importlib.util.spec_from_file_location("phase_c_prepare_pik_data", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_prepare_pik_builds_contract_artifacts_without_rollouts(tmp_path: Path) -> None:
    module = _load_prepare_pik_module()
    input_path = tmp_path / "train.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "sample_id": "strategyqa:10",
                "dataset": "strategyqa",
                "split": "train",
                "prompt_text": "Question: Is water wet?\nAnswer:",
                "target_text": "Water is described as wet.\nFinal answer: yes",
                "answer": "yes",
                "question": "Is water wet?",
            }
        ],
    )
    output_root = tmp_path / "phase_c_pik_data"

    exit_code = module.main(
        [
            "--input-jsonl",
            str(input_path),
            "--output-root",
            str(output_root),
            "--run-name",
            "smoke",
            "--max-samples",
            "1",
            "--no-build-rollouts",
        ]
    )
    assert exit_code == 0

    run_dirs = list((output_root / "strategyqa").glob("smoke__*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "questions.jsonl").exists()
    assert (run_dir / "errors.jsonl").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.json").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_questions"] == 1
    assert summary["target_summary"] is None


def test_rollout_config_to_dict_is_json_serializable() -> None:
    module = _load_prepare_pik_module()
    cfg = module.RolloutConfig(
        model_path="assets/models/Qwen2.5-7B-Instruct",
        adapter_path=None,
        batch_size=256,
        rollout_count=20,
        max_new_tokens=96,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        seed=42,
        dtype="bfloat16",
        device_map="auto",
        require_cuda=True,
        oom_backoff=True,
        log_every=64,
    )
    payload = cfg.to_dict()
    assert payload["rollout_count"] == 20
    assert payload["batch_size"] == 256
