"""Unit tests for scripts/phase_a_prepare.py helper behavior."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ours.data.schema import CanonicalSample
from ours.phase_a.splitting import SplitConfig


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_a_prepare.py"
    spec = importlib.util.spec_from_file_location("phase_a_prepare", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class _PreparedStub:
    """Minimal prepared-sample stub used by the unit test."""

    sample_id: str
    split: str
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "metadata": self.metadata,
        }


def test_prepare_one_dataset_uses_effective_official_split_from_loader(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()

    sample = CanonicalSample(
        id="gsm8k:test:0",
        dataset="gsm8k",
        question="Q",
        answer="A",
        metadata={
            "source_split": "test",
            "requested_split": "validation",
        },
    )

    monkeypatch.setattr(module, "load_dataset_canonical", lambda **_: [sample])

    def _build_prepared_sample(**kwargs):
        item = kwargs["sample"]
        return _PreparedStub(
            sample_id=item.id,
            split=str(kwargs["split"]),
            metadata=dict(kwargs.get("extra_metadata", {}) or {}),
        )

    monkeypatch.setattr(module, "build_prepared_sample", _build_prepared_sample)

    run_dir = tmp_path / "prepared"
    summary = module._prepare_one_dataset(
        dataset="gsm8k",
        source_split="validation",
        split_policy="official",
        limit=None,
        dataset_root=tmp_path / "datasets",
        cache_dir=tmp_path / "cache",
        dataset_kwargs={},
        target_style="answer_only",
        template_id="qa_direct",
        template_version="1.0.0",
        split_cfg=SplitConfig(train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, seed=42),
        run_dir=run_dir,
        run_spec={"dataset": "gsm8k"},
        run_fingerprint="abc123",
        resume=False,
        overwrite=False,
    )

    assert summary is not None
    assert summary["requested_source_split"] == "validation"
    assert summary["effective_source_splits"] == {"test": 1}
    assert summary["split_counts"] == {"train": 0, "validation": 0, "test": 1}

    test_rows = [
        json.loads(line)
        for line in (run_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(test_rows) == 1
    assert test_rows[0]["split"] == "test"
    assert test_rows[0]["metadata"]["source_split"] == "test"
    assert test_rows[0]["metadata"]["requested_source_split"] == "validation"

    validation_rows = [
        line for line in (run_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert validation_rows == []
