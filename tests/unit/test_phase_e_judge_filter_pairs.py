"""Unit tests for scripts/phase_e_judge_filter_pairs.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_e_judge_filter_pairs.py"
    spec = importlib.util.spec_from_file_location("phase_e_judge_filter_pairs", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_judge_payload_accepts_valid_contract() -> None:
    module = _load_module()
    result = module._normalize_judge_payload(
        {
            "overall_verdict": "correct",
            "first_incorrect_step": None,
            "confidence": 0.85,
            "reason": "all steps valid so far",
        },
        raw_text='{"overall_verdict":"correct"}',
        parse_error=None,
    )
    assert result.overall_verdict == "correct"
    assert result.first_incorrect_step is None
    assert result.confidence == 0.85
    assert result.parse_error is None


def test_should_keep_audited_pair_requires_correct_vs_incorrect() -> None:
    module = _load_module()
    chosen = module.PrefixJudgeResult(
        overall_verdict="correct",
        first_incorrect_step=None,
        confidence=0.7,
        reason="",
        parse_error=None,
        raw_output="{}",
    )
    rejected = module.PrefixJudgeResult(
        overall_verdict="incorrect",
        first_incorrect_step=3,
        confidence=0.8,
        reason="",
        parse_error=None,
        raw_output="{}",
    )
    keep, reason = module._should_keep_audited_pair(
        chosen_result=chosen,
        rejected_result=rejected,
        min_confidence=0.5,
    )
    assert keep is True
    assert reason == "judge_agree"

    rejected_bad = module.PrefixJudgeResult(
        overall_verdict="correct",
        first_incorrect_step=None,
        confidence=0.8,
        reason="",
        parse_error=None,
        raw_output="{}",
    )
    keep, reason = module._should_keep_audited_pair(
        chosen_result=chosen,
        rejected_result=rejected_bad,
        min_confidence=0.5,
    )
    assert keep is False
    assert reason == "rejected_not_incorrect"


def test_extract_json_recovers_fenced_payload() -> None:
    module = _load_module()
    payload, error = module._extract_json(
        "```json\n{\"overall_verdict\":\"incorrect\",\"first_incorrect_step\":2,\"confidence\":0.9,\"reason\":\"bad\"}\n```"
    )
    assert error is None
    assert payload["overall_verdict"] == "incorrect"
    assert payload["first_incorrect_step"] == 2
