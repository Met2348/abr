"""Unit tests for `scripts/phase_c_score_prm_teacher.py`."""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import torch
import pytest


def _load_phase_d_teacher_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "phase_c_score_prm_teacher.py"
    spec = importlib.util.spec_from_file_location("phase_c_score_prm_teacher", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    """Minimal tokenizer stub for helper-function tests."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is False
        return f"{messages[0]['content']} || {messages[1]['content']} || {messages[2]['content']}"


class _FailingTemplateTokenizer:
    """Tokenizer stub that forces fallback path in `_build_teacher_chat_text`."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        raise RuntimeError("no template")


def test_split_reasoning_steps_normalizes_list_markers() -> None:
    module = _load_phase_d_teacher_module()
    steps = module._split_reasoning_steps("- alpha\n\n* beta\n gamma ")
    assert steps == ["alpha", "beta", "gamma"]


def test_render_assistant_step_text_appends_separator() -> None:
    module = _load_phase_d_teacher_module()
    rendered = module._render_assistant_step_text(
        steps=["a", "b"],
        step_separator_token="<extra_0>",
    )
    assert rendered == "a <extra_0>\nb <extra_0>"


def test_build_teacher_chat_text_uses_template_when_available() -> None:
    module = _load_phase_d_teacher_module()
    text = module._build_teacher_chat_text(
        tokenizer=_FakeTokenizer(),
        system_prompt="sys",
        question="q",
        steps=["r1", "r2"],
        step_separator_token="<extra_0>",
    )
    assert "sys || q ||" in text
    assert "<extra_0>" in text


def test_build_teacher_chat_text_falls_back_without_template() -> None:
    module = _load_phase_d_teacher_module()
    text = module._build_teacher_chat_text(
        tokenizer=_FailingTemplateTokenizer(),
        system_prompt="sys",
        question="q",
        steps=["r1"],
        step_separator_token="<extra_0>",
    )
    assert "[SYSTEM]" in text
    assert "[USER]" in text
    assert "[ASSISTANT]" in text


def test_extract_step_scores_from_probs_reads_separator_positions() -> None:
    module = _load_phase_d_teacher_module()
    probs = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        dtype=torch.float32,
    )
    input_ids = torch.tensor(
        [
            [9, 100, 9, 100],
            [100, 9, 9, 100],
        ],
        dtype=torch.long,
    )
    rows = module._extract_step_scores_from_probs(
        probs=probs,
        input_ids=input_ids,
        sep_token_id=100,
    )
    assert rows[0] == [pytest.approx(0.2), pytest.approx(0.4)]
    assert rows[1] == [pytest.approx(0.5), pytest.approx(0.8)]


def test_resolve_input_paths_allows_missing_manifest_when_enabled(tmp_path: Path) -> None:
    module = _load_phase_d_teacher_module()
    phase_c_dir = tmp_path / "phase_c_missing_manifest"
    phase_c_dir.mkdir(parents=True, exist_ok=True)
    (phase_c_dir / "prefixes.jsonl").write_text(
        '{"prefix_id":"p0","question":"q","prompt_text":"p","prefix_target_text":""}\n',
        encoding="utf-8",
    )
    (phase_c_dir / "corruptions.jsonl").write_text("", encoding="utf-8")

    args = SimpleNamespace(
        phase_c_dir=phase_c_dir,
        prefixes_jsonl=None,
        corruptions_jsonl=None,
    )
    resolved = module._resolve_input_paths(
        args=args,
        allow_missing_manifest=True,
    )
    assert resolved[0] == phase_c_dir.resolve()
    assert resolved[1] == (phase_c_dir / "prefixes.jsonl").resolve()
    assert resolved[2] == (phase_c_dir / "corruptions.jsonl").resolve()
    assert resolved[3] is None
