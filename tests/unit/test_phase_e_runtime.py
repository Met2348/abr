"""Unit tests for Phase E runtime helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ours.phase_e.runtime import (
    _maybe_resize_embeddings_for_tokenizer,
    build_max_memory_map,
    resolve_backbone_loader_family,
    resolve_checkpoint_path,
)


def test_resolve_checkpoint_path_best_prefers_best_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    best_path = run_dir / "best_value_head.pt"
    final_path = run_dir / "final_value_head.pt"
    best_path.write_text("best\n", encoding="utf-8")
    final_path.write_text("final\n", encoding="utf-8")
    manifest = {
        "output_files": {
            "best_value_head": str(best_path),
            "final_value_head": str(final_path),
        }
    }

    resolved = resolve_checkpoint_path(
        value_run_dir=run_dir,
        run_manifest=manifest,
        checkpoint_name="best",
    )

    assert resolved == best_path


def test_resolve_checkpoint_path_best_falls_back_to_final_when_best_missing(tmp_path: Path) -> None:
    """Audit the current silent fallback because it can skew experiment claims."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    final_path = run_dir / "final_value_head.pt"
    final_path.write_text("final\n", encoding="utf-8")
    manifest = {
        "output_files": {
            "best_value_head": str(run_dir / "missing_best_value_head.pt"),
            "final_value_head": str(final_path),
        }
    }

    resolved = resolve_checkpoint_path(
        value_run_dir=run_dir,
        run_manifest=manifest,
        checkpoint_name="best",
    )

    assert resolved == final_path


def test_build_max_memory_map_supports_gpu_and_cpu_caps() -> None:
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(device_count=lambda: 2),
    )

    payload = build_max_memory_map(
        torch_module=torch_stub,
        max_gpu_memory_gib=48,
        max_cpu_memory_gib=96,
    )

    assert payload == {0: "48GiB", 1: "48GiB", "cpu": "96GiB"}


def test_build_max_memory_map_returns_none_when_unset() -> None:
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(device_count=lambda: 1),
    )

    payload = build_max_memory_map(
        torch_module=torch_stub,
        max_gpu_memory_gib=None,
        max_cpu_memory_gib=None,
    )

    assert payload is None


def test_resolve_backbone_loader_family_detects_reward_model(monkeypatch) -> None:
    class _Config:
        architectures = ["Qwen2ForProcessRewardModel"]

    class _AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _Config()

    import sys

    transformers_stub = SimpleNamespace(AutoConfig=_AutoConfig)
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)

    family = resolve_backbone_loader_family(model_path="dummy")

    assert family == "process_reward_model"


def test_resolve_backbone_loader_family_defaults_to_causal_lm(monkeypatch) -> None:
    class _Config:
        architectures = ["Qwen2ForCausalLM"]

    class _AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _Config()

    import sys

    transformers_stub = SimpleNamespace(AutoConfig=_AutoConfig)
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)

    family = resolve_backbone_loader_family(model_path="dummy")

    assert family == "causal_lm"


def test_maybe_resize_embeddings_for_tokenizer_when_vocab_grows() -> None:
    class _Embeddings:
        num_embeddings = 10

    class _Backbone:
        def __init__(self) -> None:
            self.resize_calls: list[int] = []

        def get_input_embeddings(self):
            return _Embeddings()

        def resize_token_embeddings(self, size: int):
            self.resize_calls.append(int(size))

    class _Tokenizer:
        def __len__(self) -> int:
            return 11

    backbone = _Backbone()

    resized = _maybe_resize_embeddings_for_tokenizer(
        backbone=backbone,
        tokenizer=_Tokenizer(),
    )

    assert resized is True
    assert backbone.resize_calls == [11]
