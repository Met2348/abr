"""Unit tests for Phase E runtime helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ours.phase_e.runtime import (
    _maybe_resize_embeddings_for_tokenizer,
    _is_retryable_cuda_capacity_error,
    build_max_memory_map,
    resolve_checkpoint_resolution,
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


def test_resolve_checkpoint_path_best_fails_by_default_when_best_missing(tmp_path: Path) -> None:
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

    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_path(
            value_run_dir=run_dir,
            run_manifest=manifest,
            checkpoint_name="best",
        )


def test_resolve_checkpoint_resolution_can_explicitly_fallback_to_final(tmp_path: Path) -> None:
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

    resolved = resolve_checkpoint_resolution(
        value_run_dir=run_dir,
        run_manifest=manifest,
        checkpoint_name="best",
        checkpoint_missing_policy="fallback_final",
    )

    assert resolved["resolved_checkpoint_name"] == "final"
    assert resolved["fallback_to_final"] is True
    assert resolved["resolved_checkpoint_path"] == str(final_path)


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


def test_is_retryable_cuda_capacity_error_accepts_async_capacity_signals() -> None:
    assert _is_retryable_cuda_capacity_error(RuntimeError("CUDA out of memory"))
    assert _is_retryable_cuda_capacity_error(RuntimeError("CUDA error: device-side assert triggered"))
    assert _is_retryable_cuda_capacity_error(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"))


def test_is_retryable_cuda_capacity_error_rejects_non_capacity_errors() -> None:
    assert not _is_retryable_cuda_capacity_error(RuntimeError("index out of bounds"))
