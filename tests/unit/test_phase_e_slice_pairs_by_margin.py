from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("_tmp_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_module(str(Path(__file__).resolve().parents[2] / 'scripts' / 'phase_e_slice_pairs_by_margin.py'))


def test_selection_key_lowest_abs_margin_prefers_small_absolute_gap() -> None:
    rows = [
        (0.40, MODULE._selection_key(margin=0.40, mode='lowest_abs_margin')),
        (-0.05, MODULE._selection_key(margin=-0.05, mode='lowest_abs_margin')),
        (0.10, MODULE._selection_key(margin=0.10, mode='lowest_abs_margin')),
    ]
    ordered = [margin for margin, _ in sorted(rows, key=lambda item: item[1])]
    assert ordered == [-0.05, 0.10, 0.40]


def test_selection_key_lowest_margin_prioritizes_contradictions() -> None:
    rows = [
        (0.01, MODULE._selection_key(margin=0.01, mode='lowest_margin')),
        (-0.20, MODULE._selection_key(margin=-0.20, mode='lowest_margin')),
        (-0.01, MODULE._selection_key(margin=-0.01, mode='lowest_margin')),
    ]
    ordered = [margin for margin, _ in sorted(rows, key=lambda item: item[1])]
    assert ordered == [-0.20, -0.01, 0.01]
