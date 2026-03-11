from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("_tmp_hardslice_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_module(str(Path(__file__).resolve().parents[2] / 'scripts' / 'phase_e_prepare_processbench_hardslice_pairs.py'))


def test_bucket_error_position() -> None:
    assert MODULE._bucket_error_position(label=0, num_steps=6) == 'early'
    assert MODULE._bucket_error_position(label=2, num_steps=6) == 'mid'
    assert MODULE._bucket_error_position(label=5, num_steps=6) == 'late'


def test_round_robin_limit_balances_buckets() -> None:
    rows = [
        {'id': 'a1', 'error_bucket': 'early'},
        {'id': 'a2', 'error_bucket': 'early'},
        {'id': 'm1', 'error_bucket': 'mid'},
        {'id': 'l1', 'error_bucket': 'late'},
        {'id': 'l2', 'error_bucket': 'late'},
    ]
    selected = MODULE._round_robin_limit(rows, limit=4)
    assert [row['id'] for row in selected] == ['a1', 'l1', 'm1', 'a2']
