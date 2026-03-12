from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / 'scripts'
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    script_path = scripts_dir / 'wait_for_summary_status.py'
    spec = importlib.util.spec_from_file_location('wait_for_summary_status', script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load module spec from {script_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_status_reads_markdown_status_line() -> None:
    module = _load_module()
    text = "# Summary\n\n- group_id: X\n- status: ok\n- suite_log_file: foo\n"
    assert module.parse_status(text) == 'ok'


def test_parse_status_strips_backticks() -> None:
    module = _load_module()
    text = "- status: `failed`\n"
    assert module.parse_status(text) == 'failed'
