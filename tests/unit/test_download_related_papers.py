"""Unit tests for scripts/download_related_papers.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "download_related_papers.py"
    spec = importlib.util.spec_from_file_location("download_related_papers", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pmlr_article_url_is_detected_and_resolved() -> None:
    module = _load_module()
    url = "https://proceedings.mlr.press/v267/jia25f.html"

    assert module._is_paperish_url(url) is True
    assert module._resolve_pdf_url(url) == "https://proceedings.mlr.press/v267/jia25f/jia25f.pdf"


def test_pmlr_html_meta_fallback_extracts_real_pdf_url() -> None:
    module = _load_module()
    html = (
        '<meta name="citation_pdf_url" '
        'content="https://raw.githubusercontent.com/mlresearch/v267/main/assets/jia25f/jia25f.pdf">'
    )

    assert module._extract_pmlr_pdf_url(html) == "https://raw.githubusercontent.com/mlresearch/v267/main/assets/jia25f/jia25f.pdf"


def test_pmlr_slugifies_to_pdf_basename() -> None:
    module = _load_module()
    url = "https://proceedings.mlr.press/v70/guo17a.html"
    pdf_url = module._resolve_pdf_url(url)

    assert pdf_url == "https://proceedings.mlr.press/v70/guo17a/guo17a.pdf"
    assert module._slugify_filename_from_url(url, pdf_url) == "guo17a.pdf"
