#!/usr/bin/env python3
"""Download paper PDFs referenced by repo docs into `docs/relatedPapers/`.

这个脚本存在的原因：
1. 仓库里的研究文档会持续追加新论文链接；
2. 只在 markdown 里放 URL 不利于长期归档、离线查阅和复现实验背景；
3. 需要一个统一入口，把 docs 中提到但本地还没有的论文下载到 `docs/relatedPapers/`。

This script exists because:
1. research notes keep accumulating paper links,
2. leaving them only as URLs makes archival and offline review weak,
3. we need one repeatable entry point that downloads referenced papers into
   `docs/relatedPapers/`.

核心行为：
1. 扫描 `docs/**/*.md` 以及可选的根目录 `README.md`；
2. 识别常见论文链接（arXiv / ACL / OpenReview / direct PDF）；
3. 解析出 PDF URL；
4. 下载缺失论文，并维护一个 `index.json` 清单。

Key behavior:
1. scan `docs/**/*.md` and optionally root `README.md`,
2. detect common paper links (arXiv / ACL / OpenReview / direct PDF),
3. resolve them to PDF URLs,
4. download missing papers and maintain an `index.json` manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


URL_PATTERN = re.compile(r"https?://[^\s<>()\]`\"']+")
ARXIV_ID_PATTERN = re.compile(r"(?P<id>\d{4}\.\d{4,5})(v\d+)?")
PMLR_CITATION_PDF_PATTERN = re.compile(r'citation_pdf_url" content="(?P<url>[^"\s]+\.pdf(?:\?[^"\s]*)?)"')
PMLR_HREF_PDF_PATTERN = re.compile(r'href="(?P<url>[^"\s]+\.pdf(?:\?[^"\s]*)?)"')


@dataclass(slots=True)
class PaperEntry:
    """One normalized paper reference discovered in repo docs.

    一个从仓库文档里提取并规范化后的论文条目。
    """

    source_url: str
    pdf_url: str
    filename: str
    origin_paths: list[str]
    status: str
    note: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download related papers referenced by repo docs.")
    parser.add_argument("--docs-root", type=Path, default=Path("docs"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/relatedPapers"))
    parser.add_argument("--include-root-readme", action="store_true")
    parser.add_argument("--download", action="store_true", help="Actually download missing papers.")
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--max-files", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--run-name", default="related_papers_sync")
    return parser.parse_args(argv)


def _iter_markdown_paths(docs_root: Path, *, include_root_readme: bool) -> list[Path]:
    paths = sorted(docs_root.rglob("*.md"))
    if include_root_readme:
        root_readme = Path("README.md")
        if root_readme.exists():
            paths.append(root_readme)
    return paths


def _extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for raw in URL_PATTERN.findall(text):
        cleaned = raw.rstrip(").,;]>")
        if cleaned in seen:
            continue
        seen.add(cleaned)
        urls.append(cleaned)
    return urls


def _is_paperish_url(url: str) -> bool:
    lowered = url.lower()
    return any(
        token in lowered
        for token in (
            "arxiv.org/abs/",
            "arxiv.org/pdf/",
            "aclanthology.org/",
            "openreview.net/forum",
            "openreview.net/pdf",
            "proceedings.mlr.press/",
            "cdn.openai.com/",
        )
    )


def _extract_pmlr_pdf_url(html_text: str) -> str | None:
    """Extract the canonical PDF URL from a PMLR landing page.

    从 PMLR 论文落地页里提取站点声明的 PDF 链接。
    """

    for pattern in (PMLR_CITATION_PDF_PATTERN, PMLR_HREF_PDF_PATTERN):
        match = pattern.search(html_text)
        if match:
            return match.group("url")
    return None


def _resolve_pdf_url(url: str, *, timeout_sec: float | None = None) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path

    if "arxiv.org" in host:
        if "/pdf/" in path:
            match = ARXIV_ID_PATTERN.search(path)
            if match:
                return f"https://arxiv.org/pdf/{match.group('id')}.pdf"
            if path.endswith(".pdf"):
                return url
        if "/abs/" in path:
            match = ARXIV_ID_PATTERN.search(path)
            if match:
                return f"https://arxiv.org/pdf/{match.group('id')}.pdf"
        return None

    if "aclanthology.org" in host:
        if path.endswith(".pdf"):
            return url
        trimmed = path.rstrip("/")
        if not trimmed:
            return None
        slug = trimmed.split("/")[-1]
        return f"https://aclanthology.org/{slug}.pdf"

    if "openreview.net" in host:
        if "/pdf" in path:
            return url
        if "/forum" in path:
            query = parse_qs(parsed.query)
            paper_id = (query.get("id") or [""])[0].strip()
            if paper_id:
                return f"https://openreview.net/pdf?id={paper_id}"
        return None

    if "proceedings.mlr.press" in host:
        if path.endswith(".pdf"):
            return url
        article_path = path.rstrip("/")
        if article_path.endswith(".html"):
            if timeout_sec is not None:
                try:
                    landing_text = _http_get_bytes(url, timeout_sec=timeout_sec).decode("utf-8", errors="ignore")
                except (HTTPError, URLError, TimeoutError, UnicodeDecodeError, ValueError):
                    landing_text = ""
                resolved = _extract_pmlr_pdf_url(landing_text) if landing_text else None
                if resolved:
                    return resolved
            slug = article_path.split("/")[-1].removesuffix(".html")
            volume = article_path.split("/")[-2] if "/" in article_path.strip("/") else ""
            if volume and slug:
                return f"https://proceedings.mlr.press/{volume}/{slug}/{slug}.pdf"
        return None

    if "cdn.openai.com" in host and path.endswith(".pdf"):
        return url

    if path.endswith(".pdf"):
        return url
    return None


def _slugify_filename_from_url(url: str, pdf_url: str) -> str:
    parsed = urlparse(url)
    pdf_parsed = urlparse(pdf_url)

    arxiv_match = ARXIV_ID_PATTERN.search(url)
    if arxiv_match:
        return f"{arxiv_match.group('id')}.pdf"

    if "openreview.net" in parsed.netloc.lower():
        query = parse_qs(parsed.query)
        paper_id = (query.get("id") or ["openreview"])[0].strip() or "openreview"
        return f"openreview_{paper_id}.pdf"

    if "aclanthology.org" in parsed.netloc.lower():
        slug = pdf_parsed.path.rstrip("/").split("/")[-1] or "aclan_paper.pdf"
        if not slug.endswith(".pdf"):
            slug = f"{slug}.pdf"
        return slug

    if "proceedings.mlr.press" in parsed.netloc.lower() or "proceedings.mlr.press" in pdf_parsed.netloc.lower():
        basename = Path(pdf_parsed.path).name or "pmlr_paper.pdf"
        if basename.endswith(".pdf"):
            return basename
        return f"{basename}.pdf"

    basename = Path(pdf_parsed.path).name or "paper.pdf"
    if basename.endswith(".pdf"):
        return basename
    return f"{basename}.pdf"


def _build_existing_keys(output_dir: Path) -> set[str]:
    keys: set[str] = set()
    if not output_dir.exists():
        return keys
    for path in output_dir.glob("*.pdf"):
        keys.add(path.name)
        match = ARXIV_ID_PATTERN.search(path.name)
        if match:
            keys.add(f"{match.group('id')}.pdf")
    return keys


def _http_get_bytes(url: str, *, timeout_sec: float) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 (Codex related-papers sync)"})
    with urlopen(request, timeout=timeout_sec) as response:
        return response.read()


def _download_one(entry: PaperEntry, *, output_dir: Path, timeout_sec: float) -> tuple[str, str]:
    output_path = output_dir / entry.filename
    payload = _http_get_bytes(entry.pdf_url, timeout_sec=timeout_sec)
    if not payload.startswith(b"%PDF"):
        raise ValueError(f"Downloaded payload is not a PDF: {entry.pdf_url}")
    output_path.write_bytes(payload)
    sha256 = hashlib.sha256(payload).hexdigest()
    return str(output_path), sha256


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    docs_root = Path(args.docs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    origin_map: dict[str, set[str]] = {}
    for path in _iter_markdown_paths(docs_root, include_root_readme=bool(args.include_root_readme)):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for url in _extract_urls(text):
            if not _is_paperish_url(url):
                continue
            origin_map.setdefault(url, set()).add(str(path))

    existing_keys = _build_existing_keys(output_dir)
    entries: list[PaperEntry] = []
    for source_url, origin_paths in sorted(origin_map.items()):
        pdf_url = _resolve_pdf_url(source_url, timeout_sec=float(args.timeout_sec))
        if not pdf_url:
            entries.append(
                PaperEntry(
                    source_url=source_url,
                    pdf_url="",
                    filename="",
                    origin_paths=sorted(origin_paths),
                    status="unresolved",
                    note="Could not map URL to a direct PDF endpoint.",
                )
            )
            continue
        filename = _slugify_filename_from_url(source_url, pdf_url)
        status = "already_present" if filename in existing_keys else "pending"
        entries.append(
            PaperEntry(
                source_url=source_url,
                pdf_url=pdf_url,
                filename=filename,
                origin_paths=sorted(origin_paths),
                status=status,
            )
        )

    downloaded = 0
    skipped = 0
    failed = 0
    unresolved = 0
    max_files = int(args.max_files)

    for entry in entries:
        if entry.status == "unresolved":
            unresolved += 1
            continue
        if entry.status == "already_present":
            skipped += 1
            continue
        if not args.download:
            continue
        if max_files > 0 and downloaded >= max_files:
            break
        try:
            path_text, sha256 = _download_one(entry, output_dir=output_dir, timeout_sec=float(args.timeout_sec))
            entry.status = "downloaded"
            entry.note = f"path={path_text}; sha256={sha256}"
            downloaded += 1
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            entry.status = "failed"
            entry.note = str(exc)
            failed += 1
        time.sleep(float(args.sleep_sec))

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_name": str(args.run_name),
        "docs_root": str(docs_root),
        "output_dir": str(output_dir),
        "download_enabled": bool(args.download),
        "counts": {
            "total_urls": len(entries),
            "already_present": skipped,
            "downloaded": downloaded,
            "failed": failed,
            "unresolved": unresolved,
            "pending": sum(1 for item in entries if item.status == "pending"),
        },
        "entries": [asdict(entry) for entry in entries],
    }
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 88)
    print("Related Papers Sync")
    print("=" * 88)
    print(f"run_name          : {args.run_name}")
    print(f"total_urls        : {len(entries)}")
    print(f"already_present   : {skipped}")
    print(f"downloaded        : {downloaded}")
    print(f"failed            : {failed}")
    print(f"unresolved        : {unresolved}")
    print(f"index_path        : {index_path}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
