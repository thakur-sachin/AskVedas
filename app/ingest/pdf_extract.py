from __future__ import annotations

from pathlib import Path

import fitz


def get_pdf_page_count(path: str) -> int:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f'PDF not found: {path}')
    with fitz.open(pdf_path) as doc:
        return len(doc)


def iter_text_by_page(path: str, start_page: int = 1) -> list[tuple[int, int, str]]:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f'PDF not found: {path}')

    pages: list[tuple[int, int, str]] = []
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        safe_start = max(1, start_page)
        for i in range(safe_start - 1, total_pages):
            page = doc.load_page(i)
            pages.append((i + 1, total_pages, page.get_text('text')))
    return pages


def extract_text_by_page(path: str) -> list[tuple[int, str]]:
    return [(page, text) for page, _, text in iter_text_by_page(path, start_page=1)]
