from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Iterator

import fitz
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


def iter_ocr_pdf_pages(
    path: str,
    languages: str = 'hin+san',
    dpi: int = 240,
    start_page: int = 1,
) -> Iterator[tuple[int, int, str]]:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f'PDF not found: {path}')

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        safe_start = max(1, start_page)
        for i in range(safe_start - 1, total_pages):
            page = doc.load_page(i)
            page_number = i + 1
            try:
                pix = page.get_pixmap(dpi=dpi)
                image = Image.open(BytesIO(pix.tobytes('png')))
                text = pytesseract.image_to_string(image, lang=languages)
                yield (page_number, total_pages, text)
            except Exception:
                logger.exception('ocr_failed page=%s path=%s', page_number, path)
                yield (page_number, total_pages, '')


def ocr_pdf_by_page(path: str, languages: str = 'hin+san', dpi: int = 240) -> list[tuple[int, str]]:
    return [(page, text) for page, _, text in iter_ocr_pdf_pages(path, languages=languages, dpi=dpi, start_page=1)]
