from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Callable
from typing import Iterable

from qdrant_client.http import models as qm

from app.core.config import get_settings
from app.ingest.chunking import chunk_hindi_ocr, detect_text_type
from app.ingest.cleaning import clean_text
from app.ingest.ocr import iter_ocr_pdf_pages
from app.services.embeddings import get_embedding_provider
from app.services.library_service import DOC_REGISTRY
from app.vector.upsert import delete_doc_chunks, upsert_chunks

logger = logging.getLogger(__name__)


def _chunk_id(doc: str, page: int, idx: int, text: str) -> str:
    digest = hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]
    return f'{doc}:{page}:{idx}:{digest}'


def _point_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def ingest_hindi_docs(doc_names: Iterable[str], force_rebuild: bool = False) -> int:
    return ingest_hindi_docs_with_progress(
        doc_names=doc_names,
        force_rebuild=force_rebuild,
    )


def ingest_hindi_docs_with_progress(
    doc_names: Iterable[str],
    force_rebuild: bool = False,
    start_pages: dict[str, int] | None = None,
    on_page_done: Callable[[str, int, int], None] | None = None,
    on_doc_done: Callable[[str], None] | None = None,
) -> int:
    settings = get_settings()
    embedder = get_embedding_provider()
    total = 0

    for doc_name in doc_names:
        meta = DOC_REGISTRY.get(doc_name)
        if not meta or meta.language != 'hi':
            continue
        start_page = (start_pages or {}).get(meta.doc, 1)
        logger.info('ingest_hi_start doc=%s path=%s', meta.doc, meta.path)
        if force_rebuild:
            delete_doc_chunks(meta.collection, meta.doc)

        points: list[qm.PointStruct] = []
        pages = iter_ocr_pdf_pages(
            meta.path,
            languages=settings.ocr_languages,
            dpi=settings.ocr_dpi,
            start_page=start_page,
        )
        for page_number, total_pages, raw_text in pages:
            cleaned = clean_text(raw_text)
            if not cleaned:
                if on_page_done:
                    on_page_done(meta.doc, page_number, total_pages)
                continue
            chunks = chunk_hindi_ocr(raw_text)
            cleaned_chunks = [clean_text(ch) for ch in chunks if clean_text(ch)]
            if not cleaned_chunks:
                if on_page_done:
                    on_page_done(meta.doc, page_number, total_pages)
                continue
            vectors = embedder.encode_passages(cleaned_chunks)
            for idx, (chunk_text, vector) in enumerate(zip(cleaned_chunks, vectors, strict=False), start=1):
                chunk_id = _chunk_id(meta.doc, page_number, idx, chunk_text)
                payload = {
                    'language': 'hi',
                    'script': 'devanagari',
                    'doc': meta.doc,
                    'work': meta.doc,
                    'page': page_number,
                    'text_type': detect_text_type(chunk_text),
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                }
                points.append(qm.PointStruct(id=_point_uuid(chunk_id), vector=vector, payload=payload))
                total += 1

            if len(points) >= 64:
                upsert_chunks(meta.collection, points)
                points = []
            if on_page_done:
                on_page_done(meta.doc, page_number, total_pages)

        if points:
            upsert_chunks(meta.collection, points)
        logger.info('ingest_hi_done doc=%s total=%s', meta.doc, total)
        if on_doc_done:
            on_doc_done(meta.doc)

    return total


if __name__ == '__main__':
    docs = ['rigved', 'yajurved', 'samved', 'arthved-part-1', 'atharva-2']
    count = ingest_hindi_docs(docs, force_rebuild=False)
    print(f'Ingested Hindi chunks: {count}')
