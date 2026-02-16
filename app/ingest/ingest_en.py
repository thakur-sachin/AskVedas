from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections.abc import Callable
from typing import Iterable

from qdrant_client.http import models as qm

from app.ingest.chunking import chunk_english, detect_text_type
from app.ingest.cleaning import clean_text
from app.ingest.pdf_extract import iter_text_by_page
from app.services.embeddings import get_embedding_provider
from app.services.library_service import DOC_REGISTRY
from app.vector.upsert import delete_doc_chunks, upsert_chunks

logger = logging.getLogger(__name__)


def _detect_work(page_text: str, fallback: str) -> str:
    lines = [ln.strip() for ln in page_text.split('\n') if ln.strip()]
    for line in lines[:10]:
        if re.search(r'upanishad', line, re.IGNORECASE):
            return line[:120]
    for line in lines[:5]:
        if len(line.split()) <= 8 and line.isupper():
            return line.title()[:120]
    return fallback


def _chunk_id(doc: str, page: int, idx: int, text: str) -> str:
    digest = hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]
    return f'{doc}:{page}:{idx}:{digest}'


def _point_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def ingest_english_docs(doc_names: Iterable[str], force_rebuild: bool = False) -> int:
    return ingest_english_docs_with_progress(
        doc_names=doc_names,
        force_rebuild=force_rebuild,
    )


def ingest_english_docs_with_progress(
    doc_names: Iterable[str],
    force_rebuild: bool = False,
    start_pages: dict[str, int] | None = None,
    on_page_done: Callable[[str, int, int], None] | None = None,
    on_doc_done: Callable[[str], None] | None = None,
) -> int:
    embedder = get_embedding_provider()
    total = 0

    for doc_name in doc_names:
        meta = DOC_REGISTRY.get(doc_name)
        if not meta or meta.language != 'en':
            continue
        start_page = (start_pages or {}).get(meta.doc, 1)
        logger.info('ingest_en_start doc=%s path=%s', meta.doc, meta.path)
        if force_rebuild:
            delete_doc_chunks(meta.collection, meta.doc)

        page_points: list[qm.PointStruct] = []
        for page_number, total_pages, raw_text in iter_text_by_page(meta.path, start_page=start_page):
            cleaned_page = clean_text(raw_text)
            if not cleaned_page:
                if on_page_done:
                    on_page_done(meta.doc, page_number, total_pages)
                continue
            work = _detect_work(raw_text, meta.doc)
            chunks = chunk_english(raw_text)
            cleaned_chunks = [clean_text(ch) for ch in chunks if clean_text(ch)]
            if not cleaned_chunks:
                if on_page_done:
                    on_page_done(meta.doc, page_number, total_pages)
                continue
            vectors = embedder.encode_passages(cleaned_chunks)
            for idx, (chunk_text, vector) in enumerate(zip(cleaned_chunks, vectors, strict=False), start=1):
                chunk_id = _chunk_id(meta.doc, page_number, idx, chunk_text)
                payload = {
                    'language': 'en',
                    'script': 'latin',
                    'doc': meta.doc,
                    'work': work,
                    'page': page_number,
                    'text_type': detect_text_type(chunk_text),
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                }
                page_points.append(qm.PointStruct(id=_point_uuid(chunk_id), vector=vector, payload=payload))
                total += 1

            if len(page_points) >= 64:
                upsert_chunks(meta.collection, page_points)
                page_points = []
            if on_page_done:
                on_page_done(meta.doc, page_number, total_pages)

        if page_points:
            upsert_chunks(meta.collection, page_points)
        logger.info('ingest_en_done doc=%s total=%s', meta.doc, total)
        if on_doc_done:
            on_doc_done(meta.doc)

    return total


if __name__ == '__main__':
    count = ingest_english_docs(['108-upanishads', '18-puranas'], force_rebuild=False)
    print(f'Ingested English chunks: {count}')
