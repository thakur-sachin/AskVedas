from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import BackgroundTasks
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import IngestCheckpoint, Job
from app.db.session import SessionLocal
from app.ingest.ingest_en import ingest_english_docs_with_progress
from app.ingest.ingest_hi_ocr import ingest_hindi_docs_with_progress
from app.models.schemas import IngestRequest, JobStatusResponse
from app.services.library_service import DOC_REGISTRY
from app.vector.collections import ensure_collections

logger = logging.getLogger(__name__)


def _update_job(job_id: str, *, status: str, message: str | None = None) -> None:
    with SessionLocal() as db:
        row = db.scalar(select(Job).where(Job.job_id == job_id))
        if not row:
            return
        row.status = status
        row.message = message
        row.updated_at = datetime.now(timezone.utc)
        db.add(row)
        db.commit()


def _set_checkpoint(doc: str, language: str, last_page: int, status: str) -> None:
    with SessionLocal() as db:
        row = db.scalar(select(IngestCheckpoint).where(IngestCheckpoint.doc == doc))
        if row is None:
            row = IngestCheckpoint(
                doc=doc,
                language=language,
                last_page=max(0, int(last_page)),
                status=status,
            )
        else:
            row.language = language
            row.last_page = max(row.last_page, int(last_page))
            row.status = status
            row.updated_at = datetime.now(timezone.utc)
        db.add(row)
        db.commit()


def _mark_checkpoint_completed(doc: str) -> None:
    with SessionLocal() as db:
        row = db.scalar(select(IngestCheckpoint).where(IngestCheckpoint.doc == doc))
        if not row:
            return
        row.status = 'completed'
        row.updated_at = datetime.now(timezone.utc)
        db.add(row)
        db.commit()


def _clear_checkpoints(docs: list[str]) -> None:
    with SessionLocal() as db:
        db.execute(delete(IngestCheckpoint).where(IngestCheckpoint.doc.in_(docs)))
        db.commit()


def _resume_plan(docs: list[str]) -> tuple[list[str], dict[str, int], list[str]]:
    with SessionLocal() as db:
        rows = db.scalars(select(IngestCheckpoint).where(IngestCheckpoint.doc.in_(docs))).all()
    checkpoint_map = {row.doc: row for row in rows}

    active_docs: list[str] = []
    skipped_docs: list[str] = []
    start_pages: dict[str, int] = {}
    for doc in docs:
        cp = checkpoint_map.get(doc)
        if cp and cp.status == 'completed':
            skipped_docs.append(doc)
            continue
        active_docs.append(doc)
        start_pages[doc] = (cp.last_page + 1) if cp else 1
    return active_docs, start_pages, skipped_docs


def _run_ingest_job(job_id: str, docs: list[str], force_rebuild: bool) -> None:
    _update_job(job_id, status='running', message='Ingestion started')
    try:
        ensure_collections()

        if force_rebuild:
            _clear_checkpoints(docs)
            active_docs = docs
            start_pages = {doc: 1 for doc in docs}
            skipped_docs: list[str] = []
        else:
            active_docs, start_pages, skipped_docs = _resume_plan(docs)

        if not active_docs:
            _update_job(job_id, status='completed', message='All selected docs are already ingested.')
            return

        if skipped_docs:
            _update_job(job_id, status='running', message=f'Skipping completed docs: {", ".join(skipped_docs)}')

        def _on_page_done(doc: str, page: int, total_pages: int) -> None:
            if page % 10 != 0 and page != total_pages:
                return
            meta = DOC_REGISTRY[doc]
            _set_checkpoint(doc=doc, language=meta.language, last_page=page, status='running')
            _update_job(job_id, status='running', message=f'Ingesting {doc}: page {page}/{total_pages}')

        def _on_doc_done(doc: str) -> None:
            _mark_checkpoint_completed(doc)
            _update_job(job_id, status='running', message=f'Completed doc: {doc}')

        en_docs = [d for d in active_docs if DOC_REGISTRY[d].language == 'en']
        hi_docs = [d for d in active_docs if DOC_REGISTRY[d].language == 'hi']
        en_count = ingest_english_docs_with_progress(
            en_docs,
            force_rebuild=force_rebuild,
            start_pages=start_pages,
            on_page_done=_on_page_done,
            on_doc_done=_on_doc_done,
        )
        hi_count = ingest_hindi_docs_with_progress(
            hi_docs,
            force_rebuild=force_rebuild,
            start_pages=start_pages,
            on_page_done=_on_page_done,
            on_doc_done=_on_doc_done,
        )
        _update_job(
            job_id,
            status='completed',
            message=f'Ingestion finished. en={en_count}, hi={hi_count}, skipped={len(skipped_docs)}',
        )
    except Exception as exc:
        logger.exception('ingest_job_failed job_id=%s', job_id)
        _update_job(job_id, status='failed', message=str(exc))


class AdminService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def enqueue_ingest(self, payload: IngestRequest, background_tasks: BackgroundTasks) -> tuple[str, str]:
        docs = payload.docs or list(DOC_REGISTRY.keys())
        unknown = [d for d in docs if d not in DOC_REGISTRY]
        if unknown:
            raise ValueError(f'Unknown docs: {unknown}')

        job_id = uuid.uuid4().hex
        row = Job(
            job_id=job_id,
            status='queued',
            docs=json.dumps(docs, ensure_ascii=False),
            force_rebuild=1 if payload.force_rebuild else 0,
            message='Queued',
        )
        self.db.add(row)
        self.db.commit()

        background_tasks.add_task(_run_ingest_job, job_id, docs, payload.force_rebuild)
        return job_id, 'queued'

    def get_job(self, job_id: str) -> JobStatusResponse:
        row = self.db.scalar(select(Job).where(Job.job_id == job_id))
        if not row:
            raise ValueError('Job not found')
        docs = json.loads(row.docs)
        return JobStatusResponse(
            job_id=row.job_id,
            status=row.status,
            docs=docs,
            force_rebuild=bool(row.force_rebuild),
            message=row.message,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
