from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.router import router as api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import init_db
from app.services.embeddings import validate_embedding_dimension
from app.vector.collections import ensure_collections, get_collection_count

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.include_router(api_router, prefix=settings.api_prefix)


@app.on_event('startup')
def startup() -> None:
    init_db()
    try:
        ensure_collections()
    except Exception:
        logger.exception('qdrant_not_ready_startup')
    try:
        result = validate_embedding_dimension()
        if result['issues']:
            for issue in result['issues']:
                logger.error('embedding_validation_issue: %s', issue)
        else:
            logger.info(
                'embedding_validation_ok provider=%s dimension=%s',
                result['provider'],
                result['actual_dimension'],
            )
        for coll_name in (settings.collection_en, settings.collection_hi):
            count = get_collection_count(coll_name)
            if count == 0:
                logger.warning(
                    'collection_empty name=%s — run ingestion via POST /api/v1/admin/ingest',
                    coll_name,
                )
            else:
                logger.info('collection_ready name=%s points=%s', coll_name, count)
    except Exception:
        logger.exception('startup_validation_failed')


@app.get('/')
def root() -> dict[str, str]:
    return {'name': settings.app_name, 'version': settings.app_version}
