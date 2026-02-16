from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.router import router as api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import init_db
from app.vector.collections import ensure_collections

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


@app.get('/')
def root() -> dict[str, str]:
    return {'name': settings.app_name, 'version': settings.app_version}
