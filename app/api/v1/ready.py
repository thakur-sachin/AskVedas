from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.vector.collections import collection_exists

router = APIRouter()


@router.get('/ready')
def ready() -> dict[str, object]:
    settings = get_settings()
    missing_files = settings.ensure_data_files_exist()
    checks = {
        settings.collection_en: collection_exists(settings.collection_en),
        settings.collection_hi: collection_exists(settings.collection_hi),
    }
    is_ready = all(checks.values()) and not missing_files
    return {
        'ready': is_ready,
        'collections': checks,
        'missing_files': missing_files,
        'embedding_provider': settings.embedding_provider,
        'llm_provider': settings.llm_provider,
    }
