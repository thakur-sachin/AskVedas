from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.vector.qdrant_client import get_qdrant_client

router = APIRouter()


@router.get('/health')
def health() -> dict[str, str]:
    settings = get_settings()
    qdrant_status = 'ok'
    try:
        get_qdrant_client().get_collections()
    except Exception:
        qdrant_status = 'down'

    return {
        'status': 'ok' if qdrant_status == 'ok' else 'degraded',
        'qdrant': qdrant_status,
        'llm': settings.llm_provider,
        'version': settings.app_version,
    }
