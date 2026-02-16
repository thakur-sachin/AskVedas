from __future__ import annotations

import logging

from qdrant_client.http import models as qm

from app.core.config import get_settings
from app.vector.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)


PAYLOAD_INDEXES: tuple[tuple[str, qm.PayloadSchemaType], ...] = (
    ('doc', qm.PayloadSchemaType.KEYWORD),
    ('language', qm.PayloadSchemaType.KEYWORD),
    ('script', qm.PayloadSchemaType.KEYWORD),
    ('work', qm.PayloadSchemaType.KEYWORD),
    ('text_type', qm.PayloadSchemaType.KEYWORD),
    ('chunk_id', qm.PayloadSchemaType.KEYWORD),
    ('page', qm.PayloadSchemaType.INTEGER),
)


def collection_exists(name: str) -> bool:
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        return any(c.name == name for c in collections)
    except Exception:
        return False


def ensure_payload_indexes(collection_name: str) -> None:
    client = get_qdrant_client()
    for field_name, schema in PAYLOAD_INDEXES:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
        except Exception as exc:
            message = str(exc).lower()
            if 'already exists' in message:
                continue
            raise


def ensure_collections() -> None:
    settings = get_settings()
    client = get_qdrant_client()
    for collection_name in (settings.collection_en, settings.collection_hi):
        if not collection_exists(collection_name):
            logger.info('creating_collection name=%s', collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qm.VectorParams(size=settings.vector_size, distance=qm.Distance.COSINE),
            )
        ensure_payload_indexes(collection_name)


def get_collection_count(name: str) -> int:
    try:
        client = get_qdrant_client()
        info = client.get_collection(name)
        return int(info.points_count or 0)
    except Exception:
        return 0
