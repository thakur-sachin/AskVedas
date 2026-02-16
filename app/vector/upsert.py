from __future__ import annotations

import logging
import time
from typing import Iterable

from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import ResponseHandlingException

from app.vector.qdrant_client import get_qdrant_client

logger = logging.getLogger(__name__)

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0
MIN_SPLIT_BATCH = 16


def _is_timeout_error(exc: Exception) -> bool:
    return 'timeout' in str(exc).lower()


def _upsert_with_retry(collection: str, points: list[qm.PointStruct]) -> None:
    client = get_qdrant_client()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Avoid waiting for full write application on remote cloud nodes.
            client.upsert(collection_name=collection, points=points, wait=False)
            return
        except ResponseHandlingException as exc:
            if _is_timeout_error(exc) and attempt < MAX_RETRIES:
                sleep_s = RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    'upsert_timeout_retry collection=%s batch=%s attempt=%s sleep_s=%.1f',
                    collection,
                    len(points),
                    attempt,
                    sleep_s,
                )
                time.sleep(sleep_s)
                continue
            if _is_timeout_error(exc) and len(points) > MIN_SPLIT_BATCH:
                mid = len(points) // 2
                logger.warning(
                    'upsert_timeout_split collection=%s batch=%s left=%s right=%s',
                    collection,
                    len(points),
                    mid,
                    len(points) - mid,
                )
                _upsert_with_retry(collection, points[:mid])
                _upsert_with_retry(collection, points[mid:])
                return
            raise


def upsert_chunks(collection: str, points: Iterable[qm.PointStruct]) -> None:
    points_list = list(points)
    if not points_list:
        return
    _upsert_with_retry(collection, points_list)
    logger.info('upserted_points collection=%s count=%s', collection, len(points_list))


def delete_doc_chunks(collection: str, doc: str) -> None:
    client = get_qdrant_client()
    filter_selector = qm.FilterSelector(
        filter=qm.Filter(
            must=[
                qm.FieldCondition(
                    key='doc',
                    match=qm.MatchValue(value=doc),
                )
            ]
        )
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.delete(collection_name=collection, points_selector=filter_selector, wait=True)
            break
        except ResponseHandlingException as exc:
            if _is_timeout_error(exc) and attempt < MAX_RETRIES:
                sleep_s = RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    'delete_timeout_retry collection=%s doc=%s attempt=%s sleep_s=%.1f',
                    collection,
                    doc,
                    attempt,
                    sleep_s,
                )
                time.sleep(sleep_s)
                continue
            raise
    logger.info('deleted_doc_chunks collection=%s doc=%s', collection, doc)
