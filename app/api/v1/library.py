from __future__ import annotations

from fastapi import APIRouter

from app.services.library_service import DOC_REGISTRY, library_summary

router = APIRouter()


@router.get('/library')
def library() -> dict[str, object]:
    return library_summary()


@router.get('/library/docs')
def library_docs() -> dict[str, object]:
    docs = []
    for meta in DOC_REGISTRY.values():
        docs.append(
            {
                'doc': meta.doc,
                'title': meta.title,
                'path': meta.path,
                'language': meta.language,
                'script': meta.script,
                'collection': meta.collection,
                'labels': list(meta.labels),
            }
        )
    return {'docs': docs}
