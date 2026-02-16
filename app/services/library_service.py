from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from app.core.config import get_settings
from app.vector.collections import get_collection_count


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc: str
    path: str
    language: str
    script: str
    collection: str
    labels: tuple[str, ...]
    title: str


settings = get_settings()

DOC_REGISTRY: dict[str, DocMeta] = {
    '108-upanishads': DocMeta(
        doc='108-upanishads',
        path=settings.upanishads_path,
        language='en',
        script='latin',
        collection=settings.collection_en,
        labels=('upanishads', 'vedas'),
        title='108 Upanishads (English)',
    ),
    '18-puranas': DocMeta(
        doc='18-puranas',
        path=settings.puranas_path,
        language='en',
        script='latin',
        collection=settings.collection_en,
        labels=('puranas',),
        title='18 Puranas (English)',
    ),
    'rigved': DocMeta(
        doc='rigved',
        path=settings.rigved_path,
        language='hi',
        script='devanagari',
        collection=settings.collection_hi,
        labels=('vedas',),
        title='Rigved (Hindi/Sanskrit)',
    ),
    'yajurved': DocMeta(
        doc='yajurved',
        path=settings.yajurved_path,
        language='hi',
        script='devanagari',
        collection=settings.collection_hi,
        labels=('vedas',),
        title='Yajurved (Hindi/Sanskrit)',
    ),
    'samved': DocMeta(
        doc='samved',
        path=settings.samved_path,
        language='hi',
        script='devanagari',
        collection=settings.collection_hi,
        labels=('vedas',),
        title='Samved (Hindi/Sanskrit)',
    ),
    'arthved-part-1': DocMeta(
        doc='arthved-part-1',
        path=settings.arthved_path,
        language='hi',
        script='devanagari',
        collection=settings.collection_hi,
        labels=('vedas',),
        title='Arthved Part 1 (Hindi/Sanskrit)',
    ),
    'atharva-2': DocMeta(
        doc='atharva-2',
        path=settings.atharva2_path,
        language='hi',
        script='devanagari',
        collection=settings.collection_hi,
        labels=('vedas',),
        title='Atharva 2 (Hindi/Sanskrit)',
    ),
}


def all_docs() -> list[DocMeta]:
    return list(DOC_REGISTRY.values())


def get_docs_by_names(names: Iterable[str]) -> list[DocMeta]:
    return [DOC_REGISTRY[n] for n in names if n in DOC_REGISTRY]


def resolve_docs_from_sources(upanishads: bool, puranas: bool, vedas: bool) -> set[str]:
    active: set[str] = set()
    for doc, meta in DOC_REGISTRY.items():
        if upanishads and 'upanishads' in meta.labels:
            active.add(doc)
        if puranas and 'puranas' in meta.labels:
            active.add(doc)
        if vedas and 'vedas' in meta.labels:
            active.add(doc)
    return active


def library_summary() -> dict[str, object]:
    en_docs = [d.doc for d in all_docs() if d.collection == settings.collection_en]
    hi_docs = [d.doc for d in all_docs() if d.collection == settings.collection_hi]
    return {
        'collections': {
            settings.collection_en: {
                'docs': en_docs,
                'points_count': get_collection_count(settings.collection_en),
            },
            settings.collection_hi: {
                'docs': hi_docs,
                'points_count': get_collection_count(settings.collection_hi),
            },
        }
    }
