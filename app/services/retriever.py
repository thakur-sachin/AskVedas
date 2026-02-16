from __future__ import annotations

import re
from typing import Iterable

from qdrant_client.http import models as qm

from app.core.config import get_settings
from app.models.domain import RetrievedChunk
from app.services.embeddings import get_embedding_provider
from app.services.language import resolve_language
from app.services.library_service import DOC_REGISTRY
from app.vector.qdrant_client import get_qdrant_client

TOKEN_PATTERN = re.compile(r"[A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\-']+")
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
STOP_TERMS = {
    'what',
    'is',
    'the',
    'a',
    'an',
    'of',
    'in',
    'on',
    'for',
    'to',
    'about',
    'please',
    'tell',
    'me',
    'who',
    'are',
    'you',
    'meaning',
    'define',
}


def _is_existential_identity_query(query: str) -> bool:
    q = query.lower().strip()
    return any(
        p in q
        for p in (
            'who am i',
            'who i am',
            'what am i',
            'my true self',
            'real self',
            'true self',
            'मैं कौन हूँ',
            'मैं कौन हूं',
            'मैं क्या हूँ',
            'मैं क्या हूं',
        )
    )


def _expand_query_for_retrieval(query: str) -> str:
    if not _is_existential_identity_query(query):
        return query
    if DEVANAGARI_RE.search(query):
        return f'{query} आत्मा जीवात्मा ब्रह्म आत्मतत्व स्वभाव'
    return f'{query} atman jivatma brahman self soul true self'


def route_collection(lang_pref: str, query: str) -> tuple[str, str]:
    settings = get_settings()
    detected_language = resolve_language(lang_pref, query)
    if detected_language == 'hi':
        return detected_language, settings.collection_hi
    return detected_language, settings.collection_en


def _build_filter(docs: Iterable[str]) -> qm.Filter | None:
    doc_list = list(docs)
    if not doc_list:
        return None
    return qm.Filter(
        must=[
            qm.FieldCondition(
                key='doc',
                match=qm.MatchAny(any=doc_list),
            )
        ]
    )


def _is_definition_query(query: str) -> bool:
    q = query.lower().strip()
    patterns = (
        'what is',
        'who am i',
        'define',
        'meaning of',
        'what does',
        'kya hai',
        'क्या है',
        'अर्थ क्या',
    )
    return any(p in q for p in patterns)


def _query_terms(query: str) -> set[str]:
    terms = {t.lower() for t in TOKEN_PATTERN.findall(query)}
    filtered = {t for t in terms if t not in STOP_TERMS and len(t) >= 3}
    if filtered:
        return filtered
    if _is_existential_identity_query(query):
        if DEVANAGARI_RE.search(query):
            return {'आत्मा', 'जीवात्मा', 'ब्रह्म'}
        return {'atman', 'jivatma', 'brahman'}
    return filtered


def _text_terms(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_PATTERN.findall(text)}


def _overlap_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    text_l = text.lower()
    hits = 0
    for term in query_terms:
        variants = {term}
        if len(term) >= 5:
            variants.add(term[:-1])
        if term.endswith('a') and len(term) >= 5:
            variants.add(f'{term}n')
        if any(v in text_l for v in variants):
            hits += 1
    return hits / max(len(query_terms), 1)


def _rerank(chunks: list[RetrievedChunk], query: str, limit: int) -> list[RetrievedChunk]:
    if not chunks:
        return []
    q_terms = _query_terms(query)
    definition_query = _is_definition_query(query)

    rescored: list[tuple[float, float, RetrievedChunk]] = []
    for chunk in chunks:
        overlap = _overlap_score(q_terms, chunk.text)
        lexical_weight = 0.45 if definition_query else 0.22
        semantic_weight = 0.75
        title_boost = 0.0
        if q_terms:
            work_doc = f'{chunk.work} {chunk.doc}'.lower()
            if any(t in work_doc for t in q_terms):
                title_boost = 0.08
        combined = (chunk.score * semantic_weight) + (overlap * lexical_weight) + title_boost
        if definition_query and q_terms and overlap == 0.0:
            combined -= 0.08
        chunk.score = combined
        rescored.append((combined, overlap, chunk))

    # For definition-style prompts, strongly prefer chunks that explicitly mention query terms.
    if definition_query and q_terms and any(overlap > 0.0 for _, overlap, _ in rescored):
        rescored = [item for item in rescored if item[1] > 0.0]

    rescored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, _, chunk in rescored[:limit]]


def retrieve(
    collection: str,
    query: str,
    k: int,
    allowed_docs: Iterable[str],
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    client = get_qdrant_client()
    embedder = get_embedding_provider()
    retrieval_query = _expand_query_for_retrieval(query)
    query_vector = embedder.encode_query(retrieval_query)
    query_filter = _build_filter(allowed_docs)

    query_limit = max(k * 3, k + 4)
    if hasattr(client, 'search'):
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=query_limit,
            with_payload=True,
            with_vectors=False,
        )
    else:
        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=query_filter,
            limit=query_limit,
            with_payload=True,
            with_vectors=False,
        )
        results = response.points

    chunks: list[RetrievedChunk] = []
    for result in results:
        raw_score = float(result.score)
        if raw_score < min_score:
            continue
        payload = result.payload or {}
        chunks.append(
            RetrievedChunk(
                chunk_id=str(payload.get('chunk_id', result.id)),
                doc=str(payload.get('doc', 'unknown')),
                work=str(payload.get('work', payload.get('doc', 'unknown'))),
                page=int(payload.get('page', 1)),
                language=str(payload.get('language', 'en')),
                script=str(payload.get('script', 'latin')),
                text=str(payload.get('text', '')),
                text_type=str(payload.get('text_type', 'mixed')),
                score=raw_score,
            )
        )
    return _rerank(chunks, query=query, limit=k)


def retrieve_cross_collection_snippets(query: str, k: int = 2) -> list[RetrievedChunk]:
    settings = get_settings()
    chunks = retrieve(
        collection=settings.collection_hi,
        query=query,
        k=k,
        allowed_docs=[
            doc for doc, meta in DOC_REGISTRY.items() if meta.collection == settings.collection_hi
        ],
    )
    return chunks
