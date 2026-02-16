from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.models.schemas import QueryRequest, QueryResponse
from app.services.generator import NO_ANSWER, generate_answer, make_citations, to_original_snippets
from app.services.library_service import DOC_REGISTRY, resolve_docs_from_sources
from app.services.retriever import retrieve, retrieve_cross_collection_snippets, route_collection

router = APIRouter()
TOKEN_PATTERN = re.compile(r"[A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\-']+")
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


def _is_definition_query(query: str) -> bool:
    q = query.lower().strip()
    return any(p in q for p in ('what is', 'who am i', 'define', 'meaning of', 'kya hai', 'क्या है', 'अर्थ क्या'))


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


def _effective_min_similarity(query: str, base: float) -> float:
    if _is_existential_identity_query(query):
        return max(base, 0.38)
    if _is_definition_query(query):
        return max(base, 0.45)
    if len(re.findall(r'\w+', query)) <= 3:
        return max(base, 0.4)
    return base


def _query_terms(query: str) -> set[str]:
    terms = {t.lower() for t in TOKEN_PATTERN.findall(query)}
    return {t for t in terms if t not in STOP_TERMS and len(t) >= 3}


def _term_matches_text(term: str, text: str) -> bool:
    text_l = text.lower()
    variants = {term}
    if len(term) >= 5:
        variants.add(term[:-1])
    if term.endswith('a') and len(term) >= 5:
        variants.add(f'{term}n')
    return any(v in text_l for v in variants)


def _has_lexical_grounding(query: str, chunks: list) -> bool:
    if _is_existential_identity_query(query):
        return True
    terms = _query_terms(query)
    if not terms:
        return True
    hits = 0
    for term in terms:
        if any(_term_matches_text(term, c.text) for c in chunks):
            hits += 1
    ratio = hits / max(len(terms), 1)
    term_count = len(terms)
    if term_count >= 3:
        return hits >= 2 and ratio >= 0.4
    if term_count == 2:
        return hits >= 1 and ratio >= 0.5
    return hits >= 1


@router.post('/query', response_model=QueryResponse)
def query_scripture(payload: QueryRequest) -> QueryResponse:
    settings = get_settings()

    detected_language, collection_used = route_collection(payload.lang_pref, payload.query)
    allowed_docs = resolve_docs_from_sources(
        upanishads=payload.sources.upanishads,
        puranas=payload.sources.puranas,
        vedas=payload.sources.vedas,
    )
    allowed_docs = {doc for doc in allowed_docs if DOC_REGISTRY[doc].collection == collection_used}

    if not allowed_docs:
        raise HTTPException(status_code=400, detail='No documents selected for the current routing.')

    k = min(payload.k, settings.retrieval_k_max)
    chunks = retrieve(collection=collection_used, query=payload.query, k=k, allowed_docs=allowed_docs)
    min_similarity = _effective_min_similarity(payload.query, settings.min_similarity)
    chunks = [c for c in chunks if c.score >= min_similarity]
    if chunks and not _has_lexical_grounding(payload.query, chunks):
        chunks = []
    citations = make_citations(chunks, max_chars=settings.quote_max_chars)

    if not citations:
        answer = generate_answer(query=payload.query, mood=payload.mood, tone=payload.tone, citations=[])
        return QueryResponse(
            answer=answer,
            citations=[],
            original_snippets=[],
            routing={'detected_language': detected_language, 'collection_used': collection_used},
            safety={'disclaimer': NO_ANSWER},
        )

    answer = generate_answer(query=payload.query, mood=payload.mood, tone=payload.tone, citations=citations)
    original_snippets = []
    if payload.include_original and collection_used == settings.collection_en:
        hi_chunks = retrieve_cross_collection_snippets(payload.query, k=2)
        hi_chunks = [c for c in hi_chunks if c.score >= settings.cross_collection_similarity]
        original_snippets = to_original_snippets(hi_chunks, max_chars=settings.quote_max_chars)

    return QueryResponse(
        answer=answer,
        citations=citations,
        original_snippets=original_snippets,
        routing={'detected_language': detected_language, 'collection_used': collection_used},
        safety={
            'disclaimer': 'Guidance is scripture-grounded and informational, not a substitute for professional medical/legal advice.'
        },
    )
