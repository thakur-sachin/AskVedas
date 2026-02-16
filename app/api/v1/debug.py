from __future__ import annotations

import re

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.schemas import QueryRequest
from app.services.embeddings import get_embedding_provider, validate_embedding_dimension
from app.services.library_service import DOC_REGISTRY, resolve_docs_from_sources
from app.services.retriever import retrieve, route_collection
from app.vector.collections import collection_exists, get_collection_count

router = APIRouter(prefix='/debug', tags=['debug'])

TOKEN_PATTERN = re.compile(r"[A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F\-']+")
STOP_TERMS = {
    'what', 'is', 'the', 'a', 'an', 'of', 'in', 'on', 'for', 'to',
    'about', 'please', 'tell', 'me', 'who', 'are', 'you', 'meaning', 'define',
}


@router.get('/status')
def debug_status() -> dict[str, object]:
    """Full diagnostic status: embeddings, collections, data counts."""
    settings = get_settings()
    embedding_info = validate_embedding_dimension()

    collections_info = {}
    for name in (settings.collection_en, settings.collection_hi):
        exists = collection_exists(name)
        count = get_collection_count(name) if exists else 0
        collections_info[name] = {'exists': exists, 'points_count': count}

    return {
        'embedding': embedding_info,
        'collections': collections_info,
        'config': {
            'vector_size': settings.vector_size,
            'embedding_model': settings.embedding_model_name,
            'embedding_provider': settings.embedding_provider,
            'min_similarity': settings.min_similarity,
            'llm_provider': settings.llm_provider,
        },
    }


@router.post('/retrieval')
def debug_retrieval(payload: QueryRequest) -> dict[str, object]:
    """Step-by-step retrieval pipeline debug for a given query."""
    settings = get_settings()

    # Step 1: Route collection
    detected_language, collection_used = route_collection(payload.lang_pref, payload.query)

    # Step 2: Resolve docs
    allowed_docs = resolve_docs_from_sources(
        upanishads=payload.sources.upanishads,
        puranas=payload.sources.puranas,
        vedas=payload.sources.vedas,
    )
    allowed_docs_for_collection = {
        doc for doc in allowed_docs if DOC_REGISTRY[doc].collection == collection_used
    }

    # Step 3: Check collection has data
    collection_count = get_collection_count(collection_used)

    # Step 4: Get embedding info
    embedder = get_embedding_provider()
    embedding_dim = embedder.get_dimension()

    # Step 5: Retrieve WITHOUT min_score filtering (raw results)
    k = min(payload.k, settings.retrieval_k_max)
    raw_chunks = retrieve(
        collection=collection_used,
        query=payload.query,
        k=k,
        allowed_docs=allowed_docs_for_collection,
        min_score=0.0,
    )

    # Step 6: Also retrieve WITH min_score filtering
    filtered_chunks = retrieve(
        collection=collection_used,
        query=payload.query,
        k=k,
        allowed_docs=allowed_docs_for_collection,
        min_score=settings.min_similarity,
    )

    # Step 7: Query term analysis
    terms = {t.lower() for t in TOKEN_PATTERN.findall(payload.query)}
    content_terms = {t for t in terms if t not in STOP_TERMS and len(t) >= 3}

    # Step 8: Lexical grounding analysis
    grounding_hits = {}
    for term in content_terms:
        found_in = []
        for i, c in enumerate(raw_chunks):
            if term.lower() in c.text.lower():
                found_in.append(i)
        grounding_hits[term] = found_in

    return {
        'routing': {
            'detected_language': detected_language,
            'collection_used': collection_used,
            'collection_points': collection_count,
        },
        'doc_filter': {
            'allowed_docs': sorted(allowed_docs_for_collection),
        },
        'embedding': {
            'provider': type(embedder).__name__,
            'dimension': embedding_dim,
            'configured_vector_size': settings.vector_size,
            'dimension_match': embedding_dim == settings.vector_size,
        },
        'raw_results': [
            {
                'rank': i + 1,
                'score': round(c.score, 4),
                'doc': c.doc,
                'work': c.work,
                'page': c.page,
                'text_preview': c.text[:120],
            }
            for i, c in enumerate(raw_chunks)
        ],
        'filtered_results_count': len(filtered_chunks),
        'min_similarity': settings.min_similarity,
        'query_terms': sorted(content_terms),
        'lexical_grounding': grounding_hits,
    }
