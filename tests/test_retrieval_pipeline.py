"""Tests for the retrieval pipeline bugs:
1. Vector dimension mismatch (config vs model)
2. Score threshold applied after reranking deflates scores
3. Overly strict lexical grounding filter
"""
from __future__ import annotations

import types

import pytest

from app.models.domain import RetrievedChunk


# ---------------------------------------------------------------------------
# Bug #1: Vector dimension must match embedding model
# ---------------------------------------------------------------------------


def test_vector_size_matches_e5_base():
    """VECTOR_SIZE default must be 768 for intfloat/multilingual-e5-base."""
    from app.core.config import Settings

    s = Settings(
        embedding_model_name='intfloat/multilingual-e5-base',
        _env_file=None,
    )
    assert s.vector_size == 768, (
        f'VECTOR_SIZE={s.vector_size} but intfloat/multilingual-e5-base '
        f'produces 768-dim vectors. Collections created with the wrong '
        f'dimension will reject vectors or return no results.'
    )


def test_vector_size_env_example():
    """Ensure .env.example has the correct VECTOR_SIZE."""
    import pathlib

    env_example = pathlib.Path('/.env.example')
    # Try repo root locations
    for candidate in [
        pathlib.Path('.env.example'),
        pathlib.Path(__file__).parent.parent / '.env.example',
    ]:
        if candidate.exists():
            env_example = candidate
            break
    if not env_example.exists():
        pytest.skip('.env.example not found')

    content = env_example.read_text()
    assert 'VECTOR_SIZE=768' in content, (
        '.env.example should have VECTOR_SIZE=768 to match '
        'intfloat/multilingual-e5-base (768-dim embeddings)'
    )
    assert 'VECTOR_SIZE=1024' not in content, (
        '.env.example still has VECTOR_SIZE=1024 which mismatches '
        'the intfloat/multilingual-e5-base model (768-dim)'
    )


# ---------------------------------------------------------------------------
# Bug #2: Score threshold must be applied on RAW Qdrant scores,
#          not after reranking deflates them
# ---------------------------------------------------------------------------


def _make_chunk(score: float, text: str = 'upanishad self knowledge atman', doc: str = '108-upanishads') -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id='test:1:1:abc',
        doc=doc,
        work=doc,
        page=1,
        language='en',
        script='latin',
        text=text,
        text_type='mixed',
        score=score,
    )


def test_rerank_does_not_drop_above_threshold_chunks():
    """Chunks with raw score above min_similarity must survive reranking.

    _rerank() multiplies the raw score by 0.75, which used to push
    scores below the threshold applied afterward. Now retrieve()
    filters on raw scores before reranking.
    """
    from app.services.retriever import _rerank

    chunk = _make_chunk(score=0.50)
    result = _rerank([chunk], query='What do the Upanishads say about self-knowledge?', limit=6)
    assert len(result) == 1, 'Chunk with raw score 0.50 should survive reranking'
    # The reranked score may be lower than 0.50 (due to 0.75 weight),
    # but the chunk must still be present.


def test_retrieve_filters_on_raw_score(monkeypatch):
    """retrieve() should discard chunks BELOW min_score on their raw Qdrant
    score, before reranking deflates the score further."""
    from app.services import retriever

    fake_result = types.SimpleNamespace(
        id='fake-id',
        score=0.30,  # Below min_similarity=0.35
        payload={
            'chunk_id': 'test:1:1:abc',
            'doc': '108-upanishads',
            'work': '108-upanishads',
            'page': 1,
            'language': 'en',
            'script': 'latin',
            'text': 'upanishad self knowledge',
            'text_type': 'mixed',
        },
    )
    good_result = types.SimpleNamespace(
        id='good-id',
        score=0.50,  # Above min_similarity
        payload={
            'chunk_id': 'test:2:1:def',
            'doc': '108-upanishads',
            'work': '108-upanishads',
            'page': 2,
            'language': 'en',
            'script': 'latin',
            'text': 'upanishad self knowledge atman',
            'text_type': 'mixed',
        },
    )

    class FakeClient:
        def search(self, **kwargs):
            return [fake_result, good_result]

    class FakeEmbedder:
        def encode_query(self, text):
            return [0.1] * 768

    monkeypatch.setattr(retriever, 'get_qdrant_client', lambda: FakeClient())
    monkeypatch.setattr(retriever, 'get_embedding_provider', lambda: FakeEmbedder())

    chunks = retriever.retrieve(
        collection='scriptures_en',
        query='upanishad knowledge',
        k=6,
        allowed_docs=['108-upanishads'],
        min_score=0.35,
    )
    # Only the good_result (score=0.50) should remain
    assert len(chunks) == 1
    assert chunks[0].page == 2


# ---------------------------------------------------------------------------
# Bug #3: Lexical grounding must not be overly strict
# ---------------------------------------------------------------------------


def test_lexical_grounding_passes_with_one_match():
    """_has_lexical_grounding should pass if at least one query term
    appears in the chunks (relaxed from 40% / 2+ hits)."""
    from app.api.v1.query import _has_lexical_grounding

    # "upanishads" matches, but "say" and "self-knowledge" do not
    chunks = [_make_chunk(score=0.5, text='The Upanishads teach that Brahman is the ultimate reality.')]
    result = _has_lexical_grounding('What do the Upanishads say about self-knowledge?', chunks)
    assert result is True, (
        'Lexical grounding should pass if at least one content term '
        'matches (previously required 40%+ which is too strict for '
        'natural language queries about scripture)'
    )


def test_lexical_grounding_fails_with_zero_matches():
    """_has_lexical_grounding should fail if no query terms match."""
    from app.api.v1.query import _has_lexical_grounding

    chunks = [_make_chunk(score=0.5, text='The cosmos is vast and infinite.')]
    result = _has_lexical_grounding('What do the Upanishads say about self-knowledge?', chunks)
    assert result is False


def test_lexical_grounding_existential_always_passes():
    """Existential identity queries bypass lexical grounding."""
    from app.api.v1.query import _has_lexical_grounding

    chunks = [_make_chunk(score=0.5, text='Random text with no matching terms.')]
    result = _has_lexical_grounding('Who am I according to scripture?', chunks)
    assert result is True


# ---------------------------------------------------------------------------
# Integration: full query_scripture pipeline with mocked retrieval
# ---------------------------------------------------------------------------


def test_query_returns_citations_when_chunks_found(monkeypatch):
    """End-to-end: if retrieval returns good chunks, the response
    must contain citations (not the 'no passage found' default)."""
    from app.api.v1 import query as query_module
    from app.services import retriever
    from app.services import generator as gen_module
    from app.models.schemas import QueryRequest

    good_chunks = [
        _make_chunk(score=0.60, text='The Upanishads declare: Tat Tvam Asi – thou art that.'),
        _make_chunk(score=0.55, text='Self-knowledge (Atma Jnana) is the highest pursuit in the Upanishads.'),
    ]

    def fake_retrieve(collection, query, k, allowed_docs, min_score=0.0):
        return [c for c in good_chunks if c.score >= min_score]

    monkeypatch.setattr(query_module, 'retrieve', fake_retrieve)
    # Use mock LLM to avoid Ollama dependency
    monkeypatch.setattr(gen_module, 'generate_answer', gen_module.deterministic_answer)

    request = QueryRequest(
        mode='ask',
        query='What do the Upanishads say about self-knowledge?',
        sources={'upanishads': True, 'puranas': False, 'vedas': True},
        k=6,
        include_original=False,
    )
    response = query_module.query_scripture(request)
    assert len(response.citations) > 0, (
        'Query should return citations when retrieval finds relevant chunks'
    )
    assert 'don\'t find' not in response.answer.summary.lower()


def test_query_returns_no_answer_when_collection_empty(monkeypatch):
    """If retrieval returns nothing, the response should be the
    no-answer template (not a crash)."""
    from app.api.v1 import query as query_module
    from app.services import generator as gen_module
    from app.models.schemas import QueryRequest

    def fake_retrieve(collection, query, k, allowed_docs, min_score=0.0):
        return []

    monkeypatch.setattr(query_module, 'retrieve', fake_retrieve)
    monkeypatch.setattr(gen_module, 'generate_answer', gen_module.deterministic_answer)

    request = QueryRequest(
        mode='ask',
        query='What do the Upanishads say about self-knowledge?',
        sources={'upanishads': True, 'puranas': False, 'vedas': True},
        k=6,
        include_original=False,
    )
    response = query_module.query_scripture(request)
    assert len(response.citations) == 0
    assert 'don\'t find' in response.answer.summary.lower()
