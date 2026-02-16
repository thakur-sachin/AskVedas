"""Tests for embedding provider validation and dimension checking."""
from __future__ import annotations

from app.services.embeddings import HashEmbeddingProvider, validate_embedding_dimension


def test_hash_provider_dimension():
    """HashEmbeddingProvider must produce vectors of the configured size."""
    provider = HashEmbeddingProvider(768)
    vec = provider.encode_query('test query')
    assert len(vec) == 768

    provider_1024 = HashEmbeddingProvider(1024)
    vec_1024 = provider_1024.encode_query('test query')
    assert len(vec_1024) == 1024


def test_hash_provider_get_dimension():
    provider = HashEmbeddingProvider(768)
    assert provider.get_dimension() == 768


def test_hash_vectors_are_normalized():
    """Hash vectors should be L2-normalized."""
    import math
    provider = HashEmbeddingProvider(768)
    vec = provider.encode_query('upanishads self knowledge')
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-6, f'Hash vector should be normalized, got norm={norm}'


def test_validate_embedding_dimension_reports_mismatch(monkeypatch):
    """validate_embedding_dimension should flag dimension mismatches."""
    from app.services import embeddings as emb_module
    from app.core import config as config_module

    class FakeSettings:
        vector_size = 1024
        embedding_provider = 'hash'
        embedding_model_name = 'intfloat/multilingual-e5-base'
        embedding_local_files_only = True
        embedding_fallback_provider = 'hash'

    fake_provider = HashEmbeddingProvider(768)

    monkeypatch.setattr(emb_module, 'get_settings', lambda: FakeSettings())
    # Clear the lru_cache so our monkeypatch takes effect
    emb_module.get_embedding_provider.cache_clear()
    monkeypatch.setattr(emb_module, 'get_embedding_provider', lambda: fake_provider)

    result = validate_embedding_dimension()
    assert result['actual_dimension'] == 768
    assert result['configured_dimension'] == 1024
    assert result['dimension_match'] is False
    assert len(result['issues']) > 0
    assert 'mismatch' in result['issues'][0].lower()
