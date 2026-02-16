from __future__ import annotations

import hashlib
import logging
import math
from functools import lru_cache
from typing import Sequence

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    def encode_query(self, text: str) -> list[float]:
        raise NotImplementedError

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def get_dimension(self) -> int:
        vec = self.encode_query('dimension check')
        return len(vec)


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str, *, local_files_only: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, local_files_only=local_files_only)

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _prefix(self, text: str, is_query: bool) -> str:
        if 'e5' in self.model_name.lower():
            return f"{'query' if is_query else 'passage'}: {text}"
        return text

    def encode_query(self, text: str) -> list[float]:
        vec = self.model.encode(self._prefix(text, is_query=True), normalize_embeddings=True)
        return vec.tolist()

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        prepared = [self._prefix(t, is_query=False) for t in texts]
        vectors = self.model.encode(prepared, normalize_embeddings=True)
        return vectors.tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def encode_query(self, text: str) -> list[float]:
        raise RuntimeError('OpenAI embeddings adapter is not configured for this local MVP.')

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError('OpenAI embeddings adapter is not configured for this local MVP.')


class HashEmbeddingProvider(EmbeddingProvider):
    def __init__(self, vector_size: int) -> None:
        self.vector_size = vector_size

    def get_dimension(self) -> int:
        return self.vector_size

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.vector_size
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode('utf-8')).digest()
            idx = int.from_bytes(digest[:4], 'big') % self.vector_size
            sign = -1.0 if (digest[4] & 1) else 1.0
            weight = 1.0 + (digest[5] / 255.0)
            vec[idx] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def encode_query(self, text: str) -> list[float]:
        return self._embed(text)

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider == 'hash':
        logger.warning(
            'using_hash_embeddings vector_size=%s — hash embeddings do not capture '
            'semantic meaning; retrieval quality will be very poor',
            settings.vector_size,
        )
        return HashEmbeddingProvider(settings.vector_size)
    if settings.embedding_provider == 'openai':
        if not settings.openai_api_key:
            raise RuntimeError('OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai')
        return OpenAIEmbeddingProvider(settings.openai_api_key)
    try:
        provider = SentenceTransformerProvider(
            settings.embedding_model_name,
            local_files_only=settings.embedding_local_files_only,
        )
        logger.info(
            'embedding_provider_loaded model=%s dimension=%s',
            settings.embedding_model_name,
            provider.get_dimension(),
        )
        return provider
    except Exception as exc:
        if settings.embedding_fallback_provider == 'error':
            raise
        logger.warning(
            'sentence_transformers_unavailable fallback=hash model=%s reason=%s — '
            'hash embeddings do not capture semantic meaning; retrieval quality '
            'will be very poor',
            settings.embedding_model_name,
            str(exc),
        )
        return HashEmbeddingProvider(settings.vector_size)


def validate_embedding_dimension() -> dict[str, object]:
    settings = get_settings()
    provider = get_embedding_provider()
    actual_dim = provider.get_dimension()
    configured_dim = settings.vector_size
    is_hash = isinstance(provider, HashEmbeddingProvider) and settings.embedding_provider != 'hash'
    issues: list[str] = []
    if actual_dim != configured_dim:
        issues.append(
            f'Embedding dimension mismatch: model produces {actual_dim}-dim vectors '
            f'but VECTOR_SIZE={configured_dim}. Qdrant collections use {configured_dim} dims. '
            f'Set VECTOR_SIZE={actual_dim} in .env and recreate collections.'
        )
    if is_hash:
        issues.append(
            f'Using hash fallback embeddings because sentence_transformers model '
            f'"{settings.embedding_model_name}" failed to load. Hash embeddings '
            f'have no semantic meaning — retrieval will not work. '
            f'Download the model or set EMBEDDING_LOCAL_FILES_ONLY=false.'
        )
    return {
        'provider': type(provider).__name__,
        'model': settings.embedding_model_name,
        'actual_dimension': actual_dim,
        'configured_dimension': configured_dim,
        'dimension_match': actual_dim == configured_dim,
        'is_hash_fallback': is_hash,
        'issues': issues,
    }
