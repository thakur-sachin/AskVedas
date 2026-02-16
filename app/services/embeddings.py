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


class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str, *, local_files_only: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, local_files_only=local_files_only)

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
        return HashEmbeddingProvider(settings.vector_size)
    if settings.embedding_provider == 'openai':
        if not settings.openai_api_key:
            raise RuntimeError('OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai')
        return OpenAIEmbeddingProvider(settings.openai_api_key)
    try:
        return SentenceTransformerProvider(
            settings.embedding_model_name,
            local_files_only=settings.embedding_local_files_only,
        )
    except Exception as exc:
        if settings.embedding_fallback_provider == 'error':
            raise
        logger.warning(
            'sentence_transformers_unavailable fallback=hash model=%s reason=%s',
            settings.embedding_model_name,
            str(exc),
        )
        return HashEmbeddingProvider(settings.vector_size)
