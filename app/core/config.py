from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SourceSelection(BaseModel):
    upanishads: bool = True
    puranas: bool = True
    vedas: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = 'Scripture RAG API'
    app_version: str = '0.1.0'
    api_prefix: str = '/api/v1'

    qdrant_host: str = 'qdrant'
    qdrant_port: int = 6333
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_timeout_seconds: float = 60.0

    collection_en: str = 'scriptures_en'
    collection_hi: str = 'scriptures_hi'
    vector_size: int = 768

    embedding_provider: Literal['sentence_transformers', 'openai', 'hash'] = 'sentence_transformers'
    embedding_model_name: str = 'intfloat/multilingual-e5-base'
    openai_api_key: str | None = None
    embedding_local_files_only: bool = True
    embedding_fallback_provider: Literal['hash', 'error'] = 'hash'

    llm_provider: Literal['ollama', 'mock'] = 'ollama'
    ollama_base_url: str = 'http://host.docker.internal:11434'
    ollama_model: str = 'llama3.1:8b'

    sqlite_path: str = 'app.db'
    log_level: str = 'INFO'
    admin_token: str = 'change-me'

    retrieval_k_default: int = 6
    retrieval_k_max: int = 12
    min_similarity: float = 0.35
    cross_collection_similarity: float = 0.45

    quote_max_chars: int = 240

    data_dir: str = 'data'
    upanishads_path: str = 'data/108-upanishads.pdf'
    puranas_path: str = 'data/18 Puranas.pdf'
    rigved_path: str = 'data/rigved.pdf'
    yajurved_path: str = 'data/yajurved.pdf'
    samved_path: str = 'data/samved.pdf'
    arthved_path: str = 'data/arthved-part-1.pdf'
    atharva2_path: str = 'data/atharva-2.pdf'

    ocr_languages: str = 'hin+san'
    ocr_dpi: int = 240

    @property
    def sqlite_url(self) -> str:
        return f'sqlite:///{self.sqlite_path}'

    def ensure_data_files_exist(self) -> list[str]:
        paths = [
            self.upanishads_path,
            self.puranas_path,
            self.rigved_path,
            self.yajurved_path,
            self.samved_path,
            self.arthved_path,
            self.atharva2_path,
        ]
        missing: list[str] = []
        for path in paths:
            if not Path(path).exists():
                missing.append(path)
        return missing


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
