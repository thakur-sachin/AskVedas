from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    doc: str
    work: str
    page: int
    language: str
    script: str
    text: str
    text_type: str
    score: float


@dataclass(slots=True)
class DocRecord:
    doc: str
    path: str
    language: str
    script: str
    collection: str
