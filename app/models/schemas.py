from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class SourceFilter(BaseModel):
    upanishads: bool = True
    puranas: bool = True
    vedas: bool = True


class QueryRequest(BaseModel):
    mode: Literal['ask', 'mood', 'followup']
    query: str = Field(min_length=1)
    mood: Literal['stress', 'confusion', 'fear', 'career', 'anger', 'grief'] | None = None
    lang_pref: Literal['auto', 'en', 'hi'] = 'auto'
    tone: Literal['modern', 'devotional', 'neutral'] = 'neutral'
    sources: SourceFilter = Field(default_factory=SourceFilter)
    k: int = Field(default=6, ge=1, le=12)
    include_original: bool = True
    session_id: str | None = None


class Citation(BaseModel):
    quote: str
    source: str
    doc: str
    page: int
    chunk_id: str
    language: Literal['en', 'hi']


class OriginalSnippet(BaseModel):
    quote: str
    doc: str
    page: int
    chunk_id: str
    language: Literal['en', 'hi']


class AnswerBlock(BaseModel):
    summary: str
    explanation: str
    action_step: str
    reflection_question: str


class QueryResponse(BaseModel):
    answer: AnswerBlock
    citations: list[Citation]
    original_snippets: list[OriginalSnippet]
    routing: dict[str, str]
    safety: dict[str, str]


class DailyResponse(BaseModel):
    date: date
    lang: Literal['en', 'hi', 'auto']
    verse: str
    meaning: str
    reflection_question: str
    citations: list[Citation]


class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    mode: Literal['ask', 'mood', 'followup', 'daily']
    rating: Literal[1, -1]
    flags: list[Literal['hallucination', 'bad_ocr', 'irrelevant', 'offensive']] = Field(default_factory=list)
    comment: str | None = None
    cited_chunk_ids: list[str] = Field(default_factory=list)


class FeedbackStatsResponse(BaseModel):
    total: int
    positive: int
    negative: int
    by_flag: dict[str, int]


class IngestRequest(BaseModel):
    docs: list[str] = Field(default_factory=list)
    force_rebuild: bool = False


class IngestResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    docs: list[str]
    force_rebuild: bool
    message: str | None
    created_at: datetime
    updated_at: datetime
