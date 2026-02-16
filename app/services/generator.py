from __future__ import annotations

import json
import re

import requests

from app.core.config import get_settings
from app.models.domain import RetrievedChunk
from app.models.schemas import AnswerBlock, Citation, OriginalSnippet

NO_ANSWER = 'I don\'t find a direct passage in the indexed scriptures for this question.'
SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\?\!।॥])\s+')

def _extract_first_sentence(text: str) -> str:
    clean = text.strip().replace('\n', ' ')
    if not clean:
        return ''
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(clean) if p.strip()]
    return parts[0] if parts else clean


def make_citations(chunks: list[RetrievedChunk], max_chars: int) -> list[Citation]:
    citations: list[Citation] = []
    for chunk in chunks:
        quote = chunk.text.strip().replace('\n', ' ')
        if not quote:
            continue
        trimmed = quote[:max_chars]
        citations.append(
            Citation(
                quote=trimmed,
                source=f"{chunk.work} (p.{chunk.page})",
                doc=chunk.doc,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                language='hi' if chunk.language == 'hi' else 'en',
            )
        )
    return citations


def to_original_snippets(chunks: list[RetrievedChunk], max_chars: int) -> list[OriginalSnippet]:
    snippets: list[OriginalSnippet] = []
    for chunk in chunks:
        text = chunk.text.strip().replace('\n', ' ')
        if not text:
            continue
        snippets.append(
            OriginalSnippet(
                quote=text[:max_chars],
                doc=chunk.doc,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                language='hi' if chunk.language == 'hi' else 'en',
            )
        )
    return snippets


def deterministic_answer(query: str, mood: str | None, tone: str, citations: list[Citation]) -> AnswerBlock:
    if not citations:
        return AnswerBlock(
            summary=NO_ANSWER,
            explanation=NO_ANSWER,
            action_step='Reframe the question with a scripture name or key terms for stronger retrieval.',
            reflection_question='Which exact term from the scriptures should be queried next?',
        )

    top = citations[0]
    top_sentence = _extract_first_sentence(top.quote)
    summary = f'Based on the cited passages: {top_sentence}' if top_sentence else NO_ANSWER

    supporting_sources = ', '.join([f'{c.doc} p.{c.page}' for c in citations[:3]])
    mood_clause = f'For {mood}, ' if mood else ''
    return AnswerBlock(
        summary=f'{mood_clause}{summary}',
        explanation=(
            f'Grounded in retrieved citations only. Primary support: {top.doc} p.{top.page}. '
            f'Additional support: {supporting_sources}.'
        ),
        action_step=f'Read citation {top.doc} page {top.page} and reflect in a {tone} tone.',
        reflection_question='What one principle from these cited lines can you practice today?',
    )


def ollama_answer(
    query: str,
    mood: str | None,
    tone: str,
    citations: list[Citation],
) -> AnswerBlock:
    settings = get_settings()
    context = '\n'.join([f"[{i+1}] {c.quote}" for i, c in enumerate(citations[:6])])
    prompt = (
        'You are a strict scripture-grounded assistant.\n'
        'HARD RULES:\n'
        '1) Use only the provided Context lines.\n'
        '2) Do not use outside knowledge, assumptions, or paraphrases not supported by Context.\n'
        '3) Every claim must be directly grounded in Context; do not invent verses, citations, or concepts.\n'
        '4) If direct support is missing or weak, return the no-answer text in all four fields.\n'
        '5) Keep the response relevant to the query and complete within available evidence.\n'
        '6) Output strict JSON only with keys: summary, explanation, action_step, reflection_question.\n\n'
        '7) Summary must directly answer the user query in one clear sentence.\n'
        f"Query: {query}\n"
        f"Mood: {mood or 'none'}\n"
        f"Tone: {tone}\n"
        f"Context:\n{context}\n"
        f"No-answer text: {NO_ANSWER}\n"
        'Validation: If any field cannot be grounded in Context, set all four fields to the no-answer text.'
    )
    payload = {
        'model': settings.ollama_model,
        'prompt': prompt,
        'stream': False,
        'format': 'json',
        'options': {'temperature': 0},
    }
    response = requests.post(f"{settings.ollama_base_url}/api/generate", json=payload, timeout=20)
    response.raise_for_status()
    body = response.json()
    text = body.get('response', '{}')
    data = json.loads(text)
    return AnswerBlock(
        summary=str(data.get('summary', NO_ANSWER)),
        explanation=str(data.get('explanation', NO_ANSWER)),
        action_step=str(data.get('action_step', NO_ANSWER)),
        reflection_question=str(data.get('reflection_question', NO_ANSWER)),
    )


def generate_answer(query: str, mood: str | None, tone: str, citations: list[Citation]) -> AnswerBlock:
    settings = get_settings()
    if not citations:
        return deterministic_answer(query=query, mood=mood, tone=tone, citations=[])
    if settings.llm_provider == 'mock':
        return deterministic_answer(query=query, mood=mood, tone=tone, citations=citations)
    try:
        return ollama_answer(query=query, mood=mood, tone=tone, citations=citations)
    except Exception:
        return deterministic_answer(query=query, mood=mood, tone=tone, citations=citations)
