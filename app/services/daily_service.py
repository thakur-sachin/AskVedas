from __future__ import annotations

import hashlib
import json
from datetime import date

from sqlalchemy import and_, delete, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import DailyCache
from app.models.schemas import Citation, DailyResponse
from app.vector.qdrant_client import get_qdrant_client


class DailyService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()

    def get_daily(self, lang: str = 'auto') -> DailyResponse:
        date_key = date.today().isoformat()
        cached = self._load_cache(date_key=date_key, lang=lang)
        if cached:
            return cached
        return self.refresh_daily(lang=lang)

    def refresh_daily(self, lang: str = 'auto') -> DailyResponse:
        date_key = date.today().isoformat()
        resolved_lang = self._resolve_daily_language(lang)
        quote, citation = self._pick_passage(date_key=date_key, lang=resolved_lang)

        meaning = (
            'I don\'t find a direct passage in the indexed scriptures for this question.'
            if not quote
            else f"This passage points to disciplined reflection and practice: {quote[:180]}"
        )
        reflection_question = (
            'Which scripture keyword should you explore next?'
            if not quote
            else 'What one action can align your day with this teaching?'
        )

        response = DailyResponse(
            date=date.fromisoformat(date_key),
            lang='auto' if lang == 'auto' else resolved_lang,
            verse=quote or 'No indexed verse is available yet. Run ingestion first.',
            meaning=meaning,
            reflection_question=reflection_question,
            citations=[citation] if citation else [],
        )

        self.db.execute(delete(DailyCache).where(and_(DailyCache.date_key == date_key, DailyCache.lang == lang)))
        self.db.add(
            DailyCache(
                date_key=date_key,
                lang=lang,
                verse=response.verse,
                meaning=response.meaning,
                reflection_question=response.reflection_question,
                citations=json.dumps([c.model_dump() for c in response.citations], ensure_ascii=False),
            )
        )
        self.db.commit()
        return response

    def _load_cache(self, date_key: str, lang: str) -> DailyResponse | None:
        row = self.db.scalar(select(DailyCache).where(and_(DailyCache.date_key == date_key, DailyCache.lang == lang)))
        if not row:
            return None
        citations_raw = json.loads(row.citations)
        citations = [Citation(**c) for c in citations_raw]
        return DailyResponse(
            date=date.fromisoformat(row.date_key),
            lang='auto' if row.lang == 'auto' else row.lang,
            verse=row.verse,
            meaning=row.meaning,
            reflection_question=row.reflection_question,
            citations=citations,
        )

    def _resolve_daily_language(self, lang: str) -> str:
        if lang in {'en', 'hi'}:
            return lang
        # Deterministic alternation by date hash for auto mode.
        digest = hashlib.sha256(date.today().isoformat().encode('utf-8')).hexdigest()
        return 'hi' if int(digest[:2], 16) % 2 == 0 else 'en'

    def _pick_passage(self, date_key: str, lang: str) -> tuple[str, Citation | None]:
        collection = self.settings.collection_hi if lang == 'hi' else self.settings.collection_en
        client = get_qdrant_client()
        points, _ = client.scroll(collection_name=collection, limit=1000, with_payload=True, with_vectors=False)
        if not points and collection == self.settings.collection_hi:
            points, _ = client.scroll(collection_name=self.settings.collection_en, limit=1000, with_payload=True, with_vectors=False)
        if not points:
            return '', None

        idx_seed = int(hashlib.sha256(f'{date_key}:{collection}'.encode('utf-8')).hexdigest(), 16)
        point = points[idx_seed % len(points)]
        payload = point.payload or {}
        quote = str(payload.get('text', '')).strip().replace('\n', ' ')[: self.settings.quote_max_chars]
        if not quote:
            return '', None
        citation = Citation(
            quote=quote,
            source=f"{payload.get('work', payload.get('doc', 'scripture'))} (p.{payload.get('page', 1)})",
            doc=str(payload.get('doc', 'unknown')),
            page=int(payload.get('page', 1)),
            chunk_id=str(payload.get('chunk_id', point.id)),
            language='hi' if str(payload.get('language', 'en')) == 'hi' else 'en',
        )
        return quote, citation
