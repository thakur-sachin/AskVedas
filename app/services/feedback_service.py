from __future__ import annotations

import json

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Feedback
from app.models.schemas import FeedbackRequest, FeedbackStatsResponse


class FeedbackService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_feedback(self, payload: FeedbackRequest) -> None:
        record = Feedback(
            session_id=payload.session_id,
            query=payload.query,
            mode=payload.mode,
            rating=payload.rating,
            flags=json.dumps(payload.flags, ensure_ascii=False),
            comment=payload.comment,
            cited_chunk_ids=json.dumps(payload.cited_chunk_ids, ensure_ascii=False),
        )
        self.db.add(record)
        self.db.commit()

    def stats(self) -> FeedbackStatsResponse:
        total = self.db.scalar(select(func.count()).select_from(Feedback)) or 0
        positive = self.db.scalar(select(func.count()).select_from(Feedback).where(Feedback.rating == 1)) or 0
        negative = self.db.scalar(select(func.count()).select_from(Feedback).where(Feedback.rating == -1)) or 0

        rows = self.db.scalars(select(Feedback.flags)).all()
        by_flag: dict[str, int] = {}
        for flags_json in rows:
            try:
                flags = json.loads(flags_json)
            except Exception:
                flags = []
            for flag in flags:
                by_flag[flag] = by_flag.get(flag, 0) + 1

        return FeedbackStatsResponse(total=int(total), positive=int(positive), negative=int(negative), by_flag=by_flag)
