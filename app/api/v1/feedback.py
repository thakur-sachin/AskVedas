from __future__ import annotations

from fastapi import APIRouter, Depends, Header
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AdminAuthError
from app.db.session import get_db_session
from app.models.schemas import FeedbackRequest, FeedbackStatsResponse
from app.services.feedback_service import FeedbackService

router = APIRouter()


def _admin_guard(x_admin_token: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if x_admin_token != settings.admin_token:
        raise AdminAuthError()


@router.post('/feedback')
def create_feedback(payload: FeedbackRequest, db: Session = Depends(get_db_session)) -> dict[str, str]:
    service = FeedbackService(db)
    service.create_feedback(payload)
    return {'status': 'ok'}


@router.get('/feedback/stats', response_model=FeedbackStatsResponse, dependencies=[Depends(_admin_guard)])
def feedback_stats(db: Session = Depends(get_db_session)) -> FeedbackStatsResponse:
    service = FeedbackService(db)
    return service.stats()
