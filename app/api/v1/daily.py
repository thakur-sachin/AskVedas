from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Header
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AdminAuthError
from app.db.session import get_db_session
from app.models.schemas import DailyResponse
from app.services.daily_service import DailyService

router = APIRouter()


def _admin_guard(x_admin_token: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if x_admin_token != settings.admin_token:
        raise AdminAuthError()


@router.get('/daily', response_model=DailyResponse)
def get_daily(lang: Literal['auto', 'en', 'hi'] = 'auto', db: Session = Depends(get_db_session)) -> DailyResponse:
    service = DailyService(db)
    return service.get_daily(lang=lang)


@router.post('/daily/refresh', response_model=DailyResponse, dependencies=[Depends(_admin_guard)])
def refresh_daily(lang: Literal['auto', 'en', 'hi'] = 'auto', db: Session = Depends(get_db_session)) -> DailyResponse:
    service = DailyService(db)
    return service.refresh_daily(lang=lang)
