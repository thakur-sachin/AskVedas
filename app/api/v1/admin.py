from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import AdminAuthError
from app.db.session import get_db_session
from app.models.schemas import IngestRequest, IngestResponse, JobStatusResponse
from app.services.admin_service import AdminService

router = APIRouter()


def _admin_guard(x_admin_token: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if x_admin_token != settings.admin_token:
        raise AdminAuthError()


@router.post('/admin/ingest', response_model=IngestResponse, dependencies=[Depends(_admin_guard)])
def admin_ingest(
    payload: IngestRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_session),
) -> IngestResponse:
    service = AdminService(db)
    try:
        job_id, status = service.enqueue_ingest(payload, background_tasks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse(job_id=job_id, status=status)


@router.get('/admin/jobs/{job_id}', response_model=JobStatusResponse, dependencies=[Depends(_admin_guard)])
def admin_job(job_id: str, db: Session = Depends(get_db_session)) -> JobStatusResponse:
    service = AdminService(db)
    try:
        return service.get_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
