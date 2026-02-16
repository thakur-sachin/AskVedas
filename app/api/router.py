from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import admin, daily, debug, feedback, health, library, query, ready

router = APIRouter()
router.include_router(health.router)
router.include_router(ready.router)
router.include_router(query.router)
router.include_router(daily.router)
router.include_router(library.router)
router.include_router(feedback.router)
router.include_router(admin.router)
router.include_router(debug.router)
