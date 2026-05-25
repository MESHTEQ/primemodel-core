"""
app/routers/health.py
----------------------
GET /health — lightweight liveness check.

Used by Railway healthcheck and upstream monitoring.
Returns the app environment and version so infra can confirm the right build is running.
"""

from fastapi import APIRouter
from app.schemas.health import HealthResponse
from app.config import get_settings

router = APIRouter()

# Increment this when shipping a new release
_VERSION = "1.0.0"


@router.get("/", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """
    Liveness probe.

    Returns:
        status "ok" with version and environment strings.
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=_VERSION,
        environment=settings.app_env,
    )
