"""
app/utils/admin_auth.py
-----------------------
Fail-closed authentication dependency for the /admin router.

Resolves TD-ADMIN-001: every /admin endpoint requires a static admin key
supplied via the X-Admin-Key request header, matched against the
ADMIN_API_KEY environment variable (settings.admin_api_key).

FAIL-CLOSED design:
    - If ADMIN_API_KEY is not configured on the server, ALL /admin requests
      are refused with 503 "Admin API not configured". The router is never
      open by accident — a missing key locks it down, it does not unlock it.
    - A missing or mismatched X-Admin-Key header returns 401.

Applied at ROUTER level (dependencies=[Depends(require_admin_key)] on the
include_router call in app/main.py), so every current AND future /admin
endpoint is protected automatically without per-endpoint wiring.

Implementation notes:
    - get_settings() is called INSIDE the dependency (request time), not at
      module import. This keeps the check live against the lru_cached
      settings singleton and allows tests to cache_clear() between cases.
    - secrets.compare_digest is used for constant-time comparison.
    - The submitted key value is NEVER logged.
"""

import secrets

from fastapi import Header, HTTPException

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def require_admin_key(x_admin_key: str = Header(None)) -> None:
    """
    FastAPI dependency — validate the X-Admin-Key header for /admin routes.

    Raises:
        HTTPException 503 — server has no ADMIN_API_KEY configured (fail-closed).
        HTTPException 401 — header missing or does not match the configured key.
    """
    settings = get_settings()  # request-time read — do not move to module level

    if not settings.admin_api_key:
        logger.warning("Admin request refused: ADMIN_API_KEY not configured (fail-closed)")
        raise HTTPException(status_code=503, detail="Admin API not configured")

    if not x_admin_key or not secrets.compare_digest(
        x_admin_key, settings.admin_api_key
    ):
        # Never log the submitted key value.
        logger.warning("Admin request refused: invalid or missing X-Admin-Key header")
        raise HTTPException(status_code=401, detail="Invalid or missing admin key")
