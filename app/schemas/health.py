"""
app/schemas/health.py
----------------------
Pydantic schema for the GET /health endpoint.
"""

from pydantic import BaseModel
from typing import Optional


class HealthResponse(BaseModel):
    """
    Health check response.
    status is always "ok" when the service is running normally.
    """
    status: str
    version: Optional[str] = None
    environment: Optional[str] = None
