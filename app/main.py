"""
app/main.py
------------
PrimeModel AI Engine — FastAPI application entry point.

Registers all routers and configures the app for Railway deployment.

Routes:
    GET  /health             — liveness probe
    POST /analyse            — core per-uplink analysis endpoint
    GET  /rul/{meter_id}     — RUL for a specific meter
    GET  /models/status      — layer activation status across all meters
    POST /admin/retrain      — trigger background model retraining
"""

from fastapi import Depends, FastAPI
from app.routers import health, analyse, rul, models, admin
from app.utils.admin_auth import require_admin_key
from app.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="PrimeModel AI Engine",
    description=(
        "Neural network stack for water leak detection and meter RUL estimation. "
        "Provides progressive model activation: statistical (Day 1) → "
        "LSTM Autoencoder (Day 30) → LSTM Forecast (Day 60) → CNN Pattern (Day 90)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    redirect_slashes=False,
)

app.include_router(health.router, prefix="/health")
app.include_router(analyse.router, prefix="/analyse")
app.include_router(rul.router, prefix="/rul")
app.include_router(models.router, prefix="/models")
# Router-level auth — every current and future /admin endpoint is protected
# by the fail-closed X-Admin-Key check (resolves TD-ADMIN-001).
app.include_router(admin.router, prefix="/admin", dependencies=[Depends(require_admin_key)])

logger.info("PrimeModel AI Engine started")
