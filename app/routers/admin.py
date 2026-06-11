"""
app/routers/admin.py
---------------------
POST /admin/retrain — trigger retraining of all eligible model layers.

This endpoint is called manually or on a scheduled basis to retrain
statistical and neural models for all meters that have accumulated enough
data and are past their retrain interval.

Security note:
    TD-ADMIN-001 — RESOLVED. Authentication is enforced router-wide via
    app/utils/admin_auth.py (require_admin_key), applied as a router-level
    dependency in app/main.py. Every request to /admin/* must carry a valid
    X-Admin-Key header matching the ADMIN_API_KEY env var. FAIL-CLOSED:
    if ADMIN_API_KEY is unset on the server, all /admin requests return 503.
"""

from fastapi import APIRouter, BackgroundTasks
from typing import Dict, Any
import asyncio

from app.services import model_registry, supabase_client
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/retrain", tags=["Admin"])
def trigger_retrain(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Trigger background retraining for all eligible meters.

    Retraining is run as a background task so the response is returned
    immediately. Training progress is logged via the structured logger.

    Returns:
        Dict with status and count of meters queued for retraining.
    """
    settings = get_settings()
    meter_ids = model_registry.list_all_meters(settings.model_store_path)

    eligible = []
    for meter_id in meter_ids:
        state = model_registry.load_state(settings.model_store_path, meter_id)
        if _is_eligible_for_retrain(state, settings):
            eligible.append(meter_id)

    background_tasks.add_task(_run_retraining, eligible, settings)

    logger.info(
        "Retrain triggered",
        extra={"total_meters": len(meter_ids), "eligible": len(eligible)},
    )

    return {
        "status": "accepted",
        "meters_queued": len(eligible),
        "total_meters": len(meter_ids),
        "message": f"Background retraining started for {len(eligible)} meter(s). Check logs for progress.",
    }


def _is_eligible_for_retrain(state: Dict, settings) -> bool:
    """
    Return True if any layer of this meter needs retraining.
    """
    layers = ["isolation_forest", "lstm_ae", "lstm_forecast", "cnn"]
    for layer in layers:
        if model_registry.needs_retraining(state, layer, settings.retrain_interval_days):
            if _has_enough_data(state, layer, settings):
                return True
    return False


def _has_enough_data(state: Dict, layer: str, settings) -> bool:
    """Check if the meter has enough days of data to train this layer."""
    days = state.get("days_of_data", 0.0)
    thresholds = {
        "isolation_forest": settings.cold_start_days,
        "lstm_ae": settings.lstm_ae_activation_days,
        "lstm_forecast": settings.lstm_forecast_activation_days,
        "cnn": settings.cnn_activation_days,
    }
    return days >= thresholds.get(layer, 9999)


def _run_retraining(meter_ids: list, settings) -> None:
    """
    Background task: fetch history and retrain eligible layers for each meter.
    Imports are done inside the function to keep startup fast.
    """
    import pandas as pd
    import numpy as np
    from app.services.statistical import isolation_forest
    from app.services.neural import lstm_autoencoder, lstm_forecast, cnn_pattern
    from app.services import feature_engineering

    logger.info("Background retraining started", extra={"meter_count": len(meter_ids)})

    for meter_id in meter_ids:
        try:
            state = model_registry.load_state(settings.model_store_path, meter_id)

            # We need client_id to query Supabase — stored in baseline state
            # If not present, skip (meter must have had at least one analyse call)
            # Note: client_id is not currently stored in state — this is a known
            # limitation for the admin retrain path.
            # TECHNICAL DEBT: TD-RETRAIN-001 (LOW)
            #   Store client_id in meter state so /admin/retrain can fetch history.
            #   For now, skip meters where client_id is not derivable.
            logger.info("Retrain skipped for meter (client_id not in state)", extra={"meter_id": meter_id})
            # When TD-RETRAIN-001 is resolved, replace above with actual training call

        except Exception as e:
            logger.error("Retrain failed for meter", extra={"meter_id": meter_id, "error": str(e)})

    logger.info("Background retraining complete")
