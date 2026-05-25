"""
app/routers/models.py
----------------------
GET /models/status — activation status of all model layers across all meters.

Returns a per-meter summary of which NN layers are active and how many
days of data each meter has accumulated.

Useful for the UTP team to monitor the system's progressive activation
and know when each meter's neural network layers will come online.
"""

from fastapi import APIRouter, Query
from typing import List, Dict, Any

from app.services import model_registry
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/status", tags=["Models"])
def get_models_status() -> Dict[str, Any]:
    """
    Return activation status of all model layers for all known meters.

    Reads all baseline state files from models_store/baselines/
    and returns a summary of layer activation per meter.

    Returns:
        Dict with "meters" list, each containing meter_id, days_of_data,
        and layer activation statuses.
    """
    settings = get_settings()
    meter_ids = model_registry.list_all_meters(settings.model_store_path)

    meters_status = []
    for meter_id in meter_ids:
        state = model_registry.load_state(settings.model_store_path, meter_id)
        layer_status = model_registry.get_activation_status(
            state,
            settings.cold_start_days,
            settings.lstm_ae_activation_days,
            settings.lstm_forecast_activation_days,
            settings.cnn_activation_days,
        )
        meters_status.append({
            "meter_id": meter_id,
            "days_of_data": round(state.get("days_of_data", 0.0), 1),
            "first_reading_at": state.get("first_reading_at"),
            "layer_status": layer_status,
            "last_trained": {
                "isolation_forest": state.get("isolation_forest_trained_at"),
                "lstm_autoencoder": state.get("lstm_ae_trained_at"),
                "lstm_forecast": state.get("lstm_forecast_trained_at"),
                "cnn_pattern": state.get("cnn_trained_at"),
            },
        })

    return {
        "total_meters": len(meters_status),
        "meters": meters_status,
    }
