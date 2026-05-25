"""
app/services/model_registry.py
--------------------------------
Model registry — tracks activation status, training state, and baseline
statistics for all layers on a per-meter basis.

Each meter has a JSON state file at:
    models_store/baselines/{meter_id}.json

This file contains:
    - first_reading_at: ISO timestamp of the meter's first received uplink
    - days_of_data: float — days since first_reading_at
    - layer activation flags: statistical_active, lstm_ae_active, lstm_forecast_active, cnn_active
    - isolation_forest_trained_at: ISO timestamp of last IF train
    - lstm_ae_trained_at: ISO timestamp of last AE train
    - lstm_forecast_trained_at: ISO timestamp of last forecast train
    - cnn_trained_at: ISO timestamp of last CNN train
    - cusum_state: current CUSUM state dict
    - ewma_state: current EWMA state dict
    - burst_state: burst detector baseline
    - isolation_forest_threshold: anomaly threshold for IF (if applicable)
    - lstm_ae_threshold_stats: MAE mean/std/threshold for AE
    - lstm_forecast_baseline: RMSE mean/std for forecast
    - last_flow_rate: most recent flow reading (for delta computation)
    - flow_history: last N flow readings for rolling stats

The registry does not load ML models — it only manages metadata and state.
ML model loading is done in each service module.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from app.utils.time_utils import utcnow, days_between, parse_iso_timestamp
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Number of recent flow readings to keep in history for rolling stats and burst detection
FLOW_HISTORY_MAX = 200


def _baseline_path(model_store: str, meter_id: str) -> str:
    """Return the full path for a meter's baseline JSON file."""
    safe_id = meter_id.replace("/", "_").replace("\\", "_")
    return os.path.join(model_store, "baselines", f"{safe_id}.json")


def load_state(model_store: str, meter_id: str) -> Dict[str, Any]:
    """
    Load the meter state from disk, or initialise a fresh state if none exists.

    Args:
        model_store: Root model store path.
        meter_id: Meter identifier.

    Returns:
        State dict for the meter.
    """
    path = _baseline_path(model_store, meter_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
            return state
        except Exception as e:
            logger.error("Failed to load meter state", extra={"path": path, "error": str(e)})

    # Initialise fresh state
    return {
        "meter_id": meter_id,
        "first_reading_at": None,
        "days_of_data": 0.0,
        "statistical_active": False,
        "lstm_ae_active": False,
        "lstm_forecast_active": False,
        "cnn_active": False,
        "isolation_forest_trained_at": None,
        "lstm_ae_trained_at": None,
        "lstm_forecast_trained_at": None,
        "cnn_trained_at": None,
        "cusum_state": None,
        "ewma_state": None,
        "burst_state": None,
        "isolation_forest_threshold": None,
        "lstm_ae_threshold_stats": None,
        "lstm_forecast_baseline": None,
        "last_flow_rate": None,
        "flow_history": [],
    }


def save_state(model_store: str, meter_id: str, state: Dict[str, Any]) -> None:
    """
    Persist the meter state to disk.

    Args:
        model_store: Root model store path.
        meter_id: Meter identifier.
        state: State dict to save.
    """
    path = _baseline_path(model_store, meter_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        logger.error("Failed to save meter state", extra={"path": path, "error": str(e)})


def record_uplink(
    state: Dict[str, Any],
    flow_rate: float,
    timestamp: str,
    cold_start_days: int,
    lstm_ae_activation_days: int,
    lstm_forecast_activation_days: int,
    cnn_activation_days: int,
) -> Dict[str, Any]:
    """
    Update state with a new uplink and compute days_of_data.
    Sets activation flags based on accumulated days.

    Args:
        state: Current meter state.
        flow_rate: Flow rate from this uplink.
        timestamp: ISO-8601 timestamp of this uplink.
        cold_start_days: Days until statistical layer activates.
        lstm_ae_activation_days: Days until LSTM AE activates.
        lstm_forecast_activation_days: Days until LSTM Forecast activates.
        cnn_activation_days: Days until CNN activates.

    Returns:
        Updated state dict (not yet persisted — caller must call save_state).
    """
    # Record first reading timestamp
    if state.get("first_reading_at") is None:
        state["first_reading_at"] = timestamp
        logger.info("First uplink recorded for meter", extra={"meter_id": state.get("meter_id")})

    # Compute days of data
    try:
        first_dt = parse_iso_timestamp(state["first_reading_at"])
        current_dt = parse_iso_timestamp(timestamp)
        state["days_of_data"] = days_between(first_dt, current_dt)
    except Exception as e:
        logger.warning("Could not compute days_of_data", extra={"error": str(e)})

    days = state.get("days_of_data", 0.0)

    # Update activation flags
    state["statistical_active"] = days >= cold_start_days
    state["lstm_ae_active"] = days >= lstm_ae_activation_days
    state["lstm_forecast_active"] = days >= lstm_forecast_activation_days
    state["cnn_active"] = days >= cnn_activation_days

    # Append flow to history (trim to FLOW_HISTORY_MAX)
    history: List[float] = state.get("flow_history", [])
    history.append(float(flow_rate))
    if len(history) > FLOW_HISTORY_MAX:
        history = history[-FLOW_HISTORY_MAX:]
    state["flow_history"] = history
    state["last_flow_rate"] = float(flow_rate)

    return state


def get_activation_status(
    state: Dict[str, Any],
    cold_start_days: int,
    lstm_ae_activation_days: int,
    lstm_forecast_activation_days: int,
    cnn_activation_days: int,
) -> Dict[str, Any]:
    """
    Return the model layer activation status and days-until-activation for each layer.

    Used by GET /models/status and included in every AnalyseResponse.

    Args:
        state: Meter state dict.
        *_days: Activation thresholds from config.

    Returns:
        Dict matching the ModelLayerStatuses schema.
    """
    days = float(state.get("days_of_data", 0.0))

    def days_until(threshold: int) -> int:
        remaining = threshold - days
        return max(0, int(remaining))

    return {
        "statistical": "active" if state.get("statistical_active") else "warming_up",
        "lstm_autoencoder": "active" if state.get("lstm_ae_active") else "warming_up",
        "lstm_forecast": "active" if state.get("lstm_forecast_active") else "warming_up",
        "cnn_pattern": "active" if state.get("cnn_active") else "warming_up",
        "days_until_autoencoder": days_until(lstm_ae_activation_days),
        "days_until_forecast": days_until(lstm_forecast_activation_days),
        "days_until_cnn": days_until(cnn_activation_days),
    }


def needs_retraining(
    state: Dict[str, Any],
    layer: str,
    retrain_interval_days: int,
) -> bool:
    """
    Check if a layer is due for retraining based on the retrain interval.

    Args:
        state: Meter state dict.
        layer: One of "isolation_forest", "lstm_ae", "lstm_forecast", "cnn".
        retrain_interval_days: How often to retrain.

    Returns:
        True if the layer has never been trained, or was last trained more
        than retrain_interval_days ago.
    """
    key = f"{layer}_trained_at"
    trained_at = state.get(key)

    if trained_at is None:
        return True

    try:
        last_trained = parse_iso_timestamp(trained_at)
        days_since = days_between(last_trained)
        return days_since >= retrain_interval_days
    except Exception:
        return True


def mark_trained(state: Dict[str, Any], layer: str) -> Dict[str, Any]:
    """
    Record that a layer was just trained.

    Args:
        state: Meter state dict (mutated in place and returned).
        layer: Layer name ("isolation_forest", "lstm_ae", "lstm_forecast", "cnn").

    Returns:
        Updated state dict.
    """
    key = f"{layer}_trained_at"
    state[key] = utcnow().isoformat()
    return state


def list_all_meters(model_store: str) -> List[str]:
    """
    List all meter IDs that have a baseline state file.

    Returns:
        List of meter IDs (unsanitised filename stems).
    """
    baselines_dir = os.path.join(model_store, "baselines")
    if not os.path.exists(baselines_dir):
        return []
    return [
        f.replace(".json", "")
        for f in os.listdir(baselines_dir)
        if f.endswith(".json")
    ]
