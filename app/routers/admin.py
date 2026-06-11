"""
app/routers/admin.py
---------------------
Admin control-plane endpoints for PrimeModel.ai.

Endpoints:
    POST /admin/retrain              — trigger legacy bulk retrain (all eligible meters)
    POST /admin/train/{deveui}       — queue LSTM-AE training for a specific device
    GET  /admin/models/status        — filesystem scan of persisted model artifacts
    GET  /admin/training/history     — tail of the JSONL training history log
    GET  /admin/devices/{deveui}/scores — most-recent analysis_results rows for a device

Security note:
    TD-ADMIN-001 — RESOLVED. Authentication is enforced router-wide via
    app/utils/admin_auth.py (require_admin_key), applied as a router-level
    dependency in app/main.py. Every request to /admin/* must carry a valid
    X-Admin-Key header matching the ADMIN_API_KEY env var. FAIL-CLOSED:
    if ADMIN_API_KEY is unset on the server, all /admin requests return 503.

    DO NOT add per-endpoint auth here — the router-level dependency in main.py
    already covers every route on this router.
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from app.services import device_registry, model_registry, supabase_client
from app.services import training
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# P3 endpoints — device-grain training control plane
# ---------------------------------------------------------------------------

@router.post("/train/{deveui}", tags=["Admin"])
def train_device(
    deveui: str,
    background_tasks: BackgroundTasks,
    param: Optional[str] = None,
) -> JSONResponse:
    """
    Queue an LSTM Autoencoder training job for a specific device.

    The job runs as a FastAPI BackgroundTask (synchronous, same process).
    A per-device in-process lock prevents concurrent training runs for the
    same device — duplicate requests while a job is running return 409.

    Results land in:
    - Training history log: GET /admin/training/history
    - Model store:          GET /admin/models/status

    Args:
        deveui: Device EUI (case-insensitive — normalised to uppercase).
        param:  Optional query parameter — name of the sensor parameter to
                train on (e.g. ``temperature``).  When omitted the first
                decoded numeric parameter is used.

    Returns:
        202 Accepted with ``{"status": "training_started", "deveui": ..., "param": ...}``.

    Raises:
        409 if training is already in progress for this device.
    """
    deveui = deveui.strip().upper()

    if not training.try_acquire_training_lock(deveui):
        raise HTTPException(
            status_code=409,
            detail="training already in progress for this device",
        )

    background_tasks.add_task(training.run_training_job, deveui, param)

    return JSONResponse(
        status_code=202,
        content={"status": "training_started", "deveui": deveui, "param": param},
    )


@router.get("/models/status", tags=["Admin"])
def models_status() -> Dict[str, Any]:
    """
    Return a filesystem scan of persisted model artifacts across all layers
    and all registered devices.

    Scans the model store directory for known layer subdirectories:
    - isolation_forest: ``{deveui}_{param}.joblib`` files
    - lstm_autoencoder: ``{deveui}_{param}/`` SavedModel dirs +
                        ``{deveui}_{param}_stats.json`` threshold stat files
    - lstm_forecast:    ``{deveui}_{param}/`` SavedModel dirs
    - cnn_pattern:      ``{deveui}_{param}/`` SavedModel dirs

    Reports filesystem truth only — never reads state files.

    Returns:
        {
            "model_store": str,
            "devices": {
                "<DEVEUI>": {
                    "layers": {
                        "isolation_forest": {"params": [...]},
                        "lstm_autoencoder": {"params": [...], "stats_params": [...]},
                        "lstm_forecast":    {"params": [...]},
                        "cnn_pattern":      {"params": [...]},
                    },
                    "training": bool
                },
                ...
            }
        }
    """
    settings = get_settings()
    store = settings.model_store_path
    deveuis = device_registry.list_registered_devices()

    # Layer directory names — derived from the actual module save paths.
    # isolation_forest:  {store}/isolation_forest/{safe_id}.joblib
    # lstm_autoencoder:  {store}/lstm_autoencoder/{safe_id}/ + {safe_id}_stats.json
    # lstm_forecast:     {store}/lstm_forecast/{safe_id}/
    # cnn_pattern:       {store}/cnn_pattern/{safe_id}/
    layer_dirs = {
        "isolation_forest": os.path.join(store, "isolation_forest"),
        "lstm_autoencoder": os.path.join(store, "lstm_autoencoder"),
        "lstm_forecast":    os.path.join(store, "lstm_forecast"),
        "cnn_pattern":      os.path.join(store, "cnn_pattern"),
    }

    devices: Dict[str, Any] = {}

    for deveui in deveuis:
        prefix = deveui + "_"  # deveuis never contain underscores — safe prefix

        # --- isolation_forest ---
        if_params: List[str] = []
        if_dir = layer_dirs["isolation_forest"]
        if os.path.isdir(if_dir):
            for name in os.listdir(if_dir):
                if name.startswith(prefix) and name.endswith(".joblib"):
                    # Strip prefix and suffix to recover param name
                    param_name = name[len(prefix):-len(".joblib")]
                    if param_name:
                        if_params.append(param_name)

        # --- lstm_autoencoder ---
        ae_params: List[str] = []
        ae_stats_params: List[str] = []
        ae_dir = layer_dirs["lstm_autoencoder"]
        if os.path.isdir(ae_dir):
            for name in os.listdir(ae_dir):
                full = os.path.join(ae_dir, name)
                if name.startswith(prefix):
                    if name.endswith("_stats.json"):
                        # Stats file: {deveui}_{param}_stats.json
                        # Strip prefix, then strip "_stats.json" suffix
                        tail = name[len(prefix):]
                        param_name = tail[:-len("_stats.json")]
                        if param_name:
                            ae_stats_params.append(param_name)
                    elif os.path.isdir(full):
                        # SavedModel dir: {deveui}_{param}/
                        param_name = name[len(prefix):]
                        if param_name:
                            ae_params.append(param_name)

        # --- lstm_forecast ---
        lf_params: List[str] = []
        lf_dir = layer_dirs["lstm_forecast"]
        if os.path.isdir(lf_dir):
            for name in os.listdir(lf_dir):
                full = os.path.join(lf_dir, name)
                if name.startswith(prefix) and os.path.isdir(full):
                    param_name = name[len(prefix):]
                    if param_name:
                        lf_params.append(param_name)

        # --- cnn_pattern ---
        cnn_params: List[str] = []
        cnn_dir = layer_dirs["cnn_pattern"]
        if os.path.isdir(cnn_dir):
            for name in os.listdir(cnn_dir):
                full = os.path.join(cnn_dir, name)
                if name.startswith(prefix) and os.path.isdir(full):
                    param_name = name[len(prefix):]
                    if param_name:
                        cnn_params.append(param_name)

        devices[deveui] = {
            "layers": {
                "isolation_forest": {"params": sorted(if_params)},
                "lstm_autoencoder": {
                    "params": sorted(ae_params),
                    "stats_params": sorted(ae_stats_params),
                },
                "lstm_forecast": {"params": sorted(lf_params)},
                "cnn_pattern":   {"params": sorted(cnn_params)},
            },
            "training": training.is_training(deveui),
        }

    return {"model_store": store, "devices": devices}


@router.get("/training/history", tags=["Admin"])
def get_training_history(limit: int = 50) -> Dict[str, Any]:
    """
    Return the most recent training job history entries from the JSONL log.

    Args:
        limit: Number of entries to return.  Clamped to [1, 200].

    Returns:
        ``{"history": [...]}`` — list of training result dicts, newest first.
    """
    limit = max(1, min(limit, 200))
    return {"history": training.read_training_history(limit)}


@router.get("/devices/{deveui}/scores", tags=["Admin"])
def get_device_scores(deveui: str, limit: int = 20) -> Dict[str, Any]:
    """
    Return the most recent analysis_results rows for a specific device.

    Fetches from Supabase ``analysis_results`` table, selecting the 10
    columns: created_at, ensemble_score, layer1_scores, layer2_score,
    layer3_score, layer4_score, active_layers, anomaly_detected,
    readings_used, days_of_data.

    Args:
        deveui: Device EUI (case-insensitive — normalised to uppercase).
        limit:  Number of rows to return.  Clamped to [1, 200].

    Returns:
        ``{"deveui": str, "count": int, "scores": [...]}``

    Raises:
        404 if no analysis results exist for this device.
    """
    deveui = deveui.strip().upper()
    limit = max(1, min(limit, 200))

    rows = supabase_client.fetch_analysis_results(deveui, limit)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail="no analysis results for this device",
        )

    return {"deveui": deveui, "count": len(rows), "scores": rows}


# ---------------------------------------------------------------------------
# Legacy bulk retrain
# ---------------------------------------------------------------------------

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
