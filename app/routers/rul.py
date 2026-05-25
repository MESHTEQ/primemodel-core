"""
app/routers/rul.py
-------------------
GET /rul/{meter_id} — Remaining Useful Life for a specific meter.

Returns battery and drift RUL based on persisted meter state.
The state is updated continuously by /analyse, so this endpoint
is a read-only view of the latest computed RUL values.
"""

from fastapi import APIRouter, HTTPException, Query
from app.schemas.rul import RULResponse
from app.services import model_registry, battery_rul as battery_rul_svc, supabase_client
from app.config import get_settings
from app.utils.time_utils import parse_iso_timestamp, days_between
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/{meter_id}", response_model=RULResponse, tags=["RUL"])
def get_rul(
    meter_id: str,
    client_id: str = Query(..., description="Tenant identifier"),
) -> RULResponse:
    """
    Return battery and drift RUL estimates for a specific meter.

    Args:
        meter_id: Meter identifier (path parameter).
        client_id: Tenant identifier (required query parameter).

    Returns:
        RULResponse with battery_rul_days, drift_rul_days, and status classifications.

    Raises:
        404 if the meter has no recorded state (no uplinks received yet).
    """
    settings = get_settings()

    state = model_registry.load_state(settings.model_store_path, meter_id)

    if state.get("first_reading_at") is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for meter '{meter_id}'. No uplinks received yet.",
        )

    # --- Battery RUL ---
    battery_rul_days_val = None
    battery_rul_status_val = "ok"
    battery_last_percent = state.get("last_flow_rate")  # Note: this is flow — battery is in history

    try:
        battery_rows = supabase_client.fetch_battery_history(meter_id, client_id, limit=200)
        if battery_rows:
            battery_last_percent = battery_rows[-1].get("battery_level")
            b_levels = [r["battery_level"] for r in battery_rows]
            first_ts = parse_iso_timestamp(battery_rows[0]["timestamp"])
            b_times = [
                days_between(first_ts, parse_iso_timestamp(r["timestamp"]))
                for r in battery_rows
            ]
            battery_rul_days_val, battery_rul_status_val = battery_rul_svc.estimate_rul(
                b_levels, b_times,
                eol_percent=settings.battery_eol_percent,
                warning_days=settings.battery_warning_days,
                critical_days=settings.battery_critical_days,
            )
        else:
            battery_rul_status_val = "ok"
    except Exception as e:
        logger.warning("Battery RUL fetch failed in /rul", extra={"meter_id": meter_id, "error": str(e)})

    # --- Drift RUL (from persisted state) ---
    drift_state = state.get("drift_state", {})
    offset_history = drift_state.get("offset_history", [])
    drift_offset_val = offset_history[-1] if offset_history else None

    from app.services.drift_rul import estimate_rul as drift_estimate
    drift_rul_days_val = None
    drift_rul_status_val = "warming_up"
    if offset_history:
        drift_rul_days_val, drift_rul_status_val = drift_estimate(
            offset_history,
            settings.drift_accuracy_threshold_percent,
        )

    # --- Explanation ---
    explanation_parts = []

    if battery_rul_days_val is not None:
        explanation_parts.append(
            f"Battery RUL: {battery_rul_days_val:.0f} days ({battery_rul_status_val})."
        )
    else:
        explanation_parts.append("Battery RUL: insufficient history for regression.")

    if drift_rul_status_val == "warming_up":
        explanation_parts.append("Drift RUL: warming up — insufficient baseline data.")
    elif drift_rul_status_val == "degraded":
        explanation_parts.append("Drift RUL: meter accuracy threshold exceeded — calibration recommended.")
    elif drift_rul_days_val is not None:
        explanation_parts.append(
            f"Drift RUL: {drift_rul_days_val:.0f} days until accuracy limit. Current offset: {drift_offset_val:.2f}%."
            if drift_offset_val is not None
            else f"Drift RUL: {drift_rul_days_val:.0f} days."
        )
    else:
        explanation_parts.append("Drift: no significant trend detected.")

    return RULResponse(
        meter_id=meter_id,
        client_id=client_id,
        battery_rul_days=round(battery_rul_days_val, 1) if battery_rul_days_val is not None else None,
        battery_rul_status=battery_rul_status_val,
        battery_last_reading_percent=battery_last_percent,
        drift_rul_days=round(drift_rul_days_val, 1) if drift_rul_days_val is not None else None,
        drift_rul_status=drift_rul_status_val,
        drift_offset_percent=round(drift_offset_val, 3) if drift_offset_val is not None else None,
        explanation=" ".join(explanation_parts),
    )
