"""
app/schemas/rul.py
-------------------
Pydantic schemas for the GET /rul/{meter_id} endpoint.
"""

from pydantic import BaseModel, Field
from typing import Optional
from app.schemas.uplink import DriftRulStatus, BatteryRulStatus


class RULResponse(BaseModel):
    """
    Remaining Useful Life indicators for a specific meter.
    Returned by GET /rul/{meter_id}.
    """
    meter_id: str
    client_id: str

    # Battery
    battery_rul_days: Optional[float] = Field(
        None,
        description="Estimated battery RUL in days. Null if insufficient history."
    )
    battery_rul_status: BatteryRulStatus = Field(
        ...,
        description="ok | warning | critical — based on configured thresholds"
    )
    battery_last_reading_percent: Optional[float] = Field(
        None,
        description="Most recent battery_level reading (%)"
    )

    # Drift
    drift_rul_days: Optional[float] = Field(
        None,
        description="Estimated metrological drift RUL in days. Null during warm-up."
    )
    drift_rul_status: DriftRulStatus = Field(
        ...,
        description="Drift model status: warming_up | active | degraded"
    )
    drift_offset_percent: Optional[float] = Field(
        None,
        description="Current estimated systematic offset as % of baseline (null if warming up)"
    )

    explanation: str = Field(
        ...,
        description="Human-readable summary of RUL indicators"
    )
