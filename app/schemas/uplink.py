"""
app/schemas/uplink.py
----------------------
Pydantic schemas for the POST /analyse endpoint.

AnalyseRequest  — single meter uplink payload from ThingPark via Supabase Edge Function.
AnalyseResponse — scored result returned to the Edge Function, which writes to anomaly_scores table.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class MeterType(str, Enum):
    PANDA = "panda"
    BOVE_B39 = "bove_b39"


class LeakSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PatternType(str, Enum):
    NORMAL = "normal"
    BURST = "burst"
    BACKGROUND = "background"
    INTERMITTENT = "intermittent"


class LayerStatus(str, Enum):
    WARMING_UP = "warming_up"
    ACTIVE = "active"


class DriftRulStatus(str, Enum):
    WARMING_UP = "warming_up"
    ACTIVE = "active"
    DEGRADED = "degraded"


class BatteryRulStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


class ModelLayerStatuses(BaseModel):
    """
    Activation status of each model layer for this meter.
    days_until_* fields show how many more days until activation (0 means active).
    """
    statistical: LayerStatus
    lstm_autoencoder: LayerStatus
    lstm_forecast: LayerStatus
    cnn_pattern: LayerStatus
    days_until_autoencoder: int = Field(0, ge=0)
    days_until_forecast: int = Field(0, ge=0)
    days_until_cnn: int = Field(0, ge=0)


class AnalyseRequest(BaseModel):
    """
    Single uplink from a water meter, forwarded by the Supabase Edge Function.
    All fields from ThingPark are mapped here before being passed to services.
    """
    meter_id: str = Field(..., description="Unique meter identifier (e.g. device EUI)")
    client_id: str = Field(..., description="Tenant identifier — scopes all DB queries")
    meter_type: MeterType = Field(..., description="Meter hardware type determines decoder")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp of the uplink")
    flow_rate: float = Field(..., ge=0.0, description="Instantaneous flow rate (m³/h)")
    cumulative_volume: float = Field(..., ge=0.0, description="Cumulative volume counter reading (m³)")
    battery_level: float = Field(..., ge=0.0, le=100.0, description="Battery charge percentage")
    alarm_flags: int = Field(0, ge=0, description="Bitmask of meter alarm flags (0 = no alarm)")
    raw_payload: Optional[str] = Field(None, description="Hex-encoded LoRaWAN payload (optional, for decoder audit)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "meter_id": "0018B2000000ABCD",
                "client_id": "utp",
                "meter_type": "panda",
                "timestamp": "2026-05-12T02:15:00Z",
                "flow_rate": 0.42,
                "cumulative_volume": 12345.67,
                "battery_level": 85.0,
                "alarm_flags": 0,
                "raw_payload": "AABB001234",
            }
        }
    }


class AnalyseResponse(BaseModel):
    """
    Scored analysis result returned per uplink.
    Written to the anomaly_scores table in Supabase by the Edge Function.
    """
    meter_id: str
    timestamp: str = Field(..., description="Echo of the uplink timestamp")

    # --- Anomaly detection (Isolation Forest + ensemble) ---
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Normalised anomaly score (0=normal, 1=max anomaly)")
    is_anomaly: bool = Field(..., description="True if anomaly_score exceeds trained threshold")

    # --- MNF analysis ---
    mnf_flag: bool = Field(..., description="True if MNF is elevated above baseline")
    mnf_value: Optional[float] = Field(None, description="Mean flow during MNF window (null outside window or insufficient data)")

    # --- Burst detection ---
    burst_detected: bool = Field(..., description="True if a flow-rate spike exceeded burst threshold")

    # --- Neural network ensemble ---
    leak_probability: float = Field(..., ge=0.0, le=1.0, description="Composite leak probability [0, 1]")
    leak_severity: LeakSeverity = Field(..., description="Qualitative leak severity classification")
    pattern_type: PatternType = Field(..., description="CNN-classified flow pattern type")

    # --- RUL indicators ---
    battery_rul_days: Optional[float] = Field(None, description="Estimated battery remaining useful life in days")
    battery_rul_status: BatteryRulStatus = Field(..., description="Battery RUL status classification")
    drift_rul_days: Optional[float] = Field(None, description="Estimated metrological drift RUL in days (null if warming up)")
    drift_rul_status: DriftRulStatus = Field(..., description="Drift RUL computation status")

    # --- Model layer statuses ---
    model_status: ModelLayerStatuses = Field(..., description="Activation status of each model layer")

    # --- Result metadata ---
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in the scored result")
    explanation: str = Field(..., description="Human-readable summary of the analysis result")

    model_config = {
        "json_schema_extra": {
            "example": {
                "meter_id": "0018B2000000ABCD",
                "timestamp": "2026-05-12T02:15:00Z",
                "anomaly_score": 0.12,
                "is_anomaly": False,
                "mnf_flag": False,
                "mnf_value": 0.38,
                "burst_detected": False,
                "leak_probability": 0.08,
                "leak_severity": "none",
                "pattern_type": "normal",
                "battery_rul_days": 312.5,
                "battery_rul_status": "ok",
                "drift_rul_days": None,
                "drift_rul_status": "warming_up",
                "model_status": {
                    "statistical": "active",
                    "lstm_autoencoder": "warming_up",
                    "lstm_forecast": "warming_up",
                    "cnn_pattern": "warming_up",
                    "days_until_autoencoder": 18,
                    "days_until_forecast": 48,
                    "days_until_cnn": 78,
                },
                "confidence": 0.72,
                "explanation": "Flow within normal range. No MNF elevation or burst detected. Statistical layer only — NN layers warming up.",
            }
        }
    }
