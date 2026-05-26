"""
app/schemas/analyse.py
-----------------------
Pydantic schemas for the sensor-agnostic POST /analyse endpoint.

These replace the water-meter-specific schemas in uplink.py for the new
generalised analysis pipeline. The legacy schemas in uplink.py are retained
for the /analyse/legacy endpoint.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any


class AnalyseRequest(BaseModel):
    """
    Request body for the sensor-agnostic POST /analyse endpoint.

    Only deveui is required. The endpoint looks up device type automatically
    from the device registry using the deveui.
    """
    deveui: str = Field(
        ...,
        description="Device EUI — used to fetch history and look up device type",
    )
    client_id: str = Field(
        "meshteq",
        description="Tenant identifier — used for multi-tenant scoping (default: meshteq)",
    )
    force_layers: List[str] = Field(
        default_factory=list,
        description=(
            "Optional override — list of layer names to force-run regardless of activation threshold. "
            "Valid values: 'statistical', 'lstm_ae', 'lstm_forecast', 'cnn'. "
            "Empty list = use normal activation thresholds."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "deveui": "24E124136D355878",
                "client_id": "meshteq",
                "force_layers": [],
            }
        }
    }


class AnalyseResponse(BaseModel):
    """
    Analysis result for a single sensor device.

    Contains per-parameter anomaly scores from each active model layer,
    an ensemble score, and metadata about how many readings and days of data
    were used.
    """

    # --- Device identity ---
    deveui: str = Field(..., description="Device EUI that was analysed")
    device_type: str = Field(..., description="Device type resolved from device registry")

    # --- Data coverage ---
    parameters_analysed: List[str] = Field(
        ...,
        description="List of numeric parameter names extracted from decoded_payload (e.g. ['temperature', 'humidity'])",
    )
    readings_used: int = Field(
        ...,
        ge=0,
        description="Number of uplink rows used in this analysis",
    )
    days_of_data: float = Field(
        ...,
        ge=0.0,
        description="Calendar days between the first and last reading used",
    )

    # --- Layer scores ---
    layer1_scores: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Statistical layer anomaly score per parameter. "
            "Keys are parameter names (e.g. 'temperature'), values are normalised scores [0, 1]."
        ),
    )
    layer2_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="LSTM Autoencoder anomaly score [0, 1]. Null if layer not yet active (requires 30 days of data).",
    )
    layer3_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="LSTM Forecast anomaly score [0, 1]. Null if layer not yet active (requires 60 days of data).",
    )
    layer4_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="CNN Pattern anomaly score [0, 1]. Null if layer not yet active (requires 90 days of data).",
    )

    # --- Ensemble result ---
    ensemble_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Weighted ensemble score across all active layers [0, 1]. Null if no layers active.",
    )
    anomaly_detected: bool = Field(
        ...,
        description="True if ensemble_score exceeds the anomaly detection threshold.",
    )
    anomaly_details: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Optional detail dict describing which signals triggered the anomaly flag. "
            "Keys vary by device type and active layers."
        ),
    )

    # --- Metadata ---
    analysis_timestamp: str = Field(
        ...,
        description="ISO-8601 UTC timestamp of when this analysis was computed",
    )
    active_layers: List[str] = Field(
        default_factory=list,
        description="List of layer names that ran in this analysis (e.g. ['statistical', 'lstm_ae'])",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "deveui": "24E124136D355878",
                "device_type": "temp_humidity",
                "parameters_analysed": ["temperature", "humidity"],
                "readings_used": 142,
                "days_of_data": 14.3,
                "layer1_scores": {"temperature": 0.12, "humidity": 0.08},
                "layer2_score": None,
                "layer3_score": None,
                "layer4_score": None,
                "ensemble_score": 0.10,
                "anomaly_detected": False,
                "anomaly_details": None,
                "analysis_timestamp": "2026-05-26T18:00:00Z",
                "active_layers": ["statistical"],
            }
        }
    }
