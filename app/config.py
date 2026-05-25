"""
app/config.py
-------------
Centralised configuration for PrimeModel AI Engine.
All values sourced from environment variables via Pydantic Settings.
Never hardcode secrets — all sensitive values live in .env only.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # --- Supabase ---
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_service_key: str = Field(..., description="Supabase service-role key (never anon key)")

    # --- App ---
    app_env: str = Field("development", description="production | development | test")
    log_level: str = Field("INFO", description="Logging level")
    model_store_path: str = Field("./models_store", description="Path for serialised models and baselines")

    # --- MNF Config ---
    mnf_window_start: int = Field(2, description="Hour (24h) start of MNF window")
    mnf_window_end: int = Field(4, description="Hour (24h) end of MNF window")
    mnf_baseline_days: int = Field(30, description="Days of history for MNF baseline")
    cusum_threshold_sigma: float = Field(3.0, description="CUSUM alert threshold in std deviations")
    burst_threshold_sigma: float = Field(3.0, description="Burst detection threshold in std deviations")
    ewma_lambda: float = Field(0.2, description="EWMA smoothing factor (0 < lambda <= 1)")

    # --- Model Activation Thresholds ---
    cold_start_days: int = Field(30, description="Days before Isolation Forest trains")
    lstm_ae_activation_days: int = Field(30, description="Days of data before LSTM Autoencoder activates")
    lstm_forecast_activation_days: int = Field(60, description="Days of data before LSTM Forecasting activates")
    cnn_activation_days: int = Field(90, description="Days of data before 1D CNN activates")
    retrain_interval_days: int = Field(7, description="How often to retrain models")
    isolation_forest_contamination: float = Field(0.05, description="Expected anomaly fraction for IsolationForest")

    # --- Battery RUL ---
    battery_warning_days: int = Field(30, description="Battery RUL threshold for warning severity")
    battery_critical_days: int = Field(7, description="Battery RUL threshold for critical severity")
    battery_eol_percent: float = Field(20.0, description="Battery % level considered end-of-life")

    # --- Drift RUL ---
    drift_accuracy_threshold_percent: float = Field(
        2.0,
        description="Metrological accuracy class limit (±%). RUL = estimated days until drift exceeds this."
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Return a cached Settings singleton.
    Using lru_cache ensures we only parse .env once per process lifetime.
    """
    return Settings()
