"""
app/models/db/tables.py
------------------------
Reference definitions of the Supabase PostgreSQL tables used by PrimeModel.

These are NOT ORM models — PrimeModel communicates with Supabase via the
supabase-py client (REST/PostgREST), not SQLAlchemy.

This file serves as the authoritative schema reference so developers and the
UTP facilities team can understand exactly what is stored without reading SQL.

IMPORTANT: The actual tables must be created in Supabase via the SQL editor
or migrations.  Copy the CREATE TABLE statements from the docstrings below.

All tables are scoped per-client.  In the multi-tenant model each client has
their own Supabase project, so there is no tenant discriminator column inside
a shared table — the Supabase URL/key pair IS the tenant boundary.
"""

from typing import TypedDict, Optional


# ---------------------------------------------------------------------------
# meter_uplinks
# ---------------------------------------------------------------------------
# CREATE TABLE meter_uplinks (
#     id              BIGSERIAL PRIMARY KEY,
#     meter_id        TEXT NOT NULL,
#     client_id       TEXT NOT NULL,
#     meter_type      TEXT NOT NULL,
#     timestamp       TIMESTAMPTZ NOT NULL,
#     flow_rate       DOUBLE PRECISION NOT NULL,
#     cumulative_vol  DOUBLE PRECISION NOT NULL,
#     battery_level   DOUBLE PRECISION NOT NULL,
#     alarm_flags     INTEGER NOT NULL DEFAULT 0,
#     raw_payload     TEXT,
#     created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
# );
# CREATE INDEX idx_uplinks_meter_ts ON meter_uplinks (meter_id, timestamp DESC);

class MeterUplinkRow(TypedDict):
    """TypedDict mirror of the meter_uplinks table row."""
    id: int
    meter_id: str
    client_id: str
    meter_type: str
    timestamp: str          # ISO-8601 UTC string
    flow_rate: float
    cumulative_vol: float
    battery_level: float
    alarm_flags: int
    raw_payload: Optional[str]
    created_at: str


# ---------------------------------------------------------------------------
# anomaly_scores
# ---------------------------------------------------------------------------
# CREATE TABLE anomaly_scores (
#     id                  BIGSERIAL PRIMARY KEY,
#     meter_id            TEXT NOT NULL,
#     client_id           TEXT NOT NULL,
#     timestamp           TIMESTAMPTZ NOT NULL,
#     anomaly_score       DOUBLE PRECISION NOT NULL,
#     is_anomaly          BOOLEAN NOT NULL DEFAULT FALSE,
#     mnf_flag            BOOLEAN NOT NULL DEFAULT FALSE,
#     mnf_value           DOUBLE PRECISION,
#     burst_detected      BOOLEAN NOT NULL DEFAULT FALSE,
#     leak_probability    DOUBLE PRECISION NOT NULL DEFAULT 0.0,
#     leak_severity       TEXT NOT NULL DEFAULT 'none',
#     battery_rul_days    DOUBLE PRECISION,
#     drift_rul_days      DOUBLE PRECISION,
#     drift_rul_status    TEXT NOT NULL DEFAULT 'warming_up',
#     confidence          DOUBLE PRECISION NOT NULL DEFAULT 0.0,
#     model_status        TEXT NOT NULL DEFAULT 'cold_start',
#     explanation         TEXT NOT NULL DEFAULT '',
#     created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
# );
# CREATE INDEX idx_scores_meter_ts ON anomaly_scores (meter_id, timestamp DESC);
# -- Enable Supabase Realtime on this table for dashboard streaming:
# ALTER TABLE anomaly_scores REPLICA IDENTITY FULL;

class AnomalyScoreRow(TypedDict):
    """TypedDict mirror of the anomaly_scores table row."""
    id: int
    meter_id: str
    client_id: str
    timestamp: str
    anomaly_score: float
    is_anomaly: bool
    mnf_flag: bool
    mnf_value: Optional[float]
    burst_detected: bool
    leak_probability: float
    leak_severity: str
    battery_rul_days: Optional[float]
    drift_rul_days: Optional[float]
    drift_rul_status: str
    confidence: float
    model_status: str
    explanation: str
    created_at: str


# ---------------------------------------------------------------------------
# meter_registry
# ---------------------------------------------------------------------------
# CREATE TABLE meter_registry (
#     meter_id        TEXT PRIMARY KEY,
#     client_id       TEXT NOT NULL,
#     meter_type      TEXT NOT NULL,
#     location_label  TEXT,
#     building_block  TEXT,
#     is_active       BOOLEAN NOT NULL DEFAULT TRUE,
#     installed_at    TIMESTAMPTZ,
#     notes           TEXT,
#     created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
# );

class MeterRegistryRow(TypedDict):
    """TypedDict mirror of the meter_registry table row."""
    meter_id: str
    client_id: str
    meter_type: str
    location_label: Optional[str]
    building_block: Optional[str]
    is_active: bool
    installed_at: Optional[str]
    notes: Optional[str]
    created_at: str
