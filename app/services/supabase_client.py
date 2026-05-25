"""
app/services/supabase_client.py
---------------------------------
Supabase client — multi-tenant aware read/write operations.

All queries are scoped by client_id to enforce tenant isolation.
Supabase uses the service-role key for server-side operations (bypasses RLS).

Security note:
    The SUPABASE_SERVICE_KEY is a server-side secret.
    It is NEVER exposed to any frontend or client-facing endpoint.
    Never log the key value — the logger will log "***" instead.

Tables used:
    uplinks:          Raw meter uplink history per client
    anomaly_scores:   Output from /analyse, written back here
    meter_registry:   Meter metadata (topology, zone_id, etc.)
    battery_history:  Battery readings for RUL regression

Multi-tenant design:
    Each client has a separate Supabase DB (or schema — depending on deployment).
    client_id is always passed as a filter in every query.
    The Edge Function passes client_id from its own verified context.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy-initialise Supabase client to avoid import errors at startup
# if SUPABASE_URL / SUPABASE_SERVICE_KEY are not yet set in the environment.
_client = None


def _get_client():
    """
    Return the Supabase client instance, initialising on first call.
    Uses the service-role key — this client bypasses all RLS policies.
    """
    global _client
    if _client is None:
        from supabase import create_client, Client
        settings = get_settings()
        logger.info("Initialising Supabase client", extra={"url": settings.supabase_url})
        _client = create_client(settings.supabase_url, settings.supabase_service_key)
    return _client


# ---------------------------------------------------------------------------
# Uplink history
# ---------------------------------------------------------------------------

def fetch_uplink_history(
    meter_id: str,
    client_id: str,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    Fetch recent uplink records for a meter from Supabase.

    Args:
        meter_id: Meter identifier.
        client_id: Tenant identifier — scopes the query.
        limit: Maximum number of rows to return (ordered by timestamp desc).

    Returns:
        List of dicts with at least: timestamp, flow_rate, cumulative_volume,
        battery_level, alarm_flags.
    """
    try:
        client = _get_client()
        response = (
            client.table("uplinks")
            .select("timestamp, flow_rate, cumulative_volume, battery_level, alarm_flags")
            .eq("client_id", client_id)
            .eq("meter_id", meter_id)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        rows = response.data or []
        # Return in chronological order (oldest first)
        return list(reversed(rows))
    except Exception as e:
        logger.error(
            "Failed to fetch uplink history",
            extra={"meter_id": meter_id, "client_id": client_id, "error": str(e)},
        )
        return []


def insert_uplink(
    meter_id: str,
    client_id: str,
    timestamp: str,
    flow_rate: float,
    cumulative_volume: float,
    battery_level: float,
    alarm_flags: int,
    raw_payload: Optional[str] = None,
) -> bool:
    """
    Insert a new uplink record into the uplinks table.

    This is called by the /analyse endpoint before analysis begins,
    so that the uplink is persisted even if analysis fails.

    Args:
        meter_id: Meter identifier.
        client_id: Tenant identifier.
        timestamp: ISO-8601 timestamp.
        flow_rate: Flow rate in m³/h.
        cumulative_volume: Cumulative volume in m³.
        battery_level: Battery percentage.
        alarm_flags: Alarm bitmask.
        raw_payload: Optional hex payload string.

    Returns:
        True if insert succeeded, False otherwise.
    """
    try:
        client = _get_client()
        client.table("uplinks").insert({
            "meter_id": meter_id,
            "client_id": client_id,
            "timestamp": timestamp,
            "flow_rate": flow_rate,
            "cumulative_volume": cumulative_volume,
            "battery_level": battery_level,
            "alarm_flags": alarm_flags,
            "raw_payload": raw_payload,
        }).execute()
        return True
    except Exception as e:
        logger.error(
            "Failed to insert uplink",
            extra={"meter_id": meter_id, "client_id": client_id, "error": str(e)},
        )
        return False


# ---------------------------------------------------------------------------
# Anomaly scores
# ---------------------------------------------------------------------------

def write_anomaly_score(
    meter_id: str,
    client_id: str,
    score_record: Dict[str, Any],
) -> bool:
    """
    Write an anomaly score record to the anomaly_scores table.

    The score_record should match the AnalyseResponse schema.
    The Edge Function reads from this table via Supabase Realtime.

    Args:
        meter_id: Meter identifier.
        client_id: Tenant identifier.
        score_record: Dict containing all fields from AnalyseResponse.

    Returns:
        True if write succeeded, False otherwise.
    """
    try:
        client = _get_client()
        payload = {
            "meter_id": meter_id,
            "client_id": client_id,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            **score_record,
        }
        client.table("anomaly_scores").insert(payload).execute()
        return True
    except Exception as e:
        logger.error(
            "Failed to write anomaly score",
            extra={"meter_id": meter_id, "client_id": client_id, "error": str(e)},
        )
        return False


# ---------------------------------------------------------------------------
# Meter registry
# ---------------------------------------------------------------------------

def fetch_meter_metadata(
    meter_id: str,
    client_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch meter metadata from the meter_registry table.

    Returns dict with at least: meter_id, zone_id, meter_type, topology_role.
    Returns None if meter not found.
    """
    try:
        client = _get_client()
        response = (
            client.table("meter_registry")
            .select("*")
            .eq("client_id", client_id)
            .eq("meter_id", meter_id)
            .single()
            .execute()
        )
        return response.data
    except Exception as e:
        logger.warning(
            "Meter metadata not found",
            extra={"meter_id": meter_id, "client_id": client_id, "error": str(e)},
        )
        return None


# ---------------------------------------------------------------------------
# Battery history
# ---------------------------------------------------------------------------

def fetch_battery_history(
    meter_id: str,
    client_id: str,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """
    Fetch battery level history for RUL regression.

    Returns list of dicts with keys: timestamp, battery_level.
    Ordered chronologically (oldest first).
    """
    try:
        client = _get_client()
        response = (
            client.table("uplinks")
            .select("timestamp, battery_level")
            .eq("client_id", client_id)
            .eq("meter_id", meter_id)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        rows = response.data or []
        return list(reversed(rows))
    except Exception as e:
        logger.error(
            "Failed to fetch battery history",
            extra={"meter_id": meter_id, "client_id": client_id, "error": str(e)},
        )
        return []
