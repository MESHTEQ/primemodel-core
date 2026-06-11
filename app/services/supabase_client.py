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

def fetch_uplink_history_legacy(
    meter_id: str,
    client_id: str,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    [LEGACY] Fetch recent uplink records for a water meter from the "uplinks" table.

    Retained for backward compatibility with the legacy /analyse/legacy endpoint.
    New code should use fetch_sensor_history() which queries lorawan_uplinks.

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
            "Failed to fetch uplink history (legacy)",
            extra={"meter_id": meter_id, "client_id": client_id, "error": str(e)},
        )
        return []


def fetch_sensor_history(
    deveui: str,
    limit: int = 2000,
) -> List[Dict[str, Any]]:
    """
    Fetch recent uplink records for any LoRaWAN sensor from the lorawan_uplinks table.

    This is the primary history fetch function for the sensor-agnostic /analyse endpoint.
    Returns rows ordered chronologically (oldest first).

    Args:
        deveui: Device EUI string — used to filter rows.
        limit: Maximum number of rows to return.

    Returns:
        List of dicts with keys: deveui, decoded_payload (dict), created_at (str).
        Returns empty list on error or no data.
    """
    try:
        client = _get_client()
        # Supabase PostgREST hard-caps single requests at 500 rows by default.
        # We paginate in 500-row batches until we reach the limit or exhaust data.
        all_rows: List[Dict[str, Any]] = []
        page_size = 500
        offset = 0
        while len(all_rows) < limit:
            batch_size = min(page_size, limit - len(all_rows))
            response = (
                client.table("lorawan_uplinks")
                .select("deveui, decoded_payload, created_at")
                .eq("deveui", deveui)
                .order("created_at", desc=False)   # oldest first — no reversal needed
                .range(offset, offset + batch_size - 1)
                .execute()
            )
            batch = response.data or []
            all_rows.extend(batch)
            if len(batch) < batch_size:
                break  # no more rows
            offset += batch_size

        logger.info(
            "Sensor history fetched",
            extra={"deveui": deveui, "rows": len(all_rows)},
        )
        return all_rows
    except Exception as e:
        logger.error(
            "Failed to fetch sensor history",
            extra={"deveui": deveui, "error": str(e)},
        )
        return []


def fetch_analysis_results(
    deveui: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fetch recent analysis results for a device from the analysis_results table.

    Returns the most recent rows (newest first), selecting only the 10 columns
    needed by the admin scores endpoint.  This avoids pulling large jsonb blobs
    that are not required for the scores view.

    Schema verified read-only via Supabase MCP (2026-06-11) against project
    pkloaajhalichjopzlob.  Columns confirmed present: created_at, ensemble_score,
    layer1_scores, layer2_score, layer3_score, layer4_score, active_layers,
    anomaly_detected, readings_used, days_of_data.

    Args:
        deveui: Device EUI string (uppercase hex, no underscores).
        limit:  Maximum number of rows to return (default 20).

    Returns:
        List of dicts with the 10 selected columns, ordered newest first.
        Returns empty list on error or no data.
    """
    try:
        client = _get_client()
        response = (
            client.table("analysis_results")
            .select(
                "created_at,"
                "ensemble_score,"
                "layer1_scores,"
                "layer2_score,"
                "layer3_score,"
                "layer4_score,"
                "active_layers,"
                "anomaly_detected,"
                "readings_used,"
                "days_of_data"
            )
            .eq("deveui", deveui)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = response.data or []
        logger.info(
            "Analysis results fetched",
            extra={"deveui": deveui, "rows": len(rows)},
        )
        return rows
    except Exception as e:
        logger.error(
            "Failed to fetch analysis results",
            extra={"deveui": deveui, "error": str(e)},
        )
        return []


def write_analysis_result(
    deveui: str,
    result: dict,
) -> bool:
    """
    Write an analysis result to the analysis_results table.

    # TODO: Create analysis_results table in Supabase and enable write.
    Table schema (planned):
        id          bigint (auto)
        deveui      text
        created_at  timestamptz
        result      jsonb   ← full AnalyseResponse dict

    For now, logs the result and returns True without writing to Supabase.

    Args:
        deveui: Device EUI string.
        result: Dict representation of AnalyseResponse.

    Returns:
        True always (write is stubbed).
    """
    logger.info(
        "Analysis result (write stubbed — table not yet created)",
        extra={"deveui": deveui, "ensemble_score": result.get("ensemble_score"), "anomaly_detected": result.get("anomaly_detected")},
    )
    # TODO: Uncomment once analysis_results table is created in Supabase:
    # try:
    #     client = _get_client()
    #     client.table("analysis_results").insert({
    #         "deveui": deveui,
    #         "created_at": datetime.now(tz=timezone.utc).isoformat(),
    #         "result": result,
    #     }).execute()
    # except Exception as e:
    #     logger.error("Failed to write analysis result", extra={"deveui": deveui, "error": str(e)})
    #     return False
    return True


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
