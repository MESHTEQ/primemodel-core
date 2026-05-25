"""
app/services/decoder_panda.py
------------------------------
STUB — Panda POC meter LoRaWAN payload decoder.

STATUS: Payload format PENDING official spec from Panda hardware supplier.
        This stub returns mock/zero values so the rest of the pipeline runs
        during the Panda POC testing phase.

TECHNICAL DEBT: TD-PANDA-001 (HIGH)
    Replace byte-offset constants and decode() implementation when the
    official Panda payload specification is received.
    Expected fields: flow_rate, cumulative_volume, battery_level, alarm_flags.
    Contact: Meshteq hardware integration team.

When the real spec arrives:
1. Replace the constants below with actual byte offsets and scale factors.
2. Remove the "STUB" warning log.
3. Update tests/test_analyse_endpoint.py with a real hex sample.
"""

from app.utils.logger import get_logger

logger = get_logger(__name__)

# --- STUB constants (replace with real offsets from Panda spec) ---
_STUB_FLOW_RATE_OFFSET = None       # byte index in payload
_STUB_FLOW_RATE_SCALE = None        # divide raw int by this to get m³/h
_STUB_VOLUME_OFFSET = None
_STUB_VOLUME_SCALE = None
_STUB_BATTERY_OFFSET = None
_STUB_ALARM_OFFSET = None


def decode(raw_payload: str) -> dict:
    """
    Decode a Panda POC meter hex payload.

    Args:
        raw_payload: Hex-encoded LoRaWAN payload string (e.g. "AABB001234").
                     May be empty or None during integration testing.

    Returns:
        dict with keys:
            flow_rate         (float)  — m³/h
            cumulative_volume (float)  — m³
            battery_level     (float)  — %
            alarm_flags       (int)    — bitmask

    Notes:
        Currently returns zeros because the Panda payload spec has not been
        received.  The /analyse endpoint accepts flow_rate etc. directly in
        the request body, so this decoder is only invoked when raw_payload
        is present and the caller wants to re-derive values from bytes.
    """
    logger.warning(
        "Panda decoder is a STUB — returning zero values. "
        "Replace when official payload spec received. TD-PANDA-001"
    )
    # STUB: when spec arrives, replace this block with real byte unpacking:
    # e.g.
    #   raw_bytes = bytes.fromhex(raw_payload or "")
    #   flow_raw  = int.from_bytes(raw_bytes[0:2], "big")
    #   flow_rate = flow_raw / _STUB_FLOW_RATE_SCALE
    #   ...
    return {
        "flow_rate": 0.0,
        "cumulative_volume": 0.0,
        "battery_level": 0.0,
        "alarm_flags": 0,
    }
