"""
app/services/decoder_bove.py
-----------------------------
STUB — Bove B39 VW-M ultrasonic LoRaWAN water meter payload decoder.

STATUS: Payload format PENDING — decoder specification arrives with physical
        hardware.  This stub returns zero values so the pipeline runs end-to-end
        before the hardware is on-site.

TECHNICAL DEBT: TD-BOVE-001 (HIGH)
    Replace decode() when Bove B39 decoder documentation arrives with hardware.
    The Bove B39 is an ultrasonic meter with LoRaWAN class A output.
    Expected payload fields: flow_rate (m³/h), cumulative_volume (m³),
    battery_level (%), alarm_flags (bitmask).
    Contact: AR Technology hardware procurement team.

When the real spec arrives:
1. Consult Bove B39 integration guide (typically 12–20 byte payload).
2. Implement byte extraction below.
3. Add a test hex sample to tests/test_analyse_endpoint.py.
4. Remove TD-BOVE-001 from MEMORY.md once validated with live hardware.
"""

from app.utils.logger import get_logger

logger = get_logger(__name__)

# --- STUB constants (replace with real offsets from Bove B39 spec) ---
_STUB_FLOW_OFFSET = None
_STUB_FLOW_SCALE = None
_STUB_VOLUME_OFFSET = None
_STUB_VOLUME_SCALE = None
_STUB_BATTERY_OFFSET = None
_STUB_ALARM_OFFSET = None


def decode(raw_payload: str) -> dict:
    """
    Decode a Bove B39 VW-M ultrasonic meter hex payload.

    Args:
        raw_payload: Hex-encoded LoRaWAN payload string.
                     May be empty or None if not yet available.

    Returns:
        dict with keys:
            flow_rate         (float)  — m³/h
            cumulative_volume (float)  — m³
            battery_level     (float)  — %
            alarm_flags       (int)    — bitmask

    Notes:
        STUB implementation — returns zeros until spec received. TD-BOVE-001.
    """
    logger.warning(
        "Bove B39 decoder is a STUB — returning zero values. "
        "Replace when hardware + decoder spec is received. TD-BOVE-001"
    )
    # STUB: replace with real byte unpacking when spec received
    return {
        "flow_rate": 0.0,
        "cumulative_volume": 0.0,
        "battery_level": 0.0,
        "alarm_flags": 0,
    }
