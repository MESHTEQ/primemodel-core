"""
app/services/decoder_registry.py
----------------------------------
Sensor payload decoder registry.

Each decoder takes a decoded_payload dict (from lorawan_uplinks.decoded_payload)
and returns a dict of {parameter_name: float} containing only numeric values
suitable for time-series analysis.

Non-numeric fields (strings, bools, nulls) are silently skipped.

Usage:
    from app.services.decoder_registry import get_decoder
    decoder = get_decoder("temp_humidity")
    numeric = decoder({"temperature": 27.7, "humidity": 77.5})
    # → {"temperature": 27.7, "humidity": 77.5}
"""

from typing import Dict, Any, Callable
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

DecoderFn = Callable[[Dict[str, Any]], Dict[str, float]]


# ---------------------------------------------------------------------------
# Built-in decoders
# ---------------------------------------------------------------------------

def _decode_temp_humidity(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Decoder for temperature/humidity sensors.
    Expected keys: temperature (float), humidity (float).
    Example: {"temperature": 27.7, "humidity": 77.5}
    """
    result = {}
    if "temperature" in payload and isinstance(payload["temperature"], (int, float)):
        result["temperature"] = float(payload["temperature"])
    if "humidity" in payload and isinstance(payload["humidity"], (int, float)):
        result["humidity"] = float(payload["humidity"])
    return result


def _decode_distance(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Decoder for distance/ultrasonic sensors.
    Expected keys: distance (float), battery (float, optional).
    Example: {"battery": 100, "distance": 112, "position": "tilt"}
    Non-numeric fields like "position" are silently skipped.
    """
    result = {}
    if "distance" in payload and isinstance(payload["distance"], (int, float)):
        result["distance"] = float(payload["distance"])
    if "battery" in payload and isinstance(payload["battery"], (int, float)):
        result["battery"] = float(payload["battery"])
    return result


def _decode_bove_water_meter(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Stub decoder for Bove B39 water meters.
    Full decoder spec arrives with hardware — see TD-BOVE-001.
    """
    logger.warning(
        "Bove water meter decoder is a stub — no numeric values extracted. "
        "Awaiting official payload spec (TD-BOVE-001)."
    )
    return {}


def _decode_panda_water_meter(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Stub decoder for Panda POC water meters.
    Replace when official payload spec received — see TD-PANDA-001.
    """
    logger.warning(
        "Panda water meter decoder is a stub — no numeric values extracted. "
        "Awaiting official payload spec (TD-PANDA-001)."
    )
    return {}


def _decode_generic(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Generic fallback decoder.
    Iterates all keys in decoded_payload and extracts any value that is int or float.
    Skips strings, bools, nulls, and nested dicts/lists silently.

    Example:
        {"alarm": "power down"} → {}
        {"battery": 100, "leakage_status": "normal"} → {"battery": 100.0}
        {"temperature": 27.7, "humidity": 77.5} → {"temperature": 27.7, "humidity": 77.5}
    """
    result = {}
    for key, value in payload.items():
        # Explicitly exclude bools — bool is a subclass of int in Python
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            result[key] = float(value)
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, DecoderFn] = {
    "temp_humidity":    _decode_temp_humidity,
    "distance":         _decode_distance,
    "bove_water_meter": _decode_bove_water_meter,
    "panda_water_meter":_decode_panda_water_meter,
    "generic":          _decode_generic,
}


def get_decoder(device_type: str) -> DecoderFn:
    """
    Return the decoder function for the given device_type string.
    Falls back to the generic decoder if the type is not registered.

    Args:
        device_type: String identifier for the device type (e.g. "temp_humidity").

    Returns:
        Callable that accepts decoded_payload dict and returns {param: float}.
    """
    decoder = _REGISTRY.get(device_type)
    if decoder is None:
        logger.warning(
            "No decoder registered for device_type — falling back to generic",
            extra={"device_type": device_type},
        )
        return _REGISTRY["generic"]
    return decoder


def list_registered_types() -> list:
    """Return list of all registered device type strings."""
    return list(_REGISTRY.keys())
