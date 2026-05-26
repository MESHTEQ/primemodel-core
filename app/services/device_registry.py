# TODO: Move to Supabase device_registry table — Phase 2
"""
app/services/device_registry.py
---------------------------------
Maps device EUI to device type and metadata.

Currently an in-memory dict. Will be migrated to a Supabase
device_registry table in Phase 2 to allow dynamic registration
without redeployment.

Usage:
    from app.services.device_registry import get_device_info
    info = get_device_info("24E124136D355878")
    # → {"device_type": "temp_humidity", "name": "Temp & Humidity Sensor", "location": "Miri"}
"""

from typing import Dict, Any
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Device map — keyed by deveui (uppercase)
# ---------------------------------------------------------------------------

DEVICE_MAP: Dict[str, Dict[str, Any]] = {
    "24E124136D355878": {
        "device_type": "temp_humidity",
        "name": "Temp & Humidity Sensor",
        "location": "Miri",
    },
    "24E124713D321914": {
        "device_type": "distance",
        "name": "Distance Sensor",
        "location": "Miri",
    },
    "24E124747D260328": {
        "device_type": "power_monitor",
        "name": "Power Monitor",
        "location": "Miri",
    },
    "24E124993D091615": {
        "device_type": "leak_detector",
        "name": "Leak Detector",
        "location": "Miri",
    },
}

# Default returned when a deveui is not in the map
_UNKNOWN_DEVICE: Dict[str, Any] = {
    "device_type": "generic",
    "name": "Unknown Device",
}


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def get_device_info(deveui: str) -> Dict[str, Any]:
    """
    Return device metadata for the given deveui.

    Lookup is case-insensitive (normalised to uppercase).
    Returns a default "generic" entry if the deveui is not registered.

    Args:
        deveui: Device EUI string (e.g. "24E124136D355878").

    Returns:
        Dict with at least: device_type (str), name (str).
        May also include: location (str), and any future fields.
    """
    normalised = deveui.upper().strip()
    info = DEVICE_MAP.get(normalised)
    if info is None:
        logger.warning(
            "Device EUI not found in registry — using generic decoder",
            extra={"deveui": deveui},
        )
        return dict(_UNKNOWN_DEVICE)
    return dict(info)


def list_registered_devices() -> list:
    """Return list of all registered deveui strings."""
    return list(DEVICE_MAP.keys())
