"""
app/services/battery_rul.py
-----------------------------
Battery Remaining Useful Life (RUL) estimation.

Method:
    Linear regression on the battery_level time series to estimate the
    depletion rate (% per day). RUL = (current_level - eol_percent) / depletion_rate.

    Requires at least 3 data points to fit the regression.
    Returns None for RUL if insufficient data (graceful degradation).

Battery status severity thresholds (from config):
    ok:       rul_days >= battery_warning_days
    warning:  battery_critical_days <= rul_days < battery_warning_days
    critical: rul_days < battery_critical_days (or level already <= eol_percent)
"""

import numpy as np
from typing import Optional, Tuple, List

from app.utils.logger import get_logger

logger = get_logger(__name__)


def estimate_rul(
    battery_levels: List[float],
    timestamps_days: List[float],
    eol_percent: float = 20.0,
    warning_days: int = 30,
    critical_days: int = 7,
) -> Tuple[Optional[float], str]:
    """
    Estimate battery RUL from a history of (time, battery_level) readings.

    Args:
        battery_levels: List of battery percentage readings (most recent last).
        timestamps_days: List of elapsed days since first reading (same order as battery_levels).
        eol_percent: Battery level considered end-of-life.
        warning_days: RUL threshold for "warning" status.
        critical_days: RUL threshold for "critical" status.

    Returns:
        Tuple of (rul_days, status).
        rul_days: Estimated days remaining, or None if insufficient data.
        status: "ok" | "warning" | "critical"
    """
    current_level = battery_levels[-1] if battery_levels else 100.0

    # Already at or below EOL
    if current_level <= eol_percent:
        return 0.0, "critical"

    if len(battery_levels) < 3:
        # Not enough data to estimate rate — return None but classify by level
        if current_level < eol_percent + 10:
            return None, "warning"
        return None, "ok"

    # Fit linear regression: level = a + b * days
    x = np.array(timestamps_days, dtype=np.float64)
    y = np.array(battery_levels, dtype=np.float64)

    # Least-squares fit
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if abs(denominator) < 1e-9:
        # All readings at same time — can't fit
        return None, "ok"

    slope = numerator / denominator  # % per day

    if slope >= 0:
        # Battery not depleting (possibly replaced or noisy readings)
        logger.info("Battery slope non-negative — assuming stable", extra={"slope": slope})
        return None, "ok"

    depletion_rate = abs(slope)  # % per day (positive)
    rul_days = (current_level - eol_percent) / depletion_rate

    rul_days = max(0.0, float(rul_days))

    # Classify status
    if rul_days < critical_days:
        status = "critical"
    elif rul_days < warning_days:
        status = "warning"
    else:
        status = "ok"

    logger.debug(
        "Battery RUL computed",
        extra={
            "rul_days": round(rul_days, 1),
            "status": status,
            "depletion_rate_pct_per_day": round(depletion_rate, 4),
            "current_level": current_level,
        },
    )

    return rul_days, status


def classify_from_level(
    battery_level: float,
    eol_percent: float = 20.0,
    warning_days: int = 30,
    critical_days: int = 7,
) -> str:
    """
    Classify battery status based solely on current level when history is unavailable.

    A simple fallback: anything within 20% of EOL is "warning", at EOL is "critical".

    Args:
        battery_level: Current battery percentage.
        eol_percent: End-of-life percentage.
        warning_days: Not used directly — included for API consistency.
        critical_days: Not used directly.

    Returns:
        "ok" | "warning" | "critical"
    """
    if battery_level <= eol_percent:
        return "critical"
    elif battery_level <= eol_percent + 15:
        return "warning"
    return "ok"
