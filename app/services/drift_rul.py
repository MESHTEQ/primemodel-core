"""
app/services/drift_rul.py
--------------------------
Metrological drift RUL estimation.

Water meters drift in accuracy over time due to bearing wear and particulate
contamination. The drift manifests as a systematic positive offset in the
cumulative volume counter (meter under-registers real flow).

Detection method:
    Compute a daily mean flow from uplinks and compare it against the
    established baseline mean flow. A growing positive offset (meter reads
    less than expected) indicates forward drift.

    drift_offset_percent = (baseline_mean - current_smoothed_mean) / baseline_mean * 100
    (Positive = meter reading lower than baseline = under-registration = drift)

    drift_rate_per_day is estimated from a rolling regression on the offset series.
    RUL = (accuracy_threshold_percent - current_offset_percent) / drift_rate_per_day

Status:
    warming_up: fewer than mnf_baseline_days of data
    active:     model operating normally
    degraded:   offset already exceeds accuracy threshold

IMPORTANT: Drift cannot be measured without pressure — we infer it from cumulative
volume trends against the baseline. This is an estimation, not a calibration.
Label it clearly in outputs.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum days of offset history before RUL regression is meaningful
MIN_HISTORY_FOR_RUL = 14


def compute_offset(
    baseline_mean_flow: float,
    current_smoothed_flow: float,
) -> float:
    """
    Compute the current systematic offset as a percentage of baseline.

    Positive value = meter reading lower than baseline (under-registration).

    Args:
        baseline_mean_flow: Mean flow during the baseline period.
        current_smoothed_flow: EWMA-smoothed current mean flow.

    Returns:
        Offset as percentage (float). 0.0 if baseline is zero.
    """
    if baseline_mean_flow == 0.0:
        return 0.0
    return float((baseline_mean_flow - current_smoothed_flow) / baseline_mean_flow * 100.0)


def estimate_rul(
    offset_history: List[float],
    accuracy_threshold_percent: float = 2.0,
) -> Tuple[Optional[float], str]:
    """
    Estimate drift RUL from a history of daily offset values.

    Args:
        offset_history: List of daily offset_percent values (most recent last).
                        Must have at least MIN_HISTORY_FOR_RUL entries.
        accuracy_threshold_percent: Maximum allowable drift (%).

    Returns:
        Tuple of (rul_days, status).
        rul_days: Days until drift exceeds threshold, or None if insufficient history.
        status: "warming_up" | "active" | "degraded"
    """
    current_offset = offset_history[-1] if offset_history else 0.0

    if abs(current_offset) >= accuracy_threshold_percent:
        logger.warning(
            "Meter drift exceeds accuracy threshold",
            extra={"current_offset": current_offset, "threshold": accuracy_threshold_percent},
        )
        return 0.0, "degraded"

    if len(offset_history) < MIN_HISTORY_FOR_RUL:
        return None, "warming_up"

    # Fit linear regression on offset trend
    x = np.arange(len(offset_history), dtype=np.float64)
    y = np.array(offset_history, dtype=np.float64)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if abs(denominator) < 1e-9 or abs(numerator) < 1e-9:
        # No trend detectable
        return None, "active"

    drift_rate = numerator / denominator  # percent per day

    if drift_rate <= 0:
        # Offset is not growing — drift not progressing
        return None, "active"

    remaining = accuracy_threshold_percent - current_offset
    if remaining <= 0:
        return 0.0, "degraded"

    rul_days = float(remaining / drift_rate)
    rul_days = max(0.0, rul_days)

    logger.debug(
        "Drift RUL computed",
        extra={
            "rul_days": round(rul_days, 1),
            "drift_rate_pct_per_day": round(drift_rate, 5),
            "current_offset": round(current_offset, 3),
        },
    )

    return rul_days, "active"


def update_drift_state(
    state: Dict[str, Any],
    daily_mean_flow: float,
    mnf_baseline_days: int = 30,
    accuracy_threshold_percent: float = 2.0,
) -> Tuple[Dict[str, Any], Optional[float], str, Optional[float]]:
    """
    Update the drift tracking state with a new daily mean flow observation.

    Initialises the baseline from the first mnf_baseline_days days.
    Appends offset values once baseline is established.

    Args:
        state: Current drift state dict (from model_registry or empty {}).
                Expected keys: "baseline_mean", "baseline_days_count",
                               "offset_history", "drift_ewma".
        daily_mean_flow: Today's mean flow.
        mnf_baseline_days: Days to use for baseline establishment.
        accuracy_threshold_percent: Drift threshold.

    Returns:
        Tuple of (updated_state, rul_days, drift_status, offset_percent).
    """
    # Initialise state keys if missing
    state.setdefault("baseline_flow_history", [])
    state.setdefault("baseline_mean", None)
    state.setdefault("offset_history", [])
    state.setdefault("drift_ewma", daily_mean_flow)

    # Phase 1: accumulate baseline
    if state["baseline_mean"] is None:
        state["baseline_flow_history"].append(daily_mean_flow)
        if len(state["baseline_flow_history"]) >= mnf_baseline_days:
            state["baseline_mean"] = float(np.mean(state["baseline_flow_history"]))
            logger.info("Drift baseline established", extra={"baseline_mean": state["baseline_mean"]})
        return state, None, "warming_up", None

    # Phase 2: track offset
    # Update EWMA of current flow (lambda=0.1 for slow smoothing)
    lam = 0.1
    state["drift_ewma"] = lam * daily_mean_flow + (1.0 - lam) * state["drift_ewma"]

    offset = compute_offset(state["baseline_mean"], state["drift_ewma"])
    state["offset_history"].append(offset)

    # Limit history to 365 days
    if len(state["offset_history"]) > 365:
        state["offset_history"] = state["offset_history"][-365:]

    rul_days, drift_status = estimate_rul(
        state["offset_history"],
        accuracy_threshold_percent,
    )

    return state, rul_days, drift_status, offset
