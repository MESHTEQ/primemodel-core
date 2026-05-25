"""
app/services/statistical/burst_detector.py
--------------------------------------------
Burst (pipe failure) event detection via first-order difference analysis.

A burst is characterised by a sudden, large positive spike in flow rate —
fundamentally different from a slow background leak. We detect it by
monitoring the first-order difference (delta) of flow rate readings and
flagging when the delta exceeds the trained spike threshold.

Threshold: mean(delta) + burst_threshold_sigma * std(delta)

State (baseline delta statistics) is computed from cold-start data and
persisted in the model_registry baseline JSON.
"""

import numpy as np
from typing import Dict, Any, Tuple, List

from app.utils.logger import get_logger

logger = get_logger(__name__)


def initialise_state(
    flow_history: List[float],
) -> Dict[str, Any]:
    """
    Compute baseline delta statistics from historical flow readings.

    Args:
        flow_history: Ordered list of flow_rate readings (most recent last).
                      Requires at least 2 readings.

    Returns:
        Dict with keys: delta_mean, delta_std.
        delta_std is set to a small epsilon if zero to avoid division errors.
    """
    if len(flow_history) < 2:
        return {"delta_mean": 0.0, "delta_std": 1.0}

    arr = np.array(flow_history, dtype=np.float32)
    deltas = np.abs(np.diff(arr))  # absolute first differences
    delta_mean = float(np.mean(deltas))
    delta_std = float(np.std(deltas))
    # Minimum std: 5% of mean flow to avoid triggering on completely flat signals.
    # A flat-flow meter (zero variation) cannot meaningfully detect a burst until
    # data shows natural variation. We set a floor so the threshold is not 3 * epsilon.
    mean_flow_ref = float(np.mean(np.abs(arr))) if len(arr) > 0 else 1.0
    min_std = max(1e-3, mean_flow_ref * 0.05)
    if delta_std < min_std:
        delta_std = min_std

    return {"delta_mean": delta_mean, "delta_std": delta_std}


def detect(
    state: Dict[str, Any],
    current_flow: float,
    previous_flow: float,
    threshold_sigma: float = 3.0,
) -> Tuple[float, bool]:
    """
    Evaluate whether the current flow reading represents a burst event.

    Args:
        state: Baseline state dict from initialise_state.
        current_flow: Current uplink flow rate.
        previous_flow: Flow rate from the immediately prior uplink.
        threshold_sigma: Number of standard deviations above baseline delta mean.

    Returns:
        Tuple of:
            burst_score: Normalised score [0, 1] — higher = more burst-like.
            burst_detected: True if delta exceeds threshold.
    """
    delta_mean = state.get("delta_mean", 0.0)
    delta_std = state.get("delta_std", 1.0)

    current_delta = abs(current_flow - previous_flow)
    threshold = delta_mean + threshold_sigma * delta_std

    burst_detected = current_delta > threshold

    if burst_detected:
        logger.warning(
            "Burst detected",
            extra={
                "current_delta": current_delta,
                "threshold": threshold,
                "current_flow": current_flow,
                "previous_flow": previous_flow,
            },
        )

    # Normalise: score is proportion of how far delta is above threshold
    if threshold <= 0:
        burst_score = 0.0
    else:
        burst_score = float(np.clip(current_delta / (threshold + 1e-9), 0.0, 1.0))

    return burst_score, burst_detected


def compute_burst_flags_from_series(
    flow_series: np.ndarray,
    threshold_sigma: float = 3.0,
) -> np.ndarray:
    """
    Compute burst flags over a full historical flow series.
    Used offline during synthetic data generation and testing.

    Args:
        flow_series: 1D array of flow rate readings.
        threshold_sigma: Burst detection threshold.

    Returns:
        Boolean 1D array, True at each index where a burst was detected.
    """
    if len(flow_series) < 2:
        return np.zeros(len(flow_series), dtype=bool)

    state = initialise_state(list(flow_series))
    flags = np.zeros(len(flow_series), dtype=bool)

    for i in range(1, len(flow_series)):
        _, flag = detect(state, float(flow_series[i]), float(flow_series[i - 1]), threshold_sigma)
        flags[i] = flag

    return flags
