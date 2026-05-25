"""
app/services/statistical/mnf_cusum.py
---------------------------------------
CUSUM (Cumulative Sum) control chart applied to daily MNF (Minimum Night Flow) series.

CUSUM is well-suited for detecting a sustained shift in night-time flow —
the hallmark of a background leak that worsens gradually.

Algorithm:
    At each new daily MNF observation:
        S(t) = max(0, S(t-1) + (x(t) - mu - k))
    where:
        x(t)  = today's MNF value
        mu    = baseline mean MNF (from first cold_start_days of data)
        k     = allowance = 0.5 * sigma (half the expected shift to detect)
        sigma = baseline std dev of MNF

    Alert condition: S(t) > threshold_sigma * sigma

State per meter is stored in the model_registry baseline JSON, not in memory,
so it persists across process restarts.
"""

import numpy as np
from typing import Dict, Any, Tuple

from app.utils.logger import get_logger

logger = get_logger(__name__)


def initialise_state(mnf_values: np.ndarray) -> Dict[str, Any]:
    """
    Compute baseline statistics from the first N days of MNF data.

    Args:
        mnf_values: 1D numpy array of daily MNF values (m³/h) from cold-start period.

    Returns:
        Dict with keys: mu, sigma, cusum_state, k, alert_count.
        cusum_state is initialised to 0.0.
    """
    mu = float(np.mean(mnf_values))
    sigma = float(np.std(mnf_values))
    if sigma == 0.0:
        sigma = 1e-6  # guard against flat signal

    k = 0.5 * sigma  # allowance — detect shifts of at least 1 sigma

    return {
        "mu": mu,
        "sigma": sigma,
        "k": k,
        "cusum_state": 0.0,
        "alert_count": 0,
    }


def update(
    state: Dict[str, Any],
    mnf_value: float,
    threshold_sigma: float = 3.0,
) -> Tuple[Dict[str, Any], float, bool]:
    """
    Update the CUSUM state with a new daily MNF observation.

    Args:
        state: Current CUSUM state dict (from initialise_state or a previous update).
        mnf_value: Today's mean MNF value (m³/h).
        threshold_sigma: Alert threshold expressed in standard deviations.

    Returns:
        Tuple of:
            updated_state: New state dict (safe to persist).
            cusum_score: Current cumulative sum S(t) normalised by sigma.
            alert: True if S(t) > threshold_sigma * sigma.
    """
    mu = state["mu"]
    sigma = state["sigma"]
    k = state["k"]
    s_prev = state["cusum_state"]

    # One-sided upper CUSUM (detects upward shifts = increased leak flow)
    s_new = max(0.0, s_prev + (mnf_value - mu - k))

    alert = s_new > threshold_sigma * sigma
    if alert:
        state["alert_count"] = state.get("alert_count", 0) + 1
        logger.warning(
            "CUSUM MNF alert",
            extra={
                "cusum_state": s_new,
                "threshold": threshold_sigma * sigma,
                "mnf_value": mnf_value,
                "baseline_mu": mu,
            },
        )

    updated_state = {**state, "cusum_state": s_new}
    # Normalise score to [0, 1] against the threshold for ensemble use
    cusum_score = float(np.clip(s_new / (threshold_sigma * sigma + 1e-9), 0.0, 1.0))
    return updated_state, cusum_score, alert


def compute_cusum_from_series(
    mnf_series: np.ndarray,
    threshold_sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run CUSUM over a full historical MNF series.
    Used during offline training/backfill to reconstruct the CUSUM trajectory.

    Args:
        mnf_series: 1D array of daily MNF values, chronological order.
        threshold_sigma: Alert threshold in sigma.

    Returns:
        Tuple of:
            cusum_values: 1D array of S(t) at each day.
            alerts: Boolean 1D array, True on alert days.
    """
    if len(mnf_series) < 2:
        return np.zeros(len(mnf_series)), np.zeros(len(mnf_series), dtype=bool)

    # Use first third as baseline (at least 5 days)
    baseline_n = max(5, len(mnf_series) // 3)
    baseline = mnf_series[:baseline_n]
    state = initialise_state(baseline)

    cusum_values = np.zeros(len(mnf_series))
    alerts = np.zeros(len(mnf_series), dtype=bool)

    for i, val in enumerate(mnf_series):
        state, score, alert = update(state, float(val), threshold_sigma)
        cusum_values[i] = state["cusum_state"]
        alerts[i] = alert

    return cusum_values, alerts
