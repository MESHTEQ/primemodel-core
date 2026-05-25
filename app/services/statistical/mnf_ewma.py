"""
app/services/statistical/mnf_ewma.py
--------------------------------------
EWMA (Exponentially Weighted Moving Average) applied to the daily MNF series.

EWMA provides a smoothed trend line for night-flow, making it easier to
identify gradual increases that CUSUM may not flag immediately. The smoothed
value is compared against the baseline mean to produce an anomaly contribution.

Formula:
    Z(t) = lambda * x(t) + (1 - lambda) * Z(t-1)
    where lambda in (0, 1] — lower lambda = more smoothing (slower response)

State is persisted in the baseline JSON per meter.
"""

import numpy as np
from typing import Dict, Any, Tuple

from app.utils.logger import get_logger

logger = get_logger(__name__)


def initialise_state(
    initial_mnf_values: np.ndarray,
    ewma_lambda: float = 0.2,
) -> Dict[str, Any]:
    """
    Initialise EWMA state from historical MNF values.

    The initial EWMA value is set to the mean of the baseline period.
    This avoids a cold-start transient caused by starting from zero.

    Args:
        initial_mnf_values: 1D array of daily MNF values from the baseline period.
        ewma_lambda: Smoothing factor. 0.2 = respond slowly (good for leak detection).

    Returns:
        Dict with keys: ewma_value, mu, sigma, lambda.
    """
    mu = float(np.mean(initial_mnf_values))
    sigma = float(np.std(initial_mnf_values))
    # Use a minimum sigma of 5% of the mean so flat baselines don't produce
    # astronomically large normalised scores from tiny deviations.
    min_sigma = max(1e-3, abs(mu) * 0.05)
    if sigma < min_sigma:
        sigma = min_sigma

    return {
        "ewma_value": mu,  # start at baseline mean
        "mu": mu,
        "sigma": sigma,
        "lambda": ewma_lambda,
    }


def update(
    state: Dict[str, Any],
    mnf_value: float,
    threshold_sigma: float = 3.0,
) -> Tuple[Dict[str, Any], float, bool]:
    """
    Update the EWMA with a new daily MNF observation.

    Args:
        state: Current EWMA state dict.
        mnf_value: Today's MNF value (m³/h).
        threshold_sigma: Flag as elevated if EWMA > mu + threshold_sigma * sigma.

    Returns:
        Tuple of:
            updated_state: New state dict.
            ewma_score: Normalised EWMA deviation score [0, 1].
            mnf_elevated: True if EWMA exceeds the threshold.
    """
    lam = state["lambda"]
    z_prev = state["ewma_value"]
    mu = state["mu"]
    sigma = state["sigma"]

    z_new = lam * mnf_value + (1.0 - lam) * z_prev

    deviation = z_new - mu
    mnf_elevated = deviation > threshold_sigma * sigma

    if mnf_elevated:
        logger.warning(
            "EWMA MNF elevated",
            extra={
                "ewma_value": z_new,
                "baseline_mu": mu,
                "deviation_sigma": deviation / sigma,
            },
        )

    # Normalise deviation against the threshold band
    ewma_score = float(np.clip(deviation / (threshold_sigma * sigma + 1e-9), 0.0, 1.0))
    updated_state = {**state, "ewma_value": z_new}
    return updated_state, ewma_score, mnf_elevated


def compute_ewma_from_series(
    mnf_series: np.ndarray,
    ewma_lambda: float = 0.2,
    threshold_sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute EWMA over a full historical MNF series.
    Used for offline analysis and plotting.

    Args:
        mnf_series: 1D array of daily MNF values.
        ewma_lambda: Smoothing factor.
        threshold_sigma: Elevation threshold.

    Returns:
        Tuple of (ewma_values, elevated_flags).
    """
    if len(mnf_series) == 0:
        return np.array([]), np.array([], dtype=bool)

    baseline_n = max(5, len(mnf_series) // 3)
    state = initialise_state(mnf_series[:baseline_n], ewma_lambda)

    ewma_values = np.zeros(len(mnf_series))
    elevated = np.zeros(len(mnf_series), dtype=bool)

    for i, val in enumerate(mnf_series):
        state, _, flag = update(state, float(val), threshold_sigma)
        ewma_values[i] = state["ewma_value"]
        elevated[i] = flag

    return ewma_values, elevated
