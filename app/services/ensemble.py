"""
app/services/ensemble.py
--------------------------
Ensemble scoring — combines all layer outputs into a final leak probability.

Weighting strategy:
    Weights are proportional to the reliability of each layer.
    Layers in warming_up state contribute weight=0 and are excluded from the normalisation.
    The final score is always normalised against the sum of active weights.

Default active weights:
    statistical:    0.35  (always present — baseline reliability)
    lstm_ae:        0.25  (reconstruction error — strong unsupervised signal)
    lstm_forecast:  0.25  (forecast error — directional leak signal)
    cnn_pattern:    0.15  (pattern recognition — confirmatory)

Leak severity thresholds (on final probability):
    none:   < 0.25
    low:    0.25 – 0.50
    medium: 0.50 – 0.75
    high:   >= 0.75

Confidence reflects how many layers are active:
    1 layer  (statistical only): 0.60
    2 layers:                    0.72
    3 layers:                    0.84
    4 layers:                    0.96
"""

from typing import Dict, Any, Tuple

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Base weights — must sum to 1.0
_BASE_WEIGHTS = {
    "statistical": 0.35,
    "lstm_ae": 0.25,
    "lstm_forecast": 0.25,
    "cnn_pattern": 0.15,
}

# Confidence per number of active layers
_CONFIDENCE_MAP = {1: 0.60, 2: 0.72, 3: 0.84, 4: 0.96}

# Leak severity thresholds
_SEVERITY_THRESHOLDS = [
    (0.75, "high"),
    (0.50, "medium"),
    (0.25, "low"),
    (0.0, "none"),
]


def compute_ensemble(
    statistical_score: float,
    autoencoder_score: float,
    forecast_score: float,
    cnn_score: float,
    statistical_active: bool = True,
    lstm_ae_active: bool = False,
    lstm_forecast_active: bool = False,
    cnn_active: bool = False,
) -> Dict[str, Any]:
    """
    Combine layer scores into a final leak probability and severity.

    Inactive layers contribute 0 to the weighted sum and are excluded from
    the weight normalisation. Statistical layer is always considered active —
    if somehow called when not active, it defaults to 0 contribution.

    Args:
        statistical_score: Score from Isolation Forest + CUSUM + EWMA + burst [0, 1].
        autoencoder_score: LSTM AE reconstruction error score [0, 1].
        forecast_score: LSTM Forecast error score [0, 1].
        cnn_score: CNN leak pattern score [0, 1].
        statistical_active: Whether statistical layer is active.
        lstm_ae_active: Whether LSTM AE is active.
        lstm_forecast_active: Whether LSTM Forecast is active.
        cnn_active: Whether CNN is active.

    Returns:
        Dict with keys:
            leak_probability (float): Final ensemble probability [0, 1].
            leak_severity (str): "none" | "low" | "medium" | "high"
            confidence (float): Model confidence based on active layer count.
            active_layers (list): Names of active layers used in this computation.
    """
    active_layers = []
    weighted_sum = 0.0
    weight_total = 0.0

    layer_inputs = [
        ("statistical", statistical_score, statistical_active),
        ("lstm_ae", autoencoder_score, lstm_ae_active),
        ("lstm_forecast", forecast_score, lstm_forecast_active),
        ("cnn_pattern", cnn_score, cnn_active),
    ]

    for name, score_val, is_active in layer_inputs:
        if is_active:
            w = _BASE_WEIGHTS[name]
            weighted_sum += w * float(score_val)
            weight_total += w
            active_layers.append(name)

    if weight_total == 0.0:
        # Edge case: nothing active — return 0 score with low confidence
        logger.warning("Ensemble called with no active layers — returning zero score")
        return {
            "leak_probability": 0.0,
            "leak_severity": "none",
            "confidence": 0.40,
            "active_layers": [],
        }

    leak_probability = float(weighted_sum / weight_total)
    leak_probability = max(0.0, min(1.0, leak_probability))

    # Determine severity
    leak_severity = "none"
    for threshold, severity_label in _SEVERITY_THRESHOLDS:
        if leak_probability >= threshold:
            leak_severity = severity_label
            break

    # Confidence
    n_active = len(active_layers)
    confidence = _CONFIDENCE_MAP.get(n_active, 0.60)

    logger.debug(
        "Ensemble computed",
        extra={
            "leak_probability": leak_probability,
            "leak_severity": leak_severity,
            "confidence": confidence,
            "active_layers": active_layers,
        },
    )

    return {
        "leak_probability": leak_probability,
        "leak_severity": leak_severity,
        "confidence": confidence,
        "active_layers": active_layers,
    }


def combine_statistical_scores(
    isolation_forest_score: float,
    cusum_score: float,
    ewma_score: float,
    burst_score: float,
) -> float:
    """
    Combine the four sub-scores of the statistical layer into one composite score.

    Burst events are given extra weight because they represent acute failures.
    CUSUM and EWMA are complementary and contribute roughly equally.
    Isolation Forest is the primary anomaly detector.

    Weights: IF=0.40, CUSUM=0.25, EWMA=0.20, burst=0.15 (but burst overrides to 0.8 min)

    Args:
        isolation_forest_score: [0, 1]
        cusum_score: [0, 1]
        ewma_score: [0, 1]
        burst_score: [0, 1]

    Returns:
        Combined statistical score [0, 1].
    """
    weighted = (
        0.40 * isolation_forest_score
        + 0.25 * cusum_score
        + 0.20 * ewma_score
        + 0.15 * burst_score
    )

    # If a burst is clearly detected, floor the statistical score at 0.80
    if burst_score > 0.8:
        weighted = max(weighted, 0.80)

    return float(max(0.0, min(1.0, weighted)))
