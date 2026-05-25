"""
app/services/statistical/isolation_forest.py
----------------------------------------------
Isolation Forest anomaly detection — per-meter model.

The Isolation Forest is the backbone of the statistical layer.
It trains on the 6-element feature vector:
    [flow_rate, flow_delta, hour_of_day, day_of_week, rolling_mean_1h, rolling_std_1h]

Training: triggered when a meter accumulates >= cold_start_days of readings.
Inference: called on every uplink once trained.
Persistence: models saved as joblib files to models_store/isolation_forest/{meter_id}.joblib

The anomaly score is normalised to [0, 1] using the raw decision_function output.
Higher score = more anomalous.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from typing import Optional, Tuple

from app.utils.logger import get_logger

logger = get_logger(__name__)

# File name template — meter_id is sanitised before use
_MODEL_FILE_TEMPLATE = "{model_store}/isolation_forest/{meter_id}.joblib"


def _model_path(model_store: str, meter_id: str) -> str:
    """Return the full path for a meter's Isolation Forest model file."""
    safe_id = meter_id.replace("/", "_").replace("\\", "_")
    return _MODEL_FILE_TEMPLATE.format(model_store=model_store, meter_id=safe_id)


def train(
    X: np.ndarray,
    contamination: float = 0.05,
) -> IsolationForest:
    """
    Fit an Isolation Forest on feature matrix X.

    Args:
        X: numpy array of shape (n_samples, 6) — feature vectors.
        contamination: Expected fraction of anomalies in training data.

    Returns:
        Fitted IsolationForest instance.
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    logger.info("IsolationForest trained", extra={"n_samples": len(X), "contamination": contamination})
    return model


def save_model(model: IsolationForest, model_store: str, meter_id: str) -> str:
    """
    Persist the Isolation Forest model to disk.

    Args:
        model: Fitted IsolationForest.
        model_store: Root path of the model store directory.
        meter_id: Meter identifier (used as filename).

    Returns:
        Path where the model was saved.
    """
    path = _model_path(model_store, meter_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info("IsolationForest saved", extra={"path": path, "meter_id": meter_id})
    return path


def load_model(model_store: str, meter_id: str) -> Optional[IsolationForest]:
    """
    Load a persisted Isolation Forest model for a meter.

    Returns:
        IsolationForest instance, or None if no model exists yet.
    """
    path = _model_path(model_store, meter_id)
    if not os.path.exists(path):
        return None
    try:
        model = joblib.load(path)
        logger.info("IsolationForest loaded", extra={"path": path, "meter_id": meter_id})
        return model
    except Exception as e:
        logger.error("Failed to load IsolationForest", extra={"path": path, "error": str(e)})
        return None


def score(
    model: IsolationForest,
    feature_vector: np.ndarray,
) -> Tuple[float, bool]:
    """
    Score a single feature vector using the fitted Isolation Forest.

    The raw decision_function output is in (-inf, +inf) where:
      - Negative values = more anomalous
      - Positive values = more normal
    We normalise to [0, 1] via a sigmoid-like mapping so anomaly_score = 1 is maximum anomaly.

    Args:
        model: Fitted IsolationForest.
        feature_vector: 1D numpy array of shape (6,).

    Returns:
        Tuple of (anomaly_score, is_anomaly).
        anomaly_score: float in [0, 1].
        is_anomaly: True if model predicts -1 (anomaly).
    """
    x = feature_vector.reshape(1, -1)
    raw_score = model.decision_function(x)[0]  # typically in [-0.5, 0.5]
    prediction = model.predict(x)[0]  # -1 = anomaly, 1 = normal

    # Normalise: raw decision_function output is higher for normals, lower for anomalies.
    # We remap to [0,1] where 0=normal, 1=anomalous.
    # Typical range of decision_function is roughly [-0.5, 0.5].
    # We clip to [-0.5, 0.5] first so extreme outliers don't invert on the sigmoid.
    clipped = float(np.clip(raw_score, -0.5, 0.5))
    normalised = float(1.0 / (1.0 + np.exp(clipped * 10.0)))
    normalised = float(np.clip(normalised, 0.0, 1.0))

    is_anomaly = prediction == -1
    return normalised, is_anomaly
