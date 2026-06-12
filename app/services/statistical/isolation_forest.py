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
import json
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from typing import Optional, Tuple, Dict, Any

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
) -> Tuple[IsolationForest, Dict[str, Any]]:
    """
    Fit an Isolation Forest on feature matrix X.

    Args:
        X: numpy array of shape (n_samples, 6) — feature vectors.
        contamination: Expected fraction of anomalies in training data.

    Returns:
        Tuple of (fitted IsolationForest instance, calibration_stats dict).
        calibration_stats: {"center": float, "scale": float} — robust z-score
        calibration parameters derived from the training decision_function scores.
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    # Compute robust calibration stats for z-score normalisation
    raw_scores = model.decision_function(X)
    center = float(np.median(raw_scores))
    mad = float(np.median(np.abs(raw_scores - center)))
    scale = float(1.4826 * mad)
    if scale == 0 or not np.isfinite(scale):
        scale = float(np.std(raw_scores))
    if scale == 0 or not np.isfinite(scale):
        scale = 1.0
    calibration_stats: Dict[str, Any] = {"center": center, "scale": scale}

    logger.info(
        "IsolationForest trained",
        extra={"n_samples": len(X), "contamination": contamination, "center": center, "scale": scale},
    )
    return model, calibration_stats


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


def save_calibration_stats(model_store: str, meter_id: str, stats: Dict[str, Any]) -> None:
    """
    Persist calibration stats for a meter's Isolation Forest model.

    Path: {model_store}/isolation_forest/{safe_id}_stats.json
    Keys: {"center": float, "scale": float}

    Args:
        model_store: Root path of the model store directory.
        meter_id: Meter identifier (sanitised before use as filename).
        stats: Dict with "center" and "scale" keys (from train()).
    """
    safe_id = meter_id.replace("/", "_").replace("\\", "_")
    dir_path = os.path.join(model_store, "isolation_forest")
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{safe_id}_stats.json")
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh)
    logger.info(
        "IsolationForest calibration stats saved",
        extra={"path": file_path, "meter_id": meter_id},
    )


def load_calibration_stats(model_store: str, meter_id: str) -> Optional[Dict[str, Any]]:
    """
    Load persisted calibration stats for a meter's Isolation Forest model.

    Returns:
        Dict with "center" and "scale" keys, or None if the file does not exist
        or cannot be read.  A None return means the model was trained before P10
        and the caller should rely on the legacy sigmoid path in score().
    """
    safe_id = meter_id.replace("/", "_").replace("\\", "_")
    file_path = os.path.join(model_store, "isolation_forest", f"{safe_id}_stats.json")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.warning(
            "Failed to load IsolationForest calibration stats",
            extra={"path": file_path, "meter_id": meter_id, "error": str(e)},
        )
        return None


def score(
    model: IsolationForest,
    feature_vector: np.ndarray,
    calibration_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[float, bool]:
    """
    Score a single feature vector using the fitted Isolation Forest.

    The raw decision_function output is in (-inf, +inf) where:
      - Negative values = more anomalous
      - Positive values = more normal
    We normalise to [0, 1] via a sigmoid-like mapping so anomaly_score = 1 is maximum anomaly.

    Args:
        model: Fitted IsolationForest.
        feature_vector: 1D numpy array of shape (6,) or list.
        calibration_stats: Optional dict with "center" and "scale" from train().
            When provided, uses z-score squash: z = (center - raw) / scale,
            score = sigmoid(z - 3.0) clamped to [0, 1].
            When None, falls back to the legacy sigmoid path (models trained
            before P10 calibration — triggers a warning log).

    Returns:
        Tuple of (anomaly_score, is_anomaly).
        anomaly_score: float in [0, 1].
        is_anomaly: True if model predicts -1 (anomaly).
    """
    raw_score = float(model.decision_function([feature_vector])[0])
    is_anomaly = bool(model.predict([feature_vector])[0] == -1)

    if calibration_stats is not None:
        center = calibration_stats["center"]
        scale = calibration_stats["scale"]
        z = (center - raw_score) / scale  # lower raw = more anomalous → higher z
        score_val = float(np.clip(1.0 / (1.0 + np.exp(-(z - 3.0))), 0.0, 1.0))
    else:
        # Legacy path — existing sigmoid, unchanged
        logger.warning("calibration_stats missing for IF score — using legacy sigmoid (model needs retrain)")
        # Normalise: raw decision_function output is higher for normals, lower for anomalies.
        # We remap to [0,1] where 0=normal, 1=anomalous.
        # Typical range of decision_function is roughly [-0.5, 0.5].
        # We clip to [-0.5, 0.5] first so extreme outliers don't invert on the sigmoid.
        clipped = float(np.clip(raw_score, -0.5, 0.5))
        normalised = float(1.0 / (1.0 + np.exp(clipped * 10.0)))
        score_val = float(np.clip(normalised, 0.0, 1.0))

    return score_val, is_anomaly
