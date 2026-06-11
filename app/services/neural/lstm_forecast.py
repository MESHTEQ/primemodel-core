"""
app/services/neural/lstm_forecast.py
--------------------------------------
LSTM Forecasting model — predicts next N flow readings, compares against actual.

Architecture:
    Stacked LSTM: LSTM(128, return_sequences=True) → LSTM(64) → Dense(n_forecast)
    Input:  (batch, 96, 4) — last 24h of [flow_rate, flow_delta, hour_sin, hour_cos]
    Output: (batch, 4) — next 4 flow readings (1 hour ahead at 15-min intervals)

Leak signal: A persistent positive forecast error (actual > predicted) indicates
more flow is leaving the system than expected — consistent with a leak.

The normalised RMSE against a rolling baseline is used as the anomaly contribution.

Cold-start: returns model_status="warming_up" and score=0.0 if not yet trained.
"""

import os
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

from app.utils.logger import get_logger

logger = get_logger(__name__)

_tf = None


def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


# Number of timesteps to predict ahead (1 hour at 15-min intervals)
N_FORECAST = 4


def build_model(
    seq_len: int = 96,
    n_features: int = 4,
    n_forecast: int = N_FORECAST,
) -> "tf.keras.Model":
    """
    Construct the stacked LSTM forecasting model.

    Args:
        seq_len: Input sequence length.
        n_features: Input features per timestep.
        n_forecast: Number of timesteps to predict.

    Returns:
        Uncompiled Keras model.
    """
    tf = _get_tf()
    from tensorflow.keras.layers import Input, LSTM, Dense
    from tensorflow.keras.models import Model

    inputs = tf.keras.Input(shape=(seq_len, n_features), name="forecast_input")
    x = LSTM(128, return_sequences=True, name="lstm1")(inputs)
    x = LSTM(64, return_sequences=False, name="lstm2")(x)
    # Predict only flow_rate (first feature) N steps ahead
    outputs = Dense(n_forecast, name="forecast_output")(x)

    model = Model(inputs, outputs, name="lstm_forecast")
    return model


def prepare_training_pairs(
    sequences: np.ndarray,
    n_forecast: int = N_FORECAST,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare (X, y) training pairs from a set of sequences.

    Each sequence of length seq_len provides:
        X = sequence[:-n_forecast]  (context window)
        y = sequence[-n_forecast:, 0]  (next N flow_rate values = feature index 0)

    Args:
        sequences: Array of shape (n_seqs, seq_len, n_features).
        n_forecast: Number of steps to predict.

    Returns:
        Tuple (X, y) where X.shape=(n, seq_len-n_forecast, n_features), y.shape=(n, n_forecast).
    """
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for seq in sequences:
        x_part = seq[:-n_forecast]   # context
        y_part = seq[-n_forecast:, 0]  # target: flow_rate only
        X_list.append(x_part)
        y_list.append(y_part)

    return np.array(X_list), np.array(y_list)


def train(
    sequences: np.ndarray,
    n_features: int = 4,
    n_forecast: int = N_FORECAST,
    epochs: int = 30,
    batch_size: int = 32,
    validation_split: float = 0.1,
) -> Tuple["tf.keras.Model", Dict[str, Any]]:
    """
    Train the LSTM forecasting model.

    Args:
        sequences: Array of shape (n_seqs, seq_len, n_features) — normal flow sequences.
        n_features: Feature count.
        n_forecast: Steps ahead to predict.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        validation_split: Fraction for validation.

    Returns:
        Tuple of (trained_model, baseline_stats).
        baseline_stats: Dict with "rmse_mean", "rmse_std" computed on training set.
    """
    tf = _get_tf()

    X, y = prepare_training_pairs(sequences, n_forecast)
    seq_len = X.shape[1]

    model = build_model(seq_len, n_features, n_forecast)
    model.compile(optimizer="adam", loss="mse")

    logger.info(
        "Training LSTM Forecast",
        extra={"n_sequences": len(X), "epochs": epochs, "n_forecast": n_forecast},
    )

    model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        ],
    )

    # Compute baseline RMSE on training data
    predictions = model.predict(X, verbose=0)
    errors = np.sqrt(np.mean((y - predictions) ** 2, axis=1))
    rmse_mean = float(errors.mean())
    rmse_std = float(errors.std())

    baseline_stats = {"rmse_mean": rmse_mean, "rmse_std": rmse_std}
    logger.info("LSTM Forecast trained", extra=baseline_stats)
    return model, baseline_stats


def save_model(model: "tf.keras.Model", model_store: str, meter_id: str) -> str:
    """
    Save the LSTM Forecast model using the Keras 3 single-file format.

    The model is written as a single ``{safe_id}.keras`` file inside
    ``{model_store}/lstm_forecast/``.
    """
    safe_id = meter_id.replace("/", "_").replace("\\", "_")
    path = os.path.join(model_store, "lstm_forecast", f"{safe_id}.keras")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    logger.info("LSTM Forecast saved", extra={"path": path, "meter_id": meter_id})
    return path


def load_model(model_store: str, meter_id: str) -> Optional["tf.keras.Model"]:
    """
    Load a persisted LSTM Forecast model from a ``.keras`` single-file.
    Returns None if no ``.keras`` file exists.
    """
    tf = _get_tf()
    safe_id = meter_id.replace("/", "_").replace("\\", "_")
    path = os.path.join(model_store, "lstm_forecast", f"{safe_id}.keras")
    if not os.path.isfile(path):
        return None
    try:
        model = tf.keras.models.load_model(path)
        logger.info("LSTM Forecast loaded", extra={"path": path, "meter_id": meter_id})
        return model
    except Exception as e:
        logger.error("Failed to load LSTM Forecast", extra={"path": path, "error": str(e)})
        return None


def score(
    model: "tf.keras.Model",
    context_sequence: np.ndarray,
    actual_next_flows: np.ndarray,
    baseline_stats: Dict[str, Any],
) -> Tuple[float, bool]:
    """
    Score by comparing predicted vs actual next N flow readings.

    Args:
        model: Trained LSTM Forecast model.
        context_sequence: Array of shape (seq_len, n_features) — history window.
        actual_next_flows: 1D array of shape (n_forecast,) — actual next readings.
        baseline_stats: Dict with "rmse_mean", "rmse_std".

    Returns:
        Tuple of (forecast_score, is_anomaly).
        forecast_score: float in [0, 1]. Higher = actual persistently above predicted.
    """
    x = context_sequence[np.newaxis, ...]  # (1, seq_len, n_features)
    predicted = model.predict(x, verbose=0)[0]  # (n_forecast,)

    rmse = float(np.sqrt(np.mean((actual_next_flows - predicted) ** 2)))
    rmse_mean = baseline_stats.get("rmse_mean", 0.0)
    rmse_std = baseline_stats.get("rmse_std", 1.0)

    # Persistent positive error (actual > predicted) is the key signal
    mean_error = float(np.mean(actual_next_flows - predicted))
    persistent_leak_signal = mean_error > 0  # actual consistently above predicted

    # Threshold: rmse > mean + 3*std
    threshold = rmse_mean + 3.0 * (rmse_std if rmse_std > 0 else 1.0)
    is_anomaly = rmse > threshold and persistent_leak_signal

    # Normalise score
    forecast_score = float(np.clip(rmse / (threshold + 1e-9), 0.0, 1.0))
    return forecast_score, is_anomaly
