"""
app/services/neural/lstm_autoencoder.py
-----------------------------------------
LSTM Autoencoder — unsupervised anomaly detection via reconstruction error.

Architecture:
    Encoder:  LSTM(64) → LSTM(32) → bottleneck (RepeatVector)
    Decoder:  LSTM(32) → LSTM(64) → TimeDistributed Dense(4)
    Input:    (batch, 96, 4) — 24h of [flow_rate, flow_delta, hour_sin, hour_cos]
    Output:   (batch, 96, 4) — reconstructed sequence

The model learns the normal shape of water consumption. When a leak or burst
produces an abnormal pattern, the reconstruction error (MAE) is elevated.

Threshold: mean + 3*std of MAE over the training set.
           Stored in the per-meter baseline JSON alongside the model.

Cold-start: returns model_status="warming_up" and score=0.0 if not yet trained.
"""

import os
import json
import numpy as np
from typing import Optional, Tuple, Dict, Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy-import TensorFlow to avoid import errors if running tests without GPU
_tf = None
_keras = None


def _get_tf():
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf
        import tensorflow.keras as keras
        _tf = tf
        _keras = keras
    return _tf, _keras


def build_model(seq_len: int = 96, n_features: int = 4) -> "tf.keras.Model":
    """
    Construct the LSTM Autoencoder architecture.

    Args:
        seq_len: Sequence length (default 96 = 24h at 15-min intervals).
        n_features: Number of features per timestep.

    Returns:
        Uncompiled Keras model.
    """
    tf, keras = _get_tf()
    from tensorflow.keras.layers import (
        Input, LSTM, RepeatVector, TimeDistributed, Dense
    )
    from tensorflow.keras.models import Model

    inputs = Input(shape=(seq_len, n_features), name="encoder_input")

    # Encoder
    x = LSTM(64, return_sequences=True, name="enc_lstm1")(inputs)
    x = LSTM(32, return_sequences=False, name="enc_lstm2")(x)

    # Bottleneck — broadcast the latent vector across all timesteps
    x = RepeatVector(seq_len, name="bottleneck")(x)

    # Decoder
    x = LSTM(32, return_sequences=True, name="dec_lstm1")(x)
    x = LSTM(64, return_sequences=True, name="dec_lstm2")(x)
    outputs = TimeDistributed(Dense(n_features), name="reconstruction")(x)

    model = Model(inputs, outputs, name="lstm_autoencoder")
    return model


def train(
    sequences: np.ndarray,
    seq_len: int = 96,
    n_features: int = 4,
    epochs: int = 30,
    batch_size: int = 32,
    validation_split: float = 0.1,
) -> Tuple["tf.keras.Model", Dict[str, Any]]:
    """
    Train the LSTM Autoencoder on a set of normal flow sequences.

    Args:
        sequences: numpy array of shape (n_sequences, seq_len, n_features).
                   Should contain only normal (non-anomalous) data.
        seq_len: Sequence length.
        n_features: Feature count.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        validation_split: Fraction of data held out for validation.

    Returns:
        Tuple of (trained_model, threshold_stats).
        threshold_stats: Dict with keys "mae_mean", "mae_std", "threshold".
    """
    tf, keras = _get_tf()
    model = build_model(seq_len, n_features)
    model.compile(optimizer="adam", loss="mae")

    logger.info(
        "Training LSTM Autoencoder",
        extra={"n_sequences": len(sequences), "epochs": epochs},
    )

    model.fit(
        sequences,
        sequences,  # target = input (reconstruction)
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

    # Compute reconstruction error on training set to set threshold
    reconstructions = model.predict(sequences, verbose=0)
    mae_per_sequence = np.mean(np.abs(sequences - reconstructions), axis=(1, 2))
    mae_mean = float(mae_per_sequence.mean())
    mae_std = float(mae_per_sequence.std())
    threshold = mae_mean + 3.0 * mae_std

    threshold_stats = {
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "threshold": threshold,
    }

    logger.info("LSTM Autoencoder trained", extra=threshold_stats)
    return model, threshold_stats


def save_model(
    model: "tf.keras.Model",
    model_store: str,
    meter_id: str,
) -> str:
    """
    Save the LSTM Autoencoder model to disk using the Keras 3 single-file format.

    The model is written as a single ``{safe_id}.keras`` file inside
    ``{model_store}/lstm_autoencoder/``.

    Args:
        model: Trained Keras model.
        model_store: Root model store path.
        meter_id: Meter identifier.

    Returns:
        Full path to the saved ``.keras`` file.
    """
    safe_id = meter_id.replace("/", "_").replace("\\", "_")
    path = os.path.join(model_store, "lstm_autoencoder", f"{safe_id}.keras")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    logger.info("LSTM Autoencoder saved", extra={"path": path, "meter_id": meter_id})
    return path


def load_model(model_store: str, meter_id: str) -> Optional["tf.keras.Model"]:
    """
    Load a persisted LSTM Autoencoder model from a ``.keras`` single-file.

    Returns:
        Loaded Keras model, or None if no saved ``.keras`` file exists.
    """
    tf, keras = _get_tf()
    safe_id = meter_id.replace("/", "_").replace("\\", "_")
    path = os.path.join(model_store, "lstm_autoencoder", f"{safe_id}.keras")
    if not os.path.isfile(path):
        return None
    try:
        model = tf.keras.models.load_model(path)
        logger.info("LSTM Autoencoder loaded", extra={"path": path, "meter_id": meter_id})
        return model
    except Exception as e:
        logger.error("Failed to load LSTM Autoencoder", extra={"path": path, "error": str(e)})
        return None


def save_threshold_stats(
    stats: Dict[str, Any],
    model_store: str,
    model_key: str,
) -> None:
    """
    Persist threshold stats to disk so the agnostic /analyse endpoint can load
    calibrated values instead of the hardcoded fallback defaults.

    Path: {model_store}/lstm_autoencoder/{safe_id}_stats.json
    The same ``/`` and ``\\`` sanitisation used in ``save_model`` is applied
    to ``model_key`` to derive ``safe_id``.

    Args:
        stats:       Dict with keys ``mae_mean``, ``mae_std``, ``threshold``
                     (as returned by ``train``).
        model_store: Root model store path (settings.model_store_path).
        model_key:   Key string, e.g. ``"{deveui}_{param}"`` — same value
                     used to save/load the Keras model.
    """
    safe_id = model_key.replace("/", "_").replace("\\", "_")
    dir_path = os.path.join(model_store, "lstm_autoencoder")
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{safe_id}_stats.json")
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh)
    logger.info(
        "LSTM AE threshold stats saved",
        extra={"path": file_path, "model_key": model_key},
    )


def load_threshold_stats(
    model_store: str,
    model_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Load persisted threshold stats for the given model key.

    Returns ``None`` when the file is absent, unreadable, or contains invalid
    JSON — a warning is logged in those cases.  Callers should fall back to
    hardcoded defaults and log that calibration is missing.

    These stats calibrate the agnostic /analyse scoring (Step 7).

    Args:
        model_store: Root model store path.
        model_key:   Same key passed to ``save_threshold_stats``.

    Returns:
        Dict with ``mae_mean``, ``mae_std``, ``threshold``, or ``None``.
    """
    safe_id = model_key.replace("/", "_").replace("\\", "_")
    file_path = os.path.join(model_store, "lstm_autoencoder", f"{safe_id}_stats.json")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.warning(
            "Failed to load LSTM AE threshold stats",
            extra={"path": file_path, "model_key": model_key, "error": str(e)},
        )
        return None


def score(
    model: "tf.keras.Model",
    sequence: np.ndarray,
    threshold_stats: Dict[str, Any],
) -> Tuple[float, bool]:
    """
    Score a single sequence using reconstruction error.

    Args:
        model: Trained LSTM Autoencoder.
        sequence: numpy array of shape (seq_len, n_features).
        threshold_stats: Dict with "mae_mean", "mae_std", "threshold".

    Returns:
        Tuple of (autoencoder_score, is_anomaly).
        autoencoder_score: float in [0, 1] — higher = more anomalous.
    """
    x = sequence[np.newaxis, ...]  # shape: (1, seq_len, n_features)
    reconstruction = model.predict(x, verbose=0)
    mae = float(np.mean(np.abs(x - reconstruction)))

    threshold = threshold_stats.get("threshold", 1.0)
    mae_mean = threshold_stats.get("mae_mean", 0.0)
    mae_std = threshold_stats.get("mae_std", 1.0)

    # Z-score squash using per-model calibration stats
    if mae_std > 0:
        z = (mae - mae_mean) / mae_std
        normalised = float(np.clip(1.0 / (1.0 + np.exp(-(z - 3.0))), 0.0, 1.0))
    else:
        logger.warning("mae_std is 0 or missing — falling back to legacy MAE/threshold normalisation")
        normalised = float(np.clip(mae / (threshold + 1e-9), 0.0, 2.0) / 2.0)
    is_anomaly = mae > threshold

    return normalised, is_anomaly
