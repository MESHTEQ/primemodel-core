"""
app/services/neural/cnn_pattern.py
------------------------------------
1D CNN Pattern Recognition — classifies flow window shape into leak pattern types.

Architecture:
    Conv1D(32 filters, kernel=5) → MaxPool1D(2)
    → Conv1D(64 filters, kernel=3) → GlobalAveragePooling1D
    → Dense(64, relu) → Dense(1, sigmoid)

Input:  (batch, 96, 1) — normalised 24h flow window
Output: (batch, 1) — leak_pattern_score [0, 1]

The sigmoid output is thresholded + a secondary softmax head classifies
pattern_type: "burst" | "background" | "intermittent" | "normal"

Cold-start training uses synthetic labelled data from generate_synthetic_data.py.
Once real confirmed events accumulate, the model can be retrained on real data.

Scope: Per-zone (not per-meter) — CNNs generalise better across meters in the same DMA zone.
"""

import os
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

from app.utils.logger import get_logger

logger = get_logger(__name__)

_tf = None

# Pattern type index mapping
PATTERN_TYPES = ["normal", "burst", "background", "intermittent"]
PATTERN_TYPE_MAP = {i: p for i, p in enumerate(PATTERN_TYPES)}


def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


def build_model(seq_len: int = 96, n_features: int = 1) -> "tf.keras.Model":
    """
    Construct the 1D CNN with dual output heads:
    - leak_score: sigmoid (0=normal, 1=leak)
    - pattern_type: softmax over 4 classes

    Args:
        seq_len: Input sequence length.
        n_features: 1 (normalised flow only).

    Returns:
        Uncompiled Keras model.
    """
    tf = _get_tf()
    from tensorflow.keras.layers import (
        Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout
    )
    from tensorflow.keras.models import Model

    inputs = tf.keras.Input(shape=(seq_len, n_features), name="cnn_input")

    x = Conv1D(32, kernel_size=5, activation="relu", padding="same", name="conv1")(inputs)
    x = MaxPooling1D(pool_size=2, name="pool1")(x)
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same", name="conv2")(x)
    x = GlobalAveragePooling1D(name="global_pool")(x)
    x = Dense(64, activation="relu", name="dense1")(x)
    x = Dropout(0.3, name="dropout")(x)

    # Head 1: leak probability
    leak_score = Dense(1, activation="sigmoid", name="leak_score")(x)

    # Head 2: pattern type classification
    pattern_logits = Dense(len(PATTERN_TYPES), activation="softmax", name="pattern_type")(x)

    model = Model(inputs, [leak_score, pattern_logits], name="cnn_pattern")
    return model


def train(
    sequences: np.ndarray,
    labels_binary: np.ndarray,
    labels_pattern: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    validation_split: float = 0.1,
) -> "tf.keras.Model":
    """
    Train the 1D CNN on labelled sequences.

    Args:
        sequences: Array of shape (n, seq_len, 1) — normalised flow windows.
        labels_binary: 1D array of shape (n,) — 0=normal, 1=leak.
        labels_pattern: 1D array of shape (n,) — integer class index (0–3).
        epochs: Training epochs.
        batch_size: Mini-batch size.
        validation_split: Validation fraction.

    Returns:
        Trained Keras model.
    """
    tf = _get_tf()
    model = build_model(seq_len=sequences.shape[1])
    model.compile(
        optimizer="adam",
        loss={
            "leak_score": "binary_crossentropy",
            "pattern_type": "sparse_categorical_crossentropy",
        },
        loss_weights={"leak_score": 1.0, "pattern_type": 0.5},
        metrics={"leak_score": "accuracy"},
    )

    logger.info(
        "Training 1D CNN Pattern",
        extra={"n_sequences": len(sequences), "epochs": epochs},
    )

    model.fit(
        sequences,
        {"leak_score": labels_binary.astype(np.float32), "pattern_type": labels_pattern},
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

    logger.info("1D CNN Pattern trained")
    return model


def save_model(model: "tf.keras.Model", model_store: str, zone_id: str) -> str:
    """
    Save the CNN model using the Keras 3 single-file format.

    The model is written as a single ``{safe_id}.keras`` file inside
    ``{model_store}/cnn_pattern/``.

    Note: scope is zone_id, not meter_id.
    """
    safe_id = zone_id.replace("/", "_").replace("\\", "_")
    path = os.path.join(model_store, "cnn_pattern", f"{safe_id}.keras")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    logger.info("CNN Pattern saved", extra={"path": path, "zone_id": zone_id})
    return path


def load_model(model_store: str, zone_id: str) -> Optional["tf.keras.Model"]:
    """
    Load a persisted CNN Pattern model from a ``.keras`` single-file.
    Returns None if no ``.keras`` file exists.
    """
    tf = _get_tf()
    safe_id = zone_id.replace("/", "_").replace("\\", "_")
    path = os.path.join(model_store, "cnn_pattern", f"{safe_id}.keras")
    if not os.path.isfile(path):
        return None
    try:
        model = tf.keras.models.load_model(path)
        logger.info("CNN Pattern loaded", extra={"path": path, "zone_id": zone_id})
        return model
    except Exception as e:
        logger.error("Failed to load CNN Pattern", extra={"path": path, "error": str(e)})
        return None


def score(
    model: "tf.keras.Model",
    sequence: np.ndarray,
) -> Tuple[float, str]:
    """
    Score a single normalised flow window.

    Args:
        model: Trained CNN model.
        sequence: numpy array of shape (seq_len, 1).

    Returns:
        Tuple of (leak_pattern_score, pattern_type_str).
        leak_pattern_score: float in [0, 1].
        pattern_type_str: "normal" | "burst" | "background" | "intermittent"
    """
    x = sequence[np.newaxis, ...]  # (1, seq_len, 1)
    outputs = model.predict(x, verbose=0)

    # Handle both list and tuple output from dual-head model
    if isinstance(outputs, (list, tuple)):
        leak_score = float(outputs[0][0][0])
        pattern_probs = outputs[1][0]
    else:
        leak_score = float(outputs[0])
        pattern_probs = np.array([1.0, 0.0, 0.0, 0.0])

    pattern_idx = int(np.argmax(pattern_probs))
    pattern_type = PATTERN_TYPE_MAP.get(pattern_idx, "normal")

    return leak_score, pattern_type
