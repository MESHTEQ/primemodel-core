"""
scripts/train_cnn_pattern.py
------------------------------
Train the 1D CNN Pattern Recognition model from labelled synthetic data.

Unlike the LSTM models, the CNN is trained on LABELLED data:
    "normal" → class 0
    "burst"  → class 1
    "background_leak" → class 2
    "intermittent_leak" → class 3

This is the only model that requires labelled sequences.
The synthetic data from generate_synthetic_data.py provides all four classes.

CNN scope: zone_id (not per-meter). For the synthetic/cold-start case,
zone_id defaults to "synthetic".

Usage:
    python scripts/train_cnn_pattern.py \
        --data data/synthetic/synthetic_uplinks.csv \
        --zone_id synthetic \
        --model_store ./models_store

Output:
    models_store/cnn_pattern/synthetic/ (SavedModel format)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.neural import cnn_pattern
from app.services import feature_engineering
from app.utils.logger import get_logger

logger = get_logger("train_cnn_pattern")

SEQ_LEN = feature_engineering.SEQUENCE_LENGTH
STRIDE = 8  # shorter stride for CNN — needs more labelled samples

LABEL_MAP = {
    "normal": 0,
    "burst": 1,
    "background_leak": 2,
    "intermittent_leak": 3,
}
# Map to binary: 0=normal, 1=any leak
BINARY_MAP = {
    "normal": 0,
    "burst": 1,
    "background_leak": 1,
    "intermittent_leak": 1,
}


def extract_labelled_sequences(df: pd.DataFrame) -> tuple:
    """
    Extract (seq_len, 1) windows and their labels.

    A window's label is determined by the majority label within it.
    Sequences where the majority is "normal" are labelled normal;
    otherwise the most common non-normal label is used.

    Returns:
        sequences: numpy array (n, seq_len, 1)
        labels_binary: numpy array (n,) — 0 or 1
        labels_pattern: numpy array (n,) — integer 0–3
    """
    sequences = []
    labels_binary = []
    labels_pattern = []

    for start in range(0, len(df) - SEQ_LEN, STRIDE):
        window = df.iloc[start: start + SEQ_LEN]
        seq = feature_engineering.build_cnn_sequence(window, seq_len=SEQ_LEN)
        if seq is None:
            continue

        # Determine window label by majority vote
        if "label_type" in window.columns:
            label_counts = window["label_type"].value_counts()
            majority_label = label_counts.index[0]
        else:
            majority_label = "normal"

        sequences.append(seq)
        labels_binary.append(BINARY_MAP.get(majority_label, 0))
        labels_pattern.append(LABEL_MAP.get(majority_label, 0))

    if not sequences:
        return np.array([]), np.array([]), np.array([])

    return (
        np.array(sequences),
        np.array(labels_binary, dtype=np.float32),
        np.array(labels_pattern, dtype=np.int32),
    )


def main():
    parser = argparse.ArgumentParser(description="Train 1D CNN Pattern for PrimeModel")
    parser.add_argument("--data", default="data/synthetic/synthetic_uplinks.csv")
    parser.add_argument("--zone_id", default="synthetic")
    parser.add_argument("--model_store", default="./models_store")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"  {len(df)} rows loaded")

    if "label_type" not in df.columns:
        print("ERROR: 'label_type' column required for CNN training. Run generate_synthetic_data.py first.")
        sys.exit(1)

    print(f"Extracting labelled sequences (stride={STRIDE})...")
    sequences, labels_binary, labels_pattern = extract_labelled_sequences(df)
    print(f"  {len(sequences)} sequences extracted, shape: {sequences.shape}")

    if len(sequences) < 20:
        print("ERROR: Not enough sequences for training.")
        sys.exit(1)

    # Class distribution
    unique, counts = np.unique(labels_pattern, return_counts=True)
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        label_name = cnn_pattern.PATTERN_TYPE_MAP.get(u, f"class_{u}")
        print(f"  {label_name}: {c} sequences")

    print(f"\nTraining 1D CNN ({args.epochs} epochs, batch_size={args.batch_size})...")
    model = cnn_pattern.train(
        sequences,
        labels_binary,
        labels_pattern,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    path = cnn_pattern.save_model(model, args.model_store, args.zone_id)
    print(f"Model saved to: {path}")

    # Quick accuracy check on training data
    print("\nTraining set evaluation:")
    total = len(sequences)
    correct_binary = 0
    correct_pattern = 0
    for i in range(total):
        score_val, pattern = cnn_pattern.score(model, sequences[i])
        predicted_binary = 1 if score_val > 0.5 else 0
        predicted_pattern = cnn_pattern.PATTERN_TYPES.index(pattern) if pattern in cnn_pattern.PATTERN_TYPES else 0
        if predicted_binary == int(labels_binary[i]):
            correct_binary += 1
        if predicted_pattern == int(labels_pattern[i]):
            correct_pattern += 1

    print(f"  Binary (leak/normal) accuracy: {correct_binary}/{total} = {correct_binary/total:.1%}")
    print(f"  Pattern type accuracy:         {correct_pattern}/{total} = {correct_pattern/total:.1%}")
    print("\n(Note: training accuracy will be high due to in-sample evaluation — use held-out data for real validation)")


if __name__ == "__main__":
    main()
