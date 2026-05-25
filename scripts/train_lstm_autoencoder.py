"""
scripts/train_lstm_autoencoder.py
-----------------------------------
Train the LSTM Autoencoder from synthetic uplink data.

The AE is trained exclusively on NORMAL readings (label_type="normal").
This ensures the model learns to reconstruct only normal flow patterns.
Anomalous sequences should produce elevated reconstruction error.

Usage:
    python scripts/train_lstm_autoencoder.py \
        --data data/synthetic/synthetic_uplinks.csv \
        --meter_id SYNTH_001 \
        --model_store ./models_store

Output:
    models_store/lstm_autoencoder/SYNTH_001/ (SavedModel format)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.neural import lstm_autoencoder
from app.services import feature_engineering
from app.utils.logger import get_logger

logger = get_logger("train_lstm_autoencoder")

SEQ_LEN = feature_engineering.SEQUENCE_LENGTH
STRIDE = 12  # 3-hour stride for sequence extraction


def extract_sequences(df: pd.DataFrame, labels_col: str = "label_type") -> np.ndarray:
    """
    Extract (seq_len, 4) sequences from the DataFrame.
    Only uses windows where ALL readings are labelled "normal".
    """
    sequences = []
    for start in range(0, len(df) - SEQ_LEN, STRIDE):
        window = df.iloc[start: start + SEQ_LEN]
        if labels_col in window.columns and not all(window[labels_col] == "normal"):
            continue  # skip windows containing anomalous readings
        seq = feature_engineering.build_lstm_sequence(window, seq_len=SEQ_LEN)
        if seq is not None:
            sequences.append(seq)

    return np.array(sequences) if sequences else np.array([]).reshape(0, SEQ_LEN, 4)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder for PrimeModel")
    parser.add_argument("--data", default="data/synthetic/synthetic_uplinks.csv")
    parser.add_argument("--meter_id", default="SYNTH_001")
    parser.add_argument("--model_store", default="./models_store")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"  {len(df)} rows loaded")

    if "label_type" not in df.columns:
        print("  No label_type column — using all rows as normal")
        df["label_type"] = "normal"

    print(f"Extracting normal sequences (stride={STRIDE})...")
    sequences = extract_sequences(df)
    print(f"  {len(sequences)} sequences extracted, shape: {sequences.shape}")

    if len(sequences) < 10:
        print("ERROR: Not enough sequences for training (need at least 10).")
        sys.exit(1)

    print(f"Training LSTM Autoencoder ({args.epochs} epochs, batch_size={args.batch_size})...")
    model, threshold_stats = lstm_autoencoder.train(
        sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print(f"Threshold stats: {threshold_stats}")
    path = lstm_autoencoder.save_model(model, args.model_store, args.meter_id)
    print(f"Model saved to: {path}")

    # Save threshold stats alongside the model
    import json
    stats_path = os.path.join(args.model_store, "lstm_autoencoder", args.meter_id + "_threshold.json")
    with open(stats_path, "w") as f:
        json.dump(threshold_stats, f, indent=2)
    print(f"Threshold stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
