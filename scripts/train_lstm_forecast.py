"""
scripts/train_lstm_forecast.py
--------------------------------
Train the LSTM Forecasting model from synthetic uplink data.

The forecast model is trained on all data (including anomalies) because
it needs to learn both normal and abnormal regimes to detect deviations.
However, using primarily normal data is better — anomalous windows
should produce higher-than-expected forecast error.

Best practice: train on the first 18 days (pre-leak) for cleanest baseline.

Usage:
    python scripts/train_lstm_forecast.py \
        --data data/synthetic/synthetic_uplinks.csv \
        --meter_id SYNTH_001 \
        --model_store ./models_store

Output:
    models_store/lstm_forecast/SYNTH_001/ (SavedModel format)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.neural import lstm_forecast
from app.services import feature_engineering
from app.utils.logger import get_logger

logger = get_logger("train_lstm_forecast")

SEQ_LEN = feature_engineering.SEQUENCE_LENGTH
STRIDE = 12


def extract_sequences(df: pd.DataFrame) -> np.ndarray:
    """Extract overlapping sequences for forecast training."""
    sequences = []
    n_required = SEQ_LEN + lstm_forecast.N_FORECAST
    for start in range(0, len(df) - n_required, STRIDE):
        window = df.iloc[start: start + SEQ_LEN]
        seq = feature_engineering.build_lstm_sequence(window, seq_len=SEQ_LEN)
        if seq is not None:
            sequences.append(seq)
    return np.array(sequences) if sequences else np.array([]).reshape(0, SEQ_LEN, 4)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Forecast for PrimeModel")
    parser.add_argument("--data", default="data/synthetic/synthetic_uplinks.csv")
    parser.add_argument("--meter_id", default="SYNTH_001")
    parser.add_argument("--model_store", default="./models_store")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--normal_only_days",
        type=int,
        default=17,
        help="Use only first N days for training (before injected leaks begin at Day 18)"
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"  {len(df)} rows loaded")

    # Use pre-leak data for cleanest baseline
    cutoff_rows = args.normal_only_days * 96  # 96 readings per day
    df_train = df.iloc[:cutoff_rows]
    print(f"  Using first {args.normal_only_days} days = {len(df_train)} rows for training")

    print(f"Extracting sequences (stride={STRIDE})...")
    sequences = extract_sequences(df_train)
    print(f"  {len(sequences)} sequences extracted, shape: {sequences.shape}")

    if len(sequences) < 10:
        print("ERROR: Not enough sequences for training.")
        sys.exit(1)

    print(f"Training LSTM Forecast ({args.epochs} epochs)...")
    model, baseline_stats = lstm_forecast.train(
        sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print(f"Baseline stats: {baseline_stats}")
    path = lstm_forecast.save_model(model, args.model_store, args.meter_id)
    print(f"Model saved to: {path}")

    import json
    stats_path = os.path.join(args.model_store, "lstm_forecast", args.meter_id + "_baseline.json")
    with open(stats_path, "w") as f:
        json.dump(baseline_stats, f, indent=2)
    print(f"Baseline stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
