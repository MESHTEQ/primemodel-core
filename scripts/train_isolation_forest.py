"""
scripts/train_isolation_forest.py
-----------------------------------
Train the Isolation Forest model from synthetic or real uplink data.

Usage:
    python scripts/train_isolation_forest.py \
        --data data/synthetic/synthetic_uplinks.csv \
        --meter_id SYNTH_001 \
        --model_store ./models_store

Trains on the full dataset (all label_types) — Isolation Forest is
an unsupervised learner that fits to the data distribution.
The synthetic anomalies are present in the training data at a low rate,
consistent with the contamination parameter.

After training, the model is saved to:
    models_store/isolation_forest/{meter_id}.joblib
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.statistical import isolation_forest
from app.services import feature_engineering
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("train_isolation_forest")


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build the (N, 6) feature matrix from a synthetic uplinks DataFrame."""
    df = df.copy()
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True)
    flows = df["flow_rate"].values.astype(np.float32)
    deltas = np.concatenate([[0.0], np.diff(flows)]).astype(np.float32)
    hours = (df["timestamp_dt"].dt.hour + df["timestamp_dt"].dt.minute / 60.0).values.astype(np.float32)
    dows = df["timestamp_dt"].dt.weekday.values.astype(np.float32)
    rolling_means = pd.Series(flows).rolling(4, min_periods=1).mean().values.astype(np.float32)
    rolling_stds = pd.Series(flows).rolling(4, min_periods=1).std().fillna(0).values.astype(np.float32)
    return np.column_stack([flows, deltas, hours, dows, rolling_means, rolling_stds])


def main():
    parser = argparse.ArgumentParser(description="Train Isolation Forest for PrimeModel")
    parser.add_argument("--data", default="data/synthetic/synthetic_uplinks.csv")
    parser.add_argument("--meter_id", default="SYNTH_001")
    parser.add_argument("--model_store", default="./models_store")
    parser.add_argument("--contamination", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"  {len(df)} rows loaded")

    X = build_feature_matrix(df)
    print(f"  Feature matrix shape: {X.shape}")

    print("Training Isolation Forest...")
    model = isolation_forest.train(X, contamination=args.contamination)

    path = isolation_forest.save_model(model, args.model_store, args.meter_id)
    print(f"Model saved to: {path}")

    # Quick evaluation on labelled data
    if "label_type" in df.columns:
        scores = []
        for i in range(len(X)):
            s, _ = isolation_forest.score(model, X[i])
            scores.append(s)
        df["anomaly_score"] = scores
        print("\nMean anomaly scores by label:")
        print(df.groupby("label_type")["anomaly_score"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
