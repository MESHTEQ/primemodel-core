"""
app/services/feature_engineering.py
-------------------------------------
Feature extraction and sequence building for all model layers.

Responsibilities:
- Build the feature vector used by Isolation Forest
- Extract daily MNF (Minimum Night Flow) values
- Compute rolling statistics (mean, std) over a sliding window
- Build 96-timestep sequences for LSTM and CNN input
- Encode hour-of-day cyclically (sin/cos) to avoid wrap-around discontinuity

All functions are pure — they receive DataFrames/arrays and return arrays.
No I/O or DB access here.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

from app.utils.time_utils import filter_mnf_window, parse_iso_timestamp
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Number of 15-minute intervals in 24 hours — the canonical sequence length
SEQUENCE_LENGTH = 96


def encode_hour_cyclic(hour_float: float) -> Tuple[float, float]:
    """
    Encode hour of day (0–24) as (sin, cos) pair to avoid discontinuity at midnight.

    Args:
        hour_float: Hour as float, e.g. 2.25 = 02:15.

    Returns:
        Tuple of (sin_val, cos_val), both in [-1, 1].
    """
    angle = 2.0 * np.pi * hour_float / 24.0
    return float(np.sin(angle)), float(np.cos(angle))


def build_feature_vector(
    flow_rate: float,
    flow_delta: float,
    hour_of_day: float,
    day_of_week: int,
    rolling_mean_1h: float,
    rolling_std_1h: float,
) -> np.ndarray:
    """
    Construct the 6-element feature vector used by Isolation Forest.

    Features:
        [flow_rate, flow_delta, hour_of_day, day_of_week, rolling_mean_1h, rolling_std_1h]

    Args:
        flow_rate: Current instantaneous flow rate (m³/h).
        flow_delta: Difference between current and previous flow reading.
        hour_of_day: Fractional hour (0.0–23.99).
        day_of_week: Integer 0 (Monday) – 6 (Sunday).
        rolling_mean_1h: Mean flow over the past hour (4 readings at 15-min intervals).
        rolling_std_1h: Std dev of flow over the past hour.

    Returns:
        1D numpy array of shape (6,).
    """
    return np.array(
        [flow_rate, flow_delta, hour_of_day, float(day_of_week), rolling_mean_1h, rolling_std_1h],
        dtype=np.float32,
    )


def compute_rolling_stats(
    history: List[float],
    window: int = 4,
) -> Tuple[float, float]:
    """
    Compute mean and std dev of the last `window` flow readings.

    Args:
        history: Ordered list of recent flow_rate values (most recent last).
        window: Number of readings to include.

    Returns:
        Tuple of (mean, std). Returns (0.0, 0.0) if history is empty.
    """
    if not history:
        return 0.0, 0.0
    arr = np.array(history[-window:], dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def extract_daily_mnf(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    flow_col: str = "flow_rate",
    window_start_hour: int = 2,
    window_end_hour: int = 4,
) -> pd.Series:
    """
    Extract the mean daily Minimum Night Flow (MNF) from a DataFrame.

    For each calendar day present in the DataFrame, returns the mean flow
    during the MNF window (default 02:00–04:00 UTC).

    Args:
        df: DataFrame with timestamp and flow_rate columns.
        ts_col: Name of timestamp column (datetime or ISO string).
        flow_col: Name of flow column.
        window_start_hour: MNF window start hour (inclusive).
        window_end_hour: MNF window end hour (exclusive).

    Returns:
        Pandas Series indexed by date, values are mean MNF for that day.
        Empty Series if no data falls in the MNF window.
    """
    if df.empty:
        return pd.Series(dtype=float)

    work = df[[ts_col, flow_col]].copy()
    if not pd.api.types.is_datetime64_any_dtype(work[ts_col]):
        work[ts_col] = pd.to_datetime(work[ts_col], utc=True)

    mnf_rows = filter_mnf_window(work, ts_col, window_start_hour, window_end_hour)
    if mnf_rows.empty:
        return pd.Series(dtype=float)

    mnf_rows = mnf_rows.copy()
    mnf_rows["date"] = mnf_rows[ts_col].dt.date
    daily_mnf = mnf_rows.groupby("date")[flow_col].mean()
    return daily_mnf


def build_lstm_sequence(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    flow_col: str = "flow_rate",
    seq_len: int = SEQUENCE_LENGTH,
) -> Optional[np.ndarray]:
    """
    Build a (seq_len, 4) input array for LSTM models from the most recent rows.

    Features per timestep: [flow_rate, flow_delta, hour_sin, hour_cos]

    Args:
        df: DataFrame with at least `seq_len` rows, ordered chronologically.
            Must have timestamp and flow_rate columns.
        ts_col: Timestamp column name.
        flow_col: Flow rate column name.
        seq_len: Number of timesteps (default 96 = 24h at 15-min intervals).

    Returns:
        numpy array of shape (seq_len, 4), or None if df has fewer than seq_len rows.
    """
    if len(df) < seq_len:
        logger.debug(
            "Insufficient data for LSTM sequence",
            extra={"available": len(df), "required": seq_len},
        )
        return None

    work = df[[ts_col, flow_col]].copy()
    if not pd.api.types.is_datetime64_any_dtype(work[ts_col]):
        work[ts_col] = pd.to_datetime(work[ts_col], utc=True)
    work = work.sort_values(ts_col).tail(seq_len).reset_index(drop=True)

    flow_vals = work[flow_col].values.astype(np.float32)
    # First difference — pad with 0 at position 0
    flow_delta = np.concatenate([[0.0], np.diff(flow_vals)]).astype(np.float32)

    timestamps = work[ts_col]
    hour_float = timestamps.dt.hour + timestamps.dt.minute / 60.0
    hour_sin = np.sin(2.0 * np.pi * hour_float / 24.0).values.astype(np.float32)
    hour_cos = np.cos(2.0 * np.pi * hour_float / 24.0).values.astype(np.float32)

    sequence = np.stack([flow_vals, flow_delta, hour_sin, hour_cos], axis=1)
    return sequence  # shape: (seq_len, 4)


def build_training_sequences(
    history_df: pd.DataFrame,
    stride: int = 24,
) -> Optional[np.ndarray]:
    """
    Slide a window of SEQUENCE_LENGTH over ``history_df`` to produce training
    sequences for LSTM models (autoencoder or forecast).

    Stride controls overlap: a smaller stride produces more sequences but
    increases training time.  The default of 24 (6 hours at 15-min cadence)
    gives adequate coverage without redundancy.

    Args:
        history_df: DataFrame with "timestamp" and "flow_rate" columns,
                    ordered chronologically.
        stride:     Step size between consecutive windows (default 24).

    Returns:
        numpy array of shape (n_sequences, SEQUENCE_LENGTH, 4), or None if
        the history is too short to produce even one sequence.

    Note:
        The threshold stats produced by training on these sequences are
        persisted via ``lstm_autoencoder.save_threshold_stats`` /
        ``load_threshold_stats`` to calibrate the agnostic /analyse scoring.
    """
    seq_len = SEQUENCE_LENGTH
    if len(history_df) < seq_len + stride:
        return None

    all_seqs = []
    for start in range(0, len(history_df) - seq_len, stride):
        window = history_df.iloc[start: start + seq_len]
        seq = build_lstm_sequence(window, seq_len=seq_len)
        if seq is not None:
            all_seqs.append(seq)

    if not all_seqs:
        return None

    return np.array(all_seqs)  # shape: (n_sequences, seq_len, 4)


def build_cnn_sequence(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    flow_col: str = "flow_rate",
    seq_len: int = SEQUENCE_LENGTH,
) -> Optional[np.ndarray]:
    """
    Build a normalised (seq_len, 1) input window for the 1D CNN.

    The CNN operates on raw normalised flow only — simpler input than LSTM.
    Normalised using z-score of the window itself (mean=0, std=1).
    If std=0 (flat signal), returns zeros.

    Args:
        df: DataFrame with timestamp and flow_rate columns.
        ts_col: Timestamp column name.
        flow_col: Flow rate column name.
        seq_len: Window length.

    Returns:
        numpy array of shape (seq_len, 1), or None if insufficient data.
    """
    if len(df) < seq_len:
        return None

    work = df[[ts_col, flow_col]].copy()
    if not pd.api.types.is_datetime64_any_dtype(work[ts_col]):
        work[ts_col] = pd.to_datetime(work[ts_col], utc=True)
    work = work.sort_values(ts_col).tail(seq_len).reset_index(drop=True)

    flow_vals = work[flow_col].values.astype(np.float32)
    mean = float(flow_vals.mean())
    std = float(flow_vals.std())
    if std == 0.0:
        normalised = np.zeros_like(flow_vals)
    else:
        normalised = (flow_vals - mean) / std

    return normalised.reshape(seq_len, 1)
