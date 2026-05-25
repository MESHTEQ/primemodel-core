"""
tests/test_feature_engineering.py
-----------------------------------
Unit tests for feature engineering functions.
No ML models, no DB, no network — pure computation tests.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

from app.services.feature_engineering import (
    encode_hour_cyclic,
    build_feature_vector,
    compute_rolling_stats,
    extract_daily_mnf,
    build_lstm_sequence,
    build_cnn_sequence,
    SEQUENCE_LENGTH,
)


class TestEnodeHourCyclic:
    def test_midnight_sin_zero(self):
        sin_val, cos_val = encode_hour_cyclic(0.0)
        assert abs(sin_val) < 1e-6
        assert abs(cos_val - 1.0) < 1e-6

    def test_6am(self):
        sin_val, cos_val = encode_hour_cyclic(6.0)
        assert abs(sin_val - 1.0) < 1e-4  # sin(pi/2) = 1

    def test_continuity_at_midnight(self):
        """23:45 and 00:00 should be close in cyclic space."""
        sin_23_45, cos_23_45 = encode_hour_cyclic(23.75)
        sin_0, cos_0 = encode_hour_cyclic(0.0)
        # Check angular distance is small
        dot = sin_23_45 * sin_0 + cos_23_45 * cos_0
        assert dot > 0.99  # high cosine similarity = close angles


class TestBuildFeatureVector:
    def test_shape(self):
        v = build_feature_vector(0.5, 0.1, 8.0, 0, 0.48, 0.05)
        assert v.shape == (6,)

    def test_values(self):
        v = build_feature_vector(1.0, 0.2, 12.0, 3, 0.9, 0.1)
        assert v[0] == pytest.approx(1.0)
        assert v[1] == pytest.approx(0.2)
        assert v[2] == pytest.approx(12.0)
        assert v[3] == pytest.approx(3.0)


class TestComputeRollingStats:
    def test_empty_history(self):
        mean, std = compute_rolling_stats([])
        assert mean == 0.0
        assert std == 0.0

    def test_single_value(self):
        mean, std = compute_rolling_stats([5.0])
        assert mean == pytest.approx(5.0)

    def test_window_size(self):
        # Only last 4 values should be used
        history = [1.0, 2.0, 3.0, 10.0, 10.0, 10.0, 10.0]
        mean, std = compute_rolling_stats(history, window=4)
        assert mean == pytest.approx(10.0)
        assert std == pytest.approx(0.0)


def _make_df_with_timestamps(n: int, start_hour: int = 0) -> pd.DataFrame:
    """Helper: build a DataFrame with n 15-minute readings starting at start_hour."""
    base = datetime(2026, 1, 1, start_hour, 0, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=15 * i) for i in range(n)]
    return pd.DataFrame({
        "timestamp": [ts.isoformat() for ts in timestamps],
        "flow_rate": [0.5 + 0.1 * (i % 5) for i in range(n)],
    })


class TestExtractDailyMNF:
    def test_empty_df(self):
        df = pd.DataFrame(columns=["timestamp", "flow_rate"])
        result = extract_daily_mnf(df)
        assert len(result) == 0

    def test_mnf_window_filter(self):
        # Create data at 02:00–04:00 and outside
        rows = []
        base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        for i in range(96):
            ts = base + timedelta(minutes=15 * i)
            flow = 0.2 if 2 <= ts.hour < 4 else 0.8
            rows.append({"timestamp": ts.isoformat(), "flow_rate": flow})
        df = pd.DataFrame(rows)
        daily = extract_daily_mnf(df)
        assert len(daily) == 1
        assert daily.iloc[0] == pytest.approx(0.2, abs=0.05)

    def test_returns_series(self):
        df = _make_df_with_timestamps(200, start_hour=2)
        result = extract_daily_mnf(df, window_start_hour=2, window_end_hour=4)
        assert isinstance(result, pd.Series)


class TestBuildLSTMSequence:
    def test_insufficient_data_returns_none(self):
        df = _make_df_with_timestamps(50)
        result = build_lstm_sequence(df, seq_len=SEQUENCE_LENGTH)
        assert result is None

    def test_sufficient_data_returns_correct_shape(self):
        df = _make_df_with_timestamps(SEQUENCE_LENGTH + 10)
        result = build_lstm_sequence(df, seq_len=SEQUENCE_LENGTH)
        assert result is not None
        assert result.shape == (SEQUENCE_LENGTH, 4)

    def test_feature_names(self):
        df = _make_df_with_timestamps(SEQUENCE_LENGTH + 10)
        result = build_lstm_sequence(df, seq_len=SEQUENCE_LENGTH)
        # Columns: [flow_rate, flow_delta, hour_sin, hour_cos]
        # All values should be finite
        assert np.all(np.isfinite(result))

    def test_first_delta_is_zero(self):
        df = _make_df_with_timestamps(SEQUENCE_LENGTH + 10)
        result = build_lstm_sequence(df, seq_len=SEQUENCE_LENGTH)
        # flow_delta is col index 1, first row should be 0
        assert result[0, 1] == pytest.approx(0.0)


class TestBuildCNNSequence:
    def test_insufficient_data_returns_none(self):
        df = _make_df_with_timestamps(50)
        result = build_cnn_sequence(df, seq_len=SEQUENCE_LENGTH)
        assert result is None

    def test_correct_shape(self):
        df = _make_df_with_timestamps(SEQUENCE_LENGTH + 10)
        result = build_cnn_sequence(df, seq_len=SEQUENCE_LENGTH)
        assert result is not None
        assert result.shape == (SEQUENCE_LENGTH, 1)

    def test_normalised_flat_signal_returns_zeros(self):
        """Flat signal should produce all-zero normalised output."""
        rows = [{"timestamp": f"2026-01-01T{h:02d}:{m:02d}:00Z", "flow_rate": 0.5}
                for h in range(24) for m in [0, 15, 30, 45]]
        df = pd.DataFrame(rows[:SEQUENCE_LENGTH])
        result = build_cnn_sequence(df, seq_len=SEQUENCE_LENGTH)
        assert result is not None
        assert np.allclose(result, 0.0)
