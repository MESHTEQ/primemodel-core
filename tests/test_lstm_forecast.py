"""
tests/test_lstm_forecast.py
-----------------------------
Unit tests for the LSTM Forecasting model service.

Tests cover:
- Model construction
- Training pair preparation
- Training on minimal data
- Score output is in range
- Save/load round-trip

Uses shortened SEQ_LEN to keep tests fast.
"""

import numpy as np
import pytest
import tempfile

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")

from app.services.neural.lstm_forecast import (
    build_model,
    train,
    save_model,
    load_model,
    score,
    prepare_training_pairs,
    N_FORECAST,
)

SEQ_LEN = 32
N_FEATURES = 4
N_SEQUENCES = 30


def _make_sequences(n: int = N_SEQUENCES) -> np.ndarray:
    rng = np.random.default_rng(7)
    seqs = []
    for _ in range(n):
        t = np.linspace(0, 2 * np.pi, SEQ_LEN)
        flow = 0.4 + 0.15 * np.sin(t) + rng.normal(0, 0.01, SEQ_LEN)
        delta = np.concatenate([[0.0], np.diff(flow)])
        h_sin = np.sin(t)
        h_cos = np.cos(t)
        seqs.append(np.stack([flow, delta, h_sin, h_cos], axis=1))
    return np.array(seqs, dtype=np.float32)


class TestTrainingPairs:
    def test_shape(self):
        seqs = _make_sequences(10)
        X, y = prepare_training_pairs(seqs, N_FORECAST)
        assert X.shape == (10, SEQ_LEN - N_FORECAST, N_FEATURES)
        assert y.shape == (10, N_FORECAST)

    def test_y_is_flow_rate(self):
        """y should be the last N_FORECAST flow_rate values (feature index 0)."""
        seqs = _make_sequences(5)
        X, y = prepare_training_pairs(seqs, N_FORECAST)
        for i in range(5):
            expected = seqs[i, -N_FORECAST:, 0]
            np.testing.assert_array_almost_equal(y[i], expected)


class TestLSTMForecastBuild:
    def test_model_builds(self):
        model = build_model(seq_len=SEQ_LEN - N_FORECAST, n_features=N_FEATURES, n_forecast=N_FORECAST)
        assert model is not None

    def test_output_shape(self):
        model = build_model(seq_len=SEQ_LEN - N_FORECAST, n_features=N_FEATURES, n_forecast=N_FORECAST)
        model.compile(optimizer="adam", loss="mse")
        x = np.zeros((1, SEQ_LEN - N_FORECAST, N_FEATURES), dtype=np.float32)
        out = model.predict(x, verbose=0)
        assert out.shape == (1, N_FORECAST)


class TestLSTMForecastTrain:
    def test_train_returns_model_and_stats(self):
        seqs = _make_sequences()
        model, stats = train(seqs, n_features=N_FEATURES, n_forecast=N_FORECAST, epochs=2, batch_size=8)
        assert model is not None
        assert "rmse_mean" in stats
        assert "rmse_std" in stats
        assert stats["rmse_mean"] >= 0.0

    def test_trained_model_predicts(self):
        seqs = _make_sequences()
        model, stats = train(seqs, n_features=N_FEATURES, n_forecast=N_FORECAST, epochs=2, batch_size=8)
        context = seqs[0, :-N_FORECAST, :]
        actual = seqs[0, -N_FORECAST:, 0]
        s, _ = score(model, context, actual, stats)
        assert 0.0 <= s <= 1.0


class TestLSTMForecastPersistence:
    def test_save_load_roundtrip(self):
        seqs = _make_sequences()
        model, stats = train(seqs, n_features=N_FEATURES, n_forecast=N_FORECAST, epochs=2, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(model, tmpdir, "fc_test")
            loaded = load_model(tmpdir, "fc_test")
            assert loaded is not None

            context = seqs[0, :-N_FORECAST, :]
            actual = seqs[0, -N_FORECAST:, 0]
            s1, _ = score(model, context, actual, stats)
            s2, _ = score(loaded, context, actual, stats)
            assert s1 == pytest.approx(s2, abs=1e-4)

    def test_load_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert load_model(tmpdir, "missing") is None
