"""
tests/test_lstm_autoencoder.py
--------------------------------
Unit tests for the LSTM Autoencoder service.

Tests cover:
- Model construction
- Training on small synthetic data
- Scoring produces values in expected range
- Save/load round-trip

Note: TensorFlow is required. These tests will be skipped if TF is not installed.
Training tests use minimal epochs to keep CI time short.
"""

import numpy as np
import pytest
import tempfile
import os

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")


from app.services.neural.lstm_autoencoder import build_model, train, save_model, load_model, score


SEQ_LEN = 32   # Shortened for test speed
N_FEATURES = 4
N_SEQUENCES = 40


def _make_sequences(n: int = N_SEQUENCES) -> np.ndarray:
    """Generate synthetic normal-ish sequences."""
    rng = np.random.default_rng(42)
    seqs = []
    for _ in range(n):
        # Sinusoidal base + noise
        t = np.linspace(0, 2 * np.pi, SEQ_LEN)
        flow = 0.5 + 0.2 * np.sin(t) + rng.normal(0, 0.02, SEQ_LEN)
        delta = np.concatenate([[0.0], np.diff(flow)])
        h_sin = np.sin(t)
        h_cos = np.cos(t)
        seqs.append(np.stack([flow, delta, h_sin, h_cos], axis=1))
    return np.array(seqs, dtype=np.float32)


class TestLSTMAutoencoderBuild:
    def test_model_builds(self):
        model = build_model(seq_len=SEQ_LEN, n_features=N_FEATURES)
        assert model is not None

    def test_model_output_shape(self):
        model = build_model(seq_len=SEQ_LEN, n_features=N_FEATURES)
        model.compile(optimizer="adam", loss="mae")
        x = np.zeros((1, SEQ_LEN, N_FEATURES), dtype=np.float32)
        output = model.predict(x, verbose=0)
        assert output.shape == (1, SEQ_LEN, N_FEATURES)


class TestLSTMAutoencoderTrain:
    def test_train_returns_model_and_stats(self):
        sequences = _make_sequences()
        model, stats = train(
            sequences,
            seq_len=SEQ_LEN,
            n_features=N_FEATURES,
            epochs=2,
            batch_size=8,
        )
        assert model is not None
        assert "mae_mean" in stats
        assert "threshold" in stats
        assert stats["mae_mean"] >= 0.0
        assert stats["threshold"] > stats["mae_mean"]

    def test_train_threshold_positive(self):
        sequences = _make_sequences()
        _, stats = train(sequences, seq_len=SEQ_LEN, n_features=N_FEATURES, epochs=2, batch_size=8)
        assert stats["threshold"] > 0.0


class TestLSTMAutoencoderScore:
    def test_normal_score_lower_than_anomaly(self):
        sequences = _make_sequences()
        model, stats = train(sequences, seq_len=SEQ_LEN, n_features=N_FEATURES, epochs=3, batch_size=8)

        normal_seq = sequences[0]
        # Anomalous sequence: random noise with large amplitude
        rng = np.random.default_rng(99)
        anomaly_seq = rng.normal(5.0, 2.0, (SEQ_LEN, N_FEATURES)).astype(np.float32)

        normal_score, _ = score(model, normal_seq, stats)
        anomaly_score, _ = score(model, anomaly_seq, stats)

        assert normal_score < anomaly_score

    def test_score_in_range(self):
        sequences = _make_sequences()
        model, stats = train(sequences, seq_len=SEQ_LEN, n_features=N_FEATURES, epochs=2, batch_size=8)
        s, _ = score(model, sequences[0], stats)
        assert 0.0 <= s <= 1.0


class TestLSTMAutoencoderPersistence:
    def test_save_and_load_roundtrip(self):
        sequences = _make_sequences()
        model, stats = train(sequences, seq_len=SEQ_LEN, n_features=N_FEATURES, epochs=2, batch_size=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(model, tmpdir, "test_meter")
            loaded = load_model(tmpdir, "test_meter")
            assert loaded is not None

            s1, _ = score(model, sequences[0], stats)
            s2, _ = score(loaded, sequences[0], stats)
            assert s1 == pytest.approx(s2, abs=1e-4)

    def test_load_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert load_model(tmpdir, "nonexistent") is None
