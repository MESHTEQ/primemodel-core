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


import math
from app.services.neural.lstm_autoencoder import build_model, train, save_model, load_model, score, save_threshold_stats, load_threshold_stats


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

    def test_save_load_threshold_stats_roundtrip(self, tmp_path):
        """save/load threshold stats preserves all values exactly."""
        stats_in = {"mae_mean": 0.05, "mae_std": 0.01, "threshold": 0.08}
        save_threshold_stats(stats_in, str(tmp_path), "test_key")
        stats_out = load_threshold_stats(str(tmp_path), "test_key")
        assert stats_out is not None
        assert abs(stats_out["mae_mean"] - 0.05) < 1e-9
        assert abs(stats_out["mae_std"] - 0.01) < 1e-9
        assert abs(stats_out["threshold"] - 0.08) < 1e-9

    def test_load_threshold_stats_missing_returns_none(self, tmp_path):
        assert load_threshold_stats(str(tmp_path), "no_such_key") is None


class TestLSTMAutoencoderScoreCalibration:
    """Tests for the L2 z-score squash normalisation introduced in P10."""

    def test_score_z_squash_z3(self):
        """mae == mae_mean + 3*mae_std → z=3 → score ≈ 0.5"""
        # score() requires a real Keras model (calls model.predict).
        # We build and compile a minimal model, then craft threshold_stats so
        # that the MAE the model produces maps to z=3.
        # Strategy: train a tiny model on one sequence type, check that the
        # normalised output lies in [0,1] and that the z-score formula is applied.
        sequences = _make_sequences(n=40)
        model, stats = train(
            sequences,
            seq_len=SEQ_LEN,
            n_features=N_FEATURES,
            epochs=1,
            batch_size=8,
        )
        # Inject synthetic stats that force z=3 for this model's actual MAE
        # on sequences[0], so score ≈ 0.5.
        x = sequences[0][np.newaxis, ...]
        reconstruction = model.predict(x, verbose=0)
        actual_mae = float(np.mean(np.abs(x - reconstruction)))
        # Choose mae_mean and mae_std so that z = (actual_mae - mae_mean) / mae_std = 3
        mae_std_val = max(actual_mae / 4.0, 1e-6)  # arbitrary non-zero std
        mae_mean_val = actual_mae - 3.0 * mae_std_val
        synth_stats = {
            "threshold": actual_mae * 2.0,  # above actual → is_anomaly=False
            "mae_mean": mae_mean_val,
            "mae_std": mae_std_val,
        }
        scored, _ = score(model, sequences[0], synth_stats)
        # z=3 → sigmoid(0) = 0.5
        assert abs(scored - 0.5) < 0.05, f"Expected ~0.5 for z=3, got {scored:.4f}"

    def test_score_mae_std_zero_fallback(self):
        """mae_std=0 should fall back to legacy normalisation without raising."""
        sequences = _make_sequences(n=40)
        model, _ = train(
            sequences,
            seq_len=SEQ_LEN,
            n_features=N_FEATURES,
            epochs=1,
            batch_size=8,
        )
        # mae_std=0 forces the fallback path
        fallback_stats = {"threshold": 0.5, "mae_mean": 0.0, "mae_std": 0.0}
        scored, is_anom = score(model, sequences[0], fallback_stats)
        assert 0.0 <= scored <= 1.0
        assert isinstance(is_anom, bool)
