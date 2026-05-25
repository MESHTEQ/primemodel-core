"""
tests/test_statistical_layer.py
---------------------------------
Unit tests for all four statistical layer components:
- Isolation Forest (train + score)
- CUSUM
- EWMA
- Burst detector
"""

import numpy as np
import pytest
import tempfile
import os

from app.services.statistical import isolation_forest, mnf_cusum, mnf_ewma, burst_detector


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

def _make_normal_features(n: int = 300) -> np.ndarray:
    """
    Build a realistic feature matrix that mimics diurnal water demand patterns.
    Using structured hour/dow/flow correlation gives the IF a learnable signal.
    """
    rng = np.random.default_rng(42)
    # Simulate n readings at 15-min intervals starting at midnight
    hours = np.array([(i * 15 / 60) % 24 for i in range(n)])
    dows = np.array([((i * 15 // (60 * 24)) % 7) for i in range(n)], dtype=float)

    # Diurnal flow: low at night, peak at 08:00 and 18:00
    base_flow = 0.3 + 0.2 * (
        np.exp(-0.5 * ((hours - 8.0) / 2.0) ** 2) +
        0.8 * np.exp(-0.5 * ((hours - 18.0) / 2.0) ** 2)
    )
    flows = np.clip(base_flow + rng.normal(0, 0.02, n), 0.05, 2.0).astype(np.float32)
    deltas = np.concatenate([[0.0], np.diff(flows)]).astype(np.float32)
    r_means = np.convolve(flows, np.ones(4) / 4, mode='same').astype(np.float32)
    r_stds = np.zeros(n, dtype=np.float32)
    return np.column_stack([flows, deltas, hours, dows, r_means, r_stds])


class TestIsolationForest:
    def test_train_returns_model(self):
        X = _make_normal_features()
        model = isolation_forest.train(X)
        assert model is not None

    def test_score_normal_is_low(self):
        X = _make_normal_features()
        model = isolation_forest.train(X)
        # Score multiple normal samples and check the mean is < 0.6
        scores = [isolation_forest.score(model, X[i])[0] for i in range(10)]
        assert 0.0 <= min(scores)
        assert max(scores) <= 1.0
        assert sum(scores) / len(scores) < 0.6, f"Mean normal score too high: {sum(scores)/len(scores):.3f}"

    def test_anomaly_sample_scores_higher(self):
        X = _make_normal_features()
        model = isolation_forest.train(X)
        # Mean score of 10 normal samples
        normal_scores = [isolation_forest.score(model, X[i])[0] for i in range(10)]
        mean_normal_score = sum(normal_scores) / len(normal_scores)

        # Inject extreme outlier — flow 50x higher than normal, extreme delta
        anomaly = np.array([50.0, 49.5, 3.0, 1.0, 50.0, 20.0], dtype=np.float32)
        anomaly_score, _ = isolation_forest.score(model, anomaly)
        # The anomaly should score higher than the mean of normals
        assert anomaly_score > mean_normal_score, \
            f"Anomaly score {anomaly_score:.3f} should exceed mean normal {mean_normal_score:.3f}"

    def test_save_and_load(self):
        X = _make_normal_features()
        model = isolation_forest.train(X)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "isolation_forest"))
            isolation_forest.save_model(model, tmpdir, "test_meter")
            loaded = isolation_forest.load_model(tmpdir, "test_meter")
            assert loaded is not None
            score1, _ = isolation_forest.score(model, X[0])
            score2, _ = isolation_forest.score(loaded, X[0])
            assert score1 == pytest.approx(score2, abs=1e-4)

    def test_load_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = isolation_forest.load_model(tmpdir, "nonexistent")
            assert result is None


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------

class TestCUSUM:
    def test_no_alert_on_baseline(self):
        baseline = np.array([0.15, 0.14, 0.16, 0.15, 0.13, 0.14, 0.15, 0.16, 0.14, 0.15])
        state = mnf_cusum.initialise_state(baseline)
        # Feed a value within baseline range
        updated, score, alert = mnf_cusum.update(state, 0.15, threshold_sigma=3.0)
        assert not alert
        assert 0.0 <= score <= 1.0

    def test_alert_on_sustained_increase(self):
        baseline = np.array([0.15] * 20)
        state = mnf_cusum.initialise_state(baseline)
        # Feed values far above baseline
        for _ in range(20):
            state, score, alert = mnf_cusum.update(state, 0.50, threshold_sigma=3.0)
        assert alert

    def test_cusum_state_resets_on_normal(self):
        baseline = np.array([0.15] * 20)
        state = mnf_cusum.initialise_state(baseline)
        assert state["cusum_state"] == 0.0

    def test_series_returns_arrays(self):
        series = np.array([0.15, 0.16, 0.15, 0.14, 0.35, 0.40, 0.45, 0.50])
        vals, alerts = mnf_cusum.compute_cusum_from_series(series)
        assert len(vals) == len(series)
        assert len(alerts) == len(series)


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------

class TestEWMA:
    def test_ewma_initialises_at_mean(self):
        baseline = np.array([0.30, 0.32, 0.28, 0.31, 0.29])
        state = mnf_ewma.initialise_state(baseline)
        assert state["ewma_value"] == pytest.approx(np.mean(baseline), abs=0.01)

    def test_no_elevation_flag_on_normal(self):
        baseline = np.array([0.30] * 10)
        state = mnf_ewma.initialise_state(baseline)
        _, score, flag = mnf_ewma.update(state, 0.31)
        assert not flag

    def test_elevation_flag_on_high_value(self):
        # Tight baseline, then a large step
        baseline = np.array([0.30] * 20)
        state = mnf_ewma.initialise_state(baseline, ewma_lambda=0.5)
        for _ in range(30):
            state, _, _ = mnf_ewma.update(state, 1.50)
        _, _, flag = mnf_ewma.update(state, 1.50)
        assert flag

    def test_score_in_range(self):
        baseline = np.array([0.30] * 10)
        state = mnf_ewma.initialise_state(baseline)
        _, score, _ = mnf_ewma.update(state, 0.30)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Burst Detector
# ---------------------------------------------------------------------------

class TestBurstDetector:
    def test_no_burst_on_stable_signal(self):
        history = [0.5] * 50
        state = burst_detector.initialise_state(history)
        score, detected = burst_detector.detect(state, 0.51, 0.50, threshold_sigma=3.0)
        assert not detected

    def test_burst_detected_on_spike(self):
        history = [0.5 + 0.01 * i % 3 for i in range(50)]
        state = burst_detector.initialise_state(history)
        # A massive spike
        score, detected = burst_detector.detect(state, 10.0, 0.5, threshold_sigma=3.0)
        assert detected
        assert score > 0.5

    def test_score_clipped_to_range(self):
        history = [0.5] * 50
        state = burst_detector.initialise_state(history)
        score, _ = burst_detector.detect(state, 100.0, 0.5, threshold_sigma=3.0)
        assert 0.0 <= score <= 1.0

    def test_series_burst_flags(self):
        series = np.array([0.5] * 50 + [10.0] * 3 + [0.5] * 50)
        flags = burst_detector.compute_burst_flags_from_series(series, threshold_sigma=3.0)
        assert len(flags) == len(series)
        # At least one flag should be True
        assert flags.any()
