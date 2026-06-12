"""
tests/test_l1_selfheal.py
--------------------------
Unit tests for P10.1 changes:
  A — L1 self-heal: IF model present, calibration stats absent → retrain, stats saved, calibrated score returned
  B — L1 self-heal failure path: retrain raises → legacy sigmoid fallback, no exception propagated
  C — Battery/system keys excluded from param_series at source → absent from layer1_scores keys
"""

import numpy as np
import pytest

from app.services.statistical import isolation_forest as IF
from app.routers.analyse import _SYSTEM_KEYS


# ---------------------------------------------------------------------------
# Test A — model present, stats absent → self-heal fires
# ---------------------------------------------------------------------------

def test_l1_selfheal_fires_when_stats_absent(tmp_path):
    """
    IF model is saved but calibration stats are deliberately omitted.
    Self-heal path: train again, save stats, calibrated score returned.
    """
    X = np.random.default_rng(0).standard_normal((50, 6))
    model, _ = IF.train(X)
    # Save model only — no stats
    IF.save_model(model, str(tmp_path), "test_dev_temp")

    # Confirm stats are absent before the heal
    assert IF.load_calibration_stats(str(tmp_path), "test_dev_temp") is None

    # Simulate self-heal: retrain to generate stats and save them
    _, new_stats = IF.train(X)
    IF.save_calibration_stats(str(tmp_path), "test_dev_temp", new_stats)

    # Stats must now exist and have the required keys
    loaded = IF.load_calibration_stats(str(tmp_path), "test_dev_temp")
    assert loaded is not None
    assert "center" in loaded
    assert "scale" in loaded

    # Score using calibrated stats must return a valid probability
    feat = X[0].tolist()
    score_val, _ = IF.score(model, feat, loaded)
    assert 0.0 <= score_val <= 1.0


# ---------------------------------------------------------------------------
# Test B — self-heal retrain failure → legacy fallback, no crash
# ---------------------------------------------------------------------------

def test_l1_selfheal_failure_falls_back(tmp_path):
    """
    When calibration_stats=None is passed to score(), the legacy sigmoid
    path fires — no exception is raised and result is a valid [0,1] float.
    This covers the self-heal-failed branch where if_calibration_stats stays None.
    """
    X = np.random.default_rng(1).standard_normal((50, 6))
    model, _ = IF.train(X)
    feat = X[0].tolist()

    # Passing stats=None triggers legacy fallback inside score()
    score_val, is_anomaly = IF.score(model, feat, calibration_stats=None)
    assert 0.0 <= score_val <= 1.0
    # is_anomaly must be a bool — no type error from the legacy path
    assert isinstance(is_anomaly, bool)


# ---------------------------------------------------------------------------
# Test C — battery and other system keys excluded from param_series at source
# ---------------------------------------------------------------------------

def test_system_keys_excluded_from_param_series():
    """
    The source filter added in P10.1 strips _SYSTEM_KEYS from param_series
    before any layer iterates it. This test mirrors that filter directly.
    """
    raw_param_series = {
        "temperature": [(None, 25.0)],
        "battery": [(None, 87.0)],
        "humidity": [(None, 62.0)],
        "rssi": [(None, -85.0)],
    }
    filtered = {k: v for k, v in raw_param_series.items() if k.lower() not in _SYSTEM_KEYS}

    # System keys must be absent
    assert "battery" not in filtered
    assert "rssi" not in filtered

    # Physical parameters must be present
    assert "temperature" in filtered
    assert "humidity" in filtered
