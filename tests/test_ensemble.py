"""
tests/test_ensemble.py
------------------------
Unit tests for the ensemble scoring module.

Tests cover:
- Correct normalisation when layers are warming up
- Severity thresholds
- Confidence levels by active layer count
- Statistical score combination
- Edge cases: no active layers, all active
"""

import pytest
from app.services.ensemble import compute_ensemble, combine_statistical_scores


class TestComputeEnsemble:
    def test_statistical_only_active(self):
        """When only statistical layer is active, full weight on it."""
        result = compute_ensemble(
            statistical_score=0.6,
            autoencoder_score=0.0,
            forecast_score=0.0,
            cnn_score=0.0,
            statistical_active=True,
            lstm_ae_active=False,
            lstm_forecast_active=False,
            cnn_active=False,
        )
        assert result["leak_probability"] == pytest.approx(0.6, abs=0.01)
        assert result["confidence"] == pytest.approx(0.60)
        assert result["active_layers"] == ["statistical"]

    def test_all_layers_active(self):
        result = compute_ensemble(
            statistical_score=0.5,
            autoencoder_score=0.5,
            forecast_score=0.5,
            cnn_score=0.5,
            statistical_active=True,
            lstm_ae_active=True,
            lstm_forecast_active=True,
            cnn_active=True,
        )
        assert result["leak_probability"] == pytest.approx(0.5, abs=0.01)
        assert result["confidence"] == pytest.approx(0.96)
        assert len(result["active_layers"]) == 4

    def test_warming_up_layers_contribute_zero(self):
        """Inactive layers should not pull the score up."""
        result_stat_only = compute_ensemble(
            statistical_score=0.8,
            autoencoder_score=0.0,
            forecast_score=0.0,
            cnn_score=0.0,
            statistical_active=True,
            lstm_ae_active=False,
            lstm_forecast_active=False,
            cnn_active=False,
        )
        result_all_zero = compute_ensemble(
            statistical_score=0.8,
            autoencoder_score=0.0,
            forecast_score=0.0,
            cnn_score=0.0,
            statistical_active=True,
            lstm_ae_active=True,  # active but score=0 dilutes the result
            lstm_forecast_active=False,
            cnn_active=False,
        )
        # When AE is active with score 0, it dilutes the statistical score
        assert result_stat_only["leak_probability"] > result_all_zero["leak_probability"]

    def test_severity_none(self):
        result = compute_ensemble(0.1, 0.0, 0.0, 0.0, statistical_active=True)
        assert result["leak_severity"] == "none"

    def test_severity_low(self):
        result = compute_ensemble(0.35, 0.0, 0.0, 0.0, statistical_active=True)
        assert result["leak_severity"] == "low"

    def test_severity_medium(self):
        result = compute_ensemble(0.6, 0.0, 0.0, 0.0, statistical_active=True)
        assert result["leak_severity"] == "medium"

    def test_severity_high(self):
        result = compute_ensemble(0.85, 0.0, 0.0, 0.0, statistical_active=True)
        assert result["leak_severity"] == "high"

    def test_no_active_layers_returns_zero(self):
        result = compute_ensemble(
            0.9, 0.9, 0.9, 0.9,
            statistical_active=False,
            lstm_ae_active=False,
            lstm_forecast_active=False,
            cnn_active=False,
        )
        assert result["leak_probability"] == 0.0
        assert result["leak_severity"] == "none"

    def test_probability_bounded(self):
        result = compute_ensemble(
            1.0, 1.0, 1.0, 1.0,
            statistical_active=True,
            lstm_ae_active=True,
            lstm_forecast_active=True,
            cnn_active=True,
        )
        assert 0.0 <= result["leak_probability"] <= 1.0

    def test_two_active_layers_confidence(self):
        result = compute_ensemble(
            0.5, 0.5, 0.0, 0.0,
            statistical_active=True,
            lstm_ae_active=True,
            lstm_forecast_active=False,
            cnn_active=False,
        )
        assert result["confidence"] == pytest.approx(0.72)


class TestCombineStatisticalScores:
    def test_all_zero_returns_zero(self):
        s = combine_statistical_scores(0.0, 0.0, 0.0, 0.0)
        assert s == 0.0

    def test_burst_floors_score_at_0_8(self):
        s = combine_statistical_scores(0.1, 0.1, 0.1, 0.9)
        assert s >= 0.80

    def test_output_bounded(self):
        s = combine_statistical_scores(1.0, 1.0, 1.0, 1.0)
        assert 0.0 <= s <= 1.0

    def test_weights_approximately_correct(self):
        """IF=1 only should give 0.40 contribution."""
        s = combine_statistical_scores(1.0, 0.0, 0.0, 0.0)
        assert s == pytest.approx(0.40, abs=0.01)
