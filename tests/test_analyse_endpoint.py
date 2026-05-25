"""
tests/test_analyse_endpoint.py
--------------------------------
Integration tests for the POST /analyse endpoint.

Uses FastAPI TestClient with mocked Supabase, model registry, and settings
to test the full request/response cycle without live DB or trained models.

Tests cover:
- Valid request returns 200 with correct schema
- Warming-up state (Day 0 meter) returns graceful response
- Response fields are in valid ranges
- Panda and Bove meter types both work
- Missing raw_payload is handled gracefully
- Input validation rejects bad values (422)
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Set environment variables BEFORE importing app — pydantic-settings reads at import time
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key")

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Default valid request payload
VALID_REQUEST = {
    "meter_id": "TEST_METER_001",
    "client_id": "utp",
    "meter_type": "panda",
    "timestamp": "2026-05-12T02:15:00Z",
    "flow_rate": 0.42,
    "cumulative_volume": 12345.67,
    "battery_level": 85.0,
    "alarm_flags": 0,
}


def _mock_state_cold_start():
    """Return a meter state that is in cold-start (no layers active yet)."""
    return {
        "meter_id": "TEST_METER_001",
        "first_reading_at": "2026-05-12T02:15:00Z",
        "days_of_data": 0.5,
        "statistical_active": False,
        "lstm_ae_active": False,
        "lstm_forecast_active": False,
        "cnn_active": False,
        "isolation_forest_trained_at": None,
        "lstm_ae_trained_at": None,
        "lstm_forecast_trained_at": None,
        "cnn_trained_at": None,
        "cusum_state": None,
        "ewma_state": None,
        "burst_state": None,
        "isolation_forest_threshold": None,
        "lstm_ae_threshold_stats": None,
        "lstm_forecast_baseline": None,
        "last_flow_rate": None,
        "flow_history": [0.42],
        "drift_state": {},
    }


def _make_patches():
    """Return a list of patch context managers needed for endpoint tests."""
    return [
        patch("app.routers.analyse.supabase_client"),
        patch("app.routers.analyse.model_registry.load_state", return_value=_mock_state_cold_start()),
        patch("app.routers.analyse.model_registry.save_state"),
        patch("app.routers.analyse.model_registry.record_uplink", return_value=_mock_state_cold_start()),
        patch("app.routers.analyse.model_registry.needs_retraining", return_value=False),
        patch("app.routers.analyse.model_registry.get_activation_status", return_value={
            "statistical": "warming_up",
            "lstm_autoencoder": "warming_up",
            "lstm_forecast": "warming_up",
            "cnn_pattern": "warming_up",
            "days_until_autoencoder": 29,
            "days_until_forecast": 59,
            "days_until_cnn": 89,
        }),
        patch("app.routers.analyse.isolation_forest.load_model", return_value=None),
        patch("app.routers.analyse.lstm_autoencoder.load_model", return_value=None),
        patch("app.routers.analyse.lstm_forecast.load_model", return_value=None),
        patch("app.routers.analyse.cnn_pattern.load_model", return_value=None),
    ]


class TestAnalyseEndpoint:

    def test_cold_start_returns_200(self):
        with patch("app.routers.analyse.supabase_client") as mock_supa, \
             patch("app.routers.analyse.model_registry") as mock_reg:

            mock_reg.load_state.return_value = _mock_state_cold_start()
            mock_reg.record_uplink.return_value = _mock_state_cold_start()
            mock_reg.save_state.return_value = None
            mock_reg.needs_retraining.return_value = False
            mock_reg.get_activation_status.return_value = {
                "statistical": "warming_up",
                "lstm_autoencoder": "warming_up",
                "lstm_forecast": "warming_up",
                "cnn_pattern": "warming_up",
                "days_until_autoencoder": 29,
                "days_until_forecast": 59,
                "days_until_cnn": 89,
            }
            mock_supa.fetch_uplink_history.return_value = []
            mock_supa.fetch_battery_history.return_value = []
            mock_supa.fetch_meter_metadata.return_value = None
            mock_supa.insert_uplink.return_value = True
            mock_supa.write_anomaly_score.return_value = True

            with patch("app.routers.analyse.isolation_forest.load_model", return_value=None), \
                 patch("app.routers.analyse.lstm_autoencoder.load_model", return_value=None), \
                 patch("app.routers.analyse.lstm_forecast.load_model", return_value=None), \
                 patch("app.routers.analyse.cnn_pattern.load_model", return_value=None):

                response = client.post("/analyse/", json=VALID_REQUEST)

        assert response.status_code == 200

    def _post_analyse(self):
        """Helper: POST /analyse with all relevant services mocked."""
        with patch("app.routers.analyse.supabase_client") as mock_supa, \
             patch("app.routers.analyse.model_registry") as mock_reg, \
             patch("app.routers.analyse.isolation_forest.load_model", return_value=None), \
             patch("app.routers.analyse.lstm_autoencoder.load_model", return_value=None), \
             patch("app.routers.analyse.lstm_forecast.load_model", return_value=None), \
             patch("app.routers.analyse.cnn_pattern.load_model", return_value=None):

            mock_reg.load_state.return_value = _mock_state_cold_start()
            mock_reg.record_uplink.return_value = _mock_state_cold_start()
            mock_reg.save_state.return_value = None
            mock_reg.needs_retraining.return_value = False
            mock_reg.get_activation_status.return_value = {
                "statistical": "warming_up",
                "lstm_autoencoder": "warming_up",
                "lstm_forecast": "warming_up",
                "cnn_pattern": "warming_up",
                "days_until_autoencoder": 29,
                "days_until_forecast": 59,
                "days_until_cnn": 89,
            }
            mock_supa.fetch_uplink_history.return_value = []
            mock_supa.fetch_battery_history.return_value = []
            mock_supa.fetch_meter_metadata.return_value = None
            mock_supa.insert_uplink.return_value = True
            mock_supa.write_anomaly_score.return_value = True

            return client.post("/analyse/", json=VALID_REQUEST)

    def test_response_schema_fields_present(self):
        response = self._post_analyse()
        assert response.status_code == 200
        data = response.json()

        required_fields = [
            "meter_id", "timestamp", "anomaly_score", "is_anomaly",
            "mnf_flag", "burst_detected", "leak_probability", "leak_severity",
            "pattern_type", "battery_rul_status", "drift_rul_status",
            "model_status", "confidence", "explanation",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_response_values_in_range(self):
        response = self._post_analyse()
        assert response.status_code == 200
        data = response.json()

        assert 0.0 <= data["anomaly_score"] <= 1.0
        assert 0.0 <= data["leak_probability"] <= 1.0
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["leak_severity"] in ["none", "low", "medium", "high"]
        assert data["pattern_type"] in ["normal", "burst", "background", "intermittent"]

    def test_bove_meter_type_accepted(self):
        with patch("app.routers.analyse.supabase_client") as mock_supa, \
             patch("app.routers.analyse.model_registry") as mock_reg, \
             patch("app.routers.analyse.isolation_forest.load_model", return_value=None), \
             patch("app.routers.analyse.lstm_autoencoder.load_model", return_value=None), \
             patch("app.routers.analyse.lstm_forecast.load_model", return_value=None), \
             patch("app.routers.analyse.cnn_pattern.load_model", return_value=None):

            mock_reg.load_state.return_value = _mock_state_cold_start()
            mock_reg.record_uplink.return_value = _mock_state_cold_start()
            mock_reg.save_state.return_value = None
            mock_reg.needs_retraining.return_value = False
            mock_reg.get_activation_status.return_value = {
                "statistical": "warming_up", "lstm_autoencoder": "warming_up",
                "lstm_forecast": "warming_up", "cnn_pattern": "warming_up",
                "days_until_autoencoder": 29, "days_until_forecast": 59, "days_until_cnn": 89,
            }
            mock_supa.fetch_uplink_history.return_value = []
            mock_supa.fetch_battery_history.return_value = []
            mock_supa.fetch_meter_metadata.return_value = None
            mock_supa.insert_uplink.return_value = True
            mock_supa.write_anomaly_score.return_value = True

            req = {**VALID_REQUEST, "meter_type": "bove_b39"}
            response = client.post("/analyse/", json=req)

        assert response.status_code == 200

    def test_model_status_structure(self):
        response = self._post_analyse()
        assert response.status_code == 200
        ms = response.json()["model_status"]

        assert ms["statistical"] in ["active", "warming_up"]
        assert ms["lstm_autoencoder"] in ["active", "warming_up"]
        assert ms["lstm_forecast"] in ["active", "warming_up"]
        assert ms["cnn_pattern"] in ["active", "warming_up"]
        assert isinstance(ms["days_until_autoencoder"], int)
        assert isinstance(ms["days_until_forecast"], int)
        assert isinstance(ms["days_until_cnn"], int)

    def test_explanation_is_string(self):
        response = self._post_analyse()
        assert response.status_code == 200
        assert isinstance(response.json()["explanation"], str)
        assert len(response.json()["explanation"]) > 0

    def test_invalid_flow_rate_returns_422(self):
        req = {**VALID_REQUEST, "flow_rate": -1.0}
        response = client.post("/analyse/", json=req)
        assert response.status_code == 422

    def test_invalid_battery_level_returns_422(self):
        req = {**VALID_REQUEST, "battery_level": 150.0}
        response = client.post("/analyse/", json=req)
        assert response.status_code == 422

    def test_missing_meter_id_returns_422(self):
        req = {k: v for k, v in VALID_REQUEST.items() if k != "meter_id"}
        response = client.post("/analyse/", json=req)
        assert response.status_code == 422

    def test_no_raw_payload_still_works(self):
        response = self._post_analyse()
        assert response.status_code == 200


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_has_version(self):
        response = client.get("/health/")
        assert "version" in response.json()
