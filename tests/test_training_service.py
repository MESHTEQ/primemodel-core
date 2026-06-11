"""
tests/test_training_service.py
--------------------------------
Tests for the LSTM AE training service (app/services/training.py).

TF-free tests (a)-(e) run on any machine.
Test (f) exercises end-to-end training and is skipped when TensorFlow is
absent — the pytestmark skipif is NOT applied module-wide so the other five
tests always execute.

Environment variables are set before any app import to satisfy pydantic-settings.
"""

import json
import os

# Required before any app import — pydantic-settings reads at import time.
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key")

import numpy as np
import pytest

from app.config import get_settings
from app.services.neural.lstm_autoencoder import save_threshold_stats, load_threshold_stats
from app.services.training import (
    _append_history,
    read_training_history,
    train_lstm_ae_for_device,
)

# ---------------------------------------------------------------------------
# TF availability flag — used only for test (f)
# ---------------------------------------------------------------------------
try:
    import tensorflow  # noqa: F401
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_DEVEUI = "AABBCCDDEEFF0011"

# 20 synthetic rows whose decoded_payload the generic decoder can handle.
# The generic decoder extracts any int/float key — {"flow_rate": float} works.
def _make_rows(n: int = 20) -> list:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        ts = f"2026-01-{(i // 24) + 1:02d}T{(i % 24):02d}:00:00Z"
        rows.append({
            "deveui": _FAKE_DEVEUI,
            "decoded_payload": {"flow_rate": float(rng.uniform(0.3, 0.7))},
            "created_at": ts,
        })
    return rows


def _make_rows_sine(n: int = 400) -> list:
    """
    Generate enough rows for >= 10 training sequences.
    SEQUENCE_LENGTH=96, stride=24 → need len >= 96 + 24 = 120 to start,
    and start positions 0..216 for 10 sequences → need len >= 96 + 9*24 = 312.
    400 rows provides a comfortable margin.
    """
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        day = i // 96
        hour = (i % 96) // 4
        minute = (i % 4) * 15
        ts = f"2026-01-{(day % 28) + 1:02d}T{hour:02d}:{minute:02d}:00Z"
        t = 2 * np.pi * i / 96
        flow = float(0.5 + 0.2 * np.sin(t) + rng.normal(0, 0.02))
        rows.append({
            "deveui": _FAKE_DEVEUI,
            "decoded_payload": {"flow_rate": flow},
            "created_at": ts,
        })
    return rows


# ---------------------------------------------------------------------------
# (a) insufficient_sequences
# ---------------------------------------------------------------------------

class TestInsufficientSequences:
    """20 rows decode fine but produce far fewer than 10 LSTM sequences."""

    def test_returns_failed_reason_insufficient(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.training.supabase_client.fetch_sensor_history",
            lambda deveui, **kw: _make_rows(20),
        )
        monkeypatch.setattr(
            "app.services.training.device_registry.get_device_info",
            lambda deveui: {"device_type": "generic", "name": "Test"},
        )

        # Robust no-TF guard: if training is ever called, the test fails.
        import app.services.training as training_mod
        monkeypatch.setattr(
            training_mod.lstm_autoencoder,
            "train",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("TF training should not be reached")
            ),
        )

        result = train_lstm_ae_for_device(_FAKE_DEVEUI)
        assert result["status"] == "failed"
        assert result["reason"] == "insufficient_sequences"
        assert result["readings_used"] == 20

    def test_n_sequences_reported(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.training.supabase_client.fetch_sensor_history",
            lambda deveui, **kw: _make_rows(20),
        )
        monkeypatch.setattr(
            "app.services.training.device_registry.get_device_info",
            lambda deveui: {"device_type": "generic", "name": "Test"},
        )
        result = train_lstm_ae_for_device(_FAKE_DEVEUI)
        assert "n_sequences" in result
        assert result["n_sequences"] >= 0


# ---------------------------------------------------------------------------
# (b) no_data
# ---------------------------------------------------------------------------

class TestNoData:
    def test_empty_history_returns_no_data(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.training.supabase_client.fetch_sensor_history",
            lambda deveui, **kw: [],
        )
        monkeypatch.setattr(
            "app.services.training.device_registry.get_device_info",
            lambda deveui: {"device_type": "generic", "name": "Test"},
        )
        result = train_lstm_ae_for_device(_FAKE_DEVEUI)
        assert result["status"] == "failed"
        assert result["reason"] == "no_data"
        assert result["deveui"] == _FAKE_DEVEUI


# ---------------------------------------------------------------------------
# (c) unknown_param
# ---------------------------------------------------------------------------

class TestUnknownParam:
    def test_explicit_param_not_in_series(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.training.supabase_client.fetch_sensor_history",
            lambda deveui, **kw: _make_rows(20),
        )
        monkeypatch.setattr(
            "app.services.training.device_registry.get_device_info",
            lambda deveui: {"device_type": "generic", "name": "Test"},
        )
        result = train_lstm_ae_for_device(_FAKE_DEVEUI, param="does_not_exist")
        assert result["status"] == "failed"
        assert result["reason"] == "unknown_param"
        assert "available_params" in result
        assert isinstance(result["available_params"], list)
        # The generic decoder extracted flow_rate from our payloads
        assert "flow_rate" in result["available_params"]


# ---------------------------------------------------------------------------
# (d) stats round-trip
# ---------------------------------------------------------------------------

class TestStatsRoundTrip:
    def test_save_and_load_equal(self, tmp_path):
        stats = {"mae_mean": 0.042, "mae_std": 0.011, "threshold": 0.075}
        key = "AABBCCDD_flow_rate"
        save_threshold_stats(stats, str(tmp_path), key)
        loaded = load_threshold_stats(str(tmp_path), key)
        assert loaded == stats

    def test_load_missing_returns_none(self, tmp_path):
        result = load_threshold_stats(str(tmp_path), "nonexistent_key")
        assert result is None

    def test_slash_in_key_sanitised(self, tmp_path):
        """model_key containing '/' must still save/load correctly."""
        stats = {"mae_mean": 0.1, "mae_std": 0.05, "threshold": 0.25}
        key = "AA/BB/CC_flow_rate"
        save_threshold_stats(stats, str(tmp_path), key)
        # File must exist with sanitised name
        safe_id = key.replace("/", "_").replace("\\", "_")
        expected_path = tmp_path / "lstm_autoencoder" / f"{safe_id}_stats.json"
        assert expected_path.exists()
        loaded = load_threshold_stats(str(tmp_path), key)
        assert loaded == stats


# ---------------------------------------------------------------------------
# (e) training history
# ---------------------------------------------------------------------------

class TestTrainingHistory:
    def test_no_data_training_appends_history(self, monkeypatch, tmp_path):
        """Running a failed training should append a line to the JSONL log."""
        settings = get_settings()
        monkeypatch.setattr(settings, "model_store_path", str(tmp_path))

        monkeypatch.setattr(
            "app.services.training.supabase_client.fetch_sensor_history",
            lambda deveui, **kw: [],
        )
        monkeypatch.setattr(
            "app.services.training.device_registry.get_device_info",
            lambda deveui: {"device_type": "generic", "name": "Test"},
        )

        train_lstm_ae_for_device("DEV000")
        train_lstm_ae_for_device("DEV001")
        train_lstm_ae_for_device("DEV002")

        history = read_training_history(limit=50)
        assert len(history) == 3

    def test_history_newest_first(self, monkeypatch, tmp_path):
        """read_training_history must return entries newest-first."""
        settings = get_settings()
        monkeypatch.setattr(settings, "model_store_path", str(tmp_path))

        monkeypatch.setattr(
            "app.services.training.supabase_client.fetch_sensor_history",
            lambda deveui, **kw: [],
        )
        monkeypatch.setattr(
            "app.services.training.device_registry.get_device_info",
            lambda deveui: {"device_type": "generic", "name": "Test"},
        )

        for idx in range(5):
            train_lstm_ae_for_device(f"DEV{idx:03d}")

        history = read_training_history(limit=50)
        devs = [h["deveui"] for h in history]
        # Newest entry is DEV004 (last appended), must appear first
        assert devs[0] == "DEV004"
        assert devs[-1] == "DEV000"

    def test_history_limit_respected(self, monkeypatch, tmp_path):
        """limit parameter must cap the number of returned entries."""
        settings = get_settings()
        monkeypatch.setattr(settings, "model_store_path", str(tmp_path))

        monkeypatch.setattr(
            "app.services.training.supabase_client.fetch_sensor_history",
            lambda deveui, **kw: [],
        )
        monkeypatch.setattr(
            "app.services.training.device_registry.get_device_info",
            lambda deveui: {"device_type": "generic", "name": "Test"},
        )

        for idx in range(10):
            train_lstm_ae_for_device(f"DEV{idx:03d}")

        history = read_training_history(limit=3)
        assert len(history) == 3

    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        """Missing JSONL file must return empty list — not raise."""
        settings = get_settings()
        monkeypatch.setattr(settings, "model_store_path", str(tmp_path))
        # No file has been written; read_training_history must return []
        result = read_training_history()
        assert result == []

    def test_append_history_directly(self, tmp_path, monkeypatch):
        """_append_history writes valid JSON lines."""
        settings = get_settings()
        monkeypatch.setattr(settings, "model_store_path", str(tmp_path))

        _append_history({"key": "value_1", "n": 1})
        _append_history({"key": "value_2", "n": 2})

        log_path = tmp_path / "training_history.jsonl"
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["key"] == "value_1"
        assert json.loads(lines[1])["key"] == "value_2"


# ---------------------------------------------------------------------------
# (f) end-to-end real training — TF required
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
class TestEndToEndTraining:
    def test_completed_with_artifacts(self, monkeypatch, tmp_path):
        """
        Full training run with 400 synthetic rows.
        Asserts: status=completed, model dir exists, stats file exists,
        history line appended.
        """
        settings = get_settings()
        monkeypatch.setattr(settings, "model_store_path", str(tmp_path))

        monkeypatch.setattr(
            "app.services.training.supabase_client.fetch_sensor_history",
            lambda deveui, **kw: _make_rows_sine(400),
        )
        monkeypatch.setattr(
            "app.services.training.device_registry.get_device_info",
            lambda deveui: {"device_type": "generic", "name": "Sine Wave Sensor"},
        )

        result = train_lstm_ae_for_device(_FAKE_DEVEUI, param="flow_rate")

        assert result["status"] == "completed", f"Unexpected result: {result}"
        assert result["param"] == "flow_rate"
        assert result["n_sequences"] >= 10
        assert "threshold_stats" in result
        assert result["threshold_stats"]["threshold"] > 0.0

        # Model file must exist (.keras single-file format)
        safe_id = f"{_FAKE_DEVEUI}_flow_rate".replace("/", "_").replace("\\", "_")
        model_file = tmp_path / "lstm_autoencoder" / f"{safe_id}.keras"
        assert model_file.exists(), f"Model file missing: {model_file}"

        # Stats file must exist
        stats_file = tmp_path / "lstm_autoencoder" / f"{safe_id}_stats.json"
        assert stats_file.exists(), f"Stats file missing: {stats_file}"

        # At least one history line appended
        history = read_training_history(limit=5)
        assert len(history) >= 1
        assert history[0]["status"] == "completed"
        assert history[0]["deveui"] == _FAKE_DEVEUI
