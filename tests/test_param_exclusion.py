"""
tests/test_param_exclusion.py
------------------------------
Unit tests for the _SYSTEM_KEYS parameter exclusion logic added in P10.

These tests exercise the module-level constant directly — no router or
HTTP fixtures required, so they run without Supabase or TF dependencies.
"""

import pytest

from app.routers.analyse import _SYSTEM_KEYS


class TestSystemKeyExclusion:
    def test_battery_excluded(self):
        assert "battery" in _SYSTEM_KEYS

    def test_rssi_excluded(self):
        assert "rssi" in _SYSTEM_KEYS

    def test_snr_excluded(self):
        assert "snr" in _SYSTEM_KEYS

    def test_system_key_exclusion_filter(self):
        """battery + rssi + snr excluded; temperature + humidity retained."""
        payload_keys = {"temperature", "humidity", "battery", "rssi", "snr"}
        filtered = [p for p in payload_keys if p.lower() not in _SYSTEM_KEYS]
        assert "battery" not in filtered
        assert "rssi" not in filtered
        assert "snr" not in filtered
        assert "temperature" in filtered
        assert "humidity" in filtered

    def test_case_insensitive_match(self):
        """Keys are matched case-insensitively via .lower()."""
        payload_keys = {"BATTERY", "Battery", "RSSI", "SNR", "Temperature"}
        filtered = [p for p in payload_keys if p.lower() not in _SYSTEM_KEYS]
        assert "BATTERY" not in filtered
        assert "Battery" not in filtered
        assert "RSSI" not in filtered
        assert "SNR" not in filtered
        assert "Temperature" in filtered

    def test_all_variants_excluded(self):
        """All declared battery/radio aliases are present in _SYSTEM_KEYS."""
        expected = {"battery", "battery_level", "batt", "bat", "rssi", "signal", "snr", "fport", "fcnt"}
        assert expected.issubset(_SYSTEM_KEYS)

    def test_physical_params_not_excluded(self):
        """Physical measurement keys are never in _SYSTEM_KEYS."""
        physical = {"temperature", "humidity", "pressure", "co2", "voc", "pm2_5", "flow_rate"}
        for key in physical:
            assert key not in _SYSTEM_KEYS, f"{key} should not be in _SYSTEM_KEYS"
