"""
tests/test_admin_auth.py
------------------------
Tests for the fail-closed /admin router authentication (TD-ADMIN-001).

Covers the require_admin_key dependency applied router-wide in app/main.py:
    (a) ADMIN_API_KEY unset            -> 503 "Admin API not configured"
    (b) key set, no header             -> 401
    (c) key set, wrong header value    -> 401
    (d) key set, correct X-Admin-Key   -> 200 (endpoint executes)

Settings handling:
    get_settings() is lru_cached AND Settings reads an optional .env file
    (env_file in model_config), so deleting the process env var alone is not
    a reliable way to simulate "unset" — a developer .env could leak the key
    back in. Instead we monkeypatch the admin_api_key attribute ON the cached
    Settings singleton: the dependency calls get_settings() at request time
    and receives that same object, so the patch is always seen, and pytest's
    monkeypatch restores it automatically after each test.
"""

import os
from unittest.mock import patch

# Required settings must exist before app import (same pattern as
# test_analyse_endpoint.py) — Settings has required Supabase fields.
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key")

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

client = TestClient(app)

TEST_KEY = "test-admin-key-123"


def test_admin_key_unset_returns_503(monkeypatch):
    """(a) Fail-closed: no ADMIN_API_KEY configured -> every /admin call is 503."""
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", None)

    resp = client.post("/admin/retrain")

    assert resp.status_code == 503
    assert resp.json()["detail"] == "Admin API not configured"


def test_missing_header_returns_401(monkeypatch):
    """(b) Key configured but X-Admin-Key header absent -> 401."""
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)

    resp = client.post("/admin/retrain")

    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid or missing admin key"


def test_wrong_header_returns_401(monkeypatch):
    """(c) Key configured but X-Admin-Key value mismatches -> 401."""
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)

    resp = client.post("/admin/retrain", headers={"X-Admin-Key": "wrong-key"})

    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid or missing admin key"


def test_correct_header_passes_auth(monkeypatch):
    """(d) Correct X-Admin-Key -> auth passes and the endpoint runs (200)."""
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)

    # Keep the endpoint side-effect-free: empty meter store -> nothing queued.
    with patch(
        "app.routers.admin.model_registry.list_all_meters", return_value=[]
    ):
        resp = client.post("/admin/retrain", headers={"X-Admin-Key": TEST_KEY})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["meters_queued"] == 0
