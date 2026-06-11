"""
tests/test_admin_endpoints.py
------------------------------
Tests for the four P3 admin endpoints added to app/routers/admin.py.

Covers:
    (a) POST /admin/train/{deveui}     — 202, training job runs, result verifiable
    (b) POST /admin/train/{deveui}     — 409 when lock already held
    (c) Lock-release guarantees        — after successful job; after job that raises
    (d) GET  /admin/models/status      — filesystem scan with fake artifacts
    (e) GET  /admin/training/history   — patched history + limit clamping
    (f) GET  /admin/devices/{deveui}/scores — 200 with rows; 404 on empty
    (g) Auth inheritance               — all four new endpoints return 401 without key

Setup pattern mirrors test_admin_auth.py:
    - os.environ.setdefault before any app import
    - module-level TestClient(app)
    - monkeypatch on the cached get_settings() singleton for admin_api_key
    - TestClient runs BackgroundTasks SYNCHRONOUSLY, so assertions on side-effects
      (lock released, training function called) are valid after client.post() returns.
"""

import os
import threading
import pytest

# Required env vars must be set BEFORE app import — Settings has required fields.
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key")

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import training

client = TestClient(app)

TEST_KEY = "test-admin-key-123"
TEST_DEVEUI = "24E124136D355878"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auth_headers():
    return {"X-Admin-Key": TEST_KEY}


# ---------------------------------------------------------------------------
# (a) POST /admin/train/{deveui} — 202, job runs, correct deveui forwarded
# ---------------------------------------------------------------------------

def test_train_device_202_and_job_runs(monkeypatch, tmp_path):
    """
    Valid X-Admin-Key + unlocked device → 202 with status=training_started.
    The background task runs synchronously in TestClient — verify
    train_lstm_ae_for_device was called exactly once with the normalised deveui.
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)
    # Point model store to tmp so training history write succeeds
    monkeypatch.setattr(settings, "model_store_path", str(tmp_path))

    called_with = {}

    def mock_train(deveui, param=None):
        called_with["deveui"] = deveui
        called_with["param"] = param
        return {"status": "completed"}

    monkeypatch.setattr(
        "app.services.training.train_lstm_ae_for_device",
        mock_train,
    )

    resp = client.post(
        f"/admin/train/{TEST_DEVEUI.lower()}",  # lowercase — must be normalised
        headers=_auth_headers(),
    )

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "training_started"
    assert body["deveui"] == TEST_DEVEUI  # normalised to uppercase

    # Background task ran synchronously — mock was called
    assert called_with.get("deveui") == TEST_DEVEUI
    assert called_with.get("param") is None


def test_train_device_202_with_param(monkeypatch, tmp_path):
    """Param query string is forwarded to the training job."""
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)
    monkeypatch.setattr(settings, "model_store_path", str(tmp_path))

    called_with = {}

    def mock_train(deveui, param=None):
        called_with["deveui"] = deveui
        called_with["param"] = param
        return {"status": "completed"}

    monkeypatch.setattr("app.services.training.train_lstm_ae_for_device", mock_train)

    resp = client.post(
        f"/admin/train/{TEST_DEVEUI}",
        params={"param": "temperature"},
        headers=_auth_headers(),
    )

    assert resp.status_code == 202
    assert resp.json()["param"] == "temperature"
    assert called_with.get("param") == "temperature"


# ---------------------------------------------------------------------------
# (b) 409 — lock already held before POST
# ---------------------------------------------------------------------------

def test_train_device_409_when_locked(monkeypatch):
    """
    Manually acquire the lock before POSTing → 409.
    Lock is released in finally so other tests are not affected.
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)

    acquired = training.try_acquire_training_lock(TEST_DEVEUI)
    assert acquired, "Expected lock to be free before test"

    try:
        resp = client.post(
            f"/admin/train/{TEST_DEVEUI}",
            headers=_auth_headers(),
        )
        assert resp.status_code == 409
        assert "training already in progress" in resp.json()["detail"]
    finally:
        training.release_training_lock(TEST_DEVEUI)


# ---------------------------------------------------------------------------
# (c) Lock-release guarantees
# ---------------------------------------------------------------------------

def test_lock_released_after_successful_job(monkeypatch, tmp_path):
    """
    After a successful training job the lock must be free.
    TestClient runs the BackgroundTask synchronously so we can assert
    immediately after the response returns.
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)
    monkeypatch.setattr(settings, "model_store_path", str(tmp_path))

    monkeypatch.setattr(
        "app.services.training.train_lstm_ae_for_device",
        lambda deveui, param=None: {"status": "completed"},
    )

    resp = client.post(f"/admin/train/{TEST_DEVEUI}", headers=_auth_headers())
    assert resp.status_code == 202

    # Lock must be free after synchronous background task completed
    assert not training.is_training(TEST_DEVEUI)


def test_lock_released_after_job_raises(monkeypatch, tmp_path):
    """
    Even when train_lstm_ae_for_device raises an exception, the finally block
    in run_training_job must release the lock.

    TestClient propagates background-task exceptions; we catch any exception
    from client.post() and still assert the lock is free (proving the finally
    path ran).
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)
    monkeypatch.setattr(settings, "model_store_path", str(tmp_path))

    def boom(deveui, param=None):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr("app.services.training.train_lstm_ae_for_device", boom)

    # TestClient may propagate the background exception; catch it either way.
    try:
        resp = client.post(f"/admin/train/{TEST_DEVEUI}", headers=_auth_headers())
        # If TestClient does NOT propagate — check 202 was returned
        assert resp.status_code == 202
    except Exception:
        # TestClient propagated the RuntimeError — still valid; test the lock below
        pass

    # The finally block in run_training_job must have run regardless
    assert not training.is_training(TEST_DEVEUI), (
        "Lock was not released after training job raised an exception"
    )


# ---------------------------------------------------------------------------
# (d) GET /admin/models/status — filesystem scan
# ---------------------------------------------------------------------------

def test_models_status_with_fake_artifacts(monkeypatch, tmp_path):
    """
    Create fake artifacts in tmp_path model store.
    Verify the response reports the correct param under the correct deveui
    and that training=False.
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)
    monkeypatch.setattr(settings, "model_store_path", str(tmp_path))

    # Create fake lstm_autoencoder SavedModel dir and stats file
    ae_dir = tmp_path / "lstm_autoencoder"
    ae_dir.mkdir(parents=True)
    # SavedModel dir: 24E124136D355878_temperature/
    (ae_dir / "24E124136D355878_temperature").mkdir()
    # Stats file: 24E124136D355878_temperature_stats.json
    (ae_dir / "24E124136D355878_temperature_stats.json").write_text('{"threshold": 0.5}')

    resp = client.get("/admin/models/status", headers=_auth_headers())
    assert resp.status_code == 200

    body = resp.json()
    assert body["model_store"] == str(tmp_path)

    device_entry = body["devices"]["24E124136D355878"]
    assert device_entry["training"] is False

    ae_entry = device_entry["layers"]["lstm_autoencoder"]
    assert "temperature" in ae_entry["params"]
    assert "temperature" in ae_entry["stats_params"]


def test_models_status_isolation_forest_artifacts(monkeypatch, tmp_path):
    """
    Isolation forest .joblib files for a device are reported in params.
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)
    monkeypatch.setattr(settings, "model_store_path", str(tmp_path))

    if_dir = tmp_path / "isolation_forest"
    if_dir.mkdir(parents=True)
    (if_dir / "24E124136D355878_humidity.joblib").write_text("")
    (if_dir / "24E124136D355878_temperature.joblib").write_text("")

    resp = client.get("/admin/models/status", headers=_auth_headers())
    assert resp.status_code == 200

    if_entry = resp.json()["devices"]["24E124136D355878"]["layers"]["isolation_forest"]
    assert "humidity" in if_entry["params"]
    assert "temperature" in if_entry["params"]


# ---------------------------------------------------------------------------
# (e) GET /admin/training/history — patched history + limit clamping
# ---------------------------------------------------------------------------

def test_training_history_returns_entries(monkeypatch):
    """Patched history list is returned in order."""
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)

    fake_history = [
        {"timestamp": "2026-06-11T10:00:00+00:00", "deveui": TEST_DEVEUI, "status": "completed"},
        {"timestamp": "2026-06-11T09:00:00+00:00", "deveui": TEST_DEVEUI, "status": "failed"},
        {"timestamp": "2026-06-11T08:00:00+00:00", "deveui": TEST_DEVEUI, "status": "completed"},
    ]

    monkeypatch.setattr(
        "app.routers.admin.training.read_training_history",
        lambda limit: fake_history[:limit],
    )

    resp = client.get("/admin/training/history", headers=_auth_headers())
    assert resp.status_code == 200
    body = resp.json()
    assert "history" in body
    assert len(body["history"]) == 3
    assert body["history"][0]["status"] == "completed"


def test_training_history_limit_clamped_to_200(monkeypatch):
    """
    limit=999 query param must be clamped to 200 before forwarding to
    read_training_history.
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)

    captured_limit = {}

    def mock_history(limit):
        captured_limit["value"] = limit
        return []

    monkeypatch.setattr("app.routers.admin.training.read_training_history", mock_history)

    resp = client.get(
        "/admin/training/history",
        params={"limit": 999},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    assert captured_limit["value"] == 200


def test_training_history_limit_clamped_min_1(monkeypatch):
    """limit=0 or negative must be clamped to 1."""
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)

    captured_limit = {}

    def mock_history(limit):
        captured_limit["value"] = limit
        return []

    monkeypatch.setattr("app.routers.admin.training.read_training_history", mock_history)

    resp = client.get(
        "/admin/training/history",
        params={"limit": 0},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    assert captured_limit["value"] == 1


# ---------------------------------------------------------------------------
# (f) GET /admin/devices/{deveui}/scores
# ---------------------------------------------------------------------------

def test_device_scores_200_with_rows(monkeypatch):
    """Patched fetch returning 2 rows → 200 with count=2."""
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)

    fake_rows = [
        {"created_at": "2026-06-11T10:00:00+00:00", "ensemble_score": 0.3, "anomaly_detected": False},
        {"created_at": "2026-06-11T09:00:00+00:00", "ensemble_score": 0.8, "anomaly_detected": True},
    ]

    monkeypatch.setattr(
        "app.routers.admin.supabase_client.fetch_analysis_results",
        lambda deveui, limit: fake_rows,
    )

    resp = client.get(f"/admin/devices/{TEST_DEVEUI}/scores", headers=_auth_headers())
    assert resp.status_code == 200
    body = resp.json()
    assert body["deveui"] == TEST_DEVEUI
    assert body["count"] == 2
    assert len(body["scores"]) == 2


def test_device_scores_404_on_empty(monkeypatch):
    """Patched fetch returning [] → 404."""
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)

    monkeypatch.setattr(
        "app.routers.admin.supabase_client.fetch_analysis_results",
        lambda deveui, limit: [],
    )

    resp = client.get(f"/admin/devices/{TEST_DEVEUI}/scores", headers=_auth_headers())
    assert resp.status_code == 404
    assert "no analysis results" in resp.json()["detail"]


def test_device_scores_deveui_normalised(monkeypatch):
    """Lowercase deveui in URL is normalised to uppercase in response."""
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)

    monkeypatch.setattr(
        "app.routers.admin.supabase_client.fetch_analysis_results",
        lambda deveui, limit: [{"created_at": "2026-06-11T10:00:00+00:00"}],
    )

    resp = client.get(
        f"/admin/devices/{TEST_DEVEUI.lower()}/scores",
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    assert resp.json()["deveui"] == TEST_DEVEUI


def test_device_scores_limit_clamped(monkeypatch):
    """limit=999 is clamped to 200 before Supabase call."""
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)

    captured = {}

    def mock_fetch(deveui, limit):
        captured["limit"] = limit
        return [{"created_at": "t"}]

    monkeypatch.setattr("app.routers.admin.supabase_client.fetch_analysis_results", mock_fetch)

    resp = client.get(
        f"/admin/devices/{TEST_DEVEUI}/scores",
        params={"limit": 999},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    assert captured["limit"] == 200


# ---------------------------------------------------------------------------
# (g) Auth inheritance — all four new endpoints return 401 without key
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,path", [
    ("post",  f"/admin/train/{TEST_DEVEUI}"),
    ("get",   "/admin/models/status"),
    ("get",   "/admin/training/history"),
    ("get",   f"/admin/devices/{TEST_DEVEUI}/scores"),
])
def test_no_key_returns_401(monkeypatch, method, path):
    """
    With admin_api_key configured but NO X-Admin-Key header sent,
    all four new endpoints must return 401 (auth inherited from router).
    """
    settings = get_settings()
    monkeypatch.setattr(settings, "admin_api_key", TEST_KEY)

    fn = getattr(client, method)
    resp = fn(path)  # no headers — no X-Admin-Key

    assert resp.status_code == 401, (
        f"Expected 401 for {method.upper()} {path}, got {resp.status_code}"
    )
