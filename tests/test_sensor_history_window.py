"""
tests/test_sensor_history_window.py
-----------------------------------
Regression tests for the P4 fetch_sensor_history window bug.

Prior bug: the query ordered created_at ASCENDING and paginated from offset 0,
so a device with more rows than `limit` always got the FIRST `limit` uplinks
ever recorded — a frozen historical window (production: identical ensemble
scores across 41 analyses, readings_used pinned at 2000, days_of_data frozen).

Contract under test: fetch_sensor_history returns the MOST RECENT `limit`
rows, in chronological (oldest-first) order.

Mock seam: app.services.supabase_client._get_client — a fake client that
serves a synthetic dataset through the real pagination loop, honouring the
order direction and range() slices the production code requests.
"""

import os

os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key")

from app.services import supabase_client


# ---------------------------------------------------------------------------
# Fake Supabase client — replays a canned dataset through the query chain
# ---------------------------------------------------------------------------

def _make_dataset(n: int):
    """n rows with lexically-sortable ascending created_at timestamps."""
    return [
        {
            "deveui": "TESTDEVICE000001",
            "decoded_payload": {"flow_rate": float(i)},
            "created_at": f"2026-01-01T00:00:00.{i:06d}+00:00",
        }
        for i in range(n)
    ]


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    """Chainable stand-in for the PostgREST query builder."""

    def __init__(self, dataset, calls):
        self._dataset = dataset
        self._calls = calls          # shared list recording (desc, start, end)
        self._desc = None
        self._start = None
        self._end = None

    def select(self, *_args, **_kw):
        return self

    def eq(self, *_args, **_kw):
        return self

    def order(self, _col, desc=False):
        self._desc = desc
        return self

    def range(self, start, end):
        self._start, self._end = start, end
        return self

    def execute(self):
        self._calls.append((self._desc, self._start, self._end))
        rows = sorted(
            self._dataset,
            key=lambda r: r["created_at"],
            reverse=bool(self._desc),
        )
        return _FakeResponse(rows[self._start : self._end + 1])


class _FakeClient:
    def __init__(self, dataset, calls):
        self._dataset = dataset
        self._calls = calls

    def table(self, _name):
        return _FakeQuery(self._dataset, self._calls)


def _install(monkeypatch, n_rows):
    calls = []
    client = _FakeClient(_make_dataset(n_rows), calls)
    monkeypatch.setattr(supabase_client, "_get_client", lambda: client)
    return calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_over_limit_returns_latest_window_ascending(monkeypatch):
    """1200 rows, limit 1000 → exactly the LATEST 1000, oldest-first."""
    calls = _install(monkeypatch, 1200)

    rows = supabase_client.fetch_sensor_history("TESTDEVICE000001", limit=1000)

    # Exactly the latest 1000 rows: indices 200..1199 of the ascending dataset
    assert len(rows) == 1000
    assert rows[0]["decoded_payload"]["flow_rate"] == 200.0   # oldest of the window
    assert rows[-1]["decoded_payload"]["flow_rate"] == 1199.0  # newest row overall

    # Ascending chronological order throughout
    timestamps = [r["created_at"] for r in rows]
    assert timestamps == sorted(timestamps)

    # Pagination: two descending pages of 500 from offset 0
    assert calls == [(True, 0, 499), (True, 500, 999)]


def test_under_limit_returns_all_ascending(monkeypatch):
    """300 rows, limit 1000 → all 300 returned, oldest-first."""
    calls = _install(monkeypatch, 300)

    rows = supabase_client.fetch_sensor_history("TESTDEVICE000001", limit=1000)

    assert len(rows) == 300
    assert rows[0]["decoded_payload"]["flow_rate"] == 0.0
    assert rows[-1]["decoded_payload"]["flow_rate"] == 299.0

    timestamps = [r["created_at"] for r in rows]
    assert timestamps == sorted(timestamps)

    # Single descending page request; short batch terminates the loop
    assert calls == [(True, 0, 499)]
