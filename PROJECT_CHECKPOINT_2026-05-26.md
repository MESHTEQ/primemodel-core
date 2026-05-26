# PrimeModel Development Checkpoint
Date: 26 May 2026
Previous Checkpoint: 25 May 2026
Owner: Radzeery Fahmi Dzulkifli
Venture: Meshteq Sdn Bhd

---

## Session Summary

Major refactor completed. The /analyse endpoint is now sensor-type agnostic,
tested live against real Supabase data, and committed to main.

---

## Git State

**Branch:** main
**Remote:** github.com/MESHTEQ/primemodel-core (note: repo moved from meshteq-dev-01)
**Status:** Clean — nothing to commit, working tree clean
**Last 3 commits:**

| Hash | Date | Description |
|------|------|-------------|
| 26c7ad6 | 26 May 2026 | Fix Layer 1 activation logic, pin supabase dep, clean gitignore |
| 19454a0 | 26 May 2026 | feat: Introduce sensor-agnostic analysis endpoint and retain legacy endpoint |
| f3dfac0 | 25 May 2026 | Merge branch 'main' from GitHub |

---

## What Was Built This Session (P1–P5)

### P1 — Sensor-agnostic architecture
| File | Change |
|------|--------|
| app/services/decoder_registry.py | NEW — 5 decoder types + generic fallback |
| app/services/device_registry.py | NEW — deveui → device_type + metadata map |
| app/schemas/analyse.py | NEW — AnalyseRequest / AnalyseResponse schemas |
| app/services/supabase_client.py | Renamed legacy fetch, added fetch_sensor_history + write_analysis_result |
| app/routers/analyse.py | New POST /analyse (sensor-agnostic) + POST /analyse/legacy (water-meter, preserved) |

### P2 — .env and live Supabase connection
- .env created (not committed — in .gitignore)
- .gitignore created
- supabase Python package installed in venv

### P3 — Fetch and dependency fixes
- fetch_sensor_history: paginated fetch bypasses Supabase 500-row cap
- Router limit hardcode removed — uses function default (2000)
- requirements.txt: supabase==2.30.0, storage3==0.7.7 pinned

### P4 — Layer 1 activation fix
- Isolation Forest: trains + scores at >= 50 readings (no calendar gate)
- Burst Detector: activates at >= 10 readings
- IF trains and scores in the same request on first call
- CUSUM/EWMA: water devices only, existing calendar gate unchanged

---

## Live Test Results (26 May 2026)

### Temp/Humidity Sensor — 24E124136D355878
- 2,000 readings | 13.92 days
- layer1_scores: temperature=0.3599, humidity=0.305
- ensemble_score: 0.3325 | anomaly_detected: true
- active_layers: ["statistical"]
- IF models saved: models_store/isolation_forest/24E124136D355878_temperature.joblib
                                                 24E124136D355878_humidity.joblib

### Distance Sensor — 24E124713D321914
- 2,000 readings | 11.28 days
- layer1_scores: distance=0.4255, battery=0.3132
- ensemble_score: 0.3693 | anomaly_detected: true
- active_layers: ["statistical"]
- IF models saved: models_store/isolation_forest/24E124713D321914_distance.joblib
                                                 24E124713D321914_battery.joblib

---

## Registered Devices (in-memory, Phase 2 → Supabase)

| deveui | device_type | name | Location |
|--------|-------------|------|----------|
| 24E124136D355878 | temp_humidity | Temp & Humidity Sensor | Miri |
| 24E124713D321914 | distance | Distance Sensor | Miri |
| 24E124747D260328 | power_monitor | Power Monitor | Miri |
| 24E124993D091615 | leak_detector | Leak Detector | Miri |

---

## Supabase State

- **Project URL:** https://pkloaajhalichjopzlob.supabase.co
- **Table used:** lorawan_uplinks (deveui, decoded_payload, created_at)
- **analysis_results table:** NOT YET CREATED — write is stubbed
- **Total rows (24E124136D355878):** ~2,130 as of 26 May 2026
- **Data range:** 11 May 2026 → present

---

## Architecture State

| Dimension | Status |
|-----------|--------|
| Layer 1 (Statistical) | LIVE — IF + Burst active for all sensors |
| Layer 2 (LSTM AE) | Warming up — needs 30 calendar days |
| Layer 3 (LSTM Forecast) | Warming up — needs 60 calendar days |
| Layer 4 (CNN Pattern) | Warming up — needs 90 calendar days |
| Supabase write-back | STUBBED — analysis_results table not created yet |
| Railway deployment | NOT YET — Dockerfile present, not validated |
| MNF/CUSUM/EWMA | Water devices only — not applicable to current sensor set |

---

## Open Decisions (flagged, pending Radz)

| # | Decision | Context |
|---|----------|---------|
| 1 | anomaly_detected threshold | Currently flags "low" severity (score >= 0.25). Recommend raising to "medium" (>= 0.5) to reduce false positives |
| 2 | isolation_forest_contamination | Currently 0.05 (5%). May be too loose — consider 0.02 |
| 3 | Create analysis_results table in Supabase | Needed before write-back is live |
| 4 | Move device_registry to Supabase table | Currently in-memory dict — Phase 2 |
| 5 | Deploy to Railway | Dockerfile updated but not validated |
| 6 | power_monitor / leak_detector decoders | Both fall back to generic — power_monitor has no numeric fields; leak_detector only extracts battery |

---

## Technical Debt (Active)

| ID | Severity | Description |
|----|----------|-------------|
| TD-PANDA-001 | HIGH | Panda decoder is stub — awaiting payload spec |
| TD-BOVE-001 | HIGH | Bove B39 decoder is stub — awaiting hardware spec |
| TD-ADMIN-001 | MEDIUM | POST /admin/retrain has no auth |
| TD-007 | LOW | MNF window is UTC — may need tz_offset param |

---

## Resume Instruction

When resuming from this checkpoint:

1. Server: `cd primemodel-core && .venv/Scripts/uvicorn app.main:app --port 8000 --reload`
2. Test: `curl -X POST http://127.0.0.1:8000/analyse/ -H "Content-Type: application/json" -d "{\"deveui\": \"24E124136D355878\"}"`
3. Next priority: create analysis_results table in Supabase and enable write-back
4. Note: repo moved to github.com/MESHTEQ/primemodel-core — update remote if needed

To resume, say:
**Resume PrimeModel from 26 May 2026 checkpoint**
