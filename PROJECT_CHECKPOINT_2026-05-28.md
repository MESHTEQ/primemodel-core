# PrimeModel Development Checkpoint
Date: 28 May 2026
Previous Checkpoint: 26 May 2026
Owner: Radzeery Fahmi Dzulkifli
Venture: Meshteq Sdn Bhd

---

## Session Summary

Docker build validated and confirmed production-ready. Two deployment fixes
applied and pushed — Railway PORT expansion and trailing slash redirect
suppression. Remote URL corrected to MESHTEQ org. Ready for Railway deployment.

---

## Git State

**Branch:** main
**Remote:** https://github.com/MESHTEQ/primemodel-core.git
**Status:** Clean — nothing to commit, working tree clean
**Last 6 commits:**

| Hash | Date | Description |
|------|------|-------------|
| c835b6f | 28 May 2026 | Disable trailing slash redirect on all routes |
| 51c8b86 | 28 May 2026 | Fix Railway start command — shell wrap for PORT expansion |
| e393ba0 | 27 May 2026 | feat: Add development checkpoint for major refactor and sensor-agnostic analysis endpoint |
| ea07c85 | 27 May 2026 | Validate and fix Dockerfile for Railway deployment |
| 26c7ad6 | 26 May 2026 | Fix Layer 1 activation logic, pin supabase dep, clean gitignore |
| 19454a0 | 26 May 2026 | feat: Introduce sensor-agnostic analysis endpoint and retain legacy endpoint |

---

## What Was Completed This Session (P6–P8)

### P6 — Docker Build Validation
- Docker image built and container started on port 8000
- /health/ endpoint confirmed live
- /analyse/ endpoint tested against both real sensors
- Stale local uvicorn (PID on 127.0.0.1:8000) was masking Docker results — resolved by testing via [::1]:8000
- Docker results bit-for-bit identical to local P4 results — build confirmed correct
- .dockerignore created: .env excluded from image
- Dockerfile updated: curl installed, explicit mkdir for runtime dirs, improved HEALTHCHECK
- requirements.txt: storage3==2.30.0 pinned for Linux/Docker compatibility
- Committed: ea07c85

### P7 — Railway Start Command Fix
| File | Change |
|------|--------|
| railway.toml | startCommand wrapped in `sh -c "..."` so $PORT expands correctly at runtime |

- Without shell wrap, Railway passes `$PORT` as a literal string — uvicorn fails to bind
- Committed: 51c8b86

### P7-fix — Remote URL Correction
- `git remote set-url origin https://github.com/MESHTEQ/primemodel-core.git`
- Eliminates redirect warning on every push — repo had moved from meshteq-dev-01 org to MESHTEQ org

### P8 — Trailing Slash Redirect Fix
| File | Change |
|------|--------|
| app/main.py | `redirect_slashes=False` added to FastAPI() init |

- Prevents 307 redirect when caller omits or includes trailing slash
- Both `/analyse` and `/analyse/` now respond directly
- Committed: c835b6f

---

## Live Test Results (27 May 2026 — Docker container)

### Temp/Humidity Sensor — 24E124136D355878
- 2,000 readings | ~13.9 days
- ensemble_score: 0.3325 | anomaly_detected: true
- active_layers: ["statistical"]

### Distance Sensor — 24E124713D321914
- 2,000 readings | ~11.3 days
- ensemble_score: 0.3693 | anomaly_detected: true
- active_layers: ["statistical"]

Layer 2, 3, 4 remain null — warming up (requires 30/60/90 calendar days).

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
- **Row count (24E124136D355878):** ~2,130+ as of 26 May 2026

---

## Architecture State

| Dimension | Status |
|-----------|--------|
| Layer 1 (Statistical) | LIVE — IF + Burst active for all sensors |
| Layer 2 (LSTM AE) | Warming up — needs 30 calendar days |
| Layer 3 (LSTM Forecast) | Warming up — needs 60 calendar days |
| Layer 4 (CNN Pattern) | Warming up — needs 90 calendar days |
| Supabase write-back | STUBBED — analysis_results table not created yet |
| Docker build | VALIDATED — image confirmed production-ready |
| Railway deployment | READY — P7 and P8 fixes applied, pending deploy trigger |
| MNF/CUSUM/EWMA | Water devices only — not applicable to current sensor set |

---

## Open Decisions (flagged, pending Radz)

| # | Decision | Context |
|---|----------|---------|
| 1 | anomaly_detected threshold | Currently flags "low" severity (score >= 0.25). Recommend raising to "medium" (>= 0.5) to reduce false positives |
| 2 | isolation_forest_contamination | Currently 0.05 (5%). May be too loose — consider 0.02 |
| 3 | Create analysis_results table in Supabase | Needed before write-back is live |
| 4 | Move device_registry to Supabase table | Currently in-memory dict — Phase 2 |
| 5 | power_monitor / leak_detector decoders | Both fall back to generic — power_monitor has no numeric fields; leak_detector only extracts battery |

---

## Technical Debt (Active)

| ID | Severity | Description |
|----|----------|-------------|
| TD-PANDA-001 | HIGH | Panda decoder is stub — awaiting payload spec |
| TD-BOVE-001 | HIGH | Bove B39 decoder is stub — awaiting hardware spec |
| TD-ADMIN-001 | MEDIUM | POST /admin/retrain has no auth |
| TD-007 | LOW | MNF window is UTC — may need tz_offset param |

---

## Next Priority: P9 — Railway Deployment

The image is validated, both deployment fixes are committed. Steps to deploy:

1. Push to Railway (via GitHub auto-deploy or Railway CLI)
2. Set environment variables in Railway dashboard:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
3. Confirm /health/ returns 200 on Railway-assigned URL
4. Test POST /analyse/ against both sensor deveuis on live Railway URL

---

## Resume Instruction

When resuming from this checkpoint:

1. Local dev: `cd primemodel-core && .venv/Scripts/uvicorn app.main:app --port 8000 --reload`
2. Test local: `curl -X POST http://127.0.0.1:8000/analyse/ -H "Content-Type: application/json" -d "{\"deveui\": \"24E124136D355878\"}"`
3. Next: trigger Railway deployment and validate live endpoint
4. After Railway is live: create analysis_results table in Supabase and enable write-back

To resume, say:
**Resume PrimeModel from 28 May 2026 checkpoint**
