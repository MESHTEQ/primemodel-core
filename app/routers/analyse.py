"""
app/routers/analyse.py
-----------------------
POST /analyse        — sensor-agnostic analysis endpoint (new)
POST /analyse/legacy — original water-meter endpoint (retained)

New endpoint:
    Accepts any LoRaWAN sensor by deveui. Looks up device type from the
    device registry, fetches history from lorawan_uplinks, decodes numeric
    parameters, and runs the full Layer 1–4 pipeline on each parameter.

Legacy endpoint:
    Original water-meter-specific pipeline (Panda/Bove decoders, MNF window,
    battery RUL, drift RUL). Kept intact — do not modify.

Design principles (both endpoints):
    - Any individual layer failure is caught and logged — never crashes the endpoint
    - Warming-up layers return score=0.0 and are excluded from the ensemble
    - State is persisted to disk after every uplink
    - Supabase writes are fire-and-forget with error logging, never blocking the response
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone

# Legacy schemas (water meter)
from app.schemas.uplink import (
    AnalyseRequest as LegacyAnalyseRequest,
    AnalyseResponse as LegacyAnalyseResponse,
    ModelLayerStatuses,
)

# New sensor-agnostic schemas
from app.schemas.analyse import AnalyseRequest, AnalyseResponse

from app.config import get_settings
from app.utils.logger import get_logger
from app.utils.time_utils import parse_iso_timestamp, days_between, is_in_mnf_window

# Services
from app.services import (
    supabase_client,
    feature_engineering,
    model_registry,
    ensemble,
    battery_rul,
    drift_rul,
)
from app.services import decoder_registry, device_registry
from app.services.statistical import isolation_forest, mnf_cusum, mnf_ewma, burst_detector
from app.services.neural import lstm_autoencoder, lstm_forecast, cnn_pattern

# Legacy decoders
from app.services.decoder_panda import decode as decode_panda
from app.services.decoder_bove import decode as decode_bove

logger = get_logger(__name__)
router = APIRouter()

# Parameters that represent radio/hardware telemetry and must never be scored
# as sensor physics. Matched case-insensitively against decoded payload keys.
_SYSTEM_KEYS = {
    "battery", "battery_level", "batt", "bat",
    "rssi", "signal", "snr", "fport", "fcnt",
}


# ===========================================================================
# NEW — Sensor-agnostic POST /analyse
# ===========================================================================

@router.post("/", response_model=AnalyseResponse, tags=["Analysis"])
def analyse(request: AnalyseRequest) -> AnalyseResponse:
    """
    Sensor-agnostic analysis endpoint.

    Accepts any registered LoRaWAN device by deveui. Looks up device type,
    fetches history from lorawan_uplinks, decodes numeric parameters, and
    runs the progressive Layer 1–4 pipeline on each parameter time series.

    Args:
        request: AnalyseRequest with deveui and optional client_id / force_layers.

    Returns:
        AnalyseResponse with per-parameter scores, ensemble score, and metadata.
    """
    settings = get_settings()
    deveui = request.deveui.upper().strip()
    analysis_timestamp = datetime.now(tz=timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Step 1: Resolve device type
    # ------------------------------------------------------------------
    device_info = device_registry.get_device_info(deveui)
    device_type = device_info["device_type"]

    # ------------------------------------------------------------------
    # Step 2: Fetch sensor history from lorawan_uplinks
    # ------------------------------------------------------------------
    rows = supabase_client.fetch_sensor_history(deveui)

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for device {deveui}",
        )

    # ------------------------------------------------------------------
    # Step 3: Decode each row — extract numeric parameters only
    # ------------------------------------------------------------------
    decoder = decoder_registry.get_decoder(device_type)

    # Build per-parameter time series: {param: [(created_at, value), ...]}
    param_series: dict = {}
    timestamps: list = []

    for row in rows:
        payload = row.get("decoded_payload") or {}
        created_at = row.get("created_at", "")
        if created_at:
            timestamps.append(created_at)

        numeric = decoder(payload)
        for param, value in numeric.items():
            if param not in param_series:
                param_series[param] = []
            param_series[param].append((created_at, value))

    # Strip system/health keys so all layers see a clean channel list
    param_series = {k: v for k, v in param_series.items() if k.lower() not in _SYSTEM_KEYS}

    parameters_analysed = [
        p for p in param_series.keys()
        if p.lower() not in _SYSTEM_KEYS
    ]
    readings_used = len(rows)

    # ------------------------------------------------------------------
    # Step 4: Calculate days_of_data
    # ------------------------------------------------------------------
    days_of_data = 0.0
    if len(timestamps) >= 2:
        try:
            first_dt = parse_iso_timestamp(timestamps[0])
            last_dt = parse_iso_timestamp(timestamps[-1])
            days_of_data = days_between(first_dt, last_dt)
        except Exception as e:
            logger.warning("Could not compute days_of_data", extra={"deveui": deveui, "error": str(e)})

    logger.info(
        f"Analysing device {deveui} ({device_type}) — {readings_used} readings, {days_of_data:.1f} days",
        extra={"deveui": deveui, "device_type": device_type, "readings": readings_used, "days": days_of_data},
    )

    # ------------------------------------------------------------------
    # Step 5: Determine active layers
    # Layer 1 (IF + Burst): data-count based — no calendar gate
    # Layers 2/3/4: calendar based (30/60/90 days) — unchanged
    # ------------------------------------------------------------------
    force_layers = set(request.force_layers or [])

    # Layers 2–4 still use calendar thresholds
    ae_active  = days_of_data >= settings.lstm_ae_activation_days         or "lstm_ae"        in force_layers
    fc_active  = days_of_data >= settings.lstm_forecast_activation_days   or "lstm_forecast"  in force_layers
    cnn_active = days_of_data >= settings.cnn_activation_days             or "cnn"            in force_layers

    # Layer 1 active_layers entry added after the loop once we know if any method ran
    active_layers: list = []
    if ae_active:
        active_layers.append("lstm_ae")
    if fc_active:
        active_layers.append("lstm_forecast")
    if cnn_active:
        active_layers.append("cnn")

    # ------------------------------------------------------------------
    # Step 6: Layer 1 — Statistical analysis per parameter
    #
    # Activation rules (independent per method — no shared calendar gate):
    #   Isolation Forest: trains and scores when len(values) >= 50
    #   Burst Detector:   runs when len(values) >= 10
    #   CUSUM / EWMA:     water devices only, existing calendar + MNF window logic
    #
    # "statistical" is added to active_layers if at least one method ran.
    # ------------------------------------------------------------------
    layer1_scores: dict = {}
    anomaly_details: dict = {}
    is_water_device = "water" in device_type.lower()
    layer1_ran = False   # tracks whether any stat method produced a real score

    for param, series in param_series.items():
        values = [v for (_, v) in series]
        if len(values) < 2:
            layer1_scores[param] = 0.0
            continue

        param_score = 0.0
        param_flags = {}

        # Build a minimal DataFrame for this parameter
        param_df = pd.DataFrame({
            "timestamp": [ts for (ts, _) in series],
            "flow_rate": values,   # reuse flow_rate column name for compat with feature_engineering
        })

        # --- Isolation Forest — activates at >= 50 readings ---
        if len(values) >= 50 or "statistical" in force_layers:
            try:
                prev_val = values[-2] if len(values) >= 2 else values[-1]
                val_delta = values[-1] - prev_val
                rolling_mean, rolling_std = feature_engineering.compute_rolling_stats(values)
                try:
                    current_dt = parse_iso_timestamp(series[-1][0])
                except Exception:
                    current_dt = datetime.now(tz=timezone.utc)
                hour_of_day = current_dt.hour + current_dt.minute / 60.0
                day_of_week = current_dt.weekday()
                feat_vec = feature_engineering.build_feature_vector(
                    flow_rate=values[-1],
                    flow_delta=val_delta,
                    hour_of_day=hour_of_day,
                    day_of_week=day_of_week,
                    rolling_mean_1h=rolling_mean,
                    rolling_std_1h=rolling_std,
                )
                model_key = f"{deveui}_{param}"

                # Pre-compute training matrix X — used in both fresh-train and self-heal
                # paths below. Building it here avoids repetition; it is cheap (in memory).
                _flows = np.array(values, dtype=float)
                _deltas = np.concatenate([[0.0], np.diff(_flows)])
                _ts_index = pd.to_datetime(
                    [ts for (ts, _) in series], utc=True, errors="coerce"
                )
                _hours_arr = _ts_index.hour + _ts_index.minute / 60.0
                _dows_arr = _ts_index.dayofweek
                X = np.column_stack([
                    _flows,
                    _deltas,
                    _hours_arr.to_numpy(dtype=float),
                    _dows_arr.to_numpy(dtype=float),
                    pd.Series(_flows).rolling(4, min_periods=1).mean().values,
                    pd.Series(_flows).rolling(4, min_periods=1).std().fillna(0).values,
                ])

                if_model = isolation_forest.load_model(settings.model_store_path, model_key)
                if_calibration_stats = None

                if if_model is None:
                    # Fresh train — model and stats are both produced here
                    if_model, if_calibration_stats = isolation_forest.train(
                        X, contamination=settings.isolation_forest_contamination
                    )
                    isolation_forest.save_model(if_model, settings.model_store_path, model_key)
                    isolation_forest.save_calibration_stats(
                        settings.model_store_path, model_key, if_calibration_stats
                    )
                    logger.info(
                        "IF trained and scoring in same request",
                        extra={"deveui": deveui, "param": param, "n": len(values)},
                    )
                else:
                    # Model exists — load calibration stats
                    if_calibration_stats = isolation_forest.load_calibration_stats(
                        settings.model_store_path, model_key
                    )
                    if if_calibration_stats is None:
                        # Self-heal: model artifact present but calibration stats file is
                        # absent (e.g. stats lost after volume migration or partial write).
                        # Retrain in-request to regenerate stats; replace the model artifact
                        # as well so both files are consistent.
                        logger.info(
                            "[L1 self-heal] calibration stats missing for %s — retraining in-request",
                            model_key,
                        )
                        try:
                            if_model, if_calibration_stats = isolation_forest.train(
                                X, contamination=settings.isolation_forest_contamination
                            )
                            isolation_forest.save_model(
                                if_model, settings.model_store_path, model_key
                            )
                            isolation_forest.save_calibration_stats(
                                settings.model_store_path, model_key, if_calibration_stats
                            )
                        except Exception as heal_err:
                            logger.warning(
                                "[L1 self-heal] retrain failed for %s: %s — using legacy sigmoid",
                                model_key,
                                heal_err,
                            )
                            if_calibration_stats = None  # explicit — legacy fallback fires in score()

                # Score — if_calibration_stats may be None here only when self-heal failed;
                # isolation_forest.score() handles None via its legacy sigmoid fallback.
                if_score, is_if_anomaly = isolation_forest.score(
                    if_model, feat_vec, if_calibration_stats
                )
                param_score = max(param_score, if_score)
                layer1_ran = True
                if is_if_anomaly:
                    param_flags["isolation_forest"] = True
            except Exception as e:
                logger.warning("IF scoring failed", extra={"deveui": deveui, "param": param, "error": str(e)})

        # --- Burst Detector — activates at >= 10 readings ---
        if len(values) >= 10 or "statistical" in force_layers:
            try:
                burst_state = burst_detector.initialise_state(values)
                prev_val = values[-2] if len(values) >= 2 else values[-1]
                burst_score_val, burst_detected_flag = burst_detector.detect(
                    burst_state, values[-1], prev_val, settings.burst_threshold_sigma
                )
                param_score = max(param_score, burst_score_val)
                layer1_ran = True
                if burst_detected_flag:
                    param_flags["burst_detected"] = True
            except Exception as e:
                logger.warning("Burst detection failed", extra={"deveui": deveui, "param": param, "error": str(e)})

        # --- CUSUM / EWMA — water devices only, calendar + MNF window gate unchanged ---
        if is_water_device and days_of_data >= settings.cold_start_days:
            try:
                current_dt_parsed = parse_iso_timestamp(series[-1][0])
                in_mnf_window = is_in_mnf_window(
                    current_dt_parsed,
                    settings.mnf_window_start,
                    settings.mnf_window_end,
                )
                if in_mnf_window:
                    daily_mnf = feature_engineering.extract_daily_mnf(
                        param_df,
                        window_start_hour=settings.mnf_window_start,
                        window_end_hour=settings.mnf_window_end,
                    )
                    if len(daily_mnf) >= 3:
                        cusum_state = mnf_cusum.initialise_state(daily_mnf.values)
                        cusum_state, cusum_score, mnf_flag_cusum = mnf_cusum.update(
                            cusum_state, values[-1], settings.cusum_threshold_sigma
                        )
                        ewma_state = mnf_ewma.initialise_state(daily_mnf.values, settings.ewma_lambda)
                        ewma_state, ewma_score, mnf_flag_ewma = mnf_ewma.update(
                            ewma_state, values[-1], settings.cusum_threshold_sigma
                        )
                        mnf_score = max(cusum_score, ewma_score)
                        param_score = max(param_score, mnf_score)
                        layer1_ran = True
                        if mnf_flag_cusum or mnf_flag_ewma:
                            param_flags["mnf_flag"] = True
            except Exception as e:
                logger.warning("MNF analysis failed", extra={"deveui": deveui, "param": param, "error": str(e)})

        layer1_scores[param] = round(min(param_score, 1.0), 4)
        if param_flags:
            anomaly_details[param] = param_flags

    # Add "statistical" to active_layers if at least one method ran
    if layer1_ran:
        active_layers.insert(0, "statistical")

    # ------------------------------------------------------------------
    # Step 7: Layer 2 — LSTM Autoencoder (multi-parameter)
    # Scores every parameter that has a trained model artifact.
    # layer2_scores: per-param dict. layer2_score: MAX across params (worst-case).
    # Silent skip if no model exists for a param — no log, no entry in dict.
    # ------------------------------------------------------------------
    layer2_scores: Dict[str, float] = {}
    if ae_active and parameters_analysed:
        for param in parameters_analysed:
            try:
                ae_model_key = f"{deveui}_{param}"
                ae_model = lstm_autoencoder.load_model(settings.model_store_path, ae_model_key)
                if ae_model is None:
                    continue
                param_values = [v for (_, v) in param_series[param]]
                param_df = pd.DataFrame({"timestamp": [ts for (ts, _) in param_series[param]], "flow_rate": param_values})
                sequence = feature_engineering.build_lstm_sequence(param_df)
                if sequence is None:
                    continue
                ae_stats = lstm_autoencoder.load_threshold_stats(settings.model_store_path, ae_model_key)
                if ae_stats is not None:
                    ae_threshold_stats = ae_stats
                else:
                    ae_threshold_stats = {"mae_mean": 0.0, "mae_std": 1.0, "threshold": 1.0}
                    logger.warning(
                        "LSTM AE scoring without calibrated threshold stats — results unreliable",
                        extra={"model_key": ae_model_key},
                    )
                ae_score_val, _ = lstm_autoencoder.score(ae_model, sequence, ae_threshold_stats)
                layer2_scores[param] = round(float(ae_score_val), 4)
            except Exception as e:
                logger.warning("LSTM AE scoring failed", extra={"deveui": deveui, "param": param, "error": str(e)})

    layer2_score: Optional[float] = max(layer2_scores.values()) if layer2_scores else None

    # ------------------------------------------------------------------
    # Step 8: Layer 3 — LSTM Forecast
    # ------------------------------------------------------------------
    layer3_score: Optional[float] = None
    if fc_active and parameters_analysed:
        try:
            primary_param = parameters_analysed[0]
            primary_values = [v for (_, v) in param_series[primary_param]]
            primary_df = pd.DataFrame({"timestamp": [ts for (ts, _) in param_series[primary_param]], "flow_rate": primary_values})
            if len(primary_df) >= feature_engineering.SEQUENCE_LENGTH + 4:
                context_seq = feature_engineering.build_lstm_sequence(primary_df.iloc[:-4])
                actual_next = primary_df["flow_rate"].values[-4:].astype(np.float32)
                if context_seq is not None:
                    fc_model_key = f"{deveui}_{primary_param}"
                    fc_model = lstm_forecast.load_model(settings.model_store_path, fc_model_key)
                    if fc_model is not None:
                        fc_baseline = {"rmse_mean": 0.0, "rmse_std": 1.0}
                        fc_score_val, _ = lstm_forecast.score(fc_model, context_seq, actual_next, fc_baseline)
                        layer3_score = round(float(fc_score_val), 4)
        except Exception as e:
            logger.warning("LSTM Forecast scoring failed", extra={"deveui": deveui, "error": str(e)})

    # ------------------------------------------------------------------
    # Step 9: Layer 4 — CNN Pattern
    # ------------------------------------------------------------------
    layer4_score: Optional[float] = None
    if cnn_active and parameters_analysed:
        try:
            primary_param = parameters_analysed[0]
            primary_values = [v for (_, v) in param_series[primary_param]]
            primary_df = pd.DataFrame({"timestamp": [ts for (ts, _) in param_series[primary_param]], "flow_rate": primary_values})
            cnn_sequence = feature_engineering.build_cnn_sequence(primary_df)
            if cnn_sequence is not None:
                cnn_model = cnn_pattern.load_model(settings.model_store_path, deveui)
                if cnn_model is not None:
                    cnn_score_val, _ = cnn_pattern.score(cnn_model, cnn_sequence)
                    layer4_score = round(float(cnn_score_val), 4)
        except Exception as e:
            logger.warning("CNN scoring failed", extra={"deveui": deveui, "error": str(e)})

    # ------------------------------------------------------------------
    # Step 10: Ensemble score
    # ------------------------------------------------------------------
    ensemble_score: Optional[float] = None
    anomaly_detected = False

    try:
        # Compute a single statistical_score from layer1 scores (mean across params)
        stat_score = float(np.mean(list(layer1_scores.values()))) if layer1_scores else 0.0

        ensemble_result = ensemble.compute_ensemble(
            statistical_score=stat_score,
            autoencoder_score=layer2_score or 0.0,
            forecast_score=layer3_score or 0.0,
            cnn_score=layer4_score or 0.0,
            statistical_active=layer1_ran,
            lstm_ae_active=ae_active and layer2_score is not None,
            lstm_forecast_active=fc_active and layer3_score is not None,
            cnn_active=cnn_active and layer4_score is not None,
        )
        ensemble_score = round(float(ensemble_result["leak_probability"]), 4)
        anomaly_detected = ensemble_result["leak_severity"] not in ("none",)
    except Exception as e:
        logger.warning("Ensemble scoring failed", extra={"deveui": deveui, "error": str(e)})

    # ------------------------------------------------------------------
    # Step 11: Build response and write result (stubbed)
    # ------------------------------------------------------------------
    response = AnalyseResponse(
        deveui=deveui,
        device_type=device_type,
        parameters_analysed=parameters_analysed,
        readings_used=readings_used,
        days_of_data=round(days_of_data, 2),
        layer1_scores=layer1_scores,
        layer2_score=layer2_score,
        layer2_scores=layer2_scores,
        layer3_score=layer3_score,
        layer4_score=layer4_score,
        ensemble_score=ensemble_score,
        anomaly_detected=anomaly_detected,
        anomaly_details=anomaly_details if anomaly_details else None,
        analysis_timestamp=analysis_timestamp,
        active_layers=active_layers,
    )

    try:
        supabase_client.write_analysis_result(deveui, response.model_dump())
    except Exception as e:
        logger.error("write_analysis_result failed (non-fatal)", extra={"deveui": deveui, "error": str(e)})

    return response


# ===========================================================================
# LEGACY — Water-meter POST /analyse/legacy (original code, untouched)
# ===========================================================================

@router.post("/legacy", response_model=LegacyAnalyseResponse, tags=["Analysis (Legacy)"])
def analyse_legacy(request: LegacyAnalyseRequest) -> LegacyAnalyseResponse:
    """
    Analyse a single meter uplink and return a scored result.

    This is the primary endpoint called by the Supabase Edge Function
    after every ThingPark uplink arrives.

    Args:
        request: AnalyseRequest payload from the Edge Function.

    Returns:
        AnalyseResponse with anomaly scores, leak probability, RUL, and model status.
    """
    settings = get_settings()
    meter_id = request.meter_id
    client_id = request.client_id
    timestamp = request.timestamp

    logger.info(
        "Analyse request received",
        extra={"meter_id": meter_id, "client_id": client_id, "timestamp": timestamp},
    )

    # ------------------------------------------------------------------
    # Step 1: Decode raw payload if present
    # The decoded values are advisory — the request body values take precedence
    # because the Edge Function has already extracted them from ThingPark.
    # ------------------------------------------------------------------
    if request.raw_payload:
        try:
            if request.meter_type.value == "panda":
                _decoded = decode_panda(request.raw_payload)
            else:
                _decoded = decode_bove(request.raw_payload)
            # Only log — decoded values are not overriding request values (STUB decoders)
            logger.debug("Payload decoded (stub)", extra={"decoded": _decoded, "meter_type": request.meter_type.value})
        except Exception as e:
            logger.warning("Payload decode failed", extra={"error": str(e), "meter_id": meter_id})

    # ------------------------------------------------------------------
    # Step 2: Persist uplink to Supabase (fire-and-forget)
    # ------------------------------------------------------------------
    try:
        supabase_client.insert_uplink(
            meter_id=meter_id,
            client_id=client_id,
            timestamp=timestamp,
            flow_rate=request.flow_rate,
            cumulative_volume=request.cumulative_volume,
            battery_level=request.battery_level,
            alarm_flags=request.alarm_flags,
            raw_payload=request.raw_payload,
        )
    except Exception as e:
        logger.error("Uplink insert failed (non-fatal)", extra={"meter_id": meter_id, "error": str(e)})

    # ------------------------------------------------------------------
    # Step 3: Load meter state and update with this uplink
    # ------------------------------------------------------------------
    state = model_registry.load_state(settings.model_store_path, meter_id)
    state = model_registry.record_uplink(
        state=state,
        flow_rate=request.flow_rate,
        timestamp=timestamp,
        cold_start_days=settings.cold_start_days,
        lstm_ae_activation_days=settings.lstm_ae_activation_days,
        lstm_forecast_activation_days=settings.lstm_forecast_activation_days,
        cnn_activation_days=settings.cnn_activation_days,
    )

    # ------------------------------------------------------------------
    # Step 4: Fetch uplink history for feature computation
    # ------------------------------------------------------------------
    history_rows = supabase_client.fetch_uplink_history_legacy(meter_id, client_id, limit=500)
    history_df = pd.DataFrame(history_rows) if history_rows else pd.DataFrame(
        columns=["timestamp", "flow_rate", "cumulative_volume", "battery_level"]
    )

    # ------------------------------------------------------------------
    # Step 5: Statistical layer
    # ------------------------------------------------------------------
    if_score = 0.0
    is_if_anomaly = False
    cusum_score = 0.0
    ewma_score = 0.0
    burst_score_val = 0.0
    burst_detected = False
    mnf_flag = False
    mnf_value = None

    if state["statistical_active"]:
        # --- Feature vector for Isolation Forest ---
        flow_history: list = state.get("flow_history", [request.flow_rate])
        prev_flow = flow_history[-2] if len(flow_history) >= 2 else request.flow_rate
        flow_delta = request.flow_rate - prev_flow

        rolling_mean, rolling_std = feature_engineering.compute_rolling_stats(flow_history)

        try:
            current_dt = parse_iso_timestamp(timestamp)
        except Exception:
            current_dt = datetime.now(tz=timezone.utc)

        hour_of_day = current_dt.hour + current_dt.minute / 60.0
        day_of_week = current_dt.weekday()

        feat_vec = feature_engineering.build_feature_vector(
            flow_rate=request.flow_rate,
            flow_delta=flow_delta,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            rolling_mean_1h=rolling_mean,
            rolling_std_1h=rolling_std,
        )

        # --- Isolation Forest ---
        if_model = isolation_forest.load_model(settings.model_store_path, meter_id)
        if if_model is None and model_registry.needs_retraining(state, "isolation_forest", settings.retrain_interval_days):
            if_model = _train_isolation_forest(history_df, settings, state, meter_id)

        if if_model is not None:
            try:
                if_calibration_stats = isolation_forest.load_calibration_stats(settings.model_store_path, meter_id)
                if_score, is_if_anomaly = isolation_forest.score(if_model, feat_vec, if_calibration_stats)
            except Exception as e:
                logger.warning("IF scoring failed", extra={"meter_id": meter_id, "error": str(e)})

        # --- Burst detection ---
        burst_state = state.get("burst_state")
        if burst_state is None and len(flow_history) >= 10:
            burst_state = burst_detector.initialise_state(flow_history)
            state["burst_state"] = burst_state

        if burst_state is not None:
            try:
                burst_score_val, burst_detected = burst_detector.detect(
                    burst_state, request.flow_rate, prev_flow, settings.burst_threshold_sigma
                )
            except Exception as e:
                logger.warning("Burst detection failed", extra={"meter_id": meter_id, "error": str(e)})

        # --- MNF check ---
        try:
            current_dt_parsed = parse_iso_timestamp(timestamp)
            in_mnf_window = is_in_mnf_window(
                current_dt_parsed,
                settings.mnf_window_start,
                settings.mnf_window_end,
            )
        except Exception:
            in_mnf_window = False

        if in_mnf_window:
            mnf_value = request.flow_rate
            daily_mnf = feature_engineering.extract_daily_mnf(
                history_df,
                window_start_hour=settings.mnf_window_start,
                window_end_hour=settings.mnf_window_end,
            )
            if len(daily_mnf) >= 3:
                mnf_baseline_mean = float(daily_mnf.mean())
                mnf_baseline_std = float(daily_mnf.std()) + 1e-9
                # Update CUSUM
                cusum_state = state.get("cusum_state")
                if cusum_state is None:
                    cusum_state = mnf_cusum.initialise_state(daily_mnf.values)
                try:
                    cusum_state, cusum_score, mnf_flag_cusum = mnf_cusum.update(
                        cusum_state, request.flow_rate, settings.cusum_threshold_sigma
                    )
                    state["cusum_state"] = cusum_state
                    if mnf_flag_cusum:
                        mnf_flag = True
                except Exception as e:
                    logger.warning("CUSUM update failed", extra={"error": str(e)})

                # Update EWMA
                ewma_state = state.get("ewma_state")
                if ewma_state is None:
                    ewma_state = mnf_ewma.initialise_state(daily_mnf.values, settings.ewma_lambda)
                try:
                    ewma_state, ewma_score, mnf_flag_ewma = mnf_ewma.update(
                        ewma_state, request.flow_rate, settings.cusum_threshold_sigma
                    )
                    state["ewma_state"] = ewma_state
                    if mnf_flag_ewma:
                        mnf_flag = True
                except Exception as e:
                    logger.warning("EWMA update failed", extra={"error": str(e)})
    else:
        logger.debug("Statistical layer warming up", extra={"meter_id": meter_id, "days": state.get("days_of_data", 0.0)})

    statistical_score = ensemble.combine_statistical_scores(
        isolation_forest_score=if_score,
        cusum_score=cusum_score,
        ewma_score=ewma_score,
        burst_score=burst_score_val,
    )

    # ------------------------------------------------------------------
    # Step 6: LSTM Autoencoder
    # ------------------------------------------------------------------
    ae_score = 0.0
    is_ae_active = state["lstm_ae_active"]

    if is_ae_active:
        sequence = feature_engineering.build_lstm_sequence(history_df)
        if sequence is not None:
            ae_model = lstm_autoencoder.load_model(settings.model_store_path, meter_id)
            if ae_model is None and model_registry.needs_retraining(state, "lstm_ae", settings.retrain_interval_days):
                ae_model = _train_lstm_autoencoder(history_df, settings, state, meter_id)
            if ae_model is not None:
                ae_threshold_stats = state.get("lstm_ae_threshold_stats", {"mae_mean": 0.0, "mae_std": 1.0, "threshold": 1.0})
                try:
                    ae_score, _ = lstm_autoencoder.score(ae_model, sequence, ae_threshold_stats)
                except Exception as e:
                    logger.warning("AE scoring failed", extra={"meter_id": meter_id, "error": str(e)})

    # ------------------------------------------------------------------
    # Step 7: LSTM Forecast
    # ------------------------------------------------------------------
    fc_score = 0.0
    is_fc_active = state["lstm_forecast_active"]

    if is_fc_active and len(history_df) >= feature_engineering.SEQUENCE_LENGTH + 4:
        # We need seq_len context + 4 actual future readings
        context_seq = feature_engineering.build_lstm_sequence(
            history_df.iloc[:-4],  # exclude last 4 rows
        )
        actual_next = history_df["flow_rate"].values[-4:].astype(np.float32)

        if context_seq is not None:
            fc_model = lstm_forecast.load_model(settings.model_store_path, meter_id)
            if fc_model is None and model_registry.needs_retraining(state, "lstm_forecast", settings.retrain_interval_days):
                fc_model = _train_lstm_forecast(history_df, settings, state, meter_id)
            if fc_model is not None:
                fc_baseline = state.get("lstm_forecast_baseline", {"rmse_mean": 0.0, "rmse_std": 1.0})
                try:
                    fc_score, _ = lstm_forecast.score(fc_model, context_seq, actual_next, fc_baseline)
                except Exception as e:
                    logger.warning("Forecast scoring failed", extra={"meter_id": meter_id, "error": str(e)})

    # ------------------------------------------------------------------
    # Step 8: 1D CNN Pattern Recognition
    # ------------------------------------------------------------------
    cnn_score_val = 0.0
    pattern_type = "normal"
    is_cnn_active = state["cnn_active"]

    # zone_id defaults to client_id if meter has no zone assignment
    meter_meta = None
    try:
        meter_meta = supabase_client.fetch_meter_metadata(meter_id, client_id)
    except Exception:
        pass
    zone_id = (meter_meta or {}).get("zone_id", client_id)

    if is_cnn_active:
        cnn_sequence = feature_engineering.build_cnn_sequence(history_df)
        if cnn_sequence is not None:
            cnn_model = cnn_pattern.load_model(settings.model_store_path, zone_id)
            if cnn_model is not None:
                try:
                    cnn_score_val, pattern_type = cnn_pattern.score(cnn_model, cnn_sequence)
                except Exception as e:
                    logger.warning("CNN scoring failed", extra={"meter_id": meter_id, "error": str(e)})

    # ------------------------------------------------------------------
    # Step 9: Ensemble
    # ------------------------------------------------------------------
    result = ensemble.compute_ensemble(
        statistical_score=statistical_score,
        autoencoder_score=ae_score,
        forecast_score=fc_score,
        cnn_score=cnn_score_val,
        statistical_active=state["statistical_active"],
        lstm_ae_active=is_ae_active,
        lstm_forecast_active=is_fc_active,
        cnn_active=is_cnn_active,
    )

    # ------------------------------------------------------------------
    # Step 10: Battery RUL
    # ------------------------------------------------------------------
    battery_rul_days_val = None
    battery_rul_status_val = "ok"
    try:
        battery_rows = supabase_client.fetch_battery_history(meter_id, client_id, limit=200)
        if battery_rows:
            b_levels = [r["battery_level"] for r in battery_rows]
            first_ts = parse_iso_timestamp(battery_rows[0]["timestamp"])
            b_times = [
                days_between(first_ts, parse_iso_timestamp(r["timestamp"]))
                for r in battery_rows
            ]
            battery_rul_days_val, battery_rul_status_val = battery_rul.estimate_rul(
                b_levels, b_times,
                eol_percent=settings.battery_eol_percent,
                warning_days=settings.battery_warning_days,
                critical_days=settings.battery_critical_days,
            )
        else:
            battery_rul_status_val = battery_rul.classify_from_level(
                request.battery_level,
                eol_percent=settings.battery_eol_percent,
            )
    except Exception as e:
        logger.warning("Battery RUL computation failed", extra={"meter_id": meter_id, "error": str(e)})
        battery_rul_status_val = "ok"

    # ------------------------------------------------------------------
    # Step 11: Drift RUL
    # ------------------------------------------------------------------
    drift_rul_days_val = None
    drift_rul_status_val = "warming_up"
    drift_offset_val = None
    try:
        drift_state = state.get("drift_state", {})
        if not history_df.empty:
            today_mean = float(history_df["flow_rate"].mean())
            drift_state, drift_rul_days_val, drift_rul_status_val, drift_offset_val = drift_rul.update_drift_state(
                drift_state,
                today_mean,
                mnf_baseline_days=settings.mnf_baseline_days,
                accuracy_threshold_percent=settings.drift_accuracy_threshold_percent,
            )
            state["drift_state"] = drift_state
    except Exception as e:
        logger.warning("Drift RUL computation failed", extra={"meter_id": meter_id, "error": str(e)})

    # ------------------------------------------------------------------
    # Step 12: Persist state
    # ------------------------------------------------------------------
    model_registry.save_state(settings.model_store_path, meter_id, state)

    # ------------------------------------------------------------------
    # Step 13: Build response
    # ------------------------------------------------------------------
    layer_status = model_registry.get_activation_status(
        state,
        settings.cold_start_days,
        settings.lstm_ae_activation_days,
        settings.lstm_forecast_activation_days,
        settings.cnn_activation_days,
    )

    explanation = _build_explanation(
        statistical_active=state["statistical_active"],
        is_if_anomaly=is_if_anomaly,
        mnf_flag=mnf_flag,
        burst_detected=burst_detected,
        is_ae_active=is_ae_active,
        ae_score=ae_score,
        is_fc_active=is_fc_active,
        fc_score=fc_score,
        is_cnn_active=is_cnn_active,
        cnn_score=cnn_score_val,
        pattern_type=pattern_type,
        leak_probability=result["leak_probability"],
        leak_severity=result["leak_severity"],
        days_of_data=state.get("days_of_data", 0.0),
    )

    response = AnalyseResponse(
        meter_id=meter_id,
        timestamp=timestamp,
        anomaly_score=round(statistical_score, 4),
        is_anomaly=is_if_anomaly,
        mnf_flag=mnf_flag,
        mnf_value=round(mnf_value, 4) if mnf_value is not None else None,
        burst_detected=burst_detected,
        leak_probability=round(result["leak_probability"], 4),
        leak_severity=result["leak_severity"],
        pattern_type=pattern_type,
        battery_rul_days=round(battery_rul_days_val, 1) if battery_rul_days_val is not None else None,
        battery_rul_status=battery_rul_status_val,
        drift_rul_days=round(drift_rul_days_val, 1) if drift_rul_days_val is not None else None,
        drift_rul_status=drift_rul_status_val,
        model_status=ModelLayerStatuses(**layer_status),
        confidence=round(result["confidence"], 4),
        explanation=explanation,
    )

    # Write score back to Supabase (fire-and-forget)
    try:
        supabase_client.write_anomaly_score(meter_id, client_id, response.model_dump())
    except Exception as e:
        logger.error("Anomaly score write failed (non-fatal)", extra={"meter_id": meter_id, "error": str(e)})

    logger.info(
        "Analyse complete",
        extra={
            "meter_id": meter_id,
            "leak_probability": response.leak_probability,
            "leak_severity": response.leak_severity,
            "burst_detected": burst_detected,
            "active_layers": result.get("active_layers", []),
        },
    )

    return response


# ---------------------------------------------------------------------------
# Private training helpers
# ---------------------------------------------------------------------------

def _train_isolation_forest(history_df, settings, state, meter_id):
    """Train Isolation Forest from uplink history and persist."""
    try:
        if history_df.empty or len(history_df) < 50:
            return None

        flows = history_df["flow_rate"].values
        deltas = np.concatenate([[0.0], np.diff(flows)])
        hours = pd.to_datetime(history_df["timestamp"], utc=True).dt.hour + \
                pd.to_datetime(history_df["timestamp"], utc=True).dt.minute / 60.0
        dows = pd.to_datetime(history_df["timestamp"], utc=True).dt.weekday

        rolling_means = pd.Series(flows).rolling(4, min_periods=1).mean().values
        rolling_stds = pd.Series(flows).rolling(4, min_periods=1).std().fillna(0).values

        X = np.column_stack([flows, deltas, hours.values, dows.values, rolling_means, rolling_stds])
        model, calibration_stats = isolation_forest.train(X, contamination=settings.isolation_forest_contamination)
        isolation_forest.save_model(model, settings.model_store_path, meter_id)
        isolation_forest.save_calibration_stats(settings.model_store_path, meter_id, calibration_stats)
        model_registry.mark_trained(state, "isolation_forest")
        logger.info("Isolation Forest trained and saved", extra={"meter_id": meter_id})
        return model
    except Exception as e:
        logger.error("IF training failed", extra={"meter_id": meter_id, "error": str(e)})
        return None


def _train_lstm_autoencoder(history_df, settings, state, meter_id):
    """Train LSTM Autoencoder from uplink history and persist."""
    try:
        sequences = feature_engineering.build_training_sequences(history_df)
        if sequences is None or len(sequences) < 10:
            return None

        model, threshold_stats = lstm_autoencoder.train(sequences)
        lstm_autoencoder.save_model(model, settings.model_store_path, meter_id)
        state["lstm_ae_threshold_stats"] = threshold_stats
        model_registry.mark_trained(state, "lstm_ae")
        logger.info("LSTM AE trained and saved", extra={"meter_id": meter_id})
        return model
    except Exception as e:
        logger.error("LSTM AE training failed", extra={"meter_id": meter_id, "error": str(e)})
        return None


def _train_lstm_forecast(history_df, settings, state, meter_id):
    """Train LSTM Forecast model from uplink history and persist."""
    try:
        sequences = feature_engineering.build_training_sequences(history_df)
        if sequences is None or len(sequences) < 10:
            return None

        model, baseline_stats = lstm_forecast.train(sequences)
        lstm_forecast.save_model(model, settings.model_store_path, meter_id)
        state["lstm_forecast_baseline"] = baseline_stats
        model_registry.mark_trained(state, "lstm_forecast")
        logger.info("LSTM Forecast trained and saved", extra={"meter_id": meter_id})
        return model
    except Exception as e:
        logger.error("LSTM Forecast training failed", extra={"meter_id": meter_id, "error": str(e)})
        return None


def _build_explanation(
    statistical_active: bool,
    is_if_anomaly: bool,
    mnf_flag: bool,
    burst_detected: bool,
    is_ae_active: bool,
    ae_score: float,
    is_fc_active: bool,
    fc_score: float,
    is_cnn_active: bool,
    cnn_score: float,
    pattern_type: str,
    leak_probability: float,
    leak_severity: str,
    days_of_data: float,
) -> str:
    """Build a concise human-readable explanation of the analysis result."""
    parts = []

    if not statistical_active:
        parts.append(f"Cold start — {days_of_data:.0f} days of data accumulated (need 30). Statistical layer not yet active.")
        return " ".join(parts)

    # Severity summary
    if leak_severity == "none":
        parts.append("No leak detected.")
    elif leak_severity == "low":
        parts.append("Low-level anomaly detected — monitor.")
    elif leak_severity == "medium":
        parts.append("Elevated leak probability — investigation recommended.")
    else:
        parts.append("HIGH leak probability — immediate investigation required.")

    # Specific signals
    if burst_detected:
        parts.append("Burst event detected (sudden flow spike).")
    if mnf_flag:
        parts.append("Elevated night flow detected (MNF above baseline).")
    if is_if_anomaly:
        parts.append("Isolation Forest flagged abnormal flow pattern.")

    # NN layers
    if is_ae_active and ae_score > 0.5:
        parts.append(f"LSTM Autoencoder: abnormal reconstruction (score={ae_score:.2f}).")
    if is_fc_active and fc_score > 0.5:
        parts.append(f"LSTM Forecast: actual flow exceeds prediction (score={fc_score:.2f}).")
    if is_cnn_active and pattern_type != "normal":
        parts.append(f"CNN pattern: {pattern_type} signature detected.")

    # Warm-up status for inactive layers
    warming = []
    if not is_ae_active:
        warming.append("LSTM-AE")
    if not is_fc_active:
        warming.append("Forecast")
    if not is_cnn_active:
        warming.append("CNN")
    if warming:
        parts.append(f"Warming up: {', '.join(warming)} (statistical layer only).")

    return " ".join(parts)
