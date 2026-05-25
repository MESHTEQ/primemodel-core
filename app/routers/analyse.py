"""
app/routers/analyse.py
-----------------------
POST /analyse — core endpoint.

Called by the Supabase Edge Function on every ThingPark uplink.
Orchestrates the full analysis pipeline:

1. Decode raw payload (if present)
2. Persist uplink to Supabase
3. Load meter state from model registry
4. Run statistical layer (always)
5. Run LSTM Autoencoder (if active)
6. Run LSTM Forecast (if active)
7. Run 1D CNN (if active)
8. Compute ensemble score
9. Compute RUL (battery + drift)
10. Build and return AnalyseResponse

Design principles:
    - Any individual layer failure is caught and logged — it never crashes the endpoint
    - Warming-up layers return score=0.0 and are excluded from the ensemble
    - State is persisted to disk after every uplink
    - Supabase writes are fire-and-forget with error logging, never blocking the response
"""

import numpy as np
import pandas as pd
from typing import Optional
from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone

from app.schemas.uplink import AnalyseRequest, AnalyseResponse, ModelLayerStatuses
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
from app.services.statistical import isolation_forest, mnf_cusum, mnf_ewma, burst_detector
from app.services.neural import lstm_autoencoder, lstm_forecast, cnn_pattern

# Decoders
from app.services.decoder_panda import decode as decode_panda
from app.services.decoder_bove import decode as decode_bove

logger = get_logger(__name__)
router = APIRouter()


@router.post("/", response_model=AnalyseResponse, tags=["Analysis"])
def analyse(request: AnalyseRequest) -> AnalyseResponse:
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
    history_rows = supabase_client.fetch_uplink_history(meter_id, client_id, limit=500)
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
                if_score, is_if_anomaly = isolation_forest.score(if_model, feat_vec)
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
        model = isolation_forest.train(X, contamination=settings.isolation_forest_contamination)
        isolation_forest.save_model(model, settings.model_store_path, meter_id)
        model_registry.mark_trained(state, "isolation_forest")
        logger.info("Isolation Forest trained and saved", extra={"meter_id": meter_id})
        return model
    except Exception as e:
        logger.error("IF training failed", extra={"meter_id": meter_id, "error": str(e)})
        return None


def _train_lstm_autoencoder(history_df, settings, state, meter_id):
    """Train LSTM Autoencoder from uplink history and persist."""
    try:
        sequences = _build_sequences_from_df(history_df)
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
        sequences = _build_sequences_from_df(history_df)
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


def _build_sequences_from_df(history_df, stride: int = 24) -> np.ndarray:
    """
    Slide a window of SEQUENCE_LENGTH over history_df to produce training sequences.
    Stride controls overlap (smaller stride = more sequences but slower training).

    Returns:
        numpy array of shape (n_seqs, SEQUENCE_LENGTH, 4), or None if insufficient data.
    """
    seq_len = feature_engineering.SEQUENCE_LENGTH
    if len(history_df) < seq_len + stride:
        return None

    all_seqs = []
    for start in range(0, len(history_df) - seq_len, stride):
        window = history_df.iloc[start: start + seq_len]
        seq = feature_engineering.build_lstm_sequence(window, seq_len=seq_len)
        if seq is not None:
            all_seqs.append(seq)

    if not all_seqs:
        return None

    return np.array(all_seqs)  # (n_seqs, seq_len, 4)


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
