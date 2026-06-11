"""
app/services/training.py
--------------------------
Reusable device-grain LSTM Autoencoder training service.

P3 wires the admin endpoint that exposes this over HTTP.
Future automatic triggers (BackgroundTasks, scheduled jobs) can import
and call ``train_lstm_ae_for_device`` directly without any modification here.

Design rules:
- TF stays lazy — this module must import cleanly on a TF-less machine.
- Never raises — every outcome is captured in the return dict.
- Training history is appended to a JSONL file for auditability.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from app.config import get_settings
from app.services import decoder_registry, device_registry, supabase_client
from app.services import feature_engineering
from app.services.neural import lstm_autoencoder
from app.services.neural.lstm_autoencoder import save_threshold_stats
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Training history helpers
# ---------------------------------------------------------------------------

def _append_history(entry: dict) -> None:
    """
    Append a single training history entry (one JSON line) to the JSONL log.

    Path: {model_store_path}/training_history.jsonl

    Each line contains: timestamp (UTC ISO), deveui, param, layer, status,
    n_sequences, duration_seconds, and optionally reason / error when failed.

    Append errors are caught and logged — they must never break the training
    caller.
    """
    try:
        settings = get_settings()
        store = settings.model_store_path
        os.makedirs(store, exist_ok=True)
        log_path = os.path.join(store, "training_history.jsonl")
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.error(
            "Failed to append training history",
            extra={"error": str(exc)},
        )


def read_training_history(limit: int = 50) -> List[dict]:
    """
    Return training history entries, newest first.

    Args:
        limit: Maximum number of entries to return.

    Returns:
        List of dicts (most recent first).  Returns empty list if the log
        file does not exist or every line is unparseable.
    """
    try:
        settings = get_settings()
        log_path = os.path.join(settings.model_store_path, "training_history.jsonl")
        if not os.path.exists(log_path):
            return []
        lines: List[dict] = []
        with open(log_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    lines.append(json.loads(raw))
                except Exception:
                    pass  # skip unparseable lines
        # Newest-first: reverse the chronological order, then apply limit
        return list(reversed(lines))[:limit]
    except Exception as exc:
        logger.error(
            "Failed to read training history",
            extra={"error": str(exc)},
        )
        return []


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_lstm_ae_for_device(
    deveui: str,
    param: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train an LSTM Autoencoder for a single device and persist the result.

    This function NEVER raises.  Every outcome (success or failure) is
    returned as a structured dict and appended to the training history log.

    Args:
        deveui: Device EUI string (e.g. "24E124136D355878").
        param:  Optional parameter name to train on.  When omitted, the first
                decoded numeric parameter from insertion order is used.

    Returns:
        Dict with at least "status" ("completed" | "failed") and "deveui".

        On success:
            {
                "status": "completed",
                "deveui": str,
                "param": str,
                "readings_used": int,
                "n_sequences": int,
                "threshold_stats": {"mae_mean": float, "mae_std": float, "threshold": float},
                "duration_seconds": float,
            }

        On failure:
            {
                "status": "failed",
                "reason": str,   # no_data | unknown_param | insufficient_sequences | error
                "deveui": str,
                ... (additional context fields depending on reason)
            }
    """
    start_time = time.monotonic()

    # Shared history fields populated progressively
    history_entry: Dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "deveui": deveui,
        "param": param,
        "layer": "lstm_ae",
        "status": "failed",
        "n_sequences": 0,
        "duration_seconds": 0.0,
    }

    try:
        # ------------------------------------------------------------------
        # (a) Resolve device type
        # ------------------------------------------------------------------
        info = device_registry.get_device_info(deveui)
        device_type = info["device_type"]

        # ------------------------------------------------------------------
        # (b) Fetch sensor history
        # ------------------------------------------------------------------
        rows = supabase_client.fetch_sensor_history(deveui)
        logger.info(
            "Training: fetched sensor history",
            extra={"deveui": deveui, "rows": len(rows)},
        )
        if not rows:
            result: Dict[str, Any] = {
                "status": "failed",
                "reason": "no_data",
                "deveui": deveui,
            }
            history_entry.update({
                "status": "failed",
                "reason": "no_data",
                "duration_seconds": round(time.monotonic() - start_time, 3),
            })
            _append_history(history_entry)
            return result

        # ------------------------------------------------------------------
        # (c) Decode payloads — mirror agnostic endpoint Step 3
        # ------------------------------------------------------------------
        decoder = decoder_registry.get_decoder(device_type)
        param_series: Dict[str, List] = {}

        for row in rows:
            payload = row.get("decoded_payload") or {}
            if not payload:
                continue
            created_at = row.get("created_at", "")
            numeric = decoder(payload)
            for p_name, value in numeric.items():
                if p_name not in param_series:
                    param_series[p_name] = []
                param_series[p_name].append((created_at, value))

        # ------------------------------------------------------------------
        # (d) Determine target parameter
        # ------------------------------------------------------------------
        if param is not None:
            if param not in param_series:
                available = list(param_series.keys())
                result = {
                    "status": "failed",
                    "reason": "unknown_param",
                    "deveui": deveui,
                    "requested_param": param,
                    "available_params": available,
                }
                history_entry.update({
                    "status": "failed",
                    "reason": "unknown_param",
                    "param": param,
                    "duration_seconds": round(time.monotonic() - start_time, 3),
                })
                _append_history(history_entry)
                return result
            target_param = param
        else:
            if not param_series:
                # All rows decoded to empty dicts — treat as no usable data
                result = {
                    "status": "failed",
                    "reason": "no_data",
                    "deveui": deveui,
                }
                history_entry.update({
                    "status": "failed",
                    "reason": "no_data",
                    "duration_seconds": round(time.monotonic() - start_time, 3),
                })
                _append_history(history_entry)
                return result
            target_param = next(iter(param_series))  # insertion order (Python 3.7+)

        history_entry["param"] = target_param

        # ------------------------------------------------------------------
        # (e) Build DataFrame and training sequences
        #     DataFrame uses column names "timestamp" / "flow_rate" to match
        #     the convention expected by build_lstm_sequence inside
        #     build_training_sequences.
        # ------------------------------------------------------------------
        series = param_series[target_param]
        readings_used = len(rows)

        param_df = pd.DataFrame({
            "timestamp": [ts for (ts, _) in series],
            "flow_rate": [v for (_, v) in series],
        })

        sequences = feature_engineering.build_training_sequences(param_df)

        # ------------------------------------------------------------------
        # (f) Guard: must have >= 10 sequences BEFORE any TF import
        # ------------------------------------------------------------------
        n_sequences = 0 if sequences is None else len(sequences)
        if sequences is None or n_sequences < 10:
            result = {
                "status": "failed",
                "reason": "insufficient_sequences",
                "deveui": deveui,
                "param": target_param,
                "readings_used": readings_used,
                "n_sequences": n_sequences,
            }
            history_entry.update({
                "status": "failed",
                "reason": "insufficient_sequences",
                "n_sequences": n_sequences,
                "duration_seconds": round(time.monotonic() - start_time, 3),
            })
            _append_history(history_entry)
            return result

        # ------------------------------------------------------------------
        # (g) Train — TF only touched after the guard passes
        # ------------------------------------------------------------------
        settings = get_settings()
        model_key = f"{deveui}_{target_param}"

        model, stats = lstm_autoencoder.train(sequences)
        lstm_autoencoder.save_model(model, settings.model_store_path, model_key)
        save_threshold_stats(stats, settings.model_store_path, model_key)

        # ------------------------------------------------------------------
        # (h) Build success result
        # ------------------------------------------------------------------
        duration = round(time.monotonic() - start_time, 3)
        result = {
            "status": "completed",
            "deveui": deveui,
            "param": target_param,
            "readings_used": readings_used,
            "n_sequences": n_sequences,
            "threshold_stats": stats,
            "duration_seconds": duration,
        }
        history_entry.update({
            "status": "completed",
            "n_sequences": n_sequences,
            "duration_seconds": duration,
        })
        _append_history(history_entry)

        logger.info(
            "LSTM AE training completed",
            extra={
                "deveui": deveui,
                "param": target_param,
                "n_sequences": n_sequences,
                "duration_seconds": duration,
            },
        )
        return result

    except Exception as exc:
        # ------------------------------------------------------------------
        # (i) Catch-all — unexpected exceptions must never surface to callers
        # ------------------------------------------------------------------
        logger.error(
            "Unexpected error during LSTM AE training",
            extra={"deveui": deveui, "error": str(exc)},
        )
        duration = round(time.monotonic() - start_time, 3)
        result = {
            "status": "failed",
            "reason": "error",
            "deveui": deveui,
            "error": str(exc),
        }
        history_entry.update({
            "status": "failed",
            "reason": "error",
            "error": str(exc),
            "duration_seconds": duration,
        })
        _append_history(history_entry)
        return result
