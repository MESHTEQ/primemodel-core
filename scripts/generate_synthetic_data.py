"""
scripts/generate_synthetic_data.py
------------------------------------
Generate a realistic 30-day synthetic water meter dataset for PrimeModel cold-start training.

Produces: data/synthetic/synthetic_uplinks.csv

The dataset simulates a single meter (meter_id="SYNTH_001", client_id="synthetic")
transmitting uplinks at 15-minute intervals for 30 days.

Realistic patterns embedded:
    1. Diurnal demand profile — peak morning (07:00–09:00) and evening (17:00–19:00)
    2. Weekly pattern — lower demand on weekends
    3. Three injected anomaly scenarios:
       - Scenario A (Day 8):   Single burst event — sudden spike lasting 1–2 hours
       - Scenario B (Day 18):  Background leak — sustained +15% MNF elevation from Day 18 onward
       - Scenario C (Day 25):  Intermittent leak — elevated flow every 4–6 hours, variable

Output columns:
    meter_id, client_id, timestamp, flow_rate, cumulative_volume,
    battery_level, alarm_flags, label_type

label_type values: "normal" | "burst" | "background_leak" | "intermittent_leak"

Usage:
    python scripts/generate_synthetic_data.py

The output file is used by:
    - scripts/train_cnn_pattern.py  (CNN cold-start training)
    - scripts/train_isolation_forest.py
    - scripts/train_lstm_autoencoder.py
    - scripts/train_lstm_forecast.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# Add project root to path so local imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- Configuration ---
METER_ID = "SYNTH_001"
CLIENT_ID = "synthetic"
START_DATE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
DAYS = 30
INTERVAL_MINUTES = 15
READINGS_PER_DAY = 96  # 24h / 15min
TOTAL_READINGS = DAYS * READINGS_PER_DAY

# Seed for reproducibility
RNG = np.random.default_rng(seed=42)

# --- Baseline flow parameters ---
BASE_FLOW_MEAN = 0.45        # m³/h — typical apartment block DMA
BASE_FLOW_STD = 0.08         # natural variation
MNF_FLOOR = 0.12             # minimum night flow (leakage floor, m³/h)
BATTERY_START = 95.0
BATTERY_DRAIN_PER_DAY = 0.07  # ~% per day (simulates slow depletion)


def diurnal_profile(hour: float) -> float:
    """
    Return a multiplier (0.4–1.8) based on hour of day.
    Peaks at morning (08:00) and evening (18:00).
    """
    # Gaussian peaks
    morning_peak = 1.4 * np.exp(-0.5 * ((hour - 8.0) / 1.5) ** 2)
    evening_peak = 1.3 * np.exp(-0.5 * ((hour - 18.0) / 1.5) ** 2)
    night_low = 0.35  # base level from MNF_FLOOR contribution
    return night_low + morning_peak + evening_peak


def weekly_factor(dow: int) -> float:
    """
    Return a demand multiplier for day of week.
    Monday=0, Sunday=6. Weekend demand is ~15% lower.
    """
    if dow >= 5:  # Saturday, Sunday
        return 0.87
    return 1.0


def generate_base_flow(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Generate the normal (no-leak) flow signal."""
    flows = np.zeros(len(timestamps))
    for i, ts in enumerate(timestamps):
        hour = ts.hour + ts.minute / 60.0
        dow = ts.weekday()
        profile = diurnal_profile(hour)
        weekly = weekly_factor(dow)
        noise = RNG.normal(0, BASE_FLOW_STD)
        flow = max(0.0, BASE_FLOW_MEAN * profile * weekly + noise)
        # MNF floor — always some base leakage
        flow = max(flow, MNF_FLOOR + RNG.uniform(0, 0.02))
        flows[i] = flow
    return flows.astype(np.float32)


def inject_burst(flows: np.ndarray, timestamps: pd.DatetimeIndex, day: int) -> tuple:
    """
    Inject a burst event on `day`.
    A burst is a sudden flow rate spike of 3–8x normal for 1–2 hours (4–8 readings).

    Returns:
        Modified flows array and label mask.
    """
    labels = np.full(len(flows), "normal", dtype=object)
    # Pick a start index within the day, between 06:00 and 22:00
    day_start_idx = day * READINGS_PER_DAY
    # 06:00 = reading 24, 22:00 = reading 88
    burst_start = day_start_idx + RNG.integers(24, 72)
    burst_duration = RNG.integers(4, 9)  # 4–8 readings = 1–2 hours

    burst_multiplier = RNG.uniform(3.0, 8.0)
    for j in range(burst_duration):
        idx = burst_start + j
        if idx < len(flows):
            flows[idx] = flows[idx] * burst_multiplier
            labels[idx] = "burst"

    print(f"  Burst injected at reading {burst_start} (day {day}, approx {burst_start % 96 * 15 // 60:02d}:00)")
    return flows, labels


def inject_background_leak(flows: np.ndarray, labels: np.ndarray, start_day: int) -> tuple:
    """
    Inject a sustained background leak from start_day onward.
    MNF increases by 15–25%. Leak is present at all times but most visible in MNF window.
    """
    leak_offset = RNG.uniform(0.05, 0.10)  # m³/h constant addition
    start_idx = start_day * READINGS_PER_DAY

    for j in range(start_idx, len(flows)):
        flows[j] += leak_offset
        if labels[j] == "normal":
            labels[j] = "background_leak"

    print(f"  Background leak injected from day {start_day} onwards (offset={leak_offset:.3f} m³/h)")
    return flows, labels


def inject_intermittent_leak(flows: np.ndarray, labels: np.ndarray, start_day: int) -> tuple:
    """
    Inject intermittent leak events — elevated flow every 4–6 hours, variable duration.
    Simulates a valve that opens and closes erratically.
    """
    start_idx = start_day * READINGS_PER_DAY
    idx = start_idx
    while idx < len(flows):
        # Period of elevated flow: 2–6 readings
        leak_duration = RNG.integers(2, 7)
        leak_magnitude = RNG.uniform(0.08, 0.25)
        for j in range(leak_duration):
            if idx + j < len(flows):
                flows[idx + j] += leak_magnitude
                if labels[idx + j] == "normal":
                    labels[idx + j] = "intermittent_leak"
        # Gap between events: 16–24 readings (4–6 hours)
        idx += leak_duration + RNG.integers(16, 25)

    print(f"  Intermittent leak injected from day {start_day} onwards")
    return flows, labels


def generate_cumulative_volume(flows: np.ndarray, interval_hours: float = 0.25) -> np.ndarray:
    """Convert flow rate series (m³/h) to cumulative volume (m³)."""
    increments = flows * interval_hours
    cumulative = np.cumsum(increments)
    return cumulative.astype(np.float32)


def generate_battery_levels(n: int) -> np.ndarray:
    """Generate realistic battery level readings with slow drain."""
    levels = np.zeros(n)
    for i in range(n):
        day = i / READINGS_PER_DAY
        drain = day * BATTERY_DRAIN_PER_DAY
        noise = RNG.normal(0, 0.1)
        levels[i] = max(0.0, min(100.0, BATTERY_START - drain + noise))
    return levels.astype(np.float32)


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "synthetic_uplinks.csv")

    print(f"Generating {TOTAL_READINGS} synthetic readings ({DAYS} days at {INTERVAL_MINUTES}-min intervals)...")

    # Generate timestamps
    timestamps = pd.date_range(
        start=START_DATE,
        periods=TOTAL_READINGS,
        freq=f"{INTERVAL_MINUTES}min",
        tz="UTC",
    )

    # Generate base flow
    flows = generate_base_flow(timestamps)
    labels = np.full(TOTAL_READINGS, "normal", dtype=object)

    # Inject anomaly scenarios
    print("\nInjecting anomaly scenarios:")

    # Scenario A: Burst on Day 8
    flows, labels = inject_burst(flows, timestamps, day=8)

    # Scenario B: Background leak from Day 18
    flows, labels = inject_background_leak(flows, labels, start_day=18)

    # Scenario C: Intermittent leak from Day 25
    flows, labels = inject_intermittent_leak(flows, labels, start_day=25)

    # Generate cumulative volume and battery
    cumulative = generate_cumulative_volume(flows)
    battery_levels = generate_battery_levels(TOTAL_READINGS)

    # Alarm flags — set to 1 during burst events
    alarm_flags = np.where(labels == "burst", 1, 0).astype(int)

    # Build DataFrame
    df = pd.DataFrame({
        "meter_id": METER_ID,
        "client_id": CLIENT_ID,
        "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "flow_rate": np.round(flows, 4),
        "cumulative_volume": np.round(cumulative, 4),
        "battery_level": np.round(battery_levels, 2),
        "alarm_flags": alarm_flags,
        "label_type": labels,
    })

    df.to_csv(output_path, index=False)

    # Summary
    print(f"\nDataset written to: {output_path}")
    print(f"Total readings:     {len(df)}")
    print(f"\nLabel distribution:")
    print(df["label_type"].value_counts().to_string())
    print(f"\nFlow rate summary:")
    print(df["flow_rate"].describe().round(4).to_string())
    print(f"\nBattery summary:")
    print(df["battery_level"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
