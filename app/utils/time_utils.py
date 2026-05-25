"""
app/utils/time_utils.py
------------------------
Time-window helpers for PrimeModel AI Engine.

Key concepts:
- MNF window: Minimum Night Flow, typically 02:00–04:00 local time.
  The exact hours are configurable via MNF_WINDOW_START / MNF_WINDOW_END env vars.
- All timestamps entering the system are expected as UTC ISO-8601 strings.
  Time-of-day filtering is done in UTC. If UTP later requests local-time MNF
  windows, inject a tz_offset parameter here — noted as TD-007.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)


def parse_iso_timestamp(ts: str) -> datetime:
    """
    Parse an ISO-8601 timestamp string into a timezone-aware UTC datetime.
    Handles both 'Z' suffix and '+00:00' offset forms.

    Raises ValueError if the string cannot be parsed.
    """
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        # Assume UTC if no timezone info provided — log a warning
        logger.warning("Timestamp has no timezone info, assuming UTC", extra={"raw_ts": ts})
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_in_mnf_window(
    dt: datetime,
    window_start_hour: int = 2,
    window_end_hour: int = 4,
) -> bool:
    """
    Return True if `dt` falls within the MNF window [window_start_hour, window_end_hour).
    Both bounds are inclusive on the start, exclusive on the end.

    Args:
        dt: timezone-aware datetime (UTC).
        window_start_hour: Hour (24h) the window opens.  Default 2 = 02:00.
        window_end_hour: Hour (24h) the window closes. Default 4 = 04:00.
    """
    hour = dt.hour + dt.minute / 60.0
    return window_start_hour <= hour < window_end_hour


def filter_mnf_window(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    window_start_hour: int = 2,
    window_end_hour: int = 4,
) -> pd.DataFrame:
    """
    Filter a DataFrame to rows whose timestamp falls inside the MNF window.

    Args:
        df: DataFrame with a datetime column.
        ts_col: Name of the timestamp column.  Values must be datetime objects or
                parseable strings.  If strings, they are parsed via parse_iso_timestamp.
        window_start_hour: Start of MNF window (inclusive).
        window_end_hour: End of MNF window (exclusive).

    Returns:
        Filtered DataFrame (copy).  Empty DataFrame if no rows match.
    """
    if df.empty:
        return df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df = df.copy()
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

    hour_float = df[ts_col].dt.hour + df[ts_col].dt.minute / 60.0
    mask = (hour_float >= window_start_hour) & (hour_float < window_end_hour)
    return df[mask].copy()


def resample_to_interval(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    value_col: str = "flow_rate",
    interval_minutes: int = 15,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Resample a time-series DataFrame to a regular interval.

    Args:
        df: Input DataFrame with timestamp and value columns.
        ts_col: Timestamp column name.
        value_col: Column to aggregate.
        interval_minutes: Target interval in minutes.
        agg: Aggregation function name ('mean', 'sum', 'max', 'min').

    Returns:
        Resampled DataFrame with ts_col as index, reset to a column.
        Empty DataFrame if input is empty.
    """
    if df.empty:
        return df.copy()

    work = df[[ts_col, value_col]].copy()
    if not pd.api.types.is_datetime64_any_dtype(work[ts_col]):
        work[ts_col] = pd.to_datetime(work[ts_col], utc=True)

    work = work.set_index(ts_col).sort_index()
    rule = f"{interval_minutes}min"
    resampled = getattr(work[value_col].resample(rule), agg)()
    result = resampled.reset_index()
    result.columns = [ts_col, value_col]
    return result


def utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def days_between(dt_earlier: datetime, dt_later: Optional[datetime] = None) -> float:
    """
    Return the number of days between two UTC datetimes.
    If dt_later is None, uses the current UTC time.
    """
    if dt_later is None:
        dt_later = utcnow()
    delta: timedelta = dt_later - dt_earlier
    return delta.total_seconds() / 86400.0
