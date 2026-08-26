"""Time-consistent features for physical targets and canonical coordinates."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Callable

import numpy as np

from editable.registration import (
    MINUTES_PER_DAY,
    MonotoneWarp,
    SeasonalWarps,
)


WarpProvider = (
    MonotoneWarp
    | SeasonalWarps
    | Callable[[date], MonotoneWarp]
)


def minute_of_day(timestamp: datetime) -> float:
    return (
        timestamp.hour * 60.0
        + timestamp.minute
        + timestamp.second / 60.0
    )


def absolute_physical_minutes(timestamp: datetime) -> float:
    """Timezone-independent absolute minute based on the calendar ordinal."""
    return timestamp.date().toordinal() * MINUTES_PER_DAY + minute_of_day(
        timestamp
    )


def absolute_canonical_minutes(
    timestamp: datetime, warp: WarpProvider
) -> float:
    return float(
        _physical_to_canonical_absolute(
            absolute_physical_minutes(timestamp), warp
        )
    )


def build_time_features(
    issue_timestamp: datetime,
    target_future_hours: float,
    warp: WarpProvider,
    *,
    history_last_offset_minutes: int = 0,
) -> dict[str, float]:
    history_end = issue_timestamp + timedelta(
        minutes=history_last_offset_minutes
    )
    target = issue_timestamp + timedelta(hours=target_future_hours)
    current_canonical = absolute_canonical_minutes(history_end, warp)
    target_canonical = absolute_canonical_minutes(target, warp)
    return {
        "predict_hour": minute_of_day(target) / 60.0,
        "canonical_current_hour": (
            current_canonical % MINUTES_PER_DAY
        ) / 60.0,
        "canonical_target_hour": (
            target_canonical % MINUTES_PER_DAY
        ) / 60.0,
        "canonical_horizon_hours": (
            target_canonical - current_canonical
        ) / 60.0,
        "predict_month": float(target.month),
    }


def prepare_power_history(
    history: np.ndarray,
    capacity: float,
    input_len: int,
    warp: WarpProvider,
    *,
    register_history: bool = False,
    history_end_timestamp: datetime | None = None,
    point_minutes: int = 15,
) -> np.ndarray:
    values = np.asarray(history, dtype=np.float64).reshape(-1)
    if len(values) < input_len:
        raise ValueError("History is shorter than input_len")
    physical = values[-input_len:] / float(capacity)
    if not register_history:
        return physical.astype(np.float32)
    if history_end_timestamp is None:
        raise ValueError(
            "history_end_timestamp is required when register_history=True"
        )
    if len(values) <= input_len:
        raise ValueError(
            "Registered history needs interpolation margin; retain more than "
            "input_len physical points"
        )

    normalized = values / float(capacity)
    physical_end = absolute_physical_minutes(history_end_timestamp)
    physical_times = physical_end - np.arange(
        len(values) - 1, -1, -1, dtype=np.float64
    ) * float(point_minutes)
    canonical_end = float(
        _physical_to_canonical_absolute(physical_end, warp)
    )
    canonical_query = canonical_end - np.arange(
        input_len - 1, -1, -1, dtype=np.float64
    ) * float(point_minutes)
    required_physical = _canonical_to_physical_absolute(
        canonical_query, warp
    )
    tolerance = 1e-7
    if (
        required_physical.min() < physical_times.min() - tolerance
        or required_physical.max() > physical_times.max() + tolerance
    ):
        raise ValueError(
            "Retained history does not cover the registered interpolation range"
        )
    return np.interp(
        required_physical,
        physical_times,
        normalized,
    ).astype(np.float32)


def _warp_for_day(warp: WarpProvider, ordinal_day: int) -> MonotoneWarp:
    calendar_date = date.fromordinal(int(ordinal_day))
    if isinstance(warp, MonotoneWarp):
        return warp
    if isinstance(warp, SeasonalWarps):
        return warp.for_month(calendar_date.month)
    selected = warp(calendar_date)
    if not isinstance(selected, MonotoneWarp):
        raise TypeError("Warp provider must return MonotoneWarp")
    return selected


def _physical_to_canonical_absolute(values, warp: WarpProvider):
    values = np.asarray(values, dtype=np.float64)
    days = np.floor(values / MINUTES_PER_DAY).astype(np.int64)
    physical_fraction = (values - days * MINUTES_PER_DAY) / MINUTES_PER_DAY
    canonical_fraction = np.empty_like(physical_fraction)
    for day in np.unique(days):
        mask = days == day
        canonical_fraction[mask] = _warp_for_day(
            warp, int(day)
        ).physical_to_canonical(physical_fraction[mask])
    result = days * MINUTES_PER_DAY + canonical_fraction * MINUTES_PER_DAY
    return float(result) if result.ndim == 0 else result


def _canonical_to_physical_absolute(values, warp: WarpProvider):
    values = np.asarray(values, dtype=np.float64)
    days = np.floor(values / MINUTES_PER_DAY).astype(np.int64)
    canonical_fraction = (values - days * MINUTES_PER_DAY) / MINUTES_PER_DAY
    physical_fraction = np.empty_like(canonical_fraction)
    for day in np.unique(days):
        mask = days == day
        physical_fraction[mask] = _warp_for_day(
            warp, int(day)
        ).canonical_to_physical(canonical_fraction[mask])
    result = days * MINUTES_PER_DAY + physical_fraction * MINUTES_PER_DAY
    return float(result) if result.ndim == 0 else result
