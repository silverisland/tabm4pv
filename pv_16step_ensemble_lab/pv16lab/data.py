from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    frame: pd.DataFrame
    station: str
    origins: pd.DatetimeIndex
    capacity: np.ndarray
    power_history: np.ndarray
    raw_targets: np.ndarray
    targets: np.ndarray
    historical_weather: dict[str, np.ndarray]
    forecast_weather: dict[str, np.ndarray]
    weather_columns: list[str]
    quality: pd.DataFrame
    target_conflicts: pd.DataFrame
    missing_origins: pd.DatetimeIndex


def _paths(spec: str | list[str], fmt: str) -> list[Path]:
    specs = spec if isinstance(spec, list) else [spec]
    found: list[Path] = []
    suffix = {"parquet": "*.parquet", "csv": "*.csv", "pickle": "*.pkl"}[fmt]
    for item in specs:
        path = Path(item)
        if path.is_dir():
            found.extend(sorted(path.glob(suffix)))
        elif any(token in item for token in "*?["):
            found.extend(Path(match) for match in sorted(glob.glob(item)))
        elif path.is_file():
            found.append(path)
    unique = list(dict.fromkeys(path.resolve() for path in found))
    if not unique:
        raise FileNotFoundError(f"No input files found for: {spec}")
    return unique


def _read(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "pickle":
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported data format: {fmt}")


def _array(value: Any, length: int) -> np.ndarray:
    result = np.full(length, np.nan, dtype=np.float32)
    try:
        source = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return result
    width = min(length, len(source))
    result[:width] = source[:width]
    return result


def _array_matrix(series: pd.Series, length: int) -> np.ndarray:
    return np.vstack([_array(value, length) for value in series])


def _array_length(value: Any) -> int:
    if value is None or isinstance(value, (str, bytes)):
        return -1
    try:
        return int(np.asarray(value).size)
    except (TypeError, ValueError):
        return -1


def _timestamp(values: pd.Series, timezone: str) -> pd.Series:
    result = pd.to_datetime(values, errors="coerce")
    if isinstance(result.dtype, pd.DatetimeTZDtype):
        result = result.dt.tz_convert(timezone).dt.tz_localize(None)
    return result


def _select_weather(frame: pd.DataFrame, config: dict) -> tuple[list[str], list[str]]:
    weather = config["weather"]
    historical = list(weather.get("historical_columns") or [])
    forecast = list(weather.get("forecast_columns") or [])
    suffix = weather.get("forecast_suffix", "_predict")
    target_col = config["data"]["target_col"]
    if not forecast:
        forecast = sorted(
            column
            for column in frame.columns
            if column.endswith(suffix) and column != target_col
        )
    if not historical:
        historical = [
            column[: -len(suffix)]
            for column in forecast
            if column[: -len(suffix)] in frame.columns
        ]
    missing = sorted(set(historical + forecast).difference(frame.columns))
    if missing:
        raise KeyError(f"Configured weather columns are missing: {missing}")
    if not forecast:
        raise ValueError("No future weather columns found; configure weather.forecast_columns")
    return historical, forecast


def load_dataset(config: dict) -> Dataset:
    data_cfg = config["data"]
    fmt = str(data_cfg["format"]).lower()
    frames = []
    for path in _paths(data_cfg["path"], fmt):
        frame = _read(path, fmt)
        frame["__source_file"] = str(path)
        frames.append(frame)
    all_data = pd.concat(frames, ignore_index=True)

    required = [
        data_cfg["station_col"], data_cfg["time_col"], data_cfg["capacity_col"],
        data_cfg["history_power_col"], data_cfg["target_col"],
    ]
    missing = sorted(set(required).difference(all_data.columns))
    if missing:
        raise KeyError(f"Input data is missing required columns: {missing}")

    station_col = data_cfg["station_col"]
    requested = data_cfg.get("station")
    if requested is None:
        counts = all_data.groupby(station_col, dropna=True).size().sort_values(ascending=False)
        if counts.empty:
            raise ValueError("No valid station value found")
        requested = str(counts.index[0])
    frame = all_data.loc[all_data[station_col].astype(str).eq(str(requested))].copy()
    if frame.empty:
        available = all_data[station_col].dropna().astype(str).unique()[:20]
        raise ValueError(f"Station {requested!r} not found; examples={available.tolist()}")

    time_col = data_cfg["time_col"]
    frame[time_col] = _timestamp(frame[time_col], data_cfg["timezone"])
    if frame[time_col].isna().any():
        raise ValueError(f"{time_col} contains invalid timestamps")
    duplicate = frame.duplicated([station_col, time_col], keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, time_col].iloc[0]
        raise ValueError(f"Duplicate (station, timestamp_win), first example: {sample}")
    frame = frame.sort_values(time_col).reset_index(drop=True)
    origins = pd.DatetimeIndex(frame[time_col])

    history_length = int(data_cfg["history_length"])
    future_length = int(data_cfg["future_length"])
    horizons = int(data_cfg["forecast_horizons"])
    historical_cols, forecast_cols = _select_weather(frame, config)
    array_specs = {
        data_cfg["history_power_col"]: history_length,
        data_cfg["target_col"]: future_length,
        **{column: history_length for column in historical_cols},
        **{column: future_length for column in forecast_cols},
    }

    quality_rows: list[dict[str, Any]] = []
    for column, expected in array_specs.items():
        lengths = frame[column].map(_array_length)
        quality_rows.append({
            "check": "array_length",
            "column": column,
            "expected": expected,
            "actual_or_count": int((lengths != expected).sum()),
            "status": "PASS" if bool((lengths == expected).all()) else "WARN",
        })

    interval = pd.Timedelta(minutes=int(data_cfg["interval_minutes"]))
    full = pd.date_range(origins.min(), origins.max(), freq=interval)
    missing_times = full.difference(origins)
    quality_rows.extend([
        {"check": "rows", "column": "", "expected": len(full),
         "actual_or_count": len(frame), "status": "PASS"},
        {"check": "missing_origins", "column": time_col, "expected": 0,
         "actual_or_count": len(missing_times),
         "status": "PASS" if len(missing_times) == 0 else "WARN"},
        {"check": "time_range", "column": time_col,
         "expected": str(origins.min()), "actual_or_count": str(origins.max()),
         "status": "INFO"},
    ])

    capacity = pd.to_numeric(frame[data_cfg["capacity_col"]], errors="coerce").to_numpy(float)
    if not np.isfinite(capacity).all() or np.any(capacity <= 0):
        raise ValueError("cap_power_on must contain positive finite values")
    power_history = _array_matrix(frame[data_cfg["history_power_col"]], history_length)
    raw_targets = _array_matrix(frame[data_cfg["target_col"]], future_length)[:, :horizons]
    historical_weather = {
        column: _array_matrix(frame[column], history_length) for column in historical_cols
    }
    forecast_weather = {
        column: _array_matrix(frame[column], future_length)[:, :horizons]
        for column in forecast_cols
    }

    minutes = int(data_cfg["interval_minutes"])
    truth_index = pd.Series(
        raw_targets[:, 0], index=origins + pd.Timedelta(minutes=minutes)
    )
    targets = np.full((len(frame), horizons), np.nan, dtype=np.float32)
    conflicts = []
    for horizon in range(1, horizons + 1):
        target_time = origins + pd.Timedelta(minutes=minutes * horizon)
        central = truth_index.reindex(target_time).to_numpy(dtype=np.float32)
        targets[:, horizon - 1] = central
        raw = raw_targets[:, horizon - 1]
        comparable = np.isfinite(raw) & np.isfinite(central)
        conflicts.append({
            "horizon": horizon,
            "comparable": int(comparable.sum()),
            "conflicts": int((comparable & (np.abs(raw - central) > 1e-6)).sum()),
            "missing_index0_truth": int((~np.isfinite(central)).sum()),
        })

    return Dataset(
        frame=frame,
        station=str(requested),
        origins=origins,
        capacity=capacity,
        power_history=power_history,
        raw_targets=raw_targets,
        targets=targets,
        historical_weather=historical_weather,
        forecast_weather=forecast_weather,
        weather_columns=historical_cols + forecast_cols,
        quality=pd.DataFrame(quality_rows),
        target_conflicts=pd.DataFrame(conflicts),
        missing_origins=missing_times,
    )
