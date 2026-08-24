from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import Config
from .data import DataInput, array_at, history_values, iter_data_frames


def weather_columns(data: pd.DataFrame, config: Config) -> list[str]:
    selected = config["features"].get("weather_columns")
    if selected:
        return list(selected)
    suffix = config["features"].get("weather_suffix", "_predict")
    return sorted(column for column in data.columns if column.endswith(suffix))


def weighted_weather_features(
    stations: pd.DataFrame,
    origins: pd.DatetimeIndex,
    columns: list[str],
    horizon_index: int,
    config: Config,
) -> pd.DataFrame:
    names = config["data"]["columns"]
    timestamp, capacity_col = names["timestamp"], names["capacity"]
    rows = stations.copy()
    rows["__capacity"] = pd.to_numeric(rows[capacity_col], errors="coerce").where(
        lambda value: value > 0
    )
    result = pd.DataFrame(index=origins)

    for column in columns:
        rows["__value"] = rows[column].map(
            lambda value: array_at(value, horizon_index)
        )
        valid = rows.dropna(subset=["__capacity", "__value"]).copy()
        valid["__weighted"] = valid["__capacity"] * valid["__value"]
        capacity = valid.groupby(timestamp)["__capacity"].sum()
        weighted = valid.groupby(timestamp)["__weighted"].sum() / capacity
        result[f"weighted__{column}__mean"] = weighted.reindex(origins)
    return result.astype(np.float32)


def _file_features(
    data: pd.DataFrame,
    config: Config,
    horizons: list[int],
    weather: list[str],
) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    names = config["data"]["columns"]
    timestamp, station = names["timestamp"], names["station"]
    province = (
        data[data[station].eq(config["data"]["province_station"])]
        .drop_duplicates(timestamp, keep="last")
        .sort_values(timestamp)
        .reset_index(drop=True)
    )
    stations = (
        data[
            data[station].str.fullmatch(
                re.compile(config["data"]["plant_station_pattern"]), na=False
            )
        ]
        .drop_duplicates([timestamp, station], keep="last")
        .copy()
    )
    origins = pd.DatetimeIndex(province[timestamp])
    history_length = int(config["features"]["history_length"])
    history_columns = [
        f"power_lag_{lag}" for lag in range(history_length, 0, -1)
    ]
    history = pd.DataFrame(
        np.stack(
            province[names["power_history"]].map(
                lambda value: history_values(value, history_length)
            )
        ),
        index=origins,
        columns=history_columns,
    )

    parts = [history]
    columns_by_horizon: dict[int, list[str]] = {}
    minutes = int(config["features"]["minutes_per_point"])
    for horizon in horizons:
        suffix = f"__h{horizon:02d}"
        current = weighted_weather_features(
            stations, origins, weather, horizon - 1, config
        ).add_suffix(suffix)
        target_time = origins + pd.Timedelta(minutes=horizon * minutes)
        hour = target_time.hour.to_numpy() + target_time.minute.to_numpy() / 60.0
        current[f"time__hour{suffix}"] = target_time.hour.to_numpy()
        current[f"time__hour_sin{suffix}"] = np.sin(2 * np.pi * hour / 24).astype(
            np.float32
        )
        current[f"time__hour_cos{suffix}"] = np.cos(2 * np.pi * hour / 24).astype(
            np.float32
        )
        columns_by_horizon[horizon] = history_columns + current.columns.tolist()
        current[f"target_timestamp{suffix}"] = target_time.to_numpy()
        if names["power_future"] in province:
            current[f"target_power{suffix}"] = province[names["power_future"]].map(
                lambda value: array_at(value, horizon - 1)
            ).to_numpy(dtype=np.float32)
        parts.append(current)

    return pd.concat(parts, axis=1).reset_index(names="timestamp"), columns_by_horizon


def build_feature_data(
    data: DataInput | None,
    config: Config,
    horizons: list[int],
    *,
    date_range: str | None = None,
) -> tuple[pd.DataFrame, dict[int, list[str]], list[str]]:
    """Build one wide feature table and one column list per horizon."""
    batches: list[pd.DataFrame] = []
    columns_by_horizon: dict[int, list[str]] = {}
    selected_weather: list[str] = []

    for frame in iter_data_frames(data, config, date_range=date_range):
        if not selected_weather:
            selected_weather = weather_columns(frame, config)
        batch, columns_by_horizon = _file_features(
            frame, config, horizons, selected_weather
        )
        batches.append(batch)
        print(f"当前文件特征完成：rows={len(batch):,}")

    result = (
        pd.concat(batches, ignore_index=True)
        .drop_duplicates("timestamp", keep="last")
        .replace([np.inf, -np.inf], np.nan)
        .sort_values("timestamp", ignore_index=True)
    )
    print(f"特征完成：rows={len(result):,}, columns={len(result.columns):,}")
    return result, columns_by_horizon, selected_weather
