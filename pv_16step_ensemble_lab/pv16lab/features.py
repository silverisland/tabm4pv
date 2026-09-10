from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Dataset


@dataclass
class FeatureSet:
    by_horizon: dict[int, np.ndarray]
    names_by_horizon: dict[int, list[str]]
    persistence: np.ndarray


def _safe_stat(values: np.ndarray, function) -> np.ndarray:
    with np.errstate(all="ignore"):
        return function(values, axis=1)


def build_features(dataset: Dataset, config: dict) -> FeatureSet:
    data_cfg = config["data"]
    horizons = int(data_cfg["forecast_horizons"])
    minutes = int(data_cfg["interval_minutes"])
    points_per_day = 24 * 60 // minutes
    history = dataset.power_history / dataset.capacity[:, None]
    lags = sorted(
        {int(value) for value in config["features"]["power_lags"] if 0 < int(value) <= history.shape[1]}
    )
    windows = sorted(
        {int(value) for value in config["features"]["stat_windows"] if 0 < int(value) <= history.shape[1]}
    )
    persistence = np.full((len(history), horizons), np.nan, dtype=np.float32)
    by_horizon: dict[int, np.ndarray] = {}
    names_by_horizon: dict[int, list[str]] = {}

    common_values: list[np.ndarray] = []
    common_names: list[str] = []
    for lag in lags:
        common_values.append(history[:, -lag])
        common_names.append(f"power_ratio_lag_{lag}")
    for window in windows:
        block = history[:, -window:]
        for name, function in (
            ("mean", np.nanmean), ("std", np.nanstd),
            ("min", np.nanmin), ("max", np.nanmax),
        ):
            common_values.append(_safe_stat(block, function))
            common_names.append(f"power_ratio_{name}_{window}")
    current = history[:, -1]
    for lag in (1, 4, 16, 96):
        if lag <= history.shape[1]:
            common_values.append(current - history[:, -lag])
            common_names.append(f"power_ratio_delta_{lag}")
    common_values.append(dataset.capacity)
    common_names.append("capacity")

    for column, values in dataset.historical_weather.items():
        common_values.append(values[:, -1])
        common_names.append(f"{column}__current")
        for window in (4, 16, points_per_day):
            if window <= values.shape[1]:
                common_values.append(_safe_stat(values[:, -window:], np.nanmean))
                common_names.append(f"{column}__mean_{window}")

    base = np.column_stack(common_values).astype(np.float32)
    for horizon in range(1, horizons + 1):
        values = [base]
        names = list(common_names)
        same_time_lag = points_per_day - horizon + 1
        if 0 < same_time_lag <= history.shape[1]:
            previous_day = history[:, -same_time_lag]
            persistence[:, horizon - 1] = previous_day
            values.append(previous_day[:, None])
            names.append("power_ratio_previous_day_same_time")

        target_time = dataset.origins + pd.Timedelta(minutes=minutes * horizon)
        hour = target_time.hour.to_numpy() + target_time.minute.to_numpy() / 60.0
        day = target_time.dayofyear.to_numpy()
        time_features = np.column_stack([
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day / 365.25), np.cos(2 * np.pi * day / 365.25),
        ]).astype(np.float32)
        values.append(time_features)
        names.extend(["hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos"])

        for column, forecast in dataset.forecast_weather.items():
            point = forecast[:, horizon - 1]
            mean_to_point = _safe_stat(forecast[:, :horizon], np.nanmean)
            delta = point - forecast[:, 0]
            values.append(np.column_stack([point, mean_to_point, delta]))
            names.extend([
                f"{column}__horizon", f"{column}__mean_to_horizon",
                f"{column}__delta_from_p1",
            ])
            historical_name = column.removesuffix(config["weather"].get("forecast_suffix", "_predict"))
            if historical_name in dataset.historical_weather:
                historical = dataset.historical_weather[historical_name]
                if same_time_lag <= historical.shape[1]:
                    values.append(historical[:, -same_time_lag, None])
                    names.append(f"{historical_name}__previous_day_same_time")

        by_horizon[horizon] = np.column_stack(values).astype(np.float32)
        names_by_horizon[horizon] = names
    return FeatureSet(by_horizon, names_by_horizon, persistence)


def correction_features(
    dataset: Dataset,
    features: FeatureSet,
    base_prediction: np.ndarray,
    horizon: int,
) -> np.ndarray:
    index = horizon - 1
    history = dataset.power_history / dataset.capacity[:, None]
    compact = [
        base_prediction[:, index] / dataset.capacity,
        features.persistence[:, index],
        history[:, -1],
        np.nanmean(history[:, -4:], axis=1),
        np.nanmean(history[:, -16:], axis=1),
        history[:, -1] - history[:, -4],
        history[:, -1] - history[:, -16],
    ]
    for values in dataset.forecast_weather.values():
        compact.append(values[:, index])
    return np.column_stack(compact).astype(np.float32)

