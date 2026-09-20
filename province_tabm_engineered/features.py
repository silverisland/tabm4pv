from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import Config
from .data import DataInput, array_at, iter_data_frames


def feature_indices(rule: dict, horizon: int) -> list[int]:
    """Resolve explicit indices; horizon is one-based, array indices are zero-based."""
    if ("index" in rule) == ("indices" in rule):
        raise ValueError("每个字段必须且只能配置 index 或 indices")
    if "indices" in rule:
        indices = rule["indices"]
        if isinstance(indices, dict):
            start, stop = indices["start"], indices["stop"]
            to_end = stop is None and start < 0
            if to_end:
                stop = 0
            elif stop is None or (start < 0) != (stop < 0):
                raise ValueError("indices.start/stop 必须同为负数或同为非负数；跨边界请用索引列表")
            indices = list(range(start, stop, indices.get("step", 1)))
    else:
        index = rule["index"]
        if isinstance(index, dict):
            index = index["base"] + (horizon if index.get("horizon_offset", False) else 0)
        indices = [index]
    if not indices or any(type(index) is not int for index in indices):
        raise ValueError("索引必须是非空的整数列表")
    if len(set(indices)) != len(indices):
        raise ValueError("同一字段的索引不能重复")
    return list(indices)


def _sequence_features(
    province: pd.DataFrame,
    stations: pd.DataFrame,
    origins: pd.DatetimeIndex,
    field: str,
    rule: dict,
    horizon: int,
    config: Config,
) -> pd.DataFrame:
    indices = feature_indices(rule, horizon)
    weighted = rule.get("capacity_weighted", False)
    prefix = "weighted__" if weighted else ""
    columns = [f"{prefix}{field}__index_{index}" for index in indices]
    if weighted:
        values = [
            weighted_weather_features(stations, origins, [field], index, config)
            .iloc[:, 0].to_numpy()
            for index in indices
        ]
        matrix = np.column_stack(values)
    else:
        # Convert each sequence once, then select all configured points together.
        def select(value: object) -> np.ndarray:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
            positions = np.asarray(indices, dtype=int)
            valid = (positions >= -len(array)) & (positions < len(array))
            result = np.full(len(positions), np.nan, dtype=np.float32)
            result[valid] = array[positions[valid]]
            return result

        matrix = np.stack(province[field].map(select))
    return pd.DataFrame(matrix, index=origins, columns=columns, dtype=np.float32)


def weighted_weather_features(
    stations: pd.DataFrame,
    origins: pd.DatetimeIndex,
    columns: list[str],
    array_index: int,
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
            lambda value: array_at(value, array_index)
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
    feature_config = config["features"]
    common, varying = [], []
    for section in ("history", "future"):
        for field, rule in feature_config.get(section, {}).items():
            index = rule.get("index")
            if isinstance(index, dict) and index.get("horizon_offset", False):
                varying.append((section, field, rule))
            else:
                common.append(
                    _sequence_features(province, stations, origins, field, rule, 0, config)
                    .add_prefix(f"{section}__")
                )
    shared = pd.concat(common, axis=1) if common else pd.DataFrame(index=origins)
    parts = [shared]
    columns_by_horizon: dict[int, list[str]] = {}
    minutes = int(feature_config["minutes_per_point"])
    for horizon in horizons:
        suffix = f"__h{horizon:02d}"
        specific = [
            _sequence_features(province, stations, origins, field, rule, horizon, config)
            .add_prefix(f"{section}__").add_suffix(suffix)
            for section, field, rule in varying
        ]
        current = pd.concat(specific, axis=1) if specific else pd.DataFrame(index=origins)
        target_time = origins + pd.Timedelta(minutes=horizon * minutes)
        hour = target_time.hour.to_numpy() + target_time.minute.to_numpy() / 60.0
        time_values = {
            "hour": target_time.hour.to_numpy(),
            "hour_sin": np.sin(2 * np.pi * hour / 24).astype(np.float32),
            "hour_cos": np.cos(2 * np.pi * hour / 24).astype(np.float32),
        }
        for name in feature_config.get("time", []):
            current[f"time__{name}{suffix}"] = time_values[name]
        columns_by_horizon[horizon] = shared.columns.tolist() + current.columns.tolist()
        current[f"target_timestamp{suffix}"] = target_time.to_numpy()
        if names["power_future"] in province:
            current[f"target_power{suffix}"] = province[names["power_future"]].map(
                lambda value: array_at(value, horizon - 1)
            ).to_numpy(dtype=np.float32)
        # Consolidate each horizon before joining a wide table (16/20 horizons).
        parts.append(current.copy())

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
    features = config["features"]
    if not ("history" in features or "future" in features):
        raise ValueError("请将 features 改为 history/future 字段与 index/indices 配置")
    selected_weather = [
        field for section in ("history", "future")
        for field, rule in features.get(section, {}).items()
        if rule.get("capacity_weighted", False)
    ]
    print(f"特征规则：horizons={horizons}, features={features}")

    for frame in iter_data_frames(data, config, date_range=date_range):
        batch, columns_by_horizon = _file_features(
            frame, config, horizons
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
