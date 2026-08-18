from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import Config
from .data import DataInput, array_at, history_values, iter_data_frames


def weather_columns(data: pd.DataFrame, config: Config) -> list[str]:
    selected = config["features"].get("weather_columns")
    if selected:
        missing = sorted(set(selected).difference(data.columns))
        if missing:
            raise KeyError(f"输入数据缺少气象列: {missing}")
        return list(selected)
    suffix = config["features"].get("weather_suffix", "_predict")
    result = sorted(column for column in data.columns if column.endswith(suffix))
    if not result:
        raise ValueError(f"没有发现以 {suffix!r} 结尾的气象列")
    return result


def province_table(data: pd.DataFrame, config: Config) -> pd.DataFrame:
    cols, province = config["data"]["columns"], config["data"]["province_station"]
    timestamp, station = cols["timestamp"], cols["station"]
    rows = data[data[station].eq(province)].copy()
    rows = (
        rows.sort_values([timestamp, "source_file"])
        .drop_duplicates(timestamp, keep="last")
        .sort_values(timestamp)
        .reset_index(drop=True)
    )
    if rows.empty:
        raise ValueError(f"没有省级行: {province}")
    return rows


def weighted_weather_features(
    station_rows: pd.DataFrame,
    origins: pd.DatetimeIndex,
    columns: list[str],
    horizon_index: int,
    config: Config,
) -> pd.DataFrame:
    names = config["data"]["columns"]
    timestamp = names["timestamp"]
    station = names["station"]
    capacity_col = names["capacity"]
    rows = (
        station_rows.sort_values([timestamp, station, "source_file"])
        .drop_duplicates([timestamp, station], keep="last")
        .copy()
    )
    capacity = pd.to_numeric(rows[capacity_col], errors="coerce")
    rows["__capacity"] = capacity.where(capacity > 0)
    total_capacity = rows.groupby(timestamp)["__capacity"].sum(min_count=1)
    result = pd.DataFrame(index=origins)
    direction_keys = tuple(
        config["features"].get(
            "wind_direction_keywords",
            ["wind_direction", "winddirection", "wd_"],
        )
    )
    for column in columns:
        rows["__value"] = rows[column].map(lambda value: array_at(value, horizon_index))
        valid_rows = rows.loc[
            rows["__capacity"].notna() & rows["__value"].notna(),
            [timestamp, "__capacity", "__value"],
        ].copy()
        denominator = valid_rows.groupby(timestamp)["__capacity"].sum(min_count=1)
        prefix = f"weighted__{column}"
        result[f"{prefix}__capacity_coverage"] = (
            denominator / total_capacity
        ).reindex(origins)
        if any(key in column.lower() for key in direction_keys):
            radians = np.deg2rad(valid_rows["__value"])
            valid_rows["__sin"] = np.sin(radians) * valid_rows["__capacity"]
            valid_rows["__cos"] = np.cos(radians) * valid_rows["__capacity"]
            grouped = valid_rows.groupby(timestamp)
            result[f"{prefix}__sin"] = (
                grouped["__sin"].sum() / denominator
            ).reindex(origins)
            result[f"{prefix}__cos"] = (
                grouped["__cos"].sum() / denominator
            ).reindex(origins)
        else:
            valid_rows["__weighted"] = valid_rows["__value"] * valid_rows["__capacity"]
            result[f"{prefix}__mean"] = (
                valid_rows.groupby(timestamp)["__weighted"].sum() / denominator
            ).reindex(origins)
    return result.astype(np.float32)


def build_samples(
    data: pd.DataFrame,
    config: Config,
    horizon: int,
    columns: list[str],
    *,
    require_target: bool,
) -> tuple[pd.DataFrame, list[str]]:
    data_cfg, feature_cfg = config["data"], config["features"]
    names = data_cfg["columns"]
    timestamp, station = names["timestamp"], names["station"]
    province = province_table(data, config)
    origins = pd.DatetimeIndex(province[timestamp])
    history_length = int(feature_cfg["history_length"])
    histories = np.stack(
        province[names["power_history"]].map(
            lambda value: history_values(value, history_length)
        )
    )
    feature_names = [f"power_lag_{lag}" for lag in range(history_length, 0, -1)]
    features = pd.DataFrame(histories, index=origins, columns=feature_names)

    plant_pattern = re.compile(data_cfg["plant_station_pattern"])
    station_rows = data[
        data[station].map(lambda value: bool(plant_pattern.fullmatch(str(value))))
    ]
    weather = weighted_weather_features(
        station_rows, origins, columns, horizon - 1, config
    )
    features = features.join(weather)
    feature_names.extend(weather.columns.tolist())

    target_timestamp = origins + pd.Timedelta(
        minutes=horizon * int(feature_cfg["minutes_per_point"])
    )
    hour = (
        target_timestamp.hour.to_numpy()
        + target_timestamp.minute.to_numpy() / 60.0
    )
    features["time__hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    features["time__hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    feature_names.extend(["time__hour_sin", "time__hour_cos"])

    samples = features.reset_index(names="timestamp")
    samples["target_timestamp"] = target_timestamp.to_numpy()
    samples["source_file"] = province["source_file"].to_numpy()
    if names["power_future"] in province:
        samples["target_power"] = province[names["power_future"]].map(
            lambda value: array_at(value, horizon - 1)
        ).to_numpy(dtype=np.float32)
    elif require_target:
        raise KeyError(f"训练/测试数据缺少列: {names['power_future']}")
    samples = samples.replace([np.inf, -np.inf], np.nan)
    if require_target:
        samples = samples.dropna(subset=["target_power"])
    return samples.reset_index(drop=True), feature_names


def build_sample_sets(
    data: DataInput | None,
    config: Config,
    horizons: list[int],
    *,
    require_target: bool,
    date_range: str | None = None,
) -> tuple[dict[int, pd.DataFrame], dict[int, list[str]], list[str]]:
    """Build all requested horizons while each raw parquet file is in memory."""
    if not horizons:
        raise ValueError("horizons 不能为空")
    batches: dict[int, list[pd.DataFrame]] = {horizon: [] for horizon in horizons}
    feature_names: dict[int, list[str]] = {}
    selected_weather: list[str] | None = None
    seen_origins: dict[pd.Timestamp, str] = {}
    names = config["data"]["columns"]
    timestamp_col = names["timestamp"]

    for frame in iter_data_frames(
        data,
        config,
        require_target=require_target,
        date_range=date_range,
    ):
        current_weather = weather_columns(frame, config)
        if selected_weather is None:
            selected_weather = current_weather
        elif current_weather != selected_weather:
            raise ValueError(
                f"文件气象列 {current_weather} 与之前的 {selected_weather} 不一致"
            )

        province = province_table(frame, config)
        source_file = str(province["source_file"].iloc[0])
        origins = pd.DatetimeIndex(province[timestamp_col])
        overlap = sorted(set(origins).intersection(seen_origins))
        if overlap:
            examples = [
                {
                    "timestamp": str(origin),
                    "first_file": seen_origins[origin],
                    "current_file": source_file,
                }
                for origin in overlap[:10]
            ]
            raise ValueError(
                "同一起报时刻出现在多个文件中，无法保证逐文件气象加权等价："
                f"{examples}"
            )
        seen_origins.update({origin: source_file for origin in origins})

        for horizon in horizons:
            samples, current_names = build_samples(
                frame,
                config,
                horizon,
                selected_weather,
                require_target=require_target,
            )
            expected_names = feature_names.setdefault(horizon, current_names)
            if current_names != expected_names:
                raise RuntimeError(
                    f"horizon={horizon} 在不同文件中生成的特征列不一致"
                )
            batches[horizon].append(samples)
        print(
            f"单文件全部时效特征构造完成：file={source_file}, "
            f"origins={len(origins):,}, horizons={horizons}"
        )
        del frame, province

    if selected_weather is None:
        raise ValueError("输入数据为空，无法构造特征")
    result: dict[int, pd.DataFrame] = {}
    for horizon, frames in batches.items():
        result[horizon] = pd.concat(frames, ignore_index=True).sort_values(
            ["timestamp", "source_file"], ignore_index=True
        )
    print(
        f"逐文件样本构造完成：origins={len(seen_origins):,}, "
        f"horizons={horizons}；场站级原始数据已逐文件释放"
    )
    return result, feature_names, selected_weather
