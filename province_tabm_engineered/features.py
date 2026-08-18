from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .data import DataInput, array_at, history_values, iter_data_frames


@dataclass
class FeatureData:
    common: pd.DataFrame
    horizons: dict[int, pd.DataFrame]
    feature_names: list[str]
    weather_columns: list[str]
    require_target: bool

    @property
    def origins(self) -> pd.Index:
        return self.common.index

    def samples(self, horizon: int) -> pd.DataFrame:
        samples = self.common.join(self.horizons[horizon]).reset_index(
            names="timestamp"
        )
        samples = samples.replace([np.inf, -np.inf], np.nan)
        if self.require_target:
            samples = samples.dropna(subset=["target_power"])
        return samples.reset_index(drop=True)


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


def _model_rows(
    data: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols, province = config["data"]["columns"], config["data"]["province_station"]
    timestamp, station = cols["timestamp"], cols["station"]
    province_rows = (
        data[data[station].eq(province)]
        .drop_duplicates(timestamp, keep="last")
        .sort_values(timestamp)
        .reset_index(drop=True)
    )
    if province_rows.empty:
        raise ValueError(f"没有省级行: {province}")
    plant_pattern = re.compile(config["data"]["plant_station_pattern"])
    plant_rows = (
        data[data[station].str.fullmatch(plant_pattern, na=False)]
        .drop_duplicates([timestamp, station], keep="last")
        .sort_values([timestamp, station])
        .copy()
    )
    return province_rows, plant_rows


def weighted_weather_features(
    station_rows: pd.DataFrame,
    origins: pd.DatetimeIndex,
    columns: list[str],
    horizon_index: int,
    config: Config,
) -> pd.DataFrame:
    names = config["data"]["columns"]
    timestamp = names["timestamp"]
    capacity_col = names["capacity"]
    rows = station_rows.copy()
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


def _common_features(
    province: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, list[str]]:
    names = config["data"]["columns"]
    timestamp = names["timestamp"]
    origins = pd.DatetimeIndex(province[timestamp])
    history_length = int(config["features"]["history_length"])
    histories = np.stack(
        province[names["power_history"]].map(
            lambda value: history_values(value, history_length)
        )
    )
    feature_names = [f"power_lag_{lag}" for lag in range(history_length, 0, -1)]
    return pd.DataFrame(histories, index=origins, columns=feature_names), feature_names


def _horizon_features(
    province: pd.DataFrame,
    weather: pd.DataFrame,
    config: Config,
    horizon: int,
    *,
    require_target: bool,
) -> pd.DataFrame:
    feature_cfg = config["features"]
    names = config["data"]["columns"]
    origins = weather.index
    features = weather.copy()

    target_timestamp = origins + pd.Timedelta(
        minutes=horizon * int(feature_cfg["minutes_per_point"])
    )
    hour = (
        target_timestamp.hour.to_numpy()
        + target_timestamp.minute.to_numpy() / 60.0
    )
    features["time__hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    features["time__hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    features["target_timestamp"] = target_timestamp.to_numpy()
    if names["power_future"] in province:
        features["target_power"] = province[names["power_future"]].map(
            lambda value: array_at(value, horizon - 1)
        ).to_numpy(dtype=np.float32)
    elif require_target:
        raise KeyError(f"训练/测试数据缺少列: {names['power_future']}")
    return features


def build_feature_data(
    data: DataInput | None,
    config: Config,
    horizons: list[int],
    *,
    require_target: bool,
    date_range: str | None = None,
) -> FeatureData:
    """Aggregate weather per file, then build common and horizon features."""
    if not horizons:
        raise ValueError("horizons 不能为空")
    province_batches: list[pd.DataFrame] = []
    weather_batches: dict[int, list[pd.DataFrame]] = {
        horizon: [] for horizon in horizons
    }
    selected_weather: list[str] | None = None
    seen_origins: set[pd.Timestamp] = set()
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

        province, station_rows = _model_rows(frame, config)
        origins = pd.DatetimeIndex(province[timestamp_col])
        overlap = sorted(set(origins).intersection(seen_origins))
        if overlap:
            raise ValueError(
                "同一起报时刻出现在多个文件中，无法保证逐文件气象加权等价："
                f"{[str(origin) for origin in overlap[:10]]}"
            )
        seen_origins.update(origins)

        for horizon in horizons:
            weather_batches[horizon].append(
                weighted_weather_features(
                    station_rows,
                    origins,
                    selected_weather,
                    horizon - 1,
                    config,
                )
            )
        province_batches.append(province)
        print(
            f"当前文件加权天气计算完成："
            f"origins={len(origins):,}, horizons={horizons}"
        )
        del frame, province, station_rows

    if selected_weather is None or not province_batches:
        raise ValueError("输入数据为空，无法构造特征")

    province = pd.concat(province_batches, ignore_index=True).sort_values(
        timestamp_col, ignore_index=True
    )
    common, common_names = _common_features(province, config)
    horizon_features: dict[int, pd.DataFrame] = {}
    weather_names: list[str] | None = None
    for horizon in horizons:
        weather = pd.concat(weather_batches[horizon]).sort_index()
        horizon_features[horizon] = _horizon_features(
            province,
            weather,
            config,
            horizon,
            require_target=require_target,
        )
        current_names = weather.columns.tolist()
        if weather_names is None:
            weather_names = current_names
        elif current_names != weather_names:
            raise RuntimeError(f"horizon={horizon} 的特征列不一致")

    if weather_names is None:
        raise RuntimeError("内部错误：未构造任何 horizon 特征")
    feature_names = common_names + weather_names + [
        "time__hour_sin",
        "time__hour_cos",
    ]
    print(
        f"特征构造完成：origins={len(seen_origins):,}, horizons={horizons}；"
        "场站级原始数据已逐文件释放，公共历史功率特征只保存一份"
    )
    return FeatureData(
        common,
        horizon_features,
        feature_names,
        selected_weather,
        require_target,
    )
