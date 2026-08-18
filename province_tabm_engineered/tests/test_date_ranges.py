from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from province_tabm_engineered.data import load_data
from province_tabm_engineered.features import build_sample_sets, build_samples


def _config(data_dir: Path) -> dict:
    return {
        "data": {
            "path": str(data_dir),
            "file_glob": "plantid=*.parquet",
            "file_date_regex": r"plantid=(\d{4}-\d{2}-\d{2})\.parquet$",
            "strict_file_dates": True,
            "date_ranges": {
                "train": {"start": "2026-08-01", "end": "2026-08-02"},
                "validation": {"start": "2026-08-03", "end": "2026-08-03"},
                "test": {"start": "2026-08-04", "end": "2026-08-05"},
            },
            "province_station": "province_guangxi_solar",
            "plant_station_pattern": r"^plant_guangfu\d{4}$",
            "province_capacity": 100.0,
            "capacity_csv": None,
            "columns": {
                "timestamp": "timestamp_win",
                "station": "station",
                "capacity": "cap_power_on",
                "power_history": "observe_power",
                "power_future": "observe_power_future",
            },
        }
    }


def _frame(day: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_win": [pd.Timestamp(day)],
            "station": ["province_guangxi_solar"],
            "cap_power_on": [100.0],
            "observe_power": [np.arange(4, dtype=np.float32)],
            "observe_power_future": [np.arange(2, dtype=np.float32)],
        }
    )


def test_directory_files_are_selected_by_filename_date(tmp_path: Path):
    config = _config(tmp_path)
    for day in pd.date_range("2026-08-01", periods=5, freq="D"):
        day_text = day.strftime("%Y-%m-%d")
        _frame(day_text).to_parquet(tmp_path / f"plantid={day_text}.parquet")

    train = load_data(
        None, config, require_target=True, date_range="train"
    )
    validation = load_data(
        None, config, require_target=True, date_range="validation"
    )
    test = load_data(None, config, require_target=True, date_range="test")

    assert train["timestamp_win"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-08-01",
        "2026-08-02",
    ]
    assert validation["timestamp_win"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-08-03"
    ]
    assert test["timestamp_win"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-08-04",
        "2026-08-05",
    ]


def test_dataframe_is_selected_by_timestamp_date(tmp_path: Path):
    config = _config(tmp_path)
    data = pd.concat(
        [_frame(day.strftime("%Y-%m-%d")) for day in pd.date_range("2026-08-01", periods=5)],
        ignore_index=True,
    )
    selected = load_data(
        data, config, require_target=True, date_range="validation"
    )
    assert selected["timestamp_win"].tolist() == [pd.Timestamp("2026-08-03")]


def _streaming_config(data_dir: Path) -> dict:
    config = _config(data_dir)
    config["data"]["validate_file_content_date"] = True
    config["features"] = {
        "history_length": 4,
        "n_horizons": 2,
        "minutes_per_point": 15,
        "weather_columns": ["ghi_predict"],
        "wind_direction_keywords": ["wind_direction"],
    }
    return config


def _feature_frame(day: str, hour: int = 0) -> pd.DataFrame:
    origin = pd.Timestamp(day) + pd.Timedelta(hours=hour)
    return pd.DataFrame(
        {
            "timestamp_win": [origin, origin, origin],
            "station": [
                "province_guangxi_solar",
                "plant_guangfu0001",
                "plant_guangfu0002",
            ],
            "cap_power_on": [100.0, 25.0, 75.0],
            "observe_power": [
                np.arange(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
            ],
            "observe_power_future": [
                np.array([10.0, 20.0], dtype=np.float32),
                np.zeros(2, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
            ],
            "ghi_predict": [
                np.zeros(2, dtype=np.float32),
                np.array([1.0, 2.0], dtype=np.float32),
                np.array([5.0, 6.0], dtype=np.float32),
            ],
        }
    )


def test_streamed_samples_match_full_frame_recipe(tmp_path: Path):
    config = _streaming_config(tmp_path)
    for day in ("2026-08-01", "2026-08-02"):
        _feature_frame(day).to_parquet(tmp_path / f"plantid={day}.parquet")

    streamed, streamed_names, weather = build_sample_sets(
        None,
        config,
        [1, 2],
        require_target=True,
        date_range="train",
    )
    full_frame = load_data(
        None,
        config,
        require_target=True,
        date_range="train",
    )

    assert weather == ["ghi_predict"]
    for horizon in (1, 2):
        expected, expected_names = build_samples(
            full_frame,
            config,
            horizon,
            weather,
            require_target=True,
        )
        assert streamed_names[horizon] == expected_names
        pd.testing.assert_frame_equal(streamed[horizon], expected)


def test_filename_date_must_match_content_date(tmp_path: Path):
    config = _streaming_config(tmp_path)
    _feature_frame("2026-08-01").to_parquet(
        tmp_path / "plantid=2026-08-02.parquet"
    )

    with pytest.raises(ValueError, match="文件名日期为 2026-08-02"):
        build_sample_sets(None, config, [1, 2], require_target=True)


def test_origin_cannot_span_multiple_files(tmp_path: Path):
    config = _streaming_config(tmp_path)
    config["data"]["validate_file_content_date"] = False
    duplicated = _feature_frame("2026-08-01")
    duplicated.to_parquet(tmp_path / "plantid=2026-08-01.parquet")
    duplicated.to_parquet(tmp_path / "plantid=2026-08-02.parquet")

    with pytest.raises(ValueError, match="同一起报时刻出现在多个文件中"):
        build_sample_sets(None, config, [1, 2], require_target=True)
