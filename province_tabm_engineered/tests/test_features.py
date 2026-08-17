from __future__ import annotations

import numpy as np
import pandas as pd

from province_tabm_engineered.features import build_samples


def test_build_inference_samples_without_future_target():
    config = {
        "data": {
            "province_station": "province_guangxi_solar",
            "plant_station_pattern": r"^plant_guangfu\d{4}$",
            "columns": {
                "timestamp": "timestamp_win",
                "station": "station",
                "capacity": "cap_power_on",
                "power_history": "observe_power",
                "power_future": "observe_power_future",
            },
        },
        "features": {
            "history_length": 4,
            "minutes_per_point": 15,
            "wind_direction_keywords": ["wind_direction"],
        },
    }
    timestamp = pd.Timestamp("2026-08-17 12:00:00")
    data = pd.DataFrame(
        {
            "timestamp_win": [timestamp, timestamp],
            "station": ["province_guangxi_solar", "plant_guangfu0001"],
            "cap_power_on": [15000.0, 100.0],
            "observe_power": [np.arange(4), np.arange(4)],
            "ghi_predict": [np.arange(2), np.array([10.0, 20.0])],
            "source_file": ["input", "input"],
        }
    )
    samples, names = build_samples(
        data, config, 2, ["ghi_predict"], require_target=False
    )
    assert len(samples) == 1
    assert "target_power" not in samples
    assert samples.loc[0, "target_timestamp"] == timestamp + pd.Timedelta(minutes=30)
    assert samples.loc[0, "weighted__ghi_predict__mean"] == 20.0
    assert names[:4] == ["power_lag_4", "power_lag_3", "power_lag_2", "power_lag_1"]


def test_v2_feature_values_match_original_recipe():
    config = {
        "data": {
            "province_station": "province_guangxi_solar",
            "plant_station_pattern": r"^plant_guangfu\d{4}$",
            "columns": {
                "timestamp": "timestamp_win",
                "station": "station",
                "capacity": "cap_power_on",
                "power_history": "observe_power",
                "power_future": "observe_power_future",
            },
        },
        "features": {
            "history_length": 4,
            "minutes_per_point": 15,
            "wind_direction_keywords": ["wind_direction", "winddirection", "wd_"],
        },
    }
    timestamp = pd.Timestamp("2026-08-17 06:00:00")
    data = pd.DataFrame(
        {
            "timestamp_win": [timestamp] * 3,
            "station": [
                "province_guangxi_solar",
                "plant_guangfu0001",
                "plant_guangfu0002",
            ],
            "cap_power_on": [15000.0, 100.0, 300.0],
            "observe_power": [
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.zeros(4),
                np.zeros(4),
            ],
            "observe_power_future": [
                np.array([50.0, 60.0]),
                np.zeros(2),
                np.zeros(2),
            ],
            "ghi_predict": [
                np.zeros(2),
                np.array([10.0, 20.0]),
                np.array([30.0, np.nan]),
            ],
            "source_file": ["input"] * 3,
        }
    )

    samples, names = build_samples(
        data, config, 2, ["ghi_predict"], require_target=True
    )

    assert names == [
        "power_lag_4",
        "power_lag_3",
        "power_lag_2",
        "power_lag_1",
        "weighted__ghi_predict__capacity_coverage",
        "weighted__ghi_predict__mean",
        "time__hour_sin",
        "time__hour_cos",
    ]
    np.testing.assert_allclose(samples.loc[0, names[:4]], [1.0, 2.0, 3.0, 4.0])
    assert samples.loc[0, "weighted__ghi_predict__capacity_coverage"] == 0.25
    assert samples.loc[0, "weighted__ghi_predict__mean"] == 20.0
    assert samples.loc[0, "target_power"] == 60.0
    expected_hour = 6.5
    np.testing.assert_allclose(
        samples.loc[0, ["time__hour_sin", "time__hour_cos"]],
        [np.sin(2 * np.pi * expected_hour / 24), np.cos(2 * np.pi * expected_hour / 24)],
        rtol=1e-6,
    )
