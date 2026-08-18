from __future__ import annotations

import numpy as np
import pandas as pd

from province_tabm_engineered.features import build_feature_data


def test_build_inference_samples_without_future_target():
    config = {
        "data": {
            "province_station": "province_guangxi_solar",
            "plant_station_pattern": r"^plant_guangfu\d{4}$",
            "province_capacity": 15000.0,
            "capacity_csv": None,
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
        }
    )
    features, columns, _ = build_feature_data(
        data, config, [2], require_target=False
    )
    assert len(features) == 1
    assert "target_power__h02" not in features
    assert features.loc[0, "target_timestamp__h02"] == timestamp + pd.Timedelta(
        minutes=30
    )
    assert features.loc[0, "weighted__ghi_predict__mean__h02"] == 20.0
    assert columns[2][:4] == [
        "power_lag_4",
        "power_lag_3",
        "power_lag_2",
        "power_lag_1",
    ]


def test_v2_feature_values_match_original_recipe():
    config = {
        "data": {
            "province_station": "province_guangxi_solar",
            "plant_station_pattern": r"^plant_guangfu\d{4}$",
            "province_capacity": 15000.0,
            "capacity_csv": None,
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
        }
    )

    features, columns, _ = build_feature_data(data, config, [2], require_target=True)
    names = columns[2]

    assert features.filter(like="power_lag_").columns.tolist() == names[:4]

    assert names == [
        "power_lag_4",
        "power_lag_3",
        "power_lag_2",
        "power_lag_1",
        "weighted__ghi_predict__capacity_coverage__h02",
        "weighted__ghi_predict__mean__h02",
        "time__hour_sin__h02",
        "time__hour_cos__h02",
    ]
    np.testing.assert_allclose(features.loc[0, names[:4]], [1.0, 2.0, 3.0, 4.0])
    assert features.loc[0, "weighted__ghi_predict__capacity_coverage__h02"] == 0.25
    assert features.loc[0, "weighted__ghi_predict__mean__h02"] == 20.0
    assert features.loc[0, "target_power__h02"] == 60.0
    expected_hour = 6.5
    np.testing.assert_allclose(
        features.loc[0, ["time__hour_sin__h02", "time__hour_cos__h02"]],
        [np.sin(2 * np.pi * expected_hour / 24), np.cos(2 * np.pi * expected_hour / 24)],
        rtol=1e-6,
    )
