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
            "minutes_per_point": 15,
            "history": {"observe_power": {"indices": {"start": -4, "stop": None}}},
            "future": {"ghi_predict": {
                "index": {"base": -1, "horizon_offset": True},
                "capacity_weighted": True,
            }},
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
    features, columns, _ = build_feature_data(data, config, [2])
    assert len(features) == 1
    assert "target_power__h02" not in features
    assert features.loc[0, "target_timestamp__h02"] == timestamp + pd.Timedelta(
        minutes=30
    )
    assert features.loc[0, "future__weighted__ghi_predict__index_1__h02"] == 20.0
    assert columns[2][:4] == [
        "history__observe_power__index_-4",
        "history__observe_power__index_-3",
        "history__observe_power__index_-2",
        "history__observe_power__index_-1",
    ]


def test_feature_values_and_column_order():
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
            "minutes_per_point": 15,
            "history": {"observe_power": {"indices": {"start": -4, "stop": None}}},
            "future": {"ghi_predict": {
                "index": {"base": -1, "horizon_offset": True},
                "capacity_weighted": True,
            }},
            "time": ["hour", "hour_sin", "hour_cos"],
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

    features, columns, _ = build_feature_data(data, config, [2])
    names = columns[2]

    assert features.filter(like="history__observe_power").columns.tolist() == names[:4]

    assert names == [
        "history__observe_power__index_-4",
        "history__observe_power__index_-3",
        "history__observe_power__index_-2",
        "history__observe_power__index_-1",
        "future__weighted__ghi_predict__index_1__h02",
        "time__hour__h02",
        "time__hour_sin__h02",
        "time__hour_cos__h02",
    ]
    np.testing.assert_allclose(features.loc[0, names[:4]], [1.0, 2.0, 3.0, 4.0])
    assert features.loc[0, "future__weighted__ghi_predict__index_1__h02"] == 20.0
    assert features.loc[0, "target_power__h02"] == 60.0
    expected_hour = 6.5
    assert features.loc[0, "time__hour__h02"] == 6
    np.testing.assert_allclose(
        features.loc[0, ["time__hour_sin__h02", "time__hour_cos__h02"]],
        [np.sin(2 * np.pi * expected_hour / 24), np.cos(2 * np.pi * expected_hour / 24)],
        rtol=1e-6,
    )


def test_horizon_aligned_history_weather_is_capacity_weighted():
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
            "minutes_per_point": 15,
            "history": {
                "observe_power": {"indices": [-4, -3]},
                "GHI_SOLARGIS": {
                    "index": {"base": -96, "horizon_offset": True},
                    "capacity_weighted": True,
                },
            },
            "future": {"ghi_predict": {
                "index": {"base": -1, "horizon_offset": True},
                "capacity_weighted": True,
            }},
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
            "observe_power": [np.zeros(4)] * 3,
            "GHI_SOLARGIS": [
                np.zeros(96),
                np.arange(96, dtype=np.float32),
                np.arange(96, dtype=np.float32) + 100.0,
            ],
            "ghi_predict": [np.zeros(16)] * 3,
        }
    )

    features, columns, _ = build_feature_data(data, config, [1, 16])

    h01 = "history__weighted__GHI_SOLARGIS__index_-95__h01"
    h16 = "history__weighted__GHI_SOLARGIS__index_-80__h16"
    assert h01 in columns[1]
    assert h16 in columns[16]
    assert features.loc[0, h01] == 76.0
    assert features.loc[0, h16] == 91.0
