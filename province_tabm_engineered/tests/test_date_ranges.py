from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from province_tabm_engineered.data import load_data


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
