from __future__ import annotations

import numpy as np
import pandas as pd

from province_tabm_engineered.evaluation import (
    calculate_official_metrics,
    evaluate_saved_predictions,
    load_saved_predictions,
)


def _config() -> dict:
    return {
        "data": {"province_capacity": 500.0},
        "features": {"minutes_per_point": 15},
        "evaluation": {
            "capacity_floor_ratio": 0.2,
            "official_horizons": 16,
            "missing_normalized_error": 1.0,
        },
    }


def _full_day(prediction: float = 90.0):
    times = pd.date_range("2026-08-01", periods=96, freq="15min")
    predictions = pd.MultiIndex.from_product(
        [times, range(1, 17)], names=["target_timestamp", "horizon"]
    ).to_frame(index=False)
    predictions["prediction"] = prediction
    truth = pd.DataFrame({"target_timestamp": times, "available_power": 100.0})
    return predictions, truth


def test_official_daily_and_monthly_metric():
    predictions, truth = _full_day()
    daily, monthly = calculate_official_metrics(predictions, truth, _config())
    assert len(daily) == 1
    assert len(monthly) == 1
    np.testing.assert_allclose(daily.loc[0, "accuracy"], 0.9)
    np.testing.assert_allclose(monthly.loc[0, "monthly_average_accuracy"], 0.9)


def test_missing_prediction_has_normalized_error_one():
    predictions, truth = _full_day(prediction=100.0)
    predictions.loc[0, "prediction"] = np.nan
    daily, _ = calculate_official_metrics(predictions, truth, _config())
    np.testing.assert_allclose(daily.loc[0, "accuracy"], 1.0 - 1.0 / (96 * 16))
    assert daily.loc[0, "available_forecasts"] == 96 * 16 - 1


def test_completely_missing_horizon_is_scored_as_missing():
    origins = pd.date_range("2026-07-31 20:00", "2026-08-01 23:30", freq="15min")
    predictions = pd.DataFrame(
        [
            {
                "target_timestamp": origin + pd.Timedelta(minutes=15 * horizon),
                "horizon": horizon,
                "prediction": 100.0,
            }
            for origin in origins
            for horizon in range(1, 16)
        ]
    )
    truth = pd.DataFrame(
        {
            "target_timestamp": pd.date_range(
                "2026-08-01", periods=96, freq="15min"
            ),
            "available_power": 100.0,
        }
    )
    daily, _ = calculate_official_metrics(predictions, truth, _config())
    np.testing.assert_allclose(daily.loc[0, "accuracy"], 15 / 16)


def test_independent_interface_writes_excel(tmp_path):
    prediction_dir = tmp_path / "forecasts"
    prediction_dir.mkdir()
    origins = pd.date_range("2026-07-31 20:00", "2026-08-01 23:30", freq="15min")
    for origin in origins:
        dtime = pd.date_range(origin + pd.Timedelta(minutes=15), periods=16, freq="15min")
        pd.DataFrame(
            {
                "dtime": dtime,
                "prediction_power": np.full(16, 100.0, dtype=np.float32),
            }
        ).to_parquet(
            prediction_dir
            / f"hw_nuoya_{origin:%Y%m%d%H%M}_ultra_short_province_20260801_tabm.parquet",
            index=False,
        )

    available_path = tmp_path / "plantid=2026-08-01.parquet"
    pd.DataFrame(
        {
            "timestamp_win": origins,
            "station": "province",
            "cap_power_on": 500.0,
            "observe_power_future": [np.full(16, 100.0, dtype=np.float32)]
            * len(origins),
        }
    ).to_parquet(available_path, index=False)
    config = {
        "data": {
            "file_glob": "plantid=*.parquet",
            "province_station": "province",
            "province_capacity": 500.0,
            "capacity_csv": None,
            "columns": {
                "timestamp": "timestamp_win",
                "station": "station",
                "capacity": "cap_power_on",
                "power_future": "observe_power_future",
            },
        },
        "features": {"minutes_per_point": 15},
        "model": {},
        "training": {},
        "evaluation": {
            "capacity_floor_ratio": 0.2,
            "official_horizons": 16,
            "missing_normalized_error": 1.0,
        },
        "output": {"prediction_column": "prediction_power"},
    }
    output = evaluate_saved_predictions(
        prediction_dir, available_path, config, tmp_path / "metrics.xlsx"
    )
    assert output.exists()
    assert pd.ExcelFile(output).sheet_names == ["日指标", "月平均", "计算说明"]
    daily = pd.read_excel(output, sheet_name="日指标")
    np.testing.assert_allclose(daily.loc[0, "超短期预测准确率"], 1.0)


def test_evaluator_prefers_consolidated_predictions(tmp_path):
    pd.DataFrame({"unexpected": [1]}).to_parquet(tmp_path / "delivery.parquet")
    origin = pd.Timestamp("2026-08-01 00:00")
    pd.DataFrame(
        {
            "forecast_origin": [origin] * 16,
            "dtime": pd.date_range(
                origin + pd.Timedelta(minutes=15), periods=16, freq="15min"
            ),
            # Object/string input reproduces environments that otherwise expose
            # horizon as a pandas nullable integer to NumPy validation.
            "horizon": np.arange(1, 17).astype(str),
            "prediction_power": 100.0,
        }
    ).to_parquet(tmp_path / "evaluation_predictions.parquet", index=False)
    config = {
        "features": {"minutes_per_point": 15},
        "evaluation": {"official_horizons": 16},
        "output": {"prediction_column": "prediction_power"},
    }
    result = load_saved_predictions(tmp_path, config)
    assert len(result) == 16
    assert result["horizon"].tolist() == list(range(1, 17))
