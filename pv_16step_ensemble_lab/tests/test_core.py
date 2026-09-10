from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pv16lab.data import Dataset
from pv16lab.delivery import ds_frame, read_ds
from pv16lab.metrics import _values


class CoreTests(unittest.TestCase):
    def test_metric_is_perfect_for_perfect_prediction(self):
        frame = pd.DataFrame({
            "groundtruth": [0.0, 100.0, 200.0],
            "prediction": [0.0, 100.0, 200.0],
            "cap_power_on": [500.0, 500.0, 500.0],
        })
        config = {"evaluation": {"capacity_floor_ratio": 0.2}}
        result = _values(frame, config)
        self.assertAlmostEqual(result["official_accuracy"], 1.0)
        self.assertAlmostEqual(result["rmse"], 0.0)

    def test_ds_round_trip_keeps_sixteen_point_arrays(self):
        origins = pd.date_range("2025-01-01", periods=3, freq="15min")
        empty = np.empty((3, 16), dtype=np.float32)
        dataset = Dataset(
            frame=pd.DataFrame(index=range(3)), station="S1", origins=origins,
            capacity=np.full(3, 100.0), power_history=np.empty((3, 0)),
            raw_targets=empty, targets=empty, historical_weather={},
            forecast_weather={}, weather_columns=[], quality=pd.DataFrame(),
            target_conflicts=pd.DataFrame(), missing_origins=pd.DatetimeIndex([]),
        )
        prediction = np.arange(48, dtype=np.float32).reshape(3, 16)
        frame = ds_frame(dataset, prediction, np.ones(3, dtype=bool), 16)
        self.assertEqual(frame.columns.tolist(), ["timestamp_win", "station", "prediction"])
        self.assertEqual(frame.loc[0, "prediction"].shape, (16,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_ds.parquet"
            frame.to_parquet(path, index=False)
            restored = read_ds(path)
            self.assertIsInstance(restored.loc[0, "prediction"], np.ndarray)
            np.testing.assert_array_equal(restored.loc[2, "prediction"], prediction[2])


if __name__ == "__main__":
    unittest.main()
