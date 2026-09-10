from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pv16lab.config import load_config
from pv16lab.data import load_dataset


class AlignmentTests(unittest.TestCase):
    def test_p2_uses_later_rows_index0_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origins = pd.date_range("2025-01-01", periods=20, freq="15min")
            rows = []
            for index, origin in enumerate(origins):
                future = np.full(16, -999.0, dtype=np.float32)
                future[0] = index + 100.0
                rows.append({
                    "station": "S1", "timestamp_win": origin,
                    "cap_power_on": 1000.0,
                    "observe_power": np.zeros(8, dtype=np.float32),
                    "observe_power_future": future,
                    "GHI": np.zeros(8, dtype=np.float32),
                    "GHI_predict": np.zeros(16, dtype=np.float32),
                })
            data_path = root / "data.parquet"
            pd.DataFrame(rows).to_parquet(data_path, index=False)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "data": {
                    "path": str(data_path), "history_length": 8,
                    "future_length": 16, "forecast_horizons": 16,
                },
                "output": {"dir": str(root / "out")},
            }), encoding="utf-8")
            dataset = load_dataset(load_config(config_path))
            self.assertEqual(dataset.targets[0, 1], 101.0)
            self.assertEqual(dataset.targets[0, 15], 115.0)
            self.assertEqual(dataset.target_conflicts.loc[1, "conflicts"], 19)


if __name__ == "__main__":
    unittest.main()
