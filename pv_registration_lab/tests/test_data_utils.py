import tempfile
import unittest
from pathlib import Path

import pandas as pd

from adapters.data_utils import (
    add_station_identity,
    discover_station_files,
    load_capacity_map,
    select_target_periods,
)


class DataContractTests(unittest.TestCase):
    def contract(self, station_info_path):
        return {
            "parquet_root": "/unused",
            "parquet_glob": "station=*.parquet",
            "station_info_path": str(station_info_path),
            "station_info_station_column": "plantid",
            "station_info_capacity_column": "GCCAPACITY",
            "station_aliases": {
                "雅砻江": "雅砻江",
                "雅砻江解放站": "雅砻江",
            },
            "capacity_overrides": {"雅砻江": 465.0},
            "target_history_raw_station": "雅砻江",
            "target_history_start": "2024-01-01",
            "target_history_end": "2024-12-31 23:59:59",
            "target_evaluation_raw_station": "雅砻江解放站",
            "target_evaluation_start": "2025-01-01",
            "target_evaluation_end": "2025-12-31 23:59:59",
        }

    def test_capacity_override_and_string_plantid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "station_info.csv"
            pd.DataFrame(
                {"plantid": [101, "source_b"], "GCCAPACITY": [10.0, 20.0]}
            ).to_csv(path, index=False)
            capacities = load_capacity_map(self.contract(path))
        self.assertEqual(capacities["101"], 10.0)
        self.assertEqual(capacities["雅砻江"], 465.0)

    def test_target_alias_and_periods_are_disjoint(self):
        contract = self.contract("unused.csv")
        frame = pd.DataFrame(
            {
                "station_col": ["雅砻江", "雅砻江解放站"],
                "timestamp_win": ["2024-06-01", "2025-06-01"],
            }
        )
        frame = add_station_identity(
            frame, "station_col", contract["station_aliases"]
        )
        history, evaluation = select_target_periods(
            frame, "timestamp_win", contract
        )
        self.assertEqual(history.iloc[0]["station"], "雅砻江")
        self.assertEqual(evaluation.iloc[0]["station"], "雅砻江")
        self.assertEqual(history.iloc[0]["station_raw"], "雅砻江")
        self.assertEqual(
            evaluation.iloc[0]["station_raw"], "雅砻江解放站"
        )

    def test_station_files_are_discovered_inside_configured_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "station=a.parquet").touch()
            (root / "ignore.csv").touch()
            contract = self.contract("unused.csv")
            contract["parquet_root"] = str(root)
            files = discover_station_files(contract)
        self.assertEqual([path.name for path in files], ["station=a.parquet"])


if __name__ == "__main__":
    unittest.main()
