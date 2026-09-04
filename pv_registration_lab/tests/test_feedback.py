import json
import tempfile
import unittest
from pathlib import Path

from pvreglab.feedback import export_feedback


class FeedbackExportTests(unittest.TestCase):
    def test_feedback_removes_station_names_and_paths(self):
        records = [
            {
                "request": {
                    "stage": "full_loso",
                    "mode": "baseline",
                    "held_out_station": "private_station_a",
                    "seed": 0,
                    "implementation_hash": "private-baseline",
                    "final_test": False,
                },
                "result": {
                    "status": "ok",
                    "score": 0.9,
                    "runtime_seconds": 1.0,
                    "diagnostics": {"max_shift_minutes": 0.0},
                    "per_month": {"1": 0.9},
                },
            }
        ]
        config = {
            "source_stations": ["private_station_a", "private_station_b"],
            "target_station": "雅砻江",
            "identity_score_tolerance": 0.0003,
            "acceptance": {
                "min_mean_ood_gain": 0.001,
                "min_positive_station_ratio": 0.5,
                "max_worst_station_drop": 0.002,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            markdown, payload = export_feedback(
                records, config, Path(directory)
            )
            combined = markdown.read_text() + payload.read_text()
            parsed = json.loads(payload.read_text())
        self.assertNotIn("private_station_a", combined)
        self.assertNotIn("雅砻江", combined)
        self.assertEqual(
            parsed["records"][0]["request"]["held_out_station"],
            "station_001",
        )


if __name__ == "__main__":
    unittest.main()
