import unittest

from pvreglab.report import build_report


class ReportTests(unittest.TestCase):
    def test_full_loso_is_not_mixed_with_quick_stage(self):
        records = [
            self.record("quick_ablation", "baseline", 0.8),
            self.record("quick_ablation", "candidate", 0.9),
            self.record("full_loso", "baseline", 0.90),
            self.record("full_loso", "candidate", 0.902),
        ]
        config = {
            "identity_score_tolerance": 0.0003,
            "acceptance": {
                "min_mean_ood_gain": 0.001,
                "min_positive_station_ratio": 0.5,
                "max_worst_station_drop": 0.002,
            },
        }
        report = build_report(records, config)
        self.assertIn("Primary comparison stage: `full_loso`", report)
        self.assertIn("+0.002000", report)
        self.assertNotIn("+0.100000", report)

    def test_legacy_records_without_implementation_are_ignored(self):
        legacy = self.record("full_loso", "candidate", 0.99)
        del legacy["request"]["implementation_hash"]
        report = build_report(
            [self.record("full_loso", "baseline", 0.90), legacy],
            {
                "identity_score_tolerance": 0.0003,
                "acceptance": {
                    "min_mean_ood_gain": 0.001,
                    "min_positive_station_ratio": 0.5,
                    "max_worst_station_drop": 0.002,
                },
            },
        )
        self.assertIn("Ignored 1 successful legacy record", report)
        self.assertNotIn("0.990000", report)

    @staticmethod
    def record(stage, mode, score):
        return {
            "request": {
                "stage": stage,
                "mode": mode,
                "held_out_station": "a",
                "seed": 0,
                "implementation_hash": (
                    "private-baseline" if mode == "baseline" else "candidate123"
                ),
                "final_test": False,
            },
            "result": {"status": "ok", "score": score},
        }


if __name__ == "__main__":
    unittest.main()
