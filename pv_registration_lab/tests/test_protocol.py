import unittest

from protected.split_protocol import make_request


class SplitProtocolTests(unittest.TestCase):
    def test_held_out_station_never_trains(self):
        request = make_request(
            stage="test",
            mode="baseline",
            seed=0,
            source_stations=["a", "b", "c"],
            held_out_station="b",
            target_station="target",
            calibration_days=21,
            implementation_hash="impl-a",
        )
        self.assertEqual(request.training_stations, ["a", "c"])
        self.assertNotIn("b", request.training_stations)

    def test_unknown_pseudo_target_is_rejected(self):
        with self.assertRaises(ValueError):
            make_request(
                stage="test",
                mode="baseline",
                seed=0,
                source_stations=["a", "b"],
                held_out_station="target",
                target_station="target",
                calibration_days=21,
                implementation_hash="impl-a",
            )

    def test_final_can_train_on_disjoint_target_history(self):
        request = make_request(
            stage="final_target_test",
            mode="candidate",
            seed=0,
            source_stations=["a", "b"],
            held_out_station="雅砻江",
            target_station="雅砻江",
            calibration_days=21,
            implementation_hash="impl-a",
            target_history_role="train_and_registration",
            final_test=True,
        )
        self.assertIn("雅砻江", request.training_stations)
        self.assertEqual(request.target_history_role, "train_and_registration")


if __name__ == "__main__":
    unittest.main()
