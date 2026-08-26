import unittest

from protected.evaluator import ResultContractError, validate_adapter_result


class EvaluatorContractTests(unittest.TestCase):
    def audit(self):
        return {
            "train_row_count": 10,
            "validation_row_count": 2,
            "evaluation_row_count": 3,
            "calibration_row_count": 4,
            "train_rows_hash": "trainhash",
            "validation_rows_hash": "validhash",
            "evaluation_rows_hash": "evalhash1",
            "calibration_rows_hash": "calibhash",
            "target_index": 15,
            "weather_index": 15,
            "capacity_map_hash": "capacityhash",
            "model_config_hash": "modelhash",
            "preprocessing_hash": "preprocesshash",
            "evaluation_rows_in_train": 0,
            "calibration_eval_overlap": 0,
        }

    def request(self):
        return {
            "experiment_id": "abc",
            "metadata": {"weather": {"future_index": 15}},
        }

    def test_aggregate_result_is_allowed(self):
        payload = {
            "experiment_id": "abc",
            "status": "ok",
            "score": 0.9,
            "audit": self.audit(),
            "diagnostics": {"max_shift_minutes": 15.0},
            "per_month": {"1": 0.9},
        }
        self.assertEqual(validate_adapter_result(payload, self.request()), payload)

    def test_raw_predictions_are_rejected(self):
        payload = {
            "experiment_id": "abc",
            "status": "ok",
            "score": 0.9,
            "audit": self.audit(),
            "diagnostics": {},
            "per_month": {},
            "predictions": [1.0, 2.0],
        }
        with self.assertRaises(ResultContractError):
            validate_adapter_result(payload, self.request())

    def test_nested_raw_predictions_are_rejected(self):
        payload = {
            "experiment_id": "abc",
            "status": "ok",
            "score": 0.9,
            "audit": self.audit(),
            "diagnostics": {"debug": {"predictions": [1.0]}},
            "per_month": {},
        }
        with self.assertRaises(ResultContractError):
            validate_adapter_result(payload, self.request())

    def test_row_leakage_is_rejected(self):
        audit = self.audit()
        audit["evaluation_rows_in_train"] = 1
        payload = {
            "experiment_id": "abc",
            "status": "ok",
            "score": 0.9,
            "audit": audit,
            "diagnostics": {},
            "per_month": {},
        }
        with self.assertRaises(ResultContractError):
            validate_adapter_result(payload, self.request())

    def test_target_and_weather_indices_must_match(self):
        payload = {
            "experiment_id": "abc",
            "status": "ok",
            "score": 0.9,
            "audit": self.audit(),
            "diagnostics": {},
            "per_month": {},
        }
        payload["audit"]["weather_index"] = 16
        with self.assertRaises(ResultContractError):
            validate_adapter_result(payload, self.request())

    def test_indices_must_match_configured_future_index(self):
        payload = {
            "experiment_id": "abc",
            "status": "ok",
            "score": 0.9,
            "audit": self.audit(),
            "diagnostics": {},
            "per_month": {},
        }
        request = self.request()
        request["metadata"]["weather"]["future_index"] = 16
        with self.assertRaises(ResultContractError):
            validate_adapter_result(payload, request)

    def test_numeric_timestamp_offset_diagnostic_is_allowed(self):
        payload = {
            "experiment_id": "abc",
            "status": "ok",
            "score": 0.9,
            "audit": self.audit(),
            "diagnostics": {"target_timestamp_offset_minutes": 0.0},
            "per_month": {},
        }
        self.assertEqual(validate_adapter_result(payload, self.request()), payload)

    def test_raw_timestamp_string_is_rejected(self):
        payload = {
            "experiment_id": "abc",
            "status": "ok",
            "score": 0.9,
            "audit": self.audit(),
            "diagnostics": {"target_timestamp": "2025-01-01 12:00:00"},
            "per_month": {},
        }
        with self.assertRaises(ResultContractError):
            validate_adapter_result(payload, self.request())


if __name__ == "__main__":
    unittest.main()
