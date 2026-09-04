import unittest
from datetime import datetime

import numpy as np

from editable.registration import (
    DEFAULT_KNOTS,
    MonotoneWarp,
    fit_three_point,
    normalize_shape,
)
from editable.registration_features import (
    build_time_features,
    prepare_power_history,
)


class RegistrationInvariantTests(unittest.TestCase):
    def test_identity_preserves_curve(self):
        curve = np.linspace(0.0, 1.0, 96)
        warp = MonotoneWarp.identity()
        np.testing.assert_allclose(warp.apply_curve(curve), curve)

    def test_identity_preserves_last_96_history(self):
        history = np.arange(192, dtype=float)
        actual = prepare_power_history(
            history, capacity=2.0, input_len=96,
            warp=MonotoneWarp.identity(), register_history=False,
        )
        np.testing.assert_allclose(actual, history[-96:] / 2.0)

    def test_timestamp_aware_identity_registration_matches_last_96(self):
        history = np.arange(192, dtype=float)
        actual = prepare_power_history(
            history,
            capacity=2.0,
            input_len=96,
            warp=MonotoneWarp.identity(),
            register_history=True,
            history_end_timestamp=datetime(2025, 6, 1, 13, 0),
        )
        np.testing.assert_allclose(actual, history[-96:] / 2.0)

    def test_registered_history_requires_timestamp_and_margin(self):
        with self.assertRaises(ValueError):
            prepare_power_history(
                np.arange(192), 1.0, 96, MonotoneWarp.identity(),
                register_history=True,
            )
        with self.assertRaises(ValueError):
            prepare_power_history(
                np.arange(96), 1.0, 96, MonotoneWarp.identity(),
                register_history=True,
                history_end_timestamp=datetime(2025, 6, 1, 13, 0),
            )

    def test_identity_horizon_is_physical_horizon(self):
        features = build_time_features(
            datetime(2025, 6, 1, 8, 15),
            4.0,
            MonotoneWarp.identity(),
        )
        self.assertAlmostEqual(features["canonical_horizon_hours"], 4.0)
        self.assertAlmostEqual(features["predict_hour"], 12.25)

    def test_nonidentity_horizon_is_explicit(self):
        warp = MonotoneWarp(
            DEFAULT_KNOTS.copy(),
            np.array([0.0, 0.24, 0.52, 0.77, 1.0]),
        )
        features = build_time_features(
            datetime(2025, 6, 1, 8, 0), 4.0, warp
        )
        self.assertNotAlmostEqual(features["canonical_horizon_hours"], 4.0)

    def test_shape_normalization_removes_amplitude(self):
        x = np.arange(96) / 4.0
        curve = np.exp(-0.5 * ((x - 12.0) / 2.5) ** 2)
        np.testing.assert_allclose(
            normalize_shape(curve), normalize_shape(curve * 0.55)
        )

    def test_three_point_warp_is_monotone_and_bounded(self):
        x = np.arange(96) / 4.0
        template = np.exp(-0.5 * ((x - 12.0) / 2.5) ** 2)
        curve = np.exp(-0.5 * ((x - 12.5) / 2.5) ** 2)
        warp = fit_three_point(curve, template)
        self.assertTrue(np.all(np.diff(warp.physical_knots) > 0))
        slopes = warp.local_slopes()
        self.assertGreaterEqual(slopes.min(), 0.85 - 1e-12)
        self.assertLessEqual(slopes.max(), 1.18 + 1e-12)


if __name__ == "__main__":
    unittest.main()
