from __future__ import annotations

import numpy as np

from province_tabm_engineered.metrics import metric_values


def _config() -> dict:
    return {
        "data": {"province_capacity": 100.0},
        "evaluation": {
            "primary_metric": "official_accuracy",
            "metrics": ["rmse", "mae", "official_accuracy"],
            "capacity_floor_ratio": 0.2,
        },
    }


def test_metric_values_follow_official_denominator():
    values = metric_values(
        np.array([100.0, 0.0]),
        np.array([80.0, 20.0]),
        _config(),
    )
    assert values["rmse"] == np.sqrt(400.0)
    assert values["mae"] == 20.0
    np.testing.assert_allclose(values["official_accuracy"], 0.4)
