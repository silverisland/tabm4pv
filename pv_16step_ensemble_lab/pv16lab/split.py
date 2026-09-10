from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SplitPlan:
    train: np.ndarray
    calibration_a: np.ndarray
    calibration_b: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    summary: pd.DataFrame


def build_split(origins: pd.DatetimeIndex, config: dict) -> SplitPlan:
    split = config["split"]
    data = config["data"]
    days = pd.DatetimeIndex(origins.normalize().unique()).sort_values()
    if len(days) < 8:
        raise ValueError("At least 8 calendar days are required for train/calibration/test")

    if split["mode"] == "dates":
        validation_start = pd.Timestamp(split["train_end"]).normalize() + pd.Timedelta(days=1)
        test_start = pd.Timestamp(split["validation_end"]).normalize() + pd.Timedelta(days=1)
    else:
        train_days = max(3, int(np.floor(len(days) * float(split["train_ratio"]))))
        validation_days = max(2, int(np.floor(len(days) * float(split["validation_ratio"]))))
        if train_days + validation_days >= len(days):
            validation_days = 2
            train_days = len(days) - validation_days - 1
        validation_start = days[train_days]
        test_start = days[train_days + validation_days]

    validation_days_index = days[(days >= validation_start) & (days < test_start)]
    if len(validation_days_index) < 2:
        raise ValueError("Validation interval needs at least two calendar days")
    cut_index = max(1, int(np.floor(len(validation_days_index) * float(split["calibration_a_ratio"]))))
    if cut_index >= len(validation_days_index):
        cut_index = len(validation_days_index) - 1
    calibration_b_start = validation_days_index[cut_index]

    lead = pd.Timedelta(
        minutes=int(data["interval_minutes"]) * int(data["forecast_horizons"])
    )
    train = np.asarray((origins + lead) < validation_start)
    calibration_a = np.asarray((origins >= validation_start) & ((origins + lead) < calibration_b_start))
    calibration_b = np.asarray((origins >= calibration_b_start) & ((origins + lead) < test_start))
    validation = calibration_a | calibration_b
    test = np.asarray(origins >= test_start)
    for name, mask in (
        ("train", train), ("calibration_a", calibration_a),
        ("calibration_b", calibration_b), ("test", test),
    ):
        if not mask.any():
            raise ValueError(f"Split {name} is empty; provide a longer range or explicit dates")

    rows = []
    for name, mask in (
        ("train", train), ("calibration_a", calibration_a),
        ("calibration_b", calibration_b), ("validation", validation), ("test", test),
    ):
        selected = origins[mask]
        rows.append({
            "split": name, "rows": len(selected),
            "start": selected.min(), "end": selected.max(),
        })
    return SplitPlan(train, calibration_a, calibration_b, validation, test, pd.DataFrame(rows))

