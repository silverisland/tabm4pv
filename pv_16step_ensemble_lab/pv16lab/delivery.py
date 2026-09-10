from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import Dataset
from .models import ModelResult
from .split import SplitPlan


def ds_frame(
    dataset: Dataset, prediction: np.ndarray, mask: np.ndarray, horizons: int
) -> pd.DataFrame:
    indices = np.flatnonzero(mask)
    frame = pd.DataFrame({
        "timestamp_win": dataset.origins[indices],
        "station": dataset.station,
        "prediction": [
            np.asarray(prediction[index, :horizons], dtype=np.float32)
            for index in indices
        ],
    })
    if frame["timestamp_win"].duplicated().any():
        raise ValueError("DS output contains duplicate timestamp_win")
    if not frame["prediction"].map(lambda value: isinstance(value, np.ndarray) and value.shape == (horizons,)).all():
        raise ValueError("DS prediction must be a one-dimensional ndarray of configured length")
    return frame.sort_values(["station", "timestamp_win"], ignore_index=True)


def save_ds_outputs(
    dataset: Dataset,
    split: SplitPlan,
    result: ModelResult,
    output_dir: Path,
    config: dict,
) -> list[Path]:
    ds_dir = output_dir / "ds"
    ds_dir.mkdir(parents=True, exist_ok=True)
    horizons = int(config["data"]["forecast_horizons"])
    paths = []
    for name, prediction in result.predictions.items():
        frame = ds_frame(dataset, prediction, split.test, horizons)
        path = ds_dir / f"{name}_ds.parquet"
        frame.to_parquet(path, index=False)
        paths.append(path)
    return paths


def read_ds(path: str | Path, horizons: int = 16) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"timestamp_win", "station", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"DS file is missing columns: {sorted(missing)}")
    frame["timestamp_win"] = pd.to_datetime(frame["timestamp_win"])
    frame["station"] = frame["station"].astype(str)
    frame["prediction"] = frame["prediction"].map(
        lambda value: np.asarray(value, dtype=np.float32).reshape(-1)
    )
    invalid = frame["prediction"].map(len).ne(horizons)
    if invalid.any():
        raise ValueError(f"DS prediction length must be {horizons}")
    return frame

