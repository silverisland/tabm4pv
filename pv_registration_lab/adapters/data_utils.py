"""Private-environment helpers for station aliases and installed capacity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeatherSpec:
    future_columns: tuple[str, ...]
    future_index: int

    @property
    def minimum_array_length(self) -> int:
        return self.future_index + 1


def weather_spec_from_request(request: dict[str, Any]) -> WeatherSpec:
    """Read the single authoritative weather definition from a request."""
    try:
        raw = request["metadata"]["weather"]
        columns = tuple(str(value).strip() for value in raw["future_columns"])
        future_index = raw["future_index"]
    except (KeyError, TypeError) as error:
        raise ValueError("Request is missing metadata.weather") from error
    if not columns or any(not column for column in columns):
        raise ValueError("Weather columns must be non-empty")
    if len(columns) != len(set(columns)):
        raise ValueError("Weather columns must be unique")
    if (
        not isinstance(future_index, int)
        or isinstance(future_index, bool)
        or future_index < 0
    ):
        raise ValueError("Weather future_index must be a nonnegative int")
    return WeatherSpec(columns, future_index)


def extract_future_weather(
    frame: pd.DataFrame, spec: WeatherSpec
) -> pd.DataFrame:
    """Extract physical target-time weather using one shared array index."""
    missing = [column for column in spec.future_columns if column not in frame]
    if missing:
        raise KeyError(f"Missing configured weather columns: {missing}")
    extracted: dict[str, np.ndarray] = {}
    for column in spec.future_columns:
        extracted[f"{column}_target"] = _extract_array_index(
            frame[column], column, spec.future_index
        )
    return pd.DataFrame(extracted, index=frame.index)


def extract_future_target(
    frame: pd.DataFrame,
    target_column: str,
    spec: WeatherSpec,
) -> pd.Series:
    """Extract target power at exactly the configured weather index."""
    if target_column not in frame:
        raise KeyError(f"Missing target column: {target_column}")
    values = _extract_array_index(
        frame[target_column], target_column, spec.future_index
    )
    return pd.Series(values, index=frame.index, name=f"{target_column}_target")


def _extract_array_index(
    series: pd.Series, column: str, future_index: int
) -> np.ndarray:
    values = []
    minimum_length = future_index + 1
    for row_number, raw in enumerate(series):
        array = np.asarray(raw).reshape(-1)
        if len(array) < minimum_length:
            raise ValueError(
                f"{column} row {row_number} has length {len(array)}; "
                f"need at least {minimum_length}"
            )
        values.append(float(array[future_index]))
    return np.asarray(values, dtype=np.float32)


def discover_station_files(data_contract: dict[str, Any]) -> list[Path]:
    """Find private parquet inputs from configuration, never parent code."""
    root = Path(data_contract["parquet_root"])
    pattern = str(data_contract["parquet_glob"])
    if not root.is_dir():
        raise FileNotFoundError(f"Parquet root does not exist: {root}")
    files = sorted(path for path in root.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"No parquet files match {pattern} under {root}")
    return files


def canonical_station(value: Any, aliases: dict[str, str]) -> str:
    raw = str(value).strip()
    return str(aliases.get(raw, raw))


def load_capacity_map(data_contract: dict[str, Any]) -> dict[str, float]:
    path = Path(data_contract["station_info_path"])
    station_col = str(data_contract["station_info_station_column"])
    capacity_col = str(data_contract["station_info_capacity_column"])
    aliases = {
        str(key).strip(): str(value).strip()
        for key, value in data_contract["station_aliases"].items()
    }
    frame = pd.read_csv(path, dtype={station_col: "string"})
    missing_columns = [
        column
        for column in (station_col, capacity_col)
        if column not in frame.columns
    ]
    if missing_columns:
        raise KeyError(f"station_info is missing columns: {missing_columns}")

    frame = frame[[station_col, capacity_col]].copy()
    frame[station_col] = frame[station_col].astype(str).str.strip()
    frame[capacity_col] = pd.to_numeric(frame[capacity_col], errors="coerce")
    frame = frame.dropna(subset=[station_col, capacity_col])
    frame = frame[frame[capacity_col] > 0]
    frame["canonical_station"] = frame[station_col].map(
        lambda value: canonical_station(value, aliases)
    )

    capacity_map: dict[str, float] = {}
    for station, group in frame.groupby("canonical_station"):
        values = group[capacity_col].astype(float).unique()
        if len(values) != 1:
            raise ValueError(
                f"Conflicting capacities for canonical station={station}: "
                f"{values.tolist()}"
            )
        capacity_map[str(station)] = float(values[0])

    for station, capacity in data_contract["capacity_overrides"].items():
        capacity = float(capacity)
        if capacity <= 0:
            raise ValueError(f"Invalid capacity override for {station}")
        capacity_map[canonical_station(station, aliases)] = capacity
    return capacity_map


def add_station_identity(
    frame: pd.DataFrame,
    station_column: str,
    aliases: dict[str, str],
) -> pd.DataFrame:
    result = frame.copy()
    result["station_raw"] = result[station_column].astype(str).str.strip()
    result["station"] = result["station_raw"].map(
        lambda value: canonical_station(value, aliases)
    )
    return result


def select_target_periods(
    frame: pd.DataFrame,
    timestamp_column: str,
    data_contract: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "station_raw" not in frame.columns:
        raise KeyError("Call add_station_identity before select_target_periods")
    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    history_mask = (
        frame["station_raw"].eq(data_contract["target_history_raw_station"])
        & timestamps.between(
            data_contract["target_history_start"],
            data_contract["target_history_end"],
        )
    )
    evaluation_mask = (
        frame["station_raw"].eq(
            data_contract["target_evaluation_raw_station"]
        )
        & timestamps.between(
            data_contract["target_evaluation_start"],
            data_contract["target_evaluation_end"],
        )
    )
    history = frame.loc[history_mask].copy()
    evaluation = frame.loc[evaluation_mask].copy()
    if history.empty:
        raise ValueError("Target history period is empty")
    if evaluation.empty:
        raise ValueError("Target evaluation period is empty")
    if set(history.index) & set(evaluation.index):
        raise ValueError("Target history and evaluation rows overlap")
    return history, evaluation
