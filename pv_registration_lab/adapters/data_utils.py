"""Private-environment helpers for station aliases and installed capacity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


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
