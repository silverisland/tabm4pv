from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {"name": "pv_16step_ensemble", "random_seed": 42},
    "data": {
        "format": "parquet",
        "station": None,
        "station_col": "station",
        "time_col": "timestamp_win",
        "capacity_col": "cap_power_on",
        "history_power_col": "observe_power",
        "target_col": "observe_power_future",
        "timezone": "Asia/Shanghai",
        "interval_minutes": 15,
        "history_length": 672,
        "future_length": 192,
        "forecast_horizons": 16,
    },
    "weather": {
        "historical_columns": [],
        "forecast_columns": [],
        "forecast_suffix": "_predict",
    },
    "split": {
        "mode": "auto",
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "calibration_a_ratio": 0.50,
        "train_end": None,
        "validation_end": None,
    },
    "features": {
        "power_lags": [1, 2, 4, 8, 16, 32, 96, 192, 288, 672],
        "stat_windows": [4, 16, 96, 192, 672],
    },
    "models": {
        "persistence": {"enabled": True},
        "ridge": {"enabled": True, "alpha": 2.0},
        "lightgbm": {
            "enabled": True,
            "n_estimators": 350,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "min_child_samples": 30,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        "external_ds": [],
    },
    "correction": {
        "enabled": True,
        "base_model": "lightgbm",
        "alpha": 5.0,
    },
    "fusion": {"enabled": True, "granularity": "horizon"},
    "evaluation": {
        "capacity_floor_ratio": 0.20,
        "missing_normalized_error": 1.0,
        "rise_ratio": 1.30,
        "drop_ratio": 0.70,
    },
    "output": {"dir": "../outputs/pv_16step_ensemble_v1", "save_models": True},
}


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _resolve_path(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    config = _merge(DEFAULT_CONFIG, supplied)
    if not config["data"].get("path"):
        raise ValueError("config.data.path is required")
    if isinstance(config["data"]["path"], list):
        config["data"]["path"] = [
            _resolve_path(item, config_path.parent) for item in config["data"]["path"]
        ]
    else:
        config["data"]["path"] = _resolve_path(
            config["data"]["path"], config_path.parent
        )
    config["output"]["dir"] = _resolve_path(
        config["output"]["dir"], config_path.parent
    )
    for item in config["models"].get("external_ds", []):
        item["path"] = _resolve_path(item["path"], config_path.parent)
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    data = config["data"]
    horizons = int(data["forecast_horizons"])
    if horizons <= 0 or horizons > int(data["future_length"]):
        raise ValueError("forecast_horizons must be in 1..future_length")
    if int(data["interval_minutes"]) <= 0:
        raise ValueError("interval_minutes must be positive")
    split = config["split"]
    if split["mode"] not in {"auto", "dates"}:
        raise ValueError("split.mode must be 'auto' or 'dates'")
    if split["mode"] == "dates" and not (
        split.get("train_end") and split.get("validation_end")
    ):
        raise ValueError("date split requires train_end and validation_end")
    enabled = [
        name
        for name in ("persistence", "ridge", "lightgbm")
        if config["models"].get(name, {}).get("enabled", False)
    ]
    enabled.extend(item["name"] for item in config["models"].get("external_ds", []))
    if not enabled:
        raise ValueError("At least one component model must be enabled")


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}


def dump_config(config: Mapping[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(public_config(config), handle, allow_unicode=True, sort_keys=False)

