from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


ConfigInput = str | Path | Mapping[str, Any]
Config = dict[str, Any]


def _require_keys(section: Mapping[str, Any], section_name: str, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise ValueError(f"config.{section_name} 缺少配置项: {missing}")


def _positive(value: Any, name: str) -> None:
    if float(value) <= 0:
        raise ValueError(f"config.{name} 必须大于 0")


def validate_config(config: Config) -> None:
    """Fail early for malformed values that would otherwise fail deep in training."""
    sections = ("data", "features", "model", "training", "output")
    for section in sections:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"config 缺少字典配置段: {section}")

    data = config["data"]
    _require_keys(
        data,
        "data",
        ("province_station", "plant_station_pattern", "province_capacity", "columns"),
    )
    _require_keys(
        data["columns"],
        "data.columns",
        ("timestamp", "station", "capacity", "power_history", "power_future"),
    )
    _positive(data["province_capacity"], "data.province_capacity")

    features = config["features"]
    _require_keys(features, "features", ("history_length", "n_horizons", "minutes_per_point"))
    for key in ("history_length", "n_horizons", "minutes_per_point"):
        _positive(features[key], f"features.{key}")

    model = config["model"]
    _require_keys(model, "model", ("target_scale", "prediction_clip", "device"))
    _positive(model["target_scale"], "model.target_scale")
    clip = model["prediction_clip"]
    if (
        not isinstance(clip, (list, tuple))
        or len(clip) != 2
        or float(clip[0]) > float(clip[1])
    ):
        raise ValueError(
            "config.model.prediction_clip 必须是 [lower, upper] 且 lower <= upper"
        )
    reserved = {"n_num_features", "cat_cardinalities", "d_out", "num_embeddings"}
    conflicts = sorted(reserved.intersection(model.get("architecture", {})))
    if conflicts:
        raise ValueError(
            f"config.model.architecture 不允许覆盖数据相关参数: {conflicts}"
        )

    training = config["training"]
    _require_keys(
        training,
        "training",
        (
            "seed",
            "epochs",
            "batch_size",
            "inference_batch_size",
            "early_stopping_patience",
            "learning_rate",
            "weight_decay",
            "split",
        ),
    )
    for key in (
        "epochs",
        "batch_size",
        "inference_batch_size",
        "early_stopping_patience",
        "learning_rate",
    ):
        _positive(training[key], f"training.{key}")
    split = training["split"]
    validation_start, test_start = split.get("validation_start"), split.get("test_start")
    if bool(validation_start) != bool(test_start):
        raise ValueError("validation_start 和 test_start 必须同时设置或同时留空")
    if not validation_start:
        _require_keys(split, "training.split", ("validation_days", "test_days"))
        _positive(split["validation_days"], "training.split.validation_days")
        _positive(split["test_days"], "training.split.test_days")

    _require_keys(config["output"], "output", ("checkpoint_dir", "prediction_column"))


def load_config(config: ConfigInput) -> Config:
    """Load a YAML file or copy a mapping, then validate required sections."""
    if isinstance(config, Mapping):
        result = deepcopy(dict(config))
    else:
        path = Path(config).expanduser().resolve()
        with path.open("r", encoding="utf-8") as file:
            result = yaml.safe_load(file) or {}
    validate_config(result)
    return result


def dump_config(config: Mapping[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(dict(config), file, allow_unicode=True, sort_keys=False)
