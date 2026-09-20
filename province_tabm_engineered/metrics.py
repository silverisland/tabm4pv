from __future__ import annotations

import numpy as np

from .config import Config


SUPPORTED_METRICS = {"rmse", "mae", "official_accuracy"}


def primary_metric(config: Config) -> str:
    name = config.get("evaluation", {}).get("primary_metric", "rmse")
    if name not in SUPPORTED_METRICS:
        raise ValueError(f"不支持的评价指标：{name}")
    return name


def requested_metrics(config: Config) -> list[str]:
    names = list(config.get("evaluation", {}).get("metrics", ["rmse", "mae"]))
    primary = primary_metric(config)
    if primary not in names:
        names.append(primary)
    unsupported = set(names).difference(SUPPORTED_METRICS)
    if unsupported:
        raise ValueError(f"不支持的评价指标：{sorted(unsupported)}")
    return list(dict.fromkeys(names))


def normalized_absolute_errors(
    target: np.ndarray,
    prediction: np.ndarray,
    config: Config,
) -> np.ndarray:
    truth = np.asarray(target, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    capacity = float(config["data"]["province_capacity"])
    ratio = float(config.get("evaluation", {}).get("capacity_floor_ratio", 0.2))
    if truth.shape != forecast.shape:
        raise ValueError("真实功率与预测功率的形状不一致")
    if capacity <= 0 or ratio < 0:
        raise ValueError("装机容量必须大于0，capacity_floor_ratio 不能小于0")
    if not np.isfinite(truth).all():
        raise ValueError("官方指标的可用功率存在空值或非有限值")

    errors = np.ones_like(truth)
    valid = np.isfinite(forecast)
    errors[valid] = np.abs(forecast[valid] - truth[valid]) / np.maximum(
        truth[valid], ratio * capacity
    )
    return errors


def metric_values(
    target: np.ndarray,
    prediction: np.ndarray,
    config: Config,
) -> dict[str, float]:
    truth = np.asarray(target, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    error = forecast - truth
    values: dict[str, float] = {}
    for name in requested_metrics(config):
        if name == "rmse":
            values[name] = float(np.sqrt(np.mean(error**2)))
        elif name == "mae":
            values[name] = float(np.mean(np.abs(error)))
        else:
            values[name] = float(
                1.0 - normalized_absolute_errors(truth, forecast, config).mean()
            )
    return values


def better_score(current: float, best: float, metric: str) -> bool:
    return current > best if metric == "official_accuracy" else current < best


def initial_best_score(metric: str) -> float:
    return -float("inf") if metric == "official_accuracy" else float("inf")
