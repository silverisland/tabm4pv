from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Dataset
from .models import ModelResult
from .split import SplitPlan


@dataclass
class Evaluation:
    predictions: pd.DataFrame
    horizon_metrics: pd.DataFrame
    segment_metrics: pd.DataFrame
    regime_metrics: pd.DataFrame
    curve_metrics: pd.DataFrame
    daily_metrics: pd.DataFrame
    monthly_metrics: pd.DataFrame
    fusion_comparison: pd.DataFrame


def prediction_long(
    dataset: Dataset, split: SplitPlan, result: ModelResult, config: dict
) -> pd.DataFrame:
    horizons = int(config["data"]["forecast_horizons"])
    minutes = int(config["data"]["interval_minutes"])
    test_indices = np.flatnonzero(split.test)
    rows = []
    for name, prediction in result.predictions.items():
        for row_index in test_indices:
            origin = dataset.origins[row_index]
            for horizon in range(1, horizons + 1):
                rows.append({
                    "station": dataset.station,
                    "timestamp_win": origin,
                    "target_timestamp": origin + pd.Timedelta(minutes=minutes * horizon),
                    "horizon": horizon,
                    "model": name,
                    "groundtruth": float(dataset.targets[row_index, horizon - 1]),
                    "raw_groundtruth": float(dataset.raw_targets[row_index, horizon - 1]),
                    "prediction": float(prediction[row_index, horizon - 1]),
                    "cap_power_on": float(dataset.capacity[row_index]),
                    "split": "test",
                })
    return pd.DataFrame(rows)


def _values(group: pd.DataFrame, config: dict) -> dict[str, float | int]:
    valid = np.isfinite(group["groundtruth"]) & np.isfinite(group["prediction"])
    truth = group.loc[valid, "groundtruth"].to_numpy(float)
    forecast = group.loc[valid, "prediction"].to_numpy(float)
    capacity = group.loc[valid, "cap_power_on"].to_numpy(float)
    if not len(truth):
        return {name: np.nan for name in (
            "mae", "rmse", "nmae_capacity", "nrmse_capacity", "bias",
            "official_accuracy",
        )} | {"samples": 0, "coverage": 0.0}
    error = forecast - truth
    denominator = np.maximum(
        truth, float(config["evaluation"]["capacity_floor_ratio"]) * capacity
    )
    return {
        "samples": int(len(truth)),
        "coverage": float(valid.mean()),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "nmae_capacity": float(np.mean(np.abs(error) / capacity)),
        "nrmse_capacity": float(np.sqrt(np.mean((error / capacity) ** 2))),
        "bias": float(np.mean(error)),
        "official_accuracy": float(1.0 - np.mean(np.abs(error) / denominator)),
    }


def _group_metrics(frame: pd.DataFrame, keys: list[str], config: dict) -> pd.DataFrame:
    rows = []
    grouper = keys[0] if len(keys) == 1 else keys
    for values, group in frame.groupby(grouper, sort=True, dropna=False):
        values = (values,) if len(keys) == 1 else values
        rows.append(dict(zip(keys, values)) | _values(group, config))
    return pd.DataFrame(rows)


def _regime_by_date(dataset: Dataset, config: dict) -> pd.Series:
    minutes = int(config["data"]["interval_minutes"])
    truth = pd.DataFrame({
        "target_timestamp": dataset.origins + pd.Timedelta(minutes=minutes),
        "groundtruth": dataset.raw_targets[:, 0],
    }).dropna()
    truth["date"] = truth["target_timestamp"].dt.normalize()
    peaks = truth.groupby("date")["groundtruth"].max().sort_index()
    previous = peaks.shift(1)
    contiguous = peaks.index.to_series().diff().eq(pd.Timedelta(days=1)).to_numpy()
    ratio = peaks / previous.where(previous > 0)
    labels = pd.Series("unknown", index=peaks.index, dtype=object)
    labels.loc[contiguous & (ratio >= float(config["evaluation"]["rise_ratio"]))] = "rise"
    labels.loc[contiguous & (ratio <= float(config["evaluation"]["drop_ratio"]))] = "drop"
    labels.loc[contiguous & ratio.between(
        float(config["evaluation"]["drop_ratio"]),
        float(config["evaluation"]["rise_ratio"]), inclusive="neither"
    )] = "stable"
    return labels


def _daily_metrics(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    horizons = int(config["data"]["forecast_horizons"])
    minutes = int(config["data"]["interval_minutes"])
    points_per_day = 24 * 60 // minutes
    missing_error = float(config["evaluation"]["missing_normalized_error"])
    floor = float(config["evaluation"]["capacity_floor_ratio"])
    rows = []
    for model, source in frame.groupby("model", sort=True):
        coverage = source.groupby("horizon")["target_timestamp"].agg(["min", "max"])
        start_raw, end_raw = coverage["min"].max(), coverage["max"].min()
        start = start_raw.normalize()
        if start_raw != start:
            start += pd.Timedelta(days=1)
        end = end_raw.normalize()
        final_point = end + pd.Timedelta(minutes=minutes * (points_per_day - 1))
        if end_raw < final_point:
            end -= pd.Timedelta(days=1)
        if end < start:
            continue
        target_times = pd.date_range(
            start, end + pd.Timedelta(days=1), freq=f"{minutes}min", inclusive="left"
        )
        expected = pd.MultiIndex.from_product(
            [target_times, range(1, horizons + 1)],
            names=["target_timestamp", "horizon"],
        ).to_frame(index=False)
        values = source[["target_timestamp", "horizon", "prediction"]]
        grid = expected.merge(values, on=["target_timestamp", "horizon"], how="left", validate="one_to_one")
        truth_map = source.dropna(subset=["groundtruth"]).drop_duplicates("target_timestamp").set_index("target_timestamp")["groundtruth"]
        capacity_map = source.drop_duplicates("target_timestamp").set_index("target_timestamp")["cap_power_on"]
        grid["groundtruth"] = grid["target_timestamp"].map(truth_map)
        grid["capacity"] = grid["target_timestamp"].map(capacity_map)
        valid = grid["prediction"].notna() & grid["groundtruth"].notna() & grid["capacity"].notna()
        grid["normalized_error"] = missing_error
        grid.loc[valid, "normalized_error"] = (
            (grid.loc[valid, "prediction"] - grid.loc[valid, "groundtruth"]).abs()
            / np.maximum(grid.loc[valid, "groundtruth"], floor * grid.loc[valid, "capacity"])
        )
        grid["date"] = grid["target_timestamp"].dt.normalize()
        daily = grid.groupby("date", as_index=False).agg(
            mean_normalized_error=("normalized_error", "mean"),
            available_forecasts=("prediction", "count"),
            available_targets=("groundtruth", "count"),
        )
        daily["model"] = model
        daily["accuracy"] = 1.0 - daily["mean_normalized_error"]
        daily["expected_forecasts"] = points_per_day * horizons
        daily["forecast_coverage"] = daily["available_forecasts"] / daily["expected_forecasts"]
        daily["complete_day"] = (
            (daily["available_forecasts"] == daily["expected_forecasts"])
            & (daily["available_targets"] == daily["expected_forecasts"])
        )
        rows.append(daily)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _monthly_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    source = daily.copy()
    source["month"] = source["date"].dt.to_period("M").dt.to_timestamp()
    return source.groupby(["model", "month"], as_index=False).agg(
        monthly_average_accuracy=("accuracy", "mean"),
        monthly_average_normalized_error=("mean_normalized_error", "mean"),
        days_included=("date", "count"),
        complete_days=("complete_day", "sum"),
        forecast_coverage=("forecast_coverage", "mean"),
    )


def _curve_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    interval_hours = 0.25
    rows = []
    for (model, origin), group in frame.groupby(["model", "timestamp_win"], sort=True):
        group = group.sort_values("horizon")
        valid = np.isfinite(group["groundtruth"]) & np.isfinite(group["prediction"])
        if valid.sum() != len(group):
            continue
        truth = group["groundtruth"].to_numpy(float)
        prediction = group["prediction"].to_numpy(float)
        rows.append({
            "model": model,
            "timestamp_win": origin,
            "peak_error": float(prediction.max() - truth.max()),
            "absolute_peak_error": float(abs(prediction.max() - truth.max())),
            "energy_error": float((prediction.sum() - truth.sum()) * interval_hours),
            "absolute_energy_error": float(abs(prediction.sum() - truth.sum()) * interval_hours),
            "ramp_mae": float(np.mean(np.abs(np.diff(prediction) - np.diff(truth)))),
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    return detail.groupby("model", as_index=False).agg(
        curves=("timestamp_win", "count"),
        mean_peak_error=("peak_error", "mean"),
        mean_absolute_peak_error=("absolute_peak_error", "mean"),
        mean_energy_error=("energy_error", "mean"),
        mean_absolute_energy_error=("absolute_energy_error", "mean"),
        mean_ramp_mae=("ramp_mae", "mean"),
    )


def evaluate(dataset: Dataset, split: SplitPlan, result: ModelResult, config: dict) -> Evaluation:
    predictions = prediction_long(dataset, split, result, config)
    horizon = _group_metrics(predictions, ["model", "horizon"], config)
    bins = [0, 4, 8, 12, 16]
    labels = ["P1-P4", "P5-P8", "P9-P12", "P13-P16"]
    predictions["segment"] = pd.cut(predictions["horizon"], bins=bins, labels=labels)
    segment = _group_metrics(predictions, ["model", "segment"], config)
    regimes = _regime_by_date(dataset, config)
    predictions["target_date"] = predictions["target_timestamp"].dt.normalize()
    predictions["regime"] = predictions["target_date"].map(regimes).fillna("unknown")
    regime = _group_metrics(predictions, ["model", "regime"], config)
    curve = _curve_metrics(predictions)
    daily = _daily_metrics(predictions, config)
    monthly = _monthly_metrics(daily)
    comparison_rows = []
    if "fusion" in set(horizon.get("model", [])):
        for point, group in horizon.groupby("horizon", sort=True):
            fusion_row = group.loc[group["model"].eq("fusion")]
            components = group.loc[~group["model"].eq("fusion")]
            if fusion_row.empty or components.empty:
                continue
            fusion_row = fusion_row.iloc[0]
            best_accuracy = components.loc[components["official_accuracy"].idxmax()]
            best_rmse = components.loc[components["rmse"].idxmin()]
            comparison_rows.append({
                "horizon": int(point),
                "fusion_accuracy": fusion_row["official_accuracy"],
                "best_component_by_accuracy": best_accuracy["model"],
                "best_component_accuracy": best_accuracy["official_accuracy"],
                "accuracy_delta": fusion_row["official_accuracy"] - best_accuracy["official_accuracy"],
                "fusion_rmse": fusion_row["rmse"],
                "best_component_by_rmse": best_rmse["model"],
                "best_component_rmse": best_rmse["rmse"],
                "rmse_reduction": best_rmse["rmse"] - fusion_row["rmse"],
            })
    comparison = pd.DataFrame(comparison_rows)
    return Evaluation(predictions, horizon, segment, regime, curve, daily, monthly, comparison)
