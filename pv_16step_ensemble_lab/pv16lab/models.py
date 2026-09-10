from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Dataset
from .features import FeatureSet, correction_features
from .split import SplitPlan


@dataclass
class ModelResult:
    predictions: dict[str, np.ndarray]
    fusion_weights: pd.DataFrame


def _clip_ratio(value: np.ndarray) -> np.ndarray:
    return np.clip(value, 0.0, 1.2)


def _valid_rows(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return mask & np.isfinite(y) & np.isfinite(x).any(axis=1)


def _load_external(
    path: str, dataset: Dataset, horizons: int, timezone: str
) -> np.ndarray:
    frame = pd.read_parquet(path)
    required = {"timestamp_win", "station", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"External DS {path} missing columns: {sorted(missing)}")
    frame = frame.loc[frame["station"].astype(str).eq(dataset.station)].copy()
    frame["timestamp_win"] = pd.to_datetime(frame["timestamp_win"])
    if isinstance(frame["timestamp_win"].dtype, pd.DatetimeTZDtype):
        frame["timestamp_win"] = (
            frame["timestamp_win"].dt.tz_convert(timezone).dt.tz_localize(None)
        )
    if frame["timestamp_win"].duplicated().any():
        raise ValueError(f"External DS {path} has duplicate timestamp_win")
    values = {}
    for row in frame.itertuples(index=False):
        array = np.asarray(row.prediction, dtype=np.float32).reshape(-1)
        if len(array) != horizons:
            raise ValueError(f"External DS {path}: prediction length must be {horizons}")
        values[pd.Timestamp(row.timestamp_win)] = array
    return np.vstack([
        values.get(origin, np.full(horizons, np.nan, dtype=np.float32))
        for origin in dataset.origins
    ])


def _fit_base_models(
    dataset: Dataset,
    features: FeatureSet,
    split: SplitPlan,
    config: dict,
    model_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    horizons = int(config["data"]["forecast_horizons"])
    y_ratio = dataset.targets / dataset.capacity[:, None]
    predictions: dict[str, np.ndarray] = {}
    artifacts: dict[str, object] = {}
    models_cfg = config["models"]

    if models_cfg["persistence"].get("enabled", True):
        predictions["persistence"] = _clip_ratio(features.persistence) * dataset.capacity[:, None]

    for model_name in ("ridge", "lightgbm"):
        if not models_cfg.get(model_name, {}).get("enabled", False):
            continue
        output = np.full((len(dataset.frame), horizons), np.nan, dtype=np.float32)
        fitted = []
        for horizon in range(1, horizons + 1):
            x = features.by_horizon[horizon]
            y = y_ratio[:, horizon - 1]
            valid = _valid_rows(x, y, split.train)
            if valid.sum() < 20:
                raise ValueError(f"{model_name} P{horizon}: fewer than 20 valid training rows")
            if model_name == "ridge":
                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    Ridge(alpha=float(models_cfg[model_name].get("alpha", 2.0))),
                )
            else:
                try:
                    from lightgbm import LGBMRegressor
                except ImportError as error:
                    raise ImportError("lightgbm model is enabled but lightgbm is not installed") from error
                params = {
                    key: value for key, value in models_cfg[model_name].items()
                    if key != "enabled"
                }
                model = LGBMRegressor(
                    objective="regression", random_state=int(config["experiment"]["random_seed"]),
                    verbosity=-1, n_jobs=-1, **params,
                )
            model.fit(x[valid], y[valid])
            output[:, horizon - 1] = _clip_ratio(model.predict(x))
            fitted.append(model)
        predictions[model_name] = output * dataset.capacity[:, None]
        artifacts[model_name] = fitted
        if config["output"].get("save_models", True):
            joblib.dump(fitted, model_dir / f"{model_name}.joblib")

    for item in models_cfg.get("external_ds", []):
        name = str(item["name"])
        if name in predictions:
            raise ValueError(f"Duplicate model name: {name}")
        predictions[name] = _load_external(
            item["path"], dataset, horizons, config["data"]["timezone"]
        )
    return predictions, artifacts


def _fit_correction(
    dataset: Dataset,
    features: FeatureSet,
    split: SplitPlan,
    predictions: dict[str, np.ndarray],
    config: dict,
    model_dir: Path,
) -> None:
    correction = config["correction"]
    if not correction.get("enabled", True):
        return
    base_name = correction.get("base_model", "lightgbm")
    if base_name not in predictions:
        candidates = [name for name in predictions if name != "persistence"] or list(predictions)
        base_name = candidates[0]
    base = predictions[base_name]
    horizons = int(config["data"]["forecast_horizons"])
    output = np.full_like(base, np.nan)
    fitted = []
    for horizon in range(1, horizons + 1):
        x = correction_features(dataset, features, base, horizon)
        target_residual = (dataset.targets[:, horizon - 1] - base[:, horizon - 1]) / dataset.capacity
        valid = _valid_rows(x, target_residual, split.calibration_a) & np.isfinite(base[:, horizon - 1])
        if valid.sum() < 10:
            output[:, horizon - 1] = base[:, horizon - 1]
            fitted.append(None)
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            Ridge(alpha=float(correction.get("alpha", 5.0))),
        )
        model.fit(x[valid], target_residual[valid])
        corrected_ratio = base[:, horizon - 1] / dataset.capacity + model.predict(x)
        output[:, horizon - 1] = _clip_ratio(corrected_ratio) * dataset.capacity
        fitted.append(model)
    predictions["recent_correction"] = output
    if config["output"].get("save_models", True):
        joblib.dump({"base_model": base_name, "models": fitted}, model_dir / "recent_correction.joblib")


def _convex_weights(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    count = matrix.shape[1]
    initial = np.full(count, 1.0 / count)
    result = minimize(
        lambda weight: float(np.mean((matrix @ weight - target) ** 2)),
        initial, method="SLSQP", bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weight: weight.sum() - 1.0},
    )
    return result.x if result.success else initial


def _fit_fusion(
    dataset: Dataset,
    split: SplitPlan,
    predictions: dict[str, np.ndarray],
    config: dict,
) -> pd.DataFrame:
    if not config["fusion"].get("enabled", True) or len(predictions) < 2:
        return pd.DataFrame()
    component_names = list(predictions)
    horizons = int(config["data"]["forecast_horizons"])
    fused = np.full((len(dataset.frame), horizons), np.nan, dtype=np.float32)
    weight_rows = []
    for horizon in range(1, horizons + 1):
        matrix = np.column_stack([predictions[name][:, horizon - 1] for name in component_names])
        target = dataset.targets[:, horizon - 1]
        valid = split.calibration_b & np.isfinite(target) & np.isfinite(matrix).all(axis=1)
        weights = _convex_weights(matrix[valid], target[valid]) if valid.sum() >= 10 else np.full(len(component_names), 1 / len(component_names))
        can_predict = np.isfinite(matrix).all(axis=1)
        fused[can_predict, horizon - 1] = matrix[can_predict] @ weights
        weight_rows.extend(
            {"horizon": horizon, "model": name, "weight": float(weight)}
            for name, weight in zip(component_names, weights)
        )
    predictions["fusion"] = fused
    return pd.DataFrame(weight_rows)


def train_and_predict(
    dataset: Dataset,
    features: FeatureSet,
    split: SplitPlan,
    config: dict,
    model_dir: Path,
) -> ModelResult:
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions, _ = _fit_base_models(dataset, features, split, config, model_dir)
    _fit_correction(dataset, features, split, predictions, config, model_dir)
    weights = _fit_fusion(dataset, split, predictions, config)
    if not weights.empty:
        weights.to_csv(model_dir / "fusion_weights.csv", index=False)
    return ModelResult(predictions, weights)
