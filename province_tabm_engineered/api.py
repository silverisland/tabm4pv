from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, ConfigInput, dump_config, load_config
from .data import DataInput, load_data
from .features import build_samples, weather_columns
from .model import infer_array, load_one, resolve_device, seed_everything, train_one, transform


def _horizons(config: Config) -> list[int]:
    value = config["model"].get("horizons", "all")
    count = int(config["features"]["n_horizons"])
    if value == "all":
        return list(range(1, count + 1))
    result = sorted({int(item) for item in value})
    if not result or result[0] < 1 or result[-1] > count:
        raise ValueError(f"model.horizons 必须位于 1..{count}")
    return result


def _split(
    samples: pd.DataFrame, config: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = config["training"]["split"]
    reference = samples["timestamp"] + pd.Timedelta(
        minutes=int(config["features"]["n_horizons"])
        * int(config["features"]["minutes_per_point"])
    )
    day = reference.dt.normalize()
    validation_start = split.get("validation_start")
    test_start = split.get("test_start")
    if validation_start and test_start:
        validation_start = pd.Timestamp(validation_start)
        test_start = pd.Timestamp(test_start)
    elif not validation_start and not test_start:
        days = pd.Index(day.unique()).sort_values()
        validation_days = int(split["validation_days"])
        test_days = int(split["test_days"])
        if len(days) < validation_days + test_days + 1:
            raise ValueError("可用日期不足以生成训练、验证和测试集")
        validation_start = days[-(validation_days + test_days)]
        test_start = days[-test_days]
    else:
        raise ValueError("validation_start 和 test_start 必须同时设置或同时留空")
    train = samples[day < validation_start].copy()
    validation = samples[(day >= validation_start) & (day < test_start)].copy()
    test = samples[day >= test_start].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("训练、验证、测试切分后均必须非空")
    return train, validation, test


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
    }


def train(config: ConfigInput, data: DataInput | None = None) -> dict[str, Any]:
    """Train all configured horizons and return artifact paths and metrics."""
    cfg = load_config(config)
    seed_everything(int(cfg["training"]["seed"]))
    frame = load_data(data, cfg, require_target=True)
    weather = weather_columns(frame, cfg)
    output_dir = Path(cfg["output"]["checkpoint_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, output_dir / "config_resolved.yaml")

    metrics, test_predictions = [], []
    for horizon in _horizons(cfg):
        samples, names = build_samples(
            frame, cfg, horizon, weather, require_target=True
        )
        train_frame, validation_frame, test_frame = _split(samples, cfg)
        fit_result = train_one(
            train_frame, validation_frame, names, horizon, cfg, output_dir
        )
        prediction = _predict_horizon(output_dir, test_frame, horizon, cfg)
        score = _metrics(test_frame["target_power"].to_numpy(), prediction)
        metrics.append(
            {
                "horizon": horizon,
                "minutes_ahead": horizon
                * int(cfg["features"]["minutes_per_point"]),
                "feature_count": len(names),
                **fit_result,
                "test_rmse": score["rmse"],
                "test_mae": score["mae"],
            }
        )
        result = test_frame[["timestamp", "target_timestamp", "target_power"]].copy()
        result["horizon"] = horizon
        result[cfg["output"]["prediction_column"]] = prediction
        test_predictions.append(result)

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(test_predictions, ignore_index=True)
    metrics_df.to_csv(output_dir / "metrics_by_horizon.csv", index=False)
    predictions_df.to_parquet(output_dir / "test_predictions.parquet", index=False)
    metadata = {
        "artifact_version": 1,
        "model_type": "TabM",
        "horizons": _horizons(cfg),
        "weather_columns": weather,
        "mean_test_rmse": float(metrics_df["test_rmse"].mean()),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return {
        "checkpoint_dir": output_dir,
        "metrics": metrics_df,
        "predictions": predictions_df,
    }


def _checkpoint(
    checkpoint: str | Path,
) -> tuple[Path, list[Path], dict[str, Any]]:
    path = Path(checkpoint).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".pt" or path.parent.name != "models":
            raise ValueError("单文件 checkpoint 必须是 models/model_hXX.pt")
        checkpoint_dir, selected_model = path.parent.parent, path
    else:
        checkpoint_dir, selected_model = path, None

    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"checkpoint 缺少 metadata.json: {checkpoint_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if selected_model is not None:
        models = [selected_model]
    else:
        models = [
            checkpoint_dir / "models" / f"model_h{int(horizon):02d}.pt"
            for horizon in metadata["horizons"]
        ]
    missing = [str(model) for model in models if not model.exists()]
    if missing:
        raise FileNotFoundError(f"checkpoint 元数据引用的模型不存在: {missing}")
    return checkpoint_dir, models, metadata


def _predict_horizon(
    checkpoint_dir: Path, samples: pd.DataFrame, horizon: int, config: Config
) -> np.ndarray:
    device = resolve_device(config["model"].get("device", "auto"))
    model_path = checkpoint_dir / "models" / f"model_h{horizon:02d}.pt"
    model, preprocessor, payload = load_one(model_path, checkpoint_dir, device)
    names = list(payload["feature_names"])
    missing = sorted(set(names).difference(samples.columns))
    if missing:
        raise RuntimeError(f"推理特征与 checkpoint 不一致，缺少: {missing}")
    x = transform(preprocessor, samples[names].to_numpy(dtype=np.float32))
    lower, upper = map(float, config["model"]["prediction_clip"])
    return np.clip(
        infer_array(
            model,
            x,
            device=device,
            batch_size=int(config["training"]["inference_batch_size"]),
            target_scale=float(payload["target_scale"]),
        ),
        lower,
        upper,
    )


def _validate_weather(
    metadata: dict[str, Any], weather: list[str]
) -> None:
    expected_weather = metadata["weather_columns"]
    if weather != expected_weather:
        raise ValueError(f"推理气象列 {weather} 与训练时 {expected_weather} 不一致")


def _predict_loaded_frame(
    checkpoint_dir: Path,
    model_paths: list[Path],
    frame: pd.DataFrame,
    config: Config,
    weather: list[str],
    *,
    include_target: bool,
) -> pd.DataFrame:
    outputs = []
    prediction_col = config["output"]["prediction_column"]
    for model_path in model_paths:
        horizon = int(model_path.stem.removeprefix("model_h"))
        samples, _ = build_samples(
            frame,
            config,
            horizon,
            weather,
            require_target=include_target,
        )
        result_columns = ["timestamp", "target_timestamp"]
        if include_target:
            result_columns.append("target_power")
        result = samples[result_columns].copy()
        result["horizon"] = horizon
        result[prediction_col] = _predict_horizon(
            checkpoint_dir, samples, horizon, config
        )
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["timestamp", "horizon"], ignore_index=True
    )


def predict(
    ckpt_path: str | Path, data: DataInput, config: ConfigInput
) -> pd.DataFrame:
    """Run inference and return a long-form prediction DataFrame."""
    cfg = load_config(config)
    checkpoint_dir, model_paths, metadata = _checkpoint(ckpt_path)
    frame = load_data(data, cfg, require_target=False)
    weather = weather_columns(frame, cfg)
    _validate_weather(metadata, weather)
    return _predict_loaded_frame(
        checkpoint_dir,
        model_paths,
        frame,
        cfg,
        weather,
        include_target=False,
    )


def evaluate(
    ckpt_path: str | Path, data: DataInput, config: ConfigInput
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a checkpoint on labeled data; return (metrics, predictions)."""
    cfg = load_config(config)
    checkpoint_dir, model_paths, metadata = _checkpoint(ckpt_path)
    frame = load_data(data, cfg, require_target=True)
    weather = weather_columns(frame, cfg)
    _validate_weather(metadata, weather)
    predictions = _predict_loaded_frame(
        checkpoint_dir,
        model_paths,
        frame,
        cfg,
        weather,
        include_target=True,
    )
    metrics = []
    prediction_col = cfg["output"]["prediction_column"]
    for horizon, current in predictions.groupby("horizon", sort=True):
        score = _metrics(
            current["target_power"].to_numpy(),
            current[prediction_col].to_numpy(),
        )
        metrics.append({"horizon": int(horizon), "sample_count": len(current), **score})
    return pd.DataFrame(metrics), predictions


def test(
    ckpt_path: str | Path, data: DataInput, config: ConfigInput
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explicit test interface; equivalent to evaluate()."""
    return evaluate(ckpt_path, data, config)
