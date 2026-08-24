from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, ConfigInput, dump_config, load_config
from .data import DataInput
from .delivery import (
    INTERNAL_PREDICTION_COLUMN,
    combine_delivery_frames,
    delivery_frames,
    save_delivery_frames,
)
from .features import build_feature_data
from .model import infer_array, load_one, resolve_device, seed_everything, train_one, transform


def _print_config(task: str, source: ConfigInput, config: Config) -> None:
    source_text = (
        "<Python mapping>"
        if isinstance(source, Mapping)
        else str(Path(source).expanduser().resolve())
    )
    print(f"{task}参数：config={source_text}")
    print(
        f"{task}参数：data={config['data'].get('path')}, "
        f"date_ranges={config['data'].get('date_ranges')}, "
        f"checkpoint={config['output']['checkpoint_dir']}"
    )
    print(
        f"{task}参数：device={config['model'].get('device', 'auto')}, "
        f"horizons={config['model'].get('horizons', 'all')}, "
        f"history_length={config['features']['history_length']}, "
        f"weather={config['features'].get('weather_columns', '<自动发现>')}"
    )


def _horizons(config: Config) -> list[int]:
    value = config["model"].get("horizons", "all")
    if value == "all":
        return list(range(1, int(config["features"]["n_horizons"]) + 1))
    return sorted({int(horizon) for horizon in value})


def _split_by_date(data: pd.DataFrame, config: Config) -> pd.DataFrame:
    split = config["training"]["split"]
    reference = data["timestamp"] + pd.Timedelta(
        minutes=int(config["features"]["n_horizons"])
        * int(config["features"]["minutes_per_point"])
    )
    days = reference.dt.normalize()
    validation_start = split.get("validation_start")
    test_start = split.get("test_start")
    if validation_start:
        validation_start = pd.Timestamp(validation_start)
        test_start = pd.Timestamp(test_start)
    else:
        unique_days = pd.Index(days.unique()).sort_values()
        validation_days = int(split["validation_days"])
        test_days = int(split["test_days"])
        validation_start = unique_days[-(validation_days + test_days)]
        test_start = unique_days[-test_days]
    result = data.copy()
    result["__split"] = np.where(
        days < validation_start,
        "train",
        np.where(days < test_start, "validation", "test"),
    )
    return result


def _training_data(
    data: DataInput | None, config: Config, horizons: list[int]
) -> tuple[pd.DataFrame, dict[int, list[str]], list[str]]:
    if not config["data"].get("date_ranges"):
        frame, columns, weather = build_feature_data(data, config, horizons)
        return _split_by_date(frame, config), columns, weather

    parts = []
    columns: dict[int, list[str]] = {}
    weather: list[str] = []
    for split in ("train", "validation", "test"):
        frame, columns, weather = build_feature_data(
            data, config, horizons, date_range=split
        )
        frame["__split"] = split
        parts.append(frame)
    return pd.concat(parts, ignore_index=True), columns, weather


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


def _checkpoint(
    checkpoint: str | Path,
) -> tuple[Path, list[Path], dict[str, Any]]:
    path = Path(checkpoint).expanduser().resolve()
    checkpoint_dir = path.parent.parent if path.is_file() else path
    metadata = json.loads(
        (checkpoint_dir / "metadata.json").read_text(encoding="utf-8")
    )
    model_paths = (
        [path]
        if path.is_file()
        else [
            checkpoint_dir / "models" / f"model_h{int(horizon):02d}.pt"
            for horizon in metadata["horizons"]
        ]
    )
    print(
        f"checkpoint={checkpoint_dir}, "
        f"models={[str(model) for model in model_paths]}"
    )
    return checkpoint_dir, model_paths, metadata


def _predict_horizon(
    checkpoint_dir: Path,
    model_path: Path,
    data: pd.DataFrame,
    config: Config,
) -> np.ndarray:
    device = resolve_device(config["model"].get("device", "auto"))
    model, preprocessor, payload = load_one(model_path, checkpoint_dir, device)
    values = data[payload["feature_names"]].to_numpy(dtype=np.float32)
    values = transform(preprocessor, values)
    lower, upper = map(float, config["model"]["prediction_clip"])
    return np.clip(
        infer_array(
            model,
            values,
            device=device,
            batch_size=int(config["training"]["inference_batch_size"]),
            target_scale=float(payload["target_scale"]),
        ),
        lower,
        upper,
    )


def _predict_all(
    checkpoint_dir: Path,
    model_paths: list[Path],
    data: pd.DataFrame,
    config: Config,
    *,
    include_target: bool,
) -> pd.DataFrame:
    outputs = []
    for model_path in model_paths:
        horizon = int(model_path.stem.removeprefix("model_h"))
        suffix = f"__h{horizon:02d}"
        target = f"target_power{suffix}"
        current = data.dropna(subset=[target]) if include_target else data
        result = current[["timestamp", f"target_timestamp{suffix}"]].rename(
            columns={f"target_timestamp{suffix}": "target_timestamp"}
        )
        if include_target:
            result["target_power"] = current[target].to_numpy()
        result["horizon"] = horizon
        result[INTERNAL_PREDICTION_COLUMN] = _predict_horizon(
            checkpoint_dir, model_path, current, config
        )
        outputs.append(result)
        print(f"horizon={horizon:02d} 推理完成：rows={len(result):,}")
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["timestamp", "horizon"], ignore_index=True
    )


def train(config: ConfigInput, data: DataInput | None = None) -> dict[str, Any]:
    cfg = load_config(config)
    _print_config("训练", config, cfg)
    seed_everything(int(cfg["training"]["seed"]))
    horizons = _horizons(cfg)
    frame, columns_by_horizon, weather = _training_data(data, cfg, horizons)

    output_dir = Path(cfg["output"]["checkpoint_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, output_dir / "config_resolved.yaml")

    metrics, predictions = [], []
    for horizon in horizons:
        suffix = f"__h{horizon:02d}"
        target = f"target_power{suffix}"
        current = frame.dropna(subset=[target])
        train_frame = current[current["__split"].eq("train")]
        validation_frame = current[current["__split"].eq("validation")]
        test_frame = current[current["__split"].eq("test")]
        columns = columns_by_horizon[horizon]
        print(
            f"horizon={horizon:02d}：features={len(columns)}, "
            f"train={len(train_frame):,}, validation={len(validation_frame):,}, "
            f"test={len(test_frame):,}"
        )
        fit = train_one(
            train_frame,
            validation_frame,
            columns,
            target,
            horizon,
            cfg,
            output_dir,
        )
        model_path = output_dir / "models" / f"model_h{horizon:02d}.pt"
        prediction = _predict_horizon(output_dir, model_path, test_frame, cfg)
        score = _metrics(test_frame[target].to_numpy(), prediction)
        metrics.append(
            {
                "horizon": horizon,
                "minutes_ahead": horizon
                * int(cfg["features"]["minutes_per_point"]),
                "feature_count": len(columns),
                **fit,
                "test_rmse": score["rmse"],
                "test_mae": score["mae"],
            }
        )
        result = test_frame[["timestamp", f"target_timestamp{suffix}", target]].rename(
            columns={
                f"target_timestamp{suffix}": "target_timestamp",
                target: "target_power",
            }
        )
        result["horizon"] = horizon
        result[INTERNAL_PREDICTION_COLUMN] = prediction
        predictions.append(result)

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(predictions, ignore_index=True)
    metrics_df.to_csv(output_dir / "metrics_by_horizon.csv", index=False)
    forecasts = delivery_frames(predictions_df, cfg, skip_incomplete=True)
    save_delivery_frames(forecasts, output_dir, cfg)
    metadata = {
        "horizons": horizons,
        "weather_columns": weather,
        "mean_test_rmse": float(metrics_df["test_rmse"].mean()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"训练完成：checkpoint={output_dir}")
    return {
        "checkpoint_dir": output_dir,
        "metrics": metrics_df,
        "predictions": predictions_df,
    }


def predict(
    ckpt_path: str | Path, data: DataInput, config: ConfigInput
) -> pd.DataFrame:
    cfg = load_config(config)
    _print_config("推理", config, cfg)
    checkpoint_dir, model_paths, _ = _checkpoint(ckpt_path)
    horizons = sorted(
        int(path.stem.removeprefix("model_h")) for path in model_paths
    )
    frame, _, _ = build_feature_data(data, cfg, horizons)
    origins = frame["timestamp"].unique()
    if len(origins) != 1:
        raise ValueError(f"predict() 只允许一个起报时刻，当前为 {len(origins)} 个")
    predictions = _predict_all(
        checkpoint_dir, model_paths, frame, cfg, include_target=False
    )
    result = next(
        iter(delivery_frames(predictions, cfg, skip_incomplete=False).values())
    )
    print(f"推理完成：origin={origins[0]}, rows={len(result):,}，不保存文件")
    return result


def test(
    ckpt_path: str | Path, data: DataInput | None, config: ConfigInput
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(config)
    _print_config("测试", config, cfg)
    checkpoint_dir, model_paths, _ = _checkpoint(ckpt_path)
    horizons = sorted(
        int(path.stem.removeprefix("model_h")) for path in model_paths
    )
    frame, _, _ = build_feature_data(
        data, cfg, horizons, date_range="test"
    )
    predictions = _predict_all(
        checkpoint_dir, model_paths, frame, cfg, include_target=True
    )
    metrics = []
    for horizon, current in predictions.groupby("horizon", sort=True):
        score = _metrics(
            current["target_power"].to_numpy(),
            current[INTERNAL_PREDICTION_COLUMN].to_numpy(),
        )
        metrics.append({"horizon": int(horizon), "sample_count": len(current), **score})
        print(
            f"horizon={int(horizon):02d}：rmse={score['rmse']:.6f}, "
            f"mae={score['mae']:.6f}"
        )
    frames = delivery_frames(predictions, cfg, skip_incomplete=True)
    save_delivery_frames(frames, checkpoint_dir, cfg)
    return pd.DataFrame(metrics), combine_delivery_frames(frames)
