from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, ConfigInput, dump_config, load_config
from .data import DataInput, has_date_ranges, load_data
from .delivery import (
    INTERNAL_PREDICTION_COLUMN,
    combine_delivery_frames,
    delivery_frames,
    expected_horizons,
    save_delivery_frames,
    save_delivery_files,
)
from .features import build_samples, weather_columns
from .model import infer_array, load_one, resolve_device, seed_everything, train_one, transform


def _config_source(config: ConfigInput) -> str:
    if isinstance(config, Mapping):
        return "<Python mapping>"
    return str(Path(config).expanduser().resolve())


def _print_common_parameters(task: str, config_input: ConfigInput, config: Config) -> None:
    data = config["data"]
    features = config["features"]
    model = config["model"]
    capacity_csv = data.get("capacity_csv")
    if capacity_csv:
        capacity_csv = str(Path(capacity_csv).expanduser().resolve())
    else:
        capacity_csv = "<使用输入数据容量列>"

    print(f"{task}参数[config]：source={_config_source(config_input)}")
    print(
        f"{task}参数[data]：province_station={data['province_station']}, "
        f"province_capacity={data['province_capacity']}, "
        f"capacity_csv={capacity_csv}, columns={data['columns']}, "
        f"file_glob={data.get('file_glob', '*.parquet')}, "
        f"date_ranges={data.get('date_ranges')}"
    )
    print(
        f"{task}参数[features]：history_length={features['history_length']}, "
        f"n_horizons={features['n_horizons']}, "
        f"minutes_per_point={features['minutes_per_point']}, "
        f"weather_columns={features.get('weather_columns') or '<按后缀自动发现>'}"
    )
    print(
        f"{task}参数[model]：device={model.get('device', 'auto')}, "
        f"target_scale={model['target_scale']}, "
        f"prediction_clip={model['prediction_clip']}, "
        f"architecture={model.get('architecture', '<TabM defaults>')}"
    )


def _print_checkpoint_parameters(
    task: str,
    ckpt_input: str | Path,
    checkpoint_dir: Path,
    model_paths: list[Path],
) -> None:
    print(
        f"{task}参数[checkpoint]：input={ckpt_input}, "
        f"resolved_dir={checkpoint_dir}, "
        f"model_files={[str(path.resolve()) for path in model_paths]}"
    )


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
    _print_common_parameters("训练", config, cfg)
    training = cfg["training"]
    print(
        f"训练参数[optimizer]：seed={training['seed']}, epochs={training['epochs']}, "
        f"batch_size={training['batch_size']}, "
        f"inference_batch_size={training['inference_batch_size']}, "
        f"learning_rate={training['learning_rate']}, "
        f"weight_decay={training['weight_decay']}, "
        f"gradient_clipping_norm={training.get('gradient_clipping_norm')}, "
        f"early_stopping_patience={training['early_stopping_patience']}"
    )
    print(
        f"训练参数[split]：{training['split']}；"
        f"preprocessing={training.get('preprocessing', {})}"
    )
    if has_date_ranges(cfg):
        print(
            "已启用 data.date_ranges：training.split 的日期/末尾天数配置将被忽略"
        )
    seed_everything(int(cfg["training"]["seed"]))
    range_frames: dict[str, pd.DataFrame] | None = None
    if has_date_ranges(cfg):
        range_frames = {
            name: load_data(
                data,
                cfg,
                require_target=True,
                date_range=name,
            )
            for name in ("train", "validation", "test")
        }
        weather = weather_columns(range_frames["train"], cfg)
        for name in ("validation", "test"):
            selected_weather = weather_columns(range_frames[name], cfg)
            if selected_weather != weather:
                raise ValueError(
                    f"{name} 气象列 {selected_weather} 与训练气象列 {weather} 不一致"
                )
        print(
            "按文件日期范围划分数据："
            f"train_rows={len(range_frames['train']):,}, "
            f"validation_rows={len(range_frames['validation']):,}, "
            f"test_rows={len(range_frames['test']):,}"
        )
        frame = None
    else:
        frame = load_data(data, cfg, require_target=True)
        weather = weather_columns(frame, cfg)
    output_dir = Path(cfg["output"]["checkpoint_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "config_resolved.yaml"
    dump_config(cfg, resolved_config_path)
    selected_horizons = _horizons(cfg)
    print(
        f"训练任务开始：horizons={selected_horizons}, weather={weather}, "
        f"checkpoint_dir={output_dir}"
    )
    print(f"解析后配置已保存：{resolved_config_path}")

    metrics, test_predictions = [], []
    for horizon in selected_horizons:
        if range_frames is not None:
            train_frame, names = build_samples(
                range_frames["train"],
                cfg,
                horizon,
                weather,
                require_target=True,
            )
            validation_frame, validation_names = build_samples(
                range_frames["validation"],
                cfg,
                horizon,
                weather,
                require_target=True,
            )
            test_frame, test_names = build_samples(
                range_frames["test"],
                cfg,
                horizon,
                weather,
                require_target=True,
            )
            if validation_names != names or test_names != names:
                raise RuntimeError(
                    f"horizon={horizon} 的训练、验证、测试特征列不一致"
                )
            total_samples = len(train_frame) + len(validation_frame) + len(test_frame)
        else:
            if frame is None:
                raise RuntimeError("内部错误：未加载训练数据")
            samples, names = build_samples(
                frame, cfg, horizon, weather, require_target=True
            )
            train_frame, validation_frame, test_frame = _split(samples, cfg)
            total_samples = len(samples)
        print(
            f"horizon={horizon:02d} 样本构造完成：total={total_samples:,}, "
            f"train={len(train_frame):,}, validation={len(validation_frame):,}, "
            f"test={len(test_frame):,}, features={len(names)}"
        )
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
        result[INTERNAL_PREDICTION_COLUMN] = prediction
        test_predictions.append(result)
        print(
            f"horizon={horizon:02d} 完成：test_rmse={score['rmse']:.6f}, "
            f"test_mae={score['mae']:.6f}"
        )

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(test_predictions, ignore_index=True)
    metrics_path = output_dir / "metrics_by_horizon.csv"
    predictions_path = output_dir / "test_predictions.parquet"
    metadata_path = output_dir / "metadata.json"
    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_parquet(predictions_path, index=False)
    delivery_paths = save_delivery_files(predictions_df, output_dir, cfg)
    metadata = {
        "artifact_version": 1,
        "model_type": "TabM",
        "horizons": selected_horizons,
        "weather_columns": weather,
        "mean_test_rmse": float(metrics_df["test_rmse"].mean()),
        "delivery_file_count": len(delivery_paths),
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"训练指标已保存：{metrics_path.resolve()}")
    print(f"测试预测已保存：{predictions_path.resolve()}")
    print(f"checkpoint 元数据已保存：{metadata_path.resolve()}")
    print(
        f"训练任务完成：mean_test_rmse={metadata['mean_test_rmse']:.6f}, "
        f"checkpoint_dir={output_dir}"
    )
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
        result[INTERNAL_PREDICTION_COLUMN] = _predict_horizon(
            checkpoint_dir, samples, horizon, config
        )
        outputs.append(result)
        print(f"horizon={horizon:02d} 推理完成：rows={len(result):,}")
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["timestamp", "horizon"], ignore_index=True
    )


def predict(
    ckpt_path: str | Path, data: DataInput, config: ConfigInput
) -> pd.DataFrame:
    """Run inference and return a long-form prediction DataFrame."""
    cfg = load_config(config)
    _print_common_parameters("推理", config, cfg)
    checkpoint_dir, model_paths, metadata = _checkpoint(ckpt_path)
    _print_checkpoint_parameters("推理", ckpt_path, checkpoint_dir, model_paths)
    actual_horizons = {
        int(model_path.stem.removeprefix("model_h")) for model_path in model_paths
    }
    required_horizons = expected_horizons(cfg)
    if actual_horizons != required_horizons:
        raise ValueError(
            "predict() 必须加载完整 horizon checkpoint："
            f"expected={sorted(required_horizons)}, actual={sorted(actual_horizons)}"
        )
    print(
        f"推理参数[runtime]：inference_batch_size="
        f"{cfg['training']['inference_batch_size']}, "
        f"prediction_column={cfg['output']['prediction_column']}, "
        f"metadata_horizons={metadata['horizons']}, "
        f"metadata_weather_columns={metadata['weather_columns']}"
    )
    print(
        f"推理任务开始：checkpoint_dir={checkpoint_dir}, "
        f"models={len(model_paths)}"
    )
    frame = load_data(data, cfg, require_target=False)
    columns = cfg["data"]["columns"]
    province_rows = frame[
        frame[columns["station"]].eq(cfg["data"]["province_station"])
    ]
    input_origins = pd.Index(province_rows[columns["timestamp"]].unique())
    if len(input_origins) != 1:
        raise ValueError(
            "predict() 只允许一个起报时刻；"
            f"当前发现 {len(input_origins)} 个：{input_origins.astype(str).tolist()}"
        )
    weather = weather_columns(frame, cfg)
    _validate_weather(metadata, weather)
    result = _predict_loaded_frame(
        checkpoint_dir,
        model_paths,
        frame,
        cfg,
        weather,
        include_target=False,
    )
    origins = pd.Index(result["timestamp"].unique())
    formatted = next(
        iter(delivery_frames(result, cfg, skip_incomplete=False).values())
    )
    print(
        f"推理任务完成：origin={origins[0]}, rows={len(formatted):,}, "
        f"columns={formatted.columns.tolist()}；不保存本地文件"
    )
    return formatted


def evaluate(
    ckpt_path: str | Path, data: DataInput | None, config: ConfigInput
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a checkpoint on labeled data; return (metrics, predictions)."""
    cfg = load_config(config)
    _print_common_parameters("测试", config, cfg)
    checkpoint_dir, model_paths, metadata = _checkpoint(ckpt_path)
    _print_checkpoint_parameters("测试", ckpt_path, checkpoint_dir, model_paths)
    print(
        f"测试参数[runtime]：inference_batch_size="
        f"{cfg['training']['inference_batch_size']}, "
        f"prediction_column={cfg['output']['prediction_column']}, "
        f"metadata_horizons={metadata['horizons']}, "
        f"metadata_weather_columns={metadata['weather_columns']}"
    )
    print(
        f"测试任务开始：checkpoint_dir={checkpoint_dir}, "
        f"models={len(model_paths)}"
    )
    frame = load_data(
        data,
        cfg,
        require_target=True,
        date_range="test",
    )
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
    prediction_col = INTERNAL_PREDICTION_COLUMN
    for horizon, current in predictions.groupby("horizon", sort=True):
        score = _metrics(
            current["target_power"].to_numpy(),
            current[prediction_col].to_numpy(),
        )
        metrics.append({"horizon": int(horizon), "sample_count": len(current), **score})
        print(
            f"horizon={int(horizon):02d} 测试指标：samples={len(current):,}, "
            f"rmse={score['rmse']:.6f}, mae={score['mae']:.6f}"
        )
    print(
        f"评估计算完成：prediction_rows={len(predictions):,}；"
        "evaluate() 本身不保存，test() 会继续生成正式交付文件"
    )
    return pd.DataFrame(metrics), predictions


def test(
    ckpt_path: str | Path, data: DataInput | None, config: ConfigInput
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate, save original-format delivery files, and return delivery frames."""
    cfg = load_config(config)
    checkpoint_dir, _, _ = _checkpoint(ckpt_path)
    metrics, predictions = evaluate(ckpt_path, data, config)
    frames = delivery_frames(predictions, cfg, skip_incomplete=True)
    save_delivery_frames(frames, checkpoint_dir, cfg)
    return metrics, combine_delivery_frames(frames)
