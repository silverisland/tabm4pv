from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


INTERNAL_PREDICTION_COLUMN = "prediction_power"


def expected_horizons(config: Config) -> set[int]:
    return set(range(1, int(config["features"]["n_horizons"]) + 1))


def delivery_frame(group: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Convert one complete forecast origin to the original two-column format."""
    ordered = group.sort_values("horizon")
    actual = set(ordered["horizon"].astype(int))
    expected = expected_horizons(config)
    if len(ordered) != len(expected) or actual != expected:
        raise ValueError(
            f"交付结果时效不完整：expected={sorted(expected)}, actual={sorted(actual)}"
        )
    return pd.DataFrame(
        {
            "dtime": pd.to_datetime(ordered["target_timestamp"]).to_numpy(),
            config["output"]["prediction_column"]: ordered[
                INTERNAL_PREDICTION_COLUMN
            ].to_numpy(dtype=np.float32),
        }
    )


def delivery_frames(
    predictions: pd.DataFrame,
    config: Config,
    *,
    skip_incomplete: bool,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Build one original-format DataFrame for each forecast origin."""
    result: dict[pd.Timestamp, pd.DataFrame] = {}
    for origin, group in predictions.groupby("timestamp", sort=True):
        origin_timestamp = pd.Timestamp(origin)
        try:
            result[origin_timestamp] = delivery_frame(group, config)
        except ValueError as error:
            if not skip_incomplete:
                raise ValueError(f"起报时刻 {origin_timestamp}：{error}") from error
            print(f"跳过不完整的交付起报时刻 {origin_timestamp}：{error}")
    return result


def forecast_directory(checkpoint_dir: Path, config: Config) -> Path:
    configured = config["output"].get("forecast_dir")
    if configured:
        return Path(configured).expanduser().resolve()
    return checkpoint_dir / "forecasts"


def delivery_filename(origin: pd.Timestamp, config: Config) -> str:
    output = config["output"]
    date_tag = output.get("date_tag", "auto")
    if not date_tag or str(date_tag).lower() == "auto":
        date_tag = pd.Timestamp.now().strftime("%Y%m%d")
    method_version = output.get("method_version", "tabm_v2")
    province = config["data"]["province_station"]
    origin_text = origin.strftime("%Y%m%d%H%M")
    return (
        f"hw_nuoya_{origin_text}_ultra_short_{province}_"
        f"{date_tag}_{method_version}.parquet"
    )


def save_delivery_frames(
    frames: dict[pd.Timestamp, pd.DataFrame],
    checkpoint_dir: Path,
    config: Config,
) -> list[Path]:
    """Save already-formatted delivery frames without rebuilding them."""
    output_dir = forecast_directory(checkpoint_dir, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for origin, frame in frames.items():
        path = output_dir / delivery_filename(origin, config)
        frame.to_parquet(path, index=False)
        paths.append(path)
        print(f"交付预测已保存：{path.resolve()}")
    print(f"交付文件生成完成：count={len(paths)}, directory={output_dir.resolve()}")
    return paths


def combine_delivery_frames(
    frames: dict[pd.Timestamp, pd.DataFrame],
) -> pd.DataFrame:
    """Combine test deliveries while keeping the two delivery columns unchanged."""
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return next(iter(frames.values())).reset_index(drop=True)
    return pd.concat(frames, names=["forecast_origin", "row"])
