from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import Config
from .logging_utils import log


DataInput = pd.DataFrame | str | Path | Sequence[str | Path]


def _paths(value: str | Path | Sequence[str | Path], file_glob: str) -> list[Path]:
    values = [value] if isinstance(value, (str, Path)) else list(value)
    result: list[Path] = []
    for item in values:
        path = Path(item).expanduser()
        result.extend(sorted(path.glob(file_glob)) if path.is_dir() else [path])
    if not result:
        raise FileNotFoundError(f"没有找到输入数据: {value}")
    return result


def load_data(
    data: DataInput | None, config: Config, *, require_target: bool
) -> pd.DataFrame:
    """Load a DataFrame, parquet file, directory, or a sequence of parquet files."""
    data_cfg = config["data"]
    source = data if data is not None else data_cfg.get("path")
    if source is None:
        raise ValueError("必须通过 data 参数或 data.path 提供数据")

    if isinstance(source, pd.DataFrame):
        frame = source.copy()
        if "source_file" not in frame:
            frame["source_file"] = "<dataframe>"
        log(f"已接收 DataFrame：rows={len(frame):,}, columns={len(frame.columns)}")
    else:
        frames = []
        paths = _paths(source, data_cfg.get("file_glob", "*.parquet"))
        log(f"开始读取 parquet：files={len(paths)}, source={source}")
        for path in paths:
            item = pd.read_parquet(path)
            item["source_file"] = path.name
            frames.append(item)
        frame = pd.concat(frames, ignore_index=True)
        log(f"parquet 读取完成：rows={len(frame):,}, columns={len(frame.columns)}")

    cols = data_cfg["columns"]
    required = [cols["timestamp"], cols["station"], cols["power_history"]]
    if not data_cfg.get("capacity_csv"):
        required.append(cols["capacity"])
    if require_target:
        required.append(cols["power_future"])
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(f"输入数据缺少列: {missing}")

    capacity_csv = data_cfg.get("capacity_csv")
    if capacity_csv:
        capacity_path = Path(capacity_csv).expanduser().resolve()
        log(f"使用容量表覆盖场站容量：{capacity_path}")
        cap = pd.read_csv(capacity_path)
        station_key = data_cfg.get("capacity_station_column", "plant_pointname")
        value_key = data_cfg.get("capacity_value_column", "GCCAPACITY")
        mapping = cap.drop_duplicates(station_key).set_index(station_key)[value_key]
        # Preserve the original recipe: when a capacity table is configured,
        # station capacities come exclusively from that table.
        frame[cols["capacity"]] = frame[cols["station"]].map(mapping)

    timestamp_col, station_col = cols["timestamp"], cols["station"]
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="coerce")
    if frame[timestamp_col].isna().any():
        raise ValueError(f"{timestamp_col!r} 包含无效时间")
    frame[station_col] = frame[station_col].astype("string").str.strip().str.lower()
    province = data_cfg["province_station"]
    frame.loc[frame[station_col].eq(province), cols["capacity"]] = data_cfg[
        "province_capacity"
    ]
    pattern = re.compile(data_cfg["plant_station_pattern"])
    valid = frame[station_col].eq(province) | frame[station_col].map(
        lambda value: bool(pattern.fullmatch(str(value)))
    )
    if not valid.all():
        bad = frame.loc[~valid, station_col].dropna().unique().tolist()
        raise ValueError(f"发现非法 station 标识: {bad[:10]}")
    log(
        "数据校验完成："
        f"rows={len(frame):,}, stations={frame[station_col].nunique():,}, "
        f"time_range=[{frame[timestamp_col].min()}, {frame[timestamp_col].max()}]"
    )
    return frame


def array_at(value: object, index: int) -> float:
    try:
        values = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return float("nan")
    return float(values[index]) if len(values) > index else float("nan")


def history_values(value: object, length: int) -> np.ndarray:
    try:
        values = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError("省级历史功率数组格式错误") from error
    if len(values) < length:
        raise ValueError(f"历史功率只有 {len(values)} 点，需要至少 {length} 点")
    return values[-length:]
