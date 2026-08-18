from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import Config


DataInput = pd.DataFrame | str | Path | Sequence[str | Path]


def has_date_ranges(config: Config) -> bool:
    return bool(config["data"].get("date_ranges"))


def _date_bounds(config: Config, range_name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    ranges = config["data"].get("date_ranges")
    if not ranges or range_name not in ranges:
        raise ValueError(f"config.data.date_ranges 缺少范围: {range_name}")
    selected = ranges[range_name]
    return (
        pd.Timestamp(selected["start"]).normalize(),
        pd.Timestamp(selected["end"]).normalize(),
    )


def _filter_paths_by_date(
    paths: list[Path], config: Config, range_name: str
) -> list[Path]:
    start, end = _date_bounds(config, range_name)
    selected: list[Path] = []
    unmatched: list[str] = []
    for path in paths:
        file_date = _filename_date(path, config)
        if file_date is None:
            unmatched.append(path.name)
            continue
        if start <= file_date <= end:
            selected.append(path)
    if unmatched and config["data"].get("strict_file_dates", True):
        raise ValueError(
            "以下 parquet 文件名无法提取日期，请检查 data.file_date_regex："
            f"{unmatched[:10]}"
        )
    if not selected:
        raise FileNotFoundError(
            f"{range_name} 日期范围 [{start.date()}, {end.date()}] 没有匹配文件"
        )
    print(
        f"{range_name} 文件筛选：date_range=[{start.date()}, {end.date()}], "
        f"files={len(selected)}, first={selected[0].name}, last={selected[-1].name}"
    )
    return selected


def _filename_date(path: Path, config: Config) -> pd.Timestamp | None:
    pattern = re.compile(
        config["data"].get(
            "file_date_regex", r"plantid=(\d{4}-\d{2}-\d{2})\.parquet$"
        )
    )
    match = pattern.search(path.name)
    return pd.Timestamp(match.group(1)).normalize() if match else None


def _filter_frame_by_date(
    frame: pd.DataFrame, config: Config, range_name: str
) -> pd.DataFrame:
    start, end = _date_bounds(config, range_name)
    timestamp_col = config["data"]["columns"]["timestamp"]
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
    day = timestamps.dt.normalize()
    selected = frame[(day >= start) & (day <= end)].copy()
    if selected.empty:
        raise ValueError(
            f"DataFrame 在 {range_name} 日期范围 "
            f"[{start.date()}, {end.date()}] 内没有数据"
        )
    print(
        f"{range_name} DataFrame 筛选：date_range=[{start.date()}, {end.date()}], "
        f"rows={len(selected):,}"
    )
    return selected


def _paths(value: str | Path | Sequence[str | Path], file_glob: str) -> list[Path]:
    values = [value] if isinstance(value, (str, Path)) else list(value)
    result: list[Path] = []
    for item in values:
        path = Path(item).expanduser()
        result.extend(sorted(path.glob(file_glob)) if path.is_dir() else [path])
    if not result:
        raise FileNotFoundError(f"没有找到输入数据: {value}")
    return result


def _capacity_mapping(config: Config) -> pd.Series | None:
    data_cfg = config["data"]
    capacity_csv = data_cfg.get("capacity_csv")
    if not capacity_csv:
        return None
    capacity_path = Path(capacity_csv).expanduser().resolve()
    print(f"使用容量表覆盖场站容量：{capacity_path}")
    cap = pd.read_csv(capacity_path)
    station_key = data_cfg.get("capacity_station_column", "plant_pointname")
    value_key = data_cfg.get("capacity_value_column", "GCCAPACITY")
    return cap.drop_duplicates(station_key).set_index(station_key)[value_key]


def _prepare_frame(
    frame: pd.DataFrame,
    config: Config,
    *,
    require_target: bool,
    source_file: str | None,
    capacity_mapping: pd.Series | None,
) -> pd.DataFrame:
    data_cfg = config["data"]
    cols = data_cfg["columns"]
    frame = frame.copy()
    if source_file is None:
        if "source_file" not in frame:
            frame["source_file"] = "<dataframe>"
        source_label = "<dataframe>"
    else:
        frame["source_file"] = source_file
        source_label = source_file

    required = [cols["timestamp"], cols["station"], cols["power_history"]]
    if capacity_mapping is None:
        required.append(cols["capacity"])
    if require_target:
        required.append(cols["power_future"])
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise KeyError(f"输入数据缺少列: {missing}")

    if capacity_mapping is not None:
        # Preserve the original recipe: when a capacity table is configured,
        # station capacities come exclusively from that table.
        frame[cols["capacity"]] = frame[cols["station"]].map(capacity_mapping)

    timestamp_col, station_col = cols["timestamp"], cols["station"]
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="coerce")
    if frame[timestamp_col].isna().any():
        raise ValueError(f"{timestamp_col!r} 包含无效时间: {source_label}")
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
        raise ValueError(f"发现非法 station 标识: {bad[:10]}，source={source_label}")
    return frame


def _validate_file_content_date(
    frame: pd.DataFrame, path: Path, config: Config
) -> None:
    if not config["data"].get("validate_file_content_date", True):
        return
    file_date = _filename_date(path, config)
    if file_date is None:
        return
    timestamp_col = config["data"]["columns"]["timestamp"]
    content_dates = pd.DatetimeIndex(frame[timestamp_col]).normalize().unique()
    mismatched = content_dates[content_dates != file_date]
    if len(mismatched):
        values = [str(value.date()) for value in mismatched[:10]]
        raise ValueError(
            f"文件 {path.name} 的文件名日期为 {file_date.date()}，"
            f"但包含其他起报日期: {values}"
        )


def iter_data_frames(
    data: DataInput | None,
    config: Config,
    *,
    require_target: bool,
    date_range: str | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield validated frames one at a time so callers can release raw station rows."""
    data_cfg = config["data"]
    source = data if data is not None else data_cfg.get("path")
    if source is None:
        raise ValueError("必须通过 data 参数或 data.path 提供数据")
    capacity_mapping = _capacity_mapping(config)

    if isinstance(source, pd.DataFrame):
        frame = source.copy()
        if date_range is not None and has_date_ranges(config):
            frame = _filter_frame_by_date(frame, config, date_range)
        frame = _prepare_frame(
            frame,
            config,
            require_target=require_target,
            source_file=None,
            capacity_mapping=capacity_mapping,
        )
        print(f"已接收 DataFrame：rows={len(frame):,}, columns={len(frame.columns)}")
        yield frame
        del frame
        return

    paths = _paths(source, data_cfg.get("file_glob", "*.parquet"))
    if date_range is not None and has_date_ranges(config):
        paths = _filter_paths_by_date(paths, config, date_range)
    print(f"开始逐文件读取 parquet：files={len(paths)}, source={source}")
    for index, path in enumerate(paths, start=1):
        frame = _prepare_frame(
            pd.read_parquet(path),
            config,
            require_target=require_target,
            source_file=path.name,
            capacity_mapping=capacity_mapping,
        )
        _validate_file_content_date(frame, path, config)
        print(
            f"parquet [{index}/{len(paths)}] 读取并校验完成："
            f"file={path.name}, rows={len(frame):,}"
        )
        yield frame
        del frame


def load_data(
    data: DataInput | None,
    config: Config,
    *,
    require_target: bool,
    date_range: str | None = None,
) -> pd.DataFrame:
    """Load a DataFrame, parquet file, directory, or a sequence of parquet files."""
    frames = list(
        iter_data_frames(
            data,
            config,
            require_target=require_target,
            date_range=date_range,
        )
    )
    frame = pd.concat(frames, ignore_index=True)
    cols = config["data"]["columns"]
    timestamp_col, station_col = cols["timestamp"], cols["station"]
    print(
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
