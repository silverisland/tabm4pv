from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, ConfigInput, load_config
from .data import DataInput, array_at, iter_data_frames
from .delivery import DEFAULT_EVALUATION_FILENAME


_ORIGIN_PATTERN = re.compile(r"hw_nuoya_(\d{12})_ultra_short_")
_POINTS_PER_DAY = 96


def _parquet_files(path: str | Path, file_glob: str = "*.parquet") -> list[Path]:
    source = Path(path).expanduser().resolve()
    files = sorted(source.glob(file_glob)) if source.is_dir() else [source]
    files = [file for file in files if file.is_file()]
    if not files:
        raise FileNotFoundError(f"没有找到 parquet 文件：{source}")
    return files


def load_saved_predictions(path: str | Path, config: Config) -> pd.DataFrame:
    """Read the evaluation table, falling back to one-file-per-origin results."""
    input_path = Path(path).expanduser().resolve()
    aggregate_name = config["output"].get(
        "evaluation_filename", DEFAULT_EVALUATION_FILENAME
    )
    aggregate = input_path / aggregate_name
    files = (
        [aggregate]
        if input_path.is_dir() and aggregate.is_file()
        else _parquet_files(path)
    )
    if files == [aggregate]:
        print(f"检测到指标汇总文件，优先读取：{aggregate}")
    time_column = "dtime"
    value_column = config["output"]["prediction_column"]
    horizon_count = int(config.get("evaluation", {}).get("official_horizons", 16))
    minutes = int(config["features"]["minutes_per_point"])
    parts: list[pd.DataFrame] = []

    for file in files:
        source = pd.read_parquet(file)
        missing = {time_column, value_column}.difference(source.columns)
        if missing:
            raise KeyError(f"预测文件 {file} 缺少列：{sorted(missing)}")
        current = pd.DataFrame(
            {
                "target_timestamp": pd.to_datetime(
                    source[time_column], errors="coerce"
                ),
                "prediction": pd.to_numeric(source[value_column], errors="coerce"),
            }
        )
        if current["target_timestamp"].isna().any():
            raise ValueError(f"预测文件 {file} 存在无效 dtime")

        if "horizon" in source:
            raw_horizon = pd.to_numeric(source["horizon"], errors="coerce")
            horizon_values = raw_horizon.to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            rounded_horizon = np.rint(horizon_values)
            if not np.isfinite(horizon_values).all() or not np.allclose(
                horizon_values, rounded_horizon
            ):
                raise ValueError(f"指标汇总文件 {file} 存在无效 horizon")
            current["horizon"] = rounded_horizon.astype(np.int64)
            if "forecast_origin" in source:
                origin = pd.to_datetime(source["forecast_origin"], errors="coerce")
                offset = (
                    (current["target_timestamp"] - origin)
                    / pd.Timedelta(minutes=minutes)
                ).to_numpy(dtype=float)
                if origin.isna().any() or not np.allclose(
                    offset, current["horizon"].to_numpy(dtype=np.float64)
                ):
                    raise ValueError(f"指标汇总文件 {file} 的时间与 horizon 不对齐")
            current = current.sort_values(
                ["target_timestamp", "horizon"], ignore_index=True
            )
            parts.append(current[current["horizon"].between(1, horizon_count)])
            continue

        current = current.sort_values("target_timestamp", ignore_index=True)
        if current["target_timestamp"].duplicated().any():
            raise ValueError(f"预测文件 {file} 存在重复 dtime")

        match = _ORIGIN_PATTERN.search(file.name)
        if match:
            origin = pd.to_datetime(match.group(1), format="%Y%m%d%H%M")
            offset = (
                (current["target_timestamp"] - origin) / pd.Timedelta(minutes=minutes)
            ).to_numpy(dtype=float)
            horizon = np.rint(offset).astype(int)
            if not np.allclose(offset, horizon):
                raise ValueError(f"预测文件 {file} 的 dtime 与起报时刻不对齐")
            current["horizon"] = horizon
        else:
            expected = pd.date_range(
                current["target_timestamp"].iloc[0],
                periods=len(current),
                freq=f"{minutes}min",
            )
            if len(current) < horizon_count or not current["target_timestamp"].equals(
                pd.Series(expected)
            ):
                raise ValueError(
                    f"无法从文件名识别起报时刻，且文件不是完整连续预测：{file}"
                )
            current["horizon"] = np.arange(1, len(current) + 1)

        parts.append(current[current["horizon"].between(1, horizon_count)])

    result = pd.concat(parts, ignore_index=True)
    duplicate = result.duplicated(["target_timestamp", "horizon"], keep=False)
    if duplicate.any():
        row = result.loc[duplicate].iloc[0]
        raise ValueError(
            "预测结果重复："
            f"dtime={row['target_timestamp']}, horizon={int(row['horizon'])}"
        )
    print(
        f"预测结果读取完成：path={input_path}, "
        f"files={len(files)}, rows={len(result):,}"
    )
    return result


def load_available_power(data: DataInput, config: Config) -> pd.DataFrame:
    """Extract one available-power value per target time from normal model input."""
    data_config = deepcopy(config)
    # Metric extraction only needs the province row; do not require a station capacity CSV.
    data_config["data"]["capacity_csv"] = None
    data_config["data"]["capacity_mapping"] = None
    names = data_config["data"]["columns"]
    horizon_count = int(
        data_config.get("evaluation", {}).get("official_horizons", 16)
    )
    minutes = int(data_config["features"]["minutes_per_point"])
    parts: list[pd.DataFrame] = []

    for frame in iter_data_frames(data, data_config):
        province = (
            frame[frame[names["station"]].eq(data_config["data"]["province_station"])]
            .drop_duplicates(names["timestamp"], keep="last")
            .sort_values(names["timestamp"])
        )
        if names["power_future"] not in province:
            raise KeyError(f"可用功率数据缺少列：{names['power_future']}")
        for horizon in range(1, horizon_count + 1):
            parts.append(
                pd.DataFrame(
                    {
                        "target_timestamp": province[names["timestamp"]]
                        + pd.Timedelta(minutes=horizon * minutes),
                        "horizon": horizon,
                        "available_power": province[names["power_future"]].map(
                            lambda value, index=horizon - 1: array_at(value, index)
                        ),
                    }
                )
            )

    if not parts:
        raise ValueError("可用功率数据中没有省级记录")
    candidates = pd.concat(parts, ignore_index=True).dropna(subset=["available_power"])
    if candidates.empty:
        raise ValueError("没有提取到有效可用功率")

    tolerance = float(
        config.get("evaluation", {}).get("groundtruth_tolerance", 1e-6)
    )
    spread = candidates.groupby("target_timestamp")["available_power"].agg(
        lambda values: float(values.max() - values.min())
    )
    conflicts = spread[spread > tolerance]
    if not conflicts.empty:
        print(
            "可用功率提示：同一目标时刻在不同 horizon 中数值不一致，"
            f"按最短 horizon 取值，conflicts={len(conflicts):,}"
        )

    # The shortest horizon is closest to the target time and is used as the canonical truth.
    result = (
        candidates.sort_values(["target_timestamp", "horizon"])
        .drop_duplicates("target_timestamp", keep="first")
        [["target_timestamp", "available_power"]]
        .reset_index(drop=True)
    )
    print(
        f"可用功率读取完成：path={Path(data).expanduser().resolve()}, "
        f"points={len(result):,}"
    )
    return result


def _complete_date_bounds(
    predictions: pd.DataFrame, horizon_count: int, minutes: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    # Recover issue times so even a completely missing horizon can be scored as missing.
    origins = predictions["target_timestamp"] - pd.to_timedelta(
        predictions["horizon"] * minutes, unit="min"
    )
    shared_start = origins.min() + pd.Timedelta(minutes=horizon_count * minutes)
    shared_end = origins.max() + pd.Timedelta(minutes=minutes)
    start = shared_start.normalize()
    if shared_start > start:
        start += pd.Timedelta(days=1)
    end = shared_end.normalize()
    if shared_end < end + pd.Timedelta(hours=23, minutes=45):
        end -= pd.Timedelta(days=1)
    if end < start:
        raise ValueError(
            f"预测结果中没有{horizon_count}个 horizon 共同覆盖的完整自然日"
        )
    return start, end


def calculate_official_metrics(
    predictions: pd.DataFrame,
    available_power: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate revised official daily accuracy and its calendar-month average."""
    evaluation = config.get("evaluation", {})
    horizon_count = int(evaluation.get("official_horizons", 16))
    missing_error = float(evaluation.get("missing_normalized_error", 1.0))
    capacity = float(config["data"]["province_capacity"])
    floor_ratio = float(evaluation.get("capacity_floor_ratio", 0.2))
    minutes = int(config["features"]["minutes_per_point"])
    if capacity <= 0 or floor_ratio < 0 or missing_error < 0:
        raise ValueError("装机容量必须大于0，分母比例和缺失误差不能小于0")

    start, end = _complete_date_bounds(predictions, horizon_count, minutes)
    days = pd.date_range(start, end, freq="D")
    target_times = pd.date_range(
        start,
        end + pd.Timedelta(days=1),
        freq=f"{minutes}min",
        inclusive="left",
    )
    expected = pd.MultiIndex.from_product(
        [target_times, range(1, horizon_count + 1)],
        names=["target_timestamp", "horizon"],
    ).to_frame(index=False)
    grid = expected.merge(
        predictions,
        on=["target_timestamp", "horizon"],
        how="left",
        validate="one_to_one",
    ).merge(
        available_power,
        on="target_timestamp",
        how="left",
        validate="many_to_one",
    )
    grid["normalized_error"] = missing_error
    valid = grid["prediction"].notna() & grid["available_power"].notna()
    denominator = np.maximum(
        grid.loc[valid, "available_power"], floor_ratio * capacity
    )
    grid.loc[valid, "normalized_error"] = (
        (grid.loc[valid, "prediction"] - grid.loc[valid, "available_power"]).abs()
        / denominator
    )
    grid["date"] = grid["target_timestamp"].dt.normalize()

    daily = (
        grid.groupby("date", as_index=False)
        .agg(
            mean_normalized_error=("normalized_error", "mean"),
            available_power_points=(
                "available_power",
                lambda values: int(values.notna().sum() / horizon_count),
            ),
            available_forecasts=("prediction", "count"),
        )
        .set_index("date")
        .reindex(days)
        .rename_axis("date")
        .reset_index()
    )
    daily["expected_target_points"] = _POINTS_PER_DAY
    daily["expected_forecasts"] = _POINTS_PER_DAY * horizon_count
    truth_complete = daily["available_power_points"].eq(_POINTS_PER_DAY)
    daily["accuracy"] = 1.0 - daily["mean_normalized_error"]
    daily.loc[~truth_complete, ["accuracy", "mean_normalized_error"]] = np.nan
    daily["forecast_coverage"] = (
        daily["available_forecasts"] / daily["expected_forecasts"]
    )
    daily["complete_day"] = truth_complete & daily["available_forecasts"].eq(
        daily["expected_forecasts"]
    )
    daily = daily[
        [
            "date",
            "accuracy",
            "mean_normalized_error",
            "available_power_points",
            "expected_target_points",
            "available_forecasts",
            "expected_forecasts",
            "forecast_coverage",
            "complete_day",
        ]
    ]

    valid_daily = daily.dropna(subset=["accuracy"]).copy()
    if valid_daily.empty:
        raise ValueError("评价区间内没有可用功率完整的自然日")
    valid_daily["month"] = valid_daily["date"].dt.to_period("M").dt.to_timestamp()
    monthly = valid_daily.groupby("month", as_index=False).agg(
        monthly_average_accuracy=("accuracy", "mean"),
        monthly_average_normalized_error=("mean_normalized_error", "mean"),
        days_included=("date", "count"),
        complete_days=("complete_day", "sum"),
        available_forecasts=("available_forecasts", "sum"),
        expected_forecasts=("expected_forecasts", "sum"),
    )
    monthly["forecast_coverage"] = (
        monthly["available_forecasts"] / monthly["expected_forecasts"]
    )
    return daily, monthly


def write_metric_report(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    output_path: str | Path,
    config: Config,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    daily_export = daily.rename(
        columns={
            "date": "日期",
            "accuracy": "超短期预测准确率",
            "mean_normalized_error": "平均归一化误差",
            "available_power_points": "有效可用功率点数",
            "expected_target_points": "应有时刻数",
            "available_forecasts": "有效预测数",
            "expected_forecasts": "应有预测数",
            "forecast_coverage": "预测覆盖率",
            "complete_day": "是否完整日",
        }
    )
    monthly_export = monthly.rename(
        columns={
            "month": "月份",
            "monthly_average_accuracy": "月平均准确率",
            "monthly_average_normalized_error": "月平均归一化误差",
            "days_included": "纳入天数",
            "complete_days": "完整天数",
            "available_forecasts": "有效预测数",
            "expected_forecasts": "应有预测数",
            "forecast_coverage": "预测覆盖率",
        }
    )
    evaluation = config.get("evaluation", {})
    notes = pd.DataFrame(
        [
            ("日指标", "1 - 96个时刻、16个horizon的归一化绝对误差均值"),
            ("月指标", "有效日指标的算术平均"),
            ("装机容量", config["data"]["province_capacity"]),
            ("分母下限比例", evaluation.get("capacity_floor_ratio", 0.2)),
            ("官方horizon数", evaluation.get("official_horizons", 16)),
            ("缺失预测归一化误差", evaluation.get("missing_normalized_error", 1.0)),
            ("真值选择", "同一目标时刻优先使用最短horizon对应的可用功率"),
        ],
        columns=["项目", "说明/数值"],
    )
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        daily_export.to_excel(writer, sheet_name="日指标", index=False)
        monthly_export.to_excel(writer, sheet_name="月平均", index=False)
        notes.to_excel(writer, sheet_name="计算说明", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cells in sheet.columns:
                width = min(
                    42,
                    max(10, max(len(str(cell.value or "")) for cell in cells) + 2),
                )
                sheet.column_dimensions[cells[0].column_letter].width = width
        for column in ("B", "H"):
            for cell in writer.book["日指标"][column][1:]:
                cell.number_format = "0.0000%"
        for column in ("B", "G"):
            for cell in writer.book["月平均"][column][1:]:
                cell.number_format = "0.0000%"
    return output


def evaluate_saved_predictions(
    prediction_path: str | Path,
    available_power_path: str | Path,
    config: ConfigInput,
    output_path: str | Path,
) -> Path:
    """Independent metric interface used by the CLI."""
    cfg = load_config(config)
    print(
        "指标计算参数："
        f"predictions={Path(prediction_path).expanduser().resolve()}, "
        f"available_power={Path(available_power_path).expanduser().resolve()}, "
        f"output={Path(output_path).expanduser().resolve()}"
    )
    print(
        "指标计算参数："
        f"capacity={cfg['data']['province_capacity']}, "
        f"capacity_floor_ratio={cfg.get('evaluation', {}).get('capacity_floor_ratio', 0.2)}, "
        f"official_horizons={cfg.get('evaluation', {}).get('official_horizons', 16)}"
    )
    predictions = load_saved_predictions(prediction_path, cfg)
    truth = load_available_power(available_power_path, cfg)
    daily, monthly = calculate_official_metrics(predictions, truth, cfg)
    output = write_metric_report(daily, monthly, output_path, cfg)
    print(
        f"指标计算完成：daily={len(daily)}, monthly={len(monthly)}, output={output}"
    )
    return output
