#!/usr/bin/env python3
"""Evaluate ultra-short-term power forecasts using the 2026 revised metric.

Each input file must be named ``*_index{i}.parquet``.  ``index=0`` is the
15-minute-ahead forecast, ``index=1`` is the 30-minute-ahead forecast, and so
on through ``index=15`` (4 hours ahead).
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


INDEX_RE = re.compile(r"_index(?P<index>\d+)\.parquet$", re.IGNORECASE)
PREDICTION_COLUMN_CANDIDATES = ("prediction", "preidction")
EXPECTED_HORIZONS = 16
EXPECTED_POINTS_PER_DAY = 96


def discover_input_files(pattern: str) -> list[Path]:
    """Return unique input files sorted by forecast index."""
    paths = [Path(item) for item in glob.glob(pattern, recursive=True)]
    paths = [path for path in paths if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"No parquet files matched: {pattern}")

    indexed: list[tuple[int, Path]] = []
    seen: dict[int, Path] = {}
    for path in paths:
        match = INDEX_RE.search(path.name)
        if match is None:
            continue
        index = int(match.group("index"))
        if not 0 <= index < EXPECTED_HORIZONS:
            raise ValueError(
                f"Forecast index must be 0..{EXPECTED_HORIZONS - 1}: {path}"
            )
        if index in seen:
            raise ValueError(
                f"More than one file was found for index={index}: "
                f"{seen[index]} and {path}"
            )
        seen[index] = path
        indexed.append((index, path))

    if not indexed:
        raise ValueError(
            "Matched files do not follow the required '*_index{i}.parquet' naming pattern"
        )
    return [path for _, path in sorted(indexed)]


def _forecast_index(path: Path) -> int:
    match = INDEX_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot determine forecast index from file name: {path}")
    return int(match.group("index"))


def _to_timezone_naive(values: pd.Series, column_name: str) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce")
    if timestamps.isna().any():
        bad_count = int(timestamps.isna().sum())
        raise ValueError(f"Column '{column_name}' contains {bad_count} invalid timestamps")
    if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
        timestamps = timestamps.dt.tz_localize(None)
    return timestamps


def load_forecast_records(
    files: Iterable[Path],
    *,
    timestamp_column: str = "timestamp_win",
    prediction_column: str | None = None,
    groundtruth_column: str = "groundtruth",
) -> pd.DataFrame:
    """Load files and convert issue times to their corresponding target times."""
    frames: list[pd.DataFrame] = []
    for path in files:
        index = _forecast_index(path)
        source = pd.read_parquet(path)

        if prediction_column is None:
            matches = [name for name in PREDICTION_COLUMN_CANDIDATES if name in source]
            if not matches:
                raise KeyError(
                    f"{path} has neither 'prediction' nor the common typo 'preidction'"
                )
            selected_prediction_column = matches[0]
        else:
            selected_prediction_column = prediction_column

        required = {timestamp_column, selected_prediction_column, groundtruth_column}
        missing = required.difference(source.columns)
        if missing:
            raise KeyError(f"{path} is missing columns: {sorted(missing)}")

        issue_time = _to_timezone_naive(source[timestamp_column], timestamp_column)
        target_time = issue_time + pd.to_timedelta((index + 1) * 15, unit="min")
        frame = pd.DataFrame(
            {
                "target_timestamp": target_time,
                "forecast_index": index,
                "prediction": pd.to_numeric(
                    source[selected_prediction_column], errors="coerce"
                ),
                "groundtruth": pd.to_numeric(
                    source[groundtruth_column], errors="coerce"
                ),
                "source_file": str(path),
            }
        )
        duplicate = frame.duplicated(["target_timestamp", "forecast_index"], keep=False)
        if duplicate.any():
            sample = frame.loc[duplicate, "target_timestamp"].iloc[0]
            raise ValueError(
                f"{path} contains duplicate rows for target timestamp {sample}"
            )
        frames.append(frame)

    records = pd.concat(frames, ignore_index=True)
    if records.empty:
        raise ValueError("The matched parquet files contain no rows")
    return records


def _validate_groundtruth(records: pd.DataFrame) -> pd.Series:
    """Return one ground-truth value per target and reject inconsistent copies."""
    available = records.dropna(subset=["groundtruth"])
    if available.empty:
        raise ValueError("No valid groundtruth values were found")

    grouped = available.groupby("target_timestamp")["groundtruth"]
    spread = grouped.max() - grouped.min()
    inconsistent = spread[spread > 1e-6]
    if not inconsistent.empty:
        first = inconsistent.index[0]
        raise ValueError(
            "groundtruth differs between forecast-index files for target timestamp "
            f"{first}"
        )
    return grouped.first()


def _date_bounds(
    records: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    coverage = records.groupby("forecast_index")["target_timestamp"].agg(["min", "max"])
    shared_start = coverage["min"].max()
    shared_end = coverage["max"].min()

    # Without explicit bounds, use complete natural days inside the time range
    # shared by all available horizons.  This prevents shifted forecast horizons
    # from creating artificial missing values on the first and last day.
    automatic_start = shared_start.normalize()
    if shared_start != automatic_start:
        automatic_start += pd.Timedelta(days=1)
    automatic_end = shared_end.normalize()
    if shared_end < automatic_end + pd.Timedelta(hours=23, minutes=45):
        automatic_end -= pd.Timedelta(days=1)

    start = (
        pd.Timestamp(start_date).normalize()
        if start_date
        else automatic_start
    )
    end = (
        pd.Timestamp(end_date).normalize()
        if end_date
        else automatic_end
    )
    if end < start:
        raise ValueError(
            "No complete natural day is shared by the input forecast horizons. "
            "Provide --start-date and --end-date to evaluate an explicit interval."
        )
    return start, end


def calculate_daily_metrics(
    records: pd.DataFrame,
    *,
    capacity: float,
    start_date: str | None = None,
    end_date: str | None = None,
    missing_normalized_error: float = 1.0,
) -> pd.DataFrame:
    """Calculate revised daily accuracy, padding every day to 96 x 16 values."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if missing_normalized_error < 0:
        raise ValueError("missing_normalized_error must be non-negative")

    records = records.copy()
    records["target_timestamp"] = _to_timezone_naive(
        records["target_timestamp"], "target_timestamp"
    )
    bad_index = ~records["forecast_index"].between(0, EXPECTED_HORIZONS - 1)
    if bad_index.any():
        raise ValueError("forecast_index must be in the range 0..15")

    truth_by_target = _validate_groundtruth(records)
    start, end = _date_bounds(records, start_date, end_date)
    days = pd.date_range(start, end, freq="D")
    target_times = pd.date_range(
        start, end + pd.Timedelta(days=1), freq="15min", inclusive="left"
    )
    expected = pd.MultiIndex.from_product(
        [target_times, range(EXPECTED_HORIZONS)],
        names=["target_timestamp", "forecast_index"],
    ).to_frame(index=False)

    values = records[["target_timestamp", "forecast_index", "prediction"]]
    grid = expected.merge(
        values, on=["target_timestamp", "forecast_index"], how="left", validate="one_to_one"
    )
    grid["groundtruth"] = grid["target_timestamp"].map(truth_by_target)
    valid = grid["prediction"].notna() & grid["groundtruth"].notna()
    denominator = np.maximum(grid["groundtruth"], 0.2 * capacity)
    grid["normalized_error"] = missing_normalized_error
    grid.loc[valid, "normalized_error"] = (
        (grid.loc[valid, "groundtruth"] - grid.loc[valid, "prediction"]).abs()
        / denominator.loc[valid]
    )

    per_target = (
        grid.groupby("target_timestamp", as_index=False)
        .agg(
            mean_normalized_error=("normalized_error", "mean"),
            available_forecasts=("prediction", "count"),
            groundtruth_available=("groundtruth", lambda values: int(values.notna().any())),
        )
    )
    per_target["date"] = per_target["target_timestamp"].dt.normalize()
    daily = (
        per_target.groupby("date", as_index=False)
        .agg(
            mean_normalized_error=("mean_normalized_error", "mean"),
            groundtruth_points=("groundtruth_available", "sum"),
            available_forecasts=("available_forecasts", "sum"),
        )
        .set_index("date")
        .reindex(days)
        .rename_axis("date")
        .reset_index()
    )
    daily["expected_target_points"] = EXPECTED_POINTS_PER_DAY
    daily["expected_forecasts"] = EXPECTED_POINTS_PER_DAY * EXPECTED_HORIZONS
    daily["accuracy"] = 1.0 - daily["mean_normalized_error"]
    daily["forecast_coverage"] = (
        daily["available_forecasts"] / daily["expected_forecasts"]
    )
    daily["complete_day"] = (
        (daily["groundtruth_points"] == daily["expected_target_points"])
        & (daily["available_forecasts"] == daily["expected_forecasts"])
    )
    return daily[
        [
            "date",
            "accuracy",
            "mean_normalized_error",
            "groundtruth_points",
            "expected_target_points",
            "available_forecasts",
            "expected_forecasts",
            "forecast_coverage",
            "complete_day",
        ]
    ]


def calculate_monthly_metrics(
    daily: pd.DataFrame, *, complete_days_only: bool = False
) -> pd.DataFrame:
    """Average daily metric values by calendar month."""
    source = daily[daily["complete_day"]].copy() if complete_days_only else daily.copy()
    if source.empty:
        raise ValueError("No daily rows are eligible for monthly aggregation")
    source["month"] = source["date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        source.groupby("month", as_index=False)
        .agg(
            monthly_average_accuracy=("accuracy", "mean"),
            monthly_average_normalized_error=("mean_normalized_error", "mean"),
            days_included=("date", "count"),
            complete_days=("complete_day", "sum"),
            available_forecasts=("available_forecasts", "sum"),
            expected_forecasts=("expected_forecasts", "sum"),
        )
    )
    monthly["forecast_coverage"] = (
        monthly["available_forecasts"] / monthly["expected_forecasts"]
    )
    return monthly


def _autosize_sheet(worksheet) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill

    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for column_cells in worksheet.columns:
        width = min(
            42,
            max(10, max(len(str(cell.value or "")) for cell in column_cells) + 2),
        )
        worksheet.column_dimensions[column_cells[0].column_letter].width = width


def write_excel_report(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    output_path: Path,
    *,
    capacity: float,
    input_files: list[Path],
    missing_normalized_error: float,
    complete_days_only: bool,
) -> None:
    """Write daily, monthly, and calculation-description worksheets."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    daily_export = daily.rename(
        columns={
            "date": "日期",
            "accuracy": "超短期预测准确率",
            "mean_normalized_error": "平均归一化误差",
            "groundtruth_points": "有效真值点数",
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
    discovered_indices = sorted(_forecast_index(path) for path in input_files)
    notes = pd.DataFrame(
        [
            ("指标公式", "ACC = (1 - (1/96) * sum_i(E_i)) * 100%"),
            (
                "时刻误差",
                "E_i = (1/16) * sum_j(abs(P_M,i - P_hat_i,j) / max(P_M,i, 0.2*C))",
            ),
            ("装机容量 C", capacity),
            ("提前量映射", "index=0 对应 +15分钟；index=15 对应 +4小时"),
            ("缺失预测处理", f"归一化误差记为 {missing_normalized_error}"),
            ("月平均口径", "仅完整日" if complete_days_only else "全部日指标的算术平均"),
            ("识别到的 index", ", ".join(map(str, discovered_indices))),
            ("输入文件数", len(input_files)),
        ],
        columns=["项目", "说明/数值"],
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        daily_export.to_excel(writer, sheet_name="日指标", index=False)
        monthly_export.to_excel(writer, sheet_name="月平均", index=False)
        notes.to_excel(writer, sheet_name="计算说明", index=False)

        daily_sheet = writer.book["日指标"]
        monthly_sheet = writer.book["月平均"]
        notes_sheet = writer.book["计算说明"]
        for sheet in (daily_sheet, monthly_sheet, notes_sheet):
            _autosize_sheet(sheet)

        daily_sheet.column_dimensions["A"].width = 13
        monthly_sheet.column_dimensions["A"].width = 11
        for cell in daily_sheet["A"][1:]:
            cell.number_format = "yyyy-mm-dd"
        for cell in monthly_sheet["A"][1:]:
            cell.number_format = "yyyy-mm"
        for column in ("B", "H"):
            for cell in daily_sheet[column][1:]:
                cell.number_format = "0.0000%"
        for column in ("B", "H"):
            for cell in monthly_sheet[column][1:]:
                cell.number_format = "0.0000%"
        for cell in daily_sheet["C"][1:]:
            cell.number_format = "0.000000"
        for cell in monthly_sheet["C"][1:]:
            cell.number_format = "0.000000"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate the revised ultra-short-term forecast metric"
    )
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Quoted glob such as '/data/xxxxx_index*.parquet'",
    )
    parser.add_argument("--capacity", required=True, type=float, help="Installed capacity C")
    parser.add_argument(
        "--output", type=Path, default=Path("forecast_metrics.xlsx"), help="Output Excel"
    )
    parser.add_argument("--timestamp-column", default="timestamp_win")
    parser.add_argument(
        "--prediction-column",
        default=None,
        help="Defaults to auto-detecting 'prediction' or 'preidction'",
    )
    parser.add_argument("--groundtruth-column", default="groundtruth")
    parser.add_argument("--start-date", help="Optional first target date, YYYY-MM-DD")
    parser.add_argument("--end-date", help="Optional last target date, YYYY-MM-DD")
    parser.add_argument(
        "--missing-normalized-error",
        type=float,
        default=1.0,
        help="Error assigned to a missing prediction; default 1 means zero accuracy",
    )
    parser.add_argument(
        "--monthly-complete-days-only",
        action="store_true",
        help="Use only days with all 96*16 forecasts in the monthly average",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    files = discover_input_files(args.input_glob)
    records = load_forecast_records(
        files,
        timestamp_column=args.timestamp_column,
        prediction_column=args.prediction_column,
        groundtruth_column=args.groundtruth_column,
    )
    daily = calculate_daily_metrics(
        records,
        capacity=args.capacity,
        start_date=args.start_date,
        end_date=args.end_date,
        missing_normalized_error=args.missing_normalized_error,
    )
    monthly = calculate_monthly_metrics(
        daily, complete_days_only=args.monthly_complete_days_only
    )
    write_excel_report(
        daily,
        monthly,
        args.output,
        capacity=args.capacity,
        input_files=files,
        missing_normalized_error=args.missing_normalized_error,
        complete_days_only=args.monthly_complete_days_only,
    )
    print(f"Wrote {len(daily)} daily rows and {len(monthly)} monthly rows to {args.output}")


if __name__ == "__main__":
    main()
