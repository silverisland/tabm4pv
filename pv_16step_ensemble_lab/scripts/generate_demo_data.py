#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _signals(times: pd.DatetimeIndex, capacity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hour = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    day = times.dayofyear.to_numpy()
    clear = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))
    cloud = np.clip(0.72 + 0.18 * np.sin(day * 0.7) + 0.08 * np.cos(hour * 1.3), 0.25, 1.0)
    ghi = 950.0 * clear * cloud
    temperature = 18.0 + 9.0 * clear + 3.0 * np.sin(day / 5.0)
    power = capacity * np.clip(ghi / 1050.0, 0.0, 1.0)
    return power.astype(np.float32), ghi.astype(np.float32), temperature.astype(np.float32)


def generate(path: Path, days: int) -> None:
    capacity = 10_000.0
    origins = pd.date_range("2025-01-01", periods=days * 96, freq="15min")
    rows = []
    for origin_index, origin in enumerate(origins):
        history_time = pd.date_range(end=origin, periods=672, freq="15min")
        future_time = pd.date_range(start=origin + pd.Timedelta(minutes=15), periods=192, freq="15min")
        history_power, history_ghi, history_temp = _signals(history_time, capacity)
        future_power, future_ghi, future_temp = _signals(future_time, capacity)
        forecast_bias = 1.0 + 0.04 * np.sin(origin_index / 31.0)
        rows.append({
            "station": "DEMO_PV_STATION",
            "timestamp_win": origin,
            "cap_power_on": capacity,
            "observe_power": history_power,
            "observe_power_future": future_power,
            "GHI_SOLARGIS": history_ghi,
            "TEMP_SOLARGIS": history_temp,
            "GHI_SOLARGIS_predict": (future_ghi * forecast_bias).astype(np.float32),
            "TEMP_SOLARGIS_predict": (future_temp + 0.5).astype(np.float32),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"Wrote {len(rows):,} rows to {path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/demo_station.parquet"))
    parser.add_argument("--days", type=int, default=14)
    args = parser.parse_args()
    generate(args.output, args.days)


if __name__ == "__main__":
    main()

