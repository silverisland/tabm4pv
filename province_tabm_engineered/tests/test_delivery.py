from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from province_tabm_engineered.delivery import (
    combine_delivery_frames,
    delivery_filename,
    delivery_frames,
    save_delivery_frames,
)


def _config(output_dir: Path) -> dict:
    return {
        "data": {"province_station": "province_guangxi_solar"},
        "features": {"n_horizons": 2},
        "output": {
            "prediction_column": "predict_power_province_guangxi_solar",
            "forecast_dir": str(output_dir),
            "date_tag": "20260818",
            "method_version": "tabm_v2",
        },
    }


def test_delivery_format_and_filename_match_original(tmp_path: Path):
    config = _config(tmp_path)
    origin = pd.Timestamp("2026-08-18 12:30:00")
    predictions = pd.DataFrame(
        {
            "timestamp": [origin, origin],
            "target_timestamp": [
                origin + pd.Timedelta(minutes=15),
                origin + pd.Timedelta(minutes=30),
            ],
            "horizon": [1, 2],
            "prediction_power": [10.5, 20.5],
        }
    )

    frames = delivery_frames(predictions, config, skip_incomplete=False)
    frame = frames[origin]
    assert frame.columns.tolist() == [
        "dtime",
        "predict_power_province_guangxi_solar",
    ]
    assert frame["predict_power_province_guangxi_solar"].dtype == np.float32
    assert delivery_filename(origin, config) == (
        "hw_nuoya_202608181230_ultra_short_province_guangxi_solar_"
        "20260818_tabm_v2.parquet"
    )

    paths = save_delivery_frames(frames, tmp_path, config)
    assert len(paths) == 1
    saved = pd.read_parquet(paths[0])
    pd.testing.assert_frame_equal(saved, frame)
    pd.testing.assert_frame_equal(combine_delivery_frames(frames), frame)
