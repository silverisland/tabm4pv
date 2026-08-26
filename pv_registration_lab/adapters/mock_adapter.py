"""Deterministic adapter used only to validate the portable framework."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


MODE_GAIN = {
    "baseline": 0.0,
    "identity": 0.00002,
    "phase_only": 0.0012,
    "seasonal_shift": 0.0017,
    "seasonal_three_point": 0.0020,
    "history_warp": -0.0040,
}


def stable_noise(*parts: object) -> float:
    payload = "|".join(map(str, parts)).encode("utf-8")
    raw = int(hashlib.sha256(payload).hexdigest()[:8], 16)
    return ((raw % 2001) - 1000) / 10_000_000.0


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    station = request["held_out_station"]
    mode = request["mode"]
    seed = request["seed"]
    score = (
        0.906
        + stable_noise("station", station)
        + stable_noise("seed", seed)
        + MODE_GAIN.get(mode, 0.0005)
    )
    payload = {
        "experiment_id": request["experiment_id"],
        "status": "ok",
        "score": score,
        "runtime_seconds": 0.01,
        "audit": {
            "train_row_count": 10000,
            "validation_row_count": 1000,
            "evaluation_row_count": 2000,
            "calibration_row_count": 21 * 96,
            "train_rows_hash": stable_hash("train", station, seed),
            "validation_rows_hash": stable_hash("validation", station, seed),
            "evaluation_rows_hash": stable_hash("evaluation", station, seed),
            "calibration_rows_hash": stable_hash("calibration", station),
            "target_index": 15,
            "weather_index": 15,
            "capacity_map_hash": stable_hash("capacity-map-v1"),
            "model_config_hash": stable_hash("tabm-config-v1"),
            "preprocessing_hash": stable_hash("preprocessing-v1"),
            "evaluation_rows_in_train": 0,
            "calibration_eval_overlap": 0
        },
        "diagnostics": {
            "max_shift_minutes": 0.0 if mode in {"baseline", "identity"} else 15.0,
            "canonical_horizon_min": 3.8,
            "canonical_horizon_mean": 4.0,
            "canonical_horizon_max": 4.2,
        },
        "per_month": {str(month): score for month in range(1, 13)},
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
