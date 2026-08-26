# Private TabM Adapter Contract

Only `adapters/project_adapter.py` needs project-specific integration. The
controller writes a request JSON and expects one aggregate result JSON.

## Request

Important fields:

```json
{
  "experiment_id": "...",
  "stage": "quick_ablation",
  "mode": "phase_only",
  "seed": 0,
  "implementation_hash": "...",
  "training_stations": ["..."],
  "held_out_station": "...",
  "target_station": "...",
  "calibration_days": 21,
  "final_test": false,
  "hypothesis": "..."
}
```

For pseudo-OOD evaluation, train TabM only on `training_stations`. The held-out
station may use only its calibration subset to estimate registration. Evaluate
on a disjoint held-out period. Do not include held-out labels in TabM training.

## Current private target identity

The supplied data contract defines one physical target with two raw names:

```text
雅砻江          2024 target history
雅砻江解放站    2025 target evaluation
```

Canonicalize both to `雅砻江`, but preserve `station_raw`. Read capacities from
`/home/ma-user/work/tabm_multi_stations/station_info.csv` using `plantid` as a
string and numeric `GCCAPACITY`. Neither target raw name is present, so apply
the configured canonical capacity override `雅砻江: 465.0`.

For the sealed final request, `target_history_role=train_and_registration`
means the 2024 `雅砻江` segment participates in TabM training and registration;
the 2025 `雅砻江解放站` segment is evaluation-only. All available source-station
dates are allowed by the requested offline protocol, including 2025--2026.
Report this as offline multi-station transfer, not strict chronological
backtesting.

Modes have these minimum meanings:

- `baseline`: exact private TabM baseline, with no registration code path.
- `identity`: the candidate pipeline with a strict identity mapping.
- `phase_only`: physical history, physical target/weather, plus canonical time
  coordinates and canonical horizon.
- `seasonal_shift`: season-specific low-degree phase shift; physical history by
  default.
- `seasonal_three_point`: season-specific morning/noon/evening monotone warp.
- `history_warp`: explicitly experimental resampling of history; target and
  weather remain at the physical requested time.

## Result

Write aggregate data only:

```json
{
  "experiment_id": "...",
  "status": "ok",
  "score": 0.908,
  "runtime_seconds": 123.4,
  "audit": {
    "train_row_count": 100000,
    "validation_row_count": 10000,
    "evaluation_row_count": 20000,
    "calibration_row_count": 2016,
    "train_rows_hash": "sha256...",
    "validation_rows_hash": "sha256...",
    "evaluation_rows_hash": "sha256...",
    "calibration_rows_hash": "sha256...",
    "target_index": 15,
    "weather_index": 15,
    "capacity_map_hash": "sha256...",
    "model_config_hash": "sha256...",
    "preprocessing_hash": "sha256...",
    "evaluation_rows_in_train": 0,
    "calibration_eval_overlap": 0
  },
  "diagnostics": {
    "max_shift_minutes": 15.0,
    "min_local_slope": 0.92,
    "max_local_slope": 1.08,
    "roundtrip_rmse_capacity": 0.0002,
    "canonical_horizon_min": 3.8,
    "canonical_horizon_mean": 4.0,
    "canonical_horizon_max": 4.2
  },
  "per_month": {"1": 0.90, "2": 0.91}
}
```

The row hashes must be deterministic hashes of stable row identifiers, not of
the raw rows themselves. Use the same ordering and hashing procedure for every
mode. `target_index` and `weather_index` are the physical future-array indices;
they must match. The capacity, model, and preprocessing hashes must describe
the effective configuration. Baseline and identity must return identical audit
fingerprints. The controller rejects missing fingerprints and nonzero overlap.

Do not return raw predictions, ground truth, timestamps, power arrays, rows, or
data paths. Keep detailed private artifacts in an environment-local directory
outside the result payload if policy allows them.

Use the same score definition, train/validation logic, model hyperparameters,
random seed behavior, clipping, and numerical embeddings for baseline and
identity. The identity audit is invalid if anything else changes.

When `history_warp` is used, pass a real `history_end_timestamp` and retain
more than the final 96 points. Registration resamples a canonical 24-hour
window back onto physical observations, so extra interpolation margin is
required, especially across midnight. Never reinterpret a rolling 96-point
window as a midnight-to-midnight daily curve.
