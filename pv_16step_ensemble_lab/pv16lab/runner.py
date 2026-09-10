from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import dump_config, load_config
from .data import load_dataset
from .delivery import save_ds_outputs
from .features import build_features
from .metrics import evaluate
from .models import train_and_predict
from .report import write_report
from .split import build_split


def run_experiment(config_path: str | Path) -> Path:
    config = load_config(config_path)
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, output_dir / "run_config.yaml")

    print("[1/6] Loading and auditing data")
    dataset = load_dataset(config)
    pd.DataFrame({"timestamp_win": dataset.missing_origins}).to_csv(
        output_dir / "missing_timestamps.csv", index=False
    )
    print(
        f"station={dataset.station}, rows={len(dataset.frame):,}, "
        f"range={dataset.origins.min()} -> {dataset.origins.max()}"
    )

    print("[2/6] Building leakage-safe features and chronological split")
    features = build_features(dataset, config)
    split = build_split(dataset.origins, config)
    print(split.summary.to_string(index=False))

    print("[3/6] Training component, correction and fusion models")
    result = train_and_predict(dataset, features, split, config, output_dir / "models")

    print("[4/6] Calculating horizon, segment, regime, daily and monthly metrics")
    evaluation = evaluate(dataset, split, result, config)
    evaluation.predictions.to_parquet(output_dir / "predictions.parquet", index=False)

    print("[5/6] Writing DS files and Excel report")
    ds_paths = save_ds_outputs(dataset, split, result, output_dir, config)
    report_path = output_dir / "evaluation_report.xlsx"
    write_report(report_path, dataset, split, result, evaluation, config)

    manifest = {
        "project": "PV 16-Step Ensemble Lab",
        "version": "0.1.0",
        "created_at": datetime.now().astimezone().isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "station": dataset.station,
        "data_start": dataset.origins.min().isoformat(),
        "data_end": dataset.origins.max().isoformat(),
        "row_count": len(dataset.frame),
        "models": list(result.predictions),
        "truth_policy": "index0_by_target_timestamp",
        "ds_files": [str(path) for path in ds_paths],
        "report": str(report_path),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"[6/6] Complete: {output_dir.resolve()}")
    return output_dir
