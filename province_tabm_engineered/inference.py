from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__:
    from .api import _checkpoint, _print_config
    from .config import ConfigInput, load_config
    from .features import build_feature_data
    from .model import infer_array, load_one, resolve_device, transform
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from province_tabm_engineered.api import _checkpoint, _print_config
    from province_tabm_engineered.config import ConfigInput, load_config
    from province_tabm_engineered.features import build_feature_data
    from province_tabm_engineered.model import (
        infer_array,
        load_one,
        resolve_device,
        transform,
    )


class Model:
    """Load all horizon models once and run batched provincial inference."""

    def __init__(self, config: ConfigInput, ckpt_path: str | Path):
        self.config = load_config(config)
        _print_config("推理模型初始化", config, self.config)
        self.checkpoint_dir, model_paths, _ = _checkpoint(ckpt_path)
        self.device = resolve_device(self.config["model"].get("device", "auto"))

        model_paths = sorted(
            model_paths,
            key=lambda path: int(path.stem.removeprefix("model_h")),
        )
        self.horizons = [
            int(path.stem.removeprefix("model_h")) for path in model_paths
        ]
        expected = list(range(1, int(self.config["features"]["n_horizons"]) + 1))
        if self.horizons != expected:
            raise ValueError(
                f"inference() 需要完整 horizon：expected={expected}, "
                f"actual={self.horizons}"
            )

        self.loaded_models: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
        for model_path in model_paths:
            self.loaded_models.append(
                load_one(model_path, self.checkpoint_dir, self.device)
            )
        print(
            f"推理模型初始化完成：device={self.device}, "
            f"horizons={self.horizons}, models={len(self.loaded_models)}"
        )

    def inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict all origins and return one length-16 array per origin."""
        capacity_col = self.config["data"]["columns"]["capacity"]
        if capacity_col not in df.columns:
            raise ValueError(f"输入 DataFrame 缺少容量列：{capacity_col}")

        df = df.copy()

        def capacity_float(value: Any) -> float:
            if isinstance(value, (np.ndarray, list, tuple)):
                values = np.asarray(value).reshape(-1)
                if len(values) == 0:
                    raise ValueError(f"{capacity_col} 不能是空数组")
                value = values[0]
            try:
                return float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{capacity_col} 必须是数值或非空数组") from error

        df[capacity_col] = df[capacity_col].map(capacity_float)
        frame, _, _ = build_feature_data(df, self.config, self.horizons)
        batch_size = int(self.config["training"]["inference_batch_size"])
        lower, upper = map(float, self.config["model"]["prediction_clip"])
        horizon_predictions = []

        for horizon, (model, preprocessor, payload) in zip(
            self.horizons, self.loaded_models
        ):
            values = frame[payload["feature_names"]].to_numpy(dtype=np.float32)
            values = transform(preprocessor, values)
            prediction = infer_array(
                model,
                values,
                device=self.device,
                batch_size=batch_size,
                target_scale=float(payload["target_scale"]),
            )
            horizon_predictions.append(np.clip(prediction, lower, upper))
            print(f"horizon={horizon:02d} 推理完成：rows={len(frame):,}")

        prediction_matrix = np.column_stack(horizon_predictions).astype(
            np.float32, copy=False
        )
        result = pd.DataFrame(
            {
                "timestamp_win": pd.to_datetime(frame["timestamp"]).to_numpy(),
                "station": [self.config["data"]["province_station"]] * len(frame),
                "observe_power_predict": [row.copy() for row in prediction_matrix],
            }
        )
        print(
            f"批量推理完成：origins={len(result):,}, "
            f"prediction_length={prediction_matrix.shape[1]}"
        )
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="省级超短期批量推理示例")
    parser.add_argument("--config", required=True, help="config.yaml 路径")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 目录")
    parser.add_argument("--data", required=True, help="输入 parquet 路径")
    args = parser.parse_args()

    input_df = pd.read_parquet(args.data)
    model = Model(args.config, args.checkpoint)
    result = model.inference(input_df)
    print(result)
    print(result["observe_power_predict"].map(len))
