"""Export existing training checkpoints to config + unified safetensors.

Run once in the training environment, where sklearn/joblib are available.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from safetensors.torch import save_file

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from province_tabm_engineered.backbone import ProvinceTabMBackbone
    from province_tabm_engineered.config import load_config
    from province_tabm_engineered.data import _capacity_mapping
else:
    from .backbone import ProvinceTabMBackbone
    from .config import load_config
    from .data import _capacity_mapping


def _json_default(value):
    # YAML loads unquoted dates as date/datetime objects.
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def export_checkpoint(checkpoint_dir: str | Path, output_dir: str | Path,
                      config=None, *, overwrite: bool = False) -> Path:
    """Package fitted preprocessors and weights without refitting or training.

    config defaults to the saved config_resolved.yaml. Feature rules always
    come from metadata when available, matching the existing inference API.
    Training passes overwrite=True to refresh its two deployment files.
    """
    source = Path(checkpoint_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not overwrite and destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"导出目录必须为空，避免覆盖已有部署文件：{destination}")
    cfg = load_config(config if config is not None else source / "config_resolved.yaml")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if "features" in metadata:
        cfg["features"] = deepcopy(metadata["features"])
    if not ("history" in cfg["features"] or "future" in cfg["features"]):
        raise ValueError("此导出器需要当前 history/future 特征配置版本")
    horizons = sorted(int(h) for h in metadata["horizons"])
    if horizons != list(range(1, int(cfg["features"]["n_horizons"]) + 1)):
        raise ValueError("导出需要完整的 horizon checkpoint")

    # Snapshot capacity data to avoid relying on training-machine file paths.
    capacity = _capacity_mapping(cfg)
    if capacity is not None:
        cfg["data"]["capacity_mapping"] = {
            str(key): float(value) if np.isfinite(value) else None
            for key, value in capacity.items()
        }
    cfg["data"]["capacity_csv"] = None
    cfg["deployment_version"] = 1
    cfg["horizon_specs"] = []
    state = {}
    print(f"导出 checkpoint：source={source}, destination={destination}, horizons={horizons}")
    for horizon in horizons:
        model_path = source / "models" / f"model_h{horizon:02d}.pt"
        preprocessor_path = source / "preprocessors" / f"preprocessor_h{horizon:02d}.joblib"
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        preprocessor = joblib.load(preprocessor_path)
        imputer = preprocessor.get("imputer")
        quantile = preprocessor["quantile_transformer"]
        names = list(payload["feature_names"])
        if (int(payload["horizon"]) != horizon
                or int(payload["n_num_features"]) != len(names)
                or quantile.quantiles_.shape[1] != len(names)):
            raise ValueError(f"horizon={horizon} checkpoint 的特征维数或编号不一致")
        if imputer is not None and (
            imputer.strategy != "median" or imputer.add_indicator
            or not np.isnan(imputer.missing_values)
            or len(imputer.statistics_) != len(names)
            or not np.isfinite(imputer.statistics_).all()
        ):
            raise ValueError("仅支持当前训练流程的 median Imputer（保留所有特征且无 indicator）")
        if not np.isfinite(quantile.quantiles_).all():
            raise ValueError(f"horizon={horizon} 分位点含非有限值")
        cfg["horizon_specs"].append({
            "horizon": horizon,
            "feature_names": names,
            "architecture": dict(payload.get("architecture", {})),
            "target_scale": float(payload["target_scale"]),
            "preprocessing": {
                "n_quantiles": int(quantile.n_quantiles_),
                "use_imputer": imputer is not None,
                "output_distribution": quantile.output_distribution,
            },
        })
        key = f"h{horizon:02d}"
        for name, tensor in payload["model_state_dict"].items():
            state[f"models.{key}.{name}"] = tensor.detach().cpu().contiguous().clone()
        prefix = f"preprocessors.{key}."
        state[prefix + "medians"] = torch.tensor(
            imputer.statistics_ if imputer is not None else [], dtype=torch.float32
        )
        state[prefix + "quantiles"] = torch.tensor(quantile.quantiles_, dtype=torch.float64).contiguous()
        state[prefix + "references"] = torch.tensor(quantile.references_, dtype=torch.float64)
        print(f"horizon={horizon:02d}：model={model_path}, preprocessor={preprocessor_path}, use_imputer={imputer is not None}")

    # Validate the exact host pipeline before writing deployment artifacts.
    model = ProvinceTabMBackbone(cfg)
    model.load_state_dict(state, strict=True)
    config_text = json.dumps(
        cfg, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default
    )
    destination.mkdir(parents=True, exist_ok=True)
    save_file(state, str(destination / "model.safetensors"))
    (destination / "model_config.json").write_text(config_text, encoding="utf-8")
    print(f"导出完成：{destination / 'model.safetensors'}；{destination / 'model_config.json'}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出省级 TabM 到推理 backbone")
    parser.add_argument("--checkpoint", required=True, help="原训练 checkpoint 目录")
    parser.add_argument("--output", required=True, help="新的部署目录")
    parser.add_argument("--config", help="可选配置；默认使用 checkpoint/config_resolved.yaml")
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output, args.config)
