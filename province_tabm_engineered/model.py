from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import rtdl_num_embeddings
import sklearn.impute
import sklearn.preprocessing
import tabm
import torch
import torch.nn as nn

from .config import Config


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"配置要求使用 {value}，但 CUDA 不可用")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 2)


def make_model(
    n_features: int, device: torch.device, architecture: dict | None = None
) -> torch.nn.Module:
    embeddings = rtdl_num_embeddings.LinearReLUEmbeddings(n_features)
    return tabm.TabM.make(
        n_num_features=n_features,
        cat_cardinalities=[],
        d_out=1,
        num_embeddings=embeddings,
        **(architecture or {}),
    ).to(device)


def fit_preprocessor(
    x_train: np.ndarray, seed: int, config: Config
) -> tuple[object, object, np.ndarray]:
    imputer = sklearn.impute.SimpleImputer(strategy="median", keep_empty_features=True)
    imputed = imputer.fit_transform(x_train).astype(np.float32)
    preprocessing = config["training"].get("preprocessing", {})
    noise = np.random.default_rng(seed).normal(
        0.0, float(preprocessing.get("noise_std", 1e-5)), imputed.shape
    ).astype(np.float32)
    n_quantiles = min(
        max(
            min(
                len(imputed) // int(preprocessing.get("samples_per_quantile", 30)),
                int(preprocessing.get("max_quantiles", 1000)),
            ),
            int(preprocessing.get("min_quantiles", 10)),
        ),
        len(imputed),
    )
    transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        subsample=int(preprocessing.get("quantile_subsample", 10**9)),
        random_state=seed,
    ).fit(imputed + noise)
    return imputer, transformer, transformer.transform(imputed).astype(np.float32)


def transform(preprocessor: dict[str, Any], values: np.ndarray) -> np.ndarray:
    imputed = preprocessor["imputer"].transform(values).astype(np.float32)
    return preprocessor["quantile_transformer"].transform(imputed).astype(np.float32)


@torch.inference_mode()
def infer_array(
    model: torch.nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    target_scale: float,
) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(x, device=device)
    indices = torch.arange(len(tensor), device=device)
    outputs = []
    for batch in indices.split(batch_size):
        outputs.append(model(tensor[batch], None).squeeze(-1).float())
    if not outputs:
        return np.empty(0, dtype=np.float32)
    return torch.cat(outputs).cpu().numpy().mean(axis=1) * target_scale


def train_one(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_names: list[str],
    horizon: int,
    config: Config,
    checkpoint_dir: Path,
) -> dict[str, float | int]:
    train_cfg, model_cfg = config["training"], config["model"]
    seed = int(train_cfg["seed"])
    device = resolve_device(model_cfg.get("device", "auto"))
    x_train_raw = train_frame[feature_names].to_numpy(dtype=np.float32)
    x_val_raw = validation_frame[feature_names].to_numpy(dtype=np.float32)
    y_train = np.array(train_frame["target_power"], dtype=np.float32, copy=True)
    y_val = validation_frame["target_power"].to_numpy(dtype=np.float32)
    if not len(train_frame) or not len(validation_frame):
        raise ValueError(f"horizon={horizon} 的训练集或验证集为空")

    imputer, transformer, x_train = fit_preprocessor(
        x_train_raw, seed + horizon, config
    )
    preprocessor = {"imputer": imputer, "quantile_transformer": transformer}
    x_val = transform(preprocessor, x_val_raw)
    x_train_t = torch.as_tensor(x_train, device=device)
    y_train_t = torch.as_tensor(y_train, device=device) / float(
        model_cfg["target_scale"]
    )

    torch.manual_seed(seed + 2 + horizon)
    architecture = dict(model_cfg.get("architecture", {}))
    model = make_model(len(feature_names), device, architecture)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_state = deepcopy(model.state_dict())
    best_epoch, best_rmse = -1, float("inf")
    patience = int(train_cfg["early_stopping_patience"])
    batch_size = int(train_cfg["batch_size"])
    target_scale = float(model_cfg["target_scale"])
    lower, upper = map(float, model_cfg["prediction_clip"])

    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        for batch in torch.randperm(len(x_train_t), device=device).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_train_t[batch], None).squeeze(-1).float()
            target = y_train_t[batch].repeat_interleave(model.backbone.k)
            loss = nn.functional.mse_loss(prediction.flatten(0, 1), target)
            loss.backward()
            clip = train_cfg.get("gradient_clipping_norm")
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            optimizer.step()

        validation_prediction = np.clip(
            infer_array(
                model,
                x_val,
                device=device,
                batch_size=int(train_cfg["inference_batch_size"]),
                target_scale=target_scale,
            ),
            lower,
            upper,
        )
        rmse = float(np.sqrt(np.mean((validation_prediction - y_val) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch
            best_state = deepcopy(model.state_dict())
            patience = int(train_cfg["early_stopping_patience"])
        else:
            patience -= 1
            if patience <= 0:
                break

    checkpoint_dir.joinpath("models").mkdir(parents=True, exist_ok=True)
    checkpoint_dir.joinpath("preprocessors").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "n_num_features": len(feature_names),
            "feature_names": feature_names,
            "horizon": horizon,
            "target_scale": target_scale,
            "best_epoch": best_epoch,
            "architecture": architecture,
        },
        checkpoint_dir / "models" / f"model_h{horizon:02d}.pt",
    )
    joblib.dump(
        preprocessor,
        checkpoint_dir / "preprocessors" / f"preprocessor_h{horizon:02d}.joblib",
    )
    return {"best_epoch": best_epoch, "validation_rmse": best_rmse}


def load_one(
    model_path: Path, checkpoint_dir: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    payload = torch.load(model_path, map_location=device, weights_only=True)
    model = make_model(
        int(payload["n_num_features"]),
        device,
        dict(payload.get("architecture", {})),
    )
    model.load_state_dict(payload["model_state_dict"])
    horizon = int(payload["horizon"])
    preprocessor = joblib.load(
        checkpoint_dir / "preprocessors" / f"preprocessor_h{horizon:02d}.joblib"
    )
    return model, preprocessor, payload
