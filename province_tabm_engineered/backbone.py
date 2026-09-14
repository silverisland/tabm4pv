"""Standalone nn.Module inference; no sklearn, joblib or checkpoint I/O."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn

from .architecture import make_model
from .features import build_feature_data


class TorchPreprocessor(nn.Module):
    """Frozen median imputation and sklearn-compatible quantile transformation.

    Buffers retain fitted float64 landmarks. Interpolation is bidirectional for
    repeated quantiles; intermediate probabilities round to float32 exactly as
    in the training pipeline. Only forward transformation is needed here.
    """

    def __init__(self, n_features: int, n_quantiles: int, use_imputer: bool,
                 output_distribution: str = "normal"):
        super().__init__()
        if output_distribution not in ("normal", "uniform"):
            raise ValueError(f"不支持的分位数分布：{output_distribution}")
        self.output_distribution = output_distribution
        self.register_buffer("medians", torch.zeros(n_features if use_imputer else 0))
        self.register_buffer("quantiles", torch.zeros(n_quantiles, n_features, dtype=torch.float64))
        self.register_buffer("references", torch.zeros(n_quantiles, dtype=torch.float64))

    @staticmethod
    def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        # Features are batched; last dimension holds samples / quantile points.
        if xp.shape[1] == 1:
            return fp[0].expand_as(x)
        index = torch.searchsorted(xp.contiguous(), x.contiguous(), right=True)
        left = (index - 1).clamp(0, xp.shape[1] - 2)
        right = left + 1
        low, high = xp.gather(1, left), xp.gather(1, right)
        width = high - low
        fraction = (x - low) / torch.where(width == 0, 1.0, width)
        result = fp[left] + fraction * (fp[right] - fp[left])
        result = torch.where(x < xp[:, :1], fp[0], result)
        return torch.where(x >= xp[:, -1:], fp[-1], result)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.medians.numel():
            x = torch.where(torch.isnan(x), self.medians, x)
        if not torch.isfinite(x).all():
            raise ValueError("预处理后仍有 NaN/Inf；未启用 Imputer 时须在上游完成缺失值处理")
        points = self.quantiles.T
        query = x.T.double()
        forward = self._interp(query, points, self.references)
        backward = self._interp(-query, -points.flip(1), -self.references.flip(0))
        probability = ((forward - backward) * 0.5).T.float()
        if self.output_distribution == "normal":
            lower = (x - 1e-7).double() < self.quantiles[0]
            upper = (x + 1e-7).double() > self.quantiles[-1]
        else:
            lower = x.double() == self.quantiles[0]
            upper = x.double() == self.quantiles[-1]
        probability = torch.where(upper, 1.0, probability)
        probability = torch.where(lower, 0.0, probability)
        if self.output_distribution == "uniform":
            return probability
        # sklearn uses norm.ppf(1e-7 - np.spacing(1)) and its upper counterpart.
        return torch.special.ndtri(probability.double()).float().clamp(
            -5.199337582605575, 5.19933758270342
        )


class ProvinceTabMBackbone(nn.Module):
    """Build on CPU, load a unified state_dict, then call .to(device).eval()."""

    def __init__(self, model_config: dict):
        super().__init__()
        self.config = deepcopy(model_config)
        if self.config.get("deployment_version") != 1:
            raise ValueError("请使用 export_checkpoint 导出的 model_config.yaml")
        self.specs = self.config["horizon_specs"]
        self.horizons = [int(spec["horizon"]) for spec in self.specs]
        expected = list(range(1, int(self.config["features"]["n_horizons"]) + 1))
        if self.horizons != expected:
            raise ValueError(f"部署模型需要完整且有序的 horizon：{expected}")
        self.models = nn.ModuleDict()
        self.preprocessors = nn.ModuleDict()
        for spec in self.specs:
            key = f"h{spec['horizon']:02d}"
            n_features = len(spec["feature_names"])
            self.models[key] = make_model(n_features, torch.device("cpu"), spec["architecture"])
            self.preprocessors[key] = TorchPreprocessor(n_features, **spec["preprocessing"])
        self.batch_size = int(self.config["training"]["inference_batch_size"])
        if self.batch_size < 1:
            raise ValueError("inference_batch_size 必须为正整数")
        self.eval()
        print(f"构建 ProvinceTabMBackbone：horizons={self.horizons}；等待 load_state_dict 加载权重和预处理参数")

    def _to_dataframe(
        self, data: pd.DataFrame | Mapping[str, torch.Tensor | list[str]]
    ) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if not isinstance(data, Mapping):
            raise TypeError("forward() 输入必须是 DataFrame 或 dict[str, Tensor | list[str]]")

        names = self.config["data"]["columns"]
        timestamp_col, station_col = names["timestamp"], names["station"]
        if timestamp_col not in data or station_col not in data:
            raise ValueError(f"输入 dict 必须包含 {timestamp_col} 和 {station_col}")

        columns = {}
        row_count = None
        for name, value in data.items():
            if name == station_col:
                if not isinstance(value, (list, tuple)) or not all(
                    isinstance(station, str) for station in value
                ):
                    raise TypeError(f"{station_col} 必须是 list[str]")
                column = list(value)
            else:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{name} 必须是 torch.Tensor")
                tensor = value.detach().cpu()
                if tensor.ndim == 0:
                    raise ValueError(f"{name} 至少需要 batch 维度")
                if name != timestamp_col and tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                if name == timestamp_col:
                    if tensor.ndim == 2 and tensor.shape[1] == 1:
                        tensor = tensor[:, 0]
                    if tensor.ndim != 1 or tensor.dtype != torch.int64:
                        raise TypeError(f"{timestamp_col} 必须是形状 [B] 或 [B, 1] 的 int64 Tensor")
                    unit = self.config["data"].get("tensor_timestamp_unit", "ns")
                    column = pd.to_datetime(tensor.numpy(), unit=unit, errors="raise")
                elif tensor.ndim == 1:
                    column = tensor.numpy()
                else:
                    column = [row.numpy().copy() for row in tensor]

            if row_count is None:
                row_count = len(column)
            elif len(column) != row_count:
                raise ValueError(
                    f"输入列长度不一致：expected={row_count}, {name}={len(column)}"
                )
            columns[name] = column
        return pd.DataFrame(columns)

    @torch.inference_mode()
    def forward(
        self, data: pd.DataFrame | Mapping[str, torch.Tensor | list[str]]
    ) -> pd.DataFrame:
        """Predict from a DataFrame or a column-name-to-tensor batch dict."""
        df = self._to_dataframe(data)
        capacity_col = self.config["data"]["columns"]["capacity"]
        if capacity_col not in df:
            raise ValueError(f"输入 DataFrame 缺少容量列：{capacity_col}")

        def capacity_float(value):
            if isinstance(value, (np.ndarray, list, tuple)):
                value = np.asarray(value).reshape(-1)
                if not len(value):
                    raise ValueError(f"{capacity_col} 不能是空数组")
                value = value[0]
            return float(value)

        df[capacity_col] = df[capacity_col].map(capacity_float)
        # Inference must not depend on optional labels (including null targets).
        df = df.drop(columns=[self.config["data"]["columns"]["power_future"]], errors="ignore")
        frame, columns, _ = build_feature_data(df, self.config, self.horizons)
        lower, upper = map(float, self.config["model"]["prediction_clip"])
        predictions = []
        for spec in self.specs:
            horizon = spec["horizon"]
            key = f"h{horizon:02d}"
            if columns[horizon] != spec["feature_names"]:
                raise ValueError(f"horizon={horizon} 特征规则与 checkpoint 的列顺序不一致")
            model = self.models[key]
            parameter = next(model.parameters())
            values = frame[spec["feature_names"]].to_numpy(dtype=np.float32)
            batches = []
            for start in range(0, len(values), self.batch_size):
                x = torch.as_tensor(values[start:start + self.batch_size], device=parameter.device)
                x = self.preprocessors[key](x).to(dtype=parameter.dtype)
                prediction = model(x, None).squeeze(-1).float().mean(dim=1)
                batches.append((prediction * float(spec["target_scale"])).clamp(lower, upper).cpu().numpy())
            predictions.append(np.concatenate(batches) if batches else np.empty(0, dtype=np.float32))
            print(f"horizon={horizon:02d} 推理完成：rows={len(frame):,}, device={parameter.device}")
        matrix = np.column_stack(predictions).astype(np.float32, copy=False)
        return pd.DataFrame({
            "timestamp_win": pd.to_datetime(frame["timestamp"]).to_numpy(
                dtype="datetime64[ns]"
            ),
            "station": [self.config["data"]["province_station"]] * len(frame),
            "observe_power_predict": [row.copy() for row in matrix],
        })

    def inference(
        self, data: pd.DataFrame | Mapping[str, torch.Tensor | list[str]]
    ) -> pd.DataFrame:
        return self(data)


def build_model(model_name: str, model_config: dict) -> nn.Module:
    """Registration example for the host inference framework."""
    if model_name != "province_tabm":
        raise ValueError(f"不支持的 model_name：{model_name}")
    return ProvinceTabMBackbone(model_config)
