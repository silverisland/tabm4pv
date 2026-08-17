# 省级超短期 TabM 工程版

本目录由原始 `train_general.py` 拆分而来，提供训练、测试和推理三个稳定接口。原始版本的 v2 特征保持不变：省级最近 96 点功率、各场站按有效装机容量加权的逐时效气象、容量覆盖率和目标时刻周期特征。

## 与原版的一致性

默认配置保持原脚本的有效建模逻辑：

- TabM 使用 `LinearReLUEmbeddings`，默认结构为 2 个 block、`d_block=512`、`dropout=0.1`、`k=32` 和 `arch_type=tabm`。
- 每个 horizon 独立训练一个模型，目标按 15000 缩放，预测裁剪到 `[0, 15750]`。
- 特征顺序保持为 96 点历史功率、装机容量加权气象、气象容量覆盖率、目标时刻 hour sin/cos。
- 缺失值中位数填充与 `QuantileTransformer` 只在训练集上拟合；默认 quantile 数量、微小噪声和 `subsample=10**9` 与原脚本一致。
- 默认仍从 `/jtdata/products/data/info.csv` 覆盖场站容量；如果传入数据的 `cap_power_on` 已经可信，可将 `data.capacity_csv` 设置为 `null`。

唯一有意调整的是数据切分：原脚本中真正执行的硬编码条件会让验证集包含测试集，并且忽略已经计算好的日期边界。工程版改为 `training.split` 配置，保证训练、验证、测试互斥；这不改变模型结构和特征工程。

## 安装

在本目录的上一级执行：

```bash
pip install -r province_tabm_engineered/requirements.txt
```

## Python 接口

```python
import pandas as pd

from province_tabm_engineered import predict, test, train

# 训练：data 可省略，此时读取 config.yaml 的 data.path。
result = train("province_tabm_engineered/config.yaml", data="/path/to/train_parquets")
print(result["checkpoint_dir"])
print(result["metrics"])

# 测试：输入必须含 observe_power_future。
metrics, test_predictions = test(
    result["checkpoint_dir"],
    "/path/to/test.parquet",
    "province_tabm_engineered/config.yaml",
)

# 推理：严格按要求输入 checkpoint 路径、数据和 config，返回 DataFrame。
input_df = pd.read_parquet("/path/to/inference.parquet")
prediction_df = predict(
    ckpt_path=result["checkpoint_dir"],
    data=input_df,
    config="province_tabm_engineered/config.yaml",
)
```

`predict()` 的返回列为：

| 列 | 含义 |
|---|---|
| `timestamp` | 预测起报时刻 |
| `target_timestamp` | 目标时刻 |
| `horizon` | 第几个 15 分钟时效 |
| `predict_power_province_guangxi_solar` | 省级功率预测值 |

`ckpt_path` 可以是完整 checkpoint 目录，也可以是某一个 `models/model_hXX.pt`；传单模型文件时只返回该时效。

推理数据不要求 `observe_power_future`，但每个起报时刻必须包含一条省级行（提供历史功率数组）和对应场站行（提供气象数组及容量）。数组至少要覆盖配置中的 16 个时效。

训练和推理可以使用不同设备；只需分别在配置中设置 `model.device` 为 `cpu`、`cuda:0` 或 `auto`。模型结构与特征配置应保持一致。

## 命令行接口

```bash
python -m province_tabm_engineered.cli train \
  --config province_tabm_engineered/config.yaml \
  --data /path/to/train_data

python -m province_tabm_engineered.cli test \
  --config province_tabm_engineered/config.yaml \
  --checkpoint artifacts/tabm_v2 \
  --data /path/to/test.parquet \
  --output evaluation_predictions.parquet

python -m province_tabm_engineered.cli predict \
  --config province_tabm_engineered/config.yaml \
  --checkpoint artifacts/tabm_v2 \
  --data /path/to/inference.parquet \
  --output predictions.parquet
```

## Checkpoint 结构

```text
artifacts/tabm_v2/
├── config_resolved.yaml
├── metadata.json
├── metrics_by_horizon.csv
├── test_predictions.parquet
├── models/model_h01.pt ... model_h16.pt
└── preprocessors/preprocessor_h01.joblib ... preprocessor_h16.joblib
```

推理时模型特征名、目标缩放和 horizon 从 checkpoint 读取，配置用于定义输入列、特征构造、设备与输出字段。气象列与训练元数据不一致时会直接报错，避免静默产生错误预测。

## 测试

```bash
pip install -r province_tabm_engineered/requirements-dev.txt
python -m pytest province_tabm_engineered/tests
```

测试包含原版 v2 特征数值与列顺序回归检查，以及默认 TabM/预处理参数检查。
