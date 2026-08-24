# 省级超短期 TabM 工程版

本目录由原始 `train_general.py` 拆分而来，提供训练、测试和推理三个接口。当前特征包括省级最近 96 点功率、各场站按有效装机容量加权的逐时效气象，以及目标时刻的小时和周期特征。

## 与原版的一致性

默认配置保持原脚本的有效建模逻辑：

- TabM 使用 `LinearReLUEmbeddings`，默认结构为 2 个 block、`d_block=512`、`dropout=0.1`、`k=32` 和 `arch_type=tabm`。
- 每个 horizon 独立训练一个模型，目标按 15000 缩放，预测裁剪到 `[0, 15750]`。
- 特征顺序为 96 点历史功率、装机容量加权气象、目标时刻 hour/hour sin/hour cos。
- 缺失值中位数填充与 `QuantileTransformer` 只在训练集上拟合；默认 quantile 数量、微小噪声和 `subsample=10**9` 与原脚本一致。
- 默认仍从 `/jtdata/products/data/info.csv` 覆盖场站容量；如果传入数据的 `cap_power_on` 已经可信，可将 `data.capacity_csv` 设置为 `null`。

训练、验证和测试通过 `data.date_ranges` 或 `training.split` 划分。

## 安装

在本目录的上一级执行：

```bash
pip install -r province_tabm_engineered/requirements.txt
```

## Python 接口

```python
import pandas as pd

from province_tabm_engineered import predict, test, train

# 训练：省略 data 时读取 config.yaml 的 data.path，并按 date_ranges 划分。
result = train("province_tabm_engineered/config.yaml")
print(result["checkpoint_dir"])
print(result["metrics"])

# 测试：输入必须含 observe_power_future；同时逐起报时刻保存正式交付文件。
metrics, test_deliveries = test(
    result["checkpoint_dir"],
    None,  # 使用同一个 config.data.path，只读取 date_ranges.test
    "province_tabm_engineered/config.yaml",
)

# 推理：只允许一个起报时刻，返回正式交付格式 DataFrame，不保存文件。
input_df = pd.read_parquet("/path/to/inference.parquet")
prediction_df = predict(
    ckpt_path=result["checkpoint_dir"],
    data=input_df,
    config="province_tabm_engineered/config.yaml",
)
```

`predict()` 的返回列与原版单个交付 parquet 完全一致：

| 列 | 含义 |
|---|---|
| `dtime` | 目标时刻 |
| `predict_power_province_guangxi_solar` | 省级功率预测值 |

`predict()` 一般传完整 checkpoint 目录，并且输入数据只允许一个起报时刻。

推理数据不要求 `observe_power_future`，但必须只包含一个省级起报时刻，并包含该时刻的省级行（提供历史功率数组）和对应场站行（提供气象数组及容量）。数组至少要覆盖配置中的 16 个时效。

`test()` 可以处理多个起报时刻。它会在 `checkpoint_dir/forecasts` 下为每个拥有完整 16 个 horizon 的起报时刻生成一个文件：

```text
hw_nuoya_{YYYYMMDDHHMM}_ultra_short_province_guangxi_solar_{date_tag}_tabm_v2.parquet
```

每个文件严格只有 `dtime` 和预测值两列。`test()` 返回的第二个 DataFrame 也只有这两列；多个起报时刻时使用 `forecast_origin` 多级索引区分，不会把该索引写入交付文件。

## 按文件名日期划分数据

训练、验证和测试共用一个 `data.path`。配置 `date_ranges` 后，程序会先解析 `plantid=YYYY-MM-DD.parquet` 中的日期，再选择文件读取；三个范围的 `start/end` 都是闭区间：

```yaml
data:
  path: /path/to/all_parquets
  file_glob: "plantid=*.parquet"
  file_date_regex: "plantid=(\\d{4}-\\d{2}-\\d{2})\\.parquet$"
  date_ranges:
    train:
      start: 2026-06-01
      end: 2026-07-31
    validation:
      start: 2026-08-01
      end: 2026-08-07
    test:
      start: 2026-08-08
      end: 2026-08-15
```

启用后：

- `train(config)` 分别读取三个范围并进行训练、早停验证和最终测试；
- `test(ckpt, None, config)` 只读取 `test` 范围；
- 直接传入 DataFrame 时，使用配置的 timestamp 列按日期执行相同过滤；
- `date_ranges: null` 时继续使用旧的 `training.split` 自动切分。

目录输入会逐文件计算全部时效的加权气象特征，文件处理完后即释放场站级原始数据。

训练和推理可以使用不同设备；只需分别在配置中设置 `model.device` 为 `cpu`、`cuda:0` 或 `auto`。模型结构与特征配置应保持一致。

## Checkpoint 结构

```text
artifacts/tabm_v2/
├── config_resolved.yaml
├── metadata.json
├── metrics_by_horizon.csv
├── models/model_h01.pt ... model_h16.pt
└── preprocessors/preprocessor_h01.joblib ... preprocessor_h16.joblib
```

推理时模型特征名、目标缩放和 horizon 从 checkpoint 读取，配置用于定义输入列、特征构造、设备与输出字段。

## print 运行信息

训练、测试和推理直接使用 Python 原生 `print()` 向标准输出打印运行信息，不使用 `logging` 库或日志封装。输出包括数据规模、设备、horizon 进度和文件的绝对保存路径。训练期间可通过以下参数控制验证集打印频率：

```yaml
training:
  log_every_n_epochs: 10
```

首个 epoch 始终打印；之后每隔指定 epoch 打印一次。模型与预处理器每次保存、加载时都会直接 `print` 完整路径，方便定位产物。

任务启动时会打印 config、数据路径、日期范围、checkpoint、设备、horizon 和主要特征参数；模型保存和加载时打印完整文件路径。

## 测试

```bash
pip install -r province_tabm_engineered/requirements-dev.txt
python -m pytest province_tabm_engineered/tests
```

测试包含原版 v2 特征数值与列顺序回归检查，以及默认 TabM/预处理参数检查。
