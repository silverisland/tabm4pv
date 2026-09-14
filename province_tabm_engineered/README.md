# 省级超短期 TabM 工程版

本目录由原始 `train_general.py` 拆分而来，提供训练、测试和推理接口。序列特征由原始字段、索引和容量加权规则配置，一份 config 生成所有 horizon 的输入列。

## 与原版的一致性

模型结构与预处理沿用原版，特征和训练指标由配置决定：

- TabM 使用 `LinearReLUEmbeddings`，默认结构为 2 个 block、`d_block=512`、`dropout=0.1`、`k=32` 和 `arch_type=tabm`。
- 每个 horizon 独立训练一个模型，目标按 15000 缩放，预测裁剪到 `[0, 15750]`。
- 默认功率取 `observe_power[-96:-2]`（94点），排除最新两个可能被治理填补的点；另加入逐 horizon 的历史、未来加权气象和时间特征。
- 缺失值中位数填充与 `QuantileTransformer` 只在训练集上拟合；默认 quantile 数量、微小噪声和 `subsample=10**9` 与原脚本一致。
- 默认仍从 `/jtdata/products/data/info.csv` 覆盖场站容量；如果传入数据的 `cap_power_on` 已经可信，可将 `data.capacity_csv` 设置为 `null`。

训练、验证和测试通过 `data.date_ranges` 或 `training.split` 划分。

## 配置序列特征

```yaml
features:
  n_horizons: 16
  minutes_per_point: 15
  history:
    observe_power:
      indices: {start: -96, stop: -2}
      capacity_weighted: false
    GHI_SOLARGIS:
      index: {base: -96, horizon_offset: true}
      capacity_weighted: true
  future:
    GHI_SOLARGIS_predict:
      index: {base: -1, horizon_offset: true}
      capacity_weighted: true
  time: [hour, hour_sin, hour_cos]
```

- `index: -3`：固定取倒数第3点；也支持 `index: {base: -3, horizon_offset: false}`。
- `indices: [-96, -80, -4, -3]`：按列表顺序取点。
- `indices: {start: -96, stop: -2, step: 1}`：依次取-96到-3，stop不包含在内；`stop: null` 表示负索引取到-1。start/stop应同为负数或同为非负数；跨边界使用显式列表。
- `index: {base: -96, horizon_offset: true}`：实际索引为 `base + horizon`。horizon从1开始，H1取-95、H16取-80、H20取-76。
- `capacity_weighted: true`：每个站先取指定点，再按该点有效容量加权；false或省略时取省级行。缺失值不参与站级加权；所有站该点缺失时结果为NaN。
- `time` 可以选择 hour/hour_sin/hour_cos，省略或设为 `[]` 时不增加时间特征。hour为整数小时，sin/cos包含分钟。

`history` 和 `future` 都是显式字段配置，不再按字段后缀自动发现。删除 `history.GHI_SOLARGIS` 即可停用历史气象；恢复最新两点可将功率的stop改为null。

固定索引只构造一次；只有开启horizon_offset的字段和时间特征按horizon生成。返回值仍为 `df, columns_by_horizon, weighted_fields`。例如共享列 `history__observe_power__index_-96`，H1列 `future__weighted__GHI_SOLARGIS_predict__index_0__h01`。原始字段名、加权方式和实际索引包含在列名中。

索引按原始数组从旧到新解释，索引的实际时刻需与上游核对；越界点记为NaN，不能凭配置创造缺失观测。训练标签仍取 `data.columns.power_future` 的 `horizon-1` 点，目标时刻仍为起报时刻加 `horizon * minutes_per_point`；history/future配置只控制模型输入。

这一版使用新的特征列名，需要重新训练；旧的history_length/weather_columns/history_weather_columns配置应迁移到上述格式。旧checkpoint请使用与其训练匹配的旧代码和配置，不能直接搭配这份新配置。

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

## 评价指标

评价指标由 config 控制：

```yaml
evaluation:
  primary_metric: official_accuracy
  metrics: [rmse, mae, official_accuracy]
  capacity_floor_ratio: 0.2
```

`official_accuracy` 在每个 horizon 模型内部独立计算：`1 - mean(abs(预测功率 - 可用功率) / max(可用功率, 0.2 * 装机容量))`。不同 horizon 的指标不会再次合并。

训练 loss 默认与该分母对齐：

```yaml
training:
  loss: weighted_mae
```

`weighted_mae` 在物理功率尺度上计算每个 TabM 成员的 `abs(预测功率 - 可用功率) / max(可用功率, 0.2 * 装机容量)` 并求平均。它不先平均 TabM 成员，也不先平均 horizon。需要恢复原训练损失时，将 `loss` 改为 `mse`。

训练时，每个 horizon 使用它自己的归一化准确率选择最佳 epoch；测试时也只输出各 horizon 自己的 RMSE、MAE 和 `official_accuracy`。如需保持原来的早停逻辑，只需将 `primary_metric` 改为 `rmse`。

## Checkpoint 结构

```text
artifacts/tabm_v2/
├── config_resolved.yaml
├── metadata.json
├── metrics_by_horizon.csv
├── models/model_h01.pt ... model_h16.pt
└── preprocessors/preprocessor_h01.joblib ... preprocessor_h16.joblib
```

新checkpoint的metadata.json保存完整features配置，config_resolved.yaml保存完整训练配置。test()/predict()/Model.inference()自动沿用checkpoint中的features；输入配置不同时会print提示。每个子模型仍按其保存的有序feature_names选列。设备、数据路径、容量表及输出配置仍来自调用方；原始字段映射应与训练一致。

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
