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

`predict()` 必须加载完整的 1–16 horizon checkpoint；传单模型文件或输入多个起报时刻会报错。`ckpt_path` 一般传完整 checkpoint 目录。

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
  strict_file_dates: true
  validate_file_content_date: true
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

日期范围必须按照 train、validation、test 的顺序且不能重叠。启用后：

- `train(config)` 分别读取三个范围并进行训练、早停验证和最终测试；
- `test(ckpt, None, config)` 只读取 `test` 范围；
- 直接传入 DataFrame 时，使用配置的 timestamp 列按日期执行相同过滤；
- `date_ranges: null` 时继续使用旧的 `training.split` 自动切分。

目录输入会逐文件完成容量覆盖、数据校验和全部时效的加权气象特征，只拼接体积较小的省级样本，文件处理完后即释放场站级原始数据。为保证结果与全量计算一致，同一起报时刻不能跨文件出现；`validate_file_content_date: true` 还会校验文件名日期与文件内起报日期一致。

训练和推理可以使用不同设备；只需分别在配置中设置 `model.device` 为 `cpu`、`cuda:0` 或 `auto`。模型结构与特征配置应保持一致。

## 命令行接口

```bash
python -m province_tabm_engineered.cli train \
  --config province_tabm_engineered/config.yaml \
  --data /path/to/train_data

python -m province_tabm_engineered.cli test \
  --config province_tabm_engineered/config.yaml \
  --checkpoint artifacts/tabm_v2

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

## print 运行信息

训练、测试和推理直接使用 Python 原生 `print()` 向标准输出打印运行信息，不使用 `logging` 库或日志封装。输出包括数据规模、设备、horizon 进度和文件的绝对保存路径。训练期间可通过以下参数控制验证集打印频率：

```yaml
training:
  log_every_n_epochs: 10
```

首个 epoch 始终打印；之后每隔指定 epoch 打印一次。模型与预处理器每次保存、加载时都会直接 `print` 完整路径，方便定位产物。

任务启动时还会通过 `print` 输出参数审计信息，包括：

- config 输入来源和解析后的绝对路径；
- checkpoint 原始输入、解析后的目录及实际加载的模型文件列表；
- checkpoint 内的 horizon、特征数、target scale、best epoch 和模型结构；
- DataFrame 列映射、容量来源、省级容量及气象列；
- history length、horizon 数量、时间间隔、设备和预测裁剪范围；
- batch size、学习率、权重衰减、early stopping 和数据切分参数；
- Imputer 输入特征数及 QuantileTransformer 的 quantile 数量和输出分布。

这些日志同时展示“调用时传入的参数”和“checkpoint 中实际保存的参数”，可以用于排查传错 config、ckpt、设备或特征配置的问题。

## 测试

```bash
pip install -r province_tabm_engineered/requirements-dev.txt
python -m pytest province_tabm_engineered/tests
```

测试包含原版 v2 特征数值与列顺序回归检查，以及默认 TabM/预处理参数检查。
