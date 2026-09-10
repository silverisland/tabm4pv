# PV 16-Step Ensemble Lab

面向单个光伏实验站点的超短期功率预测评估项目。项目比较组件模型在
P1～P16上的效果，使用独立时间区间训练近期功率修正与融合权重，并输出正式
日/月指标、分析明细和统一DS推理结果。

## 模型与防泄漏流程

1. 按自然日将数据顺序切分为训练、校准A、校准B和测试集。
2. 切分边界预留4小时隔离带，避免同一目标时刻同时出现在两个集合。
3. 训练集训练昨日同刻、Ridge和16个分点LightGBM组件。
4. 校准A训练近期功率残差修正模型。
5. 校准B为P1～P16分别学习非负且和为1的融合权重。
6. 测试集只评估，不训练、不选模、不调权。

TabM、TiDE等已有模型可以通过标准DS文件配置为`external_ds`组件。

## 数据约定

- 输入路径只从YAML配置的`data.path`读取，支持文件、目录、glob或文件列表。
- `timestamp_win`为北京时间起报时刻，数据分辨率为15分钟。
- `observe_power`为历史7天、672点功率序列，最后一点对应起报时刻。
- `observe_power_future`为未来2天、192点功率序列。
- `prediction[k]`对应`timestamp_win + (k + 1) * 15分钟`。
- P2～P16真值按目标时间重新映射为对应时刻的Index0真值；原始真值冲突单独报告。
- `cap_power_on`为正数，当前实验按固定装机容量解释。
- 未显式配置气象列时，自动发现`*_predict`及其同名历史字段。

## 使用方法

安装依赖：

```bash
python -m pip install -r requirements.txt
```

复制并修改[配置示例](configs/station_experiment_v1.yaml)，至少设置`data.path`，
然后在项目目录运行：

```bash
python run_station_experiment.py --config configs/station_experiment_v1.yaml
```

`split.mode: auto`会根据实际日期范围自动切分。需要固定回测边界时，将其改为
`dates`，并设置包含式的`train_end`和`validation_end`自然日。

## 输出

```text
outputs/station_component_v1/
├── evaluation_report.xlsx
├── predictions.parquet
├── missing_timestamps.csv
├── run_config.yaml
├── run_manifest.json
├── models/
│   ├── ridge.joblib
│   ├── lightgbm.joblib
│   ├── recent_correction.joblib
│   └── fusion_weights.csv
└── ds/
    ├── persistence_ds.parquet
    ├── ridge_ds.parquet
    ├── lightgbm_ds.parquet
    ├── recent_correction_ds.parquet
    └── fusion_ds.parquet
```

Excel包含实验摘要、数据质量、真值冲突、数据切分、逐点指标、分段指标、场景
指标、16点曲线指标、每日指标、月度指标、融合对比和融合权重。

DS文件严格包含三列：

- `timestamp_win`：起报时间；
- `station`：站名；
- `prediction`：一维`np.ndarray`，长度16，依次为P1～P16。

Parquet底层会使用嵌套数组表示该列。通过`pv16lab.delivery.read_ds`读取时会统一
恢复成`np.ndarray(dtype=np.float32)`。

## 指标口径

单点归一化误差为：

\[
e=\frac{|P_M-P_P|}{\max(P_M,0.2C)}
\]

日准确率按目标日96个时刻、每个时刻16次预测求平均：

\[
A_d=1-\frac{1}{96\times16}\sum e
\]

月指标是符合评价区间的日准确率算术平均。缺失预测默认按归一化误差1计入，
并在Excel中同时报告覆盖率和完整日标记。

## 测试

```bash
python -m unittest discover -s tests -v
```
