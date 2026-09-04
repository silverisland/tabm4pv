import math
import random
from copy import deepcopy
from typing import Any, Literal, NamedTuple, Optional

import numpy as np
import os
import pandas as pd
import rtdl_num_embeddings  # https://github.com/yandex-research/rtdl-num-embeddings
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import tabm
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor


seed = 0
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)


# >>> Dataset.
DATA_ROOT_PATH = '/path/to/private/data/'
DATA_FILE_SUFFIX = '_v1.parquet'
DATA_FILE_PREFIX = 'mkv82'

HISTORY_WINDOW_HOURS = 24 * 1
POINT_PER_HOUR = 4
TARGET_FUTURE_HOUR = 4
POINTS_PER_DAY = 24 * 4

HISTORY_DAYS_COUNT = 7
INPUT_LEN = HISTORY_WINDOW_HOURS * POINT_PER_HOUR
TARGET_INDEX = (TARGET_FUTURE_HOUR * POINT_PER_HOUR) - 1


COV_COLUMNS = [
   'GHI_SOLARGIS', 'TEMP_SOLARGIS', 'WS_SOLARGIS', 'WD_SOLARGIS'
]

FU_COV_COLUMNS = [
   'GHI_SOLARGIS_predict',
   'TEMP_SOLARGIS_predict',
   'WS_SOLARGIS_predict',
   'WD_SOLARGIS_predict'
]

PV_COL = 'Power'
TARGET_COL = 'Power_predict'

LabelNormalization = Literal['standard', 'scale', 'none']
LABEL_NORMALIZATION: LabelNormalization = 'scale'
LABEL_SCALE_VALUE = 500.0

PREDICTION_CLIP_LOWER = 0.0
PREDICTION_CLIP_UPPER = 465.0
SCORE_CAPACITY = 465.0

def process_single_file(df):
    processed_dfs = []

    if 'timestamp_win' in df.columns:
       processed_dfs.append(df[['timestamp_win']])

    for col_hist in COV_COLUMNS + [PV_COL]:
       raw_hist_data = df[col_hist].tolist()
       hist_array = np.array(raw_hist_data)
       input_features = hist_array[:, -INPUT_LEN:]
       feat_cols = [f"{col_hist}_lag_{i}" for i in range(INPUT_LEN, 0, -1)]
       df_feat = pd.DataFrame(input_features, columns = feat_cols, index = df.index)
       processed_dfs.append(df_feat)

    for col_fut in FU_COV_COLUMNS + [TARGET_COL]:
       raw_fut_data = df[col_fut].tolist()
       fut_array = np.array(raw_fut_data)

       target_values = fut_array[:, TARGET_INDEX]
       df_target = pd.DataFrame(target_values, columns = [f"{col_fut}_target"], index = df.index)

       processed_dfs.append(df_target)

    full_df = pd.concat(processed_dfs, axis = 1)
    return full_df

df_transform_list = []
for file in os.listdir(DATA_ROOT_PATH):
   if not file.endswith(DATA_FILE_SUFFIX): continue
   if DATA_FILE_PREFIX and not file.startswith(DATA_FILE_PREFIX): continue
   df = pd.read_parquet(os.path.join(DATA_ROOT_PATH, file))
   df_transform = process_single_file(df)
   df_transform_list.append(df_transform)
df_transform = pd.concat(df_transform_list)
df_transform['timestamp_win'] = pd.to_datetime(df_transform['timestamp_win'])
df_transform['predict_hour'] = (df_transform['timestamp_win'].dt.hour + 4) % 24
df_transform['predict_month'] = df_transform['timestamp_win'].dt.month

x_columns = []
for col in COV_COLUMNS:
   x_columns.append(f'{col}_predict_target')
x_columns += [f"{PV_COL}_lag_{i}" for i in range(INPUT_LEN, 0, -1)]
x_columns += ['predict_hour', "predict_month"]
y_columns = f'{TARGET_COL}_target'
input_dataset = df_transform.sort_values(by = ['timestamp_win'])
train_dataset, test_dataset = input_dataset[(input_dataset['timestamp_win'].dt.year == 2024) & (input_dataset['timestamp_win'].dt.month >= 9)], input_dataset[input_dataset['timestamp_win'].dt.year == 2025]

TaskType = Literal['regression', 'binclass', 'multiclass']

def get_seasonal_split(df, val_days = 5, date_col = 'timestamp_win'):
    data = df.copy()
    if date_col:
        data[date_col] = pd.to_datetime(data[date_col])
        data.set_index(date_col, inplace = True)
    days_remaining = data.index.days_in_month - data.index.day
    val_mask = days_remaining < val_days

    val_df = data[val_mask]
    train_df = data[~val_mask]

    val_months = val_df.index.month.unique()
    if len(val_months) < 12:
       print(f"警告： 验证集仅覆盖{len(val_months)}个月份")
    else:
       print(f"确认： 验证集成功覆盖全面1-12月")
    return train_df, val_df


# Regression.
task_type: TaskType = 'regression'
n_classes = None

train_ds, valid_ds = get_seasonal_split(train_dataset, val_days=5)
X_num_train, Y_train = train_ds[x_columns].values.astype(np.float32), train_ds[y_columns].values.astype(np.float32)
X_num_valid, Y_valid = valid_ds[x_columns].values.astype(np.float32), valid_ds[y_columns].values.astype(np.float32)
X_num_test, Y_test = test_dataset[x_columns].values.astype(np.float32), test_dataset[y_columns].values.astype(np.float32)
n_num_features = X_num_test.shape[1]
task_is_regression = task_type == 'regression'

data_numpy = {
    'train': {'x_num': X_num_train, 'y': Y_train},
    'val': {'x_num': X_num_valid, 'y': Y_valid},
    'test': {'x_num': X_num_test, 'y': Y_test},
}


for part, part_data in data_numpy.items():
    for key, value in part_data.items():
        print(f'{part:<5}    {key:<5}    {value.shape!r:<10}    {value.dtype}')
        del key, value
    del part, part_data

x_num_train_numpy = data_numpy['train']['x_num']
noise = (
    np.random.default_rng(0)
    .normal(0.0, 1e-5, x_num_train_numpy.shape)
    .astype(x_num_train_numpy.dtype)
)

preprocessing = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=max(min(X_num_train.shape[0] // 30, 1000), 10),
    output_distribution='normal',
    subsample=10**9,
).fit(x_num_train_numpy + noise)
del x_num_train_numpy

# Apply the preprocessing.
for part in data_numpy:
    data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num'])


# Label preprocessing.
class RegressionLabelStats(NamedTuple):
    mean: float
    std: float


def normalize_regression_labels(y_train: np.ndarray) -> tuple[np.ndarray, Optional[RegressionLabelStats]]:
    if LABEL_NORMALIZATION == 'standard':
        stats = RegressionLabelStats(y_train.mean().item(), y_train.std().item())
        if stats.std == 0.0:
            raise ValueError('Training labels have zero std; cannot use standard label normalization.')
        return (y_train - stats.mean) / stats.std, stats
    if LABEL_NORMALIZATION == 'scale':
        if LABEL_SCALE_VALUE <= 0.0:
            raise ValueError(f'LABEL_SCALE_VALUE must be positive, got {LABEL_SCALE_VALUE}.')
        return y_train / LABEL_SCALE_VALUE, None
    if LABEL_NORMALIZATION == 'none':
        return y_train, None
    raise ValueError(f'Unknown LABEL_NORMALIZATION: {LABEL_NORMALIZATION}')


def inverse_transform_regression_predictions(y_pred: np.ndarray) -> np.ndarray:
    if LABEL_NORMALIZATION == 'standard':
        assert regression_label_stats is not None
        return y_pred * regression_label_stats.std + regression_label_stats.mean
    if LABEL_NORMALIZATION == 'scale':
        return y_pred * LABEL_SCALE_VALUE
    if LABEL_NORMALIZATION == 'none':
        return y_pred
    raise ValueError(f'Unknown LABEL_NORMALIZATION: {LABEL_NORMALIZATION}')


if task_type == 'regression':
    # For regression tasks, it is recommended to transform training labels.
    Y_train, regression_label_stats = normalize_regression_labels(
        data_numpy['train']['y'].copy()
    )
else:
    Y_train = data_numpy['train']['y'].copy()
    regression_label_stats = None

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Convert data to tensors
data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}
Y_train = torch.as_tensor(Y_train, device=device)
if task_type == 'regression':
    for part in data:
        data[part]['y'] = data[part]['y'].float()
    Y_train = Y_train.float()

# Automatic mixed precision (AMP)
# torch.float16 is implemented for completeness,
# but it was not tested in the project,
# so torch.bfloat16 is used by default.
amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else None
)
# Changing False to True can speed up training
# of large enough models on compatible hardware.
amp_enabled = False and amp_dtype is not None
grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore

# torch.compile
compile_model = False

# fmt: off
print(f'Device:        {device.type.upper()}')
print(f'AMP:           {amp_enabled}{f" ({amp_dtype})"if amp_enabled else ""}')
print(f'torch.compile: {compile_model}')

# No embeddings.
# num_embeddings = None

# Simple embeddings.
num_embeddings = rtdl_num_embeddings.LinearReLUEmbeddings(n_num_features)

# Periodic embeddings.
# num_embeddings = rtdl_num_embeddings.PeriodicEmbeddings(n_num_features, lite=False)

# Piecewise-linear embeddings.
# num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
#     rtdl_num_embeddings.compute_bins(data['train']['x_num'], n_bins=48),
#     d_embedding=16,
#     activation=False,
#     version='B',
# )

model = tabm.TabM.make(
    n_num_features=n_num_features,
    cat_cardinalities=[],
    d_out=1 if n_classes is None else n_classes,
    num_embeddings=num_embeddings,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
gradient_clipping_norm: Optional[float] = 1.0

if compile_model:
    # NOTE
    # `torch.compile(model, mode="reduce-overhead")` caused issues during training,
    # so the `mode` argument is not used.
    model = torch.compile(model)
    evaluation_mode = torch.no_grad
else:
    evaluation_mode = torch.inference_mode

# A quick reminder: TabM represents an ensemble of k MLPs.
#
# The option below determines if the MLPs are trained
# on the same batches (share_training_batches=True) or
# on different batches. Technically, this option determines:
# - How the loss function is implemented.
# - How the training batches are constructed.
#
# `True` is recommended by default because of better training efficiency.
# On some tasks, `False` may provide better performance.
share_training_batches = True
@torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
def apply_model(part: str, idx: Tensor) -> Tensor:
    return (
        model(
            data[part]['x_num'][idx],
            data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
        )
        .squeeze(-1)  # Remove the last dimension for regression tasks.
        .float()
    )


base_loss_fn = (
    nn.functional.mse_loss if task_is_regression else nn.functional.cross_entropy
)


def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    # TabM produces k predictions. Each of them must be trained separately.

    # Regression:     (batch_size, k)            -> (batch_size * k,)
    # Classification: (batch_size, k, n_classes) -> (batch_size * k, n_classes)
    y_pred = y_pred.flatten(0, 1)

    if share_training_batches:
        # (batch_size,) -> (batch_size * k,)
        y_true = y_true.repeat_interleave(model.backbone.k)
    else:
        # (batch_size, k) -> (batch_size * k,)
        y_true = y_true.flatten(0, 1)

    return base_loss_fn(y_pred, y_true)


@evaluation_mode()
def evaluate(part: str) :
    model.eval()

    # When using torch.compile, you may need to reduce the evaluation batch size.
    eval_batch_size = 512
    y_pred: np.ndarray = (
        torch.cat(
            [
                apply_model(part, idx)
                for idx in torch.arange(len(data[part]['y']), device=device).split(
                    eval_batch_size
                )
            ]
        )
        .cpu()
        .numpy()
    )
    if task_type == 'regression':
        y_pred = inverse_transform_regression_predictions(y_pred)

    # Compute the mean of the k predictions.
    if not task_is_regression:
        # For classification, the mean must be computed in the probability space.
        y_pred = scipy.special.softmax(y_pred, axis=-1)
    y_pred = y_pred.mean(1)

    y_true = data[part]['y'].cpu().numpy()
    score = (
        -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
        if task_type == 'regression'
        else sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
    )
    return float(score)  # The higher -- the better.

@evaluation_mode()
def inference(part: str):
    model.eval()

    # When using torch.compile, you may need to reduce the evaluation batch size.
    eval_batch_size = 512
    y_pred: np.ndarray = (
        torch.cat(
            [
                apply_model(part, idx)
                for idx in torch.arange(len(data[part]['y']), device=device).split(
                    eval_batch_size
                )
            ]
        )
        .cpu()
        .numpy()
    )
    if task_type == 'regression':
        y_pred = inverse_transform_regression_predictions(y_pred)

    # Compute the mean of the k predictions.
    if not task_is_regression:
        # For classification, the mean must be computed in the probability space.
        y_pred = scipy.special.softmax(y_pred, axis=-1)
    y_pred = y_pred.mean(1)

    y_true = data[part]['y'].cpu().numpy()
    score = (
        -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
        if task_type == 'regression'
        else sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
    )
    return float(score), y_true, y_pred


print(f'Test score before training: {evaluate("test"):.4f}')

n_epochs = 200
train_size = X_num_train.shape[0]
batch_size = 512
epoch_size = math.ceil(train_size / batch_size)

epoch = -1
metrics = {'val': -math.inf, 'test': -math.inf}


def make_checkpoint() -> dict[str, Any]:
    return deepcopy(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
        }
    )


best_checkpoint = make_checkpoint()

# Early stopping: the training stops if the validation score
# does not improve for more than `patience` consecutive epochs.
patience = 10
remaining_patience = patience

for epoch in range(n_epochs):
    batches = (
        # Create one standard batch sequence.
        torch.randperm(train_size, device=device).split(batch_size)
        if share_training_batches
        # Create k independent batch sequences.
        else (
            torch.rand((train_size, model.backbone.k), device=device)
            .argsort(dim=0)
            .split(batch_size, dim=0)
        )
    )
    for batch_idx in batches:
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(apply_model('train', batch_idx), Y_train[batch_idx])

        if grad_scaler is None:
            loss.backward()
        else:
            grad_scaler.scale(loss).backward()  # type: ignore

        if gradient_clipping_norm is not None:
            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), gradient_clipping_norm
            )

        if grad_scaler is None:
            optimizer.step()
        else:
            grad_scaler.step(optimizer)
            grad_scaler.update()

    metrics = {part: evaluate(part) for part in ['val', 'test']}
    val_score_improved = metrics['val'] > best_checkpoint['metrics']['val']

    print(
        f'{"*" if val_score_improved else " "}'
        f' [epoch] {epoch:<3}'
        f' [val] {metrics["val"]:.3f}'
        f' [test] {metrics["test"]:.3f}'
    )

    if val_score_improved:
        best_checkpoint = make_checkpoint()
        remaining_patience = patience
    else:
        remaining_patience -= 1

    if remaining_patience < 0:
        break

# To make final predictions, load the best checkpoint.
model.load_state_dict(best_checkpoint['model'])

print('\n[Summary]')
print(f'best epoch:  {best_checkpoint["epoch"]}')
print(f'val score:  {best_checkpoint["metrics"]["val"]}')
print(f'test score: {best_checkpoint["metrics"]["test"]}')

_, groundtruth, prediction = inference('test')

test_dataset['groundtruth'] = groundtruth
test_dataset['prediction'] = prediction

test_dataset['prediction'] = test_dataset['prediction'].clip(
    lower = PREDICTION_CLIP_LOWER,
    upper = PREDICTION_CLIP_UPPER,
)
test_dataset.to_parquet('best_prediction.parquet', index = False)

def rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

result = (
    test_dataset.groupby(test_dataset["timestamp_win"].dt.date)
    .apply(lambda g: pd.Series({
        "rmse_final_pred": rmse(g["prediction"], g["groundtruth"]),
    }))
    .reset_index()
    .rename(columns = {"timestamp_win": "date"})
)

result["date"] = pd.to_datetime(result['date'])
score_list = []
for i in range(12):
    sub = result[result['date'].dt.month == (i+1)]
    s_post = 1 - sub['rmse_final_pred'].mean() / SCORE_CAPACITY
    print(f'{i+1} month score post process: {s_post: .4f}')
    score_list.append(s_post)

print(f'score mean: {sum(score_list) / len(score_list)}')
