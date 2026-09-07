#!/usr/bin/env python3
"""Standalone H16 experiment derived from tabm4pv.py.

Only Power_predict[15] is trained/evaluated (origin + 4 hours).
Daily ACC16 = 1 - mean_96(abs(P - prediction) / max(P, 0.2 * C)).
Monthly ACC16 is the arithmetic mean of daily ACC16, grouped by target year-month.
This is the H16 component of the revised metric, NOT the full 16-horizon ACC.
The notification does not specify monthly aggregation; equal daily weighting is
used here. C=465 and power/500 label scaling are retained from the original demo.
Power_predict is assumed to contain the available-power reference required by
that notification. Input files must describe ONE plant/aggregate with this C.

Incomplete days are saved in daily_metrics.csv but excluded from the main monthly
score (which requires 96 distinct target slots). A missing prediction at an
existing target slot contributes zero accuracy, i.e. normalized error=1; this is
an explicit per-slot convention, not a claim about the organizer's partial-outage
rules. Entire absent days are not inferred. Coverage is reported, never hidden.

Dependencies: numpy pandas pyarrow scikit-learn torch tabm rtdl_num_embeddings joblib
Run: python THIS_FILE.py --data-root /path/to/parquet
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
import rtdl_num_embeddings
from sklearn.preprocessing import QuantileTransformer
import tabm
import torch
import torch.nn.functional as F

EXPERIMENT = 'tabm4pv_h16_huber'
LOSS_KIND = 'normalized_huber'
HORIZON_STEP = 16
TARGET_INDEX = 15
INPUT_LEN = 96
POINTS_PER_DAY = 96
LABEL_SCALE_VALUE = 500.0
SCORE_CAPACITY = 465.0
COV_COLUMNS = ['GHI_SOLARGIS', 'TEMP_SOLARGIS', 'WS_SOLARGIS', 'WD_SOLARGIS']
X_COLUMNS = ([f'{c}_predict_target' for c in COV_COLUMNS]
             + [f'Power_lag_{i}' for i in range(INPUT_LEN, 0, -1)]
             + ['predict_hour', 'predict_month'])
Y_COLUMN = 'Power_predict_target'


def training_loss(prediction, target_scaled, *, capacity=SCORE_CAPACITY,
                  label_scale=LABEL_SCALE_VALUE, huber_delta=0.1):
    """prediction [B,K]; scaled target [B]. Train every TabM member separately.

    MSE retains original normalized-label MSE. MAE and Huber use the physical
    normalized residual r=(prediction_power-target_power)/max(target_power,.2C).
    Huber is F.huber_loss(r,0,delta)/delta, hence unit slope outside delta.
    delta=0.1 is in dimensionless normalized-error units, not MW.
    """
    target = target_scaled[:, None].expand_as(prediction)
    if LOSS_KIND == 'mse':
        return F.mse_loss(prediction, target)
    target_power = target * label_scale
    denominator = target_power.clamp_min(0.2 * capacity)
    residual = (prediction * label_scale - target_power) / denominator
    if LOSS_KIND == 'weighted_mae':
        return residual.abs().mean()
    if LOSS_KIND == 'normalized_huber':
        if huber_delta <= 0:
            raise ValueError('huber_delta must be positive')
        return F.huber_loss(residual, torch.zeros_like(residual),
                            delta=huber_delta) / huber_delta
    raise ValueError(f'Unknown loss: {LOSS_KIND}')


def normalized_errors(groundtruth, prediction, capacity=SCORE_CAPACITY):
    truth = np.asarray(groundtruth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if capacity <= 0 or truth.shape != pred.shape or truth.size == 0:
        raise ValueError('Positive capacity and nonempty aligned arrays required')
    if not np.isfinite(truth).all():
        raise ValueError('Missing/nonfinite available-power reference')
    valid = np.isfinite(pred)
    errors = np.ones_like(truth)  # Explicit zero-score convention for missing forecasts.
    errors[valid] = np.abs(pred[valid] - truth[valid]) / np.maximum(
        truth[valid], 0.2 * capacity)
    return errors


def metric_tables(predictions, capacity=SCORE_CAPACITY):
    """No averaging over horizons or over signed prediction errors."""
    records = predictions.copy()
    records['target_timestamp'] = pd.to_datetime(records['target_timestamp'])
    timestamps = records['target_timestamp']
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError('Target times must be nonmissing and unique for one plant')
    if not timestamps.eq(timestamps.dt.floor('15min')).all():
        raise ValueError('Target timestamps must be aligned to 15-minute slots')
    records['normalized_absolute_error'] = normalized_errors(
        records['groundtruth'], records['prediction'], capacity)
    records['missing_prediction'] = ~np.isfinite(records['prediction'])
    records['date'] = timestamps.dt.normalize()
    daily = records.groupby('date', sort=True).agg(
        sample_count=('normalized_absolute_error', 'size'),
        mean_normalized_absolute_error=('normalized_absolute_error', 'mean'),
        missing_prediction_count=('missing_prediction', 'sum'),
    ).reset_index()
    daily['complete_day'] = daily['sample_count'].eq(POINTS_PER_DAY)
    daily['observed_slot_accuracy'] = 1 - daily['mean_normalized_absolute_error']
    daily['accuracy'] = daily['observed_slot_accuracy'].where(daily['complete_day'])
    daily['accuracy_percent'] = 100 * daily['accuracy']
    daily['month'] = daily['date'].dt.strftime('%Y-%m')
    monthly = daily.groupby('month', sort=True).agg(
        observed_days=('date', 'size'), scored_days=('complete_day', 'sum'),
        sample_count=('sample_count', 'sum'),
        missing_prediction_count=('missing_prediction_count', 'sum'),
        accuracy=('accuracy', 'mean'),
    ).reset_index()
    monthly['excluded_incomplete_days'] = monthly['observed_days'] - monthly['scored_days']
    monthly['accuracy_percent'] = 100 * monthly['accuracy']
    return records, daily, monthly


def load_dataset(data_root, prefix, suffix):
    paths = sorted(p for p in Path(data_root).glob(f'*{suffix}')
                   if p.name.startswith(prefix))
    if not paths:
        raise FileNotFoundError(f'No {prefix}*{suffix} files in {data_root}')
    frames = []
    for path in paths:
        raw = pd.read_parquet(path)
        frame = pd.DataFrame({'timestamp_win': pd.to_datetime(raw['timestamp_win'])})
        for name in COV_COLUMNS + ['Power']:
            array = np.asarray(raw[name].tolist(), dtype=np.float32)
            if array.ndim != 2 or array.shape[1] < INPUT_LEN:
                raise ValueError(f'{path}: {name} needs at least {INPUT_LEN} history points')
            lag = pd.DataFrame(array[:, -INPUT_LEN:], index=raw.index,
                               columns=[f'{name}_lag_{i}' for i in range(INPUT_LEN, 0, -1)])
            frame = pd.concat([frame, lag], axis=1)
        for name in [f'{c}_predict' for c in COV_COLUMNS] + ['Power_predict']:
            array = np.asarray(raw[name].tolist(), dtype=np.float32)
            if array.ndim != 2 or array.shape[1] <= TARGET_INDEX:
                raise ValueError(f'{path}: {name} needs at least 16 future points')
            frame[f'{name}_target'] = array[:, TARGET_INDEX]
        frame['source_file'] = path.name
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True).sort_values('timestamp_win').reset_index(drop=True)
    if result['timestamp_win'].isna().any() or result['timestamp_win'].duplicated().any():
        raise ValueError('One plant/aggregate required; missing or duplicate origin times found')
    result['target_timestamp'] = result['timestamp_win'] + pd.Timedelta(hours=4)
    # Correct month rollover in the original demo's predict_month feature.
    result['predict_hour'] = result['target_timestamp'].dt.hour
    result['predict_month'] = result['target_timestamp'].dt.month
    if not np.isfinite(result[X_COLUMNS + [Y_COLUMN]].to_numpy(dtype=np.float32)).all():
        raise ValueError('Nonfinite features/labels; clean input before training')
    return result


def split_dataset(dataset):
    # Split all three experiments by forecast-origin time: 2024 development,
    # with each month's final five calendar days held out, and 2025 test.
    ts = dataset['timestamp_win']
    training = dataset.loc[ts.dt.year.eq(2024)].copy()
    test = dataset.loc[ts.dt.year.eq(2025)].copy()
    time = training['timestamp_win']
    is_validation = (time.dt.days_in_month - time.dt.day) < 5
    partitions = {'train': training.loc[~is_validation].copy(),
                  'val': training.loc[is_validation].copy(), 'test': test}
    for name, frame in partitions.items():
        if frame.empty:
            raise ValueError(f'Empty {name} partition under full-year 2024/2025 split')
    return partitions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='/data/hjs/1219_report_onv8/')
    parser.add_argument('--prefix', default='mkv82')
    parser.add_argument('--suffix', default='_v1.parquet')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'experiments_h16' / EXPERIMENT)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=2e-3)
    parser.add_argument('--huber-delta', type=float, default=0.1)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.learning_rate, args.huber_delta) <= 0 or args.patience < 0:
        parser.error('epochs/batch-size/learning-rate/huber-delta must be positive; patience >= 0')
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed + 1)
    torch.manual_seed(args.seed + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 3)
    partitions = split_dataset(load_dataset(args.data_root, args.prefix, args.suffix))
    train_x = partitions['train'][X_COLUMNS].to_numpy(dtype=np.float32)
    noise = np.random.default_rng(0).normal(0, 1e-5, train_x.shape).astype(np.float32)
    preprocessor = QuantileTransformer(
        n_quantiles=max(min(len(train_x) // 30, 1000), 10),
        output_distribution='normal', subsample=10**9,
    ).fit(train_x + noise)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = {name: torch.as_tensor(preprocessor.transform(frame[X_COLUMNS].to_numpy(
        dtype=np.float32)).astype(np.float32), device=device) for name, frame in partitions.items()}
    target = torch.as_tensor(partitions['train'][Y_COLUMN].to_numpy(dtype=np.float32)
                             / LABEL_SCALE_VALUE, device=device)
    model = tabm.TabM.make(n_num_features=len(X_COLUMNS), cat_cardinalities=[], d_out=1,
                          num_embeddings=rtdl_num_embeddings.LinearReLUEmbeddings(len(X_COLUMNS))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=3e-4)

    @torch.inference_mode()
    def predict(part):
        model.eval()
        values = torch.cat([model(batch, None).squeeze(-1).float()
                            for batch in data[part].split(512)]).cpu().numpy()
        # Average TabM members only; no cross-horizon averaging.
        power = (values * LABEL_SCALE_VALUE).mean(axis=1)
        return np.clip(power, 0.0, SCORE_CAPACITY)

    def evaluate(part):
        frame = partitions[part]
        records = frame[['timestamp_win', 'target_timestamp', 'source_file']].copy()
        records['horizon_step'] = HORIZON_STEP
        records['groundtruth'] = frame[Y_COLUMN].to_numpy()
        records['prediction'] = predict(part)
        records, daily, monthly = metric_tables(records)
        if monthly['accuracy'].notna().sum() == 0:
            raise ValueError(f'{part}: no complete 96-slot target days available')
        # Same monthly aggregation for early stopping and final reporting.
        return float(monthly['accuracy'].mean()), records, daily, monthly

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, best_state = -float('inf'), -1, None
    history, remaining = [], args.patience
    print(f'{EXPERIMENT}: H16 only, loss={LOSS_KIND}, device={device}; '
          f'early stopping=mean monthly revised ACC16')
    for epoch in range(args.epochs):
        total_loss, total_samples = 0.0, 0
        for indices in torch.randperm(len(target), device=device).split(args.batch_size):
            model.train()
            optimizer.zero_grad()
            prediction = model(data['train'][indices], None).squeeze(-1).float()
            loss = training_loss(prediction, target[indices], huber_delta=args.huber_delta)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(indices)
            total_samples += len(indices)
        score, _, _, _ = evaluate('val')
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            best_state = deepcopy(model.state_dict())
            remaining = args.patience
        else:
            remaining -= 1
        history.append({'epoch': epoch, 'training_loss': total_loss / total_samples,
                        'validation_mean_monthly_accuracy': score, 'improved': improved})
        print(f'epoch={epoch:03d} loss={total_loss / total_samples:.6f} '
              f'val_ACC16={score:.6f} best={best_score:.6f}')
        if remaining < 0:
            break
    model.load_state_dict(best_state)
    # Test data is evaluated once, after validation-only checkpoint selection.
    score, records, daily, monthly = evaluate('test')
    records.to_parquet(args.output_dir / 'best_prediction.parquet', index=False)
    daily.to_csv(args.output_dir / 'daily_metrics.csv', index=False)
    monthly.to_csv(args.output_dir / 'monthly_metrics.csv', index=False)
    pd.DataFrame(history).to_csv(args.output_dir / 'training_history.csv', index=False)
    torch.save({'model': best_state, 'epoch': best_epoch, 'experiment': EXPERIMENT,
                'loss_kind': LOSS_KIND, 'label_scale': LABEL_SCALE_VALUE,
                'capacity': SCORE_CAPACITY, 'feature_names': X_COLUMNS,
                'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
               args.output_dir / 'best_model.pt')
    joblib.dump(preprocessor, args.output_dir / 'preprocessor.joblib')
    summary = {'experiment': EXPERIMENT, 'loss_kind': LOSS_KIND, 'horizon_step': 16,
               'best_epoch': best_epoch, 'validation_mean_monthly_accuracy': best_score,
               'test_mean_monthly_accuracy': score,
               'test_excluded_incomplete_days': int((~daily['complete_day']).sum()),
               'monthly_aggregation': 'equal mean of complete-day ACC16; year-month groups',
               'overall_aggregation': 'equal mean of months with complete-day scores',
               'test_unscored_months': monthly.loc[monthly['accuracy'].isna(), 'month'].tolist(),
               'missing_prediction_policy': 'existing slot with nonfinite forecast gets accuracy zero',
               'split_clock': 'origin timestamp (original demo)', 'report_clock': 'target timestamp',
               'huber_delta': args.huber_delta if LOSS_KIND == 'normalized_huber' else None}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(monthly.to_string(index=False))
    print(f'Best epoch={best_epoch}; mean monthly ACC16={score:.6f} ({score:.2%})')
    print(f'Incomplete target days excluded: {summary["test_excluded_incomplete_days"]}; '
          'see daily_metrics.csv for coverage. No absent month is replaced with zero.')
    print(f'Outputs: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
