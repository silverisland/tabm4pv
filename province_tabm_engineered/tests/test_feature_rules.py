from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from province_tabm_engineered.api import predict, test as evaluate, train
from province_tabm_engineered.config import load_config
from province_tabm_engineered.features import build_feature_data, feature_indices
from province_tabm_engineered.inference import Model


def _config():
    cfg = load_config(Path(__file__).parents[1] / "config.yaml")
    cfg["data"]["capacity_csv"] = None
    cfg["features"]["n_horizons"] = 20
    return cfg


def _data(cfg):
    rows = []
    for day in range(1, 4):
        for step in range(4):
            origin = pd.Timestamp(f"2026-08-0{day} 12:00") + pd.Timedelta(minutes=15 * step)
            for station, capacity, offset in [
                ("province_guangxi_solar", 15000.0, 900.0),
                ("plant_guangfu0001", 100.0, 0.0),
                ("plant_guangfu0002", 300.0, 100.0),
            ]:
                row = {
                    "timestamp_win": origin,
                    "station": station,
                    "cap_power_on": capacity,
                    "observe_power": np.arange(96, dtype=np.float32) + 1000 + step,
                    "observe_power_future": np.arange(20, dtype=np.float32) + 1100 + step,
                    "GHI_SOLARGIS": np.arange(672, dtype=np.float32) + offset,
                }
                for field in cfg["features"]["future"]:
                    row[field] = np.arange(20, dtype=np.float32) + offset
                rows.append(row)
    return pd.DataFrame(rows)


def test_twenty_horizons_exclude_unreliable_power_and_align_weather():
    cfg = _config()
    data = _data(cfg)
    frame, columns, _ = build_feature_data(data, cfg, list(range(1, 21)))
    shared = [name for name in frame if name.startswith("history__observe_power__")]
    assert len(shared) == 94  # Stored only once, regardless of horizon count.
    np.testing.assert_array_equal(frame.loc[0, shared], np.arange(94) + 1000)
    for h in range(1, 21):
        assert columns[h][:94] == shared
        assert frame.loc[0, f"history__weighted__GHI_SOLARGIS__index_{-96+h}__h{h:02d}"] == 672 - 96 + h + 75
        assert frame.loc[0, f"future__weighted__GHI_SOLARGIS_predict__index_{h-1}__h{h:02d}"] == h - 1 + 75
        assert frame.loc[0, f"target_power__h{h:02d}"] == 1100 + h - 1

    changed = data.copy(deep=True)
    changed["observe_power"] = changed["observe_power"].map(
        lambda x: np.concatenate([x[:-2], [np.nan, 999999.0]])
    )
    actual, actual_columns, _ = build_feature_data(changed, cfg, list(range(1, 21)))
    pd.testing.assert_frame_equal(frame, actual)
    assert columns == actual_columns


def test_fixed_indices_weighting_missing_values_and_optional_features():
    cfg = _config()
    data = _data(cfg).iloc[:3].copy()
    cfg["features"] = {
        "n_horizons": 20,
        "minutes_per_point": 15,
        "history": {
            "observe_power": {"indices": [-96, -3]},
            "GHI_SOLARGIS": {"index": -1, "capacity_weighted": True},
        },
        "future": {
            "GHI_SOLARGIS_predict": {"index": 0, "capacity_weighted": False},
        },
        "time": [],
    }
    data.at[1, "GHI_SOLARGIS"][-1] = np.nan
    frame, columns, _ = build_feature_data(data, cfg, [1, 20])
    assert columns[1] == columns[20]
    assert len(columns[1]) == 4
    assert frame.loc[0, "history__weighted__GHI_SOLARGIS__index_-1"] == 771.0
    assert frame.loc[0, "future__GHI_SOLARGIS_predict__index_0"] == 900.0
    assert not any(name.startswith("time__") for name in frame)

    data.at[2, "GHI_SOLARGIS"][-1] = np.nan
    frame, _, _ = build_feature_data(data, cfg, [1])
    assert np.isnan(frame.loc[0, "history__weighted__GHI_SOLARGIS__index_-1"])

    # No history weather, no station rows and no future labels are required here.
    del cfg["features"]["history"]["GHI_SOLARGIS"]
    data = data.iloc[:1].drop(columns=["observe_power_future", "GHI_SOLARGIS"])
    frame, columns, _ = build_feature_data(data, cfg, [20])
    assert len(columns[20]) == 3
    assert "target_power__h20" not in frame


def test_index_rules_reject_ambiguous_or_empty_selection():
    assert feature_indices({"index": {"base": -3, "horizon_offset": False}}, 20) == [-3]
    assert feature_indices({"indices": {"start": -6, "stop": -2, "step": 2}}, 1) == [-6, -4]
    assert feature_indices({"indices": {"start": -3, "stop": None}}, 1) == [-3, -2, -1]
    for rule in [
        {"index": -1, "indices": [-1]},
        {"indices": []},
        {"indices": [1, 1]},
        {"index": 1.5},
        {"indices": {"start": -3, "stop": 2}},
    ]:
        try:
            feature_indices(rule, 1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid rule accepted: {rule}")


def test_train_test_predict_and_batch_inference_roundtrip(tmp_path):
    cfg = _config()
    cfg["data"]["date_ranges"] = {
        split: {"start": f"2026-08-0{day}", "end": f"2026-08-0{day}"}
        for day, split in enumerate(["train", "validation", "test"], 1)
    }
    cfg["model"]["device"] = "cpu"
    cfg["model"]["architecture"].update(n_blocks=1, d_block=16, k=2, dropout=0.0)
    cfg["training"].update(epochs=1, batch_size=4, inference_batch_size=4)
    cfg["training"]["preprocessing"].update(min_quantiles=2, max_quantiles=4)
    cfg["output"]["checkpoint_dir"] = str(tmp_path / "checkpoint")
    data = _data(cfg)
    trained = train(cfg, data)
    checkpoint = trained["checkpoint_dir"]
    assert len(list((checkpoint / "models").glob("*.pt"))) == 20
    assert len(list((checkpoint / "preprocessors").glob("*.joblib"))) == 20
    assert len(trained["metrics"]) == 20

    # Caller edits cannot silently change the saved feature selection.
    caller = deepcopy(cfg)
    caller["features"]["history"]["observe_power"]["indices"] = [-1]
    caller["features"]["n_horizons"] = 1
    metrics, deliveries = evaluate(checkpoint, data, caller)
    assert len(metrics) == 20
    assert len(deliveries) == 80
    assert caller["features"]["n_horizons"] == 1  # Caller config is not mutated.

    origin = pd.Timestamp("2026-08-03 12:00")
    single = data[data["timestamp_win"].eq(origin)].drop(columns="observe_power_future")
    predicted = predict(checkpoint, single, caller)
    assert len(predicted) == 20
    np.testing.assert_allclose(
        predicted[cfg["output"]["prediction_column"]],
        deliveries.loc[origin][cfg["output"]["prediction_column"]],
    )
    model = Model(caller, checkpoint)
    batch = model.inference(data.drop(columns="observe_power_future"))
    assert len(batch) == 12
    assert batch["observe_power_predict"].map(len).eq(20).all()
    np.testing.assert_allclose(
        batch.loc[batch["timestamp_win"].eq(origin), "observe_power_predict"].iloc[0],
        predicted[cfg["output"]["prediction_column"]],
    )
    single["observe_power"] = single["observe_power"].map(
        lambda x: np.concatenate([x[:-2], [np.nan, 999999.0]])
    )
    np.testing.assert_array_equal(
        model.inference(single)["observe_power_predict"].iloc[0],
        predicted[cfg["output"]["prediction_column"]].to_numpy(),
    )
