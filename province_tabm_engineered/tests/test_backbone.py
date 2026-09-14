"""Deployment parity: sklearn preprocessing, strict load, and full predictions."""

import json
from datetime import date
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.preprocessing import QuantileTransformer

from province_tabm_engineered.backbone import TorchPreprocessor, build_model
from province_tabm_engineered.export_checkpoint import export_checkpoint
from province_tabm_engineered.inference import Model
from province_tabm_engineered.api import train
from province_tabm_engineered.tests.test_feature_rules import _config, _data


def test_torch_quantiles_match_sklearn_at_duplicates_and_boundaries():
    rng = np.random.default_rng(7)
    train_values = rng.normal(size=(500, 6)).astype(np.float32)
    train_values[:, 1] = np.round(train_values[:, 1])
    train_values[:, 2] = 0
    train_values[:, 3] = 17
    train_values[:, 4] *= 10000
    train_values[:, 5] *= 1e-6
    for distribution in ("normal", "uniform"):
        for n_quantiles in (1, 31, 500):
            reference = QuantileTransformer(
                n_quantiles=n_quantiles, output_distribution=distribution,
                subsample=None,
            ).fit(train_values)
            actual = TorchPreprocessor(6, n_quantiles, False, distribution)
            actual.quantiles.copy_(torch.tensor(reference.quantiles_))
            actual.references.copy_(torch.tensor(reference.references_))
            points = reference.quantiles_.astype(np.float32)
            values = np.concatenate([
                train_values, rng.normal(size=(100, 6)).astype(np.float32),
                points, np.nextafter(points, -np.inf), np.nextafter(points, np.inf),
                np.full((1, 6), -1e9, dtype=np.float32),
                np.full((1, 6), 1e9, dtype=np.float32),
            ])
            expected = reference.transform(values)
            np.testing.assert_allclose(
                actual(torch.tensor(values)).numpy(), expected, atol=1e-6, rtol=1e-6
            )
            if torch.cuda.is_available():
                np.testing.assert_allclose(
                    actual.cuda()(torch.tensor(values, device="cuda")).cpu().numpy(),
                    expected, atol=1e-6, rtol=1e-6,
                )


def test_export_and_strict_backbone_inference_parity(tmp_path):
    # Both preprocessor modes and 16/20 complete horizon sets are supported.
    for use_imputer, horizons in ((True, 20), (False, 16)):
        cfg = _config()
        cfg["features"]["n_horizons"] = horizons
        cfg["data"]["date_ranges"] = {
            split: {
                bound: date(2026, 8, day) if use_imputer else f"2026-08-0{day}"
                for bound in ("start", "end")
            }
            for day, split in enumerate(["train", "validation", "test"], 1)
        }
        cfg["model"]["device"] = "cpu"
        cfg["model"]["architecture"].update(n_blocks=1, d_block=16, k=2, dropout=0.0)
        cfg["training"].update(epochs=1, batch_size=4, inference_batch_size=3)
        cfg["training"]["preprocessing"].update(
            min_quantiles=2, max_quantiles=4, use_imputer=use_imputer,
        )
        cfg["output"]["checkpoint_dir"] = str(tmp_path / f"checkpoint_{horizons}")
        data = _data(cfg)
        # Include an external capacity file: deploy must snapshot it into config.
        capacity_path = tmp_path / f"capacity_{horizons}.csv"
        pd.DataFrame({
            "plant_pointname": ["plant_guangfu0001", "plant_guangfu0002"],
            "GCCAPACITY": [123.0, 456.0],
        }).to_csv(capacity_path, index=False)
        cfg["data"]["capacity_csv"] = str(capacity_path)
        torch.manual_seed(7)
        trained = train(cfg, data)
        if horizons == 16:
            # Reusing the training directory refreshes deployment automatically.
            trained = train(cfg, data)
        checkpoint = trained["checkpoint_dir"]
        deployment = trained["deployment_dir"]
        assert deployment == checkpoint / "deployment"
        if horizons == 20:
            manual = export_checkpoint(checkpoint, tmp_path / "manual_deployment")
            assert (manual / "model_config.json").read_text() == (deployment / "model_config.json").read_text()
            manual_state = load_file(str(manual / "model.safetensors"))
            for key, tensor in load_file(str(deployment / "model.safetensors")).items():
                torch.testing.assert_close(manual_state[key], tensor)
        model_config = json.loads((deployment / "model_config.json").read_text())
        assert model_config["data"]["date_ranges"]["train"]["start"] == "2026-08-01"
        assert sorted(p.name for p in deployment.iterdir()) == ["model.safetensors", "model_config.json"]
        assert model_config["data"]["capacity_csv"] is None
        assert model_config["data"]["capacity_mapping"]["plant_guangfu0001"] == 123
        new = build_model("province_tabm", model_config)
        assert isinstance(new, torch.nn.Module)
        state = load_file(str(deployment / "model.safetensors"))
        new.load_state_dict(state, strict=True)
        new.to("cpu").eval()
        invalid = dict(state)
        invalid.pop(next(iter(invalid)))
        with np.testing.assert_raises(RuntimeError):
            new.load_state_dict(invalid, strict=True)
        new.load_state_dict(state, strict=True)
        old = Model(cfg, checkpoint)
        inputs = data.drop(columns="observe_power_future").copy()
        inputs["cap_power_on"] = inputs["cap_power_on"].map(lambda x: np.array([x, x]))
        if use_imputer:
            inputs["observe_power"] = inputs["observe_power"].map(
                lambda x: np.concatenate([[np.nan], x[1:]])
            )
        expected = old.inference(inputs)
        # Prove the exported model doesn't need the source CSV at inference.
        capacity_path.rename(capacity_path.with_suffix(".unused"))
        actual = new(inputs)
        pd.testing.assert_frame_equal(actual.iloc[:, :2], expected.iloc[:, :2])
        np.testing.assert_allclose(
            np.stack(actual["observe_power_predict"]),
            np.stack(expected["observe_power_predict"]), rtol=1e-5, atol=0.01,
        )
        assert actual["observe_power_predict"].map(len).eq(horizons).all()
        # Null target cells cannot affect inference.
        inputs["observe_power_future"] = None
        np.testing.assert_array_equal(
            np.stack(new.inference(inputs)["observe_power_predict"]),
            np.stack(actual["observe_power_predict"]),
        )
        if torch.cuda.is_available():
            np.testing.assert_allclose(
                np.stack(new.cuda().eval()(inputs)["observe_power_predict"]),
                np.stack(expected["observe_power_predict"]), rtol=1e-4, atol=0.1,
            )
        # Import and run the actual deployment in a process blocking sklearn/joblib.
        input_path = tmp_path / f"inputs_{horizons}.parquet"
        inputs.drop(columns="observe_power_future").to_parquet(input_path)
        subprocess.run([
            sys.executable, "-c", """
import sys, importlib.abc
class BlockTrainingDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sklearn', 'joblib'}:
            raise ImportError('Deployment imported training dependency: ' + fullname)
sys.meta_path.insert(0, BlockTrainingDependencies())
import json
import pandas as pd
from safetensors.torch import load_file
from province_tabm_engineered.backbone import build_model
from pathlib import Path
directory = Path(sys.argv[1])
model = build_model('province_tabm', json.loads((directory / 'model_config.json').read_text()))
model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
model.eval()
result = model(pd.read_parquet(sys.argv[2]))
assert result['observe_power_predict'].map(len).eq(int(sys.argv[3])).all()
""", str(deployment), str(input_path), str(horizons),
        ], check=True, cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)

    cfg["data"]["capacity_csv"] = None
    cfg["model"]["horizons"] = [1]
    cfg["output"]["checkpoint_dir"] = str(tmp_path / "partial_checkpoint")
    partial = train(cfg, data)
    assert partial["deployment_dir"] is None
    assert not (partial["checkpoint_dir"] / "deployment").exists()
