from __future__ import annotations

from pathlib import Path

import numpy as np
import rtdl_num_embeddings
import sklearn.impute
import sklearn.preprocessing
import tabm
import torch

from province_tabm_engineered.config import load_config
from province_tabm_engineered.model import fit_preprocessor, make_model


CONFIG_PATH = Path(__file__).parents[1] / "config.yaml"


def test_default_model_is_state_dict_equivalent_to_original_construction():
    config = load_config(CONFIG_PATH)
    n_features = 8

    torch.manual_seed(123)
    reference_embeddings = rtdl_num_embeddings.LinearReLUEmbeddings(n_features)
    reference = tabm.TabM.make(
        n_num_features=n_features,
        cat_cardinalities=[],
        d_out=1,
        num_embeddings=reference_embeddings,
    )

    torch.manual_seed(123)
    actual = make_model(n_features, torch.device("cpu"), config["model"]["architecture"])

    assert reference.state_dict().keys() == actual.state_dict().keys()
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], expected)


def test_default_preprocessor_matches_original_fit_recipe():
    config = load_config(CONFIG_PATH)
    seed = 2027
    values = np.random.default_rng(42).normal(size=(300, 8)).astype(np.float32)
    values[::17, 0] = np.nan

    reference_imputer = sklearn.impute.SimpleImputer(
        strategy="median", keep_empty_features=True
    )
    imputed = reference_imputer.fit_transform(values).astype(np.float32)
    noise = np.random.default_rng(seed).normal(
        0.0, 1e-5, imputed.shape
    ).astype(np.float32)
    reference_transformer = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(len(imputed) // 30, 1000), 10),
        output_distribution="normal",
        subsample=10**9,
        random_state=seed,
    ).fit(imputed + noise)
    expected = reference_transformer.transform(imputed).astype(np.float32)

    _, _, actual = fit_preprocessor(values, seed, config)
    np.testing.assert_allclose(actual, expected)
