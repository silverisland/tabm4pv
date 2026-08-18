from __future__ import annotations

from pathlib import Path

from province_tabm_engineered.config import load_config, validate_config


CONFIG_PATH = Path(__file__).parents[1] / "config.yaml"


def test_defaults_match_original_model_and_preprocessing():
    config = load_config(CONFIG_PATH)

    assert config["model"]["architecture"] == {
        "n_blocks": 2,
        "d_block": 512,
        "dropout": 0.1,
        "activation": "ReLU",
        "k": 32,
        "arch_type": "tabm",
        "start_scaling_init": "normal",
    }
    assert config["training"]["preprocessing"]["quantile_subsample"] == 10**9
    assert config["features"]["weather_columns"] == [
        "GHI_SOLARGIS_predict",
        "TEMP_SOLARGIS_predict",
        "ssrd_pos_1_predict",
        "t2m_pos_1_predict",
    ]


def test_non_overlapping_date_ranges_are_valid():
    config = load_config(CONFIG_PATH)
    config["data"]["date_ranges"] = {
        "train": {"start": "2026-06-01", "end": "2026-07-31"},
        "validation": {"start": "2026-08-01", "end": "2026-08-07"},
        "test": {"start": "2026-08-08", "end": "2026-08-15"},
    }
    validate_config(config)
