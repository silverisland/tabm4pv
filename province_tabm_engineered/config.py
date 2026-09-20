from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


ConfigInput = str | Path | Mapping[str, Any]
Config = dict[str, Any]


def validate_config(config: Config) -> None:
    required = {"data", "features", "model", "training", "output"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"config 缺少配置段: {missing}")


def load_config(config: ConfigInput) -> Config:
    if isinstance(config, Mapping):
        result = deepcopy(dict(config))
    else:
        with Path(config).expanduser().open("r", encoding="utf-8") as file:
            result = yaml.safe_load(file) or {}
    validate_config(result)
    return result


def dump_config(config: Mapping[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(dict(config), file, allow_unicode=True, sort_keys=False)
