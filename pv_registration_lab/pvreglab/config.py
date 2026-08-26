from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise ConfigError(
            f"Missing config: {config_path}. Copy config.example.json to "
            "config.json and edit only the local paths and stations."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    return config, config_path


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "project_root",
        "state_dir",
        "adapter_command",
        "source_stations",
        "target_station",
        "baseline_implementation_id",
        "data_contract",
        "calibration_days",
        "quick_seeds",
        "full_seeds",
        "candidate_modes",
        "identity_score_tolerance",
        "acceptance",
        "budget",
        "protected_paths",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ConfigError(f"Missing config keys: {missing}")

    stations = [str(x) for x in config["source_stations"]]
    if len(stations) < 2 or len(stations) != len(set(stations)):
        raise ConfigError("source_stations must contain at least two unique stations")
    if str(config["target_station"]) in stations:
        raise ConfigError("target_station must not be included in source_stations")
    if not str(config["baseline_implementation_id"]).strip():
        raise ConfigError("baseline_implementation_id must be non-empty")

    data_contract = config["data_contract"]
    required_data_keys = {
        "station_info_path",
        "station_info_station_column",
        "station_info_capacity_column",
        "station_aliases",
        "capacity_overrides",
        "target_history_raw_station",
        "target_history_start",
        "target_history_end",
        "target_evaluation_raw_station",
        "target_evaluation_start",
        "target_evaluation_end",
        "target_history_role",
        "source_time_policy",
        "allows_future_source_data",
    }
    missing_data = sorted(required_data_keys - set(data_contract))
    if missing_data:
        raise ConfigError(f"data_contract is missing: {missing_data}")
    if data_contract["target_history_role"] not in {
        "registration_only",
        "train_and_registration",
    }:
        raise ConfigError("Invalid target_history_role")
    aliases = {
        str(key): str(value)
        for key, value in data_contract["station_aliases"].items()
    }
    history_raw = str(data_contract["target_history_raw_station"])
    evaluation_raw = str(data_contract["target_evaluation_raw_station"])
    target = str(config["target_station"])
    if aliases.get(history_raw) != target or aliases.get(evaluation_raw) != target:
        raise ConfigError(
            "Both target raw station names must alias to target_station"
        )
    capacity = data_contract["capacity_overrides"].get(target)
    if not isinstance(capacity, (int, float)) or float(capacity) <= 0:
        raise ConfigError("target_station needs a positive capacity override")
    if int(config["calibration_days"]) < 7:
        raise ConfigError("calibration_days must be at least 7")
    if not isinstance(config["adapter_command"], list):
        raise ConfigError("adapter_command must be a JSON list, not a shell string")
    command = [str(x) for x in config["adapter_command"]]
    if "{request}" not in command or "{output}" not in command:
        raise ConfigError("adapter_command must contain {request} and {output}")

    acceptance = config["acceptance"]
    for key in (
        "min_mean_ood_gain",
        "min_positive_station_ratio",
        "max_worst_station_drop",
    ):
        if key not in acceptance:
            raise ConfigError(f"acceptance is missing {key}")

    budget = config["budget"]
    for key in (
        "max_iterations",
        "max_consecutive_failures",
        "command_timeout_seconds",
    ):
        if key not in budget:
            raise ConfigError(f"budget is missing {key}")


def resolve_paths(
    config: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    resolved = dict(config)
    base = config_path.parent
    project_root = Path(config["project_root"])
    if not project_root.is_absolute():
        project_root = (base / project_root).resolve()
    state_dir = Path(config["state_dir"])
    if not state_dir.is_absolute():
        state_dir = (base / state_dir).resolve()
    resolved["project_root"] = str(project_root)
    resolved["state_dir"] = str(state_dir)
    resolved["python"] = sys.executable
    resolved["_config_path"] = str(config_path)
    resolved["_lab_root"] = str(base.resolve())
    return resolved
