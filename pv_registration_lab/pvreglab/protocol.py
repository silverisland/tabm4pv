from __future__ import annotations

from typing import Iterable

from protected.split_protocol import make_request
from pvreglab.models import ExperimentRequest


def implementation_for_mode(config: dict, mode: str) -> str:
    if mode == "baseline":
        return str(config["baseline_implementation_id"])
    return str(config["_implementation_hash"])


def request_metadata(config: dict) -> dict:
    """Configuration passed identically to every adapter mode."""
    return {
        "data_contract": config["data_contract"],
        "weather": config["weather"],
    }


def identity_audit_requests(config: dict) -> list[ExperimentRequest]:
    station = config["source_stations"][0]
    seed = int(config["quick_seeds"][0])
    common = dict(
        stage="identity_audit",
        seed=seed,
        source_stations=list(config["source_stations"]),
        held_out_station=station,
        target_station=str(config["target_station"]),
        calibration_days=int(config["calibration_days"]),
        metadata=request_metadata(config),
    )
    return [
        make_request(
            mode="baseline",
            implementation_hash=implementation_for_mode(config, "baseline"),
            **common,
        ),
        make_request(
            mode="identity",
            implementation_hash=implementation_for_mode(config, "identity"),
            **common,
        ),
    ]


def ablation_requests(config: dict) -> list[ExperimentRequest]:
    count = min(
        int(config.get("quick_held_out_count", 2)),
        len(config["source_stations"]),
    )
    stations = list(config["source_stations"])[:count]
    modes = ["baseline", "identity"] + list(config["candidate_modes"])
    return list(
        _cross_requests(
            config,
            stage="quick_ablation",
            stations=stations,
            modes=modes,
            seeds=config["quick_seeds"],
        )
    )


def loso_requests(
    config: dict, modes: list[str] | None = None
) -> list[ExperimentRequest]:
    selected_modes = modes or ["baseline"] + list(config["candidate_modes"])
    return list(
        _cross_requests(
            config,
            stage="full_loso",
            stations=list(config["source_stations"]),
            modes=selected_modes,
            seeds=config["full_seeds"],
        )
    )


def final_request(config: dict, mode: str, seed: int) -> ExperimentRequest:
    return make_request(
        stage="final_target_test",
        mode=mode,
        seed=seed,
        source_stations=list(config["source_stations"]),
        held_out_station=str(config["target_station"]),
        target_station=str(config["target_station"]),
        calibration_days=int(config["calibration_days"]),
        implementation_hash=implementation_for_mode(config, mode),
        target_history_role=str(
            config["data_contract"]["target_history_role"]
        ),
        final_test=True,
        metadata=request_metadata(config),
    )


def _cross_requests(
    config: dict,
    *,
    stage: str,
    stations: Iterable[str],
    modes: Iterable[str],
    seeds: Iterable[int],
) -> Iterable[ExperimentRequest]:
    for held_out in stations:
        for seed in seeds:
            for mode in modes:
                yield make_request(
                    stage=stage,
                    mode=str(mode),
                    seed=int(seed),
                    source_stations=list(config["source_stations"]),
                    held_out_station=str(held_out),
                    target_station=str(config["target_station"]),
                    calibration_days=int(config["calibration_days"]),
                    implementation_hash=implementation_for_mode(config, str(mode)),
                    metadata=request_metadata(config),
                )
