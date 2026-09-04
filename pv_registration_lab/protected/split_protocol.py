"""Protected station split rules for pseudo-OOD and final evaluation."""

from __future__ import annotations

from pvreglab.models import ExperimentRequest
from typing import Any


def make_request(
    *,
    stage: str,
    mode: str,
    seed: int,
    source_stations: list[str],
    held_out_station: str,
    target_station: str,
    calibration_days: int,
    implementation_hash: str,
    target_history_role: str = "registration_only",
    final_test: bool = False,
    hypothesis: str = "",
    metadata: dict[str, Any] | None = None,
) -> ExperimentRequest:
    if final_test:
        if held_out_station != target_station:
            raise ValueError("Final test must use target_station")
        training_stations = list(source_stations)
        if target_history_role == "train_and_registration":
            training_stations.append(target_station)
    else:
        if held_out_station not in source_stations:
            raise ValueError("Pseudo-OOD station must be one of source_stations")
        training_stations = [
            station for station in source_stations if station != held_out_station
        ]
    if (
        held_out_station in training_stations
        and not (final_test and target_history_role == "train_and_registration")
    ):
        raise ValueError("Held-out station leaked into training_stations")
    return ExperimentRequest(
        stage=stage,
        mode=mode,
        seed=int(seed),
        training_stations=training_stations,
        held_out_station=held_out_station,
        target_station=target_station,
        calibration_days=int(calibration_days),
        implementation_hash=str(implementation_hash),
        target_history_role=target_history_role,
        final_test=bool(final_test),
        hypothesis=hypothesis,
        metadata=dict(metadata or {}),
    )
