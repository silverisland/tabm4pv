from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ExperimentRequest:
    stage: str
    mode: str
    seed: int
    training_stations: list[str]
    held_out_station: str
    target_station: str
    calibration_days: int
    implementation_hash: str
    target_history_role: str = "registration_only"
    final_test: bool = False
    hypothesis: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def experiment_id(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["experiment_id"] = self.experiment_id()
        data["created_at"] = datetime.now(timezone.utc).isoformat()
        return data


@dataclass
class ExperimentResult:
    experiment_id: str
    status: str
    score: float | None
    runtime_seconds: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    per_month: dict[str, float] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
