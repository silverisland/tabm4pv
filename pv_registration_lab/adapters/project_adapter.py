"""Local integration point for the confidential experiment environment.

Implement ``run_local_experiment`` inside the private environment. Keep raw
data inside the existing project and return aggregate metrics only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def run_local_experiment(request: dict[str, Any]) -> dict[str, Any]:
    """Run the existing TabM pipeline for one controlled request.

    Required request fields include mode, seed, training_stations,
    held_out_station, calibration_days and final_test. The implementation must
    use training_stations for fitting TabM, use only calibration_days of a
    pseudo-target to fit registration, and evaluate on its disjoint period.
    For the sealed real target, target_history_role=train_and_registration
    means canonical station 雅砻江 contributes only its configured 2024 history;
    雅砻江解放站 contributes only its configured 2025 evaluation rows.
    The returned aggregate result must include the audit fingerprints defined
    in references/adapter-contract.md; baseline and identity fingerprints must
    match exactly.
    """
    raise NotImplementedError(
        "Connect this adapter to the private TabM entrypoint. Read "
        "references/adapter-contract.md before implementing it."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    try:
        payload = run_local_experiment(request)
        payload = dict(payload)
        payload.update(
            experiment_id=request["experiment_id"],
            status="ok",
        )
    except Exception as error:
        payload = {
            "experiment_id": request["experiment_id"],
            "status": "failed",
            "score": None,
            "runtime_seconds": 0.0,
            "diagnostics": {},
            "per_month": {},
            "error": f"{type(error).__name__}: {error}",
        }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
