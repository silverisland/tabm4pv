from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from protected.evaluator import validate_adapter_result
from pvreglab.implementation import (
    implementation_hash,
    snapshot_implementation,
)
from pvreglab.models import ExperimentRequest
from pvreglab.protection import verify_config_lock, verify_manifest
from pvreglab.registry import Registry


class ExperimentRunner:
    def __init__(self, root: Path, config: dict[str, Any]):
        self.root = root.resolve()
        self.config = config
        self.state_dir = Path(config["state_dir"])
        self.requests_dir = self.state_dir / "requests"
        self.outputs_dir = self.state_dir / "outputs"
        self.logs_dir = self.state_dir / "logs"
        for path in (self.requests_dir, self.outputs_dir, self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.registry = Registry(self.state_dir)

    def run_many(
        self,
        requests: Iterable[ExperimentRequest],
        *,
        rerun: bool = False,
    ) -> list[dict[str, Any]]:
        records = []
        completed = self.registry.completed_ids()
        for request in requests:
            if request.experiment_id() in completed and not rerun:
                print(f"skip completed {request.experiment_id()} {request.mode}")
                continue
            records.append(self.run_one(request))
        return records

    def run_one(self, request: ExperimentRequest) -> dict[str, Any]:
        protection_errors = verify_manifest(
            self.root, self.state_dir
        ) + verify_config_lock(
            Path(self.config["_config_path"]), self.state_dir
        )
        if protection_errors:
            raise RuntimeError("; ".join(protection_errors))

        baseline_id = str(self.config["baseline_implementation_id"])
        if request.implementation_hash != baseline_id:
            current_hash = implementation_hash(self.root / "editable")
            if request.implementation_hash != current_hash:
                raise RuntimeError(
                    "Experiment request implementation does not match editable/"
                )
            snapshot_implementation(self.root / "editable", self.state_dir)

        request_payload = request.to_dict()
        experiment_id = request_payload["experiment_id"]
        request_path = self.requests_dir / f"{experiment_id}.json"
        output_path = self.outputs_dir / f"{experiment_id}.json"
        log_path = self.logs_dir / f"{experiment_id}.log"
        request_path.write_text(
            json.dumps(request_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if output_path.exists():
            output_path.unlink()

        replacements = {
            "request": str(request_path),
            "output": str(output_path),
            "project_root": str(self.config["project_root"]),
            "lab_root": str(self.root),
            "python": str(self.config["python"]),
        }
        command = [
            str(part).format(**replacements)
            for part in self.config["adapter_command"]
        ]
        print(
            f"run {experiment_id} stage={request.stage} mode={request.mode} "
            f"held_out={request.held_out_station} seed={request.seed}"
        )
        start = time.monotonic()
        process_error = ""
        try:
            completed = subprocess.run(
                command,
                cwd=self.root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=int(self.config["budget"]["command_timeout_seconds"]),
                check=False,
            )
            returncode = completed.returncode
            output = completed.stdout
        except subprocess.TimeoutExpired as error:
            returncode = -1
            output = error.stdout or ""
            process_error = "adapter timed out"
        except OSError as error:
            returncode = -1
            output = ""
            process_error = f"adapter could not start: {error}"
        runtime = time.monotonic() - start
        log_path.write_text(str(output), encoding="utf-8")

        if returncode != 0 or not output_path.exists():
            result_payload = {
                "experiment_id": experiment_id,
                "status": "failed",
                "score": None,
                "runtime_seconds": runtime,
                "diagnostics": {},
                "per_month": {},
                "error": (
                    process_error
                    or f"adapter exit={returncode}; see {log_path}"
                ),
            }
        else:
            try:
                result_payload = json.loads(
                    output_path.read_text(encoding="utf-8")
                )
                result_payload.setdefault("runtime_seconds", runtime)
                result_payload = validate_adapter_result(
                    result_payload, request_payload
                )
            except (ValueError, json.JSONDecodeError) as error:
                result_payload = {
                    "experiment_id": experiment_id,
                    "status": "failed",
                    "score": None,
                    "runtime_seconds": runtime,
                    "diagnostics": {},
                    "per_month": {},
                    "error": f"invalid adapter result: {error}",
                }

        after_errors = verify_manifest(
            self.root, self.state_dir
        ) + verify_config_lock(
            Path(self.config["_config_path"]), self.state_dir
        )
        if after_errors:
            result_payload = {
                "experiment_id": experiment_id,
                "status": "failed",
                "score": None,
                "runtime_seconds": runtime,
                "diagnostics": {},
                "per_month": {},
                "error": "protected files changed: " + "; ".join(after_errors),
            }

        record = {"request": request_payload, "result": result_payload}
        self.registry.append(record)
        score = result_payload.get("score")
        print(f"result status={result_payload['status']} score={score}")
        return record
