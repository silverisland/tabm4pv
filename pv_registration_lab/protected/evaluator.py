"""Protected result contract and aggregate-only privacy checks."""

from __future__ import annotations

import math
from typing import Any


FORBIDDEN_RESULT_KEYS = {
    "predictions",
    "groundtruth",
    "raw_rows",
    "raw_power",
    "timestamps",
    "data_path",
}
FORBIDDEN_KEY_FRAGMENTS = (
    "prediction",
    "groundtruth",
    "raw_power",
    "raw_row",
    "data_path",
)
PRIVATE_STRING_MARKERS = ("/home/", ".parquet", "file://", "station=")
MAX_COLLECTION_LENGTH = 100
MAX_STRING_LENGTH = 2048
REQUIRED_AUDIT_KEYS = {
    "train_row_count",
    "validation_row_count",
    "evaluation_row_count",
    "calibration_row_count",
    "train_rows_hash",
    "validation_rows_hash",
    "evaluation_rows_hash",
    "calibration_rows_hash",
    "target_index",
    "weather_index",
    "capacity_map_hash",
    "model_config_hash",
    "preprocessing_hash",
    "evaluation_rows_in_train",
    "calibration_eval_overlap",
}


class ResultContractError(ValueError):
    pass


def validate_adapter_result(
    payload: dict[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    _scan_private_content(payload)
    experiment_id = str(request["experiment_id"])
    if payload.get("experiment_id") != experiment_id:
        raise ResultContractError("Adapter returned the wrong experiment_id")
    status = str(payload.get("status", ""))
    if status not in {"ok", "failed"}:
        raise ResultContractError("status must be 'ok' or 'failed'")
    if status == "ok":
        score = payload.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ResultContractError("Successful result needs a finite score")
        audit = payload.get("audit")
        if not isinstance(audit, dict):
            raise ResultContractError("Successful result needs an audit dictionary")
        missing_audit = sorted(REQUIRED_AUDIT_KEYS - set(audit))
        if missing_audit:
            raise ResultContractError(f"audit is missing: {missing_audit}")
        for key in (
            "train_row_count",
            "validation_row_count",
            "evaluation_row_count",
            "calibration_row_count",
        ):
            if not isinstance(audit[key], int) or audit[key] < 0:
                raise ResultContractError(f"audit.{key} must be a nonnegative int")
        for key in (
            "train_rows_hash",
            "validation_rows_hash",
            "evaluation_rows_hash",
            "calibration_rows_hash",
            "capacity_map_hash",
            "model_config_hash",
            "preprocessing_hash",
        ):
            if not isinstance(audit[key], str) or len(audit[key]) < 8:
                raise ResultContractError(f"audit.{key} must be a hash string")
        for key in ("target_index", "weather_index"):
            if not isinstance(audit[key], int) or audit[key] < 0:
                raise ResultContractError(f"audit.{key} must be a nonnegative int")
        if audit["target_index"] != audit["weather_index"]:
            raise ResultContractError(
                "Target power and forecast weather use different physical indices"
            )
        if audit["evaluation_rows_in_train"] != 0:
            raise ResultContractError("Evaluation rows leaked into training")
        if audit["calibration_eval_overlap"] != 0:
            raise ResultContractError("Calibration and evaluation rows overlap")
    diagnostics = payload.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        raise ResultContractError("diagnostics must be an aggregate dictionary")
    per_month = payload.get("per_month", {})
    if not isinstance(per_month, dict):
        raise ResultContractError("per_month must be a dictionary")
    return payload


def _scan_private_content(value: Any, path: str = "result") -> None:
    if isinstance(value, dict):
        if len(value) > MAX_COLLECTION_LENGTH:
            raise ResultContractError(f"Too many fields at {path}")
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_RESULT_KEYS or any(
                fragment in normalized for fragment in FORBIDDEN_KEY_FRAGMENTS
            ):
                raise ResultContractError(f"Forbidden result field at {path}.{key}")
            if "timestamp" in normalized and isinstance(
                child, (str, list, tuple, dict)
            ):
                raise ResultContractError(f"Raw timestamp field at {path}.{key}")
            _scan_private_content(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        if len(value) > MAX_COLLECTION_LENGTH:
            raise ResultContractError(f"Collection is too long at {path}")
        for index, child in enumerate(value):
            _scan_private_content(child, f"{path}[{index}]")
    elif isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            raise ResultContractError(f"String is too long at {path}")
        lowered = value.lower()
        if any(marker in lowered for marker in PRIVATE_STRING_MARKERS):
            raise ResultContractError(f"Private path-like string at {path}")
