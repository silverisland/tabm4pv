from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pvreglab.report import build_report


def export_feedback(
    records: list[dict[str, Any]], config: dict, destination: Path
) -> tuple[Path, Path]:
    aliases = {
        station: f"station_{index:03d}"
        for index, station in enumerate(
            sorted(map(str, config["source_stations"])), start=1
        )
    }
    aliases[str(config["target_station"])] = "target_station"
    sanitized = []
    for record in records:
        request = record.get("request", {})
        if request.get("final_test"):
            continue
        result = record.get("result", {})
        sanitized.append(
            {
                "request": {
                    "stage": str(request.get("stage", "")),
                    "mode": str(request.get("mode", "")),
                    "held_out_station": aliases.get(
                        str(request.get("held_out_station", "")),
                        "station_unknown",
                    ),
                    "seed": int(request.get("seed", 0)),
                    "implementation_hash": str(
                        request.get("implementation_hash", "")
                    )[:12],
                    "final_test": False,
                },
                "result": {
                    "status": str(result.get("status", "failed")),
                    "score": result.get("score"),
                    "runtime_seconds": result.get("runtime_seconds"),
                    "diagnostics": _aggregate_only(
                        result.get("diagnostics", {})
                    ),
                    "per_month": _aggregate_only(
                        result.get("per_month", {})
                    ),
                },
            }
        )

    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "feedback.json"
    markdown_path = destination / "feedback.md"
    json_path.write_text(
        json.dumps(
            {
                "privacy": (
                    "station identifiers are pseudonymized; final target "
                    "records, code, paths, fingerprints, rows, and arrays are excluded"
                ),
                "records": sanitized,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(
        build_report(sanitized, config), encoding="utf-8"
    )
    return markdown_path, json_path


def _aggregate_only(value: Any):
    if isinstance(value, dict):
        return {
            str(key): cleaned
            for key, child in value.items()
            if (cleaned := _aggregate_only(child)) is not None
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 24:
            return None
        cleaned = [_aggregate_only(child) for child in value]
        return [child for child in cleaned if child is not None]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    return None
