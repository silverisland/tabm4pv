#!/usr/bin/env python3
"""Aggregate-only checks for array-index and canonical-time alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(metadata: dict) -> dict:
    points_per_hour = int(metadata["points_per_hour"])
    horizon = int(metadata["target_future_hour"])
    target_index = int(metadata["target_index"])
    future_zero_offset_minutes = int(metadata["future_zero_offset_minutes"])
    actual_minutes = future_zero_offset_minutes + target_index * (
        60 // points_per_hour
    )
    expected_minutes = horizon * 60
    return {
        "expected_target_offset_minutes": expected_minutes,
        "actual_target_offset_minutes": actual_minutes,
        "target_index_aligned": actual_minutes == expected_minutes,
        "history_last_offset_minutes": int(
            metadata["history_last_offset_minutes"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_json")
    args = parser.parse_args()
    metadata = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
    print(json.dumps(audit(metadata), indent=2))


if __name__ == "__main__":
    main()
