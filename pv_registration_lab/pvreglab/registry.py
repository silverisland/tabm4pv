from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class Registry:
    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / "experiments.jsonl"

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def completed_ids(self) -> set[str]:
        return {
            str(record["request"]["experiment_id"])
            for record in self.records()
            if record.get("result", {}).get("status") == "ok"
        }

    def select(self, **conditions: Any) -> Iterable[dict[str, Any]]:
        for record in self.records():
            request = record.get("request", {})
            if all(request.get(key) == value for key, value in conditions.items()):
                yield record
