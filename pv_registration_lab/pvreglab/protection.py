from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


MANIFEST_NAME = "protected_manifest.json"
CONFIG_LOCK_NAME = "config_lock.json"


def _iter_files(root: Path, relative_paths: Iterable[str]):
    for relative in relative_paths:
        path = (root / relative).resolve()
        if path.is_file():
            yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and "__pycache__" not in child.parts:
                    yield child
        else:
            raise FileNotFoundError(f"Protected path does not exist: {path}")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_manifest(
    root: Path, state_dir: Path, protected_paths: Iterable[str]
) -> Path:
    root = root.resolve()
    manifest = {
        str(path.relative_to(root)): _digest(path)
        for path in _iter_files(root, protected_paths)
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    destination = state_dir / MANIFEST_NAME
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return destination


def verify_manifest(root: Path, state_dir: Path) -> list[str]:
    manifest_path = state_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return ["Protected manifest is missing; run controller.py init"]
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for relative, digest in expected.items():
        path = root / relative
        if not path.exists():
            errors.append(f"missing protected file: {relative}")
        elif _digest(path) != digest:
            errors.append(f"modified protected file: {relative}")
    return errors


def create_config_lock(config_path: Path, state_dir: Path) -> Path:
    payload = {
        "path": str(config_path.resolve()),
        "sha256": _digest(config_path.resolve()),
    }
    destination = state_dir / CONFIG_LOCK_NAME
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return destination


def verify_config_lock(config_path: Path, state_dir: Path) -> list[str]:
    lock_path = state_dir / CONFIG_LOCK_NAME
    if not lock_path.exists():
        return ["Config lock is missing; run controller.py init"]
    expected = json.loads(lock_path.read_text(encoding="utf-8"))
    actual_path = config_path.resolve()
    if str(actual_path) != expected.get("path"):
        return ["A different config file is being used after initialization"]
    if _digest(actual_path) != expected.get("sha256"):
        return ["The initialized config file was modified"]
    return []
