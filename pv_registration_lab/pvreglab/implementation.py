from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def implementation_hash(editable_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in _files(editable_dir):
        relative = path.relative_to(editable_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def snapshot_implementation(editable_dir: Path, state_dir: Path) -> Path:
    current_hash = implementation_hash(editable_dir)
    destination = state_dir / "implementations" / current_hash
    snapshot_dir = destination / "editable"
    if destination.exists():
        if implementation_hash(snapshot_dir) != current_hash:
            raise RuntimeError(f"Corrupted implementation snapshot: {current_hash}")
        return destination
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    for source in _files(editable_dir):
        relative = source.relative_to(editable_dir)
        target = snapshot_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    metadata = {
        "implementation_hash": current_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [
            path.relative_to(editable_dir).as_posix()
            for path in _files(editable_dir)
        ],
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return destination


def restore_implementation(
    editable_dir: Path, state_dir: Path, requested_hash: str
) -> str:
    snapshot_dir = state_dir / "implementations" / requested_hash / "editable"
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Unknown implementation: {requested_hash}")
    actual_hash = implementation_hash(snapshot_dir)
    if actual_hash != requested_hash:
        raise RuntimeError("Implementation snapshot hash does not match its name")

    expected_files = {
        path.relative_to(snapshot_dir) for path in _files(snapshot_dir)
    }
    for current in reversed(list(_files(editable_dir))):
        relative = current.relative_to(editable_dir)
        if relative not in expected_files:
            current.unlink()
    for source in _files(snapshot_dir):
        relative = source.relative_to(snapshot_dir)
        target = editable_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    restored_hash = implementation_hash(editable_dir)
    if restored_hash != requested_hash:
        raise RuntimeError("Restored implementation failed hash verification")
    return restored_hash


def list_implementations(state_dir: Path) -> list[dict]:
    root = state_dir / "implementations"
    if not root.exists():
        return []
    result = []
    for directory in sorted(root.iterdir()):
        metadata = directory / "metadata.json"
        if metadata.exists():
            result.append(json.loads(metadata.read_text(encoding="utf-8")))
    return result


def _files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(directory)
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and not path.name.endswith(".pyc")
    )
