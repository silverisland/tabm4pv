from __future__ import annotations

from datetime import datetime


def log(message: str) -> None:
    """Print a timestamped progress message and flush it immediately."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"[{timestamp}] [province-tabm] {message}", flush=True)
