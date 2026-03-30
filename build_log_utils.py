"""Append-only build log helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


BUILD_LOG_PATH = Path("build.log")


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_build_log_entry(action: str, target: str, description: str) -> None:
    """Append a formatted entry to the build log."""
    BUILD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BUILD_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_timestamp()}] {action:<8} {target}  — {description}\n")


def log_file_event(action: str, path: str | Path, description: str) -> None:
    """Log a file create or modify event."""
    append_build_log_entry(action, str(path), description)


def log_run_event(command: str, description: str) -> None:
    """Log a pipeline execution event."""
    append_build_log_entry("RUN", command, description)
