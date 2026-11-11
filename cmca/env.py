"""Credentials come from the environment or from a dotenv file; values are never printed."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_FILE = Path(".env")


def load_env(path: Path | None = None) -> None:
    """Set variables from a dotenv file without overriding ones already set.

    With no path, ./.env is read if it exists."""
    path = path or (DEFAULT_ENV_FILE if DEFAULT_ENV_FILE.exists() else None)
    if path is None:
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
