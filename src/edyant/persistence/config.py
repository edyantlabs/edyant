"""Configuration helpers for persistence backends."""
from __future__ import annotations

import os
from pathlib import Path


def default_data_dir() -> Path:
    """Return the base directory for persistence data on this machine.

    Resolution order (all optional):
    1. EDYANT_DATA_DIR
    2. XDG_DATA_HOME
    3. ~/.local/share/edyant
    """

    env_dir = os.environ.get("EDYANT_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    xdg_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_home).expanduser() if xdg_home else Path.home() / ".local" / "share"
    return (base / "edyant" / "persistence").resolve()
