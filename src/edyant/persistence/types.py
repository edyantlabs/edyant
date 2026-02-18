"""Shared dataclasses for persistence-facing model IO."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelOutput:
    """Model response payload with optional raw provider data."""

    text: str
    raw: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)
