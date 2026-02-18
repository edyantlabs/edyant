"""Public interfaces and data types for the persistence layer."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence

from edyant.persistence.types import ModelOutput


@dataclass
class MemoryHit:
    """A retrieved memory node."""

    node_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """Stored prompt/response pair with metadata."""

    node_id: str
    prompt: str
    response: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryStore(abc.ABC):
    """Abstract memory interface used by the framework and adapters."""

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryHit]:
        """Return the top_k memory hits for a query."""

    @abc.abstractmethod
    def record_episode(
        self,
        prompt: str,
        output: ModelOutput,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist a prompt/response pair and return the new node id."""

    @abc.abstractmethod
    def update_edges(
        self,
        source_id: str,
        related_ids: Sequence[str],
        weight: float = 1.0,
    ) -> None:
        """Adjust relationship weights between nodes."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources held by the store."""


class NullMemoryStore(MemoryStore):
    """No-op store that disables persistence (useful for tests)."""

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryHit]:
        return []

    def record_episode(
        self,
        prompt: str,
        output: ModelOutput,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return ""

    def update_edges(self, source_id: str, related_ids: Sequence[str], weight: float = 1.0) -> None:
        return None

    def close(self) -> None:  # pragma: no cover - nothing to close
        return None
