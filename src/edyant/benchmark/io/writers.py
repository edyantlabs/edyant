"""Result writers for benchmark runs."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..types import RunRecord


class ResultWriter(ABC):
    """Abstract base class for result writers."""

    @abstractmethod
    def write(self, record: RunRecord) -> None:
        """Write a single run record."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        """Close the writer if it owns resources."""
        return None


class JsonlResultWriter(ResultWriter):
    """Write results as newline-delimited JSON."""

    def __init__(self, path: str | Path, append: bool = False) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._handle = self._path.open(mode, encoding="utf-8")

    def write(self, record: RunRecord) -> None:
        """Append a single record to the JSONL file."""
        payload = json.dumps(record.to_dict(), ensure_ascii=False)
        self._handle.write(payload + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the JSONL file handle."""
        if not self._handle.closed:
            self._handle.close()


class JsonResultWriter(ResultWriter):
    """Write results as a single JSON array."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []

    def write(self, record: RunRecord) -> None:
        """Collect a record for later JSON serialization."""
        self._records.append(record.to_dict())

    def close(self) -> None:
        """Write all collected records to the JSON file."""
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(self._records, handle, ensure_ascii=False, indent=2)


class InMemoryResultWriter(ResultWriter):
    """Store result records in memory."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: RunRecord) -> None:
        """Append a record to the in-memory list."""
        self.records.append(record.to_dict())
