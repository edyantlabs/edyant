"""Result writers for benchmark runs."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


def _redact_fields(obj: Any, paths: list[list[str]]) -> None:
    """In-place redaction of specified dotted paths.

    Supports wildcard '*' to apply to all items of a list. Dict navigation only.
    """

    for path in paths:
        _redact_path(obj, path)


def _redact_path(obj: Any, path: list[str]) -> None:
    if not path:
        return

    key = path[0]
    rest = path[1:]

    if isinstance(obj, dict):
        if key == "*":
            for value in obj.values():
                _redact_path(value, rest)
            return
        if key not in obj:
            return
        if rest:
            _redact_path(obj[key], rest)
        else:
            obj.pop(key, None)
        return

    if isinstance(obj, list):
        if key == "*":
            for item in obj:
                _redact_path(item, rest)
        else:
            # numeric index is not supported; treat as wildcard fallback
            for item in obj:
                _redact_path(item, path)


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

    def __init__(
        self,
        path: str | Path,
        append: bool = False,
        exclude_keys: set[str] | None = None,
        redact_paths: list[list[str]] | None = None,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._handle = self._path.open(mode, encoding="utf-8")
        self._exclude_keys = set(exclude_keys or [])
        self._redact_paths = redact_paths or []

    def write(self, record: RunRecord) -> None:
        """Append a single record to the JSONL file."""
        payload_dict = record.to_dict()
        for key in self._exclude_keys:
            payload_dict.pop(key, None)
        if self._redact_paths:
            _redact_fields(payload_dict, self._redact_paths)
        payload = json.dumps(payload_dict, ensure_ascii=False)
        self._handle.write(payload + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the JSONL file handle."""
        if not self._handle.closed:
            self._handle.close()


class JsonResultWriter(ResultWriter):
    """Write results as a single JSON array."""

    def __init__(
        self,
        path: str | Path,
        exclude_keys: set[str] | None = None,
        redact_paths: list[list[str]] | None = None,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []
        self._exclude_keys = set(exclude_keys or [])
        self._redact_paths = redact_paths or []

    def write(self, record: RunRecord) -> None:
        """Collect a record for later JSON serialization."""
        payload = record.to_dict()
        for key in self._exclude_keys:
            payload.pop(key, None)
        if self._redact_paths:
            _redact_fields(payload, self._redact_paths)
        self._records.append(payload)

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
