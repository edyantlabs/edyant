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

    def __init__(self, path: str | Path, append: bool = False, exclude_keys: set[str] | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._handle = self._path.open(mode, encoding="utf-8")
        self._exclude_keys = set(exclude_keys or [])

    def write(self, record: RunRecord) -> None:
        """Append a single record to the JSONL file."""
        payload_dict = record.to_dict()
        self._promote_judge_metadata(payload_dict)
        for key in self._exclude_keys:
            payload_dict.pop(key, None)
        payload = json.dumps(payload_dict, ensure_ascii=False)
        self._handle.write(payload + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the JSONL file handle."""
        if not self._handle.closed:
            self._handle.close()

    @staticmethod
    def _promote_judge_metadata(payload: dict[str, Any]) -> None:
        """Move judge metadata out of judge_raw and drop judge_raw."""
        evaluations = payload.get("evaluations")
        if not isinstance(evaluations, list):
            return

        for evaluation in evaluations:
            if not isinstance(evaluation, dict):
                continue
            details = evaluation.get("details")
            if not isinstance(details, dict):
                continue
            judge_raw = details.get("judge_raw")
            if not isinstance(judge_raw, dict):
                continue

            # Copy selected fields so they remain available after removing the raw payload.
            if "model" in judge_raw:
                details["model"] = judge_raw.get("model")
            if "created_at" in judge_raw:
                details["created_at"] = judge_raw.get("created_at")
            if "done" in judge_raw:
                details["done"] = judge_raw.get("done")
            if "done_reason" in judge_raw:
                details["done_reason"] = judge_raw.get("done_reason")

            # Remove bulky raw payload now that key fields are preserved.
            details.pop("judge_raw", None)


class JsonResultWriter(ResultWriter):
    """Write results as a single JSON array."""

    def __init__(self, path: str | Path, exclude_keys: set[str] | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []
        self._exclude_keys = set(exclude_keys or [])

    def write(self, record: RunRecord) -> None:
        """Collect a record for later JSON serialization."""
        payload = record.to_dict()
        JsonlResultWriter._promote_judge_metadata(payload)
        for key in self._exclude_keys:
            payload.pop(key, None)
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
