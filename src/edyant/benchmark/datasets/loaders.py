"""Dataset loaders for JSON, JSONL, and CSV prompt suites."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from ..types import Dataset, PromptExample


def load_dataset(path: str | Path, name: str | None = None) -> Dataset:
    """Load a dataset from a file path, inferring format by extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".json":
        return _load_json(path, name=name)
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path, name=name)
    if path.suffix.lower() == ".csv":
        return _load_csv(path, name=name)

    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _load_json(path: Path, name: str | None = None) -> Dataset:
    """Load a dataset from a JSON list or object with examples."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    dataset_name = name or path.stem
    metadata: dict[str, Any] = {}

    if isinstance(payload, dict):
        metadata = dict(payload.get("metadata") or {})
        dataset_name = payload.get("name") or dataset_name
        raw_examples = payload.get("examples")
        if raw_examples is None:
            raise ValueError("JSON dataset requires an 'examples' key when using object format")
    elif isinstance(payload, list):
        raw_examples = payload
    else:
        raise ValueError("JSON dataset must be a list or an object with an 'examples' key")

    examples = _normalize_examples(raw_examples)
    return Dataset(name=dataset_name, examples=examples, metadata=metadata)


def _load_jsonl(path: Path, name: str | None = None) -> Dataset:
    """Load a dataset from JSON Lines (one object per line)."""
    examples: list[PromptExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {index}") from exc
            examples.append(_normalize_example(payload, index))

    dataset_name = name or path.stem
    return Dataset(name=dataset_name, examples=examples, metadata={})


def _load_csv(path: Path, name: str | None = None) -> Dataset:
    """Load a dataset from CSV with a prompt column."""
    examples: list[PromptExample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, 1):
            examples.append(_normalize_example(row, index))

    dataset_name = name or path.stem
    return Dataset(name=dataset_name, examples=examples, metadata={})


def _normalize_examples(raw_examples: Iterable[dict[str, Any]]) -> list[PromptExample]:
    """Normalize a list of dict payloads into PromptExample objects."""
    examples: list[PromptExample] = []
    for index, payload in enumerate(raw_examples, 1):
        examples.append(_normalize_example(payload, index))
    return examples


def _normalize_example(payload: dict[str, Any], index: int) -> PromptExample:
    """Normalize a single payload dict into a PromptExample."""
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset item {index} must be an object")

    if "prompt" not in payload:
        raise ValueError(f"Dataset item {index} missing required 'prompt' field")

    example_id = str(payload.get("id") or f"item_{index}")
    category = payload.get("category")

    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"id", "prompt", "category"}
    }

    return PromptExample(
        id=example_id,
        prompt=str(payload["prompt"]),
        category=category,
        metadata=metadata,
    )
