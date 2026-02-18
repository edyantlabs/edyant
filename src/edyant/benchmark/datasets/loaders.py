"""Dataset loaders for JSON, JSONL, and CSV prompt suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..types import Dataset, PromptItem


def load_dataset(path: str | Path, name: str | None = None) -> Dataset:
    """Load a dataset from a file path, inferring format by extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".json":
        return _load_json(path, name=name)
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path, name=name)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _load_json(path: Path, name: str | None = None) -> Dataset:
    """Load a dataset from a JSON list or object with prompts."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    dataset_name = name or path.stem
    metadata: dict[str, Any] = {}

    if isinstance(payload, dict):
        metadata = dict(payload.get("metadata") or {})
        dataset_name = payload.get("name") or dataset_name
        raw_prompts = payload.get("examples")
        if raw_prompts is None:
            raise ValueError("JSON dataset requires an 'examples' key when using object format")
    elif isinstance(payload, list):
        raw_prompts = payload
    else:
        raise ValueError("JSON dataset must be a list or an object with an 'examples' key")

    prompts = _normalize_prompts(raw_prompts)
    return Dataset(name=dataset_name, prompts=prompts, metadata=metadata)


def _load_jsonl(path: Path, name: str | None = None) -> Dataset:
    """Load a dataset from JSON Lines (one object per line)."""
    prompts: list[PromptItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {index}") from exc
            prompts.append(_normalize_prompt(payload, index))

    dataset_name = name or path.stem
    return Dataset(name=dataset_name, prompts=prompts, metadata={})


def _normalize_prompts(raw_prompts: Iterable[dict[str, Any]]) -> list[PromptItem]:
    """Normalize a list of dict payloads into PromptItem objects."""
    prompts: list[PromptItem] = []
    for index, payload in enumerate(raw_prompts, 1):
        prompts.append(_normalize_prompt(payload, index))
    return prompts


def _normalize_prompt(payload: dict[str, Any], index: int) -> PromptItem:
    """Normalize a single payload dict into a PromptItem."""
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset item {index} must be an object")

    if "prompt" not in payload:
        raise ValueError(f"Dataset item {index} missing required 'prompt' field")

    prompt_id = str(payload.get("id") or f"item_{index}")
    category = payload.get("category")

    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"id", "prompt", "category"}
    }

    return PromptItem(
        id=prompt_id,
        prompt=str(payload["prompt"]),
        category=category,
        metadata=metadata,
    )
