"""Adapter interfaces and registry for model providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from ..types import ModelOutput


class AdapterError(RuntimeError):
    """Raised when an adapter cannot complete a request."""

    pass


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Return the model name associated with this adapter."""
        return self._name

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> ModelOutput:
        """Generate a model response for a prompt."""
        raise NotImplementedError


_ADAPTERS: dict[str, type[ModelAdapter]] = {}


def register_adapter(key: str, adapter_cls: type[ModelAdapter]) -> None:
    """Register a model adapter class under a key."""
    if key in _ADAPTERS:
        raise ValueError(f"Adapter already registered: {key}")
    _ADAPTERS[key] = adapter_cls


def get_adapter(key: str) -> type[ModelAdapter]:
    """Fetch a registered adapter class by key."""
    if key not in _ADAPTERS:
        available = ", ".join(sorted(_ADAPTERS))
        raise KeyError(f"Unknown adapter '{key}'. Available: {available}")
    return _ADAPTERS[key]


def create_adapter(key: str, **kwargs: Any) -> ModelAdapter:
    """Instantiate a registered adapter."""
    adapter_cls = get_adapter(key)
    return adapter_cls(**kwargs)


def available_adapters() -> list[str]:
    """Return a sorted list of available adapter keys."""
    return sorted(_ADAPTERS)


def lazy_register(key: str, loader: Callable[[], type[ModelAdapter]]) -> None:
    """Register an adapter lazily via a loader function."""
    if key in _ADAPTERS:
        return
    _ADAPTERS[key] = loader()
