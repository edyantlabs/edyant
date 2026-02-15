from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from ..types import ModelOutput


class AdapterError(RuntimeError):
    pass


class ModelAdapter(ABC):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> ModelOutput:
        raise NotImplementedError


_ADAPTERS: dict[str, type[ModelAdapter]] = {}


def register_adapter(key: str, adapter_cls: type[ModelAdapter]) -> None:
    if key in _ADAPTERS:
        raise ValueError(f"Adapter already registered: {key}")
    _ADAPTERS[key] = adapter_cls


def get_adapter(key: str) -> type[ModelAdapter]:
    if key not in _ADAPTERS:
        available = ", ".join(sorted(_ADAPTERS))
        raise KeyError(f"Unknown adapter '{key}'. Available: {available}")
    return _ADAPTERS[key]


def create_adapter(key: str, **kwargs: Any) -> ModelAdapter:
    adapter_cls = get_adapter(key)
    return adapter_cls(**kwargs)


def available_adapters() -> list[str]:
    return sorted(_ADAPTERS)


def lazy_register(key: str, loader: Callable[[], type[ModelAdapter]]) -> None:
    if key in _ADAPTERS:
        return
    _ADAPTERS[key] = loader()
