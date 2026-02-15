"""Model adapter registry and built-in adapters."""

from .base import (
    AdapterError,
    ModelAdapter,
    available_adapters,
    create_adapter,
    get_adapter,
    register_adapter,
)
from .ollama import OllamaAdapter

__all__ = [
    "AdapterError",
    "ModelAdapter",
    "OllamaAdapter",
    "available_adapters",
    "create_adapter",
    "get_adapter",
    "register_adapter",
]
