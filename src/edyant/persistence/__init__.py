"""Persistence module for edyant."""
from .api import Episode, MemoryHit, MemoryStore, NullMemoryStore
from .memory_adapter import MemoryAugmentedAdapter
from .storage import SqliteMemoryStore
from .config import default_data_dir
from .types import ModelOutput
from .adapters import (
    AdapterError,
    ModelAdapter,
    OllamaAdapter,
    available_adapters,
    create_adapter,
    get_adapter,
    lazy_register,
    register_adapter,
)

__all__ = [
    "Episode",
    "MemoryHit",
    "MemoryStore",
    "NullMemoryStore",
    "MemoryAugmentedAdapter",
    "SqliteMemoryStore",
    "ModelOutput",
    "AdapterError",
    "ModelAdapter",
    "OllamaAdapter",
    "available_adapters",
    "create_adapter",
    "get_adapter",
    "lazy_register",
    "register_adapter",
    "default_data_dir",
]
