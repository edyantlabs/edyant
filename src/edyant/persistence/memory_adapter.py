"""Adapter wrapper that injects persistence-backed memory into any ModelAdapter."""
from __future__ import annotations

from typing import Callable, Iterable

from edyant.persistence.adapters.base import ModelAdapter
from edyant.persistence.types import ModelOutput

from .api import MemoryHit, MemoryStore


Formatter = Callable[[str, Iterable[MemoryHit]], str]


def default_formatter(prompt: str, hits: Iterable[MemoryHit]) -> str:
    """Prepend lightweight context to the original prompt."""
    context_lines = []
    for hit in hits:
        context_lines.append(f"- {hit.text}")
    if not context_lines:
        return prompt
    context_block = "\n".join(context_lines)
    return f"Context (from memory):\n{context_block}\n\nUser:\n{prompt}"


class MemoryAugmentedAdapter(ModelAdapter):
    def __init__(
        self,
        base_adapter: ModelAdapter,
        store: MemoryStore,
        context_k: int = 5,
        formatter: Formatter | None = None,
    ) -> None:
        super().__init__(name=f"{base_adapter.name}+mem")
        self._base = base_adapter
        self._store = store
        self._context_k = context_k
        self._format = formatter or default_formatter

    def generate(self, prompt: str, **kwargs) -> ModelOutput:
        hits = self._store.retrieve(prompt, top_k=self._context_k)
        enriched_prompt = self._format(prompt, hits)
        output = self._base.generate(enriched_prompt, **kwargs)
        episode_id = self._store.record_episode(
            prompt=prompt,
            output=output,
            metadata={"adapter": self._base.name},
        )
        if hits:
            self._store.update_edges(episode_id, [h.node_id for h in hits], weight=1.0)
        return output

    def close(self) -> None:
        close = getattr(self._base, "close", None)
        if callable(close):
            close()
        self._store.close()
