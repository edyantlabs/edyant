"""Ollama adapter for local model execution (persistence namespace)."""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from .base import AdapterError, ModelAdapter, register_adapter
from edyant.persistence.types import ModelOutput


class OllamaAdapter(ModelAdapter):
    """Adapter for the Ollama HTTP API."""

    def __init__(
        self,
        model: str,
        url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
    ) -> None:
        super().__init__(model)
        self._model = model
        self._url = url or os.getenv("OLLAMA_API_URL")
        if not self._url:
            raise ValueError(
                "OllamaAdapter requires a URL via the 'url' argument or the OLLAMA_API_URL environment variable"
            )
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_sleep = retry_sleep

    def generate(self, prompt: str, **kwargs: Any) -> ModelOutput:
        """Send a prompt to Ollama and return the response."""
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        payload.update(kwargs)

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                req = request.Request(self._url, data=body, headers=headers)
                with request.urlopen(req, timeout=self._timeout) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                text = raw.get("response", "")
                return ModelOutput(text=text, raw=raw)
            except (HTTPError, URLError, socket.timeout, ValueError) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    time.sleep(self._retry_sleep)

        raise AdapterError(f"Ollama request failed after {self._max_retries} attempts") from last_error


register_adapter("ollama", OllamaAdapter)
