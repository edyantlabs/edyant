"""Benchmark runner that orchestrates datasets, models, and evaluators."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Iterable

from ..adapters.base import ModelAdapter
from ..evaluators.base import Evaluator
from ..io.writers import ResultWriter
from ..types import Dataset, RunRecord


class BenchmarkRunner:
    """Run a dataset against a model adapter and evaluators."""

    def __init__(
        self,
        adapter: ModelAdapter,
        evaluators: Iterable[Evaluator] | None = None,
        run_id: str | None = None,
        run_metadata: dict | None = None,
        generation_kwargs: dict | None = None,
        throttle_seconds: float = 0.0,
    ) -> None:
        self._adapter = adapter
        self._evaluators = list(evaluators) if evaluators else []
        self._run_id = run_id or uuid.uuid4().hex
        self._run_metadata = dict(run_metadata or {})
        self._generation_kwargs = dict(generation_kwargs or {})
        self._throttle_seconds = throttle_seconds

    @property
    def run_id(self) -> str:
        """Return the run identifier."""
        return self._run_id

    def run(self, dataset: Dataset, writer: ResultWriter | None = None) -> list[RunRecord]:
        """Execute a dataset and return run records."""
        records: list[RunRecord] = []
        run_metadata = {
            "adapter": self._adapter.__class__.__name__,
            "dataset_metadata": dataset.metadata,
            "dataset_size": dataset.size(),
        }
        run_metadata.update(self._run_metadata)

        for example in dataset.examples:
            start = time.perf_counter()
            output = self._adapter.generate(example.prompt, **self._generation_kwargs)
            latency_ms = int((time.perf_counter() - start) * 1000)

            evaluations = [
                evaluator.evaluate(example, output) for evaluator in self._evaluators
            ]

            record = RunRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                run_id=self._run_id,
                model=self._adapter.name,
                dataset=dataset.name,
                example_id=example.id,
                category=example.category,
                prompt=example.prompt,
                response=output.text,
                response_raw=output.raw,
                latency_ms=latency_ms,
                evaluations=evaluations,
                example_metadata=example.metadata,
                run_metadata=run_metadata,
            )

            if writer is not None:
                writer.write(record)

            records.append(record)

            if self._throttle_seconds:
                time.sleep(self._throttle_seconds)

        if writer is not None:
            writer.close()

        return records
