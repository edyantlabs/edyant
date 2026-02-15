from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class PromptExample:
    id: str
    prompt: str
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Dataset:
    name: str
    examples: list[PromptExample]
    metadata: dict[str, Any] = field(default_factory=dict)

    def size(self) -> int:
        return len(self.examples)


@dataclass(frozen=True)
class ModelOutput:
    text: str
    raw: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    name: str
    score: float | None = None
    passed: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunRecord:
    timestamp: str
    run_id: str
    model: str
    dataset: str
    example_id: str
    category: str | None
    prompt: str
    response: str
    response_raw: dict[str, Any] | None = None
    latency_ms: int | None = None
    evaluations: list[EvaluationResult] = field(default_factory=list)
    example_metadata: dict[str, Any] = field(default_factory=dict)
    run_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evaluations"] = [asdict(item) for item in self.evaluations]
        return data


def summarize_results(records: Iterable[RunRecord]) -> dict[str, Any]:
    total = 0
    passed = 0
    failed = 0
    unknown = 0

    for record in records:
        total += 1
        if not record.evaluations:
            unknown += 1
            continue
        result_flags = [e.passed for e in record.evaluations if e.passed is not None]
        if not result_flags:
            unknown += 1
        elif all(result_flags):
            passed += 1
        else:
            failed += 1

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "unknown": unknown,
    }
