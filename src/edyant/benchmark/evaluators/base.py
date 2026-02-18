"""Evaluator interfaces for grading benchmark responses."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import EvaluationResult, ModelOutput, PromptItem


class Evaluator(ABC):
    """Base class for response evaluators."""

    name: str

    @abstractmethod
    def evaluate(self, prompt_item: PromptItem, output: ModelOutput) -> EvaluationResult:
        """Evaluate a single model output."""
        raise NotImplementedError


class NoopEvaluator(Evaluator):
    """Evaluator that records no score and always returns unknown."""

    name = "noop"

    def evaluate(self, prompt_item: PromptItem, output: ModelOutput) -> EvaluationResult:
        """Return an empty evaluation result."""
        return EvaluationResult(name=self.name)
