from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import EvaluationResult, ModelOutput, PromptExample


class Evaluator(ABC):
    name: str

    @abstractmethod
    def evaluate(self, example: PromptExample, output: ModelOutput) -> EvaluationResult:
        raise NotImplementedError


class NoopEvaluator(Evaluator):
    name = "noop"

    def evaluate(self, example: PromptExample, output: ModelOutput) -> EvaluationResult:
        return EvaluationResult(name=self.name)
