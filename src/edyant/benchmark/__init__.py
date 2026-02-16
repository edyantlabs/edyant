"""Benchmarking toolkit for edyant.

Provides adapters, dataset loaders, evaluators, runners, and result writers.
Use this module as the main entry point for running benchmark suites.
"""

from .adapters import (
    AdapterError,
    ModelAdapter,
    OllamaAdapter,
    OllamaJudgeAdapter,
    available_adapters,
    create_adapter,
)
from .datasets import Dataset, PromptExample, load_dataset
from .evaluators import Evaluator, NoopEvaluator, RefusalEvaluator, JudgeEvaluator
from .io import InMemoryResultWriter, JsonResultWriter, JsonlResultWriter, ResultWriter
from .runners import BenchmarkRunner
from .types import EvaluationResult, ModelOutput, RunRecord, summarize_results

__all__ = [
    "AdapterError",
    "BenchmarkRunner",
    "Dataset",
    "Evaluator",
    "EvaluationResult",
    "InMemoryResultWriter",
    "JsonResultWriter",
    "JsonlResultWriter",
    "ModelAdapter",
    "ModelOutput",
    "NoopEvaluator",
    "JudgeEvaluator",
    "OllamaAdapter",
    "OllamaJudgeAdapter",
    "PromptExample",
    "RefusalEvaluator",
    "ResultWriter",
    "RunRecord",
    "available_adapters",
    "create_adapter",
    "load_dataset",
    "summarize_results",
]
