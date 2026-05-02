"""
Evaluation Pipeline — Intrinsic & Extrinsic Metrics
Evaluation & Testing Lead: Leen Safi (STU ID: 2309116117)
"""

import math
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional


@dataclass
class IntrinsicMetrics:
    """Token-level language model quality metrics."""
    perplexity: float
    avg_log_likelihood: float
    token_count: int


@dataclass
class ExtrinsicMetrics:
    """Task-completion and end-to-end performance metrics."""
    task_name: str
    success: bool
    latency_ms: float
    response_length: int
    safety_passed: bool
    risk_score: float


@dataclass
class EvaluationResult:
    intrinsic: Optional[IntrinsicMetrics] = None
    extrinsic: list[ExtrinsicMetrics] = field(default_factory=list)

    @property
    def task_success_rate(self) -> float:
        if not self.extrinsic:
            return 0.0
        return sum(1 for r in self.extrinsic if r.success) / len(self.extrinsic)

    @property
    def avg_latency_ms(self) -> float:
        if not self.extrinsic:
            return 0.0
        return sum(r.latency_ms for r in self.extrinsic) / len(self.extrinsic)

    def to_dict(self) -> dict:
        return {
            "intrinsic": asdict(self.intrinsic) if self.intrinsic else None,
            "extrinsic_results": [asdict(r) for r in self.extrinsic],
            "summary": {
                "task_success_rate": round(self.task_success_rate, 4),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "total_tasks": len(self.extrinsic),
            },
        }


class PerplexityEvaluator:
    """
    Intrinsic evaluation using pseudo-perplexity from token log-probabilities.
    Requires an LLM that returns logprobs (e.g., OpenAI with logprobs=True).
    """

    def compute_from_logprobs(self, log_probs: list[float]) -> IntrinsicMetrics:
        if not log_probs:
            raise ValueError("log_probs list is empty.")
        avg_ll = sum(log_probs) / len(log_probs)
        perplexity = math.exp(-avg_ll)
        return IntrinsicMetrics(
            perplexity=round(perplexity, 4),
            avg_log_likelihood=round(avg_ll, 6),
            token_count=len(log_probs),
        )


class TaskEvaluator:
    """
    Extrinsic evaluation — runs a task through the orchestrator and scores the result.
    Pass a `success_fn` that takes the agent response string and returns bool.
    """

    def __init__(self, orchestrator_run_fn: Callable[[str], dict]):
        self.run = orchestrator_run_fn

    def evaluate_task(
        self,
        task_name: str,
        query: str,
        success_fn: Callable[[str], bool],
    ) -> ExtrinsicMetrics:
        start = time.perf_counter()
        result = self.run(query)
        latency_ms = (time.perf_counter() - start) * 1000

        response = result.get("response", "")
        safety = result.get("safety", {})

        return ExtrinsicMetrics(
            task_name=task_name,
            success=success_fn(response),
            latency_ms=round(latency_ms, 2),
            response_length=len(response),
            safety_passed=safety.get("passed", True),
            risk_score=safety.get("risk_score", 0.0),
        )

    def run_suite(self, test_cases: list[dict]) -> list[ExtrinsicMetrics]:
        """
        test_cases: list of dicts with keys:
          - name: str
          - query: str
          - success_fn: Callable[[str], bool]
        """
        return [
            self.evaluate_task(tc["name"], tc["query"], tc["success_fn"])
            for tc in test_cases
        ]


class EvaluationRunner:
    """Aggregates intrinsic + extrinsic results and exports a JSON report."""

    def __init__(self):
        self.result = EvaluationResult()
        self.perplexity_evaluator = PerplexityEvaluator()

    def add_intrinsic(self, log_probs: list[float]) -> None:
        self.result.intrinsic = self.perplexity_evaluator.compute_from_logprobs(log_probs)

    def add_extrinsic(self, metrics: list[ExtrinsicMetrics]) -> None:
        self.result.extrinsic.extend(metrics)

    def export(self, path: str = "eval/report.json") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.result.to_dict(), f, indent=2)
        print(f"Evaluation report saved to {path}")

    def print_summary(self) -> None:
        summary = self.result.to_dict()["summary"]
        print("\n--- Evaluation Summary ---")
        print(f"  Task Success Rate : {summary['task_success_rate'] * 100:.1f}%")
        print(f"  Avg Latency       : {summary['avg_latency_ms']:.0f} ms")
        print(f"  Total Tasks Run   : {summary['total_tasks']}")
        if self.result.intrinsic:
            print(f"  Perplexity        : {self.result.intrinsic.perplexity}")
