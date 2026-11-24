"""Inference performance measurement: a benchmark client over the streaming API,
plus raw-observation storage and aggregate metrics (p50/p95/p99/mean/stddev,
failure rate). See docs/benchmarking.md for metric definitions and policies.
"""

from inference_optimizer.benchmark.aggregate import summarize
from inference_optimizer.benchmark.client import BenchmarkClient
from inference_optimizer.benchmark.models import (
    BenchmarkResult,
    BenchmarkSummary,
    MetricStats,
    RequestObservation,
)
from inference_optimizer.benchmark.token_counter import (
    FixedTokenCounter,
    HFTokenizerCounter,
    TokenCounter,
    WhitespaceApproxCounter,
)

__all__ = [
    "BenchmarkClient",
    "BenchmarkResult",
    "BenchmarkSummary",
    "MetricStats",
    "RequestObservation",
    "FixedTokenCounter",
    "HFTokenizerCounter",
    "TokenCounter",
    "WhitespaceApproxCounter",
    "summarize",
]
