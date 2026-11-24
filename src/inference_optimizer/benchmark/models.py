"""Typed schemas for benchmark measurements.

RequestObservation is the raw per-request record (one per prompt sent). MetricStats
and BenchmarkSummary are the aggregate computed from a set of observations. Both are
stored -- raw observations let later days re-aggregate with different policies
(e.g. a different warmup count) without re-running the benchmark.

Terminology (see docs/benchmarking.md for the full writeup):
- TTFT (time to first token): start -> first non-empty generated content.
- End-to-end latency: start -> stream completion (includes TTFT).
- TPOT (time per output token): the average per-token cost of the generation phase
  *after* the first token, i.e. (latency - ttft) / (token_count - 1). Needs >= 2
  output tokens; this is a different quantity from "inter-token latency" (the
  distribution of individual gaps between tokens), which this system does not
  measure because SSE chunk boundaries don't line up with token boundaries.
- Throughput: token_count / (latency_ms / 1000), tokens per second over the whole
  request including TTFT (a vendor-standard definition, not generation-phase-only).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

TokenCountMethod = Literal["tokenizer", "fixed", "approximate", "unavailable"]


class RequestObservation(BaseModel):
    request_index: int
    is_warmup: bool = False
    success: bool
    error: str | None = None

    started_at: datetime
    """Wall-clock time the request was issued (for human-readable provenance only;
    all durations below are computed from time.perf_counter(), not this field)."""

    start_perf: float
    first_token_perf: float | None = None
    end_perf: float

    ttft_ms: float | None = None
    latency_ms: float

    output_text_length: int
    """len() of the concatenated output text -- NOT a token count; kept for
    sanity-checking the token counter's output, never used in aggregates."""

    output_token_count: int | None = None
    token_count_method: TokenCountMethod = "unavailable"

    tpot_ms: float | None = None
    output_tokens_per_second: float | None = None


class MetricStats(BaseModel):
    count: int
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    mean: float | None = None
    stddev: float | None = None


class BenchmarkSummary(BaseModel):
    total_requests: int
    warmup_excluded: int
    measured_requests: int
    successful_requests: int
    failed_requests: int
    failure_rate: float

    ttft_ms: MetricStats
    latency_ms: MetricStats
    tpot_ms: MetricStats
    throughput_tokens_per_second: MetricStats


class BenchmarkResult(BaseModel):
    observations: list[RequestObservation]
    summary: BenchmarkSummary
    config: dict[str, object]
    """Run conditions (base_url, prompt, params, warmup_count, token_count_method
    requested, ...) -- recorded for reproducibility per the project's benchmarking
    rules, not used in any calculation."""
