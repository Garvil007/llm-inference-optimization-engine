"""Aggregation policies, stated explicitly rather than left implicit in the math:

1. Warmup exclusion: observations with `is_warmup=True` (the first `warmup_count`
   requests of a run) are excluded from every aggregate entirely -- they measure
   cold-start effects (model/connection setup), not steady-state performance, and
   including them would bias percentiles upward. They remain in the raw
   observation list.

2. Failed requests: excluded from every per-metric aggregate (ttft_ms, latency_ms,
   tpot_ms, throughput), but counted in `failure_rate`, which is computed over all
   non-warmup requests (`failed / measured`). A failed request's partial data
   (e.g. a ttft recorded before a mid-stream error) is never used in the
   aggregates -- only successful requests contribute measurements.

3. Missing token counts: handled per-metric, not per-request. If
   `output_token_count` is None (counter unavailable) or `tpot_ms`/
   `output_tokens_per_second` is None (fewer than 2 output tokens, see
   RequestObservation), that *specific* metric's sample list skips the
   observation -- but the same observation's ttft_ms/latency_ms still
   contribute to those aggregates if present. No request is dropped wholesale
   for a single missing field.

4. Percentile calculation: linear interpolation between the two closest ranks
   (the same method as numpy's default `percentile(..., method="linear")`),
   computed over the sorted sample. With a sample of size 1, every percentile
   equals that one value. With an empty sample, MetricStats has count=0 and
   every statistic is None (not 0 -- 0 would imply a measured zero duration).

5. Standard deviation: sample stdev (ddof=1), which needs >= 2 data points;
   with exactly 1 data point stddev is reported as 0.0 (no evidence of
   variance from a single observation, which is meaningfully different from
   "no data" -- hence None only when count == 0).
"""

from __future__ import annotations

import math
import statistics

from inference_optimizer.benchmark.models import BenchmarkSummary, MetricStats, RequestObservation


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * fraction
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    lower_weight = sorted_values[lower] * (upper - rank)
    upper_weight = sorted_values[upper] * (rank - lower)
    return lower_weight + upper_weight


def _stats(values: list[float]) -> MetricStats:
    if not values:
        return MetricStats(count=0)
    sorted_values = sorted(values)
    return MetricStats(
        count=len(values),
        p50=_percentile(sorted_values, 0.50),
        p95=_percentile(sorted_values, 0.95),
        p99=_percentile(sorted_values, 0.99),
        mean=statistics.fmean(values),
        stddev=statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def summarize(observations: list[RequestObservation]) -> BenchmarkSummary:
    measured = [observation for observation in observations if not observation.is_warmup]
    successful = [observation for observation in measured if observation.success]
    failed_count = len(measured) - len(successful)

    ttft_values = [o.ttft_ms for o in successful if o.ttft_ms is not None]
    latency_values = [o.latency_ms for o in successful]
    tpot_values = [o.tpot_ms for o in successful if o.tpot_ms is not None]
    throughput_values = [
        o.output_tokens_per_second for o in successful if o.output_tokens_per_second is not None
    ]

    return BenchmarkSummary(
        total_requests=len(observations),
        warmup_excluded=len(observations) - len(measured),
        measured_requests=len(measured),
        successful_requests=len(successful),
        failed_requests=failed_count,
        failure_rate=(failed_count / len(measured)) if measured else 0.0,
        ttft_ms=_stats(ttft_values),
        latency_ms=_stats(latency_values),
        tpot_ms=_stats(tpot_values),
        throughput_tokens_per_second=_stats(throughput_values),
    )
