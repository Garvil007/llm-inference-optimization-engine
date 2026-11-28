"""Recomputes BenchmarkSummary from raw per-request observations (reusing Day
4's summarize(), unmodified) and compares it against the stored aggregate for
the same repeat -- the brief's "validate numerical aggregates from stored raw
requests." A mismatch means the stored aggregate doesn't match what the raw
data actually supports, which is always worth surfacing, never silently
trusted.
"""

from __future__ import annotations

from inference_optimizer.baseline.models import BaselineReport
from inference_optimizer.benchmark import summarize
from inference_optimizer.benchmark.models import MetricStats
from inference_optimizer.report.models import RawObservationRecord, ValidationResult

_TOLERANCE = 1e-6

_METRIC_FIELDS = ("ttft_ms", "latency_ms", "tpot_ms", "throughput_tokens_per_second")
_SCALAR_FIELDS = (
    "total_requests",
    "warmup_excluded",
    "measured_requests",
    "successful_requests",
    "failed_requests",
)


def _stats_mismatches(recomputed: MetricStats, stored: MetricStats) -> list[str]:
    mismatches: list[str] = []
    for field in ("count", "p50", "p95", "p99", "mean", "stddev"):
        recomputed_value = getattr(recomputed, field)
        stored_value = getattr(stored, field)
        if recomputed_value is None and stored_value is None:
            continue
        if recomputed_value is None or stored_value is None:
            mismatches.append(f"{field}: recomputed={recomputed_value} stored={stored_value}")
        elif abs(recomputed_value - stored_value) > _TOLERANCE:
            mismatches.append(f"{field}: recomputed={recomputed_value} stored={stored_value}")
    return mismatches


def validate_aggregates(
    baseline_report: BaselineReport, raw_records: list[RawObservationRecord]
) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    for repeat in baseline_report.repeats:
        observations = [
            record.observation
            for record in raw_records
            if record.repeat_index == repeat.repeat_index
        ]
        recomputed = summarize(observations)
        stored = repeat.summary

        mismatches: list[str] = []
        for metric_name in _METRIC_FIELDS:
            mismatches.extend(
                f"{metric_name}.{entry}"
                for entry in _stats_mismatches(
                    getattr(recomputed, metric_name), getattr(stored, metric_name)
                )
            )
        for scalar_field in _SCALAR_FIELDS:
            recomputed_value = getattr(recomputed, scalar_field)
            stored_value = getattr(stored, scalar_field)
            if recomputed_value != stored_value:
                mismatches.append(
                    f"{scalar_field}: recomputed={recomputed_value} stored={stored_value}"
                )
        if abs(recomputed.failure_rate - stored.failure_rate) > _TOLERANCE:
            mismatches.append(
                f"failure_rate: recomputed={recomputed.failure_rate} stored={stored.failure_rate}"
            )

        results.append(
            ValidationResult(
                repeat_index=repeat.repeat_index,
                matches=not mismatches,
                mismatches=mismatches,
            )
        )
    return results
