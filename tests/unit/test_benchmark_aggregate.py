"""Aggregate math verified by hand against a known small sample -- the percentile
method is linear interpolation (numpy's default "linear"), so these expected values
are computed the same way by hand in the comments below.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from inference_optimizer.benchmark.aggregate import summarize
from inference_optimizer.benchmark.models import RequestObservation


def _obs(
    index: int,
    latency_ms: float,
    ttft_ms: float | None = None,
    tpot_ms: float | None = None,
    throughput: float | None = None,
    success: bool = True,
    is_warmup: bool = False,
    error: str | None = None,
) -> RequestObservation:
    return RequestObservation(
        request_index=index,
        is_warmup=is_warmup,
        success=success,
        error=error,
        started_at=datetime.now(UTC),
        start_perf=0.0,
        first_token_perf=None,
        end_perf=latency_ms / 1000,
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        output_text_length=0,
        output_token_count=None,
        tpot_ms=tpot_ms,
        output_tokens_per_second=throughput,
    )


class TestPercentilesMeanStddev:
    def test_matches_hand_computed_linear_interpolation(self) -> None:
        # latency_ms = [100, 200, 300, 400, 500]; rank = (n-1)*fraction
        # p50: rank=2.0 -> sorted[2] = 300
        # p95: rank=3.8 -> sorted[3]*0.2 + sorted[4]*0.8 = 80 + 400 = 480
        # p99: rank=3.96 -> sorted[3]*0.04 + sorted[4]*0.96 = 16 + 480 = 496
        # mean = 300; sample stdev (ddof=1) = sqrt(100000/4) = 158.1138830...
        observations = [_obs(i, latency) for i, latency in enumerate([100, 200, 300, 400, 500])]

        summary = summarize(observations)

        assert summary.latency_ms.count == 5
        assert summary.latency_ms.p50 == pytest.approx(300.0)
        assert summary.latency_ms.p95 == pytest.approx(480.0)
        assert summary.latency_ms.p99 == pytest.approx(496.0)
        assert summary.latency_ms.mean == pytest.approx(300.0)
        assert summary.latency_ms.stddev == pytest.approx(158.1138830, rel=1e-6)

    def test_single_sample_every_percentile_equals_that_value(self) -> None:
        summary = summarize([_obs(0, 42.0)])
        assert summary.latency_ms.count == 1
        assert summary.latency_ms.p50 == 42.0
        assert summary.latency_ms.p95 == 42.0
        assert summary.latency_ms.p99 == 42.0
        assert summary.latency_ms.stddev == 0.0

    def test_empty_sample_is_all_none_not_zero(self) -> None:
        summary = summarize([_obs(0, 10.0, success=False, error="boom")])
        assert summary.latency_ms.count == 0
        assert summary.latency_ms.p50 is None
        assert summary.latency_ms.mean is None
        assert summary.latency_ms.stddev is None


class TestWarmupExclusion:
    def test_warmup_observations_excluded_from_every_aggregate(self) -> None:
        warmups = [_obs(i, 9999.0, is_warmup=True) for i in range(2)]
        measured = [_obs(i + 2, 100.0) for i in range(3)]

        summary = summarize(warmups + measured)

        assert summary.total_requests == 5
        assert summary.warmup_excluded == 2
        assert summary.measured_requests == 3
        assert summary.latency_ms.count == 3
        assert summary.latency_ms.mean == pytest.approx(100.0)


class TestFailedRequests:
    def test_failed_requests_excluded_from_aggregates_but_counted_in_failure_rate(self) -> None:
        observations = [
            _obs(0, 100.0, ttft_ms=10.0),
            _obs(1, 200.0, ttft_ms=20.0),
            _obs(2, 50.0, success=False, error="connection reset", ttft_ms=5.0),
        ]

        summary = summarize(observations)

        assert summary.measured_requests == 3
        assert summary.failed_requests == 1
        assert summary.failure_rate == pytest.approx(1 / 3)
        # The failed request's ttft_ms=5.0 must not pollute the ttft aggregate.
        assert summary.ttft_ms.count == 2
        assert summary.ttft_ms.mean == pytest.approx(15.0)

    def test_all_failed_gives_failure_rate_one_and_empty_aggregates(self) -> None:
        observations = [_obs(i, 10.0, success=False, error="x") for i in range(3)]
        summary = summarize(observations)
        assert summary.failure_rate == 1.0
        assert summary.latency_ms.count == 0

    def test_no_measured_requests_gives_failure_rate_zero_not_div_by_zero(self) -> None:
        summary = summarize([_obs(0, 10.0, is_warmup=True)])
        assert summary.measured_requests == 0
        assert summary.failure_rate == 0.0


class TestMissingTokenCounts:
    def test_missing_tpot_or_throughput_excluded_per_metric_not_per_request(self) -> None:
        observations = [
            _obs(0, 100.0, ttft_ms=10.0, tpot_ms=None, throughput=None),
            _obs(1, 200.0, ttft_ms=20.0, tpot_ms=15.0, throughput=50.0),
        ]

        summary = summarize(observations)

        # Both requests contribute to latency_ms/ttft_ms...
        assert summary.latency_ms.count == 2
        assert summary.ttft_ms.count == 2
        # ...but only the second contributes to tpot/throughput.
        assert summary.tpot_ms.count == 1
        assert summary.throughput_tokens_per_second.count == 1
