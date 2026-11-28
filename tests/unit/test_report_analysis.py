"""Tests for the pure analysis functions over a parsed BaselineReport:
validate_aggregates (recompute vs. stored), compute_variability (repeat-to-
repeat spread), compute_flags (incomplete/failed/mismatched/contended/
non-comparable), and build_comparisons (internal consistency only, no
external SLA claims).
"""

from __future__ import annotations

from inference_optimizer.backends.models import BackendConfig, ModelConfig
from inference_optimizer.baseline.models import (
    BaselineReport,
    MemorySample,
    RepeatResult,
    RunMetadata,
    SoftwareVersions,
)
from inference_optimizer.benchmark.models import BenchmarkSummary, MetricStats, RequestObservation
from inference_optimizer.hardware.models import (
    CPUInfo,
    DiskInfo,
    GPUInfo,
    HardwareReport,
    MemoryInfo,
    SoftwareInfo,
)
from inference_optimizer.report.comparison import build_comparisons
from inference_optimizer.report.flags import compute_flags
from inference_optimizer.report.models import RawObservationRecord
from inference_optimizer.report.validate import validate_aggregates
from inference_optimizer.report.variability import compute_variability


def _hardware(available_gb: float = 8.0, gpu_available: bool = False) -> HardwareReport:
    return HardwareReport(
        os_name="TestOS",
        os_version="1.0",
        python_version="3.12.0",
        cpu=CPUInfo(physical_cores=4, logical_cores=8, architecture="AMD64"),
        memory=MemoryInfo(total_gb=16.0, available_gb=available_gb),
        gpu=GPUInfo(available=gpu_available, name="Fake GPU" if gpu_available else None),
        disk=DiskInfo(total_gb=100.0, free_gb=50.0, path="."),
        software=SoftwareInfo(docker_available=False),
    )


def _run_metadata(available_gb: float = 8.0) -> RunMetadata:
    return RunMetadata(
        run_id="run-1",
        started_at="2026-11-27T00:00:00+00:00",
        git_commit_hash="abc123",
        hardware=_hardware(available_gb=available_gb),
        software=SoftwareVersions(python_version="3.12.0", packages={}),
        model_config_snapshot=ModelConfig(
            name="fake-model",
            tokenizer="fake/tokenizer",
            revision="deadbeef",
            context_limit=2048,
            quantization="Q4_K_M",
            license="apache-2.0",
        ),
        backend_config_snapshot=BackendConfig(backend_type="llama-cpp"),
    )


def _observation(
    index: int, ttft_ms: float, latency_ms: float, success: bool = True
) -> RequestObservation:
    return RequestObservation(
        request_index=index,
        is_warmup=False,
        success=success,
        error=None if success else "boom",
        started_at="2026-11-27T00:00:00+00:00",
        start_perf=0.0,
        first_token_perf=ttft_ms / 1000 if success else None,
        end_perf=latency_ms / 1000,
        ttft_ms=ttft_ms if success else None,
        latency_ms=latency_ms,
        output_text_length=10,
        output_token_count=5 if success else None,
        token_count_method="fixed" if success else "unavailable",
        tpot_ms=(latency_ms - ttft_ms) / 4 if success else None,
        output_tokens_per_second=(5 / (latency_ms / 1000)) if success else None,
    )


def _repeat(repeat_index: int, observations: list[RequestObservation]) -> RepeatResult:
    successful = [o for o in observations if o.success]
    failed = len(observations) - len(successful)

    def _stats(values: list[float]) -> MetricStats:
        if not values:
            return MetricStats(count=0)
        return MetricStats(
            count=len(values),
            p50=values[0],
            p95=values[0],
            p99=values[0],
            mean=sum(values) / len(values),
            stddev=0.0,
        )

    summary = BenchmarkSummary(
        total_requests=len(observations),
        warmup_excluded=0,
        measured_requests=len(observations),
        successful_requests=len(successful),
        failed_requests=failed,
        failure_rate=failed / len(observations) if observations else 0.0,
        ttft_ms=_stats([o.ttft_ms for o in successful if o.ttft_ms is not None]),
        latency_ms=_stats([o.latency_ms for o in successful]),
        tpot_ms=_stats([o.tpot_ms for o in successful if o.tpot_ms is not None]),
        throughput_tokens_per_second=_stats(
            [o.output_tokens_per_second for o in successful if o.output_tokens_per_second]
        ),
    )
    return RepeatResult(
        repeat_index=repeat_index,
        wall_time_seconds=1.0,
        requests_per_second=len(observations),
        memory_before=MemorySample(
            label=f"repeat_{repeat_index}_before",
            process_rss_mb=100.0,
            gpu_memory_used_mb=None,
            gpu_memory_source="unavailable",
        ),
        memory_after=MemorySample(
            label=f"repeat_{repeat_index}_after",
            process_rss_mb=100.0,
            gpu_memory_used_mb=None,
            gpu_memory_source="unavailable",
        ),
        summary=summary,
    )


class TestValidateAggregates:
    def test_matching_raw_data_reports_no_mismatches(self) -> None:
        observations = [_observation(0, 10.0, 20.0), _observation(1, 10.0, 20.0)]
        repeat = _repeat(0, observations)
        report = BaselineReport(
            run_metadata=_run_metadata(),
            config={},
            workload_record_ids=["a", "b"],
            repeats=[repeat],
            status="completed",
        )
        raw_records = [
            RawObservationRecord(repeat_index=0, observation=o) for o in observations
        ]

        results = validate_aggregates(report, raw_records)

        assert len(results) == 1
        assert results[0].matches is True
        assert results[0].mismatches == []

    def test_tampered_stored_aggregate_is_flagged_as_mismatch(self) -> None:
        observations = [_observation(0, 10.0, 20.0), _observation(1, 10.0, 20.0)]
        repeat = _repeat(0, observations)
        repeat.summary.latency_ms.p50 = 999.0  # tamper with the stored aggregate
        report = BaselineReport(
            run_metadata=_run_metadata(),
            config={},
            workload_record_ids=["a", "b"],
            repeats=[repeat],
            status="completed",
        )
        raw_records = [
            RawObservationRecord(repeat_index=0, observation=o) for o in observations
        ]

        results = validate_aggregates(report, raw_records)

        assert results[0].matches is False
        assert any("latency_ms" in m for m in results[0].mismatches)


class TestComputeVariability:
    def test_single_repeat_reports_none_not_zero(self) -> None:
        report = BaselineReport(
            run_metadata=_run_metadata(),
            config={},
            workload_record_ids=["a"],
            repeats=[_repeat(0, [_observation(0, 10.0, 20.0)])],
            status="completed",
        )

        stats = compute_variability(report)

        latency_stat = next(s for s in stats if s.metric == "latency_ms_p50")
        assert latency_stat.mean is None
        assert latency_stat.coefficient_of_variation is None

    def test_multiple_repeats_computes_spread(self) -> None:
        report = BaselineReport(
            run_metadata=_run_metadata(),
            config={},
            workload_record_ids=["a"],
            repeats=[
                _repeat(0, [_observation(0, 10.0, 20.0)]),
                _repeat(1, [_observation(0, 10.0, 40.0)]),
            ],
            status="completed",
        )

        stats = compute_variability(report)

        latency_stat = next(s for s in stats if s.metric == "latency_ms_p50")
        assert latency_stat.per_repeat_values == [20.0, 40.0]
        assert latency_stat.mean == 30.0
        assert latency_stat.coefficient_of_variation is not None


class TestComputeFlags:
    def test_unavailable_run_only_gets_run_unavailable_flag(self) -> None:
        report = BaselineReport(
            run_metadata=_run_metadata(),
            config={},
            workload_record_ids=[],
            repeats=[],
            status="unavailable",
            unavailable_reason="model file missing",
        )

        flags = compute_flags(report, [], [])

        assert len(flags) == 1
        assert flags[0].code == "run_unavailable"

    def test_fewer_than_three_repeats_flags_incomplete(self) -> None:
        report = BaselineReport(
            run_metadata=_run_metadata(),
            config={},
            workload_record_ids=["a"],
            repeats=[_repeat(0, [_observation(0, 10.0, 20.0)])],
            status="completed",
        )

        flags = compute_flags(report, [], [])

        assert any(f.code == "incomplete_repeats" for f in flags)

    def test_failed_requests_are_flagged(self) -> None:
        observations = [_observation(0, 10.0, 20.0, success=False)]
        repeat = _repeat(0, observations)
        report = BaselineReport(
            run_metadata=_run_metadata(),
            config={},
            workload_record_ids=["a"],
            repeats=[repeat, _repeat(1, [_observation(0, 10.0, 20.0)])],
            status="completed",
        )

        flags = compute_flags(report, [], [])

        assert any(f.code == "requests_failed" for f in flags)

    def test_low_available_memory_is_flagged(self) -> None:
        report = BaselineReport(
            run_metadata=_run_metadata(available_gb=1.0),
            config={},
            workload_record_ids=["a"],
            repeats=[_repeat(0, [_observation(0, 10.0, 20.0)])],
            status="completed",
        )

        flags = compute_flags(report, [], [])

        assert any(f.code == "low_available_memory" for f in flags)


class TestBuildComparisons:
    def test_ttft_less_than_or_equal_latency_passes(self) -> None:
        report = BaselineReport(
            run_metadata=_run_metadata(),
            config={},
            workload_record_ids=["a"],
            repeats=[_repeat(0, [_observation(0, 10.0, 20.0)])],
            status="completed",
        )

        comparisons = build_comparisons(report)

        ttft_check = next(c for c in comparisons if c.metric == "ttft_vs_latency")
        assert ttft_check.passed is True

    def test_includes_no_external_baseline_disclaimer(self) -> None:
        report = BaselineReport(
            run_metadata=_run_metadata(),
            config={},
            workload_record_ids=["a"],
            repeats=[_repeat(0, [_observation(0, 10.0, 20.0)])],
            status="completed",
        )

        comparisons = build_comparisons(report)

        assert any(c.metric == "no_external_baseline" for c in comparisons)
