"""Parser tests: loading real baseline artifact files back into typed objects.
No real model/backend involved -- these build small synthetic BaselineReport/
observation files on disk, matching exactly what baseline/runner.py writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.backends.models import BackendConfig, ModelConfig
from inference_optimizer.baseline.models import (
    BaselineReport,
    MemorySample,
    RepeatResult,
    RunMetadata,
    SoftwareVersions,
)
from inference_optimizer.benchmark.models import BenchmarkSummary, MetricStats, RequestObservation
from inference_optimizer.exceptions import ConfigurationError
from inference_optimizer.hardware.models import (
    CPUInfo,
    DiskInfo,
    GPUInfo,
    HardwareReport,
    MemoryInfo,
    SoftwareInfo,
)
from inference_optimizer.report.parser import load_baseline_report, load_raw_observations


def _hardware() -> HardwareReport:
    return HardwareReport(
        os_name="TestOS",
        os_version="1.0",
        python_version="3.12.0",
        cpu=CPUInfo(physical_cores=4, logical_cores=8, architecture="AMD64"),
        memory=MemoryInfo(total_gb=16.0, available_gb=8.0),
        gpu=GPUInfo(available=False),
        disk=DiskInfo(total_gb=100.0, free_gb=50.0, path="."),
        software=SoftwareInfo(docker_available=False),
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="fake-model",
        tokenizer="fake/tokenizer",
        revision="deadbeef",
        context_limit=2048,
        quantization="Q4_K_M",
        license="apache-2.0",
    )


def _baseline_report() -> BaselineReport:
    summary = BenchmarkSummary(
        total_requests=2,
        warmup_excluded=0,
        measured_requests=2,
        successful_requests=2,
        failed_requests=0,
        failure_rate=0.0,
        ttft_ms=MetricStats(count=2, p50=10.0, p95=10.0, p99=10.0, mean=10.0, stddev=0.0),
        latency_ms=MetricStats(count=2, p50=20.0, p95=20.0, p99=20.0, mean=20.0, stddev=0.0),
        tpot_ms=MetricStats(count=2, p50=5.0, p95=5.0, p99=5.0, mean=5.0, stddev=0.0),
        throughput_tokens_per_second=MetricStats(
            count=2, p50=50.0, p95=50.0, p99=50.0, mean=50.0, stddev=0.0
        ),
    )
    run_metadata = RunMetadata(
        run_id="test-run-id",
        started_at="2026-11-27T00:00:00+00:00",
        git_commit_hash="abc123",
        hardware=_hardware(),
        software=SoftwareVersions(python_version="3.12.0", packages={"pydantic": "2.0.0"}),
        model_config_snapshot=_model_config(),
        backend_config_snapshot=BackendConfig(backend_type="llama-cpp"),
    )
    return BaselineReport(
        run_metadata=run_metadata,
        config={},
        workload_record_ids=["a", "b"],
        repeats=[
            RepeatResult(
                repeat_index=0,
                wall_time_seconds=1.0,
                requests_per_second=2.0,
                memory_before=MemorySample(
                    label="repeat_0_before",
                    process_rss_mb=100.0,
                    gpu_memory_used_mb=None,
                    gpu_memory_source="unavailable",
                ),
                memory_after=MemorySample(
                    label="repeat_0_after",
                    process_rss_mb=100.0,
                    gpu_memory_used_mb=None,
                    gpu_memory_source="unavailable",
                ),
                summary=summary,
            )
        ],
        status="completed",
    )


def test_load_baseline_report_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_baseline_report(tmp_path / "missing.json")


def test_load_baseline_report_parses_real_file(tmp_path: Path) -> None:
    report = _baseline_report()
    path = tmp_path / "report.json"
    path.write_text(report.model_dump_json())

    loaded = load_baseline_report(path)

    assert loaded.run_metadata.run_id == "test-run-id"
    assert loaded.status == "completed"
    assert len(loaded.repeats) == 1


def test_load_raw_observations_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_raw_observations(tmp_path / "missing.jsonl")


def test_load_raw_observations_restores_repeat_index(tmp_path: Path) -> None:
    observation = RequestObservation(
        request_index=0,
        is_warmup=False,
        success=True,
        started_at="2026-11-27T00:00:00+00:00",
        start_perf=0.0,
        first_token_perf=0.1,
        end_perf=0.2,
        ttft_ms=100.0,
        latency_ms=200.0,
        output_text_length=5,
        output_token_count=3,
        token_count_method="fixed",
        tpot_ms=50.0,
        output_tokens_per_second=15.0,
    )
    payload = observation.model_dump(mode="json")
    payload["repeat_index"] = 1
    path = tmp_path / "observations.jsonl"
    path.write_text(json.dumps(payload) + "\n")

    records = load_raw_observations(path)

    assert len(records) == 1
    assert records[0].repeat_index == 1
    assert records[0].observation.latency_ms == 200.0
