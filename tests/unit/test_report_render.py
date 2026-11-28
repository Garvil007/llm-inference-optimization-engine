"""Markdown/HTML renderer tests: given a small synthetic ReportData, assert
each renderer includes the hardware/model tables, flags, variability,
comparisons, and (for the unavailable case) the exact unavailable reason --
without re-checking the analysis logic itself (covered in
test_report_analysis.py).
"""

from __future__ import annotations

from pathlib import Path

from inference_optimizer.backends.models import BackendConfig, ModelConfig
from inference_optimizer.baseline.models import (
    BaselineReport,
    MemorySample,
    RepeatResult,
    RunMetadata,
    SoftwareVersions,
)
from inference_optimizer.benchmark.models import BenchmarkSummary, MetricStats
from inference_optimizer.hardware.models import (
    CPUInfo,
    DiskInfo,
    GPUInfo,
    HardwareReport,
    MemoryInfo,
    SoftwareInfo,
)
from inference_optimizer.report.html_renderer import render_html
from inference_optimizer.report.markdown_renderer import render_markdown
from inference_optimizer.report.models import ComparisonNote, ReportData, RunFlag, VariabilityStat


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


def _baseline_report(status: str = "completed") -> BaselineReport:
    run_metadata = RunMetadata(
        run_id="run-xyz",
        started_at="2026-11-27T00:00:00+00:00",
        git_commit_hash="deadbeefcafe",
        hardware=_hardware(),
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
    if status == "unavailable":
        return BaselineReport(
            run_metadata=run_metadata,
            config={},
            workload_record_ids=[],
            repeats=[],
            status="unavailable",
            unavailable_reason="model file missing",
        )

    stats = MetricStats(count=1, p50=10.0, p95=10.0, p99=10.0, mean=10.0, stddev=0.0)
    summary = BenchmarkSummary(
        total_requests=1,
        warmup_excluded=0,
        measured_requests=1,
        successful_requests=1,
        failed_requests=0,
        failure_rate=0.0,
        ttft_ms=stats,
        latency_ms=stats,
        tpot_ms=stats,
        throughput_tokens_per_second=stats,
    )
    repeat = RepeatResult(
        repeat_index=0,
        wall_time_seconds=1.0,
        requests_per_second=1.0,
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
    return BaselineReport(
        run_metadata=run_metadata,
        config={},
        workload_record_ids=["a"],
        repeats=[repeat],
        status="completed",
    )


def _report_data(status: str = "completed", plot_paths: dict | None = None) -> ReportData:
    return ReportData(
        baseline_report=_baseline_report(status),
        validation=[],
        variability=[
            VariabilityStat(
                metric="latency_ms_p50", per_repeat_values=[10.0], mean=None,
                stddev=None, coefficient_of_variation=None,
            )
        ],
        flags=[RunFlag(level="warning", code="incomplete_repeats", message="only 1 repeat")],
        comparisons=[
            ComparisonNote(
                metric="ttft_vs_latency", description="ttft <= latency", passed=True, detail="ok"
            )
        ],
        plot_paths=plot_paths or {},
    )


def test_markdown_includes_hardware_model_flags_and_commit() -> None:
    markdown = render_markdown(_report_data())

    assert "deadbeefcafe" in markdown
    assert "fake-model" in markdown
    assert "incomplete_repeats" in markdown
    assert "ttft <= latency" in markdown


def test_markdown_unavailable_run_shows_reason_and_stops() -> None:
    markdown = render_markdown(_report_data(status="unavailable"))

    assert "model file missing" in markdown
    assert "## Hardware" not in markdown


def test_html_includes_hardware_model_flags_and_commit() -> None:
    html_doc = render_html(_report_data())

    assert "deadbeefcafe" in html_doc
    assert "fake-model" in html_doc
    assert "incomplete_repeats" in html_doc
    assert "<table>" in html_doc


def test_html_embeds_plot_image_references(tmp_path: Path) -> None:
    plot_path = tmp_path / "latency_distribution.png"
    plot_path.write_bytes(b"fake-png-bytes")

    html_doc = render_html(_report_data(plot_paths={"latency_ms": plot_path}))

    assert "latency_distribution.png" in html_doc
