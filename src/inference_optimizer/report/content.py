"""Shared structured content used by both the Markdown and HTML renderers, so
table layouts live in one place instead of being duplicated per output
format.
"""

from __future__ import annotations

from inference_optimizer.baseline.models import BaselineReport


def hardware_table(baseline_report: BaselineReport) -> list[tuple[str, str]]:
    hw = baseline_report.run_metadata.hardware
    gpu = (
        f"{hw.gpu.name} ({hw.gpu.vram_total_mb} MiB VRAM)"
        if hw.gpu.available
        else "none detected"
    )
    return [
        ("OS", f"{hw.os_name} ({hw.os_version})"),
        ("Python", hw.python_version),
        ("CPU", f"{hw.cpu.physical_cores} physical / {hw.cpu.logical_cores} logical cores"),
        ("RAM", f"{hw.memory.total_gb} GB total, {hw.memory.available_gb} GB available"),
        ("GPU", gpu),
        ("Disk", f"{hw.disk.free_gb} GB free of {hw.disk.total_gb} GB"),
    ]


def model_table(baseline_report: BaselineReport) -> list[tuple[str, str]]:
    model = baseline_report.run_metadata.model_config_snapshot
    backend = baseline_report.run_metadata.backend_config_snapshot
    return [
        ("Model", model.name),
        ("Quantization", model.quantization),
        ("Context limit", f"{model.context_limit} tokens"),
        ("Tokenizer", model.tokenizer),
        ("Revision", model.revision),
        ("Backend", backend.backend_type),
        ("Git commit", baseline_report.run_metadata.git_commit_hash or "unavailable"),
    ]


def metrics_table(baseline_report: BaselineReport) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repeat in baseline_report.repeats:
        summary = repeat.summary
        rows.append(
            {
                "repeat": repeat.repeat_index,
                "latency_p50_ms": summary.latency_ms.p50,
                "latency_p95_ms": summary.latency_ms.p95,
                "latency_p99_ms": summary.latency_ms.p99,
                "ttft_p50_ms": summary.ttft_ms.p50,
                "tpot_p50_ms": summary.tpot_ms.p50,
                "throughput_p50_tps": summary.throughput_tokens_per_second.p50,
                "requests_per_second": repeat.requests_per_second,
                "failure_rate": summary.failure_rate,
            }
        )
    return rows
