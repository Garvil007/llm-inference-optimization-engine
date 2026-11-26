"""Typed schemas for a baseline run: provenance metadata, per-repeat results, and
the overall report. Reuses Day 4's RequestObservation/BenchmarkSummary directly
(the per-request measurement contract doesn't change) and adds run-level
provenance (hardware, software versions, git commit) plus per-repeat memory
samples that Day 4's single-shot API benchmark didn't need.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from inference_optimizer.backends.models import BackendConfig, ModelConfig
from inference_optimizer.benchmark.models import BenchmarkSummary
from inference_optimizer.hardware.models import HardwareReport


class SoftwareVersions(BaseModel):
    python_version: str
    packages: dict[str, str | None]
    """Package name -> installed version, or None if not installed in this
    environment. Never a guessed/fabricated version."""


class RunMetadata(BaseModel):
    run_id: str
    started_at: str
    git_commit_hash: str | None
    git_commit_unavailable_reason: str | None = None
    hardware: HardwareReport
    software: SoftwareVersions
    model_config_snapshot: ModelConfig
    backend_config_snapshot: BackendConfig


class MemorySample(BaseModel):
    label: str
    process_rss_mb: float | None
    gpu_memory_used_mb: float | None
    gpu_memory_source: Literal["pynvml", "unavailable"]


class RepeatResult(BaseModel):
    repeat_index: int
    wall_time_seconds: float
    requests_per_second: float | None
    memory_before: MemorySample
    memory_after: MemorySample
    summary: BenchmarkSummary


class BaselineReport(BaseModel):
    run_metadata: RunMetadata
    config: dict[str, object]
    workload_record_ids: list[str]
    repeats: list[RepeatResult]
    status: Literal["completed", "unavailable"]
    unavailable_reason: str | None = None
