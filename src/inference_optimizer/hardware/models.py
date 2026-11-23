"""Typed data models for a hardware detection report.

Fields: os_name (str), os_version (str), python_version (str), architecture (str),
cpu.physical_cores (int), cpu.logical_cores (int), memory.total_gb (float),
memory.available_gb (float), gpu.available (bool), gpu.name (str|None),
gpu.vram_total_mb (int|None), gpu.driver_version (str|None), gpu.cuda_version (str|None),
gpu.compute_capability (str|None), disk.total_gb (float), disk.free_gb (float),
software.docker_available (bool), software.cuda_toolkit_available (bool).
detected_at is an ISO-8601 UTC timestamp string (e.g. "2026-11-23T01:10:38+00:00").
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class CPUInfo(BaseModel):
    physical_cores: int | None
    logical_cores: int | None
    architecture: str


class MemoryInfo(BaseModel):
    total_gb: float
    available_gb: float


class GPUInfo(BaseModel):
    available: bool
    name: str | None = None
    vram_total_mb: int | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    compute_capability: str | None = None
    detection_method: str | None = None


class DiskInfo(BaseModel):
    total_gb: float
    free_gb: float
    path: str


class SoftwareInfo(BaseModel):
    docker_available: bool
    docker_version: str | None = None


class HardwareReport(BaseModel):
    os_name: str
    os_version: str
    python_version: str
    cpu: CPUInfo
    memory: MemoryInfo
    gpu: GPUInfo
    disk: DiskInfo
    software: SoftwareInfo
    detected_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
