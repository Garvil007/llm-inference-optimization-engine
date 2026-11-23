"""Unit tests for inference_optimizer.hardware.detector.

pynvml is not installed in the dev/test environment (it lives in the optional
`gpu` extra), so detect_gpu() naturally exercises the nvidia-smi fallback path
here; the nvidia-smi subprocess call itself is mocked so tests don't depend on
actual hardware.
"""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import pytest

from inference_optimizer.hardware import detector
from inference_optimizer.hardware.models import HardwareReport


class TestDetectCPU:
    def test_reports_physical_and_logical_cores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            detector.psutil, "cpu_count", lambda logical=True: 8 if logical else 4
        )
        monkeypatch.setattr(detector.platform, "machine", lambda: "AMD64")

        cpu = detector.detect_cpu()

        assert cpu.physical_cores == 4
        assert cpu.logical_cores == 8
        assert cpu.architecture == "AMD64"


class TestDetectMemory:
    def test_converts_bytes_to_gb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_vm = SimpleNamespace(total=16_000_000_000, available=8_000_000_000)
        monkeypatch.setattr(detector.psutil, "virtual_memory", lambda: fake_vm)

        memory = detector.detect_memory()

        assert memory.total_gb == 16.0
        assert memory.available_gb == 8.0


class TestDetectDisk:
    def test_converts_bytes_to_gb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_usage = shutil._ntuple_diskusage(
            total=100_000_000_000, used=40_000_000_000, free=60_000_000_000
        )
        monkeypatch.setattr(detector.shutil, "disk_usage", lambda path: fake_usage)

        disk = detector.detect_disk(path="/fake")

        assert disk.total_gb == 100.0
        assert disk.free_gb == 60.0
        assert disk.path == "/fake"


class TestDetectGPU:
    def test_falls_back_to_nvidia_smi_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(detector.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
        fake_result = SimpleNamespace(
            stdout="NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, 566.36, 8.6\n"
        )
        monkeypatch.setattr(
            detector.subprocess, "run", lambda *args, **kwargs: fake_result
        )

        gpu = detector.detect_gpu()

        assert gpu.available is True
        assert gpu.name == "NVIDIA GeForce RTX 3050 Laptop GPU"
        assert gpu.vram_total_mb == 4096
        assert gpu.driver_version == "566.36"
        assert gpu.compute_capability == "8.6"
        assert gpu.detection_method == "nvidia-smi"

    def test_reports_unavailable_when_no_gpu_tooling_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(detector.shutil, "which", lambda name: None)

        gpu = detector.detect_gpu()

        assert gpu.available is False
        assert gpu.name is None

    def test_reports_unavailable_when_nvidia_smi_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(detector.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

        def _raise(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

        monkeypatch.setattr(detector.subprocess, "run", _raise)

        gpu = detector.detect_gpu()

        assert gpu.available is False
        assert gpu.detection_method == "nvidia-smi-error"


class TestDetectDocker:
    def test_reports_available_with_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(detector.shutil, "which", lambda name: "/usr/bin/docker")
        fake_result = SimpleNamespace(stdout="Docker version 27.4.0, build bde2b89\n")
        monkeypatch.setattr(
            detector.subprocess, "run", lambda *args, **kwargs: fake_result
        )

        software = detector.detect_docker()

        assert software.docker_available is True
        assert software.docker_version == "Docker version 27.4.0, build bde2b89"

    def test_reports_unavailable_when_binary_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(detector.shutil, "which", lambda name: None)

        software = detector.detect_docker()

        assert software.docker_available is False
        assert software.docker_version is None


class TestDetectHardware:
    def test_assembles_a_complete_report(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(detector, "detect_cpu", lambda: detector.CPUInfo(
            physical_cores=4, logical_cores=8, architecture="AMD64"
        ))
        monkeypatch.setattr(detector, "detect_memory", lambda: detector.MemoryInfo(
            total_gb=16.0, available_gb=8.0
        ))
        monkeypatch.setattr(detector, "detect_gpu", lambda: detector.GPUInfo(available=False))
        monkeypatch.setattr(detector, "detect_disk", lambda path=".": detector.DiskInfo(
            total_gb=100.0, free_gb=50.0, path=path
        ))
        monkeypatch.setattr(detector, "detect_docker", lambda: detector.SoftwareInfo(
            docker_available=True, docker_version="Docker version 27.4.0"
        ))

        report = detector.detect_hardware()

        assert isinstance(report, HardwareReport)
        assert report.cpu.logical_cores == 8
        assert report.memory.total_gb == 16.0
        assert report.gpu.available is False
        assert report.disk.total_gb == 100.0
        assert report.software.docker_available is True
        assert report.detected_at  # non-empty ISO timestamp
