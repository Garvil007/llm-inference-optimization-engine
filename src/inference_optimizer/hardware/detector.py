"""Hardware probing: CPU, memory, GPU, disk, and Docker availability.

Every probe degrades gracefully: if a signal can't be read (missing driver,
missing binary, permission error) the corresponding field is left None/False
and a warning is logged, rather than raising. detect_hardware() itself never
raises for environmental reasons — only HardwareDetectionError on a logic bug.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys

import psutil

from inference_optimizer.hardware.models import (
    CPUInfo,
    DiskInfo,
    GPUInfo,
    HardwareReport,
    MemoryInfo,
    SoftwareInfo,
)
from inference_optimizer.logging.setup import get_logger

logger = get_logger(__name__)

_SUBPROCESS_TIMEOUT_SECONDS = 5
# On Windows, `docker`/`nvidia-smi` on PATH can resolve to a .cmd/.bat shim,
# which CreateProcess can't exec directly without a shell (WinError 193).
_NEEDS_SHELL = sys.platform == "win32"


def detect_cpu() -> CPUInfo:
    return CPUInfo(
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        architecture=platform.machine(),
    )


def detect_memory() -> MemoryInfo:
    virtual_memory = psutil.virtual_memory()
    return MemoryInfo(
        total_gb=round(virtual_memory.total / 1e9, 2),
        available_gb=round(virtual_memory.available / 1e9, 2),
    )


def detect_disk(path: str = ".") -> DiskInfo:
    usage = shutil.disk_usage(path)
    return DiskInfo(
        total_gb=round(usage.total / 1e9, 2),
        free_gb=round(usage.free / 1e9, 2),
        path=path,
    )


def _detect_gpu_via_pynvml() -> GPUInfo | None:
    try:
        import pynvml
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() == 0:
                return GPUInfo(available=False, detection_method="pynvml")

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode()
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)

            return GPUInfo(
                available=True,
                name=name,
                vram_total_mb=round(memory_info.total / (1024 * 1024)),
                driver_version=driver_version,
                compute_capability=f"{major}.{minor}",
                detection_method="pynvml",
            )
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # pynvml raises its own NVMLError hierarchy
        logger.warning("gpu_detection_pynvml_failed", error=str(exc))
        return None


def _detect_gpu_via_nvidia_smi() -> GPUInfo:
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        return GPUInfo(available=False, detection_method="none")

    query_fields = "name,memory.total,driver_version,compute_cap"
    try:
        result = subprocess.run(  # noqa: S603 - fixed args, no user input
            [nvidia_smi_path, f"--query-gpu={query_fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            check=True,
            shell=_NEEDS_SHELL,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("gpu_detection_nvidia_smi_failed", error=str(exc))
        return GPUInfo(available=False, detection_method="nvidia-smi-error")

    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 4:
        return GPUInfo(available=False, detection_method="nvidia-smi-unparsed")

    name, vram_raw, driver_version, compute_capability = parts
    vram_total_mb = None
    try:
        vram_total_mb = int(vram_raw.split()[0])
    except (ValueError, IndexError):
        pass

    return GPUInfo(
        available=True,
        name=name,
        vram_total_mb=vram_total_mb,
        driver_version=driver_version,
        compute_capability=compute_capability,
        detection_method="nvidia-smi",
    )


def detect_gpu() -> GPUInfo:
    """Prefer pynvml (structured, no subprocess); fall back to parsing nvidia-smi."""
    gpu_info = _detect_gpu_via_pynvml()
    if gpu_info is not None:
        return gpu_info
    return _detect_gpu_via_nvidia_smi()


def detect_docker() -> SoftwareInfo:
    docker_path = shutil.which("docker")
    if docker_path is None:
        return SoftwareInfo(docker_available=False)

    try:
        result = subprocess.run(  # noqa: S603 - fixed args, no user input
            [docker_path, "--version"],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
            check=True,
            shell=_NEEDS_SHELL,
        )
        return SoftwareInfo(docker_available=True, docker_version=result.stdout.strip())
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("docker_detection_failed", error=str(exc))
        return SoftwareInfo(docker_available=False)


def detect_hardware(disk_path: str = ".") -> HardwareReport:
    """Run all probes and assemble a single hardware report."""
    return HardwareReport(
        os_name=platform.system(),
        os_version=platform.platform(),
        python_version=sys.version.split()[0],
        cpu=detect_cpu(),
        memory=detect_memory(),
        gpu=detect_gpu(),
        disk=detect_disk(disk_path),
        software=detect_docker(),
    )
