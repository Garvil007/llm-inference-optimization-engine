"""Process and GPU memory sampling for a baseline run. Degrades to None/
"unavailable" exactly like hardware/detector.py's GPU probing -- a baseline
run must never report a fabricated memory number when the signal can't be read
in this environment (no pynvml, no NVIDIA GPU, or a read error).
"""

from __future__ import annotations

import os

import psutil

from inference_optimizer.baseline.models import MemorySample
from inference_optimizer.logging import get_logger

logger = get_logger(__name__)


def _process_rss_mb() -> float | None:
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 2)
    except psutil.Error as exc:
        logger.warning("process_memory_sample_failed", error=str(exc))
        return None


def _gpu_memory_used_mb() -> tuple[float | None, str]:
    try:
        import pynvml
    except ImportError:
        return None, "unavailable"

    try:
        pynvml.nvmlInit()
        try:
            if pynvml.nvmlDeviceGetCount() == 0:
                return None, "unavailable"
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return round(memory_info.used / 1e6, 2), "pynvml"
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # pynvml raises its own NVMLError hierarchy
        logger.warning("gpu_memory_sample_failed", error=str(exc))
        return None, "unavailable"


def sample_memory(label: str) -> MemorySample:
    gpu_used_mb, gpu_source = _gpu_memory_used_mb()
    return MemorySample(
        label=label,
        process_rss_mb=_process_rss_mb(),
        gpu_memory_used_mb=gpu_used_mb,
        gpu_memory_source=gpu_source,  # type: ignore[arg-type]
    )
