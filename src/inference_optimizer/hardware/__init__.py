"""Hardware detection: CPU, memory, GPU, disk, and software availability."""

from inference_optimizer.hardware.detector import detect_hardware
from inference_optimizer.hardware.models import HardwareReport

__all__ = ["detect_hardware", "HardwareReport"]
