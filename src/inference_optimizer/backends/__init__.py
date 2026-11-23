"""Inference backends: a common interface over llama.cpp and vLLM, both served
over OpenAI-compatible HTTP endpoints.
"""

from inference_optimizer.backends.base import InferenceBackend
from inference_optimizer.backends.config_loader import load_model_config
from inference_optimizer.backends.models import (
    BackendConfig,
    GenerationParams,
    GenerationResult,
    HealthStatus,
    ModelConfig,
)
from inference_optimizer.backends.registry import create_backend

__all__ = [
    "InferenceBackend",
    "BackendConfig",
    "GenerationParams",
    "GenerationResult",
    "HealthStatus",
    "ModelConfig",
    "create_backend",
    "load_model_config",
]
