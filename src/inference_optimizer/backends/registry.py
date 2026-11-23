"""Factory for constructing an InferenceBackend by name."""

from __future__ import annotations

from inference_optimizer.backends.base import InferenceBackend
from inference_optimizer.backends.llama_cpp_backend import LlamaCppBackend
from inference_optimizer.backends.models import BackendConfig, ModelConfig
from inference_optimizer.backends.vllm_backend import VLLMBackend

_BACKEND_CLASSES: dict[str, type[InferenceBackend]] = {
    "llama-cpp": LlamaCppBackend,
    "vllm": VLLMBackend,
}


def create_backend(
    backend_type: str, model_config: ModelConfig, backend_config: BackendConfig | None = None
) -> InferenceBackend:
    backend_class = _BACKEND_CLASSES.get(backend_type)
    if backend_class is None:
        raise ValueError(
            f"Unknown backend_type {backend_type!r}; choose one of "
            f"{sorted(_BACKEND_CLASSES)}."
        )
    resolved_config = backend_config or BackendConfig(backend_type=backend_type)  # type: ignore[arg-type]
    return backend_class(model_config, resolved_config)
