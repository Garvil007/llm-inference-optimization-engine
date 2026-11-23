"""Unit tests for backends.registry.create_backend."""

from __future__ import annotations

import pytest

from inference_optimizer.backends.llama_cpp_backend import LlamaCppBackend
from inference_optimizer.backends.models import ModelConfig
from inference_optimizer.backends.registry import create_backend
from inference_optimizer.backends.vllm_backend import VLLMBackend


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="test-model",
        tokenizer="test/tokenizer",
        revision="deadbeef",
        context_limit=2048,
        quantization="Q4_K_M",
        license="apache-2.0",
    )


class TestCreateBackend:
    def test_creates_llama_cpp_backend(self) -> None:
        backend = create_backend("llama-cpp", _model_config())

        assert isinstance(backend, LlamaCppBackend)
        assert backend.backend_config.backend_type == "llama-cpp"

    def test_creates_vllm_backend(self) -> None:
        backend = create_backend("vllm", _model_config())

        assert isinstance(backend, VLLMBackend)

    def test_raises_on_unknown_backend_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend_type"):
            create_backend("does-not-exist", _model_config())
