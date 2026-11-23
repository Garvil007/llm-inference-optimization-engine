"""Unit tests for LlamaCppBackend using a mocked HTTP transport (no real
subprocess/server). The real-server path is covered by the optional
integration smoke test in tests/integration/test_llama_cpp_smoke.py.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from inference_optimizer.backends.llama_cpp_backend import LlamaCppBackend
from inference_optimizer.backends.models import (
    BackendConfig,
    LlamaCppModelSource,
    ModelConfig,
)
from inference_optimizer.exceptions import BackendConnectionError, ModelLoadError


def _model_config(model_path: Path) -> ModelConfig:
    return ModelConfig(
        name="test-model",
        tokenizer="test/tokenizer",
        revision="deadbeef",
        context_limit=2048,
        quantization="Q4_K_M",
        license="apache-2.0",
        llama_cpp=LlamaCppModelSource(
            repo_id="test/repo", filename="model.gguf", model_path=model_path
        ),
    )


def _backend_config() -> BackendConfig:
    return BackendConfig(backend_type="llama-cpp", port=18731)


class TestStartValidation:
    @pytest.mark.asyncio
    async def test_raises_model_load_error_when_gguf_missing(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "does-not-exist.gguf"
        backend = LlamaCppBackend(_model_config(missing_path), _backend_config())

        with pytest.raises(ModelLoadError, match="not found"):
            await backend.start()

    @pytest.mark.asyncio
    async def test_raises_model_load_error_when_no_llama_cpp_source(self) -> None:
        config = ModelConfig(
            name="test-model",
            tokenizer="test/tokenizer",
            revision="deadbeef",
            context_limit=2048,
            quantization="Q4_K_M",
            license="apache-2.0",
        )
        backend = LlamaCppBackend(config, _backend_config())

        with pytest.raises(ModelLoadError, match="no 'llama_cpp' source"):
            await backend.start()


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_reports_unhealthy_when_never_started(self, tmp_path: Path) -> None:
        backend = LlamaCppBackend(_model_config(tmp_path / "x.gguf"), _backend_config())

        status = await backend.health_check()

        assert status.healthy is False
        assert status.backend_type == "llama-cpp"

    @pytest.mark.asyncio
    async def test_reports_healthy_with_mocked_transport(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"object": "list", "data": [{"id": "test-model"}]})

        backend = LlamaCppBackend(_model_config(tmp_path / "x.gguf"), _backend_config())
        backend._client = httpx.AsyncClient(  # noqa: SLF001 - test injects transport directly
            transport=httpx.MockTransport(handler), base_url=backend.base_url
        )

        status = await backend.health_check()

        assert status.healthy is True
        assert status.model_loaded is True


class TestGenerateStream:
    @pytest.mark.asyncio
    async def test_raises_connection_error_when_not_started(self, tmp_path: Path) -> None:
        backend = LlamaCppBackend(_model_config(tmp_path / "x.gguf"), _backend_config())

        with pytest.raises(BackendConnectionError):
            async for _ in backend.generate_stream("hello"):
                pass

    @pytest.mark.asyncio
    async def test_yields_chunks_from_sse_stream(self, tmp_path: Path) -> None:
        sse_body = (
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            "data: [DONE]\n\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse_body)

        backend = LlamaCppBackend(_model_config(tmp_path / "x.gguf"), _backend_config())
        backend._client = httpx.AsyncClient(  # noqa: SLF001 - test injects transport directly
            transport=httpx.MockTransport(handler), base_url=backend.base_url
        )

        chunks = [chunk async for chunk in backend.generate_stream("hi")]

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_generate_collects_stream_into_full_text(self, tmp_path: Path) -> None:
        sse_body = (
            'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":" there"}}]}\n\n'
            "data: [DONE]\n\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse_body)

        backend = LlamaCppBackend(_model_config(tmp_path / "x.gguf"), _backend_config())
        backend._client = httpx.AsyncClient(  # noqa: SLF001 - test injects transport directly
            transport=httpx.MockTransport(handler), base_url=backend.base_url
        )

        result = await backend.generate("hi")

        assert result.text == "Hi there"

    @pytest.mark.asyncio
    async def test_raises_connection_error_on_http_failure(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        backend = LlamaCppBackend(_model_config(tmp_path / "x.gguf"), _backend_config())
        backend._client = httpx.AsyncClient(  # noqa: SLF001 - test injects transport directly
            transport=httpx.MockTransport(handler), base_url=backend.base_url
        )

        with pytest.raises(BackendConnectionError):
            async for _ in backend.generate_stream("hi"):
                pass
