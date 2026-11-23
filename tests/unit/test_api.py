"""API tests against a fake InferenceBackend -- no real subprocess/model is spawned.
Covers: valid requests, malformed input, timeouts, backend unavailability, concurrency
admission, request IDs, and /metrics.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_optimizer.api.app import create_app
from inference_optimizer.api.config import ApiSettings
from inference_optimizer.backends.base import InferenceBackend
from inference_optimizer.backends.models import BackendConfig, HealthStatus, ModelConfig
from inference_optimizer.exceptions import BackendConnectionError


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="fake-model",
        tokenizer="fake/tokenizer",
        revision="deadbeef",
        context_limit=2048,
        quantization="Q4_K_M",
        license="apache-2.0",
    )


class FakeBackend(InferenceBackend):
    """Controllable stand-in: configurable chunks, per-chunk delay, health status,
    and an optional error to raise instead of streaming.
    """

    def __init__(
        self,
        chunks: list[str] | None = None,
        delay: float = 0.0,
        health: HealthStatus | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        super().__init__(_model_config(), BackendConfig(backend_type="llama-cpp"))
        self._chunks = chunks if chunks is not None else ["Hello", " world"]
        self._delay = delay
        self._health = health or HealthStatus(
            healthy=True, backend_type="llama-cpp", detail="ok", model_loaded=True
        )
        self._stream_error = stream_error

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def health_check(self) -> HealthStatus:
        return self._health

    async def generate_stream(self, prompt: str, **params: object) -> AsyncIterator[str]:
        if self._stream_error is not None:
            raise self._stream_error
        for chunk in self._chunks:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield chunk


def _client(app: FastAPI) -> TestClient:
    return TestClient(app)


@asynccontextmanager
async def _running_async_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


class TestHealthAndModels:
    def test_health_reports_healthy_backend(self) -> None:
        app = create_app(settings=ApiSettings(), backend=FakeBackend())
        with _client(app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["healthy"] is True

    def test_health_reports_unhealthy_backend_without_raising(self) -> None:
        unhealthy = HealthStatus(
            healthy=False, backend_type="llama-cpp", detail="not reachable", model_loaded=False
        )
        app = create_app(settings=ApiSettings(), backend=FakeBackend(health=unhealthy))
        with _client(app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["healthy"] is False

    def test_models_lists_the_loaded_model(self) -> None:
        app = create_app(settings=ApiSettings(), backend=FakeBackend())
        with _client(app) as client:
            response = client.get("/models")
        assert response.status_code == 200
        body = response.json()
        assert body == [
            {
                "name": "fake-model",
                "backend": "llama-cpp",
                "context_limit": 2048,
                "quantization": "Q4_K_M",
                "license": "apache-2.0",
            }
        ]


class TestGenerate:
    def test_valid_request_returns_full_text(self) -> None:
        app = create_app(settings=ApiSettings(), backend=FakeBackend())
        with _client(app) as client:
            response = client.post("/generate", json={"prompt": "hi"})
        assert response.status_code == 200
        body = response.json()
        assert body["text"] == "Hello world"
        assert body["model"] == "fake-model"
        assert body["backend"] == "llama-cpp"
        assert "request_id" in body

    def test_empty_prompt_is_rejected_as_malformed(self) -> None:
        app = create_app(settings=ApiSettings(), backend=FakeBackend())
        with _client(app) as client:
            response = client.post("/generate", json={"prompt": ""})
        assert response.status_code == 422

    def test_out_of_range_temperature_is_rejected(self) -> None:
        app = create_app(settings=ApiSettings(), backend=FakeBackend())
        with _client(app) as client:
            response = client.post("/generate", json={"prompt": "hi", "temperature": 5.0})
        assert response.status_code == 422

    def test_missing_prompt_field_is_rejected(self) -> None:
        app = create_app(settings=ApiSettings(), backend=FakeBackend())
        with _client(app) as client:
            response = client.post("/generate", json={})
        assert response.status_code == 422

    def test_backend_unavailable_returns_503(self) -> None:
        backend = FakeBackend(stream_error=BackendConnectionError("server not reachable"))
        app = create_app(settings=ApiSettings(), backend=backend)
        with _client(app) as client:
            response = client.post("/generate", json={"prompt": "hi"})
        assert response.status_code == 503
        body = response.json()
        assert body["error"] == "BackendConnectionError"

    def test_slow_backend_past_timeout_returns_504(self) -> None:
        backend = FakeBackend(chunks=["a", "b", "c"], delay=0.2)
        app = create_app(settings=ApiSettings(), backend=backend)
        with _client(app) as client:
            response = client.post(
                "/generate", json={"prompt": "hi", "timeout_seconds": 0.05}
            )
        assert response.status_code == 504

    def test_response_carries_request_id_header(self) -> None:
        app = create_app(settings=ApiSettings(), backend=FakeBackend())
        with _client(app) as client:
            response = client.post("/generate", json={"prompt": "hi"})
        assert "X-Request-ID" in response.headers

    def test_request_id_header_is_echoed_back(self) -> None:
        app = create_app(settings=ApiSettings(), backend=FakeBackend())
        with _client(app) as client:
            response = client.post(
                "/generate", json={"prompt": "hi"}, headers={"X-Request-ID": "fixed-id-123"}
            )
        assert response.headers["X-Request-ID"] == "fixed-id-123"
        assert response.json()["request_id"] == "fixed-id-123"


class TestGenerateStream:
    def test_streams_sse_chunks_and_terminates_with_done(self) -> None:
        app = create_app(settings=ApiSettings(), backend=FakeBackend(chunks=["foo", "bar"]))
        with _client(app) as client, client.stream(
            "POST", "/generate/stream", json={"prompt": "hi"}
        ) as response:
            assert response.status_code == 200
            lines = [line for line in response.iter_lines() if line]
        assert any('"text": "foo"' in line for line in lines)
        assert any('"text": "bar"' in line for line in lines)
        assert lines[-1] == "data: [DONE]"

    def test_stream_backend_error_yields_error_event(self) -> None:
        backend = FakeBackend(stream_error=BackendConnectionError("down"))
        app = create_app(settings=ApiSettings(), backend=backend)
        with _client(app) as client, client.stream(
            "POST", "/generate/stream", json={"prompt": "hi"}
        ) as response:
            assert response.status_code == 200
            lines = [line for line in response.iter_lines() if line]
        assert any('"error"' in line for line in lines)


class TestConcurrencyAdmission:
    @pytest.mark.asyncio
    async def test_rejects_second_request_when_limit_saturated(self) -> None:
        backend = FakeBackend(chunks=["a"], delay=0.3)
        app = create_app(settings=ApiSettings(concurrency_limit=1), backend=backend)

        async with _running_async_client(app) as client:
            first = asyncio.create_task(client.post("/generate", json={"prompt": "hi"}))
            await asyncio.sleep(0.05)
            second = await client.post("/generate", json={"prompt": "hi"})
            first_response = await first

        assert first_response.status_code == 200
        assert second.status_code == 429


class TestMetrics:
    def test_metrics_endpoint_reports_prometheus_text(self) -> None:
        app = create_app(settings=ApiSettings(), backend=FakeBackend())
        with _client(app) as client:
            client.get("/health")
            response = client.get("/metrics")
        assert response.status_code == 200
        assert "inference_optimizer_requests_total" in response.text
        assert "inference_optimizer_requests_in_flight" in response.text
