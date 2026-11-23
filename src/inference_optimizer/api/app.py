"""FastAPI service: a thin HTTP layer over InferenceBackend. This module has zero
knowledge of llama.cpp/vLLM internals -- only the InferenceBackend interface -- so the
backend stays testable and usable outside of FastAPI (e.g. from the CLI, as it already
is).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from inference_optimizer.api.concurrency import AdmissionRejected, ConcurrencyLimiter
from inference_optimizer.api.config import ApiSettings, get_api_settings
from inference_optimizer.api.metrics import Metrics
from inference_optimizer.api.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfo,
)
from inference_optimizer.backends import InferenceBackend, create_backend, load_model_config
from inference_optimizer.exceptions import (
    BackendConnectionError,
    BackendStartupError,
    InferenceOptimizerError,
    ModelLoadError,
    UnsupportedHardwareError,
)
from inference_optimizer.logging import configure_logging, get_logger

logger = get_logger(__name__)

_SERVICE_UNAVAILABLE_ERRORS = (
    BackendConnectionError,
    BackendStartupError,
    ModelLoadError,
    UnsupportedHardwareError,
)


def create_app(
    settings: ApiSettings | None = None, backend: InferenceBackend | None = None
) -> FastAPI:
    """Build the FastAPI app.

    `backend` lets tests inject a fake InferenceBackend directly, so the test suite
    never spawns a real llama.cpp/vLLM subprocess. Production code (the `serve` CLI
    command) leaves it None and the lifespan constructs a real one from `settings`.
    """
    resolved_settings = settings or get_api_settings()
    configure_logging()

    state: dict[str, object] = {
        "metrics": Metrics(),
        "limiter": ConcurrencyLimiter(resolved_settings.concurrency_limit),
    }

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        app_backend = backend
        if app_backend is None:
            model_config = load_model_config(resolved_settings.model_config_path)
            app_backend = create_backend(resolved_settings.backend_type, model_config)
        await app_backend.start()
        state["backend"] = app_backend
        logger.info("api_backend_ready", backend=resolved_settings.backend_type)
        try:
            yield
        finally:
            await app_backend.stop()

    app = FastAPI(
        title="Inference Optimizer API",
        description=(
            "Local inference serving API. No authentication: intended for local/dev "
            "use behind a trusted network boundary. See README for deployment "
            "security assumptions before exposing this beyond localhost."
        ),
        lifespan=lifespan,
    )

    def _backend() -> InferenceBackend:
        return state["backend"]  # type: ignore[return-value]

    def _limiter() -> ConcurrencyLimiter:
        return state["limiter"]  # type: ignore[return-value]

    def _metrics() -> Metrics:
        return state["metrics"]  # type: ignore[return-value]

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.monotonic()
        metrics = _metrics()
        await metrics.inc_in_flight()
        with structlog.contextvars.bound_contextvars(request_id=request_id):
            logger.info("request_started", method=request.method, path=request.url.path)
            try:
                response = await call_next(request)
            except Exception:
                duration = time.monotonic() - start
                await metrics.observe_request(request.method, request.url.path, 500, duration)
                await metrics.dec_in_flight()
                logger.exception(
                    "request_failed", method=request.method, path=request.url.path
                )
                raise
            duration = time.monotonic() - start
            await metrics.observe_request(
                request.method, request.url.path, response.status_code, duration
            )
            await metrics.dec_in_flight()
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "request_finished",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_seconds=round(duration, 4),
            )
            return response

    def _error_response(
        status_code: int, error: str, detail: str, request_id: str
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(error=error, detail=detail, request_id=request_id).model_dump(),
        )

    @app.exception_handler(AdmissionRejected)
    async def _handle_admission_rejected(
        request: Request, exc: AdmissionRejected
    ) -> JSONResponse:
        request_id = request.headers.get("X-Request-ID", "unknown")
        return _error_response(429, "too_many_requests", str(exc), request_id)

    @app.exception_handler(TimeoutError)
    async def _handle_timeout(request: Request, exc: TimeoutError) -> JSONResponse:
        request_id = request.headers.get("X-Request-ID", "unknown")
        return _error_response(
            504, "generation_timeout", "The backend did not respond in time.", request_id
        )

    @app.exception_handler(InferenceOptimizerError)
    async def _handle_backend_error(
        request: Request, exc: InferenceOptimizerError
    ) -> JSONResponse:
        request_id = request.headers.get("X-Request-ID", "unknown")
        status_code = 503 if isinstance(exc, _SERVICE_UNAVAILABLE_ERRORS) else 500
        return _error_response(status_code, type(exc).__name__, str(exc), request_id)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        status = await _backend().health_check()
        return HealthResponse(**status.model_dump())

    @app.get("/models", response_model=list[ModelInfo])
    async def models() -> list[ModelInfo]:
        model_config = _backend().model_info
        return [
            ModelInfo(
                name=model_config.name,
                backend=resolved_settings.backend_type,
                context_limit=model_config.context_limit,
                quantization=model_config.quantization,
                license=model_config.license,
            )
        ]

    def _build_params(body: GenerateRequest) -> dict[str, object]:
        params: dict[str, object] = {}
        if body.max_tokens is not None:
            params["max_tokens"] = body.max_tokens
        if body.temperature is not None:
            params["temperature"] = body.temperature
        if body.top_p is not None:
            params["top_p"] = body.top_p
        if body.seed is not None:
            params["seed"] = body.seed
        return params

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        timeout = body.timeout_seconds or resolved_settings.default_request_timeout_seconds
        start = time.monotonic()
        async with _limiter().admit():
            result = await asyncio.wait_for(
                _backend().generate(body.prompt, **_build_params(body)), timeout=timeout
            )
        return GenerateResponse(
            text=result.text,
            finish_reason=result.finish_reason,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            model=_backend().model_info.name,
            backend=resolved_settings.backend_type,
            request_id=request_id,
            latency_seconds=round(time.monotonic() - start, 4),
        )

    @app.post("/generate/stream")
    async def generate_stream(body: GenerateRequest, request: Request) -> StreamingResponse:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        timeout = body.timeout_seconds or resolved_settings.default_request_timeout_seconds

        # Admission must be checked *before* StreamingResponse is constructed: once the
        # ASGI response starts streaming, headers (status 200) are already committed and
        # an exception raised inside the generator can no longer become a 429 response.
        admission = _limiter().admit()
        await admission.__aenter__()

        async def event_source() -> AsyncIterator[str]:
            try:
                deadline = time.monotonic() + timeout
                stream = _backend().generate_stream(body.prompt, **_build_params(body))
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("stream exceeded timeout_seconds")
                    try:
                        chunk = await asyncio.wait_for(
                            stream.__anext__(), timeout=remaining
                        )
                    except StopAsyncIteration:
                        break
                    yield f"data: {json.dumps({'text': chunk, 'request_id': request_id})}\n\n"
                yield "data: [DONE]\n\n"
            except (InferenceOptimizerError, TimeoutError) as exc:
                logger.error("stream_failed", request_id=request_id, error=str(exc))
                yield f"data: {json.dumps({'error': str(exc), 'request_id': request_id})}\n\n"
            finally:
                await admission.__aexit__(None, None, None)

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        return Response(
            content=_metrics().render_prometheus(), media_type="text/plain; version=0.0.4"
        )

    return app
