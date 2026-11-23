"""Typed request/response schemas for the HTTP API. Separate from backends/models.py:
those describe backend-internal config, these describe the wire contract, and the two
are allowed to diverge (e.g. the API exposes `seed`, which ModelConfig.default_params
does not need).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    max_tokens: int | None = Field(default=None, gt=0, le=8192)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    seed: int | None = Field(default=None, description="Deterministic sampling seed, if supported.")
    timeout_seconds: float | None = Field(
        default=None, gt=0, le=600, description="Per-request override of the server default."
    )


class GenerateResponse(BaseModel):
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str
    backend: str
    request_id: str
    latency_seconds: float


class ModelInfo(BaseModel):
    name: str
    backend: str
    context_limit: int
    quantization: str
    license: str


class HealthResponse(BaseModel):
    healthy: bool
    backend_type: str
    detail: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str
