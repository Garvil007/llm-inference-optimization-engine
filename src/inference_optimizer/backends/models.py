"""Typed schemas shared by every inference backend.

Fields: ModelConfig.name (str), .tokenizer (HF repo id, str), .revision (git sha, str),
.context_limit (int tokens), .quantization (str, e.g. "Q4_K_M"), .license (str),
.default_params.{temperature,top_p,max_tokens}, .llama_cpp.{repo_id,filename,model_path},
.vllm.{repo_id,quantization}. HealthStatus.{healthy (bool), backend_type (str),
detail (str), model_loaded (bool)}. GenerationResult.{text (str), finish_reason (str|None),
prompt_tokens (int|None), completion_tokens (int|None)}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class GenerationParams(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256


class LlamaCppModelSource(BaseModel):
    """Where to get the GGUF file for the llama.cpp backend."""

    repo_id: str
    filename: str
    model_path: Path


class VLLMModelSource(BaseModel):
    """Where to get the full-precision/HF weights for the vLLM backend."""

    repo_id: str
    quantization: str = "none"


class ModelConfig(BaseModel):
    name: str
    tokenizer: str
    revision: str
    context_limit: int
    quantization: str
    license: str
    default_params: GenerationParams = GenerationParams()
    llama_cpp: LlamaCppModelSource | None = None
    vllm: VLLMModelSource | None = None


class BackendConfig(BaseModel):
    backend_type: Literal["llama-cpp", "vllm"]
    host: str = "127.0.0.1"
    port: int = 8731
    startup_timeout_seconds: float = 60.0
    request_timeout_seconds: float = 120.0


class HealthStatus(BaseModel):
    healthy: bool
    backend_type: str
    detail: str
    model_loaded: bool = False


class GenerationResult(BaseModel):
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
