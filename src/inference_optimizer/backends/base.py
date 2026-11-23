"""Abstract interface every inference backend implements.

All backends serve over an OpenAI-compatible HTTP endpoint (/v1/chat/completions)
so the benchmarking/orchestration code added in later days can treat llama.cpp,
vLLM, or anything else added later identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from types import TracebackType

from inference_optimizer.backends.models import (
    BackendConfig,
    GenerationResult,
    HealthStatus,
    ModelConfig,
)


class InferenceBackend(ABC):
    """Lifecycle: start() -> [health_check(), generate_stream()]* -> stop()."""

    def __init__(self, model_config: ModelConfig, backend_config: BackendConfig) -> None:
        self._model_config = model_config
        self._backend_config = backend_config

    @property
    def model_info(self) -> ModelConfig:
        """Model loading metadata: name, tokenizer, revision, context limit, quantization."""
        return self._model_config

    @property
    def backend_config(self) -> BackendConfig:
        return self._backend_config

    @property
    def base_url(self) -> str:
        return f"http://{self._backend_config.host}:{self._backend_config.port}"

    @abstractmethod
    async def start(self) -> None:
        """Bring the backend up (launch a server process, etc). Idempotent."""

    @abstractmethod
    async def stop(self) -> None:
        """Tear the backend down. Safe to call even if start() was never called."""

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check whether the backend is reachable and has a model loaded."""

    @abstractmethod
    def generate_stream(self, prompt: str, **params: object) -> AsyncIterator[str]:
        """Stream generated text chunks for a single-turn user prompt."""

    async def generate(self, prompt: str, **params: object) -> GenerationResult:
        """Non-streaming convenience built on generate_stream(); collects all chunks."""
        chunks: list[str] = []
        async for chunk in self.generate_stream(prompt, **params):
            chunks.append(chunk)
        return GenerationResult(text="".join(chunks))

    async def __aenter__(self) -> InferenceBackend:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.stop()
