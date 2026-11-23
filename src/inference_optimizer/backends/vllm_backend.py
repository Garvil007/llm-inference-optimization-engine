"""vLLM backend scaffold: same lifecycle/HTTP contract as LlamaCppBackend, but vLLM has
no official native Windows build and needs a CUDA GPU, so start() checks hardware and
package availability up front and raises UnsupportedHardwareError with a concrete
alternative instead of attempting (and silently failing) a launch.

Verified on this machine: platform is Windows -> start() raises immediately, by design.
The subprocess/HTTP plumbing below is otherwise identical to LlamaCppBackend and would
run unmodified on a Linux host with a CUDA GPU and `pip install vllm`.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator

import httpx

from inference_optimizer.backends.base import InferenceBackend
from inference_optimizer.backends.models import HealthStatus
from inference_optimizer.exceptions import (
    BackendConnectionError,
    BackendStartupError,
    ModelLoadError,
    UnsupportedHardwareError,
)
from inference_optimizer.hardware import detect_hardware
from inference_optimizer.logging import get_logger

logger = get_logger(__name__)

_POLL_INTERVAL_SECONDS = 1.0
_SUPPORTED_OS_NAMES = {"Linux"}


class VLLMBackend(InferenceBackend):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._process: asyncio.subprocess.Process | None = None
        self._client: httpx.AsyncClient | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._output_tail: list[str] = []

    def _check_hardware_support(self) -> None:
        report = detect_hardware()
        if report.os_name not in _SUPPORTED_OS_NAMES:
            raise UnsupportedHardwareError(
                f"vLLM has no official {report.os_name} build; it requires Linux "
                f"(native or WSL2) with a CUDA GPU. Alternative: use the llama-cpp "
                f"backend (CPU-capable), or run this under WSL2 with the `gpu` extra "
                f"installed (`pip install -e '.[gpu]'`)."
            )
        if not report.gpu.available:
            raise UnsupportedHardwareError(
                "vLLM requires a CUDA-capable GPU; none was detected. "
                "Alternative: use the llama-cpp backend instead."
            )
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            raise UnsupportedHardwareError(
                "vllm is not installed. Install it with `pip install -e '.[gpu]'` "
                "on a Linux host with a matching CUDA toolkit."
            ) from exc

    async def start(self) -> None:
        if self._process is not None:
            return

        self._check_hardware_support()

        source = self._model_config.vllm
        if source is None:
            raise ModelLoadError(
                f"Model {self._model_config.name!r} has no 'vllm' source configured."
            )

        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            source.repo_id,
            "--host",
            self._backend_config.host,
            "--port",
            str(self._backend_config.port),
            "--max-model-len",
            str(self._model_config.context_limit),
        ]
        logger.info("vllm_backend_starting", command=" ".join(command))

        self._process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=self._backend_config.request_timeout_seconds
        )
        # Drain stdout concurrently so a chatty startup log can't deadlock the
        # subprocess by filling the OS pipe buffer (see LlamaCppBackend for the
        # same issue observed and fixed against a real server this session).
        self._reader_task = asyncio.create_task(self._pump_output())
        await self._wait_until_ready()

    async def _pump_output(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        async for raw_line in self._process.stdout:
            line = raw_line.decode(errors="replace").rstrip()
            self._output_tail.append(line)
            self._output_tail = self._output_tail[-200:]

    async def _wait_until_ready(self) -> None:
        assert self._client is not None
        deadline = time.monotonic() + self._backend_config.startup_timeout_seconds

        while time.monotonic() < deadline:
            if self._process is not None and self._process.returncode is not None:
                raise BackendStartupError(
                    f"vllm server exited early (code {self._process.returncode}): "
                    + "\n".join(self._output_tail[-20:])
                )
            try:
                response = await self._client.get("/v1/models")
                if response.status_code == 200:
                    logger.info("vllm_backend_ready", base_url=self.base_url)
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

        await self.stop()
        raise BackendStartupError(
            f"vllm server did not become ready within "
            f"{self._backend_config.startup_timeout_seconds}s on {self.base_url}."
        )

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._process is not None:
            if self._process.returncode is None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=10)
                except TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            self._process = None
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None

    async def health_check(self) -> HealthStatus:
        if self._client is None:
            return HealthStatus(healthy=False, backend_type="vllm", detail="Backend not started.")
        try:
            response = await self._client.get("/v1/models")
            response.raise_for_status()
            models = response.json().get("data", [])
            return HealthStatus(
                healthy=True,
                backend_type="vllm",
                detail=f"Reachable at {self.base_url}.",
                model_loaded=len(models) > 0,
            )
        except (httpx.HTTPError, ValueError) as exc:
            return HealthStatus(
                healthy=False, backend_type="vllm", detail=f"Health check failed: {exc}"
            )

    async def generate_stream(self, prompt: str, **params: object) -> AsyncIterator[str]:
        if self._client is None:
            raise BackendConnectionError("Backend not started; call start() first.")

        defaults = self._model_config.default_params
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.get("temperature", defaults.temperature),
            "top_p": params.get("top_p", defaults.top_p),
            "max_tokens": params.get("max_tokens", defaults.max_tokens),
            "stream": True,
        }
        if params.get("seed") is not None:
            payload["seed"] = params["seed"]
        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
        except httpx.HTTPError as exc:
            raise BackendConnectionError(f"Generation request failed: {exc}") from exc
