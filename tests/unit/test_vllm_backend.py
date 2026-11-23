"""Unit tests for VLLMBackend. The only path actually exercisable on this dev
machine (Windows, no CUDA GPU) is the hardware-gating failure; the HTTP/stream
plumbing is identical to LlamaCppBackend and tested the same way with a mocked
transport, bypassing _check_hardware_support directly.
"""

from __future__ import annotations

import httpx
import pytest

from inference_optimizer.backends.models import BackendConfig, ModelConfig, VLLMModelSource
from inference_optimizer.backends.vllm_backend import VLLMBackend
from inference_optimizer.exceptions import UnsupportedHardwareError
from inference_optimizer.hardware.models import (
    CPUInfo,
    DiskInfo,
    GPUInfo,
    HardwareReport,
    MemoryInfo,
    SoftwareInfo,
)


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="test-model",
        tokenizer="test/tokenizer",
        revision="deadbeef",
        context_limit=2048,
        quantization="none",
        license="apache-2.0",
        vllm=VLLMModelSource(repo_id="test/repo"),
    )


def _backend_config() -> BackendConfig:
    return BackendConfig(backend_type="vllm", port=18732)


def _fake_report(os_name: str, gpu_available: bool) -> HardwareReport:
    return HardwareReport(
        os_name=os_name,
        os_version=f"{os_name} test",
        python_version="3.12.0",
        cpu=CPUInfo(physical_cores=4, logical_cores=8, architecture="x86_64"),
        memory=MemoryInfo(total_gb=16.0, available_gb=8.0),
        gpu=GPUInfo(available=gpu_available, name="fake-gpu" if gpu_available else None),
        disk=DiskInfo(total_gb=100.0, free_gb=50.0, path="."),
        software=SoftwareInfo(docker_available=False),
    )


class TestHardwareGating:
    def test_rejects_non_linux_os(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import inference_optimizer.backends.vllm_backend as module

        monkeypatch.setattr(module, "detect_hardware", lambda: _fake_report("Windows", True))
        backend = VLLMBackend(_model_config(), _backend_config())

        with pytest.raises(UnsupportedHardwareError, match="Windows"):
            backend._check_hardware_support()  # noqa: SLF001 - testing the gate directly

    def test_rejects_missing_gpu_on_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import inference_optimizer.backends.vllm_backend as module

        monkeypatch.setattr(module, "detect_hardware", lambda: _fake_report("Linux", False))
        backend = VLLMBackend(_model_config(), _backend_config())

        with pytest.raises(UnsupportedHardwareError, match="CUDA"):
            backend._check_hardware_support()  # noqa: SLF001 - testing the gate directly

    def test_rejects_when_vllm_package_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        import inference_optimizer.backends.vllm_backend as module

        monkeypatch.setattr(module, "detect_hardware", lambda: _fake_report("Linux", True))
        real_import = builtins.__import__

        def _fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "vllm":
                raise ImportError("no module named vllm")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        backend = VLLMBackend(_model_config(), _backend_config())

        with pytest.raises(UnsupportedHardwareError, match="not installed"):
            backend._check_hardware_support()  # noqa: SLF001 - testing the gate directly

    @pytest.mark.asyncio
    async def test_start_raises_before_spawning_any_subprocess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import inference_optimizer.backends.vllm_backend as module

        monkeypatch.setattr(module, "detect_hardware", lambda: _fake_report("Windows", False))
        backend = VLLMBackend(_model_config(), _backend_config())

        with pytest.raises(UnsupportedHardwareError):
            await backend.start()

        assert backend._process is None  # noqa: SLF001 - confirms no subprocess was launched


class TestGenerateStreamPlumbing:
    """Same OpenAI-compatible SSE parsing as LlamaCppBackend; exercised directly
    against a mocked transport since start() is hardware-gated on this machine.
    """

    @pytest.mark.asyncio
    async def test_yields_chunks_from_sse_stream(self) -> None:
        sse_body = (
            'data: {"choices":[{"delta":{"content":"Bonjour"}}]}\n\n'
            "data: [DONE]\n\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse_body)

        backend = VLLMBackend(_model_config(), _backend_config())
        backend._client = httpx.AsyncClient(  # noqa: SLF001 - test injects transport directly
            transport=httpx.MockTransport(handler), base_url=backend.base_url
        )

        chunks = [chunk async for chunk in backend.generate_stream("hi")]

        assert chunks == ["Bonjour"]
