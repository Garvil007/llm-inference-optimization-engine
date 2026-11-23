"""Optional real-backend smoke test: actually launches llama_cpp.server against the
downloaded Qwen2.5-0.5B-Instruct GGUF and performs one real streaming generation.

Skipped by default (slow, needs the model file + the `serving` extra installed).
Run explicitly with:
    RUN_REAL_BACKEND_TESTS=1 pytest tests/integration/test_llama_cpp_smoke.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from inference_optimizer.backends.config_loader import load_model_config
from inference_optimizer.backends.registry import create_backend

_DEFAULT_CONFIG_PATH = Path("configs/models/qwen2.5-0.5b-instruct.yaml")

pytestmark = pytest.mark.real_backend

_SKIP_REASON = "Set RUN_REAL_BACKEND_TESTS=1 to run the real llama.cpp smoke test."


@pytest.mark.skipif(os.environ.get("RUN_REAL_BACKEND_TESTS") != "1", reason=_SKIP_REASON)
class TestLlamaCppRealBackend:
    @pytest.mark.asyncio
    async def test_completes_a_real_prompt_response_cycle(self) -> None:
        model_config = load_model_config(_DEFAULT_CONFIG_PATH)
        assert model_config.llama_cpp is not None
        if not model_config.llama_cpp.model_path.exists():
            pytest.skip(
                f"GGUF not found at {model_config.llama_cpp.model_path}; download it first "
                f"(see README.md)."
            )

        backend = create_backend("llama-cpp", model_config)
        try:
            await backend.start()
            status = await backend.health_check()
            assert status.healthy is True

            result = await backend.generate("Say the single word: hello")
            assert len(result.text.strip()) > 0
        finally:
            await backend.stop()
