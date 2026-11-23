"""API-specific settings, separate from the core Settings (which is about CLI/results
paths, not HTTP serving). Same pydantic-settings pattern, its own env prefix so API
concerns don't collide with CLI/backend env vars.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_MODEL_CONFIG = Path("configs/models/qwen2.5-0.5b-instruct.yaml")


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_OPTIMIZER_API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    backend_type: str = "llama-cpp"
    model_config_path: Path = _DEFAULT_MODEL_CONFIG
    concurrency_limit: int = 4
    default_request_timeout_seconds: float = 120.0
    host: str = "127.0.0.1"
    port: int = 8000


def get_api_settings() -> ApiSettings:
    """Load fresh from the environment on every call (same rationale as get_settings())."""
    return ApiSettings()
