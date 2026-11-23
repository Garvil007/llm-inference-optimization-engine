"""Typed, environment-driven application settings.

Chose pydantic-settings over raw os.environ/configparser/dynaconf because the
project already depends on pydantic for the hardware report models, so this
adds validation (type coercion, required-field errors) with no new dependency
family. All fields are overridable via INFERENCE_OPTIMIZER_* env vars or a
.env file (see .env.example).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_VALID_ENVIRONMENTS = {"development", "test", "production"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_OPTIMIZER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"
    log_level: str = "INFO"
    results_dir: Path = Path("results")
    model_cache_dir: Path = Path(".cache/models")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, got {value!r}"
            )
        return normalized

    @field_validator("environment")
    @classmethod
    def _validate_environment(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in _VALID_ENVIRONMENTS:
            raise ValueError(
                f"environment must be one of {sorted(_VALID_ENVIRONMENTS)}, got {value!r}"
            )
        return normalized


def get_settings() -> Settings:
    """Load settings fresh from the environment / .env file on every call.

    No module-level singleton: tests mutate env vars per-case and must see
    the change, and this is cheap enough to not need caching.
    """
    return Settings()
