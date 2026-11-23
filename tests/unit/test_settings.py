"""Unit tests for inference_optimizer.config.settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from inference_optimizer.config.settings import Settings


class TestSettingsDefaults:
    def test_uses_sane_defaults_with_no_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("INFERENCE_OPTIMIZER_ENVIRONMENT", raising=False)
        monkeypatch.delenv("INFERENCE_OPTIMIZER_LOG_LEVEL", raising=False)

        settings = Settings(_env_file=None)

        assert settings.environment == "development"
        assert settings.log_level == "INFO"
        assert settings.results_dir == Path("results")
        assert settings.model_cache_dir == Path(".cache/models")


class TestSettingsFromEnv:
    def test_reads_overrides_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INFERENCE_OPTIMIZER_ENVIRONMENT", "production")
        monkeypatch.setenv("INFERENCE_OPTIMIZER_LOG_LEVEL", "debug")
        monkeypatch.setenv("INFERENCE_OPTIMIZER_RESULTS_DIR", "/tmp/results")

        settings = Settings(_env_file=None)

        assert settings.environment == "production"
        assert settings.log_level == "DEBUG"
        assert settings.results_dir == Path("/tmp/results")


class TestSettingsValidation:
    def test_rejects_invalid_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INFERENCE_OPTIMIZER_LOG_LEVEL", "NOT_A_LEVEL")

        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_rejects_invalid_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INFERENCE_OPTIMIZER_ENVIRONMENT", "staging-ish")

        with pytest.raises(ValidationError):
            Settings(_env_file=None)
