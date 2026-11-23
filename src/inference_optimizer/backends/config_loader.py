"""Loads a ModelConfig from a YAML file under configs/models/."""

from __future__ import annotations

from pathlib import Path

import yaml

from inference_optimizer.backends.models import ModelConfig
from inference_optimizer.exceptions import ConfigurationError


def load_model_config(path: str | Path) -> ModelConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Model config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        return ModelConfig.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError, yaml structural issues
        raise ConfigurationError(f"Invalid model config at {config_path}: {exc}") from exc
