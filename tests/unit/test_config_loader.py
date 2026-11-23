"""Unit tests for backends.config_loader.load_model_config."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.backends.config_loader import load_model_config
from inference_optimizer.exceptions import ConfigurationError

_VALID_YAML = """
name: test-model
tokenizer: test/tokenizer
revision: abc123
context_limit: 2048
quantization: Q4_K_M
license: apache-2.0
default_params:
  temperature: 0.5
  top_p: 0.8
  max_tokens: 128
llama_cpp:
  repo_id: test/repo
  filename: model.gguf
  model_path: .cache/models/model.gguf
"""


class TestLoadModelConfig:
    def test_loads_valid_yaml_into_model_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "model.yaml"
        config_file.write_text(_VALID_YAML, encoding="utf-8")

        config = load_model_config(config_file)

        assert config.name == "test-model"
        assert config.context_limit == 2048
        assert config.default_params.temperature == 0.5
        assert config.llama_cpp is not None
        assert config.llama_cpp.repo_id == "test/repo"

    def test_raises_configuration_error_when_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_model_config(tmp_path / "missing.yaml")

    def test_raises_configuration_error_on_invalid_structure(self, tmp_path: Path) -> None:
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("name: test-model\n", encoding="utf-8")  # missing required fields

        with pytest.raises(ConfigurationError, match="Invalid model config"):
            load_model_config(config_file)
