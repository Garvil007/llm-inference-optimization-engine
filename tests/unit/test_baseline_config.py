"""BaselineConfig loading: defaults, overrides, missing file, invalid YAML."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.baseline.config import BaselineConfig, load_baseline_config
from inference_optimizer.exceptions import ConfigurationError


class TestLoadBaselineConfig:
    def test_missing_file_raises_configuration_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_baseline_config(tmp_path / "missing.yaml")

    def test_invalid_yaml_content_raises_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.yaml"
        path.write_text("repeats: not-an-int\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Invalid baseline config"):
            load_baseline_config(path)

    def test_empty_file_yields_all_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.yaml"
        path.write_text("", encoding="utf-8")
        config = load_baseline_config(path)
        assert config == BaselineConfig()

    def test_overrides_are_applied(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.yaml"
        path.write_text(
            "sample_count: 4\nsample_seed: 7\nrepeats: 5\nwarmup_count: 1\n"
            "backend_type: llama-cpp\ncategories: [short_qa]\n",
            encoding="utf-8",
        )
        config = load_baseline_config(path)
        assert config.sample_count == 4
        assert config.sample_seed == 7
        assert config.repeats == 5
        assert config.warmup_count == 1
        assert config.categories == ["short_qa"]
