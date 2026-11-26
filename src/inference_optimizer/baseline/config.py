"""BaselineConfig: typed settings for a Day 6 baseline run, loaded from YAML
(see configs/baseline.yaml). Mirrors backends/config_loader.py's load pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError

from inference_optimizer.exceptions import ConfigurationError
from inference_optimizer.workload.models import Category, InputLengthTarget


class BaselineConfig(BaseModel):
    model_config_path: Path = Path("configs/models/qwen2.5-0.5b-instruct.yaml")
    backend_type: Literal["llama-cpp", "vllm"] = "llama-cpp"
    workload_path: Path = Path("data/workloads/prompts_v1.jsonl")
    sample_count: int = 10
    sample_seed: int = 0
    categories: list[Category] | None = None
    input_length_targets: list[InputLengthTarget] | None = None
    warmup_count: int = 2
    repeats: int = 3
    tokenizer_repo: str | None = None
    max_tokens_override: int | None = None
    output_dir: Path = Path("results/baseline")


def load_baseline_config(path: Path) -> BaselineConfig:
    if not path.exists():
        raise ConfigurationError(f"Baseline config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return BaselineConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid baseline config at {path}: {exc}") from exc
