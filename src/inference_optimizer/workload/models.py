"""Typed schemas for the workload dataset and its manifest.

A PromptRecord's `input_length_target`/`output_target_tokens` are the categorical
bucket labels from the Day 5 brief (128/512/2048 input, 64/128/256 output) --
`input_tokens_measured` is what the actual tokenizer actually counted for the
generated text. These are *not* always equal: the 2048-token input tier is clamped
during generation to fit this model's 2048-token context window alongside its
output budget (see generator.py), so its measured length is intentionally lower
than its label. The distinction is kept explicit end-to-end, mirroring Day 4's
measured-vs-target discipline for output tokens.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

Category = Literal[
    "short_qa",
    "summarization",
    "code_generation",
    "structured_extraction",
    "long_context_qa",
]
InputLengthTarget = Literal[128, 512, 2048]
OutputTargetTokens = Literal[64, 128, 256]
LengthCalculationMethod = Literal["tokenizer", "approximate"]


class PromptRecord(BaseModel):
    id: str
    category: Category
    input_length_target: InputLengthTarget
    output_target_tokens: OutputTargetTokens
    prompt: str
    input_tokens_measured: int
    length_calculation_method: LengthCalculationMethod


class WorkloadManifest(BaseModel):
    dataset_version: str
    dataset_path: str
    tokenizer_repo_id: str
    tokenizer_method: LengthCalculationMethod
    generated_at: datetime
    seed: int
    record_count: int
    generation_settings: dict[str, object]
    """Everything needed to reproduce this exact dataset: categories, length tiers,
    output-target mapping, tolerance, context_limit/safety_margin used for clamping.
    """
    source_file_hashes: dict[str, str]
    """sha256 of the generator source and the produced dataset file, so a consumer
    can confirm the dataset on disk matches what this manifest describes."""
