"""Typed schemas for the Day 7 report pipeline: everything computed about a
parsed baseline run before rendering to Markdown/HTML -- a raw-observation
wrapper that restores the `repeat_index` tag the baseline runner stamped onto
each JSONL line, per-repeat validation results, repeat-to-repeat variability,
quality flags, and internal-consistency comparison notes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from inference_optimizer.baseline.models import BaselineReport
from inference_optimizer.benchmark.models import RequestObservation

FlagLevel = Literal["info", "warning", "error"]


class RawObservationRecord(BaseModel):
    repeat_index: int
    observation: RequestObservation


class ValidationResult(BaseModel):
    repeat_index: int
    matches: bool
    mismatches: list[str] = []


class VariabilityStat(BaseModel):
    metric: str
    per_repeat_values: list[float]
    mean: float | None
    stddev: float | None
    coefficient_of_variation: float | None
    """stddev / mean across repeats -- a unitless measure of repeat-to-repeat
    spread, distinct from the within-repeat stddev already in MetricStats.
    None when fewer than 2 repeats exist (nothing to compute) or mean is 0."""


class RunFlag(BaseModel):
    level: FlagLevel
    code: str
    message: str


class ComparisonNote(BaseModel):
    metric: str
    description: str
    passed: bool
    detail: str


class ReportData(BaseModel):
    baseline_report: BaselineReport
    validation: list[ValidationResult]
    variability: list[VariabilityStat]
    flags: list[RunFlag]
    comparisons: list[ComparisonNote]
    plot_paths: dict[str, Path]
