"""Day 7 report pipeline: parses Day 6 baseline artifacts, validates aggregates
against raw observations, computes repeat-to-repeat variability and quality
flags, builds internal-consistency comparisons (no external SLA exists yet),
and renders Markdown + HTML reports with real-measurement distribution plots.
See docs/reporting.md.
"""

from inference_optimizer.report.generator import generate_report
from inference_optimizer.report.models import (
    ComparisonNote,
    RawObservationRecord,
    ReportData,
    RunFlag,
    ValidationResult,
    VariabilityStat,
)

__all__ = [
    "generate_report",
    "ReportData",
    "RunFlag",
    "ValidationResult",
    "VariabilityStat",
    "ComparisonNote",
    "RawObservationRecord",
]
