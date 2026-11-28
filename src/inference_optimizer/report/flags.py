"""Automatic flags for runs that shouldn't be read as a clean, comparable
baseline -- an incomplete repeat count, any failed requests, a stored
aggregate that doesn't match its own raw data, a resource-contended machine,
or high repeat-to-repeat variability. Flags only annotate; they never change
the stored numbers.
"""

from __future__ import annotations

from inference_optimizer.baseline.models import BaselineReport
from inference_optimizer.report.models import RunFlag, ValidationResult, VariabilityStat

MIN_REPEATS_FOR_A_CLEAN_BASELINE = 3
LOW_AVAILABLE_MEMORY_GB = 3.0
HIGH_VARIABILITY_CV = 0.3


def compute_flags(
    baseline_report: BaselineReport,
    validation: list[ValidationResult],
    variability: list[VariabilityStat],
) -> list[RunFlag]:
    flags: list[RunFlag] = []

    if baseline_report.status == "unavailable":
        flags.append(
            RunFlag(
                level="error",
                code="run_unavailable",
                message=f"Backend unavailable: {baseline_report.unavailable_reason}",
            )
        )
        return flags

    if len(baseline_report.repeats) < MIN_REPEATS_FOR_A_CLEAN_BASELINE:
        flags.append(
            RunFlag(
                level="warning",
                code="incomplete_repeats",
                message=(
                    f"Only {len(baseline_report.repeats)} repeat(s) run "
                    f"(target: >= {MIN_REPEATS_FOR_A_CLEAN_BASELINE})."
                ),
            )
        )

    for repeat in baseline_report.repeats:
        if repeat.summary.failure_rate > 0:
            flags.append(
                RunFlag(
                    level="error" if repeat.summary.failure_rate == 1.0 else "warning",
                    code="requests_failed",
                    message=(
                        f"Repeat {repeat.repeat_index}: failure_rate="
                        f"{repeat.summary.failure_rate:.2f} "
                        f"({repeat.summary.failed_requests}/{repeat.summary.measured_requests})."
                    ),
                )
            )

    for result in validation:
        if not result.matches:
            flags.append(
                RunFlag(
                    level="error",
                    code="aggregate_mismatch",
                    message=(
                        f"Repeat {result.repeat_index}: stored aggregate does not match "
                        f"recomputation from raw observations: {result.mismatches}."
                    ),
                )
            )

    available_gb = baseline_report.run_metadata.hardware.memory.available_gb
    if available_gb < LOW_AVAILABLE_MEMORY_GB:
        flags.append(
            RunFlag(
                level="warning",
                code="low_available_memory",
                message=(
                    f"Only {available_gb} GB RAM free at run time; results may reflect "
                    "machine contention rather than clean baseline performance."
                ),
            )
        )

    for stat in variability:
        cv = stat.coefficient_of_variation
        if cv is not None and cv > HIGH_VARIABILITY_CV:
            flags.append(
                RunFlag(
                    level="warning",
                    code="high_repeat_variability",
                    message=(
                        f"{stat.metric}: coefficient of variation {cv:.2f} across repeats "
                        f"exceeds {HIGH_VARIABILITY_CV} -- repeats are not directly comparable."
                    ),
                )
            )

    if baseline_report.run_metadata.hardware.gpu.available:
        flags.append(
            RunFlag(
                level="info",
                code="gpu_memory_not_process_attributed",
                message=(
                    "GPU memory samples (if present) report whole-device usage via pynvml, "
                    "not this run's own attribution -- see docs/baseline.md."
                ),
            )
        )

    return flags
