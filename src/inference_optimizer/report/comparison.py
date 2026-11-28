"""Internal-consistency checks only -- there is no documented SLA or vendor
baseline for this model+hardware combination yet (this IS the first baseline
run), so nothing here compares measured numbers against an external target.
Each check instead verifies that the measured numbers are mutually
consistent, which is the only "expectation" that can be stated without
fabricating a claim this project hasn't earned yet.
"""

from __future__ import annotations

from inference_optimizer.baseline.models import BaselineReport
from inference_optimizer.report.models import ComparisonNote

_TOLERANCE_MS = 0.01


def build_comparisons(baseline_report: BaselineReport) -> list[ComparisonNote]:
    notes: list[ComparisonNote] = []

    for repeat in baseline_report.repeats:
        summary = repeat.summary
        if summary.ttft_ms.p50 is not None and summary.latency_ms.p50 is not None:
            passed = summary.ttft_ms.p50 <= summary.latency_ms.p50 + _TOLERANCE_MS
            notes.append(
                ComparisonNote(
                    metric="ttft_vs_latency",
                    description=f"repeat {repeat.repeat_index}: TTFT p50 <= latency p50",
                    passed=passed,
                    detail=(
                        f"ttft_p50={summary.ttft_ms.p50}ms latency_p50={summary.latency_ms.p50}ms"
                    ),
                )
            )

    notes.append(
        ComparisonNote(
            metric="no_external_baseline",
            description=(
                "No documented SLA or expected-performance target exists yet for this "
                "model+hardware combination -- this is the first baseline run."
            ),
            passed=True,
            detail="Informational only; not a pass/fail check against an external target.",
        )
    )
    return notes
