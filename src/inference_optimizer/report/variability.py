"""Repeat-to-repeat measurement variability -- distinct from the within-repeat
stddev already in each BenchmarkSummary's MetricStats. Needs >= 2 repeats;
with exactly 1 repeat there is no repeat-to-repeat spread to report (None,
never fabricated as 0).
"""

from __future__ import annotations

import statistics

from inference_optimizer.baseline.models import BaselineReport
from inference_optimizer.report.models import VariabilityStat

_TRACKED_METRICS = (
    ("latency_ms", "latency_ms_p50"),
    ("ttft_ms", "ttft_ms_p50"),
    ("tpot_ms", "tpot_ms_p50"),
    ("throughput_tokens_per_second", "throughput_tokens_per_second_p50"),
)


def compute_variability(baseline_report: BaselineReport) -> list[VariabilityStat]:
    stats: list[VariabilityStat] = []
    for summary_field, label in _TRACKED_METRICS:
        values = [
            p50
            for repeat in baseline_report.repeats
            if (p50 := getattr(repeat.summary, summary_field).p50) is not None
        ]
        if len(values) < 2:
            stats.append(
                VariabilityStat(
                    metric=label,
                    per_repeat_values=values,
                    mean=None,
                    stddev=None,
                    coefficient_of_variation=None,
                )
            )
            continue

        mean = statistics.fmean(values)
        stddev = statistics.stdev(values)
        stats.append(
            VariabilityStat(
                metric=label,
                per_repeat_values=values,
                mean=mean,
                stddev=stddev,
                coefficient_of_variation=(stddev / mean) if mean else None,
            )
        )
    return stats
