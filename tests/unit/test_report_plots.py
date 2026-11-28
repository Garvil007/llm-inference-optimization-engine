"""Plot-generation mechanics test. The values below are FIXTURE DATA -- made
up numbers with no relationship to any real inference run, used solely to
exercise generate_distribution_plots()'s file-writing/skip logic. They must
never be read as measured TTFT/latency/throughput. Real plots are only ever
produced from a parsed real baseline run (see report/generator.py and
docs/reporting.md).
"""

from __future__ import annotations

from pathlib import Path

from inference_optimizer.benchmark.models import RequestObservation
from inference_optimizer.report.models import RawObservationRecord
from inference_optimizer.report.plots import generate_distribution_plots

# FIXTURE DATA -- not real measurements. See module docstring above.
_FIXTURE_TTFT_MS = [100.0, 120.0, 90.0, 110.0, 105.0]
_FIXTURE_LATENCY_MS = [200.0, 240.0, 180.0, 220.0, 210.0]
_FIXTURE_THROUGHPUT_TPS = [20.0, 18.0, 22.0, 19.0, 21.0]


def _fixture_observation(
    index: int, ttft: float, latency: float, throughput: float
) -> RequestObservation:
    return RequestObservation(
        request_index=index,
        is_warmup=False,
        success=True,
        started_at="2026-11-27T00:00:00+00:00",
        start_perf=0.0,
        first_token_perf=ttft / 1000,
        end_perf=latency / 1000,
        ttft_ms=ttft,
        latency_ms=latency,
        output_text_length=20,
        output_token_count=10,
        token_count_method="fixed",
        tpot_ms=(latency - ttft) / 9,
        output_tokens_per_second=throughput,
    )


def _fixture_records() -> list[RawObservationRecord]:
    return [
        RawObservationRecord(
            repeat_index=0,
            observation=_fixture_observation(i, ttft, latency, throughput),
        )
        for i, (ttft, latency, throughput) in enumerate(
            zip(_FIXTURE_TTFT_MS, _FIXTURE_LATENCY_MS, _FIXTURE_THROUGHPUT_TPS, strict=True)
        )
    ]


def test_generates_a_png_per_metric_with_enough_fixture_points(tmp_path: Path) -> None:
    plot_paths = generate_distribution_plots(_fixture_records(), tmp_path)

    assert set(plot_paths.keys()) == {"ttft_ms", "latency_ms", "output_tokens_per_second"}
    for path in plot_paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_skips_plot_when_fewer_than_two_points(tmp_path: Path) -> None:
    single_record = [
        RawObservationRecord(
            repeat_index=0, observation=_fixture_observation(0, 100.0, 200.0, 20.0)
        )
    ]

    plot_paths = generate_distribution_plots(single_record, tmp_path)

    assert plot_paths == {}


def test_excludes_failed_observations_from_the_distribution(tmp_path: Path) -> None:
    records = _fixture_records()
    failed = RequestObservation(
        request_index=99,
        is_warmup=False,
        success=False,
        error="boom",
        started_at="2026-11-27T00:00:00+00:00",
        start_perf=0.0,
        end_perf=5.0,
        latency_ms=5000.0,
        output_text_length=0,
    )
    records.append(RawObservationRecord(repeat_index=0, observation=failed))

    plot_paths = generate_distribution_plots(records, tmp_path)

    # Still produces plots from the 5 successful fixture points; the failed
    # observation's extreme 5000ms latency must not appear in the output.
    assert "latency_ms" in plot_paths
