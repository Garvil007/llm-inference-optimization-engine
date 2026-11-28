"""Distribution plots (TTFT, end-to-end latency, output tokens/sec) rendered
with matplotlib's non-interactive Agg backend (no display needed, works
headless/CI). Only ever called with real measured observations from a parsed
baseline run in production code paths -- tests pass explicitly
fixture-labeled synthetic data to exercise the plotting mechanics only (see
tests/unit/test_report_plots.py), never claiming it as a real measurement.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - backend must be set before this import

from inference_optimizer.report.models import RawObservationRecord

_MIN_POINTS_FOR_A_DISTRIBUTION = 2

_PLOTS: tuple[tuple[str, str, str], ...] = (
    ("ttft_ms", "ttft_distribution.png", "Time to First Token (ms)"),
    ("latency_ms", "latency_distribution.png", "End-to-End Latency (ms)"),
    ("output_tokens_per_second", "throughput_distribution.png", "Output Tokens/sec"),
)


def _histogram(values: list[float], xlabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=min(10, max(len(values), 1)), color="#4C72B0", edgecolor="white")
    ax.set_title(xlabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Requests")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def generate_distribution_plots(
    raw_records: list[RawObservationRecord], output_dir: Path
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    successful = [record.observation for record in raw_records if record.observation.success]

    plot_paths: dict[str, Path] = {}
    for field_name, filename, xlabel in _PLOTS:
        values = [v for o in successful if (v := getattr(o, field_name)) is not None]
        if len(values) < _MIN_POINTS_FOR_A_DISTRIBUTION:
            continue  # not enough points for a meaningful distribution plot
        output_path = output_dir / filename
        _histogram(values, xlabel, output_path)
        plot_paths[field_name] = output_path
    return plot_paths
