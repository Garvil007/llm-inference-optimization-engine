"""Day 7 report pipeline entry point: parses real baseline artifacts, validates
aggregates against raw observations, computes repeat-to-repeat variability,
flags incomplete/failed/non-comparable runs, builds internal-consistency
comparisons, renders distribution plots, and writes a Markdown + HTML report
(plus the plot PNGs) to an output directory.
"""

from __future__ import annotations

from pathlib import Path

from inference_optimizer.exceptions import ConfigurationError
from inference_optimizer.report.comparison import build_comparisons
from inference_optimizer.report.flags import compute_flags
from inference_optimizer.report.html_renderer import render_html
from inference_optimizer.report.markdown_renderer import render_markdown
from inference_optimizer.report.models import ReportData
from inference_optimizer.report.parser import load_baseline_report, load_raw_observations
from inference_optimizer.report.plots import generate_distribution_plots
from inference_optimizer.report.validate import validate_aggregates
from inference_optimizer.report.variability import compute_variability


def generate_report(
    baseline_report_path: Path,
    observations_path: Path | None,
    output_dir: Path,
) -> ReportData:
    baseline_report = load_baseline_report(baseline_report_path)

    if baseline_report.status == "unavailable":
        report_data = ReportData(
            baseline_report=baseline_report,
            validation=[],
            variability=[],
            flags=compute_flags(baseline_report, [], []),
            comparisons=[],
            plot_paths={},
        )
    else:
        if observations_path is None:
            raise ConfigurationError(
                "observations_path is required for a completed baseline run"
            )
        raw_records = load_raw_observations(observations_path)
        validation = validate_aggregates(baseline_report, raw_records)
        variability = compute_variability(baseline_report)
        flags = compute_flags(baseline_report, validation, variability)
        comparisons = build_comparisons(baseline_report)
        plot_paths = generate_distribution_plots(raw_records, output_dir)
        report_data = ReportData(
            baseline_report=baseline_report,
            validation=validation,
            variability=variability,
            flags=flags,
            comparisons=comparisons,
            plot_paths=plot_paths,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(render_markdown(report_data), encoding="utf-8")
    (output_dir / "report.html").write_text(render_html(report_data), encoding="utf-8")
    return report_data
