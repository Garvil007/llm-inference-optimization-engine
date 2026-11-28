"""Renders a ReportData into a Markdown document: hardware/model tables,
per-repeat metrics table, repeat-to-repeat variability, flags, internal
consistency comparisons, validation results, and (if any) distribution plot
images referenced by relative filename.
"""

from __future__ import annotations

from inference_optimizer.report.content import hardware_table, metrics_table, model_table
from inference_optimizer.report.models import ReportData


def _table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_markdown(report_data: ReportData) -> str:
    report = report_data.baseline_report
    lines: list[str] = [f"# Baseline report: run {report.run_metadata.run_id}", ""]
    lines.append(f"Status: **{report.status}**")
    if report.status == "unavailable":
        lines.append(f"Unavailable reason: {report.unavailable_reason}")
        return "\n".join(lines)

    lines += [
        "",
        "## Hardware",
        _table(["Field", "Value"], [[k, v] for k, v in hardware_table(report)]),
    ]
    lines += [
        "",
        "## Model & backend",
        _table(["Field", "Value"], [[k, v] for k, v in model_table(report)]),
    ]

    metric_rows = metrics_table(report)
    if metric_rows:
        headers = list(metric_rows[0].keys())
        lines += [
            "",
            "## Per-repeat metrics",
            _table(headers, [[row[h] for h in headers] for row in metric_rows]),
        ]

    lines += ["", "## Repeat-to-repeat variability"]
    for stat in report_data.variability:
        if stat.coefficient_of_variation is not None:
            lines.append(
                f"- {stat.metric}: mean={stat.mean:.2f}, stddev={stat.stddev:.2f}, "
                f"CV={stat.coefficient_of_variation:.3f} (values: {stat.per_repeat_values})"
            )
        else:
            lines.append(
                f"- {stat.metric}: not enough repeats to compute variability "
                f"(values: {stat.per_repeat_values})"
            )

    lines += ["", "## Flags"]
    if report_data.flags:
        for flag in report_data.flags:
            lines.append(f"- **[{flag.level.upper()}] {flag.code}**: {flag.message}")
    else:
        lines.append("- none")

    lines += ["", "## Comparisons (internal consistency only)"]
    for note in report_data.comparisons:
        status = "PASS" if note.passed else "FAIL"
        lines.append(f"- [{status}] {note.description} -- {note.detail}")

    if report_data.plot_paths:
        lines += ["", "## Distribution plots"]
        for name, path in report_data.plot_paths.items():
            lines.append(f"![{name}]({path.name})")

    lines += ["", "## Validation against raw observations"]
    for result in report_data.validation:
        status = "OK" if result.matches else "MISMATCH"
        suffix = f" {result.mismatches}" if result.mismatches else ""
        lines.append(f"- repeat {result.repeat_index}: {status}{suffix}")

    return "\n".join(lines)
